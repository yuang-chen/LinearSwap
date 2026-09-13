"""Stage 1½ — distill: recover the pretrained function for an *inexact* swap.

    python linswap.py distill --kernel deltanet                 # layer alignment, then KL distillation
    python linswap.py distill --kernel mamba2 --stages kl --kl_steps 500

Teacher: the original model (``gdn`` kernel, exact copy of the HF backbone).
Student: the swapped model.  Two stages, both on the SFT corpus truncated to
``--max_length`` tokens (all positions, not only assistant tokens):

  layer  every linear-attention layer of the student is fed the *teacher's*
         input to that layer and trained (MSE) to reproduce the teacher layer's
         output; only the linear layers' parameters are updated.  Cheap and
         stable — this is the usual first step of attention-linearisation
         recipes (Mamba-in-Llama, LoLCATs).
  kl     end-to-end KL(teacher ‖ student) on the next-token distributions,
         computed in vocabulary chunks so the full logits are never
         materialised; trains all parameters (``--kl_train linear`` restricts it).

The final checkpoint (``outputs/<kernel>/distill/checkpoint-N``, config
``sft_mode = "distill"``) is the starting point for ``posttrain --init_ckpt``;
``run`` does this automatically for kernels registered with
``exact_init=False``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..data import DEFAULT_DATA_DIR, ensure_sft_data
from ..load_weights import DEFAULT_BASE_MODEL_DIR, REPO_ROOT, build_model
from ..registry import get_kernel
from ..sft_utils import TruncatedDataset, collate_fn, evaluate
from .posttrain import checkpoint_config, log_jsonl


def add_args(ap):
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--output_dir", default=None, help="default outputs/<kernel>/distill")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))
    ap.add_argument("--teacher", default="gdn", help="kernel used as the teacher (exact copy of the original)")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--data_max_length", type=int, default=262144)
    ap.add_argument("--max_length", type=int, default=8192, help="distillation sequence length")
    ap.add_argument("--stages", default="layer,kl")
    ap.add_argument("--layer_steps", type=int, default=200)
    ap.add_argument("--layer_lr", type=float, default=1e-4)
    ap.add_argument("--kl_steps", type=int, default=300)
    ap.add_argument("--kl_lr", type=float, default=2e-5)
    ap.add_argument("--kl_train", choices=["all", "linear"], default="all")
    ap.add_argument("--kl_temperature", type=float, default=1.0)
    ap.add_argument("--grad_accum_steps", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_batches", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ce_chunk_size", type=int, default=2048)


# ------------------------------------------------------------------- helpers
def linear_blocks(model):
    return [(i, b) for i, b in enumerate(model.trf_blocks) if b.layer_type == "linear_attention"]


def linear_params(model):
    return [p for _, b in linear_blocks(model) for p in b.token_mixer.parameters()]


@torch.no_grad()
def teacher_layer_io(teacher, ids):
    """Inputs and outputs of every linear-attention layer of the teacher."""
    ins, outs, hooks = {}, {}, []
    for i, b in linear_blocks(teacher):
        hooks.append(b.norm1.register_forward_hook(lambda m, a, o, i=i: ins.__setitem__(i, o.detach())))
        hooks.append(b.token_mixer.register_forward_hook(lambda m, a, o, i=i: outs.__setitem__(i, o[0].detach())))
    teacher(ids, return_hidden_before_norm=True)
    for h in hooks:
        h.remove()
    return ins, outs


def chunked_kl_with_backward(s_hidden, t_hidden, s_head, t_head, chunk_size=2048, temperature=1.0, loss_scale=1.0):
    """KL(teacher ‖ student) averaged over positions, back-propagated in vocabulary-chunks."""
    b, T, d = s_hidden.shape
    s_flat = s_hidden.reshape(-1, d)
    t_flat = t_hidden.reshape(-1, d)
    grad = torch.zeros_like(s_flat)
    total, n = 0.0, s_flat.shape[0]
    scale = loss_scale / n
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        with torch.no_grad():
            t_logp = F.log_softmax(F.linear(t_flat[start:end], t_head.weight).float() / temperature, dim=-1)
        h = s_flat[start:end].detach().requires_grad_(True)
        s_logp = F.log_softmax(F.linear(h, s_head.weight).float() / temperature, dim=-1)
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).sum()
        (kl * scale * temperature ** 2).backward()
        grad[start:end] = h.grad
        total += kl.item()
    s_hidden.backward(grad.view(b, T, d))
    return total / n


# --------------------------------------------------------------------- stages
def run_stage(stage, args, teacher, student, train_loader, val_loader, out_dir, log_path, step0):
    device = next(student.parameters()).device
    steps = args.layer_steps if stage == "layer" else args.kl_steps
    lr = args.layer_lr if stage == "layer" else args.kl_lr
    if stage == "layer" or args.kl_train == "linear":
        params = linear_params(student)
    else:
        params = list(student.parameters())
    ids_ = {id(p) for p in params}
    for p in student.parameters():
        p.requires_grad_(id(p) in ids_)
    student.gradient_checkpointing = stage == "kl"
    optimizer = AdamW([{"params": params, "lr": lr, "weight_decay": 0.0}])
    print(f"[distill] stage={stage} steps={steps} lr={lr} trainable={sum(p.numel() for p in params)/1e6:.1f}M")

    train_iter = iter(train_loader)
    start = time.time()
    step = 0
    while step < steps:
        acc = 0.0
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            ids = batch["input_ids"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if stage == "layer":
                    t_in, t_out = teacher_layer_io(teacher, ids)
                    loss = 0.0
                    for i, b in linear_blocks(student):
                        o, _, _ = b.token_mixer(t_in[i])
                        loss = loss + F.mse_loss(o.float(), t_out[i].float())
                    (loss / args.grad_accum_steps).backward()
                    acc += loss.item()
                else:
                    with torch.no_grad():
                        t_hidden = teacher(ids, return_hidden=True)
                    s_hidden = student(ids, return_hidden=True)
                    acc += chunked_kl_with_backward(s_hidden, t_hidden, student.out_head, teacher.out_head,
                                                    args.ce_chunk_size, args.kl_temperature,
                                                    1.0 / args.grad_accum_steps)
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        rec = {"stage": stage, "step": step0 + step, "loss": acc / args.grad_accum_steps, "grad_norm": float(gn),
               "elapsed_s": round(time.time() - start, 1)}
        print(f"  {stage} step {step}/{steps}: loss={rec['loss']:.4f} grad_norm={rec['grad_norm']:.3f} "
              f"elapsed={rec['elapsed_s']/60:.1f}min", flush=True)
        log_jsonl(log_path, rec)
        last = step == steps
        if step % args.eval_every == 0 or last:
            student.eval()
            val = evaluate(student, val_loader, max_batches=args.eval_batches, chunk_size=args.ce_chunk_size)
            student.train()
            print(f"  {stage} step {step}: val_loss={val:.4f}", flush=True)
            log_jsonl(log_path, {"stage": stage, "step": step0 + step, "val_loss": val})
        if step % args.save_every == 0 or last:
            save(student, args, out_dir, step0 + step)
    return step0 + step


def save(student, args, out_dir, step):
    ckpt = Path(out_dir) / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), ckpt / "model.pt")
    with open(ckpt / "config.json", "w") as f:
        json.dump(checkpoint_config(student, args, "distill"), f, indent=1)
    print(f"  saved {ckpt}", flush=True)
    return ckpt


def main(args) -> Path:
    spec = get_kernel(args.kernel)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out_dir = (Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / args.kernel / "distill").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=1)
    log_path = out_dir / "train_log.jsonl"
    print(f"[distill] kernel={spec.name} (exact_init={spec.exact_init}) teacher={args.teacher} -> {out_dir}")

    data_dir = ensure_sft_data(args.base_model_dir, args.data_max_length, args.data_dir)
    train_ds = TruncatedDataset(load_from_disk(data_dir / "train"), args.max_length)
    val_ds = TruncatedDataset(load_from_disk(data_dir / "validation"), args.max_length)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, generator=g)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    teacher = build_model(args.teacher, base_model_dir=args.base_model_dir, device=device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = build_model(args.kernel, base_model_dir=args.base_model_dir, device=device).train()

    val0 = evaluate(student, val_loader, max_batches=args.eval_batches, chunk_size=args.ce_chunk_size)
    print(f"  step 0: val_loss={val0:.4f}")
    log_jsonl(log_path, {"stage": "init", "step": 0, "val_loss": val0})

    step = 0
    for stage in [s for s in args.stages.split(",") if s]:
        if stage not in ("layer", "kl"):
            raise SystemExit(f"distill: unknown stage {stage!r}")
        step = run_stage(stage, args, teacher, student, train_loader, val_loader, out_dir, log_path, step)
    ckpt = save(student, args, out_dir, step)
    print(f"[distill] done: {ckpt}")
    del teacher, student
    torch.cuda.empty_cache()
    return ckpt
