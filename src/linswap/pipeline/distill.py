"""Stage 2 — distill: train the swapped model to reproduce the original one.

    linswap distill --kernel rwkv7                 # the standard three-step recipe below
    linswap distill --kernel mamba2 --stages layer # first step only (cheap sanity run)

Teacher: the original model (``gdn`` kernel, an exact copy of the HF backbone).  Student: the swapped
model.  Three steps on packed generic web text (``--text_data``, default DCLM), no instruction data and
no chat template anywhere:

  layer   every linear-attention layer of the student is fed the *teacher's* input to that layer and
          trained (L2) to reproduce the teacher layer's output; all layers at once, one optimizer, only
          the swapped layers' parameters.  100M tokens at length 512, lr 1e-3 decayed to 1e-5 (cosine).
  kl      end-to-end KL(teacher ‖ student) on the next-token distributions, computed in vocabulary
          chunks so the full logits are never materialised.  500M tokens at length 512, flat lr 1e-5,
          *all* parameters (freezing the MLPs or the embeddings costs accuracy).
  ce      context-length extension: plain next-token cross-entropy without the teacher, 100M tokens at
          length 16384, flat lr 1e-5, all parameters.

Budgets are given in tokens (``--<step>_tokens``) or in optimizer steps (``--<step>_steps``); the
sequence length, sequences per step and schedule of each step have their own defaults (see ``add_args``)
and can be overridden with ``--stage_length`` / ``--stage_batch`` / ``--stage_micro`` / ``--stage_schedule``.

The final checkpoint (``outputs/<kernel>/distill/checkpoint-N``) is what ``evaluate`` and ``lmeval``
score; no supervised fine-tuning follows.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..load_weights import DEFAULT_BASE_MODEL_DIR, REPO_ROOT, build_model
from ..registry import get_kernel
from ..textdata import DEFAULT_TEXT_DIR, ensure_text_data
from ..train_utils import PackedDataset, chunked_cross_entropy_with_backward, collate_fn, evaluate

STAGES = ("layer", "kl", "ce")
# per-step defaults: sequence length, sequences per optimizer step, sequences per micro-batch, lr schedule
STAGE_DEFAULTS = {
    "layer": dict(length=512, batch=32, micro=32, schedule="cosine"),
    "kl": dict(length=512, batch=96, micro=96, schedule="flat"),
    "ce": dict(length=16384, batch=96, micro=8, schedule="flat"),
}


def add_args(ap):
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--output_dir", default=None, help="default outputs/<kernel>/distill")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))
    ap.add_argument("--teacher", default="gdn", help="kernel used as the teacher (exact copy of the original)")
    ap.add_argument("--text_data", default="dclm", help="generic-text corpus (see linswap.textdata.CORPORA)")
    ap.add_argument("--text_shards", type=int, default=10, help="corpus shards to tokenise (~110M tokens each)")
    ap.add_argument("--text_dir", default=str(DEFAULT_TEXT_DIR))
    ap.add_argument("--stages", default=",".join(STAGES), help="comma list of layer / kl / ce")
    ap.add_argument("--layer_tokens", type=float, default=100e6)
    ap.add_argument("--layer_lr", type=float, default=1e-3)
    ap.add_argument("--layer_steps", type=int, default=None, help="overrides --layer_tokens")
    ap.add_argument("--kl_tokens", type=float, default=500e6)
    ap.add_argument("--kl_lr", type=float, default=1e-5)
    ap.add_argument("--kl_steps", type=int, default=None)
    ap.add_argument("--kl_temperature", type=float, default=1.0)
    ap.add_argument("--ce_tokens", type=float, default=100e6)
    ap.add_argument("--ce_lr", type=float, default=1e-5)
    ap.add_argument("--ce_steps", type=int, default=None)
    ap.add_argument("--lr_final", type=float, default=1e-5, help="final lr of cosine schedules")
    ap.add_argument("--stage_length", default=None, help="per-step sequence length, e.g. layer:512,ce:16384")
    ap.add_argument("--stage_batch", default=None, help="per-step sequences per optimizer step, e.g. kl:96")
    ap.add_argument("--stage_micro", default=None, help="per-step sequences per micro-batch (accumulation = batch/micro)")
    ap.add_argument("--stage_schedule", default=None, help="per-step lr schedule 'cosine' or 'flat'")
    ap.add_argument("--adam_betas", default="0.9,0.95")
    ap.add_argument("--adam_eps", type=float, default=1e-8)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--eval_batches", type=int, default=10, help="held-out packed sequences for the validation loss")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--val_length", type=int, default=8192, help="length of the held-out packed sequences")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ce_chunk_size", type=int, default=2048)
    return ap


# ------------------------------------------------------------------ helpers
def linear_blocks(model):
    return [(i, b) for i, b in enumerate(model.layers) if b.layer_type == "linear_attention"]


def linear_params(model):
    return [p for _, b in linear_blocks(model) for p in b.linear_attn.parameters()]


def checkpoint_config(model, args, stage="distill"):
    cfg = {k: (str(v) if k == "dtype" else v) for k, v in model.cfg.items()}
    cfg.update({
        "linear_kernel": model.kernel_name,
        "base_model_dir": str(args.base_model_dir),
        "sft_mode": stage,                      # kept for checkpoints written before this field was renamed
        "new_param_names": list(model.new_param_names),
    })
    return cfg


def log_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


@torch.no_grad()
def teacher_layer_io(teacher, ids):
    """Inputs and outputs of every linear-attention layer of the teacher."""
    ins, outs, hooks = {}, {}, []
    for i, b in linear_blocks(teacher):
        hooks.append(b.input_layernorm.register_forward_hook(lambda m, a, o, i=i: ins.__setitem__(i, o.detach())))
        hooks.append(b.linear_attn.register_forward_hook(lambda m, a, o, i=i: outs.__setitem__(i, o[0].detach())))
    teacher(ids, return_hidden_before_norm=True)
    for h in hooks:
        h.remove()
    return ins, outs


def chunked_kl_with_backward(s_hidden, t_hidden, s_head, t_head, chunk_size=2048, temperature=1.0, loss_scale=1.0):
    """KL(teacher ‖ student) averaged over positions, back-propagated in vocabulary-chunks."""
    b, T, d = s_hidden.shape
    s_flat, t_flat = s_hidden.reshape(-1, d), t_hidden.reshape(-1, d)
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


def _stage_map(spec):
    """'layer:512,kl:96' -> {'layer': '512', 'kl': '96'}"""
    return {k: v for k, v in (kv.split(":") for kv in (spec or "").split(",") if kv)}


def stage_plan(args, stage):
    """(sequence length, micro-batch, accumulation, schedule) for a step."""
    d = STAGE_DEFAULTS[stage]
    length = int(_stage_map(args.stage_length).get(stage, d["length"]))
    batch = int(_stage_map(args.stage_batch).get(stage, d["batch"]))
    micro = min(int(_stage_map(args.stage_micro).get(stage, d["micro"])), batch)
    schedule = _stage_map(args.stage_schedule).get(stage, d["schedule"])
    return length, micro, max(1, batch // micro), schedule


def stage_steps(args, stage, length, micro, accum):
    steps = getattr(args, f"{stage}_steps")
    if steps:
        return steps
    return max(1, int(float(getattr(args, f"{stage}_tokens")) // (micro * accum * length)))


# --------------------------------------------------------------------- steps
def run_stage(stage, args, teacher, student, train_loader, val_loader, out_dir, log_path, step0, steps,
              seq_length, micro, accum, schedule):
    device = next(student.parameters()).device
    lr = getattr(args, f"{stage}_lr")
    params = linear_params(student) if stage == "layer" else list(student.parameters())
    ids_ = {id(p) for p in params}
    for p in student.parameters():
        p.requires_grad_(id(p) in ids_)
    student.gradient_checkpointing = stage in ("kl", "ce") and student.supports_activation_checkpointing
    betas = tuple(float(b) for b in args.adam_betas.split(","))
    optimizer = AdamW([{"params": params, "lr": lr, "weight_decay": 0.0}], betas=betas, eps=args.adam_eps)
    tag = stage if stage != "ce" else f"ce@{seq_length}"
    tok_per_step = micro * accum * seq_length
    print(f"[distill] step={tag} steps={steps} lr={lr} ({schedule}) betas={betas} micro={micro} accum={accum} "
          f"trainable={sum(p.numel() for p in params)/1e6:.1f}M tokens/step={tok_per_step/1e3:.0f}K "
          f"total={steps*tok_per_step/1e6:.0f}M", flush=True)

    train_iter = iter(train_loader)
    start, step = time.time(), 0
    while step < steps:
        acc = 0.0
        for _ in range(accum):
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
                        o, _, _ = b.linear_attn(t_in[i])
                        loss = loss + F.mse_loss(o.float(), t_out[i].float())
                    (loss / accum).backward()
                    acc += loss.item()
                elif stage == "kl":
                    with torch.no_grad():
                        t_hidden = teacher(ids, return_hidden=True)
                    s_hidden = student(ids, return_hidden=True)
                    acc += chunked_kl_with_backward(s_hidden, t_hidden, student.lm_head, teacher.lm_head,
                                                    args.ce_chunk_size, args.kl_temperature, 1.0 / accum)
                else:  # ce
                    s_hidden = student(ids, return_hidden=True)
                    acc += chunked_cross_entropy_with_backward(s_hidden, batch["labels"].to(device), student.lm_head,
                                                               chunk_size=args.ce_chunk_size, loss_scale=1.0 / accum)
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if schedule == "cosine":
            frac = step / max(steps - 1, 1)
            for g in optimizer.param_groups:
                g["lr"] = args.lr_final + 0.5 * (lr - args.lr_final) * (1 + math.cos(math.pi * frac))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        rec = {"stage": tag, "step": step0 + step, "loss": acc / accum, "grad_norm": float(gn),
               "lr": optimizer.param_groups[0]["lr"], "elapsed_s": round(time.time() - start, 1),
               "tokens": (step0 + step) * tok_per_step}
        print(f"  {tag} step {step}/{steps}: loss={rec['loss']:.4f} grad_norm={rec['grad_norm']:.3f} "
              f"elapsed={rec['elapsed_s']/60:.1f}min", flush=True)
        log_jsonl(log_path, rec)
        last = step == steps
        if step % args.eval_every == 0 or last:
            student.eval()
            val = evaluate(student, val_loader, max_batches=args.eval_batches, chunk_size=args.ce_chunk_size)
            student.train()
            print(f"  {tag} step {step}: val_loss={val:.4f}", flush=True)
            log_jsonl(log_path, {"stage": tag, "step": step0 + step, "val_loss": val})
        if step % args.save_every == 0 or last:
            save(student, args, out_dir, step0 + step)
    return step0 + step


def save(student, args, out_dir, step):
    ckpt = Path(out_dir) / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), ckpt / "model.pt")
    with open(ckpt / "config.json", "w") as f:
        json.dump(checkpoint_config(student, args), f, indent=1)
    print(f"  saved {ckpt}", flush=True)
    return ckpt


def packed_loader(raw, length, batch_size, seed):
    return DataLoader(PackedDataset(raw, length, seed=seed), batch_size=batch_size, collate_fn=collate_fn)


def fixed_packed_val(raw, length, n):
    """The first ``n`` packed sequences of the held-out documents, materialised for a stable validation loss."""
    it = iter(PackedDataset(raw, length, seed=0))
    return DataLoader([next(it) for _ in range(n)], batch_size=1, shuffle=False, collate_fn=collate_fn)


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

    stages = [s for s in args.stages.split(",") if s]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"distill: unknown step {s!r} (choose from {STAGES})")

    text_dir = ensure_text_data(args.base_model_dir, args.text_data, args.text_shards, args.text_dir)
    raw_train = load_from_disk(str(text_dir / "train"))
    val_loader = fixed_packed_val(load_from_disk(str(text_dir / "validation")), args.val_length, args.eval_batches)
    print(f"[distill] corpus {text_dir} ({len(raw_train)} docs); validation = {args.eval_batches} packed "
          f"sequences of {args.val_length}")

    teacher = build_model(args.teacher, base_model_dir=args.base_model_dir, device=device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = build_model(args.kernel, base_model_dir=args.base_model_dir, device=device).train()

    val_t = evaluate(teacher, val_loader, max_batches=args.eval_batches, chunk_size=args.ce_chunk_size)
    val0 = evaluate(student, val_loader, max_batches=args.eval_batches, chunk_size=args.ce_chunk_size)
    print(f"  step 0: val_loss={val0:.4f} (teacher {val_t:.4f})")
    log_jsonl(log_path, {"stage": "init", "step": 0, "val_loss": val0, "teacher_val_loss": val_t})

    step = 0
    for stage in stages:
        length, micro, accum, schedule = stage_plan(args, stage)
        loader = packed_loader(raw_train, length, micro, args.seed + step)
        steps = stage_steps(args, stage, length, micro, accum)
        step = run_stage(stage, args, teacher, student, loader, val_loader, out_dir, log_path, step, steps,
                         length, micro, accum, schedule)
    ckpt = save(student, args, out_dir, step)
    print(f"[distill] done: {ckpt}")
    del teacher, student
    torch.cuda.empty_cache()
    return ckpt
