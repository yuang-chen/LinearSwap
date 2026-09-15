"""Stage 1½ — distill: recover the pretrained function for an *inexact* swap.

    linswap distill --kernel deltanet                 # layer alignment, then KL distillation (SFT corpus)
    linswap distill --kernel mamba2 --stages kl --kl_steps 500
    linswap distill --kernel mamba2 --kl_schedule 8192:200,65536:100     # long-context curriculum (packed)
    linswap distill --kernel mamba2 --text_data fineweb-edu --stages layer,hidden,kl,ce \\
        --max_length 2048 --batch_size 16 --layer_tokens 50e6 --hidden_tokens 50e6 --kl_tokens 300e6 \\
        --ce_length 16384 --ce_tokens 100e6                        # literature recipe on generic text

Teacher: the original model (``gdn`` kernel, exact copy of the HF backbone).
Student: the swapped model.  Stages (``--stages``), each on packed fixed-length sequences
when ``--text_data`` names a generic-text corpus (``linswap.textdata``), otherwise on the
SFT corpus truncated to ``--max_length`` tokens (all positions supervised):

  layer   every linear-attention layer of the student is fed the *teacher's* input to that
          layer and trained (MSE) to reproduce the teacher layer's output; only the linear
          layers' parameters are updated (token-mixer alignment: RADLADS step 1, HALO stage 1,
          Distill-then-Replace's blockwise local distillation).
  hidden  end-to-end forward; per-layer L2 between the student's and the teacher's residual
          stream after every linear-attention block, normalised by the teacher's variance
          (hidden-state alignment: "What matters in linearizing LMs" stage 2, HyLo's ILD).
          Trains the linear layers (``--hidden_train all`` trains everything).
  kl      end-to-end KL(teacher ‖ student) on the next-token distributions, computed in
          vocabulary chunks so the full logits are never materialised, optionally plus
          ``--kl_ce_weight`` × cross-entropy on the tokens; trains all parameters
          (``--kl_train linear`` restricts it).
  ce      plain next-token cross-entropy without the teacher at ``--ce_length`` (long-context
          fine-tuning: RADLADS step 3, HALO stage 3, HydraHead stage 3).

Budgets are given in optimizer steps (``--<stage>_steps``) or in tokens (``--<stage>_tokens``,
which override steps as tokens / (batch × accumulation × length)).

The final checkpoint (``outputs/<kernel>/distill/checkpoint-N``, config ``sft_mode = "distill"``)
is the starting point for ``posttrain --init_ckpt``; ``run`` does this automatically for
kernels registered with ``exact_init=False``.
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
from ..sft_utils import PackedDataset, TruncatedDataset, chunked_cross_entropy_with_backward, collate_fn, evaluate
from ..textdata import DEFAULT_TEXT_DIR, ensure_text_data
from .posttrain import checkpoint_config, log_jsonl

STAGES = ("layer", "hidden", "kl", "ce")


def add_args(ap):
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--output_dir", default=None, help="default outputs/<kernel>/distill")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))
    ap.add_argument("--teacher", default="gdn", help="kernel used as the teacher (exact copy of the original)")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--datasets", default="all", help="SFT mixture subset, e.g. longalign,longalpaca (ablation: no anti-haystack)")
    ap.add_argument("--data_max_length", type=int, default=262144)
    ap.add_argument("--text_data", default=None, help="generic-text corpus instead of the SFT mixture, e.g. fineweb-edu")
    ap.add_argument("--text_shards", type=int, default=1, help="number of corpus shards to tokenise (~700M tokens each)")
    ap.add_argument("--text_dir", default=str(DEFAULT_TEXT_DIR))
    ap.add_argument("--text_mix", type=float, default=0.0,
                    help="fraction of documents drawn from the SFT mixture (chat format, all tokens supervised) when packing "
                         "generic text, so long generic-text training does not erase the instruction format (replay)")
    ap.add_argument("--max_length", type=int, default=8192, help="sequence length for layer / hidden / kl")
    ap.add_argument("--batch_size", type=int, default=1, help="sequences per micro-batch")
    ap.add_argument("--stages", default="layer,kl", help="comma list of layer / hidden / kl / ce")
    ap.add_argument("--layer_steps", type=int, default=200)
    ap.add_argument("--layer_lr", type=float, default=1e-4)
    ap.add_argument("--layer_tokens", type=float, default=None)
    ap.add_argument("--hidden_steps", type=int, default=200)
    ap.add_argument("--hidden_lr", type=float, default=1e-4)
    ap.add_argument("--hidden_tokens", type=float, default=None)
    ap.add_argument("--hidden_train", choices=["all", "linear"], default="linear")
    ap.add_argument("--kl_steps", type=int, default=300)
    ap.add_argument("--kl_schedule", default=None,
                    help="packed long-context KL curriculum 'len:steps,len:steps' (e.g. 8192:200,65536:100); "
                         "replaces --max_length/--kl_steps for the kl stage")
    ap.add_argument("--kl_lr", type=float, default=2e-5)
    ap.add_argument("--kl_tokens", type=float, default=None)
    ap.add_argument("--kl_train", choices=["all", "linear"], default="all")
    ap.add_argument("--kl_temperature", type=float, default=1.0)
    ap.add_argument("--kl_ce_weight", type=float, default=0.0, help="add this × token cross-entropy to the KL loss")
    ap.add_argument("--ce_steps", type=int, default=100)
    ap.add_argument("--ce_lr", type=float, default=1e-5)
    ap.add_argument("--ce_tokens", type=float, default=None)
    ap.add_argument("--ce_length", type=int, default=16384, help="sequence length of the ce stage")
    ap.add_argument("--ce_batch_size", type=int, default=None, help="default: max(1, batch_size*max_length//ce_length)")
    ap.add_argument("--ce_schedule", default=None,
                    help="staged long-context CE 'len:tokens,len:tokens' (e.g. 8192:300e6,16384:100e6); replaces --ce_length/--ce_tokens")
    ap.add_argument("--grad_accum_steps", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_batches", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ce_chunk_size", type=int, default=2048)
    return ap


# ------------------------------------------------------------------ helpers
def linear_blocks(model):
    return [(i, b) for i, b in enumerate(model.layers) if b.layer_type == "linear_attention"]


def linear_params(model):
    return [p for _, b in linear_blocks(model) for p in b.linear_attn.parameters()]


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


def block_outputs(model, ids, detach):
    """Residual stream after every linear-attention block (forward hooks on the blocks)."""
    outs, hooks = {}, []
    for i, b in linear_blocks(model):
        hooks.append(b.register_forward_hook(
            lambda m, a, o, i=i: outs.__setitem__(i, o[0].detach() if detach else o[0])))
    model(ids, return_hidden_before_norm=True)
    for h in hooks:
        h.remove()
    return outs


def chunked_kl_with_backward(s_hidden, t_hidden, s_head, t_head, chunk_size=2048, temperature=1.0, loss_scale=1.0,
                             labels=None, ce_weight=0.0):
    """KL(teacher ‖ student) averaged over positions (+ ``ce_weight`` × mean token cross-entropy when
    ``labels`` is given), back-propagated in vocabulary-chunks.  Returns (mean KL, mean CE)."""
    b, T, d = s_hidden.shape
    s_flat = s_hidden.reshape(-1, d)
    t_flat = t_hidden.reshape(-1, d)
    grad = torch.zeros_like(s_flat)
    total, total_ce, n = 0.0, 0.0, s_flat.shape[0]
    scale = loss_scale / n
    lab = labels.reshape(-1) if labels is not None else None
    n_lab = int((lab != -100).sum().item()) if lab is not None else 0
    ce_scale = loss_scale * ce_weight / max(n_lab, 1)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        with torch.no_grad():
            t_logp = F.log_softmax(F.linear(t_flat[start:end], t_head.weight).float() / temperature, dim=-1)
        h = s_flat[start:end].detach().requires_grad_(True)
        s_logits = F.linear(h, s_head.weight).float()
        s_logp = F.log_softmax(s_logits / temperature, dim=-1)
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).sum()
        loss = kl * scale * temperature ** 2
        if ce_weight > 0 and lab is not None:
            ce = F.cross_entropy(s_logits, lab[start:end], ignore_index=-100, reduction="sum")
            loss = loss + ce * ce_scale
            total_ce += ce.item()
        loss.backward()
        grad[start:end] = h.grad
        total += kl.item()
    s_hidden.backward(grad.view(b, T, d))
    return total / n, total_ce / max(n_lab, 1)


def stage_steps(args, stage, length, batch_size):
    tokens = getattr(args, f"{stage}_tokens")
    if tokens:
        return max(1, int(float(tokens) // (batch_size * args.grad_accum_steps * length)))
    return getattr(args, f"{stage}_steps")


# --------------------------------------------------------------------- stages
def run_stage(stage, args, teacher, student, train_loader, val_loader, out_dir, log_path, step0, steps, tag=None,
              seq_length=None, batch_size=1):
    device = next(student.parameters()).device
    tag = tag or stage
    lr = getattr(args, f"{stage}_lr")
    train_linear = stage == "layer" or (stage == "hidden" and args.hidden_train == "linear") or \
        (stage == "kl" and args.kl_train == "linear")
    params = linear_params(student) if train_linear else list(student.parameters())
    ids_ = {id(p) for p in params}
    for p in student.parameters():
        p.requires_grad_(id(p) in ids_)
    student.gradient_checkpointing = stage in ("kl", "ce") and student.supports_activation_checkpointing
    optimizer = AdamW([{"params": params, "lr": lr, "weight_decay": 0.0}])
    tok_per_step = batch_size * args.grad_accum_steps * (seq_length or 0)
    print(f"[distill] stage={tag} steps={steps} lr={lr} trainable={sum(p.numel() for p in params)/1e6:.1f}M "
          f"tokens/step={tok_per_step/1e3:.0f}K total={steps*tok_per_step/1e6:.0f}M", flush=True)

    train_iter = iter(train_loader)
    start = time.time()
    step = 0
    while step < steps:
        acc, acc_ce = 0.0, 0.0
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
                        o, _, _ = b.linear_attn(t_in[i])
                        loss = loss + F.mse_loss(o.float(), t_out[i].float())
                    (loss / args.grad_accum_steps).backward()
                    acc += loss.item()
                elif stage == "hidden":
                    with torch.no_grad():
                        t_out = block_outputs(teacher, ids, detach=True)
                    s_out = block_outputs(student, ids, detach=False)
                    loss = 0.0
                    for i in s_out:
                        t = t_out[i].float()
                        loss = loss + F.mse_loss(s_out[i].float(), t) / (t.pow(2).mean() + 1e-6)
                    loss = loss / len(s_out)
                    (loss / args.grad_accum_steps).backward()
                    acc += loss.item()
                elif stage == "kl":
                    with torch.no_grad():
                        t_hidden = teacher(ids, return_hidden=True)
                    s_hidden = student(ids, return_hidden=True)
                    kl, ce = chunked_kl_with_backward(s_hidden, t_hidden, student.lm_head, teacher.lm_head,
                                                      args.ce_chunk_size, args.kl_temperature,
                                                      1.0 / args.grad_accum_steps,
                                                      labels=batch["labels"].to(device), ce_weight=args.kl_ce_weight)
                    acc += kl
                    acc_ce += ce
                else:  # ce
                    s_hidden = student(ids, return_hidden=True)
                    acc += chunked_cross_entropy_with_backward(s_hidden, batch["labels"].to(device), student.lm_head,
                                                               chunk_size=args.ce_chunk_size,
                                                               loss_scale=1.0 / args.grad_accum_steps)
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        rec = {"stage": tag, "step": step0 + step, "loss": acc / args.grad_accum_steps, "grad_norm": float(gn),
               "elapsed_s": round(time.time() - start, 1), "tokens": (step0 + step) * tok_per_step}
        if args.kl_ce_weight > 0 and stage == "kl":
            rec["ce"] = acc_ce / args.grad_accum_steps
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
        json.dump(checkpoint_config(student, args, "distill"), f, indent=1)
    print(f"  saved {ckpt}", flush=True)
    return ckpt


class MixedDocs:
    """Document source for ``PackedDataset``: each index maps to a document of ``primary`` or, for a fixed
    fraction of indices, of ``replay`` (round-robin), so packing interleaves the two corpora at the given ratio."""

    def __init__(self, primary, replay, fraction):
        self.primary, self.replay = primary, replay
        self.every = max(2, int(round(1.0 / fraction)))     # every k-th document comes from the replay set
        self.n = len(primary) + len(primary) // (self.every - 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i % self.every == self.every - 1:
            return {"input_ids": self.replay[(i // self.every) % len(self.replay)]["input_ids"]}
        j = i - i // self.every
        return {"input_ids": self.primary[j % len(self.primary)]["input_ids"]}


def packed_loader(raw, length, batch_size, seed):
    return DataLoader(PackedDataset(raw, length, seed=seed), batch_size=batch_size, collate_fn=collate_fn)


def fixed_packed_val(raw, length, n):
    """The first ``n`` packed sequences of the validation documents, materialised for a stable val loss."""
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
            raise SystemExit(f"distill: unknown stage {s!r} (choose from {STAGES})")

    if args.text_data:
        text_dir = ensure_text_data(args.base_model_dir, args.text_data, args.text_shards, args.text_dir)
        raw_train = load_from_disk(str(text_dir / "train"))
        raw_val = load_from_disk(str(text_dir / "validation"))
        if args.text_mix > 0:
            sft_dir = ensure_sft_data(args.base_model_dir, args.data_max_length, args.data_dir, args.datasets)
            raw_train = MixedDocs(raw_train, load_from_disk(sft_dir / "train"), args.text_mix)
            print(f"[distill] replay: {args.text_mix:.0%} of packed documents come from the SFT mixture ({sft_dir})")
        val_loader = fixed_packed_val(raw_val, args.max_length, args.eval_batches)
        print(f"[distill] text corpus {text_dir} ({len(raw_train)} docs); val = {args.eval_batches} packed "
              f"sequences of {args.max_length}")
    else:
        data_dir = ensure_sft_data(args.base_model_dir, args.data_max_length, args.data_dir, args.datasets)
        raw_train = load_from_disk(data_dir / "train")
        train_ds = TruncatedDataset(raw_train, args.max_length)
        val_ds = TruncatedDataset(load_from_disk(data_dir / "validation"), args.max_length)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

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
        if stage == "kl" and args.kl_schedule:
            for phase in [p for p in args.kl_schedule.split(",") if p]:
                length, n = (int(v) for v in phase.split(":"))
                loader = packed_loader(raw_train, length, 1, args.seed + step)
                step = run_stage("kl", args, teacher, student, loader, val_loader, out_dir, log_path, step,
                                 steps=n, tag=f"kl@{length}", seq_length=length, batch_size=1)
            continue
        if stage == "ce" and args.ce_schedule:
            for phase in [p for p in args.ce_schedule.split(",") if p]:
                length, tokens = phase.split(":")
                length = int(length)
                bs = args.ce_batch_size or max(1, args.batch_size * args.max_length // length)
                n = max(1, int(float(tokens) // (bs * args.grad_accum_steps * length)))
                loader = packed_loader(raw_train, length, bs, args.seed + step)
                step = run_stage("ce", args, teacher, student, loader, val_loader, out_dir, log_path, step, n,
                                 tag=f"ce@{length}", seq_length=length, batch_size=bs)
            continue
        if stage == "ce":
            length = args.ce_length
            bs = args.ce_batch_size or max(1, args.batch_size * args.max_length // length)
        else:
            length, bs = args.max_length, args.batch_size
        if args.text_data or stage == "ce":
            loader = packed_loader(raw_train, length, bs, args.seed + step)
        else:
            g = torch.Generator().manual_seed(args.seed + step)
            loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, generator=g)
        steps = stage_steps(args, stage, length, bs)
        step = run_stage(stage, args, teacher, student, loader, val_loader, out_dir, log_path, step, steps,
                         tag=stage if stage != "ce" else f"ce@{length}", seq_length=length, batch_size=bs)
    ckpt = save(student, args, out_dir, step)
    print(f"[distill] done: {ckpt}")
    del teacher, student
    torch.cuda.empty_cache()
    return ckpt
