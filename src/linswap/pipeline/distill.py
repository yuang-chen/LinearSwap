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

Every run writes ``train_log.jsonl`` and a TensorBoard run (``<output_dir>/tensorboard``, see
``linswap.tb_logger``): loss, grad norm (total and per parameter group: the kernel's new parameters,
the swapped layers' shared projections, the rest of the backbone), lr, in the ``layer`` and ``kl``
steps the per-layer ``layer_mse`` (MSE between each student linear-attention layer's output and the
teacher's, each model running on its own hidden states, so it includes error carried in from earlier
layers), and every ``--eval_every`` steps the validation loss, KL(teacher ‖ student) / top-1
agreement / entropies on the held-out sequences and the groups' parameter norm, drift from the swap
init and update size.  ``tensorboard --logdir outputs`` overlays the kernels.
"""

from __future__ import annotations

import contextlib
import itertools
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
from ..train_utils import PackedDataset, chunked_cross_entropy_with_backward, collate_fn

from ..tb_logger import (TBLogger, drift_from_init, evaluate_vs_teacher, group_grad_norms, group_param_norms,
                         init_state, param_groups, snapshot, teacher_reference, update_ratio)

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
    ap.add_argument("--layer_skip_below", type=float, default=1e-4,
                    help="skip the layer step when its loss on the first batch is already below this "
                         "(a function-preserving swap has nothing to align); 0 always runs it")
    ap.add_argument("--init_from", default=None,
                    help="resume from this checkpoint dir: finished steps are skipped and a partial one continues "
                         "on the same data (optimizer state restarts)")
    ap.add_argument("--init_step", type=int, default=None,
                    help="global step of --init_from (default: parsed from its checkpoint-N name)")
    ap.add_argument("--tensorboard_dir", default=None, help="default <output_dir>/tensorboard")
    ap.add_argument("--no_tensorboard", action="store_true")
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


@torch.no_grad()
def layer_alignment_loss(teacher, student, ids):
    """The layer step's own objective on one batch, without training: how far the swapped mixers sit
    from the teacher's on the teacher's own input."""
    student.eval()
    t_in, t_out, _ = teacher_layer_io(teacher, ids, return_hidden_before_norm=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = sum(F.mse_loss(b.linear_attn(t_in[i])[0].float(), t_out[i].float())
                   for i, b in linear_blocks(student))
    student.train()
    return float(loss)


def log_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


@torch.no_grad()
def teacher_layer_io(teacher, ids, inputs=True, **forward_kw):
    """Inputs (if ``inputs``) and outputs of every linear-attention layer of the teacher, and what
    ``teacher(ids, **forward_kw)`` returns."""
    ins, outs, hooks = {}, {}, []
    for i, b in linear_blocks(teacher):
        if inputs:
            hooks.append(b.input_layernorm.register_forward_hook(lambda m, a, o, i=i: ins.__setitem__(i, o.detach())))
        hooks.append(b.linear_attn.register_forward_hook(lambda m, a, o, i=i: outs.__setitem__(i, o[0].detach())))
    ret = teacher(ids, **forward_kw)
    for h in hooks:
        h.remove()
    return ins, outs, ret


@contextlib.contextmanager
def record_layer_mse(model, t_out, acc, scale=1.0):
    """While ``model`` runs on its own hidden states, add ``scale`` × MSE(linear-attention output of layer
    i, ``t_out[i]``) to ``acc[i]`` (a tensor, so nothing synchronises).  Only the first call of a layer
    counts: activation checkpointing re-runs the blocks in the backward."""
    seen, hooks = set(), []

    def hook(m, a, o, i):
        if i in seen:
            return
        seen.add(i)
        with torch.no_grad():
            mse = F.mse_loss(o[0].float(), t_out[i].float()) * scale
        acc[i] = acc[i] + mse if i in acc else mse

    for i, b in linear_blocks(model):
        hooks.append(b.linear_attn.register_forward_hook(lambda m, a, o, i=i: hook(m, a, o, i)))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


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
def run_stage(stage, args, teacher, student, train_loader, teacher_ref, out_dir, log_path, step0, steps,
              seq_length, micro, accum, schedule, done=0, tb=None, init_sd=None):
    device = next(student.parameters()).device
    lr = getattr(args, f"{stage}_lr")
    params = linear_params(student) if stage == "layer" else list(student.parameters())
    groups = param_groups(student, params)
    tb = tb or TBLogger(None, enabled=False)
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
          f"total={steps*tok_per_step/1e6:.0f}M" + (f" (resuming after {done})" if done else ""), flush=True)
    tb.text("stage", f"step {step0 + done}: {tag} ({steps} steps, lr {lr} {schedule}, {seq_length} x {micro} x {accum})",
            step0 + done)

    train_iter = iter(train_loader)
    start, step = time.time(), done
    while step < steps:
        acc, mse_acc, t_step = 0.0, {}, time.time()
        for _ in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            ids = batch["input_ids"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if stage == "layer":
                    t_in, t_out, _ = teacher_layer_io(teacher, ids, return_hidden_before_norm=True)
                    loss = 0.0
                    for i, b in linear_blocks(student):
                        o, _, _ = b.linear_attn(t_in[i])
                        loss = loss + F.mse_loss(o.float(), t_out[i].float())
                    (loss / accum).backward()
                    acc += loss.item()
                    del t_in
                    with torch.no_grad(), record_layer_mse(student, t_out, mse_acc, 1.0 / accum):
                        student(ids, return_hidden_before_norm=True)
                    del t_out
                elif stage == "kl":
                    _, t_out, t_hidden = teacher_layer_io(teacher, ids, inputs=False, return_hidden=True)
                    with record_layer_mse(student, t_out, mse_acc, 1.0 / accum):
                        s_hidden = student(ids, return_hidden=True)
                    del t_out
                    acc += chunked_kl_with_backward(s_hidden, t_hidden, student.lm_head, teacher.lm_head,
                                                    args.ce_chunk_size, args.kl_temperature, 1.0 / accum)
                else:  # ce
                    s_hidden = student(ids, return_hidden=True)
                    acc += chunked_cross_entropy_with_backward(s_hidden, batch["labels"].to(device), student.lm_head,
                                                               chunk_size=args.ce_chunk_size, loss_scale=1.0 / accum)
        group_gn = group_grad_norms(groups)                    # before clipping, like ``gn``
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if schedule == "cosine":
            frac = step / max(steps - 1, 1)
            for g in optimizer.param_groups:
                g["lr"] = args.lr_final + 0.5 * (lr - args.lr_final) * (1 + math.cos(math.pi * frac))
        last = step + 1 == steps
        diag = (step + 1) % args.eval_every == 0 or last
        before = snapshot(groups) if diag else None
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        rec = {"stage": tag, "step": step0 + step, "loss": acc / accum, "grad_norm": float(gn),
               "lr": optimizer.param_groups[0]["lr"], "elapsed_s": round(time.time() - start, 1),
               "tokens": (step0 + step) * tok_per_step}
        rec.update({f"grad_norm_{g}": v for g, v in group_gn.items()})
        layer_mse = {i: float(v) for i, v in sorted(mse_acc.items())}
        if layer_mse:
            rec["layer_mse"] = {str(i): v for i, v in layer_mse.items()}
        print(f"  {tag} step {step}/{steps}: loss={rec['loss']:.4f} grad_norm={rec['grad_norm']:.3f} "
              f"elapsed={rec['elapsed_s']/60:.1f}min", flush=True)
        log_jsonl(log_path, rec)
        gstep = step0 + step
        tb.scalars({"loss": rec["loss"], "grad_norm": rec["grad_norm"], "lr": rec["lr"], "tokens": rec["tokens"],
                    "step_time_s": time.time() - t_step, "stage": STAGES.index(stage)}, gstep, "train/")
        tb.scalars(group_gn, gstep, "grad_norm/")
        tb.scalars({f"layer_{i}": v for i, v in layer_mse.items()}, gstep, "layer_mse/")
        if diag:
            stats = evaluate_vs_teacher(student, teacher_ref, args.ce_chunk_size)
            val = stats["val_loss"]
            print(f"  {tag} step {step}: val_loss={val:.4f}", flush=True)
            log_jsonl(log_path, {"stage": tag, "step": step0 + step, "val_loss": val})
            pstats = {"update_ratio": update_ratio(groups, before), "norm": group_param_norms(groups),
                      "drift_from_init": drift_from_init(groups, init_sd) if init_sd else {}}
            del before
            log_diagnostics(tb, log_path, tag, gstep, stats, pstats)
        if step % args.save_every == 0 or last:
            save(student, args, out_dir, step0 + step)
    return step0 + step


def log_diagnostics(tb, log_path, stage, step, stats, pstats):
    """``stats``: validation / output-distribution scalars (``val_loss`` -> ``val/loss``);
    ``pstats``: ``{kind: {group: value}}`` parameter-group scalars -> ``params/<kind>/<group>``."""
    tb.scalars({("loss" if k == "val_loss" else k): v for k, v in stats.items()}, step, "val/")
    for kind, per_group in pstats.items():
        tb.scalars(per_group, step, f"params/{kind}/")
    rec = {"stage": stage, "step": step, **{k: v for k, v in stats.items() if k != "val_loss"}}
    rec.update({f"{kind}_{g}": v for kind, per_group in pstats.items() for g, v in per_group.items()})
    log_jsonl(log_path, rec)
    print(f"  {stage} step {step}: kl={stats.get('kl_teacher_student', float('nan')):.4f} "
          f"top1={stats.get('top1_agreement', float('nan')):.3f}", flush=True)


def save(student, args, out_dir, step):
    ckpt = Path(out_dir) / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), ckpt / "model.pt")
    with open(ckpt / "config.json", "w") as f:
        json.dump(checkpoint_config(student, args), f, indent=1)
    print(f"  saved {ckpt}", flush=True)
    return ckpt


class _Skip(torch.utils.data.IterableDataset):
    def __init__(self, ds, n):
        self.ds, self.n = ds, n

    def __iter__(self):
        return itertools.islice(iter(self.ds), self.n, None)


def packed_loader(raw, length, batch_size, seed, skip=0):
    """``skip`` sequences are dropped from the front, so a resumed step sees the same data as the original run."""
    ds = PackedDataset(raw, length, seed=seed)
    return DataLoader(_Skip(ds, skip) if skip else ds, batch_size=batch_size, collate_fn=collate_fn)


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
    step = 0
    if args.init_from:
        step = args.init_step if args.init_step is not None else int(Path(args.init_from).name.split("-")[-1])
    with open(out_dir / ("args.json" if not args.init_from else f"args_resume{step}.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    log_path = out_dir / "train_log.jsonl"
    tb = TBLogger(args.tensorboard_dir or out_dir / "tensorboard", enabled=not args.no_tensorboard,
                  purge_step=step if args.init_from else None)
    tb.args(args, step)
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
    student = build_model(args.kernel, base_model_dir=args.base_model_dir, ckpt_dir=args.init_from,
                          device=device).train()
    # Drift is measured from the swap init, so a resumed run rebuilds it (on CPU) instead of taking the checkpoint.
    init_sd = init_state(build_model(args.kernel, base_model_dir=args.base_model_dir, device="cpu")
                         if args.init_from else student)

    teacher_ref = teacher_reference(teacher, val_loader, args.eval_batches, args.ce_chunk_size)
    stats0 = evaluate_vs_teacher(student, teacher_ref, args.ce_chunk_size)
    val_t, val0 = teacher_ref["loss"], stats0["val_loss"]
    print(f"  step {step}: val_loss={val0:.4f} (teacher {val_t:.4f})" + (f" [from {args.init_from}]" if args.init_from else ""))
    log_jsonl(log_path, {"stage": "init" if not args.init_from else "resume", "step": step, "val_loss": val0,
                         "teacher_val_loss": val_t})
    groups0 = param_groups(student, list(student.parameters()))
    log_diagnostics(tb, log_path, "init" if not args.init_from else "resume", step,
                    {**stats0, "teacher_loss": val_t},
                    {"norm": group_param_norms(groups0), "drift_from_init": drift_from_init(groups0, init_sd)})

    resume, step = step, 0
    for stage in stages:
        length, micro, accum, schedule = stage_plan(args, stage)
        steps = stage_steps(args, stage, length, micro, accum)
        done = min(max(resume - step, 0), steps)
        if done == steps:
            print(f"[distill] step={stage}: already done in {args.init_from}")
            step += steps
            continue
        if done:
            print(f"[distill] step={stage}: fast-forwarding the data past {done} steps", flush=True)
        loader = packed_loader(raw_train, length, micro, args.seed + step, skip=done * micro * accum)
        if stage == "layer" and not done and args.layer_skip_below > 0:
            # An exact init starts this step at bf16 noise (~1e-6) with nothing to learn, and Adam's
            # normalised update at lr 1e-3 then walks the weights away from a function-preserving
            # solution: the objective rises, and the model does not recover it in the later steps.
            probe = layer_alignment_loss(teacher, student, next(iter(loader))["input_ids"].to(device))
            if probe < args.layer_skip_below:
                print(f"[distill] step=layer: skipped (initial loss {probe:.2e} < {args.layer_skip_below:.0e}, "
                      f"the swap already reproduces the teacher's layers)", flush=True)
                log_jsonl(log_path, {"stage": "layer", "step": step, "skipped": True, "initial_loss": probe})
                step += steps          # keep the step counter, data order and checkpoint numbering unchanged
                continue
            print(f"[distill] step=layer: initial loss {probe:.2e} >= {args.layer_skip_below:.0e}, running it",
                  flush=True)
        step = run_stage(stage, args, teacher, student, loader, teacher_ref, out_dir, log_path, step, steps,
                         length, micro, accum, schedule, done, tb=tb, init_sd=init_sd)
    ckpt = save(student, args, out_dir, step)
    tb.close()
    print(f"[distill] done: {ckpt}")
    del teacher, student
    torch.cuda.empty_cache()
    return ckpt
