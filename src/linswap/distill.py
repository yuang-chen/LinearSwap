"""distill: bring a swapped model to the teacher's function with as little compute as possible.

    linswap distill --kernel gla                      # -> outputs/distill/gla/{swap.safetensors,linswap.json}

A kernel whose init is exact (``kda``, ``gdn2``, ``rwkv7``, and ``gdn`` itself) already *is* the teacher and
is not trained: ``distill`` says so and stops.  For the others, only the swapped layers are trained — the rest of the model
is the teacher's, frozen, so nothing outside them can drift — against the teacher's activations on the
inputs it handles (``linswap.data``: web text up to 16K tokens, chat, image chats).  No labels and no
ground-truth loss: the target is the teacher, not the data.

Two phases, each ending by measurement:

  local   every swapped layer learns to reproduce the teacher layer's output from the teacher layer's
          input (18 independent regressions on one teacher forward).  No vocabulary head and no backward
          pass through the depth of the model, so it is the cheapest signal per token and runs at any length.
  loop    the student runs on its own hidden states, which are matched to the teacher's after every block
          (normalised MSE).  This removes the error that accumulates when each layer was only fitted to
          clean inputs.  The logits are not part of the loss — adding a KL on them bought nothing — so
          training never touches the vocabulary head.

A step is ``--step_tokens`` tokens of batches, each weighted by its tokens (``--weight tokens``), so the
objective is the mean loss per token of the stream's mix — a four-image chat of 2.4K tokens does not weigh as
much as a 16K-token sequence.

The gap — KL(teacher ‖ student) of the next-token distribution on the held-out set, averaged over its four
kinds (text, chat, image chats of every subset, one 32K sequence) — is measured at token milestones (0.3M,
1M, 3M, ... of each phase).  Positions holding an image token are left out of it: nothing is ever predicted
from them, so their logits are arbitrary, while what they pass on is seen at the text positions that follow.
The local phase is a warm start of at most ``--local_tokens``.  In either phase, when a milestone improves
the gap by less than ``--min_gain``, the learning rate is cut once, and the second time the phase ends; the
run stops at ``--max_tokens`` at the latest and keeps the weights of its best milestone.  The milestones grow
apart by sqrt(10), so a 100M-token run is judged six times in the loop: the rule is a safeguard against
training on a plateau, and a run that reaches the cap was not stopped by it.  The learning rate is warmed
up and then lowered linearly to zero over the phase's token budget (``--no-decay`` keeps it constant, and
then the gap of a single milestone moves by several percent with the last batches trained on).  Seeds differ
by a few percent, and the best milestone is chosen on the set it is reported on: read differences of that
size between two runs as none.

Precision.  The trained layers keep float32 master weights, because a small step on a bf16 weight rounds
away; the forward pass and the kernels' states run in bf16 as at inference.  The gap is measured with the
trained weights cast to bf16, and that cast is what ``swap.safetensors`` holds.

``train_log.jsonl`` records every step and milestone with the work done so far (tokens, GPU seconds of the
training steps, and the wall clock including the measurements); ``linswap.json`` in the output directory
carries the totals and the gap curve, ``manifest.json`` what the run was made of and what else was on its
GPU — the seconds are only GPU-seconds if nothing was.  With ``tensorboard``
installed the same values go to the output directory as scalars against training tokens — ``train/*`` per
step (loss, gradient norm, learning rate), ``gap/*`` per milestone (the gap, its four parts, top-1) — so
``tensorboard --logdir outputs/distill`` shows the runs side by side while they train.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import data
from .manifest import gpu_state, model_identity, write_manifest
from .model import DEFAULT_BASE_MODEL_DIR, REPO_ROOT, build_model, mixers, resolve, save_checkpoint
from .registry import get_kernel

PHASES = ("local", "loop")
WARMUP = 10                    # optimizer steps of linear learning-rate warm-up in each phase
TRAIN = {"mixer": "the swapped layers", "block": "the 18 blocks holding them (MLP and norms too)",
         "all": "the whole language model"}


def add_args(ap):
    ap.add_argument("--kernel", required=True, help="kernel, or a checkpoint directory to continue from")
    ap.add_argument("--output_dir", default=None, help="default: outputs/distill/<kernel>, or "
                    "outputs/distill/<backbone>/<kernel> for another backbone than the default")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))
    ap.add_argument("--phases", default="local,loop")
    ap.add_argument("--train", default="mixer", choices=tuple(TRAIN), help="what is trained: " + "; ".join(
        f"{k} = {v}" for k, v in TRAIN.items()))
    ap.add_argument("--max_tokens", type=float, default=100e6, help="cap on the training tokens of the whole run")
    ap.add_argument("--local_tokens", type=float, default=3e6, help="cap on the local phase")
    ap.add_argument("--local_lr", type=float, default=1e-3)
    ap.add_argument("--loop_lr", type=float, default=1e-4)
    ap.add_argument("--step_tokens", type=int, default=65536, help="tokens per optimizer step")
    ap.add_argument("--weight", default="tokens", choices=("tokens", "batches"), help="what a batch counts for in a "
                    "step's gradient: its tokens (the objective is the mean over the step's tokens), or one (a "
                    "four-image chat then weighs as much as a 16K-token sequence)")
    ap.add_argument("--decay", action=argparse.BooleanOptionalAction, default=True, help="lower the learning rate "
                    "linearly to zero over a phase's budget (--no-decay: constant after the warm-up)")
    ap.add_argument("--first_milestone", type=float, default=0.3e6, help="tokens into a phase at its first "
                    "measurement; each next one is sqrt(10) times further")
    ap.add_argument("--min_gain", type=float, default=0.05, help="relative improvement of the gap a milestone must bring")
    ap.add_argument("--force", action="store_true", help="train a kernel whose init is exact")
    ap.add_argument("--seed", type=int, default=0)


def nmse(got, want):
    """Squared error relative to the target's power, so every layer weighs the same."""
    got, want = got.float(), want.float()
    return (got - want).pow(2).mean() / want.pow(2).mean()


@contextlib.contextmanager
def deployed(params):
    """The trained weights as they are saved and run: cast to bf16."""
    master = [p.data for p in params]
    for p in params:
        p.data = p.data.to(torch.bfloat16)
    try:
        yield
    finally:
        for p, d in zip(params, master):
            p.data = d


def scalar_writer(out_dir):
    """A TensorBoard writer for the run, or None without the ``tensorboard`` package."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter(str(out_dir), flush_secs=10)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


class Distiller:
    def __init__(self, args):
        from transformers import AutoProcessor

        self.args = args
        self.started = time.time()                             # wall clock of the run, measurements included
        self.device = torch.device("cuda")
        self.processor = AutoProcessor.from_pretrained(args.base_model_dir)
        self.teacher = build_model("gdn", args.base_model_dir, self.device)
        self.student = build_model(args.kernel, args.base_model_dir, self.device)
        self.student.requires_grad_(False)
        text = self.student.model.language_model
        scope = {"mixer": list(mixers(self.student).values()), "block": [text.layers[i] for i in mixers(self.student)],
                 "all": [text]}[args.train]
        self.params = [p for m in scope for p in m.parameters()]
        for p in self.params:
            p.data = p.data.float()                            # float32 master weights
            p.requires_grad_(True)
        ids = {id(p) for p in self.params}
        self.trained = [n for n, p in self.student.named_parameters() if id(p) in ids]
        if all(getattr(m, "checkpointable", True) for m in mixers(self.student).values()):
            self.student.gradient_checkpointing_enable()       # the loop phase back-propagates through 16K tokens
        self.val = [(kind, to_device(b, self.device)) for kind, b in data.validation(self.processor, args.base_model_dir)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.val_ref = [self.teacher.model(**b, use_cache=False).last_hidden_state for _, b in self.val]
        self.image_token = self.student.config.image_token_id
        self.tokens, self.seconds = 0, 0.0

    def predicting(self, batch):
        """Flat indices of the positions a next token is predicted from: all but the image tokens."""
        return (batch["input_ids"].flatten() != self.image_token).nonzero().squeeze(1)

    # --------------------------------------------------------------------- measurement
    @torch.no_grad()
    def gap(self) -> dict:
        """KL(teacher ‖ student) per kind of held-out input and their mean, top-1 agreement likewise — with
        the trained weights in bf16."""
        self.student.eval()
        kl, top1, count = {}, {}, {}
        with deployed(self.params), torch.autocast("cuda", dtype=torch.bfloat16):
            for (kind, batch), ref in zip(self.val, self.val_ref):
                hidden, at = self.student.model(**batch, use_cache=False).last_hidden_state, self.predicting(batch)
                for s, t in zip(hidden.flatten(0, 1)[at].split(4096), ref.flatten(0, 1)[at].split(4096)):
                    ls = F.log_softmax(self.student.lm_head(s).float(), -1)
                    lt = F.log_softmax(self.teacher.lm_head(t).float(), -1)
                    for k in dict.fromkeys((kind.split(":")[0], kind)):      # "image:vqav2" counts for image and for itself
                        kl[k] = kl.get(k, 0.0) + F.kl_div(ls, lt, log_target=True, reduction="sum").item()
                        top1[k] = top1.get(k, 0) + (ls.argmax(-1) == lt.argmax(-1)).sum().item()
                        count[k] = count.get(k, 0) + len(s)
        kinds = [k for k in kl if ":" not in k]
        out = {f"kl_{k}": kl[k] / count[k] for k in kinds}
        out["gap"] = sum(out.values()) / len(out)
        out["top1"] = sum(top1[k] / count[k] for k in kinds) / len(kinds)
        out.update({k.replace(":", "_"): kl[k] / count[k] for k in kl if ":" in k})    # the parts of a kind, not in the mean
        return out

    # ---------------------------------------------------------------------------- losses
    def local_loss(self, batch, weight=1.0):
        """Teacher-forced: each student mixer on the teacher mixer's input.  Backward per layer."""
        seen = {}
        hooks = [m.register_forward_hook(lambda _, a, kw, out, i=i: seen.__setitem__(i, (kw["hidden_states"], out)),
                                         with_kwargs=True) for i, m in mixers(self.teacher).items()]
        with torch.no_grad():
            self.teacher.model(**batch, use_cache=False)
        for h in hooks:
            h.remove()
        total = 0.0
        for i, mixer in mixers(self.student).items():
            x, want = seen.pop(i)
            loss = nmse(mixer(hidden_states=x), want)
            (loss * weight).backward()
            total += loss.item()
        return {"loss": total / len(mixers(self.student))}

    def loop_loss(self, batch, weight=1.0):
        """Student-forced: the hidden states after every block against the teacher's."""
        with torch.no_grad():
            want = self.teacher.model(**batch, use_cache=False, output_hidden_states=True).hidden_states[1:]
        got = self.student.model(**batch, use_cache=False, output_hidden_states=True).hidden_states[1:]
        loss = sum(nmse(s, t) for s, t in zip(got, want)) / len(want)
        (loss * weight).backward()
        return {"loss": loss.item()}

    # ----------------------------------------------------------------------------- phase
    def run_phase(self, phase, stream, log, budget, history):
        """Train until the gap stops improving (one learning-rate cut, then the end) or ``budget`` tokens."""
        a = self.args
        lr = getattr(a, f"{phase}_lr")
        optimizer = torch.optim.AdamW(self.params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
        loss_fn = getattr(self, f"{phase}_loss")
        milestone, cut, start, step = a.first_milestone, False, self.tokens, 0
        print(f"[distill] phase {phase}: lr {lr:g}, {a.step_tokens:,} tokens per step, up to {budget / 1e6:.0f}M tokens", flush=True)
        while self.tokens - start < budget:
            self.student.train()
            t0 = time.time()
            got, parts = self.accumulate(loss_fn, stream)
            norm = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            step += 1
            now = lr * min(1.0, step / WARMUP) * (max(0.0, 1 - (self.tokens - start) / budget) if a.decay else 1.0)
            for g in optimizer.param_groups:
                g["lr"] = now
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.tokens += got
            self.seconds += time.time() - t0
            log({"phase": phase, "step": step, "tokens": self.tokens, "seconds": round(self.seconds, 1), "lr": now,
                 "grad_norm": float(norm), **{k: v / got for k, v in parts.items()}})
            if self.tokens - start >= milestone or self.tokens - start >= budget:
                gap = self.measure(phase, log, history)
                milestone *= math.sqrt(10)
                if step > WARMUP and len(history) > 1 and gap > (1 - a.min_gain) * history[-2]["gap"]:   # not judged while warming up
                    if cut:
                        print(f"[distill] phase {phase}: no further gain, done", flush=True)
                        return
                    cut, lr = True, lr * 0.3
                    print(f"[distill] phase {phase}: gain under {a.min_gain:.0%}, learning rate -> {lr:g}", flush=True)

    def accumulate(self, loss_fn, stream):
        """The gradient of one optimizer step, left in ``.grad``: batches are drawn until they hold
        ``--step_tokens`` and weighted by their tokens (``--weight tokens``: the gradient of the mean loss per
        token, which is also what is logged) or equally.  Returns the tokens drawn and the summed, token-weighted
        losses."""
        a, got, total, parts = self.args, 0, 0.0, {}
        while got < a.step_tokens:
            _, batch = next(stream)
            batch = to_device(batch, self.device)
            n = batch["input_ids"].numel()
            weight = n / a.step_tokens if a.weight == "tokens" else 1.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = loss_fn(batch, weight)
            for k, v in out.items():
                parts[k] = parts.get(k, 0.0) + v * n
            got += n
            total += weight
        for p in self.params:                                  # gradients were summed over the batches
            if p.grad is not None:
                p.grad.div_(total)
        return got, parts

    def measure(self, phase, log, history):
        rec = {"phase": phase, "tokens": self.tokens, "seconds": round(self.seconds, 1),
               "wall": round(time.time() - self.started, 1), **self.gap()}
        history.append(rec)
        log(rec)
        best = min(h["gap"] for h in history)
        if rec["gap"] == best:
            self.save(history)
        print(f"[distill] {phase:5s} {self.tokens / 1e6:7.2f}M tokens  {self.seconds / 3600:5.2f} GPU-h  gap {rec['gap']:.4f}  "
              + "  ".join(f"{k[3:]} {v:.4f}" for k, v in rec.items() if k.startswith("kl_")) + f"  top-1 {rec['top1']:.4f}"
              + ("  *" if rec["gap"] == best else ""), flush=True)
        return rec["gap"]

    def save(self, history):
        with deployed(self.params):
            save_checkpoint(self.student, self.out_dir, trained=self.trained, train=self.args.train, tokens=self.tokens,
                            gpu_hours=round(self.seconds / 3600, 3), wall_hours=round((time.time() - self.started) / 3600, 3),
                            weight=self.args.weight, decay=self.args.decay, gap=history[-1]["gap"], history=history)


def main(args) -> Path | None:
    _, kernel, ckpt = resolve(args.kernel)
    spec = get_kernel(kernel)
    if spec.exact_init and ckpt is None and not args.force:
        print(f"[distill] {kernel}: the init is exact, the swap already computes the teacher's function — nothing to "
              f"train (check with `linswap verify --kernel {kernel}`; --force trains it anyway)")
        return None
    torch.manual_seed(args.seed)
    base = Path(args.base_model_dir).resolve()
    family = () if base == Path(DEFAULT_BASE_MODEL_DIR).resolve() else (base.name,)
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT.joinpath("outputs", "distill", *family, kernel)
    if ckpt is not None and out_dir.resolve() == ckpt:
        raise SystemExit(f"distill: {ckpt} is the checkpoint to continue from; writing there would replace its "
                         f"linswap.json and append to its train_log.jsonl — pass another --output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = [p for p in args.phases.split(",") if p]
    if any(p not in PHASES for p in phases):
        raise SystemExit(f"distill: phases are {PHASES}")
    write_manifest(out_dir / "manifest.json", kernel=kernel, args=vars(args), base_model=model_identity(args.base_model_dir),
                   mix=[dict(zip(("source", "length", "batch", "share"), m)) for m in data.MIX],
                   validation={"image_subsets": list(data.CAULDRON[1]), "image_chats_per_subset": data.VAL_CHATS},
                   gpu=gpu_state())
    d = Distiller(args)
    d.out_dir = out_dir
    print(f"[distill] {kernel} -> {out_dir}; training {TRAIN[args.train]}: {sum(p.numel() for p in d.params) / 1e6:.1f}M "
          f"parameters in float32, the rest frozen", flush=True)
    stream = data.stream(d.processor, args.base_model_dir, args.seed)
    history, board = [], scalar_writer(out_dir)
    with open(out_dir / "train_log.jsonl", "a") as f:
        def log(rec):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            for key, value in rec.items() if board else ():
                if key not in ("phase", "step", "tokens"):
                    board.add_scalar(f"{'train' if 'step' in rec else 'gap'}/{key}", value, rec["tokens"])

        d.measure("init", log, history)
        for phase in phases:
            budget = args.max_tokens - d.tokens
            d.run_phase(phase, stream, log, min(budget, args.local_tokens) if phase == "local" else budget, history)
    if board:
        board.close()
    best = min(history, key=lambda h: h["gap"])
    print(f"[distill] done: gap {history[0]['gap']:.4f} -> {best['gap']:.4f} at {best['tokens'] / 1e6:.1f}M tokens "
          f"({best['seconds'] / 3600:.2f} GPU-h); trained {d.tokens / 1e6:.1f}M tokens in all, "
          f"{(time.time() - d.started) / 3600:.2f} h wall with the measurements -> {out_dir}")
    return out_dir
