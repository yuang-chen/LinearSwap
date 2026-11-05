"""A resumed ``distill`` run against the same run left alone (one GPU, ~3 min).

A short ``kl`` + ``ce`` run goes through; its mid-``ce`` checkpoint is copied aside the moment it is
written, as an interruption would leave it.  ``--init_from`` that copy must replay the data order,
restore the optimizer state and carry the work counters on, so that it ends where the uninterrupted
run ends.  A second continuation starts from the same checkpoint *without* ``optimizer.pt`` (what
every checkpoint looked like before the state was saved).

Training on the GPU is not bit-reproducible (two identical runs already differ after two steps), so
"ends where the uninterrupted run ends" cannot be equality: the test asks that the continuation with
the optimizer state lands at most half as far from the uninterrupted run as the one without it, and
prints both distances.

    python tests/test_resume.py            # LINSWAP_TEST_TMP=<dir> for the ~12 GB of checkpoints
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from linswap.pipeline import distill  # noqa: E402

KL_STEPS, CE_STEPS, SAVE_EVERY = 2, 6, 3
KL_TOKENS, CE_TOKENS = 4 * 256, 4 * 512                 # per optimizer step
CUT = KL_STEPS + SAVE_EVERY                             # nominal step of the mid-ce checkpoint
END = KL_STEPS + CE_STEPS


def args_for(out_dir, init_from=None):
    ap = argparse.ArgumentParser()
    distill.add_args(ap)
    argv = ["--kernel", "kda", "--output_dir", str(out_dir), "--stages", "kl,ce",
            "--kl_steps", str(KL_STEPS), "--ce_steps", str(CE_STEPS),
            "--stage_length", "kl:256,ce:512", "--stage_batch", "kl:4,ce:4", "--stage_micro", "kl:4,ce:2",
            "--save_every", str(SAVE_EVERY), "--eval_every", "1000", "--eval_batches", "1", "--val_length", "512",
            "--no_tensorboard"]
    if init_from:
        argv += ["--init_from", str(init_from)]
    return ap.parse_args(argv)


def run(out_dir, init_from=None, keep=None):
    """``distill.main``; ``keep = (step, dir)`` copies that checkpoint to ``dir`` right after it is written
    (later saves prune its optimizer state)."""
    original = distill.save

    def save_and_keep(student, args, out, step, *a, **kw):
        ckpt = original(student, args, out, step, *a, **kw)
        if keep is not None and step == keep[0] and not Path(keep[1]).exists():
            shutil.copytree(ckpt, keep[1])
        return ckpt

    distill.save = save_and_keep
    try:
        distill.main(args_for(out_dir, init_from))
    finally:
        distill.save = original
    gc.collect()
    torch.cuda.empty_cache()


def distance(a, b):
    """(largest |difference|, fraction of weights that differ) between two checkpoints' weights."""
    sa = torch.load(Path(a) / "model.pt", map_location="cpu", weights_only=True)
    sb = torch.load(Path(b) / "model.pt", map_location="cpu", weights_only=True)
    worst, differ, total = 0.0, 0, 0
    for k in sa:
        d = (sa[k].float() - sb[k].float()).abs()
        worst, differ, total = max(worst, d.max().item()), differ + int((d > 0).sum()), total + d.numel()
    return worst, differ / total


def state(ckpt):
    with open(Path(ckpt) / "train_state.json") as f:
        return json.load(f)


def main():
    ok = True

    def check(cond, what):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {what}")

    with tempfile.TemporaryDirectory(dir=os.environ.get("LINSWAP_TEST_TMP")) as tmp:
        tmp = Path(tmp)
        whole, cut, cold = tmp / "whole", tmp / "cut", tmp / "cold"
        run(whole, keep=(CUT, cut / f"checkpoint-{CUT}"))
        check((cut / f"checkpoint-{CUT}" / "optimizer.pt").exists(), "a checkpoint inside a step holds the optimizer state")
        st = state(cut / f"checkpoint-{CUT}")
        check((st["stage"], st["stage_step"], st["updates"]) == ("ce", SAVE_EVERY, CUT)
              and st["tokens"] == KL_STEPS * KL_TOKENS + SAVE_EVERY * CE_TOKENS
              and st["teacher_tokens"] == KL_STEPS * KL_TOKENS,
              f"its train_state.json: {st['updates']} updates, {st['tokens']} tokens, {st['teacher_tokens']} teacher tokens")

        (cold / f"checkpoint-{CUT}").mkdir(parents=True)        # the same checkpoint as it was written before
        for name in ("model.pt", "config.json", "train_state.json"):
            shutil.copy(cut / f"checkpoint-{CUT}" / name, cold / f"checkpoint-{CUT}" / name)
        run(cut, init_from=cut / f"checkpoint-{CUT}")
        run(cold, init_from=cold / f"checkpoint-{CUT}")

        end = f"checkpoint-{END}"
        want = {"updates": END, "tokens": KL_STEPS * KL_TOKENS + CE_STEPS * CE_TOKENS,
                "teacher_tokens": KL_STEPS * KL_TOKENS}
        for name, d in (("uninterrupted", whole), ("resumed", cut)):
            st = state(d / end)
            check(all(st[k] == v for k, v in want.items()),
                  f"{name}: {st['updates']} updates, {st['tokens']} tokens, {st['teacher_tokens']} teacher tokens")
        check(not list(whole.glob("checkpoint-*/optimizer.pt")) and not list(cut.glob("checkpoint-*/optimizer.pt")),
              "a finished run keeps no optimizer state")
        with open(whole / "train_log.jsonl") as f:
            last = [json.loads(line) for line in f if '"loss"' in line][-1]
        check(last["tokens"] == want["tokens"] and last["step"] == END,
              f"log: step {last['step']} at {last['tokens']} tokens (step x tokens/step would say {END * CE_TOKENS})")
        with open(whole / "manifest.json") as f:
            manifest = json.load(f)
        check([r["tokens"] for r in manifest["stages"]] == [KL_STEPS * KL_TOKENS, CE_STEPS * CE_TOKENS]
              and manifest["base_model"]["tokenizer_sha256"] == manifest["data"]["tokenizer_sha256"],
              "manifest: stage token totals, and the corpus was tokenised with the base model's tokenizer")

        worst, frac = distance(whole / end, cut / end)
        worst_cold, frac_cold = distance(whole / end, cold / end)
        print(f"  resumed with optimizer state vs uninterrupted: max |diff| {worst:.3g}, {frac:.2%} of weights differ")
        print(f"  resumed without it           vs uninterrupted: max |diff| {worst_cold:.3g}, {frac_cold:.2%} of weights differ")
        check(frac <= 0.5 * frac_cold, "the continuation with the optimizer state lands closer than the one without")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
