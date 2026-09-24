"""Stage 0 — run: verify → distill → evaluate → lmeval for one kernel with the standard recipe.

    linswap run --kernel rwkv7                     # the whole pipeline, results in outputs/eval/rwkv7
    linswap run --kernel mamba2 --samples 25       # same, cheaper evaluation

``verify`` checks that the swap reproduces the original model (exactly, for kernels registered with
``exact_init=True``), ``distill`` runs the three training steps on generic text, and the distilled
checkpoint is then scored on long-context retrieval and on the short-context suite next to the
unmodified backbone.  Any ``distill`` / ``evaluate`` / ``lmeval`` option can be passed through.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import distill, evaluate, lmeval, verify


def _namespace(add_args_fn, argv):
    ap = argparse.ArgumentParser(add_help=False)
    add_args_fn(ap)
    return ap.parse_args(argv)


def add_args(ap):
    distill.add_args(ap)
    ap.add_argument("--name", default=None, help="evaluation run name (default: the kernel name)")
    ap.add_argument("--skip_verify", action="store_true")
    ap.add_argument("--skip_distill", action="store_true", help="evaluate the base swap only")
    ap.add_argument("--verify_checks", default="layer,logits,cache,generation")
    ap.add_argument("--verify_lengths", default="8,512,4096")
    ap.add_argument("--baseline", default="gdn", help="second kernel for the verify comparison")
    ap.add_argument("--tasks", default=evaluate.DEFAULT_TASKS)
    ap.add_argument("--lengths", default="4096,16384,65536,131072", help="RULER sequence lengths")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--chat_template", action="store_true")
    ap.add_argument("--ruler_jobs", type=int, default=3, help="RULER tasks to run at once (see evaluate)")
    ap.add_argument("--skip_ruler", action="store_true")
    ap.add_argument("--skip_lmeval", action="store_true")
    ap.add_argument("--lmeval_tasks", default=lmeval.DEFAULT_TASKS)


def main(args):
    name = args.name or args.kernel
    if not args.skip_verify:
        v = _namespace(verify.add_args, ["--kernel", args.kernel, "--checks", args.verify_checks,
                                         "--lengths", args.verify_lengths, "--base_model_dir", args.base_model_dir]
                       + (["--baseline", args.baseline] if args.baseline else []))
        verify.main(v)

    model = args.kernel
    if not args.skip_distill:
        ckpt = distill.main(args)
        model = f"{args.kernel}-distilled={ckpt}"

    e = _namespace(evaluate.add_args, ["--models", model, "--name", name,
                                       "--base_model_dir", args.base_model_dir, "--tasks", args.tasks,
                                       "--lengths", args.lengths, "--samples", str(args.samples),
                                       "--ruler_jobs", str(args.ruler_jobs)]
                   + (["--no_cache"] if args.no_cache else [])
                   + (["--chat_template"] if args.chat_template else [])
                   + (["--skip_ruler"] if args.skip_ruler else []))
    rows = evaluate.main(e)

    if not args.skip_lmeval:
        le = _namespace(lmeval.add_args, ["--models", args.baseline or "gdn", model, "--name", f"{name}-lmeval",
                                          "--base_model_dir", args.base_model_dir, "--tasks", args.lmeval_tasks])
        lmeval.main(le)
    return rows
