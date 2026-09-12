"""Stage 0 — run: verify → posttrain → evaluate for one kernel with the standard recipe.

    python linswap.py run --kernel kda                      # full pipeline, results in outputs/eval/kda
    python linswap.py run --kernel kda --modes full --tasks niah_single_1 --lengths 4096 --samples 10

The base swap and every checkpoint produced by ``posttrain`` are evaluated.
Any ``posttrain`` / ``evaluate`` option can be passed through (they share the
argument names; ``--lengths`` is RULER's, ``--verify_lengths`` the verify one).
"""

from __future__ import annotations

import argparse

from . import evaluate, posttrain, verify


def _namespace(add_args_fn, argv):
    ap = argparse.ArgumentParser(add_help=False)
    add_args_fn(ap)
    return ap.parse_args(argv)


def add_args(ap):
    posttrain.add_args(ap)
    ap.add_argument("--name", default=None, help="evaluation run name (default: the kernel name)")
    ap.add_argument("--skip_verify", action="store_true")
    ap.add_argument("--verify_checks", default="layer,logits,cache,generation")
    ap.add_argument("--verify_lengths", default="8,512,4096")
    ap.add_argument("--baseline", default=None, help="second kernel for the verify comparison, e.g. gdn")
    ap.add_argument("--tasks", default=evaluate.DEFAULT_TASKS)
    ap.add_argument("--lengths", default="131072", help="RULER sequence lengths")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--val_batches", type=int, default=40)
    ap.add_argument("--skip_ruler", action="store_true")


def main(args):
    if not args.skip_verify:
        v = _namespace(verify.add_args, ["--kernel", args.kernel, "--checks", args.verify_checks,
                                         "--lengths", args.verify_lengths, "--base_model_dir", args.base_model_dir]
                       + (["--baseline", args.baseline] if args.baseline else []))
        verify.main(v)

    ckpts = posttrain.main(args)

    e = _namespace(evaluate.add_args, ["--models", args.kernel, *map(str, ckpts),
                                       "--name", args.name or args.kernel,
                                       "--base_model_dir", args.base_model_dir, "--tasks", args.tasks,
                                       "--lengths", args.lengths, "--samples", str(args.samples),
                                       "--val_batches", str(args.val_batches), "--data_dir", args.data_dir,
                                       "--data_max_length", str(args.data_max_length)]
                   + (["--no_cache"] if args.no_cache else []) + (["--skip_ruler"] if args.skip_ruler else []))
    return evaluate.main(e)
