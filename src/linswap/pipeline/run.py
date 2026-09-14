"""Stage 0 — run: verify → (distill) → posttrain → evaluate for one kernel with the standard recipe.

    linswap run --kernel kda                      # full pipeline, results in outputs/eval/kda
    linswap run --kernel kda --modes full --tasks niah_single_1 --lengths 4096 --samples 10

Kernels registered with ``exact_init=False`` (or ``--distill``) are distilled from
the original model first and SFT starts from the distilled checkpoint.  The base
swap, the distilled checkpoint and every SFT checkpoint are evaluated.
Any ``posttrain`` / ``evaluate`` option can be passed through (they share the
argument names; ``--lengths`` is RULER's, ``--verify_lengths`` the verify one).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import distill, evaluate, posttrain, verify
from ..registry import get_kernel


def _namespace(add_args_fn, argv):
    ap = argparse.ArgumentParser(add_help=False)
    add_args_fn(ap)
    return ap.parse_args(argv)


def add_args(ap):
    posttrain.add_args(ap)
    ap.add_argument("--name", default=None, help="evaluation run name (default: the kernel name)")
    ap.add_argument("--distill", choices=["auto", "yes", "no"], default="auto",
                    help="distil before SFT: auto = only for inexact kernels")
    ap.add_argument("--distill_stages", default="layer,kl")
    ap.add_argument("--layer_steps", type=int, default=200)
    ap.add_argument("--kl_steps", type=int, default=300)
    ap.add_argument("--distill_length", type=int, default=8192)
    ap.add_argument("--distill_kl_schedule", default=None, help="e.g. 8192:200,65536:100 (packed long-context KL)")
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

    models = [args.kernel]
    do_distill = args.distill == "yes" or (args.distill == "auto" and not get_kernel(args.kernel).exact_init)
    if do_distill:
        d = _namespace(distill.add_args, ["--kernel", args.kernel, "--base_model_dir", args.base_model_dir,
                                          "--data_dir", args.data_dir, "--data_max_length", str(args.data_max_length),
                                          "--datasets", args.datasets, "--stages", args.distill_stages, "--layer_steps", str(args.layer_steps),
                                          "--kl_steps", str(args.kl_steps), "--max_length", str(args.distill_length),
                                          "--seed", str(args.seed)]
                       + (["--kl_schedule", args.distill_kl_schedule] if args.distill_kl_schedule else [])
                       + (["--output_dir", str(Path(args.output_dir) / "distill")] if args.output_dir else []))
        args.init_ckpt = str(distill.main(d))
        models.append(args.init_ckpt)

    ckpts = posttrain.main(args)
    models += [str(c) for c in ckpts]

    e = _namespace(evaluate.add_args, ["--models", *models,
                                       "--name", args.name or args.kernel,
                                       "--base_model_dir", args.base_model_dir, "--tasks", args.tasks,
                                       "--lengths", args.lengths, "--samples", str(args.samples),
                                       "--val_batches", str(args.val_batches), "--data_dir", args.data_dir,
                                       "--data_max_length", str(args.data_max_length), "--datasets", args.datasets]
                   + (["--no_cache"] if args.no_cache else []) + (["--skip_ruler"] if args.skip_ruler else []))
    return evaluate.main(e)
