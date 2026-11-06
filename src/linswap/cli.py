"""``linswap`` command line.

    linswap kernels                               # list the kernels
    linswap verify  --kernel kda                  # is the swap faithful, on text and on images?
    linswap distill --kernel gla                  # train the swapped layers towards the teacher
    linswap eval    --models native kda outputs/distill/gla      # standard test sets (--smoke: a quick check)
"""

import argparse
import importlib

from .registry import get_kernel, list_kernels

STAGES = {"verify": "verify", "distill": "distill", "eval": "evaluate"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="linswap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    mods = {name: importlib.import_module(f".{mod}", __package__) for name, mod in STAGES.items()}
    for name, mod in mods.items():
        sp = sub.add_parser(name, help=mod.__doc__.strip().splitlines()[0], description=mod.__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        mod.add_args(sp)
    sub.add_parser("kernels", help="list the kernels")
    args = ap.parse_args(argv)
    if args.stage == "kernels":
        for k in list_kernels():
            s = get_kernel(k)
            print(f"{k:9s} exact_init={str(s.exact_init):5s} {s.description}")
        return
    mods[args.stage].main(args)


if __name__ == "__main__":
    main()
