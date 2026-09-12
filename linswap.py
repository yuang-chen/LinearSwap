#!/usr/bin/env python
"""qwen-linswap command line: verify → posttrain → evaluate (or `run` for all three).

    python linswap.py verify    --kernel kda --baseline gdn
    python linswap.py posttrain --kernel kda
    python linswap.py evaluate  --models kda outputs/kda/sft_full/checkpoint-50
    python linswap.py run       --kernel kda
    python linswap.py kernels                       # list registered kernels
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qwen_linswap.pipeline import STAGES  # noqa: E402
from qwen_linswap.registry import get_kernel, list_kernels  # noqa: E402


def main():
    ap = argparse.ArgumentParser(prog="linswap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name, mod in STAGES.items():
        sp = sub.add_parser(name, help=mod.__doc__.strip().splitlines()[0], description=mod.__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        mod.add_args(sp)
    sub.add_parser("kernels", help="list registered kernels")
    args = ap.parse_args()
    if args.stage == "kernels":
        for k in list_kernels():
            s = get_kernel(k)
            print(f"{k:13s} exact_init={str(s.exact_init):5s} new_params={s.new_param_names}  {s.description}")
        return
    STAGES[args.stage].main(args)


if __name__ == "__main__":
    main()
