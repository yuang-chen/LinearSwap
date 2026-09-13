"""``linswap`` command line: verify → (distill) → posttrain → evaluate, ``run`` for the whole chain,
``export`` to a Hugging Face checkpoint, ``kernels`` to list what is registered.

    linswap verify    --kernel kda --baseline gdn
    linswap distill   --kernel mamba2
    linswap posttrain --kernel kda
    linswap evaluate  --models kda outputs/kda/sft_full/checkpoint-50
    linswap run       --kernel kda
    linswap export    --kernel kda --ckpt outputs/kda/sft_full/checkpoint-50 --out hf/Qwen3.5-0.8B-KDA
    linswap kernels
"""

import argparse

from .pipeline import STAGES
from .registry import get_kernel, list_kernels


def main(argv=None):
    ap = argparse.ArgumentParser(prog="linswap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name, mod in STAGES.items():
        sp = sub.add_parser(name, help=mod.__doc__.strip().splitlines()[0], description=mod.__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        mod.add_args(sp)
    sub.add_parser("kernels", help="list registered kernels")
    args = ap.parse_args(argv)
    if args.stage == "kernels":
        for k in list_kernels():
            s = get_kernel(k)
            print(f"{k:13s} exact_init={str(s.exact_init):5s} new_params={s.new_param_names}  {s.description}")
        return
    STAGES[args.stage].main(args)


if __name__ == "__main__":
    main()
