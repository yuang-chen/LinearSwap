"""``linswap`` command line: verify → distill → evaluate, ``run`` for the whole pipeline.

    linswap kernels                                      # list the registered kernels
    linswap verify   --kernel rwkv7 --baseline gdn       # is the swap a faithful replacement?
    linswap distill  --kernel rwkv7                      # three training steps on generic text
    linswap evaluate --models gdn rwkv7-distilled=outputs/rwkv7/distill/checkpoint-16338
    linswap lmeval   --models gdn rwkv7-distilled=outputs/rwkv7/distill/checkpoint-16338
    linswap run      --kernel rwkv7                      # all of the above
    linswap export   --ckpt outputs/rwkv7/distill/checkpoint-16338 --out hf/Qwen3.5-0.8B-RWKV7
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
