"""Register a kernel-swapped model for RULER under outputs/ruler_models/<name>.

    python scripts/register_ruler_model.py --name kda-base --kernel kda
    python scripts/register_ruler_model.py --name kda-full-50 --ckpt outputs/sft_kda_full/checkpoint-50

Afterwards ``cd RULER/scripts && bash run.sh linswap-<name> synthetic`` evaluates it
(``linswap-nocache-<name>`` for no-cache decoding).  The directory only holds a
config.json pointing at the kernel / checkpoint; weights are not copied.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_linswap import DEFAULT_BASE_MODEL_DIR, get_kernel, read_checkpoint_kernel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--ckpt", default=None, help="directory containing model.pt (+ config.json)")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))
    args = ap.parse_args()

    kernel = args.kernel or (read_checkpoint_kernel(args.ckpt) if args.ckpt else None)
    if kernel is None:
        raise SystemExit("--kernel is required when --ckpt has no config.json with linear_kernel")
    get_kernel(kernel)  # validate
    out = REPO / "outputs" / "ruler_models" / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"linear_kernel": kernel, "base_model_dir": str(Path(args.base_model_dir).resolve())}
    if args.ckpt:
        cfg["ckpt_dir"] = str(Path(args.ckpt).resolve())
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"registered {out} -> {cfg}\nrun: cd RULER/scripts && bash run.sh linswap-{args.name} synthetic")


if __name__ == "__main__":
    main()
