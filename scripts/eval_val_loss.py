"""Validation cross-entropy / perplexity for registered swap models.

    python scripts/eval_val_loss.py --models kda-base kda-full-50 gdn2-full-50 --batches 40

``--models`` are names under outputs/ruler_models (see register_ruler_model.py)
or checkpoint directories.  All models see the same first ``--batches``
validation examples (left-truncated to ``--max_length``).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_linswap import DEFAULT_BASE_MODEL_DIR, build_model  # noqa: E402
from qwen_linswap.sft_utils import TruncatedDataset, collate_fn, evaluate  # noqa: E402


def resolve(name):
    d = Path(name)
    if not d.exists():
        d = REPO / "outputs" / "ruler_models" / name
    cfg = json.load(open(d / "config.json"))
    ckpt = Path(cfg.get("ckpt_dir", d))
    return cfg["linear_kernel"], Path(cfg.get("base_model_dir", DEFAULT_BASE_MODEL_DIR)), ckpt if (ckpt / "model.pt").exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data_dir", default=str(REPO / "data/sft/len262144"))
    ap.add_argument("--max_length", type=int, default=131072)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--out", default=None, help="optional json file for the results")
    args = ap.parse_args()

    val_ds = TruncatedDataset(load_from_disk(Path(args.data_dir) / "validation"), args.max_length)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    results = {}
    print(f"{'model':28s} {'kernel':13s} {'val CE':>8s} {'ppl':>7s}")
    for name in args.models:
        kernel, base, ckpt = resolve(name)
        model = build_model(kernel, base_model_dir=base, ckpt_dir=ckpt).eval()
        ce = evaluate(model, val_loader, max_batches=args.batches)
        results[name] = {"kernel": kernel, "val_ce": ce, "ppl": math.exp(ce)}
        print(f"{name:28s} {kernel:13s} {ce:8.4f} {math.exp(ce):7.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
