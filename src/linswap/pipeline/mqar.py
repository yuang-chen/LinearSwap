"""Stage — mqar: in-context multi-query associative recall versus number of key-value pairs.

    linswap mqar --models gdn kda mamba2-sft=outputs/mamba2/sft_full/checkpoint-50 --pairs 16,64,256,1024,4096
"""

from __future__ import annotations

import csv
import json
import time

import torch

from ..load_weights import DEFAULT_BASE_MODEL_DIR, REPO_ROOT, build_model
from ..mqar import mqar_accuracy
from .evaluate import resolve_model


def add_args(ap):
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--pairs", default="16,64,256,1024,4096")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))


def main(args):
    from transformers import AutoTokenizer

    name = args.name or time.strftime("mqar-%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [int(p) for p in args.pairs.split(",") if p]
    tok = AutoTokenizer.from_pretrained(args.base_model_dir)
    rows = []
    for spec in args.models:
        disp, kernel, base, ckpt = resolve_model(spec, args.base_model_dir)
        model = build_model(kernel, base_model_dir=base, ckpt_dir=ckpt).eval()
        row = {"model": disp, "kernel": kernel}
        for n in pairs:
            r = mqar_accuracy(model, tok, n, args.samples)
            row[f"acc@{n}"] = round(r["acc"], 4)
            row[f"len@{n}"] = r["max_len"]
        print(f"  {disp}: " + ", ".join(f"{n}:{row[f'acc@{n}']:.3f}" for n in pairs), flush=True)
        rows.append(row)
        del model
        torch.cuda.empty_cache()
        with open(out_dir / "mqar.json", "w") as f:
            json.dump(rows, f, indent=1)
    cols = ["model", "kernel"] + [f"acc@{n}" for n in pairs] + [f"len@{n}" for n in pairs]
    with open(out_dir / "mqar.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (out_dir / "mqar.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    return rows
