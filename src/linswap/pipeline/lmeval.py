"""Stage — lmeval: short-context regression check with lm-eval-harness.

    linswap lmeval --models gdn kda outputs/kda/sft_full/checkpoint-50 --tasks hellaswag,piqa,arc_challenge,winogrande

Each model is exported to an HF checkpoint (cached under outputs/eval/<name>/hf/) and
evaluated with ``lm_eval.simple_evaluate(model="hf", ...)`` — `import linswap` registers the
architecture, so the harness loads it like any HF model.  Writes ``outputs/eval/<name>/lmeval.{csv,md,json}``.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from ..load_weights import DEFAULT_BASE_MODEL_DIR, REPO_ROOT
from .evaluate import resolve_model

DEFAULT_TASKS = "piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,social_iqa,lambada_openai"  # Gated DeltaNet paper, Table 3


def add_args(ap):
    ap.add_argument("--models", nargs="+", required=True, help="kernel names, checkpoint dirs, label=path")
    ap.add_argument("--name", default=None)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=None, help="examples per task (None = all)")
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--batch_size", default="8")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))


def main(args):
    import lm_eval

    from ..hf import export

    name = args.name or time.strftime("lmeval-%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in args.tasks.split(",") if t]
    rows = []
    for spec in args.models:
        disp, kernel, base, ckpt = resolve_model(spec, args.base_model_dir)
        hf_dir = out_dir / "hf" / disp
        if not (hf_dir / "config.json").exists():
            export(kernel, hf_dir, base_model_dir=base, ckpt_dir=ckpt)
        t = time.time()
        res = lm_eval.simple_evaluate(model="hf", model_args=f"pretrained={hf_dir},dtype=bfloat16,trust_remote_code=False",
                                      tasks=tasks, num_fewshot=args.num_fewshot, limit=args.limit,
                                      batch_size=args.batch_size, log_samples=False)
        (out_dir / "raw").mkdir(exist_ok=True)
        with open(out_dir / "raw" / f"{disp}.json", "w") as f:      # full lm-eval result dicts, for re-parsing
            json.dump({t: res["results"].get(t, {}) for t in tasks}, f, indent=1)
        row = {"model": disp, "kernel": kernel}
        for task in tasks:
            r = res["results"].get(task, {})
            # accuracy-like metrics first (acc_norm / acc / contains for SWDE, FDA, SQuAD-completion / exact match / F1),
            # perplexity last; LAMBADA additionally reports its perplexity in a second column.
            metric = next((k for k in ("acc_norm,none", "acc,none", "contains,none", "exact_match,none", "em,none",
                                       "f1,none", "perplexity,none") if k in r), None)
            row[task] = round(float(r[metric]), 4) if metric else None
            if "perplexity,none" in r and metric != "perplexity,none":
                row[f"{task}_ppl"] = round(float(r["perplexity,none"]), 3)
        rows.append(row)
        print(f"  {disp}: " + ", ".join(f"{k}={v}" for k, v in row.items() if k not in ("model", "kernel")) + f"  ({time.time()-t:.0f}s)", flush=True)
        with open(out_dir / "lmeval.json", "w") as f:
            json.dump(rows, f, indent=1)
    cols = ["model", "kernel"] + [c for c in rows[0] if c not in ("model", "kernel")] if rows else ["model", "kernel"]
    with open(out_dir / "lmeval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (out_dir / "lmeval.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    return rows
