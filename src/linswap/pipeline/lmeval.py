"""Stage — lmeval: short-context regression check with lm-eval-harness.

    linswap lmeval --models gdn rwkv7-distilled=outputs/rwkv7/distill/checkpoint-16338

The default task set is the standard short-context suite (LAMBADA, ARC-c/e, PIQA, WinoGrande,
HellaSwag 0-shot and MMLU 5-shot).  ``--relative_to`` adds the relative score (s - r)/(t - r) of every
model against a reference row (usually the unmodified backbone), with r the chance level of the task.

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

DEFAULT_TASKS = "lambada_openai,arc_challenge,arc_easy,piqa,winogrande,hellaswag,mmlu"


def add_args(ap):
    ap.add_argument("--models", nargs="+", required=True, help="kernel names, checkpoint dirs, label=path")
    ap.add_argument("--name", default=None)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=None, help="examples per task (None = all)")
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--fewshot_tasks", default="mmlu:5", help="tasks evaluated with their own shot count, e.g. mmlu:5")
    ap.add_argument("--relative_to", default="gdn-base", help="row whose scores are the reference t in the relative "
                                                            "score (s - r)/(t - r), r = chance; '' to disable")
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
        margs = f"pretrained={hf_dir},dtype=bfloat16,trust_remote_code=False"
        fs = {k: int(v) for k, v in (kv.split(":") for kv in args.fewshot_tasks.split(",") if kv)}
        res = lm_eval.simple_evaluate(model="hf", model_args=margs, tasks=[t for t in tasks if t not in fs],
                                      num_fewshot=args.num_fewshot, limit=args.limit, batch_size=args.batch_size,
                                      log_samples=False) if [t for t in tasks if t not in fs] else {"results": {}}
        for fs_task, n in fs.items():   # few-shot tasks run separately with their own shot count
            r2 = lm_eval.simple_evaluate(model="hf", model_args=margs, tasks=[fs_task], num_fewshot=n, limit=args.limit,
                                         batch_size=args.batch_size, log_samples=False)
            res["results"].update(r2["results"])
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
    if args.relative_to:   # relative score: (student - chance) / (reference - chance), in percent
        chance = {"lambada_openai": 0.0, "piqa": 0.5, "winogrande": 0.5, "boolq": 0.5, "arc_easy": 0.25,
                  "arc_challenge": 0.25, "hellaswag": 0.25, "mmlu": 0.25, "social_iqa": 1 / 3}
        teacher = next((r for r in rows if r["model"] == args.relative_to), None)
        if teacher is None:
            print(f"[lmeval] --relative_to {args.relative_to!r} not among the evaluated models; skipping relative scores")
        else:
            for r in rows:
                for t in tasks:
                    s_, t_ = r.get(t), teacher.get(t)
                    if s_ is not None and t_ is not None and t in chance and t_ != chance[t]:
                        r[f"{t}_rel"] = round(100 * (s_ - chance[t]) / (t_ - chance[t]), 1)
                rels = [r[k] for k in r if k.endswith("_rel")]
                if rels:
                    r["rel_avg"] = round(sum(rels) / len(rels), 1)
            with open(out_dir / "lmeval.json", "w") as f:
                json.dump(rows, f, indent=1)
    cols = ["model", "kernel"] + [c for c in rows[0] if c not in ("model", "kernel")] if rows else ["model", "kernel"]
    with open(out_dir / "lmeval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (out_dir / "lmeval.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    return rows
