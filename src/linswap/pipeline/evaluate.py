"""Stage 3 — evaluate: validation loss + RULER for any set of swapped models.

    python linswap.py evaluate --models kda outputs/kda/sft_full/checkpoint-50 gdn --name kda-vs-gdn
    python linswap.py evaluate --models kda --tasks niah_single_1 --lengths 4096,32768 --samples 20

``--models`` entries are kernel names (the base, function-preserving swap) or
checkpoint directories written by ``posttrain``.  Results go to
``outputs/eval/<name>/``: ``val_loss.json``, ``ruler/<model>/<length>/{data,pred}``
and a combined ``summary.csv`` / ``summary.md``.  RULER's own scripts are
invoked directly (no editing of its shell configs).
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from ..data import DEFAULT_DATA_DIR, ensure_sft_data
from ..load_weights import DEFAULT_BASE_MODEL_DIR, REPO_ROOT, build_model, read_checkpoint_kernel
from ..registry import get_kernel, list_kernels
from ..sft_utils import TruncatedDataset, collate_fn, evaluate as eval_loss

RULER_DIR = REPO_ROOT / "RULER" / "scripts"
DEFAULT_TASKS = "niah_single_1,niah_multikey_1,niah_multivalue"


def add_args(ap):
    ap.add_argument("--models", nargs="+", required=True,
                    help="kernel names and/or checkpoint dirs; prefix with label= to name a row")
    ap.add_argument("--name", default=None, help="run name -> outputs/eval/<name> (default: timestamp)")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))
    ap.add_argument("--tasks", default=DEFAULT_TASKS, help="RULER synthetic tasks, comma separated")
    ap.add_argument("--lengths", default="131072", help="RULER sequence lengths, comma separated")
    ap.add_argument("--samples", type=int, default=100, help="RULER samples per task")
    ap.add_argument("--no_cache", action="store_true", help="decode without KV / recurrent-state cache")
    ap.add_argument("--val_batches", type=int, default=40, help="validation examples for the loss (0 = skip)")
    ap.add_argument("--val_max_length", type=int, default=131072)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--data_max_length", type=int, default=262144)
    ap.add_argument("--skip_ruler", action="store_true")


# ----------------------------------------------------------------------------- models
def resolve_model(spec: str, base_model_dir):
    """kernel name | checkpoint dir | label=spec  ->  (display name, kernel, base_model_dir, ckpt_dir | None)."""
    label = None
    if "=" in spec:
        label, spec = spec.split("=", 1)
    out = _resolve_model(spec, base_model_dir)
    return (label or out[0],) + out[1:]


def _resolve_model(spec: str, base_model_dir):
    if spec in list_kernels():
        return f"{spec}-base", spec, Path(base_model_dir), None
    d = Path(spec).resolve()  # RULER runs from its own directory, so paths must be absolute
    if not (d / "model.pt").exists():
        raise SystemExit(f"evaluate: {spec!r} is neither a registered kernel nor a checkpoint dir with model.pt")
    with open(d / "config.json") as f:
        cfg = json.load(f)
    kernel = cfg.get("linear_kernel") or read_checkpoint_kernel(d)
    step = d.name.split("-")[-1] if d.name.startswith("checkpoint-") else d.name
    mode = cfg.get("sft_mode", "sft").replace("gate_only", "gate")
    return f"{kernel}-{mode}-{step}", kernel, Path(cfg.get("base_model_dir", base_model_dir)), d


# ------------------------------------------------------------------------------ RULER
def _ruler_env():
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def _run(cmd, log_file):
    with open(log_file, "a") as log:
        log.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        log.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=RULER_DIR, env=_ruler_env(), stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"RULER command failed (see {log_file}): {' '.join(map(str, cmd))}")


def _read_summary(pred_dir: Path) -> dict:
    files = list(pred_dir.glob("summary*.csv"))
    if not files:
        return {}
    rows = list(csv.reader(open(max(files, key=lambda p: p.stat().st_mtime))))
    tasks = next(r[1:] for r in rows if r and r[0] == "Tasks")
    scores = next(r[1:] for r in rows if r and r[0] == "Score")
    return {t: float(s) for t, s in zip(tasks, scores)}


def run_ruler(name, model_dir: Path, base_model_dir, tasks, lengths, samples, use_cache, out_dir: Path) -> dict:
    """Returns {length: {task: score}}."""
    results = {}
    log_file = out_dir / "ruler" / f"{name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    for L in lengths:
        res_dir = out_dir / "ruler" / name / str(L)
        data_dir, pred_dir = res_dir / "data", res_dir / "pred"
        data_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        failed = []
        for task in tasks:
            t = time.time()
            try:
                _run([sys.executable, "data/prepare.py", "--save_dir", data_dir, "--benchmark", "synthetic",
                      "--task", task, "--tokenizer_path", base_model_dir, "--tokenizer_type", "hf",
                      "--max_seq_length", L, "--model_template_type", "base", "--num_samples", samples], log_file)
                _run([sys.executable, "pred/call_api.py", "--data_dir", data_dir, "--save_dir", pred_dir,
                      "--benchmark", "synthetic", "--task", task,
                      "--server_type", "linswap" if use_cache else "linswap_nocache",
                      "--model_name_or_path", model_dir, "--temperature", "0.0", "--top_k", "32", "--top_p", "1.0",
                      "--batch_size", "1"], log_file)
                print(f"    {name} L={L} {task}: {time.time()-t:.0f}s", flush=True)
            except RuntimeError as e:  # keep going; the task is reported as missing
                print(f"    {name} L={L} {task}: FAILED ({e})", flush=True)
                failed.append(task)
        _run([sys.executable, "eval/evaluate.py", "--data_dir", pred_dir, "--benchmark", "synthetic"], log_file)
        results[L] = _read_summary(pred_dir)
        for task in failed:
            results[L].setdefault(task, None)
        print(f"  {name} L={L}: {results[L]}", flush=True)
    return results


# ------------------------------------------------------------------------------- main
def main(args):
    name = args.name or time.strftime("eval-%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in args.tasks.split(",") if t]
    lengths = [int(x) for x in args.lengths.split(",") if x]
    models = [resolve_model(m, args.base_model_dir) for m in args.models]
    print(f"[evaluate] {len(models)} models -> {out_dir}")

    val_loader = None
    if args.val_batches > 0:
        data_dir = ensure_sft_data(args.base_model_dir, args.data_max_length, args.data_dir)
        val_ds = TruncatedDataset(load_from_disk(data_dir / "validation"), args.val_max_length)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    rows = []
    for disp, kernel, base, ckpt in models:
        row = {"model": disp, "kernel": kernel, "exact_init": get_kernel(kernel).exact_init}
        if val_loader is not None:
            model = build_model(kernel, base_model_dir=base, ckpt_dir=ckpt).eval()
            ce = eval_loss(model, val_loader, max_batches=args.val_batches)
            row["val_ce"], row["ppl"] = round(ce, 4), round(math.exp(ce), 3)
            print(f"  {disp}: val CE {ce:.4f} (ppl {math.exp(ce):.2f})", flush=True)
            del model
            torch.cuda.empty_cache()
        if not args.skip_ruler:
            if ckpt is not None:
                model_dir = ckpt
            else:  # base swap: a config dir the RULER wrapper can read
                model_dir = out_dir / "models" / disp
                model_dir.mkdir(parents=True, exist_ok=True)
                with open(model_dir / "config.json", "w") as f:
                    json.dump({"linear_kernel": kernel, "base_model_dir": str(base)}, f)
            res = run_ruler(disp, model_dir, base, tasks, lengths, args.samples, not args.no_cache, out_dir)
            for L, scores in res.items():
                for t, s in scores.items():
                    row[f"{t}@{L}"] = s
        rows.append(row)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(rows, f, indent=1)

    cols = ["model", "kernel", "exact_init"] + sorted({k for r in rows for k in r} - {"model", "kernel", "exact_init"},
                                                     key=lambda c: (c not in ("val_ce", "ppl"), c))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\n[evaluate] written {out_dir}/summary.{{csv,md,json}}")
    return rows
