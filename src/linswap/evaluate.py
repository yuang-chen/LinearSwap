"""eval: the standard test sets, through the standard harnesses, for any set of models.

    linswap eval --models native kda outputs/distill/gla --name main        # everything, standard procedure
    linswap eval --models gla --smoke                                        # does it run? (minutes)

``--models`` entries are ``native`` (the unmodified backbone, i.e. the teacher), kernel names (the swap at
step 0), checkpoint directories written by ``distill``, or ``label=<one of those>``.  Three suites, each run
by the harness that defines its protocol, on the full test set:

  text   lm-evaluation-harness: LAMBADA, ARC-c/e, PIQA, WinoGrande, HellaSwag (0-shot), MMLU (5-shot),
         IFEval (chat template, generative)
  long   RULER's own scripts: needles, variable tracking, common / frequent words extraction, 500 samples
         per task at 4K / 16K / 64K / 128K, prompts in the chat template
  mm     lmms-eval: MME, MMBench-EN dev (its "static" variant: answers are extracted by rule, not by a GPT
         judge that needs an API key), AI2D, ChartQA, OCRBench, TextVQA val, MuirBench (multi-image)

Thinking is disabled everywhere and decoding is greedy.  ``--smoke`` keeps the code path and cuts the work:
two tasks per suite, 8 items each, one RULER length — a functional check, not a score.

Results accumulate in ``outputs/eval/<name>/results.json`` (one entry per model and suite; a finished
suite is not run again, so models can be added to a run, or run on several GPUs at once) and
``results.md`` shows the headline metric of every task with each model's difference to ``--reference``.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .model import DEFAULT_BASE_MODEL_DIR, NATIVE, REPO_ROOT, build_model, resolve

RULER_DIR = REPO_ROOT / "RULER" / "scripts"
TEXT_TASKS = "lambada_openai,arc_challenge,arc_easy,piqa,winogrande,hellaswag,mmlu,ifeval"
TEXT_FEWSHOT = {"mmlu": 5}                    # the shot count each benchmark is reported with
TEXT_CHAT = ("ifeval",)                       # generative tasks, run with the chat template
LONG_TASKS = "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,vt,cwe,fwe"
LONG_LENGTHS = "4096,16384,65536,131072"
MM_TASKS = "mme,mmbench_en_dev_static,ai2d,chartqa,ocrbench,textvqa_val,muirbench"
SMOKE = {"text": "lambada_openai,ifeval", "long": "niah_multikey_1,vt", "mm": "ocrbench,chartqa", "limit": 8,
         "lengths": "4096", "samples": 4}
# the metric a task is reported by: the first of these its result holds.  All are fractions except the two
# RULER and MMBench report in percent already.
HEADLINE = ("score", "gpt_eval_score", "prompt_level_strict_acc", "acc_norm", "acc", "exact_match", "relaxed_overall",
            "ocrbench_accuracy", "muirbench_score_overall")


def add_args(ap):
    ap.add_argument("--models", nargs="+", required=True, help="native | kernel | checkpoint dir; label= names a row")
    ap.add_argument("--name", default=None, help="run name -> outputs/eval/<name> (default: timestamp)")
    ap.add_argument("--suites", default="text,long,mm")
    ap.add_argument("--smoke", action="store_true", help="functional check: a few items of two tasks per suite")
    ap.add_argument("--reference", default=NATIVE, help="row the others are compared with in results.md")
    ap.add_argument("--text_tasks", default=TEXT_TASKS)
    ap.add_argument("--long_tasks", default=LONG_TASKS)
    ap.add_argument("--long_lengths", default=LONG_LENGTHS)
    ap.add_argument("--long_samples", type=int, default=500, help="RULER samples per task (its standard: 500)")
    ap.add_argument("--mm_tasks", default=MM_TASKS)
    ap.add_argument("--limit", type=int, default=None, help="items per text / mm task (default: the full test set)")
    ap.add_argument("--batch_size", type=int, default=8, help="text suite and RULER prompts per call (swa: 1)")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))


def slug(name):
    return re.sub(r"[^A-Za-z0-9._=-]", "_", name)


def numbers(result: dict) -> dict:
    """``{"acc,none": 0.5, "alias": ...}`` -> ``{"acc": 0.5}``: a harness result's numeric metrics."""
    return {k.split(",")[0]: round(float(v), 4) for k, v in result.items()
            if isinstance(v, (int, float)) and "stderr" not in k and k != "alias"}


# -------------------------------------------------------------------------------- text
def run_text(model, tokenizer, tasks, limit, batch_size):
    """lm-evaluation-harness on the model object.  Loglikelihood batches are right-padded (pads come after
    the scored tokens of a causal model), generation is left-padded with an attention mask."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", batch_size=batch_size)
    groups = {}
    for t in tasks:                               # one harness call per (shots, chat template)
        groups.setdefault((TEXT_FEWSHOT.get(t, 0), t in TEXT_CHAT), []).append(t)
    out = {}
    for (shots, chat), names in groups.items():
        res = lm_eval.simple_evaluate(model=lm, tasks=names, num_fewshot=shots, limit=limit, apply_chat_template=chat,
                                      log_samples=False)
        out.update({t: numbers(res["results"][t]) for t in names})
    return out


# ---------------------------------------------------------------------------------- mm
@contextlib.contextmanager
def public_datasets():
    """lmms-eval's task files ask for ``token: True`` even for public datasets, which fails on a machine that
    is not logged in to the Hub.  Without a stored token, that request is dropped."""
    import datasets
    import huggingface_hub

    load = datasets.load_dataset
    if not huggingface_hub.get_token():
        datasets.load_dataset = lambda *a, **k: load(*a, **({**k, "token": None} if k.get("token") is True else k))
    try:
        yield
    finally:
        datasets.load_dataset = load


def run_mm(model, base_model_dir, tasks, limit, work_dir: Path):
    """lmms-eval with its own Qwen3 wrapper (processor, prompt building, pixel limits, generation settings),
    whose model is exchanged for ours.  The tasks' side files (submissions, per-category results) go to
    ``work_dir``."""
    from lmms_eval.evaluator import simple_evaluate
    from lmms_eval.models.simple.qwen3_vl import Qwen3_VL

    class Wrapper(Qwen3_VL):                      # the wrapper would load the unswapped backbone itself
        def __init__(self, model, **kwargs):
            import transformers

            cls = transformers.Qwen3_5ForConditionalGeneration
            cls.from_pretrained = classmethod(lambda cls, *a, **k: model)
            try:
                super().__init__(**kwargs)
            finally:
                del cls.from_pretrained                # back to the inherited loader

    lm = Wrapper(model, pretrained=str(Path(base_model_dir).resolve()), device=str(model.device),
                 device_map=str(model.device), batch_size=1, enable_thinking=False)
    work_dir.mkdir(parents=True, exist_ok=True)
    with public_datasets(), contextlib.chdir(work_dir):
        res = simple_evaluate(model=lm, tasks=tasks, limit=limit, log_samples=False)
    return {t: numbers(res["results"][t]) for t in tasks if t in res["results"]}


# -------------------------------------------------------------------------------- long
def _ruler(cmd, log_file, env):
    env = dict(os.environ, **env)
    for var in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS"):       # several evaluations on a many-core host
        env.setdefault(var, "4")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # the KV cache grows token by token
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    with open(log_file, "a") as log:
        log.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        log.flush()
        done = subprocess.run([str(c) for c in cmd], cwd=RULER_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    if done.returncode:
        raise RuntimeError(f"RULER command failed (see {log_file}): {' '.join(map(str, cmd))}")


def run_long(spec, base_model_dir, tasks, lengths, samples, out_dir: Path, batch_size):
    """RULER's prepare / predict / evaluate scripts, per length.  Returns ``{"<task>@<length>": {"score": s}}``."""
    _, kernel, ckpt = resolve(spec)
    target = str(ckpt) if ckpt else kernel                      # RULER runs from its own directory
    log_file = out_dir / "ruler.log"
    env = {"LINSWAP_BASE_MODEL_DIR": str(Path(base_model_dir).resolve())}
    out = {}
    for length in lengths:
        data_dir, pred_dir = out_dir / str(length) / "data", out_dir / str(length) / "pred"
        data_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            t0 = time.time()
            _ruler([sys.executable, "data/prepare.py", "--save_dir", data_dir, "--benchmark", "synthetic", "--task", task,
                    "--tokenizer_path", env["LINSWAP_BASE_MODEL_DIR"], "--tokenizer_type", "hf", "--max_seq_length", length,
                    "--model_template_type", "base", "--num_samples", samples], log_file, env)
            if not (data_dir / task / "validation.jsonl").exists():     # prepare.py exits 0 when its generator fails
                raise RuntimeError(f"RULER generated no data for {task} (see {log_file}); its haystacks are fetched "
                                   f"with the download scripts in {RULER_DIR / 'data/synthetic/json'}")
            _ruler([sys.executable, "pred/call_api.py", "--data_dir", data_dir, "--save_dir", pred_dir, "--benchmark",
                    "synthetic", "--task", task, "--server_type", "linswap", "--model_name_or_path", target,
                    "--temperature", "0.0", "--top_k", "32", "--top_p", "1.0", "--batch_size", batch_size], log_file, env)
            print(f"    {task}@{length}: {time.time() - t0:.0f}s", flush=True)
        _ruler([sys.executable, "eval/evaluate.py", "--data_dir", pred_dir, "--benchmark", "synthetic"], log_file, env)
        rows = list(csv.reader(open(max(pred_dir.glob("summary*.csv"), key=lambda p: p.stat().st_mtime))))
        names, scores = (next(r[1:] for r in rows if r and r[0] == key) for key in ("Tasks", "Score"))
        out.update({f"{t}@{length}": {"score": float(s)} for t, s in zip(names, scores)})
    return out


# ------------------------------------------------------------------------------ tables
def headline(metrics: dict):
    """A task's score in percent.  MME's two sums (perception out of 2000, cognition out of 800) are added and
    given as a share of 2800."""
    if any(k.startswith("mme_") for k in metrics):
        return sum(v for k, v in metrics.items() if k.startswith("mme_")) / 28
    name = next((m for m in HEADLINE if m in metrics), None)
    return None if name is None else metrics[name] * (1 if name in HEADLINE[:2] else 100)


def write_markdown(path, table, reference):
    """Per suite one table, in percent: every task, the mean, and the mean's distance to the reference row."""
    lines = []
    for suite in ("text", "long", "mm"):
        rows = {m: {t: headline(r) for t, r in entry[suite].items()} for m, entry in table.items() if suite in entry}
        tasks = list(dict.fromkeys(t for r in rows.values() for t in r))
        if not rows:
            continue

        def mean(r):
            return sum(r[t] for t in tasks) / len(tasks) if all(r.get(t) is not None for t in tasks) else None

        ref = mean(rows[reference]) if reference in rows else None
        lines += [f"**{suite}**", "", "| model | " + " | ".join(tasks) + " | mean | vs " + reference + " |",
                  "|" + "---|" * (len(tasks) + 3)]
        for m, r in rows.items():
            mu = mean(r)
            lines.append(f"| {m} | " + " | ".join("" if r.get(t) is None else f"{r[t]:.1f}" for t in tasks)
                         + f" | {'' if mu is None else f'{mu:.1f}'} | {'' if mu is None or ref is None else f'{mu - ref:+.1f}'} |")
        lines.append("")
    Path(path).write_text("\n".join(lines))


def merge(out_dir, label, update, reference):
    """Add ``update`` to the model's entry in results.json under a lock: several processes may fill one run."""
    with open(out_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = out_dir / "results.json"
        table = json.loads(path.read_text()) if path.exists() else {}
        table[label] = {**table.get(label, {}), **update}
        path.write_text(json.dumps(table, indent=1))
        write_markdown(out_dir / "results.md", table, reference)
    return table


# -------------------------------------------------------------------------------- main
def main(args):
    import torch
    from transformers import AutoTokenizer

    name = args.name or time.strftime("smoke-%Y%m%d-%H%M%S" if args.smoke else "eval-%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    suites = [s for s in args.suites.split(",") if s]
    tasks = {s: (SMOKE[s] if args.smoke else getattr(args, f"{s}_tasks")).split(",") for s in suites}
    limit = SMOKE["limit"] if args.smoke else args.limit
    lengths = [int(x) for x in (SMOKE["lengths"] if args.smoke else args.long_lengths).split(",")]
    samples = SMOKE["samples"] if args.smoke else args.long_samples
    print(f"[eval] {name}: {', '.join(f'{s} ({len(t)} tasks)' for s, t in tasks.items())}"
          f"{' — SMOKE, not a score' if args.smoke else ''} -> {out_dir}")
    table = {}
    for spec in args.models:
        label, kernel, _ = resolve(spec)
        table = merge(out_dir, label, {"kernel": kernel}, args.reference)
        todo = [s for s in suites if s not in table[label]]
        batch = 1 if kernel == "swa" else args.batch_size        # swa cannot mask left padding
        model = None
        for suite in todo:
            t0 = time.time()
            if suite == "long":
                if model is not None:                            # RULER loads its own copy
                    model = None
                    torch.cuda.empty_cache()
                res = run_long(spec, args.base_model_dir, tasks[suite], lengths, samples, out_dir / "ruler" / slug(label), batch)
            else:
                model = model or build_model(spec, args.base_model_dir)
                if suite == "text":
                    tok = AutoTokenizer.from_pretrained(args.base_model_dir)
                    res = run_text(model, tok, tasks[suite], limit, batch)
                else:
                    res = run_mm(model, args.base_model_dir, tasks[suite], limit, out_dir / "mm" / slug(label))
            table = merge(out_dir, label, {suite: res, f"{suite}_seconds": round(time.time() - t0)}, args.reference)
            print(f"  {label} {suite}: " + ", ".join(f"{t}={headline(r):.1f}" for t, r in res.items())
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    print("\n" + (out_dir / "results.md").read_text())
    return table
