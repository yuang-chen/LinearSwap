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
Every suite is stored with the code revision, the package versions and the checkpoint's hash; a label whose
checkpoint has changed since is refused, not mixed with the old scores.  The harnesses' per-item records
(reply, extracted answer, item metrics) go to ``samples/<label>/<task>.jsonl``, RULER's to ``ruler/``.

The image suite's multiple-choice tasks are scored by the harness from the reply's text, so a model that
explains its answer loses points a model answering with the bare letter keeps.  ``results.md`` therefore
shows, next to those scores, how each model replies (share of replies that open with an option, mean
length): compare scores of models that reply differently with that in mind.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .manifest import file_sha256, git_revision, package_versions
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
# what an answer looks like in the image tasks that are scored from the reply's text
OPTIONS = {"mme": "(?i:yes|no)", "mmbench_en_dev_static": "[A-E]", "ai2d": "[A-D]", "muirbench": "[A-Z]"}
SAMPLE_DROP = ("doc", "arguments", "input_media", "token_counts", "doc_hash", "prompt_hash", "target_hash", "submission")


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
    ap.add_argument("--mm_batch_size", type=int, default=1, help="image items per call (swa: 1).  1 is lmms-eval's "
                    "default; larger is faster, but padding shifts bf16 rounding and with it some greedy answers")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))


def slug(name):
    return re.sub(r"[^A-Za-z0-9._=-]", "_", name)


def numbers(result: dict) -> dict:
    """``{"acc,none": 0.5, "acc_stderr,none": 0.01, "alias": ...}`` -> ``{"acc": 0.5, "acc_stderr": 0.01}``: a
    harness result's numeric metrics and their standard errors."""
    return {k.split(",")[0]: round(float(v), 4) for k, v in result.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "alias"}


def write_samples(folder: Path, samples: dict):
    """The harness's per-item records, without the documents and prompts: ``<task>.jsonl`` (lm-eval names
    the leaves of a group, so MMLU is one file per subject)."""
    folder.mkdir(parents=True, exist_ok=True)
    for task, rows in samples.items():
        with open(folder / f"{task}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps({k: v for k, v in row.items() if k not in SAMPLE_DROP}, ensure_ascii=False,
                                   default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


def reply_format(samples, option) -> dict:
    """How a model answers a multiple-choice task: the share of replies that are nothing but an option, the
    share that open with one, and the mean length in characters."""
    def text(resp):
        while isinstance(resp, (list, tuple)) and resp:
            resp = resp[0]
        return str(resp)

    replies = [text(s["resps"]) for s in samples] or [""]
    bare, lead = re.compile(rf"\s*\(?{option}\)?[.:]?\s*"), re.compile(rf"\s*\(?{option}([.):,]|\s*$)")
    return {"reply_bare": round(sum(bool(bare.fullmatch(r)) for r in replies) / len(replies), 4),
            "reply_leading": round(sum(bool(lead.match(r)) for r in replies) / len(replies), 4),
            "reply_chars": round(sum(map(len, replies)) / len(replies), 1)}


# -------------------------------------------------------------------------------- text
def run_text(model, tokenizer, tasks, limit, batch_size, samples_dir: Path):
    """lm-evaluation-harness on the model object.  Loglikelihood batches are right-padded (pads come after
    the scored tokens of a causal model), generation is left-padded with an attention mask."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", batch_size=batch_size, enable_thinking=False)
    groups = {}
    for t in tasks:                               # one harness call per (shots, chat template)
        groups.setdefault((TEXT_FEWSHOT.get(t, 0), t in TEXT_CHAT), []).append(t)
    out = {}
    for (shots, chat), names in groups.items():
        res = lm_eval.simple_evaluate(model=lm, tasks=names, num_fewshot=shots, limit=limit, apply_chat_template=chat,
                                      log_samples=True)
        write_samples(samples_dir, res["samples"])
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


def run_mm(model, base_model_dir, tasks, limit, work_dir: Path, samples_dir: Path, batch_size=1):
    """lmms-eval with its own Qwen3 wrapper (processor, prompt building, pixel limits, generation settings),
    whose model is exchanged for ours.  The tasks' side files (submissions, per-category results) go to
    ``work_dir``."""
    import lmms_eval.evaluator as evaluator
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

    work_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    media = evaluator.is_multimodal_content        # a logged sample keeps its document minus the media, but a *list*
    evaluator.is_multimodal_content = lambda v: media(v) or (isinstance(v, (list, tuple)) and any(map(media, v)))
    try:                                           # of images (MuirBench) passes that test and would stay in memory
        with public_datasets(), contextlib.chdir(work_dir):
            for task in tasks:          # one call per task: the wrapper reads a batch's images with its first item's
                # task, and the harness strips the model from the wrapper when a call ends
                lm = Wrapper(model, pretrained=str(Path(base_model_dir).resolve()), device=str(model.device),
                             device_map=str(model.device), batch_size=batch_size, enable_thinking=False)
                res = simple_evaluate(model=lm, tasks=[task], limit=limit, log_samples=True)
                if task in res["results"]:
                    out[task] = numbers(res["results"][task])
                    write_samples(samples_dir, res["samples"])
                    if task in OPTIONS:
                        out[task].update(reply_format(res["samples"][task], OPTIONS[task]))
    finally:
        evaluator.is_multimodal_content = media
    return out


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
        return sum(v for k, v in metrics.items() if k.startswith("mme_") and k.endswith("_score")) / 28
    name = next((m for m in HEADLINE if m in metrics), None)
    return None if name is None else metrics[name] * (1 if name in HEADLINE[:2] else 100)


def stderr(metrics: dict):
    """The harness's standard error of the headline score, in percent — None where it gives none (RULER, and
    the scores computed over groups of items: MME, MMBench, MuirBench)."""
    name = next((m for m in HEADLINE if m in metrics), None)
    if name is None or any(k.startswith("mme_") for k in metrics):
        return None
    se = metrics.get(name + "_stderr", metrics.get(name + "_stderr_clt"))
    return se * (1 if name in HEADLINE[:2] else 100) if se else None           # 0 is "not estimated" (OCRBench)


def write_markdown(path, table, reference):
    """Per suite a table in percent (RULER: one per length): every task with its distance to the reference
    row in brackets — starred when it is more than two standard errors of the difference, where the harness
    gives standard errors — and the mean.  The mean is no verdict: read the tasks."""
    lines = []
    for suite in ("text", "long", "mm"):
        entries = {m: e[suite] for m, e in table.items() if suite in e}
        rows = {m: {t: headline(r) for t, r in e.items()} for m, e in entries.items()}
        errs = {m: {t: stderr(r) for t, r in e.items()} for m, e in entries.items()}
        ref, ref_err = rows.get(reference, {}), errs.get(reference, {})
        every = list(dict.fromkeys(t for r in rows.values() for t in r))
        for block in dict.fromkeys(t.partition("@")[2] for t in every):       # "" for text and mm, a length for RULER
            tasks = [t for t in every if t.partition("@")[2] == block]

            def mean(r):
                return sum(r[t] for t in tasks) / len(tasks) if all(r.get(t) is not None for t in tasks) else None

            def cell(m, t):
                v, base = rows[m].get(t), ref.get(t)
                if v is None or m == reference or base is None:
                    return "" if v is None else f"{v:.1f}"
                se, se_ref = errs[m].get(t), ref_err.get(t)
                star = "*" if se is not None and se_ref is not None and abs(v - base) > 2 * math.hypot(se, se_ref) else ""
                return f"{v:.1f} ({v - base:+.1f}{star})"

            lines += [f"**{suite}{' @ ' + block if block else ''}**", "",
                      "| model | " + " | ".join(t.partition("@")[0] for t in tasks) + " | mean |", "|" + "---|" * (len(tasks) + 2)]
            for m, r in rows.items():
                mu, base = mean(r), mean(ref)
                lines.append(f"| {m} | " + " | ".join(cell(m, t) for t in tasks) + " | " + ("" if mu is None else f"{mu:.1f}" + (
                    "" if m == reference or base is None else f" ({mu - base:+.1f})")) + " |")
                if m == reference and any(ref_err.get(t) is not None for t in tasks):
                    lines.append("| ± (standard error) | " + " | ".join(
                        "" if ref_err.get(t) is None else f"{ref_err[t]:.1f}" for t in tasks) + " | |")
            lines.append("")
        asked = [t for t in every if any("reply_leading" in e.get(t, {}) for e in entries.values())]
        if asked:               # scored from the reply's text: how each model replies
            lines += ["How the models reply where the answer is read from the reply's text — share of replies that open "
                      "with an option / mean characters:", "", "| model | " + " | ".join(asked) + " |", "|" + "---|" * (len(asked) + 1)]
            lines += [f"| {m} | " + " | ".join(f"{e[t]['reply_leading']:.0%} / {e[t]['reply_chars']:.0f}" if "reply_leading" in e.get(t, {})
                                               else "" for t in asked) + " |" for m, e in entries.items()]
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
    table, code = {}, {**git_revision(), "packages": package_versions()}
    for spec in args.models:
        label, kernel, ckpt = resolve(spec)
        sha = file_sha256(ckpt / "swap.safetensors") if ckpt and (ckpt / "swap.safetensors").exists() else None
        table = merge(out_dir, label, {"kernel": kernel}, args.reference)
        for s in suites:        # scores (and RULER predictions, which its scripts resume) of another checkpoint
            run, started = table[label].get(f"{s}_run"), (out_dir / "ruler" / slug(label)).exists() and s == "long"
            if sha and (run["swap_sha256"] != sha if run else started):
                raise SystemExit(f"eval: {out_dir.name} holds {s} results of {label!r} for another checkpoint than {ckpt} is "
                                 f"now ({(run or {}).get('swap_sha256') or 'not recorded'} / {sha}): use another --name, "
                                 f"or another label (--models <label>={ckpt})")
        todo = [s for s in suites if s not in table[label]]
        batch = 1 if kernel == "swa" else args.batch_size        # swa cannot mask left padding
        model = None
        for suite in todo:
            t0 = time.time()
            table = merge(out_dir, label, {f"{suite}_run": {**code, "swap_sha256": sha, "limit": limit}}, args.reference)
            samples_dir = out_dir / "samples" / slug(label)
            if suite == "long":
                if model is not None:                            # RULER loads its own copy
                    model = None
                    torch.cuda.empty_cache()
                res = run_long(spec, args.base_model_dir, tasks[suite], lengths, samples, out_dir / "ruler" / slug(label), batch)
            else:
                model = model or build_model(spec, args.base_model_dir)
                if suite == "text":
                    tok = AutoTokenizer.from_pretrained(args.base_model_dir)
                    res = run_text(model, tok, tasks[suite], limit, batch, samples_dir)
                else:
                    res = run_mm(model, args.base_model_dir, tasks[suite], limit, out_dir / "mm" / slug(label), samples_dir,
                                 1 if kernel == "swa" else args.mm_batch_size)
            table = merge(out_dir, label, {suite: res, f"{suite}_seconds": round(time.time() - t0)}, args.reference)
            print(f"  {label} {suite}: " + ", ".join(f"{t}={headline(r):.1f}" for t, r in res.items())
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    print("\n" + (out_dir / "results.md").read_text())
    return table
