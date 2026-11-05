"""Run bookkeeping (CPU, no model, ~1 s): the stage plan and its token totals, the verify acceptance
gate, the tokenizer check of a prepared corpus and the run manifest.

The cases are the ones that went wrong silently: ``tokens`` in the log was the global step times the
*current* step's tokens per update (a default run ended at 25.7B "tokens" for 699M consumed), a batch
that the micro-batch does not divide was floored, ``verify`` printed its numbers without judging them,
and a tokenised corpus was reused for any backbone.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from linswap.manifest import tokenizer_fingerprint, write_manifest  # noqa: E402
from linswap.pipeline import distill, verify  # noqa: E402
from linswap.textdata import check_tokenizer  # noqa: E402

results = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    results.append(bool(ok))


def distill_args(*argv):
    ap = argparse.ArgumentParser()
    distill.add_args(ap)
    return ap.parse_args(["--kernel", "gdn", *argv])


def exits(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except SystemExit as e:
        return str(e) or "exit"
    return None


def check_plan():
    print("stage plan and token totals")
    plan = distill.stage_table(distill_args(), ["layer", "kl", "ce"])
    check([r["steps"] for r in plan] == [6103, 10172, 63], "default plan: 6103 / 10172 / 63 optimizer steps")
    check([r["start_step"] for r in plan] == [0, 6103, 16275], "nominal first steps 0 / 6103 / 16275")
    check([r["tokens"] for r in plan] == [99_991_552, 499_974_144, 99_090_432],
          "tokens per step x steps: 99,991,552 / 499,974,144 / 99,090,432")
    check([r["data_seed"] for r in plan] == [42, 42 + 6103, 42 + 16275], "data seed = seed + nominal first step")

    nolayer = distill.stage_table(distill_args(), ["kl", "ce"])
    check(sum(r["tokens"] for r in nolayer) == 599_064_576, "kl + ce consume 599,064,576 tokens (the log said 16.1B)")

    # a checkpoint written before train_state.json: the plan, minus a layer step the log says was skipped
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "checkpoint-16275"
        ckpt.mkdir()
        ran = distill.resume_counters(ckpt, plan, 16275)
        check(ran["tokens"] == 99_991_552 + 499_974_144 and ran["updates"] == 16275
              and ran["teacher_tokens"] == ran["tokens"], "resume without a log: layer + kl counted")
        (Path(d) / "train_log.jsonl").write_text(json.dumps({"stage": "layer", "step": 0, "skipped": True}) + "\n")
        skipped = distill.resume_counters(ckpt, plan, 16275)
        check(skipped["tokens"] == 499_974_144 and skipped["updates"] == 10172, "skipped layer step counts for nothing")
        mid = distill.resume_counters(ckpt, plan, 6103 + 2000)
        check(mid["tokens"] == 2000 * 96 * 512 and mid["updates"] == 2000, "partial kl step: 2000 updates")
        (ckpt / "train_state.json").write_text(json.dumps({"updates": 7, "tokens": 11, "teacher_tokens": 5}))
        own = distill.resume_counters(ckpt, plan, 16275)
        check((own["updates"], own["tokens"], own["teacher_tokens"]) == (7, 11, 5), "train_state.json wins when present")

    check(distill.stage_plan(distill_args("--stage_micro", "ce:12"), "ce")[1:3] == (12, 8), "96 / 12 -> accumulate 8")
    msg = exits(distill.stage_plan, distill_args("--stage_micro", "ce:10"), "ce")
    check(msg is not None and "not a multiple" in msg, "96 sequences with micro-batch 10 is refused, not floored to 90")
    check(distill.stage_plan(distill_args("--stage_batch", "kl:4", "--stage_micro", "kl:96"), "kl")[1:3] == (4, 1),
          "a micro-batch above the batch is clamped to it")


def check_judge():
    print("verify acceptance gate")
    base = {"layer": [{"layer": 0, "T": 64, "rel": 7.0e-3}], "logits": [{"T": 512, "kl": 8.5e-5}],
            "cache": [{"prompt": 5, "new": 20, "agreement": 1.0}],
            "generation": [{"prompt": "p", "new": 20, "agreement": 1.0}]}
    same = json.loads(json.dumps(base))
    same["logits"][0]["kl"] = 1.0e-4            # the spread between exact kernels on one machine: ~1.2x
    check(verify.judge(same, base) == [], "an exact kernel at 1.2x the baseline passes")
    off = json.loads(json.dumps(base))
    off["layer"][0]["rel"] = 1.4                # mamba2's init
    off["logits"][0]["kl"] = 5.6
    off["generation"][0]["agreement"] = 0.05
    fails = verify.judge(off, base)
    check(len(fails) == 3 and all(any(c in f for f in fails) for c in ("layer", "logits", "generation")),
          f"an inexact init fails layer, logits and generation ({len(fails)} failures)")
    nan = json.loads(json.dumps(base))
    nan["logits"][0]["kl"] = float("nan")
    check(len(verify.judge(nan, base)) == 1, "NaN fails")
    zero = {"layer": [{"layer": 0, "T": 64, "rel": 0.0}]}
    check(verify.judge(zero, zero) == [], "a baseline at exactly 0 does not fail itself")
    check(verify.judge({"logits": [{"T": 8, "kl": 9.0}]}, base) == [], "rows the baseline did not measure are not judged")


def check_tokenizer_identity():
    print("tokenizer identity of a prepared corpus")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, vocab in (("tok_a", "a b c"), ("tok_b", "a b d")):
            (d / name).mkdir()
            (d / name / "tokenizer.json").write_text(vocab)
        (d / "tok_a_copy").mkdir()
        (d / "tok_a_copy" / "tokenizer.json").write_text("a b c")
        data = d / "corpus"
        data.mkdir()

        def info(**kw):
            (data / "info.json").write_text(json.dumps({"corpus": "x", **kw}))

        info(tokenizer=str(d / "tok_a"), tokenizer_sha256=tokenizer_fingerprint(d / "tok_a"))
        check(exits(check_tokenizer, data, d / "tok_a") is None, "same tokenizer passes")
        check(exits(check_tokenizer, data, d / "tok_a_copy") is None, "same tokenizer under another path passes")
        msg = exits(check_tokenizer, data, d / "tok_b")
        check(msg is not None and "tokenised with" in msg, "another tokenizer is refused")

        info(tokenizer=str(d / "tok_a"))        # prepared before fingerprints were recorded
        check(exits(check_tokenizer, data, d / "tok_b") is not None, "legacy corpus: another tokenizer is refused")
        check(exits(check_tokenizer, data, d / "tok_a") is None
              and json.loads((data / "info.json").read_text()).get("tokenizer_sha256") == tokenizer_fingerprint(d / "tok_a"),
              "legacy corpus: matching tokenizer passes and its fingerprint is recorded")


def check_manifest():
    print("run manifest")
    with tempfile.TemporaryDirectory() as d:
        plan = distill.stage_table(distill_args(), ["kl", "ce"])
        m = write_manifest(Path(d) / "manifest.json", kernel="kda", stages=plan)
        back = json.loads((Path(d) / "manifest.json").read_text())
        check(back["stages"] == plan and back["kernel"] == "kda", "manifest round-trips the plan")
        check("torch" in m["packages"] and "git_commit" in m["code"], "records package versions and the code revision")
        fla = m["packages"].get("flash-linear-attention", "")
        check("+git." in fla, f"flash-linear-attention carries its git commit ({fla})")


def main():
    check_plan()
    check_judge()
    check_tokenizer_identity()
    check_manifest()
    ok = all(results)
    print(f"{'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)} checks)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
