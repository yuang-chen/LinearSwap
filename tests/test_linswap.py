"""Functional test of the whole path (one GPU, ~8 min): every kernel builds inside the upstream model and
runs text and an image; the exact ones reproduce the backbone; a checkpoint round-trips; the distillation
stream repeats no image batch and the held-out set covers every kind of input; a step's gradient is the
token-weighted one; a few steps of both distillation phases reduce the gap, with float32 master weights
saved as bf16.

    python tests/test_linswap.py
"""

import argparse
import hashlib
import itertools
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from linswap import DEFAULT_BASE_MODEL_DIR, NATIVE, build_model, data, distill, get_kernel, list_kernels, save_checkpoint, verify  # noqa: E402
from linswap.model import mixers  # noqa: E402

KERNELS = ("gdn", "rwkv7", "kda", "gdn2", "mamba2", "deltanet", "gla", "swa")
OPTIONAL = ("mamba3",)                  # registered only with its kernels installed (the `mamba3` extra)
results = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
    results.append(bool(ok))


def check_kernels(processor):
    print("every kernel inside transformers' Qwen3.5 model, on text and on an image")
    extra = tuple(k for k in OPTIONAL if k in list_kernels())
    check(set(list_kernels()) == set(KERNELS + extra),
          f"the registry holds exactly {', '.join(KERNELS)}" + (f" and the optional {', '.join(extra)}" if extra else ""))
    measured = {}
    for name in (NATIVE,) + KERNELS + extra:
        model = build_model(name)
        measured[name] = verify.measure(model, verify.inputs_for(processor, model.device))
        if name == "gla":
            with tempfile.TemporaryDirectory() as tmp:
                save_checkpoint(model, tmp)
                again = build_model(tmp)
                same = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), again.state_dict().values()))
                check(same and again.linswap_kernel == "gla" and (Path(tmp) / "swap.safetensors").stat().st_size < 1e9,
                      "a checkpoint (swapped layers only) loads back bit-identical")
        del model
        torch.cuda.empty_cache()
    floor = verify.compare(measured[NATIVE], measured["gdn"])
    check(floor["text@512"]["top1"] == 1.0 and floor["generation"]["vs_native"] == 1.0,
          f"gdn on FLA's kernels is the backbone: text KL {floor['text@512']['kl']:.1e}, image KL {floor['image']['kl']:.1e}")
    for name in KERNELS[1:] + extra:
        rows, spec = verify.compare(measured[NATIVE], measured[name]), get_kernel(name)
        finite = all(torch.isfinite(v).all() for k, v in measured[name].items() if k not in ("cached", "uncached"))
        fails = verify.judge(rows, floor, spec.exact_init)
        check(finite and not (spec.exact_init and fails),
              f"{name}: " + (f"exact, image KL {rows['image']['kl']:.1e} within 3x the floor" if spec.exact_init
                             else f"runs; step-0 distance from the teacher: text KL {rows['text@512']['kl']:.1f}, image KL {rows['image']['kl']:.1f}")
              + "".join(f" [{f}]" for f in fails if spec.exact_init))


def distill_args(*argv):
    ap = argparse.ArgumentParser()
    distill.add_args(ap)
    return ap.parse_args(list(argv) + ["--no-decay"])       # the runs here are a handful of steps


def check_data(processor):
    print("distillation data: the stream, the held-out set, and how a step weighs its batches")
    images = [b for kind, b in itertools.islice(data.stream(processor, DEFAULT_BASE_MODEL_DIR, 0), 60) if kind == "image"]
    keys = {hashlib.sha1(b["pixel_values"].float().numpy().tobytes()).hexdigest() for b in images}
    check(len(images) >= 20 and len(keys) == len(images), f"{len(images)} image batches from the two loader workers, no two alike")
    d = distill.Distiller(distill_args("--kernel", "mamba2"))
    kinds = {kind for kind, _ in d.val}
    check(kinds == {"text", "chat", "long"} | {f"image:{s}" for s in data.CAULDRON[1]},
          f"the held-out set has text, chat, a long sequence and image chats of all {len(data.CAULDRON[1])} subsets")
    gap = d.gap()
    check({k for k in gap if k.startswith("kl_")} == {"kl_text", "kl_chat", "kl_image", "kl_long"}
          and all(f"image_{s}" in gap for s in data.CAULDRON[1]), "the gap is the mean of four kinds; the image subsets are reported beside it")

    def grad():
        g = torch.cat([p.grad.flatten().float() for p in d.params if p.grad is not None])
        for p in d.params:
            p.grad = None
        return g

    torch.manual_seed(0)
    short, long = ({"input_ids": torch.randint(1000, 20000, (1, n))} for n in (512, 1536))
    alone = []
    for batch in (short, long):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d.loop_loss({k: v.to(d.device) for k, v in batch.items()})
        alone.append(grad())
    off = {}
    for weight, want in (("tokens", 0.25 * alone[0] + 0.75 * alone[1]), ("batches", 0.5 * alone[0] + 0.5 * alone[1])):
        d.args.weight, d.args.step_tokens = weight, 2048
        d.accumulate(d.loop_loss, iter([("text", short), ("text", long)]))
        g = grad()
        off[weight] = ((g - want).norm() / want.norm()).item()
        off[weight + "_other"] = ((g - (0.5 * alone[0] + 0.5 * alone[1] if weight == "tokens" else 0.25 * alone[0] + 0.75 * alone[1])).norm()
                                  / want.norm()).item()
    check(off["tokens"] < 0.05 and off["batches"] < 0.05 and off["tokens_other"] > 4 * off["tokens"],
          f"a step's gradient weighs batches by tokens (off by {off['tokens']:.1e}; the equal-weight gradient is "
          f"{off['tokens_other']:.1e} away) or equally with --weight batches ({off['batches']:.1e})")
    del d
    torch.cuda.empty_cache()


def check_distill():
    print("distillation: exact kernels are left alone; two phases on mamba2")
    check(distill.main(distill_args("--kernel", "kda")) is None, "kda (exact init) is not trained")
    with tempfile.TemporaryDirectory() as tmp:
        out = distill.main(distill_args("--kernel", "mamba2", "--output_dir", tmp, "--phases", "local",
                                        "--max_tokens", "2e5", "--first_milestone", "1e5", "--step_tokens", "32768"))
        import json
        meta = json.loads((out / "linswap.json").read_text())
        first, last = meta["history"][0]["gap"], meta["gap"]
        check(last < 0.8 * first, f"local phase: gap {first:.2f} -> {last:.2f} in {meta['tokens']:,} tokens")
        kinds = {k for k in meta["history"][0] if k.startswith("kl_")}
        check(kinds == {"kl_text", "kl_chat", "kl_image", "kl_long"}, "the gap is measured on text, chat, images and a long sequence")
        try:
            distill.main(distill_args("--kernel", str(out), "--output_dir", str(out)))
            refused = False
        except SystemExit:
            refused = True
        check(refused, "continuing a checkpoint into its own directory is refused")
        student = build_model(out)
        check(all(p.dtype == torch.bfloat16 for m in mixers(student).values() for p in m.parameters()),
              "the saved layers are bf16")
        del student
        out2 = distill.main(distill_args("--kernel", str(out), "--output_dir", str(Path(tmp) / "loop"), "--phases", "loop",
                                         "--max_tokens", "2e5", "--first_milestone", "1e5", "--step_tokens", "32768"))
        meta2 = json.loads((out2 / "linswap.json").read_text())
        check(abs(meta2["history"][0]["gap"] - last) < 0.05 * last and meta2["gap"] <= meta2["history"][0]["gap"],
              f"loop phase continues from the checkpoint: gap {meta2['history'][0]['gap']:.2f} -> {meta2['gap']:.2f}")


def main():
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(DEFAULT_BASE_MODEL_DIR)
    check_kernels(processor)
    check_data(processor)
    check_distill()
    ok = all(results)
    print(f"{'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)} checks)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
