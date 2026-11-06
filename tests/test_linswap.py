"""Functional test of the whole path (one GPU, ~6 min): every kernel builds inside the upstream model and
runs text and an image; the exact ones reproduce the backbone; a checkpoint round-trips; a few steps of both
distillation phases reduce the gap, with float32 master weights saved as bf16.

    python tests/test_linswap.py
"""

import argparse
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from linswap import DEFAULT_BASE_MODEL_DIR, NATIVE, build_model, distill, get_kernel, list_kernels, save_checkpoint, verify  # noqa: E402
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
    return ap.parse_args(list(argv))


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

    check_kernels(AutoProcessor.from_pretrained(DEFAULT_BASE_MODEL_DIR))
    check_distill()
    ok = all(results)
    print(f"{'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)} checks)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
