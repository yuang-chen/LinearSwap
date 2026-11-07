"""verify: is the swap a faithful replacement of the backbone, on text and on images?

    linswap verify --kernel kda
    linswap verify --kernel outputs/distill/gla          # a distilled checkpoint: does its cache decode right?

The swapped model is compared with the unmodified upstream model on identical inputs: a text passage at
512 and 4096 tokens, and a chat prompt with an image run through the upstream processor.  The ``gdn``
swap — the same recurrence on FLA's kernels instead of upstream's torch code — measures the numerical
noise floor of the comparison in the same run.

  logits       KL(native ‖ swapped) and top-1 agreement on each input
  generation   greedy decoding of the image prompt: agreement with native, and with its own uncached decode

A kernel whose init is exact must stay within ``--tolerance`` (3x) of the floor, and ``verify`` exits
non-zero when it does not.  For an inexact kernel the numbers show how far it starts from the teacher; they
are reported, and only a checkpoint's cached decoding is judged (an untrained inexact model emits noise, on
which cached and uncached decoding need not agree).

These inputs test exactness; they are not the distance of a distilled checkpoint.  The passage is two
sentences repeated, which any trained student predicts (KL 1e-3 for a checkpoint whose held-out text KL is
5e-2), and the image row counts the image-token positions, where nothing is predicted and which the gap of
``linswap distill`` leaves out (KL 0.4 for the same checkpoint, held-out image chats 0.06).  For a
checkpoint the distance is that gap and ``linswap eval``.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from .model import DEFAULT_BASE_MODEL_DIR, NATIVE, build_model, resolve
from .registry import get_kernel

PASSAGE = ("The quick brown fox jumps over the lazy dog. In 1815 the Congress of Vienna redrew the map of "
           "Europe; meanwhile, steam engines began to transform British industry. ")
QUESTION = "Which two shapes are in the picture, and what colour is each?"
NEW_TOKENS = 24


def add_args(ap):
    ap.add_argument("--kernel", required=True, help="kernel, or a checkpoint directory")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))
    ap.add_argument("--tolerance", type=float, default=3.0, help="allowed multiple of the gdn noise floor")
    ap.add_argument("--json", default=None, help="write the result here")


def test_image():
    """A picture with content a model can name: a red disc and a blue square on white, 448 x 448."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 60, 200, 220), fill=(220, 30, 30))
    draw.rectangle((250, 240, 410, 400), fill=(30, 60, 220))
    return img


def inputs_for(processor, device):
    """``{name: model inputs}``: text at two lengths and an image chat prompt."""
    ids = processor.tokenizer(PASSAGE * 200, return_tensors="pt").input_ids
    messages = [{"role": "user", "content": [{"type": "image", "image": test_image()}, {"type": "text", "text": QUESTION}]}]
    image = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                          return_tensors="pt", enable_thinking=False)
    return {"text@512": {"input_ids": ids[:, :512].to(device)}, "text@4096": {"input_ids": ids[:, :4096].to(device)},
            "image": dict(image.to(device))}


@torch.no_grad()
def measure(model, inputs):
    """Logits per input and the greedy answer to the image prompt, cached and uncached."""
    out = {name: model(**x).logits[0].float() for name, x in inputs.items()}
    n = inputs["image"]["input_ids"].shape[1]
    out["cached"] = model.generate(**inputs["image"], max_new_tokens=NEW_TOKENS, do_sample=False)[0, n:]
    out["uncached"] = model.generate(**inputs["image"], max_new_tokens=NEW_TOKENS, do_sample=False, use_cache=False)[0, n:]
    return out


def agreement(a, b):
    n = min(len(a), len(b))
    return (a[:n] == b[:n]).float().mean().item()


def compare(ref, got):
    rows = {}
    for name in ref:
        if name in ("cached", "uncached"):
            continue
        kl = F.kl_div(F.log_softmax(got[name], -1), F.log_softmax(ref[name], -1), log_target=True, reduction="none").sum(-1)
        rows[name] = {"kl": kl.mean().item(), "top1": (ref[name].argmax(-1) == got[name].argmax(-1)).float().mean().item()}
    rows["generation"] = {"vs_native": agreement(got["cached"], ref["cached"]),
                          "cached_vs_uncached": agreement(got["cached"], got["uncached"])}
    return rows


def judge(rows, floor, exact, tolerance=3.0, cache=True):
    """Reasons a model fails: the logits and the generation of an exact swap, the cache if ``cache``."""
    fails = []
    if cache and not rows["generation"]["cached_vs_uncached"] >= 0.75:
        fails.append(f"cached and uncached decoding agree on {rows['generation']['cached_vs_uncached']:.0%} of the tokens")
    if exact:
        for name, r in rows.items():
            if name == "generation":
                if not r["vs_native"] >= floor[name]["vs_native"] - 0.25:
                    fails.append(f"generation agrees with native on {r['vs_native']:.0%} (gdn: {floor[name]['vs_native']:.0%})")
            elif not r["kl"] <= tolerance * floor[name]["kl"] + 1e-7:      # also catches NaN
                fails.append(f"{name}: KL {r['kl']:.2e} is over {tolerance:g}x the gdn floor {floor[name]['kl']:.2e}")
    return fails


def main(args):
    from transformers import AutoProcessor

    label, kernel, ckpt = resolve(args.kernel)
    spec, exact = get_kernel(kernel), get_kernel(kernel).exact_init and ckpt is None
    processor = AutoProcessor.from_pretrained(args.base_model_dir)
    measured = {}
    for name in dict.fromkeys((NATIVE, "gdn", args.kernel)):
        model = build_model(name, args.base_model_dir)
        measured[name] = measure(model, inputs_for(processor, model.device))
        del model
        torch.cuda.empty_cache()
    floor = compare(measured[NATIVE], measured["gdn"])
    rows = compare(measured[NATIVE], measured[args.kernel])
    fails = judge(rows, floor, exact, args.tolerance, cache=exact or ckpt is not None)
    print(f"[verify] {label} (kernel {spec.name}, exact_init={spec.exact_init}) against the unmodified backbone; gdn = noise floor")
    for name, r in rows.items():
        if name == "generation":
            print(f"  generation  agrees with native on {r['vs_native']:.0%} of {NEW_TOKENS} tokens (gdn {floor[name]['vs_native']:.0%}); "
                  f"cached vs uncached {r['cached_vs_uncached']:.0%}")
        else:
            print(f"  {name:10s}  KL {r['kl']:.2e} (gdn {floor[name]['kl']:.2e})  top-1 {r['top1']:.4f} (gdn {floor[name]['top1']:.4f})")
    status = "fail" if fails else "pass" if exact else "not_gated"
    print(f"[verify] {status.upper()}" + "".join(f"\n  {f}" for f in fails)
          + ("" if exact else "  (the logit numbers are the distance from the teacher, reported and not judged)"))
    result = {"model": label, "kernel": spec.name, "exact_init": spec.exact_init, "status": status, "fails": fails, "rows": rows, "floor": floor}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    if fails:
        raise SystemExit(1)
    return result
