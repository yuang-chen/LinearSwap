"""Swapped mixers inside the upstream vision-language Qwen3.5 model (one GPU, ~2 min; needs torchvision
for the upstream processor).

``linswap.upstream`` replaces only ``linear_attn`` in the model ``transformers`` loads, so the checks are
against that model itself, unswapped, on identical inputs:

  untouched      every tensor outside the swapped layers is still the checkpoint's
  text           logits on text, vs native — judged against the control swap (``gdn``), whose distance
                 from native is the FLA-kernel-vs-upstream-torch noise floor
  image          the same on a real image run through the upstream processor (pixel values, grid,
                 multimodal positions), plus greedy cached generation vs native and vs its own uncached decode
  checkpoint     a text-distilled checkpoint loads strictly into the language model and gives the logits
                 ``linswap.build_model`` gives for it (skipped when no checkpoint is on disk)
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from linswap import DEFAULT_BASE_MODEL_DIR, build_model  # noqa: E402
from linswap.upstream import build_upstream_model, text_model  # noqa: E402

TOLERANCE = 3.0                     # as ``linswap verify``: an exact swap stays within 3x the control's noise
CHECKPOINTS = ("outputs/kda_nolayer/distill/checkpoint-10235", "outputs/kda/distill/checkpoint-16338")
TEXT = ("The quick brown fox jumps over the lazy dog. In 1815 the Congress of Vienna redrew the map of "
        "Europe; meanwhile, steam engines began to transform British industry. ") * 12


def test_image():
    """A picture with content a model can name: a red disc and a blue square on white, 448 x 448."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 60, 200, 220), fill=(220, 30, 30))
    draw.rectangle((250, 240, 410, 400), fill=(30, 60, 220))
    return img


def kl_top1(ref, got):
    ref, got = ref.float(), got.float()
    kl = F.kl_div(F.log_softmax(got, -1), F.log_softmax(ref, -1), log_target=True, reduction="none").sum(-1).mean()
    return kl.item(), (ref.argmax(-1) == got.argmax(-1)).float().mean().item()


@torch.no_grad()
def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ok = True

    def check(cond, what):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {what}")

    device = "cuda"
    proc = AutoProcessor.from_pretrained(DEFAULT_BASE_MODEL_DIR)
    native = AutoModelForImageTextToText.from_pretrained(DEFAULT_BASE_MODEL_DIR, dtype=torch.bfloat16).to(device).eval()
    native_sd = {k: v.clone() for k, v in native.state_dict().items()}

    text = proc.tokenizer(TEXT, return_tensors="pt").to(device)
    messages = [{"role": "user", "content": [{"type": "image", "image": test_image()},
                                             {"type": "text", "text": "Which two shapes are in the picture, and what colour is each?"}]}]
    image = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                     return_tensors="pt").to(device)
    n_img = int((image["input_ids"] == native.config.image_token_id).sum())
    print(f"inputs: text {text['input_ids'].shape[1]} tokens; image prompt {image['input_ids'].shape[1]} tokens, "
          f"{n_img} of them image tokens, grid {image['image_grid_thw'].tolist()}")
    ref_text = native(**text).logits
    ref_image = native(**image).logits
    gen_kw = dict(max_new_tokens=24, do_sample=False)
    ref_gen = native.generate(**image, **gen_kw)[0, image["input_ids"].shape[1]:]
    print(f"native answer: {proc.tokenizer.decode(ref_gen, skip_special_tokens=True)!r}")

    floor = {}
    for kernel in ("gdn", "kda", "rwkv7"):
        print(f"{kernel}: swapped into {type(native).__name__}")
        model = build_upstream_model(kernel, device=device)
        sd = model.state_dict()
        same = [k for k in native_sd if ".linear_attn." not in k]
        check(all(torch.equal(sd[k], native_sd[k]) for k in same) and any(k.startswith("model.visual") for k in same),
              f"untouched: {len(same)} tensors outside the swapped layers are the checkpoint's (vision tower included)")
        for name, inputs, ref in (("text", text, ref_text), ("image", image, ref_image)):
            kl, top1 = kl_top1(ref, model(**inputs).logits)
            if kernel == "gdn":
                floor[name] = kl
                check(kl < 1e-2 and top1 > 0.97, f"{name}: KL vs native {kl:.2e}, top-1 {top1:.4f} (the noise floor)")
            else:
                check(kl <= TOLERANCE * floor[name] and top1 > 0.97,
                      f"{name}: KL vs native {kl:.2e} (control {floor[name]:.2e}), top-1 {top1:.4f}")
        gen = model.generate(**image, **gen_kw)[0, image["input_ids"].shape[1]:]
        nocache = model.generate(**image, use_cache=False, **gen_kw)[0, image["input_ids"].shape[1]:]
        n = min(len(gen), len(ref_gen))
        agree = (gen[:n] == ref_gen[:n]).float().mean().item()
        m = min(len(gen), len(nocache))
        check(agree >= 0.75, f"image: greedy cached generation agrees with native on {agree:.0%} of {n} tokens")
        check((gen[:m] == nocache[:m]).float().mean().item() >= 0.75, "image: cached decode == uncached decode")
        del model
        torch.cuda.empty_cache()

    ckpt = next((REPO / c for c in CHECKPOINTS if (REPO / c / "model.pt").exists()), None)
    if ckpt is None:
        print("checkpoint: no distilled checkpoint on disk, skipped")
    else:
        print(f"checkpoint: {ckpt.relative_to(REPO)} in the upstream model")
        model = build_upstream_model(ckpt_dir=ckpt, device=device)
        ours = build_model(ckpt_dir=ckpt, device=device).eval()
        text_sd, prefix = text_model(model)[0].state_dict(), "model."
        ours_sd = ours.state_dict()
        check(all(torch.equal(text_sd[k[len(prefix):]], v) for k, v in ours_sd.items() if k.startswith(prefix)),
              "every language tensor equals the checkpoint's")
        sd = model.state_dict()
        check(all(torch.equal(sd[k], v) for k, v in native_sd.items() if k.startswith("model.visual")),
              "the vision tower is still the base checkpoint's")
        kl, top1 = kl_top1(ours(text["input_ids"]), model(**text).logits)
        check(kl <= TOLERANCE * floor["text"] and top1 > 0.97,
              f"text logits vs linswap.build_model: KL {kl:.2e} (control noise {floor['text']:.2e}), top-1 {top1:.4f}")
        kl, top1 = kl_top1(ref_image, model(**image).logits)
        gen = model.generate(**image, **gen_kw)[0, image["input_ids"].shape[1]:]
        print(f"  after text-only distillation, on the image prompt: KL vs native {kl:.3f}, top-1 {top1:.3f}; "
              f"answer {proc.tokenizer.decode(gen, skip_special_tokens=True)!r}  (reported, not judged)")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
