"""Fast regression test for the qwen_linswap framework (needs one GPU, ~1-2 min).

    python tests/test_kernels.py

Checks, for every registered kernel:
  * the model builds, loads the pretrained HF weights and its logits agree with the
    `gdn` control kernel (top-1 identical, small KL) on a short prompt;
  * a native checkpoint round-trip (state_dict -> model.pt -> build_model(ckpt_dir=...))
    reproduces the same logits bit-for-bit and records the kernel in config.json.
"""

import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_linswap import build_model, get_kernel, list_kernels, load_hf_state_dict  # noqa: E402


def main():
    device = torch.device("cuda")
    weights = load_hf_state_dict()
    from transformers import AutoTokenizer
    from qwen_linswap import DEFAULT_BASE_MODEL_DIR

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL_DIR)
    text = ("The quick brown fox jumps over the lazy dog. In 1815 the Congress of Vienna redrew the map of "
            "Europe; meanwhile, steam engines began to transform British industry. ") * 3
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    ref = build_model("gdn", hf_weights=weights, device=device).eval()
    with torch.no_grad():
        ref_logits = ref(ids).float()
    del ref
    failures = []
    for name in list_kernels():
        spec = get_kernel(name)
        model = build_model(name, hf_weights=weights, device=device).eval()
        with torch.no_grad():
            logits = model(ids).float()
        top1 = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
        kl = F.kl_div(F.log_softmax(logits, -1), F.log_softmax(ref_logits, -1), log_target=True,
                      reduction="none").sum(-1).mean().item()
        n_new = sum(p.numel() for _, p in model.new_parameters())
        ok = top1 >= 0.98 and kl < 5e-3 if spec.exact_init else True

        with tempfile.TemporaryDirectory() as tmp:
            torch.save(model.state_dict(), Path(tmp) / "model.pt")
            with open(Path(tmp) / "config.json", "w") as f:
                json.dump({"linear_kernel": name}, f)
            reloaded = build_model(ckpt_dir=tmp, hf_weights=weights, device=device).eval()
            with torch.no_grad():
                logits2 = reloaded(ids).float()
            roundtrip = torch.equal(logits, logits2)
            del reloaded
        ok = ok and roundtrip
        print(f"{'PASS' if ok else 'FAIL'} {name:13s} new params {n_new/1e6:6.2f}M  top1-vs-gdn {top1:.3f}  KL {kl:.2e}  "
              f"checkpoint round-trip {'exact' if roundtrip else 'MISMATCH'}")
        if not ok:
            failures.append(name)
        del model
        torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f"FAILED: {failures}")
    print("all kernels passed")


if __name__ == "__main__":
    main()
