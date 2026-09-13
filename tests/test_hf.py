"""HF integration round-trip (needs one GPU, ~2 min).

    python tests/test_hf.py [kernel] [ckpt_dir]

Checks that LinearSwapForCausalLM.from_swap matches the underlying LinearSwapModel, that
save_pretrained → AutoModelForCausalLM.from_pretrained reproduces its logits exactly, and that
HF generate (greedy, cached) equals the model's own greedy decode.
"""

import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import linswap  # noqa: E402,F401  (registers the Auto classes)
from linswap import DEFAULT_BASE_MODEL_DIR, build_model  # noqa: E402
from linswap.hf import LinearSwapForCausalLM  # noqa: E402


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kernel = sys.argv[1] if len(sys.argv) > 1 else "kda"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL_DIR)
    ids = tok("The quick brown fox jumps over the lazy dog. In 1815 the Congress of Vienna", return_tensors="pt").input_ids.to(device)

    ref = build_model(kernel, ckpt_dir=ckpt, device=device).eval()
    with torch.no_grad():
        ref_logits = ref(ids).float()
        ref_gen = ref.generate(ids, max_new_tokens=16, use_cache=True)[0, ids.shape[1]:]

    hf = LinearSwapForCausalLM.from_swap(kernel, ckpt_dir=ckpt, device=device).eval()
    with torch.no_grad():
        out = hf(ids, use_cache=False)
    same = torch.equal(out.logits.float(), ref_logits)
    print(f"from_swap logits == LinearSwapModel logits: {same}")

    with tempfile.TemporaryDirectory() as tmp:
        hf.save_pretrained(tmp, safe_serialization=True)
        tok.save_pretrained(tmp)
        files = sorted(p.name for p in Path(tmp).iterdir())
        size = sum(p.stat().st_size for p in Path(tmp).glob("*.safetensors")) / 2**30
        print(f"saved: {files} ({size:.2f} GiB of safetensors)")
        del hf
        torch.cuda.empty_cache()
        loaded = AutoModelForCausalLM.from_pretrained(tmp, dtype=torch.bfloat16).to(device).eval()
        print(f"loaded class: {type(loaded).__name__}, kernel={loaded.config.kernel}, tied={loaded.lm_head.weight.data_ptr() == loaded.model.embed_tokens.weight.data_ptr()}")
        with torch.no_grad():
            l2 = loaded(ids, use_cache=False).logits.float()
            gen = loaded.generate(ids, max_new_tokens=16, do_sample=False)[0, ids.shape[1]:]
    same2 = torch.equal(l2, ref_logits)
    same_gen = torch.equal(gen.cpu(), ref_gen.cpu())
    print(f"from_pretrained logits == reference: {same2}")
    print(f"HF generate == model.generate: {same_gen}  -> {tok.decode(gen)!r}")
    ok = same and same2 and same_gen
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
