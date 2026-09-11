"""Kernel-agnostic verification of a swapped Qwen3.5 model against HF Qwen3.5.

Usage:
    python scripts/verify.py --kernel kda [--baseline gdn] [--checks layer,logits,layerwise,cache,generation]
                                  [--lengths 8,64,512,4096]
    python scripts/verify.py --ckpt outputs/sft_kda_full/checkpoint-50 --checks cache,generation

Checks (all use the pretrained Qwen3.5-0.8B weights, function-preserving init):
    layer       pretrained linear layers 0/1/2 in isolation vs transformers'
                Qwen3_5GatedDeltaNet (recurrent path T=64 and chunk path T=1024)
    logits      full model logits vs HF at several sequence lengths
    layerwise   per-block hidden-state diff vs HF (locates a mismatch)
    cache       the swapped model's cached decode vs its own no-cache decode
    generation  greedy text vs HF

``--ckpt DIR`` verifies an SFT checkpoint instead of the base swap (the kernel is
read from its config.json; the ``layer`` check does not apply, and logits will
legitimately differ from HF after fine-tuning — ``cache`` is the useful one).

``--baseline gdn`` runs the same checks for a second kernel so the numbers can
be read against pure kernel/bf16 noise (the ``gdn`` kernel is an exact weight
copy of the original architecture on FLA Triton kernels).
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn.functional as F

from qwen_linswap import QWEN3_5_CONFIG, DEFAULT_BASE_MODEL_DIR, build_model, get_kernel, load_hf_state_dict


def fmt(x):
    return f"{x:.4g}"


# ----------------------------------------------------------------------------- checks
def check_layer(kernel, weights, device, base_model_dir, layers=(0, 1, 2), lengths=(64, 1024)):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    hf_cfg = AutoConfig.from_pretrained(base_model_dir)
    hf_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    spec = get_kernel(kernel)
    cfg = QWEN3_5_CONFIG
    rows = []
    for l in layers:
        ref = Qwen3_5GatedDeltaNet(hf_cfg, l).to(device=device, dtype=torch.bfloat16)
        prefix = f"model.language_model.layers.{l}.linear_attn."
        ref.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)})
        ours = spec.build(cfg, l)
        spec.init_from_gdn(ours, weights, l, "model.language_model")
        ours = ours.to(device=device, dtype=torch.bfloat16).eval()
        for T in lengths:
            torch.manual_seed(l * 1000 + T)
            x = torch.randn(1, T, cfg["emb_dim"], device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                y_ref = ref(x)
                y_ref = (y_ref[0] if isinstance(y_ref, tuple) else y_ref).float()
                y, _, _ = ours(x)
            d = (y.float() - y_ref).abs()
            rows.append((l, T, d.max().item(), d.mean().item(), (d.norm() / y_ref.norm()).item()))
    print(f"  [{kernel}] layer check (max|diff|, mean|diff|, rel-L2) vs HF torch GDN layer")
    for l, T, mx, mn, rel in rows:
        print(f"    layer {l:2d} T={T:5d}: max {fmt(mx):>8} mean {fmt(mn):>8} rel {fmt(rel):>8}")
    return rows


def make_ids(tokenizer, length, device):
    text = ("The quick brown fox jumps over the lazy dog. In 1815 the Congress of Vienna redrew "
            "the map of Europe; meanwhile, steam engines began to transform British industry. ")
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    reps = length // len(ids) + 1
    return ids.repeat(reps)[:length].unsqueeze(0).to(device)


def check_logits(model, hf, tokenizer, lengths, device, kernel):
    print(f"  [{kernel}] logits vs HF")
    rows = []
    for L in lengths:
        ids = make_ids(tokenizer, L, device)
        with torch.no_grad():
            ours = model(ids).float()
            theirs = hf(ids, use_cache=False).logits.float()
        d = (ours - theirs).abs()
        top1 = (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()
        kl = F.kl_div(F.log_softmax(ours, -1), F.log_softmax(theirs, -1), log_target=True, reduction="none").sum(-1).mean().item()
        rows.append((L, d.max().item(), d.mean().item(), top1, kl))
        print(f"    T={L:6d}: max {fmt(d.max().item()):>7} mean {fmt(d.mean().item()):>7} top1 {top1:.4f} KL {fmt(kl)}")
        del ours, theirs, d
        torch.cuda.empty_cache()
    return rows


def check_layerwise(model, hf, tokenizer, device, kernel, length=256):
    ids = make_ids(tokenizer, length, device)
    ours_tr, hf_tr = {}, {}

    def hook(store, name):
        def _h(mod, inp, out):
            store[name] = (out[0] if isinstance(out, tuple) else out).detach().float()
        return _h

    hs = []
    for i, blk in enumerate(model.trf_blocks):
        hs.append(blk.register_forward_hook(hook(ours_tr, i)))
    for i, blk in enumerate(hf.model.layers):
        hs.append(blk.register_forward_hook(hook(hf_tr, i)))
    with torch.no_grad():
        model(ids)
        hf(ids, use_cache=False)
    for h in hs:
        h.remove()
    print(f"  [{kernel}] per-block output diff vs HF (T={length}); L=linear, A=full attention")
    line = []
    for i in range(len(model.trf_blocks)):
        d = (ours_tr[i] - hf_tr[i]).abs()
        tag = "L" if model.trf_blocks[i].layer_type == "linear_attention" else "A"
        line.append(f"{i:02d}{tag}:{d.max().item():.3f}/{d.mean().item():.4f}")
        if len(line) == 4:
            print("    " + "  ".join(line)); line = []
    if line:
        print("    " + "  ".join(line))


def check_cache(model, tokenizer, device, kernel, cases=((5, 20), (120, 20), (4096, 16))):
    print(f"  [{kernel}] cached decode vs no-cache decode")
    for plen, new in cases:
        ids = make_ids(tokenizer, plen, device)
        t = time.time()
        a = model.generate(ids, max_new_tokens=new, use_cache=False)
        b = model.generate(ids, max_new_tokens=new, use_cache=True)
        agree = (a[0, -new:] == b[0, -new:]).float().mean().item()
        print(f"    prompt {plen:5d} + {new} new: agreement {agree:.3f}  ({time.time()-t:.1f}s)  "
              f"cached='{tokenizer.decode(b[0, plen:])[:60]!s}'")


def check_generation(model, hf, tokenizer, device, kernel, prompt="The capital of France is", new=20):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    ours = model.generate(ids, max_new_tokens=new, use_cache=False)[0, ids.shape[1]:]
    with torch.no_grad():
        theirs = hf.generate(ids, max_new_tokens=new, do_sample=False)[0, ids.shape[1]:]
    n = min(len(ours), len(theirs))
    agree = (ours[:n] == theirs[:n]).float().mean().item()
    print(f"  [{kernel}] greedy generation vs HF: token agreement {agree:.3f}")
    print(f"    ours: {tokenizer.decode(ours)!r}")
    print(f"    HF  : {tokenizer.decode(theirs)!r}")


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--ckpt", default=None, help="SFT checkpoint dir (kernel read from its config.json)")
    ap.add_argument("--baseline", default=None, help="second kernel to run for comparison, e.g. gdn")
    ap.add_argument("--checks", default="layer,logits,layerwise,cache,generation")
    ap.add_argument("--lengths", default="8,64,512,4096")
    ap.add_argument("--base_model_dir", default=str(DEFAULT_BASE_MODEL_DIR))
    args = ap.parse_args()
    if args.ckpt:
        from qwen_linswap import read_checkpoint_kernel

        args.kernel = args.kernel or read_checkpoint_kernel(args.ckpt)
    if args.kernel is None:
        ap.error("--kernel or --ckpt is required")

    checks = set(args.checks.split(","))
    if args.ckpt:
        checks.discard("layer")
    lengths = [int(x) for x in args.lengths.split(",")]
    device = torch.device("cuda")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Purpose: check that the swapped model is a function-preserving replacement of HF Qwen3.5 "
          f"(kernel={args.kernel}, baseline={args.baseline}, ckpt={args.ckpt}).")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)
    weights = load_hf_state_dict(args.base_model_dir)
    hf = None
    if checks & {"logits", "layerwise", "generation"}:
        hf = AutoModelForCausalLM.from_pretrained(args.base_model_dir, dtype=torch.bfloat16).to(device).eval()

    for kernel in [k for k in (args.kernel, args.baseline) if k]:
        spec = get_kernel(kernel)
        print(f"\n=== kernel: {kernel} — {spec.description}")
        if not spec.exact_init:
            print(f"  NOTE: init is not function preserving ({spec.notes}); deviations below are expected.")
        if "layer" in checks:
            check_layer(kernel, weights, device, args.base_model_dir)
        model = None
        if checks - {"layer"}:
            ckpt = args.ckpt if kernel == args.kernel else None
            model = build_model(kernel, hf_weights=weights, device=device, ckpt_dir=ckpt).eval()
            n_new = sum(p.numel() for _, p in model.new_parameters())
            print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
                  f"kernel gate/new params: {n_new/1e6:.2f}M ({spec.new_param_names})")
        if "logits" in checks:
            check_logits(model, hf, tokenizer, lengths, device, kernel)
        if "layerwise" in checks:
            check_layerwise(model, hf, tokenizer, device, kernel)
        if "cache" in checks:
            check_cache(model, tokenizer, device, kernel)
        if "generation" in checks:
            check_generation(model, hf, tokenizer, device, kernel)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
