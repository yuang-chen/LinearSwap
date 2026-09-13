"""Grouped value heads (num_v_heads = 2 * num_heads) in the custom kernels — a synthetic backbone with
random pretrained-style GDN weights: the RWKV-7 kernel (exact) must match FLA's GatedDeltaNet layer,
Mamba-2 must run and cache-decode consistently."""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from linswap import get_kernel  # noqa: E402


def fake_gdn_state(cfg, layer_idx, prefix="model", seed=0):
    g = torch.Generator().manual_seed(seed)
    H, HV, K, V, D = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_key_head_dim"], cfg["linear_value_head_dim"], cfg["emb_dim"]
    kd, vd = H * K, HV * V
    p = f"{prefix}.layers.{layer_idx}.linear_attn."
    r = lambda *s, scale=0.05: torch.randn(*s, generator=g) * scale
    return {p + "in_proj_qkv.weight": r(2 * kd + vd, D), p + "conv1d.weight": r(2 * kd + vd, 1, cfg["linear_conv_kernel_dim"], scale=0.3),
            p + "in_proj_a.weight": r(HV, D), p + "in_proj_b.weight": r(HV, D), p + "A_log": torch.log(torch.rand(HV, generator=g) * 4 + 1),
            p + "dt_bias": r(HV, scale=0.5), p + "in_proj_z.weight": r(vd, D), p + "norm.weight": 1 + r(V, scale=0.1), p + "out_proj.weight": r(D, vd)}


def main():
    dev = torch.device("cuda")
    cfg = {"emb_dim": 256, "linear_num_key_heads": 4, "linear_num_value_heads": 8, "linear_key_head_dim": 32,
           "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6}
    state = fake_gdn_state(cfg, 0)
    layers = {}
    for name in ["gdn", "rwkv7", "mamba2"]:
        spec = get_kernel(name)
        l = spec.build(cfg, 0); spec.init_from_gdn(l, state, 0, "model"); layers[name] = l.to(dev, torch.bfloat16).eval()
    torch.manual_seed(1)
    x = torch.randn(2, 300, cfg["emb_dim"], device=dev, dtype=torch.bfloat16)
    ok = True
    with torch.no_grad():
        outs = {k: l(x)[0].float() for k, l in layers.items()}
    for k, o in outs.items():
        print(f"{k:7s} out {tuple(o.shape)} finite={torch.isfinite(o).all().item()}")
        ok &= tuple(o.shape) == (2, 300, cfg["emb_dim"]) and bool(torch.isfinite(o).all())
    rel = ((outs["rwkv7"] - outs["gdn"]).norm() / outs["gdn"].norm()).item()
    print(f"rwkv7 vs gdn under grouped value heads: rel-L2 {rel:.4f} (exact init: expect < 0.05)")
    ok &= rel < 0.05
    # cached decode consistency for the two custom kernels
    from fla.models.utils import Cache as FLACache
    for k in ["rwkv7", "mamba2"]:
        l = layers[k]
        with torch.no_grad():
            full = l(x)[0].float()
            cache = FLACache()
            a = l(x[:, :200], past_key_values=cache, use_cache=True)[0]
            b = l(x[:, 200:], past_key_values=cache, use_cache=True)[0]
            chunked = torch.cat([a, b], 1).float()
        d = (full - chunked).abs().max().item()
        print(f"{k:7s} prefill+continue vs full: max diff {d:.4f}")
        ok &= d < 0.1
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
