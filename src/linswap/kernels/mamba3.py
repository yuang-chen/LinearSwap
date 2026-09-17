"""Mamba-3 (arXiv:2603.xxxxx, "Mamba-3: Improved sequence modeling using state space principles")
via FLA's ``Mamba3`` layer.  Registered only when ``mamba_ssm``'s Mamba-3 kernels import
(they need Triton ≥ 3.3): ``linswap kernels`` lists it when available.

Mamba-3 is a diagonal SSM with data-dependent decay ``A``, step ``Δ``, a trapezoidal
discretisation ("trap") and a rotary (complex-valued) state, read out through per-head
B / C vectors.  FLA's layer fuses ``[z, x, B, C, dd_dt, dd_A, trap, angles]`` into one
``in_proj`` and has no short convolution.

Init from GDN (inexact, ``exact_init=False``): one SSM group per head so B/C have the
shape of GDN's k/q; ``z ← in_proj_z``, ``x ← v``, ``B ← k``, ``C ← q`` rows of the fused
projection, ``dd_dt ← in_proj_a`` and ``dt_bias ← dt_bias`` so Δ equals GDN's; the
data-dependent ``A`` rows are zeroed with their bias set to the inverse softplus of
``exp(A_log)`` so the decay ``exp(A·Δ)`` starts exactly at GDN's; rotary angles and the
trapezoid coefficient are zeroed (no rotation, neutral mixing); ``B_norm`` / ``C_norm``
weights are set to ``1/√S`` (RMSNorm → GDN's L2 normalisation) with the ``1/√K`` query
scale folded into ``C_norm``; ``B_bias`` / ``C_bias`` and ``D`` start at 0; the per-head
output norm (``is_outproj_norm=True``, SiLU gate) receives the backbone's norm weight
tiled per head, and ``out_proj`` is copied.  What remains different from GDN: no
delta-rule erase, Δ-scaled instead of β-scaled writes, and no short convolution on
q/k/v — so, like Mamba-2, distil before SFT."""

import math

import torch
import torch.nn.functional as F

from .common import copy_

try:
    from fla.layers.mamba3 import Mamba3, is_fast_path_available as _mamba3_available
except Exception:  # pragma: no cover
    Mamba3, _mamba3_available = None, False


def layer_kwargs(cfg):
    hidden = cfg["emb_dim"]
    HV, V, K = cfg["linear_num_value_heads"], cfg["linear_value_head_dim"], cfg["linear_key_head_dim"]
    return dict(state_size=K, expand=HV * V / hidden, head_dim=V, n_groups=HV, rope_fraction=0.5,
                is_outproj_norm=True, use_bias=True)


def init_extra(layer, src):
    I, S, H = layer.intermediate_size, layer.ssm_state_size, layer.num_heads
    key_dim, value_dim = S * layer.n_groups, I
    W, b = layer.in_proj.weight, layer.in_proj.bias
    qw, kw, vw = torch.split(src["qkv"], [key_dim, key_dim, value_dim], dim=0)
    rows = [I, I, key_dim, key_dim, H, H, H, layer.num_rope_angles]
    off = [sum(rows[:i]) for i in range(len(rows) + 1)]
    with torch.no_grad():
        b.zero_()
        W[off[0]:off[1]].copy_(src["z"].to(W.dtype))          # z: output gate
        W[off[1]:off[2]].copy_(vw.to(W.dtype))                # x: values
        W[off[2]:off[3]].copy_(kw.to(W.dtype))                # B: keys (one group per head)
        W[off[3]:off[4]].copy_(qw.to(W.dtype))                # C: queries
        W[off[4]:off[5]].copy_(src["a"].to(W.dtype))          # dd_dt: GDN's decay logits -> Δ
        W[off[5]:off[6]].zero_()                              # dd_A: data-independent at init ...
        a = src["A_log"].float().exp()                        # ... with A = -softplus(bias) = -exp(A_log)
        b[off[5]:off[6]].copy_((a + torch.log(-torch.expm1(-a))).to(b.dtype))
        W[off[6]:off[7]].zero_()                              # trap / angles: neutral
        W[off[7]:off[8]].zero_()
        copy_(layer.dt_bias, src["dt_bias"], "dt_bias")
        layer.B_norm.weight.fill_(S ** -0.5)                  # RMSNorm * 1/sqrt(S) == L2 normalisation
        layer.C_norm.weight.fill_(S ** -0.5 * S ** -0.5)      # ... plus GDN's 1/sqrt(K) query scale (K == S)
        layer.B_bias.zero_()
        layer.C_bias.zero_()
        layer.D.zero_()
        layer.norm.weight.copy_(src["norm"].repeat(H).to(layer.norm.weight.dtype))
        copy_(layer.out_proj.weight, src["out"], "out_proj")
        if layer.out_proj.bias is not None:
            layer.out_proj.bias.zero_()


if _mamba3_available:
    from .fla_layer import register_fla_kernel

    register_fla_kernel(
        "mamba3", Mamba3,
        description="Mamba-3 (FLA Mamba3 on mamba_ssm kernels); GDN decay/projections mapped into the fused "
                    "in_proj, rotary/trapezoid neutral — NOT function preserving.",
        layer_kwargs=layer_kwargs, output_gate="native", copy_shared=False, init_extra=init_extra,
        new_param_names=("dt_bias", "D", "B_bias", "C_bias", "B_norm", "C_norm"), exact_init=False,
        notes="Inexact swap (no erase, Δ-scaled writes, no short conv): distil before SFT.",
        supports_activation_checkpointing=False,   # mamba_ssm kernels re-read ctx.saved_tensors
    )
