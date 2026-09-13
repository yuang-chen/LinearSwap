"""Shared helpers for writing kernel init recipes.

The pretrained Gated-DeltaNet layer of the backbone (HF ``Qwen3_5GatedDeltaNet`` /
Qwen3-Next naming) has the following parameters under ``{model_prefix}.layers.{i}.linear_attn.``:

    in_proj_qkv.weight  [2*key_dim + value_dim, hidden]   fused q/k/v projection
    conv1d.weight       [2*key_dim + value_dim, 1, k]     fused depthwise causal conv
    in_proj_a.weight    [num_v_heads, hidden]             per-head scalar decay logits
    in_proj_b.weight    [num_v_heads, hidden]             per-head scalar beta logits
    A_log               [num_v_heads]
    dt_bias             [num_v_heads]
    in_proj_z.weight    [value_dim, hidden]               output gate (swish-gated RMSNorm)
    norm.weight         [head_v_dim]
    out_proj.weight     [hidden, value_dim]

The recurrence is  S_t = exp(g_t) (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T  with
    g_t    = -exp(A_log) * softplus(in_proj_a x + dt_bias)   (scalar per head)
    beta_t = sigmoid(in_proj_b x)                            (scalar per head)
and q/k are L2-normalised inside the kernel with scale 1/sqrt(head_k_dim).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fla.modules import FusedRMSNormSwishGate

GDN_PARAM_NAMES = {
    "qkv": "in_proj_qkv.weight",
    "conv": "conv1d.weight",
    "a": "in_proj_a.weight",
    "b": "in_proj_b.weight",
    "A_log": "A_log",
    "dt_bias": "dt_bias",
    "z": "in_proj_z.weight",
    "norm": "norm.weight",
    "out": "out_proj.weight",
}


def linear_attn_prefix(model_prefix: str, layer_idx: int) -> str:
    return f"{model_prefix}.layers.{layer_idx}.linear_attn."


def get_gdn_source(gdn_state: dict, layer_idx: int, model_prefix: str = "model") -> dict:
    """Return the pretrained GDN tensors of one layer keyed by short names."""
    prefix = linear_attn_prefix(model_prefix, layer_idx)
    return {short: gdn_state[prefix + name] for short, name in GDN_PARAM_NAMES.items()}


def copy_(dst: torch.Tensor, src: torch.Tensor, name: str = "") -> None:
    if dst.shape != src.shape:
        raise ValueError(f"Shape mismatch for {name or 'tensor'}: dst {tuple(dst.shape)} vs src {tuple(src.shape)}")
    with torch.no_grad():
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device))


def tile_rows(w: torch.Tensor, reps: int) -> torch.Tensor:
    """[H, D] -> [H*reps, D]; output row c comes from input row c // reps."""
    return w.repeat_interleave(reps, dim=0)


def tile_vec(v: torch.Tensor, reps: int) -> torch.Tensor:
    """[H] -> [H*reps]; output entry c comes from input entry c // reps."""
    return v.repeat_interleave(reps, dim=0)


def init_lowrank_tiled(seq: nn.Sequential, rows: torch.Tensor, reps: int, name: str = "") -> None:
    """Initialise a rank-r MLP ``Linear(D->r) -> Linear(r->H*reps)`` (no bias) so that it
    computes ``tile_rows(rows, reps) @ x`` exactly: ``W1[:H] = rows`` and ``W2`` is the 0/1
    head selector.  Requires ``r >= H``.  The unused ``r - H`` bottleneck rows keep their
    random init but have zero output weights, so they are invisible at init yet receive
    gradient (through ``W2[:, H:]``) and can be used by fine-tuning."""
    w1, w2 = seq[0], seq[-1]
    H = rows.shape[0]
    r = w1.weight.shape[0]
    if r < H:
        raise ValueError(f"{name}: bottleneck {r} < {H} rows; the tiled projection is not representable")
    if w2.weight.shape[0] != H * reps:
        raise ValueError(f"{name}: output dim {w2.weight.shape[0]} != {H}*{reps}")
    with torch.no_grad():
        w1.weight[:H].copy_(rows.to(w1.weight.dtype))
        sel = torch.zeros(H * reps, r, dtype=w2.weight.dtype, device=w2.weight.device)
        sel[torch.arange(H * reps), torch.arange(H * reps) // reps] = 1.0
        w2.weight.copy_(sel)
        if getattr(w2, "bias", None) is not None:
            w2.bias.zero_()


def copy_qkv_and_conv(layer: nn.Module, src: dict) -> None:
    """Split the fused q/k/v projection of the pretrained layer and depthwise conv into the separate
    ``q_proj/k_proj/v_proj`` and ``q_conv1d/k_conv1d/v_conv1d`` modules used by
    FLA layers.  Depthwise conv + SiLU is channel-wise, so the split is exact."""
    key_dim, value_dim = layer.key_dim, layer.value_dim
    qw, kw, vw = torch.split(src["qkv"], [key_dim, key_dim, value_dim], dim=0)
    copy_(layer.q_proj.weight, qw, "q_proj")
    copy_(layer.k_proj.weight, kw, "k_proj")
    copy_(layer.v_proj.weight, vw, "v_proj")
    cq, ck, cv = torch.split(src["conv"], [key_dim, key_dim, value_dim], dim=0)
    copy_(layer.q_conv1d.weight, cq, "q_conv1d")
    copy_(layer.k_conv1d.weight, ck, "k_conv1d")
    copy_(layer.v_conv1d.weight, cv, "v_conv1d")


def use_qwen_output_gate(layer: nn.Module, hidden_size: int, value_dim: int, head_v_dim: int, eps: float) -> None:
    """Replace a layer's output gate with the backbone's parameterisation:
    full-rank ``g_proj`` (no bias) followed by RMSNorm(o) * SiLU(g)."""
    ref = layer.o_proj.weight
    layer.g_proj = nn.Linear(hidden_size, value_dim, bias=False, device=ref.device, dtype=ref.dtype)
    layer.o_norm = FusedRMSNormSwishGate(head_v_dim, eps=eps, device=ref.device, dtype=ref.dtype)


def copy_output_gate(layer: nn.Module, src: dict) -> None:
    g = layer.g_proj[0] if isinstance(layer.g_proj, nn.Sequential) else layer.g_proj
    copy_(g.weight, src["z"], "g_proj")
    copy_(layer.o_norm.weight, src["norm"], "o_norm")
    copy_(layer.o_proj.weight, src["out"], "o_proj")


def mark_hf_initialized(layer: nn.Module) -> None:
    """Prevent HF-style re-initialisation from touching loaded weights."""
    for m in layer.modules():
        if isinstance(m, nn.Linear):
            m._is_hf_initialized = True
