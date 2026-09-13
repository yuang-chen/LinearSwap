"""Kimi Delta Attention (KDA) kernel via FLA's ``KimiDeltaAttention``.

KDA (Kimi Linear, arXiv:2510.26692) is the gated delta rule with a *per-key-
channel* forget gate:

    S_t = (I - beta_t k_t k_t^T) Diag(exp(g_t)) S_{t-1} + beta_t k_t v_t^T,
    g_t[h, :] = -exp(A_log[h]) * softplus(f_proj(x)[h, :] + dt_bias[h, :])   in R^{head_k_dim}

whereas the backbone's GDN uses a scalar gate per head.  A scalar decay commutes
with the delta-rule projector, so tiling the pretrained scalar decay row
across the ``head_k_dim`` channels of each head recovers GDN exactly.  ``beta``,
``A_log``, q/k/v, the short convs and the output path carry over unchanged.

The FLA KDA layer parameterises ``f_proj`` as a low-rank MLP
``hidden -> head_v_dim -> num_v_heads*head_k_dim`` (no bias, no non-linearity).
The tiled decay matrix has rank <= num_v_heads (16) <= head_v_dim (128), so it
is representable *exactly* in that low-rank form:

    W1[:H]  = in_proj_a.weight          (H x hidden)      W1[H:] = small random (trainable slack)
    W2[c,h] = 1 if c // head_k_dim == h else 0             W2[:, H:] = 0

With ``W2[:, H:] = 0`` the extra rank is invisible at init but receives
gradient (grad W2[:, H:] = dL/da ⊗ (W1[H:] x) != 0), so SFT can use it.
Variant ``kda_fullgate`` uses a dense ``f_proj`` instead (more parameters,
same function at init).

KDA's default output gate is a low-rank *sigmoid*-gated norm; the backbone's is a
full-rank *SiLU*-gated norm, which is not representable, so ``g_proj`` and
``o_norm`` are replaced by the backbone's parameterisation.
"""

import torch
import torch.nn as nn
from fla.layers.kda import KimiDeltaAttention

from ..registry import KernelSpec, register_kernel
from .common import (
    copy_,
    copy_output_gate,
    copy_qkv_and_conv,
    get_gdn_source,
    init_lowrank_tiled,
    mark_hf_initialized,
    tile_rows,
    tile_vec,
    use_qwen_output_gate,
)


def _build(cfg, layer_idx, f_proj_mode="lowrank"):
    layer = KimiDeltaAttention(
        hidden_size=cfg["emb_dim"],
        expand_v=1.0,
        head_dim=cfg["linear_key_head_dim"],
        num_heads=cfg["linear_num_key_heads"],
        num_v_heads=cfg["linear_num_value_heads"],
        mode="chunk",
        use_short_conv=True,
        conv_size=cfg["linear_conv_kernel_dim"],
        conv_bias=False,
        layer_idx=layer_idx,
        norm_eps=cfg.get("rms_norm_eps", 1e-6),
    )
    if f_proj_mode == "full":
        ref = layer.q_proj.weight
        layer.f_proj = nn.Linear(layer.hidden_size, layer.gate_dim, bias=False, device=ref.device, dtype=ref.dtype)
    elif f_proj_mode != "lowrank":
        raise ValueError(f"Unknown f_proj_mode {f_proj_mode!r}")
    use_qwen_output_gate(layer, layer.hidden_size, layer.value_dim, layer.head_v_dim, layer.o_norm.eps)
    return layer


def build_lowrank(cfg, layer_idx):
    return _build(cfg, layer_idx, "lowrank")


def build_fullgate(cfg, layer_idx):
    return _build(cfg, layer_idx, "full")


def init_from_gdn(layer, gdn_state, layer_idx, model_prefix="model"):
    src = get_gdn_source(gdn_state, layer_idx, model_prefix)
    copy_qkv_and_conv(layer, src)

    H = layer.num_v_heads
    K = layer.head_k_dim
    a_w = src["a"]  # [H, hidden]
    if isinstance(layer.f_proj, nn.Linear):
        copy_(layer.f_proj.weight, tile_rows(a_w, K), "f_proj")
    else:
        init_lowrank_tiled(layer.f_proj, a_w, K, "f_proj")

    copy_(layer.A_log, src["A_log"], "A_log")
    copy_(layer.dt_bias, tile_vec(src["dt_bias"], K), "dt_bias")
    copy_(layer.b_proj.weight, src["b"], "b_proj")
    copy_output_gate(layer, src)
    mark_hf_initialized(layer)
    return layer


_NEW = ("f_proj", "b_proj", "A_log", "dt_bias")

register_kernel(KernelSpec(
    name="kda",
    description="Kimi Delta Attention (FLA KimiDeltaAttention); scalar decay tiled into the low-rank per-channel gate.",
    build=build_lowrank,
    init_from_gdn=init_from_gdn,
    new_param_names=_NEW,
    exact_init=True,
))

register_kernel(KernelSpec(
    name="kda_fullgate",
    description="KDA with a dense (full-rank) f_proj decay projection instead of the low-rank MLP.",
    build=build_fullgate,
    init_from_gdn=init_from_gdn,
    new_param_names=_NEW,
    exact_init=True,
))
