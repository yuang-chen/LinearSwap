"""Gated DeltaNet-2 via FLA's ``GatedDeltaNet2`` layer.

GDN2 (arXiv:2605.22791) decouples GDN's scalar beta into a key-side erase gate
``b_proj`` (key_dim channels) and a value-side write gate ``w_proj`` (value_dim
channels) and makes the decay channel-wise (``f_proj``, key_dim channels):

    S_t = (I - k_t (b_t ⊙ k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t ⊙ v_t)^T

With ``b_t = w_t = beta_t·1`` and ``g_t`` constant across channels this is GDN,
so tiling the pretrained scalar projections across channels is exact
(docs/gdn2_swap_notes.md).  As for KDA, GDN2's default low-rank gate MLPs and
sigmoid-gated output norm are replaced by the backbone's parameterisation (dense
``f_proj``/``g_proj``, SiLU-gated RMSNorm) so the init is representable.
New parameters: ~113M (three dense 2048×1024 gates per layer)."""

import torch.nn as nn
from fla.layers.gdn2 import GatedDeltaNet2

from ..registry import KernelSpec, register_kernel
from .common import (
    copy_,
    copy_output_gate,
    copy_qkv_and_conv,
    get_gdn_source,
    mark_hf_initialized,
    tile_rows,
    tile_vec,
    use_qwen_output_gate,
)


def build(cfg, layer_idx):
    layer = GatedDeltaNet2(
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
    ref = layer.q_proj.weight
    layer.f_proj = nn.Linear(layer.hidden_size, layer.key_dim, bias=False, device=ref.device, dtype=ref.dtype)
    use_qwen_output_gate(layer, layer.hidden_size, layer.value_dim, layer.head_v_dim, layer.o_norm.eps)
    return layer


def init_from_gdn(layer, gdn_state, layer_idx, model_prefix="model"):
    src = get_gdn_source(gdn_state, layer_idx, model_prefix)
    copy_qkv_and_conv(layer, src)
    K, V = layer.head_k_dim, layer.head_v_dim
    copy_(layer.f_proj.weight, tile_rows(src["a"], K), "f_proj")      # channel-wise decay
    copy_(layer.A_log, src["A_log"], "A_log")
    copy_(layer.dt_bias, tile_vec(src["dt_bias"], K), "dt_bias")
    copy_(layer.b_proj.weight, tile_rows(src["b"], K), "b_proj")      # erase gate (key side)
    copy_(layer.w_proj.weight, tile_rows(src["b"], V), "w_proj")      # write gate (value side)
    copy_output_gate(layer, src)
    mark_hf_initialized(layer)
    return layer


register_kernel(KernelSpec(
    name="gdn2",
    description="Gated DeltaNet-2 (FLA GatedDeltaNet2); scalar beta/decay tiled into channel-wise b/w/f gates.",
    build=build,
    init_from_gdn=init_from_gdn,
    new_param_names=("b_proj", "w_proj", "f_proj", "A_log", "dt_bias"),
    exact_init=True,
))
