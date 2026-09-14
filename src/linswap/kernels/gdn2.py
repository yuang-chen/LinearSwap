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
New parameters: ~113M (three dense 2048×1024 gates per layer).

Limitation: FLA's layer shares ``f``/``b`` across the value heads of a query-head group,
so backbones with grouped value heads (e.g. Qwen3.8-27B, 16 key / 48 value heads) are
rejected at build time — the exact tiled init does not exist there."""

import torch.nn as nn
from fla.layers import GatedDeltaNet2

from .common import copy_, tile_rows, tile_vec
from .fla_layer import register_fla_kernel


def dense_f_proj(layer, cfg):
    """GDN2's default low-rank decay MLP cannot hold the tiled scalar decay exactly; use a dense projection."""
    if layer.num_v_heads != layer.num_heads:
        raise NotImplementedError(
            "gdn2: FLA's GatedDeltaNet2 keeps the decay (f) and erase (b) gates per *key* head and repeats them "
            f"over each group of {layer.num_v_heads // layer.num_heads} value heads, so a GDN backbone with grouped "
            f"value heads ({layer.num_heads} key / {layer.num_v_heads} value heads) has no exact GDN2 image: the "
            "value heads of a group carry different pretrained decays.  Use kda or rwkv7 for GVA backbones.")
    ref = layer.q_proj.weight
    layer.f_proj = nn.Linear(layer.hidden_size, layer.key_dim, bias=False, device=ref.device, dtype=ref.dtype)


def init_extra(layer, src):
    K, V = layer.head_k_dim, layer.head_v_dim
    copy_(layer.f_proj.weight, tile_rows(src["a"], K), "f_proj")      # channel-wise decay
    copy_(layer.A_log, src["A_log"], "A_log")
    copy_(layer.dt_bias, tile_vec(src["dt_bias"], K), "dt_bias")
    copy_(layer.b_proj.weight, tile_rows(src["b"], K), "b_proj")      # erase gate (key side)
    copy_(layer.w_proj.weight, tile_rows(src["b"], V), "w_proj")      # write gate (value side)


register_fla_kernel(
    "gdn2", GatedDeltaNet2,
    description="Gated DeltaNet-2 (FLA GatedDeltaNet2); scalar beta/decay tiled into channel-wise b/w/f gates.",
    post_build=dense_f_proj, init_extra=init_extra,
    new_param_names=("b_proj", "w_proj", "f_proj", "A_log", "dt_bias"),
    exact_init=True,
)
