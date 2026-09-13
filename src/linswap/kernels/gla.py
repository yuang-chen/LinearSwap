"""Gated Linear Attention (GLA, arXiv:2312.06635) via FLA's ``GatedLinearAttention`` — the
"stock FLA layer, zero custom code" example.

GLA is linear attention with a data-dependent per-key-channel decay and no delta rule:
``S_t = Diag(exp(gk_t)) S_{t-1} + k_t v_t^T``, ``gk_t = logsigmoid(gk_proj(x_t)) / 16``.
Relative to GDN it lacks the erase term and its decay parameterisation cannot represent
GDN's, and it does not L2-normalise q/k, so the init is inexact: the shared weights
(q/k/v, convolutions, the output gate — GLA's ``g_proj`` + ``g_norm_swish_gate`` is the
backbone's gate under another name) are copied and the decay MLP keeps FLA's init.
Distillation is required."""

from fla.layers import GatedLinearAttention

from .fla_layer import register_fla_kernel


def layer_kwargs(cfg):
    hidden = cfg["emb_dim"]
    if cfg["linear_num_key_heads"] != cfg["linear_num_value_heads"]:
        raise ValueError("GLA has a single num_heads; key/value head counts must match")
    return dict(
        expand_k=cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"] / hidden,
        expand_v=cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"] / hidden,
        num_heads=cfg["linear_num_key_heads"],
        use_short_conv=True, use_output_gate=True, gate_fn="swish", fuse_norm=True,
    )


register_fla_kernel(
    "gla", GatedLinearAttention,
    description="Gated Linear Attention (FLA GatedLinearAttention); shared weights copied, decay MLP at FLA init — NOT function preserving.",
    layer_kwargs=layer_kwargs, output_gate="native", norm_attr="g_norm_swish_gate",
    new_param_names=("gk_proj",), exact_init=False,
    notes="Inexact swap (no erase, different decay, no q/k normalisation): distil before SFT.",
)
