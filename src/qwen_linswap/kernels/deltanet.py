"""DeltaNet (no decay gate) via FLA's ``DeltaNet`` layer.

DeltaNet (arXiv:2406.06484) is the delta rule without a forget gate:

    S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T

i.e. GDN with ``exp(g_t) ≡ 1``.  The pretrained GDN layers rely on their decay
(``exp(g_t) < 1``), so this swap is **not** function preserving: everything
except the decay branch (``in_proj_a``, ``A_log``, ``dt_bias``) is copied and the
decay is simply dropped.  ``verify.py`` therefore reports a real deviation for
this kernel and SFT has to recover the function.  It is included as the
example of an *inexact* swap in the framework (``exact_init=False``).

q/k/v projections, short convolutions, ``beta`` (``b_proj``), the SiLU-gated
output norm (``use_gate=True``) and ``o_proj`` map one-to-one; the kernel
L2-normalises q/k and uses scale 1/sqrt(head_k_dim) like GDN."""

from fla.layers.delta_net import DeltaNet

from ..registry import KernelSpec, register_kernel
from .common import copy_, copy_output_gate, copy_qkv_and_conv, get_gdn_source, mark_hf_initialized


def build(cfg, layer_idx):
    hidden = cfg["emb_dim"]
    key_dim = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    value_dim = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    if cfg["linear_num_key_heads"] != cfg["linear_num_value_heads"]:
        raise ValueError("FLA DeltaNet has a single num_heads; key/value head counts must match")
    return DeltaNet(
        mode="chunk",
        hidden_size=hidden,
        expand_k=key_dim / hidden,
        expand_v=value_dim / hidden,
        num_heads=cfg["linear_num_key_heads"],
        use_beta=True,
        use_gate=True,
        use_short_conv=True,
        conv_size=cfg["linear_conv_kernel_dim"],
        conv_bias=False,
        qk_activation="silu",
        qk_norm="l2",
        layer_idx=layer_idx,
        norm_eps=cfg.get("rms_norm_eps", 1e-6),
    )


def init_from_gdn(layer, gdn_state, layer_idx, model_prefix="model"):
    src = get_gdn_source(gdn_state, layer_idx, model_prefix)
    copy_qkv_and_conv(layer, src)
    copy_(layer.b_proj.weight, src["b"], "b_proj")
    copy_output_gate(layer, src)
    mark_hf_initialized(layer)
    return layer


register_kernel(KernelSpec(
    name="deltanet",
    description="DeltaNet (FLA DeltaNet, no decay); GDN weights copied, decay branch dropped — NOT function preserving.",
    build=build,
    init_from_gdn=init_from_gdn,
    new_param_names=("b_proj",),
    exact_init=False,
    notes="Inexact swap: expect a large deviation from HF at init; full SFT is required.",
))
