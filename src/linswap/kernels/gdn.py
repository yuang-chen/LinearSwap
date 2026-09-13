"""Original Gated DeltaNet via FLA's ``GatedDeltaNet`` layer (control kernel).

Weight copy is exact (no tiling); this reproduces the pretrained
linear layer with the FLA Triton kernels and is the natural baseline for any
other swap."""

from fla.layers.gated_deltanet import GatedDeltaNet

from ..registry import KernelSpec, register_kernel
from .common import copy_, copy_output_gate, copy_qkv_and_conv, get_gdn_source, mark_hf_initialized


def build(cfg, layer_idx):
    return GatedDeltaNet(
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


def init_from_gdn(layer, gdn_state, layer_idx, model_prefix="model"):
    src = get_gdn_source(gdn_state, layer_idx, model_prefix)
    copy_qkv_and_conv(layer, src)
    copy_(layer.a_proj.weight, src["a"], "a_proj")
    copy_(layer.b_proj.weight, src["b"], "b_proj")
    copy_(layer.A_log, src["A_log"], "A_log")
    copy_(layer.dt_bias, src["dt_bias"], "dt_bias")
    copy_output_gate(layer, src)
    mark_hf_initialized(layer)
    return layer


register_kernel(KernelSpec(
    name="gdn",
    description="Original Gated DeltaNet (FLA GatedDeltaNet); exact weight copy, control baseline.",
    build=build,
    init_from_gdn=init_from_gdn,
    new_param_names=("a_proj", "b_proj", "A_log", "dt_bias"),
    exact_init=True,
))
