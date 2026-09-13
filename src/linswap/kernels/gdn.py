"""Original Gated DeltaNet via FLA's ``GatedDeltaNet`` layer (control kernel).

Weight copy is exact (no tiling); this reproduces the pretrained linear layer
with the FLA Triton kernels and is the natural baseline for any other swap."""

from fla.layers import GatedDeltaNet

from .common import copy_
from .fla_layer import register_fla_kernel


def init_extra(layer, src):
    copy_(layer.a_proj.weight, src["a"], "a_proj")
    copy_(layer.b_proj.weight, src["b"], "b_proj")
    copy_(layer.A_log, src["A_log"], "A_log")
    copy_(layer.dt_bias, src["dt_bias"], "dt_bias")


register_fla_kernel(
    "gdn", GatedDeltaNet,
    description="Original Gated DeltaNet (FLA GatedDeltaNet); exact weight copy, control baseline.",
    output_gate="native",              # FLA's default gate (g_proj + swish-gated RMSNorm) is already the backbone's
    init_extra=init_extra,
    new_param_names=("a_proj", "b_proj", "A_log", "dt_bias"),
    exact_init=True,
)
