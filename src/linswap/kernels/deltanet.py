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

from fla.layers import DeltaNet

from .common import copy_
from .fla_layer import register_fla_kernel


def layer_kwargs(cfg):
    hidden = cfg["emb_dim"]
    if cfg["linear_num_key_heads"] != cfg["linear_num_value_heads"]:
        raise ValueError("FLA DeltaNet has a single num_heads; key/value head counts must match")
    return dict(
        expand_k=cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"] / hidden,
        expand_v=cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"] / hidden,
        num_heads=cfg["linear_num_key_heads"],
        use_beta=True, use_gate=True, qk_activation="silu", qk_norm="l2",
    )


register_fla_kernel(
    "deltanet", DeltaNet,
    description="DeltaNet (FLA DeltaNet, no decay); GDN weights copied, decay branch dropped — NOT function preserving.",
    layer_kwargs=layer_kwargs, output_gate="native",          # use_gate=True gives g_proj + swish-gated o_norm
    init_extra=lambda layer, src: copy_(layer.b_proj.weight, src["b"], "b_proj"),
    new_param_names=("b_proj",), exact_init=False,
    notes="Inexact swap: expect a large deviation from HF at init; full SFT is required.",
)
