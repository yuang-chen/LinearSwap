"""Mamba-1 (arXiv:2312.00752, selective S6) via FLA's ``Mamba`` layer.  Registered only when
``mamba_ssm``'s selective-scan CUDA kernel imports.

Mamba-1 has no heads and no q/k: each of the ``intermediate_size`` channels runs its own
diagonal SSM of size ``state_size`` whose B/C/Δ come from ``x_proj(x)`` (a function of the
convolved input, not of the residual stream), the gate ``z`` multiplies the output without
a norm, and ``A`` is a static ``[intermediate, state]`` matrix.  With ``state_size = 128``
the state has the same number of entries as GDN's (2048 channels × 128 = 16 heads × 128 × 128).

Init from GDN (inexact): ``x ← v`` and ``z ← in_proj_z`` rows of the fused ``in_proj``, the
value convolution copied into ``conv1d`` (same SiLU-activated depthwise conv), ``out_proj``
copied; ``x_proj``, ``dt_proj``, ``A_log`` (S4D-real) and ``D`` keep Mamba's own init because
they have no GDN counterpart.  The output is gated without a norm, so ``out_proj`` sees a
differently scaled input than in the backbone — distillation has to fix that too."""

import torch

from .common import copy_

try:
    from fla.layers.mamba import Mamba, is_fast_path_available as _mamba_available
except Exception:  # pragma: no cover
    Mamba, _mamba_available = None, False


def layer_kwargs(cfg):
    return dict(state_size=cfg["linear_key_head_dim"], conv_kernel=cfg["linear_conv_kernel_dim"],
                use_conv_bias=False, intermediate_size=cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"],
                use_bias=False)


def init_extra(layer, src):
    I = layer.intermediate_size
    key_dim = (src["qkv"].shape[0] - I) // 2
    _, _, vw = torch.split(src["qkv"], [key_dim, key_dim, I], dim=0)
    _, _, cv = torch.split(src["conv"], [key_dim, key_dim, I], dim=0)
    with torch.no_grad():
        layer.in_proj.weight[:I].copy_(vw.to(layer.in_proj.weight.dtype))      # x: values
        layer.in_proj.weight[I:].copy_(src["z"].to(layer.in_proj.weight.dtype))  # z: gate
        copy_(layer.conv1d.weight, cv, "conv1d")
        copy_(layer.out_proj.weight, src["out"], "out_proj")


if _mamba_available:
    from .fla_layer import register_fla_kernel

    register_fla_kernel(
        "mamba1", Mamba,
        description="Mamba-1 selective SSM (FLA Mamba on mamba_ssm kernels); values/gate/conv/out_proj copied, "
                    "SSM parameters at Mamba init — NOT function preserving.",
        layer_kwargs=layer_kwargs, output_gate="native", copy_shared=False, init_extra=init_extra,
        new_param_names=("x_proj", "dt_proj", "A_log", "D"), exact_init=False,
        notes="Inexact swap (per-channel SSM, no q/k, no output norm): distil before SFT.",
    )
