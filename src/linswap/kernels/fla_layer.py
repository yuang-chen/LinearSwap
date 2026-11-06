"""Wrap any ``fla.layers`` class as a LinearSwap kernel.

``build_fla_layer`` instantiates an FLA layer from the backbone config using FLA's
own constructor names (``hidden_size``, ``head_dim``, ``num_heads``, ``num_v_heads``,
``use_short_conv``, ``conv_size``, ``layer_idx``, ``norm_eps`` …), dropping the ones a
particular layer does not accept, and can swap in the backbone's output gate.
``register_fla_kernel`` turns that plus an init recipe into a ``KernelSpec``:

    register_fla_kernel(
        "gla", GatedLinearAttention, description="…",
        layer_kwargs=lambda cfg: dict(expand_k=2.0, expand_v=2.0, num_heads=16, use_short_conv=True),
        norm_attr="g_norm_swish_gate", output_gate="native",      # GLA already has the Qwen-style gate
        init_extra=None,                                           # only the shared weights are copied
        new_param_names=("gk_proj",), exact_init=False)

The init recipe is ``copy_shared_from_gdn`` (q/k/v, convolutions, output gate) followed by
``init_extra(layer, src)`` for the recurrence-specific parameters; kernels whose init is
function preserving set ``exact_init=True``.
"""

from __future__ import annotations

import inspect
from typing import Callable

import torch.nn as nn

from ..registry import KernelSpec, register_kernel
from .common import copy_shared_from_gdn, get_gdn_source, mark_hf_initialized, use_qwen_output_gate


def fla_default_kwargs(cfg: dict, layer_idx: int) -> dict:
    return dict(
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


def build_fla_layer(layer_cls, cfg: dict, layer_idx: int, layer_kwargs: dict | Callable[[dict], dict] | None = None,
                    output_gate: str = "qwen", gate_attr: str = "g_proj", norm_attr: str = "o_norm") -> nn.Module:
    kwargs = fla_default_kwargs(cfg, layer_idx)
    if layer_kwargs is not None:
        kwargs.update(layer_kwargs(cfg) if callable(layer_kwargs) else layer_kwargs)
    params = inspect.signature(layer_cls.__init__).parameters
    accepted = {k: v for k, v in kwargs.items() if k in params and params[k].kind is not inspect.Parameter.VAR_KEYWORD}
    layer = layer_cls(**accepted)
    if output_gate == "qwen":
        use_qwen_output_gate(layer, layer.hidden_size, layer.value_dim, layer.head_v_dim,
                             cfg.get("rms_norm_eps", 1e-6), gate_attr, norm_attr)
    elif output_gate != "native":
        raise ValueError(f"output_gate must be 'qwen' or 'native', got {output_gate!r}")
    return layer


def register_fla_kernel(name: str, layer_cls, *, description: str, new_param_names=(), exact_init: bool,
                        layer_kwargs=None, output_gate: str = "qwen", gate_attr: str = "g_proj",
                        norm_attr: str = "o_norm", post_build: Callable[[nn.Module, dict], None] | None = None,
                        init_extra: Callable[[nn.Module, dict], None] | None = None, notes: str = "") -> KernelSpec:
    def build(cfg, layer_idx):
        layer = build_fla_layer(layer_cls, cfg, layer_idx, layer_kwargs, output_gate, gate_attr, norm_attr)
        if post_build is not None:
            post_build(layer, cfg)
        return layer

    def init_from_gdn(layer, gdn_state, layer_idx, model_prefix="model"):
        src = get_gdn_source(gdn_state, layer_idx, model_prefix)
        copy_shared_from_gdn(layer, src, gate_attr, norm_attr)
        if init_extra is not None:
            init_extra(layer, src)
        mark_hf_initialized(layer)
        return layer

    return register_kernel(KernelSpec(name=name, description=description, build=build, init_from_gdn=init_from_gdn,
                                      new_param_names=tuple(new_param_names), exact_init=exact_init, notes=notes))
