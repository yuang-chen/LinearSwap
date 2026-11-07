"""Wrap any ``fla.layers`` class as a LinearSwap kernel.

``build_fla_layer`` instantiates an FLA layer from the backbone config using FLA's
own constructor names (``hidden_size``, ``head_dim``, ``num_heads``, ``num_v_heads``,
``use_short_conv``, ``conv_size``, ``layer_idx``, ``norm_eps`` …), dropping the ones a
particular layer does not accept, and can swap in the backbone's output gate.
``register_fla_kernel`` turns that plus an init recipe into a ``KernelSpec``:

    register_fla_kernel(
        "deltanet", DeltaNet, description="…",
        layer_kwargs=lambda cfg: dict(**hidden_relative(cfg), use_beta=True, use_gate=True, qk_norm="l2"),
        output_gate="native",                                      # the layer already has the Qwen-style gate
        grouped_values=False,                                      # it has a single num_heads
        init_extra=lambda layer, src: copy_(layer.b_proj.weight, src["b"], "b_proj"),
        new_param_names=("b_proj",), exact_init=False)

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


def fla_default_kwargs(cfg: dict, layer_idx: int, grouped_values: bool = True) -> dict:
    """``grouped_values=False`` for a layer that cannot share a key head among value heads (or keeps its gates
    per key head): it is built with one key head per value head, initialised as ``copy_qkv_and_conv`` says."""
    HV = cfg["linear_num_value_heads"]
    return dict(
        hidden_size=cfg["emb_dim"],
        expand_v=1.0,
        head_dim=cfg["linear_key_head_dim"],
        num_heads=cfg["linear_num_key_heads"] if grouped_values else HV,
        num_v_heads=HV,
        mode="chunk",
        use_short_conv=True,
        conv_size=cfg["linear_conv_kernel_dim"],
        conv_bias=False,
        layer_idx=layer_idx,
        norm_eps=cfg.get("rms_norm_eps", 1e-6),
    )


def hidden_relative(cfg: dict) -> dict:
    """``expand_k`` / ``expand_v`` for the FLA layers that size keys and values relative to the hidden size and
    have a single ``num_heads`` (GLA, DeltaNet): one key head per value head."""
    HV, hidden = cfg["linear_num_value_heads"], cfg["emb_dim"]
    return dict(expand_k=HV * cfg["linear_key_head_dim"] / hidden, expand_v=HV * cfg["linear_value_head_dim"] / hidden)


def build_fla_layer(layer_cls, cfg: dict, layer_idx: int, layer_kwargs: dict | Callable[[dict], dict] | None = None,
                    output_gate: str = "qwen", gate_attr: str = "g_proj", norm_attr: str = "o_norm",
                    grouped_values: bool = True) -> nn.Module:
    kwargs = fla_default_kwargs(cfg, layer_idx, grouped_values)
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
                        norm_attr: str = "o_norm", grouped_values: bool = True,
                        post_build: Callable[[nn.Module, dict], None] | None = None,
                        init_extra: Callable[[nn.Module, dict], None] | None = None, notes: str = "") -> KernelSpec:
    def build(cfg, layer_idx):
        layer = build_fla_layer(layer_cls, cfg, layer_idx, layer_kwargs, output_gate, gate_attr, norm_attr, grouped_values)
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
