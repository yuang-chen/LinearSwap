"""Kernel registry for the LinearSwap framework.

A *kernel* here is a complete linear-attention token mixer (an ``nn.Module``
with the FLA layer interface) plus a recipe for initialising it from the
pretrained Gated-DeltaNet weights of the backbone.  Registering a new kernel is the
only thing needed to make it available to the model builder, the weight
loader, the verification script, the SFT script and the RULER wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch.nn as nn

# (cfg, layer_idx) -> token mixer module.  The module must implement
#   forward(x, past_key_values=None, use_cache=False) -> (out, None, cache)
# i.e. the flash-linear-attention layer interface.
BuildFn = Callable[[dict, int], nn.Module]

# (layer, gdn_state_dict, layer_idx, model_prefix) -> layer
# Copies / tiles the pretrained GDN weights of ``layer_idx`` into ``layer``.
InitFn = Callable[[nn.Module, dict, int, str], nn.Module]


@dataclass(frozen=True)
class KernelSpec:
    name: str
    description: str
    build: BuildFn
    init_from_gdn: InitFn
    # Parameter *component* names (split on ".") that are considered the
    # kernel's gate / newly-introduced parameters.  Used by gate-only SFT.
    new_param_names: tuple = ()
    # Whether ``init_from_gdn`` is function preserving (exact up to fp error).
    exact_init: bool = True
    notes: str = ""
    # False for layers whose autograd Functions read ctx.saved_tensors more than once and therefore
    # break under torch.utils.checkpoint (mamba_ssm's Mamba-3 kernels); training then runs without it.
    supports_activation_checkpointing: bool = True

    def is_new_param(self, param_name: str) -> bool:
        parts = param_name.split(".")
        return any(p in parts for p in self.new_param_names)


_REGISTRY: Dict[str, KernelSpec] = {}


def register_kernel(spec: KernelSpec) -> KernelSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"Kernel '{spec.name}' is already registered")
    _REGISTRY[spec.name] = spec
    return spec


def _ensure_builtin_kernels_loaded():
    # Importing the package registers the built-in kernels.
    from . import kernels  # noqa: F401


def get_kernel(name: str | KernelSpec) -> KernelSpec:
    if isinstance(name, KernelSpec):
        return name
    _ensure_builtin_kernels_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown kernel '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_kernels() -> list[str]:
    _ensure_builtin_kernels_loaded()
    return sorted(_REGISTRY)
