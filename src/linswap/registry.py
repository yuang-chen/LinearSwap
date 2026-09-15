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
    """Registry lookup.  A *kernel map* string (``"gdn;mamba2@3,6,9"``) resolves to its default kernel;
    use :func:`parse_kernel_map` for the per-layer assignment."""
    if isinstance(name, KernelSpec):
        return name
    _ensure_builtin_kernels_loaded()
    if ";" in name or "@" in name:
        name = parse_kernel_map(name)[0].name
    if name not in _REGISTRY:
        raise KeyError(f"Unknown kernel '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def parse_kernel_map(spec: str | KernelSpec) -> tuple[KernelSpec, Dict[int, KernelSpec]]:
    """``"kda"`` -> (kda, {});  ``"gdn;mamba2@3,6,9;deltanet@12"`` -> (gdn, {3: mamba2, 6: mamba2, 9: mamba2, 12: deltanet}).

    Mixed-kernel models: every linear-attention layer uses the default kernel unless an
    ``<kernel>@<layer indices>`` clause names it (indices are absolute layer numbers)."""
    if isinstance(spec, KernelSpec):
        return spec, {}
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    default = get_kernel(parts[0].split("@")[0])
    overrides: Dict[int, KernelSpec] = {}
    for clause in parts[1:]:
        if "@" not in clause:
            raise ValueError(f"kernel map clause {clause!r} needs the form <kernel>@<layers>")
        name, layers = clause.split("@", 1)
        k = get_kernel(name.strip())
        for tok in layers.split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = tok.split("-")
                for i in range(int(a), int(b) + 1):
                    overrides[i] = k
            elif tok:
                overrides[int(tok)] = k
    return default, overrides


def kernel_map_name(default: KernelSpec, overrides: Dict[int, KernelSpec]) -> str:
    """Canonical string for a kernel map (the inverse of :func:`parse_kernel_map`)."""
    if not overrides:
        return default.name
    by_kernel: Dict[str, list] = {}
    for i in sorted(overrides):
        by_kernel.setdefault(overrides[i].name, []).append(str(i))
    return default.name + "".join(f";{k}@{','.join(v)}" for k, v in by_kernel.items())


def list_kernels() -> list[str]:
    _ensure_builtin_kernels_loaded()
    return sorted(_REGISTRY)
