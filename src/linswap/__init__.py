"""Swap the Gated-DeltaNet linear-attention layers of a pretrained hybrid LLM for other linear kernels.

    from linswap import build_model, list_kernels   # kernels: gdn, gdn2, kda, kda_fullgate, deltanet
    model = build_model("kda")                       # exact, function-preserving init from the backbone in models/
    model = build_model(ckpt_dir="outputs/sft_kda_full/checkpoint-50")   # SFT checkpoint (kernel from config.json)
"""

from .backbones import describe, load_backbone_config

from .load_weights import (
    DEFAULT_BASE_MODEL_DIR,
    build_model,
    load_hf_state_dict,
    load_native_checkpoint,
    load_weights_from_gdn,
    read_checkpoint_kernel,
)
from .model import LinearSwapModel, SwapCache
from .registry import KernelSpec, get_kernel, list_kernels, register_kernel
from .hf import LinearSwapCache, LinearSwapConfig, LinearSwapForCausalLM, register_auto_classes

register_auto_classes()

__all__ = [
    "describe",
    "load_backbone_config",
    "DEFAULT_BASE_MODEL_DIR",
    "KernelSpec",
    "LinearSwapCache",
    "LinearSwapConfig",
    "LinearSwapForCausalLM",
    "LinearSwapModel",
    "SwapCache",
    "build_model",
    "get_kernel",
    "list_kernels",
    "load_hf_state_dict",
    "load_native_checkpoint",
    "load_weights_from_gdn",
    "read_checkpoint_kernel",
    "register_kernel",
]
