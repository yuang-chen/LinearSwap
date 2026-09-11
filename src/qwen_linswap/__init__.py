"""Swap Qwen3.5's Gated-DeltaNet linear-attention layers for other linear kernels.

    from qwen_linswap import build_model, list_kernels   # kernels: gdn, gdn2, kda, kda_fullgate, deltanet
    model = build_model("kda")                       # exact, function-preserving init from Qwen3.5-0.8B
    model = build_model(ckpt_dir="outputs/sft_kda_full/checkpoint-50")   # SFT checkpoint (kernel from config.json)
"""

from .config import QWEN3_5_CONFIG

from .load_weights import (
    DEFAULT_BASE_MODEL_DIR,
    build_model,
    load_hf_state_dict,
    load_native_checkpoint,
    load_weights_from_gdn,
    read_checkpoint_kernel,
)
from .model import Qwen3_5LinearSwapModel, SwapCache
from .registry import KernelSpec, get_kernel, list_kernels, register_kernel

__all__ = [
    "QWEN3_5_CONFIG",
    "DEFAULT_BASE_MODEL_DIR",
    "KernelSpec",
    "Qwen3_5LinearSwapModel",
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
