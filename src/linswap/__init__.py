"""Swap the Gated-DeltaNet layers of Qwen3.5 for other linear kernels and distil them back to the teacher.

    from linswap import build_model, list_kernels
    model = build_model("kda")                       # transformers' Qwen3.5 vision-language model, mixers swapped
    model = build_model("outputs/distill/gla")       # a distilled checkpoint
"""

from .model import DEFAULT_BASE_MODEL_DIR, NATIVE, REPO_ROOT, build_model, load_checkpoint, resolve, save_checkpoint, swap_mixers
from .registry import KernelSpec, get_kernel, list_kernels, register_kernel

__all__ = ["DEFAULT_BASE_MODEL_DIR", "NATIVE", "REPO_ROOT", "KernelSpec", "build_model", "get_kernel", "list_kernels",
           "load_checkpoint", "register_kernel", "resolve", "save_checkpoint", "swap_mixers"]
