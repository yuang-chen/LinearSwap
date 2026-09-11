"""Weight loading for swapped models.

Two checkpoint formats are understood:

* **HF format** (``model.language_model.layers.{i}.linear_attn.*``): the
  pretrained Qwen3.5 checkpoint.  Non-linear layers are copied verbatim; each
  linear layer is initialised through ``kernel.init_from_gdn``.
* **Native format** (``trf_blocks.{i}.token_mixer.*``): ``state_dict()`` of a
  swapped model, as written by ``scripts/sft.py`` (``model.pt``).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import QWEN3_5_CONFIG
from .model import Qwen3_5LinearSwapModel
from .registry import get_kernel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL_DIR = REPO_ROOT / "models" / "Qwen3.5-0.8B"


def load_hf_state_dict(model_dir=DEFAULT_BASE_MODEL_DIR) -> dict:
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
    else:
        files = sorted(p.name for p in model_dir.glob("*.safetensors"))
    weights = {}
    for fn in files:
        weights.update(load_file(model_dir / fn))
    return weights


def _assign(left, right, name):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in '{name}': {tuple(left.shape)} vs {tuple(right.shape)}")
    with torch.no_grad():
        left.copy_(right.to(dtype=left.dtype, device=left.device))


def load_weights_from_gdn(model: Qwen3_5LinearSwapModel, params: dict) -> None:
    """Load an HF-format Qwen3.5 (GDN) checkpoint into a swapped model."""
    if "model.embed_tokens.weight" in params:
        model_prefix = "model"
    elif "model.language_model.embed_tokens.weight" in params:
        model_prefix = "model.language_model"
    else:
        raise KeyError("Could not find embed token weights in checkpoint.")

    def pkey(suffix):
        return f"{model_prefix}.{suffix}"

    def get(suffix):
        return params[pkey(suffix)]

    _assign(model.tok_emb.weight, get("embed_tokens.weight"), "embed_tokens")
    layer_types = model.cfg.get("layer_types", ["full_attention"] * model.cfg["n_layers"])
    kernel = model.kernel

    for l, block in enumerate(model.trf_blocks):
        if layer_types[l] == "full_attention":
            att = block.token_mixer
            _assign(att.W_query.weight, get(f"layers.{l}.self_attn.q_proj.weight"), f"{l}.q_proj")
            _assign(att.W_key.weight, get(f"layers.{l}.self_attn.k_proj.weight"), f"{l}.k_proj")
            _assign(att.W_value.weight, get(f"layers.{l}.self_attn.v_proj.weight"), f"{l}.v_proj")
            _assign(att.out_proj.weight, get(f"layers.{l}.self_attn.o_proj.weight"), f"{l}.o_proj")
            if att.q_norm is not None:
                _assign(att.q_norm.weight, get(f"layers.{l}.self_attn.q_norm.weight"), f"{l}.q_norm")
            if att.k_norm is not None:
                _assign(att.k_norm.weight, get(f"layers.{l}.self_attn.k_norm.weight"), f"{l}.k_norm")
        elif layer_types[l] == "linear_attention":
            kernel.init_from_gdn(block.token_mixer, params, l, model_prefix)
        else:
            raise ValueError(f"Unsupported layer type: {layer_types[l]}")

        _assign(block.norm1.weight, get(f"layers.{l}.input_layernorm.weight"), f"{l}.norm1")
        _assign(block.ff.fc1.weight, get(f"layers.{l}.mlp.gate_proj.weight"), f"{l}.fc1")
        _assign(block.ff.fc2.weight, get(f"layers.{l}.mlp.up_proj.weight"), f"{l}.fc2")
        _assign(block.ff.fc3.weight, get(f"layers.{l}.mlp.down_proj.weight"), f"{l}.fc3")
        _assign(block.norm2.weight, get(f"layers.{l}.post_attention_layernorm.weight"), f"{l}.norm2")

    _assign(model.final_norm.weight, get("norm.weight"), "final_norm")
    if "lm_head.weight" in params:
        _assign(model.out_head.weight, params["lm_head.weight"], "lm_head")
    elif pkey("lm_head.weight") in params:
        _assign(model.out_head.weight, get("lm_head.weight"), "lm_head")
    else:
        model.out_head.weight = model.tok_emb.weight


def load_native_checkpoint(model: Qwen3_5LinearSwapModel, ckpt_dir, strict=True) -> None:
    state = torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"[qwen_linswap] load_native_checkpoint: missing={missing[:5]} unexpected={unexpected[:5]}")


def read_checkpoint_kernel(ckpt_dir) -> str | None:
    cfg_path = Path(ckpt_dir) / "config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return json.load(f).get("linear_kernel")


def build_model(kernel: str | None = None, base_model_dir=DEFAULT_BASE_MODEL_DIR, ckpt_dir=None,
                device="cuda", dtype=torch.bfloat16, cfg=None, hf_weights: dict | None = None):
    """One-stop model construction.

    1. build ``Qwen3_5LinearSwapModel(cfg, kernel)``
    2. initialise from the pretrained HF GDN checkpoint (function preserving)
    3. if ``ckpt_dir`` contains ``model.pt``, overwrite with that native checkpoint.

    ``kernel`` may be omitted when ``ckpt_dir/config.json`` records it.
    """
    if kernel is None and ckpt_dir is not None:
        kernel = read_checkpoint_kernel(ckpt_dir)
    if kernel is None:
        raise ValueError("kernel must be given or recorded in ckpt_dir/config.json")
    cfg = cfg or QWEN3_5_CONFIG
    model = Qwen3_5LinearSwapModel(cfg, kernel)
    if hf_weights is None:
        hf_weights = load_hf_state_dict(base_model_dir)
    load_weights_from_gdn(model, hf_weights)
    if ckpt_dir is not None and (Path(ckpt_dir) / "model.pt").exists():
        load_native_checkpoint(model, ckpt_dir)
    return model.to(device=device, dtype=dtype)
