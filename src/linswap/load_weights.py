"""Weight loading for swapped models.

Two checkpoint formats are understood:

* **HF format** (``model.language_model.layers.{i}.linear_attn.*``): the
  pretrained backbone checkpoint.  Non-linear layers are copied verbatim; each
  linear layer is initialised through ``kernel.init_from_gdn``.
* **Native format** (``model.layers.{i}.linear_attn.*`` — the same keys as the HF checkpoint):
  ``state_dict()`` of a swapped model, as written by ``linswap posttrain`` (``model.pt``).  Checkpoints
  written before September 2026 (``trf_blocks.*`` keys) are converted on load.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .backbones import load_backbone_config
from .model import LinearSwapModel
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


def load_weights_from_gdn(model: LinearSwapModel, params: dict) -> None:
    """Load an HF-format GDN-hybrid checkpoint (Qwen3-Next / Qwen3.5 naming) into a swapped model."""
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

    _assign(model.embed_tokens.weight, get("embed_tokens.weight"), "embed_tokens")
    layer_types = model.cfg.get("layer_types", ["full_attention"] * model.cfg["n_layers"])
    kernel = model.kernel

    for l, block in enumerate(model.layers):
        if layer_types[l] == "full_attention":
            att = block.self_attn
            _assign(att.q_proj.weight, get(f"layers.{l}.self_attn.q_proj.weight"), f"{l}.q_proj")
            _assign(att.k_proj.weight, get(f"layers.{l}.self_attn.k_proj.weight"), f"{l}.k_proj")
            _assign(att.v_proj.weight, get(f"layers.{l}.self_attn.v_proj.weight"), f"{l}.v_proj")
            _assign(att.o_proj.weight, get(f"layers.{l}.self_attn.o_proj.weight"), f"{l}.o_proj")
            if att.q_norm is not None:
                _assign(att.q_norm.weight, get(f"layers.{l}.self_attn.q_norm.weight"), f"{l}.q_norm")
            if att.k_norm is not None:
                _assign(att.k_norm.weight, get(f"layers.{l}.self_attn.k_norm.weight"), f"{l}.k_norm")
        elif layer_types[l] == "linear_attention":
            kernel.init_from_gdn(block.linear_attn, params, l, model_prefix)
        else:
            raise ValueError(f"Unsupported layer type: {layer_types[l]}")

        _assign(block.input_layernorm.weight, get(f"layers.{l}.input_layernorm.weight"), f"{l}.input_layernorm")
        _assign(block.mlp.gate_proj.weight, get(f"layers.{l}.mlp.gate_proj.weight"), f"{l}.gate_proj")
        _assign(block.mlp.up_proj.weight, get(f"layers.{l}.mlp.up_proj.weight"), f"{l}.up_proj")
        _assign(block.mlp.down_proj.weight, get(f"layers.{l}.mlp.down_proj.weight"), f"{l}.down_proj")
        _assign(block.post_attention_layernorm.weight, get(f"layers.{l}.post_attention_layernorm.weight"), f"{l}.post_attention_layernorm")

    _assign(model.model.norm.weight, get("norm.weight"), "norm")
    if "lm_head.weight" in params:
        _assign(model.lm_head.weight, params["lm_head.weight"], "lm_head")
    elif pkey("lm_head.weight") in params:
        _assign(model.lm_head.weight, get("lm_head.weight"), "lm_head")
    else:
        model.lm_head.weight = model.embed_tokens.weight


_LEGACY_ATTN = {"W_query": "q_proj", "W_key": "k_proj", "W_value": "v_proj", "out_proj": "o_proj", "q_norm": "q_norm", "k_norm": "k_norm"}
_LEGACY_MLP = {"fc1": "gate_proj", "fc2": "up_proj", "fc3": "down_proj"}


def convert_legacy_state_dict(state: dict, layer_types) -> dict:
    """Map the pre-September-2026 native key layout (``tok_emb`` / ``trf_blocks.{i}.token_mixer`` / ``ff.fc*`` /
    ``final_norm`` / ``out_head``) onto the Qwen-style layout used now."""
    if "tok_emb.weight" not in state:
        return state
    out = {}
    for k, v in state.items():
        if k == "tok_emb.weight":
            nk = "model.embed_tokens.weight"
        elif k == "out_head.weight":
            nk = "lm_head.weight"
        elif k == "final_norm.weight":
            nk = "model.norm.weight"
        elif k.startswith("trf_blocks."):
            _, i, rest = k.split(".", 2)
            i = int(i)
            if rest.startswith("norm1."):
                rest = "input_layernorm." + rest[len("norm1."):]
            elif rest.startswith("norm2."):
                rest = "post_attention_layernorm." + rest[len("norm2."):]
            elif rest.startswith("ff."):
                sub, tail = rest[len("ff."):].split(".", 1)
                rest = f"mlp.{_LEGACY_MLP[sub]}.{tail}"
            elif rest.startswith("token_mixer."):
                tail = rest[len("token_mixer."):]
                if layer_types[i] == "full_attention":
                    sub, tail2 = tail.split(".", 1)
                    rest = f"self_attn.{_LEGACY_ATTN[sub]}.{tail2}"
                else:
                    rest = f"linear_attn.{tail}"
            nk = f"model.layers.{i}.{rest}"
        else:
            nk = k
        out[nk] = v
    return out


def load_native_checkpoint(model: LinearSwapModel, ckpt_dir, strict=True) -> None:
    state = torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu", weights_only=True)
    state = convert_legacy_state_dict(state, model.cfg.get("layer_types", ["full_attention"] * model.cfg["n_layers"]))
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"[linswap] load_native_checkpoint: missing={missing[:5]} unexpected={unexpected[:5]}")


def read_checkpoint_kernel(ckpt_dir) -> str | None:
    cfg_path = Path(ckpt_dir) / "config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return json.load(f).get("linear_kernel")


def build_model(kernel: str | None = None, base_model_dir=DEFAULT_BASE_MODEL_DIR, ckpt_dir=None,
                device="cuda", dtype=torch.bfloat16, cfg=None, hf_weights: dict | None = None):
    """One-stop model construction.

    1. read the architecture from ``base_model_dir/config.json`` and build ``LinearSwapModel(cfg, kernel)``
    2. initialise from the pretrained HF GDN checkpoint (function preserving)
    3. if ``ckpt_dir`` contains ``model.pt``, overwrite with that native checkpoint.

    ``kernel`` may be omitted when ``ckpt_dir/config.json`` records it.
    """
    if kernel is None and ckpt_dir is not None:
        kernel = read_checkpoint_kernel(ckpt_dir)
    if kernel is None:
        raise ValueError("kernel must be given or recorded in ckpt_dir/config.json")
    cfg = cfg or load_backbone_config(base_model_dir, dtype=dtype)
    model = LinearSwapModel(cfg, kernel)
    if hf_weights is None:
        hf_weights = load_hf_state_dict(base_model_dir)
    load_weights_from_gdn(model, hf_weights)
    if ckpt_dir is not None and (Path(ckpt_dir) / "model.pt").exists():
        load_native_checkpoint(model, ckpt_dir)
    return model.to(device=device, dtype=dtype)
