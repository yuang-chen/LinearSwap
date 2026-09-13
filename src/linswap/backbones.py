"""Backbone configuration read from a HF checkpoint (any Gated-DeltaNet hybrid with the Qwen3-Next / Qwen3.5 layer layout).

``load_backbone_config(model_dir)`` maps the HF ``config.json`` (or its
``text_config`` for multimodal checkpoints) onto the dict the model expects:

    vocab_size, context_length, emb_dim, n_heads, n_layers, hidden_dim, head_dim,
    qk_norm, n_kv_groups, rope_base, partial_rotary_factor, rms_norm_eps,
    linear_conv_kernel_dim, linear_key_head_dim, linear_value_head_dim,
    linear_num_key_heads, linear_num_value_heads, layer_types, dtype, tie_word_embeddings
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


REQUIRED_LINEAR_KEYS = ("linear_conv_kernel_dim", "linear_key_head_dim", "linear_value_head_dim",
                        "linear_num_key_heads", "linear_num_value_heads")


def load_backbone_config(model_dir, dtype=torch.bfloat16) -> dict:
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        hf = json.load(f)
    tie = hf.get("tie_word_embeddings", False)
    hf = hf.get("text_config", hf)
    missing = [k for k in REQUIRED_LINEAR_KEYS if k not in hf]
    if missing or "layer_types" not in hf:
        raise ValueError(f"{model_dir}: not a GDN-based hybrid config (missing {missing or ['layer_types']})")
    rope = hf.get("rope_parameters") or hf.get("rope_scaling") or {}
    head_dim = hf.get("head_dim") or hf["hidden_size"] // hf["num_attention_heads"]
    cfg = {
        "vocab_size": hf["vocab_size"],
        "context_length": hf.get("max_position_embeddings", 262_144),
        "emb_dim": hf["hidden_size"],
        "n_heads": hf["num_attention_heads"],
        "n_layers": hf["num_hidden_layers"],
        "hidden_dim": hf["intermediate_size"],
        "head_dim": head_dim,
        "qk_norm": True,
        "n_kv_groups": hf["num_key_value_heads"],
        "rope_base": float(rope.get("rope_theta", hf.get("rope_theta", 10_000_000.0))),
        "partial_rotary_factor": float(rope.get("partial_rotary_factor", hf.get("partial_rotary_factor", 1.0))),
        "rms_norm_eps": hf.get("rms_norm_eps", 1e-6),
        "linear_conv_kernel_dim": hf["linear_conv_kernel_dim"],
        "linear_key_head_dim": hf["linear_key_head_dim"],
        "linear_value_head_dim": hf["linear_value_head_dim"],
        "linear_num_key_heads": hf["linear_num_key_heads"],
        "linear_num_value_heads": hf["linear_num_value_heads"],
        "layer_types": list(hf["layer_types"]),
        "tie_word_embeddings": tie or hf.get("tie_word_embeddings", False),
        "dtype": dtype,
    }
    if hf.get("attention_bias", False):
        raise ValueError("attention_bias=True is not supported by the attention component")
    if "mlp_only_layers" in hf and hf["mlp_only_layers"]:
        raise ValueError("MoE / mlp_only_layers backbones are not supported")
    if hf.get("num_experts", 0):
        raise ValueError("MoE backbones are not supported")
    return cfg


def describe(cfg: dict) -> str:
    n_lin = sum(t == "linear_attention" for t in cfg["layer_types"])
    return (f"{cfg['n_layers']} layers ({n_lin} linear / {cfg['n_layers'] - n_lin} attention), hidden {cfg['emb_dim']}, "
            f"linear heads {cfg['linear_num_key_heads']}q/{cfg['linear_num_value_heads']}v × {cfg['linear_key_head_dim']}, "
            f"attention {cfg['n_heads']}h/{cfg['n_kv_groups']}kv × {cfg['head_dim']}, ctx {cfg['context_length']}")
