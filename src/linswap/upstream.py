"""Swapped mixers inside the *upstream* ``transformers`` Qwen3.5 model — text-only or vision-language.

    from linswap.upstream import build_upstream_model
    model = build_upstream_model("kda")                                   # Qwen3_5ForConditionalGeneration
    model = build_upstream_model(ckpt_dir="outputs/kda/distill/checkpoint-16338")
    inputs = processor.apply_chat_template(messages_with_images, ..., return_tensors="pt")
    model.generate(**inputs.to("cuda"), max_new_tokens=64)

``linswap.build_model`` rebuilds the language backbone from its own components and takes ``input_ids``
only.  This module does the opposite: it loads the checkpoint with the upstream class — decoder, full
attention, RoPE (multimodal positions included), vision tower, merger, processor contract and
``generate`` all stay upstream code and upstream weights — and replaces nothing but each decoder
layer's ``linear_attn`` with the kernel's layer, built and initialised exactly as ``build_model`` does.
It is how a swap, or a checkpoint distilled on text, is run on image and video inputs.

Scope, deliberately narrow:

* an in-memory model (no ``save_pretrained`` round trip: the upstream config does not record the kernel);
* inference: the cache bridge keeps the kernel's recurrent / convolution state in an FLA cache attached
  to the upstream cache object, which the upstream cache's own ``reorder_cache`` / ``crop`` do not see,
  so decoding is greedy or sampling only — no beam search, no assisted decoding;
* padding follows the upstream convention (hidden states of pad positions are zeroed before the mixer),
  which is exact for left padding with every kernel that writes nothing for a zero input (not ``swa``).

``tests/test_upstream.py`` checks it against the native upstream model on text and on a real processed
image, and against ``build_model`` for a trained checkpoint.
"""

from __future__ import annotations

import types
from pathlib import Path

import torch

from .backbones import load_backbone_config
from .load_weights import DEFAULT_BASE_MODEL_DIR, load_hf_state_dict, read_checkpoint_kernel
from .registry import kernel_map_name, parse_kernel_map

FLA_CACHE_ATTR = "_linswap_fla_cache"


def _upstream_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """``Qwen3_5DecoderLayer``'s call of its linear mixer -> the FLA layer interface the kernels implement.

    ``cache_params`` is the upstream cache of the whole model (or None); the kernel's own state lives in
    an FLA cache attached to it, so it is created, carried and dropped with that request's cache."""
    if attention_mask is not None:            # 2D padding mask of the current chunk, as upstream's own layer uses it
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    fla_cache = None
    if cache_params is not None:
        fla_cache = getattr(cache_params, FLA_CACHE_ATTR, None)
        if fla_cache is None:
            from fla.models.utils import Cache as FLACache

            fla_cache = FLACache()
            setattr(cache_params, FLA_CACHE_ATTR, fla_cache)
    return self._linswap_forward(hidden_states, past_key_values=fla_cache, use_cache=fla_cache is not None)[0]


def adapt_mixer(mixer):
    """Give a kernel layer the calling convention of the upstream decoder layer.  The module, its
    parameters and their names are untouched, so its state-dict keys stay ``...linear_attn.<name>``."""
    mixer._linswap_forward = mixer.forward
    mixer.forward = types.MethodType(_upstream_forward, mixer)
    return mixer


def text_model(model):
    """The upstream ``Qwen3_5TextModel`` of a causal-LM or conditional-generation model, and its key prefix."""
    inner = model.model
    if hasattr(inner, "language_model"):
        return inner.language_model, "model.language_model."
    return inner, "model."


def swap_mixers(model, kernel, base_model_dir=DEFAULT_BASE_MODEL_DIR, hf_weights=None):
    """Replace every ``linear_attn`` of an upstream Qwen3.5 model in place (function preserving for exact
    kernels).  ``kernel`` is a registry name or a kernel map; returns its canonical name."""
    default, kmap = parse_kernel_map(kernel)
    dtype = next(model.parameters()).dtype
    cfg = load_backbone_config(base_model_dir, dtype=dtype)
    weights = hf_weights if hf_weights is not None else load_hf_state_dict(base_model_dir)
    prefix = "model.language_model" if "model.language_model.embed_tokens.weight" in weights else "model"
    text, _ = text_model(model)
    for idx, layer in enumerate(text.layers):
        if not hasattr(layer, "linear_attn"):
            continue
        spec = kmap.get(idx, default)
        mixer = spec.build(cfg, idx)
        spec.init_from_gdn(mixer, weights, idx, prefix)
        layer.linear_attn = adapt_mixer(mixer.to(device=layer.mlp.gate_proj.weight.device, dtype=dtype))
    return kernel_map_name(default, kmap)


def load_language_checkpoint(model, ckpt_dir):
    """Load a ``linswap distill`` checkpoint (``model.pt``, language backbone in Qwen's layout) into the
    upstream model's language model and head.  Strict on the language side: every tensor of the checkpoint
    must land, and everything the checkpoint does not hold must belong to the vision side."""
    state = torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu", weights_only=True)
    _, prefix = text_model(model)
    mapped = {(prefix + k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if unexpected:
        raise RuntimeError(f"{ckpt_dir}: {len(unexpected)} tensors have no place in the upstream model, "
                           f"e.g. {unexpected[:3]} (was the checkpoint written by another kernel?)")
    stray = [k for k in missing if k.startswith(prefix) or k.startswith("lm_head")]
    if stray:
        raise RuntimeError(f"{ckpt_dir}: the checkpoint lacks {len(stray)} language tensors, e.g. {stray[:3]}")
    return len(mapped), len(missing)


def build_upstream_model(kernel=None, base_model_dir=DEFAULT_BASE_MODEL_DIR, ckpt_dir=None, multimodal=True,
                         device="cuda", dtype=torch.bfloat16):
    """The upstream Qwen3.5 model (``Qwen3_5ForConditionalGeneration`` with ``multimodal``, else the
    text-only causal LM) with its linear mixers swapped for ``kernel`` and, with ``ckpt_dir``, the language
    weights of a distilled checkpoint.  ``kernel`` may be omitted when the checkpoint records it."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    if kernel is None and ckpt_dir is not None:
        kernel = read_checkpoint_kernel(ckpt_dir)
    if kernel is None:
        raise ValueError("kernel must be given or recorded in ckpt_dir/config.json")
    cls = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    model = cls.from_pretrained(base_model_dir, dtype=dtype)
    model.linswap_kernel = swap_mixers(model, kernel, base_model_dir)       # before any swapped weights load
    if ckpt_dir is not None:
        load_language_checkpoint(model, ckpt_dir)
    return model.to(device).eval()
