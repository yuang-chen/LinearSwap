"""Hybrid (Gated-DeltaNet + full attention) backbone with pluggable linear-attention token mixers.

Full-attention layers, MLPs and norms come from ``components.py``; the
linear-attention layers are built by the kernel registry.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .components import FeedForward, GroupedQueryAttention, RMSNorm, compute_rope_params

from .registry import KernelSpec, get_kernel, kernel_map_name, parse_kernel_map

ROPE_MARGIN = 4096  # positions beyond the backbone's max_position_embeddings that the RoPE tables cover


class TransformerBlock(nn.Module):
    """One decoder layer, laid out like Qwen3.5's: input_layernorm -> self_attn | linear_attn -> post_attention_layernorm -> mlp."""

    def __init__(self, cfg, layer_type, layer_idx, kernel: KernelSpec):
        super().__init__()
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        if layer_type == "full_attention":
            self.self_attn = GroupedQueryAttention(
                d_in=cfg["emb_dim"],
                num_heads=cfg["n_heads"],
                head_dim=cfg["head_dim"],
                num_kv_groups=cfg["n_kv_groups"],
                qk_norm=cfg["qk_norm"],
                dtype=cfg["dtype"],
            )
        elif layer_type == "linear_attention":
            self.linear_attn = kernel.build(cfg, layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.mlp = FeedForward(cfg)

    @property
    def token_mixer(self):
        return self.self_attn if self.layer_type == "full_attention" else self.linear_attn

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None, linear_cache=None, use_cache=False):
        shortcut = x
        x = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            h, next_kv_cache = self.self_attn(
                x, mask=mask, cos=cos, sin=sin, start_pos=start_pos, cache=cache, use_cache=use_cache,
            )
        else:
            h, _, _ = self.linear_attn(x, past_key_values=linear_cache, use_cache=linear_cache is not None)
            next_kv_cache = None
        x = h + shortcut
        return self.mlp(self.post_attention_layernorm(x)) + x, next_kv_cache


class LinearSwapBackbone(nn.Module):
    """embed_tokens -> layers -> norm (the ``model`` sub-module of a Qwen-style causal LM)."""

    def __init__(self, cfg, kernel: str | KernelSpec):
        super().__init__()
        self.cfg = cfg
        # ``kernel`` may be a registry name / KernelSpec or a kernel-map string ("gdn;mamba2@3,6,9"):
        # the default kernel builds every linear-attention layer unless the map overrides that layer.
        self.kernel, self.kernel_map = parse_kernel_map(kernel)
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        layer_types = cfg.get("layer_types", ["full_attention"] * cfg["n_layers"])
        if len(layer_types) != cfg["n_layers"]:
            raise ValueError("len(layer_types) must equal n_layers")
        for i in self.kernel_map:
            if i >= len(layer_types) or layer_types[i] != "linear_attention":
                raise ValueError(f"kernel map names layer {i}, which is not a linear-attention layer")
        self.layers = nn.ModuleList([TransformerBlock(cfg, lt, idx, self.layer_kernel(idx))
                                     for idx, lt in enumerate(layer_types)])
        self.norm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        # RoPE tables are computed lazily per (device, dtype) rather than registered as buffers:
        # non-persistent buffers are not restored by HF `from_pretrained` (meta-device init).
        self._rope_tables = {}
        self.current_pos = 0
        self.gradient_checkpointing = False

    def layer_kernel(self, idx: int) -> KernelSpec:
        return self.kernel_map.get(idx, self.kernel)

    @property
    def kernel_name(self) -> str:
        """Registry name, or the canonical kernel-map string for mixed-kernel models."""
        return kernel_map_name(self.kernel, self.kernel_map)

    def rope_tables(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._rope_tables:
            cfg = self.cfg
            head_dim = cfg["emb_dim"] // cfg["n_heads"] if cfg["head_dim"] is None else cfg["head_dim"]
            cos, sin = compute_rope_params(
                # A margin beyond max_position_embeddings lets generation continue past a prompt that fills the
                # native window (RULER at 256K); HF computes RoPE per position and extrapolates the same way.
                head_dim=head_dim, theta_base=cfg["rope_base"], context_length=cfg["context_length"] + ROPE_MARGIN,
                partial_rotary_factor=cfg.get("partial_rotary_factor", 1.0), dtype=torch.float32,
            )
            self._rope_tables = {key: (cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype))}
        return self._rope_tables[key]

    def forward(self, in_idx, cache=None, use_cache=False, apply_norm=True):
        x = self.embed_tokens(in_idx)
        num_tokens = x.shape[1]
        if cache is not None or use_cache:
            start_pos = self.current_pos
            self.current_pos = start_pos + num_tokens
        else:
            start_pos = 0
        mask = None  # full attention builds its own compact causal mask; never materialise a dense one
        linear_cache = cache.linear_cache if cache is not None else None
        cos, sin = self.rope_tables(x.device, x.dtype)
        for i, block in enumerate(self.layers):
            kv_cache = cache.get(i) if cache is not None else None
            if self.gradient_checkpointing and self.training:
                x, new_kv = checkpoint(block, x, mask, cos, sin, start_pos, kv_cache, linear_cache, use_cache,
                                       use_reentrant=False)
            else:
                x, new_kv = block(x, mask=mask, cos=cos, sin=sin, start_pos=start_pos, cache=kv_cache,
                                  linear_cache=linear_cache, use_cache=use_cache)
            if cache is not None and new_kv is not None:
                cache.update(i, new_kv)
        return self.norm(x) if apply_norm else x


class LinearSwapModel(nn.Module):
    """Causal LM: ``model`` (LinearSwapBackbone) + ``lm_head`` — the same module tree and state-dict
    keys as Qwen3.5's ``Qwen3_5ForCausalLM`` (``model.embed_tokens``, ``model.layers.{i}.self_attn|linear_attn|mlp``,
    ``model.norm``, ``lm_head``), so unchanged tensors are byte-identical to the backbone checkpoint."""

    def __init__(self, cfg, kernel: str | KernelSpec = "gdn"):
        super().__init__()
        self.cfg = cfg
        self.model = LinearSwapBackbone(cfg, kernel)
        self.kernel = self.model.kernel            # default kernel (see ``kernel_map`` / ``kernel_name``)
        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    # ---- convenience accessors
    @property
    def layers(self):
        return self.model.layers

    @property
    def embed_tokens(self):
        return self.model.embed_tokens

    @property
    def out_head(self):  # backwards-compatible alias
        return self.lm_head

    @property
    def current_pos(self):
        return self.model.current_pos

    @current_pos.setter
    def current_pos(self, v):
        self.model.current_pos = v

    @property
    def gradient_checkpointing(self):
        return self.model.gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, v):
        self.model.gradient_checkpointing = v

    @property
    def linear_layers(self):
        return [b.linear_attn for b in self.model.layers if b.layer_type == "linear_attention"]

    @property
    def kernel_map(self):
        return self.model.kernel_map

    @property
    def kernel_name(self) -> str:
        return self.model.kernel_name

    def layer_kernel(self, idx: int) -> KernelSpec:
        return self.model.layer_kernel(idx)

    @property
    def supports_activation_checkpointing(self) -> bool:
        return all(k.supports_activation_checkpointing for k in [self.kernel, *self.kernel_map.values()])

    @property
    def new_param_names(self) -> tuple:
        names = []
        for k in [self.kernel, *self.kernel_map.values()]:
            names.extend(n for n in k.new_param_names if n not in names)
        return tuple(names)

    def new_parameters(self):
        """Named parameters the (per-layer) kernel spec considers new / gate parameters."""
        out = []
        for n, p in self.named_parameters():
            m = re.match(r"model\.layers\.(\d+)\.", n)
            spec = self.layer_kernel(int(m.group(1))) if m else self.kernel
            if spec.is_new_param(n):
                out.append((n, p))
        return out

    # ---- forward
    def forward(self, in_idx, cache=None, use_cache=False, return_hidden=False,
                return_hidden_before_norm=False, last_logits_only=False):
        x = self.model(in_idx, cache=cache, use_cache=use_cache, apply_norm=not return_hidden_before_norm)
        if return_hidden_before_norm or return_hidden:
            return x.to(self.cfg["dtype"])
        if last_logits_only:
            return self.lm_head(x[:, -1:, :].to(self.cfg["dtype"])).squeeze(1)
        return self.lm_head(x.to(self.cfg["dtype"]))

    # ---- generate
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, eos_token_id=None, temperature=0.0, top_k=1, top_p=1.0,
                 use_cache=True, pad_token_id=None):
        """Greedy decoding for a batch of equal-length (unpadded) prompts; sampling arguments are
        accepted for API compatibility.  Rows that hit ``eos_token_id`` (an int or a list) are
        frozen and padded with ``pad_token_id`` (default: the EOS id) until every row is finished."""
        if max_new_tokens <= 0:
            return input_ids
        eos = None
        if eos_token_id is not None:
            eos = torch.as_tensor(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id],
                                  device=input_ids.device)
        pad = pad_token_id if pad_token_id is not None else (int(eos[0]) if eos is not None else 0)
        self.reset_cache_state()
        prefix = input_ids
        B = input_ids.shape[0]
        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        cache = SwapCache(len(self.model.layers)) if use_cache else None
        next_token = None
        for step in range(max_new_tokens):
            if step == 0:
                logits = self(prefix, cache=cache, use_cache=use_cache, last_logits_only=True)
            elif use_cache:
                logits = self(next_token, cache=cache, use_cache=True, last_logits_only=True)
            else:
                logits = self(prefix, last_logits_only=True)
            next_token = logits.argmax(dim=-1, keepdim=True)
            if eos is not None:
                next_token = torch.where(finished[:, None], torch.full_like(next_token, pad), next_token)
                finished |= (next_token == eos).any(dim=-1)
            prefix = torch.cat([prefix, next_token], dim=1)
            if eos is not None and bool(finished.all()):
                break
        return prefix

    def reset_cache_state(self):
        self.model.current_pos = 0


class SwapCache:
    """Per-layer KV cache for full attention plus an FLA cache for the linear layers."""

    def __init__(self, n_layers):
        from fla.models.utils import Cache as FLACache

        self._kv = [None] * n_layers
        self.linear_cache = FLACache()

    def get(self, layer_idx):
        return self._kv[layer_idx]

    def update(self, layer_idx, value):
        self._kv[layer_idx] = value

    def reset(self):
        self._kv = [None] * len(self._kv)
        self.linear_cache.reset()
