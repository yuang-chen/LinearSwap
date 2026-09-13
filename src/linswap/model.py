"""Hybrid (Gated-DeltaNet + full attention) backbone with pluggable linear-attention token mixers.

Full-attention layers, MLPs and norms come from ``components.py``; the
linear-attention layers are built by the kernel registry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .components import FeedForward, GroupedQueryAttention, RMSNorm, compute_rope_params

from .registry import KernelSpec, get_kernel


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

    def __init__(self, cfg, kernel: KernelSpec):
        super().__init__()
        self.cfg = cfg
        self.kernel = kernel
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        layer_types = cfg.get("layer_types", ["full_attention"] * cfg["n_layers"])
        if len(layer_types) != cfg["n_layers"]:
            raise ValueError("len(layer_types) must equal n_layers")
        self.layers = nn.ModuleList([TransformerBlock(cfg, lt, idx, kernel) for idx, lt in enumerate(layer_types)])
        self.norm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        # RoPE tables are computed lazily per (device, dtype) rather than registered as buffers:
        # non-persistent buffers are not restored by HF `from_pretrained` (meta-device init).
        self._rope_tables = {}
        self.current_pos = 0
        self.gradient_checkpointing = False

    def rope_tables(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._rope_tables:
            cfg = self.cfg
            head_dim = cfg["emb_dim"] // cfg["n_heads"] if cfg["head_dim"] is None else cfg["head_dim"]
            cos, sin = compute_rope_params(
                head_dim=head_dim, theta_base=cfg["rope_base"], context_length=cfg["context_length"],
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
        self.kernel = get_kernel(kernel)
        self.model = LinearSwapBackbone(cfg, self.kernel)
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

    def new_parameters(self):
        """Named parameters the kernel spec considers new / gate parameters."""
        return [(n, p) for n, p in self.named_parameters() if self.kernel.is_new_param(n)]

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
                 use_cache=True):
        """Greedy decoding (sampling arguments are accepted for API compatibility)."""
        self.reset_cache_state()
        prefix = input_ids
        cache = SwapCache(len(self.model.layers)) if use_cache else None
        logits = self(prefix, cache=cache, use_cache=use_cache, last_logits_only=True)
        next_token = logits.argmax(dim=-1, keepdim=True)
        prefix = torch.cat([prefix, next_token], dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            return prefix
        for _ in range(max_new_tokens - 1):
            if use_cache:
                logits = self(next_token, cache=cache, use_cache=True, last_logits_only=True)
            else:
                logits = self(prefix, last_logits_only=True)
            next_token = logits.argmax(dim=-1, keepdim=True)
            prefix = torch.cat([prefix, next_token], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
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
