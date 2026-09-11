"""Qwen3.5 backbone with pluggable linear-attention token mixers.

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
    def __init__(self, cfg, layer_type, layer_idx, kernel: KernelSpec):
        super().__init__()
        self.layer_type = layer_type
        self.layer_idx = layer_idx

        if layer_type == "full_attention":
            self.token_mixer = GroupedQueryAttention(
                d_in=cfg["emb_dim"],
                num_heads=cfg["n_heads"],
                head_dim=cfg["head_dim"],
                num_kv_groups=cfg["n_kv_groups"],
                qk_norm=cfg["qk_norm"],
                dtype=cfg["dtype"],
            )
        elif layer_type == "linear_attention":
            self.token_mixer = kernel.build(cfg, layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None, linear_cache=None, use_cache=False):
        shortcut = x
        x = self.norm1(x)

        if self.layer_type == "full_attention":
            h, next_kv_cache = self.token_mixer(
                x, mask=mask, cos=cos, sin=sin, start_pos=start_pos, cache=cache, use_cache=use_cache,
            )
        else:
            h, _, _ = self.token_mixer(x, past_key_values=linear_cache, use_cache=linear_cache is not None)
            next_kv_cache = None

        x = h + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        return x + shortcut, next_kv_cache


class Qwen3_5LinearSwapModel(nn.Module):
    """Qwen3.5 with every ``linear_attention`` layer replaced by ``kernel``."""

    def __init__(self, cfg, kernel: str | KernelSpec = "gdn"):
        super().__init__()
        self.kernel = get_kernel(kernel)
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        layer_types = cfg.get("layer_types", ["full_attention"] * cfg["n_layers"])
        if len(layer_types) != cfg["n_layers"]:
            raise ValueError("len(layer_types) must equal n_layers")
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg, lt, idx, self.kernel) for idx, lt in enumerate(layer_types)]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg["emb_dim"] // cfg["n_heads"] if cfg["head_dim"] is None else cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            partial_rotary_factor=cfg.get("partial_rotary_factor", 1.0),
            dtype=torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.current_pos = 0
        self.gradient_checkpointing = False

    # ------------------------------------------------------------------ utils
    @property
    def linear_layers(self):
        return [b.token_mixer for b in self.trf_blocks if b.layer_type == "linear_attention"]

    def new_parameters(self):
        """Named parameters the kernel spec considers new / gate parameters."""
        return [(n, p) for n, p in self.named_parameters() if self.kernel.is_new_param(n)]

    # ---------------------------------------------------------------- forward
    def forward(self, in_idx, cache=None, use_cache=False, return_hidden=False,
                return_hidden_before_norm=False, last_logits_only=False):
        x = self.tok_emb(in_idx)
        num_tokens = x.shape[1]

        if cache is not None or use_cache:
            start_pos = self.current_pos
            self.current_pos = start_pos + num_tokens
        else:
            start_pos = 0
        # Full attention builds its own compact causal mask; never materialise a dense one.
        mask = None
        linear_cache = cache.linear_cache if cache is not None else None

        for i, block in enumerate(self.trf_blocks):
            kv_cache = cache.get(i) if cache is not None else None
            if self.gradient_checkpointing and self.training:
                x, new_kv = checkpoint(
                    block, x, mask, self.cos, self.sin, start_pos, kv_cache, linear_cache, use_cache,
                    use_reentrant=False,
                )
            else:
                x, new_kv = block(
                    x, mask=mask, cos=self.cos, sin=self.sin, start_pos=start_pos,
                    cache=kv_cache, linear_cache=linear_cache, use_cache=use_cache,
                )
            if cache is not None and new_kv is not None:
                cache.update(i, new_kv)

        if return_hidden_before_norm:
            return x.to(self.cfg["dtype"])
        x = self.final_norm(x)
        if return_hidden:
            return x.to(self.cfg["dtype"])
        if last_logits_only:
            return self.out_head(x[:, -1:, :].to(self.cfg["dtype"])).squeeze(1)
        return self.out_head(x.to(self.cfg["dtype"]))

    # --------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, eos_token_id=None, temperature=0.0, top_k=1, top_p=1.0,
                 use_cache=True):
        """Greedy decoding (sampling arguments are accepted for API compatibility)."""
        self.reset_cache_state()
        prefix = input_ids
        cache = SwapCache(len(self.trf_blocks)) if use_cache else None

        if use_cache:
            logits = self(prefix, cache=cache, use_cache=True, last_logits_only=True)
        else:
            logits = self(prefix, last_logits_only=True)
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
        self.current_pos = 0


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
