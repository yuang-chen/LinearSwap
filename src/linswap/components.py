import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rope_params(
    head_dim,
    theta_base=10_000,
    context_length=4096,
    partial_rotary_factor=1.0,
    dtype=torch.float32,
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    rotary_dim = int(head_dim * partial_rotary_factor)
    rotary_dim = max(2, rotary_dim - (rotary_dim % 2))

    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(0, rotary_dim, 2, dtype=dtype)[: (rotary_dim // 2)].float() / rotary_dim
        )
    )

    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    _, _, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    rot_dim = cos.shape[-1]
    if rot_dim > head_dim:
        raise ValueError(f"RoPE dim {rot_dim} cannot exceed head_dim {head_dim}.")

    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    x1 = x_rot[..., : rot_dim // 2]
    x2 = x_rot[..., rot_dim // 2 :]

    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x_rot * cos) + (rotated * sin)

    x_out = torch.cat([x_rotated, x_pass], dim=-1)
    return x_out.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(emb_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        x_norm = self._norm(x.float())
        x_norm = x_norm * (1.0 + self.weight.float())
        return x_norm.to(dtype=x.dtype)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dtype = cfg["dtype"]
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=dtype, bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=dtype, bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = F.silu(x_fc1) * x_fc2
        return self.fc3(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "d_in must be divisible by num_heads"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out * 2, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None, use_cache=False):
        b, num_tokens, _ = x.shape

        q_and_gate = self.W_query(x)
        q_and_gate = q_and_gate.view(b, num_tokens, self.num_heads, self.head_dim * 2)
        queries, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(b, num_tokens, self.d_out)

        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        prev_len = 0
        if cache is not None:
            prev_k, prev_v = cache
            if prev_k is not None:
                prev_len = prev_k.size(2)
                values_cat_raw = torch.cat([prev_v, values_new], dim=2)
            else:
                values_cat_raw = values_new
        else:
            values_cat_raw = values_new

        queries = apply_rope(queries, cos, sin, offset=start_pos)

        if cache is None:
            keys = apply_rope(keys_new, cos, sin, offset=start_pos)
            next_cache = (keys, values_new) if use_cache else None
        else:
            # RoPE only the new keys at their absolute positions, then append.
            keys_new_rope = apply_rope(keys_new, cos, sin, offset=start_pos)
            if prev_k is not None:
                keys = torch.cat([prev_k, keys_new_rope], dim=2)
                next_cache = (
                    torch.cat([prev_k, keys_new_rope], dim=2),
                    values_cat_raw,
                )
            else:
                keys = keys_new_rope
                next_cache = (keys_new_rope, values_new)

        # Expand grouped KV heads to match query heads for SDPA.
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values_cat_raw.repeat_interleave(self.group_size, dim=1)

        if cache is None:
            # Training path: use Flash/Memory-Efficient SDPA with causal masking.
            attn_output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            # Generation path with KV cache: build a 2-D causal mask.
            kv_len = keys.size(-2)
            q_pos = torch.arange(num_tokens, device=queries.device).unsqueeze(1) + start_pos
            kv_pos = torch.arange(kv_len, device=keys.device).unsqueeze(0)
            attn_mask = q_pos >= kv_pos  # True = keep
            attn_output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        context = attn_output.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        context = context * torch.sigmoid(gate)
        out = self.out_proj(context)
        return out, next_cache
