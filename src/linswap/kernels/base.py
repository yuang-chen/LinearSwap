"""Base class for kernels that are *not* an existing FLA layer but an FLA **op** (a recurrence).

``BackboneMixer`` provides everything a linear-attention layer in this backbone shares —
q/k/v projections, three short convolutions, optional q/k L2 normalisation, the FLA cache
protocol and the SiLU-gated output norm — so a kernel only implements ``recurrence`` (and
adds its own parameters in ``__init__``).  ``kernels/rwkv7.py`` and ``kernels/mamba2.py`` are
the two examples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormSwishGate, ShortConvolution
from fla.modules.l2norm import l2_norm


class BackboneMixer(nn.Module):
    """``num_v_heads`` may be a multiple of ``num_heads`` (grouped value heads, as in the larger Qwen
    backbones): q/k are computed for ``num_heads`` and repeated per value-head group before the
    recurrence, so ``recurrence`` always sees q, k, v with ``num_v_heads`` heads."""

    def __init__(self, hidden_size, head_dim, num_heads, num_v_heads=None, conv_size=4, norm_eps=1e-6,
                 layer_idx=None, qk_l2norm=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_k_dim = self.head_v_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads or num_heads
        if self.num_v_heads % num_heads:
            raise ValueError(f"num_v_heads={self.num_v_heads} must be a multiple of num_heads={num_heads}")
        self.v_groups = self.num_v_heads // num_heads
        self.key_dim = num_heads * head_dim
        self.value_dim = self.num_v_heads * head_dim
        self.layer_idx = layer_idx
        self.qk_l2norm = qk_l2norm
        self.use_short_conv = True
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, bias=False, activation="silu")
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def recurrence(self, hidden_states, q, k, v, state, use_cache, cu_seqlens=None):
        """q, k: [B, T, H, K] (L2-normalised if ``qk_l2norm``), v: [B, T, H, V]; return (o [B, T, H, V], new state).
        With ``cu_seqlens`` the batch is packed (B == 1) and ``state`` has one entry per sequence."""
        raise NotImplementedError

    @property
    def use_recurrent_kernel(self):
        return not torch.is_grad_enabled()

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        B, T = hidden_states.shape[:2]
        H = self.num_heads
        last_state = get_layer_cache(self, past_key_values)
        # A padding mask packs the batch into one varlen sequence (B == 1, ``cu_seqlens``), as FLA's layers do.
        hidden_states, indices, cu_seqlens = unpad_hidden_states(hidden_states, kwargs.get("cu_seqlens"),
                                                                 attention_mask, T)
        conv_q = conv_k = conv_v = None
        if last_state is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]
        q, conv_q = self.q_conv1d(x=self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache,
                                  cu_seqlens=cu_seqlens)
        k, conv_k = self.k_conv1d(x=self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache,
                                  cu_seqlens=cu_seqlens)
        v, conv_v = self.v_conv1d(x=self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache,
                                  cu_seqlens=cu_seqlens)
        q, k = (rearrange(x, "b t (h d) -> b t h d", h=H) for x in (q, k))
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_v_heads)
        if self.qk_l2norm:
            q, k = l2_norm(q), l2_norm(k)
        if self.v_groups > 1:  # grouped value heads: share each q/k head across its value-head group
            q, k = (x.repeat_interleave(self.v_groups, dim=2) for x in (q, k))
        state = last_state["recurrent_state"] if last_state is not None else None
        o, state = self.recurrence(hidden_states, q, k, v, state, use_cache, cu_seqlens=cu_seqlens)
        update_layer_cache(self, past_key_values, recurrent_state=state, conv_state=(conv_q, conv_k, conv_v), offset=T)
        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "b t (h d) -> b t h d", h=self.num_v_heads))
        o = self.o_proj(rearrange(o, "b t h d -> b t (h d)"))
        return repad_hidden_states(o, indices, B, T), None, past_key_values


def build_backbone_mixer(mixer_cls, cfg, layer_idx, **extra):
    if cfg["linear_key_head_dim"] != cfg["linear_value_head_dim"]:
        raise ValueError(f"{mixer_cls.__name__}: key and value head dims must match")
    return mixer_cls(hidden_size=cfg["emb_dim"], head_dim=cfg["linear_key_head_dim"], num_heads=cfg["linear_num_key_heads"],
                     num_v_heads=cfg["linear_num_value_heads"], conv_size=cfg["linear_conv_kernel_dim"],
                     norm_eps=cfg.get("rms_norm_eps", 1e-6), layer_idx=layer_idx, **extra)
