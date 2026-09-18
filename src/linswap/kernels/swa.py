"""Sliding-window softmax attention with sinks (``swa``), after "Sliding-window beats linear
attention" (arXiv 2608.28444).

That paper takes a pretrained *dense* model and only changes the attention mask: every query
attends to the previous ``w`` tokens plus the first ``s = 4`` tokens (the attention sinks of
Xiao et al., 2024, without which SWA collapses once position 0 leaves the window).  With no
training at all, SWA(64, 4) recovers 99 % of the teacher's short-context average and beats
every linear-attention retrofit they tried on needle-in-a-haystack by 2-10x.

The backbone here has no dense layers to re-mask -- the 18 swappable layers are already Gated
DeltaNet -- so this kernel is the analogous construction rather than a reproduction: softmax
attention over the *pretrained GDN layer's own* q/k/v, restricted to that mask.  Everything
except the recurrence is a pure copy (q/k/v projections, the three short convolutions, the
swish-gated output path), so ``swa`` isolates "replace the delta-rule recurrence with a
64-wide softmax window" and nothing else.  Three points where the setting forces a choice the
paper did not have to make, all switchable by environment variable at build time:

``LINSWAP_SWA_ROPE`` (default 1)
    GDN has no positional encoding -- it is a recurrence -- so its q/k were pretrained without
    RoPE.  A 64-wide softmax window without positional information is order-blind beyond the
    4-tap short convolution, which is a strictly weaker layer, so RoPE (over the linear layers'
    own head dim, at the backbone's theta) is on by default.  Set to 0 to ablate.
``LINSWAP_SWA_WINDOW`` / ``LINSWAP_SWA_SINKS`` (default 64 / 4)
    The paper's headline configuration; it also reports 128/256/512, which trade memory for
    needle accuracy.
``LINSWAP_SWA_BLOCK`` (default 1024)
    Prefill is computed one query block at a time against the (at most ``s + w + block``) keys
    that block can see, which keeps attention linear in sequence length without a fused kernel.

The init is *not* function preserving (``exact_init=False``): GDN L2-normalises q and k inside
its kernel, so the pretrained projections carry no usable scale for a softmax.  q and k are
L2-normalised here too and the logits get a learnable per-head temperature initialised to
``sqrt(head_k_dim)``, which puts the logit scale where ``q·k/sqrt(d)`` would be for
unit-variance projections.  That temperature is the kernel's only new parameter.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import FusedRMSNormSwishGate, RotaryEmbedding, ShortConvolution

from .fla_layer import register_fla_kernel


def windowed_sdpa(q, k, v, q_pos, k_pos, window: int, sinks: int):
    """Softmax attention of ``q`` over the keys it may see: ``{i <= t} & ({i < s} | {i > t - w})``.

    ``q`` is ``[B, H, Tq, Dk]``, ``k`` / ``v`` are ``[B, H, Tk, Dk] / [B, H, Tk, Dv]`` and the two
    position vectors hold the *absolute* index of every query and key, so the caller is free to
    hand over any superset of the visible keys (a cached window, a block of the current chunk) as
    long as it contains no duplicates.  Position ``t`` always sees itself, so no row is empty.
    """
    mask = (k_pos[None, :] <= q_pos[:, None]) & ((k_pos[None, :] < sinks) | (k_pos[None, :] > q_pos[:, None] - window))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None], scale=1.0)


class SlidingWindowAttention(nn.Module):
    """GDN's projections and output gate with the recurrence replaced by SWA(window, sinks).

    The cache keeps two contiguous ranges of keys and values -- the sinks ``[0, n_s)`` and the
    window ``[max(s, T - w), T)`` -- which are disjoint by construction and bounded by ``s + w``
    entries, so decoding is O(1) in memory like the linear kernels it replaces.
    """

    def __init__(self, hidden_size: int = 1024, head_dim: int = 128, num_heads: int = 16,
                 num_v_heads: int | None = None, expand_v: float = 1.0, window_size: int = 64,
                 num_sinks: int = 4, use_rope: bool = True, rope_theta: float = 1e7, block_size: int = 1024,
                 use_short_conv: bool = True, conv_size: int = 4, conv_bias: bool = False,
                 layer_idx: int | None = None, norm_eps: float = 1e-6, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.num_heads = num_heads
        self.num_v_heads = num_heads if num_v_heads is None else num_v_heads
        if self.num_v_heads % self.num_heads:
            raise ValueError(f"num_v_heads={self.num_v_heads} must be a multiple of num_heads={self.num_heads}")
        self.num_kv_groups = self.num_v_heads // self.num_heads
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.window_size, self.num_sinks, self.block_size = window_size, num_sinks, block_size
        self.use_short_conv, self.conv_size, self.layer_idx = use_short_conv, conv_size, layer_idx

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_short_conv:
            conv = dict(kernel_size=conv_size, bias=conv_bias, activation="silu")
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, **conv)
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, **conv)
            self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, **conv)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.logit_scale = nn.Parameter(torch.full((self.num_heads,), float(head_dim) ** 0.5))
        self.rotary = RotaryEmbedding(dim=self.head_k_dim, base=rope_theta) if use_rope else None

    def project(self, x, conv_states, use_cache):
        """q/k/v projections + short convolutions, as ``[B, T, H, D]``."""
        cq, ck, cv = conv_states
        if self.use_short_conv:
            q, cq = self.q_conv1d(x=self.q_proj(x), cache=cq, output_final_state=use_cache)
            k, ck = self.k_conv1d(x=self.k_proj(x), cache=ck, output_final_state=use_cache)
            v, cv = self.v_conv1d(x=self.v_proj(x), cache=cv, output_final_state=use_cache)
        else:
            q, k, v = (F.silu(p(x)) for p in (self.q_proj, self.k_proj, self.v_proj))
        q, k = (rearrange(t, "... (h d) -> ... h d", d=self.head_k_dim) for t in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        return q, k, v, (cq, ck, cv)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache: bool = False,
                output_attentions: bool = False, **kwargs):
        if attention_mask is not None:
            raise ValueError("swa expects unpadded sequences; linswap packs them without a padding mask")
        T = hidden_states.shape[1]
        last = get_layer_cache(self, past_key_values)
        state = last.get("recurrent_state") if last is not None else None
        conv_states = last["conv_state"] if last is not None and last.get("conv_state") is not None else (None,) * 3
        offset = state[-1] if state is not None else 0

        q, k, v, conv_states = self.project(hidden_states, conv_states, use_cache)
        if self.rotary is not None:
            q, k = self.rotary(q, k, seqlen_offset=offset, max_seqlen=offset + T)
        q, k = (F.normalize(t, dim=-1) for t in (q, k))
        q = q * self.logit_scale.to(q.dtype).view(1, 1, -1, 1)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # [B, H, T, D]
        if self.num_kv_groups > 1:                                 # grouped value attention: share q/k across v heads
            q, k = (t.repeat_interleave(self.num_kv_groups, dim=1) for t in (q, k))

        pos = torch.arange(offset, offset + T, device=q.device)
        cached = self.cached_keys(state, q.device)
        out = []
        for lo in range(0, T, self.block_size):
            hi = min(lo + self.block_size, T)
            win = max(0, lo - self.window_size + 1)                # first chunk-local key this block can see
            snk = min(max(self.num_sinks - offset, 0), win)        # sinks still inside the chunk, never overlapping ``win``
            ks = [*cached[0], k[:, :, :snk], k[:, :, win:hi]]
            vs = [*cached[1], v[:, :, :snk], v[:, :, win:hi]]
            ps = [*cached[2], pos[:snk], pos[win:hi]]
            out.append(windowed_sdpa(q[:, :, lo:hi], torch.cat(ks, 2), torch.cat(vs, 2),
                                     pos[lo:hi], torch.cat(ps), self.window_size, self.num_sinks))
        o = torch.cat(out, dim=2).transpose(1, 2)                  # [B, T, H, Dv]

        if use_cache:
            update_layer_cache(self, past_key_values, offset=T,
                               conv_state=conv_states if self.use_short_conv else None,
                               recurrent_state=self.next_state(state, k, v, offset, T))
        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        return self.o_proj(rearrange(o, "b t h d -> b t (h d)")), None, past_key_values

    @staticmethod
    def cached_keys(state, device):
        """``(keys, values, positions)`` of the cached sink and window ranges, or three empty lists."""
        if state is None:
            return [], [], []
        sink_k, sink_v, sink_lo, win_k, win_v, win_lo, _ = state
        span = lambda lo, t: torch.arange(lo, lo + t.shape[2], device=device)  # noqa: E731
        return ([sink_k, win_k], [sink_v, win_v], [span(sink_lo, sink_k), span(win_lo, win_k)])

    def next_state(self, state, k, v, offset: int, T: int):
        """Keys/values of the sinks ``[0, n_s)`` and the window ``[max(s, end - w), end)``.

        Both ranges are contiguous in absolute position, so they are carried as plain slices plus
        their start index -- no gather, no device synchronisation per decoded token.
        """
        end, s, w = offset + T, self.num_sinks, self.window_size
        if state is None:
            sink_k, sink_v, old_k, old_v, old_lo = k[:, :, :s], v[:, :, :s], None, None, 0
        else:
            sink_k, sink_v, _, old_k, old_v, old_lo, _ = state
            if offset < s:                                         # sinks not yet complete: extend from this chunk
                sink_k = torch.cat([sink_k, k[:, :, :s - offset]], 2)
                sink_v = torch.cat([sink_v, v[:, :, :s - offset]], 2)
        win_lo = max(s, end - w)
        parts_k, parts_v = [], []
        if old_k is not None and old_lo + old_k.shape[2] > win_lo:  # keep the still-visible tail of the cached window
            take = max(0, win_lo - old_lo)
            parts_k.append(old_k[:, :, take:])
            parts_v.append(old_v[:, :, take:])
        chunk_lo = max(win_lo - offset, max(s - offset, 0))         # window keys coming from this chunk
        if chunk_lo < T:
            parts_k.append(k[:, :, chunk_lo:])
            parts_v.append(v[:, :, chunk_lo:])
        win_k = torch.cat(parts_k, 2) if parts_k else k[:, :, T:]  # empty while every token is still a sink
        win_v = torch.cat(parts_v, 2) if parts_v else v[:, :, T:]
        return (sink_k.contiguous(), sink_v.contiguous(), 0,
                win_k.contiguous(), win_v.contiguous(), end - win_k.shape[2], end)


def swa_kwargs(cfg):
    return dict(window_size=int(os.environ.get("LINSWAP_SWA_WINDOW", "64")),
                num_sinks=int(os.environ.get("LINSWAP_SWA_SINKS", "4")),
                use_rope=os.environ.get("LINSWAP_SWA_ROPE", "1") != "0",
                block_size=int(os.environ.get("LINSWAP_SWA_BLOCK", "1024")),
                rope_theta=float(cfg["rope_base"]))


register_fla_kernel(
    "swa", SlidingWindowAttention,
    description="Sliding-window softmax attention with sinks (arXiv 2608.28444); GDN's projections, "
                "conv and output gate copied, recurrence replaced by SWA(64, 4). Window/sinks/RoPE "
                "from LINSWAP_SWA_WINDOW / _SINKS / _ROPE.",
    output_gate="native",              # the layer already builds the backbone's g_proj + swish-gated RMSNorm
    layer_kwargs=swa_kwargs,
    new_param_names=("logit_scale",),
    exact_init=False,
    notes="Not function preserving: GDN L2-normalises q/k inside its kernel, so the pretrained "
          "projections carry no softmax scale; the per-head logit temperature starts at sqrt(head_k_dim).",
)
