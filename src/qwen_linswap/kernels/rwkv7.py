"""RWKV-7 kernel: the generalised (diagonal-plus-low-rank) delta rule.

RWKV-7 (arXiv:2503.14456) evolves its state with a data-dependent
diagonal-plus-rank-one transition — FLA's ``chunk_rwkv7`` is a thin wrapper
around ``chunk_dplr_delta_rule`` computing (state stored as ``[K, V]``)

    S_t = Diag(exp(gk_t)) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T

with per-channel log decay ``gk_t``, "removal" vectors ``a_t``/``b_t`` and a
write key ``k_t``.  GDN is the special case

    a_t = k̂_t ⊙ exp(g_t),   b_t = -beta_t k̂_t,   k_t = beta_t k̂_t,   gk_t = g_t

(``k̂`` = L2-normalised key), since then ``b (a^T S) = -beta k̂ k̂^T Diag(e^g) S``
is exactly GDN's delta-rule erase and ``k v^T = beta k̂ v^T`` its write.  The
layer below is parameterised the RWKV-7 way — per-channel decay from a
low-rank projection (``w_lora``), a per-channel in-context learning rate
(``a_lora``, here ``b_proj``) and a separately modulated removal key
(``k_k``) — but keeps Qwen's q/k/v projections, short convolutions and
SiLU-gated output norm, exactly like the KDA kernel replaces KDA's output
gate.  RWKV-7's token shift, value residual and GroupNorm are not used.

Function-preserving init: the pretrained scalar decay row is tiled into the
low-rank ``f_proj`` (rank ≥ num_heads suffices, see ``init_lowrank_tiled``),
the scalar beta row into the low-rank per-channel ``b_proj``, ``dt_bias`` is
tiled, ``A_log`` copied and ``k_k = 1`` so the removal key equals ``k̂``.

FLA's own ``RWKV7Attention`` layer is not used because it fixes
``key_dim = hidden_size`` (Qwen's linear layers use key_dim = 2·hidden) and
bounds the decay to ``exp(-0.607·sigmoid(·)) ≥ 0.545`` per step, which cannot
represent the pretrained decays."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import FusedRMSNormSwishGate, ShortConvolution
from fla.modules.l2norm import l2_norm
from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule, fused_recurrent_dplr_delta_rule

from ..registry import KernelSpec, register_kernel
from .common import (
    copy_,
    copy_output_gate,
    copy_qkv_and_conv,
    get_gdn_source,
    init_lowrank_tiled,
    mark_hf_initialized,
    tile_vec,
)


class RWKV7DeltaLayer(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, conv_size=4, lora_rank=128, norm_eps=1e-6, layer_idx=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_k_dim = self.head_v_dim = head_dim
        self.num_heads = self.num_v_heads = num_heads
        self.key_dim = self.value_dim = num_heads * head_dim
        self.layer_idx = layer_idx
        self.use_short_conv = True

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, bias=False, activation="silu")

        # per-channel log decay:  g = -exp(A_log[h]) * softplus(f_proj(x) + dt_bias)
        self.f_proj = nn.Sequential(nn.Linear(hidden_size, lora_rank, bias=False),
                                    nn.Linear(lora_rank, self.key_dim, bias=False))
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(self.key_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        # per-channel in-context learning rate (RWKV-7's `a`):  beta = sigmoid(b_proj(x))
        self.b_proj = nn.Sequential(nn.Linear(hidden_size, lora_rank, bias=False),
                                    nn.Linear(lora_rank, self.key_dim, bias=False))
        # removal-key modulation (RWKV-7's k_k):  kappa = l2norm(k * k_k)
        self.k_k = nn.Parameter(torch.ones(self.key_dim))
        self.k_k._no_weight_decay = True

        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        B, T, _ = hidden_states.shape
        H, K = self.num_heads, self.head_k_dim
        last_state = get_layer_cache(self, past_key_values)
        conv_q = conv_k = conv_v = None
        if last_state is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]
        q, conv_q = self.q_conv1d(x=self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache)
        k, conv_k = self.k_conv1d(x=self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache)
        v, conv_v = self.v_conv1d(x=self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache)
        q, k, v = (rearrange(x, "b t (h d) -> b t h d", h=H) for x in (q, k, v))

        g = -self.A_log.float().exp().view(1, 1, H, 1) * F.softplus(
            self.f_proj(hidden_states).float().view(B, T, H, K) + self.dt_bias.float().view(1, 1, H, K))
        beta = torch.sigmoid(self.b_proj(hidden_states).float()).view(B, T, H, K)
        q = l2_norm(q)
        k_hat = l2_norm(k)
        kappa = l2_norm(k * self.k_k.view(1, 1, H, K).to(k.dtype))
        dt = q.dtype
        a = (kappa.float() * g.exp()).to(dt)          # read-out of the decayed state
        b = (-beta * kappa.float()).to(dt)            # erase strength (per channel)
        k_w = (beta * k_hat.float()).to(dt)           # write key
        gk = g.to(dt)

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        fn = fused_recurrent_dplr_delta_rule if (not torch.is_grad_enabled() and T <= 64) else chunk_dplr_delta_rule
        o, recurrent_state = fn(q=q, k=k_w, v=v, a=a, b=b, gk=gk, initial_state=recurrent_state,
                                output_final_state=use_cache)
        update_layer_cache(self, past_key_values, recurrent_state=recurrent_state,
                           conv_state=(conv_q, conv_k, conv_v), offset=T)

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "b t (h d) -> b t h d", h=H))
        o = self.o_proj(rearrange(o, "b t h d -> b t (h d)"))
        return o, None, past_key_values


def build(cfg, layer_idx):
    if cfg["linear_num_key_heads"] != cfg["linear_num_value_heads"]:
        raise ValueError("rwkv7: the DPLR kernel shares one head count for k and v")
    if cfg["linear_key_head_dim"] != cfg["linear_value_head_dim"]:
        raise ValueError("rwkv7: key and value head dims must match")
    return RWKV7DeltaLayer(
        hidden_size=cfg["emb_dim"],
        head_dim=cfg["linear_key_head_dim"],
        num_heads=cfg["linear_num_key_heads"],
        conv_size=cfg["linear_conv_kernel_dim"],
        norm_eps=cfg.get("rms_norm_eps", 1e-6),
        layer_idx=layer_idx,
    )


def init_from_gdn(layer, gdn_state, layer_idx, model_prefix="model"):
    src = get_gdn_source(gdn_state, layer_idx, model_prefix)
    copy_qkv_and_conv(layer, src)
    K = layer.head_k_dim
    init_lowrank_tiled(layer.f_proj, src["a"], K, "f_proj")   # per-channel decay == scalar decay
    init_lowrank_tiled(layer.b_proj, src["b"], K, "b_proj")   # per-channel lr == scalar beta
    copy_(layer.A_log, src["A_log"], "A_log")
    copy_(layer.dt_bias, tile_vec(src["dt_bias"], K), "dt_bias")
    with torch.no_grad():
        layer.k_k.fill_(1.0)                                  # removal key == write key
    copy_output_gate(layer, src)
    mark_hf_initialized(layer)
    return layer


register_kernel(KernelSpec(
    name="rwkv7",
    description="RWKV-7 generalised delta rule (FLA DPLR kernel); per-channel decay + learning rate and a "
                "separate removal key, tiled from the scalar GDN gates.",
    build=build,
    init_from_gdn=init_from_gdn,
    new_param_names=("f_proj", "b_proj", "A_log", "dt_bias", "k_k"),
    exact_init=True,
))
