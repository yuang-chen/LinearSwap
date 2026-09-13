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
(``k_k``) — but keeps the backbone's q/k/v projections, short convolutions and
SiLU-gated output norm, exactly like the KDA kernel replaces KDA's output
gate.  RWKV-7's token shift, value residual and GroupNorm are not used.

Function-preserving init: the pretrained scalar decay row is tiled into the
low-rank ``f_proj`` (rank ≥ num_heads suffices, see ``init_lowrank_tiled``),
the scalar beta row into the low-rank per-channel ``b_proj``, ``dt_bias`` is
tiled, ``A_log`` copied and ``k_k = 1`` so the removal key equals ``k̂``.

FLA's own ``RWKV7Attention`` layer is not used because it fixes
``key_dim = hidden_size`` (the backbone's linear layers use key_dim = 2·hidden) and
bounds the decay to ``exp(-0.607·sigmoid(·)) ≥ 0.545`` per step, which cannot
represent the pretrained decays."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules.l2norm import l2_norm
from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule, fused_recurrent_dplr_delta_rule

from ..registry import KernelSpec, register_kernel
from .base import BackboneMixer, build_backbone_mixer
from .common import copy_, copy_shared_from_gdn, get_gdn_source, init_lowrank_tiled, mark_hf_initialized, tile_vec


class RWKV7DeltaLayer(BackboneMixer):
    def __init__(self, hidden_size, head_dim, num_heads, conv_size=4, norm_eps=1e-6, layer_idx=None, lora_rank=128):
        super().__init__(hidden_size, head_dim, num_heads, conv_size, norm_eps, layer_idx, qk_l2norm=True)
        # per-channel log decay:  g = -exp(A_log[h]) * softplus(f_proj(x) + dt_bias)   (RWKV-7's w_lora)
        self.f_proj = nn.Sequential(nn.Linear(hidden_size, lora_rank, bias=False),
                                    nn.Linear(lora_rank, self.key_dim, bias=False))
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(self.key_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        # per-channel in-context learning rate (RWKV-7's a_lora):  beta = sigmoid(b_proj(x))
        self.b_proj = nn.Sequential(nn.Linear(hidden_size, lora_rank, bias=False),
                                    nn.Linear(lora_rank, self.key_dim, bias=False))
        # removal-key modulation (RWKV-7's k_k):  kappa = l2norm(k * k_k)
        self.k_k = nn.Parameter(torch.ones(self.key_dim))
        self.k_k._no_weight_decay = True

    def recurrence(self, hidden_states, q, k, v, state, use_cache):
        B, T, H, K = k.shape
        g = -self.A_log.float().exp().view(1, 1, H, 1) * F.softplus(
            self.f_proj(hidden_states).float().view(B, T, H, K) + self.dt_bias.float().view(1, 1, H, K))
        beta = torch.sigmoid(self.b_proj(hidden_states).float()).view(B, T, H, K)
        kappa = l2_norm(k * self.k_k.view(1, 1, H, K).to(k.dtype))
        dt = q.dtype
        a = (kappa.float() * g.exp()).to(dt)          # read-out of the decayed state
        b = (-beta * kappa.float()).to(dt)            # erase strength (per channel)
        k_w = (beta * k.float()).to(dt)               # write key
        fn = fused_recurrent_dplr_delta_rule if (self.use_recurrent_kernel and T <= 64) else chunk_dplr_delta_rule
        return fn(q=q, k=k_w, v=v, a=a, b=b, gk=g.to(dt), initial_state=state, output_final_state=use_cache)


def build(cfg, layer_idx):
    return build_backbone_mixer(RWKV7DeltaLayer, cfg, layer_idx)


def init_from_gdn(layer, gdn_state, layer_idx, model_prefix="model"):
    src = get_gdn_source(gdn_state, layer_idx, model_prefix)
    copy_shared_from_gdn(layer, src)
    K = layer.head_k_dim
    init_lowrank_tiled(layer.f_proj, src["a"], K, "f_proj")   # per-channel decay == scalar decay
    init_lowrank_tiled(layer.b_proj, src["b"], K, "b_proj")   # per-channel lr == scalar beta
    copy_(layer.A_log, src["A_log"], "A_log")
    copy_(layer.dt_bias, tile_vec(src["dt_bias"], K), "dt_bias")
    with torch.no_grad():
        layer.k_k.fill_(1.0)                                  # removal key == write key
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
