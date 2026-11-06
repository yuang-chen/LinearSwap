"""Mamba-2 kernel: the SSD (state space duality) recurrence, on FLA's GLA kernels.

Mamba-2 (arXiv:2405.21060) is a scalar-decay linear RNN per head:

    Δ_t   = softplus(dt_proj(x_t) + dt_bias)                (per head)
    S_t   = exp(-Δ_t·exp(A_log)) S_{t-1} + Δ_t B_t x_t^T     (state [N, P])
    y_t   = C_t^T S_t + D ⊙ x_t

which is linear attention with a scalar decay (``S_t = exp(g_t) S_{t-1} + k_t v_t^T``) with
``g_t = -Δ_t exp(A_log)``, ``k_t = Δ_t B_t``, ``q_t = C_t``, ``v_t = x_t``.  It runs on FLA's GLA kernels with
the scalar decay repeated over the key channels, not on the simple-GLA kernel that states exactly this: on
Hopper GPUs simple-GLA's backward goes through FLA's TileLang kernel (fla#640), which now and then never
returns (a 16K-token step hung after 150–190 good ones, three times).  The GLA kernels give the same
outputs and state, gradients that match float64 autograd to 1e-3, and take bf16 inputs.
Compared with GDN (``S_t = α_t (I - β_t k̂ k̂^T) S_{t-1} + β_t k̂ v^T``) it has the
*same decay* (``α_t = exp(-exp(A_log)·softplus(a_t + dt_bias))``) but **no
delta-rule erase** and the write is scaled by Δ_t instead of beta_t.  The
swap is therefore not function preserving: B/C/x/decay/gate/output weights are
copied from q/k/v/decay/gate/output, ``D`` starts at 0 (no skip, like GDN) and
``beta`` (``in_proj_b``) is dropped.  Distillation is required
(``exact_init=False``).

Layout choices: one SSD "group" per head (``n_groups = num_heads``) so B and C
have the same shape as the backbone's k and q; q/k are L2-normalised and scaled by
1/sqrt(N) as in the pretrained GDN (Mamba-2 itself does neither, but the
pretrained projections were trained under it); the backbone's SiLU-gated output norm
is ``norm_before_gate=True`` in Mamba-2 terms.  Mamba-2's own CUDA/Triton
kernels (``mamba_ssm``) are not needed."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.ops.gla import chunk_gla, fused_recurrent_gla

from ..registry import KernelSpec, register_kernel
from .base import BackboneMixer, build_backbone_mixer
from .common import copy_, copy_shared_from_gdn, get_gdn_source, mark_hf_initialized


class Mamba2SSDLayer(BackboneMixer):
    def __init__(self, hidden_size, head_dim, num_heads, num_v_heads=None, conv_size=4, norm_eps=1e-6, layer_idx=None,
                 qk_l2norm=True):
        super().__init__(hidden_size, head_dim, num_heads, num_v_heads, conv_size, norm_eps, layer_idx, qk_l2norm=qk_l2norm)
        HV = self.num_v_heads                       # Δ, A, D are per value head (SSD head)
        self.dt_proj = nn.Linear(hidden_size, HV, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(HV, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(HV) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.D = nn.Parameter(torch.zeros(HV))
        self.D._no_weight_decay = True

    def recurrence(self, hidden_states, q, k, v, state, use_cache, cu_seqlens=None):
        T, H, K = k.shape[1], k.shape[2], k.shape[3]
        delta = F.softplus(self.dt_proj(hidden_states).float() + self.dt_bias.float())   # Δ_t  [B, T, H]
        g = (-self.A_log.float().exp() * delta).unsqueeze(-1).expand(-1, -1, -1, K)       # log decay, on every key channel
        k_w = k * delta.unsqueeze(-1).to(k.dtype)                                           # Δ_t B_t
        common = dict(q=q, k=k_w, v=v, scale=self.head_k_dim ** -0.5, initial_state=state, output_final_state=use_cache,
                      cu_seqlens=cu_seqlens)
        if self.use_recurrent_kernel and T <= 64:
            o, state = fused_recurrent_gla(gk=g, **common)
        else:
            o, state = chunk_gla(g=g, **common)
        return o.to(v.dtype) + v * self.D.view(1, 1, H, 1).to(v.dtype), state


def build(cfg, layer_idx):
    return build_backbone_mixer(Mamba2SSDLayer, cfg, layer_idx)


def init_from_gdn(layer, gdn_state, layer_idx, model_prefix="model"):
    src = get_gdn_source(gdn_state, layer_idx, model_prefix)
    copy_shared_from_gdn(layer, src)
    copy_(layer.dt_proj.weight, src["a"], "dt_proj")
    copy_(layer.A_log, src["A_log"], "A_log")
    copy_(layer.dt_bias, src["dt_bias"], "dt_bias")
    with torch.no_grad():
        layer.D.zero_()
    mark_hf_initialized(layer)
    return layer


register_kernel(KernelSpec(
    name="mamba2",
    description="Mamba-2 SSD recurrence (FLA simple-GLA kernel); GDN decay/projections copied, delta-rule erase "
                "and beta dropped — NOT function preserving.",
    build=build,
    init_from_gdn=init_from_gdn,
    new_param_names=("dt_proj", "A_log", "dt_bias", "D"),
    exact_init=False,
    notes="Inexact swap (no delta rule, Δ-scaled writes): distil before SFT.",
))
