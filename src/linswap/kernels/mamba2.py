"""Mamba-2 kernel: the SSD (state space duality) recurrence, on FLA's simple-GLA kernel.

Mamba-2 (arXiv:2405.21060) is a scalar-decay linear RNN per head:

    Δ_t   = softplus(dt_proj(x_t) + dt_bias)                (per head)
    S_t   = exp(-Δ_t·exp(A_log)) S_{t-1} + Δ_t B_t x_t^T     (state [N, P])
    y_t   = C_t^T S_t + D ⊙ x_t

which is exactly ``chunk_simple_gla`` (``S_t = exp(g_t) S_{t-1} + k_t v_t^T``)
with ``g_t = -Δ_t exp(A_log)``, ``k_t = Δ_t B_t``, ``q_t = C_t``, ``v_t = x_t``.
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
from einops import rearrange
from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import FusedRMSNormSwishGate, ShortConvolution
from fla.modules.l2norm import l2_norm
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla

from ..registry import KernelSpec, register_kernel
from .common import copy_, copy_output_gate, copy_qkv_and_conv, get_gdn_source, mark_hf_initialized


class Mamba2SSDLayer(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, conv_size=4, norm_eps=1e-6, layer_idx=None, qk_l2norm=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_k_dim = self.head_v_dim = head_dim
        self.num_heads = self.num_v_heads = num_heads
        self.key_dim = self.value_dim = num_heads * head_dim
        self.layer_idx = layer_idx
        self.qk_l2norm = qk_l2norm
        self.use_short_conv = True

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)    # C
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)    # B
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)  # x
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, bias=False, activation="silu")

        self.dt_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.D = nn.Parameter(torch.zeros(num_heads))
        self.D._no_weight_decay = True

        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        B, T, _ = hidden_states.shape
        H = self.num_heads
        last_state = get_layer_cache(self, past_key_values)
        conv_q = conv_k = conv_v = None
        if last_state is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]
        q, conv_q = self.q_conv1d(x=self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache)
        k, conv_k = self.k_conv1d(x=self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache)
        v, conv_v = self.v_conv1d(x=self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache)
        q, k, v = (rearrange(x, "b t (h d) -> b t h d", h=H) for x in (q, k, v))
        if self.qk_l2norm:
            q, k = l2_norm(q), l2_norm(k)

        delta = F.softplus(self.dt_proj(hidden_states).float() + self.dt_bias.float())   # [B, T, H]
        g = -self.A_log.float().exp() * delta                                              # log decay per head
        k_w = (k.float() * delta.unsqueeze(-1)).to(k.dtype)                                 # Δ_t B_t

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        fn = fused_recurrent_simple_gla if (not torch.is_grad_enabled() and T <= 64) else chunk_simple_gla
        o, recurrent_state = fn(q=q, k=k_w, v=v, g=g, scale=self.head_k_dim ** -0.5,
                                initial_state=recurrent_state, output_final_state=use_cache)
        update_layer_cache(self, past_key_values, recurrent_state=recurrent_state,
                           conv_state=(conv_q, conv_k, conv_v), offset=T)

        o = o + v * self.D.view(1, 1, H, 1).to(v.dtype)
        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "b t (h d) -> b t h d", h=H))
        o = self.o_proj(rearrange(o, "b t h d -> b t (h d)"))
        return o, None, past_key_values


def build(cfg, layer_idx):
    if cfg["linear_num_key_heads"] != cfg["linear_num_value_heads"] or cfg["linear_key_head_dim"] != cfg["linear_value_head_dim"]:
        raise ValueError("mamba2: one SSD group per head requires matching key/value head counts and dims")
    return Mamba2SSDLayer(
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
    copy_(layer.dt_proj.weight, src["a"], "dt_proj")
    copy_(layer.A_log, src["A_log"], "A_log")
    copy_(layer.dt_bias, src["dt_bias"], "dt_bias")
    with torch.no_grad():
        layer.D.zero_()
    copy_output_gate(layer, src)
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
