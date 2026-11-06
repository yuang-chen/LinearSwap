"""Mamba-3 ("Mamba-3: Improved Sequence Modeling using State Space Principles"): FLA's ``Mamba3`` layer, whose
kernels come from ``mamba_ssm``, behind the backbone's short convolutions.  Optional: the kernel registers
only when those kernels import (``pip install -e ".[mamba3]"``, see the README).

Mamba-3 is a diagonal SSM with data-dependent decay ``A``, step ``Δ``, a trapezoidal discretisation and a
rotary (complex-valued) state, read out through per-group B / C vectors.  FLA's layer fuses
``[z, x, B, C, dd_dt, dd_A, trap, angles]`` into one ``in_proj``.

**The short convolution stays.**  Mamba-3 has none, but the backbone passes q / k / v through a 4-tap causal
convolution and SiLU, and for keys and values the tap on the *previous* token is as large as the one on the
current token: the pretrained layer stores "what follows what".  Without it the copied projections are
scrambled (the taps are small and of either sign per channel) and the layer cannot form those pairs at all:
per layer the untrained swap was at NMSE 70 from the teacher's output with zero correlation, and after
12.5M tokens of distillation at a gap of 0.88 where ``mamba2`` is at 0.13.  Folding the current-token tap
into the projections brought the init to NMSE 3 and changed nothing in training.  With the convolutions in
front of B / C / x — as every other kernel here keeps them — the swap starts where ``mamba2`` starts.

Init from GDN (inexact, ``exact_init=False``): one B / C group per key head, so B / C have the shape of
GDN's k / q and Mamba-3 expands them to the value heads as GDN does; ``z ← in_proj_z``, ``x ← v``,
``B ← k``, ``C ← q`` rows of the fused projection and the three convolutions copied; ``dd_dt ← in_proj_a``
and ``dt_bias ← dt_bias`` so Δ equals GDN's; the data-dependent ``A`` rows are zeroed with their bias at the
inverse softplus of ``exp(A_log)``, so the decay ``exp(A·Δ)`` starts exactly at GDN's; rotary angles and the
trapezoid logit are zero (no rotation, the library's even mix); ``B_norm`` / ``C_norm`` weights are
``1/√S`` (RMSNorm → GDN's L2 normalisation) with the ``1/√K`` query scale folded into ``C_norm``;
``B_bias`` / ``C_bias`` and ``D`` start at 0; the per-head output norm (SiLU gate, the backbone's eps)
receives the backbone's norm weight tiled per head, and ``out_proj`` is copied.  What remains different from
GDN: no delta-rule erase and Δ-scaled instead of β-scaled writes, as in ``mamba2``.

Two more things FLA's layer leaves to the caller:

* **Cached decoding.**  The prefill kernel returns the last key in float32, while the single-token kernel
  (CuTe DSL: ``nvidia-cutlass-dsl``, ``quack-kernels``) is compiled for *one* dtype shared by the cached
  key, the new key and the new query — which the layer computes in the model's dtype.  The cached key is
  handed over in the model's dtype, as the layer's own ``allocate_inference_cache`` does.  It is the
  previous token's key, overwritten at every step, so nothing accumulates in it.
* **Padding.**  With a bias in the fused projection a zeroed hidden state is not a no-op, so the layer
  takes the padding mask itself (``takes_padding_mask``) and packs the valid tokens, which is exact.
"""

import torch

from ..registry import KernelSpec, register_kernel
from .common import copy_, get_gdn_source, mark_hf_initialized

try:
    from fla.layers.mamba3 import Mamba3, is_fast_path_available as available
    from fla.modules import ShortConvolution
except Exception:  # pragma: no cover
    Mamba3, available = None, False


if available:
    class State(tuple):
        """``(angle, ssm, key, value)`` as FLA's layer unpacks it, carrying the convolution caches along."""
        conv = (None, None, None)


    class Mamba3Layer(Mamba3):
        takes_padding_mask = True              # linswap.model passes the mask instead of zeroing pad positions
        checkpointable = False                 # mamba_ssm's backward reads its saved tensors twice: no gradient checkpointing

        def __init__(self, conv_size, **kwargs):
            super().__init__(**kwargs)
            key_dim = self.ssm_state_size * self.n_groups
            self.q_conv1d = ShortConvolution(key_dim, conv_size, bias=False, activation="silu")
            self.k_conv1d = ShortConvolution(key_dim, conv_size, bias=False, activation="silu")
            self.v_conv1d = ShortConvolution(self.intermediate_size, conv_size, bias=False, activation="silu")

        def cuda_kernels_forward(self, hidden_states, last_state=None, use_cache=False, cu_seqlens=None):
            prev = last_state["recurrent_state"] if last_state is not None else None
            # read by _project_and_split, which FLA calls without arguments to spare
            self._conv_call = (getattr(prev, "conv", State.conv), use_cache or prev is not None, cu_seqlens)
            out, state = super().cuda_kernels_forward(hidden_states, last_state, use_cache, cu_seqlens)
            if state is not None:
                angle, ssm, key, value = state
                state = State((angle, ssm, key.to(value.dtype), value))       # the dtypes the step kernel expects
                state.conv = self._conv_caches
            return out, state

        def _project_and_split(self, hidden_states):
            """FLA's split, with the backbone's short convolution + SiLU on C (query), B (key) and x (value)."""
            z, x, B, C, *rest = super()._project_and_split(hidden_states)
            caches, keep, cu_seqlens = self._conv_call
            step = hidden_states.dim() == 2                                    # decoding: one token, [batch, hidden]
            out, new = [], []
            for u, conv, cache in zip((C, B, x), (self.q_conv1d, self.k_conv1d, self.v_conv1d), caches):
                y, cache = conv(u.unsqueeze(1) if step else u, cache=cache, output_final_state=keep,
                                cu_seqlens=None if step else cu_seqlens)
                out.append(y.squeeze(1) if step else y)
                new.append(cache)
            self._conv_caches = tuple(new)
            C, B, x = out
            return (z, x, B, C, *rest)


    def build(cfg, layer_idx):
        hidden, H = cfg["emb_dim"], cfg["linear_num_key_heads"]
        HV, V, K = cfg["linear_num_value_heads"], cfg["linear_value_head_dim"], cfg["linear_key_head_dim"]
        return Mamba3Layer(conv_size=cfg["linear_conv_kernel_dim"], hidden_size=hidden, state_size=K,
                           expand=HV * V / hidden, head_dim=V, n_groups=H, rope_fraction=0.5, is_outproj_norm=True,
                           use_bias=True, layer_idx=layer_idx, norm_eps=cfg.get("rms_norm_eps", 1e-6))


    def init_from_gdn(layer, gdn_state, layer_idx, model_prefix="model"):
        src = get_gdn_source(gdn_state, layer_idx, model_prefix)
        I, S, H = layer.intermediate_size, layer.ssm_state_size, layer.num_heads
        key_dim = S * layer.n_groups
        W, b = layer.in_proj.weight, layer.in_proj.bias
        qw, kw, vw = torch.split(src["qkv"], [key_dim, key_dim, I], dim=0)
        cq, ck, cv = torch.split(src["conv"], [key_dim, key_dim, I], dim=0)
        rows = [I, I, key_dim, key_dim, H, H, H, layer.num_rope_angles]
        off = [sum(rows[:i]) for i in range(len(rows) + 1)]
        with torch.no_grad():
            b.zero_()
            W[off[0]:off[1]].copy_(src["z"].to(W.dtype))          # z: output gate
            W[off[1]:off[2]].copy_(vw.to(W.dtype))                # x: values
            W[off[2]:off[3]].copy_(kw.to(W.dtype))                # B: keys, one group per key head
            W[off[3]:off[4]].copy_(qw.to(W.dtype))                # C: queries
            W[off[4]:off[5]].copy_(src["a"].to(W.dtype))          # dd_dt: GDN's decay logits -> Δ
            W[off[5]:off[6]].zero_()                              # dd_A: data-independent at init ...
            a = src["A_log"].float().exp()                        # ... with A = -softplus(bias) = -exp(A_log)
            b[off[5]:off[6]].copy_((a + torch.log(-torch.expm1(-a))).to(b.dtype))
            W[off[6]:off[8]].zero_()                              # trapezoid logit, rotary angles
            copy_(layer.q_conv1d.weight, cq, "q_conv1d")
            copy_(layer.k_conv1d.weight, ck, "k_conv1d")
            copy_(layer.v_conv1d.weight, cv, "v_conv1d")
            copy_(layer.dt_bias, src["dt_bias"], "dt_bias")
            layer.B_norm.weight.fill_(S ** -0.5)                  # RMSNorm * 1/sqrt(S) == L2 normalisation
            layer.C_norm.weight.fill_(S ** -0.5 * S ** -0.5)      # ... and GDN's 1/sqrt(K) query scale (K == S)
            layer.B_bias.zero_()
            layer.C_bias.zero_()
            layer.D.zero_()
            layer.norm.weight.copy_(src["norm"].repeat(H).to(layer.norm.weight.dtype))
            copy_(layer.out_proj.weight, src["out"], "out_proj")
            layer.out_proj.bias.zero_()
        mark_hf_initialized(layer)
        return layer


    register_kernel(KernelSpec(
        name="mamba3", build=build, init_from_gdn=init_from_gdn, exact_init=False,
        description="Mamba-3 (FLA Mamba3 on mamba_ssm kernels) behind the backbone's short convolutions; GDN decay and "
                    "projections mapped into the fused in_proj, rotary / trapezoid neutral — NOT function preserving.",
        new_param_names=("dt_bias", "D", "B_bias", "C_bias", "B_norm", "C_norm"),
        notes="Inexact swap (no erase, Δ-scaled writes): distil.",
    ))
