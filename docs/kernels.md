# Per-kernel swaps

How each target recurrence is reached from the pretrained Gated-DeltaNet layer, what maps exactly and
what does not.  The recipe that trains them is in [recipe.md](recipe.md); the numbers are in
[results.md](results.md); the framework itself is in [framework.md](framework.md).

## KDA swap

KDA (Kimi Linear, arXiv:2510.26692) is the gated delta rule with a
per-key-channel forget gate: `g_t[h,:] = -exp(A_log[h]) * softplus(f_proj(x)[h,:] + dt_bias[h,:])`
instead of GDN's scalar `g_t[h]`.  A scalar decay commutes with the delta-rule
projector, so tiling the scalar decay row across the 128 key channels of each
head is exact; `beta`, `A_log`, q/k/v, the short convolutions and the output
projection carry over unchanged.

| GDN (Qwen3.5) | KDA (FLA `KimiDeltaAttention`) | init |
|---|---|---|
| `in_proj_qkv`, `conv1d` | `q/k/v_proj`, `q/k/v_conv1d` | split (exact) |
| `in_proj_a` [16×1024] | `f_proj` = Linear(1024→128) ∘ Linear(128→2048) | `W1[:16]=in_proj_a`, `W2` = 0/1 head selector, rest of the rank inactive but trainable |
| `dt_bias` [16] | `dt_bias` [2048] | tiled |
| `A_log`, `in_proj_b` | `A_log`, `b_proj` | copied |
| `in_proj_z`, swish-gated `norm`, `out_proj` | `g_proj` (full rank), `FusedRMSNormSwishGate`, `o_proj` | copied; KDA's default low-rank sigmoid-gated output norm cannot represent the pretrained gate and is replaced |

The tiled decay matrix has rank ≤ 16 ≤ 128, so it fits KDA's low-rank
`f_proj` *exactly*; the 112 unused bottleneck dimensions get `W2[:,16:]=0`,
which keeps them invisible at init but gives them non-zero gradient so
distillation can use the extra rank.  This is Kimi Linear's own gate parameterisation, kept as
FLA ships it.  New parameters: 7.4M vs 113M for GDN2.

### Verification (`linswap verify --kernel kda --baseline gdn`)

All numbers are at the level of the `gdn` control (bf16 Triton noise vs HF's
implementation):

| check | kda | gdn (control) |
|---|---|---|
| kernel-level: `chunk_kda` with tiled gate vs `chunk_gated_delta_rule` (T=512) | max abs diff 2.4e-4 (chunk), 3.8e-6 (recurrent) | – |
| pretrained layers 0/1/2 vs HF torch layer, T=64 / 1024 | rel-L2 0.008–0.010 | rel-L2 0.008–0.010 |
| full-model logits vs HF, T=8 / 64 / 512 / 4096 | top-1 agreement 1.000, KL ≤ 1.6e-3 | same |
| cached decode vs no-cache decode (5 / 120 / 4096-token prompts) | 100 % token agreement | 100 % |
| greedy generation vs HF (20 tokens) | identical | identical |
| gradients of shared params, KDA vs GDN layer (T=256 / 4096) | rel diff ≤ 1.3 % | – |

## RWKV-7 swap (exact)

RWKV-7 (arXiv:2503.14456) uses the generalised delta rule with a
diagonal-plus-rank-one transition, which FLA exposes as ``chunk_rwkv7`` /
``chunk_dplr_delta_rule``:  ``S_t = Diag(e^{gk_t}) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T``.
GDN is the special case ``a_t = k̂ ⊙ e^{g_t}``, ``b_t = -beta_t k̂``, ``k_t = beta_t k̂``,
``gk_t = g_t``, so the swap is exact.  ``kernels/rwkv7.py`` keeps the RWKV-7
parameterisation of what the kernel needs — a low-rank per-channel decay
(``w_lora`` → ``f_proj``), a low-rank per-channel in-context learning rate
(``a_lora`` → ``b_proj``) and a separately modulated removal key (``k_k``) —
and Qwen's projections, convolutions and SiLU-gated output norm for the rest
(token shift, value residual and GroupNorm of RWKV-7 are not used).  FLA's own
``RWKV7Attention`` layer cannot be used: it fixes ``key_dim = hidden_size`` and
bounds the per-step decay to ≥ 0.545, which the pretrained decays exceed.
New parameters: 14M.  ``verify`` puts every check at the ``gdn`` noise level
(top-1 agreement 1.0 at 8/512/4096 tokens, KL ≤ 1.6e-3, cached decode and
generation identical).

## Mamba-2 swap (inexact)

Mamba-2's SSD recurrence ``S_t = exp(-Δ_t e^{A_log}) S_{t-1} + Δ_t B_t x_t^T``,
``y_t = C_t^T S_t + D x_t`` is a scalar-decay linear RNN, i.e. FLA's
``chunk_simple_gla``; ``kernels/mamba2.py`` builds it with one SSD group per
head so B/C/x line up with Qwen's k/q/v, copies the decay (identical
parameterisation to GDN's), gate and output weights, and starts ``D`` at 0.
What GDN has and Mamba-2 lacks is the delta-rule erase, and Mamba-2 scales the
write by Δ_t rather than beta_t, so the init is inexact (top-1 agreement with
the original 0.07 on the test prompt) and the model is distilled first.
``mamba_ssm`` is not required.

### Mamba-1 and Mamba-3 (via ``mamba_ssm``)

Both kernels wrap FLA's ``Mamba`` / ``Mamba3`` layers, whose scan kernels come
from ``mamba_ssm`` (selective-scan CUDA for Mamba-1, Triton for Mamba-3's SISO
path).  ``mamba_ssm`` 2.3.2 requires Triton ≥ 3.5 (its Mamba-3 kernel uses a
``tl.dot`` shape Triton 3.4 rejects), which means torch ≥ 2.9; the project
environment was therefore moved from torch 2.6 / Triton 3.2 — below FLA's own
``torch>=2.7, triton>=3.3`` requirement — to torch 2.9.1 / Triton 3.5.1, and
all test suites and the ``verify`` numbers were re-checked there (same noise
floor).  Install ``mamba_ssm`` only with ``--no-deps --no-build-isolation``:
resolved normally it pulls a different torch build.  The two kernels register
only when the kernels import (``fla.layers.mamba{,3}.is_fast_path_available``).

* ``mamba3``: one SSM group per head so B/C have k/q's shape; ``z``, ``x``, ``B``,
  ``C`` and ``dd_dt`` rows of the fused ``in_proj`` are copied from the gate,
  value, key, query and decay projections; the data-dependent ``A`` rows are
  zeroed with a bias equal to the inverse-softplus of ``exp(A_log)`` so the
  decay starts exactly at GDN's; rotary angles and the trapezoid coefficient
  are zeroed (identity rotation, neutral mixing); ``B_norm`` / ``C_norm``
  weights are ``1/√S`` so the RMSNorms act as GDN's L2 normalisation, with the
  ``1/√K`` query scale folded into ``C_norm``; ``D``, ``B_bias``, ``C_bias``
  start at 0; the per-head SiLU-gated output norm gets the backbone's norm
  weight tiled.  Still different from GDN: no erase, Δ-scaled writes, no
  short convolution — inexact, distil first.  Cached single-token decoding uses
  `mamba_ssm`'s CuTe-DSL step kernel (`nvidia-cutlass-dsl`, `quack-kernels`); with
  cutlass-dsl 4.7.1 the published `quack-kernels` releases either lack an API the
  kernel needs (0.3.x) or pass one cutlass rejects (0.6.5), so decode is
  unsupported here — evaluate Mamba-3 with `--no_cache` (full-prefix recompute
  per token) at moderate lengths until the upstream versions line up.
* ``mamba1``: per-channel selective SSM with ``state_size = 128`` (the same
  number of state entries as GDN); values, gate, the value convolution and
  ``out_proj`` are copied, the SSM parameters keep Mamba's init and the output
  is gated without a norm — the least GDN-like target in the registry.


## DeltaNet swap (inexact)

DeltaNet (arXiv:2406.06484) is the delta rule *without* a forget gate,
`S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T`, i.e. GDN with
`exp(g_t) ≡ 1`.  There is no setting of DeltaNet's parameters that reproduces
a decaying GDN layer, so `kernels/deltanet.py` copies everything that maps
one-to-one (q/k/v, convolutions, `beta`, SiLU-gated output norm, `o_proj`) and
drops the decay branch (`in_proj_a`, `A_log`, `dt_bias`).  The spec is
registered with `exact_init=False`; `verify.py` prints a notice and the
expected deviation:

| check | deltanet | gdn (control) |
|---|---|---|
| pretrained layers 0/1/2 vs HF layer, rel-L2 | 1.2–3.8 | 0.008–0.011 |
| full-model top-1 agreement with HF, T=8 / 512 / 4096 | 0.375 / 0.221 / 0.027 | 1.0 / 1.0 / 1.0 |
| validation CE at step 0 | 13.84 | 1.69 |

Without decay the state never contracts (for unit keys and β∈[0,1] the
transition is non-expansive, so it does not blow up, but stale associations
persist until they are explicitly overwritten), which is why the deviation
increases with length and gradient norms in SFT start in the thousands.  This kernel is kept as the worked example of an inexact swap and of
what post-training then has to recover (results below).
