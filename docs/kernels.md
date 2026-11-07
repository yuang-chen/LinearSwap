# Per-kernel swaps

How each target recurrence is reached from the pretrained Gated-DeltaNet layer, what maps exactly and
what does not.  How the inexact ones are distilled is in [recipe.md](recipe.md); the numbers are in
[results.md](results.md).

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

### Verification (`linswap verify --kernel kda`)

Against the unmodified upstream model, with the `gdn` swap as the noise floor of the comparison (FLA's
Triton / TileLang kernels in bf16 against upstream's torch code):

| check | kda | gdn (floor) |
|---|---|---|
| text, 512 tokens: KL to the backbone / top-1 agreement | 8.8e-5 / 1.000 | 7.5e-5 / 1.000 |
| text, 4096 tokens | 1.1e-5 / 1.000 | 9.8e-6 / 1.000 |
| chat prompt with an image (196 image tokens) | 2.4e-3 / 0.973 | 2.7e-3 / 0.973 |
| greedy answer to the image prompt vs the backbone | identical | identical |
| cached vs uncached decoding | identical | identical |

`gdn2` and `rwkv7` give the same picture (image KL 2.8e-3 and 3.8e-3).

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
(top-1 agreement 1.0 on text at 512 and 4096 tokens, KL 8.0e-5 / 1.0e-5; 3.8e-3 on the
image prompt, where the floor is 2.7e-3; cached decode and generation identical).

## Mamba-2 swap (inexact)

Mamba-2's SSD recurrence ``S_t = exp(-Δ_t e^{A_log}) S_{t-1} + Δ_t B_t x_t^T``,
``y_t = C_t^T S_t + D x_t`` is a scalar-decay linear RNN, i.e. FLA's
``chunk_simple_gla``; ``kernels/mamba2.py`` builds it with one SSD group per
head so B/C/x line up with Qwen's k/q/v, copies the decay (identical
parameterisation to GDN's), gate and output weights, and starts ``D`` at 0.
What GDN has and Mamba-2 lacks is the delta-rule erase, and Mamba-2 scales the
write by Δ_t rather than beta_t, so the init is inexact (KL 5.7 to the backbone at
512 tokens, top-1 agreement 0.25) and the model is distilled first.  The kernel runs in
float32 at inference as in training: simple-GLA keeps its state in float32, and on Hopper
GPUs FLA's only correct backward for it (TileLang, fla#640) needs q/k/v in the same dtype.
``mamba_ssm`` is not required.

## Mamba-3 swap (inexact, optional)

FLA's `Mamba3` layer on `mamba_ssm`'s kernels (`pip install -e ".[mamba3]"`; the kernel registers only when
they import), behind the backbone's short convolutions.  Mamba-3 is a diagonal state-space model with
data-dependent decay `A`, step `Δ`, a trapezoidal update and a rotary (complex-valued) state, read out
through per-group `B` / `C` vectors.  The layer fuses `[z, x, B, C, dd_dt, dd_A, trap, angles]` into one
`in_proj`.

| pretrained GDN | Mamba-3 | how |
|---|---|---|
| `in_proj_z` | `z` rows of `in_proj` | copied (output gate) |
| `v` rows of `in_proj_qkv` | `x` rows | copied |
| `k`, `q` rows of `in_proj_qkv` | `B`, `C` rows, one group per key head | copied; Mamba-3 expands groups to value heads as GDN does |
| `conv1d` (q / k / v) | `q_conv1d`, `k_conv1d`, `v_conv1d` on `C`, `B`, `x` | copied; **added to FLA's layer**, see below |
| `in_proj_a`, `dt_bias` | `dd_dt` rows, `dt_bias` | copied, so `Δ` equals GDN's |
| `A_log` | bias of the `dd_A` rows (weights zero) | `A = -softplus(bias) = -exp(A_log)`: the decay `exp(A·Δ)` starts at GDN's |
| — | `trap`, rotary `angles` rows | zero: the library's even trapezoid mix, no rotation |
| q / k L2 normalisation, `1/√K` | `B_norm`, `C_norm` weights | `1/√S`, with the query scale folded into `C_norm` |
| `norm`, `out_proj` | per-head gated `norm` (the backbone's eps), `out_proj` | tiled per head, copied |

Still different from GDN: no erase and `Δ`-scaled instead of `β`-scaled writes, as in `mamba2`.

**Why the short convolution is kept.**  Mamba-3 has none.  The backbone passes q / k / v through a 4-tap
causal convolution and SiLU, with small taps of either sign per channel, and for keys and values the tap on
the *previous* token is as large as the one on the current token: the pretrained layer writes pairs that
span two tokens.  Three variants, each layer fed the teacher layer's input (mean over the 18 layers) and
then distilled — with the recipe as it was before its faults were fixed ([results.md](results.md),
"The runs before the fixes"), so the gaps are comparable with each other and not with the current
`gla` / `mamba2` runs; `mamba3` has not been re-distilled:

| variant | layer output vs the teacher's at step 0 | gap at 12.5M tokens | at 100M |
|---|---|---|---|
| FLA's layer as is, projections copied | NMSE 70.6, cosine −0.02, 3–13× too large | 0.884 | 0.245 (2.12 GPU-hours) |
| the same, current-token tap folded into the projections | NMSE 3.1, cosine 0.26 | 0.931 | — |
| **short convolutions kept (the kernel)** | NMSE 0.56, cosine 0.74 | 0.122 | 0.068 (2.05 GPU-hours) |
| `mamba2`, for reference | NMSE 0.73, cosine 0.74 | 0.127 | 0.071 |

The trapezoid logit starts at 0, the library's even mix.  At +12 (a pure Euler step, exactly `mamba2`'s
function at step 0) a trial run was behind at every milestone: 0.174 against 0.169 at 6M tokens, 0.135
against 0.122 at 12.5M.

* The scale: the teacher's readout before its output norm has an RMS of 5e-4 – 1e-2, comparable to the
  norm's epsilon, so the norm passes magnitude through.  Without the convolution's small taps the values were
  10–40 times larger, the norm saturated, and the layer output was 3–13 times the teacher's.
* The correlation: without the per-channel sign of the taps the query and key channels no longer line up.
* A better init of FLA's layer alone does not help: fixing sign and scale brings the init 20 times closer
  and changes nothing in training, because the layer still cannot write a pair that spans two tokens.

Two more things the layer needs from us (`kernels/mamba3.py`, `linswap.model`, `linswap.distill`):

* **Cached decoding.**  The prefill kernel returns the last key in float32; the single-token kernel is
  compiled for one dtype shared by the cached key, the new key and the new query, which the layer computes
  in the model's dtype.  Our subclass hands the cached key over in the model's dtype, as the layer's own
  `allocate_inference_cache` does, and carries the convolution caches with the state.  It needs
  `nvidia-cutlass-dsl` ≥ 4.8 with `quack-kernels` 0.6.5; with 4.7 the kernel did not compile.  On a
  distilled checkpoint cached decoding matches a full forward as closely as the teacher kernel's does (KL
  9e-4 against 1e-3).
* **Padding.**  With a bias in the fused projection a zeroed hidden state still writes, so the layer gets the
  padding mask and packs the valid tokens (`takes_padding_mask`); left-padded batches then match unpadded
  rows (KL 7e-4, the teacher kernel's level).
* **No gradient checkpointing.**  `mamba_ssm`'s backward reads its saved tensors twice, which
  `torch.utils.checkpoint` refuses (`checkpointable = False`), so the closed-loop phase keeps all
  activations.

## DeltaNet swap (inexact)

DeltaNet (arXiv:2406.06484) is the delta rule *without* a forget gate,
`S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T`, i.e. GDN with
`exp(g_t) ≡ 1`.  There is no setting of DeltaNet's parameters that reproduces
a decaying GDN layer, so `kernels/deltanet.py` copies everything that maps
one-to-one (q/k/v, convolutions, `beta`, SiLU-gated output norm, `o_proj`) and
drops the decay branch (`in_proj_a`, `A_log`, `dt_bias`).  The spec is
registered with `exact_init=False`, and `linswap verify --kernel deltanet` reports the distance from
the teacher at step 0 instead of judging it: KL 6.7 at 512 tokens and 12.7 at 4096 (top-1 agreement 0.24 and
0.04), 4.5 on the image prompt.

Without decay the state never contracts (for unit keys and β∈[0,1] the
transition is non-expansive, so it does not blow up, but stale associations
persist until they are explicitly overwritten), which is why the deviation
increases with length.  This kernel is the worked example of an inexact swap and of what distillation
then has to recover.
