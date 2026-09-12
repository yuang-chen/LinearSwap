# Linear-kernel swap framework for Qwen3.5 (`qwen_linswap`)

Generalises the GDN→GDN2 in-place swap (docs/gdn2_experiment_log.md,
docs/gdn2_swap_notes.md) into a small framework that can replace Qwen3.5's Gated-DeltaNet (GDN) linear-attention
layers with *any* linear-attention kernel, initialise the new layer so that the
pretrained function is preserved, verify it, fine-tune it and benchmark it on
RULER — all through one kernel name.  The first new kernel is **Kimi Delta
Attention (KDA)**.

```
linswap.py             command line: verify | posttrain | evaluate | run | kernels
src/qwen_linswap/
  registry.py          KernelSpec + register_kernel / get_kernel / list_kernels
  kernels/common.py    helpers shared by init recipes (pretrained tensor layout, split fused qkv/conv, tiling, Qwen output gate)
  kernels/gdn.py       "gdn"          original GDN on FLA kernels (exact copy; control baseline)
  kernels/gdn2.py      "gdn2"         Gated DeltaNet-2 (scalar beta/decay tiled into b/w/f gates)
  kernels/kda.py       "kda"          Kimi Delta Attention, low-rank per-channel decay gate (default KDA)
                       "kda_fullgate" KDA with a dense decay projection
  kernels/rwkv7.py     "rwkv7"        RWKV-7 generalised delta rule (DPLR kernel), exact tiled init
  kernels/mamba2.py    "mamba2"       Mamba-2 SSD on the simple-GLA kernel — inexact swap (exact_init=False)
  kernels/deltanet.py  "deltanet"     DeltaNet, no decay — inexact swap (exact_init=False)
  model.py             Qwen3_5LinearSwapModel(cfg, kernel) + SwapCache
  components.py        RMSNorm / GQA / MLP / RoPE;  config.py  QWEN3_5_CONFIG
  load_weights.py      build_model(kernel | ckpt_dir), HF-format and native checkpoint loading
  data.py              SFT data preparation (LongAlign / LongAlpaca / anti-haystack);  sft_utils.py  chunked CE etc.
  pipeline/verify.py     stage 1: function-preservation checks vs HF Qwen3.5
  pipeline/distill.py    stage 1½ (inexact kernels): layer-wise alignment + KL distillation from the original
  pipeline/posttrain.py  stage 2: gate-only / full SFT (prepares data on first use; --init_ckpt to start from distill)
  pipeline/evaluate.py   stage 3: validation loss + RULER (calls RULER's scripts directly) -> summary table
  pipeline/run.py        the three stages chained for one kernel
tests/test_kernels.py  regression test over all registered kernels
RULER/scripts/pred/model_wrappers.py::QwenLinearSwapModelWrapper, server types qwen_linswap[_nocache]
```

## Using it

```bash
source .venv/bin/activate
python linswap.py verify    --kernel kda --baseline gdn
python linswap.py distill   --kernel mamba2                      # inexact kernels: outputs/mamba2/distill
python linswap.py posttrain --kernel kda                         # gate_only + full, outputs/kda/sft_*
python linswap.py evaluate  --models kda outputs/kda/sft_full/checkpoint-50 --name kda   # outputs/eval/kda/summary.*
python linswap.py run       --kernel kda                         # all three
```

```python
from qwen_linswap import build_model, list_kernels
model = build_model("kda")                                       # Qwen3.5-0.8B weights, exact KDA init
model = build_model(ckpt_dir="outputs/kda/sft_full/checkpoint-50")   # SFT checkpoint (kernel from config.json)
```

## Adding a kernel

Write `src/qwen_linswap/kernels/<name>.py` with

* `build(cfg, layer_idx) -> nn.Module` returning a token mixer with the FLA
  layer interface `forward(x, past_key_values=None, use_cache=False) -> (out, None, cache)`.
  Make every architectural change (replaced sub-modules, etc.) **here**, so the
  module structure is fixed before any weights are loaded and native
  checkpoints can be loaded strictly.
* `init_from_gdn(layer, hf_state_dict, layer_idx, model_prefix)` copying /
  tiling the pretrained GDN tensors (`kernels/common.py` documents their
  shapes and the GDN recurrence and provides `get_gdn_source`,
  `copy_qkv_and_conv`, `tile_rows`, `tile_vec`, `use_qwen_output_gate`,
  `copy_output_gate`).
* `register_kernel(KernelSpec(name=..., build=..., init_from_gdn=...,
  new_param_names=(...), exact_init=True/False))` and an import in
  `kernels/__init__.py`.  `new_param_names` are the parameter components
  trained by `--mode gate_only`.

Then `tests/test_kernels.py` and `python linswap.py verify --kernel <name> --baseline gdn` tell you whether the
init is function preserving: every number should sit at the same level as the
`gdn` column, which is pure Triton/bf16 noise.

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
which keeps them invisible at init but gives them non-zero gradient so SFT
can use the extra rank.  `kda_fullgate` uses a dense 2048×1024 `f_proj`
instead.  New parameters: 7.4M (`kda`) / 38M (`kda_fullgate`) vs 113M for GDN2.

### Verification (`python linswap.py verify --kernel kda --baseline gdn`)

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
| validation CE at step 0 (10 batches ≤131K) | 1.6914 | 1.6920 (gdn2: 1.6914) |

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

### Mamba-1 and Mamba-3: not available here

Both kernels exist only in ``mamba_ssm`` (selective-scan CUDA kernels for
Mamba-1, Triton/TileLang kernels for Mamba-3).  Installing ``mamba-ssm``
2.3.2 replaces torch with a CUDA-13 build, and even a ``--no-deps`` source
build cannot be imported: the package needs ``triton.set_allocator`` (Triton
≥ 3.3) while torch 2.6 pins Triton 3.2, and because FLA's ``fla.layers.mamba2``
imports it at package-import time the broken import takes ``import fla`` down
with it.  Mamba-1's per-(channel, state) decay also has no FLA equivalent.
Adding them needs a separate environment (torch ≥ 2.7, Triton ≥ 3.3, FLA
re-validated) — the kernel spec would then be a thin wrapper around FLA's
``Mamba``/``Mamba3`` layers plus a partial weight copy, and the distill stage
applies unchanged.

## Distillation for inexact swaps (`linswap.py distill`)

Teacher: ``gdn`` (exact copy of the original model).  Student: the swapped
model.  Sequences: the SFT corpus, left-truncated to ``--max_length``
(default 8192), all positions.

1. **layer** (default 200 steps, lr 1e-4, linear-layer parameters only) —
   each linear-attention layer of the student receives the teacher's input
   to the corresponding layer and is trained with MSE to reproduce the
   teacher layer's output.  All layers train in parallel from teacher
   activations, so this stage is cheap and does not depend on the rest of
   the student being right yet.
2. **kl** (default 300 steps, lr 2e-5, all parameters) — end-to-end
   ``KL(teacher ‖ student)`` on next-token distributions, computed in
   2048-token vocabulary chunks with a chunk-wise backward so the full logits
   are never materialised.

The checkpoint (``outputs/<kernel>/distill/checkpoint-N``) then seeds
``posttrain --init_ckpt``; ``run`` performs both automatically for kernels
registered with ``exact_init=False`` and evaluates base, distilled and SFT
checkpoints.

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

The un-decayed state grows without bound over long inputs, which is why the
deviation increases with length and gradient norms in SFT start in the
thousands.  This kernel is kept as the worked example of an inexact swap and of
what post-training then has to recover (results below).

## SFT and benchmark results

Recipe (identical for every kernel, `linswap.py posttrain`): data
`data/sft/len262144` left-truncated to 131072, bf16, gradient checkpointing,
micro-batch 1 × 2 accumulation, AdamW, clip 1.0, seed 42.
`gate_only`: 100 steps, lr 2e-4 on `new_param_names`, backbone frozen.
`full`: 50 steps, lr 1e-5 on everything.  Hardware here: 2× NVIDIA L20X (143 GiB);
a 131K micro-step takes ~10 s, a 262K micro-step ~36 s / 54 GiB, so every run
finishes in well under 10 minutes because most examples are short (median 9K tokens).

Environment: Python 3.11 in the `uv` venv with torch 2.6, transformers 5.16,
flash-linear-attention 0.6.0 and Triton 3.2.  `causal-conv1d` is not installed,
so FLA's short convolution falls back to PyTorch (a printed notice, not an
error); the KDA backward kernel prints benign Triton 3.2 scheduling warnings.

> Note: the original `scripts/sft.py::chunked_cross_entropy_with_backward` (now
> `qwen_linswap/sft_utils.py`) had a scaling bug (the LM-head / tied-embedding gradient was a token *sum* while the
> hidden-state gradient was a token *mean*, inflating the pre-clip grad norm to
> ~1000).  It is fixed in this revision; all runs below use the fixed loss.
> The GDN2 numbers in docs/gdn2_experiment_log.md were produced with the old loss.

### Validation cross-entropy (first 40 validation examples ≤131K)

| model | trainable params | val CE | ppl |
|---|---|---|---|
| gdn  base (exact copy) | – | 1.7432 | 5.72 |
| gdn2 base (tiled init) | – | 1.7414 | 5.71 |
| kda  base (tiled init) | – | 1.7419 | 5.71 |
| gdn  gate-only 100 steps | 0.59M | 1.4720 | 4.36 |
| kda  gate-only 100 steps | 7.41M | 1.4274 | 4.17 |
| kda_fullgate gate-only 100 steps | 38.1M | 1.4131 | 4.11 |
| gdn2 gate-only 100 steps | 113.3M | 1.3776 | 3.97 |
| gdn  full 50 steps | 752M | 1.3884 | 4.01 |
| kda  full 50 steps | 759M | 1.3884 | 4.01 |
| gdn2 full 50 steps | 865M | 1.3885 | 4.01 |
| deltanet base (inexact init) | – | 12.845 | 3.8e5 |
| deltanet full 50 steps, lr 1e-5 | 752M | 8.628 | 5586 |
| deltanet full 200 steps, lr 1e-5 | 752M | 7.174 | 1305 |
| deltanet full 200 steps, lr 1e-4 | 752M | 6.220 | 502 |

The three bases are equal within kernel noise (function-preserving init).
Under the identical recipe, full SFT lands on the same loss for all three
kernels, while gate-only SFT separates the kernels by gate capacity: the
per-channel KDA decay gate (7.4M params) recovers about half of the gap
between GDN's scalar gates and GDN2's three full-rank gates, and the dense
`kda_fullgate` variant a little more.

### RULER at 131072 tokens (100 samples per task, cached decode)

`niah_multivalue` is value-level accuracy.  (These runs predate the `evaluate` stage; today
`linswap.py evaluate` writes the same numbers to `outputs/eval/<name>/summary.csv`.)  Every model was evaluated with the
same `QwenLinearSwapModelWrapper`, chat template, greedy decoding and 128 new tokens.

| model | niah_single_1 | niah_multikey_1 | niah_multivalue |
|---|---|---|---|
| gdn  base (exact copy) | 100.0 | 100.0 | 96.5 |
| gdn2 base (tiled init) | 100.0 | 100.0 | 96.25 |
| kda  base (tiled init) | 100.0 | 100.0 | 95.75 |
| gdn  gate-only 100 (0.59M) | 100.0 | 100.0 | 96.25 |
| kda  gate-only 100 (7.4M) | 100.0 | 100.0 | 97.5 |
| kda_fullgate gate-only 100 (38M) | 100.0 | 98.0 | 99.0 |
| gdn2 gate-only 100 (113M) | 100.0 | 99.0 | 96.5 |
| gdn  full 50 | 100.0 | 100.0 | 99.0 |
| kda  full 50 | 100.0 | 100.0 | 99.5 |
| gdn2 full 50 | 100.0 | 100.0 | 99.25 |
| deltanet base (inexact) | 0.0 | 0.0 | 0.0 |
| deltanet full 200 (lr 1e-4) | 0.0 | 0.0 | 0.0 |

Reading: the three function-preserving bases are indistinguishable (single- and
multi-key retrieval saturated, multi-value 95.75–96.5, i.e. 1–2 wrong values
out of 400).  Full SFT under the identical recipe lifts multi-value retrieval
to 99–99.5 for every kernel, so on this budget the kernel choice does not
change the outcome of full fine-tuning.  Gate-only SFT is where the kernels
differ, and not in the order of validation loss: GDN's 0.59M scalar gates and
GDN2's 113M gates both leave multi-value retrieval at the base level (96.25 /
96.5) even though GDN2 reaches the lowest validation loss of all gate-only
runs, KDA's 7.4M low-rank per-channel decay gate gives a small gain (97.5),
and the dense-gate KDA variant reaches 99.0, comparable to full SFT, with 38M
trainable parameters.  Differences of ≤1 point on `niah_multivalue` (4 values
out of 400) and the 98.0 / 99.0 on `niah_multikey_1` (1–2 of 100 samples) are
within sample noise for this size of evaluation, so the robust conclusions
are: (i) the KDA swap is exact and loses nothing, (ii) full SFT equalises the
kernels, (iii) a per-channel decay gate is a much cheaper gate-only handle
than GDN2's three full-rank gates for long-context retrieval.

The inexact DeltaNet swap is the counter-example: at init the model is
unusable (validation CE 12.8, all NIAH scores 0), and 200 full-SFT steps at
lr 1e-4 (4× the budget of the exact swaps, 10× their learning rate) only bring
the loss to 6.2 with retrieval still at 0.  Dropping a component the
pretrained network depends on is not something a short post-training run
recovers; a kernel without decay would need a distillation-style schedule
(or a smarter init) rather than the recipe that works for exact swaps.

Compared with the earlier GDN2 numbers in docs/gdn2_experiment_log.md (different machine, old loss
scaling): base 92.5 → full SFT 98.25 on `niah_multivalue`; here 96.25 → 99.25.
