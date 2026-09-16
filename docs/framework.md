# LinearSwap: the kernel-swap framework (`linswap`)

Generalises the GDN→GDN2 in-place swap (docs/gdn2_experiment_log.md,
docs/gdn2_swap_notes.md) into a small framework that can replace the Gated-DeltaNet (GDN) linear-attention
layers of a pretrained hybrid backbone (Qwen3.5-0.8B in all experiments here) with *any* linear-attention kernel, initialise the new layer so that the
pretrained function is preserved, verify it, fine-tune it and benchmark it on
RULER — all through one kernel name.  The first new kernel is **Kimi Delta
Attention (KDA)**.

```
pyproject.toml         `pip install -e .` -> the `linswap` command (src/linswap/cli.py; also `python -m linswap`)
src/linswap/
  hf.py                LinearSwapConfig / LinearSwapCache / LinearSwapForCausalLM (transformers PreTrainedModel,
                       registered with AutoConfig / AutoModelForCausalLM on import) + export()
  registry.py          KernelSpec + register_kernel / get_kernel / list_kernels
  kernels/common.py    helpers shared by init recipes (pretrained tensor layout, split fused qkv/conv, tiling, output gate, copy_shared_from_gdn)
  kernels/fla_layer.py build_fla_layer / register_fla_kernel: any fla.layers class + init recipe -> KernelSpec
  kernels/base.py      BackboneMixer: projections + convs + cache + gated norm around an FLA op (custom recurrences)
  kernels/gdn.py       "gdn"          original GDN on FLA kernels (exact copy; control baseline)
  kernels/gdn2.py      "gdn2"         Gated DeltaNet-2 (scalar beta/decay tiled into b/w/f gates)
  kernels/kda.py       "kda"          Kimi Delta Attention, low-rank per-channel decay gate (default KDA)
                       "kda_fullgate" KDA with a dense decay projection
  kernels/rwkv7.py     "rwkv7"        RWKV-7 generalised delta rule (DPLR kernel), exact tiled init
  kernels/mamba2.py    "mamba2"       Mamba-2 SSD on the simple-GLA kernel — inexact swap (exact_init=False)
  kernels/deltanet.py  "deltanet"     DeltaNet, no decay — inexact swap (exact_init=False)
  kernels/gla.py       "gla"          Gated Linear Attention, stock FLA layer, no custom code — inexact swap
  kernels/mamba3.py    "mamba3"       FLA Mamba3 (mamba_ssm kernels), GDN decay mapped into the fused in_proj — inexact
                       "mamba3_min"   control: projection rows + output path copied, Mamba-3's own SSM init
  kernels/mamba2.py    "mamba2_beta"  control: SSD with GDN's beta-scaled write (only the erase is missing)
  kernels/mamba1.py    "mamba1"       FLA Mamba (mamba_ssm kernels), values/gate/conv copied — inexact
  model.py             LinearSwapBackbone (model.embed_tokens / layers / norm) + LinearSwapModel (adds lm_head)
                       — Qwen's module tree and state-dict keys; SwapCache
  components.py        RMSNorm / GQA / MLP / RoPE;  backbones.py  load_backbone_config() from the HF config
  load_weights.py      build_model(kernel | ckpt_dir), HF-format and native checkpoint loading
  data.py              SFT data preparation (LongAlign / LongAlpaca / anti-haystack);  sft_utils.py  chunked CE etc.
  pipeline/verify.py     stage 1: function-preservation checks vs HF Qwen3.5
  pipeline/distill.py    stage 1½ (inexact kernels): layer-wise alignment + KL distillation from the original
  pipeline/posttrain.py  stage 2: full SFT (gate-only mode kept for the 27B) (prepares data on first use; --init_ckpt to start from distill)
  pipeline/evaluate.py   stage 3: validation loss + RULER (calls RULER's scripts directly) -> summary table
  pipeline/run.py        the stages chained for one kernel
  pipeline/export.py     write a swapped model / checkpoint as an HF checkpoint (safetensors + tokenizer + card)
tests/                 test_kernels.py (all kernels vs control), test_hf.py (HF round-trip), test_batch.py (padded batches), test_gva.py (grouped value heads),
                       test_losses.py (chunked CE / KL vs dense autograd; CPU, no model)
examples/quickstart.ipynb
RULER/scripts/pred/model_wrappers.py::LinearSwapModelWrapper, server types linswap[_nocache]
```

## Using it

```bash
source .venv/bin/activate
linswap verify    --kernel kda --baseline gdn
linswap distill   --kernel mamba2                      # inexact kernels: outputs/mamba2/distill
linswap posttrain --kernel kda                         # full SFT, outputs/kda/sft_full
linswap evaluate  --models kda outputs/kda/sft_full/checkpoint-50 --name kda   # outputs/eval/kda/summary.*
linswap run       --kernel kda                         # all three
```

```python
from linswap import build_model, list_kernels
model = build_model("kda")                                       # Qwen3.5-0.8B weights, exact KDA init
model = build_model(ckpt_dir="outputs/kda/sft_full/checkpoint-50")   # SFT checkpoint (kernel from config.json)
```

## Adding a kernel

Write `src/linswap/kernels/<name>.py` with

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
  counted as the kernel's new parameters (parameter counts; the `gate_only` mode of `posttrain` trains only these).

Then `tests/test_kernels.py` and `linswap verify --kernel <name> --baseline gdn` tell you whether the
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
| validation CE at step 0 (10 batches ≤128K) | 1.6914 | 1.6920 (gdn2: 1.6914) |

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

## Distillation for inexact swaps (`linswap distill`)

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

Without decay the state never contracts (for unit keys and β∈[0,1] the
transition is non-expansive, so it does not blow up, but stale associations
persist until they are explicitly overwritten), which is why the deviation
increases with length and gradient norms in SFT start in the thousands.  This kernel is kept as the worked example of an inexact swap and of
what post-training then has to recover (results below).

## SFT and benchmark results

Recipe (identical for every kernel, `linswap posttrain`): data
`data/sft/len262144` left-truncated to 131072, bf16, gradient checkpointing,
micro-batch 1 × 2 accumulation, AdamW, clip 1.0, seed 42.
`full`: 50 steps, lr 1e-5 on everything.  (A `gate_only` mode — 100 steps, lr 2e-4 on the kernel's
new parameters with the backbone frozen — exists in `posttrain` but was dropped from the study: it never
beat full SFT and only added tables; it is used solely at 27B, where full SFT does not fit.)  Hardware here: 2× NVIDIA H200 (143 GiB);
a 128K micro-step takes ~10 s, a 262K micro-step ~36 s / 54 GiB, so every run
finishes in well under 10 minutes because most examples are short (median 9K tokens).

Environment: Python 3.11 in the `uv` venv.  The 0.8B results in this document
were produced with torch 2.6 / Triton 3.2 (FLA's short convolution on its
PyTorch fallback until `causal-conv1d` was built); the environment was later
moved to torch 2.9.1 / CUDA 12.8 / Triton 3.5.1 with `causal-conv1d` and
`mamba_ssm` built from source, where every exact kernel re-verifies at the
same noise floor.  The H200 reports compute capability 9.0 (Hopper class), and
FLA refuses the Triton backward of gated chunk kernels under Triton 3.4–3.7
there (issue #640); `tilelang` supplies the backward for `gdn` / `gdn2` / `kda`,
while the `mamba2` / `mamba2_beta` kernels (simple-GLA op, no TileLang backend)
were *trained* in the retained torch 2.6 / Triton 3.2 environment and evaluated
in the new one.  Mamba-1 / Mamba-3 results and everything from this point on
come from the torch 2.9 environment.

> Note: the original `scripts/sft.py::chunked_cross_entropy_with_backward` (now
> `linswap/sft_utils.py`) had a scaling bug (the LM-head / tied-embedding gradient was a token *sum* while the
> hidden-state gradient was a token *mean*, inflating the pre-clip grad norm to
> ~1000).  It is fixed in this revision; all runs below use the fixed loss.
> The GDN2 numbers in docs/gdn2_experiment_log.md were produced with the old loss.

### Validation cross-entropy (first 40 validation examples ≤128K)

| model | trainable params | val CE | ppl |
|---|---|---|---|
| gdn  base (exact copy) | – | 1.7432 | 5.72 |
| gdn2 base (tiled init) | – | 1.7414 | 5.71 |
| kda  base (tiled init) | – | 1.7419 | 5.71 |
| rwkv7 base (tiled init) | – | 1.7430 | 5.71 |
| gdn  full 50 steps | 752M | 1.3884 | 4.01 |
| kda  full 50 steps | 759M | 1.3884 | 4.01 |
| gdn2 full 50 steps | 865M | 1.3885 | 4.01 |
| rwkv7 full 50 steps | 766M | 1.3884 | 4.01 |
| mamba2 base (inexact init) | – | 6.854 | 948 |
| mamba2 distill (layer 200 + KL 300 @8K) | 752M | 1.7293 | 5.64 |
| mamba2 distill → full 50 steps | 752M | 1.4908 | 4.44 |
| mamba1 base (inexact init) | – | 13.436 | 6.8e5 |
| mamba1 distill (layer 200 + KL 300 @8K) | 752M | 2.3234 | 10.2 |
| mamba1 distill → full 50 steps | 752M | 2.0677 | 7.9 |
| mamba3 base (inexact init, GDN-matched) | – | 15.328 | 4.5e6 |
| mamba3 distill (layer 200 + KL 300 @8K) | 752M | 3.8451 | 46.8 |
| mamba3 distill → full 50 steps @32K | 752M | 3.6794 | 39.6 |
| mamba3_min (Mamba-3's own SSM init) distill → full 50 | 752M | 3.8731 | 48.1 |
| deltanet base (inexact init) | – | 12.845 | 3.8e5 |
| deltanet full 50 steps, lr 1e-5 | 752M | 8.628 | 5586 |
| deltanet full 200 / 500 steps, lr 1e-5 | 752M | 7.174 / 6.145 | 1305 / 466 |
| deltanet full 200 steps, lr 1e-4 | 752M | 6.220 | 502 |
| deltanet distill (layer 200 + KL 300 @8K) | 752M | 2.461 | 11.7 |
| deltanet distill → full 50 steps | 752M | 2.092 | 8.1 |

The three bases are equal within kernel noise (function-preserving init).
Under the identical recipe, full SFT lands on the same loss for all three
kernels.

### RULER at 128K (131,072) tokens (100 samples per task, cached decode)

`niah_multivalue` is value-level accuracy.  (These runs predate the `evaluate` stage; today
`linswap evaluate` writes the same numbers to `outputs/eval/<name>/summary.csv`.)  Every model was evaluated with the
same `LinearSwapModelWrapper`, chat template, greedy decoding and 128 new tokens.

| model | niah_single_1 | niah_multikey_1 | niah_multivalue |
|---|---|---|---|
| gdn  base (exact copy) | 100.0 | 100.0 | 96.5 |
| gdn2 base (tiled init) | 100.0 | 100.0 | 96.25 |
| kda  base (tiled init) | 100.0 | 100.0 | 95.75 |
| rwkv7 base (tiled init) | 100.0 | 100.0 | 96.75 |
| gdn  full 50 | 100.0 | 100.0 | 99.0 |
| kda  full 50 | 100.0 | 100.0 | 99.5 |
| gdn2 full 50 | 100.0 | 100.0 | 99.25 |
| rwkv7 full 50 | 100.0 | 100.0 | 99.0 |
| mamba2 base (inexact) | 0.0 | 0.0 | 0.0 |
| mamba2 distill (500) | 100.0 | 72.0 | 53.25 |
| mamba2 SFT only, 50 / 500 steps | 0.0 / 95.0 | 0.0 / 66.0 | 1.0 / 55.0 |
| mamba1 distill → full 50 | 66.0 | 20.0 | 15.0 |
| mamba2 distill → full 50 | 100.0 | 77.0 | 73.5 |
| deltanet base (inexact) | 0.0 | 0.0 | 0.0 |
| deltanet full 200 (lr 1e-4), no distill | 0.0 | 0.0 | 0.0 |
| deltanet distill (500) | 0.0 | 0.0 | 0.0 |
| deltanet SFT only, 500 steps | 0.0 | 0.0 | 0.0 |
| deltanet distill → full 50 | 0.0 | 0.0 | 0.0 |

Reading: the three function-preserving bases are indistinguishable (single- and
multi-key retrieval saturated, multi-value 95.75–96.5, i.e. 1–2 wrong values
out of 400).  Full SFT under the identical recipe lifts multi-value retrieval
to 99–99.5 for every kernel, so on this budget the kernel choice does not
change the outcome of full fine-tuning.  Differences of ≤1 point on `niah_multivalue` (4
values out of 400) and 1–2 of 100 samples on `niah_multikey_1` are within sample noise, so
the robust conclusions are: (i) the swaps are exact and lose nothing, (ii) full SFT
equalises the kernels.

Mamba-1 (torch 2.9 environment) behaves like DeltaNet: distillation takes it from an
unusable start (13.4) to 2.32 and SFT to 2.07.  Mamba-3 is the hardest target: 15.3 →
3.85 → 3.68, and the control that keeps Mamba-3's own SSM initialisation instead of the
GDN-matched mapping ends slightly worse (3.87), so the gap is the architecture (no short
convolution, trapezoidal discretisation, rotary state, Δ-scaled writes) rather than the
init.  Mamba-3's SFT ran at 32K because its kernels are incompatible with activation
checkpointing.  The β-write control confirms the attribution for Mamba-2: keeping GDN's beta-scaled
write and dropping only the erase starts far closer to the original (3.13 vs 6.85) and
ends at 1.470 after distillation + SFT, against 1.491 with Δ-scaled writes and 1.388 for
the exact kernels — so about half of Mamba-2's remaining loss gap is the write scaling and
half the missing erase.  Long-context (packed 64K) distillation does not move the
validation loss for either Mamba-2 (1.504) or DeltaNet (2.042 vs 2.092); its effect on
long-range retrieval is reported in the RULER tables.  The two other inexact swaps ablate
the pretrained recurrence.  Mamba-2 removes the
delta-rule erase and keeps the decay: distillation returns it exactly to the
original model's validation loss (6.85 → 1.73, the same 1.74 the bases have),
SFT takes it to 1.49 (exact kernels: 1.39), single-needle retrieval is fully
recovered but multi-key / multi-value retrieval stays at 77 / 73.5 against 99+
for every kernel with an erase term — consistent with a scalar-decay state
being unable to overwrite stale associations, with the caveat that the adapter
also changes the write scaling (see the Mamba-2 section).  DeltaNet removes the
decay and keeps the erase, and is the harsher counter-example: at init the model is
unusable (validation CE 12.8, all NIAH scores 0), and 200 full-SFT steps at
lr 1e-4 only bring the loss to 6.2.  Distillation (`linswap distill`,
layer alignment 200 steps + KL 300 steps at 8K tokens, ~20 min) is far more
effective — 2.46, and 2.09 after the standard 50-step SFT — yet 128K-token
retrieval stays at 0: a state without decay never forgets, so at 16× the distillation length it is
full of stale associations.
At short context the distilled model does retrieve — `niah_single_1` / `niah_multikey_1` reach 56 / 36 at 4K tokens (25 samples) and 0 / 24 at 16K — so the failure is specifically the loss of long-range forgetting, not of the mechanism itself.

### Hard RULER tasks, 4K–128K (50 samples per task)

> **Protocol correction (13 September 2026, evening).**  RULER stores every sample's
> `answer_prefix` (e.g. *"Answer: According to the chain(s) of variable assignment in the
> text above, 5 variables are assigned the value X, they are:"*) separately from the
> prompt, and the vendored `pred/call_api.py` never passed it to the model, so every
> hard-task number produced before this note was obtained *without* the answer prefix:
> the models answered free-form and the short generation budgets (30 tokens for `vt`)
> were spent on preambles.  The wrapper now opens the assistant turn with the answer
> prefix (after the chat template, thinking disabled), as RULER's own chat templates do.
> On the GDN base at 4K / 25 samples this moves `vt` from 3.6 to 91.2, `cwe` from 58.6
> to 79.6 and leaves the needle tasks and QA within noise (`niah_multikey_3` 96,
> `qa_1` 68 vs 76).  Every hard-task table in this document and in the README was
> regenerated under the fixed protocol (`outputs/eval/sweep2-*`, `hard2-*`,
> `qwen38-27b`); the old runs are kept in `outputs/eval/hard` and `sweep-*` for reference
> only.  The easy-NIAH tables were not re-run: those tasks scored ≈100 without the prefix.


**Corrected protocol, all lengths** (50 samples per task, ±7 points; `outputs/eval/sweep2-A`,
`sweep2-B`, `hard2-A`, `hard2-B`).  Tasks: `niah_multikey_2/3` (essay haystack, distractor
needles), `niah_multiquery`, `vt` (variable tracking), `cwe` / `fwe` (common / frequent word
extraction), `qa_1` (SQuAD), `qa_2` (HotpotQA).

##### 4K tokens

| model | mk2 | mk3 | mq | vt | cwe | fwe | qa1 | qa2 | avg |
|---|---|---|---|---|---|---|---|---|---|
| gdn-base | 100.0 | 98.0 | 100.0 | 93.2 | 77.0 | 98.0 | 76.0 | 50.0 | 86.5 |
| gdn-full-50 | 100.0 | 100.0 | 100.0 | 95.2 | 65.2 | 97.3 | 68.0 | 58.0 | 85.5 |
| gdn2-full-50 | 100.0 | 100.0 | 100.0 | 95.2 | 63.8 | 98.0 | 72.0 | 54.0 | 85.4 |
| kda-full-50 | 100.0 | 100.0 | 100.0 | 95.2 | 63.6 | 98.0 | 70.0 | 56.0 | 85.3 |
| rwkv7-full-50 | 100.0 | 100.0 | 100.0 | 93.6 | 67.0 | 96.7 | 70.0 | 56.0 | 85.4 |
| mamba2-distill-sft-50 | 100.0 | 90.0 | 99.5 | 53.2 | 12.2 | 67.3 | 58.0 | 52.0 | 66.5 |

##### 16K tokens

| model | mk2 | mk3 | mq | vt | cwe | fwe | qa1 | qa2 | avg |
|---|---|---|---|---|---|---|---|---|---|
| gdn-base | 100.0 | 100.0 | 100.0 | 90.0 | 89.6 | 96.0 | 62.0 | 48.0 | 85.7 |
| gdn-full-50 | 100.0 | 100.0 | 100.0 | 81.6 | 70.4 | 95.3 | 60.0 | 48.0 | 81.9 |
| gdn2-full-50 | 100.0 | 100.0 | 100.0 | 84.4 | 71.2 | 95.3 | 60.0 | 48.0 | 82.4 |
| kda-full-50 | 100.0 | 100.0 | 100.0 | 80.0 | 70.4 | 95.3 | 56.0 | 48.0 | 81.2 |
| rwkv7-full-50 | 100.0 | 100.0 | 100.0 | 81.6 | 72.0 | 95.3 | 58.0 | 48.0 | 81.9 |
| mamba2-distill-sft-50 | 96.0 | 38.0 | 94.0 | 37.6 | 0.0 | 66.7 | 48.0 | 42.0 | 52.8 |

##### 64K tokens

| model | mk2 | mk3 | mq | vt | cwe | fwe | qa1 | qa2 | avg |
|---|---|---|---|---|---|---|---|---|---|
| gdn-base | 100.0 | 100.0 | 100.0 | 80.4 | 60.6 | 90.0 | 56.0 | 42.0 | 78.6 |
| gdn-full-50 | 100.0 | 100.0 | 100.0 | 62.0 | 51.4 | 92.7 | 52.0 | 40.0 | 74.8 |
| gdn2-full-50 | 100.0 | 100.0 | 100.0 | 61.6 | 49.6 | 93.3 | 54.0 | 40.0 | 74.8 |
| kda-full-50 | 100.0 | 100.0 | 100.0 | 63.2 | 50.4 | 92.0 | 52.0 | 42.0 | 74.9 |
| rwkv7-full-50 | 100.0 | 100.0 | 100.0 | 61.2 | 50.4 | 92.0 | 52.0 | 42.0 | 74.7 |
| mamba2-distill-sft-50 | 78.0 | 8.0 | 89.5 | 38.4 | 0.2 | 68.0 | 38.0 | 40.0 | 45.0 |

##### 128K tokens (all models; val CE of the added rows: noah 1.443–1.445, mamba2 SFT-only 1.708, distill-only 1.729, mamba2_beta 1.470, mamba2_lc 1.504, deltanet 2.093, deltanet_lc 2.042, mamba1 2.068)

| model | mk2 | mk3 | mq | vt | cwe | fwe | qa1 | qa2 | avg |
|---|---|---|---|---|---|---|---|---|---|
| gdn-base | 100.0 | 100.0 | 100.0 | 77.2 | 46.0 | 98.7 | 40.0 | 38.0 | 75.0 |
| gdn-full-50 | 98.0 | 100.0 | 100.0 | 79.6 | 9.0 | 98.0 | 42.0 | 40.0 | 70.8 |
| gdn2-full-50 | 98.0 | 100.0 | 100.0 | 80.0 | 9.6 | 98.0 | 44.0 | 44.0 | 71.7 |
| kda-full-50 | 98.0 | 100.0 | 100.0 | 80.4 | 9.6 | 98.0 | 44.0 | 40.0 | 71.2 |
| rwkv7-full-50 | 98.0 | 100.0 | 100.0 | 80.0 | 9.0 | 98.0 | 42.0 | 44.0 | 71.4 |
| kda-noah-full-50 | 98.0 | 98.0 | 100.0 | 80.8 | 9.4 | 98.7 | 50.0 | 48.0 | 72.9 |
| gdn-noah-full-50 | 98.0 | 98.0 | 100.0 | 80.8 | 9.4 | 98.7 | 48.0 | 50.0 | 72.9 |
| mamba2-sft-500 | 12.0 | 0.0 | 2.0 | 1.6 | 1.2 | 30.0 | 24.0 | 20.0 | 11.3 |
| mamba2-distill-500 | 42.0 | 8.0 | 64.5 | 22.0 | 0.4 | 59.3 | 16.0 | 32.0 | 30.5 |
| mamba2-distill-sft-50 | 54.0 | 4.0 | 89.0 | 26.4 | 0.4 | 63.3 | 36.0 | 30.0 | 37.9 |
| mamba2_beta-distill-sft-50 | 84.0 | 22.0 | 94.0 | 29.2 | 0.4 | 92.7 | 42.0 | 34.0 | 49.8 |
| mamba2_lc-distill-sft-50 | 18.0 | 0.0 | 86.5 | 58.4 | 0.4 | 69.3 | 34.0 | 32.0 | 37.3 |
| mamba1-distill-sft-50 | 0.0 | 0.0 | 21.0 | 0.0 | 0.2 | 15.3 | 4.0 | 12.0 | 6.6 |
| deltanet-distill-sft-50 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| deltanet_lc-distill-sft-50 | 0.0 | 0.0 | 0.0 | 0.4 | 0.2 | 0.0 | 0.0 | 2.0 | 0.3 |

##### 256K tokens (the backbone's native window; 25 samples per task, ±10 points)

| model | mk2 | mk3 | mq | vt | cwe | fwe | qa1 | qa2 | avg |
|---|---|---|---|---|---|---|---|---|---|
| gdn-base | 100.0 | 96.0 | 99.0 | 78.4 | 6.4 | 96.0 | 32.0 | 28.0 | 67.0 |
| gdn-full-50 | 96.0 | 88.0 | 100.0 | 75.2 | 2.0 | 94.7 | 24.0 | 28.0 | 63.5 |
| gdn2-full-50 | 96.0 | 88.0 | 100.0 | 76.8 | 3.6 | 96.0 | 24.0 | 32.0 | 64.6 |
| kda-full-50 | 96.0 | 88.0 | 100.0 | 76.0 | 3.2 | 94.7 | 24.0 | 36.0 | 64.7 |
| rwkv7-full-50 | 96.0 | 88.0 | 100.0 | 75.2 | 2.0 | 94.7 | 24.0 | 32.0 | 64.0 |
| mamba2-distill-sft-50 | 28.0 | 0.0 | 69.0 | 48.0 | 0.0 | 53.3 | 12.0 | 24.0 | 29.3 |

##### average over the 8 tasks vs length

| model | 4K | 16K | 64K | 128K | 256K |
|---|---|---|---|---|---|
| gdn-base | 86.5 | 85.7 | 78.6 | 75.0 | 67.0 |
| gdn-full-50 | 85.5 | 81.9 | 74.8 | 70.8 | 63.5 |
| gdn2-full-50 | 85.4 | 82.4 | 74.8 | 71.7 | 64.6 |
| kda-full-50 | 85.3 | 81.2 | 74.9 | 71.2 | 64.7 |
| rwkv7-full-50 | 85.4 | 81.9 | 74.7 | 71.4 | 64.0 |
| mamba2-distill-sft-50 | 66.5 | 52.8 | 45.0 | 37.9 | 29.3 |

Reading.

* **Exact kernels are indistinguishable at every length.**  After the identical 50-step
  full SFT, `gdn`, `gdn2`, `kda` and `rwkv7` sit within 0.6 average points of each other at
  4K, 16K, 64K and 128K, and within noise on every individual task.  The swap is invisible;
  the backbone and the recipe decide the numbers.
* **What SFT does, by length.**  The retrieval-flavoured SFT does not improve the hard
  tasks on average — it costs 1 point at 4K and 4 points at 64K–128K relative to the base.
  The cost is concentrated in common-word extraction (`cwe`: 77 → 65 at 4K, 46 → 9 at 128K;
  the fine-tuned models answer the counting task with confident, fabricated lists) and in
  variable tracking at 64K (80 → 62); needles, `fwe` and QA are unchanged.  Under the old
  protocol (no answer prefix) the same runs looked like SFT *lifting* `vt` from 0 to 19 —
  that was a formatting artefact.
* **Mamba-2 (no erase) is the only kernel whose retrieval degrades with length.**  After
  distillation and SFT it is close to the exact kernels on single-needle recall at 4K
  (`niah_multikey_2` 100, `multiquery` 99.5) but the distractor task collapses with
  context — `niah_multikey_3` 90 → 38 → 8 → 4 from 4K to 128K — and `cwe` is at zero from
  16K on; the average falls 66.5 → 37.9.  The exact kernels' averages fall only 86 → 71–75
  over the same range, almost all of it `cwe` and QA.

* **The anti-haystack data is not what breaks `cwe`.**  Full SFT on LongAlign + LongAlpaca
  only (`*-noah-full-50`, val CE 1.443–1.445 vs 1.388 with the full mixture) gives the same
  `cwe` collapse (9.4) and the same `vt` (80.8), and slightly better QA (48–50 vs 40–44);
  the two exact kernels again tie.  The `cwe` loss comes from the long-context SFT format
  itself, not from the anti-haystack subset.
* **Distillation vs SFT alone (Mamba-2).**  At matched validation loss (1.71 SFT-only for
  500 steps vs 1.73 distill-only for 500 steps) the SFT-only model retrieves almost nothing at
  128K (`niah_multikey_2` 12, `multiquery` 2, `vt` 1.6, avg 11) while the distilled one is
  at 42 / 64.5 / 22 (avg 30.5); distillation followed by the standard 50-step SFT is best on
  every task (avg 37.9).  The validation loss on SFT data does not measure what the swap
  broke; matching the teacher's distributions transfers the retrieval behaviour that SFT
  alone does not.
* **Controls for the Mamba-2 gap.**  Keeping GDN's β-scaled writes and dropping only the erase
  (`mamba2_beta`, val CE 1.470) recovers a large part of the gap — `niah_multikey_2` 84,
  `multiquery` 94, `fwe` 92.7, avg 49.8 vs 37.9 — so roughly half of the Δ-scaled adapter's
  loss was the write scale, not the missing erase; the distractor task (`multikey_3` 22) and
  `cwe` (0.4) stay far below the exact kernels, which is the part attributable to the erase.
  Distilling at longer context (`mamba2_lc`, packed KL at 8K then 64K, val CE 1.504) helps
  `vt` (58.4 vs 26.4) but hurts the needle tasks (`multikey_2` 18 vs 54), avg 37.3 — no net
  gain from long-context distillation under this budget.  For DeltaNet neither recipe
  (`deltanet`, `deltanet_lc`) retrieves anything at 128K, and Mamba-1 reaches only
  `multiquery` 21 / `fwe` 15.

### Perplexity: WikiText-103 and PG-19 length extrapolation (`evaluate --nll pg19,wikitext`; token-weighted NLL, 20 PG-19 books and the WikiText-103 test set, ≤128K tokens per document)

WikiText perplexity is the Gated DeltaNet paper's headline language-modelling metric; the PG-19 numbers binned by position are its Figure-2-style length-extrapolation curve.  The SFT-data validation loss reported elsewhere in this document is a training diagnostic only.

| model | PG-19 NLL | 0-4k | 4k-16k | 16k-64k | 64k-128k | WikiText NLL |
|---|---|---|---|---|---|---|
| gdn-base | 2.970 | 3.108 | 3.034 | 3.011 | 2.842 | 2.572 |
| gdn-full | 2.973 | 3.115 | 3.041 | 3.012 | 2.845 | 2.529 |
| gdn2-full | 2.973 | 3.115 | 3.041 | 3.012 | 2.845 | 2.529 |
| kda-full | 2.972 | 3.115 | 3.041 | 3.012 | 2.845 | 2.529 |
| rwkv7-full | 2.972 | 3.115 | 3.040 | 3.012 | 2.845 | 2.528 |
| mamba2-distill-sft | 3.149 | 3.293 | 3.212 | 3.183 | 3.034 | 2.694 |
| mamba1-distill-sft | 3.856 | 4.028 | 3.921 | 3.888 | 3.739 | 3.487 |
| deltanet-distill-sft | 4.460 | 4.202 | 4.175 | 4.354 | 4.832 | 4.455 |
| mamba3-distill-sft | 5.036 | 5.284 | 5.078 | 5.055 | 4.936 | 5.179 |

None of these texts were used for fine-tuning.  The exact kernels are identical to
three decimals on both corpora after SFT (and identical to the untouched original on
PG-19 to within 0.003), so the fine-tuning recipe neither helps nor hurts general
language modelling for them.  The inexact kernels rank as everywhere else
(Mamba-2 < Mamba-1 < DeltaNet < Mamba-3), and the position bins show the mechanism:
every model's loss falls with more context except DeltaNet, whose PG-19 loss *rises*
from 4.20 in the first 4K tokens to 4.83 beyond 64K — the model without decay cannot
forget, so more context hurts.

### Short-context regression (`linswap lmeval`, 0-shot, acc_norm where defined)

| model | kernel | hellaswag | piqa | arc_easy | arc_challenge | winogrande | lambada_openai |
|---|---|---|---|---|---|---|---|
| gdn-base | gdn | 0.4958 | 0.6926 | 0.6111 | 0.3737 | 0.5833 | 0.4372 |
| gdn-full | gdn | 0.5154 | 0.7002 | 0.6675 | 0.3968 | 0.5872 | 0.4789 |
| gdn2-full | gdn2 | 0.5171 | 0.7008 | 0.6688 | 0.3933 | 0.5943 | 0.4768 |
| kda-full | kda | 0.5166 | 0.7013 | 0.6688 | 0.3951 | 0.5856 | 0.4778 |
| rwkv7-full | rwkv7 | 0.5161 | 0.7002 | 0.6692 | 0.3968 | 0.5912 | 0.4762 |
| mamba2-distill-sft | mamba2 | 0.4784 | 0.6839 | 0.6393 | 0.3592 | 0.5533 | 0.4205 |
| mamba1-distill-sft | mamba1 | 0.3528 | 0.6099 | 0.4162 | 0.2671 | 0.5162 | 0.2661 |
| deltanet-distill-sft | deltanet | 0.3779 | 0.6491 | 0.5383 | 0.3003 | 0.532 | 0.2581 |
| mamba3-distill-sft | mamba3 | 0.268 | 0.543 | 0.2992 | 0.2176 | 0.5012 | 0.0763 |

The exact kernels sit within half a point of each other on every task and slightly
above the untouched original (the SFT data helps LAMBADA and ARC); the inexact kernels
lose 2–4 points (Mamba-2) to 10–20 points (DeltaNet, Mamba-1) and Mamba-3 is near
chance, in line with their validation losses.

### In-context multi-query associative recall (`linswap mqar`, 10 samples per size, accuracy %)

Text MQAR: N random word → 4-digit-number pairs, then every key queried in random order;
a query counts when all value tokens are the arg-max prediction (teacher forced).  Sequence
lengths ≈ 1.2K / 4.7K / 18.8K / 75K tokens for 64 / 256 / 1024 / 4096 pairs.

| model | kernel | acc@64 | acc@256 | acc@1024 | acc@4096 |
|---|---|---|---|---|---|
| gdn-base | gdn | 100.0 | 100.0 | 99.9 | 99.6 |
| gdn-full | gdn | 100.0 | 100.0 | 99.9 | 99.5 |
| gdn2-full | gdn2 | 100.0 | 99.9 | 99.9 | 99.5 |
| kda-full | kda | 100.0 | 99.9 | 99.9 | 99.5 |
| rwkv7-full | rwkv7 | 100.0 | 99.9 | 99.9 | 99.5 |
| mamba2-distill-sft | mamba2 | 99.8 | 99.3 | 99.1 | 92.6 |
| mamba1-distill-sft | mamba1 | 72.8 | 62.0 | 37.7 | 7.9 |
| deltanet-distill-sft | deltanet | 91.4 | 68.5 | 12.3 | 0.0 |
| mamba3-distill-sft | mamba3 | 0.5 | 0.0 | 0.0 | 0.0 |

The four exact kernels are indistinguishable (≥ 99.5 % at 4096 pairs — in a hybrid the
full-attention layers carry most of the recall), and the ordering of the inexact ones is
the ordering of how much of GDN they keep: Mamba-2 (decay, no erase) 92.6 % at 4096
pairs, DeltaNet (erase, no decay) collapses beyond 256 pairs, Mamba-1 degrades steadily
and Mamba-3 never recovers recall at all under this recipe.

Compared with the earlier GDN2 numbers in docs/gdn2_experiment_log.md (different machine, old loss
scaling): base 92.5 → full SFT 98.25 on `niah_multivalue`; here 96.25 → 99.25.

## Second scale: Qwen3.8-27B

Qwen3.8-27B (dense, Apache-2.0; 64 layers = 48 linear + 16 full-attention,
hidden 5120, 16 query / 48 value linear heads with head dim 128 — i.e. grouped
value heads, untied `lm_head`, 27B parameters) is the second scale.  Its
tokenizer is identical to Qwen3.5-0.8B's (same vocabulary and ids), so the SFT
mixture in `data/sft/len262144` is reused unchanged.  On one H200 a 64K-token
forward takes 20 s / 72 GiB in bf16, and gate-only SFT at 32K takes ≈ 22 s per
optimizer step (micro-batch 1 × 2 accumulation, gradient checkpointing).  Full
SFT of 27B parameters does not fit the 143 GiB budget with AdamW, so the 27B
protocol is: exactness verification, gate-only SFT (100 steps, lr 2e-4) for the
exact kernels with new parameters (`kda` 81M, `rwkv7` 139M; `gdn2` cannot be built
here — FLA's `GatedDeltaNet2` keeps `f`/`b` per key head and repeats them over the
three value heads of a group, so the per-value-head pretrained decay has no exact
image and the kernel now refuses grouped-value-head backbones), and a
reduced RULER subset (`niah_multikey_2`, `niah_multiquery`, `vt`, `qa_1` at 16K and
64K, 25 samples per task) plus the 64K validation loss.  No 128K runs, no
inexact kernels and no distillation were attempted at this scale.

**Exactness at 27B.** All three tiled inits and the exact `gdn` copy build
and load; `verify --checks logits,layerwise` against the HF implementation gives:

| kernel | T | max &#124;Δlogit&#124; | mean &#124;Δlogit&#124; | top-1 agreement | KL(HF ‖ ours) |
|---|---|---|---|---|---|
| gdn (exact copy) | 8 | 0.16 | 0.017 | 1.000 | 1.4e-4 |
| gdn (exact copy) | 512 | 15.3 | 0.36 | 0.980 | 0.066 |
| kda (tiled) | 8 | 0.14 | 0.017 | 1.000 | 1.1e-4 |
| kda (tiled) | 512 | 16.3 | 0.37 | 0.969 | 0.095 |
| HF sdpa vs HF eager (noise floor) | 512 | 17.5 | 0.29 | 1.000 | 1.7e-3 |

At 0.8B the same check sits at the HF noise floor (ours 1e-5 vs sdpa/eager
6e-5), so the 27B's KL of 0.066 asked for a diagnosis.  Feeding every block
HF's own input (forward hooks on HF, teacher-forced through our modules, bf16,
T = 512) localizes it: on both scales the attention blocks and the MLPs are
bit-exact (rel-L2 0.0 for all layers), and the *only* per-component difference
is the linear layer: mean rel-L2 0.0056 (max 0.009) for `gdn` on the 0.8B and
0.0049 (max 0.009) on the 27B — the same magnitude.  That difference is kernel
numerics, not weights: HF's `Qwen3_5GatedDeltaNet` and FLA's `GatedDeltaNet`
run different short-convolution, gated-norm and chunked delta-rule code paths in
bf16.  The per-block drift then compounds with depth (48 recurrent layers
instead of 18, and a wider residual stream), which is what the smoothly
increasing per-block mean difference in the verify output shows.  The fp32
check below confirms that the weights are function-preserving.

**fp32 check** (same T = 512 prompt, HF model and swapped model run one after
the other on a single GPU because a 27B fp32 model needs 100 GiB; per-layer
rel-L2 on HF's own inputs for probe layers):

| model | dtype | KL(HF ‖ ours) | top-1 | max &#124;Δlogit&#124; | linear-layer rel-L2 (probe layers) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | fp32 | 1.5e-8 | 1.000 | 0.006 | 2e-5 – 4e-5 |
| Qwen3.5-0.8B | bf16 | 5.3e-5 | 1.000 | 0.42 | 4e-3 – 5e-3 |
| Qwen3.8-27B | fp32 | 2.6e-6 | 1.000 | 3.8 | 7e-6 (layer 0) – 5e-4 (layer 30) |
| Qwen3.8-27B | bf16 | 6.6e-2 | 0.980 | 15.3 | 4e-3 – 9e-3 |

In fp32 the swapped 27B reproduces HF to KL 2.6e-6 (the bf16 HF-vs-HF floor is
1.7e-3), so the exact-copy and tiled initialisations are function-preserving at
this scale too; what remains in bf16 is the mixed-kernel rounding of 48 stacked
recurrent layers.  Practically: at 27B, compare swapped models against the
*same* implementation (the `gdn` swap) rather than against HF when a
bit-level baseline is needed, or verify in fp32.

**SFT data and the 27B.** Qwen3.8-27B is a thinking model: its default chat
template prepends a "reasoning effort" system message and opens a `<think>` block,
while with `enable_thinking=False` it renders exactly the Qwen3.5 format used for
the SFT mixture (no system message, empty think block).  On the first five
validation examples (≤ 32K) the unmodified HF Qwen3.8-27B scores a per-example
mean CE of 4.79 (token-weighted 5.08; individual examples 1.3–7.6) against 1.69
for the 0.8B on the same examples; the exact-copy `gdn` swap gives 4.76 and the
tiled `kda` swap 4.77 (bf16), i.e. the swaps reproduce the HF model's loss and
the high number is the base model itself in non-thinking mode on this data, not
the swap.  The 27B gate-only runs therefore start at 4.77 and reach ≈ 0.93–0.97
after 50–100 steps (`kda`: 4.77 → 0.968 → 0.938; `rwkv7`: 4.78 → 0.960 → 0.927),
most of which is adaptation to the answer style carried by the gate parameters
alone.  RULER prompts are rendered with each backbone's own chat template with
thinking disabled (`RULER/scripts/pred/model_wrappers.py`); with the default
template the 27B spends its 32–128 generated tokens on reasoning ("We need
answer user's request…") and scores 0 on `vt` / `qa_1`, so that setting matters
for any thinking backbone.

**Gate-only SFT and retrieval at 27B** (corrected RULER protocol, 25 samples per
task, cached greedy decoding; `outputs/eval/qwen38-27b`; 16K ≈ 27 min and 64K
≈ 43 min per model for the four tasks):

| model | trainable | val CE 32K (5 ex.) | mk2 16K | mk2 64K | mq 16K | mq 64K | vt 16K | vt 64K | qa1 16K | qa1 64K |
|---|---|---|---|---|---|---|---|---|---|---|
| gdn base (exact copy) | – | 4.77 | 100 | 100 | 100 | 100 | 100 | 100 | 80 | 84 |

Reading (25 samples ⇒ ±10 points): the 27B backbone solves the distractor
needle, multi-query and variable-tracking tasks perfectly at 16K and 64K, and
the tiled KDA and RWKV-7 swaps keep every one of those scores after gate-only
SFT while their validation loss drops from 4.77 to 0.93 — the function-
preserving init holds at 27B and 100 gate-only steps on 81–139M parameters
neither break long-context retrieval nor change QA beyond noise.  Where the
0.8B base already loses on `vt` with length (93 → 77 from 4K to 128K) the 27B
is saturated, so the second scale confirms containment rather than separating
the kernels; separating them at this scale would need full SFT (does not fit
here) or harder tasks.

## Which layers can be swapped? Per-layer sensitivity and greedy mixed-kernel models

Transformer→hybrid conversions choose *which* attention layers to keep by scoring each layer's
marginal effect (KL-guided layer selection, arXiv 2512.20569) or by greedy validation-guided
replacement after a blockwise local distillation of every layer (Distill-then-Replace,
arXiv 2601.11667).  `linswap sensitivity` is the kernel-swap analogue: each of the 18
linear-attention layers of the 0.8B backbone is replaced by the target kernel *alone*
(forward only) and scored by KL(teacher ‖ student) on 16 packed 2K FineWeb-Edu sequences
and by text MQAR; then layers are added greedily in order of least KL, re-scoring the
remaining candidates each round.  Candidate layers come either from the tiled init or from
the checkpoint after the `layer` stage of the literature recipe (blockwise local distillation
on 50M tokens; `outputs/sensitivity/*`).

| layer | mamba2, tiled init | mamba2, after blockwise distillation | deltanet, after blockwise distillation |
|---|---|---|---|
| 0 | 1.798 | 0.0100 | 0.0338 |
| 1 | 0.161 | 0.0074 | 0.0395 |
| 2 | 0.080 | 0.0048 | 0.0211 |
| 4 | 0.503 | 0.0039 | 0.0094 |
| 5 | 0.041 | 0.0038 | 0.0107 |
| 6 | 0.050 | 0.0037 | 0.0098 |
| 8 | 0.022 | 0.0026 | 0.0074 |
| 9 | 0.026 | 0.0026 | 0.0053 |
| 10 | 0.144 | 0.0038 | 0.0079 |
| 12 | 0.016 | 0.0027 | 0.0039 |
| 13 | 0.053 | 0.0031 | 0.0084 |
| 14 | 0.052 | 0.0034 | 0.0079 |
| 16 | 0.054 | 0.0044 | 0.0089 |
| 17 | 0.031 | 0.0033 | 0.0037 |
| 18 | 0.029 | 0.0028 | 0.0022 |
| 20 | 0.020 | 0.0028 | 0.0046 |
| 21 | 0.023 | 0.0049 | 0.0031 |
| 22 | 0.089 | 0.0054 | 0.0043 |
| *all 18* | 5.825 | 0.084 | 0.572 |

Greedy replacement (k = number of GDN layers turned into the target; KL, MQAR accuracy at
256 and 1024 pairs ≈ 4.7K / 19K tokens):

| k | mamba2, tiled init | mamba2, after BLD | deltanet, after BLD |
|---|---|---|---|
| 3 | 0.060 / 1.00 / 1.00 | 0.007 / 1.00 / 1.00 | 0.008 / 1.00 / 1.00 |
| 6 | 0.165 / 1.00 / 1.00 | 0.014 / 1.00 / 1.00 | 0.020 / 1.00 / 1.00 |
| 9 | 0.346 / 1.00 / 0.99 | 0.024 / 1.00 / 1.00 | 0.047 / 1.00 / 0.77 |
| 12 | 0.645 / 0.23 / 0.09 | 0.038 / 1.00 / 1.00 | 0.091 / 0.98 / 0.15 |
| 15 | 1.437 / 0.17 / 0.02 | 0.055 / 1.00 / 1.00 | 0.183 / 0.92 / 0.00 |
| 18 | 5.825 / 0.00 / 0.00 | 0.084 / 1.00 / 0.99 | 0.572 / 0.01 / 0.00 |

Greedy order (first = cheapest): mamba2 after BLD 8, 9, 20, 12, 18, 17, 14, 5, 13, 6, 4, 22,
10, 16, 2, 21, 1, 0; deltanet after BLD 18, 21, 12, 17, 22, 20, 9, 8, 14, 4, 10, 16, 13, 5, 6,
0, 2, 1.

Reading.

* **Sensitivity is front-loaded.**  Before any training the first linear layer costs 1.8 nats
  of KL on its own when swapped to Mamba-2, layer 4 0.5, layer 1 0.16; the middle and late
  layers cost 0.02–0.05.  The same ordering survives distillation: layers 0, 1, 2 are the last
  ones the greedy procedure swaps for both kernels.
* **Blockwise local distillation is what makes swapping cheap, and single-layer scores are
  nearly additive for Mamba-2.**  After 50M tokens of per-layer MSE the sum of the 18
  single-layer KLs (0.067) is close to the all-swapped KL (0.084), and MQAR stays ≥ 0.99 at
  every k — the missing erase is not visible at 19K tokens on this probe.
* **DeltaNet (no decay) has a budget of about nine layers.**  MQAR at 1024 pairs is intact
  up to k = 6, drops to 0.77 at k = 9 and to 0.15 at k = 12; the early layers (5, 6, 0, 2, 1)
  cannot be swapped without losing recall.  This is the layer-level version of the
  "state saturation" of additive/undecayed recurrences reported by "What matters in
  linearizing LMs" (arXiv 2504.14366): a few undecayed layers are tolerable, a majority is not.
* **Mixed models on the hard tasks confirm the budget and sharpen it.**  The greedy sets were
  materialised as checkpoints (`outputs/mixed/*`, spec strings such as
  `gdn;deltanet@8,9,12,14,17,18,20,21,22`; only the swapped layers carry the blockwise-distilled
  weights, no further training) and run on the eight hard tasks (25 samples):

  | model (k of 18 linear layers swapped) | 4K | 16K | 64K |
  |---|---|---|---|
  | gdn base (0) | 86.5 | 85.7 | 78.6 |
  | deltanet k = 6 | 82.4 | 79.0 | 57.2 |
  | deltanet k = 9 | 65.9 | 23.7 | 5.0 |
  | deltanet k = 12 | 40.6 | 7.8 | 0.5 |
  | mamba2 k = 9 | 72.3 | 61.6 | 49.2 |
  | mamba2 k = 18 (layer stage only) | 60.6 | 42.8 | 26.0 |
  | mamba2 k = 18, full literature recipe + SFT | 78.0 | 67.3 | 52.5 |

  Six blockwise-distilled DeltaNet layers cost 4–7 points at 4K–16K and keep the needle tasks
  (96 / 96 / 99 at 16K); the ninth layer is the cliff (multi-key needles 0 at 16K), exactly where
  the MQAR probe predicted it.  For Mamba-2 nine layers cost 14–29 points, with the distractor task
  (`multikey_3` 80 at 16K, 64 at 64K) and `vt` bearing the loss, and swapping all 18 without the
  later stages costs far more than the full recipe recovers — the KL/hidden/CE stages are worth
  ~20 points at every length.  KL on 2K text ranks layers correctly but understates task damage
  by an order of magnitude (k = 6 DeltaNet: KL 0.02, hard-task loss 4–21 points), so the
  greedy selection should be run on a task metric when the target is retrieval.

## Literature recipe for inexact swaps: generic-text, four-stage distillation

The chat-data recipe above (~8M tokens of layer-MSE + KL on the SFT mixture) follows the
GDN2 tech blog.  Transformer→hybrid conversions instead distil on generic web text with a
staged recipe and 10–100× the tokens: token-mixer alignment → hidden-state alignment →
end-to-end KL → long-context fine-tuning (RADLADS arXiv 2505.03005, HALO 2601.22156,
"What matters in linearizing LMs" 2504.14366, Retrieval-Aware Distillation 2602.11374).
`linswap distill --text_data fineweb-edu --stages layer,hidden,kl,ce` implements that
recipe: FineWeb-Edu sample-10BT (one shard, 756M tokens, tokenised once), packed 2K
sequences, 16 × 2 accumulation (66K tokens per step); budgets 50M (layer, lr 1e-4, linear
params) / 50M (hidden, lr 1e-4, linear params) / 300M (KL, lr 2e-5, all params) / 100M at
16K (CE, lr 1e-5, all params); ≈ 4.5 h per kernel on one L20X.  The teacher's packed-text
validation CE is 2.645; the students reach 2.73 → 2.69 → 2.66 after the first three stages and
2.55 after the CE stage, which trains on the same distribution and therefore drops *below* the
teacher on this split (it is not a sign of surpassing the teacher).  The standard 50-step SFT
follows, then the corrected-protocol hard tasks (50 samples) and perplexity.

| model | recipe | 4K | 16K | 64K | 128K | PG-19 NLL | WikiText NLL |
|---|---|---|---|---|---|---|---|
| gdn base | – | 86.5 | 85.7 | 78.6 | 75.0 | 2.972 | 2.529 |
| mamba2 distill-only | chat, 500 steps | – | – | – | 30.5 | – | – |
| mamba2 distill → SFT | chat, 500 + 50 | 66.5 | 52.8 | 45.0 | 37.9 | 3.149 | 2.694 |
| mamba2 distill-only | **literature, 500M tokens** | 59.4 | 33.9 | 22.2 | 19.2 | **2.961** | **2.470** |
| mamba2 distill → SFT | **literature + 50 SFT** | **78.0** | **67.3** | **52.5** | **49.1** | 2.982 | 2.511 |
| deltanet distill → SFT | chat, 500 + 50 | – | – | – | 0.0 | 4.460 | 4.455 |
| deltanet distill-only | **literature, 500M tokens** | 46.2 | 18.7 | 0.3 | 0.2 | 4.229 | 4.107 |
| deltanet distill → SFT | **literature + 50 SFT** | 60.3 | 32.4 | 4.2 | 0.3 | 3.907 | 3.541 |
| mamba2_beta distill → SFT | chat, 500 + 50 | – | – | – | 49.8 | – | – |
| mamba2_beta distill-only | **literature, 500M tokens** | 69.7 | 39.1 | 26.0 | 24.0 | 2.922 | 2.440 |
| mamba2_beta distill → SFT | **literature + 50 SFT** | 75.7 | 63.6 | **55.4** | 47.7 | 2.946 | 2.477 |

Per task, Mamba-2 with the literature recipe + SFT at 16K: `niah_multikey_2` 100,
`multikey_3` 92, `multiquery` 92.5, `vt` 73.6, `fwe` 76, QA 56 / 48 (chat recipe: 96 / 38 / 94 /
37.6 / 66.7 / 48 / 42); at 128K: 58 / 2 / 88.5 / 86.4 / 92 / 36 / 30 (chat: 54 / 4 / 89 / 26.4 /
63.3 / 36 / 30).

Reading.

* **The recipe, not the kernel, was the bottleneck for Mamba-2 at short and medium context.**
  Generic-text distillation with the four stages lifts the hard-task average by 11–15 points at
  every length (66.5 → 78.0 at 4K, 52.8 → 67.3 at 16K, 45.0 → 52.5 at 64K, 37.9 → 49.1 at
  128K) and closes the perplexity gap entirely: the distilled student's PG-19 / WikiText NLL
  (2.961 / 2.470) is at or below the GDN base (2.972 / 2.529), because the last stage trains on
  web text; after SFT it is 2.982 / 2.511.
* **What remains is the erase.**  The distractor-needle task at 128K (`multikey_3` 2, vs 100 for
  the exact kernels) and `cwe` (0.2) do not move with more distillation tokens, the same tasks
  the β-write control left open; the remaining Mamba-2 gap is architectural, consistent with
  "state resolution, not budget" in arXiv 2504.14366.
* **Distill-only is not the right checkpoint to judge.**  Before SFT the literature-recipe
  student scores *lower* than the chat-distilled one on the hard tasks (19.2 vs 30.5 at 128K)
  because it has never seen the instruction format; after the identical 50-step SFT it is 11
  points better.  Loss and retrieval move independently in both directions.
* **The β-write control converges with the Δ-write adapter under the literature recipe.**  With
  enough distillation tokens the two Mamba-2 variants end within noise of each other (75.7 / 63.6 /
  55.4 / 47.7 vs 78.0 / 67.3 / 52.5 / 49.1), whereas under the chat recipe the β-write control was
  12 points ahead at 128K: the write-scale mismatch is something distillation repairs, the missing
  erase is not (both stay at 2–4 on `niah_multikey_3` at 128K and ≈ 0 on `cwe`).
* **DeltaNet (no decay) gains at short context only.**  At 4K the literature recipe + SFT
  reaches 60.3 (chat recipe ≈ 55 on the needle tasks, 0 elsewhere), but every task is still
  zero from 64K on, before and after SFT: undecayed state saturation is not a budget problem.

## `gdn_breg`: Gated DeltaNet with Bregman soft-thresholding of the state

An external, in-development kernel (`gated_breg_delta_rule/`, not shipped): GDN whose state is
soft-thresholded after every complete 64-token chunk, ``S ← sign(S)·max(|S| − λ, 0)`` (the proximal
step of an L1 penalty on the state).  Its parameters are exactly GDN's, so the swap is a weight
copy and λ = 0 is the `gdn` control; `LINSWAP_BREG_LAM` sets λ at build time.  Registered as
`gdn_breg` once the kernel's λ = 0 path was made bit-compatible with FLA's GDN (log2-domain gates)
and its decode path thresholded at the same absolute positions as training.

**λ = 0 (acceptance gate, 0.8B, vs HF):** layer rel-L2 0.005–0.008, logits T = 4096 KL 1.4e-5 /
top-1 1.000, cached decode and greedy generation agreement 1.000 — identical to `gdn`.

**λ = 0.01, zero-shot (no training):**

| metric | gdn_breg λ = 0.01 | gdn (λ = 0) |
|---|---|---|
| KL vs HF, T = 512 / 4096 | 0.011 / 0.013 | 8e-5 / 1e-5 |
| top-1 vs HF, T = 512 / 4096 | 0.994 / 0.999 | 1.000 / 1.000 |
| per-block mean drift, block 0 / 12 / 23 (T = 256) | 0.004 / 0.023 / 0.073 | 0.0002 / 0.0008 / 0.0034 |
| PG-19 NLL (bins 0–4K / 4–16K / 16–64K / 64–128K) | 3.436 (3.47 / 3.45 / 3.48 / 3.36) | 2.972 |
| WikiText NLL | 3.007 | 2.529 |
| hard RULER at 4K: mk2 / mk3 / mq / vt / cwe / fwe / qa1 / qa2 (avg) | 78 / 8 / 92 / 19.6 / 6 / 76.7 / 44 / 38 (45.3) | 100 / 98 / 100 / 93.2 / 77 / 98 / 76 / 50 (86.5) |

Reading: at λ = 0.01 the pretrained backbone does not tolerate the threshold — +0.46 nats on both
corpora, flat across position bins (the state is degraded everywhere, not only at long range),
and the 4K hard-task average halves, with the distractor needles (8) and variable tracking (20)
hit hardest.  The zero-shot sweep at 16K–128K was stopped as uninformative; the operator has to
be trained at λ.  Next: continued training with chat replay at λ = 0.01 (`distill --stages ce
--text_mix 0.1`, 300M @ 8K + 100M @ 16K) evaluated directly, against the identical λ = 0 run.

## Can a newer kernel beat GDN?  Equal-budget continued training

The question behind the project: does swapping GDN for a kernel with a richer gate (GDN2's three
dense gates, KDA's per-channel decay) pay off once the model is trained enough to use it?  Short
SFT cannot answer it (the exact kernels tie after 50 steps, see above), and the linearization
literature only sees kernel differences after 1–10B tokens.  Test: identical continued training
for `gdn`, `gdn2`, `kda` — 300M tokens of FineWeb-Edu packed at 8K then 100M at 16K, lr 1e-5,
cross-entropy on every token (`distill --stages ce --ce_schedule 8192:300e6,16384:100e6`), no
chat replay in this first pass — then the standard 50-step SFT; both checkpoints evaluated.

| model | recipe | 4K | 16K | 64K | 128K | PG-19 NLL | WikiText NLL | SFT-data CE |
|---|---|---|---|---|---|---|---|---|
| gdn base | – | 86.5 | 85.7 | 78.6 | 75.0 | 2.972 | 2.529 | 1.743 |
| gdn full SFT | base + 50 SFT | 85.5 | 81.9 | 74.8 | 70.8 | – | – | 1.388 |
| gdn | continued 400M | 65.0 | 39.7 | 35.6 | 33.5 | **2.858** | **2.392** | 2.184 |
| gdn2 | continued 400M | 64.7 | 39.5 | 35.7 | 33.6 | 2.858 | 2.392 | 2.181 |
| kda | continued 400M | 65.8 | 39.5 | 35.7 | 33.3 | 2.858 | 2.392 | 2.190 |
| gdn | continued + 50 SFT | 83.4 | 77.2 | 67.4 | 64.8 | 2.885 | 2.426 | 1.391 |
| gdn2 | continued + 50 SFT | 84.3 | 76.6 | 67.9 | 65.3 | 2.885 | 2.426 | 1.391 |
| kda | continued + 50 SFT | 83.6 | 76.9 | 67.4 | 65.3 | 2.885 | 2.426 | 1.392 |

Reading.

* **No.**  After 400M tokens of identical training the three kernels are the same model to the
  precision of the measurement: perplexities equal to three decimals on both corpora, SFT-data
  loss within 0.01, hard-task averages within 0.9 points at every length and every individual
  task within sample noise.  Whatever the extra gate capacity of GDN2 (113M parameters) or KDA
  (7.4M) could express, this budget does not find it, and the shared trajectory (loss curves
  overlap step for step) suggests the pretrained GDN solution is a stationary point for the
  richer parameterisations too.  The blog's "GDN2 beats GDN" was an artefact (see above).
* **Continued training on web text without replay trades retrieval for perplexity.**  The
  400M tokens lower PG-19 / WikiText NLL by 0.11 / 0.14 nats but move the model out of the
  instruction format (multi-key needle 100 → 46, `qa_1` 76 → 26 at 4K before SFT) and, even
  after the same SFT, leave the hard tasks 2–6 points below base + SFT at every length.  The
  10 % chat replay added afterwards (`--text_mix`) is the fix; the λ = 0 replay control below
  measures it.
* **Replay fixes the format, not the long-range erosion.**  With 10 % chat documents in the
  stream (`--text_mix 0.1`) the same 400M-token GDN run evaluates directly at 84.0 / 75.7 /
  56.4 / 50.2 (4K–128K) with no SFT — at 4K–16K within 3–10 points of the base and above
  base + SFT at 4K — while its multi-key needles still fall to 74 / 46 at 64K and 58 / 12 at
  128K (base: 100 / 100).  Training at 8K–16K context erodes retrieval beyond the training
  length whatever the data mix; recovering it needs the training context to reach the
  evaluation length (HyLo's 64K stage), which is the setting for the node runs.
* Two seeds and the 1.5B-token version of this comparison are running on the 8-GPU node
  (`docs/runbook_8gpu.md`); at 0.8B a null result at 1.5B tokens would close the question.

**λ = 0.01 after continued training with chat replay** (300M tokens @ 8K + 100M @ 16K of
FineWeb-Edu with 10 % chat documents, lr 1e-5, all parameters, evaluated directly — no SFT;
`outputs/cpt/gdn_breg`, `outputs/eval/cpt-gdn_breg`):

| model | 4K | 16K | 64K | 128K | PG-19 NLL | WikiText NLL | MQAR @ 256 / 1024 / 4096 pairs |
|---|---|---|---|---|---|---|---|
| gdn base | 86.5 | 85.7 | 78.6 | 75.0 | 2.972 | 2.529 | 1.00 / 1.00 / 1.00 |
| gdn_breg λ = 0.01, zero-shot | 45.3 | – | – | – | 3.436 | 3.007 | – |
| gdn_breg λ = 0.01, continued 400M + replay | 58.5 | 50.0 | 36.8 | 30.7 | 2.928 | 2.504 | 1.00 / 0.99 / 0.86 |
| gdn λ = 0, continued 400M, no replay | 65.0 | 39.7 | 35.6 | 33.5 | 2.858 | 2.392 | – |
| gdn λ = 0, continued 400M + replay | 84.0 | 75.7 | 56.4 | 50.2 | 2.837 | 2.399 | 1.00 / 1.00 / 0.99 |

Per task after training, λ = 0.01: 4K 80 / 66 / 97.5 / 37 / 2 / 79 / 64 / 42; 16K 82 / 36 / 90 /
34 / 1 / 77 / 44 / 36; 64K 66 / 0 / 57 / 29 / 1 / 77 / 30 / 34; 128K 44 / 0 / 36.5 / 27 / 0 / 83 /
32 / 22 (mk2 / mk3 / mq / vt / cwe / fwe / qa1 / qa2).

Reading, with the replay-matched λ = 0 control: 400M tokens of training at λ = 0.01 recover
most of the perplexity (3.436 → 2.928, still 0.09 / 0.10 nats above the identical λ = 0 run)
and the short-context needles (multikey_3 8 → 66 at 4K), but the threshold costs 20–26 points
of hard-task average at every length relative to the same recipe without it (84.0 / 75.7 /
56.4 / 50.2 → 58.5 / 50.0 / 36.8 / 30.7), the distractor needle is 0 from 64K on, `cwe` stays
at 0–2, `vt` at 27–37 (control 65–85), and MQAR at 4096 pairs drops to 0.86 (control 0.99).
At λ = 0.01 the threshold removes state the retrieval tasks need and training does not learn
around it; a smaller λ (0.001–0.003) is the next operating point, and the equal-forgetting
control (faster decay matched to the state norm) is still needed before attributing any effect
to sparsification rather than forgetting.

### Recall-intensive short-context tasks (`linswap lmeval --tasks swde,fda,squad_completion`, 500 examples each, greedy, 2K context)

The three recall-intensive tasks of the Gated DeltaNet paper (Table 4) and of the KL-guided /
HALO conversion papers; metric is lm-eval's `contains` (the gold span appears in the
generation).  `outputs/eval/recall-short`.

| model | recipe | SWDE | FDA | SQuAD-completion |
|---|---|---|---|---|
| gdn base | – | 0.884 | 0.852 | 0.730 |
| gdn | full SFT 50 | 0.880 | 0.814 | 0.730 |
| gdn2 | full SFT 50 | 0.878 | 0.816 | 0.728 |
| kda | full SFT 50 | 0.880 | 0.818 | 0.732 |
| rwkv7 | full SFT 50 | 0.882 | 0.816 | 0.724 |
| mamba2 | chat distill → SFT | 0.822 | 0.694 | 0.648 |
| mamba2_beta | chat distill → SFT | 0.862 | 0.732 | 0.670 |
| mamba2 | literature distill → SFT | 0.896 | 0.752 | 0.676 |
| deltanet | chat distill → SFT | 0.570 | 0.578 | 0.446 |
| deltanet | literature distill → SFT | 0.836 | 0.746 | 0.608 |
| mamba1 | chat distill → SFT | 0.584 | 0.460 | 0.336 |

Reading: the exact kernels tie again (±0.005), and SFT costs 4 points of FDA for all of them
(0.852 → 0.814–0.818) — the same "instruction data hurts extraction" effect as `cwe`.  The
literature recipe is worth 7 / 6 / 3 points to Mamba-2 and 27 / 17 / 16 to DeltaNet on these
2K-context tasks, i.e. DeltaNet's short-context recall is largely recoverable once it is
distilled on enough generic text, while its long-context recall (RULER ≥ 64K) is not.

### Zero-shot commonsense, second set (`linswap lmeval --tasks boolq,social_iqa,lambada_openai`; acc, LAMBADA also ppl)

Completes the Gated DeltaNet paper's Table-3 task set (the first set — PIQA, HellaSwag, ARC-e/c,
WinoGrande — is above).  `outputs/eval/csr-extra`.

| model | recipe | BoolQ | SIQA | LAMBADA acc | LAMBADA ppl |
|---|---|---|---|---|---|
| gdn base | – | 0.625 | 0.390 | 0.437 | 14.74 |
| gdn | full SFT 50 | 0.719 | 0.396 | 0.479 | 12.26 |
| gdn2 | full SFT 50 | 0.721 | 0.398 | 0.477 | 12.25 |
| kda | full SFT 50 | 0.719 | 0.395 | 0.478 | 12.27 |
| rwkv7 | full SFT 50 | 0.720 | 0.397 | 0.476 | 12.25 |
| mamba2 | chat distill → SFT | 0.760 | 0.401 | 0.421 | 19.56 |
| mamba2_beta | chat distill → SFT | 0.794 | 0.405 | 0.431 | 15.81 |
| mamba2 | literature distill → SFT | 0.785 | 0.414 | **0.496** | **12.02** |
| deltanet | chat distill → SFT | 0.596 | 0.380 | 0.258 | 71.5 |
| deltanet | literature distill → SFT | 0.736 | 0.400 | 0.451 | 14.63 |
| mamba1 | chat distill → SFT | 0.571 | 0.358 | 0.266 | 120.4 |

Reading: the exact kernels tie to the third decimal; SFT raises BoolQ and LAMBADA for all of
them.  The literature-recipe Mamba-2 is the best model on LAMBADA (0.496 / 12.0 vs 0.478 / 12.3
for the exact kernels), because its 400M tokens of generic-text training are a larger language
-modelling budget than the exact kernels' 50 SFT steps — a reminder that these short-context
scores measure recent training data more than the recurrence.  DeltaNet's LAMBADA perplexity
falls from 71.5 to 14.6 with the literature recipe.
