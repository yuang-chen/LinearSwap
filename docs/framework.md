# LinearSwap: the kernel-swap framework (`linswap`)

Generalises the GDN→GDN2 in-place swap (docs/gdn2_experiment_log.md,
docs/gdn2_swap_notes.md) into a small framework that can replace the Gated-DeltaNet (GDN) linear-attention
layers of a pretrained hybrid backbone (Qwen3.5-0.8B in all experiments here) with *any* linear-attention kernel, initialise the new layer so that the
pretrained function is preserved, verify it, fine-tune it and benchmark it on
RULER — all through one kernel name.  The first new kernel is **Kimi Delta
Attention (KDA)**.

```
pyproject.toml         `pip install -e .` -> the `linswap` command (src/linswap/cli.py); linswap.py is a checkout launcher
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
  model.py             LinearSwapBackbone (model.embed_tokens / layers / norm) + LinearSwapModel (adds lm_head)
                       — Qwen's module tree and state-dict keys; SwapCache
  components.py        RMSNorm / GQA / MLP / RoPE;  backbones.py  load_backbone_config() from the HF config
  load_weights.py      build_model(kernel | ckpt_dir), HF-format and native checkpoint loading
  data.py              SFT data preparation (LongAlign / LongAlpaca / anti-haystack);  sft_utils.py  chunked CE etc.
  pipeline/verify.py     stage 1: function-preservation checks vs HF Qwen3.5
  pipeline/distill.py    stage 1½ (inexact kernels): layer-wise alignment + KL distillation from the original
  pipeline/posttrain.py  stage 2: gate-only / full SFT (prepares data on first use; --init_ckpt to start from distill)
  pipeline/evaluate.py   stage 3: validation loss + RULER (calls RULER's scripts directly) -> summary table
  pipeline/run.py        the stages chained for one kernel
  pipeline/export.py     write a swapped model / checkpoint as an HF checkpoint (safetensors + tokenizer + card)
tests/test_kernels.py  regression test over all registered kernels;  tests/test_hf.py  HF save/load/generate round-trip
RULER/scripts/pred/model_wrappers.py::LinearSwapModelWrapper, server types linswap[_nocache]
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

Without decay the state never contracts (for unit keys and β∈[0,1] the
transition is non-expansive, so it does not blow up, but stale associations
persist until they are explicitly overwritten), which is why the deviation
increases with length and gradient norms in SFT start in the thousands.  This kernel is kept as the worked example of an inexact swap and of
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
> `linswap/sft_utils.py`) had a scaling bug (the LM-head / tied-embedding gradient was a token *sum* while the
> hidden-state gradient was a token *mean*, inflating the pre-clip grad norm to
> ~1000).  It is fixed in this revision; all runs below use the fixed loss.
> The GDN2 numbers in docs/gdn2_experiment_log.md were produced with the old loss.

### Validation cross-entropy (first 40 validation examples ≤131K)

| model | trainable params | val CE | ppl |
|---|---|---|---|
| gdn  base (exact copy) | – | 1.7432 | 5.72 |
| gdn2 base (tiled init) | – | 1.7414 | 5.71 |
| kda  base (tiled init) | – | 1.7419 | 5.71 |
| rwkv7 base (tiled init) | – | 1.7430 | 5.71 |
| gdn  gate-only 100 steps | 0.59M | 1.4720 | 4.36 |
| kda  gate-only 100 steps | 7.41M | 1.4274 | 4.17 |
| kda_fullgate gate-only 100 steps | 38.1M | 1.4131 | 4.11 |
| gdn2 gate-only 100 steps | 113.3M | 1.3776 | 3.97 |
| rwkv7 gate-only 100 steps | 14.2M | 1.4147 | 4.12 |
| gdn  full 50 steps | 752M | 1.3884 | 4.01 |
| kda  full 50 steps | 759M | 1.3884 | 4.01 |
| gdn2 full 50 steps | 865M | 1.3885 | 4.01 |
| rwkv7 full 50 steps | 766M | 1.3884 | 4.01 |
| mamba2 base (inexact init) | – | 6.854 | 948 |
| mamba2 distill (layer 200 + KL 300 @8K) | 752M | 1.7293 | 5.64 |
| mamba2 distill → full 50 steps | 752M | 1.4908 | 4.44 |
| deltanet base (inexact init) | – | 12.845 | 3.8e5 |
| deltanet full 50 steps, lr 1e-5 | 752M | 8.628 | 5586 |
| deltanet full 200 / 500 steps, lr 1e-5 | 752M | 7.174 / 6.145 | 1305 / 466 |
| deltanet full 200 steps, lr 1e-4 | 752M | 6.220 | 502 |
| deltanet distill (layer 200 + KL 300 @8K) | 752M | 2.461 | 11.7 |
| deltanet distill → full 50 steps | 752M | 2.092 | 8.1 |

The three bases are equal within kernel noise (function-preserving init).
Under the identical recipe, full SFT lands on the same loss for all three
kernels, while gate-only SFT separates the kernels by gate capacity: the
per-channel KDA decay gate (7.4M params) recovers about half of the gap
between GDN's scalar gates and GDN2's three full-rank gates, and the dense
`kda_fullgate` variant a little more.

### RULER at 131072 tokens (100 samples per task, cached decode)

`niah_multivalue` is value-level accuracy.  (These runs predate the `evaluate` stage; today
`linswap.py evaluate` writes the same numbers to `outputs/eval/<name>/summary.csv`.)  Every model was evaluated with the
same `LinearSwapModelWrapper`, chat template, greedy decoding and 128 new tokens.

| model | niah_single_1 | niah_multikey_1 | niah_multivalue |
|---|---|---|---|
| gdn  base (exact copy) | 100.0 | 100.0 | 96.5 |
| gdn2 base (tiled init) | 100.0 | 100.0 | 96.25 |
| kda  base (tiled init) | 100.0 | 100.0 | 95.75 |
| rwkv7 base (tiled init) | 100.0 | 100.0 | 96.75 |
| gdn  gate-only 100 (0.59M) | 100.0 | 100.0 | 96.25 |
| kda  gate-only 100 (7.4M) | 100.0 | 100.0 | 97.5 |
| kda_fullgate gate-only 100 (38M) | 100.0 | 98.0 | 99.0 |
| gdn2 gate-only 100 (113M) | 100.0 | 99.0 | 96.5 |
| rwkv7 gate-only 100 (14M) | 100.0 | 99.0 | 99.5 |
| gdn  full 50 | 100.0 | 100.0 | 99.0 |
| kda  full 50 | 100.0 | 100.0 | 99.5 |
| gdn2 full 50 | 100.0 | 100.0 | 99.25 |
| rwkv7 full 50 | 100.0 | 100.0 | 99.0 |
| mamba2 base (inexact) | 0.0 | 0.0 | 0.0 |
| mamba2 distill (500) | 100.0 | 72.0 | 53.25 |
| mamba2 SFT only, 50 / 500 steps | 0.0 / 95.0 | 0.0 / 66.0 | 1.0 / 55.0 |
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

The two inexact swaps ablate the pretrained recurrence.  Mamba-2 removes the
delta-rule erase and keeps the decay: distillation returns it exactly to the
original model's validation loss (6.85 → 1.73, the same 1.74 the bases have),
SFT takes it to 1.49 (exact kernels: 1.39), single-needle retrieval is fully
recovered but multi-key / multi-value retrieval stays at 77 / 73.5 against 99+
for every kernel with an erase term — consistent with a scalar-decay state
being unable to overwrite stale associations, with the caveat that the adapter
also changes the write scaling (see the Mamba-2 section).  DeltaNet removes the
decay and keeps the erase, and is the harsher counter-example: at init the model is
unusable (validation CE 12.8, all NIAH scores 0), and 200 full-SFT steps at
lr 1e-4 only bring the loss to 6.2.  Distillation (`linswap.py distill`,
layer alignment 200 steps + KL 300 steps at 8K tokens, ~20 min) is far more
effective — 2.46, and 2.09 after the standard 50-step SFT — yet 131K-token
retrieval stays at 0: a state without decay never forgets, so at 16× the distillation length it is
full of stale associations.
At short context the distilled model does retrieve — `niah_single_1` / `niah_multikey_1` reach 56 / 36 at 4K tokens (25 samples) and 0 / 24 at 16K — so the failure is specifically the loss of long-range forgetting, not of the mechanism itself.

### Hard RULER tasks at 131072 tokens (50 samples per task)

`niah_multikey_2/3` (essay haystack, distractor needles), `niah_multiquery`,
`vt` (variable tracking), `cwe` / `fwe` (common / frequent word extraction),
`qa_1` (SQuAD) and `qa_2` (HotpotQA); `outputs/eval/hard/summary.csv`.

| model | val CE | mk2 | mk3 | mq | vt | cwe | fwe | qa1 | qa2 | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| gdn-base |  | 100.0 | 98.0 | 100.0 | 0.0 | 36.4 | 87.3 | 36.0 | 36.0 | 61.7 |
| gdn-full-50 |  | 96.0 | 94.0 | 100.0 | 19.2 | 2.2 | 97.3 | 42.0 | 46.0 | 62.1 |
| gdn2-full-50 |  | 96.0 | 94.0 | 100.0 | 19.2 | 3.0 | 97.3 | 40.0 | 44.0 | 61.7 |
| kda-full-50 |  | 96.0 | 94.0 | 100.0 | 19.2 | 3.2 | 98.0 | 44.0 | 46.0 | 62.5 |
| rwkv7-full-50 |  | 96.0 | 94.0 | 100.0 | 19.2 | 2.2 | 96.7 | 42.0 | 42.0 | 61.5 |
| kda-gate-100 |  | 100.0 | 96.0 | 100.0 | 5.6 | 3.0 | 92.7 | 40.0 | 34.0 | 58.9 |
| rwkv7-gate-100 |  | 100.0 | 92.0 | 100.0 | 4.4 | 0.4 | 90.7 | 32.0 | 34.0 | 56.7 |
| mamba2-sft-50 | 2.655 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 7.3 | 2.0 | 0.0 | 1.2 |
| mamba2-sft-500 | 1.709 | 12.0 | 2.0 | 6.5 | 0.0 | 1.0 | 30.7 | 10.0 | 0.0 | 7.8 |
| mamba2-distill-500 | 1.729 | 52.0 | 16.0 | 49.0 | 0.4 | 0.4 | 6.0 | 10.0 | 24.0 | 19.7 |
| mamba2-distill-sft-50 |  | 62.0 | 6.0 | 83.5 | 20.4 | 0.4 | 59.3 | 26.0 | 32.0 | 36.2 |
| deltanet-sft-50 | 8.584 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| deltanet-sft-500 | 6.145 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| deltanet-distill-500 | 2.461 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| deltanet-distill-sft-50 | 2.092 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.0 | 0.5 |

Reading (50 samples ⇒ ±7 points): the four exact kernels' full-SFT checkpoints
are indistinguishable on every task, including the ones that separate models
(variable tracking, QA), confirming the easy-NIAH conclusion.  Mamba-2 without
the delta-rule erase collapses on the distractor-heavy needle tasks (6 on
`niah_multikey_3`) and on frequent-word extraction — consistent with the missing
delta-rule erase, although this adapter also changes β-scaled writes to Δ-scaled
writes, and a control that keeps β-scaled writes is needed before attributing
the gap to the erase alone.  Two SFT effects are visible for
every kernel: the 50-step retrieval-flavoured SFT lifts variable tracking (0 →
19) and QA, but destroys common-word extraction (36 → ~3): the fine-tuned
models answer the counting task with confident, fabricated lists.  Gate-only
checkpoints keep the base model's needle scores but gain far less on `vt`
(5) than full SFT (19).

Distillation vs SFT alone for the approximate targets (val CE = validation loss
on the same 40 examples; 25 samples per task for DeltaNet, 50 otherwise):
Mamba-2 trained with SFT only reaches the *same* validation loss after 500
steps as distillation does (1.71 vs 1.73) yet retrieves far worse (multikey_2
12 vs 52, multiquery 6.5 vs 49), and the recipe-matched 50-step SFT-only run is
at zero; distillation followed by the standard 50-step SFT is best on every
retrieval task (avg 36).  The validation loss on SFT data therefore does not
measure what the swap broke; matching the teacher's distributions transfers the
retrieval behaviour that SFT alone does not.  DeltaNet is 0 everywhere at
131K regardless of recipe (SFT-only 500 steps: 6.14 val CE; distill: 2.46;
distill+SFT: 2.09) — see the short-context numbers above.

Compared with the earlier GDN2 numbers in docs/gdn2_experiment_log.md (different machine, old loss
scaling): base 92.5 → full SFT 98.25 on `niah_multivalue`; here 96.25 → 99.25.
