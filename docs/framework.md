# LinearSwap: the kernel-swap framework (`linswap`)

Replaces the Gated-DeltaNet (GDN) linear-attention layers of a pretrained hybrid backbone
(Qwen3.5-0.8B in every experiment here) with *any* linear-attention kernel, initialises the new layer so
that the pretrained function is preserved where that is possible, verifies it, distils it back to the
original model on generic text and benchmarks it — all through one kernel name.

```
pyproject.toml         `pip install -e .` -> the `linswap` command (src/linswap/cli.py; also `python -m linswap`)
src/linswap/
  hf.py                LinearSwapConfig / LinearSwapCache / LinearSwapForCausalLM (transformers PreTrainedModel,
                       registered with AutoConfig / AutoModelForCausalLM on import) + export()
  registry.py          KernelSpec + register_kernel / get_kernel / list_kernels; kernel maps ("gdn;mamba2@3,6")
  kernels/common.py    helpers shared by init recipes (pretrained tensor layout, split fused qkv/conv, tiling, output gate)
  kernels/fla_layer.py build_fla_layer / register_fla_kernel: any fla.layers class + init recipe -> KernelSpec
  kernels/base.py      BackboneMixer: projections + convs + cache + gated norm around an FLA op (custom recurrences)
  kernels/gdn.py       "gdn"          original GDN on FLA kernels (exact copy; control and distillation teacher)
  kernels/gdn2.py      "gdn2"         Gated DeltaNet-2 (scalar beta/decay tiled into b/w/f gates)
  kernels/kda.py       "kda"          Kimi Delta Attention, low-rank per-channel decay gate (default KDA)
                       "kda_fullgate" KDA with a dense decay projection
  kernels/rwkv7.py     "rwkv7"        RWKV-7 generalised delta rule (DPLR kernel), exact tiled init
  kernels/mamba2.py    "mamba2"       Mamba-2 SSD on the simple-GLA kernel — inexact swap (exact_init=False)
  kernels/deltanet.py  "deltanet"     DeltaNet, no decay — inexact swap
  kernels/gla.py       "gla"          Gated Linear Attention, stock FLA layer, no custom code — inexact swap
  kernels/mamba3.py    "mamba3"       FLA Mamba3 (mamba_ssm kernels), GDN decay mapped into the fused in_proj — inexact
  kernels/mamba1.py    "mamba1"       FLA Mamba (mamba_ssm kernels), values/gate/conv copied — inexact
  kernels/gdn_breg.py  "gdn_breg"     GDN + Bregman soft-thresholding of the state (external package, optional)
  model.py             LinearSwapBackbone (model.embed_tokens / layers / norm) + LinearSwapModel (adds lm_head)
                       — Qwen's module tree and state-dict keys; SwapCache; per-layer kernel maps
  components.py        RMSNorm / GQA / MLP / RoPE;  backbones.py  load_backbone_config() from the HF config
  load_weights.py      build_model(kernel | ckpt_dir), HF-format and native checkpoint loading
  textdata.py          generic-text corpora (DCLM, FineWeb-Edu) tokenised once;  train_utils.py  packing + chunked losses
  pipeline/verify.py     stage 1: function-preservation checks vs the HF backbone
  pipeline/distill.py    stage 2: the three training steps (layer alignment -> KL -> context extension)
  pipeline/evaluate.py   stage 3: RULER (calls RULER's scripts directly) -> summary table
  pipeline/lmeval.py     stage 3: short-context suite through lm-eval-harness, with relative scores
  pipeline/run.py        the stages chained for one kernel
  pipeline/export.py     write a swapped model / checkpoint as an HF checkpoint (safetensors + tokenizer + card)
tools/                 throughput.py (prefill / decode speed), hard_tables.py (evaluation logs -> markdown tables)
tests/                 test_kernels.py (all kernels vs control), test_hf.py (HF round-trip), test_batch.py (padded batches),
                       test_gva.py (grouped value heads), test_losses.py (chunked CE / KL vs dense autograd; CPU, no model)
examples/quickstart.ipynb
RULER/scripts/pred/model_wrappers.py::LinearSwapModelWrapper, server types linswap[_nocache]
```

## Using it

```bash
source .venv/bin/activate
linswap verify   --kernel rwkv7 --baseline gdn        # is the swap a faithful replacement?
linswap distill  --kernel rwkv7                       # three training steps -> outputs/rwkv7/distill
linswap evaluate --models gdn rwkv7-distilled=outputs/rwkv7/distill/checkpoint-16338 --name rwkv7
linswap lmeval   --models gdn rwkv7-distilled=outputs/rwkv7/distill/checkpoint-16338
linswap run      --kernel rwkv7                       # all of the above
```

```python
from linswap import build_model, list_kernels
model = build_model("rwkv7")                                          # backbone weights, exact RWKV-7 init
model = build_model(ckpt_dir="outputs/rwkv7/distill/checkpoint-16338")  # distilled checkpoint (kernel from config.json)
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
  counted as the kernel's new parameters (used for the parameter counts below).

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
which keeps them invisible at init but gives them non-zero gradient so
distillation can use the extra rank.  `kda_fullgate` uses a dense 2048×1024 `f_proj`
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


## Distillation (`linswap distill`)

Teacher: `gdn`, an exact copy of the original model.  Student: the swapped model.  Three steps on
packed generic web text (DCLM by default; `--text_data fineweb-edu` is the alternative), no instruction
data and no chat template anywhere, no supervised fine-tuning afterwards.  Both exact and inexact swaps
use the same recipe — for an exact swap the first step starts from zero loss and the run is a
teacher-matched continuation rather than a repair.

| step | loss | tokens | length | sequences/step | lr | trained |
|---|---|---|---|---|---|---|
| `layer` | L2 between each student mixer's output and the teacher's, on the teacher's own layer input, all layers in parallel | 100M | 512 | 32 | 1e-3 → 1e-5 cosine | the swapped layers |
| `kl` | KL(teacher ‖ student) on next-token distributions, in vocabulary chunks | 500M | 512 | 96 | 1e-5 flat | all parameters |
| `ce` | plain next-token cross-entropy, no teacher (context extension) | 100M | 16384 | 96 (8 × 12) | 1e-5 flat | all parameters |

Adam(0.9, 0.95, 1e-8), clip 1.0, bf16.  Budgets are given in tokens and converted to optimizer steps;
`--stage_length` / `--stage_batch` / `--stage_micro` / `--stage_schedule` override any step.  Freezing
the MLPs or the embeddings in the KL step costs accuracy, so everything trains.  About 6 GPU-hours per
kernel at 0.8B on one L20X; the checkpoint under `outputs/<kernel>/distill/` is what gets evaluated.

## Evaluation

Two suites, both applied to the students *and* to the unmodified backbone so the comparison is
like-for-like:

* **Long-context retrieval** (`linswap evaluate`): RULER's needle tasks `niah_single_1/2/3` and
  `niah_multikey_1` at 4K / 16K / 64K / 128K, 50 samples each, cached greedy decoding.  Prompts use
  RULER's own base template (context, question, answer prefix); `--chat_template` switches to the
  backbone's chat format.  Since the students never see an instruction format during distillation,
  base prompting is the setting in which teacher and student are scored the same way.
* **Short context** (`linswap lmeval`): LAMBADA, ARC-c (acc_norm), ARC-e, PIQA, WinoGrande,
  HellaSwag (acc_norm) 0-shot and MMLU 5-shot through lm-eval-harness, reported both as accuracy and
  as a relative score (s − r)/(t − r) against a reference row, with r the chance level.

`tools/throughput.py` measures prefill and decode speed, `tools/hard_tables.py` turns evaluation logs
into markdown tables.

## Results (Qwen3.5-0.8B, one seed)

Everything below is the recipe above: 700M tokens of DCLM, no SFT, base-prompt evaluation.  The
**control** is the *unswapped* backbone put through the identical three steps — without it the
students' gains over the teacher cannot be attributed to the kernel.

**RULER / passkey** (50 samples; `niah_single_1` / `_2` / `_3` / `niah_multikey_1`):

| model | 4K | 16K | 64K | 128K |
|---|---|---|---|---|
| teacher (unmodified backbone) | 94 / 68 / 98 / 82 | 98 / 76 / 96 / 86 | 94 / 100 / 94 / 90 | 98 / 86 / 100 / 94 |
| control (`gdn`, same three steps) | 100 / 100 / 94 / 100 | 100 / 100 / 100 / 98 | 100 / 100 / 100 / 96 | 100 / 100 / 100 / 90 |
| `rwkv7` (exact init) | 100 / 100 / 98 / 96 | 100 / 100 / 100 / 96 | 100 / 100 / 100 / 94 | 100 / 100 / 92 / 92 |
| `mamba2` (no erase) | 100 / 100 / 98 / 98 | 100 / 100 / 100 / 96 | 100 / 100 / 98 / 92 | 100 / 98 / 92 / 72 |

**Short context**, accuracy and relative score against the teacher in %:

| model | LAMBADA | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | MMLU 5-shot | rel. avg |
|---|---|---|---|---|---|---|---|---|
| teacher | 0.437 (ppl 14.7) | 0.374 | 0.611 | 0.693 | 0.583 | 0.496 | 0.504 | 100.0 |
| control (`gdn`) | 0.478 (13.2) | 0.399 | 0.653 | 0.706 | 0.588 | 0.524 | 0.515 | **110.0** |
| `rwkv7` | 0.479 (13.3) | 0.391 | 0.642 | 0.705 | 0.590 | 0.521 | 0.517 | 108.7 |
| `mamba2` | 0.462 (14.2) | 0.372 | 0.610 | 0.701 | 0.578 | 0.520 | 0.501 | 101.4 |

**Throughput** (one L20X, bf16, batch 1, cached greedy decode, 256 new tokens):

| model | prefill 8K / 32K (tok/s) | decode (ms/token) | peak 8K / 32K (GiB) |
|---|---|---|---|
| `gdn` (FLA chunk kernel) | 134K / 141K | 24.5 | 2.0 / 3.4 |
| `mamba2` (FLA simple-GLA op) | 119K / 128K | 30.5 | 2.0 / 3.3 |
| `rwkv7` (FLA DPLR chunk kernel) | 79K / 76K | 33.3 | 2.7 / 6.1 |

Reading.

* **The recipe, not the kernel, is what lifts the scores above the teacher.**  The control gains as
  much as the students on both suites (relative average 110.0, needles at 100 almost everywhere), so
  the right question is what the *swap* costs on top of it.
* **With an exact init the swap is nearly free.**  `rwkv7` sits 1.3 relative points under the control
  on the short-context suite and within sample noise of it on every needle length; its 92 / 92 at 128K
  against the control's 100 / 90 is the ±7-point noise of 50 samples.
* **The missing erase still costs.**  `mamba2` trails the control by 8.6 relative points and loses the
  distractor needle at 128K (72 vs 90) — the same failure mode as under every earlier recipe, now the
  only substantial gap left.
* Decode time is length-independent for all three (constant state).  RWKV-7's DPLR kernel is the
  slowest and needs the most memory (two rank-1 terms per step, more chunk intermediates); Mamba-2's
  SSD recurrence runs through FLA's generic simple-GLA kernels rather than Mamba-2's own fused CUDA
  kernels.

## Approaches that were tried and dropped

Kept here as a record; the code for them is no longer in the tree.

* **Supervised fine-tuning on long-context chat data** (LongAlign / LongAlpaca / anti-haystack), both
  full and gate-only.  50 steps equalised every exact kernel, and the retrieval-flavoured data cost
  common-word extraction (77 → 65 at 4K, 46 → 9 at 128K) and QA at long context.  Gate-only SFT never
  beat full SFT.  A no-anti-haystack ablation scored the same, so the loss came from the format, not
  from one subset.
* **Distilling on that chat corpus** instead of generic text: ~8M tokens of layer + KL.  Replacing it
  with the generic-text recipe was worth 11–15 hard-task points for Mamba-2 at every length and 16–27
  points of short-context recall for DeltaNet.
* **Continued training on generic text without a teacher** (400M tokens, with and without 10 %
  instruction replay): it lowers perplexity but does not beat distillation, and without replay it
  drifts the instruct backbone out of its answer format (multi-key needle 100 → 46 at 4K).
* **A long-context KL curriculum** (packed 8K → 64K) and **hidden-state alignment as a separate step**:
  no measurable gain over the three steps above.
* **Equal-budget comparisons of `gdn` vs `gdn2` vs `kda`** (400M tokens each): identical to three
  decimals on perplexity and within 0.9 points on every task — a richer gate does not beat GDN at this
  scale and budget.  The earlier claim that GDN2 beats GDN came from a run with a loss-scaling bug.
* **Per-layer swap sensitivity and mixed-kernel models**: the machinery stays (kernel maps such as
  `"gdn;mamba2@3,6"`, partial checkpoint loading) but the search stage was removed.  Result worth
  remembering: six blockwise-distilled DeltaNet layers are nearly free, the ninth is a cliff.
* **A second scale (27B)**: verified function-preserving in fp32 (KL 2.6e-6), but the only
  post-training that fitted the hardware was gate-only SFT, which is gone with the rest of the SFT
  stage.

## Environment

Python 3.11, torch 2.9.1 / CUDA 12.8 / Triton 3.5.1, `flash-linear-attention` 0.6.0 (git 8e84ed4),
transformers 5.16.1; `causal-conv1d` and `mamba_ssm` built from source for the `mamba1` / `mamba3`
kernels.  On Hopper-class GPUs FLA refuses the Triton backward of its gated chunk kernels under
Triton 3.4–3.7 (issue #640): install `tilelang` (`pip install -e ".[hopper]"`) for `gdn` / `gdn2` /
`kda`, and use Triton < 3.4 or ≥ 3.7.1 to *train* `mamba2` (the simple-GLA op has no TileLang
backend; inference is unaffected).  The distillation and evaluation numbers above were produced on
2 × NVIDIA L20X (143 GiB).
