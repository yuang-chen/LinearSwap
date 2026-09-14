<div align="center">
  <img src="docs/logo.png" width="520" alt="LinearSwap">
</div>

<div align="center">

**Swap the linear-attention kernel of a pretrained hybrid LLM in place — no pretraining.**

</div>

LinearSwap replaces the linear-attention layers of a pretrained hybrid language model
(a backbone that interleaves Gated DeltaNet layers with full attention, such as
Qwen3-Next / Qwen3.5 / Qwen3.6 / Qwen3.8) with **other linear recurrences** —
Gated DeltaNet-2, Kimi Delta Attention, RWKV-7, Mamba-2, DeltaNet — and keeps the
rest of the network.  When the target recurrence contains Gated DeltaNet as a
special case, the new layer is initialised so that the model computes *exactly
the same function* at step 0 (verified to bf16 noise); when it does not, the
model is distilled from the original.  Both are then post-trained with
long-context SFT and benchmarked on RULER, all through one command line with
`--kernel <name>` as the only thing that changes between experiments.

Because every kernel starts from the same pretrained function, LinearSwap turns
"which recurrence is better?" into a controlled post-training experiment instead
of a set of incomparable pretraining runs.  The kernels come from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention).

* [Kernels](#kernels)
* [Installation](#installation)
* [Usage](#usage)
  * [Building a swapped model](#building-a-swapped-model)
  * [Token mixing layers](#token-mixing-layers)
  * [Adding a kernel](#adding-a-kernel)
* [Training](#training)
* [Evaluation](#evaluation)
* [Benchmarks](#benchmarks)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Kernels

| name | recurrence (FLA kernel) | init from the pretrained GDN layer | new params | exact |
|---|---|---|---|---|
| `gdn` | Gated DeltaNet | weight copy (control) | 0.59M | yes |
| `gdn2` | Gated DeltaNet-2 | scalar beta/decay tiled into b/w/f gates | 113M | yes ‡ |
| `kda` | Kimi Delta Attention | scalar decay tiled into the low-rank per-channel gate | 7.4M | yes |
| `kda_fullgate` | Kimi Delta Attention | … with a dense decay projection | 38M | yes |
| `rwkv7` | RWKV-7-style DPLR generalised delta rule | decay/beta tiled, removal key = key | 14M | yes |
| `mamba2` | Mamba-2-style SSD (scalar decay, no erase) | shared weights copied, erase dropped | 0.30M | no |
| `deltanet` | DeltaNet (erase, no decay) | shared weights copied, decay dropped | 0.29M | no |
| `gla` | Gated Linear Attention (per-channel decay, no erase) | shared weights copied, decay MLP at FLA init | 0.92M | no |
| `mamba3` † | Mamba-3 (data-dependent decay, trapezoidal, rotary state) | GDN decay / projections mapped into the fused `in_proj`, rotary and trapezoid neutral | 0.05M | no |
| `mamba1` † | Mamba-1 selective SSM (per-channel, no q/k) | values, gate, conv and `out_proj` copied; SSM params at Mamba init | 19M | no |

"Exact" kernels reproduce the pretrained model at initialisation and go straight
to SFT; the others are distilled first (`linswap distill`).  Parameter counts
are for the 0.8B backbone.  ‡ `gdn2` requires as many value heads as key heads: FLA's
`GatedDeltaNet2` shares its decay and erase gates across a group of value heads, so a
backbone with grouped value heads (Qwen3.8-27B: 16 key / 48 value heads) has no exact
GDN2 image and the kernel refuses to build there.  † `mamba1` / `mamba3` use `mamba_ssm`'s kernels through
FLA's `Mamba` / `Mamba3` layers and are registered only when those import (see
Installation).  Mamba-3's single-token decode step additionally needs `mamba_ssm`'s
CuTe-DSL kernel (`nvidia-cutlass-dsl` + `quack-kernels`), which did not run with the
currently published versions; prefill, training and distillation work, and `evaluate
--no_cache` recomputes the prefix per generated token instead.  Every target is the *recurrence* of the named
architecture inside a backbone-compatible block (the backbone's projections,
short convolutions and gated output norm are kept; e.g. RWKV-7's token shift and
GroupNorm are not used) — see [docs/framework.md](docs/framework.md) for each
mapping and why FLA's native RWKV-7 / Mamba-2 layers cannot express the
pretrained weights.

## Installation

Requirements: Python 3.11, PyTorch ≥ 2.7 with a matching Triton ≥ 3.3
(FLA's requirement; tested with torch 2.9.1 / CUDA 12.8 / Triton 3.5.1),
`flash-linear-attention` 0.6, `transformers` ≥ 5.16.

`flash-linear-attention` has to come from the repository, not from PyPI: its
published wheels ship only `fla/layers` and `fla/models` (no `fla/__init__.py`,
no `fla.ops`, no `fla.modules`), and the last release, 0.5.2, predates the
kernels used here anyway.  Git main calls itself 0.6.0, which is what the
dependency pin refers to; the results below were produced with commit
`8e84ed4`.

```bash
git clone https://github.com/yuang-chen/LinearSwap && cd LinearSwap
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention@8e84ed4"
uv pip install -e ".[eval]"                              # linswap + the `linswap` command; [eval] adds RULER's deps
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
python tests/test_kernels.py && python tests/test_hf.py  # kernels match the control; HF round-trip is exact
                                                         # (also tests/test_batch.py, tests/test_gva.py)
```

Optional, for the fast short-convolution path and the `mamba1` / `mamba3` kernels
(`mamba_ssm` needs Triton ≥ 3.5, i.e. torch ≥ 2.9; always build it *without*
dependency resolution or it will replace your torch):

```bash
CUDA_HOME=/usr/local/cuda MAX_JOBS=32 uv pip install --no-deps --no-build-isolation \
    --no-binary causal-conv1d --no-binary mamba-ssm causal-conv1d mamba-ssm
```

On Hopper-class GPUs (compute capability 9.x, e.g. H100) with Triton 3.4 to 3.7, FLA
rejects the Triton backward of its gated chunk kernels as incorrect (issue #640) and needs
`uv pip install -e ".[hopper]"` (TileLang) for training `gdn`, `gdn2`, `kda`; the
simple-GLA path used by `mamba2` has no TileLang backend, so its *training* needs
Triton < 3.4 or ≥ 3.7.1 (inference is unaffected).

RULER's word list and QA datasets are fetched by
`RULER/scripts/data/synthetic/json/download_*.{py,sh}`.

## Usage

### Building a swapped model

```python
from linswap import build_model, list_kernels

model = build_model("kda", base_model_dir="models/Qwen3.5-0.8B")     # exact KDA init from the pretrained weights
model = build_model(ckpt_dir="outputs/kda/sft_full/checkpoint-50")     # a checkpoint; kernel read from its config.json
logits = model(input_ids)                                              # [B, T, vocab]
out = model.generate(input_ids, max_new_tokens=32)                     # greedy, cached decode (KV + recurrent state)
```

The architecture is read from the backbone's HF `config.json`; any GDN-based
hybrid with the Qwen3-Next layer layout loads, including the grouped-value-head
configurations of the larger Qwen models (all kernels except `gla` and `deltanet`).  `linswap kernels` lists the
registered kernels.

### Hugging Face `transformers`

Swapped models are `PreTrainedModel`s (`LinearSwapForCausalLM`, architecture
`linswap`, registered with the Auto classes on `import linswap`), so they save,
load, generate and evaluate like any HF model:

```bash
linswap export --kernel kda --out hf/Qwen3.5-0.8B-KDA                        # base swap
linswap export --ckpt outputs/kda/sft_full/checkpoint-50 --out hf/Qwen3.5-0.8B-KDA-sft
```

```python
import torch, linswap                                                  # `import linswap` registers the architecture
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("hf/Qwen3.5-0.8B-KDA")
model = AutoModelForCausalLM.from_pretrained("hf/Qwen3.5-0.8B-KDA", dtype=torch.bfloat16).cuda()
ids = tok.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
print(tok.decode(model.generate(**ids, max_new_tokens=32, do_sample=False)[0]))
```

`LinearSwapForCausalLM.from_swap("kda")` does the conversion in memory;
`save_pretrained` / `push_to_hub` write a Hub-ready checkpoint (config,
safetensors, tokenizer, model card) — e.g. `huggingface-cli upload <user>/Qwen3.5-0.8B-KDA hf/Qwen3.5-0.8B-KDA`.  Checkpoints use **Qwen's tensor layout**:
`model.embed_tokens`, `model.layers.{i}.{self_attn,mlp,input_layernorm,post_attention_layernorm}`,
`model.norm`, `lm_head` are byte-identical to the backbone's tensors (194 of 194
for the `gdn` kernel), and only `model.layers.{i}.linear_attn.*` differs per
kernel — so quantisers, converters and diff tools see "Qwen with a different
linear layer".  Native `model.pt` checkpoints use the same keys (older
`trf_blocks.*` checkpoints are converted on load).  Limits: batches must be unpadded or
right-padded for loss / logits, generation takes equal-length prompts (rows
that finish are padded until all are done), and decoding is greedy or
sampling only (the model is stateful, so no beam search).

### Token mixing layers

Each kernel is an `nn.Module` with the flash-linear-attention layer interface,
usable on its own:

```python
from linswap import get_kernel, load_hf_state_dict, load_backbone_config
cfg = load_backbone_config("models/Qwen3.5-0.8B")
spec = get_kernel("rwkv7")
layer = spec.build(cfg, layer_idx=0)                                   # RWKV7DeltaLayer
spec.init_from_gdn(layer, load_hf_state_dict("models/Qwen3.5-0.8B"), 0, "model.language_model")
y, _, cache = layer(x)                                                 # x: [B, T, hidden]
```

### Adding a kernel

LinearSwap is a consumer of flash-linear-attention, not a second layer zoo.
A kernel is either a **stock FLA layer** plus an init recipe, or an **FLA op**
(a recurrence) wrapped in the shared backbone-compatible block.

*Stock FLA layer* — `register_fla_kernel` builds the layer from the backbone
config with FLA's own constructor names, copies the weights every layer shares
with GDN (q/k/v, convolutions, output gate) and calls your `init_extra` for the
rest.  GLA is registered this way in its entirety:

```python
from fla.layers import GatedLinearAttention
from linswap.kernels.fla_layer import register_fla_kernel

register_fla_kernel(
    "gla", GatedLinearAttention, description="Gated Linear Attention",
    layer_kwargs=lambda cfg: dict(expand_k=2.0, expand_v=2.0, num_heads=16, use_short_conv=True,
                                  use_output_gate=True, gate_fn="swish", fuse_norm=True),
    output_gate="native", norm_attr="g_norm_swish_gate",   # GLA already has the backbone's gate
    new_param_names=("gk_proj",), exact_init=False)
```

`gdn`, `gdn2`, `kda` and `deltanet` are registered the same way (with a
`post_build` hook where a dense gate replaces FLA's low-rank one, and an
`init_extra` that tiles the pretrained scalar gates).

*FLA op* — subclass `linswap.kernels.base.BackboneMixer`, add your parameters
and implement `recurrence(hidden_states, q, k, v, state, use_cache)`; the base
provides projections, convolutions, q/k normalisation, the FLA cache protocol
and the gated output norm.  `kernels/rwkv7.py` (DPLR delta rule) and
`kernels/mamba2.py` (SSD) are the two examples, each under 40 lines of kernel code.

Register with `exact_init=True` only if the init is function preserving, then
run `python tests/test_kernels.py` and `linswap verify --kernel <name> --baseline gdn`:
an exact swap sits at the `gdn` control's noise level on every check.
`kernels/common.py` documents the pretrained tensor layout and provides the
splitting / tiling / low-rank-embedding helpers.

## Training

The workflow is **verify → (distill) → posttrain → evaluate**, one command each;
`run` chains them for one kernel and distils automatically when the kernel's
init is not exact.

```bash
linswap verify    --kernel kda --baseline gdn        # layer / logits / layer-wise / cache / generation vs HF
linswap distill   --kernel mamba2                    # inexact kernels: layer alignment (200) + KL (300) @8K
linswap posttrain --kernel kda                       # gate-only (100 steps, 2e-4) and full SFT (50 steps, 1e-5)
linswap posttrain --kernel mamba2 --modes full --init_ckpt outputs/mamba2/distill/checkpoint-500
linswap run       --kernel rwkv7                     # everything, results in outputs/eval/rwkv7/
linswap export    --ckpt outputs/kda/sft_full/checkpoint-50 --out hf/Qwen3.5-0.8B-KDA-sft
```

(`linswap <stage>` after `pip install -e .`, `linswap <stage>` from a bare checkout.)

SFT data (LongAlign-10k, LongAlpaca-12k, anti-haystack; Qwen chat format,
non-assistant tokens masked, left-truncated) is prepared on first use.  The
recipe is identical for every kernel — bf16, gradient checkpointing, chunked
cross-entropy, micro-batch 1 (or `--batch_size N`, right-padded) × 2 accumulation, 131K training length — and every
knob is a command-line argument.  Checkpoints record their kernel in
`config.json`.  On one 143 GiB GPU a 131K micro-step takes ~10 s (262K: ~36 s,
54 GiB); most SFT examples are far shorter, so a run takes minutes.

## Evaluation

```bash
linswap evaluate --models kda outputs/kda/sft_full/checkpoint-50 gdn \
    --tasks niah_multikey_2,niah_multikey_3,niah_multiquery,vt,cwe,fwe,qa_1,qa_2 \
    --lengths 131072 --samples 100 --name kda-vs-gdn                  # -> outputs/eval/kda-vs-gdn/summary.{csv,md,json}
```

`--models` takes kernel names (the base swap) and/or checkpoint directories
(`label=path` to name a row).  Beyond RULER, `evaluate --nll pg19,wikitext`
adds token-weighted raw-text NLL (binned by position) on held-out corpora,
`linswap lmeval` runs lm-eval-harness (HellaSwag, PIQA, ARC, WinoGrande,
LAMBADA) through the HF model class, and `linswap mqar` probes in-context
multi-query associative recall versus the number of key-value pairs.
`linswap distill --kl_schedule 8192:200,65536:100` distils on packed
long-context sequences.  The stage computes validation loss and drives
RULER's own scripts (vendored under `RULER/`, with a wrapper for swapped
models) for any tasks / lengths / sample counts, then writes one summary table.

## Benchmarks

Backbone Qwen3.5-0.8B, one seed, identical recipe.  Validation loss is the
assistant-token SFT loss on 40 held-out examples (≤131K).  RULER at 131,072
tokens with cached greedy decoding; the easy set uses 100 samples per task, the
hard set 50 (≈ ±7 points).  Full tables and discussion in
[docs/framework.md](docs/framework.md).

**Easy NIAH** (`niah_single_1` / `niah_multikey_1` / `niah_multivalue`)

| model | trainable | val CE | single | multikey | multivalue |
|---|---|---|---|---|---|
| gdn base (exact copy) | – | 1.743 | 100 | 100 | 96.5 |
| gdn2 / kda / rwkv7 base (tiled init) | – | 1.741–1.743 | 100 | 100 | 95.75–96.75 |
| gdn gate-only 100 | 0.59M | 1.472 | 100 | 100 | 96.25 |
| kda gate-only 100 | 7.4M | 1.427 | 100 | 100 | 97.5 |
| rwkv7 gate-only 100 | 14M | 1.415 | 100 | 99 | 99.5 |
| kda_fullgate gate-only 100 | 38M | 1.413 | 100 | 98 | 99.0 |
| gdn2 gate-only 100 | 113M | 1.378 | 100 | 99 | 96.5 |
| gdn / gdn2 / kda / rwkv7 full 50 | all | 1.388–1.389 | 100 | 100 | 99.0–99.5 |
| mamba2 base (inexact) | – | 6.854 | 0 | 0 | 0 |
| mamba2 SFT only, 500 steps | all | 1.709 | 95 | 66 | 55 |
| mamba2 distill only (500) | all | 1.729 | 100 | 72 | 53.25 |
| mamba2 distill → full 50 | all | 1.491 | 100 | 77 | 73.5 |
| mamba1 distill → full 50 | all | 2.068 | 66 | 20 | 15 |
| mamba3 distill → full 50 (32K) | all | 3.679 | – | – | – |
| deltanet base (inexact) | – | 12.845 | 0 | 0 | 0 |
| deltanet SFT only, 500 steps | all | 6.145 | 0 | 0 | 0 |
| deltanet distill → full 50 | all | 2.092 | 0 | 0 | 0 |

**Hard RULER, average over 8 tasks vs context length** (`multikey_2` / `multikey_3` /
`multiquery` / `vt` / `cwe` / `fwe` / `qa_1` / `qa_2`; 50 samples per task, answer prefix
opens the assistant turn as in RULER's chat templates)

| model | 4K | 16K | 64K | 131K |
|---|---|---|---|---|
| gdn-base | 86.5 | 85.7 | 78.6 | 75.0 |
| gdn-full-50 | 85.5 | 81.9 | 74.8 | 70.8 |
| gdn2-full-50 | 85.4 | 82.4 | 74.8 | 71.7 |
| kda-full-50 | 85.3 | 81.2 | 74.9 | 71.2 |
| rwkv7-full-50 | 85.4 | 81.9 | 74.7 | 71.4 |
| kda-gate-100 | 86.3 | 85.8 | 71.8 | 69.8 |
| rwkv7-gate-100 | 87.7 | 83.4 | 70.3 | 68.6 |
| mamba2-distill-sft-50 | 66.5 | 52.8 | 45.0 | 37.9 |

**Hard RULER at 131K, per task** (`noah` = SFT without the anti-haystack data; `lc` = long-context distillation; `mamba2_beta` keeps β-scaled writes)

| model | mk2 | mk3 | mq | vt | cwe | fwe | qa1 | qa2 | avg |
|---|---|---|---|---|---|---|---|---|---|
| gdn-base | 100.0 | 100.0 | 100.0 | 77.2 | 46.0 | 98.7 | 40.0 | 38.0 | 75.0 |
| gdn-full-50 | 98.0 | 100.0 | 100.0 | 79.6 | 9.0 | 98.0 | 42.0 | 40.0 | 70.8 |
| gdn2-full-50 | 98.0 | 100.0 | 100.0 | 80.0 | 9.6 | 98.0 | 44.0 | 44.0 | 71.7 |
| kda-full-50 | 98.0 | 100.0 | 100.0 | 80.4 | 9.6 | 98.0 | 44.0 | 40.0 | 71.2 |
| rwkv7-full-50 | 98.0 | 100.0 | 100.0 | 80.0 | 9.0 | 98.0 | 42.0 | 44.0 | 71.4 |
| kda-gate-100 | 100.0 | 100.0 | 99.5 | 83.6 | 1.4 | 98.0 | 36.0 | 40.0 | 69.8 |
| rwkv7-gate-100 | 100.0 | 100.0 | 99.5 | 82.0 | 0.2 | 95.3 | 34.0 | 38.0 | 68.6 |
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

**Second scale: Qwen3.8-27B** (16 key / 48 value linear heads, gate-only SFT
at 32K, 100 steps; RULER `multikey_2` / `multiquery` / `vt` / `qa_1` at 16K and 64K,
25 samples; `gdn2` cannot be built on grouped value heads, see ‡)

| model | trainable | val CE | mk2 16K/64K | mq 16K/64K | vt 16K/64K | qa1 16K/64K |
|---|---|---|---|---|---|---|
| gdn base (exact copy) | – | 4.77 | 100 / 100 | 100 / 100 | 100 / 100 | 80 / 84 |
| kda gate-only 100 | 81M | 0.94 | 100 / 100 | 100 / 100 | 100 / 100 | 84 / 80 |
| rwkv7 gate-only 100 | 139M | 0.93 | 100 / 100 | 100 / 100 | 100 / 100 | 80 / 76 |

In fp32 the swapped 27B reproduces the Hugging Face model to KL 2.6e-6 (top-1
identical); the base model's high validation loss is the thinking-tuned backbone
in non-thinking mode on this SFT data (the HF model itself scores 4.79), not the
swap.

What the numbers say:

* The exact swaps lose nothing at init, and after the identical short full-SFT
  recipe all four exact kernels tie on every task at every context length (4K to
  131K, within 0.6 average points).  At this budget the backbone dominates; the
  recurrence is invisible.
* Gate-only SFT is where kernels differ on the easy set (a per-channel decay gate
  — KDA, RWKV-7 — is a cheap handle for multi-value retrieval, GDN2's three dense
  gates are not, despite the lowest validation loss); on the hard set KDA and RWKV-7
  gate-only checkpoints tie, keep common-word extraction at short context where
  full SFT loses it, and give the best variable tracking at 131K.
* Inexact swaps ablate the pretrained recurrence.  Mamba-2 (no erase) is brought
  back to the original loss by distillation but is the only kernel whose retrieval
  degrades with length (distractor needles 90 → 4 from 4K to 131K); keeping GDN's
  β-scaled writes (`mamba2_beta`) recovers half the gap (avg 50 vs 38 at 131K), the
  rest is the missing erase.  DeltaNet (no decay) recovers loss and short-context
  retrieval but nothing at 131K.  Distillation beats SFT alone at equal steps
  (DeltaNet 2.46 vs 6.70 val CE after 500 steps), and for Mamba-2 it transfers
  retrieval behaviour that SFT does not even at equal validation loss (avg 30.5 vs
  11.3 at 131K, val CE 1.73 vs 1.71).
* The retrieval-flavoured SFT does not improve the hard tasks: it costs 1 point at
  4K and 4 at 131K, almost all of it common-word extraction (46 → 9 at 131K) — the
  fine-tuning data, not the swap.

## Citation

If you use LinearSwap, please cite the repository:

```bibtex
@misc{linearswap2026,
  title  = {LinearSwap: in-place swapping of linear-attention kernels in pretrained hybrid language models},
  author = {Chen, Yuang},
  year   = {2026},
  url    = {https://github.com/yuang-chen/LinearSwap}
}
```

## Acknowledgements

LinearSwap grew out of the GDN → GDN-2 in-place swap on Qwen3.5-0.8B
([write-up](https://lutet.industries/posts/gdn2-swap/),
[code](https://github.com/lutetjeff/gdn2-in-place)), whose function-preserving
gate tiling, SFT recipe and RULER integration this project generalises; its
experiment log is kept in [docs/gdn2_experiment_log.md](docs/gdn2_experiment_log.md).
The backbone implementation started from Sebastian Raschka's
[Qwen3.5 from-scratch notebook](https://github.com/rasbt/LLMs-from-scratch).
Kernels come from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
(Songlin Yang, Yu Zhang and contributors); evaluation uses NVIDIA's
[RULER](https://github.com/NVIDIA/RULER).  The architectures swapped in are
Gated DeltaNet-2, Kimi Delta Attention (Kimi Linear), RWKV-7, Mamba-2 and DeltaNet,
by their respective authors.
