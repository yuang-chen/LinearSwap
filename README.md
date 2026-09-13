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
| `gdn2` | Gated DeltaNet-2 | scalar beta/decay tiled into b/w/f gates | 113M | yes |
| `kda` | Kimi Delta Attention | scalar decay tiled into the low-rank per-channel gate | 7.4M | yes |
| `kda_fullgate` | Kimi Delta Attention | … with a dense decay projection | 38M | yes |
| `rwkv7` | RWKV-7-style DPLR generalised delta rule | decay/beta tiled, removal key = key | 14M | yes |
| `mamba2` | Mamba-2-style SSD (scalar decay, no erase) | shared weights copied, erase dropped | 0.30M | no |
| `deltanet` | DeltaNet (erase, no decay) | shared weights copied, decay dropped | 0.29M | no |
| `gla` | Gated Linear Attention (per-channel decay, no erase) | shared weights copied, decay MLP at FLA init | 0.92M | no |

"Exact" kernels reproduce the pretrained model at initialisation and go straight
to SFT; the others are distilled first (`linswap.py distill`).  Parameter counts
are for the 0.8B backbone.  Every target is the *recurrence* of the named
architecture inside a backbone-compatible block (the backbone's projections,
short convolutions and gated output norm are kept; e.g. RWKV-7's token shift and
GroupNorm are not used) — see [docs/framework.md](docs/framework.md) for each
mapping and why FLA's native RWKV-7 / Mamba-2 layers cannot express the
pretrained weights.  Mamba-1 and Mamba-3 are not included: their kernels exist
only in `mamba_ssm`, which needs a newer Triton than the pinned torch allows.

## Installation

Requirements: Python 3.11, PyTorch 2.6 (CUDA 12.4), Triton 3.2,
`flash-linear-attention` 0.6, `transformers` ≥ 5.16.

```bash
git clone https://github.com/yuang-chen/LinearSwap && cd LinearSwap
uv venv .venv && source .venv/bin/activate
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[eval]"                              # linswap + the `linswap` command; [eval] adds RULER's deps
uv pip install --no-build-isolation causal-conv1d       # optional: fast short-conv path
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
python tests/test_kernels.py && python tests/test_hf.py  # kernels match the control; HF round-trip is exact
```

Do **not** install `mamba_ssm` into this environment (see
[docs/framework.md](docs/framework.md#mamba-1-and-mamba-3-not-available-here)).
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
hybrid with the Qwen3-Next layer layout loads.  `linswap kernels` lists the
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
safetensors, tokenizer, model card).  Checkpoints use **Qwen's tensor layout**:
`model.embed_tokens`, `model.layers.{i}.{self_attn,mlp,input_layernorm,post_attention_layernorm}`,
`model.norm`, `lm_head` are byte-identical to the backbone's tensors (194 of 194
for the `gdn` kernel), and only `model.layers.{i}.linear_attn.*` differs per
kernel — so quantisers, converters and diff tools see "Qwen with a different
linear layer".  Native `model.pt` checkpoints use the same keys (older
`trf_blocks.*` checkpoints are converted on load).  Current limits: batch size 1 without
padding, greedy or sampling decoding (the model is stateful, so no beam search).

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
python linswap.py verify    --kernel kda --baseline gdn        # layer / logits / layer-wise / cache / generation vs HF
python linswap.py distill   --kernel mamba2                    # inexact kernels: layer alignment (200) + KL (300) @8K
python linswap.py posttrain --kernel kda                       # gate-only (100 steps, 2e-4) and full SFT (50 steps, 1e-5)
python linswap.py posttrain --kernel mamba2 --modes full --init_ckpt outputs/mamba2/distill/checkpoint-500
python linswap.py run       --kernel rwkv7                     # everything, results in outputs/eval/rwkv7/
python linswap.py export    --ckpt outputs/kda/sft_full/checkpoint-50 --out hf/Qwen3.5-0.8B-KDA-sft
```

(`linswap <stage>` after `pip install -e .`, `python linswap.py <stage>` from a bare checkout.)

SFT data (LongAlign-10k, LongAlpaca-12k, anti-haystack; Qwen chat format,
non-assistant tokens masked, left-truncated) is prepared on first use.  The
recipe is identical for every kernel — bf16, gradient checkpointing, chunked
cross-entropy, micro-batch 1 × 2 accumulation, 131K training length — and every
knob is a command-line argument.  Checkpoints record their kernel in
`config.json`.  On one 143 GiB GPU a 131K micro-step takes ~10 s (262K: ~36 s,
54 GiB); most SFT examples are far shorter, so a run takes minutes.

## Evaluation

```bash
python linswap.py evaluate --models kda outputs/kda/sft_full/checkpoint-50 gdn \
    --tasks niah_multikey_2,niah_multikey_3,niah_multiquery,vt,cwe,fwe,qa_1,qa_2 \
    --lengths 131072 --samples 100 --name kda-vs-gdn                  # -> outputs/eval/kda-vs-gdn/summary.{csv,md,json}
```

`--models` takes kernel names (the base swap) and/or checkpoint directories
(`label=path` to name a row).  The stage computes validation loss and drives
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
| deltanet base (inexact) | – | 12.845 | 0 | 0 | 0 |
| deltanet SFT only, 500 steps | all | 6.145 | 0 | 0 | 0 |
| deltanet distill → full 50 | all | 2.092 | 0 | 0 | 0 |

**Hard RULER** (`multikey_2` / `multikey_3` / `multiquery` / `vt` / `cwe` / `fwe` / `qa_1` / `qa_2`, average)

| model | mk2 | mk3 | mq | vt | cwe | fwe | qa1 | qa2 | avg |
|---|---|---|---|---|---|---|---|---|---|
| gdn base | 100 | 98 | 100 | 0 | 36.4 | 87.3 | 36 | 36 | 61.7 |
| gdn full 50 | 96 | 94 | 100 | 19.2 | 2.2 | 97.3 | 42 | 46 | 62.1 |
| gdn2 full 50 | 96 | 94 | 100 | 19.2 | 3.0 | 97.3 | 40 | 44 | 61.7 |
| kda full 50 | 96 | 94 | 100 | 19.2 | 3.2 | 98.0 | 44 | 46 | 62.5 |
| rwkv7 full 50 | 96 | 94 | 100 | 19.2 | 2.2 | 96.7 | 42 | 42 | 61.5 |
| kda gate-only 100 | 100 | 96 | 100 | 5.6 | 3.0 | 92.7 | 40 | 34 | 58.9 |
| rwkv7 gate-only 100 | 100 | 92 | 100 | 4.4 | 0.4 | 90.7 | 32 | 34 | 56.7 |
| mamba2 SFT only, 50 steps | 0 | 0 | 0 | 0 | 0 | 7.3 | 2 | 0 | 1.2 |
| mamba2 SFT only, 500 steps | 12 | 2 | 6.5 | 0 | 1.0 | 30.7 | 10 | 0 | 7.8 |
| mamba2 distill only (500) | 52 | 16 | 49 | 0.4 | 0.4 | 6.0 | 10 | 24 | 19.7 |
| mamba2 distill → full 50 | 62 | 6 | 83.5 | 20.4 | 0.4 | 59.3 | 26 | 32 | 36.2 |
| deltanet (any variant) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0–4 | ~0 |

What the numbers say:

* The exact swaps lose nothing at init, and after the identical short full-SFT
  recipe all four exact kernels tie on every task, easy and hard.  At this budget
  the backbone dominates; the recurrence is invisible.
* Gate-only SFT is where kernels differ: a per-channel decay gate (KDA, RWKV-7)
  is a cheap handle for multi-value retrieval, GDN2's three dense gates are not,
  despite the lowest validation loss.
* Inexact swaps ablate the pretrained recurrence.  Mamba-2 (no erase) is brought
  back to the original loss by distillation but plateaus on distractor-heavy
  multi-key retrieval; DeltaNet (no decay) recovers loss and short-context
  retrieval but nothing at 131K.  Distillation beats SFT alone at equal steps
  (DeltaNet 2.46 vs 6.70 val CE after 500 steps), and for Mamba-2 it transfers
  retrieval behaviour that SFT does not even at equal validation loss (multikey_2
  52 vs 12 at val CE 1.73 vs 1.71).
* The retrieval-flavoured SFT lifts variable tracking (0 → 19) but destroys
  common-word extraction (36 → ~3) for every kernel — the fine-tuning data, not
  the swap.

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
