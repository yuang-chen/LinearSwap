![LinearSwap](docs/logo.png)

**Swap the linear-attention kernel of a pretrained hybrid LLM in place.**

LinearSwap replaces the Gated-DeltaNet layers of a Qwen3-Next / Qwen3.5 / 3.6 / 3.8 backbone with
another linear sequence mixer — RWKV-7, Mamba-2, Kimi Delta Attention, Gated DeltaNet-2, DeltaNet —
keeps everything else, then distils the result back to the original on generic web text and
benchmarks it. `--kernel <name>` is the only thing that changes between experiments. 

[Kernels](#kernels) · [Install](#install) · [Usage](#usage) · [Pipeline](#pipeline) ·
[Benchmarks](#benchmarks) · [Citation](#citation) · [Acknowledgements](#acknowledgements)

## Kernels

| kernel | sequence mixer | init from the pretrained layer | new params | exact |
|---|---|---|---|---|
| `gdn` | Gated DeltaNet | weight copy (control, distillation teacher) | 0.6M | yes |
| `rwkv7` | RWKV-7 generalised delta rule (DPLR) | gates tiled, removal key = key | 14M | yes |
| `kda` | Kimi Delta Attention | decay tiled into the low-rank gate | 7.4M | yes |
| `kda_fullgate` | Kimi Delta Attention | … with a dense decay projection | 38M | yes |
| `gdn2` | Gated DeltaNet-2 | scalar gates tiled into b/w/f | 113M | yes ‡ |
| `mamba2` | Mamba-2 SSD (decay, no erase) | shared weights copied, erase dropped | 0.3M | no |
| `deltanet` | DeltaNet (erase, no decay) | shared weights copied, decay dropped | 0.3M | no |
| `gla` | Gated Linear Attention | shared weights copied, decay MLP at FLA init | 0.9M | no |
| `mamba3` † | Mamba-3 | decay/projections mapped into the fused `in_proj` | 0.05M | no |
| `mamba1` † | Mamba-1 selective SSM | values, gate, conv, `out_proj` copied | 19M | no |
| `gdn_breg` § | GDN + soft-thresholded state | weight copy; `lam` from the environment | 0.6M | at `lam=0` |
| `swa` ¶ | sliding-window softmax, 64 wide + 4 sinks | weight copy; new logit temperature | 16 scalars | no |

An **exact** kernel contains Gated DeltaNet as a special case, so the swapped layer computes the
same function as the original at step 0, verified to bf16 noise; the rest start from whatever maps
and have more to recover. Every kernel then goes through the same distillation. Each target is the
*sequence mixer* of the named architecture dropped into a backbone-compatible block — the backbone's
projections, short convolutions and gated output norm are kept, so e.g. RWKV-7's token shift and
GroupNorm are not used. Counts are for the 0.8B backbone; `docs/framework.md` has every mapping.

<sub>‡ needs as many value heads as key heads; refuses grouped-value-head backbones.
† needs `mamba_ssm` (see Install); Mamba-3 has no cached decode, use `evaluate --no_cache`.
§ wraps an external in-development package, registered only when it imports.
¶ the one mixer that is not linear attention: sliding-window softmax with sinks (arXiv 2608.28444)
over the same projections, with a bounded 68-key state instead of a matrix. Window / sinks / RoPE from
`LINSWAP_SWA_WINDOW` / `_SINKS` / `_ROPE`.</sub>

## Install

Python 3.11, PyTorch ≥ 2.7 with a matching Triton ≥ 3.3; tested on torch 2.9.1 / CUDA 12.8 /
Triton 3.5.1. `flash-linear-attention` must come from git: the published wheels ship only
`fla/layers` and `fla/models`.

```bash
git clone https://github.com/yuang-chen/LinearSwap && cd LinearSwap
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention@8e84ed4"
uv pip install -e ".[eval]"                              # the `linswap` command; [eval] adds RULER's deps
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
python tests/test_kernels.py && python tests/test_hf.py  # kernels match the control, HF round-trip exact
```

Optional, for `mamba1` / `mamba3` (always build without dependency resolution, or it replaces your
torch):

```bash
CUDA_HOME=/usr/local/cuda MAX_JOBS=32 uv pip install --no-deps --no-build-isolation \
    --no-binary causal-conv1d --no-binary mamba-ssm causal-conv1d mamba-ssm
```

On Hopper GPUs with Triton 3.4–3.7, FLA rejects the Triton backward of its gated chunk kernels
(issue #640): add `uv pip install -e ".[hopper]"` (TileLang) to train `gdn` / `gdn2` / `kda`, and
use Triton < 3.4 or ≥ 3.7.1 to train `mamba2` (inference is unaffected). RULER's word list and QA
data are fetched by `RULER/scripts/data/synthetic/json/download_*.{py,sh}`.

## Usage

```python
from linswap import build_model
model = build_model("rwkv7")                                          # exact init from the backbone
model = build_model(ckpt_dir="outputs/rwkv7/distill/checkpoint-16338")  # kernel read from config.json
out = model.generate(input_ids, max_new_tokens=32)                    # greedy, cached decode
```

The architecture is read from the backbone's HF `config.json`; any GDN-based hybrid with the
Qwen3-Next layout loads, including grouped-value-head configurations (every kernel except `gdn2`,
`gla` and `deltanet`). `linswap kernels` lists what is registered.

**Hugging Face.** Swapped models are `PreTrainedModel`s (`LinearSwapForCausalLM`, registered with
the Auto classes on `import linswap`), so they save, load, generate and evaluate like any HF model:

```bash
linswap export --kernel rwkv7 --out hf/Qwen3.5-0.8B-RWKV7
```

```python
import torch, linswap                                   # registers the architecture
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("hf/Qwen3.5-0.8B-RWKV7", dtype=torch.bfloat16).cuda()
```

Checkpoints use Qwen's tensor layout: everything except `model.layers.{i}.linear_attn.*` is
byte-identical to the backbone, so quantisers and converters see "Qwen with a different linear
layer". Batches must be unpadded or right-padded, generation takes equal-length prompts, and
decoding is greedy or sampling (the model is stateful, so no beam search).

**Adding a kernel.** Two routes, one file each in `kernels/`. A stock FLA layer plus an init
recipe, where `register_fla_kernel` copies what every layer shares with GDN and calls your
`init_extra` for the rest (`gla.py`); or a bare FLA op wrapped in `kernels.base.BackboneMixer`,
where you implement `recurrence(hidden_states, q, k, v, state, use_cache)` and the base supplies
the projections, convolutions, cache protocol and gated output norm (`rwkv7.py`, `mamba2.py`). Set
`exact_init=True` only if the init is function preserving — `linswap verify --kernel <name>
--baseline gdn` must then sit at the control's noise level. Tensor layout: `kernels/common.py`.

## Pipeline

**verify → distill → evaluate → lmeval**, one command each; `run` chains all four.

```bash
linswap run --kernel rwkv7            # everything, results in outputs/eval/rwkv7
```

Distillation is three steps on packed generic web text (DCLM, prepared on first use). No
instruction data, no chat template, no supervised fine-tuning:

| step | loss | tokens | length | lr |
|---|---|---|---|---|
| `layer` | swapped layer vs original layer, on the original's own input | 100M | 512 | 1e-3 → 1e-5 |
| `kl` | KL(teacher ‖ student) on next-token distributions | 500M | 512 | 1e-5 |
| `ce` | cross-entropy, no teacher (context extension) | 100M | 16384 | 1e-5 |

Adam(0.9, 0.95), clip 1.0, bf16, about 6 GPU-hours per kernel at 0.8B. The first step trains the
swapped layers, the other two everything. Budgets are tokens (`--kl_tokens 250e6`); every per-step
knob is a flag.

Evaluation runs on the students *and* on the unmodified backbone:

```bash
linswap evaluate --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338 --name rwkv7
linswap lmeval   --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338
python tools/throughput.py --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338
```

* **Long context** — RULER `niah_single_1/2/3` and `niah_multikey_1` at 4K–128K, 50 samples,
  cached greedy decoding, RULER's base prompt template (`--chat_template` switches).
* **Short context** — LAMBADA, ARC-c/e, PIQA, WinoGrande, HellaSwag 0-shot and MMLU 5-shot,
  reported as accuracy and as a relative score (s − r)/(t − r) against a reference row.

## Benchmarks

Backbone Qwen3.5-0.8B, one seed, identical recipe for every row: 700M tokens of
DCLM, no SFT, base-prompt evaluation.  The **control** is the *unswapped*
backbone put through the same three steps — without it the students' gains over
the unmodified backbone would be read as a kernel effect when they are the
recipe.  Full tables and discussion in [docs/framework.md](docs/framework.md).

**Long-context retrieval** (`niah_single_1` / `_2` / `_3` / `niah_multikey_1`, 50 samples)

| model | 4K | 16K | 64K | 128K |
|---|---|---|---|---|
| unmodified backbone | 94 / 68 / 98 / 82 | 98 / 76 / 96 / 86 | 94 / 100 / 94 / 90 | 98 / 86 / 100 / 94 |
| control (`gdn`, same recipe) | 100 / 100 / 94 / 100 | 100 / 100 / 100 / 98 | 100 / 100 / 100 / 96 | 100 / 100 / 100 / 90 |
| `gdn2` (exact init) | 100 / 100 / 94 / 100 | 100 / 100 / 100 / 94 | 100 / 100 / 98 / 90 | 100 / 100 / 96 / 82 |
| `rwkv7` (exact init) | 100 / 100 / 98 / 96 | 100 / 100 / 100 / 96 | 100 / 100 / 100 / 94 | 100 / 100 / 92 / 92 |
| `kda` (exact init) | 100 / 100 / 94 / 96 | 100 / 100 / 98 / 94 | 100 / 100 / 100 / 94 | 100 / 100 / 92 / 88 |
| `kda_fullgate` (exact init) | 100 / 100 / 98 / 98 | 100 / 100 / 100 / 96 | 100 / 100 / 100 / 90 | 100 / 100 / 96 / 88 |
| `gdn_breg` λ=0.01 (all layers) | 100 / 100 / 100 / 94 | 100 / 100 / 100 / 96 | 96 / 98 / 98 / 80 | 94 / 98 / 96 / 84 |
| `gdn_breg` λ=0.01 (layer 0 plain) | 100 / 100 / 100 / 94 | 100 / 100 / 100 / 94 | 90 / 98 / 100 / 86 | 74 / 98 / 96 / 92 |
| `gdn_breg` λ=0.003 (all layers) | 100 / 100 / 100 / 98 | 100 / 100 / 100 / 98 | 100 / 88 / 98 / 46 | 100 / 52 / 76 / 24 |
| `gdn_breg` λ=0.003 (layer 0 plain) | 100 / 100 / 100 / 98 | 100 / 100 / 100 / 98 | 100 / 96 / 98 / 58 | 100 / 86 / 88 / 40 |
| `swa` (window 64 + 4 sinks) | 100 / 100 / 100 / 96 | 100 / 100 / 98 / 84 | 100 / 100 / 92 / 72 | 100 / 82 / 82 / 62 |
| `mamba2` (no erase) | 100 / 100 / 98 / 98 | 100 / 100 / 100 / 96 | 100 / 100 / 98 / 92 | 100 / 98 / 92 / 72 |
| `mamba3` (no erase, `--no_cache`) | 100 / 100 / 94 / 94 | 100 / 100 / 92 / 74 | – | – |
| `mamba1` (SSM init, no erase) | 100 / 100 / 94 / 66 | 100 / 100 / 100 / 60 | 32 / 92 / 82 / 58 | 30 / 12 / 8 / 16 |
| `gla` (per-channel decay, no erase) | 58 / 100 / 96 / 94 | 0 / 100 / 90 / 72 | 0 / 82 / 40 / 42 | 0 / 2 / 0 / 4 |
| `deltanet` (erase, no decay) | 100 / 100 / 90 / 98 | 96 / 100 / 76 / 82 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

**Short context**, accuracy (relative score vs the unmodified backbone in %)

| model | LAMBADA | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | MMLU | rel. avg |
|---|---|---|---|---|---|---|---|---|
| unmodified backbone | 0.437 | 0.374 | 0.611 | 0.693 | 0.583 | 0.496 | 0.504 | 100.0 |
| control (`gdn`) | 0.478 | 0.399 | 0.653 | 0.706 | 0.588 | 0.524 | 0.515 | **110.0** |
| `kda_fullgate` | 0.481 | 0.397 | 0.646 | 0.706 | 0.591 | 0.521 | 0.517 | 110.0 |
| `gdn2` | 0.481 | 0.390 | 0.644 | 0.705 | 0.597 | 0.521 | 0.514 | 109.9 |
| `kda` | 0.476 | 0.393 | 0.641 | 0.701 | 0.596 | 0.522 | 0.514 | 109.4 |
| `gdn_breg` λ=0.003 (all layers) | 0.479 | 0.395 | 0.649 | 0.703 | 0.593 | 0.519 | 0.510 | 109.4 |
| `gdn_breg` λ=0.003 (layer 0 plain) | 0.478 | 0.393 | 0.648 | 0.705 | 0.592 | 0.521 | 0.507 | 109.1 |
| `rwkv7` | 0.479 | 0.391 | 0.642 | 0.705 | 0.590 | 0.521 | 0.517 | 108.7 |
| `gdn_breg` λ=0.01 (layer 0 plain) | 0.469 | 0.385 | 0.633 | 0.711 | 0.598 | 0.518 | 0.501 | 108.1 |
| `gdn_breg` λ=0.01 (all layers) | 0.469 | 0.384 | 0.609 | 0.706 | 0.594 | 0.519 | 0.506 | 106.3 |
| `mamba2` | 0.462 | 0.372 | 0.610 | 0.701 | 0.578 | 0.520 | 0.501 | 101.4 |
| `swa` | 0.453 | 0.372 | 0.617 | 0.702 | 0.595 | 0.503 | 0.456 | 101.0 |
| `gla` | 0.454 | 0.354 | 0.593 | 0.691 | 0.568 | 0.509 | 0.482 | 94.3 |
| `mamba3` | 0.436 | 0.340 | 0.574 | 0.691 | 0.573 | 0.492 | 0.429 | 88.2 |
| `deltanet` | 0.382 | 0.331 | 0.562 | 0.694 | 0.569 | 0.475 | 0.415 | 82.8 |
| `mamba1` | 0.413 | 0.323 | 0.547 | 0.687 | 0.560 | 0.485 | 0.396 | 79.6 |

Mamba-3 has no usable single-token decode kernel here (see the note on `mamba3` above), so it is scored
with `--no_cache` and only at the two shorter lengths.  The `gdn2` / `mamba1` / `swa` / `gdn_breg` rows
are a third batch on the same recipe.  The `kda` / `kda_fullgate` / `mamba3` / `gla` / `deltanet` rows come from a second batch;
its own control reproduced the published 110.0 at 110.1 and its teacher row sat inside RULER's ±7-point
noise, so the rows are read together.

What the numbers say:

- The recipe, not the kernel, is what lifts a swapped model above the original:
  the control gains as much as the students on both suites.  The question a
  swap experiment has to answer is therefore what the swap costs *on top of the
  same training*, which is what the control row makes visible.
- With an exact init the swap is nearly free, on every kernel that has one:
  `kda_fullgate` matches the control at 110.0, `kda` is 0.6 points under it and
  `rwkv7` 1.3, and all three stay within sample noise of it at every needle
  length.  What the target recurrence *is* matters far less than whether the
  pretrained function survives the change of parameterisation.
- KDA's low-rank forget gate costs nothing.  `kda` and `kda_fullgate` differ
  only in whether the per-channel decay is produced through a rank-128
  bottleneck or a dense projection, and they land 0.6 relative points apart —
  inside noise, for 7.4M new parameters against 38M.
- `gdn2` is the fourth exact init to land on the control (109.9 vs 110.0, needles within noise to
  128K), which is the same story as `kda` / `kda_fullgate` / `rwkv7`: what the target recurrence *is*
  matters far less than whether the pretrained function survives the reparameterisation.
- Soft-thresholding the state (`gdn_breg`) trades the two suites against each other, and not
  monotonically in λ.  λ=0.003 is at the control on short context (109.4) but collapses on retrieval
  past 16K (multikey 46 at 64K, 24 at 128K); λ=0.01 gives up 3.7 relative points and holds retrieval
  to 128K (multikey 84 vs the control's 90).  Final validation loss cannot tell them apart (2.939 vs
  2.944).  Leaving layer 0 unthresholded recovers part of the cost on both settings — λ=0.01 goes
  106.3 → 108.1 and λ=0.003's 128K multikey 24 → 40 — but does not change the ordering: the larger
  threshold is still the one that keeps long-context retrieval.
- `swa` keeps short context (101.0) with a 64-token window, so most of what the suite measures is
  local; retrieval is where the bounded state shows, decaying 96 / 84 / 72 / 62 on the multikey
  needle.  Its MMLU is the outlier (0.456, 81.1 relative) — the one task here that most needs
  long-range context.
- `mamba1` is the weakest swap measured (79.6, needles 8–30 at 128K), which is what the registry's
  least GDN-like init predicts: SSM parameters at Mamba's own init, no erase, no decay gate.
- Dropping the delta-rule erase is not free: `mamba2` trails the control by 8.6
  relative points and loses the distractor needle at 128K (72 vs 90).
- The inexact kernels fail *with length*, and the short-context suite does not
  see it coming.  `deltanet` is at the control's level at 4K and scores exactly
  0 on all four needles from 64K on; `gla` decays through 65 / 41 / 1.5 (task
  average) as the context grows.  Both still look respectable on LAMBADA and
  PIQA, which is the argument for keeping a retrieval suite in the protocol:
  a swap can be nearly free at 4K and worthless at 64K.
- Decoding is length-independent for all of them (constant state).  Prefill at
  8K/32K: `gdn` 134K/141K tok/s, `mamba2` 119K/128K, `rwkv7` 79K/76K; decode
  24.5 / 30.5 / 33.3 ms per token.

Speed on one L20X, bf16, batch 1 (decode is length-independent for all three, constant state):

| model | prefill 8K / 32K | decode | peak 8K / 32K |
|---|---|---|---|
| `gdn` | 134K / 141K tok/s | 24.5 ms/token | 2.0 / 3.4 GiB |
| `mamba2` | 119K / 128K | 30.5 | 2.0 / 3.3 |
| `rwkv7` | 79K / 76K | 33.3 | 2.7 / 6.1 |


Full tables, the per-kernel mappings and the approaches that were tried and dropped:
[docs/framework.md](docs/framework.md).

## Citation

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
[code](https://github.com/lutetjeff/gdn2-in-place)), whose function-preserving gate tiling and
RULER integration this project generalises. The backbone implementation started from Sebastian
Raschka's [Qwen3.5 from-scratch notebook](https://github.com/rasbt/LLMs-from-scratch). Kernels
come from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (Songlin
Yang, Yu Zhang and contributors); evaluation uses NVIDIA's
[RULER](https://github.com/NVIDIA/RULER). The architectures swapped in are Gated DeltaNet-2, Kimi
Delta Attention (Kimi Linear), RWKV-7, Mamba-2 and DeltaNet, by their respective authors.

The distillation recipe and the evaluation protocol follow
[RADLADS](https://arxiv.org/abs/2505.03005) (Goldstein et al., *Rapid Attention Distillation to
Linear Attention Decoders at Scale*): its three steps (hidden-state alignment → logit distillation
→ context-length extension) with their token budgets, schedules and optimizer settings,
generic-text distillation data, base-prompt evaluation and the relative score against the teacher.
RADLADS converts softmax attention into linear attention; here the same recipe is applied to
swapping one linear sequence mixer for another inside a hybrid backbone.
