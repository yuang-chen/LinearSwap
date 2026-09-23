![LinearSwap](docs/logo.png)

**Swap the linear-attention kernel of a pretrained hybrid LLM in place.**

LinearSwap replaces the Gated-DeltaNet layers of a Qwen3-Next / Qwen3.5 / 3.6 / 3.8 backbone with
another linear sequence mixer — RWKV-7, Mamba-2, Kimi Delta Attention, Gated DeltaNet-2, DeltaNet —
keeps everything else, then distils the result back to the original on generic web text and
benchmarks it. `--kernel <name>` is the only thing that changes between experiments. 

[Kernels](#kernels) · [Install](#install) · [Usage](#usage) · [Pipeline](#pipeline) ·
[Benchmarks](#benchmarks) · [Citation](#citation) · [Acknowledgements](#acknowledgements)

## Kernels


| kernel         | sequence mixer                            | init from the pretrained layer                    | new params | exact      |
| -------------- | ----------------------------------------- | ------------------------------------------------- | ---------- | ---------- |
| `gdn`          | Gated DeltaNet                            | weight copy (control, distillation teacher)       | 0.6M       | yes        |
| `rwkv7`        | RWKV-7 generalised delta rule (DPLR)      | gates tiled, removal key = key                    | 14M        | yes        |
| `kda`          | Kimi Delta Attention                      | decay tiled into the low-rank gate                | 7.4M       | yes        |
| `kda_fullgate` | Kimi Delta Attention                      | … with a dense decay projection                   | 38M        | yes        |
| `gdn2`         | Gated DeltaNet-2                          | scalar gates tiled into b/w/f                     | 113M       | yes ‡      |
| `mamba2`       | Mamba-2 SSD (decay, no erase)             | shared weights copied, erase dropped              | 0.3M       | no         |
| `deltanet`     | DeltaNet (erase, no decay)                | shared weights copied, decay dropped              | 0.3M       | no         |
| `gla`          | Gated Linear Attention                    | shared weights copied, decay MLP at FLA init      | 0.9M       | no         |
| `swa` ¶        | sliding-window softmax, 64 wide + 4 sinks | weight copy; new logit temperature                | 16 scalars | no         |


An **exact** kernel contains Gated DeltaNet as a special case, so at step 0 the swapped layer
reproduces the original to bf16 noise; the rest have more to recover. Only the *sequence mixer* is
replaced — the backbone's projections, convolutions and gated output norm stay. Counts are for the
0.8B backbone; `docs/framework.md` has the per-kernel mappings.

<sub>‡ needs as many value heads as key heads. ¶ not linear attention: sliding-window softmax with
sinks (arXiv 2608.28444) over the same projections, bounded 68-key state; window / sinks / RoPE from
`LINSWAP_SWA_WINDOW` / `_SINKS` / `_ROPE`.  `linswap kernels` lists everything registered, including
kernels with no results here (`mamba1`, `mamba3`, `gdn_breg`).</sub>

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

Optional, for the `mamba1` / `mamba3` kernels (always build without dependency resolution, or it
replaces your torch):

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
`exact_init=True` only if the init is function preserving — `linswap verify --kernel <name> --baseline gdn` must then sit at the control's noise level. Tensor layout: `kernels/common.py`.

## Pipeline

**verify → distill → evaluate → lmeval**, one command each; `run` chains all four.

```bash
linswap run --kernel rwkv7            # everything, results in outputs/eval/rwkv7
```

Distillation is three steps on packed generic web text (DCLM, prepared on first use). No
instruction data, no chat template, no supervised fine-tuning:


| step    | loss                                                         | tokens | length | lr          |
| ------- | ------------------------------------------------------------ | ------ | ------ | ----------- |
| `layer` | swapped layer vs original layer, on the original's own input | 100M   | 512    | 1e-3 → 1e-5 |
| `kl`    | KL(teacher ‖ student) on next-token distributions            | 500M   | 512    | 1e-5        |
| `ce`    | cross-entropy, no teacher (context extension)                | 100M   | 16384  | 1e-5        |


Adam(0.9, 0.95), clip 1.0, bf16, about 6 GPU-hours per kernel at 0.8B. The first step trains the
swapped layers, the other two everything. Budgets are tokens (`--kl_tokens 250e6`); every per-step
knob is a flag.

Evaluation runs on the students *and* on the unmodified backbone:

```bash
linswap evaluate --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338 --name rwkv7
linswap lmeval   --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338
python tools/throughput.py --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338
```

- **Long context** — RULER `niah_single_1/2/3` and `niah_multikey_1` at 4K–128K, 500 samples,
cached greedy decoding, RULER's base prompt template (`--chat_template` switches).
- **Short context** — LAMBADA, ARC-c/e, PIQA, WinoGrande, HellaSwag 0-shot and MMLU 5-shot,
reported as accuracy and as a relative score (s − r)/(t − r) against a reference row.



## Benchmarks

Backbone Qwen3.5-0.8B, one seed, identical recipe for every row: 700M tokens of
DCLM, no SFT, base-prompt evaluation.  The **control** is the *unswapped*
backbone put through the same three steps — without it the students' gains over
the unmodified backbone would be read as a kernel effect when they are the
recipe.  Full tables and discussion in [docs/framework.md](docs/framework.md).

**Long-context retrieval** (`niah_single_1` / `_2` / `_3` / `niah_multikey_1`, 500 samples)

| model | 4K | 16K | 64K | 128K |
|---|---|---|---|---|
| unmodified backbone | 96.4 / 65.0 / 97.8 / 79.4 | 98.4 / 76.2 / 90.6 / 81.6 | 96.4 / 98.4 / 94.6 / 91.8 | 99.2 / 91.6 / 96.6 / 91.0 |
| control (`gdn`, same recipe) | 100 / 100 / 97.0 / 99.8 | 100 / 100 / 100 / 98.4 | 100 / 100 / 99.8 / 97.2 | 100 / 100 / 99.8 / 96.2 |
| `kda_fullgate` (exact init) | 100 / 100 / 95.0 / 99.6 | 100 / 100 / 100 / 97.6 | 100 / 100 / 100 / 95.4 | 100 / 100 / 99.4 / 92.4 |
| `gdn2` (exact init) | 100 / 100 / 93.6 / 100 | 100 / 100 / 99.0 / 97.0 | 100 / 100 / 98.8 / 92.8 | 100 / 99.2 / 96.6 / 86.8 |
| `rwkv7` (exact init) | 100 / 100 / 91.6 / 99.6 | 100 / 100 / 100 / 96.4 | 100 / 100 / 100 / 94.8 | 100 / 100 / 98.6 / 86.6 |
| `kda` (exact init) | 100 / 100 / 94.8 / 99.6 | 100 / 100 / 99.8 / 96.6 | 100 / 100 / 100 / 95.0 | 100 / 99.8 / 98.8 / 85.6 |
| `mamba2` (no erase) | 100 / 100 / 99.4 / 98.8 | 100 / 100 / 99.8 / 93.6 | 100 / 98.4 / 99.4 / 84.6 | 100 / 95.0 / 94.6 / 70.0 |
| `swa` (window 64 + 4 sinks) | 100 / 100 / 99.8 / 97.4 | 100 / 99.2 / 97.6 / 79.6 | 100 / 98.0 / 95.4 / 67.6 | 100 / 81.4 / 89.4 / 56.0 |
| `gla` (per-channel decay, no erase) | 53.2 / 100 / 99.0 / 97.8 | 1.4 / 100 / 85.4 / 72.2 | 0.0 / 77.8 / 51.0 / 44.0 | 0.0 / 1.2 / 1.2 / 3.8 |
| `deltanet` (erase, no decay) | 100 / 99.8 / 96.6 / 98.2 | 98.4 / 100 / 82.6 / 85.2 | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 |

**Short context**, accuracy (relative score vs the unmodified backbone in %)

| model | LAMBADA | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | MMLU | rel. avg |
|---|---|---|---|---|---|---|---|---|
| unmodified backbone | 0.437 | 0.374 | 0.611 | 0.693 | 0.583 | 0.496 | 0.504 | 100.0 |
| control (`gdn`) | 0.478 | 0.399 | 0.653 | 0.706 | 0.588 | 0.524 | 0.515 | **110.0** |
| `kda_fullgate` | 0.481 | 0.397 | 0.646 | 0.706 | 0.591 | 0.521 | 0.517 | 110.0 |
| `gdn2` | 0.481 | 0.390 | 0.644 | 0.705 | 0.597 | 0.521 | 0.514 | 109.9 |
| `kda` | 0.476 | 0.393 | 0.641 | 0.701 | 0.596 | 0.522 | 0.514 | 109.4 |
| `rwkv7` | 0.479 | 0.391 | 0.642 | 0.705 | 0.590 | 0.521 | 0.517 | 108.7 |
| `mamba2` | 0.462 | 0.372 | 0.610 | 0.701 | 0.578 | 0.520 | 0.501 | 101.4 |
| `swa` | 0.453 | 0.372 | 0.617 | 0.702 | 0.595 | 0.503 | 0.456 | 101.0 |
| `gla` | 0.454 | 0.354 | 0.593 | 0.691 | 0.568 | 0.509 | 0.482 | 94.3 |
| `deltanet` | 0.382 | 0.331 | 0.562 | 0.694 | 0.569 | 0.475 | 0.415 | 82.8 |

The short-context suite scores every task in full.  The `kda` / `kda_fullgate` / `gla` / `deltanet`
rows come from a second batch, run months later on freshly tokenised DCLM; its own control reproduced
the published 110.0 at 110.1, which is what licenses one table.

What the numbers say:

- **The recipe, not the kernel, is what lifts a swapped model above the original.**  The control gains
  as much as the students on both suites, so the question a swap has to answer is what it costs *on top
  of the same training*.
- **An exact init is nearly free, except at 128K.**  All four land within 1.3 relative points of the
  control on short context and hold every needle to 64K; on the 128K distractor needle they trail it by
  4 to 11 points (control 96.2, then `kda_fullgate` 92.4, `gdn2` 86.8, `rwkv7` 86.6, `kda` 85.6).
- **KDA's low-rank forget gate costs retrieval.**  `kda` and `kda_fullgate` differ only in whether the
  decay factors through a rank-128 bottleneck or a dense projection: 0.6 relative points apart on short
  context, but the dense gate leads at every needle length and by 6.8 points at 128K.
- **Dropping the delta-rule erase costs.**  `mamba2` trails the control by 8.6 relative points and
  loses the distractor needle at 128K (70.0 vs 96.2).
- **A 64-token window holds short context, not retrieval.**  `swa` scores 101.0 relative but decays
  97.4 / 79.6 / 67.6 / 56.0 on the multikey needle, and drops to 0.456 on MMLU (81.1) — the task here
  that most needs long context.
- **Inexact swaps fail with length, and the short suite cannot see it.**  `deltanet` matches the control
  at 4K and then scores 0 on all four needles from 64K on; `gla` decays through 87.5 / 64.8 / 43.2 / 1.6
  (task average).  Both still look respectable on LAMBADA and PIQA: a swap can be free at 4K and
  worthless at 64K.

Speed on one L20X, bf16, batch 1 (decode is length-independent for all three, constant state):


| model    | prefill 8K / 32K  | decode        | peak 8K / 32K |
| -------- | ----------------- | ------------- | ------------- |
| `gdn`    | 134K / 141K tok/s | 24.5 ms/token | 2.0 / 3.4 GiB |
| `mamba2` | 119K / 128K       | 30.5          | 2.0 / 3.3     |
| `rwkv7`  | 79K / 76K         | 33.3          | 2.7 / 6.1     |


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
