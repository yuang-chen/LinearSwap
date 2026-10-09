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
| `gdn2`         | Gated DeltaNet-2                          | scalar gates tiled into b/w/f                     | 113M       | yes ‡      |
| `mamba2`       | Mamba-2 SSD (decay, no erase)             | shared weights copied, erase dropped              | 0.3M       | no         |
| `deltanet`     | DeltaNet (erase, no decay)                | shared weights copied, decay dropped              | 0.3M       | no         |
| `gla`          | Gated Linear Attention                    | shared weights copied, decay fitted per head      | 0.9M       | no         |
| `swa` ¶        | sliding-window softmax, 64 wide + 4 sinks | weight copy; new logit temperature                | 16 scalars | no         |


An **exact** kernel contains Gated DeltaNet as a special case, so at step 0 the swapped layer
reproduces the original to bf16 noise; the rest have more to recover. Only the *sequence mixer* is
replaced — the backbone's projections, convolutions and gated output norm stay. Counts are for the
0.8B backbone; `docs/kernels.md` has the per-kernel mappings.

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
layer". Batches may be right-padded (loss / logits) or left-padded (also HF `generate`, every kernel
but `swa`), and decoding is greedy or sampling (the model is stateful, so no beam search).

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


The `layer` step measures its own loss on one batch before training and **skips itself** when that loss
is below `--layer_skip_below` (1e-4): a swap that already reproduces the pretrained layer to bf16 noise
has nothing to align, and training it there degrades the model instead of improving it.  In practice
`gdn`, `gdn2`, `kda` and `rwkv7` skip it (initial loss 0 to 8e-7) while `gla`, `mamba2`,
`swa` and `deltanet` run it (1.6e-2 to 1.1e-1).

Adam(0.9, 0.95), clip 1.0, bf16, about 6 GPU-hours per kernel at 0.8B. The first step trains the
swapped layers, the other two everything. Budgets are tokens (`--kl_tokens 250e6`); every per-step
knob is a flag.

Each run writes `outputs/<kernel>/distill/train_log.jsonl` and, with the `tensorboard` extra
(`uv pip install -e ".[tensorboard]"`), a TensorBoard run next to it:
`tensorboard --logdir outputs` overlays the kernels' loss, grad norm (total and per parameter group),
lr, validation KL(teacher ‖ student) and the swapped layers' drift from their init.

Evaluation runs on the students *and* on the unmodified backbone:

```bash
linswap evaluate --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338 --name rwkv7
linswap lmeval   --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338
python tools/throughput.py --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338
```

- **Long context** — RULER `niah_single_1/2/3` and `niah_multikey_1` (retrieval), `vt` (multi-hop
variable tracking), `cwe` / `fwe` (common / frequent words extraction) at 4K–128K, 500 samples,
cached greedy decoding, RULER's base prompt template (`--chat_template` switches).
- **Short context** — LAMBADA, ARC-c/e, PIQA, WinoGrande, HellaSwag 0-shot, MMLU 5-shot and
IFEval (chat template, generative), reported as accuracy and as a relative score (s − r)/(t − r) against a reference row.



## Benchmarks

Backbone Qwen3.5-0.8B, one seed, identical recipe for every row: 700M tokens of
DCLM, no SFT, base-prompt evaluation.  The **control** is the *unswapped*
backbone put through the same three steps — without it the students' gains over
the unmodified backbone would be read as a kernel effect when they are the
recipe.  Full tables and discussion in [docs/results.md](docs/results.md).

**Long-context retrieval** (`niah_single_1` / `_2` / `_3` / `niah_multikey_1`, 500 samples)

| model | 4K | 16K | 64K | 128K |
|---|---|---|---|---|
| unmodified backbone | 96.4 / 65.0 / 97.8 / 79.4 | 98.4 / 76.2 / 90.6 / 81.6 | 96.4 / 98.4 / 94.6 / 91.8 | 99.2 / 91.6 / 96.6 / 91.0 |
| control (`gdn`, same recipe) | 100 / 100 / 98.6 / 99.6 | 100 / 100 / 100 / 98.4 | 100 / 100 / 100 / 97.2 | 100 / 100 / 100 / 96.8 |
| `gdn2` (exact init) | 100 / 100 / 94.2 / 100 | 100 / 100 / 100 / 98.4 | 100 / 100 / 100 / 97.0 | 100 / 100 / 100 / 96.4 |
| `rwkv7` (exact init) | 100 / 100 / 96.0 / 99.6 | 100 / 100 / 100 / 98.4 | 100 / 100 / 100 / 97.2 | 100 / 100 / 100 / 96.2 |
| `kda` (exact init) | 100 / 100 / 99.2 / 99.6 | 100 / 100 / 100 / 98.4 | 100 / 100 / 100 / 97.0 | 100 / 100 / 100 / 96.4 |
| `mamba2` (no erase) | 100 / 100 / 99.4 / 98.8 | 100 / 100 / 99.8 / 93.6 | 100 / 98.4 / 99.4 / 84.6 | 100 / 95.0 / 94.6 / 70.0 |
| `swa` (window 64 + 4 sinks) | 100 / 100 / 99.8 / 97.4 | 100 / 99.2 / 97.6 / 79.6 | 100 / 98.0 / 95.4 / 67.6 | 100 / 81.4 / 89.4 / 56.0 |
| `gla` (per-channel decay, no erase) | 97.4 / 100 / 99.6 / 99.2 | 52.8 / 100 / 99.6 / 89.0 | 29.0 / 96.2 / 96.8 / 82.0 | 32.0 / 94.4 / 90.0 / 58.8 |
| `deltanet` (erase, no decay) | 100 / 99.8 / 96.6 / 98.2 | 98.4 / 100 / 82.6 / 85.2 | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 |

**Short context**, accuracy (relative score vs the unmodified backbone in %)

| model | LAMBADA | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | MMLU | rel. avg |
|---|---|---|---|---|---|---|---|---|
| unmodified backbone | 0.437 | 0.374 | 0.611 | 0.693 | 0.583 | 0.496 | 0.504 | 100.0 |
| control (`gdn`) | 0.476 | 0.402 | 0.649 | 0.704 | 0.589 | 0.525 | 0.515 | 110.1 |
| `gdn2` | 0.479 | 0.398 | 0.652 | 0.706 | 0.590 | 0.524 | 0.516 | 110.2 |
| `kda` | 0.478 | 0.399 | 0.651 | 0.705 | 0.590 | 0.525 | 0.514 | 110.2 |
| `rwkv7` | 0.480 | 0.399 | 0.655 | 0.704 | 0.590 | 0.525 | 0.513 | 110.3 |
| `mamba2` | 0.462 | 0.372 | 0.610 | 0.701 | 0.578 | 0.520 | 0.501 | 101.4 |
| `swa` | 0.453 | 0.372 | 0.617 | 0.702 | 0.595 | 0.503 | 0.456 | 101.0 |
| `gla` | 0.463 | 0.362 | 0.612 | 0.701 | 0.579 | 0.514 | 0.484 | 99.2 |
| `deltanet` | 0.382 | 0.331 | 0.562 | 0.694 | 0.569 | 0.475 | 0.415 | 82.8 |

The short-context suite scores every task in full.  The `kda` / `gla` / `deltanet`
rows come from a second batch, run months later on freshly tokenised DCLM; its own control reproduced
the published 110.0 at 110.1, which is what licenses one table.

What the numbers say:

- **The recipe, not the kernel, is what lifts a swapped model above the original.**  The control gains
  as much as the students on both suites, so the question a swap has to answer is what it costs *on top
  of the same training*.
- **An exact init is free.**  All three exact-init kernels land on the control on both suites: relative
  average 110.2–110.3 against 110.1, and the 128K distractor needle 96.2–96.4 against 96.8.  What the
  target recurrence *is* matters far less than whether the pretrained function survives the change of
  parameterisation.
- **A richer gate buys nothing.**  `gdn2` adds 113M parameters of separate erase and write gates to
  `kda`'s 7.4M low-rank per-channel decay: the two finish within 0.2 relative points of each other and
  of the control, and their needle task averages match the control to 0.1 from 16K on (`gdn2` trails it
  by 1.0 at 4K, on `niah_single_3`).
- **The exact kernels used to look costly at 128K**, because the `layer` step was damaging them.  It is
  skipped for an exact init now (see [Pipeline](#pipeline)); under the old recipe the same three kernels
  scored 85.6–86.8 at 128K.  The deficit was an artifact of the step, not a property of the kernels.
- **Dropping the delta-rule erase costs.**  `mamba2` trails the control by 8.6 relative points and
  loses the distractor needle at 128K (70.0 vs 96.2).
- **A 64-token window holds short context, not retrieval.**  `swa` scores 101.0 relative but decays
  97.4 / 79.6 / 67.6 / 56.0 on the multikey needle, and drops to 0.456 on MMLU (81.1) — the task here
  that most needs long context.
- **Inexact swaps fail with length, and the short suite cannot see it.**  `deltanet` matches the control
  at 4K and then scores 0 on all four needles from 64K on while still looking respectable on LAMBADA
  and PIQA: a swap can be free at 4K and worthless at 64K.  `gla` decays through 99.1 / 85.4 / 76.0 /
  68.8 (task average).
- **The init of the inexact part matters.**  `gla` with FLA's random gate init scored 87.5 / 64.8 / 43.2 /
  1.6 on the same recipe; fitting GDN's per-head decay into the gate at init is what lifts it, and the
  one task it still fails out of order (`niah_single_1`: 97.4 / 52.8 / 29.0 / 32.0, the repeated-noise
  haystack) is where the fit loosens — the mechanism is described in `kernels/gla.py`.  Short context
  moves from 94.3 to 99.2 relative.

Speed on one L20X, bf16, batch 1 (decode is length-independent for all three, constant state):


| model    | prefill 8K / 32K  | decode        | peak 8K / 32K |
| -------- | ----------------- | ------------- | ------------- |
| `gdn`    | 134K / 141K tok/s | 24.5 ms/token | 2.0 / 3.4 GiB |
| `mamba2` | 119K / 128K       | 30.5          | 2.0 / 3.3     |
| `rwkv7`  | 79K / 76K         | 33.3          | 2.7 / 6.1     |


Full tables and the approaches that were tried and dropped: [docs/results.md](docs/results.md).
Per-kernel mappings: [docs/kernels.md](docs/kernels.md).

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
