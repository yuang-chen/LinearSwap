![LinearSwap](docs/logo.png)

**Swap the linear-attention kernel of a pretrained hybrid LLM in place.**

LinearSwap replaces the Gated-DeltaNet layers of a Qwen3-Next / Qwen3.5 / 3.6 / 3.8 backbone with
another linear sequence mixer — RWKV-7, Mamba-2, Kimi Delta Attention, Gated DeltaNet-2, DeltaNet —
keeps everything else, then distils the result back to the original on generic web text and
benchmarks it. `--kernel <name>` is the only thing that changes between experiments. 

[Kernels](#kernels) · [Install](#install) · [Usage](#usage) · [Pipeline](#pipeline) ·
[Benchmarks](#benchmarks) · [Citation](#citation) · [Acknowledgements](#acknowledgements)

## Kernels


| kernel         | sequence mixer                            | init from the pretrained layer                    | gate params | exact      |
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

*Gate params* is the size of the kernel's own gate / recurrence group (`new_param_names`), copied gates
included: `gdn`'s 0.6M are the pretrained gates themselves, not additions. The net change of the whole
model against `gdn` (752.4M) is `gdn2` +112.7M, `rwkv7` +13.6M, `kda` +6.8M, `gla` +0.3M, `mamba2` and
`deltanet` −0.3M, `swa` −0.6M.

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

**Scope: the language backbone.** Qwen3.5-0.8B is a vision-language checkpoint
(`Qwen3_5ForConditionalGeneration`). LinearSwap loads its language model only (`text_config`,
`model.language_model.*`), takes `input_ids`, and trains and evaluates on text; the vision tower and
merger are not part of a swapped model or of its export, and no benchmark here measures image or video
inputs. Read every "reproduces the original" below as a statement about the text path.

To run a swap, or a checkpoint distilled on text, on images, `linswap.upstream` puts the swapped mixers
into the model `transformers` itself loads — vision tower, merger, multimodal RoPE, processor and
`generate` stay upstream — instead of rebuilding the backbone:

```python
from linswap.upstream import build_upstream_model            # needs the `vision` extra (torchvision)
model = build_upstream_model("kda")                          # Qwen3_5ForConditionalGeneration, mixers swapped
model = build_upstream_model(ckpt_dir="outputs/kda/distill/checkpoint-16338")   # + distilled language weights
out = model.generate(**processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                     add_generation_prompt=True, return_tensors="pt").to("cuda"))
```

At step 0 an exact kernel there matches the native model on a processed image as closely as the `gdn`
control does (`tests/test_upstream.py`), and on text its logits are bit-identical to `build_model`'s.
It is an in-memory, inference-only path (greedy / sampling); what text-only distillation does to image
tasks is not benchmarked yet.

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

`verify` is a gate, not only a printout: with `--baseline gdn` an `exact_init` kernel must keep every
layer and logit error within `--tolerance` (3×) of the baseline's own noise, measured in the same run,
or the command exits non-zero and `run` stops before training (`--json` saves the measurements).

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

`step` in the log and in `checkpoint-N` is the plan's nominal step — a skipped `layer` step keeps its
range so the numbering does not depend on the skip. The work actually done is logged next to it
(`updates`, `tokens`, `teacher_tokens`) and saved in each checkpoint's `train_state.json`; the default
recipe with the `layer` step skipped is 10,235 updates on 599M tokens. (Logs written before November
2026 carry a `tokens` field that multiplied the global step by the current step's batch; ignore it.)
`manifest.json` records the code and package revisions, the base model, the corpus and the plan.
`--init_from` a checkpoint written inside a step continues it with its optimizer state, on the same data.

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
cached greedy decoding in left-padded batches (`--batch_size`), RULER's base prompt template
(`--chat_template` switches).
- **Short context** — LAMBADA, ARC-c/e, PIQA, WinoGrande, HellaSwag 0-shot, MMLU 5-shot and
IFEval (chat template, generative), reported as accuracy and as a relative score (s − r)/(t − r) against a reference row.



## Benchmarks

Backbone Qwen3.5-0.8B, one seed, RULER base prompts, 500 samples per cell.  `gdn` is the **control**: the
*unswapped* backbone put through the same training.  The control and the exact-init kernels use the recipe
with the `layer` step skipped; the inexact kernels use the full recipe.

**These tables are selected, not representative.**  `vt` and `cwe` are the only tasks in the 500-sample runs
where the exact-init kernels differ by 5 points or more at two or more lengths.  On the needle tasks, `fwe`
and the short-context suite they sit closer together.  All the tables are in
[docs/results.md](docs/results.md).

**`vt`** (variable tracking)

| model | 4K | 16K | 64K | 128K |
|---|---|---|---|---|
| control (`gdn`) | 90.9 | 70.8 | 50.6 | 68.6 |
| `gdn2` (exact init) | 91.8 | 71.0 | 50.6 | 67.6 |
| `kda` (exact init) | 90.0 | 71.6 | 51.4 | 69.5 |
| `rwkv7` (exact init) | 89.8 | 65.0 | 45.5 | 66.4 |
| *spread, control + exact* | *1.9* | *6.6* | *5.9* | *3.2* |
| `mamba2` | 78.2 | 36.8 | 19.8 | 19.6 |
| `swa` | 37.6 | 23.7 | 28.6 | 7.2 |
| `gla` | 38.8 | 1.6 | 0.2 | 0.2 |
| `deltanet` | 51.4 | 18.7 | 0.0 | 0.0 |

**`cwe`** (common words extraction)

| model | 4K | 16K | 64K | 128K |
|---|---|---|---|---|
| control (`gdn`) | 62.5 | 68.6 | 30.7 | 8.5 |
| `gdn2` (exact init) | 63.9 | 70.6 | 26.0 | 5.0 |
| `kda` (exact init) | 62.5 | 69.2 | 31.4 | 11.2 |
| `rwkv7` (exact init) | 60.6 | 68.7 | 27.7 | 3.9 |
| *spread, control + exact* | *3.3* | *2.0* | *5.3* | *7.4* |
| `mamba2` | 38.0 | 4.0 | 0.8 | 0.4 |
| `swa` | 37.1 | 1.4 | 0.6 | 0.4 |
| `gla` | 24.7 | 7.3 | 0.3 | 0.1 |
| `deltanet` | 19.9 | 3.5 | 0.0 | 0.1 |

What the numbers say:

- **`rwkv7` trails on `vt`.**  It is 5.7 / 5.1 / 2.2 points below the control at 16K / 64K / 128K (paired
  bootstrap over samples, 95% intervals exclude zero).
- **On `cwe`, `kda` holds up at long context where `gdn2` and `rwkv7` drop.**  Against the control, `gdn2`
  is 4.6 / 3.5 below and `rwkv7` 3.0 / 4.7 below at 64K / 128K, while `kda` is level at 64K and 2.7 above
  at 128K.  Every model is near the floor at 128K, the control included.
- **The inexact kernels are much further away.**  Their gaps to the control reach 50 points or more on
  `vt` and 60 or more on `cwe`, against at most 7.4 among the exact kernels.

How far to trust them:

- **These intervals cover the RULER samples, not training.**  This is one trained checkpoint per kernel.
  Two older runs of `gdn`, `gdn2`, `kda` and `rwkv7` under a retired recipe (50 samples) show that the
  training seed can matter more: `gdn` on `vt` at 16K moves from 92.4 to 60.4 between seeds, while the four
  kernels stay within 4.8 of each other inside each seed.  `rwkv7` is the lowest of the four at every
  length in one seed and not in the other.  Do not read these tables as a settled kernel ranking.
- **The exact and inexact rows differ in recipe as well as kernel**, as described above.
- **These runs predate the RULER wrapper's `<|endoftext|>` stop** (see the changelog).  That fix was only
  checked on the backbone and on the control's needle tasks, not on these rows.

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
