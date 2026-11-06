![LinearSwap](docs/logo.png)

**Swap the linear-attention kernel of Qwen3.5 in place, and distil the result back to the teacher as
cheaply as possible.**

Qwen3.5 is a hybrid vision-language model: 18 of its 24 decoder layers are Gated DeltaNet, 6 are full
attention. LinearSwap replaces the Gated-DeltaNet layers with another sequence mixer and keeps everything
else — `transformers`' own model, vision tower and processor included. The unmodified backbone is the
**teacher**; a swapped model is a **student**, and the goal is for the student to *approach the teacher*
on text, long context and images with the least training.

[Kernels](#kernels) · [Install](#install) · [Usage](#usage) · [Distillation](#distillation) ·
[Evaluation](#evaluation) · [Results](#results) · [Adding a kernel](#adding-a-kernel)

## Kernels

| kernel     | sequence mixer                            | init from the pretrained layer               | exact |
| ---------- | ----------------------------------------- | -------------------------------------------- | ----- |
| `gdn`      | Gated DeltaNet on FLA's kernels           | weight copy — the teacher                    | yes   |
| `rwkv7`    | RWKV-7 generalised delta rule (DPLR)      | gates tiled, removal key = key               | yes   |
| `kda`      | Kimi Delta Attention                      | decay tiled into the low-rank gate           | yes   |
| `gdn2`     | Gated DeltaNet-2                          | scalar gates tiled into b/w/f                | yes   |
| `mamba2`   | Mamba-2 SSD (decay, no erase)             | shared weights copied, erase dropped         | no    |
| `deltanet` | DeltaNet (erase, no decay)                | shared weights copied, decay dropped         | no    |
| `gla`      | Gated Linear Attention                    | shared weights copied, decay fitted per head | no    |
| `swa`      | sliding-window softmax, 64 wide + 4 sinks | weight copy; new logit temperature           | no    |

An **exact** kernel contains Gated DeltaNet as a special case: at step 0 the swapped model reproduces
the teacher to bf16 noise and needs no training. The others lack something the teacher uses and have
to be distilled. [docs/kernels.md](docs/kernels.md) has the per-kernel mappings.

## Install

Python 3.11, PyTorch ≥ 2.7 with a matching Triton ≥ 3.3; tested on torch 2.9.1 / CUDA 12.8 /
Triton 3.5.1 / transformers 5.17. `flash-linear-attention` must come from git: the published wheels ship
only `fla/layers` and `fla/models`.

```bash
git clone https://github.com/yuang-chen/LinearSwap && cd LinearSwap
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-deps torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention@8e84ed4"
uv pip install -e ".[eval]"                      # the `linswap` command; [eval] adds the three harnesses
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
python tests/test_linswap.py                     # every kernel on text and an image, a short distillation
```

On Hopper GPUs with Triton 3.4–3.7, FLA rejects the Triton backward of its gated chunk kernels
(fla#640): add `uv pip install -e ".[hopper]"` (TileLang) to train. RULER's haystacks are fetched by
`RULER/scripts/data/synthetic/json/download_*.{py,sh}`; every other dataset comes from the Hub on first use.

## Usage

```bash
linswap kernels                                   # what is registered
linswap verify  --kernel kda                      # is the swap faithful, on text and on an image?
linswap distill --kernel gla                      # train the swapped layers -> outputs/distill/gla
linswap eval    --models native kda outputs/distill/gla --name main    # the standard test sets
linswap eval    --models gla --smoke              # a functional check in minutes, not a score
```

```python
from linswap import build_model
model = build_model("kda")                        # transformers' Qwen3.5 vision-language model, mixers swapped
model = build_model("outputs/distill/gla")        # a distilled checkpoint
out = model.generate(**processor.apply_chat_template(messages_with_images, tokenize=True, return_dict=True,
                                                     add_generation_prompt=True, return_tensors="pt").to("cuda"))
```

`build_model` loads the backbone with `transformers` and replaces only each decoder layer's
`linear_attn`; `"native"` is the backbone untouched. A checkpoint holds just the swapped layers
(`swap.safetensors`, plus `linswap.json` naming the kernel), since nothing else is trained.

```
src/linswap/
  model.py       the swap inside transformers' Qwen3.5 model; checkpoints; build_model
  registry.py    KernelSpec and the kernel registry
  kernels/       one file per kernel: how to build the layer and initialise it from Gated DeltaNet
  verify.py      swapped model vs the backbone on text and an image, gated for exact kernels
  data.py        the distillation stream: web text, chat, image chats
  distill.py     local matching, then closed loop, stopped by measurement
  evaluate.py    lm-eval (text), RULER (long context), lmms-eval (images); --smoke
RULER/           NVIDIA's RULER, with a model wrapper for linswap
tests/           test_linswap.py: the whole path on one GPU
```

## Distillation

`linswap distill` trains **only the swapped layers**, against **the teacher's activations**, on **the
kinds of input the teacher handles** (web text up to 16K tokens, chat, image chats). There are no labels
and no ground-truth loss, and the rest of the model stays the teacher's, frozen.

1. **local** — every swapped layer learns to reproduce the teacher layer's output from the teacher
   layer's input: 18 independent regressions on one teacher forward, without the vocabulary head and
   without a backward pass through the depth of the model.
2. **loop** — the student runs on its own hidden states, which are matched to the teacher's after every
   block.

The gap to the teacher is measured on held-out text, chat, images and a 32K sequence at token
milestones, and the run ends when more tokens stop closing it. The trained layers keep float32 master
weights (a small step on a bf16 weight rounds away) while the forward pass runs in bf16; the gap is
measured on, and the checkpoint holds, the bf16 cast. An exact kernel is not trained at all. Every run
writes `train_log.jsonl` and, with `tensorboard` installed, the same curves as scalars:
`tensorboard --logdir outputs/distill` shows the runs side by side while they train.
[docs/recipe.md](docs/recipe.md) has the reasoning and the measurements behind each choice.

## Evaluation

`linswap eval` runs each suite with the harness that defines its protocol, on the full test set:

- **text** — [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): LAMBADA,
  ARC-c/e, PIQA, WinoGrande, HellaSwag (0-shot), MMLU (5-shot), IFEval.
- **long** — [RULER](https://github.com/NVIDIA/RULER): four needle tasks, variable tracking, common and
  frequent words extraction, 500 samples per task at 4K / 16K / 64K / 128K.
- **mm** — [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): MME, MMBench-EN dev, AI2D, ChartQA,
  OCRBench, TextVQA val, MuirBench.

`--smoke` keeps the code path and cuts the work to a few items of two tasks per suite. Results
accumulate in `outputs/eval/<name>/results.{json,md}`, so models can be added to a run or spread over
GPUs.

## Results

Qwen3.5-0.8B, one seed; the gap is KL(teacher ‖ student) on held-out text, chat, image chats and a 32K
sequence.

- **Exact kernels need no training.** `kda`, `gdn2` and `rwkv7` at step 0 are at the numerical noise of
  the comparison on text and on images (gap 0.001; same greedy answers as the teacher).
- **Inexact kernels: a quarter of a GPU-hour does more than six did.** The earlier recipe (700M tokens,
  about 6 GPU-hours, the whole model trained on web text) left `gla` at a gap of 0.55, mostly on images
  (1.70). Training only the swapped layers against the teacher's hidden states brings it to 0.13 in 20M
  tokens and 0.25 GPU-hours, evenly across text, chat, images and long context.

The full distillation runs and the standard test sets are in [docs/results.md](docs/results.md).

## Adding a kernel

Write `src/linswap/kernels/<name>.py` with

* `build(cfg, layer_idx) -> nn.Module`: a token mixer with the FLA layer interface
  `forward(x, past_key_values=None, use_cache=False) -> (out, None, cache)`;
* `init_from_gdn(layer, gdn_state, layer_idx, prefix)`: copy or tile the pretrained Gated-DeltaNet
  tensors (`kernels/common.py` documents their shapes and the recurrence, and has the helpers);
* `register_kernel(KernelSpec(name=..., build=..., init_from_gdn=..., exact_init=True/False))`, and an
  import in `kernels/__init__.py`.

`linswap verify --kernel <name>` then says whether the init is function preserving: every number should
sit at the level of the `gdn` column, which is numerical noise. Where a gate cannot hold the pretrained
one exactly, fit it rather than leaving it at the library's init (`kernels/gla.py::fit_gate`) — and
if it is still inexact, `linswap distill` closes what is left.

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
[code](https://github.com/lutetjeff/gdn2-in-place)), whose function-preserving gate tiling this project
generalises. Kernels come from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
(Songlin Yang, Yu Zhang and contributors); the model is `transformers`' Qwen3.5; evaluation uses
lm-evaluation-harness, lmms-eval and NVIDIA's RULER. Layer-wise matching followed by end-to-end
distillation on generic text follows [RADLADS](https://arxiv.org/abs/2505.03005) (Goldstein et al.).
The architectures swapped in are Gated DeltaNet-2, Kimi Delta Attention, RWKV-7, Mamba-2, DeltaNet,
Gated Linear Attention and sliding-window attention, by their respective authors.
