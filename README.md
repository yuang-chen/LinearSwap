![LinearSwap](docs/logo.png)



**Swap the linear-attention kernel of a pretrained hybrid LLM in place — no pretraining.**



LinearSwap replaces the linear-attention layers of a pretrained hybrid language model
(a backbone that interleaves Gated DeltaNet layers with full attention, such as
Qwen3-Next / Qwen3.5 / Qwen3.6 / Qwen3.8) with **other linear recurrences** —
Gated DeltaNet-2, Kimi Delta Attention, RWKV-7, Mamba-2, DeltaNet — and keeps the
rest of the network.  When the target recurrence contains Gated DeltaNet as a
special case, the new layer is initialised so that the model computes *exactly
the same function* at step 0 (verified to bf16 noise); when it does not, the new
layer starts from whatever maps.  Either way the swapped model is then distilled
back to the original on generic web text and benchmarked on long-context
retrieval and the usual short-context suite, all through one command line with
`--kernel <name>` as the only thing that changes between experiments.

Because every kernel starts from the same pretrained function and goes through
the same recipe, LinearSwap turns "which recurrence is better?" into a
controlled experiment instead of a set of incomparable pretraining runs.  The kernels come from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention).

- [Kernels](#kernels)
- [Installation](#installation)
- [Usage](#usage)
  - [Building a swapped model](#building-a-swapped-model)
  - [Token mixing layers](#token-mixing-layers)
  - [Adding a kernel](#adding-a-kernel)
- [Distillation](#distillation)
- [Evaluation](#evaluation)
- [Benchmarks](#benchmarks)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)



## Kernels


| name           | recurrence (FLA kernel)                                   | init from the pretrained GDN layer                                                    | new params | exact |
|---|---|---|---|---|---|
| `gdn`          | Gated DeltaNet                                            | weight copy (control)                                                                 | 0.59M      | yes   |
| `gdn2`         | Gated DeltaNet-2                                          | scalar beta/decay tiled into b/w/f gates                                              | 113M       | yes ‡ |
| `kda`          | Kimi Delta Attention                                      | scalar decay tiled into the low-rank per-channel gate                                 | 7.4M       | yes   |
| `kda_fullgate` | Kimi Delta Attention                                      | … with a dense decay projection                                                       | 38M        | yes   |
| `rwkv7`        | RWKV-7-style DPLR generalised delta rule                  | decay/beta tiled, removal key = key                                                   | 14M        | yes   |
| `mamba2`       | Mamba-2-style SSD (scalar decay, no erase)                | shared weights copied, erase dropped                                                  | 0.30M      | no    |
| `deltanet`     | DeltaNet (erase, no decay)                                | shared weights copied, decay dropped                                                  | 0.29M      | no    |
| `gla`          | Gated Linear Attention (per-channel decay, no erase)      | shared weights copied, decay MLP at FLA init                                          | 0.92M      | no    |
| `mamba3` †     | Mamba-3 (data-dependent decay, trapezoidal, rotary state) | GDN decay / projections mapped into the fused `in_proj`, rotary and trapezoid neutral | 0.05M      | no    |
| `mamba1` †     | Mamba-1 selective SSM (per-channel, no q/k)               | values, gate, conv and `out_proj` copied; SSM params at Mamba init                    | 19M        | no    |
| `gdn_breg` §   | Gated DeltaNet + Bregman soft-threshold on the state       | weight copy (GDN's parameter set); `lam` from `LINSWAP_BREG_LAM`, 0 = GDN            | 0.59M      | at lam = 0 |


"Exact" kernels reproduce the pretrained model at initialisation; the others start
from whatever maps and have more to recover.  Every kernel then goes through the
same distillation recipe.  Parameter counts are for the 0.8B backbone.  ‡ `gdn2` requires as many value heads as key heads: FLA's
`GatedDeltaNet2` shares its decay and erase gates across a group of value heads, so a
backbone with grouped value heads (16 key / 48 value heads, as in the larger Qwen models) has no exact
GDN2 image and the kernel refuses to build there.  § `gdn_breg` wraps the external
`gated_breg_delta_rule` package (in development, not shipped) and is registered only when it
imports; at `lam = 0` it verifies at the GDN noise floor.  † `mamba1` / `mamba3` use `mamba_ssm`'s kernels through
FLA's `Mamba` / `Mamba3` layers and are registered only when those import (see
Installation).  Mamba-3's single-token decode step additionally needs `mamba_ssm`'s
CuTe-DSL kernel (`nvidia-cutlass-dsl` + `quack-kernels`), which did not run with the
currently published versions; prefill and distillation work, and `evaluate --no_cache` recomputes the prefix per generated token instead.  Every target is the *recurrence* of the named
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
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
python tests/test_kernels.py && python tests/test_hf.py  # kernels match the control; HF round-trip is exact
                                                         # (also tests/test_batch.py, tests/test_gva.py)
python tests/test_losses.py                              # chunked CE / KL vs dense autograd; CPU-only, needs no checkpoint
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
model = build_model(ckpt_dir="outputs/kda/distill/checkpoint-16338")  # a checkpoint; kernel read from its config.json
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
linswap export --ckpt outputs/kda/distill/checkpoint-16338 --out hf/Qwen3.5-0.8B-KDA-distilled
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
linear layer".  Native `model.pt` checkpoints use the same keys.  Limits: batches must be unpadded or
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

## Distillation

The workflow is **verify → distill → evaluate**, one command each; `run` chains
them for one kernel.

```bash
linswap verify   --kernel rwkv7 --baseline gdn       # layer / logits / cache / generation vs the backbone
linswap distill  --kernel rwkv7                      # the three training steps below
linswap run      --kernel rwkv7                      # verify + distill + both evaluations
linswap export   --ckpt outputs/rwkv7/distill/checkpoint-16338 --out hf/Qwen3.5-0.8B-RWKV7
```

Training is three steps on packed generic web text (DCLM by default, prepared on
first use; `--text_data fineweb-edu` is the alternative).  No instruction data,
no chat template, no supervised fine-tuning:

| step | loss | tokens | length | sequences/step | lr | trained |
|---|---|---|---|---|---|---|
| `layer` | L2 between each swapped layer's output and the original layer's, on the original layer's own input, all layers in parallel | 100M | 512 | 32 | 1e-3 → 1e-5 cosine | the swapped layers |
| `kl` | KL(teacher ‖ student) on next-token distributions, in vocabulary chunks | 500M | 512 | 96 | 1e-5 flat | all parameters |
| `ce` | plain next-token cross-entropy, no teacher (context extension) | 100M | 16384 | 96 | 1e-5 flat | all parameters |

Adam(0.9, 0.95, 1e-8), clip 1.0, bf16, ~6 GPU-hours per kernel at 0.8B on one
143 GiB GPU.  Budgets are given in tokens (`--kl_tokens 250e6`), and every
per-step knob (`--stage_length`, `--stage_batch`, `--stage_micro`,
`--stage_schedule`, the learning rates) is a command-line argument.  Checkpoints
record their kernel in `config.json`, so `build_model(ckpt_dir=...)` and
`evaluate` need nothing else.

## Evaluation

```bash
linswap evaluate --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338 --name rwkv7
linswap lmeval   --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338 --name rwkv7-lmeval
python tools/throughput.py --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338
```

`--models` takes kernel names (the base swap) and/or checkpoint directories
(`label=path` to name a row).  Two suites, both run on the students *and* on the
unmodified backbone so the comparison is like-for-like:

* **Long-context retrieval** — RULER's `niah_single_1/2/3` and `niah_multikey_1`
  at 4K / 16K / 64K / 128K, 50 samples per task, cached greedy decoding, prompts
  built with RULER's base template (`--chat_template` switches to the backbone's
  chat format; use the same setting for every model).
* **Short context** — LAMBADA, ARC-c/e, PIQA, WinoGrande, HellaSwag 0-shot and
  MMLU 5-shot through lm-eval-harness, reported as accuracy and as a relative
  score (s − r)/(t − r) against a reference row, r = chance.

Results go to `outputs/eval/<name>/summary.{csv,md,json}`.

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
| `rwkv7` (exact init) | 100 / 100 / 98 / 96 | 100 / 100 / 100 / 96 | 100 / 100 / 100 / 94 | 100 / 100 / 92 / 92 |
| `mamba2` (no erase) | 100 / 100 / 98 / 98 | 100 / 100 / 100 / 96 | 100 / 100 / 98 / 92 | 100 / 98 / 92 / 72 |

**Short context**, accuracy (relative score vs the unmodified backbone in %)

| model | LAMBADA | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | MMLU | rel. avg |
|---|---|---|---|---|---|---|---|---|
| unmodified backbone | 0.437 | 0.374 | 0.611 | 0.693 | 0.583 | 0.496 | 0.504 | 100.0 |
| control (`gdn`) | 0.478 | 0.399 | 0.653 | 0.706 | 0.588 | 0.524 | 0.515 | **110.0** |
| `rwkv7` | 0.479 | 0.391 | 0.642 | 0.705 | 0.590 | 0.521 | 0.517 | 108.7 |
| `mamba2` | 0.462 | 0.372 | 0.610 | 0.701 | 0.578 | 0.520 | 0.501 | 101.4 |

What the numbers say:

- The recipe, not the kernel, is what lifts a swapped model above the original:
  the control gains as much as the students on both suites.  The question a
  swap experiment has to answer is therefore what the swap costs *on top of the
  same training*, which is what the control row makes visible.
- With an exact init the swap is nearly free: `rwkv7` is 1.3 relative points
  under the control on the short-context suite and within sample noise of it on
  every needle length.
- Dropping the delta-rule erase is not free: `mamba2` trails the control by 8.6
  relative points and loses the distractor needle at 128K (72 vs 90).
- Decoding is length-independent for all of them (constant state).  Prefill at
  8K/32K: `gdn` 134K/141K tok/s, `mamba2` 119K/128K, `rwkv7` 79K/76K; decode
  24.5 / 30.5 / 33.3 ms per token.

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
gate tiling and RULER integration this project generalises.
The backbone implementation started from Sebastian Raschka's
[Qwen3.5 from-scratch notebook](https://github.com/rasbt/LLMs-from-scratch).
Kernels come from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
(Songlin Yang, Yu Zhang and contributors); evaluation uses NVIDIA's
[RULER](https://github.com/NVIDIA/RULER).  The architectures swapped in are
Gated DeltaNet-2, Kimi Delta Attention (Kimi Linear), RWKV-7, Mamba-2 and DeltaNet,
by their respective authors.

The distillation recipe and the evaluation protocol follow
[RADLADS](https://arxiv.org/abs/2505.03005) (Goldstein et al., *Rapid Attention
Distillation to Linear Attention Decoders at Scale*): the three steps
(hidden-state alignment → logit distillation → context-length extension) with
its token budgets, learning-rate schedules and optimizer settings, generic-text
distillation data, base-prompt evaluation and the relative score against the
teacher.  RADLADS converts softmax attention into linear attention; here the
same recipe is applied to swapping one linear recurrence for another inside a
hybrid backbone.