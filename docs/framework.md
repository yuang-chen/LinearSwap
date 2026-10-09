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
  kernels/kda.py       "kda"          Kimi Delta Attention, low-rank per-channel decay gate (as in Kimi Linear)
  kernels/rwkv7.py     "rwkv7"        RWKV-7 generalised delta rule (DPLR kernel), exact tiled init
  kernels/mamba2.py    "mamba2"       Mamba-2 SSD on the simple-GLA kernel — inexact swap (exact_init=False)
  kernels/deltanet.py  "deltanet"     DeltaNet, no decay — inexact swap
  kernels/gla.py       "gla"          Gated Linear Attention, stock FLA layer, GDN decay fitted per head into the gate — inexact swap
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

## The documents

| file | what is in it |
|---|---|
| [kernels.md](kernels.md) | per-kernel swaps: what maps exactly, what does not, and the verify numbers |
| [recipe.md](recipe.md) | the three distillation steps and the two evaluation suites |
| [results.md](results.md) | RULER and short-context tables, throughput, and the approaches that were dropped |
| [gate_diagnostics.md](gate_diagnostics.md) | what the training steps do to a swapped layer: gate spread and weight drift |
| [state_rank.md](state_rank.md) | the rank of each head's memory state on real text |

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

**Fit the gate rather than leaving it at the library's init.**  Where the target kernel's gate cannot
hold the pretrained one exactly, fitting it is worth more than any amount of training that follows.
`gla` is the worked example: FLA's `GatedLinearAttention` parameterises its decay as
`logsigmoid(gk_proj x) / 16`, which cannot represent GDN's `-exp(A_log) softplus(a + dt_bias)`, so the
kernel used to leave `gk_proj` at FLA's random init.  `kernels/gla.py::fit_gate` instead solves a
per-head least squares in retention space over the range the pretrained gate actually occupies, then
tiles the result across the head's channels.  The two inits differ by more than any training change
measured in this project:

| `gla` init | RULER task average, 4K / 16K / 64K / 128K | short context |
|---|---|---|
| FLA random gate | 87.5 / 64.8 / 43.2 / 1.6 | 94.3 |
| per-head fit | 99.1 / 85.4 / 76.0 / 68.8 | 99.2 |

**The layer step's initial loss is the number to watch.**  `linswap distill` prints it before deciding
whether to run the step, and it separates the cases cleanly: 0 for an exact copy, ~5e-7 to 8e-7 for a
function-preserving reparameterisation (`gdn2`, `kda`, `rwkv7`), 1.6e-2 for `gla`'s
fitted gate, 1.8e-2 for `mamba2`, 1.1e-1 for `swa`.  Treat it as the fidelity of the init: driving it
down by construction costs nothing at run time, and the training steps cannot buy back what a poor init
gives away.

## Environment

Python 3.11, torch 2.9.1 / CUDA 12.8 / Triton 3.5.1, `flash-linear-attention` 0.6.0 (git 8e84ed4),
transformers 5.16.1; `causal-conv1d` and `mamba_ssm` built from source for the `mamba1` / `mamba3`
kernels.  On Hopper-class GPUs FLA refuses the Triton backward of its gated chunk kernels under
Triton 3.4–3.7 (issue #640): install `tilelang` (`pip install -e ".[hopper]"`) for `gdn` / `gdn2` /
`kda`, and use Triton < 3.4 or ≥ 3.7.1 to *train* `mamba2` (the simple-GLA op has no TileLang
backend; inference is unaffected).  The distillation and evaluation numbers above were produced on
2 × NVIDIA L20X (143 GiB).
