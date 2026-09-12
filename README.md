# qwen-linswap — swapping the linear-attention kernel of Qwen3.5 in place

Qwen3.5-0.8B is a hybrid model: three Gated-DeltaNet (GDN) linear-attention
layers for every full-attention layer.  This repository replaces those GDN
layers with **other linear-attention kernels** — Gated DeltaNet-2, Kimi Delta
Attention, DeltaNet, … — *without retraining from scratch*: the new layer is
initialised from the pretrained GDN weights so that (whenever the new kernel
contains GDN as a special case) the model computes exactly the same function
at step 0, and is then post-trained with long-context SFT and benchmarked on
RULER.

Everything is driven by a kernel registry, so adding a kernel is one file and
the same verification, training and evaluation tooling applies to it.

```
kernel name    recurrence (flash-linear-attention kernel)      init from GDN                          new params  exact
gdn            Gated DeltaNet                                  weight copy (control)                  0.59M       yes
gdn2           Gated DeltaNet-2                                scalar beta/decay tiled → b/w/f gates  113M        yes
kda            Kimi Delta Attention                            scalar decay tiled → low-rank f_proj   7.4M        yes
kda_fullgate   Kimi Delta Attention                            … with a dense f_proj                  38M         yes
rwkv7          RWKV-7 generalised delta rule (DPLR)            decay/beta tiled, removal key = key    14M         yes
mamba2         Mamba-2 SSD (scalar decay, no delta rule)       shared weights copied, erase dropped   0.30M       no
deltanet       DeltaNet (delta rule, no decay)                 shared weights copied, decay dropped   0.29M       no
```

Kernels marked *exact* reproduce the pretrained model at step 0 and go
straight to SFT.  The others cannot represent the pretrained layer; for them
the pipeline first **distils** the swapped model from the original (layer-wise
output matching, then end-to-end KL) and only then fine-tunes.  Mamba-1 and
Mamba-3 are not included: their kernels only exist in `mamba_ssm`, whose
current release needs a newer Triton than torch 2.6 allows and breaks
`flash-linear-attention` when installed here (see docs/framework.md).


## Layout

- `linswap.py` — the command line: `verify`, `distill`, `posttrain`, `evaluate`, `run` (the whole chain), `kernels`.
- `src/qwen_linswap/` — the framework: kernel registry, the Qwen3.5 backbone with pluggable linear layers, weight loading, data and SFT utilities. Kernels live in `src/qwen_linswap/kernels/`, one file each; that directory is the extension point. The workflow stages live in `src/qwen_linswap/pipeline/`.
- `tests/test_kernels.py` — regression test over all registered kernels.
- `RULER/` — the RULER benchmark, vendored with a wrapper for swapped models.
- `docs/` — design, per-kernel details and all results (`framework.md`), plus the notes and log of the original GDN→GDN2 experiment.

## Workflow

The workflow is **verify → (distill) → posttrain → evaluate**; each stage is one
command, and `run` chains them for one kernel (distilling automatically when
the kernel's init is not exact).

```bash
source .venv/bin/activate
python tests/test_kernels.py                  # every kernel builds, loads, matches the GDN control, round-trips

# 1. verify: is the swap function preserving?  (the `gdn` baseline column is the bf16/Triton noise floor)
python linswap.py verify --kernel kda --baseline gdn

# 1b. distill (inexact kernels only): layer-wise alignment + KL against the original model
python linswap.py distill --kernel mamba2           # -> outputs/mamba2/distill/checkpoint-N

# 2. posttrain: gate-only and full SFT with the standard recipe (SFT data is prepared on first use)
python linswap.py posttrain --kernel kda            # -> outputs/kda/sft_gate_only, outputs/kda/sft_full
python linswap.py posttrain --kernel mamba2 --modes full --init_ckpt outputs/mamba2/distill/checkpoint-500

# 3. evaluate: validation loss + RULER for any set of base swaps / checkpoints, one summary table
python linswap.py evaluate --models kda outputs/kda/sft_full/checkpoint-50 gdn \
       --tasks niah_single_1,niah_multikey_1,niah_multivalue --lengths 131072 --samples 100 --name kda-vs-gdn
                                                    # -> outputs/eval/kda-vs-gdn/summary.{csv,md,json}

# all of the above for one kernel
python linswap.py run --kernel kda
```

Every stage accepts `--help`; the recipe knobs (steps, learning rates,
training length, RULER tasks / lengths / sample count) are arguments, so
comparisons between kernels use one command line with only `--kernel` changed.

```python
import sys; sys.path.insert(0, "src")
from qwen_linswap import build_model, list_kernels
model = build_model("kda")                                       # Qwen3.5-0.8B weights, exact KDA init
model = build_model(ckpt_dir="outputs/kda/sft_full/checkpoint-50")  # SFT checkpoint; kernel read from config.json
out = model.generate(input_ids, max_new_tokens=32)               # greedy, cached decode
```

## Adding a kernel

Create `src/qwen_linswap/kernels/<name>.py` with a `build(cfg, layer_idx)`
that returns a token mixer with the FLA layer interface
(`forward(x, past_key_values=None, use_cache=False) -> (out, None, cache)`),
an `init_from_gdn(layer, hf_state_dict, layer_idx, model_prefix)` that copies
or tiles the pretrained GDN tensors, and a `register_kernel(KernelSpec(...))`
call; import it in `kernels/__init__.py`.  Make architectural changes in
`build` (not in init) so the module structure is fixed before weights are
loaded.  `kernels/common.py` documents the pretrained tensor layout and the GDN
recurrence and provides the tiling / splitting helpers; `kernels/kda.py` is the
template for an exact swap, `kernels/deltanet.py` for an inexact one.  Then run
`tests/test_kernels.py` and `python linswap.py verify --kernel <name> --baseline gdn`.

## Results (131K context, identical SFT recipe)

RULER at 131072 tokens, 100 samples per task, cached greedy decoding
validation cross-entropy on 40 held-out examples.  Full tables and discussion in
[docs/framework.md](docs/framework.md).

| model | trainable | val CE | niah_single_1 | niah_multikey_1 | niah_multivalue |
|---|---|---|---|---|---|
| gdn base (exact copy) | – | 1.743 | 100 | 100 | 96.5 |
| gdn2 base | – | 1.741 | 100 | 100 | 96.25 |
| kda base | – | 1.742 | 100 | 100 | 95.75 |
| gdn gate-only 100 | 0.59M | 1.472 | 100 | 100 | 96.25 |
| kda gate-only 100 | 7.4M | 1.427 | 100 | 100 | 97.5 |
| kda_fullgate gate-only 100 | 38M | 1.413 | 100 | 98 | 99.0 |
| gdn2 gate-only 100 | 113M | 1.378 | 100 | 99 | 96.5 |
| gdn full 50 | 752M | 1.388 | 100 | 100 | 99.0 |
| kda full 50 | 759M | 1.388 | 100 | 100 | 99.5 |
| gdn2 full 50 | 865M | 1.389 | 100 | 100 | 99.25 |
| deltanet base (inexact) | – | 12.845 | 0 | 0 | 0 |
| deltanet full 50 (lr 1e-5) | 752M | 8.628 | – | – | – |
| deltanet full 200 (lr 1e-5) | 752M | 7.174 | – | – | – |
| deltanet full 200 (lr 1e-4) | 752M | 6.220 | 0 | 0 | 0 |

Take-aways: the exact swaps (GDN2, KDA) lose nothing at init; full SFT lands
on the same loss and retrieval scores for every exact kernel; gate-only SFT is
where kernels differ, and KDA's per-channel decay gate is a far cheaper
gate-only handle for multi-value retrieval than GDN2's three dense gates.
DeltaNet shows what an inexact swap costs: dropping the decay destroys the
pretrained function (validation CE 12.8, retrieval 0 at init) and 200 full-SFT
steps at a 10× higher learning rate do not rebuild it (CE 6.2, retrieval still
0) — inexact swaps need a distillation-style schedule, not the short recipe
that suffices for exact ones.

## Setup

Python environment: the `uv`-managed `.venv` (torch 2.6, transformers 5.16,
flash-linear-attention 0.6.0).  Put `Qwen/Qwen3.5-0.8B` in `models/`; the SFT
data is downloaded and tokenised on first use of `posttrain` / `evaluate`, and
checkpoints and evaluation results go to `outputs/`.  Hardware notes and kernel caveats
are in [docs/framework.md](docs/framework.md).

## Acknowledgements

The Qwen3.5 backbone implementation started from Sebastian Raschka's
[Qwen3.5 from-scratch notebook](https://github.com/rasbt/LLMs-from-scratch)
(LLMs-from-scratch, bonus material).  Kernels come from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention);
evaluation uses NVIDIA's [RULER](https://github.com/NVIDIA/RULER).
