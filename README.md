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
kernel name    layer (flash-linear-attention)   init from GDN                         new params
gdn            GatedDeltaNet                    exact copy (control)                  0.59M
gdn2           GatedDeltaNet2                   scalar beta/decay tiled → b/w/f gates 113M
kda            KimiDeltaAttention               scalar decay tiled → low-rank f_proj  7.4M
kda_fullgate   KimiDeltaAttention               … with a dense f_proj                 38M
deltanet       DeltaNet                         decay dropped (NOT exact)             0.29M
```

## Layout

- `src/qwen_linswap/` — the framework: kernel registry, the Qwen3.5 backbone with pluggable linear layers, weight loading, SFT utilities. Kernels live in `src/qwen_linswap/kernels/`, one file each; that directory is the extension point.
- `scripts/` — `prepare_datasets.py`, `verify.py`, `sft.py`, `eval_val_loss.py`, `register_ruler_model.py` (see `scripts/README.md`).
- `tests/test_kernels.py` — regression test over all registered kernels.
- `RULER/` — the RULER benchmark, vendored with a wrapper for swapped models.
- `docs/` — design, per-kernel details and all results (`framework.md`), plus the notes and log of the original GDN→GDN2 experiment.

## Quick start

```bash
source .venv/bin/activate                     # uv-managed venv; RULER's run.sh calls bare `python`
python tests/test_kernels.py                  # every kernel builds, loads, matches the GDN control, round-trips

# 1. verify a swap against HF Qwen3.5 (the `gdn` baseline column is the bf16/Triton noise floor)
python scripts/verify.py --kernel kda --baseline gdn

# 2. post-train it (identical recipe for every kernel; ~5 min per run on one L20X)
python scripts/prepare_datasets.py --max_length 262144           # once
python scripts/sft.py --kernel kda --mode gate_only --output_dir outputs/sft_kda_gate \
       --max_length 131072 --num_steps 100 --grad_accum_steps 2 --gate_lr 2e-4
python scripts/sft.py --kernel kda --mode full --output_dir outputs/sft_kda_full \
       --max_length 131072 --num_steps 50 --grad_accum_steps 2 --full_lr 1e-5

# 3. evaluate
python scripts/eval_val_loss.py --models kda-base kda-full-50 --batches 40
python scripts/register_ruler_model.py --name kda-base --kernel kda
python scripts/register_ruler_model.py --name kda-full-50 --ckpt outputs/sft_kda_full/checkpoint-50
(cd RULER/scripts && bash run.sh linswap-kda-full-50 synthetic)   # tasks/lengths in config_tasks.sh / config_models.sh
```

```python
import sys; sys.path.insert(0, "src")
from qwen_linswap import build_model, list_kernels
model = build_model("kda")                                          # Qwen3.5-0.8B weights, exact KDA init
model = build_model(ckpt_dir="outputs/sft_kda_full/checkpoint-50")  # SFT checkpoint; kernel read from config.json
out = model.generate(input_ids, max_new_tokens=32)                  # greedy, cached decode
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
`tests/test_kernels.py` and `scripts/verify.py --kernel <name> --baseline gdn`.

## Results (131K context, identical SFT recipe)

RULER at 131072 tokens, 100 samples per task, cached greedy decoding
(`outputs/ruler_131k_summary.csv`); validation cross-entropy on 40 held-out
examples (`outputs/val_loss_131k.json`).  Full tables and discussion in
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
flash-linear-attention 0.6.0).  Put `Qwen/Qwen3.5-0.8B` in `models/` and run
`scripts/prepare_datasets.py` once to build the SFT data; checkpoints, RULER
registrations and results go to `outputs/`.  Hardware notes and kernel caveats
are in [docs/framework.md](docs/framework.md).

## Acknowledgements

The Qwen3.5 backbone implementation started from Sebastian Raschka's
[Qwen3.5 from-scratch notebook](https://github.com/rasbt/LLMs-from-scratch)
(LLMs-from-scratch, bonus material).  Kernels come from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention);
evaluation uses NVIDIA's [RULER](https://github.com/NVIDIA/RULER).
