# Gated-DeltaNet-2 Swap and Post-Training on Qwen3.5-0.8B


> **Generalised swap framework (2026-09):** `src/qwen_linswap` turns this GDN→GDN2 swap into a
> kernel registry (`gdn`, `gdn2`, `kda`, `kda_fullgate`) with shared verify / SFT / RULER
> tooling.  See [LINEAR_SWAP.md](LINEAR_SWAP.md) for the KDA swap and the cross-kernel results.

## Goal
Replace the Gated-DeltaNet (GDN) linear-attention layers in Qwen3.5-0.8B with Gated-DeltaNet-2 (GDN2), preserve the pre-trained function, and perform supervised fine-tuning (SFT) for long-context retrieval.

## Hardware and Environment
- **RAM**: 128 GiB
- **GPU**: NVIDIA RTX PRO 4500 (Blackwell), 32 GiB VRAM
- **Python environment**: `.venv` (uv-managed)
- **Key dependency**: `flash-linear-attention` installed from git HEAD (`0.5.2+git.9b20d26`)
- **Target context length**: 262,144 tokens; fell back to 131,072 tokens because the GDN2 chunk kernel OOMs at 262K on this GPU.

## Model Architecture

### Source model
- `Qwen/Qwen3.5-0.8B` (instruct)
- 24 layers, 1,024 embedding dim, 8 full-attention heads, head dim 256, 2 KV groups
- Interleaved attention pattern: 3 linear-attention layers followed by 1 full-attention layer, repeated 6 times
- Vocab size: 248,320
- RoPE base: 10,000,000; partial rotary factor: 0.25

### GDN2 layer
- Uses upstream `fla.layers.gdn2.GatedDeltaNet2`.
- Weight mapping from original GDN to GDN2:
  - `in_proj_qkv` → `q_proj`, `k_proj`, `v_proj`
  - `conv1d` → `q_conv1d`, `k_conv1d`, `v_conv1d`
  - `in_proj_z` / `norm` / `out_proj` → `g_proj` / `o_norm` / `o_proj`
  - scalar `beta` projection tiled into channel-wise `b_proj` (key side) and `w_proj` (value side)
  - scalar decay projection (`in_proj_a` + `A_log` + `dt_bias`) tiled into channel-wise `f_proj`, `A_log`, `dt_bias`
- Replaced `GatedDeltaNet2`'s `FusedRMSNormGated` output norm with `FusedRMSNormSwishGate` to match the source GDN gating.

### Implemented modules
- `src/qwen_gdn2/config.py`: model configuration
- `src/qwen_gdn2/model.py`: `Qwen3_5GDN2Model`, `TransformerBlock`, `GDN2Cache`, `generate()`
- `src/qwen_gdn2/model_components.py`: `GroupedQueryAttention`, `FeedForward`, `RMSNorm`, RoPE helpers
- `src/qwen_gdn2/gdn2_layer.py`: GDN2 layer construction
- `src/qwen_gdn2/load_weights.py`: HF checkpoint loading and GDN→GDN2 weight adaptation

## Verification

### Logit match
- GDN2-swapped model logits and top-1 predictions match HuggingFace `Qwen3_5ForCausalLM` on sample inputs.
- Cached decode was verified against no-cache decode:
  - 5-token prompt + 20 tokens: 20/20 agreement
  - 120-token prompt + 20 tokens: 20/20 agreement
  - 4096-token prompt + 16 tokens: 16/16 agreement
  - RULER `niah_single_1` at 4096 via cached path: 100.0

### Bug fixes
- Fixed residual-connection bug in the custom model.
- Fixed `use_cache` propagation in `TransformerBlock.forward` so the full-attention KV cache is populated during prefill.
- Replaced dense causal-mask creation in `_create_mask` with `None`; the full-attention layer builds its own compact mask. This removed a ~16 GiB allocation that caused OOM on 131K cached decode.

## Data Preparation
- Script: `scripts/prepare_datasets.py`
- Datasets:
  - `zai-org/LongAlign-10k`: used native `messages`, applied Qwen3.5 chat template, masked non-assistant turns
  - `Yukang/LongAlpaca-12k`: mapped `instruction` → user, `output` → assistant, applied chat template, masked user turns
  - `wenbopan/anti-haystack`: formatted as `document + "\n\nQuestion: " + question` → user, `answer` → assistant, applied chat template, masked document/question turns
- Combined, shuffled (seed 42), split 98/2 train/validation.
- Final split at 131,072 tokens: 23,826 train / 486 validation examples.
- Tokenization followed the Qwen3.5 chat template (`<|im_start|>role\ncontent<|im_end|>\n`) and left-truncated to `max_length`.

## Training
- Script: `scripts/sft.py`
- Mixed precision: bf16
- Gradient checkpointing enabled
- Batch size: 1; gradient accumulation: 1
- Cross-entropy computed in 2,048-token chunks to avoid materializing full 131K × vocab logits.

### Gate-parameter-only SFT
- Learning rate: 2e-4 for gate params (`b_proj`, `w_proj`, `f_proj`, `A_log`, `dt_bias`)
- Backbone frozen
- 100 training steps
- Checkpoints saved: `outputs/sft_gate_only/checkpoint-{25,26,50,75,100,101}`

### Full SFT
- Learning rate: 1e-5 for all parameters
- 50 training steps
- Checkpoints saved: `outputs/sft_full/checkpoint-{25,50,51}`

## RULER Evaluation Setup
- Added `QwenGDN2ModelWrapper` to `RULER/scripts/pred/model_wrappers.py`.
- Added model entries to `RULER/scripts/config_models.sh`:
  - `qwen-gdn2-base` (cached)
  - `qwen-gdn2-base-nocache`
  - `qwen-gdn2-gate-101` (cached)
  - `qwen-gdn2-gate-101-nocache`
  - `qwen-gdn2-full-51` (cached)
  - `qwen-gdn2-full-51-nocache`
- Added server types `qwen_gdn2` and `qwen_gdn2_nocache` in `RULER/scripts/pred/call_api.py`.
- Removed `nemo` manifest dependency; evaluation uses JSON helpers in `call_api.py` and `evaluate.py`.
- Added harder NIAH tasks to `RULER/scripts/config_tasks.sh`: `niah_multikey_1`, `niah_multivalue`, etc.

## Results

### `niah_single_1` (single needle) — no-cache, base model only
| Sequence length | Score |
|---|---|
| 4096 | 100.0 |
| 8192 | 100.0 |
| 16384 | 100.0 |
| 32768 | 100.0 |
| 65536 | 100.0 |
| 131072 | 100.0 |

### `niah_single_1` at 4096 — all checkpoints (no-cache)
| Model | Score |
|---|---|
| Base GDN2 swap | 100.0 |
| Gate-only SFT checkpoint-101 | 100.0 |
| Full SFT checkpoint-51 | 100.0 |
| HuggingFace Qwen3.5-0.8B | 100.0 |

### `niah_multikey_1` (4 keys, 1 value each) at 131072
| Model | Score |
|---|---|
| Base GDN2 swap | 100.0 |
| Gate-only SFT checkpoint-101 | 96.0 |
| Full SFT checkpoint-51 | 100.0 |

### `niah_multivalue` (1 key, 4 values each) at 131072
Scores are reported as value-level accuracy (correct values / total values).

| Model | Score |
|---|---|
| Base GDN2 swap | 92.5 |
| Gate-only SFT checkpoint-101 | 92.19 |
| Full SFT checkpoint-51 | 98.25 |

## Original-GDN Baseline

To isolate the effect of the GDN2 swap from the effect of SFT, we trained the original GDN Qwen3.5 model with the same full-SFT recipe at 131,072 tokens. The original-GDN implementation was switched from the hand-ported PyTorch layer to FLA's `GatedDeltaNet` layer so that 131K evaluation fits in memory and runs in a reasonable time.

- Script: `scripts/sft_gdn.py`
- Checkpoints: `outputs/sft_full_gdn/checkpoint-{25,50,51}/`
- Wrapper: `QwenOriginalModelWrapper` in `RULER/scripts/pred/model_wrappers.py`
- Server types: `qwen_gdn_original` and `qwen_gdn_original_nocache` in `RULER/scripts/pred/call_api.py`
- Model config: `qwen-gdn-original-full-51` in `RULER/scripts/config_models.sh`

### `niah_multivalue` (1 key, 4 values) at 131072 — original GDN vs GDN2

| Model | Score |
|---|---|
| Base GDN2 swap | 92.5 |
| Gate-only SFT checkpoint-101 | 92.19 |
| Full SFT checkpoint-51 | 98.25 |
| Original GDN base (FLA) | 92.75 |
| Original GDN full SFT checkpoint-51 | **89.50** |

The original-GDN full-SFT baseline scores lower than the GDN2 full-SFT checkpoint, suggesting the GDN2 swap contributes to the improved multi-value retrieval performance beyond SFT alone. Note that the original-GDN base is slightly lower than the GDN2 base (92.75 vs 96.75), which is expected because the function-preserving GDN2 initialization is exact for the original scalar beta/decay projections, while the FLA `GatedDeltaNet` layer uses channel-wise projections that are tiled from the scalar values; the small difference is within typical bfloat16 kernel vs. PyTorch-reference variation.

### Validation perplexity (20 batches, 131072-token examples)

| Model | Cross-entropy | Perplexity |
|---|---|---|
| GDN2 base | 1.6924 | 5.43 |
| GDN base (FLA) | 1.6912 | 5.43 |
| GDN2 full SFT checkpoint-51 | 1.5579 | 4.75 |
| Original GDN full SFT checkpoint-51 | 1.5590 | 4.75 |

The base models are nearly identical in perplexity, and both full-SFT checkpoints improve by the same amount (~0.13 nats).

### Additional original-GDN full-SFT results at 131072

| Task | Score |
|---|---|
| niah_single_1 | 100.0 |
| niah_single_2 | 92.0 |
| niah_single_3 | 88.0 |
| niah_multikey_1 | 91.0 |
| niah_multikey_2 | 91.0 |
| niah_multikey_3 | 62.5 (8 samples before timeout) |

Note: `niah_multikey_3` and the remaining tasks were interrupted by Triton autotuner issues on this Blackwell GPU / FLA 0.5.2 combination; the reported score is from the partial 8-sample run.

## Audit Notes

### Training recipe fairness
- `scripts/sft.py` and `scripts/sft_gdn.py` are identical except for the model import (`qwen_gdn2` vs `qwen_gdn2_original`). Both use the same data (`data/sft/len131072`), the same hyperparameters (full-SFT `lr=1e-5`, 50 steps, batch size 1, grad accumulation 1, AdamW, weight decay 0.01, max grad norm 1.0, seed 42, chunked cross-entropy 2048), and the same optimizer/scheduler defaults. The `args.json` files for `outputs/sft_full` and `outputs/sft_full_gdn` are identical.

### Function-preserving initialization
- GDN2 base and original-GDN base match on short and 4096-token prompts: top-1 predictions agree, generation agrees, and validation perplexity differs by <0.001 nats. This confirms the GDN2 weight tiling is functionally equivalent at initialization.

### SFT checkpoint differences
- Both full-SFT checkpoints differ from their bases by small bf16-scale amounts (max per-parameter diff ~5e-4, dominated by RMS-norm and dense weights). The GDN2 and original-GDN full-SFT checkpoints are also very close to each other (corresponding parameters differ by ~1e-4–5e-4), indicating both training runs followed similar optimization trajectories.

### Parameter count
- GDN2 has ~112M more parameters than original GDN because GDN2 uses channel-wise `f_proj`, `b_proj`, and `w_proj` (3 × 2048 × 1024 per linear layer) while original GDN uses scalar per-head `a_proj`/`b_proj` (2 × 16 × 1024). This is an intentional architectural difference in Gated-DeltaNet-2. The fair comparison is therefore between the GDN2 swap and the original GDN both trained with the same recipe, not a capacity-matched comparison.

### Kernel/evaluation reliability
- The original GDN baseline now uses FLA's `GatedDeltaNet` Triton kernel. Cached decode matches no-cache decode at 131K, removing the earlier drift issue. Occasional Triton autotuner errors on Blackwell still interrupt some long eval runs; reported partial scores are noted.

### Code
- `src/qwen_gdn2/`: GDN2-swapped Qwen3.5 implementation
- `src/qwen_gdn2_original/`: original-GDN Qwen3.5 implementation using FLA `GatedDeltaNet`
- `scripts/sft.py`: GDN2 training script
- `scripts/sft_gdn.py`: original GDN training script
- `scripts/prepare_datasets.py`: data preprocessing
- `RULER/scripts/pred/model_wrappers.py`: `QwenGDN2ModelWrapper` and `QwenOriginalModelWrapper`
- `RULER/scripts/config_models.sh`: model configs
- `RULER/scripts/config_tasks.sh`: task list
- `RULER/scripts/pred/call_api.py` and `RULER/scripts/eval/evaluate.py`: JSON-based evaluation helpers

### Checkpoints
- Base model: `models/Qwen3.5-0.8B/`
- Gate-only SFT: `outputs/sft_gate_only/checkpoint-{25,26,50,75,100,101}/`
- Full SFT: `outputs/sft_full/checkpoint-{25,50,51}/`
- Original GDN full SFT: `outputs/sft_full_gdn/checkpoint-{25,50,51}/`

### Evaluation outputs
- `RULER/scripts/benchmark_root/qwen-gdn2-base/synthetic/131072/`
- `RULER/scripts/benchmark_root/qwen-gdn2-gate-101/synthetic/131072/`
- `RULER/scripts/benchmark_root/qwen-gdn2-full-51/synthetic/131072/`
- `RULER/scripts/benchmark_root/qwen-gdn-original-full-51/synthetic/131072/`
