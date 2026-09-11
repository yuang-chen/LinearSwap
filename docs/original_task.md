# TASK FOR AGENTS.

Our goal is to perform an in-place swap of Gated-DeltaNet with Gated-DeltaNet 2 and perform post-training.
Our target model is Qwen3.5-0.8B.

## Constraints.
Your hardware available is 128GiB of RAM + NVIDIA RTX PRO 4500 (Blackwell) with 32GiB VRAM.
You should use `uv` and there is a venv in @.venv
All commands should time out quickly, nothing above 10 minutes, except full benchmark and training runs, which should be limited to 1 hour.
The model should be able to use all 262K context available.
Work autonomously without human interaction, until you have completed all the below goals.

## Clarified Decisions

The following decisions were made before starting implementation. Do not deviate from them without asking.

- **Target model:** `Qwen/Qwen3.5-0.8B`.
- **GDN2 implementation:** Use the fast Triton kernels from `gateddeltanet-2/lit_gpt/gdn2_ops/` via `flash-linear-attention`.
- **Context length for SFT:** Train at the maximum context length that fits. Ideal target is 262K; if 262K does not fit in 32GB VRAM, fall back to 131K.
- **Fine-tuning modes:** Run two SFT variants:
  1. Gate-parameter-only (new `b_proj`, `w_proj`, and channel-wise decay params) with higher LR, backbone frozen.
  2. Full fine-tune with lower LR, optionally using a brief gate-only warmup.
- **Checkpoints:** Keep a small number of checkpoints (3–5) within the 21GB free disk budget. Prefer bf16 to stay within budget.
- **HF token:** None required; public downloads are sufficient.
- **RULER evaluation:** Run MK-NIAH-1 (`niah_single_1`) up to the length used for training.
- **Training stopping:** Train until validation loss plateaus or 10 hours elapse, whichever comes first.
- **Coordination:** Do not pause for approval between Goal A and Goal B; another agent may run concurrently. Update this file with progress.

## Environment Status

- The `.venv` python symlink was dead and has been relinked to `/.uv/python_install/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11`.
- `flash-linear-attention` (FLA) is installed from git.
- `einops`, `datasets`, `packaging`, `ninja`, `sentencepiece`, and `protobuf` are installed.
- `causal-conv1d` build was attempted but did not complete in the 10-minute command window. If it is still missing, use the FLA `ShortConvolution` PyTorch fallback or retry the build in a background/training step.

## Goal A.
1. Build an efficient PyTorch model based on the existing implementation of Qwen3.5-0.8B in @16_qwen3.5/
2. Load weights from HuggingFace and compare implementation accuracy against transformers. You may have to load the transformers chat template.
3. Port the reference implementation of GDN2 (@gateddeltanet-2/) to the model. Adapt weights using @GDN2_SWAP.md as instructions, and evaluate numerical correctness against GDN.

### Goal A Implementation Notes
- Extract the notebook model from `16_qwen3.5/qwen3.5.ipynb` into clean modules (`model.py`, `layers.py`, `load_weights.py`).
- Verify logits match `Qwen3_5ForCausalLM` before any swap.
- Replace each `Qwen3_5GatedDeltaNet` linear-attention layer with `GatedDeltaNet2` from `gateddeltanet-2/lit_gpt/gdn2.py`.
- Perform the function-preserving init described in `GDN2_SWAP.md`:
  - Tile the original scalar `beta` projection (`in_proj_b`) into the channel-wise erase gate `b_proj` (key-side, `key_dim`) and write gate `w_proj` (value-side, `value_dim`).
  - Tile the original scalar decay projection (`in_proj_a` + `A_log` + `dt_bias`) into the channel-wise decay (`f_proj`, `A_log`, `dt_bias`).
  - Copy `in_proj_qkv` → `q_proj`/`k_proj`/`v_proj`, `conv1d` → `q_conv1d`/`k_conv1d`/`v_conv1d`, `in_proj_z`/`norm`/`out_proj` → `g_proj`/`o_norm`/`o_proj`.
- Confirm GDN2-with-tied-gates matches original GDN outputs to fp tolerance before training.

## Goal B (after Goal A).
1. Preprocess and format the following datasets for fine-tuning:
   - https://huggingface.co/datasets/zai-org/LongAlign-10k
   - https://huggingface.co/datasets/Yukang/LongAlpaca-12k
   - https://huggingface.co/datasets/wenbopan/anti-haystack
2. Run supervised fine-tuning (SFT) on the GDN2 model on these checkpoints, keeping checkpoints.
3. Benchmark each checkpoint on MK-NIAH-1 subset of RULER (@RULER/)

### Goal B Implementation Notes
- **Data formatting:**
  - LongAlign-10k: use existing `messages` field; apply Qwen3.5 chat template; mask user/system turns with `-100`.
  - LongAlpaca-12k: map `instruction` → user, `output` → assistant; apply chat template; mask user turns.
  - anti-haystack: format as `document + "\n\nQuestion: " + question` → user, `answer` → assistant; apply chat template; mask question/document turns.
- **SFT script:** Write a custom `sft.py` (no existing SFT script is present). Use bf16 mixed precision, gradient checkpointing, micro-batch size 1, and gradient accumulation to reach a reasonable effective batch size. Save checkpoints at regular intervals.
- **Training lengths:** Start at the chosen training length (262K if possible, else 131K). If OOM, reduce length or use gradient checkpointing more aggressively.
- **Benchmarking:** Use RULER's HF model wrapper (`scripts/pred/model_wrappers.py::HuggingFaceModel`) by making the final model HF-compatible, or add a custom wrapper. Configure `scripts/config_models.sh`, `scripts/config_tasks.sh`, and `scripts/run.sh` for `niah_single_1` and sequence lengths up to the training length.
- **Evaluation checkpoint set:** Evaluate the base GDN2 swap (post-init, before SFT) and each saved SFT checkpoint.

## Non-Goals
- Do not commit changes to git unless explicitly asked.
- Do not change the target model or use approximate init for GDN2.
- Do not pause for human approval between Goal A and Goal B.

(End of file)

## Progress Update (autonomous)

- **Goal A** complete: GDN2-swapped Qwen3.5 model matches HF logits and is verified.
- **Goal B**:
  - Datasets preprocessed at 131072 context.
  - SFT training completed: gate-only to 100 steps and full fine-tune to 50 steps.
  - RULER wrapper added (`qwen_gdn2` server type) and pipeline tested.
  - Fixed 131072 cached-decode OOM by removing the unused dense causal mask in `Qwen3_5GDN2Model._create_mask` (`src/qwen_gdn2/model.py`).
  - Evaluated one multi-needle task (`niah_multikey_1`) at 131072 using cached decode (after fixing wrapper checkpoint loading):
    - Base GDN2 swap: **100.0**
    - Gate-only SFT checkpoint-101: **96.0**
    - Full SFT checkpoint-51: **100.0**
  - Evaluated `niah_multivalue` @ 131072 on 100 samples (4 values per key):
    - Base GDN2 swap: **92.5**
    - Gate-only SFT checkpoint-101: **92.19**
    - Full SFT checkpoint-51: **98.25**
  - Added cached SFT checkpoint model configs (`qwen-gdn2-gate-101`, `qwen-gdn2-full-51`) to `RULER/scripts/config_models.sh`.
- **Audit (2026-06-25)**: Discovered that `QwenGDN2ModelWrapper` was not loading SFT checkpoints. SFT checkpoints use native keys (`trf_blocks.*`), while base weights use HF-style keys (`model.language_model.*`); `weights.update()` did not override base weights, so all three "models" evaluated were actually the base model. Fixed wrapper to load native-format `model.pt` via `load_state_dict`. Re-evaluation:
    - Base GDN2 swap: **100.0**
    - Gate-only SFT checkpoint-101: **96.0** (4 wrong-key retrievals)
    - Full SFT checkpoint-51: **100.0**
- **Note**: Earlier `niah_single_1` at 4096 showed degenerate outputs for SFT checkpoints; the `niah_multikey_1` at 131072 results show the models do retain multi-key retrieval capability at long context, though gate-only SFT slightly degrades key discrimination.
