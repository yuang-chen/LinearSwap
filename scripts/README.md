# scripts

All scripts add `src/` to `sys.path` themselves; run them from anywhere with the
`.venv` Python.  Every script prints `--help`.

| script | purpose |
|---|---|
| `prepare_datasets.py` | Download LongAlign-10k / LongAlpaca-12k / anti-haystack, apply the Qwen chat template, mask non-assistant tokens, left-truncate to `--max_length`, save `data/sft/len{max_length}/{train,validation}` |
| `verify.py` | `--kernel X [--baseline gdn] [--ckpt DIR]` — function-preservation checks vs HF Qwen3.5: `layer` (pretrained layers in isolation vs transformers' `Qwen3_5GatedDeltaNet`), `logits` (several lengths), `layerwise` (per-block diff), `cache` (cached vs no-cache decode), `generation` |
| `sft.py` | `--kernel X --mode gate_only|full` — bf16 SFT with gradient checkpointing and chunked CE; `gate_only` trains the kernel's `new_param_names`; checkpoints (`model.pt`, `optimizer.pt`, `config.json` with `linear_kernel`) under `--output_dir/checkpoint-N`, `train_log.jsonl` with loss / grad-norm / val loss |
| `eval_val_loss.py` | `--models NAME_OR_CKPT ...` — validation cross-entropy and perplexity on the same examples for each model |
| `register_ruler_model.py` | `--name N (--kernel X | --ckpt DIR)` — write `outputs/ruler_models/N/config.json`; then `cd RULER/scripts && bash run.sh linswap-N synthetic` (or `linswap-nocache-N`) |

RULER's `run.sh` calls bare `python`, so `source .venv/bin/activate` first.  Tasks
are listed in `RULER/scripts/config_tasks.sh` (`synthetic=(...)`, `NUM_SAMPLES`),
sequence lengths in `RULER/scripts/config_models.sh` (`SEQ_LENGTHS`).
