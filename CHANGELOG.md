# Changelog

## 0.1.0 — 2026-09-13

First packaged release.

* `pip install -e .` and the `linswap` command (`verify`, `distill`, `posttrain`, `evaluate`, `run`, `export`, `kernels`).
* Kernels: `gdn` (control), `gdn2`, `kda`, `kda_fullgate`, `rwkv7` (exact, function-preserving init);
  `mamba2`, `deltanet`, `gla`, and — with `mamba_ssm` — `mamba3`, `mamba1` (inexact; distilled first).  Stock FLA layers register through
  `kernels/fla_layer.register_fla_kernel`, FLA ops through `kernels/base.BackboneMixer`.
* Backbone read from the HF `config.json` (Qwen3-Next / Qwen3.5 / 3.6 / 3.8 layouts; dense models);
  grouped value heads supported by all kernels except `gla` and `deltanet`.
* Hugging Face `transformers` integration: `LinearSwapForCausalLM` (`PreTrainedModel` + `GenerationMixin`),
  Qwen's checkpoint layout (`model.layers.{i}.{self_attn,mlp,linear_attn}`, `lm_head`), `linswap export`.
* Batches: right-padded batches for loss / logits, equal-length batches for generation.
* Distillation stage (layer alignment + chunked KL) for inexact kernels; SFT recipe with gate-only / full modes.
* RULER evaluation driven from `linswap evaluate`; results for Qwen3.5-0.8B in `docs/framework.md`.
* Tests: `tests/test_kernels.py`, `tests/test_hf.py`, `tests/test_batch.py`, `tests/test_gva.py`.
* Environment: torch ≥ 2.7 / Triton ≥ 3.3 (FLA's requirement); tested on torch 2.9.1 / Triton 3.5.1.

Earlier history (unpackaged): the GDN → GDN-2 in-place swap experiment this project grew out of
(`docs/gdn2_experiment_log.md`).
