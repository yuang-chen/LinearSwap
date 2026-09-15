# Changelog

## Unreleased

* **RULER protocol fix.**  The vendored `pred/call_api.py` never passed RULER's per-sample
  `answer_prefix` to the model; `LinearSwapModelWrapper` now opens the assistant turn with it
  (after the chat template, `enable_thinking=False` for thinking backbones).  Variable
  tracking on the GDN base at 4K moves from 3.6 to 91.2.  All hard-task tables were regenerated.
* `evaluate --nll pg19,wikitext` (token-weighted raw-text NLL with position bins),
  `linswap lmeval` (lm-eval-harness on exported HF checkpoints) and `linswap mqar` (text
  multi-query associative recall); `--datasets` subset selection for SFT / distill / evaluate;
  `distill --kl_schedule` (packed long-context KL stage).
* `tests/test_losses.py`: the chunked cross-entropy and distillation KL — which run their own
  backward, so no framework checks them — against dense autograd, on a stand-in with the LM head
  tied to the embedding (the configuration the old loss-scaling bug depended on).  Loss,
  tied-matrix gradient and hidden-path gradient, over uneven chunk sizes, masked prompts, right
  padding, `loss_scale`, KL temperature and an all-masked microbatch.  CPU, no checkpoint needed.
* `KernelSpec.supports_activation_checkpointing` (False for `mamba3`, whose `mamba_ssm`
  autograd functions do not compose with `torch.utils.checkpoint`).
* `gdn2` refuses backbones with grouped value heads: FLA's `GatedDeltaNet2` shares its decay and
  erase gates across a value-head group, so the tiled init has no exact image there.
* `evaluate` resolves checkpoint and base-model paths to absolute before handing them to RULER
  (RULER runs from its own directory); the RULER wrapper resolves repo-relative paths as well.
* Second backbone scale: Qwen3.8-27B (exactness in fp32 at KL 2.6e-6; gate-only SFT and a RULER
  subset for `kda` / `rwkv7`; see `docs/framework.md`).

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
