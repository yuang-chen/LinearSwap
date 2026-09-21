# Changelog

## Unreleased

* **The pipeline is now verify → distill → evaluate.**  Distillation is three steps on generic web text
  (DCLM by default): layer-output alignment (100M tokens, length 512, lr 1e-3 cosine, swapped layers only),
  logit KL (500M tokens, length 512, lr 1e-5, all parameters) and context extension by plain
  cross-entropy (100M tokens, length 16384).  The defaults are the recipe; `linswap run --kernel rwkv7`
  needs no flags.
* **Evaluation defaults**: RULER `niah_single_1/2/3` + `niah_multikey_1` at 4K–128K with RULER's base
  prompt template (`--chat_template` opts back in), and `linswap lmeval` with LAMBADA / ARC-c / ARC-e /
  PIQA / WinoGrande / HellaSwag 0-shot + MMLU 5-shot and relative scores against a reference row.
* **Removed**: the supervised fine-tuning stage (`linswap posttrain`, both gate-only and full) and its
  chat corpora (`linswap.data`); raw-text NLL (`evaluate --nll`) and the MQAR probe (`linswap mqar`);
  the per-layer sensitivity search (`linswap sensitivity`) — the kernel-map machinery it used stays;
  the `hidden` distillation step, the long-context KL curriculum and instruction replay; the
  `mamba2_beta` and `mamba3_min` control kernels; the pre-September checkpoint-key converter.
* `linswap.sft_utils` is now `linswap.train_utils`.
* `distill --init_from <checkpoint-N>` (and `run`) resumes a run: steps already finished are skipped, a
  partial one continues for its remaining optimizer steps on the same data (the loader is fast-forwarded
  past the consumed sequences); optimizer state restarts.  `--init_step` overrides the step parsed from
  the directory name.
* `evaluate` puts the repo root on `PYTHONPATH` for the RULER subprocesses, so repo-local kernel
  packages (`gated_breg_delta_rule`) register there too; before, `gdn_breg` checkpoints produced no
  predictions.
* Results for five more kernels in the README and `docs/framework.md`: `gdn2` (109.9 relative, on the
  control), `swa` (101.0), `mamba1` (79.6) and `gdn_breg` at λ = 0.01 / 0.003, each with every GDN
  layer thresholded and with layer 0 left as a plain GDN layer (`--kernel 'gdn_breg;gdn@0'`).
* `evaluate` / `lmeval` sanitise model labels before using them as directory names (`path_slug`):
  RULER builds shell command strings, so a kernel-map label such as `gdn_breg;gdn@0-distilled`
  truncated them at the `;` and no prediction files were written.


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
