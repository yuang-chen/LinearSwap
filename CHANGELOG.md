# Changelog

## Unreleased

* **New goal: the student approaches the teacher as efficiently as possible** (AGENTS.md); beating the
  teacher is dropped.  The project was rebuilt around it and is about half its former size.
* **One model.**  A swapped model is `transformers`' own Qwen3.5 vision-language model with only each
  layer's `linear_attn` replaced (`linswap.model`), for training and evaluation alike, so text, images and
  the standard harnesses work as with the backbone.  The project's own rebuilt backbone, its HF wrapper and
  `export` are gone.  A checkpoint is `swap.safetensors` (the swapped layers, bf16) plus `linswap.json`;
  checkpoints of the earlier format (`model.pt`) still load.
* **`linswap distill` is new** (docs/recipe.md): only the swapped layers are trained, against the teacher's
  activations, on web text, chat and image chats; a local layer-matching phase, then a closed loop on
  hidden states; float32 master weights saved as bf16; milestones measure the gap to the
  teacher and end the run when it stops closing.  Exact kernels are not trained.  Gone: the `layer` / `kl` /
  `ce` steps, `--ce_kl_weight`, `--replay_data`, `--new_lr_mult`, `--param_dtype`, resume with optimizer
  state.  The curves go to `train_log.jsonl` and, with `tensorboard` installed, to TensorBoard
  (`tensorboard --logdir outputs/distill`): loss, gradient norm and learning rate per step, the gap with its
  four parts and top-1 per milestone, against training tokens.
* **`linswap eval` replaces `evaluate`, `lmeval`, `mmeval` and `run`**: lm-evaluation-harness for text,
  RULER's scripts for long context (now in the chat template), lmms-eval for images, each on its full test
  set; `--smoke` for a functional check.  The hand-written image benchmark code is gone.
* **`linswap verify`** compares with the unmodified upstream model on text and on an image prompt, with
  `gdn` as the noise floor, and accepts a checkpoint.
* **Kernels: `gdn`, `rwkv7`, `kda`, `gdn2`, `mamba2`, `deltanet`, `gla`, `swa`.**  `mamba1`, `mamba3` and
  `gdn_breg` are removed, as are kernel maps (mixed-kernel models).  `mamba2` runs its kernel in float32,
  which is what lets it train on Hopper GPUs (fla#640).
* **Removed:** the state-rank and gate-diagnostics tools and notes, the throughput and table tools, the
  quickstart notebook, `docs/framework.md` (now in the README), and the earlier tests (one functional test,
  `tests/test_linswap.py`, covers the path).

* **The recipe trains about a fifth of the weights: the rest cannot move in bf16.**  The student's weights
  are stored and updated in bf16, and an lr 1e-5 Adam step is below half a bf16 grid step for any weight
  above ~2.5e-3 in magnitude, so it rounds away.  In the finished `gdn` / `kda` runs (10,235 updates, 599M
  tokens) 79 % of the backbone and shared-projection weights are bit-identical to their init, and the
  entries of `kda`'s `f_proj.1` that carry GDN's decay did not change at all.  Every published row was
  trained this way, control included, so the rows stay comparable with each other; what the tables say
  about how little the gates move needs re-measuring.  `distill --param_dtype float32` keeps
  full-precision master weights (same bf16 forward pass, float32 checkpoint); the default is unchanged
  until the recipe is re-run with it.  See `docs/recipe.md`.
* **`linswap.upstream`: the swap inside `transformers`' own Qwen3.5 model**, text-only or
  vision-language.  Only `linear_attn` is replaced (an adapter for the upstream decoder layer's mixer
  call plus a cache bridge); vision tower, merger, multimodal RoPE, processor and `generate` stay
  upstream.  `build_upstream_model(kernel | ckpt_dir=...)` also loads a text-distilled checkpoint into
  the full model.  `tests/test_upstream.py`: on a processed image the exact kernels match the native
  model at the control's noise level, and on text the logits are bit-identical to `build_model`'s.
  Inference only, in memory, greedy / sampling; needs the new `vision` extra (torchvision).
* **`tokens` in `train_log.jsonl` was wrong and is now the tokens actually consumed.**  It was the global
  step times the *current* step's tokens per update, so the default recipe ended at 25.7B (16.1B without the
  `layer` step) for 699M (599M) consumed.  The training loop and its data were never affected, only the
  logged number; ignore it in logs written before this change.  Every record now carries `updates`,
  `tokens` and `teacher_tokens`, a skipped `layer` step adds nothing to them, each checkpoint saves them
  in `train_state.json`, and `step` / `checkpoint-N` stay the plan's nominal numbering.
* **`--init_from` continues a step with its optimizer state.**  A checkpoint written inside a step holds
  `optimizer.pt` (the newest checkpoint only; it is twice the size of the weights, and a finished run keeps
  none); resuming used to restart AdamW's moments.  `tests/test_resume.py` compares an interrupted and
  resumed run with the same run left alone.
* **`distill` refuses a batch its micro-batch does not divide** (`--stage_batch ce:96 --stage_micro ce:10`
  silently trained on 90 sequences per step).
* **`distill` writes `manifest.json`**: git revision, package versions (with the git commit of
  `flash-linear-attention`), base-model and tokenizer hashes, corpus, and the stage plan with token totals.
* **A prepared corpus is tied to its tokenizer.**  `info.json` records the tokenizer's sha256 and
  `ensure_text_data` stops when `--base_model_dir` has a different one, instead of feeding another
  backbone token ids it reads differently.  Existing corpora get the fingerprint on first use.
* **`verify` judges exact kernels.**  With `--baseline`, an `exact_init` kernel must stay within
  `--tolerance` (3×) of the baseline's layer and logit errors and `--agreement_slack` of its cache /
  generation agreement; it returns and (`--json`) writes the measurements with `pass` / `fail` /
  `not_gated`, and exits non-zero on `fail`, which stops `linswap run`.
* **The teacher's decoder leaves the GPU before the `ce` step**, which never runs it; validation keeps
  its cached hidden states and head.
* `linswap run --control name=checkpoint` evaluates the same-recipe `gdn` control next to the student.
* Docs: the chunked losses run over token positions with the full vocabulary (not "vocabulary chunks");
  the README states that a swapped model is the language backbone of a vision-language checkpoint and
  splits the kernels' gate-group sizes from their net parameter change.

* **RULER scores measured before this change understate the unmodified backbone.**  The RULER wrapper now
  stops at `<|endoftext|>` as well as `<|im_end|>`.  On base prompts the backbone answers, emits
  `<|endoftext|>` and often continues into a generated chat turn with a `<think>` block; the wrapper kept
  only the text after `</think>`, which deleted the answer.  On the backbone (100 samples per cell)
  `niah_single_2` goes from 65 to 100 at 4K and 73 to 100 at 16K, `niah_multikey_1` from 85 to 100 and 80
  to 100, and `vt` is unchanged.  The distilled `gdn` control is unaffected on `niah_single_2` and
  `niah_multikey_1` at 4K (40 samples): it never emitted `<|endoftext|>`.  Other distilled kernels were not
  checked.  The backbone rows in the README and `docs/results.md` predate the fix, so do not compare them
  with new runs until they are re-measured.  `--batch_size` above 1 is not bit-identical to batch size 1:
  in the same comparison 3-28 % of predictions changed, with scores within 2 points.
* **IFEval in `linswap lmeval`** (default task set, `--chat_tasks ifeval`): generated with the chat
  template and thinking disabled, scored by prompt-level strict accuracy (`ifeval`) and
  instruction-level strict accuracy (`ifeval_inst`), chance 0 in the relative score.  Needs `langdetect`
  and `immutabledict` (added to the `eval` extra).  `--fewshot_tasks` entries not listed in `--tasks`
  are no longer evaluated.
* **Left-padded batches in the HF wrapper**, so `generate` takes prompts of different lengths (IFEval
  runs at `--batch_size`): the padding mask reaches full attention (pad keys hidden) and the linear
  layers, which unpad the prompt to varlen `cu_seqlens` (FLA layers natively, `BackboneMixer` kernels
  — `rwkv7`, `mamba2` — now too).  Not `swa`.
* **HF export writes a generation config** with the chat end-of-turn and backbone EOS ids (`<|im_end|>`,
  `<|endoftext|>`) and the pad id.  Without it `generate` never stopped a row, so batched IFEval scored the
  text finished rows kept generating after `<|im_end|>`.  Delete cached `outputs/eval/<name>/hf/` exports
  to pick it up.
* **Removed the `kda_fullgate` kernel.**  `kda` keeps the rank-128 forget-gate bottleneck Kimi Linear
  specifies; the dense-gate variant was a repo-only ablation and no longer ships, nor do its rows in the
  README and docs.
* **RULER `vt`, `cwe`, `fwe` join the default evaluation tasks** (variable tracking, common / frequent
  words extraction), scored as upstream RULER and lm-eval score them (case-insensitive substring
  recall, full generation budget, no stop words).
* `evaluate` falls back to `--base_model_dir` when a checkpoint's recorded `base_model_dir` does not
  exist (a checkpoint trained on another machine) and passes the resolved directory to the RULER
  wrapper (`LINSWAP_BASE_MODEL_DIR`); a task whose data RULER fails to generate is reported as failed
  instead of surfacing as a missing file, and a length where every task failed no longer aborts the run.
* RULER subprocesses cap the tokenizer (rayon) and OpenMP pools at 4 threads (`RAYON_NUM_THREADS`,
  `OMP_NUM_THREADS`, overridable): several evaluations in parallel on a 224-core host exhausted thread
  creation.  At 128K a `--ruler_jobs 3` evaluation peaks near 30 GiB per process (`rwkv7`), so three of
  them on one 140 GiB GPU can run out of memory.
* **TensorBoard dashboard for `distill`** (`linswap.tb_logger`, `--tensorboard_dir`, `--no_tensorboard`,
  optional extra `.[tensorboard]`): loss, grad norm, lr and, in the `layer` and `kl` steps, the MSE of every
  linear-attention layer's output against the teacher's (`layer_mse/`, each model on its own hidden states)
  every step, with the gradient norm split
  into the kernel's new parameters, the swapped layers' shared projections and the rest of the backbone;
  every `--eval_every` steps the validation loss, KL(teacher ‖ student), top-1 agreement and entropies
  on the held-out sequences (also per position bucket), and each group's parameter norm, drift from the
  swap init and update size.  A resumed run purges the events past its checkpoint and measures drift
  from the swap init, not the checkpoint.  The same scalars go into `train_log.jsonl`.
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
* **The `layer` step skips itself when there is nothing to align** (`--layer_skip_below`, default
  1e-4).  An exact init starts that step at bf16 noise (~1e-6), and Adam's normalised update at
  lr 1e-3 walks the weights 18-20 % of their norm away from a function-preserving solution: its own
  objective rises three orders of magnitude and validation degrades ~0.16 nats.  Skipping it is worth
  9-11 points on the 128K distractor needle (`gdn2` 86.8 -> 96.4, `kda` 85.6 -> 96.4, `rwkv7`
  86.6 -> 96.2, control 96.8) and puts every exact-init kernel on the
  control on both suites.  `gla`, `mamba2`, `swa` and `deltanet` start at 1.6e-2 to 1.1e-1 and run it.
* **Two earlier conclusions are retracted.**  "An exact init costs 4-11 points at 128K" and "KDA's
  low-rank decay gate costs retrieval" were both artifacts of that step; with it skipped the
  exact-init kernels are indistinguishable from the control and from each other.  The per-channel
  gate spread they showed (S = 0.09-0.30) was drift from the same step, not learned structure: the
  better checkpoints sit at 0.005-0.011.  `docs/gate_diagnostics.md` has the measurements.
* **RULER results are now at 500 samples per task** (the standard count) instead of 50, for the
  teacher, the `gdn` control, `gdn2`, `kda`, `rwkv7`, `mamba2`, `swa`, `gla` and
  `deltanet` — every row in both tables.  At 50 samples the binomial
  95% interval is about ±7 points and the exact-init kernels were an undifferentiated block of
  100s; at 500 it is about ±3 and every exact init is measurably below the control on the 128K
  distractor needle.
* `mamba1` and `mamba3` results are removed from both tables — neither is being carried forward.
  The kernels stay registered and documented.
* Results for three more kernels in the README and `docs/framework.md`: `gdn2` (109.9 relative, on the
  control), `swa` (101.0) and `mamba1` (79.6).  `gdn_breg` (λ = 0.01 / 0.003, with and without layer 0
  thresholded) was measured too but is held out of the docs pending a re-run at RULER's standard
  sample count; the runs are under `outputs/breg*` and `outputs/eval/breg*`.
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
* Kernels: `gdn` (control), `gdn2`, `kda`, `rwkv7` (exact, function-preserving init);
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
