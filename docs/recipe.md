# Recipe: distillation and evaluation

The three training steps and the two evaluation suites, identical for every kernel.  What the steps do
to a swapped layer is measured in [gate_diagnostics.md](gate_diagnostics.md); the results are in
[results.md](results.md).

## Distillation (`linswap distill`)

Teacher: `gdn`, an exact copy of the original model.  Student: the swapped model.  Three steps on
packed generic web text (DCLM by default; `--text_data fineweb-edu` is the alternative), no instruction
data and no chat template anywhere, no supervised fine-tuning afterwards.  Both exact and inexact swaps
use the same recipe — for an exact swap the first step starts from zero loss and the run is a
teacher-matched continuation rather than a repair.

| step | loss | tokens | length | sequences/step | lr | trained |
|---|---|---|---|---|---|---|
| `layer` | L2 between each student mixer's output and the teacher's, on the teacher's own layer input, all layers in parallel | 100M | 512 | 32 | 1e-3 → 1e-5 cosine | the swapped layers |
| `kl` | KL(teacher ‖ student) on next-token distributions, exact, computed 2048 positions at a time | 500M | 512 | 96 | 1e-5 flat | all parameters |
| `ce` | plain next-token cross-entropy, no teacher (context extension) | 100M | 16384 | 96 (8 × 12) | 1e-5 flat | all parameters |

The `layer` step measures its own loss on one batch before training and skips itself when that loss is
below `--layer_skip_below` (default 1e-4).  An exact init starts it at bf16 noise (~1e-6) with nothing
to align, and Adam's normalised update at lr 1e-3 then walks the weights 18-20 % of their norm away from
a function-preserving solution: its own objective rises by three orders of magnitude, validation loss
degrades by ~0.16 nats, and the later steps do not recover it.  Skipping it is worth 9-11 points of 128K
distractor-needle accuracy for `gdn2`, `kda` and `rwkv7` (see [gate_diagnostics.md](gate_diagnostics.md)).
`gla`, `mamba2`, `swa` and `deltanet` start at 1.6e-2 to 1.1e-1 and run the step as before.

Adam(0.9, 0.95, 1e-8), clip 1.0, bf16.  Budgets are given in tokens and converted to optimizer steps;
`--stage_length` / `--stage_batch` / `--stage_micro` / `--stage_schedule` override any step.  Freezing
the MLPs or the embeddings in the KL step costs accuracy, so everything trains.  About 6 GPU-hours per
kernel at 0.8B on one L20X; the checkpoint under `outputs/<kernel>/distill/` is what gets evaluated.

### Weight precision: what the recipe can and cannot move

The student's weights are stored and updated in bf16 (`--param_dtype bfloat16`, the default and what
every published row used).  bf16 keeps 8 significant bits, so a weight of magnitude `w` sits on a grid
of spacing about `w / 128`, and an Adam step of at most the learning rate, 1e-5, is less than half a
grid step for any weight above ~2.5e-3: the update rounds back to the old value.  Measured against the
swap init:

| checkpoint | weights that changed at all: backbone / shared projections | `kda` `f_proj.1` entries carrying GDN's decay |
|---|---|---|
| `kda_nolayer`, 10,235 updates (500M `kl` + 99M `ce`), bf16 | 20.6 % / 21.4 % | 0.0 % |
| `gdn_nolayer`, same recipe, bf16 | 21.2 % / 22.2 % | — |
| `kda`, 63 `ce` updates only, bf16 | 21.0 % / 22.2 % | 0.0 % |
| `kda`, 63 `ce` updates only, float32 | 100 % / 100 % | 99.7 % |

The same fifth of the weights — the small ones — is all that moves, after 63 updates or after 10,235.
Weights initialised at exactly zero (the spare directions of a tiled gate) have a fine grid and do move.
`--param_dtype float32` keeps full-precision master weights; the forward pass stays bf16 under autocast,
validation included, and the checkpoint is float32 (`build_model` casts it back on load).

A pilot on the `ce` step alone, started from the swap init on the data stream the full recipe's `ce`
step sees (`linswap distill --kernel <k> --stages ce --seed 16317 [--param_dtype float32]`, 63 updates,
99M tokens, `outputs/pilot_ce_only/`); validation NLL on the usual 10 held-out sequences, init 3.090:

| run | `gdn` | `kda` | KL(teacher ‖ student) / top-1 agreement |
|---|---|---|---|
| `kl` 500M then `ce` 99M, bf16 (the recipe without `layer`) | 2.9171 | 2.9172 | not logged |
| `ce` 99M only, bf16 | 2.9158 | 2.9156 | 0.154 / 0.82 |
| `ce` 99M only, float32 weights | 2.8853 | 2.8854 | 0.196 / 0.80 |

Three things, each one seed and validation NLL only (no RULER, no short-context suite):

* **For an exact swap the `kl` step does not show up in validation NLL.**  The `ce` step alone lands
  where 500M tokens of `kl` followed by `ce` land, and its first training loss is the same to four
  digits (3.0310 vs 3.0311).  What `kl` buys on retrieval and answer format is not measured by this.
* **Precision is worth more than the kernel here.**  float32 weights gain a further 0.03 nats on the same
  data and move further from the teacher (KL 0.196 vs 0.154), which the dropped teacher-free
  continuation in [results.md](results.md) found can cost answer format — so it needs the full
  evaluation before it becomes the default.
* **`gdn` and `kda` are indistinguishable in both settings**, to 2e-4 nats at every evaluation.
  Unfreezing the weights does not by itself separate the kernels on web text.

## Evaluation

Two suites, both applied to the students *and* to the unmodified backbone so the comparison is
like-for-like:

* **Long context** (`linswap evaluate`): RULER's needle tasks `niah_single_1/2/3` and
  `niah_multikey_1`, plus `vt` (follow a 4-hop chain of variable assignments through noise) and `cwe` /
  `fwe` (name the most common words of a list / coded text — aggregation over the whole context rather
  than retrieval of one span), at 4K / 16K / 64K / 128K, 500 samples each, cached greedy decoding,
  scored as upstream RULER does (case-insensitive substring recall, no stop words).  Prompts use
  RULER's own base template (context, question, answer prefix); `--chat_template` switches to the
  backbone's chat format.  Since the students never see an instruction format during distillation,
  base prompting is the setting in which teacher and student are scored the same way.
* **Short context** (`linswap lmeval`): LAMBADA, ARC-c (acc_norm), ARC-e, PIQA, WinoGrande,
  HellaSwag (acc_norm) 0-shot and MMLU 5-shot through lm-eval-harness, reported both as accuracy and
  as a relative score (s − r)/(t − r) against a reference row, with r the chance level.  IFEval
  (prompt-level strict accuracy, plus instruction-level in `ifeval_inst`) runs generatively with the
  chat template, thinking disabled, as left-padded batches: full attention masks the pad keys and the
  linear layers run the prompt varlen (`cu_seqlens`), so a batch decodes like each prompt alone (`swa`,
  whose sinks are absolute positions, stays at batch size 1).

`tools/throughput.py` measures prefill and decode speed, `tools/hard_tables.py` turns evaluation logs
into markdown tables.  [gate_diagnostics.md](gate_diagnostics.md) measures what the training steps do to
the swapped layers: how far the tiled per-channel gates spread, and which step moves the weights.
[state_rank.md](state_rank.md) measures the rank of every head's memory state on real text.
