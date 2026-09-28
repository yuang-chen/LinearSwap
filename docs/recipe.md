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
| `kl` | KL(teacher ‖ student) on next-token distributions, in vocabulary chunks | 500M | 512 | 96 | 1e-5 flat | all parameters |
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
  as a relative score (s − r)/(t − r) against a reference row, with r the chance level.

`tools/throughput.py` measures prefill and decode speed, `tools/hard_tables.py` turns evaluation logs
into markdown tables.  [gate_diagnostics.md](gate_diagnostics.md) measures what the training steps do to
the swapped layers: how far the tiled per-channel gates spread, and which step moves the weights.
[state_rank.md](state_rank.md) measures the rank of every head's memory state on real text.
