# Results

Qwen3.5-0.8B, one seed.  The recipe is in [recipe.md](recipe.md), the per-kernel mappings in
[kernels.md](kernels.md).  The teacher is the unmodified backbone.

## Step 0: how far each swap starts from the teacher

`linswap verify`: KL(teacher ‖ swapped) of the next-token distribution and top-1 agreement, before any
training.  `gdn` is the same recurrence on FLA's kernels, i.e. the numerical noise of the comparison.

| kernel | text, 512 tokens | text, 4096 tokens | image chat prompt | greedy answer to the image prompt |
|---|---|---|---|---|
| `gdn` | 7.5e-5 / 1.000 | 9.8e-6 / 1.000 | 2.7e-3 / 0.973 | identical to the teacher's |
| `kda` | 8.8e-5 / 1.000 | 1.1e-5 / 1.000 | 2.4e-3 / 0.973 | identical |
| `gdn2` | 9.0e-5 / 1.000 | 1.2e-5 / 1.000 | 2.8e-3 / 0.973 | identical |
| `rwkv7` | 8.0e-5 / 1.000 | 1.0e-5 / 1.000 | 3.8e-3 / 0.969 | identical |
| `mamba2` | 5.7 / 0.250 | 4.1 / 0.293 | 4.7 / 0.205 | 4 % of the tokens agree |
| `deltanet` | 6.7 / 0.242 | 12.7 / 0.038 | 4.5 / 0.402 | none |
| `gla` | 6.5 / 0.139 | 12.2 / 0.023 | 5.4 / 0.237 | none |
| `swa` | 13.9 / 0.000 | 14.7 / 0.000 | 11.2 / 0.000 | none |

The three exact kernels are the teacher to numerical noise on text and on images, so they need no
training; the other four start as noise.

## Distillation

The gap is KL(teacher ‖ student) on held-out text, chat, image chats and a 32K sequence (mean of the four).

| model | training | GPU-hours | gap | text | chat | image | long |
|---|---|---|---|---|---|---|---|
| `kda`, step 0 | none | 0 | 0.001 | 0.001 | 0.001 | 0.002 | 0.001 |
| `gla`, earlier recipe | 700M tokens | 6.0 | 0.553 | 0.160 | 0.194 | 1.699 | 0.160 |
| `mamba2`, earlier recipe | 700M tokens | ~6 | 0.425 | 0.159 | 0.165 | 1.210 | 0.167 |
| `deltanet`, earlier recipe | 700M tokens | 5.6 | 0.491 | 0.200 | 0.255 | 1.117 | 0.391 |
| `swa`, earlier recipe | 700M tokens | ~6 | 0.604 | 0.190 | 0.275 | 1.760 | 0.191 |
| `gla`, this recipe (pilot, 20M-token cap) | 20M tokens | 0.25 | 0.129 | 0.150 | 0.115 | 0.125 | 0.128 |

The full runs of the four inexact kernels (up to 100M tokens, stopped by the gap) are in progress as of
2026-11-06; this table will hold their curves.

## Standard test sets

`linswap eval` on the full test sets (lm-evaluation-harness, RULER, lmms-eval) for the teacher, the four
exact kernels at step 0, the four distilled kernels and the four checkpoints of the earlier recipe is running
as of 2026-11-06 (`outputs/eval/main/results.md`); the tables will be added here when it has finished.

## Approaches that were tried and dropped

From the earlier recipes (layer alignment → logit KL → cross-entropy, and before).  Kept as a record; the
code for them is no longer in the tree.

* **Supervised fine-tuning on long-context chat data** (LongAlign / LongAlpaca / anti-haystack), both
  full and gate-only.  50 steps equalised every exact kernel, and the retrieval-flavoured data cost
  common-word extraction (77 → 65 at 4K, 46 → 9 at 128K) and QA at long context.  Gate-only SFT never
  beat full SFT.  A no-anti-haystack ablation scored the same, so the loss came from the format, not
  from one subset.
* **Distilling on that chat corpus** instead of generic text: ~8M tokens of layer + KL.  Replacing it
  with the generic-text recipe was worth 11–15 hard-task points for Mamba-2 at every length and 16–27
  points of short-context recall for DeltaNet.
* **Continued training on generic text without a teacher** (400M tokens, with and without 10 %
  instruction replay): it lowers perplexity but does not beat distillation, and without replay it
  drifts the instruct backbone out of its answer format (multi-key needle 100 → 46 at 4K).
* **A long-context KL curriculum** (packed 8K → 64K) and **hidden-state alignment as a separate step**:
  no measurable gain over the three steps above.
* **Equal-budget comparisons of `gdn` vs `gdn2` vs `kda`** (400M tokens each): identical to three
  decimals on perplexity and within 0.9 points on every task — a richer gate does not beat GDN at this
  scale and budget.  The earlier claim that GDN2 beats GDN came from a run with a loss-scaling bug.
* **Per-layer swap sensitivity and mixed-kernel models**: the machinery stays (kernel maps such as
  `"gdn;mamba2@3,6"`, partial checkpoint loading) but the search stage was removed.  Result worth
  remembering: six blockwise-distilled DeltaNet layers are nearly free, the ninth is a cliff.
* **A second scale (27B)**: verified function-preserving in fp32 (KL 2.6e-6), but the only
  post-training that fitted the hardware was gate-only SFT, which is gone with the rest of the SFT
  stage.
