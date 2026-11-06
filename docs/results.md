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

### Qwen3.8-27B

The same comparison on the second backbone (48 Gated-DeltaNet layers of 64; 16 key heads shared by 48 value
heads), KL / top-1.  The noise floor is two orders of magnitude higher than on the 0.8B model: 48 swapped
layers in bf16, each within 1e-4 – 3e-4 of the backbone's own layer when compared alone in float32.
The image prompt of this table was built with the chat template's default, which on this backbone turns
thinking on; `linswap verify` now switches it off, as the evaluations do.

| kernel | text, 512 tokens | text, 4096 tokens | image chat prompt | greedy answer to the image prompt | swapped layers |
|---|---|---|---|---|---|
| `gdn` | 8.6e-3 / 0.998 | 1.4e-2 / 0.997 | 0.14 / 0.947 | identical to the teacher's | 5.56B parameters |
| `kda` | 5.4e-3 / 1.000 | 1.2e-2 / 0.997 | 0.043 / 0.932 | identical | 5.62B |
| `rwkv7` | 7.2e-3 / 0.996 | 9.4e-3 / 0.997 | 0.044 / 0.958 | identical | 5.68B |
| `gdn2` | 1.3e-2 / 0.998 | 1.2e-2 / 0.997 | 0.070 / 0.932 | identical | 12.08B |
| `mamba2` | 0.37 / 0.924 | 0.29 / 0.941 | 7.9 / 0.303 | 17 % of the tokens agree | 5.55B |
| `gla` | 0.42 / 0.924 | 0.62 / 0.874 | 7.7 / 0.136 | 75 % | 7.58B |
| `deltanet` | 1.3 / 0.740 | 7.2 / 0.116 | 8.2 / 0.360 | 12 % | 7.57B |
| `swa` | 11.9 / 0.020 | 11.6 / 0.004 | 11.1 / 0.019 | none | 5.54B |
| `mamba3` | 12.9 / 0.000 | 13.3 / 0.000 | 10.8 / 0.000 | none | 5.58B |

* The three exact kernels are at the floor here too.  `gdn2`, `gla` and `deltanet` have no grouped value
  heads and get one key head per value head with the key projections repeated (`kernels/common.py`), which
  is what makes `gdn2` exact and what makes these three larger; `gdn2` more than doubles the swapped layers.
* On text the kernels without an erase start far closer than on the 0.8B model (`mamba2` 0.37 against 5.7),
  on images just as far.
* This backbone is supported for use — `build_model`, `linswap verify` and `linswap eval` run on it — but it
  is not distilled or benchmarked here.  `linswap distill` as it stands would not fit it on one 140 GiB GPU
  (an estimate, not a run): the float32 master weights, their gradients and the two Adam moments of 5.6B
  parameters take 89 GB next to two copies of the 51 GiB model.

## Distillation

The gap is KL(teacher ‖ student) on held-out text, chat, image chats and a 32K sequence (mean of the four).

| model | training | GPU-hours | gap | text | chat | image | long |
|---|---|---|---|---|---|---|---|
| `kda`, step 0 | none | 0 | 0.001 | 0.001 | 0.001 | 0.002 | 0.001 |
| `gla`, earlier recipe | 700M tokens | 6.0 | 0.553 | 0.160 | 0.194 | 1.699 | 0.160 |
| `mamba2`, earlier recipe | 700M tokens | ~6 | 0.425 | 0.159 | 0.165 | 1.210 | 0.167 |
| `deltanet`, earlier recipe | 700M tokens | 5.6 | 0.491 | 0.200 | 0.255 | 1.117 | 0.391 |
| `swa`, earlier recipe | 700M tokens | ~6 | 0.604 | 0.190 | 0.275 | 1.760 | 0.191 |

The runs of this recipe (`linswap distill`, swapped layers only, cap 100M tokens), gap at the milestones:

| kernel | step 0 | 3M (end of local) | 12.5M | 33M | 100M | GPU-hours | text / chat / image / long at 100M |
|---|---|---|---|---|---|---|---|
| `gla` | 5.01 | 0.377 | 0.134 | 0.106 | **0.077** | 1.35 | 0.083 / 0.061 / 0.099 / 0.066 |
| `deltanet` | 10.95 | 1.049 | 0.261 | 0.172 | **0.107** | 1.30 | 0.129 / 0.088 / 0.084 / 0.126 |
| `swa` | 11.77 | 0.569 | 0.197 | 0.153 | **0.118** | 1.76 | 0.149 / 0.096 / 0.112 / 0.115 |

* Each run reached the cap: after one learning-rate cut the gap still fell by more than 5 % per milestone,
  so none was stopped by the plateau rule.  The return is steeply diminishing — from 33M to 100M tokens,
  two thirds of the compute, `gla` goes from 0.106 to 0.077.
* Against the earlier recipe (first table): about a quarter of the GPU time, a gap 4.6 to 7 times smaller, and
  the image gap in line with the others instead of an order of magnitude above them.
* `mamba2` and `mamba3` are still training as of 2026-11-06.  `mamba2`'s first run hung at 17M tokens inside
  FLA's TileLang backward kernel for simple-GLA; its recurrence now runs on the GLA kernels
  (`kernels/mamba2.py`) and the run was restarted.

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
