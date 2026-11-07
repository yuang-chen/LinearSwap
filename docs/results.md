# Results

Qwen3.5-0.8B, one seed unless said otherwise.  The recipe is in [recipe.md](recipe.md), the per-kernel
mappings in [kernels.md](kernels.md).  The teacher is the unmodified backbone.

Where the numbers come from (all under `outputs/`, which is not in the repository):

| record | what |
|---|---|
| `verify/Qwen3.5-0.8B/` | step-0 comparison of every kernel, and of the distilled checkpoints |
| `distill/gla`, `distill/mamba2` | the runs of the recipe as it is now |
| `distill/_prefix/` | the runs made before the faults listed under [Distillation](#distillation) were fixed |
| `ab/` | five 20M-token `gla` runs: batch weight, learning-rate decay, seeds |
| `eval/main` | standard test sets for the teacher, the exact kernels and the **earlier** distilled checkpoints; stopped before it finished |
| `eval/main2` | text and image suites for the two re-distilled checkpoints, with per-item records |
| `eval/check-kda` | `kda` rerun at the present code on parts of `main`, and its per-item text records |

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
| `mamba3` | 7.0 / 0.109 | 5.3 / 0.175 | 4.2 / 0.214 | none |
| `deltanet` | 6.7 / 0.242 | 12.7 / 0.038 | 4.5 / 0.402 | none |
| `gla` | 6.5 / 0.139 | 12.2 / 0.023 | 5.4 / 0.237 | none |
| `swa` | 13.9 / 0.000 | 14.7 / 0.000 | 11.2 / 0.000 | none |

`kda`, `gdn2` and `rwkv7` are the teacher to numerical noise on text and on images, so they need no
training; the other five start as noise.  (An untrained `deltanet` or `swa` also decodes differently with
and without the cache, 21 % and 92 % of the tokens agreeing: argmax over noise.  Their distilled
checkpoints, like `gla`'s and `mamba2`'s, agree on all tokens.)

This comparison tests exactness.  It is not the distance of a distilled checkpoint: the passage is two
sentences repeated and the image row counts the image-token positions (see `linswap/verify.py`).

### Qwen3.8-27B

No output of these runs was kept, so this table has no saved record and was not repeated for this
revision.  The same comparison on the second backbone (48 Gated-DeltaNet layers of 64; 16 key heads shared by 48 value
heads), KL / top-1.  The noise floor is two orders of magnitude higher than on the 0.8B model: 48 swapped
layers in bf16, each within 1e-4 – 3e-4 of the backbone's own layer when compared alone in float32.

| kernel | text, 512 tokens | text, 4096 tokens | image chat prompt | greedy answer to the image prompt | swapped layers |
|---|---|---|---|---|---|
| `gdn` | 8.6e-3 / 0.998 | 1.4e-2 / 0.997 | 0.074 / 0.946 | identical to the teacher's | 5.56B parameters |
| `kda` | 5.4e-3 / 1.000 | 1.2e-2 / 0.997 | 0.057 / 0.920 | identical | 5.62B |
| `rwkv7` | 7.2e-3 / 0.996 | 9.4e-3 / 0.997 | 0.095 / 0.929 | identical | 5.68B |
| `gdn2` | 1.3e-2 / 0.998 | 1.2e-2 / 0.997 | 0.051 / 0.938 | identical | 12.08B |
| `mamba2` | 0.37 / 0.924 | 0.29 / 0.941 | 7.5 / 0.263 | none | 5.55B |
| `gla` | 0.42 / 0.924 | 0.62 / 0.874 | 8.2 / 0.067 | 50 % of the tokens agree | 7.58B |
| `deltanet` | 1.3 / 0.740 | 7.2 / 0.116 | 7.1 / 0.429 | none | 7.57B |
| `mamba3` | 0.67 / 0.861 | 1.09 / 0.780 | 8.4 / 0.317 | none | 5.58B |
| `swa` | 11.9 / 0.020 | 11.6 / 0.004 | 10.9 / 0.009 | none | 5.54B |

* The three exact kernels are at the floor here too.  `gdn2`, `gla` and `deltanet` have no grouped value
  heads and get one key head per value head with the key projections repeated (`kernels/common.py`), which
  is what makes `gdn2` exact and what makes these three larger; `gdn2` more than doubles the swapped layers.
* On text the kernels without an erase start far closer than on the 0.8B model (`mamba2` 0.37 against 5.7),
  on images just as far.
* This backbone is supported for use — `build_model`, `linswap verify` and `linswap eval` run on it
  (`linswap eval --smoke` gives the teacher and `kda` the same scores on all three suites) — but it is not
  distilled or benchmarked here.  `linswap distill` as it stands would not fit it on one 140 GiB GPU
  (an estimate, not a run): the float32 master weights, their gradients and the two Adam moments of 5.6B
  parameters take 89 GB next to two copies of the 51 GiB model.

## Distillation

The gap is KL(teacher ‖ student) of the next-token distribution on held-out text, chat, image chats (8 of
each of the five subsets) and a 32K sequence, the mean of the four; image-token positions are left out.
The teacher of this measurement is the `gdn` swap, so `gdn` itself is at 0 by construction.

| model | training | GPU-hours | gap | text | chat | image | long |
|---|---|---|---|---|---|---|---|
| `kda`, step 0 | none | 0 | 0.0010 | 0.0008 | 0.0007 | 0.0018 | 0.0008 |

(`gdn2` and `rwkv7` were not measured this way; `linswap verify` puts them at the same noise.)

### The recipe as it is now

`linswap distill`, swapped layers only, 100M tokens, each run alone on its GPU; gap at the milestones:

| kernel | step 0 | 3M (end of local) | 12.6M | 33M | 98M | 100M | GPU-hours | text / chat / image / long at the end |
|---|---|---|---|---|---|---|---|---|
| `gla` | 5.30 | 0.285 | 0.106 | 0.078 | **0.050** | 0.050 | 1.30 (1.32 h wall) | 0.054 / 0.039 / 0.064 / 0.043 |
| `mamba2` | 5.19 | 0.296 | 0.103 | 0.070 | 0.045 | **0.044** | 1.38 (1.40 h wall) | 0.050 / 0.035 / 0.053 / 0.040 |

Image chats by subset at the end (photos, scene text, plots, screenshots, image pairs): `gla` 0.068 / 0.076 /
0.058 / 0.087 / 0.052, `mamba2` 0.064 / 0.067 / 0.047 / 0.063 / 0.039.

* **The return diminishes steeply.**  The first 12.6M tokens (9 GPU-minutes) take `gla` from 5.3 to 0.106;
  the last two thirds of the tokens take it from 0.078 to 0.050.
* **The end of a run is now a stable reading**: with the learning rate decayed to zero the last two
  measurements, 2M tokens apart, agree (0.0500 and 0.0501; 0.0445 and 0.0445).  With a constant rate they
  differed by 4–12 % (below).
* **`mamba2` against `gla`: 0.044 against 0.050**, one seed each.  Three seeds of a 20M-token `gla` run
  spread by ±3.6 % (one standard deviation, [recipe.md](recipe.md)), so an 11 % difference is likely real,
  but it has not been repeated at full length.
* **What is left is not small.**  A KL of 0.05 is 90 % top-1 agreement with the teacher on held-out input
  (the exact kernels: 98.5 %), and it shows on the standard test sets below.

### The runs before the fixes

The first runs of the recipe (`outputs/distill/_prefix/`) had three faults: every image batch was trained
on twice (two loader workers yielded the same stream); a step weighted its batches equally, which gave the
image chats about 60 % of the gradient for 20 % of the tokens; and the learning rate was constant.  Their
gap also measured images on 8 VQAv2 chats only.  `gla` and `mamba2` were re-distilled (above); `mamba3`,
`deltanet` and `swa` were **not**, to save compute, so their rows are of the faulty recipe.

| kernel | step 0 | 3M (end of local) | 12.5M | 33M | 100M | GPU-hours | text / chat / image (VQAv2) / long at 100M |
|---|---|---|---|---|---|---|---|
| `gla` | 5.01 | 0.377 | 0.134 | 0.106 | 0.077 | 1.35 | 0.083 / 0.061 / 0.099 / 0.066 |
| `deltanet` | 10.95 | 1.049 | 0.261 | 0.172 | 0.107 | 1.30 | 0.129 / 0.088 / 0.084 / 0.126 |
| `swa` | 11.77 | 0.569 | 0.197 | 0.153 | 0.118 | 1.76 | 0.149 / 0.096 / 0.112 / 0.115 |
| `mamba2` | 4.98 | 0.406 | 0.127 | 0.099 | 0.071 | 1.42 | 0.078 / 0.055 / 0.088 / 0.062 |
| `mamba3` | 3.46 | 0.375 | 0.122 | 0.093 | 0.068 | 2.05 | 0.076 / 0.055 / 0.083 / 0.058 |
| `mamba3`, FLA's layer as is (no short convolution) | 13.27 | 3.591 | 0.884 | 0.430 | 0.245 | 2.12 | 0.320 / 0.240 / 0.155 / 0.265 |

* **What the fixes are worth**, on the held-out rows both measurements share, same tokens:

  | kernel | | text | chat | long | image, VQAv2 | GPU-hours |
  |---|---|---|---|---|---|---|
  | `gla` | before → now | 0.083 → 0.054 | 0.061 → 0.039 | 0.066 → 0.043 | 0.099 → 0.068 | 1.35 → 1.30 |
  | `mamba2` | before → now | 0.078 → 0.050 | 0.055 → 0.035 | 0.062 → 0.040 | 0.088 → 0.064 | 1.42 → 1.38 |

  A third less distance for the same training, on every kind — images included, although image chats now
  carry a third of the gradient weight they had.  The 20M-token comparison in [recipe.md](recipe.md) splits
  this between the batch weighting and the decay; the doubled image batches were not measured separately.
* **These runs' last readings were favourable ones.**  In all five the gap fell by 4–12 % over the final
  2.2M tokens (`gla` 0.082 → 0.077, `mamba3` 0.077 → 0.068) while the training loss did not fall.  So
  "`mamba3` 0.068 against `mamba2` 0.071" is no difference, and the order of the three kernels that were
  not rerun is known only where they are far apart (`deltanet` and `swa` behind the others).
* **No run was stopped, or trained at a lower rate, by the plateau rule.**  Three runs triggered the cut at
  their last measurement, after which no step ran; every run ended at the token cap.
* **Their GPU-hours are upper bounds**: the runs shared their GPUs with evaluation jobs.  `mamba3`'s 1.4
  times `mamba2`'s time per token in particular is one reading under unknown load.
* `mamba2`'s first run hung at 17M tokens inside FLA's TileLang backward kernel for simple-GLA; its
  recurrence now runs on the GLA kernels (`kernels/mamba2.py`).
* `mamba3` as FLA's layer stands, without the backbone's short convolution, reached 0.245 after 100M
  tokens, where `mamba2` was after 4M; the kernel keeps the convolution ([kernels.md](kernels.md)).  Both
  runs had the same faults, and a gap this wide does not depend on them.

## Standard test sets

`linswap eval`, full test sets, in percent; a student's cell is its score and, in brackets, its difference
to the teacher.  Two runs:

* `main` — the teacher, `gdn`, the three exact kernels at step 0 and the distilled checkpoints **of the runs
  before the fixes**.  It was stopped during RULER: 128K is missing for every model, 64K for `mamba2` and
  `swa`, and `mamba3` has only RULER at 4K.  It kept no standard errors and no per-item records for text;
  the RULER rows below are read from RULER's own summary files (`outputs/eval/main-ruler-recovered.json`).
  The code changed while it ran; `kda` was therefore rerun at the present code on two RULER cells and 300
  MMBench items, and all 1,000 predictions and 300 replies are identical (`outputs/eval/check-kda`).
* `main2` — the re-distilled `gla` and `mamba2` on text and images, and the teacher on text, with standard
  errors, per-item records and the code revision.  The teacher's text scores reproduce `main`'s to the
  digit, except IFEval (52.3 against 52.5: one item of 541).  RULER was not rerun for the new checkpoints.

A star marks a difference of more than two standard errors where the harness gives them.

### Text (lm-evaluation-harness)

| model | LAMBADA | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | MMLU | IFEval | mean |
|---|---|---|---|---|---|---|---|---|---|
| teacher | 43.7 | 37.5 | 61.7 | 69.4 | 58.0 | 49.6 | 50.3 | 52.3 | 52.8 |
| ± standard error | 0.7 | 1.4 | 1.0 | 1.1 | 1.4 | 0.5 | 0.4 | 2.1 | |
| `gdn`, step 0 | +0.0 | +0.2 | −0.2 | −0.1 | +0.6 | +0.1 | +0.2 | +0.7 | +0.2 |
| `kda`, step 0 | +0.1 | −0.1 | −0.3 | −0.1 | +0.1 | +0.0 | −0.1 | +0.0 | −0.0 |
| `gdn2`, step 0 | +0.1 | +0.0 | +0.0 | −0.2 | −0.3 | +0.1 | +0.1 | +0.2 | −0.0 |
| `rwkv7`, step 0 | +0.0 | −0.2 | −0.1 | −0.3 | +0.7 | −0.0 | +0.0 | −0.2 | −0.0 |
| **`gla`, distilled** | 39.6 (−4.0*) | 33.3 (−4.2*) | 56.6 (−5.0*) | 67.9 (−1.5) | 56.6 (−1.4) | 48.1 (−1.5*) | 48.3 (−2.0*) | 38.6 (−13.7*) | 48.6 (−4.2) |
| **`mamba2`, distilled** | 40.1 (−3.5*) | 33.2 (−4.3*) | 55.5 (−6.2*) | 68.4 (−1.0) | 58.6 (+0.6) | 49.0 (−0.6) | 49.3 (−1.0) | 42.7 (−9.6*) | 49.6 (−3.2) |
| `gla`, before the fixes | −3.0 | −2.8 | −8.6 | −1.0 | −0.9 | −1.5 | −3.3 | −17.8 | −4.9 |
| `mamba2`, before the fixes | −3.6 | −5.6 | −9.6 | −1.0 | −1.0 | −1.1 | −2.2 | −13.5 | −4.7 |
| `deltanet`, before the fixes | −6.2 | −7.3 | −13.3 | −3.0 | −2.5 | −3.5 | −7.4 | −12.9 | −7.0 |
| `swa`, before the fixes | −7.8 | −5.5 | −9.1 | −1.5 | −2.8 | −3.1 | −7.1 | −19.4 | −7.0 |

* The exact kernels are the teacher: every difference is inside one standard error.
* **A distilled student is not.**  At a gap of 0.05 it is up to 6 points below the teacher on the likelihood
  tasks and 10–14 points below on IFEval — following instructions over a long reply is where a small
  per-token distance compounds.
* **Scores hide part of the distance.**  The share of items on which a model *chooses the option the
  teacher chooses* (LAMBADA: gets the same word right or wrong), from the per-item records:

  | model | LAMBADA | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | MMLU |
  |---|---|---|---|---|---|---|---|
  | `kda`, step 0 | 99.2 | 98.5 | 98.7 | 98.5 | 96.8 | 99.4 | 96.1 |
  | `gla`, distilled | 90.9 | 87.7 | 87.8 | 95.6 | 87.1 | 96.2 | 65.5 |
  | `mamba2`, distilled | 91.1 | 85.5 | 86.9 | 95.4 | 88.2 | 96.4 | 67.0 |

  On MMLU a student scores 1–2 points below the teacher and picks a different option on a third of the
  questions; an exact kernel differs on 4 % (bf16 noise on near-ties).
* The fixes moved the students towards the teacher on most tasks (IFEval +3.8 and +3.7, ARC-e +3.6 and
  +3.5), not on all (`gla` on LAMBADA −1.0, ARC-c −1.3).

### Long context (RULER, 500 samples per task; mean of the 7 tasks)

All distilled rows are of the checkpoints before the fixes.

| model | 4K | 16K | 64K | `vt` 4K / 16K / 64K | `cwe` | `fwe` |
|---|---|---|---|---|---|---|
| teacher | 95.8 | 96.1 | 90.4 | 94.3 / 88.9 / 76.8 | 78.3 / 86.3 / 64.1 | 97.7 / 97.5 / 91.8 |
| `gdn`, step 0 | 95.8 | 96.1 | 90.4 | 94.1 / 89.0 / 76.7 | 78.6 / 86.3 / 64.4 | 97.6 / 97.1 / 91.9 |
| `kda`, step 0 | 95.7 | 96.1 | 90.6 | 94.1 / 89.1 / 77.0 | 78.1 / 86.4 / 65.3 | 97.7 / 97.2 / 91.9 |
| `gdn2`, step 0 | 95.6 | 96.2 | 90.5 | 94.3 / 89.0 / 77.1 | 77.5 / 86.7 / 64.2 | 97.5 / 97.3 / 91.9 |
| `rwkv7`, step 0 | 95.7 | 96.1 | 90.5 | 94.2 / 89.0 / 76.4 | 78.1 / 86.3 / 65.1 | 97.7 / 97.3 / 91.8 |
| `gla`, before the fixes | 73.5 | 63.5 | 62.0 | 10.9 / 9.6 / 4.9 | 42.3 / 0.9 / 0.3 | 63.5 / 43.1 / 46.9 |
| `mamba2`, before the fixes | 80.2 | 73.2 | — | 38.0 / 43.4 / — | 43.8 / 1.1 / — | 80.2 / 75.2 / — |
| `deltanet`, before the fixes | 79.2 | 68.2 | 54.6 | 56.5 / 29.2 / 24.4 | 30.9 / 5.2 / 0.0 | 73.6 / 64.8 / 32.6 |
| `swa`, before the fixes | 71.6 | 65.0 | — | 17.7 / 15.9 / — | 17.4 / 0.5 / — | 69.3 / 58.7 / — |
| `mamba3`, before the fixes | 82.2 | — | — | 53.2 / — / — | 39.5 / — / — | 82.8 / — / — |

* The exact kernels match the teacher at every length measured.
* **The distilled students find a needle and cannot track or count**: the four needle tasks stay at
  81–100 % (except `deltanet`'s multi-key at 64K, 51 %), while variable tracking falls from 94 to 11–57 at
  4K and common-word extraction from 86 to 0.5–5 at 16K.  These tasks need state carried exactly over
  thousands of tokens, which is where the kernels differ from the teacher; a held-out KL of 0.06–0.13 on
  a 32K sequence did not show it.  Whether the re-distilled checkpoints do better here is not known.
* `gla` returned an empty answer on `fwe` for 54 / 110 / 113 of the 500 samples at 4K / 16K / 64K; every
  other prediction file holds 500 non-empty answers.

### Images (lmms-eval)

| model | MME | MMBench | AI2D | ChartQA | OCRBench | TextVQA | MuirBench |
|---|---|---|---|---|---|---|---|
| teacher | 63.2 | 62.4 | 54.8 | 70.1 | 78.4 | 68.7 | 5.1 |
| `gdn`, step 0 | +0.1 | −0.6 | +0.1 | −0.2 | +0.2 | −0.1 | +0.3 |
| `kda`, step 0 | +0.2 | −0.9 | −0.2 | +0.1 | +0.1 | +0.0 | +0.2 |
| `gdn2`, step 0 | +0.1 | −0.3 | +0.2 | −0.2 | +0.0 | −0.2 | +0.1 |
| `rwkv7`, step 0 | +0.3 | −0.6 | +0.3 | +0.2 | +0.0 | +0.0 | +0.4 |
| **`gla`, distilled** | 61.4 (−1.8) | 56.8 (−5.6) | 59.3 (+4.5) | 59.9 (−10.2) | 77.3 (−1.1) | 68.0 (−0.7) | 29.5 (+24.4) |
| **`mamba2`, distilled** | 60.6 (−2.6) | 58.8 (−3.5) | 61.6 (+6.8) | 60.2 (−9.9) | 76.7 (−1.7) | 68.2 (−0.5) | 26.8 (+21.6) |
| ± standard error of a student's score | | | 0.9 | 1.0 | 1.3 | 0.6 | |
| `gla`, before the fixes | −2.5 | −10.0 | +2.3 | −6.8 | −3.0 | −1.2 | +20.8 |
| `mamba2`, before the fixes | −1.8 | −5.6 | +6.1 | −6.5 | −2.6 | −0.5 | +23.7 |
| `deltanet`, before the fixes | −1.4 | −18.9 | −17.6 | −5.5 | −4.9 | −3.3 | +18.4 |
| `swa`, before the fixes | −3.6 | −16.0 | −3.7 | −10.2 | −4.9 | −1.9 | +19.2 |

No mean is given: three of the seven columns do not measure what they seem to.

**How the models reply.**  lmms-eval reads a multiple-choice answer from the reply's text.  Share of
replies that open with an option, and mean reply length in characters:

| model | MMBench | AI2D | MuirBench | MME (yes / no) |
|---|---|---|---|---|
| teacher | 66 % / 355 | | | |
| `kda`, `gdn`, `gdn2`, `rwkv7`, step 0 | 65–67 % / 349–366 | | | |
| `gla`, distilled | 100 % / 21 | 100 % / 8 | 100 % / 6 | 98 % / 2 |
| `mamba2`, distilled | 96 % / 41 | 100 % / 10 | 82 % / 27 | 99 % / 2 |
| `gla` / `mamba2` / `deltanet` / `swa`, before the fixes | 100 % / 22, 100 % / 21, 94 % / 69, 99 % / 27 | | | |

(`main` kept replies for MMBench only, so the teacher's cells for the other tasks are empty.)

* **The students answer in a different manner from the teacher.**  The teacher explains (a third of its
  MMBench replies do not open with an option; 355 characters on average); a student gives the letter.  The
  gap does not see this — its image chats come with The Cauldron's short answers — and the image suite ran
  4.8–7.2 hours for the teacher and the exact kernels against 1.3–2.7 hours for the students for this reason.
* **So MuirBench, AI2D and MMBench do not compare a student with the teacher.**  MuirBench counts a reply
  only if it is the option letter: the teacher gets 5 %, below chance, the students 26–30 %.  AI2D puts the
  students 4–7 points *above* the teacher; MMBench's rule-based extraction has the teacher at 62.4.  A
  student "beating" the teacher here is a difference in wording, not in ability.
* **Where the reply is free-form the students are behind**: ChartQA by 10 points, OCRBench and TextVQA by
  0.5–2, MME by 2–3.
* **The fixes did not help everywhere.**  MMBench (+4.4, +2.0) and OCRBench (+1.9, +0.9) moved towards the
  teacher; **ChartQA moved away, 63.2 → 59.9 and 63.6 → 60.2**, on both of its splits and for both kernels
  (standard error 1.0).  The image chats' share of the objective went from about 60 % to their 20 % share
  of the tokens; the held-out image KL improved all the same (0.099 → 0.068 on the VQAv2 chats), so the gap
  does not capture what ChartQA needs.  The share of image tokens in the mix (`data.MIX`) is the knob; it
  was not tuned, and should not be tuned on these test sets.

## What the evidence supports

* **Exact kernels**: `kda`, `gdn2` and `rwkv7` are the teacher at step 0 — logits, greedy answers, and every
  standard test set measured (text, images, RULER to 64K).  No training, as claimed.
* **Inexact kernels**: 100M tokens and 1.3–1.4 GPU-hours bring `gla` and `mamba2` to a held-out KL of
  0.050 and 0.044, 90 % top-1 agreement.  That is *near* the teacher on likelihood benchmarks (up to 6 points),
  and *not* near it on instruction following (10–14 points), chart reading (10 points), and — measured
  only on the earlier checkpoints — variable tracking and word counting in long context.  The gap alone
  understates the distance; read it with the tables above.
* **Not established**: the order of the inexact kernels beyond "`deltanet` and `swa` are behind" (one seed;
  three of the five not re-distilled); anything about RULER, 128K contexts or `mamba3` on the standard sets
  for the current recipe; the 27B backbone beyond "it runs".
