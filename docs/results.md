# Results

Qwen3.5-0.8B, one seed, the recipe in [recipe.md](recipe.md).  Per-kernel mappings are in
[kernels.md](kernels.md).

## Results (Qwen3.5-0.8B, one seed)

Everything below is the recipe above: 700M tokens of DCLM, no SFT, base-prompt evaluation.  The
**control** is the *unswapped* backbone put through the identical three steps — without it the
students' gains over the teacher cannot be attributed to the kernel.

**RULER / passkey** (500 samples; `niah_single_1` / `_2` / `_3` / `niah_multikey_1`):

| model | 4K | 16K | 64K | 128K |
|---|---|---|---|---|
| teacher (unmodified backbone) | 96.4 / 65.0 / 97.8 / 79.4 | 98.4 / 76.2 / 90.6 / 81.6 | 96.4 / 98.4 / 94.6 / 91.8 | 99.2 / 91.6 / 96.6 / 91.0 |
| control (`gdn`, same recipe) | 100 / 100 / 98.6 / 99.6 | 100 / 100 / 100 / 98.4 | 100 / 100 / 100 / 97.2 | 100 / 100 / 100 / 96.8 |
| `kda_fullgate` (exact init) | 100 / 100 / 96.8 / 99.6 | 100 / 100 / 100 / 98.4 | 100 / 100 / 100 / 97.0 | 100 / 100 / 99.8 / 96.6 |
| `gdn2` (exact init) | 100 / 100 / 94.2 / 100 | 100 / 100 / 100 / 98.4 | 100 / 100 / 100 / 97.0 | 100 / 100 / 100 / 96.4 |
| `rwkv7` (exact init) | 100 / 100 / 96.0 / 99.6 | 100 / 100 / 100 / 98.4 | 100 / 100 / 100 / 97.2 | 100 / 100 / 100 / 96.2 |
| `kda` (exact init) | 100 / 100 / 99.2 / 99.6 | 100 / 100 / 100 / 98.4 | 100 / 100 / 100 / 97.0 | 100 / 100 / 100 / 96.4 |
| `mamba2` (no erase) | 100 / 100 / 99.4 / 98.8 | 100 / 100 / 99.8 / 93.6 | 100 / 98.4 / 99.4 / 84.6 | 100 / 95.0 / 94.6 / 70.0 |
| `swa` (window 64 + 4 sinks) | 100 / 100 / 99.8 / 97.4 | 100 / 99.2 / 97.6 / 79.6 | 100 / 98.0 / 95.4 / 67.6 | 100 / 81.4 / 89.4 / 56.0 |
| `gla` (per-channel decay, no erase) | 97.4 / 100 / 99.6 / 99.2 | 52.8 / 100 / 99.6 / 89.0 | 29.0 / 96.2 / 96.8 / 82.0 | 32.0 / 94.4 / 90.0 / 58.8 |
| `deltanet` (erase, no decay) | 100 / 99.8 / 96.6 / 98.2 | 98.4 / 100 / 82.6 / 85.2 | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 |

500 samples puts the binomial 95% interval at about ±3 points.  `mamba1` and `mamba3` are not being
carried forward, so their rows are gone from both tables; the kernels stay in the registry.

**RULER aggregation** (`vt` / `cwe` / `fwe`, 500 samples, same protocol).  Each kernel's best
checkpoint: for the exact kernels and the control that is the recipe without the `layer` step
(`outputs/<kernel>_nolayer/distill/checkpoint-10235`, KL + CE only), the others are the full recipe above.  These runs cut each prediction where the
model restarted the task prompt; regenerating 1,800 paired samples without the cut (control, `mamba2`,
`gla` at 16K and 128K) changed 4 sample scores and no cell by more than 0.3, so the table stands for
standard RULER scoring, which is what `evaluate` now uses.

| model | 4K | 16K | 64K | 128K |
|---|---|---|---|---|
| control (`gdn`, no layer) | 90.9 / 62.5 / 87.8 | 70.8 / 68.6 / 88.5 | 50.6 / 30.7 / 76.3 | 68.6 / 8.5 / 88.2 |
| `gdn2` (no layer) | 91.8 / 63.9 / 86.1 | 71.0 / 70.6 / 86.5 | 50.6 / 26.0 / 75.7 | 67.6 / 5.0 / 85.9 |
| `kda` (no layer) | 90.0 / 62.5 / 85.5 | 71.6 / 69.2 / 83.6 | 51.4 / 31.4 / 75.5 | 69.5 / 11.2 / 84.4 |
| `kda_fullgate` (no layer) | 90.0 / 63.3 / 86.4 | 71.0 / 69.3 / 85.5 | 53.6 / 30.6 / 76.0 | 70.5 / 7.4 / 85.9 |
| `rwkv7` (no layer) | 89.8 / 60.6 / 85.1 | 65.0 / 68.7 / 84.0 | 45.5 / 27.7 / 76.4 | 66.4 / 3.9 / 88.1 |
| `mamba2` | 78.2 / 38.0 / 67.9 | 36.8 / 4.0 / 71.1 | 19.8 / 0.8 / 60.4 | 19.6 / 0.4 / 51.0 |
| `swa` | 37.6 / 37.1 / 76.7 | 23.7 / 1.4 / 54.1 | 28.6 / 0.6 / 63.3 | 7.2 / 0.4 / 69.5 |
| `gla` | 38.8 / 24.7 / 68.9 | 1.6 / 7.3 / 42.3 | 0.2 / 0.3 / 29.9 | 0.2 / 0.1 / 21.3 |
| `deltanet` | 51.4 / 19.9 / 12.7 | 18.7 / 3.5 / 9.5 | 0.0 / 0.0 / 0.0 | 0.0 / 0.1 / 0.0 |

- These tasks separate the kernels from 4K on, where the needles do not.  On the three-task average the
  four exact kernels stay within 3.4 points of the control at every length (largest single-cell gaps:
  `rwkv7` `vt` −5.7 at 16K, `gdn2` `cwe` −4.6 at 64K, `kda` `fwe` −4.9 at 16K); the inexact kernels
  are 19 (`mamba2`) to 52 (`deltanet`) points down on the average at 4K already.
- `cwe` collapses with length for every model, the control included (8.5 at 128K): 30 repeats of the
  ten common words among ~5.5K distractor words (17K list entries at 128K) is past what this 0.8B backbone counts, so beyond 16K `cwe`
  measures the backbone, not the swap.
- `vt` dips at 64K and recovers at 128K for the control and every exact kernel.  It is not the sample
  set: a fresh draw (seed 43, 100 samples) gives the control 54.6 at 64K and 66.2 at 128K.  The
  dip is one failure mode: the model answers in the 3-letter format of the few-shot example's
  variables instead of naming the queried chain — 91 of the control's 131 zero-score answers at 64K,
  16 of 46 at 128K.
  (`fwe`'s 64K dip mostly is the sample set: 80.0 vs 82.0 on the fresh draw.)
- `fwe` is the one task a short window can partly solve: under a zeta(2) word distribution the top
  three words dominate any local window, so `swa` holds 54–77 while its `vt` and `cwe` collapse.
- `deltanet` fails even at 4K: it loops on one word (`likeness likeness …`, `... ... ...`).

**Short context**, accuracy and relative score against the teacher in %:

| model | LAMBADA | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | MMLU 5-shot | rel. avg |
|---|---|---|---|---|---|---|---|---|
| teacher | 0.437 (ppl 14.7) | 0.374 | 0.611 | 0.693 | 0.583 | 0.496 | 0.504 | 100.0 |
| control (`gdn`) | 0.476 (13.2) | 0.402 | 0.649 | 0.704 | 0.589 | 0.525 | 0.515 | 110.1 |
| `kda_fullgate` | 0.479 (13.2) | 0.399 | 0.655 | 0.706 | 0.592 | 0.525 | 0.513 | **110.7** |
| `gdn2` | 0.479 (13.2) | 0.398 | 0.652 | 0.706 | 0.590 | 0.524 | 0.516 | 110.2 |
| `kda` | 0.478 (13.1) | 0.399 | 0.651 | 0.705 | 0.590 | 0.525 | 0.514 | 110.2 |
| `rwkv7` | 0.480 (13.2) | 0.399 | 0.655 | 0.704 | 0.590 | 0.525 | 0.513 | 110.3 |
| `mamba2` | 0.462 (14.2) | 0.372 | 0.610 | 0.701 | 0.578 | 0.520 | 0.501 | 101.4 |
| `swa` | 0.453 (15.3) | 0.372 | 0.617 | 0.702 | 0.595 | 0.503 | 0.456 | 101.0 |
| `gla` | 0.463 (14.3) | 0.362 | 0.612 | 0.701 | 0.579 | 0.514 | 0.484 | 99.2 |
| `deltanet` | 0.382 (21.3) | 0.331 | 0.562 | 0.694 | 0.569 | 0.475 | 0.415 | 82.8 |

The short-context suite runs each task in full, so it does not depend on the RULER sample count and
these rows are unchanged.  The second batch (`kda`, `kda_fullgate`, `gla`, `deltanet`) was run months
later on freshly tokenised DCLM and carried its own control: it came back at 110.0 -> 110.1, which is
what licenses reading the batches in one table.  The decay-fitted `gla` row is a third run
(`outputs/eval/gla_mapped-lmeval`) scored against the same backbone reference.

**Throughput** (one L20X, bf16, batch 1, cached greedy decode, 256 new tokens):

| model | prefill 8K / 32K (tok/s) | decode (ms/token) | peak 8K / 32K (GiB) |
|---|---|---|---|
| `gdn` (FLA chunk kernel) | 134K / 141K | 24.5 | 2.0 / 3.4 |
| `mamba2` (FLA simple-GLA op) | 119K / 128K | 30.5 | 2.0 / 3.3 |
| `rwkv7` (FLA DPLR chunk kernel) | 79K / 76K | 33.3 | 2.7 / 6.1 |

Reading.

* **The recipe, not the kernel, is what lifts the scores above the teacher.**  The control gains as
  much as the students on both suites (relative average 110.0, needles at 100 almost everywhere), so
  the right question is what the *swap* costs on top of it.
* **With an exact init the swap is free.**  All four exact-init kernels land on the control on both
  suites: relative average 110.2–110.7 against 110.1, and 96.2–96.6 on the 128K distractor needle
  against 96.8.  Four recurrences as different as the gated delta rule, a per-key-channel gated delta
  rule and a DPLR generalised delta rule are indistinguishable from an exact copy of the backbone,
  which says the swap is paid for by the *initialisation* and not by the target architecture.
  This corrects an earlier reading of the same kernels.  Under the previous recipe they scored
  85.6–92.4 at 128K and the deficit was attributed to the bounded state; it was the `layer` step, which
  moves a function-preserving init 18–20 % of weight norm for nothing (see
  [gate_diagnostics.md](gate_diagnostics.md)) and is skipped for such kernels now.
* **KDA's low-rank forget gate costs nothing, and neither does GDN-2's extra gate.**  `kda` and
  `kda_fullgate` differ only in whether the decay factors through the 128-dim bottleneck FLA ships or a
  dense 2048x1024 matrix; `gdn2` adds 113M parameters of separate erase and write gates.  All three
  finish within 0.5 relative points of each other and of the control, at every needle length (128K
  distractor: `kda` 96.4, `kda_fullgate` 96.6, `gdn2` 96.4, control 96.8).  A richer gate buys nothing
  at this scale and budget.  The 6.8-point gap between the KDA variants reported earlier was an
  artifact of the `layer` step, which moved the dense gate further than the low-rank one.
* **The missing erase still costs.**  `mamba2` trails the control by 8.6 relative points and loses the
  distractor needle at 128K (70.0 vs 96.2) — the same failure mode as under every earlier recipe.
* **A bounded softmax window is the same story, sharper.**  `swa` keeps 101.0 relative on short
  context with a 68-key state but decays 97.4 / 79.6 / 67.6 / 56.0 on the multikey needle across the
  four lengths, and drops to 0.456 on MMLU (81.1 relative), the largest single-task deficit here.
* **Inexact swaps fail with length, and the short-context suite cannot see it.**  `deltanet` matches
  the control at 4K (98.7 task average against 99.2) and then scores exactly 0 on all four needles from
  64K on — at 500 samples that is 2,000 attempts per length with no answer, so it is the mechanism, not
  the sampling; `gla` decays through 99.1 / 85.4 / 76.0 / 68.8.  `gla` also fails one task out of order:
  `niah_single_1` is at 52.8 at 16K and 29.0 at 64K while its other three needles are still at 82-100,
  the only row in the table where the nominally easiest needle goes first.  Both remain respectable on LAMBADA, PIQA and
  HellaSwag, so a swap validated only on short-context benchmarks can be
  entirely broken at 64K.  This is the strongest argument in these results for scoring retrieval at
  several lengths rather than reporting a single accuracy.
* **`gla`'s out-of-order failure is the gate init, and the fix is partial.**  With FLA's random
  per-channel gate init `niah_single_1` went 53.2 / 1.4 / 0 / 0 (task average 87.5 / 64.8 / 43.2 / 1.6,
  94.3 relative on short context) while the essay
  needles held to 16K.  Measured on that checkpoint: retrieval in the hybrid is done by the six
  full-attention layers (the linear layers' readout at the answer position carries under 0.3% needle in
  every kernel, control included), and what breaks is the residual stream they read from.  On a
  haystack of one sentence repeated thousands of times the GDN state reaches a fixed point (repeated
  writes are idempotent under the delta rule) and Mamba-2's per-head scalar growth is cancelled by the
  per-head RMSNorm, but GLA's per-channel decay lets channels of one head grow at different rates, so
  its output keeps changing with the repeat count and drifts from the teacher — on distillation text
  that regime never occurs.  84% of its heads had mixed horizons after distillation.  Fitting GDN's
  per-head decay into the gate MLP at init (`kernels/gla.py`) starts every channel of a head on one
  horizon; distillation regrows the spread in about a quarter of the heads, which is why the
  repeated-noise needle recovers to 97.4 / 52.8 / 29.0 / 32.0 rather than to 100.
* `deltanet`'s collapse is the predicted one: with `exp(g_t) = 1` the state never contracts, so stale
  associations survive until they are explicitly overwritten and the needle becomes unrecoverable once
  the context is long enough.  Distillation on 700M tokens moves the length at which that happens; it
  does not remove it.
* Decode time is length-independent for all three (constant state).  RWKV-7's DPLR kernel is the
  slowest and needs the most memory (two rank-1 terms per step, more chunk intermediates); Mamba-2's
  SSD recurrence runs through FLA's generic simple-GLA kernels rather than Mamba-2's own fused CUDA
  kernels.

## Approaches that were tried and dropped

Kept here as a record; the code for them is no longer in the tree.

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
