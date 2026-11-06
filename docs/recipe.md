# Recipe: distillation and evaluation

How a swapped model is brought to the teacher, and how the distance is measured.  The numbers are in
[results.md](results.md); the per-kernel mappings in [kernels.md](kernels.md).

## The goal decides the recipe

The target is the **teacher** — the unmodified Qwen3.5 backbone, a vision-language model — and not the
data.  Three things follow, and each removed a part of the earlier recipe (layer alignment → logit KL →
cross-entropy on web text, 700M tokens):

* **No ground-truth loss.**  Cross-entropy on web text pulls the student towards the corpus, away from the
  teacher: it lowered validation loss *below* the teacher's (2.96 against 3.09 for `gla`) and un-taught the
  end-of-turn token, because packed web text has no chat turns.
* **The inputs are the teacher's inputs.**  Web text alone leaves the swapped layers unfitted on what a
  vision-language chat model actually sees.  The old checkpoints are at KL 0.16–0.20 from the teacher on
  text and at 1.1–1.8 on image chats.
* **An exact kernel is finished at step 0.**  `kda`, `gdn2` and `rwkv7` reproduce the teacher to numerical
  noise (`linswap verify`), so `linswap distill` declines to train them.  Any training only moves them away.

## What the old recipe spent its compute on

The `gla` run of the earlier recipe (validation loss on held-out web text; the teacher is at 3.09):

| point in the run | tokens | GPU time | validation loss |
|---|---|---|---|
| swap, untrained | 0 | 0 | 7.45 |
| layer alignment, first 200 steps | 3.3M | ~1 min | 3.39 |
| layer alignment, end | 100M | 22 min | 3.18 |
| logit KL, end | 600M | 5.2 h | 3.14 |

94 % of what was gained came in the first minute, and the KL step — 80 % of the GPU time — moved the loss
by 0.04.  It could hardly do more: the weights were stored in bf16 and updated at lr 1e-5, and such a step
is below half a bf16 grid step for any weight larger than about 2.5e-3, so it rounds away.  In the finished
runs 79 % of the weights were bit-identical to the init.  The teacher was also only ever matched at 512
tokens, while the inexact kernels fail at 16K and beyond.

## Distillation (`linswap distill`)

**Only the swapped layers are trained**; everything else is the teacher's and frozen, so the vision
tower, the chat format and the end-of-turn behaviour cannot drift.  **The target is the teacher's
activations**, on a stream (`linswap.data`) that gives each kind of input its share of the tokens:

| source | form | share |
|---|---|---|
| web text (DCLM) | one 16,384-token sequence | 30 % |
| web text (DCLM) | eight 2,048-token sequences | 30 % |
| chat (Alpaca, in the chat template) | eight 2,048-token sequences | 20 % |
| image chats (The Cauldron: photos, scene text, plots, screenshots, image pairs) | one multi-turn chat with four images | 20 % |

None of it is used as labels, and the image subsets are not evaluation benchmarks.

Two phases:

1. **local** (3M tokens, about a minute).  Every swapped layer is given the teacher layer's input and
   asked for the teacher layer's output — 18 independent regressions on one teacher forward.  There is no
   vocabulary head and no backward pass through the depth of the model, so a token costs less than half of
   what it costs in the loop, and the fit starts from a useful place however far the init was.
2. **loop**.  The student runs on its own hidden states.  They are matched to the teacher's after every one
   of the 24 blocks (squared error relative to the teacher's power, so each block weighs the same).  This
   removes the error that accumulates when every layer was only fitted to clean inputs.  The logits are not
   in the loss, so training never touches the vocabulary head.

**Stopping is measured.**  The gap — KL(teacher ‖ student) of the next-token distribution on held-out
text, chat, image chats and one 32K sequence, averaged over the four — is taken at token milestones
(0.3M, 1M, 3M, 10M, ... into the loop).  Positions holding an image token are left out of it: nothing is
predicted from them, and what they pass on shows at the text positions that follow.  When a milestone improves it by less than 5 %, the learning rate
is cut once; the second time the run ends, at 100M tokens at the latest.  The best milestone is what is saved.

**Precision.**  The trained layers keep float32 master weights and float32 optimizer state; the forward
pass, the kernels' recurrent states and everything frozen are bf16, exactly as at inference.  The gap is
measured with the trained weights cast to bf16, and that cast is what the checkpoint holds, so the stopping
decision is made on the model that is delivered.

### What the choices are based on

`gla`, 20M tokens each, one run per variant.  "Loop" here is hidden states plus a KL on the logits at 2,048
sampled positions per batch unless said otherwise.  The first gap column counts every position, as the gap
did when these ran; the second leaves image tokens out, as it does now (measured for five of the variants).

| variant | gap, all positions | gap | GPU-hours |
|---|---|---|---|
| local only | 0.256 | 0.211 | 0.14 |
| loop only | 0.232 | 0.154 | 0.31 |
| loop only, logit KL alone (no hidden-state loss) | 0.241 | | 0.30 |
| local 3M, then loop | 0.189 | 0.138 | 0.26 |
| **local 3M, then loop on hidden states alone** | **0.180** | **0.129** | 0.25 |
| local 3M, then loop at lr 3e-4 | 0.210 | | 0.25 |
| local 3M, then loop, training the 18 blocks (MLPs and norms too) | 0.231 | 0.145 | 0.26 |
| local 3M, then loop, training the whole language model | 0.275 | | 0.27 |

* The two phases together beat either alone at equal compute: the local phase gets from 5.0 to 0.43 in
  about a minute, and the loop then closes what teacher-forced fitting cannot.
* Training more than the swapped layers is *worse*, not better, on top of risking the abilities that live
  outside them.
* Matching hidden states carries the signal.  The logit KL alone is the weakest loop, and adding it to the
  hidden states is no better than leaving it out (0.138 against 0.129), so it was removed.

For reference, on the same held-out set the earlier 700M-token checkpoints are at 0.43 (`mamba2`) to 0.60
(`swa`), and `kda` at step 0 is at 0.001.

## Evaluation (`linswap eval`)

Three suites, each run by the harness that defines its protocol, on the full test set, thinking disabled,
greedy decoding.  The swapped model is `transformers`' own model, so the harnesses drive it as they drive
the backbone.

* **text** — lm-evaluation-harness: LAMBADA, ARC-challenge / easy, PIQA, WinoGrande, HellaSwag (0-shot),
  MMLU (5-shot), and IFEval (generative, chat template).
* **long** — RULER's own scripts: `niah_single_1/2/3`, `niah_multikey_1` (needles), `vt` (variable
  tracking), `cwe` / `fwe` (common / frequent words extraction), 500 samples per task at 4K, 16K, 64K and
  128K, prompts in the chat template, left-padded batches.
* **mm** — lmms-eval with its Qwen3 wrapper: MME (perception + cognition, as a share of 2800), MMBench-EN
  dev (rule-based answer extraction, no GPT judge), AI2D, ChartQA (relaxed accuracy), OCRBench, TextVQA
  val, MuirBench (multi-image).

`--smoke` runs two tasks per suite on 8 items and one RULER length with 4 samples: a check that the path
works, in minutes, never a score.  `swa` cannot mask left padding (its sinks are absolute positions) and is
evaluated one prompt at a time.
