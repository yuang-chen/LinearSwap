# Recipe: distillation and evaluation

How a swapped model is brought to the teacher, and how the distance is measured.  The numbers are in
[results.md](results.md); the per-kernel mappings in [kernels.md](kernels.md).

## The goal decides the recipe

The target is the **teacher** — the unmodified Qwen3.5 backbone, a vision-language model — and not the
data.  Three things follow:

* **No ground-truth loss.**  Cross-entropy on web text pulls the student towards the corpus, away from the
  teacher, and packed web text has no chat turns to keep the end-of-turn token.
* **The inputs are the teacher's inputs.**  Web text alone leaves the swapped layers unfitted on what a
  vision-language chat model actually sees.
* **An exact kernel is finished at step 0.**  `kda`, `gdn2` and `rwkv7` reproduce the teacher to numerical
  noise (`linswap verify`), so `linswap distill` declines to train them.  Any training only moves them away.

## Distillation (`linswap distill`)

**Only the swapped layers are trained**; everything else is the teacher's and frozen.  That keeps the
weights of the vision tower, the embeddings and the vocabulary head — not the model's behaviour: how a
student answers is decided in the swapped layers too, and a distilled student does answer differently
from the teacher (see [results.md](results.md), "How the models reply").  **The target is the teacher's
activations**, on a stream (`linswap.data`) that gives each kind of input its share of the tokens:

| source | form | share |
|---|---|---|
| web text (DCLM) | one 16,384-token sequence | 30 % |
| web text (DCLM) | eight 2,048-token sequences | 30 % |
| chat (Alpaca, in the chat template) | eight 2,048-token sequences | 20 % |
| image chats (The Cauldron: photos, scene text, plots, screenshots, image pairs) | one multi-turn chat with four images | 20 % |

None of it is used as labels.  The image subsets are not among the evaluation benchmarks by name;
whether single images recur in them (VQAv2 and MME / MMBench both draw on COCO, TextCaps on the image
pool of TextVQA) has not been checked.  A step is 65,536 tokens of these batches and **weighs each batch by
its tokens**, so the shares above are also the shares of the objective; the two image-loader workers read
disjoint rows.

Two phases:

1. **local** (3M tokens, about a minute).  Every swapped layer is given the teacher layer's input and
   asked for the teacher layer's output — 18 independent regressions on one teacher forward.  There is no
   vocabulary head and no backward pass through the depth of the model, so a token costs about half of
   what it costs in the loop (0.56 in the `gla` run), and the fit starts from a useful place however far
   the init was.
2. **loop**.  The student runs on its own hidden states.  They are matched to the teacher's after every one
   of the 24 blocks — the last one after the final norm — (squared error relative to the teacher's power, so
   each block weighs the same).  This
   removes the error that accumulates when every layer was only fitted to clean inputs.  The logits are not
   in the loss, so training never touches the vocabulary head.

**The learning rate** is warmed up over 10 steps and then lowered linearly to zero over the phase's token
budget (`--no-decay` keeps it constant).

**The gap** — KL(teacher ‖ student) of the next-token distribution on held-out text (4 × 8,192 tokens),
chat (2 × 4 × 2,048), image chats (8 of each of the five subsets: 40 chats, 8.2K scored positions) and one
32K sequence, averaged over the four kinds — is taken at token milestones (0.3M, 1M, 3M, 10M, ... into the
loop).  Positions holding an image token are left out of it: nothing is predicted from them, and what they
pass on shows at the text positions that follow.  The teacher it is measured against is the `gdn` swap,
which `linswap verify` puts at numerical noise from the unmodified backbone.  The best milestone is what
is saved.

**The plateau rule is a safeguard, not what ends the runs.**  When a milestone improves the gap by less
than 5 %, the learning rate is cut to 0.3 of its value once, and the second time the phase ends.  The
milestones grow apart by √10, so a 100M-token run is judged six times in the loop, and no run so far has
been stopped (or trained at a cut rate) by it: every run ends at the token cap, and the cap is the budget
to choose.  A single milestone is also a noisy reading — with a constant learning rate the gap moved by
4–12 % over the last 2 % of the tokens in all five runs of the first batch, with no change in the training
loss; three seeds of one 20M-token `gla` run differ by ±3.6 % (one standard deviation) — and the best
milestone is chosen on the set it is reported on.  Differences of that size between two runs are none.
With the decay the end of a run is steady: the last two readings of the current `gla` and `mamba2` runs,
2M tokens apart, agree to 0.2 %.

**The gap is not the whole distance.**  It is a next-token KL under teacher forcing on inputs like the
training mix.  It does not see how a student words a free reply, and it did not predict ChartQA, which got
worse when the held-out image KL got better ([results.md](results.md)).  A run is judged on the standard
test sets as well.

**Precision.**  The trained layers keep float32 master weights and float32 optimizer state; the forward
pass, the kernels' recurrent states and everything frozen are bf16, exactly as at inference.  The gap is
measured with the trained weights cast to bf16, and that cast is what the checkpoint holds, so the stopping
decision is made on the model that is delivered.

### What the choices are based on

**Batch weight and learning-rate schedule** (`outputs/ab/`): `gla`, 20M tokens each, the pipeline as it is
now, gap on the held-out set described above.

| variant | gap | text | chat | image | long |
|---|---|---|---|---|---|
| batches weighted equally, constant learning rate (the behaviour before) | 0.114 | 0.140 | 0.097 | 0.113 | 0.108 |
| batches weighted by tokens, constant, seeds 0 / 1 / 2 | 0.103 / 0.101 / 0.096 | 0.124 / 0.130 / 0.110 | 0.084 / 0.078 / 0.078 | 0.114 / 0.108 / 0.110 | 0.091 / 0.087 / 0.085 |
| **batches weighted by tokens, linear decay to zero** (the default) | **0.086** | 0.092 | 0.069 | 0.107 | 0.077 |

* Weighting by tokens is ahead by four standard deviations of the seed spread (0.100 ± 0.004 against
  0.114), on text, chat and long context, and costs nothing on images (0.110 against 0.113) although image
  chats go from about 60 % of a step's gradient to their 20 % token share.
* Decay is ahead of the constant rate by another 0.014 (one run; 3.8 standard deviations of that spread),
  on all four kinds.
* Same tokens, same GPU time (0.25 GPU-hours each, five runs side by side on five GPUs).

**Phases and what is trained** (`outputs/bakeoff/`): these ran before three faults were fixed — every image
batch was drawn twice, the image part of the gap was measured on VQAv2 photos only, and batches were
weighted equally — so read the table for its order, not its values; it has not been repeated.

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

* The two phases together beat either alone at equal tokens (and about equal GPU time: 0.25 against 0.14
  and 0.31 GPU-hours): the local phase gets from 5.0 to about 0.4 in about a minute, and the loop then
  closes what teacher-forced fitting cannot.
* Training more than the swapped layers was *worse*, not better, in this one comparison.
* Matching hidden states carries the signal.  The logit KL alone is the weakest loop, and adding it to the
  hidden states is no better than leaving it out (0.138 against 0.129), so it was removed.

## Evaluation (`linswap eval`)

Three suites, each run by the harness that defines its protocol, on the full test set, thinking disabled,
greedy decoding.  The swapped model is `transformers`' own model, so the harnesses drive it as they drive
the backbone.  A run keeps, per model and suite, the code revision, the package versions and the checkpoint's
hash (`results.json`), the harness's standard errors, and the per-item records (`samples/<label>/<task>.jsonl`;
RULER's predictions under `ruler/`); a label whose checkpoint has changed is refused rather than mixed with
its old scores.  `results.md` gives every task with its difference to the teacher; the mean over tasks is
shown but decides nothing.

* **text** — lm-evaluation-harness: LAMBADA, ARC-challenge / easy, PIQA, WinoGrande, HellaSwag (0-shot),
  MMLU (5-shot), and IFEval (generative, chat template).
* **long** — RULER's own scripts: `niah_single_1/2/3`, `niah_multikey_1` (needles), `vt` (variable
  tracking), `cwe` / `fwe` (common / frequent words extraction), 500 samples per task at 4K, 16K, 64K and
  128K, prompts in the chat template, left-padded batches.
* **mm** — lmms-eval with its Qwen3 wrapper: MME (perception + cognition, as a share of 2800), MMBench-EN
  dev (rule-based answer extraction, no GPT judge), AI2D, ChartQA (relaxed accuracy), OCRBench, TextVQA
  val, MuirBench (multi-image).  One item per call, the wrapper's default.  `--mm_batch_size 8` is 2–3
  times faster, but left padding shifts bf16 rounding: on 120 items per task it changed 2–8 % of the
  answers (the wording of 46 % of the teacher's MMBench answers) and moved no score by more than three
  items, in no consistent direction — fine for a quick look, not used for the tables.

**The image suite's multiple-choice scores depend on how a model words its answer.**  lmms-eval reads the
answer from the reply's text: MuirBench counts a reply only if it *is* the option letter, AI2D and MMBench
extract a letter by rule.  The teacher often explains instead of answering with the letter (on MMBench a
third of its replies do not open with an option, and they average 355 characters), while the distilled
students almost always open with one.  So the teacher scores 5 % on MuirBench, below chance, and a student
can score above it on such a task without being better.  The scoring is left as the harness defines it;
`results.md` prints, per multiple-choice task, the share of replies that open with an option and the mean
reply length, and scores of models that reply differently are not comparable on those tasks.

`--smoke` runs two tasks per suite on 8 items and one RULER length with 4 samples: a check that the path
works, in minutes, never a score.  `swa` cannot mask left padding (its sinks are absolute positions) and is
evaluated one prompt at a time.
