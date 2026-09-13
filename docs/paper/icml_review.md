# LinearSwap: ICML paper review

Reviewed 12 September 2026. Draft: the attached nine-page “Are Recurrent Sequence Mixers Interchangeable? Recurrent Operator Transplantation in Pretrained Hybrid Language Models” (rot.pdf).

Code snapshot: [LinearSwap, commit 5a23c10323a47ca42a12e7ffaac09358ea8df2c8](https://github.com/yuang-chen/LinearSwap/tree/5a23c10323a47ca42a12e7ffaac09358ea8df2c8).

This is an author-facing assessment and static code audit, not an official conference review. Repository results are reported results, not independently reproduced measurements. No model training or GPU verification was performed: the local review environment lacks the project's PyTorch/FLA stack and weights. Source checkout was recovered with LFS downloads disabled after a benchmark asset returned 404.

## Assessment

**The topic is suitable for an ICML paper, but the current draft is a research proposal rather than a submission-ready paper. Filling numerical placeholders alone will not fix it.** The main risks are an inconsistent central equation, overlap with existing work, missing implementations of proposed contributions, and experiments that do not yet isolate those contributions.

The strongest direction is a controlled study of when pretrained recurrent computations transfer across operator families, a functional diagnostic of transfer difficulty, and a recovery method that improves on standard distillation under matched budgets.

The framework is useful infrastructure. Exact gate tiling and a collection of swaps are supporting components; they are unlikely to carry the novelty claim alone. A careful empirical discovery can carry an ICML paper without an elaborate new algorithm or a universal speedup. ICML explicitly values soundness, significance, originality, and new understanding of existing methods. The experiment sizes recommended here are specific to this project, not formal ICML requirements. [ICML reviewer guidance](https://icml.cc/Conferences/2026/ReviewerInstructions)

## 1. Strengths worth preserving

- The draft asks falsifiable questions about transferability, recovery cost, and long-context behavior.
- A shared pretrained backbone and fixed attention schedule control important confounders.
- It distinguishes recurrence algebra from complete block equivalence in the surrounding prose.
- DeltaNet and an SSD target provide useful non-exact cases.
- The code already provides a registry, initialization, cached generation, verification, block matching, global KL, SFT, and RULER integration.
- Numerical placeholders are clearly marked. Keep this discipline until results are reproducible.

## 2. Mathematical and naming corrections

### Equations (4) and (6) disagree

Page 2, Equation (4), uses

\[
S_t=(\alpha_t I-\beta_t k_tk_t^\top)S_{t-1}+\beta_t k_tv_t^\top.
\]

Page 3, Equation (6), described as its expansion, uses

\[
A_t=\alpha_t I-\alpha_t\beta_t k_tk_t^\top.
\]

These differ when decay is not one and the erase term is nonzero. Appendix A acknowledges two conventions but does not resolve the contradiction.

The repository documents and maps the convention

\[
\bar S_{t-1}=\alpha_t S_{t-1},\qquad
S_t=\bar S_{t-1}+\beta_t k_t(v_t-\bar S_{t-1}^{\top}k_t)^\top,
\]

so the consistent equation is

\[
\boxed{S_t=\alpha_t(I-\beta_t k_tk_t^\top)S_{t-1}
                 +\beta_t k_tv_t^\top.}
\]

State normalized-key conventions, query scaling, gate ranges, and orientation, and verify against the pinned FLA implementation using independent one-step and rollout tests. [Code: recurrence helpers](https://github.com/yuang-chen/LinearSwap/blob/5a23c10323a47ca42a12e7ffaac09358ea8df2c8/src/qwen_linswap/kernels/common.py)

For the code's DPLR convention \(S_t=D_tS_{t-1}+b_ta_t^\top S_{t-1}+k_t^wv_t^\top\), the map is \(D_t=\alpha_t I,\ a_t=\alpha_t k_t,\ b_t=-\beta_t k_t,\ k_t^w=\beta_t k_t\). The draft reverses the names of the rank-one factors; this is algebraically valid but needs an explicit equation-to-code mapping.

### The code does not install stock RWKV-7 or Mamba-2 blocks

The RWKV adapter keeps Qwen projections, convolutions, and output gating/normalization. It omits RWKV-7 token shift, value residual, and GroupNorm, and changes the decay parameterization. Its comments explain why FLA's native RWKV7Attention cannot directly express the desired pretrained decays.

Use **“RWKV-7-style DPLR recurrence with a Qwen-compatible block”** in tables and claims. Exactness applies to that constructed target. A native RWKV-7 comparison is optional for a recurrence-level paper, but necessary if claiming standard RWKV-7 architecture conversion or performance. [Adapter](https://github.com/yuang-chen/LinearSwap/blob/5a23c10323a47ca42a12e7ffaac09358ea8df2c8/src/qwen_linswap/kernels/rwkv7.py), [RWKV-7 paper](https://arxiv.org/abs/2503.14456)

Similarly, the SSD adapter uses simple-GLA kernels, normalized Q/K-like factors, Qwen preprocessing, and one SSD group per head. Call it a **Mamba-2-style SSD recurrence in a Qwen-compatible block**. Its initialization removes erase AND changes beta-scaled writes to delta-scaled writes. Its failure cannot be attributed solely to missing erase; add intermediate controls separating these changes. [SSD adapter](https://github.com/yuang-chen/LinearSwap/blob/5a23c10323a47ca42a12e7ffaac09358ea8df2c8/src/qwen_linswap/kernels/mamba2.py)

### DeltaNet is structurally constrained, not necessarily functionally nearby

Removing decay is a small architectural edit but can be a large functional change. Measure closeness instead of assuming it.

For unit keys and \(0\leq\beta_t\leq1\), the DeltaNet transition has spectral norm at most one. The documentation's statement that its state necessarily grows without bound is too strong. Lack of strict contraction can permit accumulation but does not establish divergence on every sequence.

A useful limited theorem is non-containment at matched state dimensions: for \(0<\alpha<1\), \(d_k>1\), and the stated gates, GDN contracts directions orthogonal to its key, whereas a single DeltaNet update leaves a nontrivial subspace unchanged. State these assumptions; do not claim impossibility for modified multi-update or larger-state architectures.

### “Zero-token” needs three separate labels

Distinguish analytic initialization without data, calibration-only initialization with all fitting costs counted, and subsequent recovery training. An optimized calibration initialization is not zero-data or zero-optimization. If an axis means “zero recovery tokens excluding calibration,” say so.

Equation (5) requires an actual optimizer, trainable set, normalization, budget, and stopping rule. A finite fitting routine is not automatically the global argmin written in the draft.

## 3. Novelty and related work

The most consequential omission is a direct recurrent-to-recurrent predecessor: the June 26, 2026 write-up “Swapping GDN for GDN-2 on Qwen3.5” already demonstrates function-preserving initialization, gate-only/full SFT, the same three training datasets, and long-context RULER evaluation on Qwen3.5-0.8B. Cite it explicitly. This overlap does not establish code provenance; separately document any reused implementation or recipe. The broad claim that prior conversion only starts from softmax attention is no longer defensible. [Original write-up](https://lutet.industries/posts/gdn2-swap/), [linked code](https://github.com/lutetjeff/gdn2-in-place)

| Prior work | Established component | Needed distinction |
|---|---|---|
| [GDN2 in-place](https://lutet.industries/posts/gdn2-swap/) | Exact GDN→GDN2 transfer and SFT | Non-containing targets, validated diagnostics, controlled recovery |
| [MOHAWK](https://arxiv.org/abs/2408.10189) | Mixing-behavior, block, prediction matching | Benefit beyond adapted staged matching |
| [LoLCATs](https://arxiv.org/abs/2410.10254) | Output matching followed by parameter-efficient recovery | Benefit beyond block matching plus LoRA |
| [Distill-then-Replace](https://arxiv.org/abs/2601.11667) | Local distillation and greedy replacement | Added value of the proposed schedule/criterion |
| [Taylor-Calibrate](https://arxiv.org/abs/2606.16429) | Teacher-statistics-based recurrent initialization | Recurrent-source calibration with stronger controls |
| [KL-guided layer selection](https://arxiv.org/abs/2512.20569) | Data-driven selection and hybrid distillation | Value beyond simple sensitivity/KL scoring |
| [HALO / HypeNet](https://arxiv.org/abs/2601.22156) | Efficient hybrid conversion and long-context evaluation | Separate recurrent-source transfer from attention/position changes |

Not every method is a directly applicable competitor. Reproduce applicable ingredients within the same recurrent-source setting, label adaptations, and document deviations. Do not invent a “corresponding full-attention teacher” for a hybrid checkpoint.

KDA and GDN2 should not be deferred just because metadata are pending: public papers exist and both are implemented here. Include KDA as an exact-transfer control and GDN2 as an additional containment/capacity control. The containment relationships are already discussed in operator work. [Kimi Linear](https://arxiv.org/abs/2510.26692), [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)

Verify bibliography metadata from primary sources. The draft's RWKV-7 author list, for example, omits authors shown on the linked paper. This targeted search does not prove absence of additional concurrent work.

## 4. Draft-to-code gap

| Draft component | Audited implementation | Required work |
|---|---|---|
| Analytic exact maps | Present for selected adapted kernels | Independent parity evidence and constraints |
| Constrained approximate projection, Eq. (5) | No solver found | Implement or remove claim |
| Local block distillation | Teacher-input MSE exists | Preserve as baseline |
| Trajectory/aligned-state loss | Not found | Sparse differentiable state supervision |
| Functional probes | Not found | Define common readout spaces and implement |
| Progressive replacement | One global kernel for recurrent layers | Per-layer operator maps and schedule |
| Frozen-backbone protocol | Default global KL/full SFT update everything | Explicit audited trainable scopes |
| Operator distance | Not found | Fitting, scoring, predictive validation |
| Token recovery curves | Step-based; partial token logs | Cumulative token/compute accounting |
| Multi-scale study | Hard-coded 0.8B default config | Actual checkpoint config loading |
| General LM/downstream evaluation | SFT validation and RULER | Raw-text NLL and task harness |
| Systems evaluation | No dedicated driver found | Synchronized timing/cache accounting |

Evidence: [distillation](https://github.com/yuang-chen/LinearSwap/blob/5a23c10323a47ca42a12e7ffaac09358ea8df2c8/src/qwen_linswap/pipeline/distill.py), [model](https://github.com/yuang-chen/LinearSwap/blob/5a23c10323a47ca42a12e7ffaac09358ea8df2c8/src/qwen_linswap/model.py), [loading](https://github.com/yuang-chen/LinearSwap/blob/5a23c10323a47ca42a12e7ffaac09358ea8df2c8/src/qwen_linswap/load_weights.py), [evaluation](https://github.com/yuang-chen/LinearSwap/blob/5a23c10323a47ca42a12e7ffaac09358ea8df2c8/src/qwen_linswap/pipeline/evaluate.py).

The current pipeline is a useful baseline for ROT, not the full method proposed in the PDF.

## 5. Experimental and implementation risks

### Metrics and data

Current validation measures assistant-token SFT loss on a small subset. The evaluation helper averages example-level mean losses equally. Exponentiating that quantity is not standard token-weighted corpus perplexity.

Compute headline NLL as total token loss divided by total valid target tokens, and PPL as its exponential, on separate raw text. Keep macro-averaged assistant SFT loss as a clearly named application metric. Separate calibration-fit, calibration-score, recovery, development and final evaluation; deduplicate documents before splitting. Add a no-anti-haystack ablation because the current SFT mixture is retrieval-oriented. [Loss utilities](https://github.com/yuang-chen/LinearSwap/blob/5a23c10323a47ca42a12e7ffaac09358ea8df2c8/src/qwen_linswap/sft_utils.py), [data code](https://github.com/yuang-chen/LinearSwap/blob/5a23c10323a47ca42a12e7ffaac09358ea8df2c8/src/qwen_linswap/data.py)

### Capacity and budgets

Reported gate trainables are approximately 0.59M for GDN, 7.4M for KDA, 38M for dense-gate KDA, and 113M for GDN2. This shows adaptation costs but does not isolate recurrence structure. Add GDN with parameter-matched adapters, rank sweeps where exact initialization remains possible, and full-fine-tuning controls. Report total/trainable parameters, state size and compute separately. Token-, parameter-, and compute-matched views are different comparisons. [Reported results](https://github.com/yuang-chen/LinearSwap/blob/5a23c10323a47ca42a12e7ffaac09358ea8df2c8/docs/framework.md)

Count calibration, repeated local passes, KD and SFT. Distinguish processed inputs from supervised targets and unique data. Maximum context length is not the actual length distribution: documentation reports many much shorter examples.

### Concrete code findings

| Finding | Consequence |
|---|---|
| CE backward slices flattened labels using a sequence-only chunk index | Batch>1 is incompatible; existing batch-one runs avoid this specific issue |
| Collator pads without a mask; model lacks general attention-mask input | Variable-length batching/packing needs explicit masking and recurrent resets |
| Generation uses next_token.item() for EOS | Batch>1 EOS handling fails; zero requested tokens also incorrectly generates one |
| SFT automatically resumes in a shared output folder; sampler uses seed+step | Runs can mix and resumption does not continue the original data stream |
| Distillation saves model/config only | No exact optimizer/RNG/sampler continuation |
| Evaluation display names omit configuration/seed identity | Distinct runs can collide |
| run.py returns final SFT checkpoints despite promising every checkpoint | Recovery curves need manifest-driven checkpoint evaluation |
| Verification prints diagnostics; regression uses one short prompt | Existing checks are insufficient for broad equality claims |
| LFS benchmark asset returned 404 during clone | Clean benchmark setup is currently not fully reproducible |

All except the observed LFS failure are static code findings, not executed bug reproductions. They do not invalidate every historical result. The documentation also identifies older results affected by a previously fixed CE-gradient scaling bug; exclude those from headline evidence.

### Evaluation breadth

Three mostly saturated NIAH tasks with 100 examples cannot establish broad improvement. Add harder multi-query recall, distractor density, overwrite and variable tracking, aggregation, and natural long-context tasks. Vary length and query position. Bootstrap complete examples; multiple answers from one example are dependent.

Evaluate teacher-input and student-input block errors, single-layer transplants, fixed groups, and all-layer transplants. This separates local mismatch, error accumulation, and compensation by retained attention.

## 6. Make the distance/predictor contribution testable

Equation (14) is a directed, data- and fitting-budget-dependent discrepancy, not automatically a symmetric or basis-invariant metric. “Transfer discrepancy” is a safer term.

- Raw Frobenius transition error can underweight important rank-one directions relative to a large diagonal component. Compare it with state-weighted one-step and multi-step readout error.
- Three target families are too few independent architecture observations for a strong predictor claim. Use held-out layer groups/scales and controlled within-family perturbations. Tokens and seeds do not create independent architecture samples.
- Compare predictive power against initial NLL/KL, block MSE, gate magnitudes, layer depth and parameter count. Account for the diagnostic's own compute.
- Predicting immediate degradation is weaker than predicting recovery cost. Validate both separately, including failures under domain/length shift.

Under identical external block inputs and a common state space, the identity

\[
E_t=A_t^{\rm target}E_{t-1}
+(A_t^{\rm target}-A_t^{\rm source})S_{t-1}^{\rm source}
+(B_t^{\rm target}-B_t^{\rm source})
\]

motivates accumulated functional mismatch. It does not prove optimizer convergence or full-model autoregressive behavior.

For unequal state spaces, define alignment and query maps before comparing states. A flexible aligner can hide mismatch: constrain it, fit only on calibration-fit data, freeze it for scoring, and use unseen sequences/probes in a common output space. Offline future probes may supervise training but must never enter deployed causal inference.

Use an absolute teacher-relative NLL tolerance as the primary recovery criterion, supported by downstream uncertainty. Closing 95% of a huge random-initialization gap can still leave an unusable model. Keep gap-closure secondary. Unrecovered runs are censored at the maximum tested budget, not assigned invented recovery times.

## 7. Systems claims must fit this setting

Swapping recurrent layers does not remove retained-attention KV. Equal state dimensions can mean identical recurrent-state storage across targets. The hybrid retains attention's length-dependent memory and computation.

As a configuration-derived illustration, 18 recurrent layers with 16 heads and 128×128 FP32 states occupy 18 MiB/sequence, excluding convolutions. Six attention layers with 2 KV heads of dimension 256 use approximately 1.5 GiB of BF16 KV at 131,072 tokens, excluding temporaries. These are calculations from the checked-in config, not peak-memory measurements.

Measure recurrent kernels, complete blocks and whole models separately. Audit repeated KV concatenation and grouped-KV expansion in components.py before attributing timing differences to operator families. Report shared-backend results and, optionally, separately labeled optimized-backend results.

An efficiency win is not mandatory for a scientific interchangeability paper. Revise Section 2's requirement that a transplant succeeds only if deployment improves: distinguish functional transfer success from deployment benefit.

## 8. Core experiments and manuscript changes

| Experiment | Question | Evidence |
|---|---|---|
| E0 parity | Which claims are exact? | Independent recurrence, block, cache and logit checks |
| E1 initialization | What survives before recovery? | Source, exact controls, DeltaNet, SSD; NLL/KL by length/layer |
| E2 recovery | Does ROT beat ordinary training/KD? | Quality versus all-in tokens and accelerator-hours |
| E3 ablations | Which ingredients matter? | Initialization×trajectory factorial; schedule separately |
| E4 mechanism | What fails at long horizons? | Recall/overwrite/tracking and position-resolved drift |
| E5 prediction | Does discrepancy add predictive value? | Group-held-out prediction versus cheap predictors |
| E6 capacity | Is improvement simply more trainable parameters? | GDN adapter controls and gate-rank sweeps |
| E7 scope/systems | Does the result generalize and help deployment? | Larger supported checkpoint and quality/latency/memory |

Start with 0.8B, three training seeds for decisive small-scale comparisons, and one larger verified checkpoint. Do not promise a “3B” checkpoint without checking its identity and configuration. Pilot at 10M/100M processed tokens and extend to 1B under a common declared rule once costs and correctness are known. A small-budget failure is not proof that a target cannot recover.

Prioritize these manuscript changes:

1. Correct the recurrence and spell out adapted target blocks.
2. Cite the direct predecessor and distinguish ROT from existing distillation ingredients.
3. Include KDA/GDN2 controls and define calibration versus recovery cost.
4. Make the solver, alignment, losses and schedule executable from their descriptions.
5. Align frozen-backbone claims with actual parameter masks.
6. Replace generic evaluation promises with fixed endpoints and claim-to-experiment mappings.
7. Drop headline components that do not help, retain negative findings, and generate tables from raw evidence.
8. Verify bibliography and the chosen submission year's official template.

**Submission decision:** pursue ICML if the study establishes a useful, reproducible finding beyond exact gate tiling and ordinary distillation. If the new loss or predictor fails, rewrite the contribution around the supported scientific findings rather than retaining unsupported claims.
