# LinearSwap: coding-agent and experiment checklist

> **Status annotation (coding agent, 13 September 2026).**  Audit baseline 5a23c10 is
> two commits behind HEAD (c59bff9) plus uncommitted work; the differences that matter
> for this list are recorded here so the tasks below are read against the real state.
> Legend: **DONE** / **PARTIAL** / **OPEN** / **DISAGREE** (see docs/paper/review_response.md).
>
> | Task | Status | Notes |
> |---|---|---|
> | A01 setup | PARTIAL | `english_words.json` LFS pointer replaced by the real asset (uncommitted); scipy added for `fwe`; HotpotQA fetched from the HF mirror (both files git-ignored, manifest/checksums still missing). No pyproject/lock/preflight yet. The CE-scaling-bug inventory exists in docs/framework.md (only `docs/gdn2_experiment_log.md` numbers are affected). |
> | A02 losses | OPEN (confirmed) | Batch>1 label slicing bug and macro-averaged `evaluate()` confirmed by reading `sft_utils.py`; all reported runs use batch 1, so results stand. Fix planned together with token-weighted NLL. Packing/masking not implemented — batches are single unpadded sequences. |
> | A03 recurrence contracts | PARTIAL | Kernel-level parity (chunk + recurrent), layer-level vs the HF layer, block/cache/logit checks exist (`verify`, `tests/test_kernels.py`) but print rather than emit structured results; no FP64 independent reference; tolerances were not pre-declared. Naming per review adopted in docs ("RWKV-7-style DPLR / Mamba-2-style SSD recurrence in a backbone-compatible block"). |
> | A04 cache/checkpoints | PARTIAL | `evaluate --models label=path` (added 13 Sep) removes the display-name collision. `current_pos` audit, EOS/batch>1 generation, explicit resume, RNG/sampler state, distill resume: OPEN. `run` docstring corrected to "final checkpoint per mode". |
> | A05 protocol | OPEN | Validation loss is assistant-token SFT loss on the first N examples; no raw-text NLL split, no dedup, no pre-declared recovery margin. |
> | A06 configs / mixed swaps | OPEN | Config is still the fixed 0.8B dict; per-layer kernel maps not implemented. Planned as part of the "backbone-agnostic" refactor (load HF text config; per-layer kernel list). |
> | A07 calibrated init | OPEN | No solver. The planned cheap control is a Mamba-2 variant that keeps beta-scaled writes (separates "erase removed" from "write scale changed"). |
> | A08 distillation baselines | PARTIAL | Named baseline exists (teacher-input block MSE → global KL). New since audit: **SFT-only** controls for both approximate targets at recipe-matched (50) and compute-matched (500) steps — DeltaNet 13.8→6.70 (SFT-only 500) vs 2.46 (distill 500); Mamba-2 1.83 (SFT-only 500) vs 1.73 (distill) / 1.49 (distill+SFT 50). Hard-task RULER for these is running. LM-only / KD-only / block+LoRA / MOHAWK-style variants: OPEN. |
> | A09 trajectory/probes | OPEN | Not started. |
> | A10 progressive replacement | OPEN | Not started (depends on A06 per-layer maps). |
> | A11 core campaign | PARTIAL | One seed, 0.8B only. Source/exact/approximate groups all have base + gate-only + full SFT + (for approximate) distill / distill+SFT / SFT-only rows; token accounting is step-based. |
> | A12 ablations | PARTIAL | Gate-rank (kda vs kda_fullgate vs gdn2 vs rwkv7) and gate-only vs full exist; GDN parameter-matched LoRA, single-layer / group swaps, overwrite / delayed-recall probes: OPEN. |
> | A13 predictor | OPEN | Not started. |
> | A14 evaluation breadth / systems | PARTIAL | Hard RULER at 131K done for 8 models (multikey_2/3, multiquery, vt, cwe, fwe, qa_1/2; 50 samples): exact kernels tie (avg ≈ 62), Mamba-2 avg 36, SFT destroys `cwe` (36→3) for every kernel. Length sweep, MQAR, lm-eval harness, bootstrap CIs, timing/memory driver: OPEN. Layer timing note: RWKV-7 DPLR layer ≈ 2× KDA fwd+bwd cost (measured, not yet in a table). |
> | A15 second scale | OPEN | Candidate checkpoints identified (Qwen3.5-4B/9B; 3.6/3.8-27B need grouped value heads and fit only gate-only SFT here); nothing run. |
> | A16 artifacts | OPEN | Tables in docs are hand-assembled from `outputs/eval/*/summary.json`; a generator script is the natural first step. |
>
> **Corrections to the audit text.** (i) `docs/original_task.md` exists at HEAD (it was `TASK.md` at the audited snapshot's parent). (ii) "run.py collects final SFT checkpoints" — correct; the docstring was the error. (iii) The predecessor write-up (GDN→GDN2 on Qwen3.5) is not only related work: this repository's code descends from that project (`docs/gdn2_experiment_log.md` is its experiment log); it is credited in the README from the next revision on.

Audit baseline: commit 5a23c10323a47ca42a12e7ffaac09358ea8df2c8, 12 September 2026.
Companion document: LinearSwap_ICML_Review.md.

This is an implementation handoff. The tasks below have not been executed in the review. Proposed paths/interfaces are suggestions, not existing commands. Read AGENTS.md, README.md and docs/framework.md; reconcile current HEAD with the audit before changing code. AGENTS.md references docs/original_task.md, absent in the audited snapshot.

## Execution order

| Phase | Tasks | Exit condition |
|---|---|---|
| P0 correctness | A01–A05 | Setup, loss, recurrence, cache and provenance gates pass |
| P1 method pilot | A06–A10 | Approximate targets and baselines train reproducibly |
| P2 evidence | A11–A14 | Recovery, mechanisms, prediction and systems measured |
| P3 generalization/release | A15–A16 | Larger-scale replication and traceable paper artifacts |

Record every task outcome and evidence path in a proposed outputs/paper/index.json. Preserve failed, unsupported, OOM and unrecovered runs. Do not treat old documentation tables as newly reproduced evidence.

## A01 — P0: reproducible setup

Existing: root setup instructions and vendored RULER.
Proposed: pyproject.toml, dependency lock, scripts/preflight.py.

- [ ] Pin Python, PyTorch, Transformers, Triton, FLA, convolution backend, model/tokenizer revisions and RULER revision. Initially preserve the working environment instead of upgrading opportunistically.
- [ ] Repair benchmark asset delivery: clean clone returned LFS 404 for RULER/scripts/data/synthetic/json/english_words.json. Supply the actual versioned asset/checksum; skipping smudge only enables source inspection.
- [ ] Record git SHA/dirty status, GPU count/model/memory, CUDA, driver, packages, kernel flags, TF32 and state/parameter precision.
- [ ] Add preflight validation of weights, benchmark assets, supported model shapes, a small forward/backward and available memory.
- [ ] Inventory historical results affected by the older CE-gradient scaling bug documented in framework.md; exclude them from headline paper evidence.

Acceptance: fresh checkout and documented setup can run a small evaluation without undocumented local files.
Outputs: environment.json, asset_manifest.json, legacy_results_audit.json.

## A02 — P0: losses, masks and token accounting

Existing: src/linswap/sft_utils.py, data.py, pipeline/distill.py, pipeline/posttrain.py.
Proposed tests: test_losses.py, test_padding.py, test_token_accounting.py.

- [ ] Fix batch>1 CE backward indexing: flatten hidden states and shifted labels consistently across batch and sequence. Current sequence-only label slicing is incompatible with flattened batch chunks.
- [ ] Return summed evaluation loss and valid-target count; aggregate globally. Keep macro-averaged SFT loss as a separately named metric.
- [ ] Compare chunked CE/KL losses and gradients with a dense small reference: hidden states, LM head, tied embeddings, temperature, loss scaling, accumulation, ignored labels, uneven chunks, batch 1/2 and all-ignored examples.
- [ ] Define token-mean accumulation explicitly; weight microbatches by target counts instead of silently averaging unequal-length sequence means.
- [ ] Track shifted CE targets, valid KL positions, input tokens, repeated exposures, unique data where measurable and layer-token applications.
- [ ] Add cumulative counters for calibration fitting, local KD, global KD and SFT. Checkpoint at predeclared token budgets, recording any overshoot.
- [ ] Add padding masks and positions for attention, convolution and recurrent state before variable-length batching. Until supported, reject padded/packed batches.
- [ ] Reset recurrent/convolution states and mask attention at packed document boundaries. Verify against separate-document execution.

Acceptance: reference loss/gradients agree under declared precision tolerances; token NLL equals direct summation; no padding/document leakage; counters distinguish input from supervised tokens.

## A03 — P0: recurrence contracts and exactness

Existing: kernels/common.py and all kernel files, registry.py, pipeline/verify.py, tests/test_kernels.py.
Proposed: analysis/reference_recurrences.py, docs/operator_contracts.md, tests/test_recurrences.py.

- [ ] Adopt one documented state orientation and record interface transposes.
- [ ] Freeze GDN as S_next = alpha × (I − beta k kᵀ) S + beta k vᵀ, with source-compatible key normalization and explicit query scaling. Reconcile draft Eq. (4), Eq. (6) and Appendix A.
- [ ] Implement independent CPU FP64 references; do not call production kernels from reference tests.
- [ ] Check normalized keys, decay order, erase/write scaling, nonzero initial states and readouts against pinned FLA.
- [ ] Record recurrence family separately from complete block adaptation. Label rwkv7 as backbone-compatible RWKV-7-style DPLR and mamba2 as backbone-compatible SSD.
- [ ] Replace a bare exact_init assertion with metadata describing supported head/state shapes, parameter ranges, preprocessing and precision.
- [ ] Test one-step and multi-step recurrences, random initial states, gate edge cases, repeated keys, all recurrent layer positions, natural text and synthetic inputs.
- [ ] Report FP64 algebra, FP32 kernels, BF16 kernels, complete block, cache and full-model errors separately. Include absolute/normwise-relative error and KL; top-1 alone is insufficient.
- [ ] Establish tolerances from source controls and direct target-vs-source comparisons before final evaluation. Do not choose thresholds after target failures.
- [ ] Make verification emit structured results and fail when required contracts fail. Non-exactness is not itself a kernel correctness failure.

Acceptance: every exact claim has a mathematical map and independent numerical evidence; approximate targets pass recurrence correctness tests.
Outputs: operator_contracts.json, parity_results.jsonl, equation-to-code note.

## A04 — P0: cache and checkpoint correctness

Existing: model.py, components.py, load_weights.py, pipeline/posttrain.py, distill.py, evaluate.py, run.py and RULER wrapper.
Proposed tests: test_cache.py, test_resume.py, test_checkpoint_identity.py.

- [ ] Compare full forward, arbitrary chunked prefill and token decode; test lengths around 63/64/65 and repeated requests.
- [ ] Audit model-global current_pos; use per-cache/request positions where needed. Test interleaved independent caches.
- [ ] Fix zero-token generation, batchwise EOS and supported EOS lists. Reject unsupported sampling settings explicitly.
- [ ] Serialize complete architecture, per-layer kernels, source/tokenizer revision, initialization and trainable scope; load strictly.
- [ ] Require explicit resume and reject incompatible output-directory reuse. Current automatic latest-checkpoint loading can mix experiments.
- [ ] Save/restore optimizer, scheduler, stage, counters, Python/NumPy/Torch/CUDA RNG, data cursor and sampler/generator state.
- [ ] Replace seed+global_step sampler restarting with real continuation. Add equivalent resume support for distillation, currently model/config-only.
- [ ] Use full-config-plus-seed run IDs in checkpoint/evaluation paths; current display names can collide.
- [ ] Evaluate every declared budget checkpoint from a manifest. Current run.py collects final SFT checkpoints despite its broader docstring.

Acceptance: interrupted small training resumes with the same data stream/counters and expected numerical agreement; requests do not leak cache state.

## A05 — P0: immutable experimental protocol

Existing: data.py and pipeline/evaluate.py.
Proposed: configs/paper/protocol.yaml, data_manifests/, analysis/metrics.py.

- [ ] Separate calibration-fit, calibration-score, recovery-train, development and final-test documents. Detect exact/near duplicates before splitting.
- [ ] Pin datasets, licensing, hashes, mixtures, tokenizer/chat template and truncation; save actual length distributions.
- [ ] Use generic text for core recovery/NLL. Keep LongAlign/LongAlpaca/anti-haystack SFT as a separate application stage; include a no-anti-haystack ablation.
- [ ] Verify exact source identity, base/instruction status, context window and layer schedule; do not infer a model's existence from draft scale placeholders.
- [ ] Define primary recovery as student NLL ≤ teacher NLL + a predeclared tolerance. A proposed pilot starting margin is 0.05 nats/token; justify/freeze it before final runs.
- [ ] Define downstream non-inferiority margins separately and report confidence intervals. Keep random-gap closure secondary.
- [ ] Record recovery crossing intervals between evaluation budgets. Require persistence at the next planned checkpoint where available, otherwise mark provisional.
- [ ] Treat never-recovered runs as right-censored at the maximum budget, not exact recovery times.
- [ ] Separate analytic zero-data initialization, calibration-only initialization and post-calibration recovery.
- [ ] Share token streams and hyperparameter-search allowances. Treat token-matched and compute-matched comparisons as distinct.
- [ ] Pilot at 0/10M/100M processed input tokens; add checkpoints to resolve crossings. Count fitting/local passes in all-in cost. Extend to 1B only under a predeclared common rule.

Acceptance: the protocol alone specifies data permissions between stages, endpoints, budgets, comparison groups and exclusions.

## A06 — P1: model configs, mixed swaps and trainable scopes

Existing: config.py, model.py, load_weights.py, registry.py.

- [ ] Load architecture from actual HF text-model configs and saved native configs; the current default is a fixed 0.8B dictionary.
- [ ] Validate grouped key/value heads, state dimensions, RoPE, attention schedule and embedding tying. Reject unsupported shapes rather than silently reshaping.
- [ ] Support per-layer kernel maps and selected transplant layers; preserve unchanged weights.
- [ ] Resolve gate/new parameters by each layer's own kernel spec.
- [ ] Add scopes for new gates, recurrent blocks, recurrent blocks plus adapters, and all parameters.
- [ ] Hash frozen parameters before/after training. Primary controlled-transfer runs freeze retained attention, FFNs, embeddings and non-transplanted norms; wider tuning is a separate regime.

Acceptance: mixed models round-trip strictly, only requested layers change at init, and named trainable masks match actual updates.

## A07 — P1: calibrated approximate initialization

Existing: kernels/deltanet.py, kernels/mamba2.py and weight-copy helpers.
Proposed: pipeline/calibrate.py and analysis/operator_factors.py.

- [ ] Expose transition/write/read operations and factor diagnostics without constructing dense transition tensors across long sequences.
- [ ] Implement named random-recurrence, projection-copy and function-aware calibrated initializations. Preserve the same shared backbone and specify which projections each copies.
- [ ] Fit only target-valid parameters on teacher-input blocks; specify normalized objectives, optimizer, bounds, tokens, steps, seed and stopping.
- [ ] For DeltaNet, calibrate available write/erase/output parameters without adding a decay gate.
- [ ] For SSD, fit decay/write/read under its declared parameterization.
- [ ] Add diagnostic intermediate recurrences separating erase removal, write-scale replacement and decay removal. Label them as controls, not native architectures.
- [ ] Compare coefficient error, state-weighted one-step error and short multi-step readout error.
- [ ] Score on disjoint calibration-score data. Report nonfinite states, stability violations and fitting costs. Do not claim the numerical solver finds a global argmin.

Acceptance: fitted models remain in the declared family; fitting and held-out residuals are distinct. Improvement is an experimental outcome, not a pass condition to manufacture.

## A08 — P1: configurable strong distillation baselines

Existing: pipeline/distill.py.
Proposed: training/objectives.py, training/parameter_groups.py and baseline configs.

- [ ] Preserve existing teacher-input block MSE → global KL as a named baseline.
- [ ] Add LM-only, logit-KD-only, block+global-KD and block+LoRA recipes with matched target definitions.
- [ ] Independently configure CE/KL/local weights, temperature, token masks, trainable scope and initial checkpoint.
- [ ] Add an adapted MOHAWK-style short-window mixing/readout alignment baseline where mathematically valid; document deviations. Avoid quadratic mixing matrices at 128K.
- [ ] Measure local fit on teacher inputs and held-out residuals on actual student inputs.
- [ ] Normalize layer losses or explicitly report their weighting; raw MSE sums can favor large-magnitude blocks.
- [ ] Include source GDN trained with the same data, objectives and scope.
- [ ] Give all baselines a comparable tuning allowance; a poorly chosen learning rate is not evidence of non-transferability.

Acceptance: differences among methods are explicitly isolated and costed.

## A09 — P1: sparse trajectory and functional probes

Proposed: training/trajectory.py, analysis/probes.py, tests/test_trajectory_loss.py.
Extension points: kernel forwards and state/cache interfaces.

- [ ] Start with identity alignment for shared equal-shape coordinates.
- [ ] Define constrained maps for different state spaces, including dimensions, rank, regularization and calibration split.
- [ ] Define common functional outputs: shared queries for identity coordinates, consistent transformed queries or shared block-output space otherwise.
- [ ] Use sparse log-spaced state anchors and fixed write/query event anchors; compare realized-query and held-out random probes.
- [ ] Verify differentiability of returned student states. Inference-cache final states may be detached or lack backward support.
- [ ] Check finite nonzero gradients for intended gate groups against a short differentiable reference rollout.
- [ ] Use memory-bounded chunk/anchor extraction; never store every head's full matrix state at every token over long contexts.
- [ ] Freeze teacher signals; constrain/freeze alignment when measuring held-out discrepancy.
- [ ] Remove auxiliary modules at inference. Offline future probes may supervise training but cannot enter deployed causal inference.
- [ ] Report extraction, alignment and probe costs.

Acceptance: reference losses/gradients agree; memory obeys the documented policy; inference does not depend on training-only modules.

## A10 — P1: progressive replacement

Existing: model.py, pipeline/distill.py and run.py.
Proposed: training/replacement.py.
Dependencies: A06 and A08.

- [ ] Implement all-at-once, shallow-to-deep, reverse, seeded random and sensitivity-based schedules.
- [ ] Score sensitivity on development/calibration data only; include its cost.
- [ ] Keep final target layer sets identical across schedules. Choosing which layers remain GDN is a separate experiment from order.
- [ ] Divide one fixed total budget among stages. Do not allocate a full baseline budget to every replacement group.
- [ ] Update optimizer groups correctly as layers are introduced/unfrozen; record previously transplanted trainables.
- [ ] Save schedule and active layer map in checkpoints.

Acceptance: schedules are replayable, preserve required frozen parameters and converge to the same final architecture.

## A11 — P2: core recovery campaign

Proposed: configs/paper/core/ and scripts/run_paper.py.
Start with one-seed 0.8B pilots for correctness/cost; freeze final configs and use three training seeds for decisive small-scale comparisons.

| Group | Targets | Methods | Purpose |
|---|---|---|---|
| Source | GDN | Untouched, same-data LM/KD/SFT, matched adapters | Adaptation-only control |
| Exact | KDA, GDN2, adapted DPLR | Analytic init and matched adaptation | Preservation and adaptation cost |
| Approximate init | DeltaNet, adapted SSD | Random recurrence, copy, calibration | Initial mismatch |
| Approximate recovery | DeltaNet, adapted SSD | LM, logit KD, block+KD, block+LoRA, adapted staged matching | Strong baselines |
| ROT | DeltaNet, adapted SSD | Calibration + local + trajectory/probe, schedule variant | Increment over baselines |

- [ ] Evaluate initialization and each declared checkpoint with the same backbone wrapper.
- [ ] Record all-in/stage tokens, accelerator-hours, parameter/state counts, NLL/KL and per-task scores.
- [ ] Include failed/censored runs and common budget-extension rules.
- [ ] Evaluate before and after any later retrieval SFT; keep these claims separate.

Acceptance: complete comparable recovery curves exist for both approximate targets, regardless of whether ROT wins.

## A12 — P2: mechanism and capacity ablations

- [ ] Run 2×2 calibrated-init on/off × trajectory/probe on/off on the same block+global baseline.
- [ ] Separate state alignment from functional probes; compare identity/raw-state matching and constrained alignment.
- [ ] Test replacement schedules separately. Avoid duplicate “no progressive” and “all-at-once” rows.
- [ ] Compare gate-only, recurrent-only and full-model training.
- [ ] Add GDN parameter-matched LoRA/adapters, plus gate-rank sweeps that preserve exact initialization where claimed.
- [ ] Distinguish state-size controls from parameter-count controls.
- [ ] Test single-layer, fixed-group and all-recurrent-layer swaps.
- [ ] Add no-anti-haystack, repeated-key overwrite, distractor interference, delayed recall and variable-tracking tests.

Acceptance: each claimed ingredient has a nonredundant ablation with costs and uncertainty.

## A13 — P2: predictive transfer discrepancy

Proposed: analysis/transfer_discrepancy.py and analysis/predict_recovery.py.

- [ ] Compute directed held-out normalized factor, state-weighted and multi-step functional discrepancies.
- [ ] Compare with cheap predictors: initial model NLL/KL, block MSE, decay/removal magnitude, layer index and parameter count.
- [ ] Build variation using layer groups/scales and controlled within-family perturbations; label artificial recurrences.
- [ ] Reserve held-out family/scale/layer groups before fitting diagnostic weights.
- [ ] Use grouped cross-validation and clustered bootstrap. Tokens, layers and seeds from the same source are not independent architecture samples.
- [ ] Handle interval-valued and right-censored recovery times; never assign a fictitious time to a failed run.
- [ ] Separate exact zero-cost cases from positive-cost prediction.
- [ ] Test length/domain shift and report diagnostic overhead/sensitivity.

Acceptance: a held-out comparison against cheap predictors exists. If discrepancy adds no predictive value, demote the headline claim.

## A14 — P2: quality breadth and systems

Existing: pipeline/evaluate.py, RULER wrapper, components.py, model.py.
Proposed: evaluation/lm_eval_adapter.py, analysis/bootstrap.py and pipeline/benchmark.py.

- [ ] Add raw-text token NLL and per-position bins.
- [ ] Add a standard task-harness adapter for loglikelihood and greedy generation; cross-check the source against its original wrapper.
- [ ] Core candidates: HellaSwag, PIQA, ARC-Challenge and WinoGrande; MMLU/GSM8K at scales with informative source performance. Pin exact task IDs/revisions/prompts/shots.
- [ ] Expand RULER categories, including tracking and aggregation; add natural long-context tasks and MQAR/overwrite diagnostics.
- [ ] Evaluate 4K/16K/32K/64K/128K where supported. Record actual post-template lengths and generation headroom.
- [ ] Reuse immutable generated examples across models; save IDs, generator seeds, raw predictions and scoring versions.
- [ ] Size final samples from pilot uncertainty; approximately 500 examples/task/length is a planning starting point where feasible, not a guarantee of resolving small effects.
- [ ] Bootstrap whole examples and distinguish sample uncertainty from training-seed variability.
- [ ] Benchmark prefill and decode separately, fixed prompt/output lengths, batch 1/4/8 where supported, warmup, synchronization and repeated trials.
- [ ] Report median/dispersion of throughput, TTFT, ITL, peak allocated/reserved memory and training cost; disclose compilation/preprocessing treatment.
- [ ] Separate microkernel, full-block and whole-model timing. Audit KV concatenation/group expansion before architecture conclusions.
- [ ] Report recurrent states, convolution states, retained-attention KV, weights and temporaries separately, including dtype.
- [ ] Compare against GDN on a common backend; label optimized-backend comparisons separately.
- [ ] Use fixed-output-length timing to avoid EOS-length confounding, alongside normal task generation.

Acceptance: measurements separate recovery, retrieval specialization and implementation speed. Equal-state-size swaps may show no memory benefit; report that result.

## A15 — P3: another supported scale

- [ ] Select an actually available larger GDN checkpoint after config inspection; do not assume the draft's “3B” exists.
- [ ] Validate source parity and grouped-head/state mappings first.
- [ ] Repeat source, exact control, strongest baseline and ROT; include both approximate targets where feasible.
- [ ] Freeze hyperparameters from small-scale development or give methods equal retuning budgets.
- [ ] Report seed counts honestly; one large run does not provide training-seed uncertainty.
- [ ] An independent backbone is valuable if practical. If all sources are Qwen, narrow claims accordingly; a larger Qwen is not cross-backbone evidence.

Acceptance: the central finding is tested beyond the tuning checkpoint, with explicit scope limits.

## A16 — P3: paper artifacts and evidence mapping

Proposed: scripts/make_paper_artifacts.py, paper/figures/, paper/tables/, docs/claims_to_evidence.md.

- [ ] Generate tables/figures from versioned logs, not hand-entered values. Missing results remain NA with reasons.
- [ ] Produce parity/containment, initialization, recovery, ablation, long-context/drift, prediction and quality/systems figures.
- [ ] Include per-task/seed tables, stage costs, parameter masks, failed runs and checkpoint manifests.
- [ ] Trace each plotted point to run IDs, source/data revisions and aggregation code.
- [ ] Compare directly with the GDN2 in-place predecessor and relevant distillation methods; label adaptations versus original reproductions.
- [ ] Record code/kernel/recipe provenance and preserve upstream licenses.
- [ ] Correct misleading documentation about target naming, divergence, budgets, checkpoint evaluation and legacy results.
- [ ] Map each manuscript claim to supported, contradicted, untested or scope-limited evidence.
- [ ] Provide one tiny end-to-end smoke command and manifest-driven full reproduction commands. Validate tables against raw totals and inspect labels/units.

Acceptance: a new agent can reproduce a representative run and regenerate the paper artifacts.

## Required result fields

Each JSONL record should contain run ID, git SHA, source/model/tokenizer revisions, recurrence family, block adaptation, per-layer map, initialization, training scope, objectives, schedule, seed, stage, step, cumulative input/supervised/teacher tokens, layer-token applications, calibration costs, accelerator-seconds, total/trainable parameters, state bytes, precision, data/evaluation hashes, checkpoint ID, metric/value/sample count and status.

Use null plus a reason for unmeasured values. Save batch/sequence shapes and backend identity for timings, and per-example predictions for uncertainty. Distinguish wall time from GPU-count × elapsed time.

## First work session

1. Inspect current HEAD and record changes since this audit.
2. Complete setup and loss/accounting checks before research runs.
3. Establish recurrence, cache and checkpoint contracts.
4. Freeze the protocol and run a small DeltaNet/SSD baseline pilot.
5. Measure pilot cost, implement the new objectives/schedule, then run fixed comparisons.

The critical path is correctness, matched baselines and evidence for the recovery mechanism. Expanding the kernel catalogue is lower priority.
