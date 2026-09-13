# Response to the ICML review (docs/paper/icml_review.md)

Written 13 September 2026 by the coding agent after re-reading the review against the
current code (HEAD c59bff9 + uncommitted work).  Verdicts: **confirmed** (true and still
open), **fixed**, **addressed in docs**, **disagree / nuance**.

## Concrete code findings (review §5)

| Finding | Verdict | Note |
|---|---|---|
| CE backward slices flattened labels with a sequence-only chunk index (batch>1 broken) | confirmed | `sft_utils.py`: `h_chunk` is `[b*chunk, d]`, `l_chunk` is `flat_labels[start:end]`. All runs use batch 1. Fix: iterate over flattened rows. |
| `evaluate()` averages example-level means (not token-weighted PPL) | confirmed | Report as "assistant-token SFT loss, macro-averaged"; add token-weighted raw-text NLL. |
| Collator pads without a mask; no attention mask / recurrent reset | confirmed | Only unpadded single sequences are used; packing must reset states. |
| `next_token.item()` EOS handling; 0 requested tokens generates 1 | confirmed | Batch-1 assumption in `model.generate`. |
| Automatic resume in a shared folder; seed+step sampler | confirmed | Make resume explicit; persist sampler/RNG state. |
| Distillation saves model/config only | confirmed | |
| Display names can collide | fixed | `evaluate --models label=path` (13 Sep). |
| `run.py` returns final checkpoints despite the docstring | confirmed | Docstring corrected; manifest-driven checkpoint lists still to do. |
| Verification prints diagnostics; regression uses one short prompt | confirmed | `verify` covers layers 0/1/2 at T=64/1024, logits at 8–4096, cache, generation — but unstructured and thresholds not pre-declared. |
| LFS asset 404 (`english_words.json`) | fixed | Real file in the working tree (uncommitted); manifest/checksum still to add. |

## Mathematics and naming (review §2)

* Recurrence convention: the code implements `S_t = α_t (I − β_t k̂ k̂ᵀ) S_{t−1} + β_t k̂ v_tᵀ`
  (decay applied before the erase, keys L2-normalised, query scale 1/√d_k); the review's boxed
  equation is the correct one and the draft's Eq. (6) should be changed to it.  The DPLR map in
  `kernels/rwkv7.py` is `a_t = k̂ ⊙ e^{g}` (read-out of the decayed state), `b_t = −β k̂`,
  `k^w_t = β k̂`, `gk_t = g` — the review's `a_t = α k_t` is the same map written per head.
* Naming: adopted.  Tables and docs say "RWKV-7-style DPLR recurrence" and "Mamba-2-style SSD
  recurrence" in a backbone-compatible block; exactness claims refer to those constructed targets.
* DeltaNet divergence wording: **agree**.  With unit keys and β∈[0,1] the DeltaNet transition is
  non-expansive; the state does not "grow without bound".  Docs are being reworded to "does not
  contract; stale associations persist until overwritten".  The non-containment statement (GDN
  contracts the orthogonal complement of the key, one DeltaNet step leaves it unchanged) is the
  right limited theorem.
* Mamba-2 attribution: **agree** that the current adapter removes the erase *and* changes
  β-scaled writes to Δ-scaled writes, so the multi-key collapse cannot be pinned on the erase alone.
  The planned control keeps β-scaled writes (one missing term).  Docs are being reworded from
  "the erase term is what … relies on" to "consistent with the missing erase; the write-scale
  change is not yet separated".
* "Zero-token": the three labels (analytic init / calibration-only / recovery) are adopted for
  the paper; the code currently has only analytic init and recovery.

## Novelty / related work (review §3)

* Predecessor: **agree, and stronger** — this repository is a continuation of that project
  (its experiment log is `docs/gdn2_experiment_log.md`; the initial task statement is
  `docs/original_task.md`).  It must be cited and credited, and the README acknowledgements are
  being extended accordingly.
* KDA and GDN2 as controls: already in every table.
* The related-work table's arXiv identifiers were not verified here; verify each against the
  primary source before citing.

## Experiments since the audit (relevant to §5, §8)

* Hard RULER at 131K (essay haystacks with distractors, multi-query, variable tracking,
  word extraction, SQuAD/HotpotQA), 8 models, 50 samples: exact kernels tie on every task
  (avg ≈ 62); Mamba-2-style SSD avg 36 (multikey_3 = 6); the retrieval-flavoured SFT destroys
  common-word extraction for every kernel (36 → ~3) — the "no-anti-haystack ablation" the
  review asks for is therefore mandatory, not optional.
* SFT-only controls for both approximate targets (50 and 500 steps): DeltaNet 6.70 vs 2.46
  with distillation at equal steps; Mamba-2 1.83 vs 1.73 (distill) / 1.49 (distill+SFT).
  Hard-task RULER for these is running.

## Where I disagree or would re-prioritise

* "Expanding the kernel catalogue is lower priority" — agreed for the paper; note that Mamba-1/3
  are blocked in this environment regardless (mamba_ssm needs Triton ≥ 3.3; see docs/framework.md).
* Systems section: agreed that equal state sizes give no memory win; the one measurable systems
  fact so far is the RWKV-7 DPLR layer costing ≈ 2× KDA in forward+backward.  Report it as such.
* Three seeds at 0.8B and a second scale are the two items most likely to change conclusions;
  most other TODO items are engineering hygiene that should be done but will not move the paper.
