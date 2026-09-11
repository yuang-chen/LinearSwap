Look at Gated DeltaNet-2: [https://arxiv.org/abs/2605.22791](https://arxiv.org/abs/2605.22791) - is it possible to do an in-place swap of gated deltanet on something like Qwen3.6 by populating the decoupled gates with the same parameter and post-training the model on long context again?

Yes — and this is one of the cleaner cases for an in-place architecture graft, because GDN-2 contains GDN as an *exact* algebraic special case, not an approximation. That changes the risk profile entirely: you can warm-start with a function-preserving init (zero loss increase at step 0), rather than distilling across a function mismatch the way softmax→linear grafts have to.

**The reduction is exact.** GDN-2's update (Eq. 10) is

S_t = (I − k_t(b_t⊙k_t)ᵀ) D_t S_{t−1} + k_t(w_t⊙v_t)ᵀ,

with erase gate b_t∈[0,1]^{d_k}, write gate w_t∈[0,1]^{d_v}, D_t=Diag(α_t). Gated Delta Rule-2 recovers KDA when b_t=β_t·1_{d_k} and w_t=β_t·1_{d_v}, and recovers Gated DeltaNet when the decay further collapses to a scalar. Setting all three to the tied values collapses Eq. 10 term-by-term to GDN's α_t(I − β_t k_t k_tᵀ)S_{t−1} + β_t k_t v_tᵀ (I checked it: the scalar α commutes with the projector, so even the decay-before-erase ordering matches). So "populating the decoupled gates with the same parameter" is exactly the right move — it's the tied subspace the authors point to, and the graft is loss-preserving in ℝ.

**Concretely, the init.** Widen the existing scalar β projection to d_k by broadcasting the same pre-activation row across all key channels → after σ, b_t=β_t·1. The decay branch depends on Qwen3.6's layer: if it's GDN with a per-head scalar A_log, tile the decay row (plus the exp(a) coefficient and dt_bias→δ) across d_k; if it's already KDA-style channel-wise, copy W_f directly. The decay-projection output width in config.json is the single fact that decides this. Conv, q/k L2-norm, and the output gate z all carry over untouched — only the recurrence-internal gates change.

**The trap is the write gate.** GDN has no value-side gate, so W_w is genuinely new — and the tempting move is to initialize it fresh/random/ones. Don't. Equivalence requires w_t=β_t at init, so W_w must be a tiled copy of the *same* β projection across d_v. Any other init breaks the function and eats a loss spike that post-training then has to climb back out of, for no reason.

**The tie won't stay tied,** which is the usual worry with identical-valued inits. It doesn't collapse here because b and w sit on structurally different gradient paths: b enters through the key-side erase read r_t=S̄_tᵀ(b⊙k) in d_k, w through the value-side target z_t=w⊙v in d_v. ∂L/∂b and ∂L/∂w differ from the first step, and the per-channel decay rows decouple for the same reason (each channel's gradient depends on its own k/state interaction). No manual symmetry-breaking needed, though an ε-jitter on W_w won't hurt and may speed divergence. Keep d_k vs d_v straight throughout — Qwen-Next GDN runs value head dim = 2× key head dim, so b and α broadcast to d_k, w to d_v, all from the one scalar.

**Will post-training actually use the new capacity?** Partially, and long-context is the right objective for it. GDN-2's reported gains are specifically multi-key long-context retrieval — the exact regime where decoupled erase/write matters — so continued long-context training supplies gradient that *rewards* pulling b and w apart. But a backbone pretrained under tied gates has organized its k/v geometry around scalar β; on a post-training budget (not a full pretrain) you'll recover a fraction of the from-scratch GDN-2 advantage, not all of it. Realistic expectation: modest retrieval gain at near-zero downside risk, not parity with native GDN-2.

Recipe:

1. Confirm GDN vs KDA decay from config (decay-proj output width).
2. Add W_w (→d_v), widen β→b (→d_k), widen-or-copy decay (→d_k).
3. Init all three by tiling the corresponding original scalar rows, including dt_bias and A_log; optional ε-jitter on W_w.
4. Swap to GDN-2's WY kernel (NVlabs repo), then **gate everything on a forward-parity check**: tiled-gate GDN-2 vs original GDN on a fixed batch should match to fp tolerance (decay is computed in fp32, WY solve has its own precision, so it won't be bit-exact — that's fine for continued training). Mismatch beyond tolerance means your tiling/layout is wrong, not your training.
5. Continue long-context training with a higher LR on the new gate params (W_w especially) and a lower LR on the backbone; optionally a brief gate-only warmup.
6. Monitor b–w channel divergence and the channel-wise decay spread. Flat curves ⇒ gate LR too low or context signal too weak.

One framing worth keeping in mind: unlike the Mamba-in-Llama / linearization grafts you know, you're not approximating anything here — the warm-start is exact, so the entire post-training budget goes to *exploiting* new expressivity rather than *recovering* lost function.
