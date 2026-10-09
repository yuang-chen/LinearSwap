# Gate and stage diagnostics

Two questions about what distillation actually does to a swapped layer, and the measurements that
answer them:

1. **Do the tiled per-channel gates learn anything?**  Several kernels give a gate more freedom than
   the backbone's (GDN has one scalar per head; GDN-2, KDA, RWKV-7 and GLA have one value per
   channel).  Every such kernel is initialised by *tiling* the pretrained scalar across the channels,
   so at step 0 the extra freedom is unused by construction.  Measuring how far the channels move
   apart says whether training uses the capacity the architecture adds.
2. **Which training step actually moves the swapped layers?**  The recipe is three steps (`layer`,
   `kl`, `ce`) with very different budgets and learning rates; the weights say which one does the work.

All numbers: Qwen3.5-0.8B backbone, one seed, the standard recipe in
[recipe.md](recipe.md#distillation-linswap-distill).

## Definitions

### `S(W)` — within-head channel spread

How much of a gate is channel-specific rather than one value per head.  Reshape the gate projection
`W ∈ R^{H·C × D}` (heads × channels × hidden) and measure each head's deviation from its own channel
mean `m_h = (1/C) Σ_c W[h, c, :]`:

```
S(W) = mean over heads and layers of   ‖W[h] − 1 m_h^T‖_F / ‖W[h]‖_F        ∈ [0, 1]
```

A tiled init makes every channel of a head a copy of `m_h`, so `S = 0` exactly.  `S = 0.30` means 30 %
of the gate's norm expresses something the backbone's per-head scalar could not.  Low-rank gates
(`Linear(D→r)` then `Linear(r→H·C)`) are measured on the effective matrix `W₂ W₁`.

`CV` is the same idea on real text instead of weights: `std_c g / |mean_c g|` over the channels of a
head, averaged over tokens, heads and layers, with `g` the gate value the kernel computes.

### `D(a, b)` — relative weight drift

How far the swapped layers moved between two checkpoints, over `.linear_attn.` parameters only (the
`layer` step trains those alone, so an unrestricted drift would not compare like with like):

```
D(a, b) = sqrt( Σ_p ‖W_p^a − W_p^b‖_F² ) / sqrt( Σ_p ‖W_p^a‖_F² )
```

`D = 0.18` means the weights moved 18 % of their own norm.  It is a magnitude, not a direction: the
same value can mean repair or damage, which is what table C separates.

## A. Do the tiled gates spread?

`S(W)` at init and after the full recipe, with the `layer` step run and with it skipped.  Every row
starts at exactly `0.0000`.  (`gla` is inexact, so it never skips the step — see the skip rule in
[recipe.md](recipe.md#distillation-linswap-distill).)

| kernel | gate | init | final, `layer` step run | final, `layer` step skipped |
|---|---|---|---|---|
| `gdn2` | `w_proj` (write) | 0.0000 | 0.3009 | 0.0106 |
| `gdn2` | `b_proj` (erase) | 0.0000 | 0.2621 | 0.0078 |
| `gdn2` | `f_proj` (decay) | 0.0000 | 0.2009 | 0.0054 |
| `rwkv7` | `b_proj` (per-channel lr) | 0.0000 | 0.1652 | 0.0077 |
| `kda` | `f_proj` (rank-128 decay) | 0.0000 | 0.1121 | 0.0054 |
| `rwkv7` | `f_proj` (decay) | 0.0000 | 0.0931 | 0.0054 |
| `gla` | `gk_proj` (decay) | 0.0000 | 0.0858 | — |

`gdn`, `mamba2`, `swa` and `deltanet` are absent because the quantity does not exist for them, not
because it was not measured: none has a per-channel gate tiled from a backbone scalar (`gdn` adds no
gate, `mamba2` keeps a per-head scalar decay, `swa` adds 16 scalars, `deltanet` has no decay branch).

GDN-2's gate values on real text, the functional check on the same kernel:

| | mean value | CV across channels |
|---|---|---|
| retention `α`, init → final | 0.848 → 0.826 | 0.0000 → 0.0640 |
| erase `b`, init → final | 0.439 → 0.573 | 0.0000 → 0.2810 |
| write `w`, init → final | 0.439 → 0.566 | 0.0000 → 0.1686 |

`b` and `w` are identical at init (both tiled from the same pretrained `beta` row) and end
`mean|b − w| / mean|b| = 0.2266` apart, so the erase/write decoupling GDN-2 exists for is used.

## B. Which step moves the swapped layers?

| kernel | `D`(init, 6103) — `layer` | `D`(6103, final) — `kl`+`ce` | ratio |
|---|---|---|---|
| `gdn` (control, exact copy) | 0.0000 | 0.0071 | — |
| `gla` | 0.0716 | 0.0016 | 45× |
| `rwkv7` | 0.1820 | 0.0038 | 48× |
| `kda` | 0.1850 | 0.0039 | 48× |
| `gdn2` | 0.2034 | 0.0047 | 43× |
| `swa` | 0.5328 | 0.0118 | 45× |
| `mamba2` | 0.5814 | 0.0113 | 51× |
| `deltanet` | 0.6056 | 0.0131 | 46× |

The `layer` step is 100M tokens at lr 1e-3; `kl` + `ce` are 600M tokens at 1e-5.  Six times the data
moves the swapped layers 43–51× less, for every kernel.  The ordering of the first column tracks how
much of GDN's function the init preserves: exact copy 0.00, exact-init kernels 0.18–0.20, inexact ones
0.53–0.61.

## C. What the `layer` step does to the model

Validation loss (10 held-out packed sequences of 8192; teacher = 3.0898) and the `layer` objective
itself, at its first step, 5 % in, and at the end.

| kernel | init | @200 steps | end of `layer` | end of `kl` | `layer` loss: first → 5 % → last |
|---|---|---|---|---|---|
| `gdn` (exact copy) | 3.0898 | 3.0898 | 3.0898 | 3.0902 | 0 → 0 → 0 |
| `gdn2` (exact init) | 3.0895 | **3.2476** | 3.1105 | 3.1005 | 7.3e-07 → 1.2e-03 → 1.7e-04 |
| `rwkv7` (exact init) | 3.0894 | **3.2108** | 3.1046 | 3.0960 | 8.2e-07 → 1.0e-03 → 1.5e-04 |
| `gla` (inexact) | 7.4508 | 3.3890 | 3.1832 | 3.1425 | 1.6e-02 → 1.9e-03 → 6.8e-04 |
| `mamba2` (inexact) | 7.5915 | 3.3877 | 3.1821 | 3.1510 | 1.8e-02 → 1.8e-03 → 5.7e-04 |
| `swa` (inexact) | 14.0398 | 3.5160 | 3.3241 | 3.3219 | 1.1e-01 → 2.1e-03 → 9.7e-04 |

`kda` and `deltanet` are absent: only their weights were transferred to this machine,
without the `train_log.jsonl` the trajectory needs.

## Reading

* **The `layer` step sets the structure; `kl` and `ce` refine around it.**  43–51× more weight movement
  from 1/6 of the tokens.  Whatever the swapped layer becomes, it becomes it in the first hour.
* **For an exact init the step is not free — it is harmful.**  `gdn2` and `rwkv7` begin at a `layer`
  loss of ~1e-6, i.e. nothing to learn, and the step still moves them 18–20 % of weight norm: the
  objective *rises* three orders of magnitude and validation degrades by ~0.16 nats before partly
  recovering.  Neither is back to its step-0 value by the end of `kl`.  The mechanism is Adam: its
  update is normalised by gradient magnitude, so a 1e-6 gradient still produces a step of order the
  learning rate, and at 1e-3 over 6103 steps the weights random-walk away from a correct solution.
  Only the `gdn` control escapes, because its gradient is identically zero (it *is* the teacher).
  Skipping it is now the default (`--layer_skip_below`), and it is worth 9-11 points on the 128K
  distractor needle: `gdn2` 86.8 → 96.4, `kda` 85.6 → 96.4, `rwkv7` 86.6 → 96.2, against the
  control's 96.8.  Short context moves 108.7–109.9 → 110.2–110.3.
* **For an inexact init the same step is the whole repair.**  `swa` 14.04 → 3.32, `mamba2` 7.59 → 3.18,
  `gla` 7.45 → 3.18, with the objective falling monotonically.  The useful predictor of which case
  applies is the loss at step 1, not `exact_init`: `gla`'s mapped decay init is formally inexact but
  starts at 1.6e-2, two orders below where `swa` starts.
* **The gates barely use the extra freedom — the earlier spread was damage.**  With the `layer` step
  skipped, every tiled gate ends at `S = 0.005–0.011` instead of 0.09–0.30: 20–40× less, and those are
  the checkpoints that score *better* (128K distractor needle 96.2–96.4 vs 85.6–86.8).  Almost all of
  the spread in the middle column was produced by the step that walks an exact init away from the
  teacher, not by the model finding a use for per-channel gates.  An earlier reading of this table —
  "distillation uses the capacity GDN-2 and KDA add" — was wrong: at this scale and budget the useful
  solution keeps the pretrained per-head gate nearly unchanged.  The residual ordering survives
  (`gdn2`'s write gate still spreads furthest, 0.0106 vs 0.0054 for its decay) at a scale that explains
  nothing.
* **But spread is not a goal in itself.**  It is meaningful only relative to a structured init: a gate
  initialised at random has a large spread and no structure, and scores far worse than the same kernel
  initialised by tiling.  Read `S` as "how far from the tiled starting point", never as "how expressive".
* **Settled.**  The question this note left open — how much of table A's spread is learned structure
  and how much is drift from a harmful step — is answered by the right-hand column: almost all of it was
  drift.  The `layer` step now skips itself for an init that starts at bf16 noise, and the four
  exact-init kernels land on the control on both suites.

## Reproducing

```python
import torch
from linswap.load_weights import build_model

def spread(sd, name, H=16):                      # S(W), averaged over heads and layers
    out = []
    for key in (k for k in sd if k.endswith(f"linear_attn.{name}.weight")):
        W = sd[key].float()
        W = W.reshape(H, W.shape[0] // H, -1)
        out.append((torch.norm(W - W.mean(1, keepdim=True), dim=(1, 2)) / torch.norm(W, dim=(1, 2))).mean())
    return torch.stack(out).mean().item()

def drift(a, b):                                 # D(a, b) over the swapped layers
    num = den = 0.0
    for k in a:
        if ".linear_attn." in k and a[k].dtype.is_floating_point:
            num += (a[k].float() - b[k].float()).pow(2).sum().item()
            den += a[k].float().pow(2).sum().item()
    return (num ** 0.5) / (den ** 0.5)

init = build_model("gdn2", device="cpu").state_dict()          # the tiled init, S == 0
mid  = torch.load("outputs/gdn2/distill/checkpoint-6103/model.pt", map_location="cpu", weights_only=True)
fin  = torch.load("outputs/gdn2/distill/checkpoint-16338/model.pt", map_location="cpu", weights_only=True)
print(spread(init, "w_proj"), spread(fin, "w_proj"), drift(init, mid), drift(mid, fin))
```

A low-rank gate is stored as `.0.weight` / `.1.weight`; use the effective matrix `W₂ W₁` in `spread`.
`build_model` on CPU costs ~22 s per kernel, so cache the init state dict if you sweep several.
