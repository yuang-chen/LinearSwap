#!/usr/bin/env python
"""Ranks of the memory states dumped by `tools/state_rank.py`: per-head CSV, heatmap figure, markdown tables.

    python tools/state_rank_report.py gdn gdn2 kda            # the first model is the reference for the Δ panels

Numerical rank counts singular values above n·eps_fp32·σ_max (the fp32 SVD floor, n = 128); the
significant rank counts those above 1 % of σ_max; the effective rank is exp(entropy of σ / Σσ)
(Roy & Vetterli, 2007).  Each head is summarised by its median over the batch.
"""
import argparse, csv, json
from pathlib import Path

import numpy as np

TOL = 128 * 2.0 ** -23


def ranks(sv):
    sv = np.asarray(sv)
    rel = sv / sv[..., :1]
    return (rel > TOL).sum(-1), (rel > 1e-2).sum(-1)                   # [B, H] each


def erank(sv):
    """Entropy effective rank exp(H(p)), p = σ / Σσ; also the share of the energy Σσ² in σ₁."""
    sv = np.asarray(sv)
    p = sv / sv.sum(-1, keepdims=True)
    return np.exp(-(p * np.log(np.where(p > 0, p, 1))).sum(-1)), sv[..., 0] ** 2 / (sv ** 2).sum(-1)


p = argparse.ArgumentParser()
p.add_argument("models", nargs="+")
p.add_argument("--dir", default="outputs/state_rank")
p.add_argument("--tokens", type=int, default=16384)
p.add_argument("--fig", default="docs/figures/state_rank.png")
a = p.parse_args()

D = {n: json.load(open(Path(a.dir) / f"{n}.json")) for n in a.models}
layers = D[a.models[0]]["layers"]
num, sig, eff, top = {}, {}, {}, {}                                      # [layer, B, H] at --tokens
with open(Path(a.dir) / "per_head_rank.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["model", "tokens", "layer", "head", "num_rank_median", "num_rank_min", "num_rank_max",
                "full_rank_seqs", "sig_rank_median", "eff_rank_median", "sigma1_energy_median"])
    for n, d in D.items():
        for m in d["marks"]:
            per = [ranks(d["sv"][f"{i}@{m}"]) for i in layers]
            ers = [erank(d["sv"][f"{i}@{m}"]) for i in layers]
            for i, (r, r1), (e, t) in zip(layers, per, ers):
                for h in range(r.shape[1]):
                    w.writerow([n, m, i, h, int(np.median(r[:, h])), r[:, h].min(), r[:, h].max(),
                                int((r[:, h] == 128).sum()), int(np.median(r1[:, h])),
                                round(float(np.median(e[:, h])), 2), round(float(np.median(t[:, h])), 3)])
            if m == a.tokens:
                num[n] = np.stack([r for r, _ in per]); sig[n] = np.stack([r1 for _, r1 in per])
                eff[n] = np.stack([e for e, _ in ers]); top[n] = np.stack([t for _, t in ers])

# ---- markdown: per-layer summary, rank vs length, per-head grid
print(f"\n#### Per layer at {a.tokens} tokens: full-rank states / median numerical rank / median significant rank\n")
print("| layer | " + " | ".join(a.models) + " |\n|---|" + "---|" * len(a.models))
for k, i in enumerate(layers):
    print(f"| {i} | " + " | ".join(f"{int((num[n][k] == 128).sum())} / {int(np.median(num[n][k]))} / "
                                   f"{int(np.median(sig[n][k]))}" for n in a.models) + " |")
print("\n#### Full-rank states (of layers × heads × batch) against prefix length\n")
marks = D[a.models[0]]["marks"]
print("| model | " + " | ".join(str(m) for m in marks) + " |\n|---|" + "---|" * len(marks))
for n, d in D.items():
    print(f"| {n} | " + " | ".join(str(sum(int((ranks(d['sv'][f'{i}@{m}'])[0] == 128).sum()) for i in layers))
                                   for m in marks) + " |")
print(f"\n#### Per layer at {a.tokens} tokens: median effective rank / median σ₁ share of the energy\n")
print("| layer | " + " | ".join(a.models) + " |\n|---|" + "---|" * len(a.models))
for k, i in enumerate(layers):
    print(f"| {i} | " + " | ".join(f"{np.median(eff[n][k]):.1f} / {np.median(top[n][k]):.2f}" for n in a.models) + " |")
print("| all | " + " | ".join(f"**{np.median(eff[n]):.1f}** / **{np.median(top[n]):.2f}**" for n in a.models) + " |")
print("\n#### Median effective rank (all layers × heads × batch) against prefix length\n")
print("| model | " + " | ".join(str(m) for m in marks) + " |\n|---|" + "---|" * len(marks))
for n, d in D.items():
    print(f"| {n} | " + " | ".join(f"{np.median(np.stack([erank(d['sv'][f'{i}@{m}'])[0] for i in layers])):.1f}"
                                   for m in marks) + " |")
print(f"\n#### Per-head median effective rank at {a.tokens} tokens ({' / '.join(a.models)})\n")
H = num[a.models[0]].shape[2]
print("| L | " + " | ".join(f"h{h}" for h in range(H)) + " |\n|---|" + "---|" * H)
for k, i in enumerate(layers):
    print(f"| {i} | " + " | ".join("/".join(f"{np.median(eff[n][k, :, h]):.0f}" for n in a.models) for h in range(H)) + " |")
print(f"\n#### Per-head median numerical rank at {a.tokens} tokens ({' / '.join(a.models)})\n")
H = num[a.models[0]].shape[2]
print("| L | " + " | ".join(f"h{h}" for h in range(H)) + " |\n|---|" + "---|" * H)
for k, i in enumerate(layers):
    cells = []
    for h in range(H):
        v = [int(np.median(num[n][k, :, h])) for n in a.models]
        cells.append(("**" + "/".join(map(str, v)) + "**") if max(v) <= 20 else "/".join(map(str, v)))
    print(f"| {i} | " + " | ".join(cells) + " |")
ref = np.median(num[a.models[0]], 1).ravel()
for n in a.models[1:]:
    x = np.median(num[n], 1).ravel()
    print(f"\n{n}: corr with {a.models[0]} per-head rank {np.corrcoef(ref, x)[0, 1]:.3f}, mean change {np.mean(x - ref):+.1f}")
    e0, e1 = np.median(eff[a.models[0]], 1).ravel(), np.median(eff[n], 1).ravel()
    print(f"{n}: corr with {a.models[0]} per-head effective rank {np.corrcoef(e0, e1)[0, 1]:.3f}, "
          f"ratio of medians {np.median(e1) / np.median(e0):.2f}, heads lower {np.mean(e1 < e0):.0%}")

# ---- figure: rows = numerical / significant rank; columns = models, then Δ against the reference
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK, MUTED, SURFACE = "#262624", "#6b6a66", "#fcfcfb"
seq = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
div = LinearSegmentedColormap.from_list("div", ["#b3302f", "#e34948", "#f0efec", "#3987e5", "#1c5cab"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "none"})
others = a.models[1:]
nm, nd = len(a.models), len(others)
widths = [1] * nm + [0.07, 0.35] + [1] * nd + [0.07]                   # model panels | colorbar | gap | Δ panels | colorbar
rows = [("numerical rank", num, 128), ("significant rank (σ > 1% σ_max)", sig, 80), ("effective rank", eff, 40)]
fig = plt.figure(figsize=(2.05 * (nm + nd) + 1.6, 3.7 * len(rows)), facecolor=SURFACE)
gs = fig.add_gridspec(len(rows), len(widths), width_ratios=widths, wspace=0.12, hspace=0.22)
cols = list(range(nm)) + list(range(nm + 2, nm + 2 + nd))
ax = np.array([[fig.add_subplot(gs[r, c]) for c in cols] for r in range(len(rows))])
ncol = nm + nd
for r, (label, R, vmax) in enumerate(rows):
    med = {n: np.median(R[n], 1) for n in a.models}                    # [layer, head]
    for c, n in enumerate(a.models):
        im = ax[r, c].imshow(med[n], cmap=seq, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
        ax[r, c].set_title(n if r == 0 else "", fontsize=11, color=INK, pad=6)
    for c, n in enumerate(others, nm):
        dm = med[n] - med[a.models[0]]
        dim = ax[r, c].imshow(dm, cmap=div, norm=TwoSlopeNorm(0, -vmax * 0.75, vmax * 0.25), aspect="auto",
                              interpolation="nearest")
        ax[r, c].set_title(f"{n} − {a.models[0]}" if r == 0 else "", fontsize=11, color=INK, pad=6)
    for c in range(ncol):
        A = ax[r, c]
        A.set_facecolor(SURFACE)
        A.set_xticks(range(0, H, 3)); A.set_xticks(np.arange(-.5, H), minor=True)
        A.set_yticks(range(len(layers))); A.set_yticklabels(layers if c in (0, nm) else [])
        A.set_yticks(np.arange(-.5, len(layers)), minor=True)
        A.grid(which="minor", color=SURFACE, linewidth=0.8); A.tick_params(which="both", length=0)
        if r == len(rows) - 1: A.set_xlabel("head")
    ax[r, 0].set_ylabel(f"{label}\n\nlayer", color=INK)
    for mappable, c, title in [(im, nm, "rank"), (dim, len(widths) - 1, "Δ rank")]:
        cb = fig.colorbar(mappable, cax=fig.add_subplot(gs[r, c]))
        cb.outline.set_visible(False); cb.ax.tick_params(length=0, colors=MUTED)
        cb.ax.set_title(title, fontsize=8, color=MUTED, pad=4)
fig.suptitle(f"Rank of each head's 128×128 memory state after {a.tokens:,} tokens "
             f"(median over {num[a.models[0]].shape[1]} DCLM sequences)", fontsize=12, color=INK, x=0.45, y=0.925)
Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.fig, dpi=160, bbox_inches="tight", facecolor=SURFACE)
print(f"\n[state_rank_report] figure -> {a.fig}")
