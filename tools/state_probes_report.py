#!/usr/bin/env python
"""Turn the state_probes.py dumps into truncation.csv, heads.csv, P_spectrum.npy, the plots and a numbers file
(outputs/state_probes/summary.json) for the write-up.  numpy / matplotlib / scipy only."""
import csv, glob, json, math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from state_probes import OUT, R_KEEP, R_DROP, SHORTLIST, T, erank, numrank

BUCKET_ORDER = ["all", "0-1K", "1K-4K", "4K-8K", "8K-16K", "retrievable"]


# ---------------------------------------------------------------------------------------------- probe 1
def load_p1():
    rows = []
    for f in sorted(glob.glob(str(OUT / "p1_rows_w*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            scope, layer, head, variant, rr = r["job"]
            rows.append(dict(scope=scope, layer=layer, head=head, variant=variant, r=rr, pos_bucket=r["bucket"],
                             n_tokens=r["n_tokens"], nll=r["nll"], nll_baseline=r["nll_baseline"], delta=r["delta"],
                             delta_std=r["delta_std"], delta_per_seq=r["delta_per_seq"]))
    return rows


def write_truncation_csv(rows):
    cols = ["scope", "layer", "head", "variant", "r", "pos_bucket", "n_tokens", "nll", "nll_baseline", "delta", "delta_std"]
    key = lambda r: (["global", "layer", "head"].index(r["scope"]), r["layer"], r["head"], r["variant"], -r["r"],
                     BUCKET_ORDER.index(r["pos_bucket"]))
    with open(OUT / "truncation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=key))


def get(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def plot_global(rows):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for b in BUCKET_ORDER:
        pts = sorted(get(rows, scope="global", pos_bucket=b), key=lambda r: r["r"])
        if not pts:
            continue
        x = [p["r"] for p in pts]; y = [p["delta"] for p in pts]; e = [p["delta_std"] for p in pts]
        style = dict(lw=2.4, color="k") if b == "all" else (dict(ls="--") if b == "retrievable" else {})
        ax.errorbar(x, y, yerr=e, marker="o", ms=3, capsize=2, label="overall" if b == "all" else b, **style)
    ax.set_xscale("log", base=2); ax.set_xticks(R_KEEP); ax.set_xticklabels(R_KEEP)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.axhline(0.01, color="grey", lw=0.8, ls=":"); ax.axhline(0.005, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("r (singular directions kept, every head of every layer)"); ax.set_ylabel("Δ NLL (nats)")
    ax.set_title("Global keep_top(r): Δ NLL vs r (mean ± std over sequences)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "plot_global.png", dpi=150); plt.close(fig)


def plot_per_layer(rows, layers):
    M = np.full((len(layers), len(R_KEEP)), np.nan)
    for i, L in enumerate(layers):
        for j, r in enumerate(R_KEEP):
            p = get(rows, scope="layer", layer=L, r=r, pos_bucket="all")
            if p:
                M[i, j] = p[0]["delta"]
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = max(np.nanmax(np.abs(M)), 1e-3)
    im = ax.imshow(M, cmap="magma_r", aspect="auto", norm=matplotlib.colors.SymLogNorm(1e-3, vmin=0, vmax=vmax))
    ax.set_xticks(range(len(R_KEEP))); ax.set_xticklabels(R_KEEP); ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(M[i, j]) > vmax / 8 else "black")
    ax.set_xlabel("r (keep_top, all heads of one layer)"); ax.set_ylabel("layer")
    ax.set_title("Per-layer keep_top(r): Δ NLL overall (nats)")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout(); fig.savefig(OUT / "plot_per_layer.png", dpi=150); plt.close(fig)
    return M


def plot_per_head(rows):
    heads = list(SHORTLIST)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharey=False)
    for ax, (L, h) in zip(axes.flat, heads):
        for variant, rs, c in (("keep_top", R_KEEP, "C0"), ("drop_top", R_DROP, "C3")):
            pts = sorted(get(rows, scope="head", layer=L, head=h, variant=variant, pos_bucket="all"), key=lambda r: r["r"])
            if pts:
                ax.errorbar([p["r"] for p in pts], [p["delta"] for p in pts], yerr=[p["delta_std"] for p in pts],
                            marker="o", ms=3, capsize=2, color=c, label=variant)
        ax.set_xscale("log", base=2); ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"L{L} h{h} ({SHORTLIST[(L, h)]})", fontsize=9); ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("r")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ NLL overall")
    fig.suptitle("Per-head truncation: keep_top(r) keeps the top r directions, drop_top(r) removes them")
    fig.tight_layout(); fig.savefig(OUT / "plot_per_head.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------------------------- probe 2
def probe2():
    z = np.load(OUT / "p2_raw.npz")
    layers = sorted({int(k[1:]) for k in z.files if k.startswith("g")})
    H = z[f"g{layers[0]}"].shape[2]
    rows, spectra = [], np.zeros((len(layers), H, 128))
    stats = lambda v: (float(np.median(v)), float(np.min(v)), float(np.max(v)))
    for li, L in enumerate(layers):
        g, beta, lastk = z[f"g{L}"], z[f"beta{L}"], z[f"lastk{L}"]                  # [B,T,H], [B,T,H], [B,H,4096,K]
        for h in range(H):
            r = {"layer": L, "head": h}
            gh, bh = g[:, :, h].astype(np.float64), beta[:, :, h].astype(np.float64)
            al = np.exp(gh)
            r["mean_log_alpha"] = gh.mean()
            seq_mla = gh.mean(1)
            r["mean_log_alpha_min"], r["mean_log_alpha_max"] = seq_mla.min(), seq_mla.max()
            r["horizon"] = -1 / r["mean_log_alpha"]
            seq_hor = -1 / seq_mla
            r["horizon_min"], r["horizon_max"] = seq_hor.min(), seq_hor.max()
            r["alpha_p10"], r["alpha_p90"] = np.percentile(al, 10), np.percentile(al, 90)
            p10s, p90s = np.percentile(al, 10, axis=1), np.percentile(al, 90, axis=1)
            r["alpha_p10_min"], r["alpha_p10_max"], r["alpha_p90_min"], r["alpha_p90_max"] = p10s.min(), p10s.max(), p90s.min(), p90s.max()
            r["mean_beta"] = bh.mean(); r["mean_beta_min"], r["mean_beta_max"] = bh.mean(1).min(), bh.mean(1).max()
            wr = (bh > 0.5).mean(1)
            r["write_rate"] = (bh > 0.5).mean(); r["write_rate_min"], r["write_rate_max"] = wr.min(), wr.max()
            for m, tag in ((1024, "1k"), (4096, "4k"), (T, "16k")):
                sS = z[f"S{L}@{m}"][:, h]; eP = z[f"P{L}@{m}"][:, h]
                for name, v in ((f"erank_S_{tag}", erank(sS)), (f"erank_P_{tag}", erank(eP))):
                    r[name], r[name + "_min"], r[name + "_max"] = stats(v)
            v = numrank(z[f"S{L}@{T}"][:, h]); r["numrank_S_16k"], r["numrank_S_16k_min"], r["numrank_S_16k_max"] = stats(v)
            v = numrank(np.clip(z[f"P{L}@{T}"][:, h], 0, None)); r["numrank_P_16k"], r["numrank_P_16k_min"], r["numrank_P_16k_max"] = stats(v)
            n = int(np.clip(round(r["horizon"]), 2, 4096)) if np.isfinite(r["horizon"]) and r["horizon"] > 0 else 4096
            k = lastk[:, h, -n:].astype(np.float64)                                   # [B, n, K]
            ev = np.linalg.eigvalsh(np.einsum("bnk,bnj->bkj", k, k))[:, ::-1]
            r["key_window_n"] = n
            r["key_erank_window"], r["key_erank_window_min"], r["key_erank_window_max"] = stats(erank(ev))
            for kk in (8, 32):
                v = z[f"ov{L}@{kk}"][:, h]
                r[f"overlap_{kk}"], r[f"overlap_{kk}_min"], r[f"overlap_{kk}_max"] = stats(v)
            eP = np.clip(z[f"P{L}@{T}"][:, h], 0, None)
            spectra[li, h] = np.median(eP / eP[:, :1], axis=0)
            rows.append(r)
    np.save(OUT / "P_spectrum.npy", spectra)
    main = ["layer", "head", "mean_log_alpha", "horizon", "alpha_p10", "alpha_p90", "mean_beta", "write_rate",
            "erank_S_1k", "erank_S_4k", "erank_S_16k", "numrank_S_16k", "erank_P_1k", "erank_P_4k", "erank_P_16k",
            "key_erank_window", "overlap_8", "overlap_32"]
    extra = ["numrank_P_16k", "key_window_n"]
    cols = main + extra + [f"{c}_{s}" for c in main[2:] + ["numrank_P_16k"] for s in ("min", "max")]
    with open(OUT / "heads.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    return rows, spectra, layers


def plots2(rows, spectra, layers):
    hor = np.array([min(r["horizon"], T) if r["horizon"] > 0 else T for r in rows])
    eS = np.array([r["erank_S_16k"] for r in rows]); eP = np.array([r["erank_P_16k"] for r in rows])
    lay = np.array([r["layer"] for r in rows])
    cmap = plt.get_cmap("viridis")
    col = [cmap(layers.index(l) / (len(layers) - 1)) for l in lay]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(hor, eS, c=[layers.index(l) for l in lay], cmap="viridis", s=14)
    ax.plot([1, T], [1, T], "k--", lw=0.8, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.8, 160); ax.set_xlim(0.8, T * 1.3)
    ax.set_xlabel("horizon = −1 / mean log α (tokens, clipped at 16384)"); ax.set_ylabel("erank S at 16K")
    cb = fig.colorbar(sc, ax=ax, ticks=range(0, len(layers), 3)); cb.ax.set_yticklabels(layers[::3]); cb.set_label("layer")
    ax.legend(); ax.set_title("Decay horizon vs effective rank of the state"); fig.tight_layout()
    fig.savefig(OUT / "plot_horizon_vs_erank.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(eP, eS, c=col, s=14)
    for r in rows:
        if (r["layer"], r["head"]) in SHORTLIST:
            ax.annotate(f"L{r['layer']}h{r['head']}", (r["erank_P_16k"], r["erank_S_16k"]), fontsize=6)
    m = max(eP.max(), eS.max()) * 1.1
    ax.plot([0.8, m], [0.8, m], "k--", lw=0.8, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("erank P at 16K (decayed key covariance, no delta term)"); ax.set_ylabel("erank S at 16K")
    ax.legend(); ax.set_title("Key-covariance rank vs state rank"); fig.tight_layout()
    fig.savefig(OUT / "plot_erankP_vs_erankS.png", dpi=150); plt.close(fig)

    order = np.argsort(eS)
    fig, ax = plt.subplots(figsize=(11, 4))
    for tag, c in (("1k", "C2"), ("4k", "C1"), ("16k", "C0")):
        ax.plot(np.arange(len(rows)), [rows[i][f"erank_S_{tag}"] for i in order], ".", ms=3, color=c, label=f"{tag}")
    ax.set_yscale("log"); ax.set_xlabel("head (sorted by erank S at 16K)"); ax.set_ylabel("erank S")
    ax.legend(title="prefix"); ax.set_title("Saturation: effective rank of each head's state at 1K / 4K / 16K tokens")
    fig.tight_layout(); fig.savefig(OUT / "plot_saturation.png", dpi=150); plt.close(fig)

    n = len(layers); cols = 6; rws = math.ceil(n / cols)
    fig, axes = plt.subplots(rws, cols, figsize=(3 * cols, 2.5 * rws), sharex=True, sharey=True)
    for ax, (li, L) in zip(axes.flat, enumerate(layers)):
        for h in range(spectra.shape[1]):
            ax.semilogy(np.arange(1, 129), np.clip(spectra[li, h], 1e-16, None), lw=0.8)
        ax.axvline(32, color="grey", lw=0.5, ls=":"); ax.axhline(1e-2, color="grey", lw=0.5, ls=":")
        ax.set_title(f"layer {L}", fontsize=9); ax.set_ylim(1e-12, 2)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Eigenvalues of P at 16K (λ_i / λ_1, median over sequences), one line per head")
    fig.tight_layout(); fig.savefig(OUT / "plot_P_spectrum.png", dpi=130); plt.close(fig)


def classify_spectra(spectra):
    """low-rank + floor: lambda_32 / lambda_1 <= 1e-2; power law: otherwise, and log lambda linear in log i (R^2 >= 0.9)."""
    flat = spectra.reshape(-1, 128)
    lowrank = flat[:, 31] <= 1e-2
    x = np.log(np.arange(1, 129))
    powerlaw = np.zeros(len(flat), bool)
    r2s = np.zeros(len(flat))
    for i, s in enumerate(flat):
        y = np.log(np.clip(s, 1e-16, None))
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss = ((y - y.mean()) ** 2).sum()
        r2s[i] = 1 - (res[0] / ss if len(res) and ss > 0 else 0)
        powerlaw[i] = (not lowrank[i]) and r2s[i] >= 0.9
    return lowrank.reshape(spectra.shape[:2]), powerlaw.reshape(spectra.shape[:2]), r2s.reshape(spectra.shape[:2])


# ---------------------------------------------------------------------------------------------- probe 1r
def load_p1r():
    """{tag: rows}; tag '' is the unmodified backbone (p1r_rows_w*.jsonl), 'control' etc. are --tag runs."""
    runs = {}
    for f in sorted(glob.glob(str(OUT / "p1r*_rows_w*.jsonl"))):
        tag = Path(f).name[len("p1r"):].split("_rows_w")[0].lstrip("_")
        runs.setdefault(tag, []).extend(json.loads(l) for l in open(f))
    return runs


def probe1r(rows, tag=""):
    """retrieval.csv (task, L, r, n, score, score_baseline, delta, delta_se) and plot_retrieval.png.  Scores are
    RULER string_match_all x 100; the baseline is r = 128 on the same samples; delta_se is the standard error of the
    paired per-sample difference."""
    by = {}
    for r in rows:   # RULER's needle files repeat a few index values, so key samples by index + reference answers
        by.setdefault((r["task"], r["L"], r["r"]), {})[(r["index"], tuple(r.get("refs", [])))] = r["score"]
    out = []
    for (task, L, rr), d in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[0][2])):
        base = by.get((task, L, 128), {})
        common = sorted(set(d) & set(base))
        diff = np.array([d[i] - base[i] for i in common]) * 100 if common else np.array([])
        out.append(dict(task=task, L=L, r=rr, n=len(d), score=round(100 * np.mean(list(d.values())), 2),
                        n_paired=len(common), score_baseline=round(100 * np.mean([base[i] for i in common]), 2) if common else float("nan"),
                        delta=round(float(diff.mean()), 2) if len(diff) else float("nan"),
                        delta_se=round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 2) if len(diff) > 1 else float("nan")))
    suffix = f"_{tag}" if tag else ""
    with open(OUT / f"retrieval{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    tasks = sorted({o["task"] for o in out})
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.8), sharey=True)
    for ax, task in zip(np.atleast_1d(axes), tasks):
        for L, c in ((16384, "C0"), (65536, "C1"), (131072, "C3")):
            pts = sorted([o for o in out if o["task"] == task and o["L"] == L], key=lambda o: o["r"])
            if pts:
                ax.errorbar([o["r"] for o in pts], [o["score"] for o in pts],
                            yerr=[0 if np.isnan(o["delta_se"]) else o["delta_se"] for o in pts],
                            marker="o", ms=4, capsize=2, color=c, label=f"{L // 1024}K (n={pts[0]['n']})")
        ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 12, 32, 128]); ax.set_xticklabels([1, 4, 12, 32, 128])
        ax.set_title(task); ax.set_xlabel("r kept per head (every layer)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("RULER score")
    fig.suptitle(f"{tag or 'unmodified backbone'}: keep_top(r) on the carried state at every {rows[0].get('seg', 256)}-token "
                 f"boundary, retrieval accuracy vs r")
    fig.tight_layout(); fig.savefig(OUT / f"plot_retrieval{suffix}.png", dpi=150); plt.close(fig)
    return out


def main():
    summary = {}
    for tag, rows1r in load_p1r().items():
        summary["retrieval" + (f"_{tag}" if tag else "")] = probe1r(rows1r, tag)
    rows1 = load_p1()
    if rows1:
        write_truncation_csv(rows1)
        plot_global(rows1)
        layers = sorted({r["layer"] for r in rows1 if r["scope"] == "layer"})
        M = plot_per_layer(rows1, layers)
        plot_per_head(rows1)
        g = {(r["r"], r["pos_bucket"]): (r["delta"], r["delta_std"]) for r in rows1 if r["scope"] == "global"}
        summary["global"] = {f"{b}@{rr}": g[(rr, b)] for (rr, b) in g}
        summary["per_layer_all"] = {f"L{L}": dict(zip(map(str, R_KEEP), M[i].tolist())) for i, L in enumerate(layers)}
        summary["per_head"] = {f"L{r['layer']}h{r['head']} {r['variant']}({r['r']})": (r["delta"], r["delta_std"])
                               for r in rows1 if r["scope"] == "head" and r["pos_bucket"] == "all"}
        summary["n_rows"] = len(rows1)
    if (OUT / "p2_raw.npz").exists():
        rows2, spectra, layers2 = probe2()
        plots2(rows2, spectra, layers2)
        hor = np.array([r["horizon"] for r in rows2]); eS = np.array([r["erank_S_16k"] for r in rows2])
        eP = np.array([r["erank_P_16k"] for r in rows2]); e4 = np.array([r["erank_S_4k"] for r in rows2])
        summary["spearman_horizon_erankS16k"] = spearmanr(np.where(hor > 0, hor, np.inf), eS).correlation
        summary["spearman_erankP16k_erankS16k"] = spearmanr(eP, eS).correlation
        low, pw, _ = classify_spectra(spectra)
        summary["frac_P_lowrank_plus_floor"] = float(low.mean()); summary["frac_P_powerlaw"] = float(pw.mean())
        summary["frac_P_other"] = float(1 - low.mean() - pw.mean())
        short = [r for r in rows2 if (r["layer"], r["head"]) in SHORTLIST]
        summary["shortlist"] = {f"L{r['layer']}h{r['head']}": {k: r[k] for k in (
            "horizon", "mean_beta", "write_rate", "erank_S_1k", "erank_S_4k", "erank_S_16k", "numrank_S_16k",
            "erank_P_16k", "key_erank_window", "overlap_8", "overlap_32")} for r in short}
        retr = [r for r in rows2 if SHORTLIST.get((r["layer"], r["head"])) == "retrieval"]
        summary["retrieval_median_overlap_32"] = float(np.median([r["overlap_32"] for r in retr]))
        summary["retrieval_median_erankS_over_erankP"] = float(np.median([r["erank_S_16k"] / r["erank_P_16k"] for r in retr]))
        summary["median_erankS_over_erankP_all"] = float(np.median(eS / eP))
        rel = (eS - e4) / np.maximum(e4, 1e-9)
        summary["frac_heads_saturated_4k_16k_within_10pct"] = float((np.abs(rel) <= 0.10).mean())
        summary["climbing_heads"] = [f"L{r['layer']}h{r['head']} ({r['erank_S_4k']:.1f}->{r['erank_S_16k']:.1f})"
                                     for r in rows2 if r["erank_S_16k"] > 1.2 * r["erank_S_4k"] and r["erank_S_16k"] - r["erank_S_4k"] >= 2]
        ema = [r for r in rows2 if r["horizon"] > 4096 and r["erank_S_16k"] < 5]
        summary["ema_heads"] = [f"L{r['layer']}h{r['head']} (horizon {r['horizon']:.0f}, erank {r['erank_S_16k']:.1f})" for r in ema]
        summary["n_heads"] = len(rows2)
        summary["heads_horizon_gt_4k"] = int((hor > 4096).sum())
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    for key in [k for k in summary if k.startswith("retrieval") and isinstance(summary[k], list)]:
        print(f"== {key}\n{'task':<16}{'L':>7}{'r':>5}{'n':>5}{'score':>8}{'base':>8}{'delta':>8}{'se':>6}")
        for o in summary[key]:
            print(f"{o['task']:<16}{o['L']:>7}{o['r']:>5}{o['n']:>5}{o['score']:>8.1f}{o['score_baseline']:>8.1f}{o['delta']:>+8.1f}{o['delta_se']:>6.1f}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("global", "per_layer_all", "per_head", "shortlist")
                      and not (k.startswith("retrieval") and isinstance(v, list))}, indent=1, default=float))


if __name__ == "__main__":
    main()
