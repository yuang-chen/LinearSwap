"""Plot a RULER summary.csv (columns ``task@length``): one subplot per task, score vs. length, one line per model.

    python tools/plot_ruler_summary.py outputs/eval/ruler_mk_agg_nolayer_1e-5/summary.csv
    python tools/plot_ruler_summary.py <csv> --merge_multikey   # one "multikey" panel = mean of _1/_2/_3
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

MULTIKEY = ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3"]
TITLES = {"niah_multikey_1": "multikey 1 (essay)", "niah_multikey_2": "multikey 2 (needle haystack)",
          "niah_multikey_3": "multikey 3 (UUID)", "multikey": "multikey (mean of 1/2/3)",
          "vt": "vt (variable tracking)", "cwe": "cwe (common words)", "fwe": "fwe (frequent words)"}


def load(path):
    """-> {model: {task: {length: score}}}, skipping empty cells and avg@ columns."""
    data = {}
    for row in csv.DictReader(open(path)):
        scores = data.setdefault(row["model"], {})
        for col, val in row.items():
            m = re.fullmatch(r"(.+)@(\d+)", col)
            if m and m.group(1) != "avg" and val not in ("", None):
                scores.setdefault(m.group(1), {})[int(m.group(2))] = float(val)
    return data


def merge_multikey(data):
    for scores in data.values():
        lengths = set.intersection(*(set(scores.get(t, {})) for t in MULTIKEY))
        scores["multikey"] = {L: sum(scores[t][L] for t in MULTIKEY) / 3 for L in lengths}
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None, help="default: <csv dir>/ruler_plot.png")
    ap.add_argument("--merge_multikey", action="store_true")
    args = ap.parse_args()

    data = load(args.csv)
    tasks = MULTIKEY + ["vt", "cwe", "fwe"]
    if args.merge_multikey:
        data = merge_multikey(data)
        tasks = ["multikey", "vt", "cwe", "fwe"]
    lengths = sorted({L for s in data.values() for t in tasks for L in s.get(t, {})})

    ncols = 3 if len(tasks) > 4 else 2
    nrows = -(-len(tasks) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), sharey=True, squeeze=False)
    for ax, task in zip(axes.flat, tasks):
        for model, scores in data.items():
            pts = sorted(scores.get(task, {}).items())
            if pts:
                ax.plot([L for L, _ in pts], [s for _, s in pts], marker="o", label=model)
        ax.set_title(TITLES.get(task, task))
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths, [f"{L // 1024}K" for L in lengths])
        ax.minorticks_off()
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.3)
        ax.set_xlabel("context length")
    for ax in axes[:, 0]:
        ax.set_ylabel("score")
    for ax in list(axes.flat)[len(tasks):]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(args.out) if args.out else Path(args.csv).with_name("ruler_plot.png")
    fig.savefig(out, dpi=150)
    print(f"written {out}")


if __name__ == "__main__":
    main()
