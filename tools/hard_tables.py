#!/usr/bin/env python
"""Markdown tables from `linswap evaluate` logs: one table per length plus the average (works on partial runs).

    python tools/hard_tables.py outputs/eval/rwkv7.log outputs/eval/mamba2.log
"""
import ast, re, sys, json, pathlib
TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
         "niah_multikey_3", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]
SHORT = ["s1", "s2", "s3", "mk1", "mk2", "mk3", "mq", "vt", "cwe", "fwe", "qa1", "qa2"]
LAB = lambda L: {4096: "4K", 16384: "16K", 65536: "64K", 131072: "128K", 262144: "256K"}.get(L, f"{L//1024}K")
res, val = {}, {}
for log in [l for l in sys.argv[1:] if pathlib.Path(l).exists()]:
    for line in open(log):
        m = re.match(r"^  ([\w.-]+) L=(\d+): (\{.*\})$", line.rstrip())
        if m:
            res[(m.group(1), int(m.group(2)))] = ast.literal_eval(m.group(3))
        m = re.match(r"^  ([\w.-]+): val CE ([\d.]+)", line)
        if m:
            val[m.group(1)] = float(m.group(2))
models = sorted({k[0] for k in res})
lengths = sorted({k[1] for k in res})
def row(m, L):
    r = res.get((m, L))
    if r is None: return None
    cells = [f"{r[t]:.1f}" if t in r else "–" for t in TASKS if any(t in x for x in res.values())]
    have = [r[t] for t in TASKS if t in r]
    return f"| {m} | " + " | ".join(cells) + f" | {sum(have)/len(have):.1f} |"
for L in lengths:
    cols = [c for t, c in zip(TASKS, SHORT) if any(t in r for r in res.values())]
    print(f"\n#### {LAB(L)} tokens\n\n| model | " + " | ".join(cols) + " | avg |\n|---|" + "---|" * (len(cols) + 1))
    for m in models:
        r = row(m, L)
        if r: print(r)
print("\n#### average over the 8 tasks vs length\n\n| model | " + " | ".join(LAB(L) for L in lengths) + " |\n|---|" + "---|" * len(lengths))
for m in models:
    cells = []
    for L in lengths:
        r = res.get((m, L)); cells.append(f"{sum(r[t] for t in TASKS if t in r)/len([t for t in TASKS if t in r]):.1f}" if r else "–")
    print(f"| {m} | " + " | ".join(cells) + " |")
if val: print("\nval CE:", val)
