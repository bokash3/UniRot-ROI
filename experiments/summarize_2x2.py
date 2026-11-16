#!/usr/bin/env python3
"""summarize_2x2.py — mean ± std over seeds for results_2x2.csv (sample std, ddof=1)."""
import argparse
import csv
from collections import defaultdict

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--csv", default="runs/results_2x2.csv")
p.add_argument("--out", default="runs/summary_2x2.csv")
a = p.parse_args()

rows = list(csv.DictReader(open(a.csv)))
keys = [k for k in rows[0] if k not in ("cell", "seed")]
by = defaultdict(list)
for r in rows:
    by[r["cell"]].append(r)
out = []
for cell in sorted(by):
    rs = by[cell]
    line = {"cell": cell, "n_seeds": len(rs), "seeds": " ".join(r["seed"] for r in rs)}
    for k in keys:
        v = np.array([float(r[k]) for r in rs])
        line[k] = f"{np.nanmean(v):.4f} ± {np.nanstd(v, ddof=1) if len(v) > 1 else 0:.4f}"
    out.append(line)
    print(cell, " | ".join(f"{k}={line[k]}" for k in keys))
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("->", a.out)
