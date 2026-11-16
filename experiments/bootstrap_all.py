#!/usr/bin/env python3
"""
bootstrap_all.py — paired bootstrap confidence intervals for every difference between ROI sources
=================================================================================================
Only numpy is needed. Every comparison is PAIRED: the same resampled identities / images are used for all
sources in one replicate, so the interval is for the DIFFERENCE, not for two independent means.
Seeds: the statistic of a source is the mean over its seeds (seed variance is reported separately, as before).

  match       cross-dataset matching (xd_match.py score files, out/xmatch/<ds>/<source>/scores_*.npz)
              unit resampled = IDENTITY (all probes of a drawn identity enter with it; gallery columns fixed)
              metrics: EER (histogram threshold sweep, 2000 bins) and Rank-1
  geom_cross  geometric accuracy on the hand-labelled VERA / SCUT images (xd_eval.py per_image_*.csv)
              unit = image (one image per person in VERA, one per hand in SCUT); metrics: IoU, angle error, failures
  geom_casia  geometric accuracy on the CASIA-MS test splits (per_image.csv of every method)
              unit = TEST SUBJECT, resampled within each seed (seeds have different test subjects), seeds averaged

Output: out/bootstrap/<part>_<dataset>.csv  with  A, B, metric, value_A, value_B, diff (A - B), 95% CI, p (two-sided)
        For EER and angle error a NEGATIVE difference means A is better; for IoU and Rank-1 a POSITIVE one.

  python bootstrap_all.py match
  python bootstrap_all.py geom_cross
  python bootstrap_all.py geom_casia
"""
from __future__ import annotations

import argparse
import csv
import itertools
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = (42, 2025, 1337)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")


def summarize_pairs(stats: dict, point: dict, metric: str, better: str, rows: list, tag: dict):
    """stats[src] = (B,) bootstrap values, point[src] = point estimate."""
    for a_, b_ in itertools.combinations(list(stats), 2):
        d = stats[a_] - stats[b_]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        diff = point[a_] - point[b_]
        win = "" if lo <= 0 <= hi else (a_ if (diff < 0) == (better == "low") else b_)
        rows.append({**tag, "metric": metric, "A": a_, "B": b_, "value_A": round(point[a_], 4),
                     "value_B": round(point[b_], 4), "diff_A_minus_B": round(diff, 4), "ci_low": round(lo, 4),
                     "ci_high": round(hi, 4), "p_two_sided": round(p, 4), "significantly_better": win})
        print(f"    {metric:9s} {a_:>10s} vs {b_:<10s} {point[a_]:8.3f} vs {point[b_]:8.3f}  diff={diff:+8.3f}  "
              f"95% CI [{lo:+8.3f}, {hi:+8.3f}]  p={p:.4f}  {('-> ' + win) if win else '(n.s.)'}")


# ============================================================================================== matching
def probe_identities(names):
    pid, probe = [], []
    for n in names:
        t = str(n).split("_")
        if t[0] == "vera":                         # vera_001_L_2_3
            pid.append(f"{t[1]}_{t[2]}")
            probe.append(t[3] == "2")
        else:                                      # scut_12_7
            pid.append(t[1])
            probe.append(int(t[2]) > 5)
    pid, probe = np.array(pid), np.array(probe)
    return pid[probe]


def load_scores(p):
    z = np.load(p)
    g = z["genuine"].astype(np.float64)
    return g, z["impostor"].astype(np.float64).reshape(len(g), -1)


def eer_from_hist(Gh, Ih):
    """rows = replicates; distance accepted if <= threshold (bin upper edge)."""
    far = np.cumsum(Ih, 1) / Ih.sum(1, keepdims=True)
    frr = 1.0 - np.cumsum(Gh, 1) / Gh.sum(1, keepdims=True)
    k = np.argmin(np.abs(far - frr), 1)
    r = np.arange(len(k))
    return 50.0 * (far[r, k] + frr[r, k])


def part_match(a):
    root = Path(a.xmatch_dir)
    rng = np.random.default_rng(a.seed)
    for ds in a.datasets.split(","):
        d = root / ds
        if not d.is_dir():
            continue
        srcdirs = sorted(p for p in d.iterdir() if p.is_dir() and (p / "crops.npz").is_file())
        names = np.load(srcdirs[0] / "crops.npz", allow_pickle=True)["names"]
        pid = probe_identities(names)
        ids, pidx = np.unique(pid, return_inverse=True)
        nI, P = len(ids), len(pid)
        C = np.stack([np.bincount(rng.integers(0, nI, nI), minlength=nI) for _ in range(a.n_boot)]).astype(np.float32)
        W = np.vstack([np.ones((1, P), np.float32), C[:, pidx]])            # row 0 = point estimate
        print(f"\n===== matching | {ds}: {nI} identities, {P} probes, {a.n_boot} replicates =====")
        rows = []
        for m in ("compcode", "resnet18", "miura"):
            ent = defaultdict(list)                                         # source -> [(gen, imp)] over seeds
            for sd in srcdirs:
                mm = re.match(r"(.+)_seed(\d+)$", sd.name)
                src = mm.group(1) if mm else sd.name
                files = sorted(sd.glob(f"scores_{m}_rnseed*.npz")) or sorted(sd.glob(f"scores_{m}.npz"))
                for f in files:
                    g, i = load_scores(f)
                    if len(g) != P:
                        raise ValueError(f"{f}: {len(g)} probes, expected {P}")
                    ent[src].append((g, i))
            if not ent:
                continue
            lo = min(min(g.min(), i.min()) for v in ent.values() for g, i in v)
            hi = max(max(g.max(), i.max()) for v in ent.values() for g, i in v)
            edges = np.linspace(lo, hi, a.bins + 1)
            eer, r1 = {}, {}
            for src, lst in ent.items():
                e_s, r_s = [], []
                for g, i in lst:
                    gb = np.clip(np.searchsorted(edges, g, "left") - 1, 0, a.bins - 1)
                    ib = np.clip(np.searchsorted(edges, i, "left") - 1, 0, a.bins - 1)
                    Gh1 = np.zeros((P, a.bins), np.float32)
                    Gh1[np.arange(P), gb] = 1
                    Ih1 = np.zeros((P * a.bins,), np.float32)
                    np.add.at(Ih1, (np.arange(P)[:, None] * a.bins + ib).ravel(), 1)
                    Ih1 = Ih1.reshape(P, a.bins)
                    e_s.append(eer_from_hist(W @ Gh1, W @ Ih1))
                    correct = (g < i.min(1)).astype(np.float32)
                    r_s.append(100.0 * (W @ correct) / W.sum(1))
                eer[src], r1[src] = np.mean(e_s, 0), np.mean(r_s, 0)
                print(f"  {m:9s} {src:10s} seeds={len(lst)}  EER={eer[src][0]:6.2f}%  "
                      f"95% CI [{np.percentile(eer[src][1:], 2.5):5.2f}, {np.percentile(eer[src][1:], 97.5):5.2f}]  "
                      f"Rank-1={r1[src][0]:6.2f}%")
            tag = {"dataset": ds, "matcher": m}
            summarize_pairs({s: v[1:] for s, v in eer.items()}, {s: v[0] for s, v in eer.items()}, "EER_pct",
                            "low", rows, tag)
            summarize_pairs({s: v[1:] for s, v in r1.items()}, {s: v[0] for s, v in r1.items()}, "Rank1_pct",
                            "high", rows, tag)
        write(Path(a.out_dir) / f"match_{ds}.csv", rows)


# ============================================================================================== geometry
def read_csv(p, key="file"):
    return {r[key]: r for r in csv.DictReader(open(p))}


def boot_geom(data: dict, units: dict, a, tag, rows):
    """data[src][seed] = {file: row}; units[seed] = {unit: [files]} (files common to all sources).
    statistic = mean over seeds of the per-seed mean over units (unit mean over its files)."""
    rng = np.random.default_rng(a.seed)
    seeds = sorted(units)
    draws = {s: [rng.integers(0, len(units[s]), len(units[s])) for _ in range(a.n_boot)] for s in seeds}
    for metric, better in (("iou", "high"), ("angle_err_deg", "low"), ("failed", "low")):
        stats, point = {}, {}
        for src in data:
            if not all(s in data[src] for s in seeds):
                continue
            per_seed_b, per_seed_p = [], []
            for s in seeds:
                ul = list(units[s].values())
                um = np.array([np.mean([float(data[src][s][f][metric]) for f in fs]) for fs in ul])
                per_seed_p.append(um.mean())
                per_seed_b.append(np.array([um[ix].mean() for ix in draws[s]]))
            stats[src], point[src] = np.mean(per_seed_b, 0), float(np.mean(per_seed_p))
        if len(stats) > 1:
            print(f"   [{metric}] " + "  ".join(f"{k}={v:.4f}" for k, v in point.items()))
            summarize_pairs(stats, point, metric, better, rows, tag)


def part_geom_cross(a):
    root = Path(a.cross_dir) / "results"
    for ds in a.datasets.split(","):
        d = root / ds
        if not d.is_dir():
            continue
        data = defaultdict(dict)
        for f in d.glob("per_image_*_seed*.csv"):
            mm = re.match(r"per_image_(.+)_seed(\d+)\.csv", f.name)
            data[mm.group(1)][int(mm.group(2))] = read_csv(f)
        seeds = sorted(set.intersection(*[set(v) for v in data.values()]))
        missing = {k: sorted(set(SEEDS) - set(v)) for k, v in data.items() if set(SEEDS) - set(v)}
        if missing:
            print(f"\n  [NOTE] {ds}: seeds missing {missing} -> only seeds {seeds} are compared (run xd_eval.py)")
        common = {s: sorted(set.intersection(*[set(data[k][s]) for k in data])) for s in seeds}
        units = {s: {f: [f] for f in common[s]} for s in seeds}
        print(f"\n===== geometry | {ds}: sources {list(data)}  seeds {seeds}  images {len(common[seeds[0]])} =====")
        rows = []
        boot_geom({k: {s: v[s] for s in seeds} for k, v in data.items()}, units, a,
                  {"dataset": ds, "seeds": "+".join(map(str, seeds))}, rows)
        if rows:
            write(Path(a.out_dir) / f"geom_cross_{ds}.csv", rows)


def part_geom_casia(a):
    pats = {"pipeline": "out/full_pipeline/seed{s}/per_image.csv",
            "yolo": "runs_yolo/pose_seed{s}/eval/per_image.csv",
            "yolo_aug": "runs_yolo/pose_aug_seed{s}/eval/per_image.csv",
            "valley": "out/baselines/valley/seed{s}/per_image_valley_offset.csv",
            "pipeline_aug": "out/full_pipeline_aug/seed{s}/per_image.csv",
            "yolo_l100": "runs_yolo/pose_l100_seed{s}/eval/per_image.csv",
            "yolo_pre": "runs_yolo/pose_pre_seed{s}/eval/per_image.csv",
            "yolo_s_pre": "runs_yolo/pose_s_pre_seed{s}/eval/per_image.csv",     # YOLOv8s-pose-PT (used only when complete)
            "yolo_m_pre": "runs_yolo/pose_m_pre_seed{s}/eval/per_image.csv",     # YOLOv8m-pose-PT
            "rawA_aug": "out/raw_aug/cellA_aug/casia_seed{s}.csv", "rawB_aug": "out/raw_aug/cellB_aug/casia_seed{s}.csv",
            "rawA_orig": "out/raw_aug/cellA_orig/casia_seed{s}.csv", "rawB_orig": "out/raw_aug/cellB_orig/casia_seed{s}.csv"}
    data = defaultdict(dict)
    for k, p in pats.items():
        for s in SEEDS:
            f = Path(p.format(s=s))
            if f.is_file():
                data[k][s] = read_csv(f)
    data = {k: v for k, v in data.items() if all(s in v for s in SEEDS)}     # optional sources only when complete
    seeds = sorted(set.intersection(*[set(v) for v in data.values()]))
    missing = {k: sorted(set(SEEDS) - set(v)) for k, v in data.items() if set(SEEDS) - set(v)}
    if missing:
        print(f"  [NOTE] seeds missing {missing} -> comparing seeds {seeds} only")
    units = {}
    for s in seeds:
        common = sorted(set.intersection(*[set(data[k][s]) for k in data]))
        g = defaultdict(list)
        for f in common:
            g[f.split("_")[0]].append(f)                       # unit = test subject
        units[s] = dict(g)
    print(f"\n===== geometry | CASIA test: sources {list(data)}  seeds {seeds}  "
          f"subjects/seed {[len(units[s]) for s in seeds]} =====")
    rows = []
    boot_geom({k: {s: v[s] for s in seeds} for k, v in data.items()}, units, a,
              {"dataset": "casia", "seeds": "+".join(map(str, seeds))}, rows)
    write(Path(a.out_dir) / "geom_casia.csv", rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("part", choices=["match", "geom_cross", "geom_casia", "all"])
    p.add_argument("--xmatch_dir", default="out/xmatch")
    p.add_argument("--cross_dir", default="out/cross")
    p.add_argument("--datasets", default="vera,scut")
    p.add_argument("--out_dir", default="out/bootstrap")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--bins", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.part in ("match", "all"):
        part_match(a)
    if a.part in ("geom_cross", "all"):
        part_geom_cross(a)
    if a.part in ("geom_casia", "all"):
        part_geom_casia(a)


if __name__ == "__main__":
    main()
