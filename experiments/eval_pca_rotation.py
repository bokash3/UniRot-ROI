#!/usr/bin/env python3
"""
eval_pca_rotation.py — classical PCA baseline for Stage II (rotation)
====================================================================

Replaces Hand-Alignment/run_pca_baseline.py, whose MAE of ~91 deg came from
comparing the raw eigenvector direction with the GT *correction* angle
(different conventions) and from not resolving the 180 deg ambiguity.

Here the PCA major axis of the predicted hand mask is turned into a
correction angle in the same convention as angles.json
    (aligned = warpAffine(raw, getRotationMatrix2D(center, -theta, 1)) has the
     fingers pointing up),
and the 180 deg ambiguity is resolved in two ways (plus a calibrated variant):
  * none        : keep the candidate in [0, 180)            (shows the ambiguity)
  * finger_runs : after rotation, the end of the hand whose rows contain more
                  separate foreground runs (spread fingers) is placed at the
                  top. (In CASIA-MS the wrist does not touch the image border,
                  so a border rule cannot be used.)
  * finger_runs+offset: as finger_runs, minus a left/right-hand offset that is
                  estimated on the TRAIN split (the PCA axis is pulled towards
                  the thumb, which biases left and right hands in opposite
                  directions). Uses the hand side from the CASIA file name.

Metrics on each seed's Stage II test split (same as Table 2): MAE, median,
p90, p95, Acc@exact-bin and Acc@+-1 bin with 5-degree bins (C = 72).

Usage:
  python eval_pca_rotation.py ^
     --masks_dir  ../data/casia/stage1_masks_resnet34 ^
     --angles     ../data/casia/angles.json ^
     --split_json ../data/casia/subject_split.json ^
     --out_csv    out\\pca_rotation_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from rx_common import circ_err, index_dir, load_angle_map, load_json, stem


def major_axis_dir(mask_b: np.ndarray):
    ys, xs = np.nonzero(mask_b)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    w, v = np.linalg.eigh(cov)
    return c, v[:, np.argmax(w)], pts


def correction_angle_for_up(d: np.ndarray) -> float:
    """Angle theta such that rotating direction d with getRotationMatrix2D(.., -theta)
    makes it point up, i.e. to (0, -1) in image coordinates (y down)."""
    # cv2 linear part for angle a: [[cos a, sin a], [-sin a, cos a]]; we use a = -theta.
    # Solve numerically on a fine grid then refine analytically.
    best, best_t = -2.0, 0.0
    for t in np.arange(0.0, 360.0, 0.25):
        a = math.radians(-t)
        x = math.cos(a) * d[0] + math.sin(a) * d[1]
        y = -math.sin(a) * d[0] + math.cos(a) * d[1]
        score = -y  # want y = -1
        if score > best:
            best, best_t = score, t
    return best_t % 360.0


def rotate_pts(pts, c, theta):
    a = math.radians(-theta)
    R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    return (pts - c) @ R.T


def _mean_runs(rows: np.ndarray) -> float:
    """Mean number of foreground runs per row (fingers -> several runs per row)."""
    if len(rows) == 0:
        return 0.0
    d = np.diff(rows.astype(np.int8), axis=1)
    return float(((d == 1).sum(1) + rows[:, 0]).mean())


def pca_angle(mask: np.ndarray, method: str) -> float:
    b = mask > 127
    if b.sum() < 50:
        return 0.0
    c, d, pts = major_axis_dir(b)
    t1 = correction_angle_for_up(d)
    t2 = (t1 + 180.0) % 360.0
    if method == "none":
        return t1 if t1 < 180 else t2
    # finger_runs: rotate with t1, then the end of the hand whose rows contain
    # more separate foreground runs (spread fingers) must be at the top.
    H, W = b.shape
    r = cv2.warpAffine(b.astype(np.uint8), cv2.getRotationMatrix2D((W / 2, H / 2), -t1, 1.0), (W, H),
                       flags=cv2.INTER_NEAREST) > 0
    ys = np.nonzero(r.any(1))[0]
    if len(ys) < 10:
        return t1
    y0, y1 = ys.min(), ys.max()
    q = max(1, int(0.25 * (y1 - y0)))
    top = _mean_runs(r[y0:y0 + q]); bot = _mean_runs(r[y1 - q:y1 + 1])
    return t1 if top >= bot else t2


def stats(errs, preds, gts, C=72):
    e = np.asarray(errs)
    bw = 360.0 / C
    pb = (np.floor(np.asarray(preds) / bw) % C).astype(int)
    gb = (np.floor(np.asarray(gts) / bw) % C).astype(int)
    d = np.minimum((pb - gb) % C, (gb - pb) % C)
    return {"MAE": e.mean(), "Median": np.median(e), "p90": np.percentile(e, 90),
            "p95": np.percentile(e, 95), "Acc@exact-bin": (d == 0).mean(), "Acc@pm1bin": (d <= 1).mean(),
            "n_flip_gt90": int((e > 90).sum()), "n": len(e)}


def hand_side(f: str) -> str:
    t = f.split("_")
    return t[1].lower() if len(t) > 1 else "?"


def signed(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--masks_dir", required=True)
    p.add_argument("--angles", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--seeds", default="42,2025,1337")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--cache", default=None, help="JSON cache of per-mask PCA angles (re-used across runs)")
    p.add_argument("--max_new", type=int, default=0, help="compute at most N new masks then stop (chunked runs)")
    a = p.parse_args()

    idx = index_dir(Path(a.masks_dir))
    gt = load_angle_map(a.angles)
    split = load_json(a.split_json)
    cache = load_json(a.cache) if a.cache and Path(a.cache).is_file() else {}
    seeds = a.seeds.split(",")
    need = sorted({stem(f) for sd in seeds for part in ("train", "test") for f in split[sd][part]}
                  & set(idx) & set(gt))
    new = 0
    for f in need:
        if f in cache:
            continue
        m = cv2.imread(str(idx[f]), cv2.IMREAD_GRAYSCALE)
        cache[f] = {"none": pca_angle(m, "none"), "finger_runs": pca_angle(m, "finger_runs")}
        new += 1
        if a.max_new and new >= a.max_new:
            break
    if a.cache:
        Path(a.cache).parent.mkdir(parents=True, exist_ok=True)
        with open(a.cache, "w") as fh:
            json.dump(cache, fh)
    missing = [f for f in need if f not in cache]
    if missing:
        print(f"[PARTIAL] {len(missing)} masks still to compute; re-run with the same --cache")
        return

    rows, per = [], []
    for seed in seeds:
        test = [stem(f) for f in split[seed]["test"] if stem(f) in cache]
        train = [stem(f) for f in split[seed]["train"] if stem(f) in cache]
        # hand-side offset learned on the TRAIN split (PCA axis is biased by the thumb)
        off = {}
        for side in ("l", "r"):
            e = [signed(cache[f]["finger_runs"], gt[f]) for f in train if hand_side(f) == side]
            off[side] = float(np.mean(e)) if e else 0.0
        for method in ("none", "finger_runs", "finger_runs+offset"):
            preds = []
            for f in test:
                if method == "finger_runs+offset":
                    pr = (cache[f]["finger_runs"] - off.get(hand_side(f), 0.0)) % 360.0
                else:
                    pr = cache[f][method]
                preds.append(pr)
            errs = [circ_err(pr, gt[f]) for pr, f in zip(preds, test)]
            per += [(seed, method, f, gt[f], pr, e) for f, pr, e in zip(test, preds, errs)]
            st = stats(errs, preds, [gt[f] for f in test])
            rows.append({"seed": seed, "method": method, **st})
            print(f"seed {seed} {method:18s} " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                                         for k, v in st.items()))
        print(f"   train-split offsets: left={off['l']:+.2f}  right={off['r']:+.2f} deg")
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(Path(a.out_csv).with_name(Path(a.out_csv).stem + "_per_sample.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["seed", "method", "file", "gt", "pred", "err"]); w.writerows(per)
    for method in ("none", "finger_runs", "finger_runs+offset"):
        r = [x for x in rows if x["method"] == method]
        f = lambda k: (np.mean([x[k] for x in r]), np.std([x[k] for x in r], ddof=1))
        print(f"MEAN±STD {method:18s} MAE={f('MAE')[0]:.3f}±{f('MAE')[1]:.3f}  Median={f('Median')[0]:.3f}"
              f"  p90={f('p90')[0]:.3f}  p95={f('p95')[0]:.3f}  Acc@exact={f('Acc@exact-bin')[0]:.3f}"
              f"  Acc@pm1={f('Acc@pm1bin')[0]:.3f}")


if __name__ == "__main__":
    main()
