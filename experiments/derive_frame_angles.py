#!/usr/bin/env python3
"""
derive_frame_angles.py
======================
Recover, for every image, the exact angle phi that was used to create
palm_roi_benchmark/aligned_masks from the raw Stage-I masks:

    aligned = warpAffine(raw_mask, getRotationMatrix2D(center, -phi, 1))

Why this is needed
------------------
The Stage III ROI labels were drawn on aligned_masks, so their coordinate
frame is defined by phi. None of the saved angle files (angles.json,
predicted_angles_seed*.json) reproduces aligned_masks exactly (mask IoU
0.89-0.93), whereas the angle recovered here reproduces it with IoU ~0.99.
Any raw-frame target for a baseline must therefore use phi, not another
angle source.

Method: coarse search on 4x down-sampled masks, then a 0.1 deg refinement
at full resolution, maximizing mask IoU.

Usage (Windows, from experiments):
  python derive_frame_angles.py ^
      --raw_masks     ../data/casia/stage1_masks ^
      --aligned_masks ../data/casia/aligned_masks ^
      --init_angles   ../data/casia/predicted_angles_seed42.json ^
      --out           out\\frame_angles.json
"""
from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from rx_common import index_dir, load_angle_map, save_json

ARGS = None


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _rot(mask, ang):
    H, W = mask.shape
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), -ang, 1.0)
    return cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST) > 127


def solve_one(item):
    st, raw_p, al_p, init = item
    raw = cv2.imread(str(raw_p), cv2.IMREAD_GRAYSCALE)
    al = cv2.imread(str(al_p), cv2.IMREAD_GRAYSCALE)
    if raw is None or al is None or raw.shape != al.shape:
        return st, None, 0.0, "read_or_shape_error"
    al_b = al > 127
    raw_s = cv2.resize(raw, (raw.shape[1] // 4, raw.shape[0] // 4), interpolation=cv2.INTER_NEAREST)
    al_s = cv2.resize(al, (al.shape[1] // 4, al.shape[0] // 4), interpolation=cv2.INTER_NEAREST) > 127

    if init is None:
        grid = np.arange(0.0, 360.0, 2.0)
    else:
        grid = np.arange(init - ARGS.search_range, init + ARGS.search_range + 1e-6, 1.0)
    best_a, best_i = None, -1.0
    for a in grid:
        i = _iou(_rot(raw_s, a), al_s)
        if i > best_i:
            best_a, best_i = a, i
    # second coarse pass over the full circle if the local search looks wrong
    if init is not None and best_i < 0.90:
        for a in np.arange(0.0, 360.0, 2.0):
            i = _iou(_rot(raw_s, a), al_s)
            if i > best_i:
                best_a, best_i = a, i
    # refinement at full resolution
    for step, span in ((0.5, 3.0), (0.1, 0.6)):
        a0 = best_a
        for a in np.arange(a0 - span, a0 + span + 1e-6, step):
            i = _iou(_rot(raw, a), al_b)
            if i > best_i:
                best_a, best_i = a, i
    return st, float(best_a % 360.0), float(best_i), "ok"


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--raw_masks", required=True)
    p.add_argument("--aligned_masks", required=True)
    p.add_argument("--init_angles", default=None, help="optional angle JSON to centre the search")
    p.add_argument("--search_range", type=float, default=30.0)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--start", type=int, default=0, help="skip the first N images (chunked runs)")
    p.add_argument("--limit", type=int, default=0, help="process only N images (testing / chunks)")
    ARGS = p.parse_args()

    raw_idx = index_dir(Path(ARGS.raw_masks))
    al_idx = index_dir(Path(ARGS.aligned_masks))
    init = load_angle_map(ARGS.init_angles) if ARGS.init_angles else {}
    stems = sorted(set(raw_idx) & set(al_idx))
    stems = stems[ARGS.start:]
    if ARGS.limit:
        stems = stems[:ARGS.limit]
    print(f"[INFO] {len(stems)} image pairs (raw={len(raw_idx)}, aligned={len(al_idx)})")
    items = [(s, raw_idx[s], al_idx[s], init.get(s)) for s in stems]

    out, rows = {}, []
    with Pool(ARGS.workers, initializer=_init, initargs=(ARGS,)) as pool:
        for k, (s, a, i, status) in enumerate(pool.imap_unordered(solve_one, items, chunksize=8), 1):
            if a is not None:
                out[s + ".png"] = {"angle": round(a, 2), "iou": round(i, 4)}
            rows.append((s, a, i, status))
            if k % 500 == 0:
                print(f"  {k}/{len(items)}")
    save_json(out, ARGS.out)
    ious = np.array([r[2] for r in rows if r[1] is not None])
    with open(Path(ARGS.out).with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "angle", "iou", "status"]); w.writerows(sorted(rows))
    print(f"[DONE] {len(out)} angles -> {ARGS.out}")
    print(f"  IoU mean={ious.mean():.4f} min={ious.min():.4f}  n(IoU<0.97)={(ious < 0.97).sum()}")
    if (ious < 0.97).sum():
        print("  Check the low-IoU files in the CSV before using them.")


def _init(args):
    global ARGS
    ARGS = args


if __name__ == "__main__":
    main()
