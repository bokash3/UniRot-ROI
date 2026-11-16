#!/usr/bin/env python3
"""
xd_valley.py - the calibrated valley-point baseline (valley+offset) on VERA and SCUT-PV, without adaptation
============================================================================================================
Requested by reviewers: the valley-point method was evaluated in-domain only. Here, for every seed:
  1. Stage I (that seed's original-recipe U-Net, as in xd_eval.py) segments the image in the CASIA frame;
  2. finger valleys are found on the mask (baseline_valley_points.find_valleys);
  3. the square is placed with the offsets calibrated on the CASIA-MS TRAIN split of the seed
     (out/baselines/valley/seed<S>/calibration.json, written by baseline_valley_points.py) - nothing is re-calibrated;
  4. if no valleys are found, the train-median square is used and the image is flagged ('failed' = 1, as in-domain).
Needs a GPU/torch environment (Stage I), like xd_eval.py.

Run from experiments:
  python xd_valley.py --seeds 42,2025,1337
Output: out\\cross\\results\\<vera|scut>\\per_image_valley_offset_seed<S>.csv ; out\\baselines\\valley\\xd_summary.csv
"""
from __future__ import annotations

import argparse
import csv
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

import full_pipeline_eval as fp
from baseline_valley_points import find_valleys, square_from_valleys
from rx_common import label_to_px, load_json, square_corners
from rx_stages import hand_side, score, write_rows

W0, H0 = 768, 576


def dump_overlays(a, dd, ds, seed, rows, lab, seg, dev):
    out = Path(a.valley_dir) / "xd_stages" / ds / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    par = load_json(Path(a.valley_dir) / f"seed{seed}" / "calibration.json")
    worst = [r["file"] for r in sorted(rows, key=lambda r: r["iou"])[:a.dump_n]]
    first = [r["file"] for r in rows[:a.dump_n]]
    for n in dict.fromkeys(first + worst):
        rgb = cv2.cvtColor(cv2.imread(str(dd / "images" / f"{n}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        m = fp.stage1_mask(seg, rgb, dev)
        v = find_valleys(m)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cl = v["clean"] if v.get("clean") is not None else m
        cnts, _ = cv2.findContours((cl > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, (0, 255, 255), 1)
        for d in v.get("selected", []):
            cv2.circle(img, tuple(int(x) for x in d["pt"]), 4, (0, 140, 255), -1)
        for j, d in enumerate(v.get("chain") or []):
            cv2.circle(img, tuple(int(x) for x in d["pt"]), 7, (255, 0, 255), 2)
        gt = label_to_px(lab[n], W0, H0)
        cv2.polylines(img, [square_corners(*gt).astype(np.int32)], True, (0, 255, 0), 2)
        if v["ok"]:
            pr = square_from_valleys(v, par, hand_side(n), True)
            cv2.polylines(img, [square_corners(*pr).astype(np.int32)], True, (255, 80, 0), 2)
        r = next(x for x in rows if x["file"] == n)
        cv2.putText(img, f"{n[:28]} IoU={r['iou']:.2f} {'FAIL:' + r['reason'] if r['failed'] else ''}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"{n}.png"), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cross_dir", default="out/cross")
    ap.add_argument("--valley_dir", default="out/baselines/valley")
    ap.add_argument("--labels_root", default="out/labels_oriented")
    ap.add_argument("--seeds", default="42,2025,1337")
    ap.add_argument("--pipeline_dir", default=r"../pipeline_models")
    ap.add_argument("--seg_ckpt_pattern", default=r"../checkpoints/basic/stage1_seg_mobilenetv2_seed{seed}.pt")
    ap.add_argument("--rot_ckpt_pattern", default=r"../checkpoints/basic/stage2_unirot_angle_seed{seed}.pt")
    ap.add_argument("--roi_ckpt_pattern", default=r"../checkpoints/basic/stage3_roi_seed{seed}.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump_n", type=int, default=0, help="write overlays (image, mask contour, candidate valleys, chosen "
                    "chain, prediction blue, reference green) for the first N and the worst N images of each dataset")
    a = ap.parse_args()
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    summ = Path(a.valley_dir) / "xd_summary.csv"
    with open(summ, "w", newline="") as f:
        csv.writer(f).writerow(["dataset", "seed", "n", "mIoU", "acc_0_7", "center_err_px", "angle_err_deg",
                                "iou_lt_0_5", "valley_failed"])
    for seed in [int(s) for s in a.seeds.split(",")]:
        par = load_json(Path(a.valley_dir) / f"seed{seed}" / "calibration.json")
        tr = load_json(Path(a.labels_root) / f"seed{seed}" / "train_labels.json")
        allp = [label_to_px(l, W0, H0) for l in tr.values()]
        fb = (float(np.median([p[0] for p in allp])), float(np.median([p[1] for p in allp])),
              float(np.median([p[2] for p in allp])), 0.0)
        ns = Namespace(pipeline_dir=a.pipeline_dir, seg_ckpt_pattern=a.seg_ckpt_pattern,
                       rot_ckpt_pattern=a.rot_ckpt_pattern, roi_ckpt_pattern=a.roi_ckpt_pattern, random_weights=False)
        (seg, _, _), _ = fp.build_models(ns, seed, dev)
        for ds in ("vera", "scut"):
            dd = Path(a.cross_dir) / ds
            lab = load_json(dd / "labels_manual.json")
            man = load_json(dd / "manifest.json")
            names = [n for n, l in lab.items() if not l.get("skip") and n in man]
            rows = []
            for n in names:
                rgb = cv2.cvtColor(cv2.imread(str(dd / "images" / f"{n}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                m = fp.stage1_mask(seg, rgb, dev)
                v = find_valleys(m)
                side = hand_side(n)
                pr = square_from_valleys(v, par, side, True) if v["ok"] else fb
                r = score(pr, label_to_px(lab[n], W0, H0), m)
                r.update(file=n, failed=int(not v["ok"]), reason=v["reason"], side=side)
                rows.append(r)
            if a.dump_n:
                dump_overlays(a, dd, ds, seed, rows, lab, seg, dev)
            write_rows(Path(a.cross_dir) / "results" / ds / f"per_image_valley_offset_seed{seed}.csv", rows)
            iou = np.array([r["iou"] for r in rows])
            ang = np.array([r["angle_err_deg"] for r in rows])
            ce = np.array([r["center_err_px"] for r in rows])
            vf = int(sum(r["failed"] for r in rows))
            print(f"[seed {seed}] {ds:5s} n={len(rows)} mIoU={iou.mean():.4f} Acc@0.7={(iou >= .7).mean():.3f} "
                  f"ctr={ce.mean():.1f}px angle={ang.mean():.1f} IoU<0.5={(iou < .5).sum()} valley-failed={vf}", flush=True)
            with open(summ, "a", newline="") as f:
                csv.writer(f).writerow([ds, seed, len(rows), f"{iou.mean():.4f}", f"{(iou >= .7).mean():.4f}",
                                        f"{ce.mean():.2f}", f"{ang.mean():.2f}", int((iou < .5).sum()), vf])


if __name__ == "__main__":
    main()
