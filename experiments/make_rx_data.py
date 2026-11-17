#!/usr/bin/env python3
"""
make_rx_data.py — build the data needed by the experiments in this folder.

Sub-commands
------------
labels   Convert the Stage III ROI labels (drawn on aligned masks) into
         RAW-frame ORIENTED labels {cx, cy, size, theta} using the frame
         angles from derive_frame_angles.py. theta is the square's rotation
         in the raw image (OpenCV sign), size is unchanged (rotation keeps
         the side length). This is the exact same square, expressed in the
         raw frame — no bounding-box inflation.

images   Rotate the raw images with the same frame angles, giving images in
         exactly the frame of aligned_masks (needed for the
         "image + alignment" cell of the 2x2 ablation).

verify   Draw GT squares on raw images (oriented labels) and on frame-aligned
         images (original labels) for a visual check.

Examples (Windows):
  python make_rx_data.py labels --labels_root ../data/casia ^
      --frame_angles out\\frame_angles.json --out_root out\\labels_oriented
  python make_rx_data.py images --images_dir ../data/casia/images ^
      --frame_angles out\\frame_angles.json --out_dir out\\images_frame_aligned
  python make_rx_data.py verify --images_dir ../data/casia/images ^
      --aligned_images out\\images_frame_aligned --labels_root ../data/casia ^
      --oriented_root out\\labels_oriented --seed 42 --out_dir out\\verify --n 12
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from rx_common import (apply_affine, index_dir, label_to_px, load_angle_map, load_json,
                       rot_matrix, save_json, square_corners, stem)

SEEDS = (42, 2025, 1337)
SPLITS = ("train", "val", "test")


def cmd_labels(a):
    phi = load_angle_map(a.frame_angles)
    for seed in SEEDS:
        src = Path(a.labels_root) / f"dataset_palm_locator_seed{seed}"
        dst = Path(a.out_root) / f"seed{seed}"
        missing = 0
        for split in SPLITS:
            lab = load_json(src / f"{split}_labels.json")
            out = {}
            for fname, l in lab.items():
                s = stem(fname)
                if s not in phi:
                    missing += 1
                    continue
                W, H = a.canvas_w, a.canvas_h
                cx, cy, side, _ = label_to_px(l, W, H)
                M = rot_matrix(+phi[s], W, H)          # aligned -> raw
                (rx, ry), = apply_affine(M, [[cx, cy]])
                out[fname] = {"cx": rx / W, "cy": ry / H, "size": float(l["size"]),
                              "theta": float(phi[s])}
            save_json(out, dst / f"{split}_labels.json")
            print(f"seed {seed} {split}: {len(out)} labels")
        shutil.copy(src / "roi_proto_split_info.json", dst / "roi_proto_split_info.json")
        if missing:
            print(f"  [WARN] seed {seed}: {missing} labels without a frame angle were skipped")


def cmd_images(a):
    phi = load_angle_map(a.frame_angles)
    idx = index_dir(Path(a.images_dir))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    n = 0
    for s, p in sorted(idx.items()):
        if s not in phi or (a.only and s not in a.only):
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        rot = cv2.warpAffine(img, rot_matrix(-phi[s], W, H), (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(str(out / (s + ".png")), rot)
        n += 1
        if n % 1000 == 0:
            print(f"  {n}")
    print(f"[DONE] {n} frame-aligned images -> {out}")


def _draw(img, corners, color):
    cv2.polylines(img, [np.round(corners).astype(np.int32)], True, color, 2)


def cmd_verify(a):
    raw_idx = index_dir(Path(a.images_dir))
    al_idx = index_dir(Path(a.aligned_images))
    lab = load_json(Path(a.labels_root) / f"dataset_palm_locator_seed{a.seed}" / "test_labels.json")
    ori = load_json(Path(a.oriented_root) / f"seed{a.seed}" / "test_labels.json")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    keys = sorted(k for k in set(lab) & set(ori) if stem(k) in al_idx); random.seed(0)
    for fname in random.sample(keys, min(a.n, len(keys))):
        s = stem(fname)
        raw = cv2.imread(str(raw_idx[s])); al = cv2.imread(str(al_idx[s]))
        H, W = raw.shape[:2]
        cx, cy, side, th = label_to_px(ori[fname], W, H)
        _draw(raw, square_corners(cx, cy, side, th), (0, 255, 0))
        cx, cy, side, _ = label_to_px(lab[fname], W, H)
        _draw(al, square_corners(cx, cy, side, 0.0), (0, 255, 0))
        cv2.imwrite(str(out / f"{s}_verify.jpg"), np.hstack([raw, al]))
    print(f"[DONE] overlays -> {out}  (left: raw + oriented GT, right: aligned + original GT)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("labels"); q.add_argument("--labels_root", required=True)
    q.add_argument("--frame_angles", required=True); q.add_argument("--out_root", required=True)
    q.add_argument("--canvas_w", type=int, default=768); q.add_argument("--canvas_h", type=int, default=576)
    q = sub.add_parser("images"); q.add_argument("--images_dir", required=True)
    q.add_argument("--frame_angles", required=True); q.add_argument("--out_dir", required=True)
    q.add_argument("--only", nargs="*", default=None, help="restrict to these file stems (testing)")
    q = sub.add_parser("verify"); q.add_argument("--images_dir", required=True)
    q.add_argument("--aligned_images", required=True); q.add_argument("--labels_root", required=True)
    q.add_argument("--oriented_root", required=True); q.add_argument("--seed", type=int, default=42)
    q.add_argument("--out_dir", required=True); q.add_argument("--n", type=int, default=12)
    a = p.parse_args()
    {"labels": cmd_labels, "images": cmd_images, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
