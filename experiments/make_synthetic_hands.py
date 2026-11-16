#!/usr/bin/env python3
"""
make_synthetic_hands.py — tiny synthetic data set to smoke-test the baseline scripts
(no real data needed). Draws a hand-like mask (palm + 4 fingers + thumb) fingers-up
in the ALIGNED frame, rotates it by phi (OpenCV, +phi = aligned -> raw, the same
convention as rx_common.py) and writes: images/, masks/, labels_oriented/seed42/*.json.
Left and right hands are mirror images. Names follow CASIA: 001_l_850_01.png.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from rx_common import apply_affine, rot_matrix, save_json

W, H = 768, 576


def hand_aligned(rng, right: bool):
    m = np.zeros((H, W), np.uint8)
    cx, cy = W // 2, H // 2 + 60
    pw, ph = rng.randint(190, 230), rng.randint(200, 240)
    cv2.rectangle(m, (cx - pw // 2, cy - ph // 2), (cx + pw // 2, cy + ph // 2), 255, -1)
    cv2.ellipse(m, (cx, cy + ph // 2), (pw // 2, 30), 0, 0, 180, 255, -1)          # wrist rounding
    fw, gap = 36, 14
    lens = [rng.randint(110, 130), rng.randint(150, 175), rng.randint(140, 160), rng.randint(100, 120)]
    x0 = cx - pw // 2 + 8
    for i in range(4):
        xl = x0 + i * (fw + gap)
        top = cy - ph // 2 - lens[i]
        cv2.rectangle(m, (xl, top + fw // 2), (xl + fw, cy - ph // 2 + 5), 255, -1)
        cv2.circle(m, (xl + fw // 2, top + fw // 2), fw // 2, 255, -1)
    # thumb: slanted capsule on the left side
    base = (cx - pw // 2 + 5, cy - 10)
    tip = (base[0] - 95, base[1] - 85)
    cv2.line(m, base, tip, 255, 40)
    cv2.circle(m, tip, 20, 255, -1)
    if right:
        m = m[:, ::-1].copy()
    gt_c = (cx, cy - 15)
    gt_s = float(min(pw, ph) * 0.85)
    if right:
        gt_c = (W - 1 - gt_c[0], gt_c[1])
    return m, gt_c, gt_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    labs = {"train": {}, "val": {}, "test": {}}
    for i in range(a.n):
        right = i % 2 == 1
        phi = rng.uniform(40, 135)
        m, gc, gs = hand_aligned(rng, right)
        M = rot_matrix(+phi, W, H)
        mr = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST)
        (rx, ry), = apply_affine(M, [gc])
        name = f"{i // 6 + 1:03d}_{'r' if right else 'l'}_850_{i % 6 + 1:02d}"
        img = (mr.astype(np.float32) * 0.55 + np.random.RandomState(i).randn(H, W) * 6 + 20).clip(0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), 2)
        cv2.imwrite(str(out / "images" / f"{name}.png"), img)
        cv2.imwrite(str(out / "masks" / f"{name}.png"), mr)
        split = "train" if i < 0.6 * a.n else ("val" if i < 0.7 * a.n else "test")
        labs[split][name + ".png"] = {"cx": rx / W, "cy": ry / H, "size": gs / min(H, W), "theta": phi}
    for s, d in labs.items():
        save_json(d, out / "labels_oriented" / "seed42" / f"{s}_labels.json")
    print({s: len(d) for s, d in labs.items()})


if __name__ == "__main__":
    main()
