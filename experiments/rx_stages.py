#!/usr/bin/env python3
"""
rx_stages.py — shared helpers for the baseline scripts (valley points, YOLO).

* identical scoring for every method (rotated-square IoU, centre/size/angle error,
  share of the predicted square outside the hand mask) — same code path as Table 3;
* helpers that WRITE EVERY INTERMEDIATE STAGE to disk (one PNG per stage + a strip
  that puts all stages of one image side by side) so that each processing step can be
  inspected by eye.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

from rx_common import circ_err, outside_fraction, polygon_iou, square_corners

GREEN, RED, BLUE, YELLOW, CYAN, MAGENTA, ORANGE, GRAY = (
    (0, 255, 0), (0, 0, 255), (255, 80, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 140, 255), (140, 140, 140))


def hand_side(fname: str) -> str:
    """'l' / 'r' from names like 001_l_850_01.png or 001_L_1_1.png ('?' if unknown)."""
    t = Path(fname).stem.split("_")
    return t[1].lower() if len(t) > 1 and t[1].lower() in ("l", "r") else "?"


# ------------------------------------------------------------------ scoring
def score(pred, gt, hand_mask=None, mod90=False) -> dict:
    """pred/gt = (cx, cy, side, theta_deg). mod90: compare the angle modulo 90 deg
    (a square has 4-fold symmetry; used for detectors that cannot tell 'up')."""
    pp = square_corners(*pred)
    gp = square_corners(*gt)
    if mod90:
        d = abs(pred[3] - gt[3]) % 90.0
        ang = min(d, 90.0 - d)
    else:
        ang = circ_err(pred[3], gt[3])
    r = {"iou": polygon_iou(pp, gp), "center_err_px": float(np.hypot(pred[0] - gt[0], pred[1] - gt[1])),
         "size_err_px": abs(pred[2] - gt[2]), "angle_err_deg": ang}
    if hand_mask is not None:
        r["outside_frac"] = outside_fraction(pp, hand_mask)
    return r


def summarize(rows: Sequence[dict]) -> dict:
    iou = np.array([r["iou"] for r in rows])
    out = {"n": len(rows), "mIoU": float(iou.mean()), "acc_0_5": float((iou >= .5).mean()),
           "acc_0_7": float((iou >= .7).mean()),
           "center_err_px": float(np.mean([r["center_err_px"] for r in rows])),
           "size_err_px": float(np.mean([r["size_err_px"] for r in rows])),
           "angle_err_deg": float(np.mean([r["angle_err_deg"] for r in rows]))}
    if "outside_frac" in rows[0]:
        out["outside_frac"] = float(np.mean([r["outside_frac"] for r in rows]))
    out["n_failed"] = int(sum(1 for r in rows if r.get("failed")))
    return out


def append_result(csv_path, method, seed, summ: dict):
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = ["n", "mIoU", "acc_0_5", "acc_0_7", "center_err_px", "size_err_px", "outside_frac",
            "angle_err_deg", "n_failed"]
    new = not p.is_file()
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["method", "seed"] + keys)
        w.writerow([method, seed] + [summ.get(k, float("nan")) for k in keys])


def write_rows(path, rows: List[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def pick_for_dump(rows: List[dict], n_random: int, n_worst: int, n_fail: int, seed: int = 0) -> List[str]:
    """Files whose stages are written: random sample + the worst IoUs + failures."""
    rnd = random.Random(seed)
    files = [r["file"] for r in rows]
    pick = [r["file"] for r in sorted(rows, key=lambda r: r["iou"])[:n_worst]]
    pick += [r["file"] for r in rows if r.get("failed")][:n_fail]
    rest = [f for f in files if f not in pick]
    rnd.shuffle(rest)
    pick += rest[:n_random]
    return list(dict.fromkeys(pick))


# ------------------------------------------------------------------ drawing / stage files
def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def draw_poly(img, poly, color, thick=2):
    cv2.polylines(img, [np.round(poly).astype(np.int32)], True, color, thick, cv2.LINE_AA)


def put_text(img, text, org, color=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def titled(img: np.ndarray, title: str, height: int = 280) -> np.ndarray:
    im = to_bgr(img)
    s = height / im.shape[0]
    im = cv2.resize(im, (max(1, int(im.shape[1] * s)), height), interpolation=cv2.INTER_AREA)
    bar = np.zeros((26, im.shape[1], 3), np.uint8)
    cv2.putText(bar, title[:max(10, im.shape[1] // 8)], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([bar, im])


def save_stages(folder: Path, stem: str, panels: List[tuple], info: dict = None):
    """panels = [(file_name_without_ext, title, image)]. Writes one PNG per panel
    and <stem>_strip.png with all panels side by side."""
    import json
    d = Path(folder) / stem
    d.mkdir(parents=True, exist_ok=True)
    for name, title, im in panels:
        cv2.imwrite(str(d / f"{name}.png"), to_bgr(im))
    strip = np.hstack([titled(im, f"{i + 1}. {t}") for i, (_, t, im) in enumerate(panels)])
    cv2.imwrite(str(Path(folder) / f"{stem}_strip.png"), strip)
    if info is not None:
        with open(d / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def oriented_crop(img: np.ndarray, cx, cy, side, theta, out=128) -> np.ndarray:
    """Upright out x out crop of the square (same warp as run_matching_v2.py)."""
    c = square_corners(cx, cy, side, theta).astype(np.float32)        # TL, TR, BR, BL
    dst = np.float32([[0, 0], [out - 1, 0], [out - 1, out - 1]])
    M = cv2.getAffineTransform(c[:3], dst)
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
