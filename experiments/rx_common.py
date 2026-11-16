#!/usr/bin/env python3
"""
rx_common.py — shared helpers for the reviewer experiments.

Coordinate conventions (identical to the existing project code):
  * Images / masks live on a W x H canvas (CASIA: 768 x 576).
  * ROI labels are JSON dicts {fname: {"cx", "cy", "size"[, "theta"]}} where
      cx   is normalized by W, cy by H, size (square side) by min(H, W).
  * The "frame angle" phi of an image is the angle such that
        aligned = cv2.warpAffine(raw, cv2.getRotationMatrix2D((W/2, H/2), -phi, 1), (W, H))
    i.e. the aligned masks in palm_roi_benchmark/aligned_masks were produced
    by rotating the raw Stage-I mask by -phi about the canvas center.
  * Hence a point p_aligned maps back to the raw frame with +phi:
        p_raw = cv2.getRotationMatrix2D(center, +phi, 1) @ [p_aligned, 1]
    and a square drawn on the aligned mask becomes, in the raw frame, a square
    rotated by +phi (OpenCV angle sign).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------------------
# file helpers
# ----------------------------------------------------------------------------
def stem(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def index_dir(folder: Path) -> Dict[str, Path]:
    """Map file stem -> path for every image file in `folder`."""
    out = {}
    for p in Path(folder).iterdir():
        if p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def load_json(p) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def load_angle_map(p) -> Dict[str, float]:
    """Load {fname: angle} or {fname: {"angle": a, ...}} keyed by file stem."""
    raw = load_json(p)
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            v = v.get("angle", v.get("angle_deg"))
        out[stem(k)] = float(v)
    return out


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def rot_matrix(angle_deg: float, W: int, H: int) -> np.ndarray:
    """OpenCV rotation about the canvas center (2x3)."""
    return cv2.getRotationMatrix2D((W / 2.0, H / 2.0), float(angle_deg), 1.0)


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def square_corners(cx_px: float, cy_px: float, side_px: float, theta_deg: float = 0.0) -> np.ndarray:
    """Corners (4x2) of a square of given center/side, rotated by theta_deg
    with the OpenCV sign convention (same as cv2.getRotationMatrix2D)."""
    h = side_px / 2.0
    base = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64)
    a = math.radians(theta_deg)
    c, s = math.cos(a), math.sin(a)
    # cv2.getRotationMatrix2D linear part for angle a: [[c, s], [-s, c]]
    R = np.array([[c, s], [-s, c]])
    return base @ R.T + np.array([cx_px, cy_px])


def polygon_iou(p1: np.ndarray, p2: np.ndarray) -> float:
    """IoU of two convex polygons (Nx2 float)."""
    p1 = np.asarray(p1, np.float32).reshape(-1, 1, 2)
    p2 = np.asarray(p2, np.float32).reshape(-1, 1, 2)
    a1 = abs(cv2.contourArea(p1))
    a2 = abs(cv2.contourArea(p2))
    inter, _ = cv2.intersectConvexConvex(p1, p2)
    inter = max(0.0, float(inter))
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def outside_fraction(poly: np.ndarray, hand_mask: np.ndarray) -> float:
    """Fraction of the polygon area that lies outside the binary hand mask."""
    H, W = hand_mask.shape[:2]
    canvas = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(canvas, np.round(poly).astype(np.int32), 1)
    area = int(canvas.sum())
    if area == 0:
        return 1.0
    inside = int(((hand_mask > 127) & (canvas > 0)).sum())
    return float(1.0 - inside / area)


def circ_err(a: float, b: float, period: float = 360.0) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


def label_to_px(lbl: dict, W: int, H: int) -> Tuple[float, float, float, float]:
    """(cx_px, cy_px, side_px, theta_deg) from a normalized label."""
    return (float(lbl["cx"]) * W, float(lbl["cy"]) * H,
            float(lbl["size"]) * min(H, W), float(lbl.get("theta", 0.0)))


# ----------------------------------------------------------------------------
# letterbox (mirrors train_roi_net_soft_letterbox_validated.py / baseline A)
# ----------------------------------------------------------------------------
def letterbox_params(h: int, w: int, T: int):
    scale = min(T / float(w), T / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_x = (T - new_w) // 2
    pad_y = (T - new_h) // 2
    return scale, new_w, new_h, pad_x, pad_y


def letterbox(img: np.ndarray, T: int, is_mask: bool) -> np.ndarray:
    h, w = img.shape[:2]
    scale, new_w, new_h, pad_x, pad_y = letterbox_params(h, w, T)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    rs = cv2.resize(img, (new_w, new_h), interpolation=interp)
    if img.ndim == 3:
        canvas = np.zeros((T, T, img.shape[2]), dtype=img.dtype)
    else:
        canvas = np.zeros((T, T), dtype=img.dtype)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = rs
    return canvas


def orig_to_lb(cx_px, cy_px, side_px, h, w, T):
    scale, _, _, pad_x, pad_y = letterbox_params(h, w, T)
    return ((cx_px * scale + pad_x) / T, (cy_px * scale + pad_y) / T, (side_px * scale) / T)


def lb_to_orig(cx_n, cy_n, side_n, h, w, T):
    scale, _, _, pad_x, pad_y = letterbox_params(h, w, T)
    return ((cx_n * T - pad_x) / scale, (cy_n * T - pad_y) / scale, (side_n * T) / scale)
