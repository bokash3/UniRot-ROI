#!/usr/bin/env python3
"""
Letterbox utilities for square input pipelines (e.g., 256x256).

This module standardizes:
- image/mask letterboxing to a fixed square size
- forward mapping of ROI labels from original image space -> letterboxed space
- inverse mapping of ROI predictions from letterboxed space -> original image space

All comments are in English as requested.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import cv2
import random
import numpy as np
import torch


@dataclass
class LetterboxMeta:
    """Geometry needed to invert the letterbox transform."""
    orig_h: int
    orig_w: int
    new_size: int
    scale: float
    pad_x: int
    pad_y: int


def letterbox_gray(
    img: np.ndarray,
    new_size: int = 256,
    pad_value: int = 0,
    interpolation: int = cv2.INTER_NEAREST,
) -> Tuple[np.ndarray, LetterboxMeta]:
    """
    Letterbox a single-channel image/mask to (new_size, new_size), preserving aspect ratio.

    Returns:
        out: (new_size, new_size) uint8/float image
        meta: LetterboxMeta for inverse mapping
    """
    if img.ndim != 2:
        raise ValueError(f"letterbox_gray expects HxW, got shape={img.shape}")

    orig_h, orig_w = img.shape[:2]
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid image size: H={orig_h}, W={orig_w}")

    # Uniform scale to fit the longer side into new_size
    scale = float(new_size) / float(max(orig_h, orig_w))
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    # Pad to square
    out = np.full((new_size, new_size), pad_value, dtype=resized.dtype)
    pad_y = (new_size - new_h) // 2
    pad_x = (new_size - new_w) // 2
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    meta = LetterboxMeta(
        orig_h=orig_h,
        orig_w=orig_w,
        new_size=new_size,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return out, meta
# -----------------------------
# Evaluation utilities
# -----------------------------
def boxes_to_masks(cx, cy, size, h, w, device):
    """
    Build binary box masks from normalized (cx, cy, size) for a batch.

    Args:
        cx, cy, size: tensors of shape (B,) in [0,1]
        h, w: height and width of the mask (int)
        device: torch device

    Returns:
        box_masks: tensor of shape (B, 1, h, w) with 1.0 inside the box and 0.0 outside.
    """
    batch_size = cx.shape[0]

    # Convert normalized coordinates to pixel units
    min_side = float(min(h, w))

    side = size * min_side
    half = side / 2.0

    cx_px = cx * float(w)
    cy_px = cy * float(h)

    x0 = torch.round(cx_px - half).clamp(0, w - 1).long()
    x1 = torch.round(cx_px + half).clamp(0, w - 1).long()
    y0 = torch.round(cy_px - half).clamp(0, h - 1).long()
    y1 = torch.round(cy_px + half).clamp(0, h - 1).long()

    box_masks = torch.zeros((batch_size, 1, h, w), device=device, dtype=torch.float32)

    for i in range(batch_size):
        xi0 = x0[i].item()
        xi1 = x1[i].item()
        yi0 = y0[i].item()
        yi1 = y1[i].item()
        if xi1 >= xi0 and yi1 >= yi0:
            box_masks[i, 0, yi0:yi1 + 1, xi0:xi1 + 1] = 1.0

    return box_masks
# -----------------------------
# Utility: set random seeds
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def orig_norm_to_orig_px(cx: float, cy: float, size: float, orig_h: int, orig_w: int) -> Tuple[float, float, float]:
    """Convert normalized (cx,cy,size) in original space to pixel units in original space."""
    cx_px = cx * float(orig_w)
    cy_px = cy * float(orig_h)
    side_px = size * float(min(orig_h, orig_w))
    return cx_px, cy_px, side_px


def orig_px_to_orig_norm(cx_px: float, cy_px: float, side_px: float, orig_h: int, orig_w: int) -> Tuple[float, float, float]:
    """Convert pixel units in original space to normalized (cx,cy,size) in original space."""
    cx = cx_px / float(orig_w)
    cy = cy_px / float(orig_h)
    size = side_px / float(min(orig_h, orig_w))
    return cx, cy, size


def orig_norm_to_lb_norm(cx: float, cy: float, size: float, meta: LetterboxMeta) -> Tuple[float, float, float]:
    """
    Map original normalized ROI params -> letterboxed normalized ROI params (in [0,1]).

    Assumes:
      - cx normalized by orig width
      - cy normalized by orig height
      - size normalized by min(orig_h, orig_w)
    """
    cx_px, cy_px, side_px = orig_norm_to_orig_px(cx, cy, size, meta.orig_h, meta.orig_w)

    # Forward map into letterboxed pixel space
    cx_lb_px = cx_px * meta.scale + float(meta.pad_x)
    cy_lb_px = cy_px * meta.scale + float(meta.pad_y)
    side_lb_px = side_px * meta.scale

    # Normalize by new_size (square canvas)
    cx_lb = cx_lb_px / float(meta.new_size)
    cy_lb = cy_lb_px / float(meta.new_size)
    size_lb = side_lb_px / float(meta.new_size)
    return cx_lb, cy_lb, size_lb


def lb_norm_to_orig_norm(cx_lb: float, cy_lb: float, size_lb: float, meta: LetterboxMeta) -> Tuple[float, float, float]:
    """Inverse map letterboxed normalized ROI params -> original normalized ROI params."""
    cx_lb_px = cx_lb * float(meta.new_size)
    cy_lb_px = cy_lb * float(meta.new_size)
    side_lb_px = size_lb * float(meta.new_size)

    # Inverse mapping
    cx_px = (cx_lb_px - float(meta.pad_x)) / meta.scale
    cy_px = (cy_lb_px - float(meta.pad_y)) / meta.scale
    side_px = side_lb_px / meta.scale

    return orig_px_to_orig_norm(cx_px, cy_px, side_px, meta.orig_h, meta.orig_w)


def square_box_from_orig_norm(cx: float, cy: float, size: float, orig_h: int, orig_w: int) -> Tuple[float, float, float, float]:
    """Return (x1,y1,x2,y2) in original pixel space from original normalized ROI params."""
    cx_px, cy_px, side_px = orig_norm_to_orig_px(cx, cy, size, orig_h, orig_w)
    half = side_px / 2.0
    x1 = cx_px - half
    x2 = cx_px + half
    y1 = cy_px - half
    y2 = cy_px + half
    # Clip to image canvas for IoU
    x1 = max(0.0, min(float(orig_w), x1))
    x2 = max(0.0, min(float(orig_w), x2))
    y1 = max(0.0, min(float(orig_h), y1))
    y2 = max(0.0, min(float(orig_h), y2))
    return x1, y1, x2, y2


def iou_xyxy(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes in (x1,y1,x2,y2) format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-8
    return float(inter / union)
