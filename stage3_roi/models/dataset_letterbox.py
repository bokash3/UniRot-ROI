#!/usr/bin/env python3
"""
Letterbox-enabled ROI Dataset for Palm ROI regression.

Key idea:
- Always feed the model a letterboxed square input (e.g., 256x256).
- Always convert labels into the SAME letterboxed coordinate system for training.
- During evaluation, you can invert predictions back to original space using meta.

Label convention (must match your JSON):
- cx normalized by original width
- cy normalized by original height
- size normalized by min(original_h, original_w)

All comments are in English as requested.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from letterbox_utils import letterbox_gray, orig_norm_to_lb_norm, LetterboxMeta


class ROIDatasetLetterbox(Dataset):
    """
    Loads grayscale masks/images, applies letterbox to new_size, and returns:
      - image tensor: (1,new_size,new_size) float32 in [0,1]
      - target tensor: (3,) float32 in [0,1] in LETTERBOX space
      - meta dict (optional): needed to map predictions back to original space
    """

    def __init__(
        self,
        images_dir: Path,
        labels: Dict[str, Dict[str, float]],
        img_size: int = 256,
        augment: bool = False,
        return_meta: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.labels = labels
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.return_meta = bool(return_meta)

        self.keys = sorted(list(labels.keys()))

    def __len__(self) -> int:
        return len(self.keys)

    def _read_gray(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return img

    def __getitem__(self, idx: int):
        name = self.keys[idx]
        path = self.images_dir / name

        img = self._read_gray(path)  # original HxW
        ann = self.labels[name]
        cx_o = float(ann["cx"])
        cy_o = float(ann["cy"])
        sz_o = float(ann["size"])

        # Letterbox image to square
        # For masks, nearest-neighbor interpolation is the safest.
        img_lb, meta = letterbox_gray(img, new_size=self.img_size, pad_value=0, interpolation=cv2.INTER_NEAREST)

        # Map original normalized label -> letterbox normalized label
        cx_lb, cy_lb, sz_lb = orig_norm_to_lb_norm(cx_o, cy_o, sz_o, meta)

        # Optional augmentation: horizontal flip (in letterbox space).
        # If you flip the image, you must flip cx in the SAME coordinate system.
        if self.augment:
            if np.random.rand() < 0.5:
                img_lb = np.ascontiguousarray(img_lb[:, ::-1])
                cx_lb = 1.0 - cx_lb

        # Normalize image to [0,1] float32
        if img_lb.dtype != np.float32:
            img_f = img_lb.astype(np.float32)
        else:
            img_f = img_lb
        if img_f.max() > 1.0:
            img_f = img_f / 255.0

        x = torch.from_numpy(img_f).unsqueeze(0)  # (1,H,W)
        y = torch.tensor([cx_lb, cy_lb, sz_lb], dtype=torch.float32)

        if not self.return_meta:
            return x, y

        meta_dict = {
            "name": name,
            "orig_h": meta.orig_h,
            "orig_w": meta.orig_w,
            "new_size": meta.new_size,
            "scale": meta.scale,
            "pad_x": meta.pad_x,
            "pad_y": meta.pad_y,
            # Store original labels too (for evaluation in original space)
            "gt_cx_orig": cx_o,
            "gt_cy_orig": cy_o,
            "gt_size_orig": sz_o,
        }
        return x, y, meta_dict
