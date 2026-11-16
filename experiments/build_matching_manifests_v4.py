#!/usr/bin/env python3
"""
build_matching_manifests.py
============================

Builds the JSON manifests consumed by matching_experiment.py.

A manifest is a small JSON describing where to read each image from and
what predicted/GT ROI to crop:

    {
      "image_root": "D:/.../data/images_aligned",
      "roi_type":   "original_normalized",
      "entries": {
        "001_l_460_01.png": {"cx": 0.51, "cy": 0.60, "size": 0.40},
        ...
      }
    }

ALL manifests for Phase 4 must share the SAME image_root (the aligned
images folder) and SAME coordinate frame (aligned-frame original-
normalized), so that one cropping function works for every source.

This script supports four sources:

  gt
    Use the seed's test_labels.json directly. Already in aligned frame.

  litepalmroi
    Run a saved PalmROINet (MobileNetV2) checkpoint on the aligned masks
    in <seed_root>/test/. Predictions come out in letterbox-normalized
    space and are inverted to aligned-frame original-normalized for the
    manifest.

  fastpalmroi
    Same as above but for SmallCNNBaseline. Both run on aligned masks,
    so both produce aligned-frame predictions.

  baseline_A
    Run the saved Baseline A (MobileNetV2 with 3-channel input) on raw
    images. Predictions come out in raw-frame original-normalized space.
    We then ROTATE THE CENTER FORWARD by -theta (the same direction your
    pipeline rotates raw -> aligned) so the manifest ends up in aligned
    frame, matching the other sources.

Each subcommand accepts --verify, which writes 30 overlay images to
<out>.verify/ so you can spot-check that the predicted ROI lands on
the palm in the aligned image. Run this for one seed before launching
Phase 4 -- silent coordinate-frame bugs are exactly what we want to
catch here.

Usage examples (run for each seed in {42, 2025, 1337}):

    # 1. GT
    python build_matching_manifests.py gt \
        --labels_json dataset_palm_locator_seed42/test_labels.json \
        --image_root  data/images_aligned \
        --out         manifests/gt_seed42.json

    # 2. Lite-PalmROI (MobileNetV2)
    python build_matching_manifests.py litepalmroi \
        --ckpt        roi_net_runs/roi_net_best_seed42.pth \
        --masks_dir   dataset_palm_locator_seed42/test \
        --labels_json dataset_palm_locator_seed42/test_labels.json \
        --image_root  data/images_aligned \
        --out         manifests/litepalmroi_seed42.json

    # 3. Fast-PalmROI (SmallCNN)
    python build_matching_manifests.py fastpalmroi \
        --ckpt        small_cnn_runs/<filename>_seed42.pth \
        --masks_dir   dataset_palm_locator_seed42/test \
        --labels_json dataset_palm_locator_seed42/test_labels.json \
        --image_root  data/images_aligned \
        --out         manifests/fastpalmroi_seed42.json

    # 4. Baseline A
    python build_matching_manifests.py baseline_A \
        --ckpt        runs/baseline_A/seed42/baseline_A_best.pth \
        --images_dir  data/images \
        --angles      data/angles.json \
        --labels_json dataset_palm_locator_seed42_raw/test_labels.json \
        --image_root  data/images_aligned \
        --canvas_w 768 --canvas_h 576 \
        --out         manifests/baseline_A_seed42.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm.auto import tqdm


# =====================================================================
# Architectures (must match the training scripts that produced .pth)
# =====================================================================
class PalmROINet(nn.Module):
    """Lite-PalmROI: MobileNetV2 backbone, 1-channel input. Matches ROINet.py."""
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        first_conv = self.backbone.features[0][0]
        new_first_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        self.backbone.features[0][0] = new_first_conv

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.backbone(x))


class SmallCNNBaseline(nn.Module):
    """Fast-PalmROI: 4 conv blocks + small MLP. Matches SmallCNN.py."""
    def __init__(self, img_size: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        feat_hw = img_size // (2 ** 4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feat_hw * feat_hw, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 3),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.head(self.features(x)))


class ImageROINet(nn.Module):
    """Baseline A: MobileNetV2 backbone, 3-channel input. Matches train_baseline_A_image_roi.py."""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        first_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        self.backbone.features[0][0] = new_conv

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.backbone(x))


# =====================================================================
# Letterbox helpers
# =====================================================================
def letterbox_meta(h: int, w: int, target: int):
    T = int(target)
    scale = min(T / float(w), T / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_x = (T - new_w) // 2
    pad_y = (T - new_h) // 2
    return scale, new_w, new_h, pad_x, pad_y


def letterbox_image(img: np.ndarray, target: int, is_color: bool) -> np.ndarray:
    h, w = img.shape[:2]
    scale, new_w, new_h, pad_x, pad_y = letterbox_meta(h, w, target)
    interp = cv2.INTER_AREA if is_color else cv2.INTER_NEAREST
    img_rs = cv2.resize(img, (new_w, new_h), interpolation=interp)
    if is_color:
        canvas = np.zeros((target, target, 3), dtype=img.dtype)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_rs
    else:
        canvas = np.zeros((target, target), dtype=img.dtype)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_rs
    return canvas


def letterbox_to_original_normalized(cx_n, cy_n, size_n, orig_h, orig_w, target):
    """
    Invert letterbox: predictions in target-normalized [0,1]^3 -> original-
    normalized (cx by W, cy by H, size by min(H,W)).
    """
    scale, _, _, pad_x, pad_y = letterbox_meta(orig_h, orig_w, target)
    T = float(target)
    cx_lb_px = cx_n * T
    cy_lb_px = cy_n * T
    side_lb_px = size_n * T

    cx_org_px = (cx_lb_px - pad_x) / (scale + 1e-12)
    cy_org_px = (cy_lb_px - pad_y) / (scale + 1e-12)
    side_org_px = side_lb_px / (scale + 1e-12)

    cx_o = float(np.clip(cx_org_px / float(orig_w), 0.0, 1.0))
    cy_o = float(np.clip(cy_org_px / float(orig_h), 0.0, 1.0))
    sz_o = float(np.clip(side_org_px / float(min(orig_h, orig_w)), 0.0, 1.0))
    return cx_o, cy_o, sz_o


# =====================================================================
# Coordinate-frame conversion: raw -> aligned (used for Baseline A)
# =====================================================================
def rotate_point_around_center(cx_px, cy_px, theta_deg, W, H):
    """
    Apply cv2.getRotationMatrix2D(..., theta_deg, 1.0) to a single point.
    Same convention as cv2.warpAffine.
    """
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), float(theta_deg), 1.0)
    new_x = M[0, 0] * cx_px + M[0, 1] * cy_px + M[0, 2]
    new_y = M[1, 0] * cx_px + M[1, 1] * cy_px + M[1, 2]
    return float(new_x), float(new_y)


def raw_to_aligned_label(cx_n, cy_n, size_n, theta_deg, W, H, mode="aabb",
                         rotation_sign="minus"):
    """
    Forward conversion: a label in RAW-frame normalized coords -> ALIGNED-frame
    normalized coords.

    For a square ROI: the rotated raw-frame square becomes a tilted square
    in the aligned frame. We approximate it with an axis-aligned square in
    aligned-frame coordinates.

    rotation_sign:
      'minus' (default): apply rotation by -theta. This is the algebraic inverse
        of Phase 1's '+theta' aligned->raw conversion.
      'plus' : apply rotation by +theta. Use this if 'minus' produces visually
        wrong overlays in the aligned image -- it most likely means the
        angles.json convention is signed opposite to what we assumed.

    mode='aabb' takes the AABB of the rotated 4 corners (consistent with
    Phase 1's --mode aabb when going aligned -> raw).
    """
    cx_raw_px = cx_n * W
    cy_raw_px = cy_n * H
    side_raw_px = size_n * float(min(H, W))

    sign = -1.0 if rotation_sign == "minus" else +1.0
    inv_theta = sign * float(theta_deg)

    if mode == "center_only":
        cx_a_px, cy_a_px = rotate_point_around_center(cx_raw_px, cy_raw_px, inv_theta, W, H)
        new_side_px = side_raw_px
    elif mode == "aabb":
        half = side_raw_px / 2.0
        corners = [
            (cx_raw_px - half, cy_raw_px - half),
            (cx_raw_px + half, cy_raw_px - half),
            (cx_raw_px + half, cy_raw_px + half),
            (cx_raw_px - half, cy_raw_px + half),
        ]
        rotated = [rotate_point_around_center(x, y, inv_theta, W, H) for x, y in corners]
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        cx_a_px = 0.5 * (max(xs) + min(xs))
        cy_a_px = 0.5 * (max(ys) + min(ys))
        new_side_px = max(max(xs) - min(xs), max(ys) - min(ys))
    else:
        raise ValueError(mode)

    cx_a_n = float(np.clip(cx_a_px / float(W), 0.0, 1.0))
    cy_a_n = float(np.clip(cy_a_px / float(H), 0.0, 1.0))
    sz_a_n = float(np.clip(new_side_px / float(min(H, W)), 0.0, 1.0))
    return cx_a_n, cy_a_n, sz_a_n


# =====================================================================
# Manifest writing + verification
# =====================================================================
def write_manifest(out_path: Path, image_root: str, entries: Dict, roi_type="original_normalized"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "image_root": str(image_root).replace("\\", "/"),
        "roi_type": roi_type,
        "entries": entries,
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] wrote {len(entries)} entries -> {out_path}")


def render_verification(image_root: Path, entries: Dict, out_dir: Path,
                        max_n: int = 30):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(entries.keys())[:max_n]
    n = 0
    for fname in names:
        img_path = image_root / fname
        if not img_path.exists():
            stem = Path(fname).stem
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                alt = image_root / (stem + ext)
                if alt.exists():
                    img_path = alt
                    break
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        ld = entries[fname]
        cx_px = ld["cx"] * W
        cy_px = ld["cy"] * H
        side = ld["size"] * min(H, W)
        half = side * 0.5
        x0 = int(round(cx_px - half)); x1 = int(round(cx_px + half))
        y0 = int(round(cy_px - half)); y1 = int(round(cy_px + half))
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / fname), img)
        n += 1
    print(f"[VERIFY] wrote {n} overlays -> {out_dir}")
    if n == 0:
        print("[WARN] No overlays were written. Check that --image_root points "
              "at a folder containing images that match the manifest entries.")


# =====================================================================
# Subcommand: gt
# =====================================================================
def cmd_gt(args):
    with open(args.labels_json) as f:
        entries = json.load(f)
    out_path = Path(args.out)
    write_manifest(out_path, args.image_root, entries)
    if args.verify:
        render_verification(Path(args.image_root), entries,
                            Path(str(out_path) + ".verify"), args.verify_count)


# =====================================================================
# Subcommand: litepalmroi (MobileNetV2-based, 1-channel mask input)
# =====================================================================
def _load_state(ckpt_path: Path, model: nn.Module, device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] {ckpt_path.name}: missing keys: {missing[:3]} ...")
    if unexpected:
        print(f"[WARN] {ckpt_path.name}: unexpected keys: {unexpected[:3]} ...")
    model.to(device)
    model.eval()


def _predict_from_masks(model, masks_dir, labels_json, target, device):
    """
    Run a 1-channel mask-input regressor on aligned masks. Returns entries
    in ALIGNED-frame original-normalized coordinates.

    The 'orig_h, orig_w' here come from the actual mask file on disk
    (which lives on the 768x576 aligned canvas in your pipeline).
    """
    with open(labels_json) as f:
        names = list(json.load(f).keys())

    entries = {}
    with torch.no_grad():
        for fname in tqdm(sorted(names), desc=Path(masks_dir).name):
            mp = Path(masks_dir) / fname
            if not mp.exists():
                continue
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            orig_h, orig_w = mask.shape[:2]
            mask_lb = letterbox_image(mask, target, is_color=False)
            t = (torch.from_numpy(mask_lb.astype(np.float32) / 255.0)
                 .unsqueeze(0).unsqueeze(0).to(device))
            pred = model(t).squeeze(0).cpu().numpy()
            cx_o, cy_o, sz_o = letterbox_to_original_normalized(
                float(pred[0]), float(pred[1]), float(pred[2]),
                orig_h, orig_w, target,
            )
            entries[fname] = {"cx": cx_o, "cy": cy_o, "size": sz_o}
    return entries


def cmd_litepalmroi(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PalmROINet()
    _load_state(Path(args.ckpt), model, device)
    entries = _predict_from_masks(model, args.masks_dir, args.labels_json,
                                  args.img_size, device)
    out_path = Path(args.out)
    write_manifest(out_path, args.image_root, entries)
    if args.verify:
        render_verification(Path(args.image_root), entries,
                            Path(str(out_path) + ".verify"), args.verify_count)


def cmd_fastpalmroi(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNNBaseline(img_size=args.img_size)
    _load_state(Path(args.ckpt), model, device)
    entries = _predict_from_masks(model, args.masks_dir, args.labels_json,
                                  args.img_size, device)
    out_path = Path(args.out)
    write_manifest(out_path, args.image_root, entries)
    if args.verify:
        render_verification(Path(args.image_root), entries,
                            Path(str(out_path) + ".verify"), args.verify_count)


# =====================================================================
# Subcommand: baseline_A (3-channel raw image input)
# =====================================================================
def cmd_baseline_A(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageROINet(in_channels=3)
    _load_state(Path(args.ckpt), model, device)

    with open(args.labels_json) as f:
        names = list(json.load(f).keys())

    images_dir = Path(args.images_dir)
    entries: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for fname in tqdm(sorted(names), desc="Baseline A"):
            ip = images_dir / fname
            if not ip.exists():
                stem = Path(fname).stem
                for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                    alt = images_dir / (stem + ext)
                    if alt.exists():
                        ip = alt
                        break
            if not ip.exists():
                continue
            img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb.shape[:2]

            img_lb = letterbox_image(img_rgb, args.img_size, is_color=True)
            t = (torch.from_numpy(img_lb.astype(np.float32) / 255.0)
                 .permute(2, 0, 1).unsqueeze(0).to(device))
            pred = model(t).squeeze(0).cpu().numpy()
            cx_r, cy_r, sz_r = letterbox_to_original_normalized(
                float(pred[0]), float(pred[1]), float(pred[2]),
                orig_h, orig_w, args.img_size,
            )
            entries[fname] = {"cx": cx_r, "cy": cy_r, "size": sz_r}

    out_path = Path(args.out)
    # image_root points at the RAW image folder. The matching script will crop
    # raw images using these raw-frame coordinates. No rotation conversion.
    write_manifest(out_path, args.image_root, entries)
    if args.verify:
        render_verification(Path(args.image_root), entries,
                            Path(str(out_path) + ".verify"), args.verify_count)


# =====================================================================
# Argparse
# =====================================================================
def make_parser():
    p = argparse.ArgumentParser("Build matching manifests")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- gt ----
    g = sub.add_parser("gt", help="Use ground-truth labels directly")
    g.add_argument("--labels_json", required=True,
                   help="dataset_palm_locator_seed<S>/test_labels.json (aligned-frame).")
    g.add_argument("--image_root", required=True,
                   help="data/images_aligned (used by Phase 4).")
    g.add_argument("--out", required=True)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--verify_count", type=int, default=30)
    g.set_defaults(func=cmd_gt)

    # ---- litepalmroi ----
    lp = sub.add_parser("litepalmroi", help="Run a saved Lite-PalmROI (MobileNetV2) checkpoint")
    lp.add_argument("--ckpt", required=True)
    lp.add_argument("--masks_dir", required=True,
                    help="dataset_palm_locator_seed<S>/test (aligned masks).")
    lp.add_argument("--labels_json", required=True,
                    help="Used only to fix the list of test filenames.")
    lp.add_argument("--image_root", required=True)
    lp.add_argument("--img_size", type=int, default=256)
    lp.add_argument("--out", required=True)
    lp.add_argument("--verify", action="store_true")
    lp.add_argument("--verify_count", type=int, default=30)
    lp.set_defaults(func=cmd_litepalmroi)

    # ---- fastpalmroi ----
    fp = sub.add_parser("fastpalmroi", help="Run a saved Fast-PalmROI (SmallCNN) checkpoint")
    fp.add_argument("--ckpt", required=True)
    fp.add_argument("--masks_dir", required=True)
    fp.add_argument("--labels_json", required=True)
    fp.add_argument("--image_root", required=True)
    fp.add_argument("--img_size", type=int, default=256)
    fp.add_argument("--out", required=True)
    fp.add_argument("--verify", action="store_true")
    fp.add_argument("--verify_count", type=int, default=30)
    fp.set_defaults(func=cmd_fastpalmroi)

    # ---- baseline_A ----
    ba = sub.add_parser("baseline_A", help="Run a saved Baseline A checkpoint, emit raw-frame manifest")
    ba.add_argument("--ckpt", required=True)
    ba.add_argument("--images_dir", required=True,
                    help="data/images (raw images). Used to read images for prediction.")
    ba.add_argument("--labels_json", required=True,
                    help="raw-frame labels (used only to fix the filename list).")
    ba.add_argument("--image_root", required=True,
                    help="data/images (raw). The matching script will crop raw images using raw-frame coords.")
    ba.add_argument("--img_size", type=int, default=256)
    ba.add_argument("--out", required=True)
    ba.add_argument("--verify", action="store_true")
    ba.add_argument("--verify_count", type=int, default=30)
    ba.set_defaults(func=cmd_baseline_A)

    return p


def main():
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
