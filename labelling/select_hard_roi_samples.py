#!/usr/bin/env python3
"""
Select hard palm ROI samples + visualize GT vs Pred
---------------------------------------------------
- Supports resize_mode: direct / letterbox (true letterbox with padding).
- Loads ROI-Net checkpoint.
- For each mask image:
    * Get prediction (cx, cy, size) in [0,1]
    * Load GT label from JSON (cx, cy, size) in [0,1] (original space convention)
    * Compute:
        - outside_fraction_input  (in model-input space after preprocessing)
        - outside_fraction_orig   (in original resolution)
        - IoU (original space)
        - center L2 (px), abs size error (px) in original space
    * Save visualization on ORIGINAL image with:
        - GT box (green)
        - Pred box (red)
- Select "hard" samples based on chosen outside_fraction space.

All comments are in English as requested.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm.auto import tqdm


# -----------------------------
# Model: Palm ROI-Net (same family as training)
# -----------------------------
class PalmROINet(nn.Module):
    """
    MobileNetV2-based ROI regression:
    Input : (B,1,H,W)
    Output: (B,3) sigmoid in [0,1] -> (cx, cy, size)
    """
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)

        # Modify first conv to 1-channel
        first_conv = self.backbone.features[0][0]
        new_first = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        self.backbone.features[0][0] = new_first

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return self.sigmoid(out)


# -----------------------------
# Geometry helpers
# -----------------------------
def params_to_box(cx: float, cy: float, size: float, H: int, W: int) -> Tuple[int, int, int, int]:
    """
    Convert normalized (cx, cy, size) in [0,1] to pixel square box (x0,y0,x1,y1).
    size is normalized by min(H, W).
    """
    side = size * float(min(H, W))
    half = 0.5 * side

    cx_px = cx * float(W)
    cy_px = cy * float(H)

    x0 = int(round(cx_px - half))
    x1 = int(round(cx_px + half))
    y0 = int(round(cy_px - half))
    y1 = int(round(cy_px + half))

    x0 = max(0, min(W - 1, x0))
    x1 = max(0, min(W - 1, x1))
    y0 = max(0, min(H - 1, y0))
    y1 = max(0, min(H - 1, y1))

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    return x0, y0, x1, y1


def box_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """IoU of two axis-aligned boxes (inclusive coords)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    iw = max(0, ix1 - ix0 + 1)
    ih = max(0, iy1 - iy0 + 1)
    inter = float(iw * ih)

    area_a = float((ax1 - ax0 + 1) * (ay1 - ay0 + 1))
    area_b = float((bx1 - bx0 + 1) * (by1 - by0 + 1))
    union = area_a + area_b - inter + 1e-8
    return float(inter / union)


def outside_fraction_on_original(mask_bin: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    """
    Compute outside fraction on ORIGINAL space:
    outside = sum(box & !hand) / sum(box)
    """
    x0, y0, x1, y1 = box
    roi = mask_bin[y0:y1 + 1, x0:x1 + 1]  # 1=hand, 0=bg
    box_area = float(roi.size) + 1e-6
    outside = float((roi == 0).sum())
    return outside / box_area


# -----------------------------
# Letterbox helpers (true letterbox with padding)
# -----------------------------
def letterbox_meta(orig_h: int, orig_w: int, target: int):
    """
    Scale to fit inside target square then pad to center.
    scale = min(T/W, T/H) == T / max(W,H)
    """
    T = int(target)
    scale = min(T / float(orig_w), T / float(orig_h))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (T - new_w) // 2
    pad_y = (T - new_h) // 2
    return scale, new_w, new_h, pad_x, pad_y


def letterbox_mask_to_square(mask_u8: np.ndarray, target: int) -> np.ndarray:
    """
    Letterbox-resize a mask to target x target square (preserve aspect ratio).
    Returns only the final square mask (uint8).
    """
    h, w = mask_u8.shape[:2]
    scale, new_w, new_h, pad_x, pad_y = letterbox_meta(h, w, target)
    resized = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    out = np.zeros((target, target), dtype=np.uint8)
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return out


def label_original_to_target_letterbox(cx: float, cy: float, size: float, orig_h: int, orig_w: int, target: int):
    """
    Convert ORIGINAL-normalized label -> TARGET(letterbox)-normalized label.
    - cx,cy normalized by orig_w, orig_h
    - size normalized by min(orig_h, orig_w)
    Output:
    - cx_t,cy_t,size_t normalized by target
    """
    T = int(target)
    scale, new_w, new_h, pad_x, pad_y = letterbox_meta(orig_h, orig_w, T)

    cx_px = cx * float(orig_w)
    cy_px = cy * float(orig_h)
    side_px = size * float(min(orig_h, orig_w))

    cx_lb = cx_px * scale + float(pad_x)
    cy_lb = cy_px * scale + float(pad_y)
    side_lb = side_px * scale

    cx_t = float(np.clip(cx_lb / float(T), 0.0, 1.0))
    cy_t = float(np.clip(cy_lb / float(T), 0.0, 1.0))
    size_t = float(np.clip(side_lb / float(T), 0.0, 1.0))
    return cx_t, cy_t, size_t


def params_target_to_original_box(cx_t: float, cy_t: float, size_t: float, orig_h: int, orig_w: int, target: int):
    """
    Inverse letterbox: target-normalized params -> ORIGINAL pixel box.
    """
    T = int(target)
    scale, new_w, new_h, pad_x, pad_y = letterbox_meta(orig_h, orig_w, T)

    cx_lb = cx_t * float(T)
    cy_lb = cy_t * float(T)
    side_lb = size_t * float(T)

    cx_org = (cx_lb - float(pad_x)) / (scale + 1e-12)
    cy_org = (cy_lb - float(pad_y)) / (scale + 1e-12)
    side_org = side_lb / (scale + 1e-12)

    half = 0.5 * side_org
    x0 = int(round(cx_org - half))
    x1 = int(round(cx_org + half))
    y0 = int(round(cy_org - half))
    y1 = int(round(cy_org + half))

    x0 = max(0, min(orig_w - 1, x0))
    x1 = max(0, min(orig_w - 1, x1))
    y0 = max(0, min(orig_h - 1, y0))
    y1 = max(0, min(orig_h - 1, y1))

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    return x0, y0, x1, y1


# -----------------------------
# Input-space box mask + outside fraction
# -----------------------------
def box_mask_from_params_inputspace(cx: float, cy: float, size: float, S: int) -> np.ndarray:
    """
    Build hard box mask in input-space (SxS).
    NOTE: Here size is interpreted as normalized by min(S,S)=S.
    """
    side = size * float(S)
    half = 0.5 * side
    cx_px = cx * float(S)
    cy_px = cy * float(S)

    x0 = int(round(cx_px - half))
    x1 = int(round(cx_px + half))
    y0 = int(round(cy_px - half))
    y1 = int(round(cy_px + half))

    x0 = max(0, min(S - 1, x0))
    x1 = max(0, min(S - 1, x1))
    y0 = max(0, min(S - 1, y0))
    y1 = max(0, min(S - 1, y1))

    m = np.zeros((S, S), dtype=np.uint8)
    m[y0:y1 + 1, x0:x1 + 1] = 1
    return m


def outside_fraction_input(hand_bin_input: np.ndarray, box_mask_input: np.ndarray) -> float:
    """
    outside = sum(box & !hand)/sum(box) in input space.
    hand_bin_input, box_mask_input are (S,S) with {0,1}
    """
    box_area = float(box_mask_input.sum()) + 1e-6
    outside = float(((box_mask_input == 1) & (hand_bin_input == 0)).sum())
    return outside / box_area


# -----------------------------
# Visualization
# -----------------------------
def draw_vis_original(
    orig_gray: np.ndarray,
    gt_box: Tuple[int, int, int, int],
    pr_box: Tuple[int, int, int, int],
    iou: float,
    out_frac_orig: float,
    out_frac_in: float,
    out_path: Path,
    title: str
) -> None:
    """
    Save visualization on ORIGINAL image:
    - GT (green), Pred (red)
    """
    img = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)

    # GT: green
    cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 2)
    # Pred: red
    cv2.rectangle(img, (pr_box[0], pr_box[1]), (pr_box[2], pr_box[3]), (0, 0, 255), 2)

    cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"IoU={iou:.4f}", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"out(orig)={out_frac_orig:.5f}", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"out(input)={out_frac_in:.5f}", (8, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# -----------------------------
# Args / Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Select hard ROI samples + visualize GT vs Pred")

    p.add_argument("--masks_dir", type=str, default="dataset_palm_locator_seed42/test",
                   help="Folder of masks to scan (e.g., .../test).")
    p.add_argument("--labels_json", type=str, default="dataset_palm_locator_seed42/test_labels.json",
                   help="GT labels JSON (filename -> {cx,cy,size}) in ORIGINAL space convention.")
    p.add_argument("--ckpt", type=str, default="ablation_runs/letterbox_noPenalty/seed42/roi_net_best.pth",
                   help="Checkpoint path (roi_net_best.pth).")

    p.add_argument("--resize_mode", type=str, default="letterbox", choices=["direct", "letterbox"],
                   help="Must match how the model was trained.")
    p.add_argument("--img_size", type=int, default=256,
                   help="Model input size (square), must match training.")

    p.add_argument("--hard_space", type=str, default="input", choices=["input", "original"],
                   help="Which outside_fraction to threshold for hard samples.")
    p.add_argument("--outside_thr", type=float, default=0.001,
                   help="Threshold for outside fraction to mark hard samples.")
    p.add_argument("--max_hard", type=int, default=999999,
                   help="Max hard samples to save/copy.")

    p.add_argument("--out_dir", type=str, default="roinet_samples_test_nopenality_lettrbox_input",
                   help="Output folder to store hard masks/vis/json.")
    p.add_argument("--extensions", type=str, nargs="+",
                   default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
                   help="Extensions to include.")

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    masks_dir = Path(args.masks_dir)
    if not masks_dir.is_dir():
        raise RuntimeError(f"masks_dir does not exist: {masks_dir}")

    labels_path = Path(args.labels_json)
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels_json not found: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as f:
        labels: Dict[str, Dict[str, float]] = json.load(f)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Collect masks
    exts = tuple(e.lower() for e in args.extensions)
    mask_paths = sorted([p for p in masks_dir.rglob("*") if p.suffix.lower() in exts])
    if not mask_paths:
        raise RuntimeError(f"No mask images found under: {masks_dir}")

    print(f"[INFO] Masks found: {len(mask_paths)}")
    print(f"[INFO] Labels found: {len(labels)}")

    # Output dirs
    out_dir = Path(args.out_dir)
    hard_masks_dir = out_dir / "hard_masks"
    hard_vis_dir = out_dir / "hard_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    hard_masks_dir.mkdir(parents=True, exist_ok=True)
    hard_vis_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = PalmROINet().to(device)
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    all_records: List[Dict] = []
    hard_records: List[Dict] = []
    hard_count = 0

    S = int(args.img_size)

    for path in tqdm(mask_paths, desc="Scanning masks", ncols=90):
        fname = path.name
        if fname not in labels:
            # Skip if no GT
            continue

        orig = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if orig is None:
            continue

        orig_h, orig_w = orig.shape[:2]
        mask_bin_orig = (orig > 0).astype(np.uint8)  # 1=hand

        gt = labels[fname]
        gt_cx = float(gt["cx"])
        gt_cy = float(gt["cy"])
        gt_sz = float(gt["size"])

        # Build model input mask according to resize_mode
        if args.resize_mode == "letterbox":
            inp_u8 = letterbox_mask_to_square(orig.astype(np.uint8), S)
        else:
            inp_u8 = cv2.resize(orig.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)

        inp_f = (inp_u8.astype(np.float32) / 255.0)
        inp_t = torch.from_numpy(inp_f).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,S,S)

        with torch.no_grad():
            pred = model(inp_t)[0].detach().cpu().numpy()
        pr_cx = float(pred[0]); pr_cy = float(pred[1]); pr_sz = float(pred[2])

        # ---- Compute boxes on ORIGINAL space for visualization + IoU ----
        gt_box_orig = params_to_box(gt_cx, gt_cy, gt_sz, orig_h, orig_w)

        if args.resize_mode == "letterbox":
            # Pred is in TARGET(letterbox) space -> map back to original
            pr_box_orig = params_target_to_original_box(pr_cx, pr_cy, pr_sz, orig_h, orig_w, S)
            # For input-space outside, use pred params directly in SxS
            pr_boxmask_in = box_mask_from_params_inputspace(pr_cx, pr_cy, pr_sz, S)
            hand_bin_in = (inp_u8 > 0).astype(np.uint8)
            out_in = outside_fraction_input(hand_bin_in, pr_boxmask_in)
        else:
            # Pred is treated as ORIGINAL-normalized (as in your evaluation)
            pr_box_orig = params_to_box(pr_cx, pr_cy, pr_sz, orig_h, orig_w)
            # Input-space outside (approx, consistent with SxS application)
            pr_boxmask_in = box_mask_from_params_inputspace(pr_cx, pr_cy, pr_sz, S)
            hand_bin_in = (inp_u8 > 0).astype(np.uint8)
            out_in = outside_fraction_input(hand_bin_in, pr_boxmask_in)

        iou = box_iou_xyxy(gt_box_orig, pr_box_orig)
        out_orig = outside_fraction_on_original(mask_bin_orig, pr_box_orig)

        # Center / size errors on ORIGINAL
        gt_cx_px = 0.5 * (gt_box_orig[0] + gt_box_orig[2])
        gt_cy_px = 0.5 * (gt_box_orig[1] + gt_box_orig[3])
        pr_cx_px = 0.5 * (pr_box_orig[0] + pr_box_orig[2])
        pr_cy_px = 0.5 * (pr_box_orig[1] + pr_box_orig[3])
        center_l2_px = float(((pr_cx_px - gt_cx_px) ** 2 + (pr_cy_px - gt_cy_px) ** 2) ** 0.5)

        gt_side = float(gt_box_orig[2] - gt_box_orig[0] + 1)
        pr_side = float(pr_box_orig[2] - pr_box_orig[0] + 1)
        abs_size_err_px = float(abs(pr_side - gt_side))

        rec = {
            "filename": str(path.relative_to(masks_dir)),
            "name": fname,
            "resize_mode": args.resize_mode,

            "gt_cx": gt_cx, "gt_cy": gt_cy, "gt_size": gt_sz,
            "pr_cx": pr_cx, "pr_cy": pr_cy, "pr_size": pr_sz,

            "iou_orig": float(iou),
            "outside_fraction_input": float(out_in),
            "outside_fraction_orig": float(out_orig),
            "center_l2_px": center_l2_px,
            "abs_size_err_px": abs_size_err_px,

            "orig_h": int(orig_h),
            "orig_w": int(orig_w),
        }
        all_records.append(rec)

        # Decide "hard" by chosen space
        hard_value = out_in if args.hard_space == "input" else out_orig
        if hard_value >= args.outside_thr and hard_count < args.max_hard:
            hard_records.append(rec)
            hard_count += 1

            # Copy original mask
            rel_path = path.relative_to(masks_dir)
            dest_mask = hard_masks_dir / rel_path
            dest_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_mask)

            # Save visualization on ORIGINAL
            vis_rel = rel_path.with_suffix(".png")
            vis_path = hard_vis_dir / vis_rel
            draw_vis_original(
                orig_gray=orig,
                gt_box=gt_box_orig,
                pr_box=pr_box_orig,
                iou=iou,
                out_frac_orig=out_orig,
                out_frac_in=out_in,
                out_path=vis_path,
                title=f"{args.resize_mode} | {fname}"
            )

    # Save JSON outputs
    all_json_path = out_dir / "all_samples_metrics.json"
    hard_json_path = out_dir / "hard_samples.json"

    with all_json_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    with hard_json_path.open("w", encoding="utf-8") as f:
        json.dump(hard_records, f, indent=2)

    print("\n[INFO] Done.")
    print(f"[INFO] Total scanned (with GT): {len(all_records)}")
    print(f"[INFO] Hard samples saved      : {len(hard_records)}")
    print(f"[INFO] Hard masks dir          : {hard_masks_dir}")
    print(f"[INFO] Hard vis dir            : {hard_vis_dir}")
    print(f"[INFO] all_samples_metrics.json: {all_json_path}")
    print(f"[INFO] hard_samples.json       : {hard_json_path}")


if __name__ == "__main__":
    main()
