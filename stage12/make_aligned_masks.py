#!/usr/bin/env python3
"""
Mask + Align — Batch Processor (Progress Bar, CSV, Final-only + Aligned Masks + Aligned Originals)
MATCHES TRAIN/EVAL PREPROCESSING (LETTERBOX + IMAGENET NORM)
----------------------------------------------------------------
- Loads ONE selected segmentation model and ONE alignment model.
- Uses the SAME preprocessing as your dataset pipeline:
  LongestMaxSize(max(H,W)) + PadIfNeeded(H,W) + Normalize(ImageNet).
- Takes the predicted mask DIRECTLY from the model (prob or binary).
- Projects the letterboxed mask back to the ORIGINAL image size.
- Rotates the original image by the predicted angle and applies the rotated mask.
- Saves:
    * ONLY the final image (aligned + masked) into: <out_dir>/final/<same relative path>
    * The aligned mask (post-rotation) into:       <out_dir>/aligned_masks/<same relative path>.png
    * The aligned original (no mask) into:         <out_dir>/aligned_images/<same relative path>
- Logs CSV with: relative_path,angle_deg (if --save_csv is provided).
- Shows a progress bar for the whole procedure.
"""

import argparse
import csv
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch
import torch.nn as nn

# Progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # Fallback: simple prints

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg: str):
    print(f"[MaskAlignBatch] {msg}")

# -----------------------------
# Import your project models
# -----------------------------
try:
    from src.models.unet_seg import UNetSeg  # segmentation
except Exception as e:
    UNetSeg = None
    log(f"Warning: could not import src.models.unet_seg: {e}")

try:
    from src.models.rot_net import RotResNet18  # alignment
except Exception as e:
    RotResNet18 = None
    log(f"Warning: could not import src.models.rot_net: {e}")

class AlignHeadType:
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    FUSION = "fusion"

# -----------------------------
# Helpers (image, tensor, io)
# -----------------------------
def rotate_image_and_mask_nocrop(img_rgb, mask_any, angle_deg,
                                 bg_rgb=(0,0,0), margin_scale=1.02,
                                 mode="center_crop"):
    """
    Rotate without losing content:
      1) نحسب حجم القماش المطلوب لاحتواء المستطيل بعد الدوران.
      2) نُدير الصورة والماسك على هذا القماش.
      3) نُعيد إلى HxW إمّا بقصّ مركزي (center_crop) أو بتصغير يحافظ على المحتوى (scale_to_fit).
    """
    H, W = img_rgb.shape[:2]

    # حساب حجم المستطيل بعد الدوران (هندسيًا)
    theta = np.deg2rad(angle_deg)
    c, s = abs(np.cos(theta)), abs(np.sin(theta))
    newW = int(np.ceil(W * c + H * s))
    newH = int(np.ceil(W * s + H * c))

    # هامش أمان
    newW = int(np.ceil(newW * margin_scale))
    newH = int(np.ceil(newH * margin_scale))

    # مصفوفة الدوران حول مركز الصورة الأصلية + إزاحة لوضع الصورة في وسط القماش الجديد
    M = cv2.getRotationMatrix2D((W/2.0, H/2.0), angle_deg, 1.0)
    M[0,2] += (newW - W) / 2.0
    M[1,2] += (newH - H) / 2.0

    rot_img_big = cv2.warpAffine(img_rgb, M, (newW, newH),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=bg_rgb)

    if mask_any.dtype == np.uint8:
        border_val = 0
        interp_m = cv2.INTER_NEAREST
    else:
        border_val = 0.0
        interp_m = cv2.INTER_LINEAR

    rot_msk_big = cv2.warpAffine(mask_any, M, (newW, newH),
                                 flags=interp_m,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=border_val)

    # الرجوع إلى الحجم الأصلي
    if mode == "center_crop":
        # قصّ مركزي بسيط إلى HxW
        y0 = (newH - H)//2; x0 = (newW - W)//2
        y1 = y0 + H;        x1 = x0 + W
        rot_img = rot_img_big[y0:y1, x0:x1]
        rot_msk = rot_msk_big[y0:y1, x0:x1]
    else:  # scale_to_fit
        # نُصغّر القماش الكبير بحيث يطابق HxW بدون فقدان
        rot_img = cv2.resize(rot_img_big, (W, H), interpolation=cv2.INTER_LINEAR)
        interp_back = cv2.INTER_NEAREST if mask_any.dtype == np.uint8 else cv2.INTER_LINEAR
        rot_msk = cv2.resize(rot_msk_big, (W, H), interpolation=interp_back)

    return rot_img, rot_msk

def rotate_image_and_mask_expand(img_rgb, mask_any, angle_deg, pad_pct=0.20, bg_rgb=(0,0,0)):
    """
    Rotate with temporary padding around the image to avoid clipping,
    then center-crop back to the original HxW.
    pad_pct: fraction of max(H,W) added to each side before rotation.
    Works for uint8 or float masks.
    """
    H, W = img_rgb.shape[:2]
    pad = int(round(max(H, W) * pad_pct))

    img_pad = cv2.copyMakeBorder(img_rgb, pad, pad, pad, pad,
                                 cv2.BORDER_CONSTANT, value=bg_rgb)

    # للماسك: إن كان float سنستخدم 0.0 كقيمة حدّية، وإن كان uint8 نستخدم 0
    if mask_any.dtype == np.uint8:
        border_val = 0
    else:
        border_val = 0.0
    msk_pad = cv2.copyMakeBorder(mask_any, pad, pad, pad, pad,
                                 cv2.BORDER_CONSTANT, value=border_val)

    Hp, Wp = img_pad.shape[:2]
    center = (Wp/2.0, Hp/2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    rot_img = cv2.warpAffine(img_pad, M, (Wp, Hp), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=bg_rgb)
    interp_mask = cv2.INTER_NEAREST if mask_any.dtype == np.uint8 else cv2.INTER_LINEAR
    rot_msk = cv2.warpAffine(msk_pad, M, (Wp, Hp), flags=interp_mask,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)

    # قصّ مركزي للعودة لحجم الصورة الأصلي
    y0 = (Hp - H)//2; x0 = (Wp - W)//2
    y1 = y0 + H;      x1 = x0 + W
    return rot_img[y0:y1, x0:x1], rot_msk[y0:y1, x0:x1]

def read_image(path: Path) -> np.ndarray:
    """Read image with OpenCV (BGR) and convert to RGB."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def to_tensor(img: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """Convert HxWxC uint8 RGB to float32 normalized tensor (1,C,H,W)."""
    x = img.astype(np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))  # C,H,W
    return torch.from_numpy(x).unsqueeze(0)

def to_mask_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert HxW float/uint8 mask to (1,1,H,W) float32 in [0,1]."""
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

def sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def resize_letterbox(img: np.ndarray, target_hw: Tuple[int, int]):
    """Letterbox resize (LongestMaxSize + PadIfNeeded). Returns (canvas, scale, (x0,y0), (nh,nw))."""
    th, tw = target_hw
    h, w = img.shape[:2]
    max_side = max(th, tw)
    scale = min(max_side / float(h), max_side / float(w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # pad to (th, tw)
    if img.ndim == 3:
        canvas = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
    else:
        canvas = np.zeros((th, tw), dtype=img.dtype)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas, scale, (x0, y0), (nh, nw)

def unletterbox_to_original(mask_lb: np.ndarray, orig_hw: Tuple[int, int], pad_xy: Tuple[int, int], new_hw: Tuple[int, int]):
    """Project a letterboxed mask back to original image size."""
    oh, ow = orig_hw
    x0, y0 = pad_xy
    nh, nw = new_hw
    crop = mask_lb[y0:y0+nh, x0:x0+nw]
    interp = cv2.INTER_NEAREST if mask_lb.dtype == np.uint8 else cv2.INTER_LINEAR
    back = cv2.resize(crop, (ow, oh), interpolation=interp)
    return back

def _to_float01_mask(mask) -> np.ndarray:
    """Ensure mask is float32 in [0,1] regardless of incoming dtype/value range."""
    if mask.dtype == np.uint8:
        return (mask.astype(np.float32) / 255.0)
    m = mask.astype(np.float32)
    return np.clip(m, 0.0, 1.0)

def apply_mask_rgb(img_rgb: np.ndarray, mask_any, bg_value=(0, 0, 0)) -> np.ndarray:
    """Apply soft/binary mask on RGB image; blends background if bg_value != 0."""
    m = _to_float01_mask(mask_any)
    m3 = np.dstack([m, m, m])
    img_f = img_rgb.astype(np.float32)
    if any(bg_value):
        bg = np.full_like(img_f, bg_value, dtype=np.float32)
        out = img_f * m3 + bg * (1.0 - m3)
    else:
        out = img_f * m3
    return np.clip(out, 0, 255).astype(img_rgb.dtype)

def to_uint8_mask(mask_any: np.ndarray) -> np.ndarray:
    """Convert a predicted/rotated mask (float [0,1] or 0/255 uint8) to 0–255 uint8."""
    if mask_any.dtype == np.uint8:
        return np.clip(mask_any, 0, 255)
    m = np.clip(mask_any.astype(np.float32), 0.0, 1.0)
    return (m * 255.0).round().astype(np.uint8)

# -----------------------------
# Angle decoding
# -----------------------------

def decode_angle_from_model(output, head_type: str, num_bins: int, angle_space: str) -> float:
    """Decode angle (degrees) from RotResNet18 outputs."""
    def cls_center_from_logits(logits_t: torch.Tensor) -> float:
        idx = int(torch.argmax(logits_t, dim=1).item())
        bin_w = 360.0 / float(num_bins)
        center = (idx + 0.5) * bin_w
        return center

    if isinstance(output, tuple) and len(output) == 2:
        logits, vec = output
    else:
        logits, vec = None, output

    if head_type == AlignHeadType.REGRESSION:
        angle = float(RotResNet18.vec_to_deg(vec).squeeze().item())
    elif head_type == AlignHeadType.CLASSIFICATION:
        if logits is None:
            raise ValueError("Classification head requested but logits=None from model output")
        angle = cls_center_from_logits(logits)
    elif head_type == AlignHeadType.FUSION:
        if logits is None or vec is None:
            raise ValueError("Fusion head requires both logits and vec outputs")
        cls_deg = cls_center_from_logits(logits)
        reg_deg = float(RotResNet18.vec_to_deg(vec).squeeze().item())
        angle = 0.5 * reg_deg + 0.5 * cls_deg
    else:
        raise ValueError(f"Unknown head type: {head_type}")

    if angle_space == "0-360":
        angle = angle % 360.0
    else:
        angle = ((angle + 180.0) % 360.0) - 180.0
    return angle

# -----------------------------
# Models
# -----------------------------

def _load_seg_unet(ckpt_path: str, encoders_try: List[str], device) -> nn.Module:
    if UNetSeg is None:
        raise RuntimeError("UNetSeg not available; ensure src.models.unet_seg is on PYTHONPATH")
    last_err = None
    sd_all = torch.load(ckpt_path, map_location=device)
    sd = sd_all.get("model", sd_all) if isinstance(sd_all, dict) else sd_all
    for enc in encoders_try:
        try:
            m = UNetSeg(encoder_name=enc).to(device)
            m.load_state_dict(sd)
            m.eval()
            log(f"Loaded UNetSeg with encoder={enc} from {ckpt_path}")
            return m
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load UNetSeg from {ckpt_path}. Last error: {last_err}")

def build_seg_model(ckpt_path: str, prefer_encoder: Optional[str], device) -> nn.Module:
    encoders_try = []
    if prefer_encoder:
        encoders_try.append(prefer_encoder)
    encoders_try += ["resnet34", "mobilenet_v2"]
    return _load_seg_unet(ckpt_path, encoders_try, device)

def build_align_model(ckpt_path: str, head_type: str, num_bins: int, device) -> nn.Module:
    if RotResNet18 is None:
        raise RuntimeError("RotResNet18 not available; ensure src.models.rot_net is on PYTHONPATH")
    mode_map = {AlignHeadType.REGRESSION: "reg", AlignHeadType.CLASSIFICATION: "cls", AlignHeadType.FUSION: "both"}
    mode = mode_map[head_type]

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    if num_bins is None:
        num_bins = cfg.get("num_bins", 36)

    m = RotResNet18(mode=mode, num_bins=num_bins, grayscale=True).to(device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    m.load_state_dict(state)
    m.eval()
    log(f"Loaded RotResNet18 mode={mode} bins={num_bins} from {ckpt_path}")
    return m

# -----------------------------
# Core per-image processing
# -----------------------------

def process_one_image(img_path: Path, seg: nn.Module, align_model: nn.Module, args):
    """Run segmentation -> alignment -> rotate and return (final_img, angle, rot_mask_u8, rot_img)."""
    img = read_image(img_path)  # RGB

    # Letterbox to training size
    seg_in = img
    pad_xy = (0, 0)
    new_hw = (img.shape[0], img.shape[1])
    if args.seg_input_hw is not None:
        seg_in, _, pad_xy, new_hw = resize_letterbox(img, args.seg_input_hw)

    # Normalize like training
    xt = to_tensor(seg_in, mean=args.seg_mean, std=args.seg_std).to(DEVICE)

    # Forward segmentation
    with torch.no_grad():
        logits = seg(xt)

    # Logits -> prob or binary (on letterboxed size)
    l = logits.detach().float().cpu().numpy()
    p = sigmoid_numpy(l)[0, 0]
    if args.seg_mask_mode == "prob":
        mask_lb = p.astype(np.float32)
    else:
        mask_lb = (p >= args.seg_thr).astype(np.uint8) * 255

    # Project mask back to original image size
    mask_orig = unletterbox_to_original(mask_lb, (img.shape[0], img.shape[1]), pad_xy, new_hw)

    # Prepare alignment input: 1-channel float [0,1]
    align_in = mask_orig.astype(np.float32) / (255.0 if mask_orig.dtype == np.uint8 else 1.0)
    if args.align_input_hw is not None:
        align_in = cv2.resize(align_in, (args.align_input_hw[1], args.align_input_hw[0]), interpolation=cv2.INTER_LINEAR)
    ta = to_mask_tensor(align_in).to(DEVICE)

    # Predict angle
    with torch.no_grad():
        out = align_model(ta)
    angle_deg = decode_angle_from_model(out, args.align_head_type, args.num_angle_bins, args.angle_space)
    apply_angle = -angle_deg if args.rotate_negate else angle_deg

    # Rotate original image + mask
    M = cv2.getRotationMatrix2D((img.shape[1] / 2.0, img.shape[0] / 2.0), apply_angle, 1.0)

        # Rotate original image + mask (geometry-aware)
    if getattr(args, "expand_on_rotate", False):
        rot_img, rot_mask = rotate_image_and_mask_nocrop(
            img_rgb=img,
            mask_any=mask_orig,
            angle_deg=apply_angle,
            bg_rgb=tuple(int(x*255) for x in args.bg_value),
            margin_scale=args.margin_scale,
            mode=args.no_crop_mode
        )
    else:
        # المسار البسيط السابق (قد يقصّ الأطراف)
        M = cv2.getRotationMatrix2D((img.shape[1]/2.0, img.shape[0]/2.0), apply_angle, 1.0)
        interp_mask = cv2.INTER_NEAREST if mask_orig.dtype == np.uint8 else cv2.INTER_LINEAR
        rot_mask = cv2.warpAffine(mask_orig, M, (img.shape[1], img.shape[0]),
                                  flags=interp_mask, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rot_img  = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=tuple(int(x*255) for x in args.bg_value))


    # Apply mask → final image only
    final_img = apply_mask_rgb(rot_img, rot_mask, bg_value=tuple(int(x * 255) for x in args.bg_value))

    # Convert rotated mask to uint8 for saving
    rot_mask_u8 = to_uint8_mask(rot_mask)

    # Return final + angle + rotated mask + rotated original image
    return final_img, float(angle_deg), rot_mask_u8, rot_img

# -----------------------------
# CLI & Batch runner
# -----------------------------

def parse_hw(text: Optional[str]) -> Optional[Tuple[int, int]]:
    if text is None:
        return None
    parts = [p.strip() for p in text.replace("x", ",").replace("X", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected HxW or H,W (e.g., 576x768)")
    return int(parts[0]), int(parts[1])

def parse_mean_std(text: str):
    parts = [float(p.strip()) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expect three comma-separated floats, e.g., 0.485,0.456,0.406")
    return tuple(parts)

def build_argparser():
    parser = argparse.ArgumentParser(description="Mask + Align — Batch Processor (Final-only, Letterbox Matched, +Aligned Masks +Aligned Originals)")

    # I/O
    parser.add_argument("--images_dir", type=str, required=True, help="Input images folder (scanned recursively).")
    parser.add_argument("--out_dir", type=str, required=True, help="Base output folder (creates 'final', 'aligned_masks', and 'aligned_images').")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional CSV filepath to log 'relative_path,angle_deg'.")

    # Segmentation selection
    parser.add_argument("--seg_source", type=str, default="seg1", choices=["seg1", "seg2", "single"],
                        help="Which segmentation model to use: seg1 | seg2 | single (explicit ckpt)")
    parser.add_argument("--seg1_ckpt", type=str, default=None, help="Seg model checkpoint for seg1")
    parser.add_argument("--seg2_ckpt", type=str, default=None, help="Seg model checkpoint for seg2")
    parser.add_argument("--seg_ckpt", type=str, default=None, help="Seg model checkpoint when --seg_source single")
    parser.add_argument("--seg1_encoder", type=str, default=None, help="Optional encoder for seg1 (e.g., resnet34)")
    parser.add_argument("--seg2_encoder", type=str, default=None, help="Optional encoder for seg2")
    parser.add_argument("--seg_encoder", type=str, default=None, help="Optional encoder when --seg_source single")

    # Preprocessing to match training/eval
    parser.add_argument("--seg_input_hw", type=parse_hw, default=(576, 768), help="Seg input size HxW (letterboxed)")
    parser.add_argument("--seg_mean", type=parse_mean_std, default=(0.485, 0.456, 0.406), help="Normalization mean (ImageNet)")
    parser.add_argument("--seg_std", type=parse_mean_std, default=(0.229, 0.224, 0.225), help="Normalization std (ImageNet)")

    # Mask mode selection
    parser.add_argument("--seg_mask_mode", type=str, default="binary", choices=["prob", "binary"],
                        help="Use 'prob' to pass raw sigmoid probabilities [0,1]. 'binary' applies --seg_thr only.")
    parser.add_argument("--seg_thr", type=float, default=0.5, help="Sigmoid threshold when --seg_mask_mode binary")

    # Alignment model
    parser.add_argument("--align_ckpt", type=str, required=True, help="Alignment model checkpoint")
    parser.add_argument("--align_head_type", type=str, default=AlignHeadType.FUSION,
                        choices=[AlignHeadType.REGRESSION, AlignHeadType.CLASSIFICATION, AlignHeadType.FUSION],
                        help="Type of alignment head to decode")
    parser.add_argument("--num_angle_bins", type=int, default=36, help="Num bins for classification/fusion")
    parser.add_argument("--angle_space", type=str, default="0-360", choices=["0-360", "-180_180"], help="Angle range produced by the model")
    parser.add_argument("--align_input_hw", type=parse_hw, default=(576, 768), help="Alignment input size HxW for mask tensor")
    parser.add_argument("--rotate_negate", action="store_true", help="Rotate by -angle instead of +angle")

    # Background fill for masked-out area
    parser.add_argument("--bg_value", type=parse_mean_std, default=(0.0, 0.0, 0.0), help="Background fill RGB in [0,1] when applying mask")
    parser.add_argument("--expand_on_rotate", action="store_true",
                    help="Pad before rotation to avoid clipping, then center-crop back")
    parser.add_argument("--no_crop_mode", type=str, default="center_crop",
                    choices=["center_crop", "scale_to_fit"],
                    help="How to bring rotated canvas back to original size")
    parser.add_argument("--margin_scale", type=float, default=1.02,
                    help="Safety margin when computing expanded canvas (e.g., 1.02–1.10)")

    return parser

def gather_images(root: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def ensure_final_path(base_out: Path, in_root: Path, img_path: Path) -> Path:
    """Return final image path mirroring input structure under out_dir/final."""
    rel = img_path.relative_to(in_root)
    final_path = base_out / "final" / rel
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return final_path

def ensure_mask_path(base_out: Path, in_root: Path, img_path: Path) -> Path:
    """Return aligned-mask path mirroring input structure under out_dir/aligned_masks (forced .png)."""
    rel = img_path.relative_to(in_root)
    rel_png = rel.with_suffix(".png")
    mask_path = base_out / "aligned_masks" / rel_png
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    return mask_path

def ensure_aligned_img_path(base_out: Path, in_root: Path, img_path: Path) -> Path:
    """Return aligned-original-image path under out_dir/aligned_images."""
    rel = img_path.relative_to(in_root)
    aligned_path = base_out / "aligned_images" / rel
    aligned_path.parent.mkdir(parents=True, exist_ok=True)
    return aligned_path

def main():
    parser = build_argparser()
    args = parser.parse_args()

    log(f"Device: {DEVICE}")

    # Build models
    if args.seg_source == "seg1":
        if not args.seg1_ckpt:
            raise SystemExit("--seg1_ckpt is required when --seg_source seg1")
        seg = build_seg_model(args.seg1_ckpt, args.seg1_encoder, DEVICE)
    elif args.seg_source == "seg2":
        if not args.seg2_ckpt:
            raise SystemExit("--seg2_ckpt is required when --seg_source seg2")
        seg = build_seg_model(args.seg2_ckpt, args.seg2_encoder, DEVICE)
    elif args.seg_source == "single":
        if not args.seg_ckpt:
            raise SystemExit("--seg_ckpt is required when --seg_source single")
        seg = build_seg_model(args.seg_ckpt, args.seg_encoder, DEVICE)
    else:
        raise SystemExit("Invalid --seg_source. Use one of: seg1, seg2, single")

    align_model = build_align_model(args.align_ckpt, args.align_head_type, args.num_angle_bins, DEVICE)

    in_root = Path(args.images_dir)
    out_root = Path(args.out_dir)
    (out_root / "final").mkdir(parents=True, exist_ok=True)
    (out_root / "aligned_masks").mkdir(parents=True, exist_ok=True)
    (out_root / "aligned_images").mkdir(parents=True, exist_ok=True)  # NEW

    # CSV logging
    csv_file = None
    csv_writer = None
    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["relative_path", "angle_deg"])  # header

    img_paths = gather_images(in_root)
    if not img_paths:
        log("No images found. Check --images_dir and extensions.")
        if csv_file: csv_file.close()
        return

    # Progress bar (tqdm if available)
    iterator = img_paths
    if tqdm is not None:
        iterator = tqdm(img_paths, desc="Processing images", unit="img")

    processed, failed = 0, 0
    for p in iterator:
        try:
            # UPDATED unpack to also get rot_img
            final_img, angle, rot_mask_u8, rot_img = process_one_image(p, seg, align_model, args)

            # Save final image (aligned + masked)
            dst_final = ensure_final_path(out_root, in_root, p)
            dst_final.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_final), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

            # Save aligned (rotated) mask as single-channel PNG
            dst_mask = ensure_mask_path(out_root, in_root, p)
            dst_mask.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_mask), rot_mask_u8)

            # Save aligned original image (no mask applied)
            dst_aligned = ensure_aligned_img_path(out_root, in_root, p)
            dst_aligned.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_aligned), cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR))

            # CSV row
            if csv_writer is not None:
                rel = str(p.relative_to(in_root)).replace("\\", "/")
                csv_writer.writerow([rel, f"{angle:.6f}"])

            processed += 1
        except Exception as e:
            failed += 1
            log(f"Failed on {p}: {e}")
            if tqdm is None:
                print(f"Processed {processed + failed}/{len(img_paths)}")

    if csv_file:
        csv_file.close()

    log(f"Done. Processed={processed}, Failed={failed}, Output dir={out_root}")

if __name__ == "__main__":
    main()
