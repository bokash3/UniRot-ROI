#!/usr/bin/env python3
"""
Mask + Align — Interactive Viewer (Tkinter GUI)
MATCHES TRAIN/EVAL PREPROCESSING (LETTERBOX + IMAGENET NORM)
----------------------------------------------------------------
- Uses ONLY ONE selected segmentation model (no combining).
- Applies the SAME preprocessing used in your dataset pipeline:
  LongestMaxSize(max(H,W)) + PadIfNeeded(H,W) + Normalize(ImageNet).
- The predicted mask is taken DIRECTLY from the model output
  (choose raw probabilities or binary threshold) — no extra post-proc.
- Projects the letterboxed mask back to the ORIGINAL image size before rotation,
  so the aligned & masked outputs are at the original resolution.

Run examples
------------
# seg1 (ResNet34), RAW probability mask (no threshold):
python mask_align_gui_letterbox.py \
  --images_dir /path/to/input_images \
  --seg_source seg1 \
  --seg1_ckpt /path/to/seg_resnet34.ckpt \
  --align_ckpt /path/to/rot_resnet18.ckpt \
  --align_head_type fusion --num_angle_bins 36 --angle_space 0-360 \
  --align_input_hw 576x768 --rotate_negate \
  --seg_input_hw 576x768 \
  --seg_mask_mode prob

# seg2 (MobileNetV2), BINARY mask @ 0.5 (no morphology):
python mask_align_gui_letterbox.py \
  --images_dir /path/to/input_images \
  --seg_source seg2 \
  --seg2_ckpt /path/to/seg_mobilenetv2.ckpt \
  --align_ckpt /path/to/rot_resnet18.ckpt \
  --align_head_type fusion --num_angle_bins 36 --angle_space 0-360 \
  --align_input_hw 576x768 --rotate_negate \
  --seg_input_hw 576x768 \
  --seg_mask_mode binary --seg_thr 0.5

"""

import argparse
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

# GUI
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg: str):
    print(f"[MaskAlignGUI] {msg}")

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
# Helpers
# -----------------------------

def read_image(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_tensor(img: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    x = img.astype(np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))  # C,H,W
    return torch.from_numpy(x).unsqueeze(0)


def to_mask_tensor(mask: np.ndarray) -> torch.Tensor:
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
    # make the longest side match max(th, tw)
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
    if mask.dtype == np.uint8:
        return (mask.astype(np.float32) / 255.0)
    m = mask.astype(np.float32)
    return np.clip(m, 0.0, 1.0)


def apply_mask_rgb(img_rgb: np.ndarray, mask_any, bg_value=(0, 0, 0)) -> np.ndarray:
    m = _to_float01_mask(mask_any)
    m3 = np.dstack([m, m, m])
    img_f = img_rgb.astype(np.float32)
    if any(bg_value):
        bg = np.full_like(img_f, bg_value, dtype=np.float32)
        out = img_f * m3 + bg * (1.0 - m3)
    else:
        out = img_f * m3
    return np.clip(out, 0, 255).astype(img_rgb.dtype)


def decode_angle_from_model(output, head_type: str, num_bins: int, angle_space: str) -> float:
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

def _load_seg_unet(ckpt_path: str, encoders_try, device) -> nn.Module:
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
# Core processing
# -----------------------------

def process_one_image(img_path: Path, seg: nn.Module, align_model: nn.Module, args):
    # Read original
    img = read_image(img_path)  # RGB

    # LETTERBOX to training size
    seg_in = img
    scale = 1.0
    pad_xy = (0, 0)
    new_hw = (img.shape[0], img.shape[1])
    if args.seg_input_hw is not None:
        seg_in, scale, pad_xy, new_hw = resize_letterbox(img, args.seg_input_hw)

    # Normalize like training
    xt = to_tensor(seg_in, mean=args.seg_mean, std=args.seg_std).to(DEVICE)

    # Segmentation forward
    with torch.no_grad():
        logits = seg(xt)

    # Logits -> prob or binary (on letterboxed size)
    l = logits.detach().float().cpu().numpy()
    p = sigmoid_numpy(l)[0, 0]
    if args.seg_mask_mode == 'prob':
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
    interp_mask = cv2.INTER_NEAREST if mask_orig.dtype == np.uint8 else cv2.INTER_LINEAR
    rot_mask = cv2.warpAffine(mask_orig, cv2.getRotationMatrix2D((img.shape[1]/2.0, img.shape[0]/2.0), apply_angle, 1.0),
                              (img.shape[1], img.shape[0]), flags=interp_mask,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rot_img = cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]/2.0, img.shape[0]/2.0), apply_angle, 1.0),
                             (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=tuple(int(x*255) for x in args.bg_value))

    # Apply mask to rotated image
    masked_img = apply_mask_rgb(rot_img, rot_mask, bg_value=tuple(int(x*255) for x in args.bg_value))

    # Make a displayable mask (uint8 0..255)
    disp_mask = (np.clip(mask_orig.astype(np.float32), 0, 1) * 255.0).astype(np.uint8) if args.seg_mask_mode=='prob' else mask_orig

    return {"original": img, "mask": disp_mask, "angle": float(angle_deg), "rotated": rot_img, "masked": masked_img}

# -----------------------------
# Tkinter GUI
# -----------------------------
class MaskAlignApp:
    def __init__(self, master, args):
        self.master = master
        self.args = args
        self.master.title("Mask + Align Viewer (Letterbox Matched)")

        # Load models
        log(f"Device: {DEVICE}")
        self.seg = self._load_selected_seg(args)
        self.align_model = build_align_model(args.align_ckpt, args.align_head_type, args.num_angle_bins, DEVICE)

        # Gather images
        self.img_paths = self._gather_images(Path(args.images_dir))
        if not self.img_paths:
            raise SystemExit("No images found. Check --images_dir and extensions.")
        self.idx = 0

        # Build UI
        self._build_layout()
        self.master.bind('n', lambda e: self.on_next())
        self.render_current()

    def _load_selected_seg(self, args) -> nn.Module:
        if args.seg_source == 'seg1':
            if not args.seg1_ckpt:
                raise SystemExit("--seg1_ckpt is required when --seg_source seg1")
            return build_seg_model(args.seg1_ckpt, args.seg1_encoder, DEVICE)
        elif args.seg_source == 'seg2':
            if not args.seg2_ckpt:
                raise SystemExit("--seg2_ckpt is required when --seg_source seg2")
            return build_seg_model(args.seg2_ckpt, args.seg2_encoder, DEVICE)
        elif args.seg_source == 'single':
            if not args.seg_ckpt:
                raise SystemExit("--seg_ckpt is required when --seg_source single")
            return build_seg_model(args.seg_ckpt, args.seg_encoder, DEVICE)
        else:
            raise SystemExit("Invalid --seg_source. Use one of: seg1, seg2, single")

    def _build_layout(self):
        ctrl = ttk.Frame(self.master); ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
        self.path_label = ttk.Label(ctrl, text=""); self.path_label.pack(side=tk.LEFT, padx=(0,8))
        ttk.Button(ctrl, text="Next ▶", command=self.on_next).pack(side=tk.RIGHT)

        angle_fr = ttk.Frame(self.master); angle_fr.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0,8))
        ttk.Label(angle_fr, text="Predicted Angle (deg):").pack(side=tk.LEFT)
        self.angle_var = tk.StringVar(value="-")
        self.angle_entry = ttk.Entry(angle_fr, textvariable=self.angle_var, width=12)
        self.angle_entry.pack(side=tk.LEFT, padx=(6,0))

        panels = ttk.Frame(self.master); panels.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.mask_canvas = self._create_panel(panels, "Predicted Mask")
        self.aligned_canvas = self._create_panel(panels, "Aligned Image")
        self.masked_canvas = self._create_panel(panels, "Masked Image")

    def _create_panel(self, parent, title: str):
        frame = ttk.Frame(parent, borderwidth=1, relief=tk.GROOVE)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        ttk.Label(frame, text=title, font=("TkDefaultFont",10,"bold")).pack(side=tk.TOP, pady=4)
        canvas = tk.Label(frame); canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return canvas

    def _gather_images(self, root: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
        return [p for p in root.rglob("*") if p.suffix.lower() in exts]

    def _np_to_tk(self, arr: np.ndarray, is_mask: bool=False, max_side: int=512) -> ImageTk.PhotoImage:
        rgb = np.stack([arr, arr, arr], axis=-1) if is_mask and arr.ndim==2 else arr
        h, w = rgb.shape[:2]
        scale = min(max_side / max(h, w), 1.0)
        nh, nw = int(h*scale), int(w*scale)
        pil_im = Image.fromarray(rgb).resize((nw, nh), Image.NEAREST if is_mask else Image.BILINEAR)
        return ImageTk.PhotoImage(pil_im)

    def render_current(self):
        path = self.img_paths[self.idx]
        self.path_label.config(text=f"{self.idx+1}/{len(self.img_paths)} — {path.name}")
        try:
            result = process_one_image(path, self.seg, self.align_model, self.args)
        except Exception as e:
            log(f"Failed on {path}: {e}")
            self.angle_var.set("error")
            self.mask_canvas.config(image=""); self.aligned_canvas.config(image=""); self.masked_canvas.config(image="")
            return
        self.angle_var.set(f"{result['angle']:.3f}")
        self.mask_imgtk = self._np_to_tk(result['mask'], is_mask=True)
        self.aligned_imgtk = self._np_to_tk(result['rotated'])
        self.masked_imgtk = self._np_to_tk(result['masked'])
        self.mask_canvas.config(image=self.mask_imgtk)
        self.aligned_canvas.config(image=self.aligned_imgtk)
        self.masked_canvas.config(image=self.masked_imgtk)

    def on_next(self):
        self.idx = (self.idx + 1) % len(self.img_paths)
        self.render_current()

# -----------------------------
# CLI
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
    parser = argparse.ArgumentParser(description="Mask + Align Interactive Viewer (Letterbox Matched)")

    # Paths
    parser.add_argument("--images_dir", type=str, required=True, help="Input images folder (scanned recursively).")

    # Segmentation selection
    parser.add_argument("--seg_source", type=str, default="seg1", choices=["seg1", "seg2", "single"],
                        help="Which segmentation model to use: seg1 | seg2 | single (explicit ckpt)")
    parser.add_argument("--seg1_ckpt", type=str, default=None, help="Seg model checkpoint for seg1")
    parser.add_argument("--seg2_ckpt", type=str, default=None, help="Seg model checkpoint for seg2")
    parser.add_argument("--seg_ckpt", type=str, default=None, help="Seg model checkpoint when --seg_source single")
    parser.add_argument("--seg1_encoder", type=str, default=None, help="Optional encoder for seg1 (e.g., resnet34)")
    parser.add_argument("--seg2_encoder", type=str, default=None, help="Optional encoder for seg2")
    parser.add_argument("--seg_encoder", type=str, default=None, help="Optional encoder when --seg_source single")

    # NOTE: match dataset HxW and normalization defaults
    parser.add_argument("--seg_input_hw", type=parse_hw, default=(576, 768), help="Seg input size HxW (letterboxed)")
    parser.add_argument("--seg_mean", type=parse_mean_std, default=(0.485, 0.456, 0.406), help="Normalization mean (ImageNet)")
    parser.add_argument("--seg_std", type=parse_mean_std, default=(0.229, 0.224, 0.225), help="Normalization std (ImageNet)")

    # Mask mode selection (no extra processing beyond optional threshold)
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

    # Output formatting for masked areas
    parser.add_argument("--bg_value", type=parse_mean_std, default=(0.0, 0.0, 0.0), help="Background fill RGB in [0,1] when applying mask")

    return parser


def main():
    parser = build_argparser(); args = parser.parse_args()
    root = tk.Tk(); app = MaskAlignApp(root, args); root.mainloop()


if __name__ == "__main__":
    main()
