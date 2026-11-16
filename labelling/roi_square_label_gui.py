#!/usr/bin/env python3
"""
Interactive Palm ROI Square Labeling GUI
----------------------------------------
- Iterates over hand mask images in a given folder.
- For each mask:
    * Shows the mask in a window.
    * Draws a square ROI overlay (center + size).
    * Three sliders:
        - cx (normalized 0..1)
        - cy (normalized 0..1)
        - size (normalized 0..1, side as fraction of min(H, W))
    * When sliders change, the square is updated live.
    * "Save & Next" button:
        - Saves the current (cx, cy, size) to a JSON file
        - Moves to the next image.

- JSON format example:
    {
      "001_l_460_01.png": {"cx": 0.53, "cy": 0.61, "size": 0.42},
      "001_l_460_02.png": {"cx": 0.51, "cy": 0.60, "size": 0.41}
    }

All comments are in English as requested.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# --- NEW: imports for ROI-Net model ---
import torch
import torch.nn as nn
from torchvision import models, transforms


# -----------------------------
# ROI-Net model definition
# -----------------------------
class PalmROINet(nn.Module):
    """
    Lightweight ROI regression network:
    - Backbone: MobileNetV2 (1-channel input).
    - Output: 3D vector in [0,1]: (cx, cy, size).
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        # We do not use pretrained ImageNet weights here, because the checkpoint
        # already contains trained weights for this task.
        self.backbone = models.mobilenet_v2(weights=None)

        # Modify first conv layer to accept 1-channel input instead of 3.
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
        out = self.backbone(x)
        out = self.sigmoid(out)
        return out


# -----------------------------
# Optional: geometric initializer
# -----------------------------
def compute_square_roi_from_mask(mask_u8: np.ndarray,
                                 margin_frac: float = 0.05
                                 ) -> Optional[Tuple[float, float, float]]:
    """
    Compute an initial square ROI inside the hand mask using distance transform.

    Args:
        mask_u8: uint8 mask (H×W), >0 = hand region.
        margin_frac: fractional margin to shrink the maximum radius
                     to stay away from the boundary a bit.

    Returns:
        (cx_norm, cy_norm, size_norm) in [0,1], or None if it fails.
    """
    m = (mask_u8 > 0).astype(np.uint8)
    H, W = m.shape[:2]

    if m.sum() == 0:
        return None

    # Keep only the largest connected component
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    largest = max(cnts, key=cv2.contourArea)
    m_largest = np.zeros_like(m)
    cv2.drawContours(m_largest, [largest], -1, 1, thickness=-1)

    # Distance transform on the largest component
    dist = cv2.distanceTransform(m_largest, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return None

    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
    r_max = float(dist[cy, cx])
    r_eff = r_max * (1.0 - margin_frac)

    low = 0.0
    high = 2.0 * r_eff  # side cannot be larger than the diameter

    def square_fits(side: float) -> bool:
        """Check if a square of given side fits completely inside the mask."""
        half = int(np.floor(side / 2.0))
        if half <= 0:
            return False
        x0 = cx - half
        x1 = cx + half
        y0 = cy - half
        y1 = cy + half

        if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
            return False

        roi = m_largest[y0:y1 + 1, x0:x1 + 1]
        expected_area = (2 * half + 1) * (2 * half + 1)
        if roi.size != expected_area:
            return False

        return np.all(roi == 1)

    # Binary search for the largest side that fits
    for _ in range(20):
        mid = 0.5 * (low + high)
        if square_fits(mid):
            low = mid
        else:
            high = mid

    side_px = int(np.floor(low))
    if side_px <= 0:
        return None

    cx_norm = cx / float(W)
    cy_norm = cy / float(H)
    size_norm = side_px / float(min(H, W))

    return cx_norm, cy_norm, size_norm


# -----------------------------
# Main GUI App
# -----------------------------
class ROISquareLabelGUI:
    def __init__(self,
                 masks_dir: str,
                 json_path: str,
                 roi_ckpt: Optional[str] = None,
                 roi_img_size: int = 256,
                 exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
                 max_display_side: int = 640):
        self.masks_dir = Path(masks_dir)
        self.json_path = Path(json_path)
        self.exts = tuple(e.lower() for e in exts)
        self.max_display_side = max_display_side

        # Collect mask paths
        self.mask_paths: List[Path] = sorted(
            [p for p in self.masks_dir.rglob("*") if p.suffix.lower() in self.exts]
        )
        if not self.mask_paths:
            raise RuntimeError(f"No mask images found in: {self.masks_dir}")

        # Load existing labels if JSON exists
        self.labels: Dict[str, Dict[str, float]] = {}
        if self.json_path.is_file():
            try:
                with self.json_path.open("r", encoding="utf-8") as f:
                    self.labels = json.load(f)
                print(f"[INFO] Loaded existing labels from {self.json_path}")
            except Exception as e:
                print(f"[WARN] Failed to load existing JSON: {e}")
                self.labels = {}
        else:
            self.labels = {}

        # -------------------------------------
        # Resume from first unlabeled image
        # -------------------------------------
        unlabeled_indices = [
            i for i, p in enumerate(self.mask_paths)
            if p.name not in self.labels
        ]
        if unlabeled_indices:
            self.idx = unlabeled_indices[0]
            print(f"[INFO] Resuming from first unlabeled image index: {self.idx}")
        else:
            # If all images are labeled, start from the last one
            self.idx = max(0, len(self.mask_paths) - 1)
            print("[INFO] All images already labeled. Starting at the last image.")

        self.current_mask: Optional[np.ndarray] = None
        self.current_display_img: Optional[np.ndarray] = None
        self.display_scale: float = 1.0  # scale from original to display size

        # Keyboard step sizes
        self.key_step_pos: float = 0.01   # step for cx, cy per arrow press
        self.key_step_size: float = 0.01  # step for size per '+' / '=' press
        # --- NEW: ROI-Net model setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.roi_img_size = int(roi_img_size)
        self.roi_model: Optional[PalmROINet]

        if roi_ckpt is not None:
            ckpt_path = Path(roi_ckpt)
            if not ckpt_path.is_file():
                print(f"[WARN] ROI checkpoint not found: {ckpt_path}. "
                      f"Geometric initializer will be used instead.")
                self.roi_model = None
            else:
                try:
                    self.roi_model = PalmROINet(pretrained=False).to(self.device)
                    ckpt = torch.load(ckpt_path, map_location=self.device)
                    if "model_state_dict" in ckpt:
                        self.roi_model.load_state_dict(ckpt["model_state_dict"])
                    else:
                        self.roi_model.load_state_dict(ckpt)
                    self.roi_model.eval()
                    if self.roi_model is not None:
                        # Use the same preprocessing as training / hard_vis script
                        self.roi_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((self.roi_img_size, self.roi_img_size),
                                            interpolation=Image.NEAREST),
                            transforms.ToTensor(),  # (1,H,W) in [0,1]
                        ])
                    else:
                        self.roi_transform = None
                    print(f"[INFO] Loaded ROI-Net from: {ckpt_path}")
                except Exception as e:
                    print(f"[WARN] Failed to load ROI-Net checkpoint: {e}")
                    self.roi_model = None
        else:
            print("[INFO] No ROI checkpoint provided. Using only geometric initializer.")
            self.roi_model = None

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("Palm ROI Square Labeling GUI")

        # State variables (normalized 0..1)
        self.cx_var = tk.DoubleVar(value=0.5)
        self.cy_var = tk.DoubleVar(value=0.5)
        self.size_var = tk.DoubleVar(value=0.3)

        self._build_ui()
        self._load_current_image()
        self._update_image()

        # Force window focus to receive key events
        self.root.focus_force()
        self.root.mainloop()

    # -------------------------
    # UI building
    # -------------------------
    def _build_ui(self):
        # Top bar: path + navigation
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        self.path_label = ttk.Label(top_frame, text="")
        self.path_label.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(top_frame, text="◄ Prev", command=self.on_prev).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top_frame, text="Next ►", command=self.on_next).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top_frame, text="Save & Next", command=self.on_save_and_next).pack(side=tk.RIGHT, padx=4)

        # Main image area
        img_frame = ttk.Frame(self.root, relief=tk.GROOVE, borderwidth=1)
        img_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.img_label = tk.Label(img_frame)
        self.img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Sliders area
        sliders = ttk.Frame(self.root)
        sliders.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        # cx slider
        ttk.Label(sliders, text="cx (0..1)").grid(row=0, column=0, sticky="w")
        tk.Scale(
            sliders,
            from_=0.0,
            to=1.0,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            variable=self.cx_var,
            command=self.on_slider_change,
            length=250,
        ).grid(row=0, column=1, padx=4, pady=2)

        # cy slider
        ttk.Label(sliders, text="cy (0..1)").grid(row=1, column=0, sticky="w")
        tk.Scale(
            sliders,
            from_=0.0,
            to=1.0,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            variable=self.cy_var,
            command=self.on_slider_change,
            length=250,
        ).grid(row=1, column=1, padx=4, pady=2)

        # size slider
        ttk.Label(sliders, text="size (0..1, side/min(H,W))").grid(row=2, column=0, sticky="w")
        tk.Scale(
            sliders,
            from_=0.05,
            to=1.0,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            variable=self.size_var,
            command=self.on_slider_change,
            length=250,
        ).grid(row=2, column=1, padx=4, pady=2)

        # Info label
        self.info_var = tk.StringVar(value="")
        # -------------------------
        # Keyboard bindings
        # -------------------------
        # Arrow keys to move the square
        self.root.bind("<Left>", self._on_key_left)
        self.root.bind("<Right>", self._on_key_right)
        self.root.bind("<Up>", self._on_key_up)
        self.root.bind("<Down>", self._on_key_down)

        # Enter to save and go to next image
        self.root.bind("<Return>", self._on_key_enter)
        self.root.bind("<KP_Enter>", self._on_key_enter)  # Numpad Enter

        # '=' or '+' to increase the square size
        self.root.bind("=", self._on_key_increase_size)
        self.root.bind("+", self._on_key_increase_size)
        self.root.bind("-", self._on_key_decrease_size)
        ttk.Label(sliders, textvariable=self.info_var).grid(row=0, column=2, rowspan=3, sticky="w", padx=8)

    # -------------------------
    # Navigation handlers
    # -------------------------
    def on_prev(self):
        if self.idx > 0:
            self.idx -= 1
        else:
            self.idx = 0
        self._load_current_image()
        self._update_image()

    def on_next(self):
        if self.idx < len(self.mask_paths) - 1:
            self.idx += 1
        else:
            self.idx = len(self.mask_paths) - 1
        self._load_current_image()
        self._update_image()

    def on_save_and_next(self):
        self._save_current_label()
        if self.idx < len(self.mask_paths) - 1:
            self.idx += 1
            self._load_current_image()
            self._update_image()
        else:
            messagebox.showinfo("Done", "You have reached the last image.")
            self._update_image()

    def on_slider_change(self, _event=None):
        self._update_image()
    # -------------------------
    # Keyboard handlers
    # -------------------------
    def _adjust_cx(self, delta: float):
        """Adjust cx by a given delta in normalized coordinates."""
        new_cx = float(self.cx_var.get()) + delta
        new_cx = max(0.0, min(1.0, new_cx))
        self.cx_var.set(new_cx)
        self._update_image()

    def _adjust_cy(self, delta: float):
        """Adjust cy by a given delta in normalized coordinates."""
        new_cy = float(self.cy_var.get()) + delta
        new_cy = max(0.0, min(1.0, new_cy))
        self.cy_var.set(new_cy)
        self._update_image()

    def _adjust_size(self, delta: float):
        """Adjust size by a given delta in normalized coordinates."""
        new_size = float(self.size_var.get()) + delta
        # Enforce the same range as the slider (0.05..1.0)
        new_size = max(0.05, min(1.0, new_size))
        self.size_var.set(new_size)
        self._update_image()

    def _on_key_left(self, event):
        # Move ROI to the left (decrease cx)
        self._adjust_cx(-self.key_step_pos)
        return "break"

    def _on_key_right(self, event):
        # Move ROI to the right (increase cx)
        self._adjust_cx(self.key_step_pos)
        return "break"

    def _on_key_up(self, event):
        # Move ROI up (decrease cy, because y increases downward)
        self._adjust_cy(-self.key_step_pos)
        return "break"

    def _on_key_down(self, event):
        # Move ROI down (increase cy)
        self._adjust_cy(self.key_step_pos)
        return "break"

    def _on_key_increase_size(self, event):
        # Increase the size of the square ROI
        self._adjust_size(self.key_step_size)
        return "break"

    def _on_key_enter(self, event):
        # Save current label and go to next image
        self.on_save_and_next()
        return "break"
    def _on_key_decrease_size(self, event):
        # Decrease the size of the square ROI (small step)
        self._adjust_size(-self.key_step_size)
        return "break"
    # -------------------------
    # ROI-Net prediction helper
    # -------------------------
    def _predict_roi_with_model(self, mask_u8: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Run the trained ROI-Net on the given mask and return (cx, cy, size) in [0,1].
        Returns None if the model is not available or an error occurs.
        """
        if self.roi_model is None or self.roi_transform is None:
            return None

        try:
            # Use the SAME preprocessing pipeline as training / hard_vis
            # mask_u8: (H,W) uint8 in [0,255]
            mask_tensor = self.roi_transform(mask_u8)      # (1,H,W) in [0,1]
            x = mask_tensor.unsqueeze(0).to(self.device)   # (1,1,H,W)

            with torch.no_grad():
                pred = self.roi_model(x).cpu().numpy()[0]  # (3,)

            cx, cy, size = float(pred[0]), float(pred[1]), float(pred[2])

            # Clamp just in case
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            size = max(0.05, min(1.0, size))

            return cx, cy, size
        except Exception as e:
            print(f"[WARN] ROI-Net prediction failed: {e}")
            return None


    # -------------------------
    # Data / image loading
    # -------------------------
    def _load_current_image(self):
        """Load mask for current index and initialize sliders."""
        p = self.mask_paths[self.idx]
        self.path_label.config(
            text=f"{self.idx+1}/{len(self.mask_paths)} — {p.name}"
        )

        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            messagebox.showerror("Error", f"Failed to read mask: {p}")
            self.current_mask = None
            return

        self.current_mask = mask

        # Prepare display version (maybe scaled)
        self.current_display_img, self.display_scale = self._prepare_display_image(mask)

        # Initialize sliders: if label exists, use it; else try ROI-Net, then geometric init
        key = p.name
        if key in self.labels:
            lbl = self.labels[key]
            cx = float(lbl.get("cx", 0.5))
            cy = float(lbl.get("cy", 0.5))
            size = float(lbl.get("size", 0.3))
            self.cx_var.set(cx)
            self.cy_var.set(cy)
            self.size_var.set(size)
        else:
            # 1) Try prediction from trained ROI-Net
            init = self._predict_roi_with_model(mask)
            if init is not None:
                cx, cy, size = init
                self.cx_var.set(cx)
                self.cy_var.set(cy)
                self.size_var.set(size)
            else:
                # 2) Fallback: geometric initializer
                geom = compute_square_roi_from_mask(mask)
                if geom is not None:
                    cx, cy, size = geom
                    self.cx_var.set(cx)
                    self.cy_var.set(cy)
                    self.size_var.set(size)
                else:
                    # 3) Final fallback: center & medium size
                    self.cx_var.set(0.5)
                    self.cy_var.set(0.5)
                    self.size_var.set(0.3)

    def _prepare_display_image(self, mask_u8: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Prepare an RGB image for display from the mask (grayscale),
        WITHOUT any resizing. We keep the original resolution.
        """
        # Convert mask to RGB (white hand on black background)
        m_norm = (mask_u8.astype(np.float32) / 255.0)
        rgb = (m_norm * 255.0).astype(np.uint8)
        rgb3 = np.stack([rgb, rgb, rgb], axis=-1)  # H×W×3

        scale = 1.0  # no resizing, display == original size
        return rgb3, scale


    # -------------------------
    # Drawing / preview
    # -------------------------
    def _update_image(self):
        """Redraw the current image with the square ROI overlay."""
        if self.current_display_img is None:
            return

        disp_img = self.current_display_img.copy()
        H, W = disp_img.shape[:2]

        cx = float(self.cx_var.get())
        cy = float(self.cy_var.get())
        size = float(self.size_var.get())

        # Clamp values to [0,1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        size = max(0.0, min(1.0, size))

        # Convert normalized params to display coordinates
        min_side = min(H, W)
        side_disp = size * float(min_side)

        half = side_disp / 2.0
        cx_px = cx * float(W)
        cy_px = cy * float(H)

        x0 = int(round(cx_px - half))
        x1 = int(round(cx_px + half))
        y0 = int(round(cy_px - half))
        y1 = int(round(cy_px + half))

        # Clamp to image borders (for display only)
        x0 = max(0, min(W - 1, x0))
        x1 = max(0, min(W - 1, x1))
        y0 = max(0, min(H - 1, y0))
        y1 = max(0, min(H - 1, y1))

        # Draw square (in red) on the preview image
        color = (255, 0, 0)  # BGR red
        cv2.rectangle(disp_img, (x0, y0), (x1, y1), color, thickness=2)

        # Convert to Tk image
        pil_img = Image.fromarray(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB))
        tk_img = ImageTk.PhotoImage(pil_img)

        # Keep a reference, otherwise Tk will garbage-collect it
        self._last_tk_image = tk_img
        self.img_label.config(image=tk_img)

        # Update info label (also show pixel values for debugging)
        info_text = (
            f"cx={cx:.3f}, cy={cy:.3f}, size={size:.3f}  "
            f"| disp W×H={W}×{H}"
        )
        self.info_var.set(info_text)

    # -------------------------
    # Saving labels
    # -------------------------
    def _save_current_label(self):
        """Save the current ROI parameters for the current image into JSON."""
        if not self.mask_paths:
            return

        p = self.mask_paths[self.idx]
        key = p.name

        cx = float(self.cx_var.get())
        cy = float(self.cy_var.get())
        size = float(self.size_var.get())

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        size = max(0.0, min(1.0, size))

        self.labels[key] = {
            "cx": float(cx),
            "cy": float(cy),
            "size": float(size),
        }

        try:
            with self.json_path.open("w", encoding="utf-8") as f:
                json.dump(self.labels, f, indent=2)
            print(f"[INFO] Saved label for {key} to {self.json_path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))


# -----------------------------
# Entry point
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser("Palm ROI Square Labeling GUI")
    parser.add_argument(
        "--masks_dir",
        type=str,
        default="images50to60",
        help="Folder containing hand masks (binary or grayscale).",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="palm_roi_label50t060.json",
        help="Path to the JSON file where labels will be stored/updated.",
    )
    parser.add_argument(
        "--roi_ckpt",
        type=str,
        default="roi_net_runs_round3/roi_net_best.pth",
        help="Path to trained PalmROINet checkpoint (roi_net_best.pth).",
    )
    parser.add_argument(
        "--roi_img_size",
        type=int,
        default=256,
        help="Input size used for ROI-Net training (H=W).",
    )
    args = parser.parse_args()

    ROISquareLabelGUI(
        masks_dir=args.masks_dir,
        json_path=args.out_json,
        roi_ckpt=args.roi_ckpt,
        roi_img_size=args.roi_img_size,
    )


if __name__ == "__main__":
    main()
