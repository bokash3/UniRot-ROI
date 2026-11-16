#!/usr/bin/env python3
# Comments are intentionally in English as requested.

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2

# -------------------------
# Project imports (must match your segmentation project)
# -------------------------
from datasets import SegmentationDataset
from models.unet_seg import UNetSeg


# -------------------------
# Utilities
# -------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_files(images_dir: str) -> List[str]:
    """Return file names (not full paths) sorted lexicographically."""
    p = Path(images_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    files = [x.name for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXTS]
    files.sort()
    if len(files) == 0:
        raise RuntimeError(f"No image files found in: {images_dir}")
    return files


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_checkpoint(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def extract_batch(batch: Any) -> Tuple[torch.Tensor, List[str]]:
    """
    Make the script robust to different dataset return formats.

    Expected (as in your eval script):
        (imgs, masks, names)

    Also supports:
        (imgs, names)
        {"image": imgs, "name": names}
        {"img": imgs, "name": names}
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            imgs, _, names = batch
            return imgs, list(names)
        if len(batch) == 2:
            imgs, names = batch
            return imgs, list(names)
        raise ValueError(f"Unsupported batch tuple length: {len(batch)}")
    if isinstance(batch, dict):
        if "image" in batch and "name" in batch:
            return batch["image"], list(batch["name"])
        if "img" in batch and "name" in batch:
            return batch["img"], list(batch["name"])
    raise ValueError(f"Unsupported batch type: {type(batch)}")


def save_mask_png(mask01: np.ndarray, out_path: str) -> None:
    """
    mask01: uint8 {0,1} or {0,255} or bool, shape (H,W)
    Saves 8-bit PNG mask with values {0,255}.
    """
    if mask01.dtype != np.uint8:
        mask01 = mask01.astype(np.uint8)
    if mask01.max() <= 1:
        mask255 = mask01 * 255
    else:
        mask255 = mask01
        # Normalize any non-binary outputs to 0/255 just in case
        mask255 = (mask255 > 0).astype(np.uint8) * 255
    ensure_dir(str(Path(out_path).parent))
    ok = cv2.imwrite(out_path, mask255)
    if not ok:
        raise RuntimeError(f"Failed to write: {out_path}")


@torch.no_grad()
def run_inference_and_save(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thr: float,
    out_dir: str,
    out_ext: str = ".png",
    save_prob: bool = False,
    prob_dir: Optional[str] = None,
) -> None:
    ensure_dir(out_dir)
    if save_prob:
        if prob_dir is None:
            prob_dir = str(Path(out_dir) / "probs")
        ensure_dir(prob_dir)

    for batch in tqdm(loader, desc="Generating masks"):
        imgs, names = extract_batch(batch)
        imgs = imgs.to(device, non_blocking=True)

        logits = model(imgs)
        probs = torch.sigmoid(logits)

        # Normalize shapes: (B,1,H,W) -> (B,H,W)
        if probs.dim() == 4 and probs.size(1) == 1:
            probs_hw = probs[:, 0, :, :]
        elif probs.dim() == 3:
            probs_hw = probs
        else:
            raise ValueError(f"Unexpected probs shape: {tuple(probs.shape)}")

        pred = (probs_hw >= thr).to(torch.uint8).cpu().numpy()  # (B,H,W)

        if save_prob:
            prob_np = probs_hw.detach().cpu().numpy()  # float32, (B,H,W)

        for i, name in enumerate(names):
            stem = Path(name).stem
            out_path = str(Path(out_dir) / f"{stem}{out_ext}")
            save_mask_png(pred[i], out_path)

            if save_prob:
                # Save probability map as 16-bit PNG for better precision
                # prob in [0,1] -> [0,65535]
                prob16 = np.clip(prob_np[i], 0.0, 1.0)
                prob16 = (prob16 * 65535.0).round().astype(np.uint16)
                prob_path = str(Path(prob_dir) / f"{stem}{out_ext}")
                ensure_dir(str(Path(prob_path).parent))
                ok = cv2.imwrite(prob_path, prob16)
                if not ok:
                    raise RuntimeError(f"Failed to write: {prob_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate segmentation masks for all images using a trained UNetSeg checkpoint."
    )

    # Core inputs
    p.add_argument("--images_dir", type=str, required=True, help="Folder with input images (e.g., dataset/images).")
    p.add_argument("--out_dir", type=str, default="generated_masks", help="Output folder to write predicted masks.")
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to trained checkpoint (e.g., seg_resnet34_s42_best.pt).")
    p.add_argument("--encoder", type=str, required=True, help="Encoder name used in training (e.g., resnet34, mobilenet_v2).")

    # Inference options
    p.add_argument("--thr", type=float, default=0.5, help="Sigmoid threshold to binarize predictions.")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device selection.")
    p.add_argument("--out_ext", type=str, default=".png", help="Extension for output masks (e.g., .png).")

    # Optional: save probability maps
    p.add_argument("--save_prob", action="store_true", help="Also save probability maps (16-bit PNG) to a 'probs' subfolder.")
    p.add_argument("--prob_dir", type=str, default=None, help="Optional directory for saving probability maps.")

    # Optional: subset control
    p.add_argument("--file_list", type=str, default=None, help="Optional text file with one image filename per line (subset).")

    # Optional: dataset compatibility
    p.add_argument("--masks_dir", type=str, default=None, help="Optional GT masks dir (not needed for inference, but keeps dataset API compatible).")

    args = p.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Collect files
    if args.file_list is not None:
        fl = Path(args.file_list)
        if not fl.is_file():
            raise FileNotFoundError(f"file_list not found: {args.file_list}")
        files = [ln.strip() for ln in fl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(files) == 0:
            raise RuntimeError("file_list is empty.")
    else:
        files = list_image_files(args.images_dir)

    # Build dataset exactly like evaluation pipeline (but masks_dir can be None)
    ds = SegmentationDataset(args.images_dir, args.masks_dir, files, is_train=False)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Load model and checkpoint (same pattern as evaluation code)
    model = UNetSeg(encoder_name=args.encoder).to(device)
    ckpt = load_checkpoint(args.ckpt_path, device=device)

    # Support both {"model": state_dict} and plain state_dict checkpoints
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    ensure_dir(args.out_dir)

    # Save a small run metadata file for reproducibility
    meta = {
        "images_dir": args.images_dir,
        "out_dir": args.out_dir,
        "ckpt_path": args.ckpt_path,
        "encoder": args.encoder,
        "thr": float(args.thr),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "device": str(device),
        "n_files": int(len(files)),
        "save_prob": bool(args.save_prob),
        "prob_dir": args.prob_dir,
    }
    with open(Path(args.out_dir) / "generated_masks_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    run_inference_and_save(
        model=model,
        loader=loader,
        device=device,
        thr=float(args.thr),
        out_dir=args.out_dir,
        out_ext=args.out_ext,
        save_prob=bool(args.save_prob),
        prob_dir=args.prob_dir,
    )

    print(f"Done. Saved masks to: {args.out_dir}")


if __name__ == "__main__":
    main()
