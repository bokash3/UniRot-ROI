"""
train_resnet18_matcher.py
=========================

Train a per-seed ResNet-18 palmprint matcher on CASIA-MS 850 nm crops.

Pipeline
--------
1. Load <split_dir>/train_labels.json and <split_dir>/val_labels.json.
2. Combine them (170 identities total) and filter to **850 nm band only**.
   Test identities (in <split_dir>/test_labels.json) are disjoint by design,
   so filtering to train+val gives a clean closed-set training pool with no
   identity leakage into evaluation.
3. Crop each image to a 128x128 ROI using the GT cx/cy/size, resize to
   224x224, ImageNet-normalize.
4. Fine-tune ImageNet-pretrained ResNet-18 with plain softmax cross-entropy
   over 170 classes for N epochs, AdamW + cosine LR schedule.
5. Save the backbone (penultimate-feature extractor) for the matching script.

Why band filtering at training time?
------------------------------------
We evaluate matching on 850 nm only (Protocol A). Training the matcher on
just 850 nm avoids forcing it to be band-invariant, which would waste model
capacity on a robustness property we will never test. With ~1020 training
crops per seed this matters -- every bit of capacity counts.

Usage
-----
    python train_resnet18_matcher.py \\
        --split_dir   dataset_palm_locator_seed42 \\
        --images_dir  data/images \\
        --band        850 \\
        --out_ckpt    checkpoints/resnet18_seed42.pt \\
        --epochs      50 \\
        --augment     true \\
        --device      cuda

Repeat for seeds 2025 and 1337 with --split_dir dataset_palm_locator_seed{S}
and --out_ckpt checkpoints/resnet18_seed{S}.pt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Filename parsing -- CASIA-MS layout: <subject>_<hand>_<band>_<sample>.<ext>
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<hand>[lr])_(?P<band>\d+|white|WHT)_(?P<sample>\d+)\.[A-Za-z]+$"
)


def parse_casia_filename(fname: str) -> dict[str, Any] | None:
    m = _FNAME_RE.match(fname)
    if m is None:
        return None
    return {
        "subject": m.group("subject"),
        "hand": m.group("hand"),
        "band": m.group("band"),  # "460", "630", "700", "850", "940", "white"
        "sample": int(m.group("sample")),  # 1..6   (1..3 = session 1, 4..6 = session 2)
        "identity": f"{m.group('subject')}_{m.group('hand')}",  # e.g. "004_l"
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    fname: str
    identity: str
    cx: float
    cy: float
    size: float


def load_labels(json_path: Path, band: str) -> list[Sample]:
    """Load ROI labels from either of two supported JSON schemas:

    Wrapped (used by test manifests):
        { "image_root": ..., "roi_type": "original_normalized",
          "entries": { "FNAME": {"cx":..., "cy":..., "size":...}, ... } }

    Flat (used by train_labels.json / val_labels.json):
        { "FNAME": {"cx":..., "cy":..., "size":...}, ... }

    Both are read into the same Sample list, filtered to the chosen band.
    The roi_type is assumed 'original_normalized' for the flat schema.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Top-level of {json_path} is not a JSON object.")

    # Detect schema.
    if "entries" in data:
        # Wrapped schema (test-manifest style).
        roi_type = data.get("roi_type", "original_normalized")
        if roi_type != "original_normalized":
            raise NotImplementedError(
                f"Unsupported roi_type {roi_type!r} in {json_path}"
            )
        items = data["entries"]
    else:
        # Flat schema (train_labels.json / val_labels.json).
        # Reject any non-filename top-level keys to be safe.
        items = {
            k: v for k, v in data.items()
            if isinstance(v, dict) and "cx" in v and "cy" in v and "size" in v
        }
        if not items:
            raise ValueError(
                f"{json_path}: flat-schema parse found no entries with cx/cy/size."
            )

    out: list[Sample] = []
    n_total = 0
    for fname, roi in items.items():
        n_total += 1
        meta = parse_casia_filename(fname)
        if meta is None:
            continue
        if meta["band"] != band:
            continue
        out.append(Sample(
            fname=fname,
            identity=meta["identity"],
            cx=float(roi["cx"]),
            cy=float(roi["cy"]),
            size=float(roi["size"]),
        ))
    print(f"  {json_path.name}: kept {len(out)} of {n_total} entries (band={band})")
    return out


def crop_roi(img: Image.Image, sx: float, sy: float, ssize: float, out_size: int = 128) -> Image.Image:
    """Crop a square ROI defined by normalized (cx, cy, size) and resize.
    Pads with edge-replicate if the crop falls outside the image, so the
    matcher never sees zero pixels (which would create artificial edges).
    """
    W, H = img.size
    cx_px = sx * W
    cy_px = sy * H
    # Project convention: `size` is normalized by min(H, W).
    side_px = ssize * float(min(H, W))
    half = side_px / 2.0
    x1, y1, x2, y2 = cx_px - half, cy_px - half, cx_px + half, cy_px + half

    pad_l = int(max(0, -x1) + 0.999)
    pad_t = int(max(0, -y1) + 0.999)
    pad_r = int(max(0, x2 - W) + 0.999)
    pad_b = int(max(0, y2 - H) + 0.999)

    if pad_l or pad_t or pad_r or pad_b:
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
        else:
            arr = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
        img = Image.fromarray(arr)
        x1 += pad_l; x2 += pad_l; y1 += pad_t; y2 += pad_t

    crop = img.crop((round(x1), round(y1), round(x2), round(y2)))
    return crop.resize((out_size, out_size), Image.BILINEAR)


class PalmCropDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        images_dir: Path,
        identity_to_class: dict[str, int],
        augment: bool,
    ) -> None:
        self.samples = samples
        self.images_dir = Path(images_dir)
        self.id2cls = identity_to_class
        self.augment = augment

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        # 224x224 input for ResNet (ImageNet expects this size).
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.04, 0.04),
                    scale=(0.95, 1.05),
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self.samples[idx]
        img = Image.open(self.images_dir / s.fname)
        crop = crop_roi(img, s.cx, s.cy, s.size, out_size=128)
        # ResNet wants 3 channels; CASIA-MS images are grayscale (palette/L mode).
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        x = self.transform(crop)
        y = self.id2cls[s.identity]
        return x, y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int) -> nn.Module:
    """ImageNet-pretrained ResNet-18 with a fresh num_classes-way classifier.
    The 512-d penultimate features (avgpool output) are what the matching
    script will use at test time."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optim.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optim.step()
        sched.step()

        running_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_dir", required=True, help="Directory with train_labels.json + val_labels.json.")
    ap.add_argument("--images_dir", required=True, help="Folder containing the raw .jpg/.png images.")
    ap.add_argument("--band", default="850", help="Band code to filter on (e.g. '850'). Default 850 nm.")
    ap.add_argument("--out_ckpt", required=True, help="Output checkpoint path (.pt).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--augment", default="true", choices=["true", "false"], help="Enable training augmentation.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for THIS training run (separate from the data-split seed).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--sanity_dir", default=None,
        help="If set, dump 8 sample crops here before training starts so the "
             "user can confirm the --images_dir / labels combination is correct. "
             "If left unset, the script picks <out_ckpt parent>/sanity_seed<S>/.",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    augment = (args.augment == "true")

    split_dir = Path(args.split_dir)
    images_dir = Path(args.images_dir)
    train_lbl = split_dir / "train_labels.json"
    val_lbl = split_dir / "val_labels.json"
    if not train_lbl.exists():
        print(f"ERROR: not found: {train_lbl}", file=sys.stderr); return 2
    if not val_lbl.exists():
        print(f"ERROR: not found: {val_lbl}", file=sys.stderr); return 2

    print(f"Loading labels (band={args.band})...")
    samples = load_labels(train_lbl, args.band) + load_labels(val_lbl, args.band)
    if not samples:
        print(f"ERROR: no samples after filtering to band={args.band}.", file=sys.stderr); return 2

    # Verify image files exist for the first few samples (cheap sanity check).
    missing = [s.fname for s in samples[:5] if not (images_dir / s.fname).exists()]
    if missing:
        print(f"ERROR: cannot find these image files in {images_dir}:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("Check --images_dir and that filenames in the JSON match files on disk "
              "(extension, case).", file=sys.stderr)
        return 2

    identities = sorted({s.identity for s in samples})
    id2cls = {ident: i for i, ident in enumerate(identities)}
    num_classes = len(identities)
    print(f"  total samples : {len(samples)}")
    print(f"  identities    : {num_classes}")
    print(f"  augmentation  : {augment}")
    print(f"  out checkpoint: {args.out_ckpt}")

    dataset = PalmCropDataset(samples, images_dir, id2cls, augment=augment)

    # --- visual sanity check: dump 8 crops without augmentation ----------------
    sanity_dir = Path(args.sanity_dir) if args.sanity_dir else \
                 Path(args.out_ckpt).parent / f"sanity_{Path(args.out_ckpt).stem}"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDumping 8 sanity crops to {sanity_dir} ...")
    step = max(1, len(samples) // 8)
    for k in range(8):
        idx = min(k * step, len(samples) - 1)
        s = samples[idx]
        img = Image.open(images_dir / s.fname)
        crop_pil = crop_roi(img, s.cx, s.cy, s.size, out_size=128)
        crop_pil.save(sanity_dir / f"sanity_{k:02d}_{s.identity}_{Path(s.fname).stem}.png")
    print("  -> open these and verify they are upright palm interiors. If they "
          "look like edges, fingers, or wildly off-target, abort and check that "
          "--images_dir matches the coordinate frame of the labels.")
    # ---------------------------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type != args.device:
        print(f"  [WARN] requested device={args.device}, falling back to {device.type}")

    model = build_model(num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    def lr_lambda(step: int) -> float:
        # Linear warmup -> cosine decay.
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    print(f"\nTraining for {args.epochs} epochs ({total_steps} steps)...")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, loader, optim, sched, device)
        last_lr = optim.param_groups[0]["lr"]
        print(f"  epoch {ep:3d}/{args.epochs}  loss={loss:.4f}  acc={acc:5.2f}%  lr={last_lr:.2e}")

    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed:.1f}s")

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save backbone state_dict only (the fc layer is dropped by the matcher).
    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "identities": identities,
        "band": args.band,
        "split_dir": str(split_dir),
        "epochs": args.epochs,
        "augment": augment,
    }, out_path)
    print(f"Saved checkpoint -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())