#!/usr/bin/env python3
"""
train_pipeline_aug.py — re-train the three stages of the mask-aligned pipeline with the strong
augmentation recipe that was given to YOLOv8n-pose ("augmented recipe"), so that the
pipeline-vs-detector comparison can be made under matched augmentation.

Only the augmentation changes (plus Stage III epochs 50 -> 100, matching the detector's 50 -> 100).
Data, subject splits, architectures, initialisation, losses, optimisers, schedules, batch sizes
and model-selection rules are those of the original stages:

  stage1  U-Net, MobileNetV2 encoder (ImageNet), image -> hand mask
          BCE + Dice, AdamW 3e-4 / 1e-4, 200 warm-up steps + cosine, batch 8, <=100 epochs,
          early stop on validation Dice (patience 15)
  stage2  UniRotNet: ResNet-18 (ImageNet; new randomly initialised 1-channel conv1), 'reg' head,
          binary mask of a ResNet-34 U-Net (preds/seg_resnet34_<seed>) -> (cos, sin) of the hand angle,
          smooth-L1 on the unit vector, AdamW 3e-4 / 1e-4, warm-up + cosine, batch 32, <=100 epochs,
          early stop on validation MAE (patience 15)
  stage3  ROI regressor (MobileNetV2, 1 channel, random init) on the aligned mask, 256 letter-box,
          smooth-L1 on (cx, cy, s), AdamW 1e-3 / 1e-4, cosine, batch 128, 100 epochs,
          lowest validation loss

Strong augmentation (Ultralytics 'augmented' recipe, transferred label-consistently):
  random affine about the image centre: scale x U(0.4, 1.6), translation U(-0.2, 0.2) x W/H,
  rotation U(-15, 15) deg, horizontal flip p = 0.5; brightness (HSV value) gain U(0.6, 1.4) for
  Stage I images. (Mosaic, saturation and random erasing are not used: they do not apply to
  single-object masks/grey images, and erasing was inactive for the detector too.)
  Label rules
    Stage I   the mask is warped with the image (nearest neighbour)
    Stage II  rotating the mask by a (cv2 convention) changes the label phi -> phi + a;
              a horizontal flip changes phi -> -phi   (verified by the 'selfcheck' command)
    Stage III centre mapped by the affine, side x scale; the residual rotation a is not represented
              (Stage III predicts an axis-aligned square), so Stage III learns to tolerate small
              alignment errors
  A transform is re-drawn (up to 10 times) if it leaves less than half of the hand in the frame
  (Stage III: or puts the square centre outside the frame); after 10 failures the sample is used
  without augmentation.

Checkpoints are written as {"model": state_dict, ...} and load directly into the classes used by
full_pipeline_eval.py (ROI-PIPELINE models: UNetSeg, RotResNet18(mode='reg'), PalmROINet).

Commands (run from experiments; see run_pipeline_aug.ps1):
  python train_pipeline_aug.py selfcheck --seed 42
  python train_pipeline_aug.py stage1 --seed 42
  python train_pipeline_aug.py stage2 --seed 42
  python train_pipeline_aug.py stage3 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rx_common import index_dir, label_to_px, letterbox, load_json, orig_to_lb, save_json, stem

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

H0, W0 = 576, 768
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
DATA = "../data/casia"          # CASIA-MS images, masks, angles, splits and labels (see README)


# =============================================================================================== utils
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    s = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(s); random.seed(s)


def fit_frame(img, interp):
    """Bring an image to 768 x 576 exactly like the original preprocessing (longest side 768, pad)."""
    h, w = img.shape[:2]
    if (h, w) == (H0, W0):
        return img
    s = W0 / max(h, w)
    img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=interp)
    out = np.zeros((H0, W0) + img.shape[2:], img.dtype)
    out[:img.shape[0], :img.shape[1]] = img
    return out


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Same schedule as src/utils.py of the original Stage I/II code (per optimisation step)."""

    def __init__(self, optimizer, warmup_steps=200, max_steps=10000, last_epoch=-1):
        self.warmup_steps, self.max_steps = warmup_steps, max_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        out = []
        for base in self.base_lrs:
            if step <= self.warmup_steps:
                sc = step / max(1, self.warmup_steps)
            else:
                t = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                sc = 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
            out.append(base * sc)
        return out


def load_split(path, seed):
    sp = load_json(path)
    key = str(seed) if str(seed) in sp else next(iter(sp))
    return {k: [stem(f) for f in sp[key][k]] for k in ("train", "val", "test")}


def log_row(path, row):
    new = not Path(path).is_file()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def import_pipeline_models(pipeline_dir):
    sys.path.insert(0, str(Path(pipeline_dir).resolve()))
    from models.unet_seg import UNetSeg          # noqa: E402
    from models.rot_net import RotResNet18       # noqa: E402
    from models.ROINet import PalmROINet          # noqa: E402
    return UNetSeg, RotResNet18, PalmROINet


# ======================================================================================= augmentation
class StrongAug:
    def __init__(self, scale=0.6, translate=0.2, degrees=15.0, fliplr=0.5, hsv_v=0.4, tries=10):
        self.scale, self.translate, self.degrees, self.fliplr, self.hsv_v, self.tries = \
            scale, translate, degrees, fliplr, hsv_v, tries

    def draw(self, W, H):
        s = np.random.uniform(1 - self.scale, 1 + self.scale)
        a = np.random.uniform(-self.degrees, self.degrees)
        tx = np.random.uniform(-self.translate, self.translate) * W
        ty = np.random.uniform(-self.translate, self.translate) * H
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), a, s)
        M[:, 2] += (tx, ty)
        flip = np.random.rand() < self.fliplr
        if flip:
            M = np.array([[-1.0, 0.0, W], [0.0, 1.0, 0.0]]) @ np.vstack([M, [0.0, 0.0, 1.0]])
        return M, a, s, flip

    @staticmethod
    def warp(img, M, W, H, nearest):
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ============================================================================================ Stage I
class SegData(Dataset):
    def __init__(self, names, img_idx, mask_idx, aug: StrongAug | None):
        self.names = [n for n in names if n in img_idx and n in mask_idx]
        if len(self.names) < len(names):
            print(f"[WARN] stage1: {len(names) - len(self.names)} files without image or mask")
        self.img_idx, self.mask_idx, self.aug = img_idx, mask_idx, aug

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        rgb = cv2.cvtColor(cv2.imread(str(self.img_idx[n]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        m = (cv2.imread(str(self.mask_idx[n]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        rgb, m = fit_frame(rgb, cv2.INTER_LINEAR), fit_frame(m, cv2.INTER_NEAREST)
        if self.aug is not None:
            area = max(1, int(m.sum()))
            for _ in range(self.aug.tries):
                M, a, s, flip = self.aug.draw(W0, H0)
                mw = self.aug.warp(m, M, W0, H0, True)
                if mw.sum() >= 0.5 * area:
                    rgb, m = self.aug.warp(rgb, M, W0, H0, False), mw
                    break
            g = np.random.uniform(1 - self.aug.hsv_v, 1 + self.aug.hsv_v)
            rgb = np.clip(rgb.astype(np.float32) * g, 0, 255).astype(np.uint8)
        x = (rgb.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(m[None].astype(np.float32)), n


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-7
    return bce + (1 - (2 * inter + 1e-7) / union).mean()


@torch.no_grad()
def dice_batch(logits, target):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * target).sum(dim=(1, 2, 3))
    return ((2 * inter + 1e-7) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-7)).mean().item()


def train_stage1(a):
    UNetSeg, _, _ = import_pipeline_models(a.pipeline_dir)
    out = Path(a.out_root) / f"stage1_seed{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    sp = load_split(a.split_json, a.seed)
    img_idx, mask_idx = index_dir(Path(a.seg_images)), index_dir(Path(a.seg_masks))
    aug = StrongAug(a.scale, a.translate, a.degrees, a.fliplr, a.hsv_v)
    tr = SegData(sp["train"], img_idx, mask_idx, aug)
    va = SegData(sp["val"], img_idx, mask_idx, None)
    dl = lambda ds, sh, bs: DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=a.workers, pin_memory=True,
                                       worker_init_fn=worker_init, persistent_workers=a.workers > 0)
    trl, val = dl(tr, True, a.batch_size or 8), dl(va, False, a.batch_size or 8)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSeg(encoder_name="mobilenet_v2", encoder_weights="imagenet").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs = a.epochs or 100
    sch = WarmupCosineLR(opt, 200, epochs * len(trl))
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and dev.type == "cuda")
    print(f"[stage1 seed {a.seed}] train/val = {len(tr)}/{len(va)}  epochs<={epochs}  amp={a.amp}")
    best, bad = -1.0, 0
    for ep in range(1, epochs + 1):
        model.train(); t0, run, k = time.time(), 0.0, 0
        for x, y, _ in trl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=a.amp and dev.type == "cuda"):
                loss = bce_dice_loss(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            run += loss.item(); k += 1
        model.eval(); d, n = 0.0, 0
        with torch.no_grad():
            for x, y, _ in val:
                d += dice_batch(model(x.to(dev)), y.to(dev)); n += 1
        d /= max(1, n)
        print(f"[stage1 seed {a.seed}] ep {ep:03d} loss={run / max(1, k):.4f} val_dice={d:.4f} ({time.time() - t0:.0f}s)")
        log_row(out / "log.csv", {"epoch": ep, "train_loss": run / max(1, k), "val_dice": d})
        torch.save({"model": model.state_dict(), "epoch": ep}, out / "last.pt")
        if d > best:
            best, bad = d, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_dice": d, "args": vars(a)}, out / "best.pt")
        else:
            bad += 1
            if bad >= (a.patience or 15):
                print(f"[stage1 seed {a.seed}] early stop at epoch {ep}")
                break
    save_json({"best_val_dice": best, "epochs_run": ep}, out / "DONE.json")


# =========================================================================================== Stage II
def load_angles(p):
    out = {}
    for k, v in load_json(p).items():
        out[stem(k)] = float(v["angle_deg"] if isinstance(v, dict) and "angle_deg" in v else
                             v.get("angle", 0.0) if isinstance(v, dict) else v)
    return out


class RotData(Dataset):
    def __init__(self, names, mask_idx, angles, aug: StrongAug | None):
        self.names = [n for n in names if n in mask_idx and n in angles]
        if len(self.names) < len(names):
            print(f"[WARN] stage2: {len(names) - len(self.names)} files without mask or angle")
        self.mask_idx, self.angles, self.aug = mask_idx, angles, aug

    def __len__(self):
        return len(self.names)

    def sample(self, i):
        """Returns (mask uint8 0/255, angle) after augmentation; also used by selfcheck."""
        n = self.names[i]
        m = ((cv2.imread(str(self.mask_idx[n]), cv2.IMREAD_GRAYSCALE) > 127) * 255).astype(np.uint8)
        m = fit_frame(m, cv2.INTER_NEAREST)
        phi = self.angles[n]
        info = None
        if self.aug is not None:
            area = max(1, int((m > 127).sum()))
            for _ in range(self.aug.tries):
                M, ang, s, flip = self.aug.draw(W0, H0)
                mw = self.aug.warp(m, M, W0, H0, True)
                if (mw > 127).sum() >= 0.5 * area:
                    m, phi = mw, phi + ang
                    if flip:
                        phi = -phi
                    info = (M, ang, s, flip)
                    break
        return m, phi % 360.0, n, info

    def __getitem__(self, i):
        m, phi, n, _ = self.sample(i)
        x = ((m > 127).astype(np.float32) - 0.5) / 0.5
        return torch.from_numpy(x[None].copy()), torch.tensor(phi, dtype=torch.float32), n


def make_unirotnet(RotResNet18):
    """ImageNet ResNet-18 with a new, randomly initialised 1-channel conv1 (as the original src/ code)."""
    m = RotResNet18(mode="reg", num_bins=72, grayscale=True, pretrained=True)
    old = m.backbone.conv1
    m.backbone.conv1 = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    return m


def circular_l1(vec, deg):
    rad = deg * math.pi / 180.0
    tgt = torch.stack([torch.cos(rad), torch.sin(rad)], 1)
    return F.smooth_l1_loss(F.normalize(vec, dim=1), tgt)


def vec_to_deg(vec):
    return (torch.atan2(vec[:, 1], vec[:, 0]) * 180.0 / math.pi) % 360.0


def ang_err(p, g):
    d = np.abs((np.asarray(p) - np.asarray(g) + 180.0) % 360.0 - 180.0)
    return d


def train_stage2(a):
    _, RotResNet18, _ = import_pipeline_models(a.pipeline_dir)
    out = Path(a.out_root) / f"stage2_seed{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    sp = load_split(a.split_json, a.seed)
    mask_idx = index_dir(Path(a.rot_masks.format(seed=a.seed)))
    angles = load_angles(a.angles_json)
    aug = StrongAug(a.scale, a.translate, a.degrees, a.fliplr, a.hsv_v)
    tr, va = RotData(sp["train"], mask_idx, angles, aug), RotData(sp["val"], mask_idx, angles, None)
    bs = a.batch_size or 32
    dl = lambda ds, sh: DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=a.workers, pin_memory=True,
                                   worker_init_fn=worker_init, persistent_workers=a.workers > 0)
    trl, val = dl(tr, True), dl(va, False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_unirotnet(RotResNet18).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs = a.epochs or 100
    sch = WarmupCosineLR(opt, 200, epochs * len(trl))
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and dev.type == "cuda")
    print(f"[stage2 seed {a.seed}] train/val = {len(tr)}/{len(va)}  epochs<={epochs}  amp={a.amp}")
    best, bad = 1e9, 0
    for ep in range(1, epochs + 1):
        model.train(); t0, run, k = time.time(), 0.0, 0
        for x, phi, _ in trl:
            x, phi = x.to(dev, non_blocking=True), phi.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=a.amp and dev.type == "cuda"):
                _, vec = model(x)
            loss = circular_l1(vec.float(), phi)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            run += loss.item(); k += 1
        model.eval(); P, G = [], []
        with torch.no_grad():
            for x, phi, _ in val:
                _, vec = model(x.to(dev))
                P += vec_to_deg(vec.float()).cpu().tolist(); G += phi.tolist()
        mae = float(ang_err(P, G).mean())
        print(f"[stage2 seed {a.seed}] ep {ep:03d} loss={run / max(1, k):.4f} val_MAE={mae:.2f} deg ({time.time() - t0:.0f}s)")
        log_row(out / "log.csv", {"epoch": ep, "train_loss": run / max(1, k), "val_mae_deg": mae})
        if mae < best:
            best, bad = mae, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_mae_deg": mae, "args": vars(a)}, out / "best.pt")
        else:
            bad += 1
            if bad >= (a.patience or 15):
                print(f"[stage2 seed {a.seed}] early stop at epoch {ep}")
                break
    save_json({"best_val_mae_deg": best, "epochs_run": ep}, out / "DONE.json")


# ========================================================================================== Stage III
class ROIData(Dataset):
    def __init__(self, labels: dict, mask_idx, aug: StrongAug | None, T=256):
        self.items = [(mask_idx[stem(f)], lab) for f, lab in sorted(labels.items()) if stem(f) in mask_idx]
        if len(self.items) < len(labels):
            print(f"[WARN] stage3: {len(labels) - len(self.items)} labels without aligned mask")
        self.aug, self.T = aug, T

    def __len__(self):
        return len(self.items)

    def sample(self, i):
        path, lab = self.items[i]
        m = ((cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127) * 255).astype(np.uint8)
        H, W = m.shape
        cx, cy, side, _ = label_to_px(lab, W, H)
        if self.aug is not None:
            area = max(1, int((m > 127).sum()))
            for _ in range(self.aug.tries):
                M, ang, s, flip = self.aug.draw(W, H)
                c = M @ np.array([cx, cy, 1.0])
                if not (0.05 * W <= c[0] <= 0.95 * W and 0.05 * H <= c[1] <= 0.95 * H):
                    continue
                mw = self.aug.warp(m, M, W, H, True)
                if (mw > 127).sum() >= 0.5 * area:
                    m, cx, cy, side = mw, float(c[0]), float(c[1]), side * s
                    break
        return m, (cx, cy, side), H, W

    def __getitem__(self, i):
        m, (cx, cy, side), H, W = self.sample(i)
        lb = letterbox(m, self.T, True)
        t = np.clip(np.array(orig_to_lb(cx, cy, side, H, W, self.T), np.float32), 0, 1)
        return torch.from_numpy((lb.astype(np.float32) / 255.0)[None].copy()), torch.from_numpy(t)


@torch.no_grad()
def eval_stage3(model, loader, dev):
    model.eval(); tot, n, ious = 0.0, 0, []
    for x, t in loader:
        x, t = x.to(dev), t.to(dev)
        p = model(x)
        tot += F.smooth_l1_loss(p, t, reduction="sum").item() / 3.0; n += x.size(0)
        pc, tc = p.cpu().numpy(), t.cpu().numpy()
        for (px, py, ps), (gx, gy, gs) in zip(pc, tc):
            ix = max(0.0, min(px + ps / 2, gx + gs / 2) - max(px - ps / 2, gx - gs / 2))
            iy = max(0.0, min(py + ps / 2, gy + gs / 2) - max(py - ps / 2, gy - gs / 2))
            inter = ix * iy
            ious.append(inter / max(1e-9, ps * ps + gs * gs - inter))
    return tot / max(1, n), float(np.mean(ious))


def train_stage3(a):
    _, _, PalmROINet = import_pipeline_models(a.pipeline_dir)
    out = Path(a.out_root) / f"stage3_seed{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    L = Path(a.roi_labels.format(seed=a.seed))
    lab = {s: load_json(L / f"{s}_labels.json") for s in ("train", "val", "test")}
    mask_idx = index_dir(Path(a.aligned_masks))
    aug = StrongAug(a.scale, a.translate, a.degrees, a.fliplr, a.hsv_v)
    tr, va, te = ROIData(lab["train"], mask_idx, aug), ROIData(lab["val"], mask_idx, None), ROIData(lab["test"], mask_idx, None)
    bs = a.batch_size or 128
    dl = lambda ds, sh: DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=a.workers, pin_memory=True,
                                   worker_init_fn=worker_init, persistent_workers=a.workers > 0)
    trl, val, tel = dl(tr, True), dl(va, False), dl(te, False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PalmROINet(pretrained=False).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = a.epochs or 100
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"[stage3 seed {a.seed}] train/val/test = {len(tr)}/{len(va)}/{len(te)}  epochs={epochs}")
    best = 1e9
    for ep in range(1, epochs + 1):
        model.train(); t0, run, k = time.time(), 0.0, 0
        for x, t in trl:
            x, t = x.to(dev, non_blocking=True), t.to(dev, non_blocking=True)
            loss = F.smooth_l1_loss(model(x), t)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item(); k += 1
        sch.step()
        vl, viou = eval_stage3(model, val, dev)
        print(f"[stage3 seed {a.seed}] ep {ep:03d} loss={run / max(1, k):.5f} val_loss={vl:.5f} "
              f"val_IoU(axis-aligned, exact angle)={viou:.4f} ({time.time() - t0:.0f}s)")
        log_row(out / "log.csv", {"epoch": ep, "train_loss": run / max(1, k), "val_loss": vl, "val_iou": viou})
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": vl, "args": vars(a)}, out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=False)["model"])
    tl, tiou = eval_stage3(model, tel, dev)
    print(f"[stage3 seed {a.seed}] TEST (exact alignment, reference masks): IoU={tiou:.4f}")
    save_json({"best_val_loss": best, "test_iou_exact_alignment": tiou}, out / "DONE.json")


# ============================================================================================ selfcheck
def principal_angle(m):
    mo = cv2.moments((m > 127).astype(np.uint8), binaryImage=True)
    if mo["m00"] < 10:
        return None
    return 0.5 * math.degrees(math.atan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"]))


def sheet(tiles, path, cols=6, w=256):
    tiles = [cv2.resize(t, (w, int(t.shape[0] * w / t.shape[1]))) for t in tiles]
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0] + 4, 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


def selfcheck(a):
    """Checks the label rules of the augmentation before any training and writes contact sheets."""
    out = Path(a.out_root) / "selfcheck"; out.mkdir(parents=True, exist_ok=True)
    np.random.seed(a.seed); random.seed(a.seed)
    sp = load_split(a.split_json, a.seed)
    aug = StrongAug(a.scale, a.translate, a.degrees, a.fliplr, a.hsv_v)
    ok = True
    # ---- Stage II. Derotating an augmented mask by its new label must undo the augmentation's rotation:
    #      the composite  R(-phi_new) . M . R(phi_old)  (M = the warp actually applied, flip included) must
    #      contain no rotation, only scale, translation and (after a flip) a mirror. This is checked exactly on
    #      the matrices, so it does not depend on the hand being fully visible (the wrist always touches the
    #      CASIA frame, which makes image-based orientation checks unreliable). The contact sheet shows the
    #      augmented mask (left) and the same mask derotated by its new label (right): the hand must be upright.
    from rx_common import rot_matrix
    mask_idx = index_dir(Path(a.rot_masks.format(seed=a.seed)))
    angles = load_angles(a.angles_json)
    ds0, ds1 = RotData(sp["train"], mask_idx, angles, None), RotData(sp["train"], mask_idx, angles, aug)
    h3 = lambda M: np.vstack([M, [0.0, 0.0, 1.0]])
    res, tiles, n_flip = [], [], 0
    for i in np.random.choice(len(ds0), min(300, len(ds0)), replace=False):
        m0, phi0, n, _ = ds0.sample(i)
        m1, phi1, _, info = ds1.sample(i)
        if info is None:
            continue
        M, ang, s, flip = info
        C = h3(rot_matrix(-phi1, W0, H0)) @ h3(M) @ h3(rot_matrix(phi0, W0, H0))
        L = C[:2, :2] @ (np.diag([-1.0, 1.0]) if flip else np.eye(2))
        res.append(abs((math.degrees(math.atan2(L[1, 0], L[0, 0])) + 180.0) % 360.0 - 180.0))
        n_flip += int(flip)
        if len(tiles) < 24:
            A1 = cv2.warpAffine(m1, rot_matrix(-phi1, W0, H0), (W0, H0), flags=cv2.INTER_NEAREST)
            tiles.append(cv2.cvtColor(np.hstack([m1, A1]), cv2.COLOR_GRAY2BGR))
    med, mx = float(np.median(res)), float(np.max(res))
    print(f"[selfcheck] stage2 label rule: n={len(res)} (flipped {n_flip})  residual rotation after derotation by "
          f"the new label: median={med:.4f} deg, max={mx:.4f} deg   (expected: max < 0.01 deg)")
    if mx > 0.01:
        ok = False
        print("[selfcheck] FAILED: the angle label does not follow the augmentation")
    p90 = mx
    sheet(tiles, out / "stage2_augmented_mask__derotated_by_new_label.jpg", cols=4, w=384)
    # ---- Stage III: draw the transformed square on the augmented aligned mask
    L = Path(a.roi_labels.format(seed=a.seed))
    ds3 = ROIData(load_json(L / "train_labels.json"), index_dir(Path(a.aligned_masks)), aug)
    tiles = []
    for i in np.random.choice(len(ds3), 24, replace=False):
        m, (cx, cy, s), H, W = ds3.sample(i)
        v = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(v, (int(cx - s / 2), int(cy - s / 2)), (int(cx + s / 2), int(cy + s / 2)), (0, 0, 255), 3)
        tiles.append(v)
    sheet(tiles, out / "stage3_augmented_aligned_mask_with_square.jpg")
    # ---- Stage I: augmented image with its warped mask outline
    img_idx, midx = index_dir(Path(a.seg_images)), index_dir(Path(a.seg_masks))
    ds = SegData(sp["train"], img_idx, midx, aug)
    tiles = []
    for i in np.random.choice(len(ds), 18, replace=False):
        x, y, _ = ds[i]
        rgb = np.clip((x.numpy().transpose(1, 2, 0) * IMNET_STD + IMNET_MEAN) * 255, 0, 255).astype(np.uint8)
        v = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cnt, _ = cv2.findContours(y.numpy()[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v, cnt, -1, (0, 255, 0), 2)
        tiles.append(v)
    sheet(tiles, out / "stage1_augmented_image_with_mask.jpg")
    print(f"[selfcheck] contact sheets -> {out}  (look at them: stage 2 right halves must show upright hands, "
          f"stage 3 squares must sit below the finger bases, stage 1 outlines must follow the hand)")
    save_json({"stage2_median_deg": med, "stage2_p90_deg": p90, "ok": ok}, out / "selfcheck.json")
    if not ok:
        raise SystemExit(2)


# ================================================================================================ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["selfcheck", "stage1", "stage2", "stage3"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_root", default=r"runs_pipeline_aug")
    p.add_argument("--pipeline_dir", default="../pipeline_models")
    p.add_argument("--split_json", default=DATA + "/subject_split.json")
    p.add_argument("--seg_images", default=DATA + "/images")
    p.add_argument("--seg_masks", default=DATA + "/masks", help="manually annotated hand masks")
    p.add_argument("--rot_masks", default=DATA + "/stage1_masks_resnet34_seed{seed}", help="Stage II training masks (as original)")
    p.add_argument("--angles_json", default=DATA + "/angles.json")
    p.add_argument("--aligned_masks", default=DATA + "/aligned_masks")
    p.add_argument("--roi_labels", default=DATA + "/dataset_palm_locator_seed{seed}")
    # augmentation (Ultralytics names / values of the detector's augmented recipe)
    p.add_argument("--scale", type=float, default=0.6)
    p.add_argument("--translate", type=float, default=0.2)
    p.add_argument("--degrees", type=float, default=15.0)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--hsv_v", type=float, default=0.4)
    # training (0 = original value of the stage)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", type=int, default=1, help="mixed precision for Stages I and II (1/0)")
    a = p.parse_args()
    a.amp = bool(a.amp)
    {"selfcheck": selfcheck, "stage1": train_stage1, "stage2": train_stage2, "stage3": train_stage3}[a.command](a)


if __name__ == "__main__":
    main()
