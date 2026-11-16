#!/usr/bin/env python3
"""
train_roi_2x2.py — fair end-to-end baseline + 2x2 ablation for Stage III
=======================================================================

One script, four cells (same MobileNetV2 regressor as Lite-PalmROI,
same optimizer, schedule, epochs, letterbox and flip augmentation):

                    | no alignment (raw frame)        | alignment (aligned frame)
  ------------------+---------------------------------+---------------------------
  input = image     | A: --input image --frame raw    | C: --input image --frame aligned
  input = mask      | B: --input mask  --frame raw    | D: --input mask  --frame aligned
                                                         (= Lite-PalmROI re-run)

* raw-frame cells predict the ORIENTED GT square (cx, cy, size, theta):
  head = 3 sigmoid outputs + a 2-D (cos, sin) vector for theta. Their
  targets come from make_rx_data.py labels (no bounding-box inflation).
  Cell A is the fair replacement of the old Image-PalmROI baseline.
* aligned-frame cells predict the axis-aligned square (cx, cy, size)
  exactly as in the paper.

Evaluation (test split, original pixel space of each cell's frame):
  mIoU / Acc@0.5 / Acc@0.7 between the predicted and GT squares (rotated
  polygons for raw-frame cells), centre error (px), side error (px),
  outside fraction (share of the predicted square outside the hand mask),
  and, for raw-frame cells, the angular error of theta.
  Because the GT square is a rigid transform of the aligned-frame label,
  these numbers are directly comparable with Table 3 of the paper.

Example (cell A, seed 42):
  python train_roi_2x2.py --input image --frame raw --seed 42 ^
     --data_dir   ../data/casia/images ^
     --labels_dir out\\labels_oriented\\seed42 ^
     --hand_masks ../data/casia/stage1_masks ^
     --save_dir   runs\\cellA_image_raw\\seed42 --results_csv runs\\results_2x2.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from rx_common import (circ_err, index_dir, label_to_px, lb_to_orig, letterbox, load_json,
                       orig_to_lb, outside_fraction, polygon_iou, square_corners, stem)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- model
class ROIRegressor(nn.Module):
    """MobileNetV2 + the Lite-PalmROI head; optional (cos, sin) angle outputs."""

    def __init__(self, in_channels: int, oriented: bool):
        super().__init__()
        self.oriented = oriented
        self.backbone = models.mobilenet_v2(weights=None)
        fc = self.backbone.features[0][0]
        if in_channels != fc.in_channels:
            self.backbone.features[0][0] = nn.Conv2d(in_channels, fc.out_channels, fc.kernel_size,
                                                     fc.stride, fc.padding, bias=fc.bias is not None)
        nf = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(nf, 64),
                                                 nn.ReLU(inplace=True), nn.Linear(64, 5 if oriented else 3))

    def forward(self, x):
        out = self.backbone(x)
        box = torch.sigmoid(out[:, :3])
        if not self.oriented:
            return box, None
        return box, F.normalize(out[:, 3:5], dim=1, eps=1e-6)


# ---------------------------------------------------------------- data
class ROIData(Dataset):
    def __init__(self, data_dir, labels: dict, T: int, is_mask: bool, oriented: bool, augment: bool):
        idx = index_dir(Path(data_dir))
        self.items = [(idx[stem(f)], f, labels[f]) for f in sorted(labels) if stem(f) in idx]
        if not self.items:
            raise RuntimeError(f"No labelled files found in {data_dir}")
        miss = len(labels) - len(self.items)
        if miss:
            print(f"[WARN] {miss} labels have no file in {data_dir}")
        self.T, self.is_mask, self.oriented, self.augment = T, is_mask, oriented, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, fname, lbl = self.items[i]
        if self.is_mask:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        cx, cy, side, th = label_to_px(lbl, W, H)
        if self.augment and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            cx = W - cx            # pixel mirror (labels are continuous)
            th = -th               # mirrored rotation
        lb = letterbox(img, self.T, self.is_mask)
        t3 = np.clip(np.array(orig_to_lb(cx, cy, side, H, W, self.T), np.float32), 0, 1)
        ang = np.array([math.cos(math.radians(th)), math.sin(math.radians(th))], np.float32)
        x = torch.from_numpy(lb.astype(np.float32) / 255.0)
        x = x.unsqueeze(0) if x.ndim == 2 else x.permute(2, 0, 1)
        return x, torch.from_numpy(t3), torch.from_numpy(ang), fname, H, W


# ---------------------------------------------------------------- loss / eval
def compute_loss(box, vec, t3, tang, w_angle):
    loss = F.smooth_l1_loss(box, t3)
    if vec is not None:
        loss = loss + w_angle * F.smooth_l1_loss(vec, tang)
    return loss


@torch.no_grad()
def evaluate(model, loader, device, T, w_angle, hand_masks=None, per_sample=False):
    model.eval()
    hidx = index_dir(Path(hand_masks)) if hand_masks else None
    tot, n = 0.0, 0
    rows = []
    for x, t3, tang, fnames, Hs, Ws in loader:
        x, t3, tang = x.to(device), t3.to(device), tang.to(device)
        box, vec = model(x)
        tot += compute_loss(box, vec, t3, tang, w_angle).item() * x.size(0)
        n += x.size(0)
        box, t3c, tangc = box.cpu().numpy(), t3.cpu().numpy(), tang.cpu().numpy()
        vecc = vec.cpu().numpy() if vec is not None else None
        for i in range(x.size(0)):
            H, W = int(Hs[i]), int(Ws[i])
            pcx, pcy, ps = lb_to_orig(*box[i], H, W, T)
            gcx, gcy, gs = lb_to_orig(*t3c[i], H, W, T)
            gth = math.degrees(math.atan2(tangc[i][1], tangc[i][0]))
            pth = math.degrees(math.atan2(vecc[i][1], vecc[i][0])) if vecc is not None else 0.0
            pp = square_corners(pcx, pcy, ps, pth)
            gp = square_corners(gcx, gcy, gs, gth)
            r = {"file": fnames[i], "iou": polygon_iou(pp, gp),
                 "center_err_px": math.hypot(pcx - gcx, pcy - gcy), "size_err_px": abs(ps - gs),
                 "angle_err_deg": circ_err(pth, gth) if vecc is not None else float("nan")}
            if hidx is not None:
                m = cv2.imread(str(hidx[stem(fnames[i])]), cv2.IMREAD_GRAYSCALE)
                r["outside_frac"] = outside_fraction(pp, m)
            rows.append(r)
    ious = np.array([r["iou"] for r in rows])
    summ = {"loss": tot / max(1, n), "mIoU": ious.mean(), "acc_0_5": (ious >= 0.5).mean(),
            "acc_0_7": (ious >= 0.7).mean(),
            "center_err_px": np.mean([r["center_err_px"] for r in rows]),
            "size_err_px": np.mean([r["size_err_px"] for r in rows]),
            "angle_err_deg": np.nanmean([r["angle_err_deg"] for r in rows]) if rows and not math.isnan(rows[0]["angle_err_deg"]) else float("nan")}
    if hidx is not None:
        summ["outside_frac"] = np.mean([r["outside_frac"] for r in rows])
    return (summ, rows) if per_sample else summ


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser("2x2 Stage III ablation / fair end-to-end baseline")
    p.add_argument("--input", choices=["image", "mask"], required=True)
    p.add_argument("--frame", choices=["raw", "aligned"], required=True)
    p.add_argument("--data_dir", required=True, help="images or masks in the chosen frame")
    p.add_argument("--labels_dir", required=True, help="folder with train/val/test_labels.json")
    p.add_argument("--hand_masks", default=None, help="hand masks in the same frame (outside fraction)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--w_angle", type=float, default=0.1,
                   help="weight of the angle loss (raw-frame cells); 0.1 keeps both terms of similar size")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--results_csv", default=None)
    p.add_argument("--mode", choices=["train", "test", "both"], default="both")
    a = p.parse_args()

    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oriented = a.frame == "raw"
    is_mask = a.input == "mask"
    cell = {("image", "raw"): "A_image_raw", ("mask", "raw"): "B_mask_raw",
            ("image", "aligned"): "C_image_aligned", ("mask", "aligned"): "D_mask_aligned"}[(a.input, a.frame)]
    print(f"[INFO] cell={cell} seed={a.seed} device={dev} oriented={oriented}")

    L = Path(a.labels_dir)
    lab = {s: load_json(L / f"{s}_labels.json") for s in ("train", "val", "test")}
    if oriented and "theta" not in next(iter(lab["train"].values())):
        raise SystemExit("raw-frame cells need oriented labels (run make_rx_data.py labels)")
    mk = lambda s, aug: ROIData(a.data_dir, lab[s], a.img_size, is_mask, oriented, aug)
    dl = lambda ds, sh: DataLoader(ds, batch_size=a.batch_size, shuffle=sh, num_workers=a.num_workers,
                                   pin_memory=dev.type == "cuda")
    tr, va, te = dl(mk("train", True), True), dl(mk("val", False), False), dl(mk("test", False), False)
    print(f"[INFO] train/val/test = {len(tr.dataset)}/{len(va.dataset)}/{len(te.dataset)}")

    model = ROIRegressor(1 if is_mask else 3, oriented).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=max(1, a.num_epochs))
    save = Path(a.save_dir); save.mkdir(parents=True, exist_ok=True)
    ckpt = save / "best.pth"

    if a.mode in ("train", "both"):
        best = math.inf
        for ep in range(1, a.num_epochs + 1):
            model.train(); run, m = 0.0, 0
            for x, t3, tang, *_ in tr:
                x, t3, tang = x.to(dev), t3.to(dev), tang.to(dev)
                box, vec = model(x)
                loss = compute_loss(box, vec, t3, tang, a.w_angle)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                run += loss.item() * x.size(0); m += x.size(0)
            sch.step()
            v = evaluate(model, va, dev, a.img_size, a.w_angle)
            print(f"[EP {ep:03d}] train={run / m:.5f} val={v['loss']:.5f} mIoU={v['mIoU']:.4f} "
                  f"ctr={v['center_err_px']:.1f}px size={v['size_err_px']:.1f}px ang={v['angle_err_deg']:.2f}")
            if v["loss"] < best:
                best = v["loss"]
                torch.save({"model": model.state_dict(), "epoch": ep,
                            "val": {k: float(x) for k, x in v.items()}, "args": vars(a)}, ckpt)

    if a.mode in ("test", "both"):
        model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
        s, rows = evaluate(model, te, dev, a.img_size, a.w_angle, a.hand_masks, per_sample=True)
        print("[TEST] " + "  ".join(f"{k}={v:.4f}" for k, v in s.items()))
        with open(save / "test_per_sample.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        if a.results_csv:
            rp = Path(a.results_csv); rp.parent.mkdir(parents=True, exist_ok=True)
            new = not rp.is_file()
            with open(rp, "a", newline="") as f:
                w = csv.writer(f)
                keys = ["mIoU", "acc_0_5", "acc_0_7", "center_err_px", "size_err_px", "outside_frac", "angle_err_deg"]
                if new:
                    w.writerow(["cell", "seed"] + keys)
                w.writerow([cell, a.seed] + [s.get(k, float("nan")) for k in keys])
            print(f"[INFO] appended -> {rp}")


if __name__ == "__main__":
    main()
