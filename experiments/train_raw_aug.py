#!/usr/bin/env python3
"""
train_raw_aug.py — does the mask-aligned frame still matter once a regressor is trained with strong augmentation?

Raw-frame cells of the 2x2 ablation (no Stage II alignment), re-trained with the SAME label-consistent strong
augmentation as UniRot-ROI and evaluated in-domain (CASIA-MS) and without adaptation (VERA, SCUT-PV):

  cell A  input = grey image (3 channels), raw frame -> oriented square (cx, cy, s, cos, sin)
  cell B  input = binary hand mask,       raw frame -> oriented square
          (training masks: the same Stage I masks as the original cell B / Stage III;
           test masks: the augmentation-trained Stage I of the same seed, as in UniRot-ROI)

Architecture, loss (smooth-L1 on the box + 0.1 x smooth-L1 on (cos, sin)), optimiser (AdamW 1e-3 / 1e-4), cosine
schedule, batch 128, letter-box 256 and model selection (lowest validation loss) are those of train_roi_2x2.py;
epochs 100 (as UniRot-ROI Stage III and the augmented detector).

Augmentation (StrongAug of train_pipeline_aug.py): affine about the image centre, scale x U(0.4, 1.6), translation
+-20%, rotation +-15 deg, horizontal flip p 0.5; brightness gain U(0.6, 1.4) for the image cell. The four corners of
the oriented square are mapped by the same affine (corner order mirrored on a flip) and the square is re-derived from
them, so the target follows the input exactly. Redrawn (<= 10 times) if the centre leaves the central 90% of the frame
or less than half of the hand stays in the frame.

Commands (run from experiments):
  python train_raw_aug.py selfcheck --cell B --seed 42
  python train_raw_aug.py train     --cell A --seed 42
  python train_raw_aug.py eval      --cell A --seed 42            # CASIA-MS test + VERA + SCUT-PV
  python train_raw_aug.py eval      --cell A --seed 42 --ckpt runs\\cellA_image_raw\\seed{seed}\\best.pth --tag _orig
       (the original-recipe cells, evaluated the same way; their test masks come from the original Stage I)
Outputs:
  runs_raw_aug\\cell<A|B>_seed<S>\\best.pt, log.csv, DONE.json
  out\\raw_aug\\cell<A|B><tag>\\casia_seed<S>.csv                      (one row per CASIA-MS test image)
  out\\cross\\results\\<ds>\\per_image_raw<A|B><tag>_seed<S>.csv        (VERA / SCUT-PV, same columns as xd_eval.py)
  runs\\results_raw_aug.csv                                           (means)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rx_common import index_dir, label_to_px, lb_to_orig, letterbox, load_json, orig_to_lb, save_json, square_corners, stem
from rx_stages import score
from train_pipeline_aug import StrongAug, set_seed, worker_init, log_row
from train_roi_2x2 import ROIRegressor

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

W0, H0 = 768, 576
IMAGES = r"../data/casia/images"
MASKS = r"../data/casia/stage1_masks"
CELL = {"A": ("image", "A_image_raw"), "B": ("mask", "B_mask_raw")}


# --------------------------------------------------------------------------------------- geometry
def square_from_corners(c: np.ndarray):
    """Inverse of square_corners: corners TL, TR, BR, BL -> (cx, cy, side, theta_deg, OpenCV sign)."""
    cx, cy = c.mean(0)
    side = float(np.mean([np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]))
    v = (c[1] - c[0]) + (c[2] - c[3])                      # averaged top and bottom edges
    th = math.degrees(math.atan2(-v[1], v[0]))
    return float(cx), float(cy), side, th


def transform_square(sq, M, flip):
    c = square_corners(*sq)
    c = np.hstack([c, np.ones((4, 1))]) @ M.T
    if flip:
        c = c[[1, 0, 3, 2]]                                # mirrored: left and right corners swap roles
    return square_from_corners(c)


# --------------------------------------------------------------------------------------- data
class RawData(Dataset):
    def __init__(self, labels: dict, cell: str, aug: StrongAug | None, T=256):
        self.kind = CELL[cell][0]
        self.img_idx = index_dir(Path(IMAGES))
        self.mask_idx = index_dir(Path(MASKS))
        src = self.img_idx if self.kind == "image" else self.mask_idx
        self.items = [(f, labels[f]) for f in sorted(labels) if stem(f) in src and stem(f) in self.mask_idx]
        if len(self.items) < len(labels):
            print(f"[WARN] {len(labels) - len(self.items)} labels without input file")
        self.aug, self.T = aug, T

    def __len__(self):
        return len(self.items)

    def sample(self, i):
        f, lab = self.items[i]
        m = ((cv2.imread(str(self.mask_idx[stem(f)]), cv2.IMREAD_GRAYSCALE) > 127) * 255).astype(np.uint8)
        if self.kind == "image":
            x = cv2.cvtColor(cv2.imread(str(self.img_idx[stem(f)]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            x = m
        H, W = m.shape
        sq = label_to_px(lab, W, H)
        if self.aug is not None:
            area = max(1, int((m > 127).sum()))
            for _ in range(self.aug.tries):
                M, a, s, flip = self.aug.draw(W, H)
                sq2 = transform_square(sq, M, flip)
                if not (0.05 * W <= sq2[0] <= 0.95 * W and 0.05 * H <= sq2[1] <= 0.95 * H):
                    continue
                mw = self.aug.warp(m, M, W, H, True)
                if (mw > 127).sum() < 0.5 * area:
                    continue
                if self.kind == "image":
                    x = self.aug.warp(x, M, W, H, False)
                    g = np.random.uniform(1 - self.aug.hsv_v, 1 + self.aug.hsv_v)
                    x = np.clip(x.astype(np.float32) * g, 0, 255).astype(np.uint8)
                else:
                    x = mw
                sq = sq2
                break
        return x, sq, H, W

    def __getitem__(self, i):
        x, (cx, cy, side, th), H, W = self.sample(i)
        lb = letterbox(x, self.T, self.kind == "mask")
        t3 = np.clip(np.array(orig_to_lb(cx, cy, side, H, W, self.T), np.float32), 0, 1)
        ang = np.array([math.cos(math.radians(th)), math.sin(math.radians(th))], np.float32)
        t = torch.from_numpy(lb.astype(np.float32) / 255.0)
        t = t.unsqueeze(0) if t.ndim == 2 else t.permute(2, 0, 1)
        return t.contiguous(), torch.from_numpy(t3), torch.from_numpy(ang)


def loss_fn(box, vec, t3, tang, w=0.1):
    return F.smooth_l1_loss(box, t3) + w * F.smooth_l1_loss(vec, tang)


# --------------------------------------------------------------------------------------- commands
def labels(seed):
    L = Path("out/labels_oriented") / f"seed{seed}"
    return {s: load_json(L / f"{s}_labels.json") for s in ("train", "val", "test")}


def cmd_selfcheck(a):
    """Exact check: corners of the transformed label == affine-mapped corners (any flip); plus a contact sheet."""
    rng = np.random.default_rng(0); np.random.seed(0)
    aug = StrongAug()
    worst = 0.0
    for _ in range(2000):
        sq = (rng.uniform(200, 560), rng.uniform(150, 420), rng.uniform(100, 260), rng.uniform(-180, 180))
        M, _, _, flip = aug.draw(W0, H0)
        sq2 = transform_square(sq, M, flip)
        c_map = np.hstack([square_corners(*sq), np.ones((4, 1))]) @ M.T
        if flip:
            c_map = c_map[[1, 0, 3, 2]]
        s = M[:, :2]; scale = math.sqrt(abs(np.linalg.det(s)))
        c_new = square_corners(*sq2)
        worst = max(worst, float(np.abs(c_new - c_map).max()) / scale)
    print(f"[selfcheck] max corner residual after re-deriving the square: {worst:.2e} px (should be ~1e-12)")
    ds = RawData(labels(a.seed)["train"], a.cell, aug)
    tiles = []
    for i in range(12):
        x, sq, H, W = ds.sample(i * 37 % len(ds))
        v = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if x.ndim == 2 else cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        c = np.round(square_corners(*sq)).astype(np.int32)
        cv2.polylines(v, [c], True, (0, 0, 255), 3)
        cv2.circle(v, tuple(((c[0] + c[1]) // 2).tolist()), 8, (0, 255, 0), -1)     # top edge
        tiles.append(cv2.resize(v, (384, 288)))
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, 12, 4)]
    out = Path("runs_raw_aug") / "selfcheck"; out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"cell{a.cell}_augmented_with_square.jpg"), np.vstack(rows))
    save_json({"max_corner_residual_px": worst, "ok": worst < 1e-6}, out / f"cell{a.cell}_selfcheck.json")
    if worst >= 1e-6:
        raise SystemExit("[selfcheck] FAILED")
    print(f"[selfcheck] passed -> look at {out}\\cell{a.cell}_augmented_with_square.jpg (green dot = top edge, "
          f"must lie on the finger side of the palm)")


def cmd_train(a):
    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path("runs_raw_aug") / f"cell{a.cell}_seed{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    lab = labels(a.seed)
    aug = StrongAug()
    tr, va = RawData(lab["train"], a.cell, aug), RawData(lab["val"], a.cell, None)
    dl = lambda d, sh: DataLoader(d, batch_size=128, shuffle=sh, num_workers=a.workers, pin_memory=True,
                                  worker_init_fn=worker_init, persistent_workers=a.workers > 0)
    tl, vl = dl(tr, True), dl(va, False)
    model = ROIRegressor(3 if CELL[a.cell][0] == "image" else 1, True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best, t0 = math.inf, time.time()
    print(f"[INFO] cell {a.cell} seed {a.seed}: train/val = {len(tr)}/{len(va)}, {a.epochs} epochs, device {dev}")
    for ep in range(1, a.epochs + 1):
        model.train(); run, n = 0.0, 0
        for x, t3, tang in tl:
            x, t3, tang = x.to(dev), t3.to(dev), tang.to(dev)
            box, vec = model(x)
            loss = loss_fn(box, vec, t3, tang)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item() * x.size(0); n += x.size(0)
        sch.step()
        model.eval(); vloss, m = 0.0, 0
        with torch.no_grad():
            for x, t3, tang in vl:
                x, t3, tang = x.to(dev), t3.to(dev), tang.to(dev)
                box, vec = model(x)
                vloss += loss_fn(box, vec, t3, tang).item() * x.size(0); m += x.size(0)
        vloss /= max(1, m)
        log_row(out / "log.csv", {"epoch": ep, "train_loss": run / max(1, n), "val_loss": vloss})
        print(f"[EP {ep:03d}] train={run / max(1, n):.5f} val={vloss:.5f} ({time.time() - t0:.0f}s)")
        if vloss < best:
            best = vloss
            torch.save({"model": model.state_dict(), "epoch": ep, "cell": a.cell, "seed": a.seed}, out / "best.pt")
    save_json({"best_val_loss": best, "epochs": a.epochs}, out / "DONE.json")


@torch.no_grad()
def predict(model, x, kind, T=256):
    H, W = x.shape[:2]
    lb = letterbox(x, T, kind == "mask")
    t = torch.from_numpy(lb.astype(np.float32) / 255.0)
    t = (t.unsqueeze(0) if t.ndim == 2 else t.permute(2, 0, 1)).unsqueeze(0).to(next(model.parameters()).device)
    box, vec = model(t)
    b, v = box[0].cpu().numpy(), vec[0].cpu().numpy()
    cx, cy, s = lb_to_orig(float(b[0]), float(b[1]), float(b[2]), H, W, T)
    return cx, cy, s, math.degrees(math.atan2(float(v[1]), float(v[0])))


def cmd_eval(a):
    import full_pipeline_eval as fp
    from argparse import Namespace
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kind = CELL[a.cell][0]
    ck = (a.ckpt or str(Path("runs_raw_aug") / f"cell{a.cell}_seed{{seed}}" / "best.pt")).format(seed=a.seed)
    model = ROIRegressor(3 if kind == "image" else 1, True).to(dev).eval()
    raw = torch.load(ck, map_location=dev, weights_only=False)
    model.load_state_dict(raw.get("model", raw), strict=True)
    seg = None
    if kind == "mask":
        ns = Namespace(pipeline_dir=a.pipeline_dir, seg_ckpt_pattern=a.seg_ckpt_pattern,
                       rot_ckpt_pattern=a.rot_ckpt_pattern, roi_ckpt_pattern=a.roi_ckpt_pattern, random_weights=False)
        (seg, _, _), _ = fp.build_models(ns, a.seed, dev)
    name = f"raw{a.cell}{a.tag}"

    def inp(rgb):
        return fp.stage1_mask(seg, rgb, dev) if kind == "mask" else rgb

    # ---- CASIA-MS test
    te = labels(a.seed)["test"]
    idx = index_dir(Path(IMAGES))
    rows = []
    for f in sorted(te):
        if stem(f) not in idx:
            continue
        rgb = cv2.cvtColor(cv2.imread(str(idx[stem(f)]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        p = predict(model, inp(rgb), kind)
        r = score(p, label_to_px(te[f], W, H))
        rows.append({"file": f, **{k: r[k] for k in ("iou", "center_err_px", "size_err_px", "angle_err_deg")},
                     "failed": int(r["iou"] < 0.5)})
    out = Path("out/raw_aug") / f"cell{a.cell}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    write(out / f"casia_seed{a.seed}.csv", rows)
    summ = {"casia": mean(rows)}
    # ---- VERA / SCUT-PV (same images, references and scoring as xd_eval.py)
    for ds in ("vera", "scut"):
        dd = Path("out/cross") / ds
        labs = load_json(dd / "labels_manual.json"); man = load_json(dd / "manifest.json")
        rows = []
        for n, l in labs.items():
            if l.get("skip") or n not in man:
                continue
            rgb = cv2.cvtColor(cv2.imread(str(dd / "images" / f"{n}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            p = predict(model, inp(rgb), kind)
            r = score(p, label_to_px(l, W0, H0))
            rows.append({"iou": r["iou"], "center_err_px": r["center_err_px"], "size_err_px": r["size_err_px"],
                         "angle_err_deg": r["angle_err_deg"], "file": n, "failed": int(r["iou"] < 0.5)})
        write(Path("out/cross/results") / ds / f"per_image_{name}_seed{a.seed}.csv", rows)
        summ[ds] = mean(rows)
    for k, v in summ.items():
        print(f"[{name} seed {a.seed}] {k:5s} mIoU={v['mIoU']:.4f} angle={v['angle']:.2f} fail={v['fail']:.3f}")
        log_row("runs/results_raw_aug.csv", {"cell": name, "seed": a.seed, "dataset": k, **v})


def mean(rows):
    return {"n": len(rows), "mIoU": float(np.mean([r["iou"] for r in rows])),
            "angle": float(np.mean([r["angle_err_deg"] for r in rows])),
            "centre": float(np.mean([r["center_err_px"] for r in rows])),
            "size": float(np.mean([r["size_err_px"] for r in rows])),
            "fail": float(np.mean([r["failed"] for r in rows]))}


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    global IMAGES, MASKS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["selfcheck", "train", "eval"])
    p.add_argument("--cell", choices=["A", "B"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ckpt", default=None, help="checkpoint pattern for eval (default: this script's best.pt)")
    p.add_argument("--tag", default="_aug", help="name suffix of the evaluated model (e.g. _aug or _orig)")
    p.add_argument("--images_dir", default=IMAGES)
    p.add_argument("--masks_dir", default=MASKS, help="Stage I masks of the CASIA-MS images (training input of cell B)")
    p.add_argument("--pipeline_dir", default=r"../pipeline_models")
    p.add_argument("--seg_ckpt_pattern", default=r"runs_pipeline_aug\stage1_seed{seed}\best.pt",
                   help="Stage I used to make test masks for cell B")
    p.add_argument("--rot_ckpt_pattern", default=r"runs_pipeline_aug\stage2_seed{seed}\best.pt")
    p.add_argument("--roi_ckpt_pattern", default=r"runs_pipeline_aug\stage3_seed{seed}\best.pt")
    a = p.parse_args()
    IMAGES, MASKS = a.images_dir, a.masks_dir
    {"selfcheck": cmd_selfcheck, "train": cmd_train, "eval": cmd_eval}[a.command](a)


if __name__ == "__main__":
    main()
