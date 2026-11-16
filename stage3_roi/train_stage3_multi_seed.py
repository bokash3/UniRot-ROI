#!/usr/bin/env python3
"""
Multi-seed training (ROI-Net + SmallCNN) using your "known-good" training logic.

This script is based on:
- train_roinet_letterbox.py
- train_small_cnn_letterbox.py

Key points (kept consistent with your working scripts)
------------------------------------------------------
- Dataset: ROIDatasetLetterbox
- Input: original masks -> letterbox to img_size x img_size (default 256)
- Targets: (cx,cy,size) in LETTERBOX-normalized space for training loss
- Validation metrics: MAE/RMSE/IoU computed in ORIGINAL space using meta (inverse letterbox)

Outputs (same naming style as your scripts)
------------------------------------------
ROI-Net:
  {roinet_save_dir}/roi_net_best_seed{seed}.pth

SmallCNN:
  {smallnet_save_dir}/small_cnn_best_seed{seed}.pth

All comments are in English as requested.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from models.ROINet import PalmROINet
from models.SmallCNN import SmallCNNBaseline
from models.dataset_letterbox import ROIDatasetLetterbox

# NOTE: In your working scripts, these helpers are imported from letterbox_utils.
from letterbox_utils import (
    LetterboxMeta,
    set_seed,
    boxes_to_masks,
    iou_xyxy,
    lb_norm_to_orig_norm,
    square_box_from_orig_norm,
)


# -----------------------------
# Utilities
# -----------------------------
def parse_seeds(seeds_str: str) -> List[int]:
    raw = seeds_str.replace(",", " ").split()
    return [int(x) for x in raw]


# -----------------------------
# ROI-Net (copied from your good script)
# -----------------------------
def roinet_train_one_epoch(model, loader, optimizer, device, loss_fn, epoch, lambda_outside: float):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for _batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        preds = model(inputs)
        reg_loss = loss_fn(preds, targets)

        if lambda_outside > 0.0:
            B, _, H, W = inputs.shape
            cx_pred, cy_pred, size_pred = preds[:, 0], preds[:, 1], preds[:, 2]
            box_masks_pred = boxes_to_masks(cx_pred, cy_pred, size_pred, H, W, device)
            hand_bin = (inputs > 0.5).float()
            outside = box_masks_pred * (1.0 - hand_bin)
            outside_loss = outside.mean()
            loss = reg_loss + lambda_outside * outside_loss
        else:
            loss = reg_loss

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * inputs.size(0)
        n_samples += inputs.size(0)

    return running_loss / max(1, n_samples)


@torch.no_grad()
def roinet_evaluate(model, loader, device, loss_fn, lambda_outside: float):
    """
    Loss in LETTERBOX space.
    Metrics (MAE/RMSE/IoU) in ORIGINAL space when meta is available.
    """
    model.eval()

    running_loss = 0.0
    n_samples = 0

    mae_sum = torch.zeros(3, device=device)
    mse_sum = torch.zeros(3, device=device)
    iou_sum = 0.0

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            imgs, targets, meta = batch
        else:
            imgs, targets = batch
            meta = None

        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(imgs)

        reg_loss = loss_fn(preds, targets)

        # Compute outside penalty in letterbox space (same as your script)
        with torch.no_grad():
            hand_bin = (imgs > 0.5).float()
        B, _, H, W = imgs.shape
        box_masks = boxes_to_masks(preds[:, 0], preds[:, 1], preds[:, 2], H, W, device)
        outside = box_masks * (1.0 - hand_bin)
        outside_loss = outside.mean()

        loss = reg_loss + lambda_outside * outside_loss
        running_loss += float(loss.item()) * B

        if meta is None:
            diff = preds - targets
            mae_sum += diff.abs().sum(dim=0)
            mse_sum += diff.pow(2).sum(dim=0)
            n_samples += B
            continue

        # Original-space metrics
        for i in range(B):
            mh = int(meta["orig_h"][i])
            mw = int(meta["orig_w"][i])
            mnew = int(meta["new_size"][i])
            mscale = float(meta["scale"][i])
            mpadx = int(meta["pad_x"][i])
            mpady = int(meta["pad_y"][i])
            lbm = LetterboxMeta(orig_h=mh, orig_w=mw, new_size=mnew, scale=mscale, pad_x=mpadx, pad_y=mpady)

            cx_o, cy_o, sz_o = lb_norm_to_orig_norm(
                float(preds[i, 0].item()),
                float(preds[i, 1].item()),
                float(preds[i, 2].item()),
                lbm
            )

            cx_g = float(meta["gt_cx_orig"][i])
            cy_g = float(meta["gt_cy_orig"][i])
            sz_g = float(meta["gt_size_orig"][i])

            dcx = cx_o - cx_g
            dcy = cy_o - cy_g
            dsz = sz_o - sz_g

            mae_sum += torch.tensor([abs(dcx), abs(dcy), abs(dsz)], device=device)
            mse_sum += torch.tensor([dcx * dcx, dcy * dcy, dsz * dsz], device=device)

            box_p = square_box_from_orig_norm(cx_o, cy_o, sz_o, mh, mw)
            box_g = square_box_from_orig_norm(cx_g, cy_g, sz_g, mh, mw)
            iou_sum += iou_xyxy(box_p, box_g)

        n_samples += B

    denom = max(1, n_samples)
    avg_loss = running_loss / denom
    mae = (mae_sum / denom).detach().cpu().tolist()
    rmse = (mse_sum / denom).sqrt().detach().cpu().tolist()
    mean_iou = iou_sum / denom

    return avg_loss, mae, rmse, mean_iou


def train_roinet_for_seed(args, seed: int) -> Path:
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ROI-NET][seed={seed}] Using device: {device}")

    data_root = Path(f"{args.data_root_base}{seed}")
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise RuntimeError(f"[ROI-NET][seed={seed}] Expected train/val under: {data_root}")

    with (data_root / "train_labels.json").open("r", encoding="utf-8") as f:
        train_labels = json.load(f)
    with (data_root / "val_labels.json").open("r", encoding="utf-8") as f:
        val_labels = json.load(f)

    print(f"[ROI-NET][seed={seed}] Loaded {len(train_labels)} train labels")
    print(f"[ROI-NET][seed={seed}] Loaded {len(val_labels)}   val labels")

    train_dataset = ROIDatasetLetterbox(train_dir, train_labels, img_size=args.img_size, augment=True, return_meta=False)
    val_dataset = ROIDatasetLetterbox(val_dir, val_labels, img_size=args.img_size, augment=False, return_meta=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PalmROINet(pretrained=False).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.roinet_lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.roinet_epochs))

    save_dir = Path(args.roinet_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / f"roi_net_best_seed{seed}.pth"

    best_val_loss = math.inf

    print(f"[ROI-NET][seed={seed}] Starting TRAINING for {args.roinet_epochs} epochs...")

    for epoch in range(1, args.roinet_epochs + 1):
        print("\n" + "=" * 60)
        print(f"[ROI-NET][seed={seed}] Epoch {epoch:03d}/{args.roinet_epochs:03d}")

        train_loss = roinet_train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, epoch, args.lambda_outside
        )
        val_loss, val_mae, val_rmse, val_iou = roinet_evaluate(
            model, val_loader, device, loss_fn, args.lambda_outside
        )
        scheduler.step()

        print(f"[ROI-NET][seed={seed}] Train Loss: {train_loss:.6f}")
        print(f"[ROI-NET][seed={seed}] Val   Loss: {val_loss:.6f}")
        print(f"[ROI-NET][seed={seed}] Val   MAE : cx={val_mae[0]:.4f}, cy={val_mae[1]:.4f}, size={val_mae[2]:.4f}")
        print(f"[ROI-NET][seed={seed}] Val  RMSE : cx={val_rmse[0]:.4f}, cy={val_rmse[1]:.4f}, size={val_rmse[2]:.4f}")
        print(f"[ROI-NET][seed={seed}] Val   IoU : {val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": float(val_loss),
                    "val_mae": val_mae,
                    "img_size": args.img_size,
                    "seed": seed,
                    "lambda_outside": float(args.lambda_outside),
                },
                best_ckpt_path,
            )
            print(f"[ROI-NET][seed={seed}] Saved new best model -> {best_ckpt_path}")

    print("\n" + "=" * 60)
    print(f"[ROI-NET][seed={seed}] DONE. Best val loss = {best_val_loss:.6f}")
    print(f"[ROI-NET][seed={seed}] Best checkpoint = {best_ckpt_path}")
    return best_ckpt_path


# -----------------------------
# SmallNet (based on your good script)
# -----------------------------
@torch.no_grad()
def smallnet_evaluate(model, loader, device, loss_fn):
    model.eval()

    running_loss = 0.0
    n_samples = 0

    mae_sum = torch.zeros(3, device=device)
    mse_sum = torch.zeros(3, device=device)
    iou_sum = 0.0

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            inputs, targets, meta = batch
        else:
            inputs, targets = batch
            meta = None

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(inputs)

        loss = loss_fn(preds, targets)
        running_loss += float(loss.item()) * inputs.size(0)

        B = inputs.size(0)

        if meta is None:
            diff = preds - targets
            mae_sum += diff.abs().sum(dim=0)
            mse_sum += diff.pow(2).sum(dim=0)
            n_samples += B
            continue

        for i in range(B):
            mh = int(meta["orig_h"][i])
            mw = int(meta["orig_w"][i])
            mnew = int(meta["new_size"][i])
            mscale = float(meta["scale"][i])
            mpadx = int(meta["pad_x"][i])
            mpady = int(meta["pad_y"][i])

            lbm = LetterboxMeta(orig_h=mh, orig_w=mw, new_size=mnew, scale=mscale, pad_x=mpadx, pad_y=mpady)

            cx_o, cy_o, sz_o = lb_norm_to_orig_norm(
                float(preds[i, 0].item()),
                float(preds[i, 1].item()),
                float(preds[i, 2].item()),
                lbm
            )

            cx_g = float(meta["gt_cx_orig"][i])
            cy_g = float(meta["gt_cy_orig"][i])
            sz_g = float(meta["gt_size_orig"][i])

            dcx = cx_o - cx_g
            dcy = cy_o - cy_g
            dsz = sz_o - sz_g

            mae_sum += torch.tensor([abs(dcx), abs(dcy), abs(dsz)], device=device)
            mse_sum += torch.tensor([dcx * dcx, dcy * dcy, dsz * dsz], device=device)

            box_p = square_box_from_orig_norm(cx_o, cy_o, sz_o, mh, mw)
            box_g = square_box_from_orig_norm(cx_g, cy_g, sz_g, mh, mw)
            iou_sum += iou_xyxy(box_p, box_g)

        n_samples += B

    denom = max(1, n_samples)
    avg_loss = running_loss / denom
    mae = (mae_sum / denom).detach().cpu().tolist()
    rmse = (mse_sum / denom).sqrt().detach().cpu().tolist()
    mean_iou = iou_sum / denom

    return avg_loss, mae, rmse, mean_iou


def smallnet_train_one_epoch(model, loader, optimizer, device, loss_fn, epoch):
    model.train()
    running = 0.0
    n = 0

    for imgs, targets in tqdm(loader, desc=f"SmallNet Train {epoch:03d}", ncols=100):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(imgs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        running += float(loss.item()) * imgs.size(0)
        n += imgs.size(0)

    return running / max(1, n)


def train_smallnet_for_seed(args, seed: int) -> Path:
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SMALLNET][seed={seed}] Using device: {device}")

    data_root = Path(f"{args.data_root_base}{seed}")
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise RuntimeError(f"[SMALLNET][seed={seed}] Expected train/val under: {data_root}")

    with (data_root / "train_labels.json").open("r", encoding="utf-8") as f:
        train_labels = json.load(f)
    with (data_root / "val_labels.json").open("r", encoding="utf-8") as f:
        val_labels = json.load(f)

    train_dataset = ROIDatasetLetterbox(train_dir, train_labels, img_size=args.img_size, augment=True, return_meta=False)
    val_dataset = ROIDatasetLetterbox(val_dir, val_labels, img_size=args.img_size, augment=False, return_meta=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    save_dir = Path(args.smallnet_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / f"small_cnn_best_seed{seed}.pth"

    model = SmallCNNBaseline(img_size=args.img_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.smallnet_lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.smallnet_epochs))

    best_val_loss = math.inf

    print(f"[SMALLNET][seed={seed}] Starting TRAINING for {args.smallnet_epochs} epochs...")

    for epoch in range(1, args.smallnet_epochs + 1):
        print("\n" + "=" * 60)
        print(f"[SMALLNET][seed={seed}] Epoch {epoch:03d}/{args.smallnet_epochs:03d}")

        train_loss = smallnet_train_one_epoch(model, train_loader, optimizer, device, loss_fn, epoch)
        val_loss, val_mae, val_rmse, val_iou = smallnet_evaluate(model, val_loader, device, loss_fn)
        scheduler.step()

        print(f"[SMALLNET][seed={seed}] Train Loss: {train_loss:.6f}")
        print(f"[SMALLNET][seed={seed}] Val   Loss: {val_loss:.6f}")
        print(f"[SMALLNET][seed={seed}] Val   MAE : cx={val_mae[0]:.4f}, cy={val_mae[1]:.4f}, size={val_mae[2]:.4f}")
        print(f"[SMALLNET][seed={seed}] Val  RMSE : cx={val_rmse[0]:.4f}, cy={val_rmse[1]:.4f}, size={val_rmse[2]:.4f}")
        print(f"[SMALLNET][seed={seed}] Val   IoU : {val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": float(val_loss),
                    "val_mae": val_mae,
                    "img_size": args.img_size,
                    "seed": seed,
                },
                best_ckpt_path,
            )
            print(f"[SMALLNET][seed={seed}] Saved new best model -> {best_ckpt_path}")

    print("\n" + "=" * 60)
    print(f"[SMALLNET][seed={seed}] DONE. Best val loss = {best_val_loss:.6f}")
    print(f"[SMALLNET][seed={seed}] Best checkpoint = {best_ckpt_path}")
    return best_ckpt_path


# -----------------------------
# CLI + main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Multi-seed training for ROI-Net and SmallNet (letterbox)")

    # Seeds / data
    p.add_argument("--seeds", type=str, default="42,2025,1337",
                   help="Comma/space-separated list, e.g. '42,2025,1337'")
    p.add_argument("--data_root_base", type=str, default="dataset_palm_locator_seed",
                   help="Per-seed root is f'{data_root_base}{seed}'")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # What to train
    p.add_argument("--train_roinet", action="store_true")
    p.add_argument("--train_smallnet", action="store_true")

    # ROI-Net hparams
    p.add_argument("--roinet_epochs", type=int, default=50)
    p.add_argument("--roinet_lr", type=float, default=1e-3)
    p.add_argument("--lambda_outside", type=float, default=0.0)
    p.add_argument("--roinet_save_dir", type=str, default="roi_net_runs")

    # SmallNet hparams
    p.add_argument("--smallnet_epochs", type=int, default=50)
    p.add_argument("--smallnet_lr", type=float, default=1e-3)
    p.add_argument("--smallnet_save_dir", type=str, default="small_cnn_runs")

    return p.parse_args()


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    if not (args.train_roinet or args.train_smallnet):
        # Default behavior: train both
        args.train_roinet = True
        args.train_smallnet = True

    print(f"[INFO] Seeds = {seeds}")
    print(f"[INFO] data_root_base = {args.data_root_base}")
    print(f"[INFO] img_size = {args.img_size} | batch_size = {args.batch_size}")

    for seed in seeds:
        print("\n" + "#" * 110)
        print(f"# SEED = {seed}")
        print("#" * 110)

        if args.train_roinet:
            train_roinet_for_seed(args, seed)
        if args.train_smallnet:
            train_smallnet_for_seed(args, seed)

    print("\n[DONE] Multi-seed training finished.")


if __name__ == "__main__":
    main()
