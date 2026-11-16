import os, argparse, math, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets import SegmentationDataset, list_images
from src.models.unet_seg import UNetSeg
from src.utils import set_seed, WarmupCosineLR, dice_score, iou_score, precision_recall
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # silence update warnings
def bce_dice_loss(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1e-7
    dice = 1 - (2*inter + 1e-7) / union
    return bce + dice.mean()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", type=str, default="resnet34")
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--masks_dir", type=str, default=None)  # optional
    p.add_argument("--split_json", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--outdir", type=str, default="runs/seg")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    import json
    with open(args.split_json, "r", encoding="utf-8") as f:
        split = json.load(f)
    # pick split for this seed (if multiple seeds exist, pick exact seed key or first key)
    key = str(args.seed) if str(args.seed) in split else next(iter(split.keys()))
    train_files = split[key]["train"]
    val_files   = split[key]["val"]

    train_ds = SegmentationDataset(args.images_dir, args.masks_dir, train_files, is_train=True)
    val_ds   = SegmentationDataset(args.images_dir, args.masks_dir, val_files,   is_train=False)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSeg(encoder_name=args.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    max_steps = args.epochs * len(train_ld)
    sch = WarmupCosineLR(opt, warmup_steps=200, max_steps=max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = -1.0
    bad = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_ld, desc=f"Epoch {epoch} [train]")
        for imgs, masks, _ in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(imgs)
                loss = bce_dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            pbar.set_postfix(loss=float(loss))

        # validation
        model.eval()
        dice, iou, prec, rec = 0.0, 0.0, 0.0, 0.0
        n = 0
        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_ld, desc=f"Epoch {epoch} [val]"):
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                dice += dice_score(logits, masks)
                iou  += iou_score(logits, masks)
                pr, rc = precision_recall(logits, masks)
                prec += pr; rec += rc
                n += 1
        dice /= max(1, n); iou /= max(1, n); prec /= max(1, n); rec /= max(1, n)

        ckpt_path = os.path.join(args.outdir, "last.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)

        if dice > best_val:
            best_val = dice
            bad = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(args.outdir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                break

    print("Training finished. Best val Dice =", best_val)

if __name__ == "__main__":
    main()
