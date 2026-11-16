import os, argparse, json, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets2 import RotationDataset
from src.models.rot_net import RotResNet18
from src.utils import set_seed, WarmupCosineLR

def circular_l1_loss(pred_vec, target_deg):
    # pred_vec: (N,2) unconstrained; normalize to unit vector
    pred = nn.functional.normalize(pred_vec, dim=1)
    # target unit vector
    rad = target_deg * math.pi / 180.0
    tgt = torch.stack([torch.cos(rad), torch.sin(rad)], dim=1)
    return nn.functional.smooth_l1_loss(pred, tgt)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="both", choices=["cls","reg","both"])
    p.add_argument("--num_bins", type=int, default=36)
    p.add_argument("--images_dir", type=str, required=False)
    p.add_argument("--angles_json", type=str, required=True)
    p.add_argument("--pred_masks_dir", type=str, default=None)  # recommended
    p.add_argument("--split_json", type=str, required=True)
    p.add_argument("--train_spectrum", type=str, default=None)  # e.g., "940"
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--outdir", type=str, default="runs/rot")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    with open(args.split_json, "r", encoding="utf-8") as f:
        split = json.load(f)
    key = str(args.seed) if str(args.seed) in split else next(iter(split.keys()))
    train_files = split[key]["train"]
    val_files   = split[key]["val"]

    # Optional: filter by spectrum for training
    if args.train_spectrum is not None:
        def keep_spectrum(name):
            toks = name.split("_")
            return len(toks) >= 3 and (toks[2].split(".")[0] == args.train_spectrum)
        train_files = [x for x in train_files if keep_spectrum(x)]
        val_files   = [x for x in val_files   if keep_spectrum(x)]

    train_ds = RotationDataset(args.images_dir, args.angles_json, train_files,
                               pred_masks_dir=args.pred_masks_dir, grayscale=True, is_train=True)
    val_ds   = RotationDataset(args.images_dir, args.angles_json, val_files,
                               pred_masks_dir=args.pred_masks_dir, grayscale=True, is_train=False)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RotResNet18(mode=args.mode, num_bins=args.num_bins, grayscale=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    max_steps = args.epochs * len(train_ld)
    sch = WarmupCosineLR(opt, warmup_steps=200, max_steps=max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05) if args.mode in {"cls","both"} else None

    def angle_to_bin(angle_deg, C):
        angle = angle_deg % 360.0
        bin_w = 360.0 / C
        b = int(angle // bin_w)
        if b >= C: b = C-1
        return b

    best = 1e9  # use val MAE proxy (compute from logits/vec at val)
    bad = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_ld, desc=f"Epoch {epoch} [train]")
        for imgs, angle_deg, _ in pbar:
            imgs = imgs.to(device)
            angle_deg = angle_deg.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits, vec = model(imgs)
                loss = 0.0
                if args.mode in {"cls","both"}:
                    bins = torch.tensor([angle_to_bin(a.item(), args.num_bins) for a in angle_deg], device=imgs.device, dtype=torch.long)
                    loss = loss + ce_loss(logits, bins)
                if args.mode in {"reg","both"}:
                    loss = loss + circular_l1_loss(vec, angle_deg)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            pbar.set_postfix(loss=float(loss))

        # validation MAE
        model.eval()
        from src.models.rot_net import RotResNet18 as _R
        preds, gts = [], []
        with torch.no_grad():
            for imgs, angle_deg, _ in tqdm(val_ld, desc=f"Epoch {epoch} [val]"):
                imgs = imgs.to(device)
                angle_deg = angle_deg.to(device)
                logits, vec = model(imgs)
                degs = []
                if args.mode == "cls":
                    bins = logits.argmax(dim=1)
                    bin_w = 360.0 / args.num_bins
                    degs = (bins.float() + 0.5) * bin_w
                elif args.mode == "reg":
                    degs = _R.vec_to_deg(vec)
                else:  # both -> fused
                    bins = logits.argmax(dim=1)
                    bin_w = 360.0 / args.num_bins
                    cls_deg = (bins.float() + 0.5) * bin_w
                    reg_deg = _R.vec_to_deg(vec)
                    degs = 0.5*reg_deg + 0.5*cls_deg
                preds.extend(degs.detach().cpu().numpy().tolist())
                gts.extend(angle_deg.detach().cpu().numpy().tolist())
        # compute MAE circular
        import numpy as np
        diffs = np.array([min(abs(p-g)%360, 360-abs(p-g)%360) for p, g in zip(preds, gts)], dtype=np.float32)
        mae = float(diffs.mean())

        if mae < best:
            best = mae; bad = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": vars(args)}, os.path.join(args.outdir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                break

    print("Training finished. Best val MAE° =", best)

if __name__ == "__main__":
    main()
