import os, argparse, json, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets import SegmentationDataset
from src.models.unet_seg import UNetSeg
from src.utils import dice_score, iou_score, precision_recall

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--masks_dir", type=str, default=None)  # optional
    p.add_argument("--split_json", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.split_json, "r", encoding="utf-8") as f:
        split = json.load(f)
    key = str(args.seed) if str(args.seed) in split else next(iter(split.keys()))
    test_files = split[key]["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # infer encoder from checkpoint? we saved only weights; require user to pass encoder? Simpler: try resnet34 then mobilenet_v2 if fails
    for enc in ["resnet34", "mobilenet_v2"]:
        try:
            model = UNetSeg(encoder_name=enc).to(device)
            sd = torch.load(args.ckpt, map_location=device)["model"]
            model.load_state_dict(sd)
            break
        except Exception as e:
            last_err = e
            model = None
    if model is None:
        raise RuntimeError(f"Failed to load checkpoint with known encoders. Last error: {last_err}")
    model.eval()

    test_ds = SegmentationDataset(args.images_dir, args.masks_dir, test_files, is_train=False)
    test_ld = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    all_rows = []
    d_sum=i_sum=p_sum=r_sum=0.0; n=0
    with torch.no_grad():
        for imgs, masks, names in tqdm(test_ld, desc="Eval [test]"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)

            # per-batch metrics
            pred = (torch.sigmoid(logits) > 0.5).float()
            inter = (pred * masks).sum(dim=(1,2,3)).cpu().numpy()
            union = (pred + masks).clamp(0,1).sum(dim=(1,2,3)).cpu().numpy()
            dice = (2*inter + 1e-7) / (pred.sum(dim=(1,2,3)).cpu().numpy() + masks.sum(dim=(1,2,3)).cpu().numpy() + 1e-7)
            iou  = (inter + 1e-7) / (union + 1e-7)

            for k in range(len(names)):
                all_rows.append({"file": names[k], "dice": float(dice[k]), "iou": float(iou[k])})

            d_sum += float(dice.mean()); i_sum += float(iou.mean())
            pr, rc = precision_recall(logits, masks)
            p_sum += pr; r_sum += rc
            n += 1

    mean_d = d_sum/max(1,n); mean_i = i_sum/max(1,n); mean_p = p_sum/max(1,n); mean_r = r_sum/max(1,n)

    # write CSV
    csv_path = os.path.join(args.outdir, "seg_metrics_test.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # summary
    with open(os.path.join(args.outdir, "seg_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Dice mean={mean_d:.4f}\nIoU mean={mean_i:.4f}\nPrecision={mean_p:.4f}\nRecall={mean_r:.4f}\n")

    print("Done. Summary:")
    print(f"Dice mean={mean_d:.4f}  IoU mean={mean_i:.4f}  Precision={mean_p:.4f}  Recall={mean_r:.4f}")

if __name__ == "__main__":
    main()
