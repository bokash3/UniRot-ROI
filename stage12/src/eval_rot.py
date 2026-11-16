import os, argparse, json, csv, math
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets import RotationDataset, load_angles
from src.models.rot_net import RotResNet18
from src.utils import mae_deg, percentile_deg

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--images_dir", type=str, required=False)
    p.add_argument("--angles_json", type=str, required=True)
    p.add_argument("--pred_masks_dir", type=str, default=None)
    p.add_argument("--split_json", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_bins", type=int, default=36)
    p.add_argument("--mode", type=str, default="both", choices=["cls","reg","both"])
    p.add_argument("--test_spectrum", type=str, default=None)  # e.g., "850" or "850,700"
    p.add_argument("--outdir", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.split_json, "r", encoding="utf-8") as f:
        split = json.load(f)
    key = str(args.seed) if str(args.seed) in split else next(iter(split.keys()))
    test_files = split[key]["test"]

    # Restrict by spectrum if requested
    if args.test_spectrum is not None:
        allowed = set([s.strip() for s in args.test_spectrum.split(",") if s.strip()])
        meta = load_angles(args.angles_json)
        def keep(f):
            m = meta.get(f, {})
            sp = m.get("spectrum", None)
            return sp in allowed
        test_files = [f for f in test_files if keep(f)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    mode = args.mode if args.mode else cfg.get("mode", "both")
    num_bins = args.num_bins if args.num_bins else cfg.get("num_bins", 36)

    model = RotResNet18(mode=mode, num_bins=num_bins, grayscale=True).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = RotationDataset(args.images_dir, args.angles_json, test_files,
                         pred_masks_dir=args.pred_masks_dir, grayscale=True, is_train=False)
    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    all_rows = []
    preds, gts = [], []
    with torch.no_grad():
        for imgs, angle_deg, names in tqdm(ld, desc="EvalRot [test]"):
            imgs = imgs.to(device)
            angle_deg = angle_deg.to(device)
            logits, vec = model(imgs)

            if mode == "cls":
                bins = logits.argmax(dim=1)
                bin_w = 360.0 / num_bins
                pred_deg = (bins.float() + 0.5) * bin_w
            elif mode == "reg":
                pred_deg = RotResNet18.vec_to_deg(vec)
            else:
                bins = logits.argmax(dim=1)
                bin_w = 360.0 / num_bins
                cls_deg = (bins.float() + 0.5) * bin_w
                reg_deg = RotResNet18.vec_to_deg(vec)
                pred_deg = 0.5*reg_deg + 0.5*cls_deg

            pd = pred_deg.detach().cpu().numpy().tolist()
            gt = angle_deg.detach().cpu().numpy().tolist()
            for k in range(len(names)):
                all_rows.append({"file": names[k], "pred_deg": float(pd[k]), "gt_deg": float(gt[k])})
            preds.extend(pd); gts.extend(gt)

    preds = np.array(preds, dtype=np.float32)
    gts   = np.array(gts, dtype=np.float32)
    mae   = float(np.mean([min(abs(p-g)%360, 360-abs(p-g)%360) for p,g in zip(preds,gts)]))
    med   = float(np.median([min(abs(p-g)%360, 360-abs(p-g)%360) for p,g in zip(preds,gts)]))
    p90   = float(np.percentile([min(abs(p-g)%360, 360-abs(p-g)%360) for p,g in zip(preds,gts)], 90))
    p95   = float(np.percentile([min(abs(p-g)%360, 360-abs(p-g)%360) for p,g in zip(preds,gts)], 95))

    # bin-based accuracies
    def acc_at_tolerance(preds, gts, C, tol_bins=1):
        bin_w = 360.0 / C
        pred_bins = np.floor((preds % 360) / bin_w).astype(int)
        gt_bins   = np.floor((gts % 360) / bin_w).astype(int)
        diff = np.abs(pred_bins - gt_bins)
        diff = np.minimum(diff, C - diff)
        acc = np.mean((diff <= tol_bins).astype(np.float32))
        return float(acc)

    acc1   = acc_at_tolerance(preds, gts, num_bins, tol_bins=0)
    accpm1 = acc_at_tolerance(preds, gts, num_bins, tol_bins=1)

    # write CSV
    csv_path = os.path.join(args.outdir, "rot_metrics_test.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    with open(os.path.join(args.outdir, "rot_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"MAE°={mae:.3f}\nMedian°={med:.3f}\np90°={p90:.3f}\np95°={p95:.3f}\nAcc@1={acc1:.3f}\nAcc@±1bin={accpm1:.3f}\n")

    print("Done. MAE°=%.3f  Median°=%.3f  p90°=%.3f  p95°=%.3f  Acc@1=%.3f  Acc@±1bin=%.3f" % (mae, med, p90, p95, acc1, accpm1))

if __name__ == "__main__":
    main()
