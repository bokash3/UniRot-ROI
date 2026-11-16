#!/usr/bin/env python3
"""
eval_checkpoints.py — re-evaluate trained Stage III regressors on the test split
with the same metric code as train_roi_2x2.py.

Two extra analyses for the 2x2 ablation:
  * --angle oracle : raw-frame cells (A, B) are scored with the GT angle instead
    of their predicted angle. This isolates the quality of (cx, cy, size) and
    puts them on the same footing as the aligned-frame cells, which receive the
    angle of the annotation frame for free.
  * --ckpt_format lite : evaluates the original Lite-PalmROI checkpoints
    (roi_net_best_seed*.pth, key 'model_state_dict') with this metric code, so
    the paper's model can be compared with cells A-C under identical scoring.

Example:
  python eval_checkpoints.py --input image --frame raw --angle oracle ^
     --ckpt runs\\cellA_image_raw\\seed42\\best.pth --data_dir ../data/casia/images ^
     --labels out\\labels_oriented\\seed42\\test_labels.json ^
     --hand_masks ../data/casia/stage1_masks --tag A_oracle --seed 42 --out_csv runs\\extra_eval.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rx_common import load_json
from train_roi_2x2 import ROIData, ROIRegressor, evaluate


class OracleAngle(torch.nn.Module):
    """Wraps a raw-frame model and replaces its angle output by the GT angle."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.gt = None

    def forward(self, x):
        box, _ = self.model(x)
        return box, self.gt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", choices=["image", "mask"], required=True)
    p.add_argument("--frame", choices=["raw", "aligned"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt_format", choices=["rx", "lite"], default="rx")
    p.add_argument("--angle", choices=["pred", "oracle"], default="pred")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--labels", required=True, help="test_labels.json")
    p.add_argument("--hand_masks", default=None)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--per_sample_csv", default=None)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oriented = a.frame == "raw"
    model = ROIRegressor(1 if a.input == "mask" else 3, oriented)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if a.ckpt_format == "lite" else ck["model"]
    model.load_state_dict(sd)
    model.to(dev).eval()

    ds = ROIData(a.data_dir, load_json(a.labels), a.img_size, a.input == "mask", oriented, False)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)

    if a.angle == "oracle":
        if not oriented:
            raise SystemExit("--angle oracle only applies to raw-frame cells")
        wrapped = OracleAngle(model)

        def batches():
            for x, t3, ta, f, H, W in dl:
                wrapped.gt = ta.to(dev)
                yield x, t3, ta, f, H, W
        summ, rows = evaluate(wrapped, batches(), dev, a.img_size, 0.1, a.hand_masks, per_sample=True)
    else:
        summ, rows = evaluate(model, dl, dev, a.img_size, 0.1, a.hand_masks, per_sample=True)

    print(f"[{a.tag} seed {a.seed}] " + "  ".join(f"{k}={v:.4f}" for k, v in summ.items()))
    keys = ["mIoU", "acc_0_5", "acc_0_7", "center_err_px", "size_err_px", "outside_frac", "angle_err_deg"]
    rp = Path(a.out_csv); rp.parent.mkdir(parents=True, exist_ok=True)
    new = not rp.is_file()
    with open(rp, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["cell", "seed"] + keys)
        w.writerow([a.tag, a.seed] + [summ.get(k, float("nan")) for k in keys])
    if a.per_sample_csv:
        with open(a.per_sample_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
