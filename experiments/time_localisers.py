#!/usr/bin/env python3
"""
time_localisers.py — batch-1 latency of every localiser with ONE timing harness (for Table 2 of the paper)

Per image, from the decoded image in memory to the oriented square, CUDA-synchronised:
  pipeline   Stage I + Stage II + rotate mask + Stage III + back-projection   (full_pipeline_eval functions)
  yolo       YOLOv8n-pose predict on one image (pre-process + inference + NMS) + corner -> square
Images: the first --n CASIA-MS test images of --seed (850 nm by default). --warmup images are run first and
not timed. Prints mean / median / p95 in ms and writes out/latency.json.

  python time_localisers.py --seed 42 --n 200
"""
from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

from rx_common import index_dir, load_json, stem


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", default=r"../data/casia/images")
    p.add_argument("--labels", default=r"out\labels_oriented\seed{seed}\test_labels.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--band", default="850")
    p.add_argument("--yolo_weights", default=r"runs_yolo\pose_aug_seed{seed}\weights\best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="out/latency.json")
    a = p.parse_args()
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    te = load_json(a.labels.format(seed=a.seed))
    idx = index_dir(Path(a.images_dir))
    files = [f for f in sorted(te) if f"_{a.band}_" in f and stem(f) in idx][:a.n + a.warmup]
    imgs = [cv2.imread(str(idx[stem(f)]), cv2.IMREAD_COLOR) for f in files]
    print(f"[INFO] {len(imgs)} images, device={dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})")
    res = {}

    # ---------------- pipeline
    import full_pipeline_eval as fp
    ns = Namespace(pipeline_dir=r"../pipeline_models",
                   seg_ckpt_pattern=r"../checkpoints/basic/stage1_seg_mobilenetv2_seed{seed}.pt",
                   rot_ckpt_pattern=r"../checkpoints/basic/stage2_unirot_angle_seed{seed}.pt",
                   roi_ckpt_pattern=r"../checkpoints/basic/stage3_roi_seed{seed}.pth",
                   random_weights=False)
    (seg, rot, roi), RotCls = fp.build_models(ns, a.seed, dev)
    t = []
    with torch.no_grad():
        for i, bgr in enumerate(imgs):
            sync(dev); t0 = time.perf_counter()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            m = fp.stage1_mask(seg, rgb, dev)
            ang = fp.stage2_angle(rot, RotCls, m, dev)
            sq = fp.to_raw(fp.stage3_square(roi, fp.rotate_mask(m, ang), dev), ang, 768, 576)
            sync(dev); t.append((time.perf_counter() - t0) * 1000)
    res["pipeline"] = t[a.warmup:]
    del seg, rot, roi

    # ---------------- YOLO
    from ultralytics import YOLO
    from baseline_yolo import decode, to_square
    model = YOLO(a.yolo_weights.format(seed=a.seed))
    t = []
    for i, bgr in enumerate(imgs):
        sync(dev); t0 = time.perf_counter()
        r = model.predict(source=bgr, imgsz=640, conf=0.001, max_det=5, device=a.device if a.device != "cuda" else 0,
                          verbose=False)[0]
        d = decode("pose", r)
        sq = to_square("pose", d[0]["pts"]) if d else None
        sync(dev); t.append((time.perf_counter() - t0) * 1000)
    res["yolo"] = t[a.warmup:]

    out = {}
    for k, v in res.items():
        v = np.array(v)
        out[k] = {"mean_ms": float(v.mean()), "median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)),
                  "n": int(len(v))}
        print(f"  {k:9s} mean {v.mean():6.2f} ms   median {np.median(v):6.2f}   p95 {np.percentile(v, 95):6.2f}   (n={len(v)})")
    out["device"] = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    Path("out").mkdir(exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
