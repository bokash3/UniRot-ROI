#!/usr/bin/env python3
"""
baseline_yolo.py — modern end-to-end detector baselines for the ROI task (Ultralytics YOLOv8)
=============================================================================================

Trains a standard single-stage detector DIRECTLY ON THE RAW IMAGE (no mask, no alignment)
to output the oriented GT palm square, with exactly the same subject-disjoint splits,
oriented labels (make_rx_data.py labels) and metrics as cells A-D of the 2x2 ablation.

Two heads (--task)
  pose : YOLOv8-pose with 4 keypoints = the ordered corners TL, TR, BR, BL of the square.
         The order encodes which side is "up", so the full 0-360 deg orientation is known and an
         upright crop can be produced. THIS is the like-for-like end-to-end baseline.
  obb  : YOLOv8-OBB (oriented bounding box). A square is 4-fold symmetric, so the detector
         cannot tell which side is up: IoU/centre/size are fair, but the angle is only known
         modulo 90 deg and no upright crop is produced (angle error is reported modulo 90).

Weights: by default the network is built from its yaml (RANDOM init), like cell A, so that
the comparison does not mix in ImageNet/COCO/DOTA pre-training. Use --pretrained to start
from the official *.pt weights (needs internet) — report both if you can.
Augmentation: horizontal flip only (mosaic, rotation, scale, translate, HSV changes are
off) — the same as cells A-D — so that the difference comes from the architecture, not from augmentation.

Sub-commands (each prints where it writes):
  prep   labels -> YOLO txt files + data.yaml (+ previews/ with the GT drawn on 16 training images)
  train  trains (Ultralytics writes weights/, results.csv, curves into runs_yolo/<task>_seed<S>/)
  eval   predicts the TEST split, scores it, writes per-image CSV, predicted labels, and
         STAGE FILES (input -> all detections -> chosen corners -> squared ROI vs GT -> crops)

Example (Windows, seed 42, pose):
  pip install ultralytics
  python baseline_yolo.py prep  --task pose --seed 42 --images_dir ../data/casia/images ^
         --labels_dir out\\labels_oriented\\seed42 --out_root out\\yolo
  python baseline_yolo.py train --task pose --seed 42 --out_root out\\yolo --epochs 50 --batch 16
  python baseline_yolo.py eval  --task pose --seed 42 --out_root out\\yolo ^
         --images_dir ../data/casia/images --labels_dir out\\labels_oriented\\seed42 ^
         --masks_dir ../data/casia/stage1_masks --results_csv runs\\results_baselines.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from rx_common import index_dir, label_to_px, load_json, save_json, square_corners, stem
from rx_stages import (BLUE, CYAN, GREEN, MAGENTA, ORANGE, RED, YELLOW, append_result, draw_poly, oriented_crop,
                       pick_for_dump, put_text, save_stages, score, summarize, to_bgr, write_rows)

SPLITS = ("train", "val", "test")


def paths(a):
    root = Path(a.out_root) / f"{a.task}_seed{a.seed}"
    tag = getattr(a, "tag", "") or ""
    return root, root / "data.yaml", Path(a.runs_dir) / f"{a.task}{tag}_seed{a.seed}"


# ------------------------------------------------------------------------- prep
def cmd_prep(a):
    root, yaml_path, _ = paths(a)
    idx = index_dir(Path(a.images_dir))
    log = {"clipped_corner_labels": 0, "labels": {}}
    for sp in SPLITS:
        lab = load_json(Path(a.labels_dir) / f"{sp}_labels.json")
        (root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (root / "labels" / sp).mkdir(parents=True, exist_ok=True)
        n = 0
        for f, l in sorted(lab.items()):
            if stem(f) not in idx:
                continue
            src = idx[stem(f)]
            dst = root / "images" / sp / (stem(f) + src.suffix.lower())
            if not dst.exists():
                try:
                    os.link(src, dst)                         # no extra disk space when possible
                except OSError:
                    shutil.copy2(src, dst)
            im = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            H, W = im.shape
            cx, cy, side, th = label_to_px(l, W, H)
            c = square_corners(cx, cy, side, th)              # TL, TR, BR, BL (rotated by theta)
            cn = c / np.array([W, H])
            if (cn < 0).any() or (cn > 1).any():
                log["clipped_corner_labels"] += 1
            cn = np.clip(cn, 0.0, 1.0)
            if a.task == "pose":
                x0, y0 = cn.min(0)
                x1, y1 = cn.max(0)
                kp = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in cn)
                line = f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f} {kp}"
            else:
                line = "0 " + " ".join(f"{v:.6f}" for v in cn.reshape(-1))
            (root / "labels" / sp / (stem(f) + ".txt")).write_text(line + "\n")
            n += 1
        log["labels"][sp] = n
        print(f"  {sp}: {n} images + labels")
    names = "names:\n  0: palm\n"
    extra = "kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\n" if a.task == "pose" else ""
    yaml_path.write_text(f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                         f"{extra}{names}")
    save_json(log, root / "conversion_log.json")
    # previews: the converted labels drawn back on the images (checks the conversion by eye)
    pv = root / "previews"
    pv.mkdir(exist_ok=True)
    files = sorted((root / "images" / "train").iterdir())[::max(1, len(list((root / "images" / "train").iterdir())) // 16)][:16]
    for p in files:
        im = to_bgr(cv2.imread(str(p), cv2.IMREAD_COLOR))
        H, W = im.shape[:2]
        v = np.array(list(map(float, (root / "labels" / "train" / (p.stem + ".txt")).read_text().split()))[1:])
        pts = (v[4:].reshape(4, 3)[:, :2] if a.task == "pose" else v.reshape(4, 2)) * np.array([W, H])
        draw_poly(im, pts, GREEN, 2)
        for j, q in enumerate(pts):
            put_text(im, str(j + 1), (int(q[0]) + 4, int(q[1]) - 4), YELLOW, 0.6, 1)
        cv2.imwrite(str(pv / f"{p.stem}.jpg"), im)
    print(f"[DONE] {yaml_path}\n       clipped labels (corner outside image): {log['clipped_corner_labels']}\n"
          f"       previews (GT corners 1=TL 2=TR 3=BR 4=BL): {pv}")


# ------------------------------------------------------------------------- train
def cmd_train(a):
    from ultralytics import YOLO
    root, yaml_path, runs = paths(a)
    model_name = a.model or (f"yolov8{a.size}-{a.task}" + (".pt" if a.pretrained else ".yaml"))
    print(f"[INFO] model={model_name}  data={yaml_path}  -> {runs}  split seed {a.seed}, training seed {a.seed if a.train_seed is None else a.train_seed}")
    model = YOLO(model_name)
    if a.aug == "strong":
        # scale / shift / small rotation / photometric jitter: the usual remedy for cross-dataset domain shift.
        # Corner keypoints keep their identity under these transforms, so orientation labels stay correct.
        aug = dict(mosaic=0.5, mixup=0.0, copy_paste=0.0, degrees=15.0, translate=0.2, scale=0.6, shear=0.0,
                   perspective=0.0, flipud=0.0, fliplr=0.5, hsv_h=0.0, hsv_s=0.4, hsv_v=0.4, erasing=0.2,
                   close_mosaic=10)
    else:
        aug = dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                   perspective=0.0, flipud=0.0, fliplr=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0,
                   close_mosaic=0)
    print(f"[INFO] augmentation preset: {a.aug}  {aug}")
    model.train(data=str(yaml_path), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=(a.seed if a.train_seed is None else a.train_seed), deterministic=True,
                project=str(runs.parent.resolve()), name=runs.name, exist_ok=True, workers=a.workers,
                device=a.device, pretrained=bool(a.pretrained), plots=True, **aug)
    print(f"[DONE] weights: {runs / 'weights' / 'best.pt'}")


# ------------------------------------------------------------------------- eval
def decode(task, res):
    """-> list of dicts sorted by confidence: {'conf', 'pts' (4x2 pixel corners)}"""
    out = []
    if task == "pose":
        if res.keypoints is None or len(res.boxes) == 0:
            return out
        kp = res.keypoints.xy.cpu().numpy()
        cf = res.boxes.conf.cpu().numpy()
        out = [{"conf": float(c), "pts": k.astype(float)} for c, k in zip(cf, kp)]
    else:
        if res.obb is None or len(res.obb) == 0:
            return out
        out = [{"conf": float(c), "pts": p.astype(float)}
               for c, p in zip(res.obb.conf.cpu().numpy(), res.obb.xyxyxyxy.cpu().numpy())]
    return sorted(out, key=lambda d: -d["conf"])


def to_square(task, pts):
    """(cx, cy, side, theta) of the ROI square from the 4 predicted corners."""
    c = pts.mean(0)
    if task == "pose":                                       # corners are ordered TL, TR, BR, BL
        side = float(np.mean([np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]))
        d = ((pts[1] - pts[0]) + (pts[2] - pts[3])) / 2.0
        th = math.degrees(math.atan2(-d[1], d[0])) % 360.0
    else:                                                    # OBB: up direction unknown -> theta mod 90
        e1, e2 = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1])
        side = float(math.sqrt(e1 * e2))
        d = pts[1] - pts[0]
        th = math.degrees(math.atan2(-d[1], d[0])) % 90.0
    return float(c[0]), float(c[1]), side, float(th)


def predict_in_chunks(model, paths, chunk=32, **kw):
    """Ultralytics 8.3.x pushes a whole list source through the GPU as ONE batch (-> CUDA out of memory
    for thousands of images). Feed it small chunks instead."""
    for i in range(0, len(paths), chunk):
        for r in model.predict(source=paths[i:i + chunk], stream=False, **kw):
            yield r


def cmd_eval(a):
    from ultralytics import YOLO
    root, yaml_path, runs = paths(a)
    w = Path(a.weights) if a.weights else runs / "weights" / "best.pt"
    model = YOLO(str(w))
    te = load_json(Path(a.labels_dir) / "test_labels.json")
    tr = load_json(Path(a.labels_dir) / "train_labels.json")
    img_idx = index_dir(Path(a.images_dir))
    mask_idx = index_dir(Path(a.masks_dir)) if a.masks_dir else {}
    files = sorted(f for f in te if stem(f) in img_idx)
    out = runs / "eval"
    out.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(img_idx[stem(files[0])]), cv2.IMREAD_GRAYSCALE)
    H0, W0 = first.shape
    allp = [label_to_px(l, W0, H0) for l in tr.values()]
    fb = (float(np.median([p[0] for p in allp])), float(np.median([p[1] for p in allp])),
          float(np.median([p[2] for p in allp])), 0.0)
    print(f"[INFO] weights={w}  test images={len(files)}  task={a.task}")

    rows, dets, preds = [], {}, {}
    srcs = [str(img_idx[stem(f)]) for f in files]
    stream = predict_in_chunks(model, srcs, chunk=a.batch, imgsz=a.imgsz, conf=a.conf, max_det=a.max_det,
                               device=a.device, verbose=False)
    for f, res in zip(files, stream):
        d = decode(a.task, res)
        H, W = res.orig_shape
        gt = label_to_px(te[f], W, H)
        mask = cv2.imread(str(mask_idx[stem(f)]), cv2.IMREAD_GRAYSCALE) if stem(f) in mask_idx else None
        failed = not d
        p = fb if failed else to_square(a.task, d[0]["pts"])
        r = score(p, gt, mask, mod90=(a.task == "obb"))
        r.update(file=f, failed=int(failed), conf=0.0 if failed else d[0]["conf"], n_det=len(d))
        rows.append(r)
        dets[f] = d
        preds[f] = {"cx": p[0] / W, "cy": p[1] / H, "size": p[2] / min(H, W), "theta": p[3]}
        if len(rows) % 200 == 0:
            print(f"  {len(rows)}/{len(files)}")
    s = summarize(rows)
    t = a.tag or ""
    method = {"pose": "YOLOv8-pose (4 corners)", "obb": "YOLOv8-OBB"}[a.task] + (" pretrained" if a.pretrained else "") + \
        (" +aug" if t == "_aug" else (f" {t.strip('_')}" if t else ""))
    print(f"[TEST {method}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items())
          + ("   (angle error is modulo 90 deg)" if a.task == "obb" else ""))
    write_rows(out / "per_image.csv", rows)
    save_json(preds, out / "pred_labels.json")
    save_json({"method": method, **s}, out / "summary.json")
    if a.results_csv:
        append_result(a.results_csv, method, a.seed, s)

    sel = files if a.dump_all else pick_for_dump(rows, a.dump_random, a.dump_worst, a.dump_fail, a.seed)
    print(f"[INFO] writing stage files for {len(sel)} images -> {out / 'stages'}")
    by = {r["file"]: r for r in rows}
    for f in sel:
        raw = cv2.imread(str(img_idx[stem(f)]), cv2.IMREAD_COLOR)
        H, W = raw.shape[:2]
        gt = label_to_px(te[f], W, H)
        d = dets[f]
        p = fb if not d else to_square(a.task, d[0]["pts"])
        panels = [("01_input", "raw image (network input)", raw)]
        p2 = raw.copy()
        for j, q in enumerate(d[:5]):
            draw_poly(p2, q["pts"], GREEN if j == 0 else ORANGE, 2)
            put_text(p2, f"{q['conf']:.2f}", (int(q["pts"][0][0]), int(q["pts"][0][1]) - 6), YELLOW, 0.55)
        if not d:
            put_text(p2, "NO DETECTION -> fallback", (10, 30), RED, 0.7, 2)
        panels.append(("02_detections", "all detections (best=green) + conf", p2))
        if d:
            p3 = raw.copy()
            draw_poly(p3, d[0]["pts"], GREEN, 2)
            for j, q in enumerate(d[0]["pts"]):
                put_text(p3, str(j + 1), (int(q[0]) + 5, int(q[1]) - 5), YELLOW, 0.7, 2)
            put_text(p3, "corner order 1=TL 2=TR 3=BR 4=BL" if a.task == "pose" else "OBB polygon (up unknown)",
                     (10, 24), (255, 255, 255), 0.55)
            panels.append(("03_chosen_corners", "best detection, corner order", p3))
        p4 = raw.copy()
        draw_poly(p4, square_corners(*gt), GREEN, 2)
        draw_poly(p4, square_corners(*p), RED, 2)
        put_text(p4, f"IoU={by[f]['iou']:.3f}  (red=pred squared, green=GT)", (10, 24), (255, 255, 255), 0.6)
        panels.append(("04_roi_vs_gt", "squared ROI vs GT", p4))
        if a.task == "pose" or not d:
            panels.append(("05_crops", "crop pred | crop GT",
                           np.hstack([oriented_crop(raw, *p), oriented_crop(raw, *gt)])))
        save_stages(out / "stages", stem(f), panels,
                    {"file": f, "gt": gt, "pred": p, "iou": by[f]["iou"], "n_detections": len(d),
                     "confidences": [q["conf"] for q in d[:5]], "failed": by[f]["failed"]})
    print(f"[DONE] {out}")



def force_fp32():
    """Disable TF32 so every script runs in the same, full-FP32 arithmetic (Ampere GPUs otherwise use TF32 in
    cuDNN convolutions by default, which changes a few pipeline outputs on out-of-domain images)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "train", "eval"):
        q = sub.add_parser(name)
        q.add_argument("--task", choices=["pose", "obb"], required=True)
        q.add_argument("--seed", type=int, default=42)
        q.add_argument("--out_root", default="out/yolo")
        q.add_argument("--runs_dir", default="runs_yolo")
        q.add_argument("--tag", default="", help="suffix of the run folder, e.g. _aug -> runs_yolo/pose_aug_seed42")
        if name == "prep":
            q.add_argument("--images_dir", required=True)
            q.add_argument("--labels_dir", required=True, help="out/labels_oriented/seed<S>")
        if name in ("train", "eval"):
            q.add_argument("--device", default="0", help="'0' (first GPU) or 'cpu'")
            q.add_argument("--imgsz", type=int, default=640)
            q.add_argument("--batch", type=int, default=16)
            q.add_argument("--pretrained", action="store_true", help="start from official .pt weights (default: random init)")
        if name == "train":
            q.add_argument("--epochs", type=int, default=50)
            q.add_argument("--workers", type=int, default=4)
            q.add_argument("--aug", choices=["none", "strong"], default="none",
                           help="strong = scale 0.6, translate 0.2, rotate 15 deg, colour jitter, mosaic 0.5")
            q.add_argument("--size", default="n", help="n/s/m (yolov8n by default)")
            q.add_argument("--model", default=None, help="override model name/path")
            q.add_argument("--train_seed", type=int, default=None,
                           help="training seed (init, order, augmentation); default = --seed. The split always follows --seed")
        if name == "eval":
            q.add_argument("--images_dir", required=True)
            q.add_argument("--labels_dir", required=True)
            q.add_argument("--masks_dir", default=None, help="raw hand masks (for the outside-hand fraction)")
            q.add_argument("--weights", default=None)
            q.add_argument("--conf", type=float, default=0.001)
            q.add_argument("--max_det", type=int, default=5)
            q.add_argument("--results_csv", default=None)
            q.add_argument("--dump_random", type=int, default=20)
            q.add_argument("--dump_worst", type=int, default=12)
            q.add_argument("--dump_fail", type=int, default=12)
            q.add_argument("--dump_all", action="store_true")
    a = p.parse_args()
    force_fp32()
    {"prep": cmd_prep, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
