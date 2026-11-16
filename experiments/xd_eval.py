#!/usr/bin/env python3
"""
xd_eval.py — zero-shot cross-dataset test (VERA, SCUT-PV) of YOLOv8-pose and the full 3-stage pipeline
=====================================================================================================
Everything is trained on CASIA-MS only and applied unchanged. References = the hand-clicked labels
made with xd_label.py (labels_manual.json, CASIA frame). Same scoring code as Tables 3/baselines.

Also reports 'casia' (if prepared): human click-label vs CASIA GT  -> how far a human label is from the dataset
convention (a ceiling for what IoU can mean across datasets).

SAVED (out_dir = <cross_dir>\\results\\<ds>):
  per_image_<method>_seed<S>.csv      one row per labelled image
  summary.json / ..\\results_cross.csv   mean metrics per method and seed
  stages\\<name>_strip.png + stages\\<name>\\0N_*.png   every processing step of every labelled image (seed --dump_seed)
     01 image + hand label (green) + YOLO (red) | 02 Stage I mask | 03 Stage II aligned mask |
     04 aligned image + Stage III square | 05 CASIA frame: pipeline (blue) vs label (green) | 06 crops label|YOLO|pipeline
"""
from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

from rx_common import label_to_px, load_json, rot_matrix, save_json, square_corners, apply_affine
from rx_stages import (BLUE, GREEN, RED, YELLOW, append_result, draw_poly, oriented_crop, put_text, save_stages,
                       score, summarize, to_bgr, write_rows)

W0, H0 = 768, 576


def load_refs(ds_dir: Path, name="labels_manual.json"):
    lab = load_json(ds_dir / name)
    man = load_json(ds_dir / "manifest.json")
    return {n: label_to_px(l, W0, H0) for n, l in lab.items() if not l.get("skip") and n in man}


def run_yolo(a, seed, names, ds_dir):
    from ultralytics import YOLO
    from baseline_yolo import decode, predict_in_chunks, to_square
    w = a.yolo_weights.format(seed=seed)
    model = YOLO(w)
    srcs = [str(ds_dir / "images" / f"{n}.png") for n in names]
    out = {}
    for n, res in zip(names, predict_in_chunks(model, srcs, chunk=16, imgsz=640, conf=0.001, max_det=5,
                                                device=a.device, verbose=False)):
        d = decode("pose", res)
        out[n] = (to_square("pose", d[0]["pts"]) + (d[0]["conf"], len(d))) if d else None
    return out


def run_pipeline(a, seed, names, ds_dir, dev):
    import full_pipeline_eval as fp
    ns = Namespace(pipeline_dir=a.pipeline_dir, seg_ckpt_pattern=a.seg_ckpt_pattern,
                   rot_ckpt_pattern=a.rot_ckpt_pattern, roi_ckpt_pattern=a.roi_ckpt_pattern, random_weights=False)
    (seg, rot, roi), RotCls = fp.build_models(ns, seed, dev)
    out = {}
    for n in names:
        rgb = cv2.cvtColor(cv2.imread(str(ds_dir / "images" / f"{n}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        m = fp.stage1_mask(seg, rgb, dev)
        ang = fp.stage2_angle(rot, RotCls, m, dev)
        al = fp.rotate_mask(m, ang)
        sq = fp.stage3_square(roi, al, dev)
        out[n] = {"mask": m, "angle": ang, "aligned_mask": al, "sq": sq, "full": fp.to_raw(sq, ang, W0, H0)}
    return out


def dump_stage(out_dir, n, ds_dir, ref, y, p):
    bgr = cv2.cvtColor(cv2.imread(str(ds_dir / "images" / f"{n}.png"), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    p1 = bgr.copy()
    draw_poly(p1, square_corners(*ref), GREEN, 2)
    if y is not None:
        draw_poly(p1, square_corners(*y[:4]), RED, 2)
    put_text(p1, "green=hand label  red=YOLO-pose", (8, 24), YELLOW, 0.55)
    p2 = to_bgr(p["mask"])
    put_text(p2, "Stage I mask", (8, 24), YELLOW, 0.55)
    p3 = to_bgr(p["aligned_mask"])
    put_text(p3, f"Stage II: rotated by -{p['angle']:.1f} deg (fingers should point UP)", (8, 24), YELLOW, 0.55)
    al = cv2.warpAffine(bgr, rot_matrix(-p["angle"], W0, H0), (W0, H0))
    p4 = al.copy()
    draw_poly(p4, square_corners(p["sq"][0], p["sq"][1], p["sq"][2], 0.0), BLUE, 2)
    put_text(p4, "Stage III square in aligned frame", (8, 24), YELLOW, 0.55)
    p5 = bgr.copy()
    draw_poly(p5, square_corners(*ref), GREEN, 2)
    draw_poly(p5, square_corners(*p["full"]), BLUE, 2)
    put_text(p5, "green=label  blue=pipeline (back in image frame)", (8, 24), YELLOW, 0.55)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    crops = [oriented_crop(gray, *ref), oriented_crop(gray, *p["full"])]
    if y is not None:
        crops.insert(1, oriented_crop(gray, *y[:4]))
    p6 = cv2.resize(np.hstack(crops), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    save_stages(out_dir / "stages", n,
                [("01_label_and_yolo", "label(g) YOLO(r)", p1), ("02_stage1_mask", "Stage I", p2),
                 ("03_stage2_aligned_mask", "Stage II aligned", p3), ("04_stage3_aligned_image", "Stage III", p4),
                 ("05_pipeline_vs_label", "pipeline(b) vs label(g)", p5), ("06_crops", "crops: label|YOLO|pipeline", p6)],
                {"name": n, "label": ref, "yolo": y, "pipeline": p["full"], "pipeline_angle": p["angle"]})



def force_fp32():
    """Disable TF32 so every script runs in the same, full-FP32 arithmetic (Ampere GPUs otherwise use TF32 in
    cuDNN convolutions by default, which changes a few pipeline outputs on out-of-domain images)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cross_dir", required=True)
    ap.add_argument("--datasets", default="vera,scut")
    ap.add_argument("--seeds", default="42,2025,1337")
    ap.add_argument("--yolo_weights", default=r"runs_yolo\pose_seed{seed}\weights\best.pt")
    ap.add_argument("--pipeline_dir", default=r"../pipeline_models")
    ap.add_argument("--seg_ckpt_pattern", default=r"../checkpoints/basic/stage1_seg_mobilenetv2_seed{seed}.pt")
    ap.add_argument("--rot_ckpt_pattern", default=r"../checkpoints/basic/stage2_unirot_angle_seed{seed}.pt")
    ap.add_argument("--roi_ckpt_pattern", default=r"../checkpoints/basic/stage3_roi_seed{seed}.pth")
    ap.add_argument("--results_csv", default=None)
    ap.add_argument("--yolo_name", default="YOLOv8-pose", help="label used in the results, e.g. 'YOLOv8-pose +aug'")
    ap.add_argument("--yolo_tag", default="", help="file suffix for per-image csv, e.g. _aug")
    ap.add_argument("--yolo_only", action="store_true", help="skip the pipeline (already evaluated)")
    ap.add_argument("--pipeline_only", action="store_true", help="skip YOLO (e.g. for a re-trained pipeline)")
    ap.add_argument("--pipeline_tag", default="", help="per_image_pipeline<tag>_seed<S>.csv, e.g. _aug")
    ap.add_argument("--skip_casia_check", action="store_true", help="skip the human-vs-GT CASIA check")
    ap.add_argument("--dump_seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    force_fp32()
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    cross = Path(a.cross_dir)
    res_csv = a.results_csv or str(cross / "results" / "results_cross.csv")

    # ---- CASIA sanity: human click label vs dataset GT
    cdir = cross / "casia"
    if (cdir / "labels_manual.json").is_file() and not a.skip_casia_check:
        man = load_json(cdir / "manifest.json")
        refs = load_refs(cdir)
        rows = []
        for n, r in refs.items():
            gt = label_to_px(man[n]["gt"], W0, H0)
            s = score(r, gt)
            s["file"] = n
            rows.append(s)
        sm = summarize(rows)
        print(f"[casia] human click-label vs CASIA GT: n={sm['n']} mIoU={sm['mIoU']:.4f} Acc@0.7={sm['acc_0_7']:.3f} "
              f"ctr={sm['center_err_px']:.1f}px size={sm['size_err_px']:.1f}px ang={sm['angle_err_deg']:.2f}deg")
        (cross / "results" / "casia").mkdir(parents=True, exist_ok=True)
        write_rows(cross / "results" / "casia" / "human_vs_GT.csv", rows)
        if (cdir / "labels_manual_pass2.json").is_file():
            from rx_common import polygon_iou
            r2 = load_refs(cdir, "labels_manual_pass2.json")
            common = sorted(set(refs) & set(r2))
            gts = {n: label_to_px(man[n]["gt"], W0, H0) for n in common}
            rep = [polygon_iou(square_corners(*refs[n]), square_corners(*r2[n])) for n in common]
            p2 = [polygon_iou(square_corners(*r2[n]), square_corners(*gts[n])) for n in common]
            avg = []
            for n in common:
                c = tuple((np.array(refs[n][:3]) + np.array(r2[n][:3])) / 2) + (refs[n][3],)
                avg.append(polygon_iou(square_corners(*c), square_corners(*gts[n])))
            print(f"[casia] repeatability (pass1 vs pass2, same person): IoU={np.mean(rep):.4f}  "
                  f"pass2 vs GT={np.mean(p2):.4f}  mean-of-passes vs GT={np.mean(avg):.4f}  (n={len(common)})")
        append_result(res_csv, "casia: human label vs GT", 0, sm)

    for ds in [d for d in a.datasets.split(",") if d]:
        ds_dir = cross / ds
        refs = load_refs(ds_dir)
        names = sorted(refs)
        out = cross / "results" / ds
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {ds}: {len(names)} labelled images ===")
        for seed in [int(s) for s in a.seeds.split(",")]:
            yo = {n: None for n in names} if a.pipeline_only else run_yolo(a, seed, names, ds_dir)
            pl = None if a.yolo_only else run_pipeline(a, seed, names, ds_dir, dev)
            rows_y, rows_p = [], []
            for n in names:
                ry = score(yo[n][:4], refs[n]) if yo[n] else {"iou": 0.0, "center_err_px": 1e3, "size_err_px": 1e3,
                                                              "angle_err_deg": 180.0}
                ry.update(file=n, failed=int(yo[n] is None or ry["iou"] < 0.5),
                          conf=yo[n][4] if yo[n] else 0.0, n_det=yo[n][5] if yo[n] else 0)
                rows_y.append(ry)
                if pl is not None:
                    rp = score(pl[n]["full"], refs[n])
                    rp.update(file=n, failed=int(rp["iou"] < 0.5), pred_angle=pl[n]["angle"])
                    rows_p.append(rp)
            if a.pipeline_only:
                rows_y = []
            else:
                write_rows(out / f"per_image_yolo{a.yolo_tag}_seed{seed}.csv", rows_y)
            if pl is not None:
                write_rows(out / f"per_image_pipeline{a.pipeline_tag}_seed{seed}.csv", rows_p)
            for meth, rows in ((a.yolo_name, rows_y), (f"Pipeline FULL{a.pipeline_tag}", rows_p)):
                if not rows:
                    continue
                sm = summarize(rows)
                print(f"  seed {seed}  {meth:14s} mIoU={sm['mIoU']:.4f} Acc@0.7={sm['acc_0_7']:.3f} "
                      f"ctr={sm['center_err_px']:.1f}px size={sm['size_err_px']:.1f}px ang={sm['angle_err_deg']:.2f}deg "
                      f"failed={sm['n_failed']}")
                append_result(res_csv, f"{ds}: {meth}", seed, sm)
            if seed == a.dump_seed and pl is not None:
                dout = out if not a.pipeline_tag else out / f"pipeline{a.pipeline_tag}"
                print(f"  writing stage files for {len(names)} images -> {dout / 'stages'}")
                for n in names:
                    dump_stage(dout, n, ds_dir, refs[n], yo[n], pl[n])
                    cv2.imwrite(str(dout / "stages" / n / "mask.png"), pl[n]["mask"])
    print(f"\nresults -> {res_csv}")


if __name__ == "__main__":
    main()
