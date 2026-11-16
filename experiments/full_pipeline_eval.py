#!/usr/bin/env python3
"""
full_pipeline_eval.py — live three-stage evaluation of the whole pipeline, per seed
===================================================================================

Stage III alone was trained and tested on one fixed folder of aligned masks: the masks predicted by
the seed-1337 Stage I of Pipeline-basic, rotated upright with the reference frame angle; the ROI
squares were drawn on them, and Stage III of every seed was trained/tested on that folder.

What this script measures: the complete pipeline, run LIVE for each seed on the test
subjects of that seed, with that seed's own models:

    raw image --Stage I (seed S)--> mask --Stage II (seed S)--> angle a --rotate by -a-->
    aligned mask --Stage III (seed S)--> square (aligned frame) --rotate back by +a-->
    oriented square in the RAW frame  -->  compared with the oriented GT square (raw frame)

i.e. exactly the quantity that YOLOv8-pose predicts, scored with the same code
(rotated-square IoU, centre / side / angle error, outside-hand fraction).

Three variants are computed from the same run, so the difference between them is
attributable:
  V0  Stage III only: GT-frame mask (stage1_masks, seed-1337 Stage I) rotated by the
      frame angle of the labels.            == Stage III-only protocol (sanity check: ~0.905)
  V1  own Stage I mask (seed S) rotated by the frame angle of the labels.
      V0 -> V1 = effect of the seed's own Stage I (and of removing train/test leakage)
  V2  FULL: own Stage I mask + own Stage II predicted angle.
      V1 -> V2 = effect of Stage II (angle error / frame mismatch)

What is saved (out_dir/full_pipeline/seed<S>/)
  per_image.csv          one row per test image: IoU of V0/V1/V2, angles, mask agreement
  summary.json           mean metrics per variant + stage timings + parameter counts
  masks/<stem>.png       Stage I output of every test image (own seed)
  angles.json            Stage II output (degrees) for every test image
  pred_labels_full.json  V2 squares in the label format (reusable for matching)
  stages/<stem>_strip.png and stages/<stem>/0N_*.png   (worst / failures / random, or --dump_all)
      01 raw image | 02 Stage I mask | 03 Stage II: mask rotated by -a (fingers up?) |
      04 aligned image | 05 Stage III: square in aligned frame (red=pred, green=GT) |
      06 raw image with both squares | 07 crop pred | crop GT

Example (Windows):
  python full_pipeline_eval.py ^
     --images_dir ../data/casia/images ^
     --gt_masks_dir ../data/casia/stage1_masks ^
     --labels_root out\\labels_oriented --out_dir out ^
     --results_csv runs\\results_baselines.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from rx_common import (apply_affine, circ_err, index_dir, label_to_px, lb_to_orig, letterbox, load_json, rot_matrix,
                       save_json, square_corners, stem)
from rx_stages import (GREEN, RED, YELLOW, append_result, draw_poly, oriented_crop, pick_for_dump, put_text,
                       save_stages, score, summarize, to_bgr, write_rows)

SEG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
SEG_STD = np.array([0.229, 0.224, 0.225], np.float32)
H0, W0 = 576, 768


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def load_state(model, path, dev):
    raw = torch.load(str(path), map_location=dev, weights_only=False)
    state = raw
    if isinstance(raw, dict):
        state = raw.get("model", raw.get("model_state_dict", raw))
    model.load_state_dict(state, strict=True)
    return model


def build_models(a, seed, dev):
    sys.path.insert(0, str(Path(a.pipeline_dir).resolve()))
    from models.unet_seg import UNetSeg
    from models.rot_net import RotResNet18
    from models.ROINet import PalmROINet
    seg = UNetSeg(encoder_name="mobilenet_v2", encoder_weights=None)
    rot = RotResNet18(mode="reg", num_bins=72, grayscale=True, pretrained=False)
    roi = PalmROINet(pretrained=False)
    if not a.random_weights:
        load_state(seg, a.seg_ckpt_pattern.format(seed=seed), dev)
        load_state(rot, a.rot_ckpt_pattern.format(seed=seed), dev)
        load_state(roi, a.roi_ckpt_pattern.format(seed=seed), dev)
    else:
        print("[WARN] --random_weights: plumbing test only, numbers are meaningless")
    return [m.to(dev).eval() for m in (seg, rot, roi)], RotResNet18


# ----------------------------------------------------------------------- the three stages
@torch.no_grad()
def stage1_mask(seg, rgb, dev, thr=0.5):
    h, w = rgb.shape[:2]
    if (h, w) != (H0, W0):                       # same as LongestMaxSize + PadIfNeeded of the original script
        s = W0 / max(h, w)
        rgb = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_LINEAR)
        pad = np.zeros((H0, W0, 3), np.uint8)
        pad[:rgb.shape[0], :rgb.shape[1]] = rgb
        rgb = pad
    x = (rgb.astype(np.float32) / 255.0 - SEG_MEAN) / SEG_STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(dev)
    p = torch.sigmoid(seg(x))[0, 0].cpu().numpy()
    return ((p >= thr).astype(np.uint8) * 255)


@torch.no_grad()
def stage2_angle(rot, RotCls, mask255, dev):
    x = ((mask255 > 127).astype(np.float32) * 255.0 / 255.0 - 0.5) / 0.5
    x = torch.from_numpy(x)[None, None].to(dev)
    _, vec = rot(x)
    return float(RotCls.vec_to_deg(vec).item())


def rotate_mask(mask255, angle_deg):
    H, W = mask255.shape
    r = cv2.warpAffine(mask255, rot_matrix(-angle_deg, W, H), (W, H), flags=cv2.INTER_NEAREST)
    return ((r > 127).astype(np.uint8) * 255)


@torch.no_grad()
def stage3_square(roi, aligned_mask255, dev, T=256):
    H, W = aligned_mask255.shape
    lb = letterbox((aligned_mask255 > 127).astype(np.uint8) * 255, T, True)
    x = torch.from_numpy(lb.astype(np.float32) / 255.0)[None, None].to(dev)
    p = torch.clamp(roi(x), 0, 1)[0].cpu().numpy()
    return lb_to_orig(float(p[0]), float(p[1]), float(p[2]), H, W, T)       # cx, cy, side (aligned frame, px)


def to_raw(sq_aligned, angle_deg, W, H):
    cx, cy, s = sq_aligned
    (rx, ry), = apply_affine(rot_matrix(+angle_deg, W, H), [[cx, cy]])
    return float(rx), float(ry), float(s), float(angle_deg)


def timed(fn, dev, store, key):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    store.setdefault(key, []).append((time.perf_counter() - t) * 1000.0)
    return out


# ----------------------------------------------------------------------- main loop
def run_seed(a, seed, dev):
    (seg, rot, roi), RotCls = build_models(a, seed, dev)
    te = load_json(Path(a.labels_root) / f"seed{seed}" / "test_labels.json")
    img_idx = index_dir(Path(a.images_dir))
    gtm_idx = index_dir(Path(a.gt_masks_dir))
    files = sorted(f for f in te if stem(f) in img_idx and stem(f) in gtm_idx)
    out = Path(a.out_dir) / f"full_pipeline{a.tag}" / f"seed{seed}"
    (out / "masks").mkdir(parents=True, exist_ok=True)
    print(f"[seed {seed}] {len(files)} test images | params seg={count_params(seg):,} rot={count_params(rot):,} "
          f"roi={count_params(roi):,}")
    rows, cache, angles, tms = [], {}, {}, {}
    for i, f in enumerate(files):
        s = stem(f)
        rgb = cv2.cvtColor(cv2.imread(str(img_idx[s]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        gt = label_to_px(te[f], W, H)                       # (cx, cy, side, theta_frame) raw frame
        gtm = cv2.imread(str(gtm_idx[s]), cv2.IMREAD_GRAYSCALE)
        # ---- Stage I (own seed), Stage II (own seed), Stage III (own seed)
        m_own = timed(lambda: stage1_mask(seg, rgb, dev), dev, tms, "stage1_ms")
        ang = timed(lambda: stage2_angle(rot, RotCls, m_own, dev), dev, tms, "stage2_ms")
        al_mask = rotate_mask(m_own, ang)
        sq = timed(lambda: stage3_square(roi, al_mask, dev), dev, tms, "stage3_ms")
        full = to_raw(sq, ang, W, H)                                                    # V2
        # ---- V1: own mask, frame angle of the label
        sq1 = stage3_square(roi, rotate_mask(m_own, gt[3]), dev)
        v1 = to_raw(sq1, gt[3], W, H)
        # ---- V0: GT-frame mask (Stage III-only protocol)
        sq0 = stage3_square(roi, rotate_mask(gtm, gt[3]), dev)
        v0 = to_raw(sq0, gt[3], W, H)
        r0, r1, r2 = (score(p, gt, gtm) for p in (v0, v1, full))
        inter = ((m_own > 127) & (gtm > 127)).sum()
        uni = ((m_own > 127) | (gtm > 127)).sum()
        rows.append({"file": f, "iou": r2["iou"], "iou_V0": r0["iou"], "iou_V1": r1["iou"], "iou_V2": r2["iou"],
                     "center_err_px": r2["center_err_px"], "size_err_px": r2["size_err_px"],
                     "angle_err_deg": r2["angle_err_deg"], "outside_frac": r2["outside_frac"],
                     "pred_angle": ang, "gt_frame_angle": gt[3], "mask_iou_vs_gtmask": float(inter / max(1, uni)),
                     "failed": int(r2["iou"] < 0.5)})
        cache[f] = (rgb, m_own, ang, al_mask, sq, full, gt, v0, v1, r0, r1)
        cv2.imwrite(str(out / "masks" / f"{s}.png"), m_own)
        angles[f] = ang
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}")
    # ---- summaries
    def summ(key_iou, extra=None):
        d = [{"iou": r[key_iou], "center_err_px": r["center_err_px"], "size_err_px": r["size_err_px"],
              "angle_err_deg": r["angle_err_deg"], "outside_frac": r["outside_frac"], "failed": r["failed"]}
             for r in rows]
        return summarize(d)
    rv = {}
    for name, key in (("V0 Stage III only (given masks+frame, Table 3)", "iou_V0"),
                      ("V1 own Stage I mask + frame angle", "iou_V1"),
                      ("V2 FULL pipeline (live Stage I+II+III)", "iou_V2")):
        d = [{"iou": r[key], "center_err_px": 0.0, "size_err_px": 0.0, "angle_err_deg": 0.0} for r in rows]
        rv[name] = summarize(d)
    full_s = summ("iou_V2")
    rv["V2 FULL pipeline (live Stage I+II+III)"] = full_s
    print("  mIoU  " + "   ".join(f"{k[:2]}={v['mIoU']:.4f}" for k, v in rv.items()))
    print(f"  FULL: Acc@0.7={full_s['acc_0_7']:.3f}  ctr={full_s['center_err_px']:.1f}px  size={full_s['size_err_px']:.1f}px  "
          f"Stage-II angle err (vs frame angle)={full_s['angle_err_deg']:.2f} deg  outside={full_s['outside_frac']:.4f}")
    print(f"  Stage I mask IoU vs generated_masks: {np.mean([r['mask_iou_vs_gtmask'] for r in rows]):.4f}")
    t = {k: float(np.mean(v[5:])) if len(v) > 10 else float(np.mean(v)) for k, v in tms.items()}
    print(f"  mean time per image (ms): " + "  ".join(f"{k}={v:.1f}" for k, v in t.items()))
    write_rows(out / "per_image.csv", rows)
    save_json(angles, out / "angles.json")
    save_json({f: {"cx": c[5][0] / W0, "cy": c[5][1] / H0, "size": c[5][2] / min(H0, W0), "theta": c[5][3]}
               for f, c in cache.items()}, out / "pred_labels_full.json")
    save_json({"variants": rv, "ms_per_image": t,
               "params": {"stage1_unet_mobilenetv2": count_params(seg), "stage2_unirotnet_resnet18": count_params(rot),
                          "stage3_lite_palmroi": count_params(roi)}}, out / "summary.json")
    if a.results_csv:
        lab = a.method_label
        append_result(a.results_csv, f"Pipeline FULL{lab} (live I+II+III)", seed, full_s)
        append_result(a.results_csv, f"Stage III{lab} only (given masks+frame)", seed, rv["V0 Stage III only (given masks+frame, Table 3)"])
        append_result(a.results_csv, f"Stage III{lab}, own Stage-I mask + frame angle", seed, rv["V1 own Stage I mask + frame angle"])

    # ---- stage files
    sel = files if a.dump_all else pick_for_dump(rows, a.dump_random, a.dump_worst, a.dump_fail, seed)
    print(f"  writing stage files for {len(sel)} images -> {out / 'stages'}")
    by = {r["file"]: r for r in rows}
    for f in sel:
        rgb, m_own, ang, al_mask, sq, full, gt, v0, v1, r0, r1 = cache[f]
        H, W = m_own.shape
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        p1 = to_bgr(m_own)
        cv2.drawContours(p1, cv2.findContours(((cache[f][1] > 127).astype(np.uint8)), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 255, 0), 1)
        p1 = np.hstack([p1]) if False else p1
        put_text(p1, f"Stage I mask  (IoU vs generated mask {by[f]['mask_iou_vs_gtmask']:.3f})", (8, 24), YELLOW, 0.55)
        p2 = to_bgr(al_mask)
        put_text(p2, f"Stage II: pred angle={ang:.1f}  frame angle={gt[3]:.1f}  err={by[f]['angle_err_deg']:.1f}", (8, 24), YELLOW, 0.55)
        al_rgb = cv2.warpAffine(bgr, rot_matrix(-ang, W, H), (W, H))
        p3 = al_rgb.copy()
        draw_poly(p3, square_corners(sq[0], sq[1], sq[2], 0.0), RED, 2)
        gtc = apply_affine(rot_matrix(-ang, W, H), [[gt[0], gt[1]]])[0]        # GT square seen in the predicted frame
        draw_poly(p3, square_corners(gtc[0], gtc[1], gt[2], gt[3] - ang), GREEN, 2)
        put_text(p3, "Stage III: red=pred (aligned frame), green=GT", (8, 24), (255, 255, 255), 0.55)
        p4 = bgr.copy()
        draw_poly(p4, square_corners(*gt), GREEN, 2)
        draw_poly(p4, square_corners(*full), RED, 2)
        put_text(p4, f"FULL IoU={by[f]['iou_V2']:.3f}  V1={by[f]['iou_V1']:.3f}  V0={by[f]['iou_V0']:.3f}", (8, 24),
                 (255, 255, 255), 0.55)
        panels = [("01_raw_image", "raw image", bgr), ("02_stage1_mask", "Stage I: mask", p1),
                  ("03_stage2_aligned_mask", "Stage II: mask rotated by -a", p2),
                  ("04_aligned_image", "aligned image (rotated by -a)", al_rgb),
                  ("05_stage3_square", "Stage III square (aligned frame)", p3),
                  ("06_raw_with_squares", "back in raw frame: pred vs GT", p4),
                  ("07_crops", "crop pred | crop GT", np.hstack([oriented_crop(bgr, *full), oriented_crop(bgr, *gt)]))]
        save_stages(out / "stages", stem(f), panels,
                    {"file": f, "pred_angle": ang, "gt_frame_angle": gt[3], "gt": gt, "pred_full": full,
                     "iou": {"V0": by[f]["iou_V0"], "V1": by[f]["iou_V1"], "V2": by[f]["iou_V2"]},
                     "mask_iou_vs_gtmask": by[f]["mask_iou_vs_gtmask"]})



def force_fp32():
    """Disable TF32 so every script runs in the same, full-FP32 arithmetic (Ampere GPUs otherwise use TF32 in
    cuDNN convolutions by default, which changes a few pipeline outputs on out-of-domain images)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--gt_masks_dir", required=True, help="generated_masks (seed-1337 Stage I masks; used for V0 and outside_frac)")
    p.add_argument("--labels_root", required=True, help="out/labels_oriented")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pipeline_dir", default=r"../pipeline_models", help="folder that contains models/")
    p.add_argument("--seg_ckpt_pattern", default=r"../checkpoints/basic/stage1_seg_mobilenetv2_seed{seed}.pt")
    p.add_argument("--rot_ckpt_pattern", default=r"../checkpoints/basic/stage2_unirot_angle_seed{seed}.pt")
    p.add_argument("--roi_ckpt_pattern", default=r"../checkpoints/basic/stage3_roi_seed{seed}.pth")
    p.add_argument("--seeds", default="42,2025,1337")
    p.add_argument("--results_csv", default=None)
    p.add_argument("--tag", default="", help="output folder suffix, e.g. _aug -> <out_dir>/full_pipeline_aug")
    p.add_argument("--method_label", default="", help="appended to the method name in results_csv, e.g. ' +aug'")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dump_random", type=int, default=20)
    p.add_argument("--dump_worst", type=int, default=12)
    p.add_argument("--dump_fail", type=int, default=12)
    p.add_argument("--dump_all", action="store_true")
    p.add_argument("--random_weights", action="store_true", help="plumbing test only (no checkpoints)")
    a = p.parse_args()
    force_fp32()
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    for s in a.seeds.split(","):
        run_seed(a, int(s), dev)


if __name__ == "__main__":
    main()
