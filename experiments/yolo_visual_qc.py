#!/usr/bin/env python3
"""
yolo_visual_qc.py — run the trained YOLOv8-pose ROI detector on EVERY raw image and
write the intermediate results of each step to separate folders, so that every ROI,
every rotation and every crop can be checked by eye.
====================================================================================

Steps and folders (out_dir/yolo_all/):
  01_roi_on_raw/    the RAW image with the predicted ROI square drawn on it
                    (red square, corners numbered 1=TL 2=TR 3=BR 4=BL, yellow arrow = "up"
                    = the direction the fingers should point; green square = GT label if given)
  02_aligned_view/  the image ROTATED so that the fingers point up, shifted so that the ROI is in
                    the centre of the canvas, with the (now axis-aligned) ROI drawn.
                    -> the palm should be upright and the red square should sit on the palm.
  03_crops/         the final upright ROI crops (square, --crop_size px). THESE are the files used
                    for matching; they are what you inspect.
  04_contact_sheets/ grids of crops (with file names) for fast visual review: sheet_0001.jpg ...
  05_flagged/       3-panel strips (raw+ROI | aligned view | crop) of every image that fails an
                    automatic check (see qc_flags.csv). Start the review here.
  predictions.csv   per image: ROI (cx, cy, side, theta), confidence, number of detections,
                    split (train/val/test of the model that was used), IoU / angle error vs GT, flags
  qc_flags.csv      only the flagged images and why
  pred_labels.json  predicted squares in the label format (for the matching scripts)

Automatic checks (a flag does NOT mean the crop is wrong, it means "look at it"):
  low_conf          best detection confidence < --min_conf
  multi_det         a second detection with high confidence at a different place
  roi_out_of_image  more than 2% of the ROI square lies outside the image
  odd_size          side outside the 1st-99th percentile of the GT sides (or 0.15-0.60 of min(H,W))
  not_square        the four predicted edges differ by more than 15%
  mirrored_corners  the corners 1-2-3-4 run counter-clockwise (up/down or left/right mirrored)
  outside_hand      more than 5% of the ROI lies outside the hand mask (if --masks_dir)
  low_iou_gt        IoU with the GT label < 0.7            (if --labels_dir)
  angle_vs_gt       angle differs from the GT frame angle by more than 10 deg (if --labels_dir)

Model choice: --seeds 42 uses the seed-42 model for all images. With several seeds, an image
is processed by the first seed in the list for which it is in the TEST split (so the crop comes
from a model that never saw that subject); otherwise by the first seed in the list.
predictions.csv has a 'split' column: only 'test' rows are honest generalisation.

Example (Windows):
  python yolo_visual_qc.py --images_dir ..\\data\\casia\\images ^
     --labels_root out\\labels_oriented --masks_dir ..\\data\\casia\\stage1_masks ^
     --out_dir out --seeds 42
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from baseline_yolo import decode, to_square
from rx_common import circ_err, index_dir, label_to_px, load_json, rot_matrix, apply_affine, save_json, square_corners, stem
from rx_stages import GREEN, RED, YELLOW, draw_poly, hand_side, oriented_crop, polygon_iou, put_text, titled, to_bgr, write_rows


def predict_in_chunks(model, paths, chunk=32, **kw):
    """Ultralytics 8.3.x pushes a whole list source through the GPU as ONE batch (-> CUDA out of memory
    for thousands of images). Feed it small chunks instead."""
    for i in range(0, len(paths), chunk):
        for r in model.predict(source=paths[i:i + chunk], stream=False, **kw):
            yield r


def frame_up(theta_deg):
    a = math.radians(theta_deg)
    return np.array([-math.sin(a), -math.cos(a)])           # image direction of "fingers up" (see rx_common)


def inside_fraction_outside_image(poly, W, H):
    c = np.zeros((H, W), np.uint8)
    full = np.round(poly).astype(np.int32)
    big = np.zeros((H * 3, W * 3), np.uint8)
    cv2.fillConvexPoly(big, full + np.array([W, H]), 1)
    area = big.sum()
    return 1.0 - (big[H:2 * H, W:2 * W].sum() / area) if area else 1.0


def aligned_view(img, cx, cy, theta):
    """Rotate by -theta about the ROI centre and move the ROI centre to the canvas centre."""
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), -theta, 1.0)
    M[0, 2] += W / 2.0 - cx
    M[1, 2] += H / 2.0 - cy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def draw_raw(img, pts, sq, gt):
    im = to_bgr(img)
    if gt is not None:
        draw_poly(im, square_corners(*gt), GREEN, 2)
    draw_poly(im, square_corners(*sq), RED, 3)
    for j, q in enumerate(square_corners(*sq)):
        put_text(im, str(j + 1), (int(q[0]) + 5, int(q[1]) - 5), YELLOW, 0.7, 2)
    u = frame_up(sq[3])
    c = np.array(sq[:2])
    cv2.arrowedLine(im, tuple(np.round(c).astype(int)), tuple(np.round(c + u * sq[2] * 0.9).astype(int)),
                    (0, 255, 255), 3, tipLength=0.25)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--out_dir", default="out")
    p.add_argument("--labels_root", default=None, help="out/labels_oriented (adds GT, split and IoU)")
    p.add_argument("--masks_dir", default=None, help="raw hand masks (for the outside_hand check)")
    p.add_argument("--seeds", default="42")
    p.add_argument("--runs_dir", default="runs_yolo")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=32, help="images per GPU call (chunk size)")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--min_conf", type=float, default=0.5)
    p.add_argument("--sheet_cols", type=int, default=8)
    p.add_argument("--sheet_rows", type=int, default=6)
    p.add_argument("--limit", type=int, default=0, help="process only the first N images (test run)")
    p.add_argument("--no_gt_overlay", action="store_true", help="do not draw GT squares even if labels are given")
    a = p.parse_args()

    from ultralytics import YOLO
    seeds = [int(s) for s in a.seeds.split(",")]
    img_idx = index_dir(Path(a.images_dir))
    stems = sorted(img_idx)
    if a.limit:
        stems = stems[:a.limit]
    mask_idx = index_dir(Path(a.masks_dir)) if a.masks_dir else {}
    # labels / split per seed
    lab, split = {}, {s: {} for s in seeds}
    if a.labels_root:
        for s in seeds:
            for sp in ("train", "val", "test"):
                d = load_json(Path(a.labels_root) / f"seed{s}" / f"{sp}_labels.json")
                for f, l in d.items():
                    split[s][stem(f)] = sp
                    lab.setdefault(stem(f), {})[s] = l
    # which model processes which image
    choice = {}
    for st in stems:
        pick = next((s for s in seeds if split[s].get(st) == "test"), seeds[0])
        choice[st] = pick
    sides_all = [l["size"] for st, d in lab.items() for s, l in d.items()] if lab else []
    lo_hi = (np.percentile(sides_all, 1), np.percentile(sides_all, 99)) if sides_all else (0.15, 0.60)

    out = Path(a.out_dir) / "yolo_all"
    dirs = {k: out / k for k in ("01_roi_on_raw", "02_aligned_view", "03_crops", "04_contact_sheets", "05_flagged")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    rows, pred_labels = [], {}
    crops_for_sheet = []
    for s in seeds:
        todo = [st for st in stems if choice[st] == s]
        if not todo:
            continue
        w = Path(a.runs_dir) / f"pose_seed{s}" / "weights" / "best.pt"
        print(f"[INFO] seed {s}: {len(todo)} images, model {w}")
        model = YOLO(str(w))
        stream = predict_in_chunks(model, [str(img_idx[t]) for t in todo], chunk=a.batch, imgsz=a.imgsz, conf=0.001,
                                   max_det=5, device=a.device, verbose=False)
        for n, (st, res) in enumerate(zip(todo, stream)):
            raw = cv2.imread(str(img_idx[st]), cv2.IMREAD_COLOR)
            H, W = raw.shape[:2]
            dets = decode("pose", res)
            flags = []
            if not dets:
                sq, conf, ndet = (W / 2, H / 2, 0.3 * min(H, W), 0.0), 0.0, 0
                pts = square_corners(*sq)
                flags.append("no_detection")
            else:
                pts = dets[0]["pts"]
                sq = to_square("pose", pts)
                conf, ndet = dets[0]["conf"], len(dets)
                if conf < a.min_conf:
                    flags.append("low_conf")
                if len(dets) > 1 and dets[1]["conf"] > 0.3 and \
                        np.linalg.norm(dets[1]["pts"].mean(0) - pts.mean(0)) > 0.3 * sq[2]:
                    flags.append("multi_det")
            poly = square_corners(*sq)
            if inside_fraction_outside_image(poly, W, H) > 0.02:
                flags.append("roi_out_of_image")
            sn = sq[2] / min(H, W)
            if sn < lo_hi[0] * 0.8 or sn > lo_hi[1] * 1.2:
                flags.append("odd_size")
            if dets:
                edges = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
                if np.std(edges) / (np.mean(edges) + 1e-9) > 0.15:
                    flags.append("not_square")
                e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
                cross = e1[0] * e2[1] - e1[1] * e2[0]
                if cross <= 0:                            # TL->TR->BR must run clockwise in image coords (y down)
                    flags.append("mirrored_corners")
            if st in mask_idx:
                m = cv2.imread(str(mask_idx[st]), cv2.IMREAD_GRAYSCALE)
                cv = np.zeros((H, W), np.uint8)
                cv2.fillConvexPoly(cv, np.round(poly).astype(np.int32), 1)
                tot = max(1, int(cv.sum()))
                ofrac = 1.0 - int(((m > 127) & (cv > 0)).sum()) / tot
                if ofrac > 0.05:
                    flags.append("outside_hand")
            else:
                ofrac = float("nan")
            gt = iou = aerr = None
            if st in lab and s in lab[st]:
                gt = label_to_px(lab[st][s], W, H)
                iou = polygon_iou(poly, square_corners(*gt))
                aerr = circ_err(sq[3], gt[3])
                if iou < 0.7:
                    flags.append("low_iou_gt")
                if aerr > 10:
                    flags.append("angle_vs_gt")
            # ---- 01 ROI on raw
            im1 = draw_raw(raw, pts, sq, None if a.no_gt_overlay else gt)
            put_text(im1, f"{st}  conf={conf:.2f}  theta={sq[3]:.1f}", (8, 24), (255, 255, 255), 0.6)
            cv2.imwrite(str(dirs["01_roi_on_raw"] / f"{st}.jpg"), im1)
            # ---- 02 aligned view
            al = aligned_view(raw, sq[0], sq[1], sq[3])
            im2 = al.copy()
            h = sq[2] / 2
            cv2.rectangle(im2, (int(W / 2 - h), int(H / 2 - h)), (int(W / 2 + h), int(H / 2 + h)), RED, 3)
            if gt is not None and not a.no_gt_overlay:
                gc = apply_affine(cv2.getRotationMatrix2D((sq[0], sq[1]), -sq[3], 1.0) + np.array([[0, 0, W / 2 - sq[0]], [0, 0, H / 2 - sq[1]]]),
                                  [[gt[0], gt[1]]])[0]
                draw_poly(im2, square_corners(gc[0], gc[1], gt[2], gt[3] - sq[3]), GREEN, 2)
            put_text(im2, "aligned view: fingers should point UP, palm in the red square", (8, 24), (255, 255, 255), 0.55)
            cv2.imwrite(str(dirs["02_aligned_view"] / f"{st}.jpg"), im2)
            # ---- 03 crop
            crop = oriented_crop(raw, *sq, out=a.crop_size)
            cv2.imwrite(str(dirs["03_crops"] / f"{st}.png"), crop)
            crops_for_sheet.append((st, crop, bool(flags)))
            if flags:
                strip = np.hstack([titled(im1, "raw + ROI"), titled(im2, "aligned view"), titled(crop, "crop " + ",".join(flags)[:60])])
                cv2.imwrite(str(dirs["05_flagged"] / f"{st}.jpg"), strip)
            rows.append({"file": st, "seed_model": s, "split": split[s].get(st, "?"), "hand": hand_side(st),
                         "cx": sq[0], "cy": sq[1], "side": sq[2], "theta": sq[3], "conf": conf, "n_det": ndet,
                         "outside_hand_frac": ofrac, "iou_gt": iou, "angle_err_gt": aerr, "flags": ";".join(flags)})
            pred_labels[st + ".png"] = {"cx": sq[0] / W, "cy": sq[1] / H, "size": sq[2] / min(H, W), "theta": sq[3]}
            if (n + 1) % 500 == 0:
                print(f"  seed {s}: {n + 1}/{len(todo)}")

    # ---- 04 contact sheets
    per = a.sheet_cols * a.sheet_rows
    cell = 150
    for k in range(0, len(crops_for_sheet), per):
        chunk = crops_for_sheet[k:k + per]
        sheet = np.zeros((a.sheet_rows * (cell + 18), a.sheet_cols * cell, 3), np.uint8)
        for j, (st, crop, flagged) in enumerate(chunk):
            r, c = divmod(j, a.sheet_cols)
            thumb = cv2.resize(to_bgr(crop), (cell, cell), interpolation=cv2.INTER_AREA)
            y0, x0 = r * (cell + 18), c * cell
            sheet[y0 + 18:y0 + 18 + cell, x0:x0 + cell] = thumb
            put_text(sheet, st, (x0 + 2, y0 + 13), (0, 0, 255) if flagged else (255, 255, 255), 0.38, 1)
        cv2.imwrite(str(dirs["04_contact_sheets"] / f"sheet_{k // per + 1:04d}.jpg"), sheet)

    write_rows(out / "predictions.csv", rows)
    fl = [r for r in rows if r["flags"]]
    write_rows(out / "qc_flags.csv", fl if fl else [{"file": "", "flags": ""}])
    save_json(pred_labels, out / "pred_labels.json")
    cnt = {}
    for r in fl:
        for f in r["flags"].split(";"):
            cnt[f] = cnt.get(f, 0) + 1
    print(f"[DONE] {len(rows)} images -> {out}")
    print(f"       flagged: {len(fl)} ({100 * len(fl) / max(1, len(rows)):.1f}%)  by reason: {cnt}")
    print("       split of processed images: " + str({sp: sum(1 for r in rows if r['split'] == sp) for sp in ('train', 'val', 'test', '?')}))


if __name__ == "__main__":
    main()
