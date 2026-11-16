#!/usr/bin/env python3
"""
baseline_valley_points.py — classical finger-valley baseline for the whole ROI task
==================================================================================

Classical contactless-palm ROI localisation (Zhang et al., 2003 style, adapted to
binary hand masks): the valleys between the fingers define a reference line, the
palm square is placed on the palm side of that line.

Input  : the Stage-I hand mask in the RAW frame (same masks that feed UniRotNet).
Output : an oriented square (cx, cy, side, theta) in the RAW frame, scored against
         the oriented GT labels (make_rx_data.py labels) with exactly the same
         metrics as cells A-D of the 2x2 ablation. No training of a network.

Algorithm (every numbered step writes a file, see "What is saved")
  1. clean the mask: keep the largest connected component, fill holes
  2. contour + convex hull + convexity defects; defects shallower than
     min_depth_frac*sqrt(area) or touching the image border are ignored
  3. candidates = the 6 deepest defects (the 3 finger valleys, the thumb-index
     valley and, often, notches at the thumb base / wrist)
  4. V1, V2, V3 = the triple of candidates (ordered along the contour) with the
     lowest cost = |gap12-gap23|/(gap12+gap23) [equal finger gaps]
     + 2*dist(V2, line V1V3)/L [V2 on the line] + 0.5*CV(depths) [similar depths]
  5. reference line V1-V3 (length L, midpoint m); 'up' = perpendicular to the line,
     pointing from the mask centroid towards m. rotation angle phi_hat from 'up'.
  6. square = centre m + d*L down the hand + e*L sideways, side rho*L.
     (d, e, rho) and the left/right-hand angle offset are MEDIANS over the TRAIN
     split of the same seed (calibration; the annotators' palm-square definition is
     not a geometric property of the valleys) -> no test information is used.
  7. upright 128x128 crop of the predicted square (and of the GT square).

Two variants are reported
  valley        : theta = phi_hat (pure geometry), centre/side calibrated
  valley+offset : theta = phi_hat - (left/right-hand angle offset from TRAIN)
Images where no valley set is found (failed=1) fall back to the train-median square
(theta = 0), are counted with their (usually low) IoU and listed in the summary.

What is saved (out_dir/valley/seed<S>/)
  calibration.json            learnt (d, e, rho, angle offsets) + how many train images were usable
  per_image_valley*.csv       one row per test image and variant
  pred_labels_valley*.json    predicted squares in the label format (reusable for matching)
  stages/<stem>_strip.png     ALL stages of one image side by side
  stages/<stem>/0N_*.png      each stage as its own file, + info.json with the numbers
Stage files are written for: the worst IoUs, all failures (up to a limit) and a
random sample (--dump_worst / --dump_fail / --dump_random; --dump_all for everything).

Example:
  python baseline_valley_points.py ^
     --masks_dir  ../data/casia/stage1_masks ^
     --images_dir ../data/casia/images ^
     --labels_root out\\labels_oriented --out_dir out\\baselines ^
     --results_csv runs\\results_baselines.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from rx_common import (apply_affine, index_dir, label_to_px, load_json, rot_matrix, save_json,
                       square_corners, stem)
from rx_stages import (BLUE, CYAN, GRAY, GREEN, MAGENTA, ORANGE, RED, YELLOW, append_result, draw_poly,
                       hand_side, oriented_crop, pick_for_dump, put_text, save_stages, score, summarize,
                       to_bgr, write_rows)


# --------------------------------------------------------------------- vision part
def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Binary uint8 (0/255): largest connected component, holes filled."""
    b = (mask > 127).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    if n <= 1:
        return b * 255
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    b = (lab == k).astype(np.uint8)
    inv = 1 - b
    ff = inv.copy()
    h, w = b.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 2)           # background reachable from the corner
    holes = (ff == 1)
    b[holes] = 1
    return b * 255


def find_valleys(mask: np.ndarray, min_depth_frac=0.05, border_margin=3, n_cand=6) -> dict:
    """Steps 1-5 of the algorithm. Always returns a dict; v['ok'] tells if it worked
    and v['reason'] explains a failure."""
    v = {"ok": False, "reason": "", "clean": None, "contour": None, "hull": None, "defects": [],
         "selected": [], "chain": [], "thumb": None}
    cl = clean_mask(mask)
    v["clean"] = cl
    H, W = cl.shape
    area = float((cl > 0).sum())
    if area < 500:
        v["reason"] = "empty_mask"
        return v
    v["area"] = area
    cnts, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)     # (N, 2), robust to OpenCV versions
    v["contour"] = c
    try:
        hidx = np.asarray(cv2.convexHull(c, returnPoints=False)).reshape(-1, 1).astype(np.int32)
        defects = cv2.convexityDefects(c, hidx)
    except cv2.error:
        v["reason"] = "hull_error"
        return v
    v["hull"] = c[hidx[:, 0]]
    if defects is None:
        v["reason"] = "no_defects"
        return v
    thr = min_depth_frac * math.sqrt(area)
    ds = []
    for s, e, f, d in np.asarray(defects).reshape(-1, 4):
        p = c[f].astype(float)
        depth = d / 256.0
        reason = ""
        if depth < thr:
            reason = "shallow"
        elif p[0] < border_margin or p[1] < border_margin or p[0] > W - 1 - border_margin or p[1] > H - 1 - border_margin:
            reason = "border"
        ds.append({"idx": int(f), "pt": p, "depth": float(depth), "rejected": reason})
    v["defects"] = ds
    good = sorted([d for d in ds if not d["rejected"]], key=lambda d: -d["depth"])
    if len(good) < 3:
        v["reason"] = f"few_defects({len(good)}<3)"
        return v
    cand = good[:n_cand]
    N = len(c)
    best, chain = None, None
    for tri in itertools.combinations(cand, 3):
        tri = sorted(tri, key=lambda d: d["idx"])
        idx = [d["idx"] for d in tri]
        gaps = [(idx[(j + 1) % 3] - idx[j]) % N for j in range(3)]
        st = (int(np.argmax(gaps)) + 1) % 3            # cut at the way round the hand
        tri = [tri[(st + j) % 3] for j in range(3)]
        p1, p2, p3 = (t["pt"] for t in tri)
        d12, d23, L13 = np.linalg.norm(p1 - p2), np.linalg.norm(p2 - p3), np.linalg.norm(p1 - p3)
        if L13 < 1e-6:
            continue
        spacing = abs(d12 - d23) / (d12 + d23 + 1e-9)                         # equal finger gaps
        u_ = (p3 - p1) / L13
        line = abs(u_[0] * (p2 - p1)[1] - u_[1] * (p2 - p1)[0]) / L13                  # V2 close to the V1-V3 line
        dep = np.array([t["depth"] for t in tri])
        cv = dep.std() / (dep.mean() + 1e-9)                                  # similar valley depths
        cost = spacing + 2.0 * line + 0.5 * cv
        if best is None or cost < best:
            best, chain = cost, tri
    v["selected"] = cand
    v["thumb"] = None
    v["chain"] = chain
    v["chain_cost"] = float(best)
    w1, w3 = chain[0]["pt"], chain[-1]["pt"]
    L = float(np.linalg.norm(w1 - w3))
    if L < 0.05 * math.sqrt(area):
        v["reason"] = "degenerate_baseline"
        return v
    m = (w1 + w3) / 2.0
    ys, xs = np.nonzero(cl)
    cen = np.array([xs.mean(), ys.mean()])
    t = (w3 - w1) / L
    n = np.array([-t[1], t[0]])
    if np.dot(m - cen, n) < 0:
        n = -n
    up = n                                           # fingers direction
    phi = math.degrees(math.atan2(-up[0], -up[1])) % 360.0   # see rx_common: up_raw = (-sin phi, -cos phi)
    v.update(ok=True, L=L, m=m, centroid=cen, up=up, phi=phi)
    return v


# --------------------------------------------------------------------- geometry of the square
def frame_vectors(phi_deg):
    a = math.radians(phi_deg)
    c, s = math.cos(a), math.sin(a)
    right = np.array([c, -s])
    up = np.array([-s, -c])
    return right, up


def local_coords(v: dict, gt_xy, gt_side, phi_deg):
    right, up = frame_vectors(phi_deg)
    dvec = np.asarray(gt_xy) - v["m"]
    return float(-dvec @ up / v["L"]), float(dvec @ right / v["L"]), float(gt_side / v["L"])   # d (down), e (right), rho


def square_from_valleys(v: dict, par: dict, side: str, use_offset: bool):
    p = par["valley+offset" if use_offset else "valley"]
    off = par["dphi"].get(side, par["dphi"]["all"]) if use_offset else 0.0
    phi = (v["phi"] + off) % 360.0                    # offset = median(gt - phi_hat) on TRAIN
    right, up = frame_vectors(phi)
    e = p["e"].get(side, p["e"]["all"])
    ctr = v["m"] + p["d"] * v["L"] * (-up) + e * v["L"] * right
    return float(ctr[0]), float(ctr[1]), float(p["rho"] * v["L"]), float(phi)


def signed(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def calibrate(files, labels, mask_idx, max_n, seed):
    """Medians over the TRAIN split. (d, e, rho) are expressed in the frame that is
    used at test time (phi_hat for 'valley', phi_hat + offset for 'valley+offset')."""
    recs = []
    files = sorted(files)
    random.Random(seed).shuffle(files)
    n_try = n_fail = 0
    for f in files:
        if stem(f) not in mask_idx:
            continue
        if max_n and n_try >= max_n:
            break
        n_try += 1
        m = cv2.imread(str(mask_idx[stem(f)]), cv2.IMREAD_GRAYSCALE)
        H, W = m.shape
        v = find_valleys(m)
        if not v["ok"]:
            n_fail += 1
            continue
        cx, cy, side, th = label_to_px(labels[f], W, H)
        recs.append({"side": hand_side(f), "m": v["m"], "L": v["L"], "phi": v["phi"], "gxy": (cx, cy),
                     "gs": side, "dphi": signed(th, v["phi"])})
    if not recs:
        raise RuntimeError("calibration failed: no train mask produced valleys")
    sides = [s for s in ("l", "r") if any(r["side"] == s for r in recs)]
    med = lambda vals: float(np.median(vals))
    dphi = {"all": med([r["dphi"] for r in recs]),
            **{s: med([r["dphi"] for r in recs if r["side"] == s]) for s in sides}}
    par = {"dphi": dphi, "n_train_tried": n_try, "n_train_used": len(recs), "n_train_failed": n_fail}
    for variant in ("valley", "valley+offset"):
        ds, es, rs = [], [], []
        for r in recs:
            ph = r["phi"] + (dphi.get(r["side"], dphi["all"]) if variant == "valley+offset" else 0.0)
            right, up = frame_vectors(ph)
            dv = np.asarray(r["gxy"]) - r["m"]
            ds.append(float(-dv @ up / r["L"])); es.append((r["side"], float(dv @ right / r["L"])))
            rs.append(r["gs"] / r["L"])
        par[variant] = {"d": med(ds), "rho": med(rs),
                        "e": {"all": med([e for _, e in es]),
                              **{s: med([e for sd, e in es if sd == s]) for s in sides}}}
    return par


# --------------------------------------------------------------------- stage images
def stage_panels(raw_img, mask, v, gt, preds, score_rows, fname):
    """Build the list of (file name, title, image) for one test image."""
    H, W = mask.shape
    panels = [("01_input_mask", "raw Stage-I mask", to_bgr(mask))]
    cl = v["clean"]
    panels.append(("02_mask_clean", "largest comp.+holes filled", to_bgr(cl)))
    p3 = to_bgr(cl)
    if v["contour"] is not None:
        cv2.drawContours(p3, [v["contour"].reshape(-1, 1, 2)], -1, CYAN, 1)
        if v["hull"] is not None:
            draw_poly(p3, v["hull"], YELLOW, 1)
        for d in v["defects"]:
            col = GRAY if d["rejected"] else ORANGE
            cv2.circle(p3, tuple(np.round(d["pt"]).astype(int)), 5, col, -1 if not d["rejected"] else 1)
            put_text(p3, f"{d['depth']:.0f}", tuple(np.round(d["pt"]).astype(int) + [6, -6]), col, 0.45)
    panels.append(("03_hull_defects", "hull(yellow) defects(orange=kept,grey=rejected)", p3))
    p4 = to_bgr(cl)
    for j, d in enumerate(v["selected"]):
        pt = tuple(np.round(d["pt"]).astype(int))
        cv2.circle(p4, pt, 7, RED, -1)
        put_text(p4, "abcdef"[j], (pt[0] + 8, pt[1] - 8), RED, 0.6)
    for j, d in enumerate(v["chain"]):
        pt = tuple(np.round(d["pt"]).astype(int))
        cv2.circle(p4, pt, 9, GREEN, 2)
        put_text(p4, f"V{j + 1}", (pt[0] - 8, pt[1] + 24), GREEN, 0.6)
    if len(v["chain"]) == 3:
        cv2.line(p4, tuple(np.round(v["chain"][0]["pt"]).astype(int)),
                 tuple(np.round(v["chain"][-1]["pt"]).astype(int)), GREEN, 2)
    if not v["ok"]:
        put_text(p4, "FAILED: " + v["reason"], (10, 30), RED, 0.7, 2)
    panels.append(("04_valleys", "6 deepest defects (red); chosen V1-V3 (green)", p4))
    if not v["ok"]:
        panels.append(("05_roi_fallback", "FAILED -> fallback square", _overlay(raw_img, gt, preds, score_rows, fname)))
        return panels
    p5 = to_bgr(cl)
    m = v["m"]
    cv2.circle(p5, tuple(np.round(v["centroid"]).astype(int)), 6, BLUE, -1)
    cv2.circle(p5, tuple(np.round(m).astype(int)), 6, GREEN, -1)
    tip = m + v["up"] * 0.6 * v["L"] * 2
    cv2.arrowedLine(p5, tuple(np.round(m).astype(int)), tuple(np.round(tip).astype(int)), RED, 3, tipLength=0.25)
    put_text(p5, f"phi_hat={v['phi']:.1f}  L={v['L']:.0f}px", (10, 30), YELLOW, 0.7, 2)
    put_text(p5, "blue=centroid green=midpoint(V1,V3) red=up", (10, H - 12), (255, 255, 255), 0.5)
    panels.append(("05_orientation", "up = centroid -> midpoint", p5))
    pr = preds["valley+offset"]
    p6 = cv2.warpAffine(to_bgr(cl), rot_matrix(-pr[3], W, H), (W, H))
    c2 = apply_affine(rot_matrix(-pr[3], W, H), [[pr[0], pr[1]]])[0]
    draw_poly(p6, square_corners(c2[0], c2[1], pr[2], 0.0), RED, 2)
    panels.append(("06_aligned_mask", "mask rotated by -theta (fingers up?) + ROI", p6))
    panels.append(("07_roi_vs_gt", "ROI: red=valley+offset green=GT", _overlay(raw_img, gt, preds, score_rows, fname)))
    cp = oriented_crop(to_bgr(raw_img), *pr)
    cg = oriented_crop(to_bgr(raw_img), *gt)
    panels.append(("08_crops", "crop pred | crop GT", np.hstack([cp, cg])))
    return panels


def _overlay(raw_img, gt, preds, score_rows, fname):
    im = to_bgr(raw_img)
    draw_poly(im, square_corners(*gt), GREEN, 2)
    cols = {"valley": ORANGE, "valley+offset": RED}
    for k, p in preds.items():
        draw_poly(im, square_corners(*p), cols[k], 2)
    y = 24
    for k in preds:
        put_text(im, f"{k}: IoU={score_rows[k]['iou']:.3f}", (8, y), cols[k], 0.6, 1)
        y += 24
    return im


# --------------------------------------------------------------------- main
def run_seed(a, seed):
    lab_dir = Path(a.labels_root) / f"seed{seed}"
    tr, te = load_json(lab_dir / "train_labels.json"), load_json(lab_dir / "test_labels.json")
    mask_idx = index_dir(Path(a.masks_dir))
    img_idx = index_dir(Path(a.images_dir)) if a.images_dir else {}
    out = Path(a.out_dir) / "valley" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[seed {seed}] calibrating on the TRAIN split ({len(tr)} labels) ...")
    par = calibrate(list(tr), tr, mask_idx, a.calib_max, seed)
    save_json(par, out / "calibration.json")
    print(f"  usable train masks {par['n_train_used']}/{par['n_train_tried']}  angle offsets (deg)={par['dphi']}")
    for k in ("valley", "valley+offset"):
        print(f"  {k:14s} d={par[k]['d']:.3f} rho={par[k]['rho']:.3f} e={par[k]['e']}")
    # fallback = train-median square
    W0, H0 = 768, 576
    allp = [label_to_px(l, W0, H0) for l in tr.values()]
    fb = (float(np.median([p[0] for p in allp])), float(np.median([p[1] for p in allp])),
          float(np.median([p[2] for p in allp])), 0.0)

    variants = ("valley", "valley+offset")
    rows = {k: [] for k in variants}
    cache = {}
    files = sorted(f for f in te if stem(f) in mask_idx)
    print(f"[seed {seed}] testing {len(files)} images ...")
    for i, f in enumerate(files):
        m = cv2.imread(str(mask_idx[stem(f)]), cv2.IMREAD_GRAYSCALE)
        H, W = m.shape
        v = find_valleys(m)
        gt = label_to_px(te[f], W, H)
        side = hand_side(f)
        preds = {}
        for k in variants:
            if v["ok"]:
                preds[k] = square_from_valleys(v, par, side, k == "valley+offset")
            else:
                preds[k] = fb
            r = score(preds[k], gt, m)
            r.update(file=f, failed=int(not v["ok"]), reason=v["reason"], side=side, variant=k,
                     phi_hat=v.get("phi", float("nan")), L=v.get("L", float("nan")))
            rows[k].append(r)
        cache[f] = (v, preds)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)}")

    for k in variants:
        s = summarize(rows[k])
        tag = k.replace("+", "_")
        write_rows(out / f"per_image_{tag}.csv", rows[k])
        save_json({f: {"cx": cache[f][1][k][0] / 768.0, "cy": cache[f][1][k][1] / 576.0,
                       "size": cache[f][1][k][2] / 576.0, "theta": cache[f][1][k][3]} for f in cache},
                  out / f"pred_labels_{tag}.json")
        ok = [r for r in rows[k] if not r["failed"]]
        s_ok = summarize(ok) if ok else {}
        print(f"  [{k:14s}] mIoU={s['mIoU']:.4f}  Acc@0.7={s['acc_0_7']:.3f}  ctr={s['center_err_px']:.1f}px  "
              f"size={s['size_err_px']:.1f}px  ang={s['angle_err_deg']:.2f}  outside={s.get('outside_frac', float('nan')):.3f}"
              f"  failed={s['n_failed']}/{s['n']}" + (f"  (mIoU on non-failed={s_ok['mIoU']:.4f})" if ok else ""))
        if a.results_csv:
            append_result(a.results_csv, "Valley-points" + ("+offset" if "offset" in k else ""), seed, s)

    # ---- stage files
    key = rows["valley+offset"]
    sel = files if a.dump_all else pick_for_dump(key, a.dump_random, a.dump_worst, a.dump_fail, seed)
    print(f"[seed {seed}] writing stage files for {len(sel)} images -> {out / 'stages'}")
    by = {k: {r["file"]: r for r in rows[k]} for k in variants}
    for f in sel:
        v, preds = cache[f]
        m = cv2.imread(str(mask_idx[stem(f)]), cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(str(img_idx[stem(f)]), cv2.IMREAD_COLOR) if stem(f) in img_idx else to_bgr(m)
        H, W = m.shape
        gt = label_to_px(te[f], W, H)
        panels = stage_panels(raw, m, v, gt, preds, {k: by[k][f] for k in variants}, f)
        info = {"file": f, "ok": v["ok"], "reason": v["reason"], "gt": gt, "pred": preds,
                "iou": {k: by[k][f]["iou"] for k in variants},
                "valleys_chain_V1_V2_V3": [d["pt"].tolist() for d in v["chain"]],
                "candidates": [d["pt"].tolist() for d in v["selected"]], "chain_cost": v.get("chain_cost"),
                "phi_hat": v.get("phi"), "L": v.get("L")}
        save_stages(out / "stages", stem(f), panels, info)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--masks_dir", required=True, help="raw-frame Stage-I hand masks (generated_masks)")
    p.add_argument("--images_dir", default=None, help="raw images (for overlays/crops); optional")
    p.add_argument("--labels_root", required=True, help="out/labels_oriented (seed*/train|test_labels.json)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seeds", default="42,2025,1337")
    p.add_argument("--results_csv", default=None)
    p.add_argument("--calib_max", type=int, default=0, help="max train masks used for calibration (0 = all)")
    p.add_argument("--dump_random", type=int, default=20)
    p.add_argument("--dump_worst", type=int, default=12)
    p.add_argument("--dump_fail", type=int, default=12)
    p.add_argument("--dump_all", action="store_true", help="write stage files for every test image")
    a = p.parse_args()
    for s in a.seeds.split(","):
        run_seed(a, int(s))


if __name__ == "__main__":
    main()
