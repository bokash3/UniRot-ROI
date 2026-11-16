#!/usr/bin/env python3
"""
baseline_mediapipe.py - MediaPipe Hands as an off-the-shelf palm localiser (zero-shot, no training on palm images)
=================================================================================================================
Requested by reviewers (external baseline). MediaPipe Hands (Zhang et al. 2020) returns 21 hand landmarks for visible-light
RGB hands. Here the NIR image is fed as grey RGB, the highest-scoring hand is kept, and a palm square is derived from four
landmarks: wrist (0) and the knuckles of index (5), middle (9), ring (13) and little finger (17).

  L        = distance from the wrist to the centroid of the four knuckles (palm length, pixels)
  phi_hand = direction wrist -> knuckle centroid ("up" of the palm)
The square (centre, side, angle) is placed relative to (wrist, L, phi_hand) with offsets that are the MEDIANS over the
CASIA-MS TRAIN split of the seed (the same recipe as the calibrated valley-point baseline); nothing else is learnt.
  centre = wrist + L * (du * up + dv * perp),  side = rho * L,  angle(up) = phi_hand + dphi
Offsets are calibrated separately for left and right hands when the file name tells the side, else pooled.
No detection -> failure to acquire (IoU 0, angle error 180 deg, flagged), as for the YOLO detectors.

Outputs
  out/baselines/mediapipe/landmarks_<ds>.json         raw detections (cached; delete to re-run)
  out/baselines/mediapipe/seed<S>/calibration.json
  out/baselines/mediapipe/seed<S>/per_image_casia.csv
  out/cross/results/<vera|scut>/per_image_mediapipe_seed<S>.csv
  out/baselines/mediapipe/summary.csv
Run (CPU):  python baseline_mediapipe.py --seeds 42,2025,1337
Needs: pip install mediapipe==0.10.14   (the legacy 'solutions' API with bundled models)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from rx_common import circ_err, label_to_px, load_json, polygon_iou, save_json, square_corners
from rx_stages import hand_side, score, summarize, write_rows

W0, H0 = 768, 576
KN = (5, 9, 13, 17)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def up_angle(sq):
    c = square_corners(*sq)
    up = (c[0] + c[1]) / 2 - (c[2] + c[3]) / 2
    return math.degrees(math.atan2(up[1], up[0]))


def theta_for_up(phi):
    """cv2-convention theta of square_corners that gives the 'up' direction phi (degrees)."""
    p0 = up_angle((0, 0, 1, 0.0))
    p1 = up_angle((0, 0, 1, 10.0))
    s = 1.0 if wrap(p1 - p0) > 0 else -1.0
    return s * wrap(phi - p0)


def detect_all(hands, items, tag, partial=None, max_seconds=0):
    """Resumable: results are written to <partial> every 500 images; with max_seconds the call stops early
    (SystemExit 3) so that a long run can be continued by calling the script again."""
    import time
    t0 = time.time()
    out = load_json(partial) if partial and Path(partial).is_file() else {}
    for i, (name, path) in enumerate(items):
        if name in out:
            continue
        if max_seconds and time.time() - t0 > max_seconds:
            save_json(out, partial)
            print(f"  [{tag}] paused at {len(out)}/{len(items)}; run again to continue", flush=True)
            raise SystemExit(3)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            out[name] = None
            continue
        H, W = img.shape[:2]
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            out[name] = None
        else:
            best, bs = None, -1.0
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                sc = hd.classification[0].score
                if sc > bs:
                    best, bs = lm, sc
            pts = {k: [best.landmark[k].x * W, best.landmark[k].y * H] for k in (0,) + KN}
            out[name] = {"pts": pts, "score": float(bs), "n_det": len(res.multi_hand_landmarks), "W": W, "H": H}
        if (i + 1) % 500 == 0:
            print(f"  [{tag}] {i + 1}/{len(items)}", flush=True)
            if partial:
                save_json(out, partial)
    return out


def hand_frame(d):
    pts = {int(k): v for k, v in d["pts"].items()}
    w = np.array(pts[0], float)
    k = np.mean([pts[j] for j in KN], axis=0)
    u = k - w
    L = float(np.linalg.norm(u))
    phi = math.degrees(math.atan2(u[1], u[0]))
    return w, L, phi


def gt_frame(gt):
    return np.array(gt[:2], float), gt[2], up_angle(gt)


def to_local(w, L, phi, c):
    a = math.radians(phi)
    up = np.array([math.cos(a), math.sin(a)])
    pp = np.array([-up[1], up[0]])
    v = (np.asarray(c) - w) / L
    return float(v @ up), float(v @ pp)


def calibrate(train, dets, W=W0, H=H0):
    acc = {"all": [], "l": [], "r": []}
    for f, lab in train.items():
        d = dets.get(Path(f).stem)
        if not d:
            continue
        w, L, phi = hand_frame(d)
        if L < 5:
            continue
        gt = label_to_px(lab, W, H)
        c, side, pgt = gt_frame(gt)
        du, dv = to_local(w, L, phi, c)
        row = (du, dv, side / L, wrap(pgt - phi))
        acc["all"].append(row)
        s = hand_side(f)
        if s in ("l", "r"):
            acc[s].append(row)
    par = {}
    for k, rows in acc.items():
        if len(rows) < 30:
            continue
        a = np.array(rows)
        dphi = float(np.median(a[:, 3]))
        par[k] = {"du": float(np.median(a[:, 0])), "dv": float(np.median(a[:, 1])),
                  "rho": float(np.median(a[:, 2])), "dphi": dphi, "n": len(rows)}
    return par


def predict(d, par, side):
    p = par.get(side, par["all"])
    w, L, phi = hand_frame(d)
    a = math.radians(phi + p["dphi"])
    up = np.array([math.cos(a), math.sin(a)])
    pp = np.array([-up[1], up[0]])
    c = w + L * (p["du"] * up + p["dv"] * pp)
    return (float(c[0]), float(c[1]), float(p["rho"] * L), theta_for_up(phi + p["dphi"]))


def evaluate(items_gt, dets, par, key_side):
    """items_gt: {name: (cx,cy,side,theta)}; returns per-image rows."""
    rows = []
    for name, gt in items_gt.items():
        d = dets.get(key_side(name)[0])
        if not d or hand_frame(d)[1] < 5:
            rows.append({"iou": 0.0, "center_err_px": float("nan"), "size_err_px": float("nan"),
                         "angle_err_deg": 180.0, "file": name, "failed": 1, "conf": 0.0, "n_det": 0})
            continue
        pr = predict(d, par, key_side(name)[1])
        r = score(pr, gt, None)
        r.update(file=name, failed=0, conf=d["score"], n_det=d["n_det"])
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--casia_images", default=r"../data/casia/images")
    ap.add_argument("--labels_root", default="out/labels_oriented")
    ap.add_argument("--cross_dir", default="out/cross")
    ap.add_argument("--out", default="out/baselines/mediapipe")
    ap.add_argument("--seeds", default="42,2025,1337")
    ap.add_argument("--min_conf", type=float, default=0.3, help="min_detection_confidence (default 0.5 is lowered for NIR)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N CASIA images")
    ap.add_argument("--max_seconds", type=float, default=0, help="stop after this many seconds (resume by re-running)")
    a = ap.parse_args()
    import mediapipe as mp
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1,
                                     min_detection_confidence=a.min_conf)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- detections (cached)
    cache = out / "landmarks_casia.json"
    if cache.is_file():
        cas = load_json(cache)
    else:
        files = sorted(Path(a.casia_images).glob("*.jpg")) + sorted(Path(a.casia_images).glob("*.png"))
        if a.limit:
            files = files[:a.limit]
        cas = detect_all(hands, [(p.stem, p) for p in files], "casia", out / "landmarks_casia_partial.json", a.max_seconds)
        if not a.limit:
            save_json(cas, cache)
    vera_scut = {}
    for ds in ("vera", "scut"):
        c = out / f"landmarks_{ds}.json"
        dd = Path(a.cross_dir) / ds
        if c.is_file():
            vera_scut[ds] = load_json(c)
        else:
            man = load_json(dd / "manifest.json")
            lab = load_json(dd / "labels_manual.json")
            names = [n for n, l in lab.items() if not l.get("skip") and n in man]
            vera_scut[ds] = detect_all(hands, [(n, dd / "images" / f"{n}.png") for n in names], ds)
            save_json(vera_scut[ds], c)
    nd = lambda D: sum(1 for v in D.values() if v)
    print(f"detections: casia {nd(cas)}/{len(cas)}  vera {nd(vera_scut['vera'])}/{len(vera_scut['vera'])}  "
          f"scut {nd(vera_scut['scut'])}/{len(vera_scut['scut'])}")

    summ = out / "summary.csv"
    with open(summ, "w", newline="") as f:
        csv.writer(f).writerow(["dataset", "seed", "n", "mIoU", "acc_0_7", "center_err_px", "angle_err_deg", "failures_iou_lt_0_5", "failed_to_acquire"])
    for seed in [int(s) for s in a.seeds.split(",")]:
        lr = Path(a.labels_root) / f"seed{seed}"
        train = load_json(lr / "train_labels.json")
        test = load_json(lr / "test_labels.json")
        par = calibrate(train, cas)
        sd = out / f"seed{seed}"
        sd.mkdir(exist_ok=True)
        save_json(par, sd / "calibration.json")
        print(f"[seed {seed}] calibration {json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in par.items()})}")
        sets = {"casia": ({f: label_to_px(l, W0, H0) for f, l in test.items()}, cas, lambda n: (Path(n).stem, hand_side(n)),
                          sd / "per_image_casia.csv")}
        for ds in ("vera", "scut"):
            dd = Path(a.cross_dir) / ds
            lab = load_json(dd / "labels_manual.json")
            man = load_json(dd / "manifest.json")
            gts = {n: label_to_px(l, W0, H0) for n, l in lab.items() if not l.get("skip") and n in man}
            sets[ds] = (gts, vera_scut[ds], lambda n: (n, hand_side(n)), Path(a.cross_dir) / "results" / ds / f"per_image_mediapipe_seed{seed}.csv")
        for ds, (gts, dets, ks, path) in sets.items():
            rows = evaluate(gts, dets, par, ks)
            write_rows(path, rows)
            iou = np.array([r["iou"] for r in rows])
            ang = np.array([r["angle_err_deg"] for r in rows])
            ce = np.array([r["center_err_px"] for r in rows], float)
            fails = int((iou < 0.5).sum())
            fa = int(sum(r["failed"] for r in rows))
            print(f"  {ds:6s} n={len(rows):5d} mIoU={iou.mean():.4f} Acc@0.7={(iou >= .7).mean():.3f} "
                  f"ctr={np.nanmean(ce):.1f}px angle={ang.mean():.1f} IoU<0.5={fails} ({100 * fails / len(rows):.1f}%) no-detection={fa}")
            with open(summ, "a", newline="") as f:
                csv.writer(f).writerow([ds, seed, len(rows), f"{iou.mean():.4f}", f"{(iou >= .7).mean():.4f}",
                                        f"{np.nanmean(ce):.2f}", f"{ang.mean():.2f}", fails, fa])


if __name__ == "__main__":
    main()
