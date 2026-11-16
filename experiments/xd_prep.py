#!/usr/bin/env python3
"""
xd_prep.py — prepare the cross-dataset generalisation test (VERA, SCUT-PV, + a CASIA sanity set)
================================================================================================

Zero-shot: nothing is retrained. Every method is run on images of datasets it has never seen.

What this script does (and SAVES, so you can look at each step)
  1. picks N random HANDS per dataset (one random image per hand -> no near-duplicates),
     fixed seed, written to  <out>/<ds>/manifest.json
  2. <out>/<ds>/upright/<name>.png   the image as a human sees it (fingers UP). The labeling tool shows this.
  3. <out>/<ds>/images/<name>.png    the image in the CASIA frame used by every method:
        fingers pointing LEFT, 768 x 576 (rotate 90 deg CCW, scale to fit, black padding)
     manifest[name]["M"] is the 2x3 matrix  upright -> CASIA-frame  (used to convert hand-clicked labels)
  4. <out>/<ds>/contact_sheet_*.jpg  thumbnails of the selected images (look at them first!)

Datasets
  vera  : <vera_root>\\NNN-X\\0S\\NNN_H_S_k.png   (640 x 480 portrait, fingers up)
  scut  : <scut_root>\\RAW\\<id>\\<id>_<k>.bmp    (640 x 480 landscape, fingers up)
  casia : a CASIA-MS check set (images already fingers-left): the SAME labeling tool is used so that the
          gap between a human click-label and the dataset GT convention is measured (= label noise ceiling).

Example
  python xd_prep.py --vera_root "../datasets/VERA-Palmvein/raw" ^
      --scut_root "../datasets/SCUT_PV_V1" ^
      --casia_images ../data/casia/images --casia_labels out\\labels_oriented\\seed1337\\test_labels.json ^
      --out_dir out\\cross --n 100 --n_casia 40
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from rx_common import index_dir, load_json, save_json, stem

W0, H0 = 768, 576


def frame_matrix(w_up: int, h_up: int):
    """upright image (w_up x h_up) -> CASIA frame (768x576, fingers left, letterboxed). returns 2x3."""
    # rotate 90 CCW: (x, y) -> (y, w_up - 1 - x); result size (h_up wide, w_up high)
    R = np.array([[0, 1, 0], [-1, 0, w_up - 1], [0, 0, 1]], np.float64)
    wr, hr = h_up, w_up
    s = min(W0 / wr, H0 / hr)
    ox, oy = (W0 - wr * s) / 2.0, (H0 - hr * s) / 2.0
    S = np.array([[s, 0, ox], [0, s, oy], [0, 0, 1]], np.float64)
    return (S @ R)[:2]


def contact_sheet(paths, out, cols=8, th=160):
    tiles = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        sc = th / im.shape[0]
        t = cv2.resize(im, (int(im.shape[1] * sc), th))
        cv2.putText(t, Path(p).stem[:14], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        tiles.append(t)
    wmax = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, wmax - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), np.vstack(rows))


def collect_vera(root: Path):
    hands = {}
    for sdir in sorted(root.iterdir()):
        if not sdir.is_dir():
            continue
        for f in sdir.rglob("*.png"):
            t = f.stem.split("_")                       # 001_L_1_1
            hands.setdefault((t[0], t[1]), []).append(f)
    return hands


def collect_scut(root: Path):
    hands = {}
    for f in (root / "RAW").rglob("*.bmp"):
        hands.setdefault((f.stem.split("_")[0], "NA"), []).append(f)
    return hands


def build(ds, hands, n, seed, out_root, one_per_subject=False, extend=False):
    rnd = random.Random(seed)
    d = out_root / ds
    for sub in ("upright", "images"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    man = {}
    if extend and (d / "manifest.json").is_file():
        old = load_json(d / "manifest.json")
        seen = set()
        dropped = {}
        for nm in sorted(old):                       # keep the first image of every subject, drop later ones
            sj = old[nm]["subject"] if one_per_subject else nm
            if sj in seen:
                dropped[nm] = old[nm]
                continue
            seen.add(sj)
            man[nm] = old[nm]
        if dropped:
            save_json(dropped, d / "manifest_dropped_duplicates.json")
        print(f"[{ds}] kept {len(man)} existing images, dropped {len(dropped)} same-subject duplicates")
    used_subjects = {v["subject"] for v in man.values()}
    used_hands = {(v["subject"], v["hand"]) for v in man.values()}
    keys = sorted(hands)
    rnd.shuffle(keys)
    cand = []
    for k in keys:
        if one_per_subject and k[0] in used_subjects:
            continue
        if (k[0], k[1]) in used_hands:
            continue
        if one_per_subject:
            used_subjects.add(k[0])
        cand.append(k)
    keys = cand[:max(0, n - len(man))]
    for k in keys:
        f = rnd.choice(sorted(hands[k]))
        up = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        h, w = up.shape
        M = frame_matrix(w, h)
        im = cv2.warpAffine(up, M, (W0, H0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        name = f"{ds}_{k[0]}_{k[1]}_{f.stem}"
        cv2.imwrite(str(d / "upright" / f"{name}.png"), up)
        cv2.imwrite(str(d / "images" / f"{name}.png"), im)
        man[name] = {"source": str(f), "hand": k[1], "subject": k[0], "M": M.tolist(), "size_upright": [w, h]}
    save_json(man, d / "manifest.json")
    names = sorted(man)
    for i in range(0, len(names), 48):
        contact_sheet([d / "upright" / f"{n}.png" for n in names[i:i + 48]], d / f"contact_sheet_{i // 48 + 1}.jpg")
    print(f"[{ds}] {len(man)} hands -> {d}")


def build_casia(a, out_root):
    gt = load_json(a.casia_labels)
    idx = index_dir(Path(a.casia_images))
    files = sorted(f for f in gt if stem(f) in idx and "_850_" in f)
    rnd = random.Random(a.seed)
    rnd.shuffle(files)
    d = out_root / "casia"
    for sub in ("upright", "images"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    man = {}
    for f in files[:a.n_casia]:
        raw = cv2.imread(str(idx[stem(f)]), cv2.IMREAD_GRAYSCALE)
        up = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)         # fingers left -> fingers up
        h, w = up.shape
        M = frame_matrix(w, h)                                # = exact inverse rotation (scale 1, no padding)
        name = "casia_" + stem(f)
        cv2.imwrite(str(d / "upright" / f"{name}.png"), up)
        cv2.imwrite(str(d / "images" / f"{name}.png"), raw)
        man[name] = {"source": str(idx[stem(f)]), "hand": "?", "subject": stem(f).split("_")[0],
                     "M": M.tolist(), "size_upright": [w, h], "gt": gt[f]}
    save_json(man, d / "manifest.json")
    contact_sheet([d / "upright" / f"{n}.png" for n in sorted(man)[:48]], d / "contact_sheet_1.jpg")
    print(f"[casia] {len(man)} images -> {d}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vera_root")
    p.add_argument("--scut_root")
    p.add_argument("--casia_images")
    p.add_argument("--casia_labels", help="oriented GT json (raw frame), e.g. out/labels_oriented/seed1337/test_labels.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=100, help="images per dataset (total target, incl. existing when --extend)")
    p.add_argument("--n_vera", type=int, default=None)
    p.add_argument("--n_scut", type=int, default=None)
    p.add_argument("--one_per_subject", action="store_true",
                   help="at most ONE image per person (VERA: one hand of each of the 110 subjects)")
    p.add_argument("--extend", action="store_true",
                   help="keep the existing selection (and its labels), drop same-subject duplicates, add new images up to --n")
    p.add_argument("--n_casia", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    out = Path(a.out_dir)
    if a.vera_root:
        build("vera", collect_vera(Path(a.vera_root)), a.n_vera or a.n, a.seed, out, a.one_per_subject, a.extend)
    if a.scut_root:
        build("scut", collect_scut(Path(a.scut_root)), a.n_scut or a.n, a.seed, out, False, a.extend)
    if a.casia_images and a.casia_labels:
        build_casia(a, out)
    print("Next: look at contact_sheet_*.jpg, then  python xd_label.py --ds_dir <out>\\<ds>")


if __name__ == "__main__":
    main()
