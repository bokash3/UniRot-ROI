#!/usr/bin/env python3
"""
xd_match.py — zero-shot cross-dataset MATCHING with the ROIs of every localiser (VERA, SCUT-PV)
==============================================================================================
Question: does a ROI localiser trained on CASIA-MS only give crops that recognise people on datasets it has never
seen, and how close does it get to the ROI the dataset owners extracted themselves (SCUT official ROI)?

No hand labels are needed: identities come from the file names. Everything trained on CASIA is used unchanged.

ROI sources  (oriented square -> upright 128 x 128 crop, same warp as every other script)
  pipeline   live Stage I + II + III, per seed
  yolo       YOLOv8-pose (no augmentation), per seed
  yolo_aug   YOLOv8-pose + strong augmentation, per seed (skipped for seeds whose weights do not exist yet)
  official   SCUT only: ROI images released with the dataset (ROI/ROIscale/scale1.0), resized to 128

Matchers  (three different families, so a conclusion cannot be an artefact of one matcher)
  compcode   Gabor orientation coding, Hamming distance with +/-2 px shift (same parameters as Table 5)
  resnet18   the CASIA-trained ResNet-18 of the same seed (zero-shot), cosine distance
  miura      vein pattern: Maximum Curvature (Miura et al. 2007) -> binary vein map -> Miura match
             (cross-correlation with +/- --miura_shift px). sigma is fixed on CASIA TRAINING identities only
             (run once with --calibrate_mc), then used unchanged on VERA / SCUT.

Protocols
  VERA : all 220 hands (identity = subject + hand), session 1 (5 images) = gallery, session 2 (5) = probes
  SCUT : the 300 hands of out/cross/scut/manifest.json (identity = hand id), images 1-5 = gallery, 6-10 = probes
  probe -> identity distance = min over that identity's gallery templates; EER over probe-vs-identity scores; Rank-1.

Everything is SAVED (out_dir/<ds>/<source>_seed<S>/):
  rois.json            oriented square of every image (CASIA frame: fingers left, 768 x 576)
  crops.npz            all 128 x 128 crops;   contact_sheet.jpg  first 48 crops (are they upright palms?)
  veins.npz            binary vein maps (miura)
  stages/<name>_strip.png   crop | Maximum-Curvature response | binary veins | veins over crop  (first --dump_n)
  scores_<matcher>.npz genuine / impostor distances (for DET curves and bootstrap)
out_dir/<ds>/compare_sources.jpg   the same images cropped by every source side by side
out_dir/results_xmatch.csv  +  results_xmatch_summary.csv  (mean +- std over seeds)

Examples (Windows, from reviewer_experiments)
  python xd_match.py --calibrate_mc --casia_images ../data/casia/images ^
         --casia_labels out\\labels_oriented\\seed42\\train_labels.json
  python xd_match.py --datasets vera,scut
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from rx_common import index_dir, label_to_px, load_json, save_json, stem
from rx_stages import oriented_crop, put_text, save_stages, YELLOW
from xd_prep import frame_matrix

W0, H0 = 768, 576
CROP = 128


# =============================================================================================== datasets
def list_vera(root: Path, max_hands: int = 0):
    items = []
    for f in sorted(Path(root).rglob("*.png")):
        t = f.stem.split("_")                                      # 001_L_1_1  subject_hand_session_sample
        if len(t) != 4:
            continue
        items.append({"name": f"vera_{f.stem}", "path": str(f), "identity": f"{t[0]}_{t[1]}",
                      "gallery": t[2] == "1"})
    if max_hands:
        keep = sorted({i["identity"] for i in items})[:max_hands]
        items = [i for i in items if i["identity"] in keep]
    return items


def list_scut(root: Path, hands):
    items = []
    for h in sorted(hands, key=lambda x: int(x)):
        for k in range(1, 11):
            f = Path(root) / "RAW" / h / f"{h}_{k}.bmp"
            if f.is_file():
                items.append({"name": f"scut_{h}_{k}", "path": str(f), "identity": h, "gallery": k <= 5,
                              "official": str(Path(root) / "ROI" / "ROIscale" / "{scale}" / h / f"{h}_{k}.jpg")})
    return items


def load_frame(item):
    """upright image (fingers up) -> CASIA frame (fingers left, 768 x 576), same transform as xd_prep.py"""
    up = cv2.imread(item["path"], cv2.IMREAD_GRAYSCALE)
    h, w = up.shape
    return cv2.warpAffine(up, frame_matrix(w, h), (W0, H0), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# =============================================================================================== localisers
def localise_yolo(weights, items, a):
    from ultralytics import YOLO
    from baseline_yolo import decode, to_square
    model = YOLO(weights)
    rois, crops, n_miss = {}, [], 0
    fb = (W0 * 0.55, H0 * 0.5, 200.0, 90.0)                       # fallback if nothing is detected
    for i in range(0, len(items), 32):
        chunk = items[i:i + 32]
        frames = [load_frame(it) for it in chunk]
        res = model.predict(source=[cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in frames], imgsz=640, conf=0.001,
                            max_det=5, device=a.device, verbose=False)
        for it, fr, r in zip(chunk, frames, res):
            d = decode("pose", r)
            sq = to_square("pose", d[0]["pts"]) if d else fb
            n_miss += 0 if d else 1
            rois[it["name"]] = list(sq) + [d[0]["conf"] if d else 0.0]
            crops.append(oriented_crop(fr, *sq, out=CROP))
        if (i // 32) % 20 == 0:
            print(f"    {min(i + 32, len(items))}/{len(items)}")
    if n_miss:
        print(f"    [WARN] no detection in {n_miss} images (fallback square used)")
    return rois, np.stack(crops)


def localise_pipeline(seed, items, a, dev, aug=False):
    import full_pipeline_eval as fp
    pats = ((a.seg_aug_pattern, a.rot_aug_pattern, a.roi_aug_pattern) if aug else
            (a.seg_ckpt_pattern, a.rot_ckpt_pattern, a.roi_ckpt_pattern))
    ns = Namespace(pipeline_dir=a.pipeline_dir, seg_ckpt_pattern=pats[0],
                   rot_ckpt_pattern=pats[1], roi_ckpt_pattern=pats[2], random_weights=False)
    (seg, rot, roi), RotCls = fp.build_models(ns, seed, dev)
    rois, crops = {}, []
    for i, it in enumerate(items):
        fr = load_frame(it)
        rgb = cv2.cvtColor(fr, cv2.COLOR_GRAY2RGB)
        m = fp.stage1_mask(seg, rgb, dev)
        ang = fp.stage2_angle(rot, RotCls, m, dev)
        sq = fp.to_raw(fp.stage3_square(roi, fp.rotate_mask(m, ang), dev), ang, W0, H0)
        rois[it["name"]] = list(sq)
        crops.append(oriented_crop(fr, *sq, out=CROP))
        if i % 500 == 0:
            print(f"    {i}/{len(items)}")
    return rois, np.stack(crops)


def load_official(items, a):
    crops = []
    for it in items:
        im = cv2.imread(it["official"].format(scale=f"scale{a.official_scale}"), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(it["official"].format(scale=f"scale{a.official_scale}"))
        if a.official_transpose:
            im = im.T.copy()                                     # the SCUT ROIs were cut from a transposed image
        crops.append(cv2.resize(im, (CROP, CROP), interpolation=cv2.INTER_AREA))
    return {}, np.stack(crops)


# =============================================================================================== vein extraction
def _profile_scores(p):
    """Miura 2007 step 2: every connected run of positive curvature scores (max curvature x width) at its peak."""
    out = np.zeros_like(p)
    pos = p > 0
    if not pos.any():
        return out
    d = np.diff(np.concatenate(([0], pos.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        seg = p[s:e]
        j = int(seg.argmax())
        out[s + j] += seg[j] * (e - s)
    return out


def max_curvature(img: np.ndarray, sigma: float):
    """Maximum Curvature (Miura, Nagasaka, Miyatake 2007). Returns (G response, binary vein map uint8)."""
    from scipy.ndimage import gaussian_filter
    f = img.astype(np.float64)
    f = (f - f.mean()) / (f.std() + 1e-6)
    fx = gaussian_filter(f, sigma, order=(0, 1))
    fy = gaussian_filter(f, sigma, order=(1, 0))
    fxx = gaussian_filter(f, sigma, order=(0, 2))
    fyy = gaussian_filter(f, sigma, order=(2, 0))
    fxy = gaussian_filter(f, sigma, order=(1, 1))
    r2 = math.sqrt(2.0)
    d1 = [fx, fy, (fx + fy) / r2, (fx - fy) / r2]                 # first derivative along 0, 90, 45, 135 deg
    d2 = [fxx, fyy, (fxx + 2 * fxy + fyy) / 2, (fxx - 2 * fxy + fyy) / 2]
    k = [b / (1 + a_ ** 2) ** 1.5 for a_, b in zip(d1, d2)]       # curvature; dark vein -> k > 0
    H, W = f.shape
    V = np.zeros_like(f)
    for y in range(H):                                            # horizontal profiles
        V[y] += _profile_scores(k[0][y])
    for x in range(W):                                            # vertical profiles
        V[:, x] += _profile_scores(k[1][:, x])
    ys, xs = np.mgrid[0:H, 0:W]
    for kk, flip in ((k[2], False), (k[3], True)):                # diagonal profiles
        K = kk[::-1] if flip else kk
        Y = ys[::-1] if flip else ys
        for o in range(-(H - 1), W):
            sc = _profile_scores(np.diagonal(K, o).copy())
            V[np.diagonal(Y, o), np.diagonal(xs, o)] += sc
    P = np.pad(V, 2)

    def sh(dy, dx):
        return P[2 + dy:2 + dy + H, 2 + dx:2 + dx + W]
    C = [np.minimum(np.maximum(sh(dy, dx), sh(2 * dy, 2 * dx)), np.maximum(sh(-dy, -dx), sh(-2 * dy, -2 * dx)))
         for dy, dx in ((0, 1), (1, 0), (1, 1), (-1, 1))]        # step 3: connect vein centres
    G = np.max(C, axis=0)
    G[:2] = G[-2:] = 0
    G[:, :2] = G[:, -2:] = 0
    pos = G[G > 0]
    thr = np.median(pos) if pos.size else 0.0
    return G, (G > thr).astype(np.uint8)


def _mc_job(args):
    crops, sigma = args
    return np.stack([max_curvature(c, sigma)[1] for c in crops])


def veins_of(crops, sigma, workers):
    chunks = [(crops[i:i + 64], sigma) for i in range(0, len(crops), 64)]
    if workers <= 1:
        return np.concatenate([_mc_job(c) for c in chunks])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return np.concatenate(list(ex.map(_mc_job, chunks)))


# =============================================================================================== distances (GPU)
def _torch(dev):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False                 # exact integer counts in the matmuls
    torch.backends.cudnn.allow_tf32 = False
    return torch


def compcode_codes(crops, dev):
    torch = _torch(dev)
    sys.path.insert(0, str(Path(ARGS.matching_dir).resolve()))
    import matching_experiment as mx
    ks = np.stack([mx._compcode_kernel(35, np.pi * i / 6) for i in range(6)])[:, None].astype(np.float32)
    w = torch.from_numpy(ks).to(dev)
    out = []
    for i in range(0, len(crops), 256):
        g = torch.from_numpy(crops[i:i + 256].astype(np.float32)).to(dev)[:, None]
        g = (g - g.mean((2, 3), keepdim=True)) / (g.std((2, 3), keepdim=True) + 1e-6)
        r = torch.nn.functional.conv2d(g, w, padding=17)           # symmetric kernels: conv == correlation
        out.append(r.argmin(1).to(torch.uint8).cpu().numpy())
    return np.concatenate(out)


def dist_compcode(Pc, Gc, shift, dev):
    torch = _torch(dev)
    oh = lambda X: torch.nn.functional.one_hot(torch.from_numpy(X).to(dev).long(), 6).permute(0, 3, 1, 2).float()
    G = oh(Gc)
    H, W = Pc.shape[1:]
    D = []
    for i in range(0, len(Pc), 400):
        P = oh(Pc[i:i + 400])
        best = torch.ones((len(P), len(G)), device=dev)
        for dy in range(-shift, shift + 1):
            for dx in range(-shift, shift + 1):
                y1a, y2a, x1a, x2a = max(0, -dy), H - max(0, dy), max(0, -dx), W - max(0, dx)
                y1b, y2b, x1b, x2b = max(0, dy), H - max(0, -dy), max(0, dx), W - max(0, -dx)
                a_ = P[:, :, y1a:y2a, x1a:x2a].reshape(len(P), -1)
                b_ = G[:, :, y1b:y2b, x1b:x2b].reshape(len(G), -1)
                best = torch.minimum(best, 1.0 - (a_ @ b_.T) / float((y2a - y1a) * (x2a - x1a)))
        D.append(best.cpu().numpy())
    return np.concatenate(D)


def dist_miura(Pv, Gv, s, dev):
    """Miura match: probe centre (crop s px each side) slid over the gallery map within +/- s px.
    score = max_t N(t) / (|probe| + |gallery window at t|) in [0, 0.5]; distance = 0.5 - score."""
    torch = _torch(dev)
    G = torch.from_numpy(Gv).to(dev).float()
    H, W = Pv.shape[1:]
    hp, wp = H - 2 * s, W - 2 * s
    D = []
    for i in range(0, len(Pv), 500):
        P = torch.from_numpy(Pv[i:i + 500]).to(dev).float()[:, s:s + hp, s:s + wp].reshape(-1, hp * wp)
        sp = P.sum(1, keepdim=True)
        bestN = torch.zeros((len(P), len(G)), device=dev)
        bestD = torch.ones((len(P), len(G)), device=dev)
        for t0 in range(2 * s + 1):
            for s0 in range(2 * s + 1):
                M = G[:, t0:t0 + hp, s0:s0 + wp].reshape(len(G), -1)
                N = P @ M.T
                den = sp + M.sum(1)[None]
                up = N > bestN
                bestN = torch.where(up, N, bestN)
                bestD = torch.where(up, den, bestD)
        D.append((0.5 - bestN / bestD.clamp(min=1.0)).cpu().numpy())
    return np.concatenate(D)


def resnet_feats(crops, seed, dev):
    sys.path.insert(0, str(Path(ARGS.matching_dir).resolve()))
    import matching_experiment as mx
    global _RN
    if "_RN" not in globals():
        _RN = mx.ResNet18Matcher(ckpt_pattern=ARGS.resnet_ckpt_pattern, device=str(dev))
    f = [_RN.encode_batch(list(crops[i:i + 64]), seed=str(seed)) for i in range(0, len(crops), 64)]
    return np.concatenate(f)


# =============================================================================================== protocol
def compute_eer(gen, imp):
    g, i = np.sort(gen), np.sort(imp)
    thr = np.linspace(min(g[0], i[0]), max(g[-1], i[-1]), 4000)
    far = np.searchsorted(i, thr, side="right") / len(i)
    frr = (len(g) - np.searchsorted(g, thr, side="left")) / len(g)
    k = int(np.argmin(np.abs(far - frr)))
    return 50.0 * float(far[k] + frr[k])


def evaluate(D, p_ids, g_ids):
    ids = sorted(set(g_ids))
    col = {x: k for k, x in enumerate(ids)}
    gk = np.array([col[x] for x in g_ids])
    PI = np.stack([D[:, gk == k].min(1) for k in range(len(ids))], 1)
    true = np.array([col.get(x, -1) for x in p_ids])
    keep = true >= 0
    PI, true = PI[keep], true[keep]
    r = np.arange(len(PI))
    gen = PI[r, true]
    m = np.ones_like(PI, bool)
    m[r, true] = False
    imp = PI[m]
    return {"EER_pct": compute_eer(gen, imp), "Rank1_pct": 100.0 * float((PI.argmin(1) == true).mean()),
            "n_ids": len(ids), "n_probes": int(keep.sum()), "n_genuine": len(gen), "n_impostor": len(imp)}, gen, imp


def run_matchers(crops, items, seed, outd, sigma, dev):
    gal = np.array([it["gallery"] for it in items])
    ids = np.array([it["identity"] for it in items])
    res = {}
    for mname in ARGS.matchers.split(","):
        if mname == "compcode":
            c = compcode_codes(crops, dev)
            D = dist_compcode(c[~gal], c[gal], ARGS.cc_shift, dev)
        elif mname == "miura":
            v = veins_of(crops, sigma, ARGS.workers)
            np.savez_compressed(outd / "veins.npz", veins=v)
            D = dist_miura(v[~gal], v[gal], ARGS.miura_shift, dev)
        elif mname == "resnet18":
            if seed is None:
                continue
            f = resnet_feats(crops, seed, dev)
            D = 1.0 - f[~gal] @ f[gal].T
        else:
            raise ValueError(mname)
        r, gen, imp = evaluate(D, ids[~gal], ids[gal])
        np.savez_compressed(outd / f"scores_{mname}{'' if seed is None else f'_rnseed{seed}'}.npz", genuine=gen,
                            impostor=imp)
        res[mname] = r
        print(f"    {mname:9s} EER={r['EER_pct']:6.2f}%  Rank-1={r['Rank1_pct']:6.2f}%  "
              f"(ids={r['n_ids']}, probes={r['n_probes']})")
    return res


# =============================================================================================== saving helpers
def sheet(imgs, path, cols=8, labels=None):
    tiles = []
    for k, im in enumerate(imgs):
        t = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im.copy()
        if labels:
            put_text(t, labels[k][:16], (2, 12), YELLOW, 0.35)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def dump_veins(outd, items, crops, sigma, n):
    for it, c in list(zip(items, crops))[:n]:
        G, b = max_curvature(c, sigma)
        g8 = (255 * G / (G.max() + 1e-9)).astype(np.uint8)
        ov = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        ov[b > 0] = (0, 0, 255)
        save_stages(outd / "stages", it["name"],
                    [("01_crop", "crop 128x128", c), ("02_max_curvature", f"Max curvature (sigma={sigma})", g8),
                     ("03_binary_veins", "binary veins", b * 255), ("04_overlay", "veins over crop", ov)],
                    {"name": it["name"], "identity": it["identity"], "sigma": sigma})


# =============================================================================================== calibration
def calibrate_mc(a):
    """choose sigma of Maximum Curvature on CASIA-MS TRAINING identities (GT crops, 850 nm)."""
    sys.path.insert(0, str(Path(a.matching_dir).resolve()))
    import matching_experiment as mx
    lab = load_json(a.casia_labels)
    idx = index_dir(Path(a.casia_images))
    items, crops = [], []
    for f in sorted(lab):
        m = mx.parse_casia_filename(Path(f).name)
        if not m or m["band"] != "850" or stem(f) not in idx:
            continue
        g = cv2.imread(str(idx[stem(f)]), cv2.IMREAD_GRAYSCALE)
        crops.append(oriented_crop(g, *label_to_px(lab[f], g.shape[1], g.shape[0]), out=CROP))
        items.append({"name": stem(f), "identity": m["identity"], "gallery": m["sample"] <= 3})
    crops = np.stack(crops)
    dev = _dev(a)
    out = Path(a.out_dir) / "mc_calibration"
    out.mkdir(parents=True, exist_ok=True)
    gal = np.array([i["gallery"] for i in items])
    ids = np.array([i["identity"] for i in items])
    print(f"[calib] CASIA train identities: {len(set(ids))}, images: {len(items)}")
    res = {}
    for s in [float(x) for x in a.sigma_grid.split(",")]:
        v = veins_of(crops, s, a.workers)
        r, _, _ = evaluate(dist_miura(v[~gal], v[gal], a.miura_shift, dev), ids[~gal], ids[gal])
        res[s] = r
        print(f"  sigma={s:4.1f}  EER={r['EER_pct']:6.2f}%  Rank-1={r['Rank1_pct']:6.2f}%  "
              f"vein pixels={100 * v.mean():.1f}%")
        dump_veins(out / f"sigma{s}", items, crops, s, 8)
    best = min(res, key=lambda s: res[s]["EER_pct"])
    save_json({"sigma": best, "miura_shift": a.miura_shift, "grid": {str(k): v for k, v in res.items()},
               "note": "chosen on CASIA-MS training identities only (seed split of --casia_labels)"},
              out / "mc_calibration.json")
    print(f"[calib] chosen sigma = {best}  -> {out / 'mc_calibration.json'}")


# =============================================================================================== main
def _dev(a):
    import torch
    return torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")



def force_fp32():
    """Disable TF32 so every script runs in the same, full-FP32 arithmetic (Ampere GPUs otherwise use TF32 in
    cuDNN convolutions by default, which changes a few pipeline outputs on out-of-domain images)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vera_root", default=r"../datasets/VERA-Palmvein/raw")
    p.add_argument("--scut_root", default=r"../datasets/SCUT_PV_V1")
    p.add_argument("--cross_dir", default=r"out\cross", help="SCUT hands are taken from <cross_dir>/scut/manifest.json")
    p.add_argument("--out_dir", default=r"out\xmatch")
    p.add_argument("--datasets", default="vera,scut")
    p.add_argument("--sources", default="pipeline,yolo,yolo_aug,official")
    p.add_argument("--seeds", default="42,2025,1337")
    p.add_argument("--matchers", default="compcode,resnet18,miura")
    p.add_argument("--yolo_weights", default=r"runs_yolo\pose_seed{seed}\weights\best.pt")
    p.add_argument("--yolo_aug_weights", default=r"runs_yolo\pose_aug_seed{seed}\weights\best.pt")
    p.add_argument("--yolo_pre_weights", default=r"runs_yolo\pose_pre_seed{seed}\weights\best.pt",
                   help="COCO-pretrained YOLOv8n-pose trained with the augmented recipe (source yolo_pre)")
    p.add_argument("--pipeline_dir", default=r"../pipeline_models")
    p.add_argument("--seg_ckpt_pattern", default=r"../checkpoints/basic/stage1_seg_mobilenetv2_seed{seed}.pt")
    p.add_argument("--rot_ckpt_pattern", default=r"../checkpoints/basic/stage2_unirot_angle_seed{seed}.pt")
    p.add_argument("--roi_ckpt_pattern", default=r"../checkpoints/basic/stage3_roi_seed{seed}.pth")
    p.add_argument("--matching_dir", default=r".")
    p.add_argument("--seg_aug_pattern", default=r"runs_pipeline_aug\stage1_seed{seed}\best.pt")
    p.add_argument("--rot_aug_pattern", default=r"runs_pipeline_aug\stage2_seed{seed}\best.pt")
    p.add_argument("--roi_aug_pattern", default=r"runs_pipeline_aug\stage3_seed{seed}\best.pt")
    p.add_argument("--resnet_ckpt_pattern", default=r"checkpoints_v2\resnet18_seed{seed}.pt")
    p.add_argument("--mc_sigma", type=float, default=None, help="default: value chosen by --calibrate_mc (else 3.0)")
    p.add_argument("--miura_shift", type=int, default=8, help="+/- px search of Miura match on 128 px maps")
    p.add_argument("--cc_shift", type=int, default=2)
    p.add_argument("--official_scale", default="1.0")
    p.add_argument("--official_transpose", type=int, default=1, help="1 = undo the transpose used by SCUT")
    p.add_argument("--vera_max_hands", type=int, default=0, help="0 = all 220 hands")
    p.add_argument("--workers", type=int, default=6, help="CPU processes for Maximum Curvature")
    p.add_argument("--dump_n", type=int, default=12, help="vein stage strips per source (seed 42 / official)")
    p.add_argument("--force", action="store_true", help="recompute ROIs/crops even if cached")
    p.add_argument("--device", default="cuda")
    p.add_argument("--calibrate_mc", action="store_true")
    p.add_argument("--casia_images", default=r"../data/casia/images")
    p.add_argument("--casia_labels", default=r"out\labels_oriented\seed42\train_labels.json")
    p.add_argument("--sigma_grid", default="1.5,2,3,4,5")
    a = p.parse_args()
    force_fp32()
    ARGS = a
    if a.calibrate_mc:
        calibrate_mc(a)
        return
    cal = Path(a.out_dir) / "mc_calibration" / "mc_calibration.json"
    sigma = a.mc_sigma if a.mc_sigma is not None else (load_json(cal)["sigma"] if cal.is_file() else 3.0)
    if "miura" in a.matchers:
        print(f"[INFO] Maximum Curvature sigma = {sigma}" + ("" if cal.is_file() or a.mc_sigma else
                                                              "  (WARNING: not calibrated, run --calibrate_mc)"))
    dev = _dev(a)
    rows = []
    res_csv = Path(a.out_dir) / "results_xmatch.csv"
    old = list(csv.DictReader(open(res_csv))) if res_csv.is_file() else []   # keep rows of earlier runs
    for ds in [d for d in a.datasets.split(",") if d]:
        if ds == "vera":
            items = list_vera(Path(a.vera_root), a.vera_max_hands)
        else:
            hands = sorted({v["subject"] for v in load_json(Path(a.cross_dir) / "scut" / "manifest.json").values()})
            items = list_scut(Path(a.scut_root), hands)
        n_id = len({i["identity"] for i in items})
        print(f"\n===== {ds}: {len(items)} images, {n_id} identities "
              f"(gallery {sum(i['gallery'] for i in items)} / probes {sum(not i['gallery'] for i in items)}) =====")
        compare = {}
        for src in [s for s in a.sources.split(",") if s]:
            if src == "official" and ds != "scut":
                continue
            seeds = [None] if src == "official" else [int(s) for s in a.seeds.split(",")]
            for seed in seeds:
                tag = f"{src}_seed{seed}" if seed is not None else src
                outd = Path(a.out_dir) / ds / tag
                outd.mkdir(parents=True, exist_ok=True)
                print(f"\n  --- {ds} | {tag} ---")
                if (outd / "crops.npz").is_file() and not a.force:
                    crops = np.load(outd / "crops.npz")["crops"]
                    print("    crops loaded from cache")
                else:
                    if src == "official":
                        rois, crops = load_official(items, a)
                    elif src.startswith("yolo"):
                        # yolo, yolo_aug, yolo_pre as before; any other yolo_<tag> (yolo_s_pre, yolo_m_pre, yolo_aug_t101, ...)
                        # reads runs_yolo\pose_<tag>_seed<S>\weights\best.pt
                        w = {"yolo": a.yolo_weights, "yolo_aug": a.yolo_aug_weights,
                             "yolo_pre": a.yolo_pre_weights}.get(
                            src, "runs_yolo\\pose_" + src[len("yolo_"):] + "_seed{seed}\\weights\\best.pt").format(seed=seed)
                        if not Path(w).is_file():
                            print(f"    [SKIP] weights not found: {w}")
                            continue
                        rois, crops = localise_yolo(w, items, a)
                    elif src in ("pipeline", "pipeline_aug"):
                        rois, crops = localise_pipeline(seed, items, a, dev, aug=src == "pipeline_aug")
                    else:
                        raise ValueError(src)
                    save_json(rois, outd / "rois.json")
                    np.savez_compressed(outd / "crops.npz", crops=crops, names=np.array([i["name"] for i in items]))
                sheet(list(crops[:48]), outd / "contact_sheet.jpg", labels=[i["name"] for i in items[:48]])
                if seed in (None, 42):
                    compare[tag] = crops[:24]
                    if "miura" in a.matchers:
                        dump_veins(outd, items, crops, sigma, a.dump_n)
                gal = np.array([i["gallery"] for i in items])
                ids = np.array([i["identity"] for i in items])
                if src == "official" and "resnet18" in a.matchers:     # official crops: ResNet of every seed
                    for s in [int(x) for x in a.seeds.split(",")]:
                        f = resnet_feats(crops, s, dev)
                        r, gen, imp = evaluate(1.0 - f[~gal] @ f[gal].T, ids[~gal], ids[gal])
                        np.savez_compressed(outd / f"scores_resnet18_rnseed{s}.npz", genuine=gen, impostor=imp)
                        print(f"    resnet18  (seed {s}) EER={r['EER_pct']:6.2f}%  Rank-1={r['Rank1_pct']:6.2f}%")
                        rows.append({"dataset": ds, "source": src, "seed": s, "matcher": "resnet18", **r})
                res = run_matchers(crops, items, seed, outd, sigma, dev)    # seed None -> resnet18 skipped here
                for m, r in res.items():
                    rows.append({"dataset": ds, "source": src, "seed": seed if seed is not None else "na",
                                 "matcher": m, **r})
                key = lambda r: (r["dataset"], r["source"], str(r["seed"]), r["matcher"])
                new_keys = {key(r) for r in rows}
                write_csv(res_csv, [r for r in old if key(r) not in new_keys] + rows)
        if compare:                                                   # same images, every source, side by side
            keys = list(compare)
            n = min(len(v) for v in compare.values())
            grid = []
            for j in range(n):
                row = [cv2.cvtColor(compare[k][j], cv2.COLOR_GRAY2BGR) for k in keys]
                grid.append(np.hstack(row))
            head = np.zeros((18, grid[0].shape[1], 3), np.uint8)
            for q, k in enumerate(keys):
                put_text(head, k[:18], (q * CROP + 2, 13), YELLOW, 0.38)
            cv2.imwrite(str(Path(a.out_dir) / ds / "compare_sources.jpg"), np.vstack([head] + grid))
    summarize(res_csv)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(res_csv):
    if not res_csv.is_file():
        return
    rows = list(csv.DictReader(open(res_csv)))
    out = []
    print("\n===== mean +- std over seeds =====")
    for key in dict.fromkeys((r["dataset"], r["source"], r["matcher"]) for r in rows):
        sel = [r for r in rows if (r["dataset"], r["source"], r["matcher"]) == key]
        e = np.array([float(r["EER_pct"]) for r in sel])
        k = np.array([float(r["Rank1_pct"]) for r in sel])
        sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
        line = {"dataset": key[0], "source": key[1], "matcher": key[2], "n_seeds": len(sel),
                "EER_pct": f"{e.mean():.2f} +- {sd(e):.2f}", "Rank1_pct": f"{k.mean():.2f} +- {sd(k):.2f}"}
        out.append(line)
        print(f"  {key[0]:5s} {key[1]:9s} {key[2]:9s} n={len(sel)}  EER={line['EER_pct']:>14s}  Rank-1={line['Rank1_pct']}")
    write_csv(res_csv.with_name("results_xmatch_summary.csv"), out)


if __name__ == "__main__":
    main()
