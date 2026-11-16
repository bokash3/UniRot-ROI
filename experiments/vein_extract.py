#!/usr/bin/env python3
"""
vein_extract.py — turn palm ROI images into vein images with Maximum Curvature (Miura et al., 2007)
===================================================================================================
Needs only numpy + opencv (no scipy, no torch).

Method (per ROI, all steps on a fixed-size crop so that sigma means the same thing for every dataset)
  1. resize the ROI to --size x --size (default 128) and normalise it (zero mean, unit std)
  2. Gaussian-derivative filters (scale sigma): first and second derivatives along 4 directions (0, 90, 45, 135 deg)
  3. curvature  k = f'' / (1 + f'^2)^(3/2)   -> a vein is a dark valley, so k > 0 across it
  4. along every row / column / diagonal: each run of positive curvature (= one valley crossing) gives
     score = (max curvature in the run) x (run width), placed at the centre (peak) of the valley; summed -> V
  5. connection: a pixel keeps a value only if the pixels on BOTH sides along a direction are also vein centres
        C_d = min( max(V[+1], V[+2]), max(V[-1], V[-2]) ),  G = max over the 4 directions
     (fills small gaps, removes isolated dots)
  6. binary veins = G > median of the positive values of G

Outputs (one sub-folder each, same relative paths / file names as the input)
  input/       the resized ROI that was processed (pixel-aligned with every other output -> training pairs)
  response/    G as an 8-bit grey image (0 = no vein, 255 = strongest vein)       <- middle column of the preview
  veins_gray/  grey vein image: the vein pixels keep their own intensity texture (ROI inverted so veins are bright),
               weighted by the response, background black. Keeps the detail that a binary map throws away.
  binary/      (--save_binary) 0/255 vein mask
  overlay/     (--save_overlay) veins in red over the ROI (for checking only)
  response_float/ (--save_float) G as float32 .npy, no 8-bit quantisation
  contact_sheet_*.jpg   input | response | veins_gray | overlay for the first images
  params.json  every parameter used (for reproducibility)

Examples
  # a folder of ROI images (any size, png/jpg/bmp, searched recursively)
  python vein_extract.py --input_dir "../datasets/SCUT_PV_V1/ROI/ROIscale/scale1.0" ^
         --out_dir out\\veins\\scut_official --sigma 3 --workers 6
  # crops cached by xd_match.py
  python vein_extract.py --npz out\\xmatch\\vera\\yolo_aug_seed42\\crops.npz --out_dir out\\veins\\vera_yolo_aug
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------------------------- Gaussian derivatives
def _kernel(sigma: float, order: int) -> np.ndarray:
    """1-D Gaussian (derivative) kernel, identical to scipy.ndimage.gaussian_filter (truncate = 4)."""
    r = int(4.0 * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    phi = np.exp(-0.5 * x * x / sigma ** 2)
    phi /= phi.sum()
    if order == 1:
        phi = -x / sigma ** 2 * phi
    elif order == 2:
        phi = (x * x - sigma ** 2) / sigma ** 4 * phi
    return phi[::-1].copy()          # cv2 correlates; scipy convolves -> flip


def gderiv(f, sigma, oy, ox):
    return cv2.sepFilter2D(f, cv2.CV_64F, _kernel(sigma, ox), _kernel(sigma, oy), borderType=cv2.BORDER_REFLECT)


# ------------------------------------------------------------------------------------- vectorised profile scoring
def _score_rows(K: np.ndarray) -> np.ndarray:
    """Every row of K is one profile. Each run of K > 0 scores max*width at its (first) maximum."""
    R, L = K.shape
    pos = K > 0
    out = np.zeros(R * L)
    if not pos.any():
        return out.reshape(R, L)
    start = pos & ~np.concatenate([np.zeros((R, 1), bool), pos[:, :-1]], 1)
    rid = (np.cumsum(start.ravel()) * pos.ravel())                  # run id per pixel, 0 = outside a run
    idx = np.flatnonzero(rid)                                       # row-major -> runs are contiguous
    ids, vals = rid[idx], K.ravel()[idx]
    bounds = np.flatnonzero(np.diff(np.concatenate(([0], ids))))   # first element of every run
    mx = np.maximum.reduceat(vals, bounds)
    width = np.diff(np.concatenate((bounds, [len(ids)])))
    run_of = np.repeat(np.arange(len(bounds)), width)
    is_max = vals == mx[run_of]
    first = np.flatnonzero(is_max)
    _, keep = np.unique(run_of[first], return_index=True)
    sel = first[keep]
    out[idx[sel]] = mx[run_of[sel]] * width[run_of[sel]]
    return out.reshape(R, L)


_DIAG = {}


def _diag_index(H, W):
    """gather indices of all diagonals (direction +x,+y), padded with -1."""
    if (H, W) not in _DIAG:
        L = min(H, W)
        rows, cols = [], []
        for o in range(-(H - 1), W):
            r = np.arange(max(0, -o), min(H, W - o))
            c = r + o
            rows.append(np.pad(r, (0, L - len(r)), constant_values=-1))
            cols.append(np.pad(c, (0, L - len(c)), constant_values=-1))
        _DIAG[(H, W)] = (np.array(rows), np.array(cols))
    return _DIAG[(H, W)]


def _score_diag(K):
    H, W = K.shape
    rr, cc = _diag_index(H, W)
    m = rr >= 0
    P = np.zeros(rr.shape)
    P[m] = K[rr[m], cc[m]]
    S = _score_rows(P)
    V = np.zeros_like(K)
    V[rr[m], cc[m]] = S[m]
    return V


def max_curvature(img: np.ndarray, sigma: float, pad: int | None = None, border: int = 2):
    """-> G (float64 response, same size as img). pad: reflect padding used during filtering (default 3*sigma).
    border: px set to 0 at the image edge (ROI files often have a dark 1-2 px frame that looks like a vein)."""
    f = img.astype(np.float64)
    f = (f - f.mean()) / (f.std() + 1e-6)
    p = int(math.ceil(3 * sigma)) if pad is None else pad
    if p:
        f = cv2.copyMakeBorder(f, p, p, p, p, cv2.BORDER_REFLECT)
    fx, fy = gderiv(f, sigma, 0, 1), gderiv(f, sigma, 1, 0)
    fxx, fyy, fxy = gderiv(f, sigma, 0, 2), gderiv(f, sigma, 2, 0), gderiv(f, sigma, 1, 1)
    r2 = math.sqrt(2.0)
    d1 = [fx, fy, (fx + fy) / r2, (fx - fy) / r2]
    d2 = [fxx, fyy, (fxx + 2 * fxy + fyy) / 2, (fxx - 2 * fxy + fyy) / 2]
    k = [b / (1 + a ** 2) ** 1.5 for a, b in zip(d1, d2)]
    V = _score_rows(k[0]) + _score_rows(k[1].T).T + _score_diag(k[2]) + _score_diag(k[3][::-1])[::-1]
    H, W = V.shape
    P = np.pad(V, 2)

    def sh(dy, dx):
        return P[2 + dy:2 + dy + H, 2 + dx:2 + dx + W]
    G = np.max([np.minimum(np.maximum(sh(dy, dx), sh(2 * dy, 2 * dx)), np.maximum(sh(-dy, -dx), sh(-2 * dy, -2 * dx)))
                for dy, dx in ((0, 1), (1, 0), (1, 1), (-1, 1))], axis=0)
    if p:
        G = G[p:-p, p:-p]
    if border:
        G[:border] = G[-border:] = 0
        G[:, :border] = G[:, -border:] = 0
    return G


# ------------------------------------------------------------------------------------- one image -> all outputs
def render(roi_u8: np.ndarray, G: np.ndarray, a) -> dict:
    pos = G[G > 0]
    thr = float(np.median(pos)) if pos.size else 0.0
    binary = (G > thr).astype(np.uint8) * 255
    scale = float(np.percentile(pos, a.norm_pct)) if (pos.size and a.norm_pct < 100) else float(G.max() + 1e-12)
    resp = np.clip(G / (scale + 1e-12), 0, 1)
    out = {"response": (255 * resp).astype(np.uint8)}
    lo, hi = np.percentile(roi_u8, [1, 99])
    inv = 1.0 - np.clip((roi_u8.astype(np.float64) - lo) / max(hi - lo, 1.0), 0, 1)   # veins (dark) -> bright
    w = resp ** a.gamma if a.gray_mode == "soft" else (binary > 0).astype(np.float64)
    out["veins_gray"] = (255 * inv * w / max(float((inv * w).max()), 1e-6)).astype(np.uint8)
    out["binary"] = binary
    ov = cv2.cvtColor(roi_u8, cv2.COLOR_GRAY2BGR)
    ov[binary > 0] = (0, 0, 255)
    out["overlay"] = ov
    return out


def process(job):
    rel, img, a = job
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if a.size:
        img = cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_AREA)
        sigma = a.sigma
    else:                                     # keep native size: sigma is defined at 128 px, scale it
        sigma = a.sigma * min(img.shape) / 128.0
    if a.clahe:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    G = max_curvature(img, sigma, border=a.border)
    o = render(img, G, a)
    base = Path(a.out_dir)
    stem = Path(rel).with_suffix(".png")
    for k in ["input", "response", "veins_gray"] + (["binary"] if a.save_binary else []) + \
             (["overlay"] if a.save_overlay else []):
        d = base / k / stem.parent
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / stem.name), img if k == "input" else o[k])
    if a.save_float:
        d = base / "response_float" / stem.parent
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / (stem.stem + ".npy"), G.astype(np.float32))
    return rel, float((o["binary"] > 0).mean())


def iter_jobs(a):
    if a.npz:
        z = np.load(a.npz, allow_pickle=True)
        crops = z["crops"]
        names = list(z["names"]) if "names" in z else [f"{i:06d}" for i in range(len(crops))]
        for n, c in zip(names, crops):
            yield str(n), c
    else:
        root = Path(a.input_dir)
        for f in sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS):
            im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                yield str(f.relative_to(root)), im


def contact_sheets(a, rels, n=32, per_sheet=8):
    base = Path(a.out_dir)
    rows = []
    for rel in rels[:n]:
        s = Path(rel).with_suffix(".png")
        ims = [cv2.imread(str(base / k / s)) for k in ("input", "response", "veins_gray")]
        b = cv2.imread(str(base / "input" / s), cv2.IMREAD_GRAYSCALE)
        ov = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        G = cv2.imread(str(base / "response" / s), cv2.IMREAD_GRAYSCALE)
        m = G > (np.median(G[G > 0]) if (G > 0).any() else 0)            # same rule as the binary map
        ov[m] = (0.4 * ov[m] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        rows.append(np.hstack(ims + [ov]))
    for i in range(0, len(rows), per_sheet):
        cv2.imwrite(str(base / f"contact_sheet_{i // per_sheet + 1}.jpg"), np.vstack(rows[i:i + per_sheet]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_dir", help="folder of ROI images (recursive)")
    src.add_argument("--npz", help="crops.npz written by xd_match.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--size", type=int, default=128, help="process at size x size (0 = native size, sigma scaled)")
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian scale at 128 px (use the value of --calib)")
    p.add_argument("--calib", default=None, help="mc_calibration.json from xd_match.py --calibrate_mc (sets sigma)")
    p.add_argument("--border", type=int, default=3, help="px zeroed at the ROI edge")
    p.add_argument("--clahe", type=int, default=0, help="1 = CLAHE before extraction (off by default)")
    p.add_argument("--norm_pct", type=float, default=100.0,
                   help="response/ grey scale: 100 = divide by the maximum (darker, like the preview); "
                        "e.g. 99 = by the 99th percentile (brighter, faint veins more visible)")
    p.add_argument("--gray_mode", choices=["soft", "hard"], default="soft",
                   help="veins_gray/: soft = weighted by the response (keeps faint veins), hard = binary mask")
    p.add_argument("--gamma", type=float, default=0.5, help="soft weighting exponent (<1 keeps faint veins)")
    p.add_argument("--save_binary", action="store_true")
    p.add_argument("--save_overlay", action="store_true")
    p.add_argument("--save_float", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.calib:
        a.sigma = float(json.load(open(a.calib))["sigma"])
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({k: v for k, v in vars(a).items()}, open(out / "params.json", "w"), indent=1)
    jobs = ((rel, im, a) for rel, im in iter_jobs(a))
    rels, frac = [], []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for k, (rel, fr) in enumerate(ex.map(process, jobs, chunksize=16)):
                rels.append(rel)
                frac.append(fr)
                if k % 500 == 0:
                    print(f"  {k} images")
    else:
        for k, j in enumerate(jobs):
            rel, fr = process(j)
            rels.append(rel)
            frac.append(fr)
            if k % 500 == 0:
                print(f"  {k} images")
    contact_sheets(a, rels)
    print(f"done: {len(rels)} images -> {out}   (sigma={a.sigma}, mean vein fraction {100 * np.mean(frac):.1f}%)")


if __name__ == "__main__":
    main()
