"""Item 12 of the review: batch-size invariance of the detectors, failures to acquire, and the TF32 effect.

Run from experiments with the GPU interpreter:
  python item12_consistency.py --part all

Parts
  yolo      every YOLO detector (light, augmented, COCO-pretrained), 3 seeds, VERA + SCUT-PV (all images of the
            recognition protocol), with batch sizes 1, 16 (geometric run) and 32 (recognition run), FP32.
            -> out/item12/yolo_summary.csv  (changes between batch sizes)
            -> out/item12/fta_yolo.csv       (images with no detection, per detector/seed/dataset, batch 32)
  pipeline  UniRot-ROI and the original-recipe pipeline, 3 seeds, VERA + SCUT-PV, FP32 vs TF32.
            -> out/item12/pipeline_tf32_summary.csv
Files are resumable: a finished (part, dataset, seed, model) block is skipped when its .npz exists.
Squares are (cx, cy, side, angle) in the CASIA frame (768 x 576); angle in degrees.
"""
import argparse, csv, json, math, sys, time
from argparse import Namespace
from pathlib import Path
import cv2
import numpy as np
import torch

import xd_match as xm
from rx_common import load_json

OUT = Path("out/item12"); OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 2025, 1337)
YOLO_W = {"light": r"runs_yolo\pose_seed{seed}\weights\best.pt",
        "augmented": r"runs_yolo\pose_aug_seed{seed}\weights\best.pt",
        "pretrained": r"runs_yolo\pose_pre_seed{seed}\weights\best.pt"}
FB = (xm.W0 * 0.55, xm.H0 * 0.5, 200.0, 90.0)            # default square of the recognition runs


def set_tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = bool(on)
    torch.backends.cudnn.allow_tf32 = bool(on)


def get_items(ds, a):
    if ds == "vera":
        items = xm.list_vera(Path(a.vera_root), 0)
        root = a.vera_root
    else:
        hands = sorted({v["subject"] for v in load_json(Path(a.cross_dir) / "scut" / "manifest.json").values()})
        items = xm.list_scut(Path(a.scut_root), hands)
        root = a.scut_root
    if not items:
        raise SystemExit(f"[ERROR] no {ds} images found under {root} (check --vera_root / --scut_root)")
    exp = {"vera": 2200, "scut": 3000}[ds]
    print(f"[{ds}] {len(items)} images under {root} (expected {exp})", flush=True)
    if len(items) != exp:
        raise SystemExit(f"[ERROR] {ds}: found {len(items)} images, expected {exp}; point --{ds if ds=='vera' else 'scut'}_root to the folder that holds the raw images only")
    return items


def frames_of(items):
    return [xm.load_frame(it) for it in items]


def yolo_run(model, frames, batch, dev):
    """returns array (n, 6): cx, cy, side, angle, conf, n_det; no detection -> nan squares, n_det 0"""
    from baseline_yolo import decode, to_square
    out = np.full((len(frames), 6), np.nan); out[:, 5] = 0
    for i in range(0, len(frames), batch):
        chunk = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in frames[i:i + batch]]
        res = model.predict(source=chunk, imgsz=640, conf=0.001, max_det=5, device=dev, verbose=False)
        for j, r in enumerate(res):
            d = decode("pose", r)
            if d:
                sq = to_square("pose", d[0]["pts"])
                out[i + j] = [*sq, d[0]["conf"], len(d)]
    return out


def angdiff(a, b):                                         # square angle is defined modulo 90 deg
    d = np.abs(a - b) % 90.0
    return np.minimum(d, 90.0 - d)


def compare(A, B):
    """A, B: (n,6). Returns counts of changes."""
    na, nb = np.isnan(A[:, 0]), np.isnan(B[:, 0])
    both = ~na & ~nb
    dc = np.hypot(A[both, 0] - B[both, 0], A[both, 1] - B[both, 1])
    ds = np.abs(A[both, 2] - B[both, 2])
    da = angdiff(A[both, 3], B[both, 3])
    return {"n": len(A), "det_status_changed": int((na != nb).sum()),
            "n_changed_any": int(((dc > 0) | (da > 0)).sum() + (na != nb).sum()),
            "n_centre_gt_1px": int((dc > 1).sum()), "n_side_gt_1px": int((ds > 1).sum()), "n_angle_gt_1deg": int((da > 1).sum()),
            "max_centre_px": float(dc.max()) if dc.size else 0.0, "max_side_px": float(ds.max()) if ds.size else 0.0,
            "max_angle_deg": float(da.max()) if da.size else 0.0,
            "mean_centre_px": float(dc.mean()) if dc.size else 0.0}


def part_yolo(a, dev):
    from ultralytics import YOLO
    rows, fta = [], []
    for ds in a.datasets:
        items = get_items(ds, a)
        frames = None
        names = [it["name"] for it in items]
        for mname, pat in YOLO_W.items():
            for seed in SEEDS:
                f = OUT / f"yolo_{ds}_{mname}_seed{seed}.npz"
                if f.is_file() and np.load(f)["b1"].shape[0] == len(items):
                    z = np.load(f)
                    R = {b: z[f"b{b}"] for b in (1, 16, 32)}
                else:
                    if frames is None:
                        frames = frames_of(items)
                    model = YOLO(pat.format(seed=seed))
                    R = {}
                    for b in (1, 16, 32):
                        t0 = time.time(); R[b] = yolo_run(model, frames, b, dev)
                        print(f"[yolo] {ds} {mname} seed{seed} batch{b}: {time.time()-t0:.0f}s", flush=True)
                    np.savez(f, names=np.array(names), **{f"b{b}": R[b] for b in R})
                for b in (1, 16):
                    c = compare(R[32], R[b]); c.update(dataset=ds, model=mname, seed=seed, compared=f"batch32_vs_batch{b}")
                    rows.append(c)
                miss = np.isnan(R[32][:, 0])
                fta.append({"dataset": ds, "model": mname, "seed": seed, "images": len(items), "no_detection_batch32": int(miss.sum()),
                            "no_detection_batch16": int(np.isnan(R[16][:, 0]).sum()), "no_detection_batch1": int(np.isnan(R[1][:, 0]).sum()),
                            "missing_names_batch32": ";".join(n for n, m in zip(names, miss) if m)})
    for fn, data in (("yolo_summary.csv", rows), ("fta_yolo.csv", fta)):
        if data:
            with open(OUT / fn, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    print("[yolo] wrote", OUT / "yolo_summary.csv", OUT / "fta_yolo.csv")


def pipeline_squares(seed, frames, a, dev, aug):
    import full_pipeline_eval as fp
    pats = ((a.seg_aug_pattern, a.rot_aug_pattern, a.roi_aug_pattern) if aug else
            (a.seg_ckpt_pattern, a.rot_ckpt_pattern, a.roi_ckpt_pattern))
    ns = Namespace(pipeline_dir=a.pipeline_dir, seg_ckpt_pattern=pats[0], rot_ckpt_pattern=pats[1],
                   roi_ckpt_pattern=pats[2], random_weights=False)
    (seg, rot, roi), RotCls = fp.build_models(ns, seed, dev)
    out = np.zeros((len(frames), 6))
    for i, fr in enumerate(frames):
        rgb = cv2.cvtColor(fr, cv2.COLOR_GRAY2RGB)
        m = fp.stage1_mask(seg, rgb, dev)
        ang = fp.stage2_angle(rot, RotCls, m, dev)
        sq = fp.to_raw(fp.stage3_square(roi, fp.rotate_mask(m, ang), dev), ang, xm.W0, xm.H0)
        out[i, :4] = sq; out[i, 4] = ang
    return out


def part_pipeline(a, dev):
    rows = []
    for ds in a.datasets:
        items = get_items(ds, a)
        frames = frames_of(items)
        if a.pipeline_max:
            frames = frames[:a.pipeline_max]
        for aug, mname in ((True, "UniRot-ROI"), (False, "Original recipe")):
            for seed in SEEDS:
                f = OUT / f"pipe_{ds}_{mname.replace(' ', '_')}_seed{seed}.npz"
                if f.is_file() and np.load(f)["fp32"].shape[0] == len(frames):
                    z = np.load(f); R = {"fp32": z["fp32"], "tf32": z["tf32"]}
                else:
                    R = {}
                    for mode in ("fp32", "tf32"):
                        set_tf32(mode == "tf32"); t0 = time.time()
                        R[mode] = pipeline_squares(seed, frames, a, dev, aug)
                        print(f"[pipeline] {ds} {mname} seed{seed} {mode}: {time.time()-t0:.0f}s", flush=True)
                    set_tf32(False)
                    np.savez(f, fp32=R["fp32"], tf32=R["tf32"])
                A, B = R["fp32"], R["tf32"]
                dc = np.hypot(A[:, 0] - B[:, 0], A[:, 1] - B[:, 1]); da = angdiff(A[:, 3], B[:, 3])
                rows.append({"dataset": ds, "model": mname, "seed": seed, "images": len(A),
                             "n_centre_gt_0.5px": int((dc > .5).sum()), "n_centre_gt_5px": int((dc > 5).sum()),
                             "n_square_angle_gt_1deg": int((da > 1).sum()), "n_frame_angle_gt_1deg": int((np.abs(A[:, 4] - B[:, 4]) > 1).sum()),
                             "max_centre_px": float(dc.max()), "mean_centre_px": float(dc.mean()), "max_square_angle_deg": float(da.max())})
    if rows:
        with open(OUT / "pipeline_tf32_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[pipeline] wrote", OUT / "pipeline_tf32_summary.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all", choices=["yolo", "pipeline", "all"])
    ap.add_argument("--datasets", default="vera,scut")
    ap.add_argument("--vera_root", default=r"../datasets/VERA-Palmvein/raw")
    ap.add_argument("--scut_root", default=r"../datasets/SCUT_PV_V1")
    ap.add_argument("--cross_dir", default=r"out\cross")
    ap.add_argument("--pipeline_dir", default=r"../pipeline_models")
    ap.add_argument("--seg_ckpt_pattern", default=r"../checkpoints/basic/stage1_seg_mobilenetv2_seed{seed}.pt")
    ap.add_argument("--rot_ckpt_pattern", default=r"../checkpoints/basic/stage2_unirot_angle_seed{seed}.pt")
    ap.add_argument("--roi_ckpt_pattern", default=r"../checkpoints/basic/stage3_roi_seed{seed}.pth")
    ap.add_argument("--seg_aug_pattern", default=r"runs_pipeline_aug\stage1_seed{seed}\best.pt")
    ap.add_argument("--rot_aug_pattern", default=r"runs_pipeline_aug\stage2_seed{seed}\best.pt")
    ap.add_argument("--roi_aug_pattern", default=r"runs_pipeline_aug\stage3_seed{seed}\best.pt")
    ap.add_argument("--pipeline_max", type=int, default=0, help="0 = all images; e.g. 500 for a quick test")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.datasets = [d for d in a.datasets.split(",") if d]
    set_tf32(False)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if a.part in ("yolo", "all"):
        part_yolo(a, "0" if dev.type == "cuda" else "cpu")
    if a.part in ("pipeline", "all"):
        part_pipeline(a, dev)


if __name__ == "__main__":
    main()
