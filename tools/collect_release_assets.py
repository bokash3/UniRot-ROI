#!/usr/bin/env python3
"""
collect_release_assets.py - gather the trained models, labels and result files of the paper into one folder
for sharing on request, with a SHA-256 manifest.

Only files that the dataset licences allow to be shared are collected:
  * no image content of any dataset (no images, crops, vein maps, contact sheets or other pictures);
  * nothing from SCUT-PV (reference squares, palm list, per-image predictions, score files), unless --include_scut
    is given for a copy that is shared only with holders of the SCUT-PV licence;
  * the CASIA-MS hand masks only with --include_masks (binary silhouettes derived from the images; check the
    CASIA-MS licence before publishing them).

Usage (paths are the folders of the original experiments):
  python tools/collect_release_assets.py --amani <experiments root> --hr <stage I/II root> --out <release folder>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SEEDS = (42, 2025, 1337)
PICTURE = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


def plan(amani: Path, hr: Path, include_masks: bool, include_scut: bool) -> list[tuple[Path, str]]:
    rx = amani / "ROI_net" / "reviewer_experiments"
    ba = amani / "ROI_net" / "basline_A"
    pb = amani / "palm_roi_benchmark"
    items: list[tuple[Path, str]] = []

    def add(src: Path, dst: str):
        items.append((src, dst))

    # ---------------- models ----------------
    for s in SEEDS:
        add(hr / f"runs/seg_mobilenetv2_{s}/best.pt", f"models/pipeline_basic/stage1_seg_mobilenetv2_seed{s}.pt")
        add(hr / f"runs/seg_resnet34_{s}/best.pt", f"models/pipeline_basic/stage1_seg_resnet34_seed{s}.pt")
        for m in ("reg", "cls", "both"):
            add(hr / f"Hand-Alignment/runsv8/rot_{s}_{m}.pt", f"models/pipeline_basic/stage2_unirot_angle_{m}_seed{s}.pt")
        add(ba / f"roi_net_runs/roi_net_best_seed{s}.pth", f"models/pipeline_basic/stage3_roi_seed{s}.pth")
        for k in (1, 2, 3):
            add(rx / f"runs_pipeline_aug/stage{k}_seed{s}/best.pt", f"models/unirot_roi/stage{k}_seed{s}.pt")
        for tag, name in (("pose", "yolo_basic"), ("pose_aug", "yolo_matched"), ("pose_pre", "yolo_matched_pt"),
                          ("pose_l100", "yolo_basic_100")):
            add(rx / f"runs_yolo/{tag}_seed{s}/weights/best.pt", f"models/{name}/seed{s}.pt")
        for cell, name in (("cellA_image_raw", "image_raw_basic"), ("cellB_mask_raw", "mask_raw_basic"),
                           ("cellC_image_aligned", "image_aligned_basic")):
            add(rx / f"runs/{cell}/seed{s}/best.pth", f"models/ablation_2x2/{name}_seed{s}.pth")
        add(ba / f"checkpoints/resnet18_seed{s}.pt", f"models/resnet18_matcher/seed{s}.pt")
    for p in sorted((rx / "runs_raw_aug").rglob("best.p*")) if (rx / "runs_raw_aug").is_dir() else []:
        add(p, "models/ablation_2x2_matched/" + "_".join(p.relative_to(rx / "runs_raw_aug").parts))

    # ---------------- labels (CASIA-MS, VERA) ----------------
    add(hr / "data/angles.json", "labels/casia/hand_angles.json")
    add(hr / "config/splits/subject_split.json", "labels/casia/subject_split_stage1_2.json")
    add(rx / "out/frame_angles.json", "labels/casia/frame_angles.json")
    if (rx / "out/frame_angles.csv").is_file():
        add(rx / "out/frame_angles.csv", "labels/casia/frame_angles.csv")
    for s in SEEDS:
        d = pb / f"dataset_palm_locator_seed{s}"
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            add(f, f"labels/casia/stage3_upright_frame/seed{s}/{f.name}")
        d = rx / f"out/labels_oriented/seed{s}"
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            add(f, f"labels/casia/raw_frame_oriented/seed{s}/{f.name}")
    for f in ("labels_manual.json", "labels_manual_pass2.json", "manifest.json"):
        add(rx / f"out/cross/casia/{f}", f"labels/casia/repeatability_40_images/{f}")
    for f in ("labels_manual.json", "manifest.json"):
        add(rx / f"out/cross/vera/{f}", f"labels/vera/{f}")
    if include_masks:
        for f in sorted((hr / "data/masks").glob("*.png")):
            add(f, f"labels/casia/hand_masks/{f.name}")

    # ---------------- results ----------------
    for f in sorted((rx / "runs").glob("*.csv")):
        add(f, f"results/tables/{f.name}")
    for f in sorted((rx / "out/bootstrap").glob("*.csv")):
        if include_scut or "scut" not in f.name:
            add(f, f"results/bootstrap/{f.name}")
    for name in ("latency.json", "pca_rotation_results.csv", "paper_tar_far/tar_far_det.json"):
        add(rx / "out" / name, f"results/{Path(name).name}")
    for f in sorted((rx / "out/item12").glob("*.csv")) if (rx / "out/item12").is_dir() else []:
        add(f, f"results/item12/{f.name}")
    for f in sorted((rx / "out/xmatch").glob("results_xmatch*.csv")):
        add(f, f"results/recognition/{f.name}")
    # per-image geometry: CASIA-MS (in-domain) and VERA
    for sub in ("full_pipeline", "full_pipeline_aug") + tuple(f"full_pipeline_sw_{x}" for x in
                                                          ("aao", "aoa", "aoo", "oaa", "oao", "ooa")):
        for s in SEEDS:
            for f in ("per_image.csv", "summary.json"):
                add(rx / f"out/{sub}/seed{s}/{f}", f"results/per_image/casia/{sub}/seed{s}/{f}")
    for tag in ("pose", "pose_aug", "pose_pre", "pose_l100"):
        for s in SEEDS:
            add(rx / f"runs_yolo/{tag}_seed{s}/eval/per_image.csv", f"results/per_image/casia/yolo_{tag}/seed{s}.csv")
    for f in sorted((rx / "out/raw_aug").rglob("casia_seed*.csv")):
        add(f, "results/per_image/casia/raw_aug/" + "_".join(f.relative_to(rx / "out/raw_aug").parts))
    for f in sorted((rx / "out/cross/results/vera").glob("per_image_*.csv")):
        add(f, f"results/per_image/vera/{f.name}")
    for f in sorted((rx / "out/xmatch/vera").rglob("scores_*.npz")):
        add(f, "results/scores/vera/" + "_".join(f.relative_to(rx / "out/xmatch/vera").parts))
    if include_scut:
        for f in sorted((rx / "out/cross/results/scut").glob("per_image_*.csv")):
            add(f, f"scut_licence_holders_only/per_image/{f.name}")
        for f in sorted((rx / "out/xmatch/scut").rglob("scores_*.npz")):
            add(f, "scut_licence_holders_only/scores/" + "_".join(f.relative_to(rx / "out/xmatch/scut").parts))
        for f in ("labels_manual.json", "manifest.json"):
            add(rx / f"out/cross/scut/{f}", f"scut_licence_holders_only/labels/{f}")
    return items


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amani", required=True, type=Path, help="folder that contains ROI_net and palm_roi_benchmark")
    ap.add_argument("--hr", required=True, type=Path, help="folder of the Stage I/II experiments (runs/, Hand-Alignment/, data/)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--include_masks", action="store_true")
    ap.add_argument("--include_scut", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="only list what would be copied")
    a = ap.parse_args()

    items = plan(a.amani, a.hr, a.include_masks, a.include_scut)
    missing = [str(s) for s, _ in items if not s.is_file()]
    found = [(s, d) for s, d in items if s.is_file() and s.suffix.lower() not in PICTURE]
    print(f"{len(found)} files to copy, {sum(s.stat().st_size for s, _ in found) / 1e9:.2f} GB; {len(missing)} not found")
    for m in missing:
        print("  missing:", m)
    if a.dry_run:
        for s, d in found:
            print(f"  {d}  <-  {s}")
        return
    manifest = {}
    for s, d in found:
        dst = a.out / d
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, dst)
        if dst.name == "manifest.json":            # keep only the file name of each source image (no local paths)
            m = json.loads(dst.read_text(encoding="utf-8"))
            for v in m.values():
                if isinstance(v, dict) and "source" in v:
                    v["source"] = str(v["source"]).replace("\\", "/").split("/")[-1]
            dst.write_text(json.dumps(m, indent=1), encoding="utf-8")
        manifest[d] = {"sha256": sha256(dst), "bytes": dst.stat().st_size}
    (a.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("written:", a.out)


if __name__ == "__main__":
    main()
