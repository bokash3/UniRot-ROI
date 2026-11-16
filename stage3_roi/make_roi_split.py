#!/usr/bin/env python3
"""
Make ROI Prototype Subset from CASIA-Multispectral Hand Masks
--------------------------------------------------------------
- Scans a root folder for hand mask images.
- Assumes filenames like: 001_l_940_01.png
    * subject_id = "001"
    * hand       = "l" or "r"
    * band       = "940" (or 460/630/850/940, etc.)
- Filters by:
    * target band (e.g., 940)
    * hands to include (e.g., both l + r, or only one hand)
- Groups images by (subject_id, hand).
- Selects up to N images per (subject, hand).
- Selects up to M subjects in total.
- Splits subjects into train / val / test (subject-wise split).
- Copies selected masks into:
    out_root/train/
    out_root/val/
    out_root/test/
- Saves:
    * roi_proto_split_info.json  (metadata + filenames per split)
    * train_labels.json          (labels for train filenames)
    * val_labels.json            (labels for val filenames)
    * test_labels.json           (labels for test filenames)

All comments are in English as requested.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


# -----------------------------
# Filename parsing
# -----------------------------
def parse_casia_filename(name: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse CASIA-style filename: subject_hand_band_index.ext
    Example: "001_l_940_01.png" -> ("001", "l", "940")

    Returns:
        (subject_id, hand, band) or None if parsing fails.
    """
    stem = Path(name).stem  # "001_l_940_01"
    parts = stem.split("_")
    if len(parts) < 4:
        # Not enough parts to parse, skip this file
        return None

    subject_id = parts[0]  # e.g. "001"
    hand = parts[1]        # e.g. "l" or "r"
    band = parts[2]        # e.g. "940"

    return subject_id, hand, band


# -----------------------------
# Main subset / split logic
# -----------------------------
def build_subset(
    masks_dir: Path,
    out_root: Path,
    target_band: Optional[str],
    hands_to_include: List[str],
    max_subjects: Optional[int],
    max_images_per_subject_hand: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    labels_json: Optional[Path] = None,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
):
    """
    Build ROI prototype subset and split into train/val/test.
    Optionally, also split labels from a global labels JSON file into
    separate JSON files for each split (train/val/test).
    """

    # 0) Load labels (if provided)
    labels_all: Dict[str, Any] = {}
    if labels_json is not None:
        if not labels_json.is_file():
            raise FileNotFoundError(f"Labels JSON not found: {labels_json}")
        print(f"[INFO] Loading labels from: {labels_json}")
        with labels_json.open("r", encoding="utf-8") as f:
            labels_all = json.load(f)
        print(f"[INFO] Loaded {len(labels_all)} labels from JSON.")

    # 1) Collect all candidate mask paths
    all_paths: List[Path] = [
        p
        for p in masks_dir.rglob("*")
        if p.suffix.lower() in exts
    ]

    if not all_paths:
        raise RuntimeError(f"No mask images found in: {masks_dir}")

    print(f"[INFO] Found {len(all_paths)} total mask files under {masks_dir}")

    # 2) Group by (subject_id, hand) and filter by band + hand
    grouped: Dict[Tuple[str, str], List[Path]] = {}
    for p in all_paths:
        parsed = parse_casia_filename(p.name)
        if parsed is None:
            # Skip files that do not follow the naming convention
            print(f"[WARN] Skipping file with unexpected name: {p.name}")
            continue

        subject_id, hand, band = parsed

        # Filter by band if requested
        if target_band is not None and band != target_band:
            continue

        # Filter by hand if requested
        if hands_to_include and hand not in hands_to_include:
            continue

        key = (subject_id, hand)
        grouped.setdefault(key, []).append(p)

    if not grouped:
        raise RuntimeError(
            f"No masks match the filters: band={target_band}, hands={hands_to_include}"
        )

    print(f"[INFO] After filtering, found {len(grouped)} (subject, hand) groups.")

    # 3) For each (subject, hand) group, limit to max_images_per_subject_hand
    rng = random.Random(seed)
    limited_grouped: Dict[Tuple[str, str], List[Path]] = {}

    for key, paths in grouped.items():
        paths_sorted = sorted(paths)
        if len(paths_sorted) > max_images_per_subject_hand:
            # Randomly select max_images_per_subject_hand images (seeded RNG)
            selected = rng.sample(paths_sorted, max_images_per_subject_hand)
        else:
            selected = paths_sorted
        limited_grouped[key] = selected

    # 4) Aggregate unique subjects from these groups
    subjects = sorted({k[0] for k in limited_grouped.keys()})  # set of subject_ids
    rng.shuffle(subjects)  # subject-wise shuffling controlled by seed

    if max_subjects is not None and max_subjects > 0:
        subjects = subjects[:max_subjects]

    print(f"[INFO] Using {len(subjects)} subjects after limiting to max_subjects={max_subjects}")

    # Keep only groups that belong to the selected subjects
    filtered_grouped: Dict[Tuple[str, str], List[Path]] = {
        k: v for k, v in limited_grouped.items() if k[0] in subjects
    }

    # 5) Split subjects into train / val / test (subject-wise)
    n_subj = len(subjects)
    n_train = int(round(train_ratio * n_subj))
    n_val = int(round(val_ratio * n_subj))
    # Ensure total = n_subj
    n_train = min(n_train, n_subj)
    n_val = min(n_val, n_subj - n_train)
    n_test = n_subj - n_train - n_val

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]

    print(f"[INFO] Subject split:")
    print(f"       train: {len(train_subjects)}")
    print(f"       val  : {len(val_subjects)}")
    print(f"       test : {len(test_subjects)}")

    subject_to_split: Dict[str, str] = {}
    for sid in train_subjects:
        subject_to_split[sid] = "train"
    for sid in val_subjects:
        subject_to_split[sid] = "val"
    for sid in test_subjects:
        subject_to_split[sid] = "test"

    # 6) Prepare output directories
    train_dir = out_root / "train"
    val_dir = out_root / "val"
    test_dir = out_root / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 7) Copy selected files into their split folders
    split_files: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    total_selected = 0
    for (subject_id, hand), paths in filtered_grouped.items():
        split = subject_to_split.get(subject_id, None)
        if split is None:
            # Should not happen if we did things correctly, but just in case
            continue

        if split == "train":
            dst_dir = train_dir
        elif split == "val":
            dst_dir = val_dir
        else:
            dst_dir = test_dir

        for src_path in paths:
            dst_path = dst_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            split_files[split].append(src_path.name)
            total_selected += 1

    print(f"[INFO] Total selected images copied: {total_selected}")
    print(f"       train: {len(split_files['train'])}")
    print(f"       val  : {len(split_files['val'])}")
    print(f"       test : {len(split_files['test'])}")

    # 8) Save split info as JSON (metadata + filenames)
    out_json = out_root / "roi_proto_split_info.json"
    meta = {
        "masks_dir": str(masks_dir),
        "out_root": str(out_root),
        "target_band": target_band,
        "hands_to_include": hands_to_include,
        "max_subjects": max_subjects,
        "max_images_per_subject_hand": max_images_per_subject_hand,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "labels_json_source": str(labels_json) if labels_json is not None else None,
    }
    payload = {
        "meta": meta,
        "splits": split_files,
        "subjects": {
            "train": train_subjects,
            "val": val_subjects,
            "test": test_subjects,
        },
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Saved split info JSON to: {out_json}")

    # 9) If labels were provided, create per-split label JSONs
    if labels_all:
        print("[INFO] Building per-split label JSONs...")
        split_label_dicts: Dict[str, Dict[str, Any]] = {
            "train": {},
            "val": {},
            "test": {},
        }

        missing_labels_count = 0

        for split_name in ["train", "val", "test"]:
            for fname in split_files[split_name]:
                if fname in labels_all:
                    split_label_dicts[split_name][fname] = labels_all[fname]
                else:
                    missing_labels_count += 1

        if missing_labels_count > 0:
            print(
                f"[WARN] {missing_labels_count} images in splits do not have labels "
                f"in the global labels JSON."
            )

        # Save per-split label JSON files
        train_labels_path = out_root / "train_labels.json"
        val_labels_path = out_root / "val_labels.json"
        test_labels_path = out_root / "test_labels.json"

        with train_labels_path.open("w", encoding="utf-8") as f:
            json.dump(split_label_dicts["train"], f, indent=2)
        with val_labels_path.open("w", encoding="utf-8") as f:
            json.dump(split_label_dicts["val"], f, indent=2)
        with test_labels_path.open("w", encoding="utf-8") as f:
            json.dump(split_label_dicts["test"], f, indent=2)

        print(f"[INFO] Saved train labels to: {train_labels_path}")
        print(f"[INFO] Saved val labels to  : {val_labels_path}")
        print(f"[INFO] Saved test labels to : {test_labels_path}")
    else:
        print("[INFO] No labels JSON provided; skipping per-split label files.")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Make ROI Prototype Subset from CASIA-Multispectral Masks")

    p.add_argument(
        "--masks_dir",
        type=str,
        default="aligned_masks",
        help="Root folder containing all CASIA hand masks.",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="dataset_palm_locator_seed42",
        help="Output root folder to store prototype subset (train/val/test).",
    )

    p.add_argument(
        "--band",
        type=str,
        default="None",
        help="Target band to filter (e.g., '940' or '850'). If 'None', use all bands.",
    )
    p.add_argument(
        "--hands",
        type=str,
        default="both",
        choices=["both", "left", "right"],
        help="Which hands to include: both / left / right.",
    )

    p.add_argument(
        "--max_subjects",
        type=int,
        default=-1,
        help="Maximum number of subjects to include. Use 0 or negative for no limit.",
    )
    p.add_argument(
        "--max_images_per_subject_hand",
        type=int,
        default=36,
        help="Maximum number of images per (subject, hand) group.",
    )

    p.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train subject ratio (e.g., 0.7).",
    )
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation subject ratio (e.g., 0.15).",
    )
    p.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test subject ratio (e.g., 0.15). Must satisfy train+val+test≈1.0.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subject selection.",
    )

    p.add_argument(
        "--labels_json",
        type=str,
        default="all_hand_labels.json",
        help="Path to JSON file containing labels for ALL masks. "
             "Keys must match filenames (e.g., '001_l_940_01.png').",
    )

    return p.parse_args()


def main():
    args = parse_args()

    masks_dir = Path(args.masks_dir)
    out_root = Path(args.out_root)

    if args.hands == "both":
        hands_to_include = ["l", "r"]
    elif args.hands == "left":
        hands_to_include = ["l"]
    else:
        hands_to_include = ["r"]

    # If max_subjects <= 0, treat as "no limit"
    max_subjects = args.max_subjects if args.max_subjects > 0 else None

    # Convert band argument: "None" (string) -> None (Python)
    target_band = None if args.band.lower() == "none" else args.band

    # Check ratios sum (~=1.0)
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-3:
        print(
            f"[WARN] train_ratio + val_ratio + test_ratio = {total_ratio:.3f} "
            f"(not exactly 1.0). It will still work, but double-check your values."
        )

    labels_json_path = Path(args.labels_json) if args.labels_json is not None else None

    build_subset(
        masks_dir=masks_dir,
        out_root=out_root,
        target_band=target_band,
        hands_to_include=hands_to_include,
        max_subjects=max_subjects,
        max_images_per_subject_hand=args.max_images_per_subject_hand,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        labels_json=labels_json_path,
    )


if __name__ == "__main__":
    main()
