#!/usr/bin/env python3
"""
Check consistency between ROI label JSONs and image folders
===========================================================

- For each split (train / val / test):
    * Compare the set of filenames in the JSON vs. the image files in the folder.
    * Report:
        - Number of labels in JSON
        - Number of image files in folder
        - How many filenames are common
        - How many labels do NOT have a matching image file
        - How many image files do NOT have a matching label

- Additionally:
    * Check for filename leakage between splits:
        - train ∩ val
        - train ∩ test
        - val ∩ test
      (Only reporting, NO modifications.)

This script does NOT modify any JSON or image file.
All comments are in English as requested.
"""

import json
from pathlib import Path
from typing import Dict, Set, Tuple, List


# ---------- Helper functions ----------

def load_json_labels(path: Path) -> Dict[str, dict]:
    """Load labels JSON file and return as a dict."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_image_filenames(folder: Path, exts: Tuple[str, ...]) -> Set[str]:
    """
    List image filenames (only names, no directories) in a folder
    that match the given extensions.
    """
    if not folder.is_dir():
        print(f"[WARN] Folder does not exist: {folder}")
        return set()

    files: Set[str] = set()
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.add(p.name)
    return files


def report_split(
    split_name: str,
    labels: Dict[str, dict],
    image_files: Set[str]
) -> None:
    """
    Print a detailed report for a single split.
    """
    label_names = set(labels.keys())

    common = label_names & image_files
    labels_without_image = label_names - image_files
    images_without_label = image_files - label_names

    print(f"\n===== SPLIT: {split_name} =====")
    print(f"  # labels in JSON           : {len(label_names)}")
    print(f"  # image files in folder    : {len(image_files)}")
    print(f"  # matching (label & image) : {len(common)}")
    print(f"  # labels WITHOUT image     : {len(labels_without_image)}")
    print(f"  # images WITHOUT label     : {len(images_without_label)}")

    if labels_without_image:
        print(f"  >> Example labels without image (up to 10):")
        for name in sorted(labels_without_image)[:10]:
            print(f"     - {name}")

    if images_without_label:
        print(f"  >> Example images without label (up to 10):")
        for name in sorted(images_without_label)[:10]:
            print(f"     - {name}")


def report_leakage(
    train_labels: Dict[str, dict],
    val_labels: Dict[str, dict],
    test_labels: Dict[str, dict]
) -> None:
    """
    Report any filenames that appear in more than one split.
    This is only a report (no modifications).
    """
    train_names = set(train_labels.keys())
    val_names = set(val_labels.keys())
    test_names = set(test_labels.keys())

    leak_train_val = train_names & val_names
    leak_train_test = train_names & test_names
    leak_val_test = val_names & test_names

    print("\n===== LEAKAGE CHECK BETWEEN SPLITS (report only) =====")
    print(f"  train ∩ val  : {len(leak_train_val)}")
    print(f"  train ∩ test : {len(leak_train_test)}")
    print(f"  val   ∩ test : {len(leak_val_test)}")

    if leak_train_val:
        print("  >> Example names in BOTH train and val (up to 10):")
        for name in sorted(leak_train_val)[:10]:
            print(f"     - {name}")

    if leak_train_test:
        print("  >> Example names in BOTH train and test (up to 10):")
        for name in sorted(leak_train_test)[:10]:
            print(f"     - {name}")

    if leak_val_test:
        print("  >> Example names in BOTH val and test (up to 10):")
        for name in sorted(leak_val_test)[:10]:
            print(f"     - {name}")


# ---------- Main ----------

def main():
    # ----- Adjust these paths if needed -----
    data_root = Path("dataset_palm_locator_seed42")
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    # JSON files to check (you can change them to *_round3.json or others)
    train_json_path = Path("dataset_palm_locator_seed42/train_labels.json")
    val_json_path = Path("dataset_palm_locator_seed42/val_labels.json")
    test_json_path = Path("dataset_palm_locator_seed42/test_labels.json")

    # Valid image extensions
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # ----- Load labels -----
    print("[INFO] Loading JSON label files...")
    train_labels = load_json_labels(train_json_path)
    val_labels = load_json_labels(val_json_path)
    test_labels = load_json_labels(test_json_path)

    # ----- List image files in each split folder -----
    print("[INFO] Listing image files in split folders...")
    train_images = list_image_filenames(train_dir, img_exts)
    val_images = list_image_filenames(val_dir, img_exts)
    test_images = list_image_filenames(test_dir, img_exts)

    # ----- Per-split reports -----
    report_split("train", train_labels, train_images)
    report_split("val", val_labels, val_images)
    report_split("test", test_labels, test_images)

    # ----- Cross-split leakage report -----
    report_leakage(train_labels, val_labels, test_labels)

    print("\n[INFO] Check finished. No files were modified.")


if __name__ == "__main__":
    main()
