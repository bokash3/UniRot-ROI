#!/usr/bin/env python3
"""
Merge manually-corrected hard ROI labels into train/val/test JSONs
+ Ensure image/label consistency and remove split leakage
------------------------------------------------------------------
- Reads:
    * train_round2/train, val, test  (mask folders)
    * palm_roi_labels_trainv2.json
    * palm_roi_labels_val2.json
    * lable31_40.json               (test labels)
    * roi_labels_hard_manual.json   (688 hard samples)

- Step 1: Merge hard labels
    * For each hard sample:
        - Detect which split (train/val/test) contains this mask.
        - Overwrite the label in that split JSON.
        - If the file does not exist in any split, add it to TRAIN labels.

- Step 2: Ensure images exist for all JSON entries
    * For each split (train/val/test):
        - For every filename in the labels JSON:
            - If the image file is missing from the split folder:
                -> Search for it under aligned_mask/ (recursive).
                -> If found, copy it into the split folder.
                -> If NOT found, remove that entry from the labels JSON.

- Step 3: Check for data leakage between splits
    * If a filename appears in more than one split:
        - train & val  -> keep in TRAIN, remove from VAL.
        - train & test -> keep in TRAIN, remove from TEST.
        - val & test   -> keep in VAL,   remove from TEST.

- Step 4: Save merged JSONs as new files:
    * palm_roi_labels_train_round3.json
    * palm_roi_labels_val_round3.json
    * palm_roi_labels_test_round3.json

All comments are in English as requested.
"""

import json
import shutil
from pathlib import Path


def load_json(path: Path):
    """Load JSON file and return dict."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    """Save dict as JSON with pretty indentation."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ensure_split_images(
    split_name: str,
    labels_dict: dict,
    split_dir: Path,
    aligned_root: Path
):
    """
    Ensure that every label in labels_dict has a corresponding image in split_dir.

    - If the image is missing:
        * Search for it under aligned_root using rglob().
        * If found, copy it to split_dir.
        * If not found, remove the label entry.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    missing_without_source = 0
    copied_from_aligned = 0

    # We convert keys to list to allow safe deletion from the dict while iterating.
    for fname in list(labels_dict.keys()):
        img_path = split_dir / fname
        if img_path.is_file():
            # Image already exists: nothing to do.
            continue

        # Try to find the image under aligned_root.
        source_path = None
        for p in aligned_root.rglob(fname):
            source_path = p
            break

        if source_path is not None and source_path.is_file():
            shutil.copy2(source_path, img_path)
            copied_from_aligned += 1
            print(f"[FIX] ({split_name}) Copied missing image '{fname}' from '{source_path}'")
        else:
            # No source image found -> remove label.
            print(f"[WARN] ({split_name}) No source image found for '{fname}' -> removing label entry")
            labels_dict.pop(fname, None)
            missing_without_source += 1

    print(
        f"[INFO] Split '{split_name}': "
        f"copied {copied_from_aligned} missing images from aligned_mask, "
        f"removed {missing_without_source} labels with no source image."
    )


def remove_leak_for_file(
    fname: str,
    split_name: str,
    split_dir: Path,
    labels_dict: dict
):
    """
    Remove a leaked file from a given split:
    - Delete label entry if present.
    - Delete image file if present.
    """
    if fname in labels_dict:
        labels_dict.pop(fname, None)

    img_path = split_dir / fname
    if img_path.is_file():
        img_path.unlink()
        print(f"[LEAK-FIX] Removed leaked '{fname}' from {split_name} (file + label)")
    else:
        print(f"[LEAK-FIX] Removed leaked '{fname}' from {split_name} (label only, no file)")


def fix_leakage_between_splits(
    train_labels: dict,
    val_labels: dict,
    test_labels: dict,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path
):
    """
    Detect and remove data leakage between splits.

    Policy:
    - train & val  -> keep in TRAIN, remove from VAL.
    - train & test -> keep in TRAIN, remove from TEST.
    - val & test   -> keep in VAL,   remove from TEST.
    """
    train_names = set(train_labels.keys())
    val_names = set(val_labels.keys())
    test_names = set(test_labels.keys())

    leak_train_val = train_names & val_names
    leak_train_test = train_names & test_names
    leak_val_test = val_names & test_names

    print("\n[INFO] Leakage check:")
    print(f"  train ∩ val : {len(leak_train_val)}")
    print(f"  train ∩ test: {len(leak_train_test)}")
    print(f"  val   ∩ test: {len(leak_val_test)}")

    # train & val  -> remove from val
    for fname in sorted(leak_train_val):
        remove_leak_for_file(fname, "val", val_dir, val_labels)

    # train & test -> remove from test
    for fname in sorted(leak_train_test):
        remove_leak_for_file(fname, "test", test_dir, test_labels)

    # val & test   -> remove from test
    # (note: if this file was also in train, it was already handled above)
    for fname in sorted(leak_val_test):
        # If still in test, remove it
        if fname in test_labels:
            remove_leak_for_file(fname, "test", test_dir, test_labels)

    # Final counts after leakage fix
    print("\n[INFO] Counts after leakage fix:")
    print(f"  train: {len(train_labels)}")
    print(f"  val  : {len(val_labels)}")
    print(f"  test : {len(test_labels)}")


def main():
    # --- base paths: adjust if needed ---
    data_root = Path("train_round23")
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    # Folder that contains all aligned masks (source for missing images)
    aligned_root = Path("aligned_masks")

    # --- JSON paths (original labels + hard labels) ---
    train_json_path = Path("palm_roi_labels_trainv2.json")
    val_json_path = Path("lable31_40.json")
    test_json_path = Path("palm_roi_labels_val2.json")  # your test labels file
    hard_json_path = Path("roi_labels_hard_manual.json")

    # --- load JSONs ---
    train_labels = load_json(train_json_path)
    val_labels = load_json(val_json_path)
    test_labels = load_json(test_json_path)
    hard_labels = load_json(hard_json_path)

    print(f"[INFO] Original counts:")
    print(f"  train: {len(train_labels)}")
    print(f"  val  : {len(val_labels)}")
    print(f"  test : {len(test_labels)}")
    print(f"[INFO] Hard labels: {len(hard_labels)}")

    updated_train = 0
    updated_val = 0
    updated_test = 0
    added_train = 0

    # --- Step 1: merge hard labels into the correct split ---
    for key, roi in hard_labels.items():
        # Key may contain subfolders (e.g. 'subdir/004_l_WHT_05.png')
        # so we keep only the file name.
        fname = Path(key).name

        # Candidate paths in each split
        p_train = train_dir / fname
        p_val = val_dir / fname
        p_test = test_dir / fname

        if p_train.is_file():
            train_labels[fname] = {
                "cx": float(roi["cx"]),
                "cy": float(roi["cy"]),
                "size": float(roi["size"]),
            }
            updated_train += 1

        elif p_val.is_file():
            val_labels[fname] = {
                "cx": float(roi["cx"]),
                "cy": float(roi["cy"]),
                "size": float(roi["size"]),
            }
            updated_val += 1

        elif p_test.is_file():
            test_labels[fname] = {
                "cx": float(roi["cx"]),
                "cy": float(roi["cy"]),
                "size": float(roi["size"]),
            }
            updated_test += 1

        else:
            # File not found in any split -> add to TRAIN labels only.
            # The actual image will be created later from aligned_mask if possible.
            print(
                f"[WARN] {fname} not found in train/val/test; "
                f"adding it to TRAIN labels only."
            )
            train_labels[fname] = {
                "cx": float(roi["cx"]),
                "cy": float(roi["cy"]),
                "size": float(roi["size"]),
            }
            added_train += 1

    print("\n[INFO] Merge hard-labels summary:")
    print(f"  updated train: {updated_train}")
    print(f"  updated val  : {updated_val}")
    print(f"  updated test : {updated_test}")
    print(f"  added->train : {added_train}")

    print("\n[INFO] Counts after merging hard labels (before image check):")
    print(f"  train: {len(train_labels)}")
    print(f"  val  : {len(val_labels)}")
    print(f"  test : {len(test_labels)}")

    # --- Step 2: ensure every label has a corresponding image file ---
    print("\n[INFO] Ensuring images for all JSON entries using 'aligned_mask' as source...")
    ensure_split_images("train", train_labels, train_dir, aligned_root)
    ensure_split_images("val", val_labels, val_dir, aligned_root)
    ensure_split_images("test", test_labels, test_dir, aligned_root)

    # --- Step 3: check and fix data leakage between splits ---
    fix_leakage_between_splits(
        train_labels, val_labels, test_labels, train_dir, val_dir, test_dir
    )

    # --- Step 4: save merged JSONs with new names (do NOT overwrite old ones) ---
    merged_train_path = Path("palm_roi_labels_train_round3.json")
    merged_val_path = Path("palm_roi_labels_val_round3.json")
    merged_test_path = Path("palm_roi_labels_test_round3.json")

    save_json(merged_train_path, train_labels)
    save_json(merged_val_path, val_labels)
    save_json(merged_test_path, test_labels)

    print("\n[INFO] Saved merged label files:")
    print(f"  {merged_train_path}")
    print(f"  {merged_val_path}")
    print(f"  {merged_test_path}")


if __name__ == "__main__":
    main()
