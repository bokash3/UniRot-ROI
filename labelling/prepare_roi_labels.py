import cv2
import numpy as np

def compute_square_roi_from_mask(mask_u8: np.ndarray,
                                 margin_frac: float = 0.05):
    """
    Compute a maximal axis-aligned square ROI inside a hand mask.

    Args:
        mask_u8: uint8 mask (H×W), >0 = hand region.
        margin_frac: fractional margin to shrink the maximum radius
                     (e.g. 0.05 = keep 95% of the theoretical max radius)
                     to stay away from the boundary a bit.

    Returns:
        (cx_norm, cy_norm, size_norm) in [0,1], or None if failed.
    """

    # 1) Ensure binary mask (0/1)
    m = (mask_u8 > 0).astype(np.uint8)

    H, W = m.shape[:2]
    if m.sum() == 0:
        return None

    # 2) Keep only the largest connected component (largest hand blob)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    largest = max(cnts, key=cv2.contourArea)
    m_largest = np.zeros_like(m)
    cv2.drawContours(m_largest, [largest], -1, 1, thickness=-1)

    # 3) Distance transform on the largest component
    dist = cv2.distanceTransform(m_largest, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return None

    # Center of maximum inscribed circle
    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
    r_max = float(dist[cy, cx])

    # Apply margin to stay away from the boundary
    r_eff = r_max * (1.0 - margin_frac)

    # 4) Binary search for the largest square side s
    # Center is fixed at (cx, cy).
    # We search s in pixels, and require the whole square to be inside m_largest.
    low = 0.0
    high = 2.0 * r_eff  # side cannot be larger than diameter of the circle

    def square_fits(side: float) -> bool:
        """Check if a square of given side fits completely inside the mask."""
        half = int(np.floor(side / 2.0))
        if half <= 0:
            return False
        x0 = cx - half
        x1 = cx + half
        y0 = cy - half
        y1 = cy + half

        # Check image bounds
        if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
            return False

        roi = m_largest[y0:y1 + 1, x0:x1 + 1]
        # Square should be (2*half+1) × (2*half+1)
        expected_area = (2 * half + 1) * (2 * half + 1)
        if roi.size != expected_area:
            return False

        # All pixels inside ROI must be 1 (inside hand)
        return np.all(roi == 1)

    # Binary search with pixel precision
    for _ in range(20):  # 20 iterations are more than enough for pixel-level
        mid = 0.5 * (low + high)
        if square_fits(mid):
            low = mid
        else:
            high = mid

    # Final integer side in pixels
    side_px = int(np.floor(low))
    if side_px <= 0:
        return None

    # 5) Normalize parameters to [0,1]
    cx_norm = cx / float(W)
    cy_norm = cy / float(H)
    size_norm = side_px / float(min(H, W))

    return cx_norm, cy_norm, size_norm
import json
from pathlib import Path

def generate_roi_labels_for_folder(masks_dir: str,
                                   out_json: str,
                                   exts=(".png", ".jpg", ".jpeg", ".bmp")):
    """
    Loop over all mask images in masks_dir, compute ROI labels,
    and save them into a JSON file.

    The JSON format will be:
    {
        "filename.png": {"cx": float, "cy": float, "size": float},
        ...
    }
    """
    masks_dir = Path(masks_dir)
    paths = [p for p in masks_dir.rglob("*") if p.suffix.lower() in exts]

    labels = {}
    num_ok = 0
    num_fail = 0

    for p in paths:
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Failed to read mask: {p}")
            num_fail += 1
            continue

        result = compute_square_roi_from_mask(mask)
        if result is None:
            print(f"[WARN] Could not compute ROI for: {p}")
            num_fail += 1
            continue

        cx_norm, cy_norm, size_norm = result
        # You can choose key as filename only or relative path.
        key = p.name   # or: str(p.relative_to(masks_dir))

        labels[key] = {
            "cx": float(cx_norm),
            "cy": float(cy_norm),
            "size": float(size_norm),
        }
        num_ok += 1

    print(f"[INFO] Successfully generated labels for {num_ok} masks; failed for {num_fail}.")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"[INFO] Saved ROI labels to: {out_json}")


if __name__ == "__main__":
    # Example usage:
    masks_folder = "aligned_masks"       # <-- ضع مسار الماسكات هنا
    out_labels = "palm_roi_labels.json"
    generate_roi_labels_for_folder(masks_folder, out_labels)
