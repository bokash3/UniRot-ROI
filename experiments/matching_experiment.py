"""
matching_experiment.py
======================

Phase 4 of the JKSU-CIS resubmission for "Mask-Guided Canonical Alignment for
Robust Contactless Palm ROI Localization".

Runs CompCode (Kong & Zhang, 2003 -- reference parameters with translation
overlay) and a per-seed fine-tuned ResNet-18 over the 12 ROI manifests
(4 ROI sources x 3 seeds), and writes one CSV row per (manifest, matcher).

Protocol (CASIA-MS, 850 nm only)
--------------------------------
For each manifest:

  1. Filter entries to a single band (default 850 nm). Other bands are dropped.
  2. Group remaining entries by identity = subject_id + hand_side.
  3. Gallery = samples 01, 02, 03 (session 1, 3 templates per identity).
     Probes  = samples 04, 05, 06 (session 2, 3 probes  per identity).
  4. For each probe, distance to a gallery identity = min over its 3 templates.
  5. Rank-1 = top-1 retrieval on probes.
     EER    = swept on (genuine, impostor) probe-vs-gallery scores.

This is the standard cross-session, single-band, multi-template protocol used
throughout the CASIA-MS palm-recognition literature.

CompCode
--------
Real Gabor filter bank, 6 orientations, Kong & Zhang reference parameters
sized for a 128x128 ROI. Codes are encoded as a (H, W) orientation index
map. Distance between codes is normalized Hamming distance with a small
horizontal/vertical translation overlay (+/- 2 px, picking the minimum) to
absorb sub-pixel ROI misalignment. The overlay is standard practice and is
worth a couple of EER points.

ResNet-18 (fine-tuned)
----------------------
For each seed, loads the matcher trained by `train_resnet18_matcher.py` on
that seed's train+val identities at 850 nm. Penultimate avgpool features
(512-d), L2-normalized, cosine distance.

Baseline-A constant pose rotation
---------------------------------
GT, Lite-PalmROI and Fast-PalmROI manifests reference aligned-frame images
(fingers pointing up, MAE ~3 deg). Baseline-A predicts ROIs in raw-frame
coordinates (fingers pointing left/right with per-pose variation). To put
all four methods on roughly comparable poses without giving Baseline-A
access to the per-image alignment angle (which would not be a fair
ablation), we apply a CONSTANT rotation to Baseline-A crops only. The
default is 90 deg (controlled by --baseline_a_rotation). The remaining
+/- per-pose variation in Baseline-A crops vs the ~3 deg precision of the
mask-guided methods directly measures the value of the alignment stage.
Set --baseline_a_rotation 0 to disable and evaluate Baseline-A as-deployed.

Usage
-----
    python matching_experiment.py \\
        --manifests_dir manifests \\
        --out_csv       results/phase4_matching.csv \\
        --project_root  . \\
        --band          850 \\
        --resnet_ckpt_pattern checkpoints/resnet18_seed{seed}.pt \\
        --device        cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Filename parsing -- CASIA-MS layout
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<hand>[lr])_(?P<band>\d+|white|WHT)_(?P<sample>\d+)\.[A-Za-z]+$"
)


def parse_casia_filename(fname: str) -> dict[str, Any] | None:
    m = _FNAME_RE.match(fname)
    if m is None:
        return None
    return {
        "subject": m.group("subject"),
        "hand": m.group("hand"),
        "band": m.group("band"),
        "sample": int(m.group("sample")),
        "identity": f"{m.group('subject')}_{m.group('hand')}",
    }


# ---------------------------------------------------------------------------
# Manifest loading + ROI cropping
# ---------------------------------------------------------------------------

@dataclass
class ManifestEntry:
    fname: str
    cx: float
    cy: float
    size: float
    identity: str
    sample: int   # 1..6
    band: str


@dataclass
class Manifest:
    path: Path
    image_root: Path
    roi_type: str
    entries: list[ManifestEntry]
    name: str


def load_manifest(path: Path, project_root: Path | None, band_filter: str | None) -> Manifest:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "image_root" not in data or "entries" not in data:
        raise ValueError(f"Manifest {path} missing 'image_root' or 'entries'.")

    image_root = Path(data["image_root"])
    if not image_root.is_absolute():
        candidates = []
        if project_root is not None:
            candidates.append((project_root / image_root).resolve())
        candidates.append((Path.cwd() / image_root).resolve())
        candidates.append((path.parent / image_root).resolve())
        chosen = next((c for c in candidates if c.exists()), None)
        if chosen is None:
            raise FileNotFoundError(
                f"image_root for {path} does not exist; tried:\n  "
                + "\n  ".join(str(c) for c in candidates)
            )
        image_root = chosen
    elif not image_root.exists():
        raise FileNotFoundError(f"image_root for {path} does not exist: {image_root}")

    roi_type = data.get("roi_type", "original_normalized")
    if roi_type != "original_normalized":
        raise NotImplementedError(
            f"Unsupported roi_type {roi_type!r} in {path}; only 'original_normalized' is implemented."
        )

    entries: list[ManifestEntry] = []
    n_total = 0
    n_unparsed = 0
    for fname, roi in data["entries"].items():
        n_total += 1
        meta = parse_casia_filename(fname)
        if meta is None:
            n_unparsed += 1
            continue
        if band_filter is not None and meta["band"] != band_filter:
            continue
        entries.append(ManifestEntry(
            fname=fname,
            cx=float(roi["cx"]),
            cy=float(roi["cy"]),
            size=float(roi["size"]),
            identity=meta["identity"],
            sample=meta["sample"],
            band=meta["band"],
        ))

    if n_unparsed:
        print(f"  [WARN] {path.name}: {n_unparsed} filenames did not match CASIA pattern.")

    return Manifest(
        path=path, image_root=image_root, roi_type=roi_type,
        entries=entries, name=path.stem,
    )


def crop_roi(
    img: Image.Image,
    entry: ManifestEntry,
    out_size: int = 128,
    rotate_deg: float = 0.0,
    pad_factor: float = 1.4,
) -> np.ndarray:
    """Crop a normalized square ROI and resize to (out_size, out_size).

    Pads with edge-replicate when the box runs off the image so the matcher
    never sees zero pixels.

    If rotate_deg != 0, the crop is taken with `pad_factor` * side enlargement
    around the same centre, then rotated, then centre-cropped back to a
    `side`-sized square before being resized to (out_size, out_size). This
    is used for Baseline-A entries to apply a constant pose rotation that
    approximates what the mask-guided alignment stage produces, without
    actually using any per-image alignment angle.
    """
    W, H = img.size
    cx_px = entry.cx * W
    cy_px = entry.cy * H
    # Project convention: `size` is normalized by min(H, W), NOT by W.
    # Every other script in the project (manifest builder, label converter,
    # baseline-A trainer) uses min(H, W). Using W here previously inflated
    # the crop side by a factor of W/min(H,W) ~= 1.33 for 768x576 CASIA-MS,
    # which leaked fingers/background into the crop.
    side_px = entry.size * float(min(H, W))

    if rotate_deg == 0.0:
        # Original tight crop path.
        half = side_px / 2.0
        x1, y1, x2, y2 = cx_px - half, cy_px - half, cx_px + half, cy_px + half
    else:
        # Pad-then-rotate-then-centre-crop. Take a larger square so corners
        # of the (rotated) target square are still inside the source.
        # For a 90 deg rotation pad_factor of sqrt(2) ~= 1.41 is enough; we
        # use 1.4 by default which is fine because we only ever rotate by
        # multiples of 90 deg (corners stay within a 1.4x box).
        large_side = side_px * pad_factor
        half = large_side / 2.0
        x1, y1, x2, y2 = cx_px - half, cy_px - half, cx_px + half, cy_px + half

    pad_l = int(max(0, -x1) + 0.999)
    pad_t = int(max(0, -y1) + 0.999)
    pad_r = int(max(0, x2 - W) + 0.999)
    pad_b = int(max(0, y2 - H) + 0.999)

    if pad_l or pad_t or pad_r or pad_b:
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
        else:
            arr = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
        img = Image.fromarray(arr)
        x1 += pad_l; x2 += pad_l; y1 += pad_t; y2 += pad_t

    crop = img.crop((round(x1), round(y1), round(x2), round(y2)))

    if rotate_deg != 0.0:
        # Rotate by rotate_deg around the centre. Positive = counter-clockwise
        # in PIL convention. Pass expand=False since we sized the crop already
        # for the corners to fit.
        crop = crop.rotate(rotate_deg, resample=Image.BILINEAR, expand=False)
        # Centre-crop back to the original side_px (pre-pad).
        cx_c, cy_c = crop.size[0] / 2.0, crop.size[1] / 2.0
        half = side_px / 2.0
        crop = crop.crop((
            round(cx_c - half), round(cy_c - half),
            round(cx_c + half), round(cy_c + half),
        ))

    crop = crop.resize((out_size, out_size), Image.BILINEAR)
    return np.array(crop)


def to_gray(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 3:
        return (
            0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
        ).astype(np.uint8)
    return crop


# ---------------------------------------------------------------------------
# CompCode -- Kong & Zhang (2003) reference parameters for 128x128
# ---------------------------------------------------------------------------

def _compcode_kernel(ksize: int, theta: float) -> np.ndarray:
    """Real Gabor kernel using Kong & Zhang (2003) parameterisation:
        omega = 0.0916          (radial frequency in rad/pixel)
        delta = 3               (half-bandwidth, octaves)
        sigma derived from omega and delta via the standard half-bandwidth
        formula. With omega=0.0916 the wavelength is ~68.6 px, appropriate
        for principal-line spacing on a 128x128 palm ROI."""
    omega = 0.0916
    delta = 3.0
    sigma = (1.0 / omega) * np.sqrt(np.log(2) / 2.0) * \
            ((2.0 ** delta + 1.0) / (2.0 ** delta - 1.0))

    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    x_t = x * np.cos(theta) + y * np.sin(theta)
    y_t = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-0.5 * (x_t ** 2 + y_t ** 2) / (sigma ** 2))
    carrier = np.cos(2.0 * np.pi * omega * x_t)
    k = envelope * carrier
    k -= k.mean()
    return k


class CompCodeMatcher:
    """Competitive code with translation-overlay Hamming distance."""

    def __init__(self, n_orient: int = 6, ksize: int = 35, shift: int = 2) -> None:
        self.n_orient = n_orient
        self.kernels = [
            _compcode_kernel(ksize, np.pi * i / n_orient)
            for i in range(n_orient)
        ]
        self.shift = shift

    def encode(self, gray: np.ndarray) -> np.ndarray:
        """Encode 128x128 grayscale into a uint8 orientation-index map."""
        from scipy.signal import fftconvolve

        gray = to_gray(gray)
        g = gray.astype(np.float64)
        g = (g - g.mean()) / (g.std() + 1e-6)

        responses = np.stack(
            [fftconvolve(g, k, mode="same") for k in self.kernels], axis=0,
        )
        return np.argmin(responses, axis=0).astype(np.uint8)

    def distance(self, code_a: np.ndarray, code_b: np.ndarray) -> float:
        """Normalized Hamming distance with +/- shift translation overlay.
        Picks the minimum over (2*shift+1)^2 horizontal/vertical shifts of
        code_b relative to code_a, comparing only the overlapping region
        at each shift."""
        H, W = code_a.shape
        s = self.shift
        best = 1.0
        for dy in range(-s, s + 1):
            for dx in range(-s, s + 1):
                y1a, y2a = max(0, -dy), H - max(0, dy)
                x1a, x2a = max(0, -dx), W - max(0, dx)
                y1b, y2b = max(0, dy), H - max(0, -dy)
                x1b, x2b = max(0, dx), W - max(0, -dx)
                a = code_a[y1a:y2a, x1a:x2a]
                b = code_b[y1b:y2b, x1b:x2b]
                d = float((a != b).mean())
                if d < best:
                    best = d
        return best


# ---------------------------------------------------------------------------
# ResNet-18 (per-seed fine-tuned)
# ---------------------------------------------------------------------------

class ResNet18Matcher:
    """Loads a per-seed fine-tuned ResNet-18 backbone on demand and caches it."""

    def __init__(self, ckpt_pattern: str, device: str = "cuda") -> None:
        import torch
        from torchvision import models, transforms
        self.torch = torch
        self.models = models
        self.device = torch.device(
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        if str(self.device) != device:
            print(f"  [WARN] requested device={device}, using {self.device}")
        self.ckpt_pattern = ckpt_pattern
        self._cache: dict[str, Any] = {}

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load(self, seed: str) -> Any:
        torch = self.torch
        if seed in self._cache:
            return self._cache[seed]

        ckpt_path = Path(self.ckpt_pattern.format(seed=seed))
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"ResNet checkpoint not found for seed={seed}: {ckpt_path}. "
                f"Did you run train_resnet18_matcher.py for that seed?"
            )

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        num_classes = int(ckpt["num_classes"])

        net = self.models.resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
        net.load_state_dict(ckpt["state_dict"])

        # Strip the classifier; keep up to and including avgpool.
        backbone = torch.nn.Sequential(*list(net.children())[:-1]).to(self.device).eval()
        self._cache[seed] = backbone
        return backbone

    def encode_batch(self, crops: list[np.ndarray], seed: str) -> np.ndarray:
        torch = self.torch
        backbone = self._load(seed)

        tensors = []
        for crop in crops:
            crop = to_gray(crop) if crop.ndim == 3 else crop
            pil = Image.fromarray(crop, mode="L").convert("RGB")
            tensors.append(self.transform(pil))
        x = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            f = backbone(x).flatten(1)
            f = torch.nn.functional.normalize(f, dim=1)
        return f.cpu().numpy()


# ---------------------------------------------------------------------------
# Protocol: gallery (samples 1-3) / probes (samples 4-6)
# ---------------------------------------------------------------------------

@dataclass
class Split:
    gallery: dict[str, list[ManifestEntry]] = field(default_factory=dict)
    probes: list[ManifestEntry] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def make_split(manifest: Manifest) -> Split:
    by_id: dict[str, list[ManifestEntry]] = defaultdict(list)
    for e in manifest.entries:
        by_id[e.identity].append(e)

    split = Split()
    for ident, items in by_id.items():
        items_sorted = sorted(items, key=lambda x: x.sample)
        gallery = [it for it in items_sorted if it.sample in (1, 2, 3)]
        probes = [it for it in items_sorted if it.sample in (4, 5, 6)]
        if not gallery or not probes:
            split.skipped.append(
                f"{ident} (gallery={len(gallery)}, probes={len(probes)})"
            )
            continue
        split.gallery[ident] = gallery
        split.probes.extend(probes)
    return split


def compute_eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """EER on a fine threshold sweep. Lower distance = better match."""
    g = np.sort(genuine)
    i = np.sort(impostor)
    n_g = max(len(g), 1)
    n_i = max(len(i), 1)
    lo = float(min(g.min(), i.min()))
    hi = float(max(g.max(), i.max()))
    thr = np.linspace(lo, hi, 4000)
    far = np.searchsorted(i, thr, side="right") / n_i
    frr = (n_g - np.searchsorted(g, thr, side="left")) / n_g
    idx = int(np.argmin(np.abs(far - frr)))
    eer = 0.5 * float(far[idx] + frr[idx])
    return 100.0 * eer, float(thr[idx])


# ---------------------------------------------------------------------------
# Per-manifest evaluation
# ---------------------------------------------------------------------------

def all_entries(split: Split) -> list[ManifestEntry]:
    out: list[ManifestEntry] = []
    for tpls in split.gallery.values():
        out.extend(tpls)
    out.extend(split.probes)
    return out


def evaluate_compcode(split: Split, matcher: CompCodeMatcher,
                      crop_cache: dict[str, np.ndarray]) -> dict[str, float]:
    print(f"  encoding {len(crop_cache)} crops with CompCode...")
    codes: dict[str, np.ndarray] = {
        fname: matcher.encode(crop) for fname, crop in crop_cache.items()
    }
    return _score(split, codes, matcher.distance, "compcode")


def evaluate_resnet(split: Split, matcher: ResNet18Matcher,
                    crop_cache: dict[str, np.ndarray], seed: str) -> dict[str, float]:
    fnames = list(crop_cache.keys())
    crops = [crop_cache[f] for f in fnames]
    print(f"  encoding {len(fnames)} crops with ResNet-18 (seed={seed})...")
    feats: dict[str, np.ndarray] = {}
    bs = 64
    for i in range(0, len(crops), bs):
        f_batch = matcher.encode_batch(crops[i:i + bs], seed=seed)
        for j, fname in enumerate(fnames[i:i + bs]):
            feats[fname] = f_batch[j]
    return _score(split, feats, None, "resnet18")


def _score(split: Split, codes: dict[str, Any], dist_fn, matcher_name: str) -> dict[str, float]:
    print(f"  scoring {len(split.probes)} probes vs {len(split.gallery)} gallery identities "
          f"(min over ~3 templates per identity)...")
    gallery_ids = sorted(split.gallery.keys())
    gallery_codes_per_id = {
        ident: [codes[t.fname] for t in split.gallery[ident]]
        for ident in gallery_ids
    }

    genuine: list[float] = []
    impostor: list[float] = []
    rank1_correct = 0

    if matcher_name == "resnet18":
        # Vectorised path. Stack all gallery feats with row->identity mapping.
        all_g = []
        ident_for_row = []
        for ident in gallery_ids:
            for f in gallery_codes_per_id[ident]:
                all_g.append(f); ident_for_row.append(ident)
        all_g_mat = np.stack(all_g, axis=0)
        for p in split.probes:
            sims = all_g_mat @ codes[p.fname]
            dists = 1.0 - sims
            per_ident_min: dict[str, float] = {}
            for d, ident in zip(dists, ident_for_row):
                if ident not in per_ident_min or d < per_ident_min[ident]:
                    per_ident_min[ident] = float(d)
            best_ident = min(per_ident_min, key=per_ident_min.get)
            if best_ident == p.identity:
                rank1_correct += 1
            genuine.append(per_ident_min[p.identity])
            for ident, d in per_ident_min.items():
                if ident != p.identity:
                    impostor.append(d)
    else:
        # CompCode -- pure-Python loop is fine for ~90 probes x 30 ids x 3 templates.
        for p in split.probes:
            p_code = codes[p.fname]
            per_ident_min: dict[str, float] = {}
            for ident in gallery_ids:
                d_min = min(dist_fn(p_code, g) for g in gallery_codes_per_id[ident])
                per_ident_min[ident] = d_min
            best_ident = min(per_ident_min, key=per_ident_min.get)
            if best_ident == p.identity:
                rank1_correct += 1
            genuine.append(per_ident_min[p.identity])
            for ident, d in per_ident_min.items():
                if ident != p.identity:
                    impostor.append(d)

    g_arr = np.asarray(genuine)
    i_arr = np.asarray(impostor)
    eer_pct, eer_thr = compute_eer(g_arr, i_arr)
    rank1_pct = 100.0 * rank1_correct / max(len(split.probes), 1)

    return {
        "eer_pct": eer_pct,
        "eer_thr": eer_thr,
        "rank1_pct": rank1_pct,
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "n_probes": len(split.probes),
        "n_gallery_identities": len(split.gallery),
        "n_gallery_templates": sum(len(v) for v in split.gallery.values()),
        "genuine_mean": float(g_arr.mean()),
        "impostor_mean": float(i_arr.mean()),
    }


# ---------------------------------------------------------------------------
# Manifest name parsing (for CSV columns)
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(
    r"^(?P<source>gt|litepalmroi|fastpalmroi|baseline_a)[_-]?seed[_-]?(?P<seed>\d+)$",
    re.IGNORECASE,
)


def parse_manifest_name(stem: str) -> tuple[str, str]:
    m = _NAME_RE.match(stem)
    if not m:
        return stem, ""
    src_map = {
        "gt": "GT",
        "litepalmroi": "Lite-PalmROI",
        "fastpalmroi": "Fast-PalmROI",
        "baseline_a": "Baseline-A",
    }
    return src_map.get(m.group("source").lower(), m.group("source")), m.group("seed")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifests", nargs="+", help="Explicit list of manifest JSON paths.")
    g.add_argument("--manifests_dir", help="Directory containing manifest JSON files.")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--project_root", default=".",
                    help="Anchor for relative image_root paths in manifests.")
    ap.add_argument("--band", default="850",
                    help="Single CASIA-MS band to evaluate on (default 850 nm).")
    ap.add_argument(
        "--matchers", nargs="+", default=["compcode", "resnet18"],
        choices=["compcode", "resnet18"],
    )
    ap.add_argument(
        "--resnet_ckpt_pattern", default="checkpoints/resnet18_seed{seed}.pt",
        help="Path template for the per-seed ResNet checkpoints; '{seed}' is substituted.",
    )
    ap.add_argument(
        "--baseline_a_rotation", type=float, default=90.0,
        help="Constant rotation (in degrees) applied to Baseline-A crops only, "
             "before encoding. This approximates pose alignment without giving "
             "Baseline-A access to the per-image alignment angle. Set to 0 to "
             "disable rotation entirely (Option 2 / pure deployment evaluation). "
             "Default 90 -- consistent with raw-frame fingers pointing left and "
             "aligned-frame fingers pointing up.",
    )
    ap.add_argument(
        "--sanity_dir", default="results/crop_sanity",
        help="Directory to dump 4 sample crops per manifest (and 4 rotated "
             "Baseline-A crops) for visual verification before scoring.",
    )
    args = ap.parse_args()

    if args.manifests:
        paths = [Path(p) for p in args.manifests]
    else:
        paths = sorted(Path(args.manifests_dir).glob("*.json"))
    if not paths:
        print("ERROR: no manifests found.", file=sys.stderr); return 2

    print(f"Found {len(paths)} manifest(s).")
    print(f"Band filter: {args.band}")

    print("\nInitialising matchers...")
    cc: CompCodeMatcher | None = None
    rn: ResNet18Matcher | None = None
    if "compcode" in args.matchers:
        cc = CompCodeMatcher()
    if "resnet18" in args.matchers:
        rn = ResNet18Matcher(ckpt_pattern=args.resnet_ckpt_pattern, device=args.device)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "manifest", "source", "seed", "matcher", "band", "rotation_deg",
                "EER_pct", "Rank1_pct",
                "n_probes", "n_gallery_ids", "n_gallery_templates",
                "n_genuine", "n_impostor",
                "eer_threshold", "genuine_mean_dist", "impostor_mean_dist",
            ])

        project_root = Path(args.project_root).resolve()

        for path in paths:
            print(f"\n--- {path.name} ---")
            try:
                manifest = load_manifest(path, project_root=project_root, band_filter=args.band)
            except Exception as e:
                print(f"  [SKIP] failed to load: {e}", file=sys.stderr); continue

            source, seed = parse_manifest_name(manifest.name)
            print(f"  source={source}  seed={seed}  entries(after band filter)={len(manifest.entries)}")
            print(f"  image_root={manifest.image_root}")

            split = make_split(manifest)
            if not split.gallery or not split.probes:
                print(f"  [SKIP] empty split: {len(split.gallery)} gallery, {len(split.probes)} probes",
                      file=sys.stderr); continue
            if split.skipped:
                print(f"  [WARN] {len(split.skipped)} identities skipped due to incomplete sessions: "
                      f"{split.skipped[:3]}{'...' if len(split.skipped) > 3 else ''}")
            print(f"  gallery: {len(split.gallery)} identities x ~3 templates  |  probes: {len(split.probes)}")

            # Decide rotation policy for this manifest.
            is_baseline_a = (source == "Baseline-A")
            rot_deg = float(args.baseline_a_rotation) if is_baseline_a else 0.0
            if is_baseline_a:
                print(f"  Baseline-A detected -> applying constant {rot_deg} deg rotation to crops.")

            # Build crop cache once per manifest -- shared across matchers.
            entries_to_load = all_entries(split)
            crop_cache: dict[str, np.ndarray] = {}
            for e in entries_to_load:
                if e.fname in crop_cache:
                    continue
                img = Image.open(manifest.image_root / e.fname)
                crop_cache[e.fname] = crop_roi(img, e, out_size=128, rotate_deg=rot_deg)

            # Sanity dump: 4 sample crops per manifest. Open these and confirm
            # they are upright palm interiors. For Baseline-A specifically:
            # if fingers still point sideways after rotation, --baseline_a_rotation
            # has the wrong sign (try -90).
            sanity_dir = Path(args.sanity_dir) / manifest.name
            sanity_dir.mkdir(parents=True, exist_ok=True)
            sample_fnames = list(crop_cache.keys())[:max(1, len(crop_cache) // 20)][:4]
            if not sample_fnames:
                sample_fnames = list(crop_cache.keys())[:4]
            for k, fname in enumerate(sample_fnames):
                arr = crop_cache[fname]
                pil = Image.fromarray(arr, mode="L") if arr.ndim == 2 else Image.fromarray(arr)
                pil.save(sanity_dir / f"sanity_{k:02d}_{Path(fname).stem}.png")

            if cc is not None:
                try:
                    m = evaluate_compcode(split, cc, crop_cache)
                    print(f"  compcode   EER = {m['eer_pct']:5.2f}%   Rank-1 = {m['rank1_pct']:6.2f}%")
                    w.writerow([
                        manifest.name, source, seed, "compcode", args.band, f"{rot_deg:.1f}",
                        f"{m['eer_pct']:.4f}", f"{m['rank1_pct']:.4f}",
                        m["n_probes"], m["n_gallery_identities"], m["n_gallery_templates"],
                        m["n_genuine"], m["n_impostor"],
                        f"{m['eer_thr']:.6f}", f"{m['genuine_mean']:.6f}", f"{m['impostor_mean']:.6f}",
                    ])
                    f.flush()
                except Exception as e:
                    print(f"  [FAIL] compcode: {e}", file=sys.stderr)

            if rn is not None:
                if not seed:
                    print(f"  [FAIL] resnet18: cannot infer seed from manifest name {manifest.name!r}",
                          file=sys.stderr)
                else:
                    try:
                        m = evaluate_resnet(split, rn, crop_cache, seed=seed)
                        print(f"  resnet18   EER = {m['eer_pct']:5.2f}%   Rank-1 = {m['rank1_pct']:6.2f}%")
                        w.writerow([
                            manifest.name, source, seed, "resnet18", args.band, f"{rot_deg:.1f}",
                            f"{m['eer_pct']:.4f}", f"{m['rank1_pct']:.4f}",
                            m["n_probes"], m["n_gallery_identities"], m["n_gallery_templates"],
                            m["n_genuine"], m["n_impostor"],
                            f"{m['eer_thr']:.6f}", f"{m['genuine_mean']:.6f}", f"{m['impostor_mean']:.6f}",
                        ])
                        f.flush()
                    except Exception as e:
                        print(f"  [FAIL] resnet18: {e}", file=sys.stderr)

    print(f"\nDone. Results appended to: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())