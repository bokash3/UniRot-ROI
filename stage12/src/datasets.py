import os, json, re, math, random
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
#import os, argparse, json, yaml, math
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # silence update warnings
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(images_dir: str) -> List[str]:
    files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS]
    files.sort()
    return files

def parse_subject_from_filename(name: str) -> str:
    # Extract leading digits before the first underscore, e.g., '001_l_940_01.jpg' -> '001'
    m = re.match(r"(\d+)_", name)
    return m.group(1) if m else os.path.splitext(name)[0]

def parse_spectrum_from_filename(name: str) -> Optional[str]:
    # Heuristic: use the 3rd token between underscores, e.g., '001_l_940_01.jpg' -> '940'
    tokens = name.split("_")
    if len(tokens) >= 3:
        token = tokens[2]
        token = os.path.splitext(token)[0]
        return token
    return None

def load_angles(angles_json: str) -> Dict[str, Any]:
    with open(angles_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "angle_deg" in v:
            out[k] = v
        else:
            out[k] = {"angle_deg": float(v)}
        if "subject_id" not in out[k]:
            out[k]["subject_id"] = parse_subject_from_filename(k)
        if "spectrum" not in out[k]:
            sp = parse_spectrum_from_filename(k)
            if sp is not None:
                out[k]["spectrum"] = sp
    return out

# ---------------- Segmentation Dataset ----------------
class SegmentationDataset(Dataset):
    """Loads image and binary mask (if available)."""
    def __init__(self, images_dir: str, masks_dir: Optional[str], files: List[str],
                 img_size=(576,768), is_train=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.files = files
        self.H, self.W = img_size
        # Mild, label-consistent augs
        if is_train:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=max(self.W, self.H)),
                A.PadIfNeeded(min_height=self.H, min_width=self.W, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1, contrast_limit=0.1),
                A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=max(self.W, self.H)),
                A.PadIfNeeded(min_height=self.H, min_width=self.W, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = cv2.imread(os.path.join(self.images_dir, name), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = None
        if self.masks_dir is not None:
            mp = os.path.join(self.masks_dir, os.path.splitext(name)[0] + ".png")
            if os.path.exists(mp):
                mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.uint8)  # binarize

        if mask is None:
            # If GT mask is missing, create a blank (used only for eval-safe behavior)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        aug = self.tf(image=img, mask=mask)
        img_t = aug["image"]
        mask_t = aug["mask"].unsqueeze(0).float()
        return img_t, mask_t, name

# ---------------- Rotation Dataset ----------------
class RotationDataset(Dataset):
    """Loads image (optionally mask-applied) and angle label."""
    def __init__(self, images_dir: str, angles_json: str, files: List[str],
                 img_size=(576,768), pred_masks_dir: Optional[str]=None,
                 grayscale=True, is_train=True):
        self.images_dir = images_dir
        self.filemeta = load_angles(angles_json)
        self.files = files
        self.pred_masks_dir = pred_masks_dir
        self.H, self.W = img_size
        self.grayscale = grayscale

        if is_train:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=max(self.W, self.H)),
                A.PadIfNeeded(min_height=self.H, min_width=self.W, border_mode=cv2.BORDER_CONSTANT, value=0),
                #A.HorizontalFlip(p=0.5),  # we DO NOT adjust angle for flips (disabled ambiguity); use small augs only
                #A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1, contrast_limit=0.1),
                #A.GaussNoise(p=0.15, var_limit=(5.0, 25.0)),
                A.Normalize(mean=(0.5,), std=(0.5,)) if grayscale else A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=max(self.W, self.H)),
                A.PadIfNeeded(min_height=self.H, min_width=self.W, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=(0.5,), std=(0.5,)) if grayscale else A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        #img = cv2.imread(os.path.join(self.images_dir, name), cv2.IMREAD_COLOR)
        #if img is None:
         #   raise FileNotFoundError(name)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # apply predicted mask if available
        if self.pred_masks_dir is not None:
            mp = os.path.join(self.pred_masks_dir, os.path.splitext(name)[0] + ".png")
            if os.path.exists(mp):
                mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = (mask > 127).astype(np.uint8)*255
                    #img = img.copy()
                    #img[mask == 0] = 0  # suppress background

        if self.grayscale:
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(mask, axis=2) #img

        aug = self.tf(image=img)
        img_t = aug["image"]
        if self.grayscale:
            img_t = img_t  # shape (1,H,W)
        angle_deg = self._get_angle(name)
        return img_t, float(angle_deg), name

    def _get_angle(self, name: str) -> float:
        meta = self.filemeta.get(name, None)
        if meta is None:
            raise KeyError(f"Angle missing for: {name}")
        if isinstance(meta, dict):
            return float(meta["angle_deg"])
        return float(meta)

# ---------------- Split helpers ----------------
def build_subject_split(files: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    # group by subject
    buckets = {}
    for f in files:
        sid = parse_subject_from_filename(f)
        buckets.setdefault(sid, []).append(f)
    subjects = list(buckets.keys())
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_test = n - n_train - n_val
    train_subj = set(subjects[:n_train])
    val_subj   = set(subjects[n_train:n_train+n_val])
    test_subj  = set(subjects[n_train+n_val:])
    train_files, val_files, test_files = [], [], []
    for s, fs in buckets.items():
        if s in train_subj: train_files += fs
        elif s in val_subj: val_files += fs
        else: test_files += fs
    return train_files, val_files, test_files
