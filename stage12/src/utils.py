import os, json, math, time, random
from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F

# ---------------- Randomness ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ---------------- Angular math ----------------
def wrap_deg(angle: float) -> float:
    """Wrap angle into [0, 360)."""
    a = angle % 360.0
    if a < 0: a += 360.0
    return a

def ang_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    d = abs(wrap_deg(a) - wrap_deg(b)) % 360.0
    return min(d, 360.0 - d)

# Vector <-> degree helpers
def deg_to_unitvec(angle_deg: float):
    rad = math.radians(angle_deg)
    return math.cos(rad), math.sin(rad)

def unitvec_to_deg(x: float, y: float):
    return wrap_deg(math.degrees(math.atan2(y, x)))

# ---------------- Metrics: Segmentation ----------------
def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    """pred_logits: (N,1,H,W) raw logits; target: (N,1,H,W) {0,1}."""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()

def iou_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = (pred + target).clamp(0,1).sum(dim=(1,2,3))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

def precision_recall(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    tp = (pred * target).sum(dim=(1,2,3))
    fp = (pred * (1 - target)).sum(dim=(1,2,3))
    fn = ((1 - pred) * target).sum(dim=(1,2,3))
    prec = (tp + eps) / (tp + fp + eps)
    rec  = (tp + eps) / (tp + fn + eps)
    return prec.mean().item(), rec.mean().item()

# ---------------- Metrics: Rotation ----------------
def mae_deg(pred_deg: np.ndarray, gt_deg: np.ndarray) -> float:
    diffs = np.array([min(abs(p-g)%360, 360-abs(p-g)%360) for p, g in zip(pred_deg, gt_deg)], dtype=np.float32)
    return float(diffs.mean())

def percentile_deg(pred_deg: np.ndarray, gt_deg: np.ndarray, q: float) -> float:
    diffs = np.array([min(abs(p-g)%360, 360-abs(p-g)%360) for p, g in zip(pred_deg, gt_deg)], dtype=np.float32)
    return float(np.percentile(diffs, q))

# ---------------- Schedulers ----------------
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup + Cosine schedule in steps."""
    def __init__(self, optimizer, warmup_steps=200, max_steps=10000, min_lr_scale=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_scale = min_lr_scale
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                scale = step / max(1, self.warmup_steps)
            else:
                t = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                scale = self.min_lr_scale + 0.5*(1-self.min_lr_scale)*(1 + math.cos(math.pi * t))
            lrs.append(base_lr * scale)
        return lrs

# ---------------- IO helpers ----------------
def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
