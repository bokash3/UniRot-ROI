import json, math, re, sys
from pathlib import Path
import numpy as np, cv2
import os as _os; _os.makedirs("results", exist_ok=True)
IN = Path("inputs")
W0, H0 = 768, 576
def corners(cx, cy, s, th):
    h = s / 2; a = math.radians(th); c, si = math.cos(a), math.sin(a)
    R = np.array([[c, si], [-si, c]])
    return np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ R.T + np.array([cx, cy])
def iou(p, g):
    p1 = np.float32(corners(*p)).reshape(-1, 1, 2); p2 = np.float32(corners(*g)).reshape(-1, 1, 2)
    a1, a2 = abs(cv2.contourArea(p1)), abs(cv2.contourArea(p2))
    inter = max(0.0, float(cv2.intersectConvexConvex(p1, p2)[0])); u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0
def wrap(d): return (d + 180.0) % 360.0 - 180.0
def refs(ds):
    lab = json.load(open(IN / f"out/cross/{ds}/labels_manual.json"))
    man = json.load(open(IN / f"out/cross/{ds}/manifest.json"))
    out = {}
    for n, l in lab.items():
        if l.get("skip") or n not in man: continue
        stem = Path(man[n]["source"].replace("\\", "/")).stem
        out[f"{ds}_{stem}"] = (l["cx"] * W0, l["cy"] * H0, l["size"] * min(W0, H0), float(l.get("theta", 0.0)), n)
    return out
def preds(ds, src, seed):
    return json.load(open(IN / f"out/xmatch/{ds}/{src}_seed{seed}/rois.json"))
