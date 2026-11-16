import sys, json, math, pickle; sys.path.insert(0, '.')
from common import corners, iou, wrap, IN
import numpy as np, pandas as pd
import os as _os; _os.makedirs("results", exist_ok=True)
W, H = 768, 576
rng = np.random.default_rng(11)
SEEDS = [42, 2025, 1337]
LOC = [("UniRot-ROI", ["P:full_pipeline_aug"]), ("Pipeline-basic", ["P:full_pipeline"]), ("YOLO-basic", ["Y:pose"]),
       ("YOLO-matched", ["Y:pose_aug", "Y:pose_aug_t101", "Y:pose_aug_t202"]),
       ("YOLO-matched-PT", ["Y:pose_pre", "Y:pose_pre_t101", "Y:pose_pre_t202"]),
       ("YOLOv8s-pose-PT", ["Y:pose_s_pre"]), ("YOLOv8m-pose-PT", ["Y:pose_m_pre"])]
px = lambda l: np.array([l["cx"] * W, l["cy"] * H, l["size"] * min(W, H), float(l.get("theta", 0.0))])
def load(src, s):
    k, n = src.split(":")
    f = IN / (f"out/{n}/seed{s}/pred_labels_full.json" if k == "P" else f"runs_yolo/{n}_seed{s}/eval/pred_labels.json")
    return json.load(open(f))
def frame(p):
    c = corners(*p); return (c[1] - c[0]) / p[2], (c[3] - c[0]) / p[2]
rows, PER = [], {}
for name, srcs in LOC:
    for src in srcs:
        for s in SEEDS:
            ref = json.load(open(IN / f"out/labels_oriented/seed{s}/test_labels.json")); P = load(src, s)
            keys = sorted(ref); subj = np.array([k.split("_")[0] for k in keys]); us = np.unique(subj)
            G = [px(ref[k]) for k in keys]; Q = [px(P[k]) for k in keys]
            n = len(keys); iou0 = np.array([iou(q, g) for q, g in zip(Q, G)])
            off = np.array([[((g[:2] - q[:2]) @ frame(q)[0]) / q[2], ((g[:2] - q[:2]) @ frame(q)[1]) / q[2],
                             math.log(g[2] / q[2]), wrap(g[3] - q[3])] for q, g in zip(Q, G)])
            ok = iou0 >= 0.5
            acc = np.zeros(n); NS = 100
            for b in range(NS):
                hs = set(rng.permutation(us)[: len(us) // 2]); half = np.array([x in hs for x in subj])
                for est, app in ((half, ~half), (~half, half)):
                    m = off[est & ok].mean(0)
                    for i in np.where(app)[0]:
                        q = Q[i]; u, v = frame(q); c = q[:2] + q[2] * (m[0] * u + m[1] * v)
                        acc[i] += iou((c[0], c[1], q[2] * math.exp(m[2]), q[3] + m[3]), G[i])
            cor = acc / NS
            full = off[ok].mean(0)
            rows.append(dict(loc=name, src=src, seed=s, mIoU0=iou0.mean(), mIoU1=cor.mean(), off_x=full[0], off_y=full[1],
                             off_size_pct=100 * (math.exp(full[2]) - 1), off_ang=full[3]))
            PER[(name, src, s)] = (iou0, cor, subj)
R = pd.DataFrame(rows); R.to_csv("results/offset_runs_casia.csv", index=False)
pickle.dump(PER, open("results/perimg_casia.pkl", "wb"))
S = R.groupby("loc", sort=False).agg(runs=("mIoU0", "size"), mIoU_before=("mIoU0", "mean"), mIoU_after=("mIoU1", "mean"),
    off_x=("off_x", "mean"), off_y=("off_y", "mean"), off_size_pct=("off_size_pct", "mean"), off_ang=("off_ang", "mean")).reset_index()
S["rank_before"] = S.mIoU_before.rank(ascending=False).astype(int); S["rank_after"] = S.mIoU_after.rank(ascending=False).astype(int)
pd.set_option("display.width", 200); print(S.round(4).to_string(index=False))
# paired, subject-level bootstrap within seeds
out = []
for det, srcs in LOC[1:]:
    for k, lab in ((0, "before"), (1, "after")):
        reps = []; diffs = []
        for s in SEEDS:
            u = PER[("UniRot-ROI", "P:full_pipeline_aug", s)]
            for src in srcs:
                d = PER[(det, src, s)]; diffs.append((u[k] - d[k], u[2]))
        point = np.mean([x.mean() for x, _ in diffs])
        bs = []
        for _ in range(2000):
            vals = []
            for x, subj in diffs:
                us = np.unique(subj); pick = rng.choice(us, len(us))
                vals.append(np.concatenate([x[subj == p] for p in pick]).mean())
            bs.append(np.mean(vals))
        out.append(dict(detector=det, when=lab, diff=point, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
O = pd.DataFrame(out); O.to_csv("results/offset_paired_casia.csv", index=False); print(O.round(4).to_string(index=False))
