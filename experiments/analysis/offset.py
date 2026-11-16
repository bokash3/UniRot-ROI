import sys; sys.path.insert(0, '.')
from common import *
import pandas as pd
import os as _os; _os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(2026)
LOC = [("UniRot-ROI", ["pipeline_aug"]), ("Pipeline-basic", ["pipeline"]), ("YOLO-basic", ["yolo"]),
       ("YOLO-matched", ["yolo_aug", "yolo_aug_t101", "yolo_aug_t202"]),
       ("YOLO-matched-PT", ["yolo_pre", "yolo_pre_t101", "yolo_pre_t202"]),
       ("YOLOv8s-pose-PT", ["yolo_s_pre"]), ("YOLOv8m-pose-PT", ["yolo_m_pre"])]
SEEDS = [42, 2025, 1337]
NSPLIT = 200
def frame(p):
    c = corners(*p[:4]); s = p[2]
    return (c[1] - c[0]) / s, (c[3] - c[0]) / s
def run(ds):
    R = refs(ds); keys = sorted(R); n = len(keys)
    rows, perimg = [], {}
    for name, srcs in LOC:
        for src in srcs:
            for sd in SEEDS:
                P = preds(ds, src, sd)
                pi = pd.read_csv(IN / f"out/cross/results/{ds}/per_image_{src}_seed{sd}.csv").set_index("file")
                pr = []
                for k in keys:
                    g = R[k]; det = not ("n_det" in pi.columns and pi.loc[g[4], "n_det"] == 0)
                    pr.append((np.array(P[k][:4], float) if det else None, np.array(g[:4], float)))
                # per-image offsets in the predicted square's frame
                off = np.full((n, 4), np.nan); iou0 = np.zeros(n)
                for i, (p, g) in enumerate(pr):
                    if p is None: continue
                    iou0[i] = iou(p, g)
                    u, v = frame(p); d = g[:2] - p[:2]
                    off[i] = [d @ u / p[2], d @ v / p[2], math.log(g[2] / p[2]), wrap(g[3] - p[3])]
                ok = (~np.isnan(off[:, 0])) & (iou0 >= 0.5)
                ious = np.zeros((NSPLIT, n)); cerr = np.zeros((NSPLIT, n))
                for b in range(NSPLIT):
                    half = rng.permutation(n) < n // 2
                    for est, app in ((half, ~half), (~half, half)):
                        m = off[est & ok].mean(0)
                        for i in np.where(app)[0]:
                            p, g = pr[i]
                            if p is None: ious[b, i] = 0.0; cerr[b, i] = np.nan; continue
                            u, v = frame(p)
                            c = p[:2] + p[2] * (m[0] * u + m[1] * v)
                            q = (c[0], c[1], p[2] * math.exp(m[2]), p[3] + m[3])
                            ious[b, i] = iou(q, g); cerr[b, i] = float(np.hypot(*(np.array(q[:2]) - g[:2])))
                cor = ious.mean(0)
                c0 = np.array([np.hypot(*(p[:2] - g[:2])) if p is not None else np.nan for p, g in pr])
                full = off[ok].mean(0)
                rows.append(dict(ds=ds, loc=name, src=src, seed=sd, mIoU0=iou0.mean(), mIoU1=cor.mean(),
                                 acc0=(iou0 >= .7).mean(), acc1=(ious >= .7).mean(), ctr0=np.nanmean(c0), ctr1=np.nanmean(cerr),
                                 off_x=full[0], off_y=full[1], off_size_pct=100 * (math.exp(full[2]) - 1), off_ang=full[3]))
                perimg[(name, src, sd)] = (iou0, cor)
    return pd.DataFrame(rows), perimg
allr, P = [], {}
for ds in ("vera", "scut"):
    r, p = run(ds); allr.append(r); P[ds] = p
R = pd.concat(allr)
R.to_csv("results/offset_runs.csv", index=False)
import pickle; pickle.dump(P, open("results/perimg.pkl", "wb"))
pd.set_option("display.width", 220)
S = R.groupby(["ds", "loc"], sort=False).agg(runs=("mIoU0", "size"), mIoU_before=("mIoU0", "mean"), mIoU_after=("mIoU1", "mean"),
      acc07_before=("acc0", "mean"), acc07_after=("acc1", "mean"), ctr_before=("ctr0", "mean"), ctr_after=("ctr1", "mean"),
      off_x=("off_x", "mean"), off_y=("off_y", "mean"), off_size_pct=("off_size_pct", "mean"), off_ang=("off_ang", "mean")).reset_index()
S["gain"] = S.mIoU_after - S.mIoU_before
for ds in ("vera", "scut"):
    s = S[S.ds == ds].copy()
    s["rank_before"] = s.mIoU_before.rank(ascending=False).astype(int); s["rank_after"] = s.mIoU_after.rank(ascending=False).astype(int)
    print(s.round(4).to_string(index=False)); print()
S.to_csv("results/offset_summary.csv", index=False)
