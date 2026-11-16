"""Label-noise analysis (run after t6_analyze.py).
Label sets: A1 (original), A2 (second annotator), P2 (annotator 1, second pass),
CONS = 0.5*mean(A1,P2) + 0.5*A2 (each person weighted equally).
For each set: raw IoU, and IoU after removing each run's constant offset (cross-fitted, split-half by subject).
UniRot-ROI minus each detector: subject-cluster bootstrap, 95% CI and 90% CI; TOST with delta = 0.5*(1 - inter-annotator mIoU).
Human-parity rate: fraction of images where the method agrees with A2 at least as well as A1 does."""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from rx_common import polygon_iou, square_corners, label_to_px
W0, H0 = 768, 576; SEEDS = [42, 2025, 1337]; NSPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 100; B = 2000
PK = Path("../labelling/second_annotator/data"); OUT = Path("out/paper/T8_label_noise"); OUT.mkdir(parents=True, exist_ok=True)
LOC = {"UniRot-ROI": ["pipeline_aug"], "Pipeline-basic": ["pipeline"], "YOLO-basic": ["yolo"],
       "YOLO-matched": ["yolo_aug", "yolo_aug_t101", "yolo_aug_t202"], "YOLO-matched-PT": ["yolo_pre", "yolo_pre_t101", "yolo_pre_t202"],
       "YOLOv8s-pose-PT": ["yolo_s_pre"], "YOLOv8m-pose-PT": ["yolo_m_pre"]}
def wrap(d): return (d + 180) % 360 - 180
def iou(p, l): return polygon_iou(square_corners(*p), square_corners(*l))
def offset(p, l):
    t = math.radians(p[3]); dx, dy = l[0] - p[0], l[1] - p[1]
    return np.array([(dx*math.cos(t) + dy*math.sin(t))/p[2], (-dx*math.sin(t) + dy*math.cos(t))/p[2], math.log(l[2]/p[2]), wrap(l[3] - p[3])])
def apply(p, o):
    t = math.radians(p[3]); u, v = o[0]*p[2], o[1]*p[2]
    return (p[0] + u*math.cos(t) - v*math.sin(t), p[1] + u*math.sin(t) + v*math.cos(t), p[2]*math.exp(o[2]), p[3] + o[3])
def cons(a1, p2, a2):
    m = lambda x, y, w: x*w + y*(1-w)
    s1 = (a1[0]+p2[0])/2, (a1[1]+p2[1])/2, math.sqrt(a1[2]*p2[2]), a1[3] + wrap(p2[3]-a1[3])/2
    return (m(s1[0], a2[0], .5), m(s1[1], a2[1], .5), math.sqrt(s1[2]*a2[2]), s1[3] + wrap(a2[3]-s1[3])/2)
def cboot(d, subj, rng, q):
    us = np.unique(subj); idx = {u: np.where(subj == u)[0] for u in us}; bs = []
    for _ in range(B):
        pick = rng.choice(us, len(us)); ii = np.concatenate([idx[u] for u in pick]); bs.append(d[ii].mean())
    return [np.percentile(bs, x) for x in q]
rng = np.random.default_rng(8); rows, par, hum = [], [], []
for ds in ("vera", "scut"):
    d = PK/ds; names = json.load(open(d/"eval_list.json")); N = len(names)
    subj = np.array([n.split("_")[1] for n in names])
    man = json.load(open(Path("out/cross")/ds/"manifest.json"))
    raw = {k: json.load(open(f)) for k, f in (("A1", Path("out/cross")/ds/"labels_manual.json"), ("A2", d/"labels_annotator2.json"), ("P2", d/"labels_pass2.json"))}
    L = {k: [label_to_px(raw[k][n], W0, H0) for n in names] for k in raw}
    L["CONS"] = [cons(a, p, b) for a, p, b in zip(L["A1"], L["P2"], L["A2"])]
    h12 = np.array([iou(a, b) for a, b in zip(L["A1"], L["A2"])]); h11 = np.array([iou(a, b) for a, b in zip(L["A1"], L["P2"])])
    delta = 0.5*(1 - h12.mean())
    hum.append(dict(ds=ds, inter_A1_A2=h12.mean(), intra_A1_P2=h11.mean(), A2_vs_P2=np.mean([iou(a, b) for a, b in zip(L["A2"], L["P2"])]), delta=delta))
    key = [f"{ds}_{Path(man[n]['source'].replace(chr(92), '/')).stem}" for n in names]
    runs = {}
    for loc, srcs in LOC.items():
        for src in srcs:
            for s in SEEDS:
                f = Path("out/xmatch")/ds/f"{src}_seed{s}"/"rois.json"
                if not f.is_file(): continue
                P = json.load(open(f)); pi = pd.read_csv(Path("out/cross/results")/ds/f"per_image_{src}_seed{s}.csv").set_index("file")
                nod = [("n_det" in pi.columns and pi.loc[n, "n_det"] == 0) for n in names]
                runs.setdefault(loc, []).append(([tuple(P[k][:4]) for k in key], nod))
    us = np.unique(subj); splits = []
    for _ in range(NSPLIT):
        a = set(rng.permutation(us)[:len(us)//2]); m = np.array([x in a for x in subj]); splits += [m, ~m]
    per = {}   # (loc, set, mode) -> per-image IoU averaged over runs
    for loc, R in runs.items():
        for ls in L:
            rawv = np.mean([[0.0 if nd[i] else iou(p[i], L[ls][i]) for i in range(N)] for p, nd in R], 0)
            acc = np.zeros(N); cnt = np.zeros(N)
            for p, nd in R:
                O = np.array([offset(p[i], L[ls][i]) if not nd[i] else np.full(4, np.nan) for i in range(N)])
                for tr in splits:
                    o = np.nanmean(O[tr], 0); te = np.where(~tr)[0]
                    for i in te: acc[i] += 0.0 if nd[i] else iou(apply(p[i], o), L[ls][i]); cnt[i] += 1
            per[(loc, ls, "raw")] = rawv; per[(loc, ls, "offset_removed")] = acc/cnt
        par.append(dict(ds=ds, localiser=loc, parity_vs_A2=np.mean(np.mean([[ (0.0 if nd[i] else iou(p[i], L["A2"][i])) >= h12[i] for i in range(N)] for p, nd in R], 0))))
    for (loc, ls, mode), v in per.items():
        rows.append(dict(ds=ds, labels=ls, mode=mode, comparison=loc, value=v.mean()))
        if loc == "UniRot-ROI": continue
        dd = per[("UniRot-ROI", ls, mode)] - v; lo95, lo90, hi90, hi95 = cboot(dd, subj, rng, [2.5, 5, 95, 97.5])
        rows.append(dict(ds=ds, labels=ls, mode=mode, comparison=f"UniRot-ROI minus {loc}", value=dd.mean(), lo95=lo95, hi95=hi95, lo90=lo90, hi90=hi90,
                         delta=delta, equivalent=(lo90 > -delta) and (hi90 < delta),
                         verdict=("better" if lo95 > 0 else "worse" if hi95 < 0 else "n.s.") + (", equivalent" if (lo90 > -delta and hi90 < delta) else "")))
R = pd.DataFrame(rows); H = pd.DataFrame(hum); PR = pd.DataFrame(par)
R.to_csv(OUT/"label_noise_results.csv", index=False); H.to_csv(OUT/"human_agreement.csv", index=False); PR.to_csv(OUT/"human_parity.csv", index=False)
pd.set_option("display.width", 250); print(H.round(4).to_string(index=False)); print(); print(PR.round(3).to_string(index=False)); print()
D = R[R.comparison.str.contains("minus")].copy(); D["cmp"] = D.comparison.str.replace("UniRot-ROI minus ", "")
for ds in ("vera", "scut"):
    print(f"== {ds}: UniRot-ROI minus detector, value [95% CI] verdict (delta={H[H.ds==ds].delta.iloc[0]:.3f})")
    for mode in ("raw", "offset_removed"):
        t = D[(D.ds == ds) & (D["mode"] == mode)].pivot(index="cmp", columns="labels", values="verdict")
        v = D[(D.ds == ds) & (D["mode"] == mode)].assign(s=lambda x: x.value.map("{:+.3f}".format) + " [" + x.lo95.map("{:+.3f}".format) + "," + x.hi95.map("{:+.3f}".format) + "]").pivot(index="cmp", columns="labels", values="s")
        print("--", mode); print((v + "  " + t)[["A1", "A2", "P2", "CONS"]].to_string()); print()
