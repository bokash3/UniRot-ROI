"""Agreement between label sets and the ranking of the localisers under each set.
  python t6_analyze.py                         # annotator 2 (labels_annotator2.json) and pass 2 (labels_pass2.json) if present
Writes out/paper/T6_T7_results/*.csv and prints a summary."""
import json, math, argparse
from pathlib import Path
import numpy as np, pandas as pd
from rx_common import polygon_iou, square_corners, label_to_px
W0, H0 = 768, 576
PK = Path("../labelling/second_annotator/data")
LOC = {"UniRot-ROI": ["pipeline_aug"], "Pipeline-basic": ["pipeline"], "YOLO-basic": ["yolo"],
       "YOLO-matched": ["yolo_aug", "yolo_aug_t101", "yolo_aug_t202"], "YOLO-matched-PT": ["yolo_pre", "yolo_pre_t101", "yolo_pre_t202"],
       "YOLOv8s-pose-PT": ["yolo_s_pre"], "YOLOv8m-pose-PT": ["yolo_m_pre"]}
SEEDS = [42, 2025, 1337]
def sq(l): return label_to_px(l, W0, H0)
def wrap(d): return (d + 180) % 360 - 180
def agree(A, B, names):
    r = []
    for n in names:
        if n not in A or n not in B or A[n].get("skip") or B[n].get("skip"): continue
        a, b = sq(A[n]), sq(B[n])
        r.append(dict(name=n, iou=polygon_iou(square_corners(*a), square_corners(*b)), centre_px=math.hypot(a[0] - b[0], a[1] - b[1]),
                      size_pct=100 * (b[2] / a[2] - 1), angle_deg=wrap(b[3] - a[3])))
    return pd.DataFrame(r)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a2", default="labels_annotator2.json"); ap.add_argument("--p2", default="labels_pass2.json")
    a = ap.parse_args()
    out = Path("out/paper/T6_T7_results"); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(6); summ, rank_rows = [], []
    for ds in ("vera", "scut"):
        d = PK / ds; names = json.load(open(d / "eval_list.json"))
        man = json.load(open(Path("out/cross") / ds / "manifest.json"))
        sets = {"A1": json.load(open(Path("out/cross") / ds / "labels_manual.json"))}
        for key, fn in (("A2", a.a2), ("A1-pass2", a.p2)):
            if (d / fn).is_file(): sets[key] = json.load(open(d / fn))
        for key in [k for k in sets if k != "A1"]:
            g = agree(sets["A1"], sets[key], names); g.to_csv(out / f"agreement_{ds}_A1_vs_{key}.csv", index=False)
            summ.append(dict(ds=ds, pair=f"A1 vs {key}", n=len(g), mean_iou=g.iou.mean(), min_iou=g.iou.min(), centre_px=g.centre_px.mean(),
                             size_bias_pct=g.size_pct.mean(), size_abs_pct=g.size_pct.abs().mean(), angle_bias=g.angle_deg.mean(), angle_abs=g.angle_deg.abs().mean()))
        # localisers scored against every label set on the same images
        key_of = {n: f"{ds}_{Path(man[n]['source'].replace(chr(92), '/')).stem}" for n in names}
        per = {}
        for loc, srcs in LOC.items():
            for src in srcs:
                for s in SEEDS:
                    f = Path("out/xmatch") / ds / f"{src}_seed{s}" / "rois.json"
                    if not f.is_file(): continue
                    P = json.load(open(f)); pi = pd.read_csv(Path("out/cross/results") / ds / f"per_image_{src}_seed{s}.csv").set_index("file")
                    for setname, L in sets.items():
                        v = []
                        for n in names:
                            if n not in L or L[n].get("skip"): v.append(np.nan); continue
                            nodet = "n_det" in pi.columns and pi.loc[n, "n_det"] == 0
                            v.append(0.0 if nodet else polygon_iou(square_corners(*P[key_of[n]][:4]), square_corners(*sq(L[n]))))
                        per[(loc, src, s, setname)] = np.array(v)
        for setname in sets:
            for loc, srcs in LOC.items():
                xs = [per[k] for k in per if k[0] == loc and k[3] == setname]
                if xs: rank_rows.append(dict(ds=ds, labels=setname, localiser=loc, mIoU=np.nanmean([x.mean() if not np.isnan(x).any() else np.nanmean(x) for x in xs])))
            # paired UniRot-ROI minus detector, image bootstrap
            for det in [l for l in LOC if l != "UniRot-ROI"]:
                D = [per[("UniRot-ROI", "pipeline_aug", s, setname)] - per[(det, src, s, setname)] for s in SEEDS for src in LOC[det]
                     if (det, src, s, setname) in per]
                if not D: continue
                D = np.stack(D); ok = ~np.isnan(D).any(0); D = D[:, ok]; n = D.shape[1]
                bs = [D[:, rng.integers(0, n, n)].mean() for _ in range(5000)]
                rank_rows.append(dict(ds=ds, labels=setname, localiser=f"UniRot-ROI minus {det}", mIoU=D.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    S = pd.DataFrame(summ); R = pd.DataFrame(rank_rows)
    S.to_csv(out / "agreement_summary.csv", index=False); R.to_csv(out / "ranking_by_label_set.csv", index=False)
    pd.set_option("display.width", 220); print(S.round(3).to_string(index=False)); print()
    for (ds, ls), g in R[~R.localiser.str.contains("minus")].groupby(["ds", "labels"]):
        g = g.sort_values("mIoU", ascending=False); print(ds, ls, " > ".join(f"{r.localiser} {r.mIoU:.3f}" for r in g.itertuples()))
    print(); print(R[R.localiser.str.contains("minus")].round(4).to_string(index=False))
if __name__ == "__main__":
    main()
