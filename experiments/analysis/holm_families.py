"""Holm families = the comparisons reported in Supplementary Data 1 (one family per dataset x metric [x matcher]).
Re-derives difference, CI and unadjusted p from the bootstrap files, applies Holm with m = family size, and checks the
published table."""
import re, numpy as np, pandas as pd
import os as _os; _os.makedirs("results", exist_ok=True)
SD1 = "inputs/Supplementary_Data_1_paired_bootstrap.csv"
KEY = {"UniRot-ROI": "pipeline_aug", "Pipeline-basic": "pipeline", "YOLO-basic": "yolo", "YOLO-basic-100": "yolo_l100",
       "YOLO-matched": "yolo_aug", "YOLO-matched-PT": "yolo_pre", "Valley": "valley", "Released SCUT-PV ROI": "official",
       "Image input, raw frame, matched": "rawA_aug", "Mask input, raw frame, matched": "rawB_aug", "YOLOv8s-pose-PT": "yolo_s_pre", "YOLOv8m-pose-PT": "yolo_m_pre"}
MET = {"IoU": "iou", "Angle (°)": "angle_err_deg", "Failure rate (points)": "failed"}
MAT = {"CompCode": "compcode", "ResNet-18": "resnet18", "Vein": "miura"}
def src(section, metric):
    if section.startswith("CASIA"): return "inputs/out/bootstrap/geom_casia.csv", None
    if section.startswith("VERA, labelled"): return "inputs/out/bootstrap/geom_cross_vera.csv", None
    if section.startswith("SCUT-PV, labelled"): return "inputs/out/bootstrap/geom_cross_scut.csv", None
    ds = "vera" if section.startswith("VERA") else "scut"
    return f"bootstrap/match_{ds}.csv", MAT[metric.split(", ")[1]]
d = pd.read_csv(SD1)
out = []
cache = {}
for r in d.itertuples(index=False):
    sec, met, comp = r[0], r[1], r[2]
    f, mt = src(sec, met)
    B = cache.setdefault(f, pd.read_csv(f))
    a_, b_ = [KEY[x.strip()] for x in comp.split(" - ")]
    m = "EER_pct" if mt else MET[met]
    sel = B[(B.metric == m) & ((B.matcher == mt) if mt else True)]
    row = sel[(sel.A == a_) & (sel.B == b_)]; sign = 1
    if row.empty: row = sel[(sel.A == b_) & (sel.B == a_)]; sign = -1
    assert len(row) == 1, (sec, met, comp)
    x = row.iloc[0]
    lo, hi = (x.ci_low, x.ci_high) if sign == 1 else (-x.ci_high, -x.ci_low)
    if m == "failed": x = x.copy(); x["diff_A_minus_B"] *= 100; lo, hi = lo * 100, hi * 100
    out.append(dict(section=sec, metric=met, comparison=comp, diff=sign * x.diff_A_minus_B, lo=lo, hi=hi, p=x.p_two_sided,
                    old_diff=r[3], old_ci=r[4], old_p=r[5], old_holm=r[6]))
# add the pairs not listed in the input table (they are part in the Holm families: 22 on CASIA-MS, 17 on VERA/SCUT-PV)
DROPPED = [("Pipeline-basic", "YOLO-matched"), ("Pipeline-basic", "Valley"), ("Pipeline-basic", "YOLO-basic-100"),
           ("Valley", "YOLO-basic"), ("Valley", "YOLO-matched"), ("Valley", "YOLO-basic-100"), ("UniRot-ROI", "YOLO-basic-100"),
           ("Image input, raw frame, matched", "YOLO-matched"), ("Mask input, raw frame, matched", "YOLO-matched")]
geo = d[~d.Metric.str.startswith("EER")][["Section", "Metric"]].drop_duplicates()
for sec, met in geo.itertuples(index=False):
    f, _ = src(sec, met); Bf = cache.setdefault(f, pd.read_csv(f)); m = MET[met]
    for a_n, b_n in DROPPED:
        a_, b_ = KEY[a_n], KEY[b_n]
        sel = Bf[Bf.metric == m]; row = sel[(sel.A == a_) & (sel.B == b_)]; sign = 1
        if row.empty: row = sel[(sel.A == b_) & (sel.B == a_)]; sign = -1
        if row.empty: continue
        x = row.iloc[0].copy(); lo, hi = (x.ci_low, x.ci_high) if sign == 1 else (-x.ci_high, -x.ci_low)
        dd = sign * x.diff_A_minus_B
        if m == "failed": dd, lo, hi = dd * 100, lo * 100, hi * 100
        out.append(dict(section=sec, metric=met, comparison=f"{a_n} - {b_n}", diff=dd, lo=lo, hi=hi, p=x.p_two_sided,
                        old_diff=f"{dd:+.4f}", old_ci="", old_p="", old_holm="restored"))
NEW = [("UniRot-ROI", "YOLOv8s-pose-PT"), ("UniRot-ROI", "YOLOv8m-pose-PT"), ("YOLOv8s-pose-PT", "YOLO-matched-PT"), ("YOLOv8m-pose-PT", "YOLOv8s-pose-PT")]
for sec, met in geo.itertuples(index=False):
    f, _ = src(sec, met); Bf = cache.setdefault(f, pd.read_csv(f)); m = MET[met]
    for a_n, b_n in NEW:
        a_, b_ = KEY[a_n], KEY[b_n]
        sel = Bf[Bf.metric == m]; row = sel[(sel.A == a_) & (sel.B == b_)]; sign = 1
        if row.empty: row = sel[(sel.A == b_) & (sel.B == a_)]; sign = -1
        assert len(row) == 1, (sec, met, a_n, b_n)
        x = row.iloc[0].copy(); lo, hi = (x.ci_low, x.ci_high) if sign == 1 else (-x.ci_high, -x.ci_low)
        dd = sign * x.diff_A_minus_B
        if m == "failed": dd, lo, hi = dd * 100, lo * 100, hi * 100
        out.append(dict(section=sec, metric=met, comparison=f"{a_n} - {b_n}", diff=dd, lo=lo, hi=hi, p=x.p_two_sided,
                        old_diff=f"{dd:+.4f}", old_ci="", old_p="", old_holm="added"))
rec = d[d.Metric.str.startswith("EER")][["Section", "Metric"]].drop_duplicates()
for sec, met in rec.itertuples(index=False):
    ds = "vera" if sec.startswith("VERA") else "scut"; mt = MAT[met.split(", ")[1]]
    Bf = cache.setdefault("A1m" + ds, pd.read_csv(f"inputs/out/bootstrap/match_{ds}.csv"))
    sel = Bf[(Bf.metric == "EER_pct") & (Bf.matcher == mt)]
    for a_n, b_n in [("UniRot-ROI", "YOLOv8s-pose-PT"), ("UniRot-ROI", "YOLOv8m-pose-PT")]:
        row = sel[(sel.A == KEY[a_n]) & (sel.B == KEY[b_n])]; assert len(row) == 1, (sec, met, a_n, b_n); x = row.iloc[0]
        out.append(dict(section=sec, metric=met, comparison=f"{a_n} - {b_n}", diff=x.diff_A_minus_B, lo=x.ci_low, hi=x.ci_high, p=x.p_two_sided,
                        old_diff=f"{x.diff_A_minus_B:+.4f}", old_ci="", old_p="", old_holm="added"))
O = pd.DataFrame(out)
B_REP = 2000
def holm(ps):
    p = np.where(ps == 0, 2 / B_REP, ps); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (m - k) * p[i])); adj[i] = run
    return adj
O["family"] = O.section + " | " + O.metric
O["m"] = O.groupby("family").comparison.transform("size")
O["holm"] = O.groupby("family").p.transform(lambda s: holm(s.values))
dec = O.metric.map(lambda m: 3 if m == "IoU" else (1 if m == "Failure rate (points)" else 2))
def fmt(v, k):
    s = f"{v:+.{k}f}"
    if abs(v) < 10 ** -k and v != 0: s = f"{v:+.{k+1}f}"
    return s.replace("-", "−")
O["Difference"] = [fmt(v, k) for v, k in zip(O["diff"], dec)]
O["95% CI"] = [f"[{fmt(a, k)}, {fmt(b, k)}]" for a, b, k in zip(O.lo, O.hi, dec)]
O["p"] = O.p.map(lambda v: "<0.001" if v == 0 else f"{v:.3f}")
O["p (Holm)"] = [(f"≤{h:.3f}" if p0 == "<0.001" else f"{h:.3f}") for h, p0 in zip(O.holm, O["p"])]
# checks against the published file
def num(s): return float(str(s).replace("−", "-").replace("+", ""))
O["diff_ok"] = [abs(round(a, k) - num(b)) < 1.5 * 10 ** -k for a, b, k in zip(O["diff"], O.old_diff, dec)]
O["sig_old"] = O.old_holm.map(lambda s: None if s in ("restored", "added") else num(str(s).replace("≤", "")) < 0.05)
O["sig_new"] = O.holm < 0.05
pd.set_option("display.width", 250, "display.max_rows", 300)
print("rows", len(O), "| difference reproduced:", O.diff_ok.all())
print(O.groupby("family").m.first().to_string())
O["old_holm_num"] = O.old_holm.map(lambda s: None if s in ("restored", "added") else num(str(s).replace("≤", "")))
O["holm_match"] = [(o is None) or abs(o - round(h, 3)) < 0.0015 for o, h in zip(O.old_holm_num, O.holm)]
print("published Holm values reproduced with the full families:", O.holm_match.all(), "| mismatches:", (~O.holm_match).sum())
ch = O[(O.sig_old.notna()) & (O.sig_old != O.sig_new)]
print("\nsignificance changes after correcting m:"); print(ch[["family", "comparison", "Difference", "p", "old_holm", "p (Holm)"]].to_string() if len(ch) else "  none")
print("\nnon-reproduced differences:"); print(O[~O.diff_ok][["family", "comparison", "diff", "old_diff"]].to_string() if (~O.diff_ok).any() else "  none")
O = O.sort_values(["section", "metric"], kind="stable")
order = list(dict.fromkeys(d.Section)); O["_s"] = O.section.map(order.index); O = O.sort_values(["_s"], kind="stable")
out = O[["section", "metric", "comparison", "Difference", "95% CI", "p", "p (Holm)", "m"]]
out.columns = ["Section", "Metric", "Comparison (A - B)", "Difference", "95% CI", "p", "p (Holm)", "Family size m"]
out.to_csv("results/Supplementary_Data_1_recomputed.csv", index=False, encoding="utf-8-sig")
O.to_csv("results/holm_check.csv", index=False)
