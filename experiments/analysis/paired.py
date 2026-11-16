import pickle, numpy as np, pandas as pd
import os as _os; _os.makedirs("results", exist_ok=True)
P = pickle.load(open("results/perimg.pkl", "rb"))
rng = np.random.default_rng(7)
SEEDS = [42, 2025, 1337]
DET = {"YOLO-matched": ["yolo_aug", "yolo_aug_t101", "yolo_aug_t202"], "YOLO-matched-PT": ["yolo_pre", "yolo_pre_t101", "yolo_pre_t202"],
       "YOLOv8s-pose-PT": ["yolo_s_pre"], "YOLOv8m-pose-PT": ["yolo_m_pre"], "Pipeline-basic": ["pipeline"], "YOLO-basic": ["yolo"]}
rows = []
for ds, p in P.items():
    for det, srcs in DET.items():
        for k, lab in ((0, "before"), (1, "after")):
            D = np.stack([p[("UniRot-ROI", "pipeline_aug", s)][k] - p[(det, src, s)][k] for s in SEEDS for src in srcs])
            n = D.shape[1]; bs = np.array([D[:, rng.integers(0, n, n)].mean() for _ in range(5000)])
            per_run = D.mean(1)
            rows.append(dict(ds=ds, detector=det, when=lab, diff=D.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             runs_pos=f"{(per_run > 0).sum()}/{len(per_run)}"))
R = pd.DataFrame(rows); R.to_csv("results/offset_paired.csv", index=False)
pd.set_option("display.width", 200); print(R.round(4).to_string(index=False))
