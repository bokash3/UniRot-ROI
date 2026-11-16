"""export_fig2_data.py - data for Fig. 2 (qualitative comparison); needs numpy + opencv only.
Writes out/paper/fig2_data.npz with, for CASIA-MS (seed-42 test, 850 nm) and VERA (seed-42 models),
two images chosen by a fixed rule:
  typical = the image whose IoU for the augmentation-trained pipeline (UniRot-ROI) is the median
  hard    = the image on which the learned localisers disagree most (largest IoU range)
For each image: the grey image, the reference square and every localiser's square (cx, cy, s, angle),
their IoUs with the reference, and the 128 x 128 crops.
Run from experiments:  python export_fig2_data.py
"""
import csv, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, ".")
from rx_common import label_to_px, square_corners, polygon_iou
from rx_stages import oriented_crop

W0, H0 = 768, 576
SRC = ("pipeline", "pipeline_aug", "yolo", "yolo_aug", "yolo_pre")


def read(p):
    return {r["file"]: r for r in csv.DictReader(open(p))}


def lab(d):
    return label_to_px(d, W0, H0)


def pick(rows):
    names = sorted(rows)
    v = np.array([rows[n]["pipeline_aug"] for n in names])
    typ = names[int(np.argsort(v)[len(v) // 2])]
    spread = np.array([max(rows[n].values()) - min(rows[n].values()) for n in names])
    hard = names[int(np.argmax(spread))]
    return [("typical", typ), ("hard", hard)]


def pack(ds, kind, name, g, sq):
    r = {k: polygon_iou(square_corners(*sq["ref"]), square_corners(*sq[k])) for k in SRC}
    crops = {k: oriented_crop(g, *sq[k], out=128) for k in ("ref",) + SRC}
    print(ds, kind, name, {k: round(v, 3) for k, v in r.items()})
    return {"ds": ds, "kind": kind, "name": name, "img": g, "sq": {k: list(map(float, v[:4])) for k, v in sq.items()},
            "iou": r, "crops": crops}


def main():
    items = []
    te = json.load(open("out/labels_oriented/seed42/test_labels.json"))
    P = {"pipeline": json.load(open("out/full_pipeline/seed42/pred_labels_full.json")),
         "pipeline_aug": json.load(open("out/full_pipeline_aug/seed42/pred_labels_full.json")),
         "yolo": json.load(open("runs_yolo/pose_seed42/eval/pred_labels.json")),
         "yolo_aug": json.load(open("runs_yolo/pose_aug_seed42/eval/pred_labels.json")),
         "yolo_pre": json.load(open("runs_yolo/pose_pre_seed42/eval/pred_labels.json"))}
    iou = {"pipeline": read("out/full_pipeline/seed42/per_image.csv"),
           "pipeline_aug": read("out/full_pipeline_aug/seed42/per_image.csv"),
           "yolo": read("runs_yolo/pose_seed42/eval/per_image.csv"),
           "yolo_aug": read("runs_yolo/pose_aug_seed42/eval/per_image.csv"),
           "yolo_pre": read("runs_yolo/pose_pre_seed42/eval/per_image.csv")}
    files = [f for f in te if "_850_" in f and all(f in iou[k] and f in P[k] for k in P)]
    rows = {f: {k: float(iou[k][f]["iou"]) for k in P} for f in files}
    imgdir = Path(r"../data/casia/images")
    if not imgdir.exists():
        imgdir = Path("../basline_A/images")
    idx = {p.stem: p for p in imgdir.iterdir()}
    for kind, f in pick(rows):
        g = cv2.imread(str(idx[Path(f).stem]), cv2.IMREAD_GRAYSCALE)
        sq = {"ref": lab(te[f]), **{k: lab(P[k][f]) for k in P}}
        items.append(pack("CASIA-MS", kind, f, g, sq))
    for ds, dsn in (("vera", "VERA"), ("scut", "SCUT-PV")):
        labs = json.load(open(f"out/cross/{ds}/labels_manual.json"))
        man = json.load(open(f"out/cross/{ds}/manifest.json"))
        iou = {k: read(f"out/cross/results/{ds}/per_image_{k}_seed42.csv") for k in SRC}
        rois = {k: json.load(open(f"out/xmatch/{ds}/{k}_seed42/rois.json")) for k in SRC}
        off = None
        if ds == "scut":
            z = np.load("out/xmatch/scut/official/crops.npz", allow_pickle=True)
            off = {str(n): c for n, c in zip(z["names"], z["crops"])}
        names = [n for n in labs if not labs[n].get("skip") and all(n in iou[k] for k in iou)]
        rows = {n: {k: float(iou[k][n]["iou"]) for k in iou} for n in names}
        for kind, n in pick(rows):
            stem_ = man[n]["source"].replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
            xn = f"{ds}_{stem_}"
            g = cv2.imread(f"out/cross/{ds}/images/{n}.png", cv2.IMREAD_GRAYSCALE)
            sq = {"ref": lab(labs[n])}
            for k in SRC:
                if xn in rois[k]:
                    sq[k] = tuple(rois[k][xn][:4])
            missing = [k for k in SRC if k not in sq]
            if missing:
                print("no square for", missing, "on", n)
                continue
            it = pack(dsn, kind, n, g, sq)
            if off is not None and xn in off:
                it["crops"]["official"] = off[xn]
            items.append(it)
    Path("out/paper").mkdir(parents=True, exist_ok=True)
    np.savez_compressed("out/paper/fig2_data.npz", items=np.array(items, dtype=object))
    print("-> out/paper/fig2_data.npz", len(items), "images")


if __name__ == "__main__":
    main()
