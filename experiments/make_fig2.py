"""Fig. 2 — qualitative comparison (runs where the images are: needs numpy + opencv only).
Rows: CASIA-MS test (seed 42), VERA, SCUT-PV. For each dataset two hands chosen by a fixed rule:
  typical = image whose YOLO+aug IoU is the median;  hard = image where the methods disagree most
  (largest IoU spread across pipeline / YOLO / YOLO+aug).
Each panel: the hand (fingers up) with the reference square (white) and the predictions, then the crops."""
import csv, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, ".")
from rx_common import label_to_px, square_corners, polygon_iou
from rx_stages import oriented_crop

W0, H0 = 768, 576
COL = {"ref": (255, 255, 255), "pipeline": (214, 120, 42), "yolo": (52, 104, 235), "yolo_aug": (122, 175, 27),
       "official": (0, 161, 237)}
NAME = {"ref": "Reference", "pipeline": "Pipeline", "yolo": "YOLO", "yolo_aug": "YOLO+aug", "official": "SCUT ROI"}


def read(p):
    return {r["file"]: r for r in csv.DictReader(open(p))}


def lab(d):
    return label_to_px(d, W0, H0)


def draw(img, sq, col, t=3):
    cv2.polylines(img, [np.round(square_corners(*sq)).astype(np.int32)], True, col, t, cv2.LINE_AA)
    c = square_corners(*sq)
    top = ((c[0] + c[1]) / 2).astype(int)                            # mark the "up" edge
    cv2.circle(img, tuple(top), 6, col, -1, cv2.LINE_AA)


def panel(gray, sqs, crops_extra=None):
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for k in ("ref", "pipeline", "yolo", "yolo_aug"):
        if k in sqs:
            draw(bgr, sqs[k], COL[k], 4 if k == "ref" else 3)
    up = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)                       # fingers up for display
    up = cv2.resize(up, (int(up.shape[1] * 300 / up.shape[0]), 300))
    tiles = []
    for k in ("ref", "pipeline", "yolo", "yolo_aug"):
        if k in sqs:
            c = cv2.cvtColor(cv2.resize(oriented_crop(gray, *sqs[k], out=128), (140, 140)), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(c, (0, 0), (139, 139), COL[k], 5)
            tiles.append(c)
    if crops_extra is not None:
        c = cv2.cvtColor(cv2.resize(crops_extra, (140, 140)), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(c, (0, 0), (139, 139), COL["official"], 5)
        tiles.append(c)
    while len(tiles) < 5:
        tiles.append(np.full((140, 140, 3), 255, np.uint8))
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:5] + [np.full((140, 140, 3), 255, np.uint8)])])
    grid = cv2.resize(grid, (int(grid.shape[1] * 300 / grid.shape[0]), 300))
    return np.hstack([up, np.full((300, 8, 3), 255, np.uint8), grid])


def pick(rows):
    """rows: name -> {src: iou}. typical = median YOLO+aug; hard = max spread."""
    names = sorted(rows)
    ya = np.array([rows[n]["yolo_aug"] for n in names])
    typ = names[int(np.argsort(ya)[len(ya) // 2])]
    spread = np.array([max(rows[n].values()) - min(rows[n].values()) for n in names])
    hard = names[int(np.argmax(spread))]
    return typ, hard


def main():
    out_rows = []
    # ---------------- CASIA (seed 42 test)
    te = json.load(open("out/labels_oriented/seed42/test_labels.json"))
    P = {"pipeline": json.load(open("out/full_pipeline/seed42/pred_labels_full.json")),
         "yolo": json.load(open("runs_yolo/pose_seed42/eval/pred_labels.json")),
         "yolo_aug": json.load(open("runs_yolo/pose_aug_seed42/eval/pred_labels.json"))}
    iou = {"pipeline": read("out/full_pipeline/seed42/per_image.csv"),
           "yolo": read("runs_yolo/pose_seed42/eval/per_image.csv"),
           "yolo_aug": read("runs_yolo/pose_aug_seed42/eval/per_image.csv")}
    files = [f for f in te if "_850_" in f and all(f in iou[k] and f in P[k] for k in P)]
    rows = {f: {k: float(iou[k][f]["iou"]) for k in P} for f in files}
    imgdir = Path(r"../data/casia/images") if Path(r"../data/casia/images").exists() \
        else Path("../basline_A/images")
    idx = {p.stem: p for p in imgdir.iterdir()}
    for f in pick(rows):
        g = cv2.imread(str(idx[Path(f).stem]), cv2.IMREAD_GRAYSCALE)
        sq = {"ref": lab(te[f]), **{k: lab(P[k][f]) for k in P}}
        r = {k: polygon_iou(square_corners(*sq["ref"]), square_corners(*sq[k])) for k in P}   # IoU of the drawn squares
        out_rows.append(("CASIA-MS", f, r, panel(g, sq)))
    # ---------------- VERA / SCUT
    for ds in ("vera", "scut"):
        labs = json.load(open(f"out/cross/{ds}/labels_manual.json"))
        man = json.load(open(f"out/cross/{ds}/manifest.json"))
        iou = {k: read(f"out/cross/results/{ds}/per_image_{k}_seed42.csv") for k in ("pipeline", "yolo", "yolo_aug")}
        rois = {k: json.load(open(f"out/xmatch/{ds}/{k}_seed42/rois.json")) for k in ("pipeline", "yolo", "yolo_aug")}
        names = [n for n in labs if not labs[n].get("skip") and all(n in iou[k] for k in iou)]
        rows = {n: {k: float(iou[k][n]["iou"]) for k in iou} for n in names}
        off = None
        if ds == "scut":
            z = np.load(f"out/xmatch/scut/official/crops.npz", allow_pickle=True)
            off = {str(n): c for n, c in zip(z["names"], z["crops"])}
        for n in pick(rows):
            stem = man[n]["source"].replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]   # Windows path
            xn = f"{ds}_{stem}"
            g = cv2.imread(f"out/cross/{ds}/images/{n}.png", cv2.IMREAD_GRAYSCALE)
            sq = {"ref": lab(labs[n]), **{k: tuple(rois[k][xn][:4]) for k in rois}}
            r = {k: polygon_iou(square_corners(*sq["ref"]), square_corners(*sq[k])) for k in rois}
            out_rows.append(({"vera": "VERA", "scut": "SCUT-PV"}[ds], n, r,
                             panel(g, sq, off[xn] if off is not None else None)))
    # ---------------- assemble: 3 rows x 2 panels
    tiles = []
    for ds_name, n, r, im in out_rows:
        head = np.full((40, im.shape[1], 3), 255, np.uint8)
        txt = f"{ds_name}  IoU: pipeline {r['pipeline']:.2f} | YOLO {r['yolo']:.2f} | YOLO+aug {r['yolo_aug']:.2f}"
        cv2.putText(head, txt, (6, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
        tiles.append(np.vstack([head, im]))
        print(ds_name, n, r)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 10, 0, w - t.shape[1] + 12, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    rows_img = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    leg = np.full((46, rows_img[0].shape[1], 3), 255, np.uint8)
    x = 8
    for k in ("ref", "pipeline", "yolo", "yolo_aug", "official"):
        cv2.rectangle(leg, (x, 12), (x + 26, 34), COL[k] if k != "ref" else (60, 60, 60), 3 if k == "ref" else -1)
        cv2.putText(leg, NAME[k] + (" (white on image)" if k == "ref" else ""), (x + 34, 31), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (30, 30, 30), 1, cv2.LINE_AA)
        x += 34 + cv2.getTextSize(NAME[k] + (" (white on image)" if k == "ref" else ""), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] + 30
    fig = np.vstack([leg] + rows_img)
    Path("out/paper").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("out/paper/Fig2_qualitative.png", fig)
    print("-> out/paper/Fig2_qualitative.png", fig.shape)


if __name__ == "__main__":
    main()
