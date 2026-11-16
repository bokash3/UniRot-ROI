#!/usr/bin/env python3
"""
xd_label.py — label the palm ROI square by hand: TWO clicks per image
=====================================================================

You see the hand upright (fingers up). For every image:
   click 1 : the CENTRE of the palm square
   click 2 : the MIDDLE OF THE TOP EDGE of the square   (this fixes both the size and the "up" direction)
The square is drawn live; the small window "crop" shows the upright 128x128 crop it would give.

Which square?  The standard palm-ROI: a square inside the palm, its top edge just BELOW the line joining the finger
bases (valleys), its bottom edge above the wrist, side about the palm width (the same square a CASIA ROI uses).
Be consistent rather than perfect: use the same rule for every image.

Fine adjust after the two clicks (the live square follows):
   i/k/j/l  move the square up/down/left/right by 1 px (hold Shift = 5 px)
   z/x      enlarge / shrink the WINDOW (display only)
   +/-      grow/shrink by 1%        ,/.   rotate by 1 degree (hold Shift = 5 deg)
Keys  [Enter]/[Space] accept   [r] redo clicks   [s] skip (bad image)   [b] back one image   [q] save and quit
Labels are saved after every image to  <ds_dir>/labels_manual.json  (resume = run again).
Output format = the CASIA label format in the CASIA frame (768x576, fingers left): cx, cy, size, theta.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from rx_common import load_json, save_json, square_corners
from rx_stages import oriented_crop

W0, H0 = 768, 576
WIN = "label (click centre, then top-edge middle)"


def to_label(M, c_up, m_up):
    M = np.asarray(M)
    cS = M @ np.array([c_up[0], c_up[1], 1.0])
    uS = M[:, :2] @ (np.asarray(m_up, float) - np.asarray(c_up, float))     # "up" vector in the CASIA frame
    side = 2.0 * float(np.hypot(*uS))
    phi = math.degrees(math.atan2(-uS[0], -uS[1])) % 360.0                   # up = (-sin phi, -cos phi)
    return {"cx": float(cS[0] / W0), "cy": float(cS[1] / H0), "size": side / min(W0, H0), "theta": phi}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ds_dir", required=True, help="e.g. out\\cross\\vera")
    p.add_argument("--zoom", type=float, default=1.0, help="display scale (upper bound; shrunk automatically to fit --max_h)")
    p.add_argument("--max_h", type=int, default=760, help="max window height in screen px; use e.g. 600 for a small laptop screen")
    p.add_argument("--names_file", default=None, help="JSON list of image names to label (default: all in the manifest)")
    p.add_argument("--shuffle_seed", type=int, default=None, help="label the images in a random order fixed by this seed")
    p.add_argument("--labels_name", default="labels_manual.json",
                   help="output file; use labels_manual_pass2.json for a second labelling pass (repeatability test)")
    a = p.parse_args()
    d = Path(a.ds_dir)
    man = load_json(d / "manifest.json")
    lab_path = d / a.labels_name
    labels = load_json(lab_path) if lab_path.is_file() else {}
    names = sorted(man)
    if a.names_file:
        keep = load_json(Path(a.names_file))
        names = [n for n in keep if n in man]
    if a.shuffle_seed is not None:
        import random as _r
        _r.Random(a.shuffle_seed).shuffle(names)
    zoom = [a.zoom]
    i = 0
    st = {"pts": []}

    def on_mouse(ev, x, y, fl, par):
        if ev == cv2.EVENT_LBUTTONDOWN and len(st["pts"]) < 2:
            st["pts"].append((x / zoom[0], y / zoom[0]))
        if ev == cv2.EVENT_MOUSEMOVE:
            st["mouse"] = (x / zoom[0], y / zoom[0])

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    while 0 <= i < len(names):
        n = names[i]
        if n in labels:
            i += 1
            continue
        st = {"pts": []}
        up = cv2.imread(str(d / "upright" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(d / "images" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        zoom[0] = min(a.zoom, a.max_h / up.shape[0])
        while True:
            vis = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
            vis = cv2.resize(vis, None, fx=zoom[0], fy=zoom[0])
            pts = list(st["pts"])
            if len(pts) == 1 and "mouse" in st:
                pts_prev = pts + [st["mouse"]]
            else:
                pts_prev = pts
            lbl = None
            if len(pts_prev) == 2:
                c, m = np.array(pts_prev[0]), np.array(pts_prev[1])
                u = m - c
                side = 2 * np.hypot(*u)
                # square in the upright view: rotate "up" vector to (0,-1)
                ang = math.degrees(math.atan2(-u[0], -u[1]))
                poly = square_corners(c[0], c[1], side, ang) * zoom[0]
                cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.arrowedLine(vis, tuple((c * zoom[0]).astype(int)), tuple((m * zoom[0]).astype(int)), (0, 255, 255), 2)
                lbl = to_label(man[n]["M"], pts_prev[0], pts_prev[1])
                cx, cy, s, th = lbl["cx"] * W0, lbl["cy"] * H0, lbl["size"] * min(W0, H0), lbl["theta"]
                crop = oriented_crop(img, cx, cy, s, th)
                cv2.imshow("crop", cv2.resize(crop, (256, 256), interpolation=cv2.INTER_NEAREST))
            cv2.putText(vis, f"{i + 1}/{len(names)} {n}  labeled={len(labels)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.imshow(WIN, vis)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                save_json(labels, lab_path)
                print(f"saved {len(labels)} labels -> {lab_path}")
                return
            if k == ord("z") or k == ord("x"):               # change display size live; points stay valid
                zoom[0] *= 1.1 if k == ord("z") else 1 / 1.1
            if k == ord("r"):
                st["pts"] = []
            if len(st["pts"]) == 2 and k != 255:
                (cx0, cy0), (mx0, my0) = st["pts"]
                step = 5.0 if chr(k).isupper() else 1.0
                ch = chr(k).lower() if k < 128 else ""
                dx = {"j": -step, "l": step}.get(ch, 0.0)
                dy = {"i": -step, "k": step}.get(ch, 0.0)
                cx0, cy0, mx0, my0 = cx0 + dx, cy0 + dy, mx0 + dx, my0 + dy
                if k in (ord("+"), ord("="), ord("-"), ord("_")):
                    f = 1.01 if k in (ord("+"), ord("=")) else 0.99
                    mx0, my0 = cx0 + (mx0 - cx0) * f, cy0 + (my0 - cy0) * f
                if k in (ord(","), ord("."), ord("<"), ord(">")):
                    da = math.radians((5.0 if k in (ord("<"), ord(">")) else 1.0) * (1 if k in (ord("."), ord(">")) else -1))
                    vx, vy = mx0 - cx0, my0 - cy0
                    mx0, my0 = cx0 + vx * math.cos(da) - vy * math.sin(da), cy0 + vx * math.sin(da) + vy * math.cos(da)
                st["pts"] = [(cx0, cy0), (mx0, my0)]
            if k == ord("s"):
                labels[n] = {"skip": True}
                break
            if k == ord("b"):
                j = max(0, i - 1)
                labels.pop(names[j], None)
                i = j - 1                      # the loop adds 1 below
                break
            if k in (13, 32) and len(st["pts"]) == 2:
                labels[n] = to_label(man[n]["M"], st["pts"][0], st["pts"][1])
                break
        save_json(labels, lab_path)
        i += 1
    save_json(labels, lab_path)
    print(f"done: {len(labels)} labels -> {lab_path}")


if __name__ == "__main__":
    main()
