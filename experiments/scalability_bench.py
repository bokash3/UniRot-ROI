#!/usr/bin/env python3
"""
scalability_bench.py - how the cost of each localiser grows with batch size (GPU) and what it costs on a CPU.

(1) GPU, network forward passes only, FP32 with TF32 disabled (same as the paper):
      Stage I   U-Net / MobileNetV2      input B x 3 x 576 x 768
      Stage II  UniRotNet / ResNet-18    input B x 1 x 576 x 768
      Stage III PalmROINet / MobileNetV2 input B x 1 x 256 x 256
      YOLOv8n-pose (fused)               input B x 3 x 480 x 640  (the letter-boxed 768 x 576 frame at imgsz 640)
    for B in --batches: mean ms per batch, images per second and peak allocated GPU memory.
    Pre- and post-processing (resize, mask rotation, NMS, back-projection) are NOT included here;
    the batch-1 end-to-end latency is in out/latency.json (time_localisers.py).
(2) CPU, batch 1, end to end with the same harness as time_localisers.py (--cpu_n images).

Run AFTER other GPU jobs have finished:
    python scalability_bench.py --seed 42
Writes out/scalability.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

PIPE = Namespace(pipeline_dir=r"../pipeline_models",
                 seg_ckpt_pattern=r"../checkpoints/basic/stage1_seg_mobilenetv2_seed{seed}.pt",
                 rot_ckpt_pattern=r"../checkpoints/basic/stage2_unirot_angle_seed{seed}.pt",
                 roi_ckpt_pattern=r"../checkpoints/basic/stage3_roi_seed{seed}.pth",
                 random_weights=False)


def bench(fn, x, reps, warm):
    with torch.no_grad():
        for _ in range(warm):
            fn(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = []
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(x)
            torch.cuda.synchronize(); t.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(t)), float(torch.cuda.max_memory_allocated() / 2 ** 20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batches", default="1,2,4,8,16,32")
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--yolo_weights", default=r"runs_yolo\pose_aug_seed{seed}\weights\best.pt")
    p.add_argument("--cpu_n", type=int, default=50)
    p.add_argument("--skip_cpu", action="store_true")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")

    import full_pipeline_eval as fp
    (seg, rot, roi), _ = fp.build_models(PIPE, a.seed, dev)
    from ultralytics import YOLO
    yolo = YOLO(a.yolo_weights.format(seed=a.seed)).model.to(dev).float().eval()
    yolo = yolo.fuse() if hasattr(yolo, "fuse") else yolo

    nets = {"stage1": (seg, (3, 576, 768)), "stage2": (rot, (1, 576, 768)),
            "stage3": (roi, (1, 256, 256)), "yolo": (yolo, (3, 480, 640))}
    res = {"device": torch.cuda.get_device_name(0), "gpu": {}}
    for B in [int(b) for b in a.batches.split(",")]:
        row = {}
        for k, (net, shp) in nets.items():
            try:
                x = torch.rand((B, *shp), device=dev)
                ms, mem = bench(net, x, a.reps, a.warmup)
                row[k] = {"ms_per_batch": ms, "img_per_s": 1000 * B / ms, "peak_mem_mib": mem}
            except torch.cuda.OutOfMemoryError:
                row[k] = {"oom": True}
                torch.cuda.empty_cache()
        if all("ms_per_batch" in row[s] for s in ("stage1", "stage2", "stage3")):
            ms = sum(row[s]["ms_per_batch"] for s in ("stage1", "stage2", "stage3"))
            row["pipeline_networks"] = {"ms_per_batch": ms, "img_per_s": 1000 * B / ms,
                                        "peak_mem_mib": max(row[s]["peak_mem_mib"] for s in ("stage1", "stage2", "stage3"))}
        res["gpu"][B] = row
        print(f"B={B:3d}  " + "  ".join(f"{k}: {v['ms_per_batch']:7.2f} ms {v['img_per_s']:7.1f} img/s {v['peak_mem_mib']:6.0f} MiB"
                                         if "ms_per_batch" in v else f"{k}: OOM" for k, v in row.items()))
    del seg, rot, roi, yolo
    torch.cuda.empty_cache()

    if not a.skip_cpu:   # end-to-end, batch 1, CPU, same harness as time_localisers.py
        import subprocess, sys
        cmd = [sys.executable, "time_localisers.py", "--seed", str(a.seed), "--n", str(a.cpu_n), "--warmup", "5",
               "--device", "cpu", "--out", "out/latency_cpu.json"]
        print("[CPU]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        res["cpu_end_to_end"] = json.load(open("out/latency_cpu.json"))
        import platform
        res["cpu_name"] = platform.processor()
        res["cpu_threads"] = torch.get_num_threads()

    Path("out").mkdir(exist_ok=True)
    json.dump(res, open("out/scalability.json", "w"), indent=1)
    print("-> out/scalability.json")


if __name__ == "__main__":
    main()
