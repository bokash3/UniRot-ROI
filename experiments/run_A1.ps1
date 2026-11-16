# run_A1.ps1  -  larger detectors: is the gain due to design or to model size / initialisation?
# Larger end-to-end detectors, trained exactly like YOLO-matched-PT (run_extra.ps1, step yolo):
#   YOLOv8s-pose and YOLOv8m-pose, official COCO-pretrained weights, matched ("strong") augmentation,
#   100 epochs, batch 16, seeds 42 / 2025 / 1337, same splits, labels, evaluation code and FP32 inference.
# Outputs (new names only; nothing published is overwritten):
#   runs_yolo\pose_s_pre_seed<S>\, runs_yolo\pose_m_pre_seed<S>\           weights + CASIA-MS eval
#   out\cross\results\<ds>\per_image_yolo_s_pre_seed<S>.csv  (and _m_pre)  VERA / SCUT-PV geometry
#   out\xmatch\<ds>\yolo_s_pre_seed<S>\  (and yolo_m_pre)                   recognition scores
#   out\bootstrap_A1\*.csv                                                  paired bootstrap (separate folder)
#   out\latency_A1_<size>.json                                              batch-1 latency
#   out\paper\A1_results.zip                                                everything above, in one zip
#
# Usage (from experiments, Anaconda Prompt or PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\run_A1.ps1 -Step smoke     # 1-epoch test of both sizes (~10 min)
#   powershell -ExecutionPolicy Bypass -File .\run_A1.ps1                 # everything (restartable)
#   powershell -ExecutionPolicy Bypass -File .\run_A1.ps1 -Step train -Sizes s     # one size only
# Steps: smoke | train | casia | cross | match | boot | time | pack | all
# Restartable: a seed whose best.pt exists is not retrained. Progress log: runs_yolo\A1_log.txt
# Expected time on the RTX 3080 Ti Laptop: s about 2 h and m about 3-4 h per seed (training), plus about 2 h of evaluation.
# The first run downloads yolov8s-pose.pt (23 MB) and yolov8m-pose.pt (51 MB) from the Ultralytics release page.

param([string]$Step = "all", [string[]]$Sizes = @("s", "m"), [int[]]$Seeds = @(42, 2025, 1337))

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$PY = "python"

$IMAGES    = "..\data\casia\images"
$RAW_MASKS = "..\data\casia\stage1_masks"
$OUT       = "out"
$ORIENT    = "$OUT\labels_oriented"
$R         = "runs\results_A1_fp32.csv"
$XR        = "$OUT\cross\results\results_cross_A1_fp32.csv"
$LOG       = "runs_yolo\A1_log.txt"
$SEEDLIST  = ($Seeds -join ",")
New-Item -ItemType Directory -Force -Path "runs_yolo", "runs" | Out-Null

function Note($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $LOG -Value $line
}
function Run($argsList) {
    & $PY @argsList
    if ($LASTEXITCODE -ne 0) { Note "FAILED: $($argsList -join ' ')"; throw "FAILED" }
}

function Do-Smoke {
    foreach ($z in $Sizes) {
        Note "smoke test: YOLOv8$z-pose from COCO, strong augmentation, 1 epoch, seed 42"
        Run @("baseline_yolo.py", "train", "--task", "pose", "--seed", "42", "--out_root", "$OUT\yolo",
              "--epochs", "1", "--batch", "16", "--aug", "strong", "--pretrained", "--size", $z, "--tag", "_${z}_smoke")
    }
    & $PY -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('ultralytics', ultralytics.__version__)"
    Note "smoke test finished; if both sizes trained one epoch without error, run the full script"
}

function Do-Train {
    foreach ($z in $Sizes) { foreach ($s in $Seeds) {
        if (Test-Path "runs_yolo\pose_${z}_pre_seed$s\weights\best.pt") { Note "[skip] YOLOv8$z-pose seed $s already trained" }
        else {
            Note "training YOLOv8$z-pose from COCO weights, strong augmentation, 100 epochs, seed $s"
            Run @("baseline_yolo.py", "train", "--task", "pose", "--seed", "$s", "--out_root", "$OUT\yolo",
                  "--epochs", "100", "--batch", "16", "--aug", "strong", "--pretrained", "--size", $z, "--tag", "_${z}_pre")
            Note "finished YOLOv8$z-pose seed $s"
        }
    } }
}

function Do-Casia {
    foreach ($z in $Sizes) { foreach ($s in $Seeds) {
        Note "CASIA-MS evaluation YOLOv8$z-pose seed $s"
        Run @("baseline_yolo.py", "eval", "--task", "pose", "--seed", "$s", "--out_root", "$OUT\yolo", "--tag", "_${z}_pre",
              "--images_dir", $IMAGES, "--labels_dir", "$ORIENT\seed$s", "--masks_dir", $RAW_MASKS, "--results_csv", $R)
    } }
}

function Do-Cross {
    foreach ($z in $Sizes) {
        Note "VERA and SCUT-PV evaluation YOLOv8$z-pose"
        Run @("xd_eval.py", "--cross_dir", "$OUT\cross", "--datasets", "vera,scut", "--seeds", $SEEDLIST,
              "--yolo_only", "--yolo_name", "YOLOv8$z-pose pretrained +aug", "--yolo_tag", "_${z}_pre", "--skip_casia_check",
              "--yolo_weights", "runs_yolo\pose_${z}_pre_seed{seed}\weights\best.pt", "--results_csv", $XR)
    }
}

function Do-Match {
    $srcs = ($Sizes | ForEach-Object { "yolo_${_}_pre" }) -join ","
    Note "recognition with the ROIs of $srcs (VERA, SCUT-PV); MC sigma from out\xmatch\mc_calibration"
    Run @("xd_match.py", "--datasets", "vera,scut", "--sources", $srcs, "--seeds", $SEEDLIST,
          "--vera_root", "..\datasets\VERA-Palmvein\raw", "--scut_root", "..\datasets\SCUT_PV_V1",
          "--cross_dir", "$OUT\cross", "--out_dir", "$OUT\xmatch")
}

function Do-Boot {
    Note "paired bootstrap including the new detectors -> out\bootstrap_A1 (the published out\bootstrap is not touched)"
    Run @("bootstrap_all.py", "all", "--out_dir", "$OUT\bootstrap_A1")
}

function Do-Time {
    foreach ($z in $Sizes) {
        Note "batch-1 latency YOLOv8$z-pose (seed 42 weights, 850 nm CASIA-MS test images, 20 warm-up)"
        Run @("time_localisers.py", "--seed", "42", "--n", "180", "--warmup", "20",
              "--yolo_weights", "runs_yolo\pose_${z}_pre_seed{seed}\weights\best.pt", "--out", "$OUT\latency_A1_$z.json")
    }
}

function Do-Pack {
    $dst = "$OUT\paper\A1_results.zip"
    $stage = "$OUT\paper\A1_results"
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    $files = @()
    $files += Get-Item $R, $XR, $LOG, "$OUT\xmatch\results_xmatch.csv", "$OUT\latency_A1_*.json" -ErrorAction SilentlyContinue
    $files += Get-ChildItem "$OUT\cross\results" -Recurse -Include "per_image_yolo_*_pre_seed*.csv" -ErrorAction SilentlyContinue
    $files += Get-ChildItem "runs_yolo" -Recurse -Include per_image.csv, summary.json, results.csv, args.yaml -ErrorAction SilentlyContinue |
              Where-Object { $_.FullName -match "pose_[sm]_pre_seed" }
    $files += Get-ChildItem "$OUT\bootstrap_A1" -Filter "*.csv" -ErrorAction SilentlyContinue
    $files += Get-ChildItem "$OUT\xmatch" -Recurse -Filter "scores_*.npz" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match "yolo_[sm]_pre" }
    foreach ($f in ($files | Select-Object -ExpandProperty FullName -Unique)) {
        $rel = (Resolve-Path -LiteralPath $f -Relative).TrimStart(".", "\")
        $target = Join-Path $stage $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
        Copy-Item -LiteralPath $f -Destination $target
    }
    if (Test-Path $dst) { Remove-Item $dst }
    Compress-Archive -Path "$stage\*" -DestinationPath $dst
    Note "results packed -> $dst"
}

switch ($Step) {
    "smoke" { Do-Smoke }
    "train" { Do-Train }
    "casia" { Do-Casia }
    "cross" { Do-Cross }
    "match" { Do-Match }
    "boot"  { Do-Boot }
    "time"  { Do-Time }
    "pack"  { Do-Pack }
    "all"   {
        Note "=== A1 started (sizes $($Sizes -join ','), seeds $SEEDLIST) ==="
        Do-Train; Do-Casia; Do-Cross; Do-Match; Do-Boot; Do-Time; Do-Pack
        Note "=== A1 finished ==="
    }
    default { throw "Unknown step $Step" }
}
