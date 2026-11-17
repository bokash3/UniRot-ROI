# run_extra.ps1
# Two control experiments (evaluation code identical to the paper):
#  (1) fairness of initialisation: YOLOv8n-pose initialised from the official COCO-pretrained weights
#      (yolov8n-pose.pt) and trained with the augmented recipe -> CASIA-MS, VERA, SCUT-PV, recognition
#  (2) is it the alignment or only the augmentation? the raw-frame regressors of the 2x2 design (no alignment;
#      A = grey image, B = hand mask) re-trained with the same label-consistent augmentation as UniRot-ROI,
#      and both these and the original raw-frame cells evaluated on CASIA-MS, VERA and SCUT-PV
#
# Usage (from experiments):
#   powershell -ExecutionPolicy Bypass -File .\run_extra.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_extra.ps1 -Step raw     # only part (2)
# Steps: selfcheck | yolo | raw | match | pack | all
# Restartable: finished trainings are skipped. Progress: runs_raw_aug\extra_log.txt
# Expected time: about 8 hours on the RTX 3080 Ti Laptop (YOLO ~1.4 h per seed, raw cells ~0.5 h each).
# The first YOLO run downloads yolov8n-pose.pt (6 MB) from the Ultralytics release page.

param([string]$Step = "all", [int[]]$Seeds = @(42, 2025, 1337))

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$PY = "python"

$IMAGES    = "../data/casia/images"
$RAW_MASKS = "../data/casia/stage1_masks"
$OUT       = "out"
$ORIENT    = "$OUT\labels_oriented"
$R         = "runs\results_baselines_fp32.csv"
$XR        = "$OUT\cross\results\results_cross_fp32.csv"
$LOGD      = "runs_raw_aug"
$LOG       = "$LOGD\extra_log.txt"
$SEEDLIST  = ($Seeds -join ",")
New-Item -ItemType Directory -Force -Path $LOGD | Out-Null

function Note($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $LOG -Value $line
}
function Run($argsList) {
    & $PY @argsList
    if ($LASTEXITCODE -ne 0) { Note "FAILED: $($argsList -join ' ')"; throw "FAILED" }
}

function Do-Selfcheck {
    foreach ($c in @("A", "B")) {
        Note "selfcheck raw-frame cell $c"
        Run @("train_raw_aug.py", "selfcheck", "--cell", $c, "--seed", "42")
    }
}

function Do-Yolo {
    foreach ($s in $Seeds) {
        if (Test-Path "runs_yolo\pose_pre_seed$s\weights\best.pt") { Note "[skip] YOLO pretrained seed $s already trained" }
        else {
            Note "training YOLOv8n-pose from COCO weights, augmented recipe, 100 epochs, seed $s"
            Run @("baseline_yolo.py", "train", "--task", "pose", "--seed", "$s", "--out_root", "$OUT\yolo",
                  "--epochs", "100", "--batch", "16", "--aug", "strong", "--pretrained", "--tag", "_pre")
        }
        Note "evaluating YOLO pretrained seed $s on CASIA-MS"
        Run @("baseline_yolo.py", "eval", "--task", "pose", "--seed", "$s", "--out_root", "$OUT\yolo", "--tag", "_pre",
              "--images_dir", $IMAGES, "--labels_dir", "$ORIENT\seed$s", "--masks_dir", $RAW_MASKS, "--results_csv", $R)
    }
    Note "evaluating YOLO pretrained on VERA and SCUT-PV"
    Run @("xd_eval.py", "--cross_dir", "$OUT\cross", "--datasets", "vera,scut", "--seeds", $SEEDLIST,
          "--yolo_only", "--yolo_name", "YOLOv8-pose pretrained +aug", "--yolo_tag", "_pre", "--skip_casia_check",
          "--yolo_weights", "runs_yolo\pose_pre_seed{seed}\weights\best.pt", "--results_csv", $XR)
}

function Do-Raw {
    foreach ($c in @("A", "B")) {
        foreach ($s in $Seeds) {
            if (Test-Path "$LOGD\cell${c}_seed$s\DONE.json") { Note "[skip] raw-frame cell $c seed $s already trained" }
            else {
                Note "training raw-frame cell $c (augmented) seed $s"
                Run @("train_raw_aug.py", "train", "--cell", $c, "--seed", "$s")
            }
            Note "evaluating raw-frame cell $c (augmented) seed $s"
            Run @("train_raw_aug.py", "eval", "--cell", $c, "--seed", "$s", "--tag", "_aug")
            Note "evaluating raw-frame cell $c (original recipe) seed $s"
            $ck = if ($c -eq "A") { "runs\cellA_image_raw\seed{seed}\best.pth" } else { "runs\cellB_mask_raw\seed{seed}\best.pth" }
            Run @("train_raw_aug.py", "eval", "--cell", $c, "--seed", "$s", "--tag", "_orig", "--ckpt", $ck,
                  "--seg_ckpt_pattern", "../checkpoints/basic/stage1_seg_mobilenetv2_seed{seed}.pt")
        }
    }
}

function Do-Match {
    Note "recognition with the pretrained detector's ROIs (VERA, SCUT-PV)"
    Run @("xd_match.py", "--datasets", "vera,scut", "--sources", "yolo_pre", "--seeds", $SEEDLIST,
          "--cross_dir", "$OUT\cross", "--out_dir", "$OUT\xmatch")
    Note "paired bootstrap of recognition"
    Run @("bootstrap_all.py", "match")
}

function Do-Pack {
    $dst = "$OUT\paper\extra_results.zip"
    $stage = "$OUT\paper\extra_results"
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    $files = @()
    $files += Get-Item $R, $XR, $LOG, "runs\results_raw_aug.csv", "$OUT\xmatch\results_xmatch.csv" -ErrorAction SilentlyContinue
    $files += Get-ChildItem "$LOGD" -Recurse -Include log.csv, DONE.json, *.jpg, *_selfcheck.json -ErrorAction SilentlyContinue
    $files += Get-ChildItem "$OUT\raw_aug" -Recurse -Filter "*.csv" -ErrorAction SilentlyContinue
    $files += Get-ChildItem "$OUT\cross\results" -Recurse -Include "per_image_raw*.csv", "per_image_yolo_pre_seed*.csv" -ErrorAction SilentlyContinue
    $files += Get-ChildItem "runs_yolo" -Recurse -Include per_image.csv, results.csv -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like "*pose_pre*" }
    $files += Get-ChildItem "$OUT\bootstrap" -Filter "match_*.csv" -ErrorAction SilentlyContinue
    $files += Get-ChildItem "$OUT\xmatch" -Recurse -Filter "scores_miura_rnseed42.npz" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like "*yolo_pre*" }
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
    "selfcheck" { Do-Selfcheck }
    "yolo"      { Do-Yolo }
    "raw"       { Do-Raw }
    "match"     { Do-Match }
    "pack"      { Do-Pack }
    "all"       {
        Note "=== extra experiments started (seeds $SEEDLIST) ==="
        Do-Selfcheck
        Do-Raw
        Do-Yolo
        Do-Match
        Do-Pack
        Note "=== extra experiments finished ==="
    }
    default     { throw "Unknown step $Step" }
}
