# run_pipeline_aug.ps1
# One batch for the reviewers' main request: the mask-aligned pipeline trained with the SAME strong
# augmentation as the augmented YOLOv8n-pose (3 stages x 3 seeds), evaluated exactly like the original
# pipeline, plus (optional) an epoch-matched control for the detector (light recipe, 100 epochs).
#
# Usage (from experiments):
#   powershell -ExecutionPolicy Bypass -File .\run_pipeline_aug.ps1                  # everything, in order
#   powershell -ExecutionPolicy Bypass -File .\run_pipeline_aug.ps1 -SkipLight100    # without the YOLO control
#   powershell -ExecutionPolicy Bypass -File .\run_pipeline_aug.ps1 -Step selfcheck  # one step only
# Steps: selfcheck | train | eval | light100 | bootstrap | pack | all
#
# The batch is restartable: every finished training stage writes DONE.json and is skipped next time,
# so after a crash or a reboot simply start the same command again.
# Progress is written to runs_pipeline_aug\batch_log.txt (one line per step with the time);
# every stage also keeps runs_pipeline_aug\stage<k>_seed<S>\log.csv (one line per epoch).

param([string]$Step = "all", [int[]]$Seeds = @(42, 2025, 1337), [switch]$SkipLight100)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$PY = "python"

$IMAGES    = "../data/casia/images"
$RAW_MASKS = "../data/casia/stage1_masks"
$OUT       = "out"
$ORIENT    = "$OUT\labels_oriented"
$RUNS      = "runs_pipeline_aug"
$LOG       = "$RUNS\batch_log.txt"
$R         = "runs\results_baselines_fp32.csv"                 # same files as the FP32 tables
$XR        = "$OUT\cross\results\results_cross_fp32.csv"
$SEG       = "$RUNS\stage1_seed{seed}\best.pt"
$ROT       = "$RUNS\stage2_seed{seed}\best.pt"
$ROI       = "$RUNS\stage3_seed{seed}\best.pt"
$SEEDLIST  = ($Seeds -join ",")

New-Item -ItemType Directory -Force -Path $RUNS | Out-Null

function Note($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $LOG -Value $line
}

function Run($argsList) {
    & $PY @argsList
    if ($LASTEXITCODE -ne 0) {
        Note "FAILED: $($argsList -join ' ')"
        throw "FAILED: $($argsList -join ' ')"
    }
}

function Do-Selfcheck {
    Note "selfcheck of the augmentation label rules (seed 42)"
    Run @("train_pipeline_aug.py", "selfcheck", "--seed", "42")
    Note "selfcheck passed -> look at $RUNS\selfcheck\*.jpg"
}

function Do-Train {
    # fast stages first, so that any problem shows up early: Stage III (~30 min/seed), Stage II, Stage I
    foreach ($k in @(3, 2, 1)) {
        foreach ($s in $Seeds) {
            if (Test-Path "$RUNS\stage${k}_seed$s\DONE.json") { Note "[skip] stage $k seed $s already done"; continue }
            Note "training stage $k seed $s"
            Run @("train_pipeline_aug.py", "stage$k", "--seed", "$s")
            Note "finished stage $k seed $s"
        }
    }
}

function Do-Eval {
    foreach ($s in $Seeds) {
        foreach ($k in @(1, 2, 3)) {
            if (-not (Test-Path "$RUNS\stage${k}_seed$s\DONE.json")) { throw "stage $k seed $s is not trained yet" }
        }
    }
    Note "evaluation 1/3: CASIA-MS in-domain (live pipeline with the re-trained stages)"
    Run @("full_pipeline_eval.py", "--images_dir", $IMAGES, "--gt_masks_dir", $RAW_MASKS, "--labels_root", $ORIENT,
          "--out_dir", $OUT, "--tag", "_aug", "--method_label", " +aug", "--results_csv", $R, "--seeds", $SEEDLIST,
          "--seg_ckpt_pattern", $SEG, "--rot_ckpt_pattern", $ROT, "--roi_ckpt_pattern", $ROI)
    Note "evaluation 2/3: VERA and SCUT-PV geometry (human reference squares)"
    Run @("xd_eval.py", "--cross_dir", "$OUT\cross", "--datasets", "vera,scut", "--seeds", $SEEDLIST,
          "--pipeline_only", "--pipeline_tag", "_aug", "--skip_casia_check", "--results_csv", $XR,
          "--seg_ckpt_pattern", $SEG, "--rot_ckpt_pattern", $ROT, "--roi_ckpt_pattern", $ROI)
    Note "evaluation 3/3: VERA and SCUT-PV recognition (CompCode, ResNet-18, vein matcher)"
    Run @("xd_match.py", "--datasets", "vera,scut", "--sources", "pipeline_aug", "--seeds", $SEEDLIST,
          "--cross_dir", "$OUT\cross", "--out_dir", "$OUT\xmatch",
          "--seg_aug_pattern", $SEG, "--rot_aug_pattern", $ROT, "--roi_aug_pattern", $ROI)
    Note "evaluation finished"
}

function Do-Light100 {
    # epoch-matched control: the light YOLOv8n-pose recipe trained for 100 epochs instead of 50
    foreach ($s in $Seeds) {
        if (Test-Path "runs_yolo\pose_l100_seed$s\weights\best.pt") { Note "[skip] YOLO light-100 seed $s already trained" }
        else {
            Note "training YOLOv8n-pose light recipe, 100 epochs, seed $s"
            Run @("baseline_yolo.py", "train", "--task", "pose", "--seed", "$s", "--out_root", "$OUT\yolo",
                  "--epochs", "100", "--batch", "16", "--aug", "none", "--tag", "_l100")
        }
        Note "evaluating YOLO light-100 seed $s on CASIA-MS"
        Run @("baseline_yolo.py", "eval", "--task", "pose", "--seed", "$s", "--out_root", "$OUT\yolo", "--tag", "_l100",
              "--images_dir", $IMAGES, "--labels_dir", "$ORIENT\seed$s", "--masks_dir", $RAW_MASKS, "--results_csv", $R)
    }
    Note "evaluating YOLO light-100 on VERA and SCUT-PV"
    Run @("xd_eval.py", "--cross_dir", "$OUT\cross", "--datasets", "vera,scut", "--seeds", $SEEDLIST,
          "--yolo_only", "--yolo_name", "YOLOv8-pose light 100ep", "--yolo_tag", "_l100", "--skip_casia_check",
          "--yolo_weights", "runs_yolo\pose_l100_seed{seed}\weights\best.pt", "--results_csv", $XR)
}

function Do-Bootstrap {
    Note "paired bootstrap (all sources, including the new ones)"
    Run @("bootstrap_all.py", "all")
}

function Do-Pack {
    # all result files, in one zip
    $dst = "$OUT\paper\pipeline_aug_results.zip"
    New-Item -ItemType Directory -Force -Path "$OUT\paper" | Out-Null
    $files = @()
    $files += Get-ChildItem "$RUNS" -Recurse -Include log.csv, DONE.json, selfcheck.json, *.jpg, batch_log.txt
    $files += Get-ChildItem "$OUT\full_pipeline_aug" -Recurse -Include per_image.csv, summary.json -ErrorAction SilentlyContinue
    $files += Get-ChildItem "$OUT\cross\results" -Recurse -Include "per_image_pipeline_aug_seed*.csv", "per_image_yolo_l100_seed*.csv", "results_cross_fp32.csv" -ErrorAction SilentlyContinue
    $files += Get-ChildItem "runs_yolo" -Recurse -Include per_image.csv -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like "*pose_l100*" }
    $files += Get-ChildItem "$OUT\xmatch" -Filter "results_xmatch.csv" -ErrorAction SilentlyContinue
    $files += Get-ChildItem "$OUT\bootstrap" -Filter "*.csv" -ErrorAction SilentlyContinue
    $files += Get-Item $R -ErrorAction SilentlyContinue
    $stage = "$OUT\paper\pipeline_aug_results"
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
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
    "train"     { Do-Train }
    "eval"      { Do-Eval }
    "light100"  { Do-Light100 }
    "bootstrap" { Do-Bootstrap }
    "pack"      { Do-Pack }
    "all"       {
        Note "=== batch started (seeds $SEEDLIST) ==="
        Do-Selfcheck
        Do-Train
        Do-Eval
        if (-not $SkipLight100) { Do-Light100 }
        Do-Bootstrap
        Do-Pack
        Note "=== batch finished ==="
    }
    default     { throw "Unknown step $Step" }
}
