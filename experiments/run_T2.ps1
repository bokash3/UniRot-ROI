# run_T2.ps1  -  detector training seeds: nine runs of YOLO-matched and YOLO-matched-PT.
# The three subject-disjoint CASIA-MS splits (42 / 2025 / 1337) are kept,
# and each detector is retrained on the SAME split with new TRAINING seeds (initialisation, data order, augmentation).
# UniRot-ROI and all published runs are untouched, so every new run stays paired with UniRot-ROI on its split.
#   YOLO-matched    = YOLOv8n-pose, random init, matched ("strong") augmentation, 100 epochs, batch 16
#   YOLO-matched-PT = the same from the COCO-pretrained yolov8n-pose.pt
# The existing runs (training seed = split seed) are run 1; this script adds training seeds 101 and 202 (runs 2 and 3),
# giving 9 runs per detector. Run 2 is done for all splits and both detectors first, so stopping after it gives 6 per detector.
# Outputs (new names only; nothing published is overwritten):
#   runs_yolo\pose_aug_t<T>_seed<S>\, runs_yolo\pose_pre_t<T>_seed<S>\      weights + CASIA-MS eval
#   out\cross\results\<ds>\per_image_yolo_aug_t<T>_seed<S>.csv (and _pre)   VERA / SCUT-PV geometry
#   out\xmatch\<ds>\yolo_aug_t<T>_seed<S>\ (and yolo_pre)                    recognition scores
#   out\paper\T2_results.zip                                                 everything above, in one zip
#
# Usage (from experiments):
#   $env:NO_ALBUMENTATIONS_UPDATE = "1"
#   powershell -ExecutionPolicy Bypass -File .\run_T2.ps1 -Step smoke     # 1 epoch with a training seed (~3 min)
#   powershell -ExecutionPolicy Bypass -File .\run_T2.ps1                 # everything (restartable)
#   powershell -ExecutionPolicy Bypass -File .\run_T2.ps1 -TrainSeeds 101 # run 2 only (6 runs per detector)
# Steps: smoke | train | casia | cross | match | pack | all.  Progress log: runs_yolo\T2_log.txt

param([string]$Step = "all", [int[]]$TrainSeeds = @(101, 202), [int[]]$Seeds = @(42, 2025, 1337),
      [string[]]$Dets = @("aug", "pre"))

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:NO_ALBUMENTATIONS_UPDATE = "1"
$PY = "python"

$IMAGES    = "..\data\casia\images"
$RAW_MASKS = "..\data\casia\stage1_masks"
$OUT       = "out"
$ORIENT    = "$OUT\labels_oriented"
$R         = "runs\results_T2_fp32.csv"
$XR        = "$OUT\cross\results\results_cross_T2_fp32.csv"
$LOG       = "runs_yolo\T2_log.txt"
$SEEDLIST  = ($Seeds -join ",")
$NAME      = @{ "aug" = "YOLO-matched"; "pre" = "YOLO-matched-PT" }
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
function Done($d, $t, $s) {
    # finished = best.pt AND 100 epochs in results.csv (best.pt already exists after epoch 1 of an interrupted run)
    $dir = "runs_yolo\pose_${d}_t${t}_seed$s"
    if (-not (Test-Path "$dir\weights\best.pt") -or -not (Test-Path "$dir\results.csv")) { return $false }
    return ((Get-Content "$dir\results.csv").Count -ge 101)
}

function TrainArgs($d) {
    $x = @("--aug", "strong", "--size", "n")
    if ($d -eq "pre") { $x += "--pretrained" }
    return $x
}

function Do-Smoke {
    Note "smoke test: YOLO-matched, split 42, training seed 101, 1 epoch"
    Run (@("baseline_yolo.py", "train", "--task", "pose", "--seed", "42", "--train_seed", "101", "--out_root", "$OUT\yolo",
           "--epochs", "1", "--batch", "16", "--tag", "_aug_tsmoke") + (TrainArgs "aug"))
    $y = Get-Content "runs_yolo\pose_aug_tsmoke_seed42\args.yaml" | Select-String -Pattern "^(seed|model|pretrained|degrees):"
    $y | ForEach-Object { Write-Host $_ }
    Note "smoke test finished; args.yaml must show seed: 101 and model: yolov8n-pose.yaml"
}

function Do-Train {
    foreach ($t in $TrainSeeds) { foreach ($d in $Dets) { foreach ($s in $Seeds) {
        $dir = "runs_yolo\pose_${d}_t${t}_seed$s"
        if (Done $d $t $s) { Note "[skip] $($NAME[$d]) split $s training seed $t already trained" }
        else {
            Note "training $($NAME[$d]) split $s training seed $t (100 epochs)"
            Run (@("baseline_yolo.py", "train", "--task", "pose", "--seed", "$s", "--train_seed", "$t", "--out_root", "$OUT\yolo",
                   "--epochs", "100", "--batch", "16", "--tag", "_${d}_t$t") + (TrainArgs $d))
            Note "finished $($NAME[$d]) split $s training seed $t"
        }
    } } }
}



function Do-Casia {
    foreach ($t in $TrainSeeds) { foreach ($d in $Dets) { foreach ($s in $Seeds) {
        if (-not (Done $d $t $s)) { Note "[skip] CASIA-MS: no weights for $d t$t split $s"; continue }
        Note "CASIA-MS evaluation $($NAME[$d]) split $s training seed $t"
        Run @("baseline_yolo.py", "eval", "--task", "pose", "--seed", "$s", "--out_root", "$OUT\yolo", "--tag", "_${d}_t$t",
              "--images_dir", $IMAGES, "--labels_dir", "$ORIENT\seed$s", "--masks_dir", $RAW_MASKS, "--results_csv", $R)
    } } }
}

function Do-Cross {
    foreach ($t in $TrainSeeds) { foreach ($d in $Dets) {
        $ok = @($Seeds | Where-Object { Done $d $t $_ })
        if ($ok.Count -eq 0) { continue }
        Note "VERA and SCUT-PV evaluation $($NAME[$d]) training seed $t (splits $($ok -join ','))"
        Run @("xd_eval.py", "--cross_dir", "$OUT\cross", "--datasets", "vera,scut", "--seeds", ($ok -join ","),
              "--yolo_only", "--yolo_name", "$($NAME[$d]) train-seed $t", "--yolo_tag", "_${d}_t$t", "--skip_casia_check",
              "--yolo_weights", "runs_yolo\pose_${d}_t${t}_seed{seed}\weights\best.pt", "--results_csv", $XR)
    } }
}

function Do-Match {
    foreach ($t in $TrainSeeds) { foreach ($d in $Dets) {
        $ok = @($Seeds | Where-Object { Done $d $t $_ })
        if ($ok.Count -eq 0) { continue }
        Note "recognition with the ROIs of $($NAME[$d]) training seed $t (splits $($ok -join ','))"
        Run @("xd_match.py", "--datasets", "vera,scut", "--sources", "yolo_${d}_t$t", "--seeds", ($ok -join ","),
              "--vera_root", "..\datasets\VERA-Palmvein\raw", "--scut_root", "..\datasets\SCUT_PV_V1",
              "--cross_dir", "$OUT\cross", "--out_dir", "$OUT\xmatch")
    } }
}

function Do-Pack {
    $dst = "$OUT\paper\T2_results.zip"
    $stage = "$OUT\paper\T2_results"
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    $files = @()
    $files += Get-Item $R, $XR, $LOG, "$OUT\xmatch\results_xmatch.csv" -ErrorAction SilentlyContinue
    $files += Get-ChildItem "$OUT\cross\results" -Recurse -Include "per_image_yolo_*_t*_seed*.csv" -ErrorAction SilentlyContinue
    $files += Get-ChildItem "runs_yolo" -Recurse -Include per_image.csv, summary.json, results.csv, args.yaml -ErrorAction SilentlyContinue |
              Where-Object { $_.FullName -match "pose_(aug|pre)_t\d+_seed" }
    $files += Get-ChildItem "$OUT\xmatch" -Recurse -Filter "scores_*.npz" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match "yolo_(aug|pre)_t\d+_seed" }
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
    "pack"  { Do-Pack }
    "all"   {
        Note "=== T2 started (detectors $($Dets -join ','), training seeds $($TrainSeeds -join ','), splits $SEEDLIST) ==="
        Do-Train; Do-Casia; Do-Cross; Do-Match; Do-Pack
        Note "=== T2 finished ==="
    }
    default { throw "Unknown step $Step" }
}
