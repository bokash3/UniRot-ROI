# run_stage_swap.ps1
# Which stage makes the difference? Evaluates the mask-aligned pipeline with every mix of
# original (o) and augmentation-trained (a) stages: code = Stage I, Stage II, Stage III.
# 'ooo' (original) and 'aaa' (augmented) are already evaluated; this runs the other six.
# Evaluation only - no training. Expected time: about 1 hour on the RTX 3080 Ti Laptop.
#
# Usage (from experiments):
#   powershell -ExecutionPolicy Bypass -File .\run_stage_swap.ps1
# Restartable: a combination whose results already exist is skipped.
# At the end it writes out\paper\stage_swap_results.zip.

param([int[]]$Seeds = @(42, 2025, 1337))

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$PY = "python"

$IMAGES    = "../data/casia/images"
$RAW_MASKS = "../data/casia/stage1_masks"
$OUT       = "out"
$ORIENT    = "$OUT\labels_oriented"
$RUNS      = "runs_pipeline_aug"
$LOG       = "$RUNS\swap_log.txt"
$R         = "runs\results_swap.csv"
$XR        = "$OUT\cross\results\results_cross_swap.csv"
$SEEDLIST  = ($Seeds -join ",")

$CK = @{
    "1o" = "../checkpoints/basic/stage1_seg_mobilenetv2_seed{seed}.pt"
    "2o" = "../checkpoints/basic/stage2_unirot_angle_seed{seed}.pt"
    "3o" = "../checkpoints/basic/stage3_roi_seed{seed}.pth"
    "1a" = "$RUNS\stage1_seed{seed}\best.pt"
    "2a" = "$RUNS\stage2_seed{seed}\best.pt"
    "3a" = "$RUNS\stage3_seed{seed}\best.pt"
}

function Note($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $LOG -Value $line
}
function Run($argsList) {
    & $PY @argsList
    if ($LASTEXITCODE -ne 0) { Note "FAILED: $($argsList -join ' ')"; throw "FAILED" }
}

Note "=== stage-swap evaluation started (seeds $SEEDLIST) ==="
foreach ($code in @("aoo", "oao", "ooa", "aao", "aoa", "oaa")) {
    $seg = $CK["1" + $code[0]]; $rot = $CK["2" + $code[1]]; $roi = $CK["3" + $code[2]]
    $tag = "_sw_$code"
    if (Test-Path "$OUT\full_pipeline$tag\seed$($Seeds[-1])\summary.json") { Note "[skip] CASIA-MS $code" }
    else {
        Note "CASIA-MS, stages $code"
        Run @("full_pipeline_eval.py", "--images_dir", $IMAGES, "--gt_masks_dir", $RAW_MASKS, "--labels_root", $ORIENT,
              "--out_dir", $OUT, "--tag", $tag, "--method_label", " $code", "--results_csv", $R, "--seeds", $SEEDLIST,
              "--seg_ckpt_pattern", $seg, "--rot_ckpt_pattern", $rot, "--roi_ckpt_pattern", $roi,
              "--dump_random", "0", "--dump_worst", "0", "--dump_fail", "0")
    }
    if (Test-Path "$OUT\cross\results\scut\per_image_pipeline${tag}_seed$($Seeds[-1]).csv") { Note "[skip] VERA/SCUT-PV $code" }
    else {
        Note "VERA and SCUT-PV, stages $code"
        Run @("xd_eval.py", "--cross_dir", "$OUT\cross", "--datasets", "vera,scut", "--seeds", $SEEDLIST,
              "--pipeline_only", "--pipeline_tag", $tag, "--skip_casia_check", "--results_csv", $XR,
              "--seg_ckpt_pattern", $seg, "--rot_ckpt_pattern", $rot, "--roi_ckpt_pattern", $roi)
    }
}

# pack
$dst = "$OUT\paper\stage_swap_results.zip"
$stage = "$OUT\paper\stage_swap_results"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
$files = @()
$files += Get-Item $R, $XR, $LOG -ErrorAction SilentlyContinue
$files += Get-ChildItem "$OUT" -Directory -Filter "full_pipeline_sw_*" | Get-ChildItem -Recurse -Include per_image.csv, summary.json
$files += Get-ChildItem "$OUT\cross\results" -Recurse -Include "per_image_pipeline_sw_*.csv"
foreach ($f in ($files | Select-Object -ExpandProperty FullName -Unique)) {
    $rel = (Resolve-Path -LiteralPath $f -Relative).TrimStart(".", "\")
    $target = Join-Path $stage $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    Copy-Item -LiteralPath $f -Destination $target
}
if (Test-Path $dst) { Remove-Item $dst }
Compress-Archive -Path "$stage\*" -DestinationPath $dst
Note "results packed -> $dst"
Note "=== stage-swap evaluation finished ==="
