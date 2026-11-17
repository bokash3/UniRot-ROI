# run_experiments.ps1
# Runs the main experiments of the paper (see README.md).
#
# Usage (from experiments):
#   powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1            # everything
#   powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Step pca  # one step
#
# Steps: prep | pca | A | B | C | D | summary | match | valley | yolo_pose | yolo_obb | fullpipe | yolo_qc | xd_prep | xd_prep_more | xd_label_casia2 | xd_label_vera | xd_label_scut | xd_label_casia | xd_eval | xmatch_calib | xmatch | fp32_rerun | all
# Training time: roughly 2-3 h per cell and seed on an RTX 3080 (A, B, C: 9 runs).

param([string]$Step = "all", [int[]]$Seeds = @(42, 2025, 1337))

$ErrorActionPreference = "Stop"
$PY = "python"

$IMAGES        = "../data/casia/images"
$RAW_MASKS     = "../data/casia/stage1_masks"
$ALIGNED_MASKS = "../data/casia/aligned_masks"
$LABELS_ROOT   = "..\data\casia"
$PRED42        = "../data/casia/predicted_angles_seed42.json"
$HA            = "../data/casia"

$OUT     = "out"
$FRAME   = "$OUT\frame_angles.json"
$ORIENT  = "$OUT\labels_oriented"
$ALIMG   = "$OUT\images_frame_aligned"
$RESULTS = "runs\results_2x2.csv"

function Run($argsList) {
    & $PY @argsList
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $($argsList -join ' ')" }
}

function Do-Prep {
    if (-not (Test-Path $FRAME)) {
        Run @("derive_frame_angles.py", "--raw_masks", $RAW_MASKS, "--aligned_masks", $ALIGNED_MASKS,
              "--init_angles", $PRED42, "--out", $FRAME, "--workers", "6")
    } else { Write-Host "[skip] $FRAME exists" }
    if (-not (Test-Path "$ORIENT\seed1337\test_labels.json")) {
        Run @("make_rx_data.py", "labels", "--labels_root", $LABELS_ROOT, "--frame_angles", $FRAME, "--out_root", $ORIENT)
    } else { Write-Host "[skip] oriented labels exist" }
    if (-not (Test-Path $ALIMG)) {
        Run @("make_rx_data.py", "images", "--images_dir", $IMAGES, "--frame_angles", $FRAME, "--out_dir", $ALIMG)
    } else { Write-Host "[skip] $ALIMG exists" }
    Run @("make_rx_data.py", "verify", "--images_dir", $IMAGES, "--aligned_images", $ALIMG,
          "--labels_root", $LABELS_ROOT, "--oriented_root", $ORIENT, "--seed", "42", "--out_dir", "$OUT\verify", "--n", "12")
    Write-Host ">>> Open $OUT\verify and check that the green squares sit on the palm in BOTH halves." -ForegroundColor Yellow
}

function Do-Match {
    # 1) re-train the per-seed ResNet-18 matchers on crops in the correct (frame-aligned) frame
    $BA = "."
    New-Item -ItemType Directory -Force -Path "checkpoints_v2" | Out-Null
    foreach ($s in $Seeds) {
        $ck = "checkpoints_v2\resnet18_seed$s.pt"
        if (Test-Path $ck) { Write-Host "[skip] $ck exists"; continue }
        Run @("$BA\train_resnet18_matcher.py", "--split_dir", "$LABELS_ROOT\dataset_palm_locator_seed$s",
              "--images_dir", $ALIMG, "--band", "850", "--out_ckpt", $ck, "--epochs", "50",
              "--batch_size", "128", "--lr", "3e-4", "--augment", "true", "--device", "cuda")
    }
    # 2) matching with oriented crops for GT, Lite-PalmROI, Fast-PalmROI and cell A
    Run @("run_matching_v2.py", "--matching_dir", $BA, "--raw_images", $IMAGES, "--frame_angles", $FRAME,
          "--labels_root", $LABELS_ROOT, "--resnet_ckpt_pattern", "checkpoints_v2\resnet18_seed{seed}.pt",
          "--yolo_pattern", "runs_yolo\pose_seed{seed}\eval\pred_labels.json",
          "--out_csv", "runs\matching_v2.csv", "--crops_dir", "$OUT\matching_crops")
    Write-Host ">>> Check $OUT\matching_crops: every crop should be an upright palm interior." -ForegroundColor Yellow
}

function Do-PCA {
    Run @("eval_pca_rotation.py", "--masks_dir", "$HA\generated_masks", "--angles", "$HA\angles.json",
          "--split_json", "$HA\subject_split.json", "--out_csv", "$OUT\pca_rotation_results.csv")
}

function Do-Cell($cell) {
    switch ($cell) {
        "A" { $inp = "image"; $frm = "raw";     $data = $IMAGES;        $hm = $RAW_MASKS;     $name = "cellA_image_raw" }
        "B" { $inp = "mask";  $frm = "raw";     $data = $RAW_MASKS;     $hm = $RAW_MASKS;     $name = "cellB_mask_raw" }
        "C" { $inp = "image"; $frm = "aligned"; $data = $ALIMG;         $hm = $ALIGNED_MASKS; $name = "cellC_image_aligned" }
        "D" { $inp = "mask";  $frm = "aligned"; $data = $ALIGNED_MASKS; $hm = $ALIGNED_MASKS; $name = "cellD_mask_aligned" }
    }
    foreach ($s in $Seeds) {
        $lab = if ($frm -eq "raw") { "$ORIENT\seed$s" } else { "$LABELS_ROOT\dataset_palm_locator_seed$s" }
        Write-Host "=== cell $cell  seed $s ===" -ForegroundColor Cyan
        $t0 = Get-Date
        Run @("train_roi_2x2.py", "--input", $inp, "--frame", $frm, "--data_dir", $data, "--labels_dir", $lab,
              "--hand_masks", $hm, "--seed", "$s", "--batch_size", "128", "--num_epochs", "50",
              "--lr", "1e-3", "--weight_decay", "1e-4", "--num_workers", "4",
              "--save_dir", "runs\$name\seed$s", "--results_csv", $RESULTS)
        Write-Host ("    done in {0:hh\:mm\:ss}" -f ((Get-Date) - $t0)) -ForegroundColor Green
    }
}

function Do-Valley {
    # classical finger-valley baseline (no training, CPU, a few minutes per seed); stages are saved to out\baselines\valley
    Run @("baseline_valley_points.py", "--masks_dir", $RAW_MASKS, "--images_dir", $IMAGES,
          "--labels_root", $ORIENT, "--out_dir", "$OUT\baselines", "--results_csv", "runs\results_baselines.csv")
}

function Do-FullPipe {
    # live Stage I + II + III per seed with each seed's own models (see full_pipeline_eval.py)
    Run @("full_pipeline_eval.py", "--images_dir", $IMAGES, "--gt_masks_dir", $RAW_MASKS,
          "--labels_root", $ORIENT, "--out_dir", $OUT, "--results_csv", "runs\results_baselines.csv")
}

function Do-YoloQC {
    # run the trained YOLO-pose on ALL raw images and save every step (ROI on raw, aligned view, crops, sheets, flags)
    Run @("yolo_visual_qc.py", "--images_dir", $IMAGES, "--labels_root", $ORIENT, "--masks_dir", $RAW_MASKS,
          "--out_dir", $OUT, "--seeds", "42,2025,1337")
}

function Do-XdPrep {
    # cross-dataset test (VERA, SCUT-PV): random hands + CASIA check set; saves upright + CASIA-frame images and contact sheets
    Run @("xd_prep.py", "--vera_root", "../datasets/VERA-Palmvein/raw",
          "--scut_root", "../datasets/SCUT_PV_V1",
          "--casia_images", $IMAGES, "--casia_labels", "$ORIENT\seed1337\test_labels.json",
          "--out_dir", "$OUT\cross", "--n", "100", "--n_casia", "40")
    Write-Host ">>> Look at $OUT\cross\<ds>\contact_sheet_*.jpg, then run the xd_label_* steps." -ForegroundColor Yellow
}

function Do-XdEval {
    Run @("xd_eval.py", "--cross_dir", "$OUT\cross", "--datasets", "vera,scut")
    Write-Host ">>> Inspect $OUT\cross\results\<ds>\stages\*_strip.png (every processing step per image)." -ForegroundColor Yellow
}

function Do-Fp32Rerun {
    # Re-run every GPU inference in full FP32 (TF32 off) so that all tables use the same arithmetic.
    $R = "runs\results_baselines_fp32.csv"
    Run @("full_pipeline_eval.py", "--images_dir", $IMAGES, "--gt_masks_dir", $RAW_MASKS,
          "--labels_root", $ORIENT, "--out_dir", $OUT, "--results_csv", $R)
    foreach ($s in $Seeds) {
        foreach ($tag in @("", "_aug")) {
            $a = @("baseline_yolo.py", "eval", "--task", "pose", "--seed", "$s", "--out_root", "$OUT\yolo",
                   "--images_dir", $IMAGES, "--labels_dir", "$ORIENT\seed$s", "--masks_dir", $RAW_MASKS, "--results_csv", $R)
            if ($tag -ne "") { $a += @("--tag", $tag) }
            Run $a
        }
    }
    $XR = "$OUT\cross\results\results_cross_fp32.csv"
    Run @("xd_eval.py", "--cross_dir", "$OUT\cross", "--seeds", "42,2025,1337", "--results_csv", $XR)
    Run @("xd_eval.py", "--cross_dir", "$OUT\cross", "--seeds", "42,2025,1337", "--yolo_only", "--yolo_name", "YOLOv8-pose +aug",
          "--yolo_tag", "_aug", "--yolo_weights", "runs_yolo\pose_aug_seed{seed}\weights\best.pt", "--results_csv", $XR)
    Run @("xd_match.py", "--datasets", "vera", "--sources", "pipeline", "--seeds", "42", "--force",
          "--cross_dir", "$OUT\cross", "--out_dir", "$OUT\xmatch")
    Run @("time_localisers.py", "--seed", "42", "--n", "200")
    Write-Host ">>> FP32 re-run finished. Re-run bootstrap_all.py to update the intervals." -ForegroundColor Yellow
}

function Do-Yolo($task) {
    # modern end-to-end detector baseline (needs: pip install ultralytics). task = pose | obb
    foreach ($s in $Seeds) {
        Write-Host "=== YOLO $task seed $s ===" -ForegroundColor Cyan
        Run @("baseline_yolo.py", "prep", "--task", $task, "--seed", "$s", "--images_dir", $IMAGES,
              "--labels_dir", "$ORIENT\seed$s", "--out_root", "$OUT\yolo")
        Run @("baseline_yolo.py", "train", "--task", $task, "--seed", "$s", "--out_root", "$OUT\yolo",
              "--epochs", "50", "--batch", "16")
        Run @("baseline_yolo.py", "eval", "--task", $task, "--seed", "$s", "--out_root", "$OUT\yolo",
              "--images_dir", $IMAGES, "--labels_dir", "$ORIENT\seed$s", "--masks_dir", $RAW_MASKS,
              "--results_csv", "runs\results_baselines.csv")
    }
}

switch ($Step) {
    "prep"    { Do-Prep }
    "pca"     { Do-PCA }
    "A"       { Do-Cell "A" }
    "B"       { Do-Cell "B" }
    "C"       { Do-Cell "C" }
    "D"       { Do-Cell "D" }
    "summary" { Run @("summarize_2x2.py", "--csv", $RESULTS, "--out", "runs\summary_2x2.csv") }
    "match"   { Do-Match }
    "valley"  { Do-Valley }
    "fullpipe" { Do-FullPipe }
    "xd_prep" { Do-XdPrep }
    "xd_prep_more" { Run @("xd_prep.py", "--vera_root", "../datasets/VERA-Palmvein/raw",
          "--scut_root", "../datasets/SCUT_PV_V1",
          "--out_dir", "$OUT\cross", "--n_vera", "110", "--n_scut", "300", "--one_per_subject", "--extend") }
    "xd_label_casia2" { Run @("xd_label.py", "--ds_dir", "$OUT\cross\casia", "--labels_name", "labels_manual_pass2.json") }
    "xd_label_vera"  { Run @("xd_label.py", "--ds_dir", "$OUT\cross\vera") }
    "xd_label_scut"  { Run @("xd_label.py", "--ds_dir", "$OUT\cross\scut") }
    "xd_label_casia" { Run @("xd_label.py", "--ds_dir", "$OUT\cross\casia") }
    "xd_eval" { Do-XdEval }
    "xmatch_calib" { Run @("xd_match.py", "--calibrate_mc", "--casia_images", $IMAGES,
                           "--casia_labels", "$ORIENT\seed42\train_labels.json", "--out_dir", "$OUT\xmatch") }
    "fp32_rerun" { Do-Fp32Rerun }
    "xmatch" { Run @("xd_match.py", "--datasets", "vera,scut", "--cross_dir", "$OUT\cross", "--out_dir", "$OUT\xmatch")
               Write-Host ">>> Check $OUT\xmatch\<ds>\compare_sources.jpg and */stages/*_strip.png" -ForegroundColor Yellow }
    "yolo_qc" { Do-YoloQC }
    "yolo_pose" { Do-Yolo "pose" }
    "yolo_obb"  { Do-Yolo "obb" }
    "all"     { Do-Prep; Do-PCA; Do-Cell "A"; Do-Cell "B"; Do-Cell "C";
                Run @("summarize_2x2.py", "--csv", $RESULTS, "--out", "runs\summary_2x2.csv") }
    default   { throw "Unknown step $Step" }
}
