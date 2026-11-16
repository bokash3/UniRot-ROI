# UniRot-ROI: matched geometric augmentation and cross-dataset evaluation of palm-vein ROI localisers

Code for the paper

> A. Almaktari, M. Bokash, W. Ghaban, A. M. Al-Hejri and V. T. Humbe, *Matched geometric augmentation changes the cross-dataset ranking of palm-vein ROI localisers* (submitted to *Scientific Reports*).

UniRot-ROI uses the three networks of UniRotNet, introduced in

> A. Almaktari, M. Bokash and V. T. Humbe, UniRotNet: rotation-invariant palm ROI extraction, *International Journal of Innovative Research in Technology* **13**(6), 2233–2252 (2026).

The repository contains everything used to produce the results of the paper: data splitting, the labelling tools, training of the three stages of UniRot-ROI and of every compared localiser, the label-consistent ("matched") augmentation and its self-check, inference, the geometric and recognition evaluation on CASIA-MS, VERA Palmvein and SCUT-PV, and the statistics (paired bootstrap, Holm, McNemar, TAR at fixed FAR).

Trained models, labels and per-image results are not part of this repository; they are available from the corresponding author on reasonable request (see [Models, labels and results](#models-labels-and-results)). The images are **not** included; they must be obtained from their providers.

## Repository layout

```
pipeline_models/models/   network definitions used at inference by all evaluation scripts
                          unet_seg.py (Stage I, U-Net), rot_net.py (Stage II, UniRotNet), ROINet.py (Stage III regressor)
stage12/                  Stages I and II with basic augmentation (Pipeline-basic): subject-wise splits, U-Net training
                          (MobileNetV2 and ResNet-34 encoders), mask inference, UniRotNet training and evaluation,
                          and the batch tool that wrote the Stage I masks and the upright (aligned) masks
stage3_roi/               Stage III with basic augmentation: split of the labelled upright masks and regressor training
labelling/                labelling and label-checking tools (ROI square on the upright mask; two-click square for VERA,
                          SCUT-PV and the repeated CASIA-MS labels; merging of corrected labels; consistency check);
                          second_annotator/: the tool and instructions given to the second annotator and for the repeat
experiments/              everything after the basic models: matched augmentation (all three stages), the 2×2 ablation,
                          stage swap, YOLOv8n-pose, Valley and MediaPipe baselines, cross-dataset preparation and evaluation,
                          the three matchers, bootstrap/TAR/DET statistics, timing, the label-agreement analyses and the PowerShell runners;
                          analysis/: removal of constant label offsets and the Holm families of Supplementary Data 1
docs/NAMES.md             internal names used in file names and code → names used in the paper
docs/DATA.md              expected data layout and the licence conditions of the three datasets
tools/                    helpers for the release (collecting models and labels, checksums)
```

## Installation

Python 3.10 or later. The paper's results were produced on Windows 11 with one NVIDIA RTX 3080 Ti Laptop GPU, PyTorch 2.5.1 (CUDA) and Ultralytics 8.3.55.

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # install the CUDA build of PyTorch first if you use a GPU
```

All evaluation runs in FP32 with TF32 disabled, as in the paper.

## Data

Request the datasets from their providers and place them as described in [docs/DATA.md](docs/DATA.md):

| Dataset | Provider | Used for |
|---|---|---|
| CASIA-MS (CASIA Multi-Spectral Palmprint V1.0) | CASIA, https://www.idealtest.org/#/datasetDetail/6 | training (all six bands) and in-domain test |
| VERA Palmvein | Idiap, https://www.idiap.ch/en/dataset/vera-palmvein | test without adaptation |
| SCUT-PV v1 | SCUT BIP Lab, https://github.com/SCUT-BIP-Lab/SCUT_PV_v1 | test without adaptation |

Default paths in the scripts are relative to the repository: `data/casia/...` for CASIA-MS and its labels, `datasets/VERA-Palmvein/raw` and `datasets/SCUT_PV_V1` for the test datasets, and `checkpoints/...` for trained models. Every path can be overridden on the command line. Scripts in `experiments/` are run from inside that folder.

## Reproducing the paper

The seeds are 42, 2025 and 1337 throughout. Commands below are for Windows PowerShell; the Python calls are the same on Linux.

### 1. Splits and labels (Methods: Datasets and orientation normalisation; Reference labels and their repeatability)

```bash
cd stage12
# subject-disjoint 70/15/15 splits for the three seeds (Stages I and II)
python -m src.build_splits --images_dir ../data/casia/images --angles_json ../data/casia/angles.json \
       --out_path ../data/casia/subject_split.json --train_ratio 0.70 --val_ratio 0.15 --test_ratio 0.15 --seeds 42 2025 1337
```

The split of the labelled upright masks used by Stage III (one per seed, same ratios, by subject) is written by
`stage3_roi/make_roi_split.py` (`--help` lists its options).

Labelling tools (one annotator, all six bands, every label checked by eye):

* hand masks: drawn with [LabelMe](https://github.com/labelmeai/labelme) using its AI-assisted outlining; a first U-Net
  (`stage12/src/train_seg.py`) was trained on part of them, its predictions were written for all images, and their errors
  were corrected by hand before Stage I was trained;
* ROI square on the upright mask: `labelling/roi_square_label_gui.py` (sliders for centre and side). In later rounds the
  sliders start from the proposal of an earlier regressor; `labelling/select_hard_roi_samples.py` lists squares that leave
  the hand mask, `labelling/merge_hard_labels_round3.py` merges the corrected labels and `labelling/check_label_consistency.py`
  checks labels against the image folders;
* reference squares on VERA and SCUT-PV, and the repeated labels on 40 CASIA-MS images: `labelling/xd_label.py` (two clicks
  for the centre and the midpoint of the top edge, then 1-pixel, 1 % and 1° adjustment; no localiser output is shown);
* frame angle of every CASIA-MS label (registration of the upright mask to the raw mask, 0.1° resolution):
  `experiments/derive_frame_angles.py`; raw-frame oriented squares: `experiments/make_rx_data.py labels`.

### 2. Pipeline-basic (three stages, basic augmentation)

```bash
cd stage12
# Stage I: U-Net with MobileNetV2 encoder (and ResNet-34 encoder, whose masks train Stage II), one run per seed
python -m src.train_seg --encoder mobilenet_v2 --images_dir ../data/casia/images --masks_dir ../data/casia/masks \
       --split_json ../data/casia/subject_split.json --seed 42 --outdir runs/seg_mobilenetv2_42
python -m src.train_seg --encoder resnet34 ... --outdir runs/seg_resnet34_42
python -m src.infer_seg --ckpt runs/seg_resnet34_42/best.pt --images_dir ../data/casia/images --out_masks_dir preds/seg_resnet34_42
# Stage II: UniRotNet (regression head, as used in the paper; the script also trains the classification and fused heads compared in the UniRotNet paper)
python src/run_rot_experiments.py
# Stage I masks and upright masks for Stage III (seed-1337 Stage I and the reference frame angle)
python generate_masks_from_trained_segmentation.py --help
python make_aligned_masks.py --help

cd ../stage3_roi
# Stage III: MobileNetV2 regressor on the upright mask, letter-boxed to 256 × 256, 50 epochs, batch 128, horizontal flip
python train_stage3_multi_seed.py --seeds 42,2025,1337 --data_root_base ../data/casia/dataset_palm_locator_seed \
       --img_size 256 --batch_size 128 --roinet_epochs 50 --train_roinet
```

### 3. UniRot-ROI (matched augmentation, label-consistent)

```powershell
cd experiments
powershell -ExecutionPolicy Bypass -File .\run_pipeline_aug.ps1 -Step selfcheck   # checks that each transform moves input and target together
powershell -ExecutionPolicy Bypass -File .\run_pipeline_aug.ps1                  # trains the three stages for the three seeds and evaluates them
```

`train_pipeline_aug.py` contains the matched augmentation for each stage (Supplementary Algorithm S1) and the self-check.

### 4. Compared localisers and ablations

| Paper | Runner step / script |
|---|---|
| YOLO-basic, YOLO-matched, YOLO-matched-PT, YOLO-basic-100 | `run_reviewer_experiments.ps1 -Step yolo_pose`, `run_extra.ps1 -Step yolo`, `run_pipeline_aug.ps1 -Step light100` (`baseline_yolo.py`) |
| Detector training seeds (nine runs of YOLO-matched and YOLO-matched-PT) | `run_T2.ps1` (`baseline_yolo.py --train_seed`) |
| Larger pretrained detectors (YOLOv8s-pose-PT, YOLOv8m-pose-PT) | `run_A1.ps1` |
| Valley baseline | `run_reviewer_experiments.ps1 -Step valley` (`baseline_valley_points.py`, `xd_valley.py`) |
| MediaPipe baseline | `baseline_mediapipe.py` |
| 2×2 ablation (input × frame), basic augmentation | `run_reviewer_experiments.ps1 -Step A`, `B`, `C`, `summary` (`train_roi_2x2.py`, `eval_checkpoints.py`) |
| 2×2 raw-frame cells with matched augmentation | `run_extra.ps1 -Step raw` (`train_raw_aug.py`) |
| Stage swap (which stage needs augmentation) | `run_stage_swap.ps1` (`full_pipeline_eval.py`) |
| Principal-axis angle baseline | `eval_pca_rotation.py` |

### 5. Evaluation without adaptation and recognition

```powershell
cd experiments
powershell -ExecutionPolicy Bypass -File .\run_reviewer_experiments.ps1 -Step xd_prep       # fixed 90° rotation + letter-box to the CASIA-MS frame
powershell -ExecutionPolicy Bypass -File .\run_reviewer_experiments.ps1 -Step xd_eval       # geometry on the labelled images
powershell -ExecutionPolicy Bypass -File .\run_reviewer_experiments.ps1 -Step xmatch_calib  # Maximum Curvature scale on CASIA-MS training identities
powershell -ExecutionPolicy Bypass -File .\run_reviewer_experiments.ps1 -Step xmatch        # CompCode, ResNet-18 and vein matcher
```

The ResNet-18 matcher is trained with `train_resnet18_matcher.py` on the training and validation identities of each seed (manifests from `build_matching_manifests_v4.py`); CompCode and the vein matcher (`vein_extract.py`, Miura matching) need no training.

### 6. Statistics and system measurements

```bash
python bootstrap_all.py geom_casia ; python bootstrap_all.py geom_cross ; python bootstrap_all.py match
python tar_far_det.py            # TAR at FAR = 0.1 % and 0.01 %, DET curves
python item12_consistency.py --part all   # batch-size invariance, failures to acquire, TF32 effect
python time_localisers.py        # batch-1 latency
```

Each script documents its inputs and outputs in its header.

### 7. Label agreement and label noise (second annotator and repeat)

The second annotator and the repeat by the first annotator used `labelling/second_annotator/label_tool.py` on 50 VERA and
50 SCUT-PV images (instructions in `labelling/second_annotator/INSTRUCTIONS.md`; the `.bat` files start the practice
round, the second annotator and the repeat). Their label files go to `labelling/second_annotator/data/`.

```bash
cd experiments
python t6_analyze.py         # agreement between label sets and the comparisons of the localisers under each set (Supplementary Data 4)
python t8_label_noise.py     # constant-offset removal (cross-fitted), equivalence margin, human-parity rate
cd analysis                  # reads copies of the result files from inputs/ and writes results/ (see analysis/README.md)
python offset.py ; python paired.py   # VERA and SCUT-PV comparisons after removing constant label offsets
python casia.py              # the same on the CASIA-MS test splits
python holm_families.py      # re-derives the Holm families of Supplementary Data 1 from the bootstrap files
```

`yolo_visual_qc.py` writes every intermediate step of a detector (square, rotation, crop) for visual inspection.

## Models, labels and results

Available from the corresponding author on reasonable request:

* trained weights of every localiser and seed (Stages I–III of Pipeline-basic and UniRot-ROI, the four YOLOv8n-pose
  detectors, the 2×2 cells, the ResNet-18 matchers);
* CASIA-MS labels: hand angles, frame angles, ROI squares (upright and raw frame), subject splits, and the repeated labels
  of 40 test images;
* VERA reference squares and the squares of the second annotator and of the repeat (coordinates only, no image content);
* per-image predictions and score files for CASIA-MS and VERA, and every result table.

The SCUT-PV licence does not allow any part of the data to be redistributed without the providers' approval, so the SCUT-PV reference squares, the squares of the second annotator and of the repeat, the list of the 300 palms and the SCUT-PV per-image files are shared only with holders of the SCUT-PV licence (contact the corresponding author).

The weights are trained on CASIA-MS and are released for non-commercial research only, in line with the CASIA-MS licence.

## Names

The code predates the final names of the paper; `docs/NAMES.md` maps file and variable names (for example `pipeline_aug`, `yolo_pre`, `rawB_aug`) to the names used in the paper (UniRot-ROI, YOLO-matched-PT, mask input / raw frame / matched). Table numbers mentioned in code comments refer to internal drafts, not to the published paper.

## Licence

Code: MIT (see `LICENSE`). Model weights and labels: CC BY-NC 4.0, subject to the dataset licences above.

## Citation

If you use this code, please cite the paper above and the UniRotNet paper (see `CITATION.cff`).
