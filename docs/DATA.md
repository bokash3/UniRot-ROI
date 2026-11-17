# Data layout and licences

## Datasets (not included)

| Dataset | Obtain from | Licence summary |
|---|---|---|
| CASIA-MS | https://www.idealtest.org/#/datasetDetail/6 | non-commercial research; no redistribution of the images |
| VERA Palmvein | https://www.idiap.ch/en/dataset/vera-palmvein (Idiap) | end-user licence agreement; no publication of corpus content |
| SCUT-PV v1 | https://github.com/SCUT-BIP-Lab/SCUT_PV_v1 | written agreement; no redistribution of any part of the data without the providers' approval |

## Expected layout (defaults of the scripts)

```
data/casia/
  images/                          raw CASIA-MS images, 768 × 576 (names like 001_l_850_01.png), all six bands
  masks/                           manually annotated hand masks (Stage I targets)
  angles.json                      recorded hand angle of every image (Stage II targets)
  subject_split.json               subject-disjoint 70/15/15 splits for seeds 42, 2025, 1337 (Stages I and II)
  stage1_masks/                    Stage I masks (seed-1337 Pipeline-basic) of every image
  stage1_masks_resnet34_seed{S}/   ResNet-34 U-Net masks on which Stage II was trained
  aligned_masks/                   upright masks on which the ROI squares were labelled
  dataset_palm_locator_seed{S}/    Stage III split and ROI labels (upright frame) for seed S
  predicted_angles_seed42.json     Stage II angles used to initialise the frame-angle registration
datasets/
  VERA-Palmvein/raw/               VERA images as distributed
  SCUT_PV_V1/                      SCUT-PV images as distributed (and its ROI folder for the released ROI)
checkpoints/basic/                 Pipeline-basic models (file names in docs/NAMES.md)
```

`experiments/xd_prep.py` brings every VERA and SCUT-PV image to the CASIA-MS frame with one fixed affine transform per
dataset (rotation by 90° and letter-box to 768 × 576); no model is re-trained or tuned on these datasets.

## Labels

The labels are available from the corresponding author on reasonable request and follow the same layout. The SCUT-PV
reference squares, the squares of the second annotator and of the repeat, the list of the 300 palms and the SCUT-PV
per-image files are shared only with holders of the SCUT-PV licence. The label files of the second annotator and of the
repeat go to `labelling/second_annotator/data/`.
