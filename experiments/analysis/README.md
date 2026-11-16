# Offset removal and Holm families

These scripts re-analyse results that the other scripts in `experiments/` have already written. They do not read the
images. Run them from this folder.

## Inputs

Copy (or link) the following folders of `experiments/` into `inputs/`, keeping their relative paths:

| Path under `inputs/` | Written by |
|---|---|
| `out/cross/<ds>/labels_manual.json`, `out/cross/<ds>/manifest.json` | `labelling/xd_label.py`, `xd_prep.py` (`<ds>` = `vera`, `scut`) |
| `out/cross/results/<ds>/per_image_<source>_seed<S>.csv` | `xd_eval.py` |
| `out/xmatch/<ds>/<source>_seed<S>/rois.json` | `xd_match.py` |
| `out/<pipeline>/seed<S>/pred_labels_full.json`, `out/labels_oriented/seed<S>/test_labels.json` | `full_pipeline_eval.py`, `make_rx_data.py` |
| `runs_yolo/<run>_seed<S>/eval/pred_labels.json` | `baseline_yolo.py` |
| `out/bootstrap/*.csv` | `bootstrap_all.py` |
| `Supplementary_Data_1_paired_bootstrap.csv` | Supplementary Data 1 of the paper |

## Scripts

| Script | Output in `results/` |
|---|---|
| `offset.py` | `offset_runs.csv`, `offset_summary.csv`, `perimg.pkl`: each run's constant offset (centre along and across the square, log side, angle), estimated by split-half cross-fitting by subject, and the IoU before and after its removal |
| `paired.py` | `offset_paired.csv`: UniRot-ROI minus each detector after offset removal, subject-cluster bootstrap (needs `perimg.pkl` from `offset.py`) |
| `casia.py` | `offset_runs_casia.csv`, `offset_paired_casia.csv`: the same on the CASIA-MS test splits |
| `holm_families.py` | `Supplementary_Data_1_recomputed.csv`, `holm_check.csv`: every comparison of Supplementary Data 1 re-derived from the bootstrap files, with Holm's adjustment within each family |

`common.py` holds the shared geometry (oriented-square corners, IoU) and the loaders.
