# Internal names → names in the paper

The scripts were written before the final terminology of the paper. File names, folder names, command-line tags and
comments use the internal names below.

| Internal name (code, files, folders) | Paper |
|---|---|
| `pipeline`, "original recipe", "mask-aligned pipeline", `full_pipeline` | Pipeline-basic (three stages, basic augmentation) |
| `pipeline_aug`, "augmented recipe", "label-consistent augmentation", `full_pipeline_aug`, `runs_pipeline_aug` | UniRot-ROI (three stages, matched augmentation) |
| `UNetSeg`, `seg_mobilenetv2_{seed}` | Stage I (U-Net, MobileNetV2 encoder) |
| `seg_resnet34_{seed}`, `preds/seg_resnet34_{seed}` | U-Net with ResNet-34 encoder whose masks train Stage II |
| `RotResNet18`, `UniRotNet`, `rot_{seed}_reg` | Stage II, UniRotNet (regression head); `_cls` and `_both` are the classification and fused heads compared in the UniRotNet paper |
| `PalmROINet`, `ROINet`, "Lite-PalmROI", "ROI-Net", `roi_net_best_seed{seed}` | Stage III regressor |
| `SmallCNNBaseline`, "Fast-PalmROI" | not reported in the paper |
| `yolo`, `pose_seed{seed}`, "light recipe" | YOLO-basic |
| `yolo_aug`, `pose_aug_seed{seed}`, "augmented recipe" | YOLO-matched |
| `yolo_pre`, `pose_pre_seed{seed}` | YOLO-matched-PT (COCO-pretrained) |
| `yolo_l100`, `pose_l100_seed{seed}` | YOLO-basic-100 |
| `yolo_aug_t101`, `yolo_aug_t202`, `yolo_pre_t101`, `yolo_pre_t202` (`--train_seed`) | extra training seeds of YOLO-matched and YOLO-matched-PT (nine runs each) |
| `yolo_s_pre`, `pose_s_pre_seed{seed}`; `yolo_m_pre`, `pose_m_pre_seed{seed}` | YOLOv8s-pose-PT; YOLOv8m-pose-PT |
| `valley`, "valley + offset" | Valley baseline |
| `mediapipe` | MediaPipe baseline |
| cell A, `cellA_image_raw`, `rawA_orig` / `rawA_aug` | Image input, raw frame, basic / matched |
| cell B, `cellB_mask_raw`, `rawB_orig` / `rawB_aug` | Mask input, raw frame, basic / matched |
| cell C, `cellC_image_aligned` | Image input, aligned frame, basic |
| cell D, variant V0 of `full_pipeline_eval.py` | Mask input, aligned frame (Stage III alone) |
| `pipeline_sw_xyz` (x, y, z ∈ {a, o} for Stages I, II, III) | Stage swap: a = matched, o = basic augmentation (Table 4a) |
| `compcode`, `resnet18`, `miura` | CompCode, ResNet-18 and vein matcher (Maximum Curvature + Miura matching) |
| `official` | ROI released with SCUT-PV |
| "frame angle" | angle that maps the raw mask to the upright mask on which the square was labelled |
| `labels_manual.json`, `labels_manual_pass2.json` | reference squares (VERA, SCUT-PV) and the two labelling passes of 40 CASIA-MS images |
| A1, `labels_annotator2.json` (A2), `labels_pass2.json` (P2), CONS | reference squares, second annotator, repeat by the first annotator, and their consensus |
| failure, `failed` | localisation failure (IoU < 0.5); an image without detection is a failure to acquire |

Table numbers in code comments refer to internal drafts, not to the published paper.
