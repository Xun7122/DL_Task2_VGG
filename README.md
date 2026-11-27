# VGG16 — Patient-group CV with a locked local holdout

## Why this experiment

The supplied overfit audit evaluated the same fixed model/threshold without augmentation, using ordinary CE on both splits: Train BA/AUROC 1.0/1.0, CE 0.01656; Validation BA 0.90249, AUROC 0.95517, CE 0.24407. This confirms a training/generalization gap, while validation performance remains strong. It does not establish a catastrophic failure. Five validation patients have within-patient AUROC 1.0; between-patient score offsets are also important. G07 negatives have max p=0.06761, positives min p=0.08036, but the global threshold was 0.53093.

## Run

Open the single-cell `vgg.ipynb` in Colab GPU, change only `DATA_DIR`, and run. No old fine-tuned checkpoint is loaded. All five folds and the final refit start from fresh ImageNet VGG16 initialization.

## Split and selection

- Fixed seed 20261128, selected using patient label counts, never model performance.
- Local Test: G01/G10/G11/G16/G19/G23, 6 patients / 674 images / 194 positives.
- Development: other 21 patients / 2274 images / 594 positives.
- The six patients previously used for model selection are excluded from the new local Test. The whole provided dataset has previously been used in development, so this is not a truly new external Test cohort and is separate from the course private Test.
- Development-only five-fold group split: each validation fold has 4 or 5 patients. Patients with different positive counts are distributed in alternating ranked strata. This custom patient-balanced split is not sklearn StratifiedGroupKFold.
- Each development image receives exactly one out-of-fold prediction per epoch. No patient's images cross training/validation within a fold.
- All-FC preparation: 3 epochs. Full fine-tuning: up to the fixed maximum of 11 epochs per fold. Class weights use that fold's training labels only. Augmentation is training-only. Frozen initial VGG feature caching does not learn from validation labels.
- Fixed cosine LR schedule prevents a held-out fold's labels changing that fold's updates. The final refit follows the same first k epochs of the schedule, where k is selected below.
- Select one common epoch by the minimum mean fold validation balanced plain CE. Then select one common threshold using pooled development OOF predictions at that epoch. Both are model selection estimates; they are not independent Test scores.
- Refit one VGG16 on all development patients using the selected epoch and threshold. Only then load/decode/evaluate the local Test images. Never optimize a Test threshold or select a Test epoch. No ensembling, TTA, or best-fold selection.

## Overfitting interpretation

CV estimates performance variability and chooses training duration. It does not automatically eliminate memorization or guarantee a higher score. The selected model's development training metrics and locked local Test metrics are evaluated with identical plain transforms and ordinary CE. Do not compare the old six-patient Validation score directly with the new six-patient Test score as if they were the same cohort.

## Time, recovery and files

Five CV models plus a final refit are trained sequentially. This takes longer than the previous two-candidate experiment. Completed folds are saved as predictions/history and reused when the exact same code/config/data are rerun. An interrupted fold restarts from initialization; optimizer-state recovery is not provided. After final evaluation completes, rerunning only returns the saved report.

Only the final `final/vgg.pt` is saved by default; `KEEP_FOLD_WEIGHTS=False` saves Drive space and checkpoint I/O. The exported `inference.py` uses the saved VGG16 and fixed threshold. `vgg_groupcv_report.zip` contains small diagnostics, not model weights or raw images. Files are saved under Drive `task2_vgg_groupcv/v1_<config hash>`; code and data hashes prevent mixing experiments.

Validation performed locally: actual-data isolation/reproducibility/OOF coverage; 100 tied-score ROC/AUC cases; development-only threshold selection; Test decoding/evaluation ordering; training/inference syntax. The local environment has no torch, so full GPU training/inference is untested. No performance result is claimed for this new experiment.
