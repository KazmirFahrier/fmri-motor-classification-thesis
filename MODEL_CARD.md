# Model Card: fMRI Motor Classification (4-Class)

## Intended Use
Research use for classifying motor-task fMRI volumes into:
- Left leg movements
- Right leg movements
- Forearm movements
- Upper arm movements

## Primary Evaluation Protocol
- Subject-wise 5-fold cross-validation
- One untouched holdout subject set
- No subject overlap across train/val/test/holdout

## Model Families
- Primary: `TemporalResNet3D`
- Baseline: `ResNet3DClassifier`
- Ablation: `CNNTransformerAblation`

## Inputs
- Pre-extracted 3D NIfTI volumes (`.nii.gz`) from fMRI runs
- Optional temporal clips built from contiguous volume IDs

## Outputs
- Class logits (no softmax in model forward)

## Metrics Reported
- Top-1 accuracy, balanced accuracy, macro-F1, MCC
- ROC-AUC (OvR macro), PR-AUC (macro)
- Per-class recall
- Bootstrap confidence intervals on holdout

## Known Limitations
- Subject-wise validation is harder and may produce lower accuracy than volume-wise leakage-prone splits.
- HRF shift heuristic may not capture all subject/task timing variability.
- Results depend on preprocessing quality of extracted volumes.

## Ethical and Scientific Considerations
- This pipeline is intended for research; not for clinical decision-making.
- Leakage control and transparent reporting are mandatory for publication integrity.
