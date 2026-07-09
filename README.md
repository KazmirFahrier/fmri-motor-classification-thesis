# fMRI Motor Classification Thesis Work

This repository tracks ongoing thesis work on 4-class motor-task fMRI classification using the public whole-body somatotopic mapping dataset (`ds004044`). The project started from a smaller 9-subject prototype and is now being extended toward full-dataset evaluation, stronger leakage control, cleaner Kaggle execution, and better-performing model variants.

## What We Are Trying To Do
- Classify four motor conditions from task-evoked fMRI volumes:
  - `Left leg movements`
  - `Right leg movements`
  - `Forearm movements`
  - `Upper arm movements`
- Use the full extracted dataset rather than only the earlier 9-subject subset.
- Compare multiple model families:
  - legacy 3D CNN + attention + transformer model from the thesis notebooks
  - cleaner 3D CNN / temporal clip baselines
  - ablations that isolate attention, transformer, and other components
  - newer whole-brain temporal models that may improve full-dataset performance
- Separate exploratory results from rigorous results:
  - pooled random-split experiments for legacy reproduction
  - subject-wise cross-validation and holdout evaluation for leakage-controlled claims
- Package the work so it can be reproduced on Kaggle, where the full dataset currently fits best.

## Current Status
- The repository contains the cleaned training pipeline, configs, scripts, and tests for leakage-aware experiments.
- The older notebooks and `notebook_code.py` preserve the earlier prototype workflow used to build the initial model idea.
- The unpublished manuscript is intentionally **not** included in this public repo.
- Phase 1 full-dataset legacy runs are complete: the pooled legacy baseline was stopped at epoch 25 by the controlled decision policy, and the subject-wise legacy run completed at chance-level holdout performance.
- The active work is now the full-scale diagnostic/modeling investigation: continuous-event baselines, subject/run nuisance control, multi-scale covariance-aware hierarchies, weak-subject forensics, and temporal-basis experiments for the remaining arm-branch errors.
- Current run state is tracked in `docs/CURRENT_STATUS.md` and `experiments/phase1_baselines/*.results.json`.

## Dataset Context
- Source dataset: OpenNeuro `ds004044`
- Public dataset paper: whole-body somatotopic mapping in healthy adults
- Full extracted working set used in this thesis:
  - `62` subjects
  - `7` extraction batches
  - `4` target motor classes
- Each motor block spans `16` seconds, corresponding to `8` fMRI volumes at `TR = 2s`
- Extracted volumes are resized to a common 3D shape and saved as class-organized NIfTI files

## Why There Are Multiple Pipelines
- The early thesis result was produced on a smaller subset with a simple random split.
- The cleaned pipeline in `src/fmri_pipeline/` is designed for stricter subject-wise evaluation to reduce leakage risk.
- Both are kept here because reproducing the old result and improving it on the full dataset are both part of the thesis story.

## Project Layout
- `src/fmri_pipeline/`: core package modules.
- `scripts/`: runnable CLIs.
- `configs/`: strict YAML experiment configs.
- `tests/`: unit and smoke tests.
- `appendices/`: leakage and reproducibility notes.
- `*.ipynb` and `notebook_code.py`: earlier thesis experimentation and Kaggle-oriented workflows.

## Main Research Directions In This Repo
1. Reproduce the original thesis-style model honestly on larger data.
2. Measure how much each architecture component actually contributes.
3. Run strict subject-wise evaluation on the full dataset.
4. Improve accuracy without relying on leakage-prone splits.
5. Understand the dataset deeply enough to distinguish model limitations, subject-specific response geometry, and true QC/outlier cases.

## Active Experiment Registry
The `experiments/` directory records what is running and what has completed. Result JSONs are intentionally lightweight: they store status, protocol, metrics, fold progress, Kaggle kernel links, and artifact dataset references, not raw data or model checkpoints.

Current Phase 1 registry entries:
- `phase1_original_subset_pooled`: historical 9-subject pooled-split baseline, complete.
- `phase1_full_dataset_pooled`: full public dataset pooled-split legacy run, complete/stopped by policy.
- `phase1_subjectwise_5fold_full`: full public dataset subject-wise 5-fold plus holdout run, complete.

Current investigation registry entries:
- `continuous_temporal_basis_arm`: planned continuous-BOLD temporal-basis extraction and nested arm-branch representation screen.

These files are meant to make the repo reflect the real research process: long-running Kaggle sessions, resumable checkpoints, and metrics updated only when a run actually finishes.

## Install
```bash
pip install -r requirements.txt
```

## 1) Build Manifest Index
```bash
python scripts/build_index.py \
  --data-roots dataset/batch_01 dataset/batch_02 dataset/batch_03 dataset/batch_04 dataset/batch_05 dataset/batch_06 dataset/batch_07 \
  --class-names "Left leg movements,Right leg movements,Forearm movements,Upper arm movements" \
  --out-index artifacts/index.parquet \
  --strict
```

## 2) Run Data QC
```bash
python scripts/check_data.py \
  --index artifacts/index.parquet \
  --class-names "Left leg movements,Right leg movements,Forearm movements,Upper arm movements" \
  --manifest-qc artifacts/index_qc.json \
  --out-report artifacts/data_qc_report.json \
  --strict
```

## 3) Create Subject-Wise Splits (5-Fold + Holdout)
```bash
python scripts/make_splits.py \
  --index artifacts/index.parquet \
  --seed 42 \
  --holdout-subject-count 8 \
  --cv-folds 5 \
  --out-splits artifacts/splits_subjectwise.json
```

## 4) Train One Fold
```bash
python scripts/train.py \
  --config configs/default_4class_clip.yaml \
  --index artifacts/index.parquet \
  --splits artifacts/splits_subjectwise.json \
  --fold 0 \
  --run-name clip_baseline \
  --out-dir artifacts/runs
```

### Optional Optuna Search (per-fold)
```bash
python scripts/train.py \
  --config configs/default_4class_clip.yaml \
  --index artifacts/index.parquet \
  --splits artifacts/splits_subjectwise.json \
  --fold 0 \
  --run-name clip_optuna \
  --out-dir artifacts/runs \
  --optuna-trials 20
```

## 5) Full CV + Automatic Holdout Evaluation
```bash
python scripts/run_cv.py \
  --config configs/default_4class_clip.yaml \
  --index artifacts/index.parquet \
  --splits artifacts/splits_subjectwise.json \
  --run-name clip_cv_main \
  --out-dir artifacts/runs
```

## 6) Evaluate a Saved Checkpoint
```bash
python scripts/evaluate.py \
  --checkpoint artifacts/runs/clip_cv_main/holdout_model/best_model.pt \
  --index artifacts/index.parquet \
  --split holdout \
  --splits artifacts/splits_subjectwise.json \
  --out-dir artifacts/runs/clip_cv_main/manual_holdout_eval
```

## 7) Build Publication Bundle
```bash
python scripts/make_report.py \
  --run-dir artifacts/runs/clip_cv_main \
  --out-dir artifacts/runs/clip_cv_main/publication_bundle
```

## Metrics
- Top-1 accuracy
- Balanced accuracy
- Macro-F1
- MCC
- ROC-AUC (OvR macro)
- PR-AUC (macro)
- Per-class recall
- Bootstrap CIs for holdout (accuracy, balanced accuracy, macro-F1, MCC)

## Reproducibility
- Strict config validation before training.
- Deterministic seeding and tracked run manifests.
- Subject-wise leakage checks in split generation.

## Notes
- Higher accuracy is useful, but only if it holds on the full dataset under a defendable evaluation protocol.
- `~90%` is aspirational under strict subject-wise validation; optimistic random-split results should not be treated as the final claim.
- Temporal modeling is performed pre-pooling in `TemporalResNet3D`.
- Legacy CNN+Transformer code is preserved because it is part of the thesis evolution and ablation story.
