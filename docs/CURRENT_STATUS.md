# Current Status

Last updated: 2026-05-11 15:27 EDT.

This repository is tracking the active full-dataset thesis runs for 4-class motor-task fMRI classification. The unpublished manuscript and private paper PDFs are intentionally not part of this public repo.

## Active Kaggle Runs

| Experiment | Kaggle kernel | Status | Purpose |
| --- | --- | --- | --- |
| Full-dataset pooled legacy baseline | [`b6uejhvvnmiwb/thesis-legacy-full-resume`](https://www.kaggle.com/code/b6uejhvvnmiwb/thesis-legacy-full-resume) | Running from the epoch-11 artifact with a resume guard | Quantify the full-data pooled-split baseline for the Phase 1 leakage-gap comparison. |
| Full-dataset subject-wise evaluation | [`kazmirfahrier/thesis-7batch-gpucompat-runner`](https://www.kaggle.com/code/kazmirfahrier/thesis-7batch-gpucompat-runner) | Complete; final artifacts refreshed | Continue leakage-aware subject-wise 5-fold evaluation plus holdout. |

## Known Progress

- Dataset manifest built successfully for all seven public batches.
- Working set size: 23,808 samples, 62 subjects, 372 runs, 4 balanced motor classes.
- Full-dataset pooled legacy run has a resumable artifact tree at `kazmirfahrier/thesis-legacy-full-artifacts`.
- The pooled legacy run resumed on a Tesla P100 from epoch 8 and saved through epoch 9 before Kaggle stopped the notebook for max allowed execution duration.
- The Niloy pooled legacy resume ran on a Kaggle Tesla P100, resumed from epoch 10, and saved through epoch 11 before Kaggle stopped it for max allowed execution duration.
- Latest pooled epoch-11 validation snapshot: accuracy 0.2386, balanced accuracy 0.2412, macro F1 0.1970, MCC -0.0140, ROC-AUC 0.4861, PR-AUC 0.2496. This is not a final result.
- Training accuracy improved to 0.3666 by epoch 11, but validation remains near chance, so the run is not yet showing useful full-dataset pooled generalization.
- The pooled artifact dataset `kazmirfahrier/thesis-legacy-full-artifacts` has been refreshed from the epoch-11 checkpoint and is ready for resume from epoch 12.
- The Lover resume ran on a Kaggle Tesla P100 but reproduced epochs 10-11 instead of advancing, likely because Kaggle mounted a stale artifact copy shortly after the dataset refresh.
- The pooled legacy resume was relaunched as `b6uejhvvnmiwb/thesis-legacy-full-resume` version 1 with a resume guard that should fail fast unless the mounted checkpoint can resume at epoch 12.
- Subject-wise fold 0 exists in `kazmirfahrier/thesis-7batch-artifacts`.
- Subject-wise fold 1 completed successfully on a Kaggle Tesla P100 after two GPU-compatibility fixes: setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` and using `training.deterministic: false`.
- Subject-wise fold 2 completed successfully on a Kaggle Tesla P100.
- Subject-wise fold 3 completed successfully on a Kaggle Tesla P100 after rebuilding the Niloy runner to remove a stale deterministic config.
- Subject-wise fold 4 completed successfully on a Kaggle Tesla P100.
- The subject-wise holdout-model stage completed successfully on `b6uejhvvnmiwb/thesis-7batch-subjectwise-gpucompat-resume` version 1 using a Kaggle Tesla P100.
- The holdout-model validation snapshot is still chance-level: accuracy 0.25, balanced accuracy 0.25, macro F1 0.10, MCC 0.0, ROC-AUC 0.4959, PR-AUC 0.2583.
- The final subject-wise holdout-evaluation hop completed as `kazmirfahrier/thesis-7batch-gpucompat-runner` version 4.
- Final subject-wise holdout metrics are chance-level: accuracy 0.25, balanced accuracy 0.25, macro F1 0.10, MCC 0.0, ROC-AUC 0.4983, PR-AUC 0.2513.
- The subject-wise artifact dataset `kazmirfahrier/thesis-7batch-artifacts` was refreshed after holdout-evaluation completion and now has `next_stage: complete`.

## Current Metrics Snapshot

| Experiment | Metric status |
| --- | --- |
| Original 9-subject pooled split | Historical baseline: accuracy 0.8522, MCC 0.8055, ROC-AUC 0.95, PR-AUC 0.88. |
| Full-dataset pooled split | Epoch 11 validation snapshot: accuracy 0.2386, balanced accuracy 0.2412, macro F1 0.1970, MCC -0.0140. Guarded resume from epoch 12 is running. |
| Full-dataset subject-wise CV + holdout | Complete. All five CV folds and final holdout are chance-level: accuracy 0.25, balanced accuracy 0.25, macro F1 0.10, MCC 0.0. |

## Why This Matters

Phase 1 is designed to separate optimistic pooled-split performance from leakage-aware generalization. The paper should not claim the original 85% result as the final full-dataset result. The key publication-grade evidence will come from subject-wise cross-validation and held-out subject evaluation.

## Next Actions

- Monitor the active B6 pooled legacy resume and download outputs when it stops.
- Refresh pooled legacy artifacts if the active run saves progress beyond epoch 11.
- Treat the completed subject-wise result as evidence that the current full-dataset subject-wise setup is not learning; investigate data/model/label issues before making publication claims.
- Download completed outputs into local status folders after each Kaggle session.
- Update `experiments/phase1_baselines/*.results.json` when new metrics or fold completions are available.
