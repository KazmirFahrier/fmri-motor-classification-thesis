# Current Status

Last updated: 2026-05-17 18:35 EDT.

This repository is tracking the active full-dataset thesis runs for 4-class motor-task fMRI classification. The unpublished manuscript and private paper PDFs are intentionally not part of this public repo.

## Active Kaggle Runs

| Experiment | Kaggle kernel | Status | Purpose |
| --- | --- | --- | --- |
| Full-dataset pooled legacy baseline | [`kazmirfahrier/thesis-legacy-full-resume`](https://www.kaggle.com/code/kazmirfahrier/thesis-legacy-full-resume) | Running from the epoch-15 artifact with an epoch-16 resume guard | Quantify the full-data pooled-split baseline for the Phase 1 leakage-gap comparison. |
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
- The B6 guarded resume passed the epoch-12 guard, used a Kaggle Tesla P100, saved epochs 12 and 13, then stopped at Kaggle's max allowed execution duration.
- Latest pooled epoch-13 validation snapshot: accuracy 0.2419, balanced accuracy 0.2437, macro F1 0.2285, MCC -0.0090, ROC-AUC 0.4931, PR-AUC 0.2513. This is not a final result.
- Training accuracy reached 0.3973 by epoch 13, while validation remains near chance.
- The pooled artifact dataset `kazmirfahrier/thesis-legacy-full-artifacts` has been refreshed from the epoch-13 checkpoint and is ready for resume from epoch 14.
- The first main-account epoch-14 attempt failed fast because Kaggle still mounted the older epoch-11 artifact; the guard prevented wasted GPU time.
- After Kaggle's dataset listing updated to the May 16 artifact files, `kazmirfahrier/thesis-legacy-full-resume` was repushed as version 2 with the same epoch-14 guard.
- The main-account guarded resume passed the epoch-14 guard, used a Kaggle Tesla P100, saved epochs 14 and 15, then stopped at Kaggle's max allowed execution duration.
- Latest pooled epoch-15 validation snapshot: accuracy 0.2365, balanced accuracy 0.2381, macro F1 0.2227, MCC -0.0167, ROC-AUC 0.4960, PR-AUC 0.2521. This is not a final result.
- Training accuracy reached 0.4273 by epoch 15, while validation remains near chance.
- The policy checker decision after epoch 15 is `continue_short_baseline`, with 10 epochs left before the 25-epoch patience stop if validation does not improve.
- The pooled artifact dataset `kazmirfahrier/thesis-legacy-full-artifacts` has been refreshed from the epoch-15 checkpoint and is ready for resume from epoch 16.
- `kazmirfahrier/thesis-legacy-full-resume` was repushed as version 3 with an epoch-16 guard and is currently running.
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
| Full-dataset pooled split | Epoch 15 validation snapshot: accuracy 0.2365, balanced accuracy 0.2381, macro F1 0.2227, MCC -0.0167. Guarded resume from epoch 16 is running. |
| Full-dataset subject-wise CV + holdout | Complete. All five CV folds and final holdout are chance-level: accuracy 0.25, balanced accuracy 0.25, macro F1 0.10, MCC 0.0. |

## Why This Matters

Phase 1 is designed to separate optimistic pooled-split performance from leakage-aware generalization. The paper should not claim the original 85% result as the final full-dataset result. The key publication-grade evidence will come from subject-wise cross-validation and held-out subject evaluation.

## Decision Policy

- Continue the pooled legacy lane only as a controlled baseline completion to early stopping.
- Stop the lane when the configured 25-epoch patience is exhausted unless validation accuracy and macro F1 both exceed 0.30.
- Do not continue toward the 200-epoch target while validation remains near chance.
- When the lane stops without meaningful validation improvement, pivot to dataset/label audit, preprocessing checks, leakage analysis, and simpler sanity-check baselines.

## Next Actions

- Monitor the active main-account pooled legacy resume and download outputs when it stops.
- Refresh pooled legacy artifacts if the active run saves progress beyond epoch 15.
- Evaluate every downloaded pooled run with `scripts/assess_pooled_legacy_policy.py` before launching another resume hop.
- Treat the completed subject-wise result as evidence that the current full-dataset subject-wise setup is not learning; investigate data/model/label issues before making publication claims.
- Download completed outputs into local status folders after each Kaggle session.
- Update `experiments/phase1_baselines/*.results.json` when new metrics or fold completions are available.
