# Current Status

Last updated: 2026-05-20 18:35 EDT.

This repository is tracking the active full-dataset thesis runs for 4-class motor-task fMRI classification. The unpublished manuscript and private paper PDFs are intentionally not part of this public repo.

## Active Kaggle Runs

| Experiment | Kaggle kernel | Status | Purpose |
| --- | --- | --- | --- |
| Full-dataset pooled legacy baseline | [`kazmirfahrierlover/thesis-legacy-full-resume`](https://www.kaggle.com/code/kazmirfahrierlover/thesis-legacy-full-resume) | Running version 2 from Lover's dedicated epoch-21 artifact dataset with a local epoch-22 resume guard | Quantify the full-data pooled-split baseline for the Phase 1 leakage-gap comparison. |
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
- The main-account version 3 guarded resume passed the epoch-16 guard, used a Kaggle Tesla P100, saved epochs 16 and 17, then stopped at Kaggle's max allowed execution duration.
- Latest pooled epoch-17 validation snapshot: accuracy 0.2470, balanced accuracy 0.2497, macro F1 0.2273, MCC -0.0003, ROC-AUC 0.5017, PR-AUC 0.2575. This is not a final result.
- Training accuracy reached 0.4603 by epoch 17, while validation remains near chance.
- The policy checker decision after epoch 17 is `continue_short_baseline`, with 8 epochs left before the 25-epoch patience stop if validation does not improve.
- The pooled artifact dataset `kazmirfahrier/thesis-legacy-full-artifacts` has been refreshed from the epoch-17 checkpoint and is ready for resume from epoch 18.
- `kazmirfahrier/thesis-legacy-full-resume` version 4 failed fast because Kaggle still mounted a stale epoch-15 copy of `kazmirfahrier/thesis-legacy-full-artifacts`; the epoch-18 guard prevented wasted training.
- A dedicated cache-busting artifact dataset, `kazmirfahrier/thesis-legacy-full-artifacts-epoch17`, was created from the verified local epoch-17 checkpoint tree.
- `kazmirfahrier/thesis-legacy-full-resume` version 5 failed fast because the embedded resume script did not auto-discover the new dedicated artifact slug, leaving no checkpoint synced into `/kaggle/working/thesis_session/thesis_legacy_full_dataset/train/last_checkpoint.pt`.
- `kazmirfahrier/thesis-legacy-full-resume` version 6 failed fast because Kaggle mounted the dedicated dataset somewhere other than the guessed `/kaggle/input/datasets/kazmirfahrier/thesis-legacy-full-artifacts-epoch17` path.
- `kazmirfahrier/thesis-legacy-full-resume` was repushed as version 7 with runtime `/kaggle/input` mount discovery and an epoch-18 guard.
- The main-account version 7 guarded resume passed the epoch-18 guard, used a Kaggle Tesla P100, saved epochs 18 and 19, then stopped at Kaggle's max allowed execution duration.
- Latest pooled epoch-19 validation snapshot: accuracy 0.2444, balanced accuracy 0.2468, macro F1 0.2326, MCC -0.0041, ROC-AUC 0.5058, PR-AUC 0.2586. This is not a final result.
- Epoch 18 briefly improved validation accuracy to 0.2545 and macro F1 to 0.2504, but this remains near chance and below the 0.30/0.30 extension threshold.
- Training accuracy reached 0.4889 by epoch 19, while validation remains near chance.
- The policy checker decision after epoch 19 is `continue_short_baseline`, with 6 epochs left before the 25-epoch patience stop if validation does not materially improve.
- A dedicated Niloy-account artifact dataset, `kazmirfahrierniloy/thesis-legacy-full-artifacts-epoch19`, was created from the verified local epoch-19 checkpoint tree.
- `kazmirfahrierniloy/thesis-legacy-full-resume` version 3 failed because the main-account private epoch-19 artifact dataset could not be attached to the Niloy kernel.
- `kazmirfahrierniloy/thesis-legacy-full-resume` version 4 reached a Tesla P100 and found the epoch-19 artifact, but failed before training because the wrapper passed an unsupported `--min-resume-epoch` argument to the bundled training script.
- `kazmirfahrierniloy/thesis-legacy-full-resume` was repushed as version 5 with the epoch-20 guard moved into the wrapper.
- The Niloy version 5 guarded resume passed the epoch-20 guard, used a Kaggle Tesla P100, saved epochs 20 and 21, then stopped at Kaggle's max allowed execution duration.
- Latest pooled epoch-21 validation snapshot: accuracy 0.2545, balanced accuracy 0.2563, macro F1 0.2495, MCC 0.0084, ROC-AUC 0.5062, PR-AUC 0.2586. This is not a final result.
- Training accuracy reached 0.5190 by epoch 21, while validation remains near chance and below the 0.30/0.30 extension threshold.
- The policy checker decision after epoch 21 is `continue_short_baseline`, with 4 epochs left before the 25-epoch patience stop if validation does not materially improve.
- A dedicated Lover-account artifact dataset, `kazmirfahrierlover/thesis-legacy-full-artifacts-epoch21`, was created from the verified local epoch-21 checkpoint tree.
- `kazmirfahrierlover/thesis-legacy-full-resume` was repushed as version 2 with an epoch-22 guard and is currently running.
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
| Full-dataset pooled split | Epoch 21 validation snapshot: accuracy 0.2545, balanced accuracy 0.2563, macro F1 0.2495, MCC 0.0084. Version 2 Lover guarded resume from epoch 22 is running. |
| Full-dataset subject-wise CV + holdout | Complete. All five CV folds and final holdout are chance-level: accuracy 0.25, balanced accuracy 0.25, macro F1 0.10, MCC 0.0. |

## Why This Matters

Phase 1 is designed to separate optimistic pooled-split performance from leakage-aware generalization. The paper should not claim the original 85% result as the final full-dataset result. The key publication-grade evidence will come from subject-wise cross-validation and held-out subject evaluation.

## Decision Policy

- Continue the pooled legacy lane only as a controlled baseline completion to early stopping.
- Stop the lane when the configured 25-epoch patience is exhausted unless validation accuracy and macro F1 both exceed 0.30.
- Do not continue toward the 200-epoch target while validation remains near chance.
- When the lane stops without meaningful validation improvement, pivot to dataset/label audit, preprocessing checks, leakage analysis, and simpler sanity-check baselines.

## Next Actions

- Monitor the active Lover-account pooled legacy resume and download outputs when it stops.
- Refresh pooled legacy artifacts if the active run saves progress beyond epoch 21.
- Evaluate every downloaded pooled run with `scripts/assess_pooled_legacy_policy.py` before launching another resume hop.
- Treat the completed subject-wise result as evidence that the current full-dataset subject-wise setup is not learning; investigate data/model/label issues before making publication claims.
- Download completed outputs into local status folders after each Kaggle session.
- Update `experiments/phase1_baselines/*.results.json` when new metrics or fold completions are available.
