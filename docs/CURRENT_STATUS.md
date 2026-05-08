# Current Status

Last updated: 2026-05-08 00:05 EDT.

This repository is tracking the active full-dataset thesis runs for 4-class motor-task fMRI classification. The unpublished manuscript and private paper PDFs are intentionally not part of this public repo.

## Active Kaggle Runs

| Experiment | Kaggle kernel | Status | Purpose |
| --- | --- | --- | --- |
| Full-dataset pooled legacy baseline | [`kazmirfahrierniloy/thesis-legacy-full-resume`](https://www.kaggle.com/code/kazmirfahrierniloy/thesis-legacy-full-resume) | Epoch-9 artifact refreshed; waiting for next sequential hop | Quantify the full-data pooled-split baseline for the Phase 1 leakage-gap comparison. |
| Full-dataset subject-wise evaluation | [`kazmirfahrier/thesis-7batch-gpucompat-runner`](https://www.kaggle.com/code/kazmirfahrier/thesis-7batch-gpucompat-runner) | Fold 2 running after fold 1 completed on GPU | Continue leakage-aware subject-wise 5-fold evaluation plus holdout. |

## Known Progress

- Dataset manifest built successfully for all seven public batches.
- Working set size: 23,808 samples, 62 subjects, 372 runs, 4 balanced motor classes.
- Full-dataset pooled legacy run has a resumable artifact tree at `kazmirfahrier/thesis-legacy-full-artifacts`.
- The pooled legacy run resumed on a Tesla P100 from epoch 8 and saved through epoch 9 before Kaggle stopped the notebook for max allowed execution duration.
- Latest pooled epoch-9 validation snapshot: accuracy 0.2453, balanced accuracy 0.2476, macro F1 0.1723. This is not a final result.
- The pooled artifact dataset `kazmirfahrier/thesis-legacy-full-artifacts` has been refreshed from the epoch-9 checkpoint and is ready for resume from epoch 10.
- Subject-wise fold 0 exists in `kazmirfahrier/thesis-7batch-artifacts`.
- Subject-wise fold 1 completed successfully on a Kaggle Tesla P100 after two GPU-compatibility fixes: setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` and using `training.deterministic: false`.
- The subject-wise artifact dataset `kazmirfahrier/thesis-7batch-artifacts` was refreshed after fold 1 and now points to `cv fold 2` as the next stage.
- The next sequential resume hop is running as `kazmirfahrier/thesis-7batch-gpucompat-runner`.

## Current Metrics Snapshot

| Experiment | Metric status |
| --- | --- |
| Original 9-subject pooled split | Historical baseline: accuracy 0.8522, MCC 0.8055, ROC-AUC 0.95, PR-AUC 0.88. |
| Full-dataset pooled split | Epoch 9 validation snapshot: accuracy 0.2453, balanced accuracy 0.2476, macro F1 0.1723. Resume from epoch 10 is ready. |
| Full-dataset subject-wise CV | Fold 0 snapshot: accuracy 0.25, balanced accuracy 0.25, macro F1 0.10. Fold 1 snapshot: accuracy 0.25, balanced accuracy 0.25, macro F1 0.10. Fold 2 is now running; final 5-fold plus holdout bundle is required before drawing conclusions. |

## Why This Matters

Phase 1 is designed to separate optimistic pooled-split performance from leakage-aware generalization. The paper should not claim the original 85% result as the final full-dataset result. The key publication-grade evidence will come from subject-wise cross-validation and held-out subject evaluation.

## Next Actions

- Monitor the active patched subject-wise run and download outputs when it stops.
- Refresh subject-wise artifacts after each completed fold and launch the next sequential account hop.
- Relaunch pooled legacy from epoch 10 after the current subject-wise hop finishes or if subject-wise is blocked by quota/errors.
- Download completed outputs into local status folders after each Kaggle session.
- Update `experiments/phase1_baselines/*.results.json` when new metrics or fold completions are available.
