# Current Status

Last updated: 2026-05-05 14:48 EDT.

This repository is tracking the active full-dataset thesis runs for 4-class motor-task fMRI classification. The unpublished manuscript and private paper PDFs are intentionally not part of this public repo.

## Active Kaggle Runs

| Experiment | Kaggle kernel | Status | Purpose |
| --- | --- | --- | --- |
| Full-dataset pooled legacy baseline | [`kazmirfahrierniloy/thesis-legacy-full-resume`](https://www.kaggle.com/code/kazmirfahrierniloy/thesis-legacy-full-resume) | Running | Quantify the full-data pooled-split baseline for the Phase 1 leakage-gap comparison. |
| Full-dataset subject-wise evaluation | [`kazmirfahrierniloy/thesis-7batch-subjectwise-gpucompat-resume`](https://www.kaggle.com/code/kazmirfahrierniloy/thesis-7batch-subjectwise-gpucompat-resume) | Running | Continue leakage-aware subject-wise 5-fold evaluation plus holdout. |

## Known Progress

- Dataset manifest built successfully for all seven public batches.
- Working set size: 23,808 samples, 62 subjects, 372 runs, 4 balanced motor classes.
- Full-dataset pooled legacy run has a resumable artifact tree at `kazmirfahrier/thesis-legacy-full-artifacts`.
- The pooled legacy run previously reached epoch 8 before Kaggle session duration stopped it.
- Subject-wise fold 0 exists in `kazmirfahrier/thesis-7batch-artifacts`; the current run should continue with fold 1.
- Kaggle live logs were not yet exposed through the output API at the time of this update, so final GPU/resume confirmation should be checked from the next log pull.

## Current Metrics Snapshot

| Experiment | Metric status |
| --- | --- |
| Original 9-subject pooled split | Historical baseline: accuracy 0.8522, MCC 0.8055, ROC-AUC 0.95, PR-AUC 0.88. |
| Full-dataset pooled split | Running; final metrics not available yet. |
| Full-dataset subject-wise CV | Running; fold 0 snapshot is weak and not a final claim. Final 5-fold plus holdout bundle is required before drawing conclusions. |

## Why This Matters

Phase 1 is designed to separate optimistic pooled-split performance from leakage-aware generalization. The paper should not claim the original 85% result as the final full-dataset result. The key publication-grade evidence will come from subject-wise cross-validation and held-out subject evaluation.

## Next Actions

- Pull Kaggle logs when available and confirm CUDA plus checkpoint resume.
- Download completed outputs into local status folders.
- Refresh the Kaggle artifact datasets after each session so the next run resumes instead of restarting.
- Update `experiments/phase1_baselines/*.results.json` when new metrics or fold completions are available.

