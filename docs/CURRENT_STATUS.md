# Current Status

Last updated: 2026-06-12.

This repository is tracking the active full-dataset thesis runs for 4-class motor-task fMRI classification. The unpublished manuscript and private paper PDFs are intentionally not part of this public repo.

## Active Kaggle Runs

| Experiment | Kaggle kernel | Status | Purpose |
| --- | --- | --- | --- |
| Full-dataset pooled legacy baseline | [`b6uejhvvnmiwb/thesis-legacy-full-resume`](https://www.kaggle.com/code/b6uejhvvnmiwb/thesis-legacy-full-resume) | Stopped by controlled policy at epoch 25 | Quantify the full-data pooled-split baseline for the Phase 1 leakage-gap comparison. |
| Full-dataset subject-wise evaluation | [`kazmirfahrier/thesis-7batch-gpucompat-runner`](https://www.kaggle.com/code/kazmirfahrier/thesis-7batch-gpucompat-runner) | Complete; final artifacts refreshed | Continue leakage-aware subject-wise 5-fold evaluation plus holdout. |
| Corrected clip feature-transfer diagnostic | [`b6uejhvvnmiwb/thesis-corrected-clip-baseline`](https://www.kaggle.com/code/b6uejhvvnmiwb/thesis-corrected-clip-baseline) | Version 16 running | Full-cohort `32³`, `clip_window_stride=1` diagnostic tests whether the spatial-resolution improvement continues beyond `24³`. |

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
- `kazmirfahrierlover/thesis-legacy-full-resume` was repushed as version 2 with an epoch-22 guard.
- The Lover version 2 guarded resume passed the epoch-22 guard, used a Kaggle Tesla P100, saved epochs 22 and 23, then stopped at Kaggle's max allowed execution duration.
- Epoch 22 produced the best recent validation snapshot so far: accuracy 0.2629, balanced accuracy 0.2640, macro F1 0.2584, MCC 0.0184. This is a small movement above chance, but still below the 0.30/0.30 extension threshold.
- Latest pooled epoch-23 validation snapshot: accuracy 0.2600, balanced accuracy 0.2613, macro F1 0.2564, MCC 0.0154, ROC-AUC 0.5132, PR-AUC 0.2648. This is not a final result.
- Training accuracy reached 0.5439 by epoch 23, while validation remains close to chance.
- The controlled policy checker decision after epoch 23 is `continue_short_baseline`, with 2 epochs left before the planned patience stop check unless validation crosses the 0.30/0.30 extension threshold.
- A dedicated B6-account artifact dataset, `b6uejhvvnmiwb/thesis-legacy-full-artifacts-epoch23`, was created from the verified local epoch-23 checkpoint tree.
- `b6uejhvvnmiwb/thesis-legacy-full-resume` was repushed as version 2 with an epoch-24 guard.
- The B6 version 2 guarded resume passed the epoch-24 guard, used a Kaggle Tesla P100, saved epochs 24 and 25, then stopped at Kaggle's max allowed execution duration.
- Latest pooled epoch-25 validation snapshot: accuracy 0.2629, balanced accuracy 0.2648, macro F1 0.2501, MCC 0.0206, ROC-AUC 0.5176, PR-AUC 0.2663.
- Training accuracy reached 0.5690 by epoch 25, while validation stayed close to chance and below the 0.30 accuracy / 0.30 macro-F1 extension threshold.
- The controlled policy checker decision after epoch 25 is `stop`: `epochs_since_best` is 25, `remaining_epochs_to_patience` is 0, and the validation extension threshold was not met.
- A final B6-account artifact dataset, `b6uejhvvnmiwb/thesis-legacy-full-artifacts-epoch25-final`, was uploaded from the verified local epoch-25 checkpoint tree. Kaggle accepted the upload; the private dataset may briefly show size `0` while Kaggle finishes processing the uploaded zip.
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
- Corrected clip-window audit confirmed that the class folders already contain two 8-volume event windows per class/run. Cleaned clip configs now use `hrf_shift: 0`; applying positive HRF shift inside `ClipDataset` only discards valid clips.
- Corrected temporal ResNet subject-holdout and same-subject run-holdout diagnostics stayed at chance: accuracy `0.25`, balanced accuracy `0.25`, macro F1 `0.10`.
- A BatchNorm train=validation overfit probe on `sub-01`, run `1`, reached train accuracy/F1 `1.00`, but eval-mode train=validation accuracy reached only `0.50`, indicating a train/eval mismatch under tiny batches.
- A GroupNorm temporal ResNet follow-up did not fix the corrected clip path: train=validation eval accuracy stayed at `0.25` and macro F1 at `0.10`.
- A no-normalization tiny temporal CNN also failed to overfit the same 24 corrected clips in eval mode, staying at `0.25` accuracy and `0.10` macro F1.
- Non-neural corrected-clip spatial features do show local signal. On `sub-01`, run `1`, nearest-centroid on clip-mean spatial features reached `0.75` train=eval accuracy and `0.7083` leave-one-clip-out accuracy.
- Broader corrected-clip feature-transfer diagnostic across `sub-01` through `sub-08` used 1,152 clips. Within subject-run leave-one-clip-out reached accuracy `0.7613`, balanced accuracy `0.7613`, macro F1 `0.7619`, MCC `0.6824`.
- The same simple features did not transfer well: same-subject run holdout reached accuracy `0.2604`, macro F1 `0.2553`; subject holdout reached accuracy `0.3160`, macro F1 `0.2981`.
- Transductive domain-alignment probes show that run-specific nuisance structure is a major failure mode. Same-subject run holdout improved to accuracy `0.5729`, macro F1 `0.5634` with per-run standardization plus cosine nearest centroids. Subject holdout improved to accuracy `0.4896`, macro F1 `0.4807` with per-run centering plus cosine nearest centroids.
- A reduced full-cohort 62-subject diagnostic at `16 x 16 x 16` with one non-overlapping clip per event window produced the same pattern: raw run holdout accuracy `0.2621`, raw subject holdout accuracy `0.2542`; per-run centering plus cosine improved these to `0.5565` and `0.4667` respectively.
- Local rotation sweeps on the downloaded full-cohort reduced feature matrix confirmed the effect: all six held-out-run splits averaged `0.2655` raw accuracy vs `0.5228` after per-run centering plus cosine; six subject folds averaged `0.2609` raw accuracy vs `0.4818` after the same alignment.
- Class-wise inspection of the aligned full-cohort sweeps shows that the gain is not only one-class collapse: all four classes are above chance after per-run centering plus cosine. Forearm movements are usually the easiest class, while left-leg and upper-arm movements remain weaker and more variable.
- The corrected clip feature-transfer kernel version 12 completed and reproduced the rotating full-cohort reduced results from repo commit `2a7bbbe`: all six held-out-run splits averaged `0.5228` accuracy / `0.5216` macro F1 after per-run centering plus cosine; six subject folds averaged `0.4818` accuracy / `0.4783` macro F1 after the same alignment. Outputs were downloaded to `/Users/USER/Documents/New project/status_2026-06-13_clip_domain_alignment_v12_complete/`.
- The corrected clip feature-transfer kernel version 13 completed from repo commit `a310b54`. Outputs were downloaded to `/Users/USER/Documents/New project/status_2026-06-13_clip_domain_alignment_v13_complete/`.
- The diagnostic script now also includes train-only alignment probes that estimate centering/standardization statistics from training data only. These are the next key check for whether the run-centering effect can become a publishable supervised baseline rather than only a transductive test-time adaptation diagnostic.
- Official version 13 train-only alignment found only small gains: same-subject held-out-run cosine nearest-centroid accuracy rose from `0.2641` raw to at most `0.3185` with train-subject standardization, while subject holdout stayed near chance at roughly `0.2646` best. This confirms that the large transductive gains mostly require validation-domain run statistics.
- A local variance decomposition on the same full-cohort reduced feature matrix found that raw feature variance is overwhelmingly dominated by subject and subject-run identity: class eta-squared was effectively `0.0000`, subject was about `0.9851`, and subject-run was about `0.9971`. After subject-run centering, class eta-squared rose to `0.0068`, supporting the run/subject nuisance interpretation.
- The corrected clip feature-transfer kernel version 14 completed with `clip_window_stride=1`; outputs were downloaded to `/Users/USER/Documents/New project/status_2026-06-13_clip_domain_alignment_v14_stride1_complete/`. Dense clips restored strong within-run separability: within subject-run leave-one-clip-out accuracy `0.7478`, macro F1 `0.7478`.
- Dense raw transfer stayed near chance: rotating held-out-run raw cosine mean accuracy `0.2631`, and rotating subject-fold raw cosine mean accuracy `0.2611`.
- Dense target-run adaptation improved over the sparse run: per-subject-run centering plus cosine reached rotating held-out-run mean accuracy `0.5691` / macro F1 `0.5679`, and rotating subject-fold mean accuracy `0.5201` / macro F1 `0.5171`.
- Event-window voting over overlapping clips improved the same target-run adaptation to trial-level mean accuracy `0.5833` for held-out runs and `0.5312` for subject folds.
- Dense train-only alignment remained weak: same-subject held-out-run train-subject centering reached about `0.3212` accuracy, while subject holdout remained around chance.
- A local subject-run centering shrinkage sweep on version 14 peaked sharply at `alpha=1.0`; partial centering at `alpha=0.9` and over-centering at `alpha=1.1` both collapsed back near `0.34-0.37` trial-level accuracy. This supports the hypothesis that the dominant nuisance is an additive subject-run offset.
- Version 14 subject-difficulty analysis showed wide subject-level variation after target-run centering: worst subject-fold trial accuracies included `sub-52` at `0.1875` and `sub-42` at `0.2083`, while best subjects included `sub-30` at `0.8125`. Run-level variation was much smaller, suggesting the remaining blocker is subject robustness.
- The corrected clip feature-transfer kernel version 15 completed with target shape `24 x 24 x 24`, `clip_window_stride=1`; outputs were downloaded to `/Users/USER/Documents/New project/status_2026-06-13_clip_domain_alignment_v15_full24_stride1_complete/`.
- Compared with dense `16³`, dense `24³` improved target-run adaptation: held-out-run per-subject-run centering plus cosine rose from `0.5691` to `0.5897` clip accuracy, and subject-fold rose from `0.5201` to `0.5562`.
- Event-window voting with dense `24³` reached `0.6001` held-out-run trial accuracy and `0.5654` subject-fold trial accuracy under per-subject-run centering plus cosine.
- Higher resolution did not fix raw or train-only transfer. Raw cosine stayed near `0.26`; train-subject centering reached only `0.3387` on same-subject held-out-run and `0.2701` on subject holdout.
- Weak-subject behavior persisted at `24³`: `sub-52` remained very poor at `0.1667` subject-fold trial accuracy, while best subjects such as `sub-30` and `sub-62` reached `0.75`.
- Same-subject leave-one-run event-level classification after per-run centering averaged `0.5813`, but `sub-52` and `sub-42` remained very poor (`0.1875` and `0.2083`). Removing the 10 worst subjects only raised dense `24³` adapted subject-fold trial accuracy from `0.5652` to `0.6038`, so weak subjects are important but not the whole problem.
- The corrected clip feature-transfer kernel was relaunched as version 16 with target shape `32 x 32 x 32`, `clip_window_stride=1`, and output directory `/kaggle/working/clip_domain_alignment_full32_stride1`. This tests whether the spatial-resolution gain observed from `16³` to `24³` continues.

## Current Metrics Snapshot

| Experiment | Metric status |
| --- | --- |
| Original 9-subject pooled split | Historical baseline: accuracy 0.8522, MCC 0.8055, ROC-AUC 0.95, PR-AUC 0.88. |
| Full-dataset pooled split | Stopped at epoch 25 by controlled policy. Final validation snapshot: accuracy 0.2629, balanced accuracy 0.2648, macro F1 0.2501, MCC 0.0206. Training accuracy reached 0.5690, so the model is fitting training data without meaningful validation generalization. |
| Full-dataset subject-wise CV + holdout | Complete. All five CV folds and final holdout are chance-level: accuracy 0.25, balanced accuracy 0.25, macro F1 0.10, MCC 0.0. |
| Corrected clip feature transfer | Dense full-cohort overlapping clips show strong within-run signal. Moving from `16³` to `24³` improves target-run adaptation: per-subject-run centering plus cosine reaches 0.5897 held-out-run and 0.5562 subject-fold mean clip accuracy, improving to 0.6001 and 0.5654 at event-window voting level. Raw/train-only transfer remains weak, and several subjects remain difficult. |

## Why This Matters

Phase 1 is designed to separate optimistic pooled-split performance from leakage-aware generalization. The paper should not claim the original 85% result as the final full-dataset result. The key publication-grade evidence will come from subject-wise cross-validation and held-out subject evaluation.

## Decision Policy

- Continue the pooled legacy lane only as a controlled baseline completion to early stopping.
- Stop the lane when the configured 25-epoch patience is exhausted unless validation accuracy and macro F1 both exceed 0.30.
- Do not continue toward the 200-epoch target while validation remains near chance.
- When the lane stops without meaningful validation improvement, pivot to dataset/label audit, preprocessing checks, leakage analysis, and simpler sanity-check baselines.

## Next Actions

- Do not launch another pooled legacy resume hop unless the 0.30 accuracy / 0.30 macro-F1 extension rule is explicitly overridden.
- Treat the pooled legacy lane as complete for Phase 1 baseline purposes and pivot to diagnosis.
- Continue dataset/label audit, preprocessing and normalization checks, leakage analysis, and simple sanity-check baselines.
- Prioritize run/subject nuisance control and domain-alignment diagnostics because corrected clip features classify well within runs but fail across held-out runs/subjects.
- Convert the transductive run-normalization gains into non-transductive experiments: training-only harmonization, explicit test-time adaptation protocol, run/session covariate control, or domain-invariant feature learning.
- Use the new train-only alignment probe outputs to decide whether a supervised run/subject harmonization baseline is viable before launching another neural model.
- Do not trust the current temporal ResNet corrected-clip recipe until a small eval-mode overfit probe succeeds.
- Treat the completed subject-wise result as evidence that the current full-dataset subject-wise setup is not learning; investigate data/model/label issues before making publication claims.
- Download completed outputs into local status folders after each Kaggle session.
- Update `experiments/phase1_baselines/*.results.json` when new metrics or fold completions are available.
