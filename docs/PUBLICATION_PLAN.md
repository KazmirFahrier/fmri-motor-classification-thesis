# Publication Plan

Last updated: 2026-08-24.

The contribution is a leakage-aware analysis of motor-task fMRI classification on the full public `ds004044` cohort. The paper must explain which preprocessing choices create transferable signal, distinguish inductive from transductive evaluation, and report both the difficult four-class subset and the full twelve-condition task. Broad architecture discovery is closed. The remaining modeling work is a bounded confirmatory comparison of preprocessing choices under nested subject validation.

## Claims

- The historical `0.8522` pooled result is not evidence of full-cohort subject generalization. The full pooled legacy model ended at `0.2629`, and the legacy subject-wise holdout ended at `0.2500`.
- The earlier frozen hierarchy reaches `0.8314` across 30 repeated nested subject folds, but its apparent advantage over a preprocessing-matched linear SVM is only `+0.0040` with a confidence interval crossing zero. It is a historical comparator, not the primary novelty claim.
- Nested native Gaussian smoothing at `24^3` reaches `0.8639` independent balanced accuracy and `0.8959` with complete-run balanced assignment. The inner selector chooses sigma `1.1` in 10 folds and `1.4` in 20 folds. On the independent protocol, the subject-level paired gain over the frozen hierarchy is `+0.0326` with 95% interval `[+0.0159, +0.0497]`; 41 subjects improve, one ties, and 20 regress.
- Nested spatial-resolution selection across `16^3`, `24^3`, `32^3`, and `48^3` chooses `48^3` in all 30 outer folds and reaches `0.8447` independent balanced accuracy without native smoothing.
- A joint nested confirmation over the `24^3` and `32^3` native-smoothing families plus the unsmoothed `48^3` winner is the final allowed internal preprocessing comparison. Its result replaces neither prior evidence nor the final primary estimate until all three validation anchors and all 30 outer folds pass.
- The complete-run repetition-consistency decoder reaches `0.8948` only when the design supplies exactly two events per class in each run. It is a transductive assignment protocol, not ordinary independent classification.
- The fixed `3:8` mean-window hierarchy remains the conservative baseline at `0.8752`.
- Five-run target-subject arm calibration is labeled personalization and must be reported separately at `0.8913` balanced and `0.8430` independent.
- All 62 subjects define the primary estimate. QC-60 is a prespecified sensitivity analysis excluding `sub-42` and `sub-52`, never a replacement cohort.
- Twelve-class independent decoding reaches `0.6838` against an empirical within-run permutation null of `0.0832`; the preprocessing-matched `smooth_3` result reaches `0.7040`.
- Subject-run centering consults unlabeled target-run data and accounts for most of the performance gain. The manuscript must describe this as test-time transductive preprocessing and quantify its data requirement.

## Evidence Package

The final benchmark must include:

- Historical and full-cohort legacy negative controls.
- Independent, balanced-assignment, repetition-consistency, and labeled-calibration results in separate rows.
- All-subject and QC-60 estimates.
- Per-class recall, all 30 fold outcomes, all five seeds, and all 62 subject outcomes.
- Subject-bootstrap confidence intervals and paired gains against the frozen mean hierarchy and lambda-zero assignment.
- Exact data counts, split-isolation checks, reproduction tolerance, artifact checksums, and the frozen protocol hash.
- Affine-aware feature-selection stability maps with an explicit warning that representative T1w geometry is not group-normalized anatomy, saliency, or causal localization.

## External Scope

The documented search in [External Confirmation](EXTERNAL_CONFIRMATION.md) found no public cohort with compatible continuous BOLD, exact left-leg/right-leg/forearm/upper-arm labels, and the required run design. The thesis therefore describes `0.8948` and `0.8314` as strong repeated nested estimates from one public cohort, not universal external-generalization results.

If a future exact cohort becomes available, freeze its label mapping and protocol before evaluation. Run the independent decoder on any exact four-class cohort and repetition consistency only when complete runs genuinely contain two events per class. If transfer fails, reopen only the implicated timing, registration, or domain-shift diagnostic.

## Finalization Checklist

1. Generate manuscript tables directly from the consolidated benchmark JSON/CSV.
2. Add methods text distinguishing independent prediction, design-constrained transductive assignment, and labeled personalization.
3. Add the all-subject/QC-60 sensitivity table, class-recall table, fold/seed stability summary, subject distribution, and bootstrap intervals.
4. Add restrained stability-map interpretation and the two irreducible response-outlier case studies without unsupported anatomical causality.
5. Reconcile every thesis number with `experiments/confirmation/investigation_closeout.results.json` and the artifact hashes.
6. Add the nested native-smoothing and nested grid confirmations, followed by the joint nested result and paired subject-level inference.
7. Run a permutation null for the final selected conventional pipeline, recomputing all label-independent preprocessing inside each permutation where applicable.
8. Prespecify a coarse motor replication on HCP or another compatible public cohort to test the centering and smoothing claim without relabeling conditions after seeing results.
9. Add continuous integration for unit tests and a public lightweight benchmark package whose hashes match the private recoverable artifacts.
10. Do not resume legacy neural training or start an unrestricted architecture sweep.

## Repository Rule

Keep private manuscript drafts, unpublished paper PDFs, Kaggle API tokens, raw data, checkpoints, and local logs out of this public repository. Track only code, configs, lightweight result summaries, and reproducibility notes.
