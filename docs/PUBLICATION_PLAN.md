# Publication Plan

Last updated: 2026-07-13.

The contribution is a leakage-aware analysis of 4-class motor-task fMRI classification on the full public `ds004044` cohort. Broad model discovery is complete. Publication work now uses the frozen protocol and consolidated benchmark described in [Investigation Closeout](INVESTIGATION_CLOSEOUT.md).

## Claims

- The historical `0.8522` pooled result is not evidence of full-cohort subject generalization. The full pooled legacy model ended at `0.2629`, and the legacy subject-wise holdout ended at `0.2500`.
- The frozen independent decoder reaches `0.8314` across 30 repeated nested subject folds and applies to incomplete, imbalanced, or online event streams.
- The frozen complete-run decoder reaches `0.8948` only when the design supplies exactly two events per class in each complete run. It is a transductive assignment protocol using unlabeled target-run event relationships, not ordinary independent classification.
- The fixed `3:8` mean-window hierarchy remains the conservative baseline at `0.8752`.
- Five-run target-subject arm calibration is labeled personalization and must be reported separately at `0.8913` balanced and `0.8430` independent.
- All 62 subjects define the primary estimate. QC-60 is a prespecified sensitivity analysis excluding `sub-42` and `sub-52`, never a replacement cohort.

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
6. Do not resume legacy neural training, retune the frozen cohort, or start an unrestricted architecture sweep.

## Repository Rule

Keep private manuscript drafts, unpublished paper PDFs, Kaggle API tokens, raw data, checkpoints, and local logs out of this public repository. Track only code, configs, lightweight result summaries, and reproducibility notes.
