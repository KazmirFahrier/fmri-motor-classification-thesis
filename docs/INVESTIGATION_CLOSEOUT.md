# Investigation Closeout

Closeout date: 2026-07-13.

## Decision

Broad discovery on the 62-subject `ds004044` cohort is closed. The project has moved to confirmation and thesis finalization. Legacy neural resumes, unrestricted architecture sweeps, and further tuning on this cohort are prohibited by the frozen protocol in `experiments/confirmation/frozen_protocol.json`.

## Frozen Claims

| Context | Frozen result | Interpretation |
| --- | ---: | --- |
| Complete balanced runs | `0.8948` balanced accuracy | Nested repetition-consistency assignment; zero target labels, exact two-per-class composition, and transductive target-run event comparison. |
| Independent/unbalanced events | `0.8314` accuracy | Nested temporal candidate selector; use for incomplete, imbalanced, or online events. |
| Conservative balanced baseline | `0.8752` balanced accuracy | Fixed `3:8` mean-window covariance hierarchy. |
| Five-run labeled personalization | `0.8913` balanced / `0.8430` independent | Validation-selected arm calibration using five labeled target-subject runs; not zero-label generalization. |
| Full legacy pooled baseline | `0.2629` validation accuracy | Stopped at epoch 25; negative evidence against the original full-dataset neural recipe. |
| Full legacy subject-wise holdout | `0.2500` accuracy | Chance-level negative control. |

## Reproducibility Commands

The raw sequence checkpoints and generated closeout bundle remain outside Git.

```bash
python scripts/run_nested_repetition_consistency_assignment.py \
  --checkpoint-dir "/path/to/continuous_event_sequence_full_cohort/subject_checkpoints" \
  --baseline-cap1024-json "/path/to/repeated_pair_family_cap1024.json" \
  --baseline-cap2048-json "/path/to/repeated_pair_family_cap2048.json" \
  --nested-selection-json "/path/to/nested_candidate_selection_repeated.json" \
  --out-json "/path/to/nested_repetition_consistency_reproduction.json" \
  --weights 0,0.25,0.5,1,2,4 \
  --inner-subject-fold-count 4 \
  --bootstrap-iterations 20000 \
  --bootstrap-seed 20260713
```

```bash
python scripts/finalize_investigation_closeout.py \
  --protocol experiments/confirmation/frozen_protocol.json \
  --checkpoint-dir "/path/to/subject_checkpoints" \
  --sequence-summary "/path/to/continuous_event_sequence_full_cohort/summary.json" \
  --legacy-original experiments/phase1_baselines/phase1_original_subset_pooled.results.json \
  --legacy-full experiments/phase1_baselines/phase1_full_dataset_pooled.results.json \
  --legacy-subjectwise experiments/phase1_baselines/phase1_subjectwise_5fold_full.results.json \
  --mean-hierarchy "/path/to/repeated_pair_family_cap1024.json" \
  --temporal-selection "/path/to/nested_candidate_selection_repeated.json" \
  --repetition-reference "/path/to/frozen_nested_repetition_consistency.json" \
  --repetition-reproduction "/path/to/nested_repetition_consistency_reproduction.json" \
  --calibration "/path/to/hierarchy_subject_calibration.json" \
  --out-dir "/path/to/investigation_closeout"
```

```bash
python scripts/make_affine_feature_stability_maps.py \
  --checkpoint-dir "/path/to/subject_checkpoints" \
  --baseline-json "/path/to/repeated_pair_family_cap1024.json" \
  --reference-image "/path/to/reference_denoised_bold.nii.gz" \
  --out-dir "/path/to/investigation_closeout/feature_stability"
```

The stability exporter reconstructs training-fold feature rankings and places their selection frequencies in a representative subject's T1w affine. These are orientation-aware model-selection maps, not normalized group anatomy, signed weights, saliency, or causal localization.

## Deployment Rule

- Use repetition consistency only for a complete run containing exactly eight events and two events from each class.
- Use the independent temporal selector for incomplete, imbalanced, or online data.
- Report labeled arm calibration in a separate personalization column.
- Report all-subject and prespecified QC-60 sensitivity estimates together; do not delete weak subjects from the primary estimate.

## External Status

No public cohort with all four exact classes and a compatible run design was identified. The candidate assessment and locked future-evaluation rules are in `docs/EXTERNAL_CONFIRMATION.md`.

## Completed Confirmation

The clean pass completed on 2026-07-13 and met every internal acceptance criterion:

- `62` checkpoints, `372` runs, and `2,976` events passed shape, finiteness, label, and run-composition checks.
- All checkpoints contain `(48, 8, 13824)` sequences under `offset_3_length_8_sequence`; their combined SHA-256 is `1a26f31440174892333add9eda8b83ea62bbed91247fdf9527e0d8133f93f166`.
- All `30` outer folds and `120` inner isolation checks passed; outer validation subjects never entered candidate or consistency-weight selection.
- The fresh repetition-consistency output is byte-identical to the frozen reference (`8c598160c39284ec21b64702fe966233125766ea5552454e2f5157f5d5e8753d`).
- Lambda zero reconstructs the independent temporal selector in all `30` folds with maximum metric difference `0.0`.
- Affine-aware stability maps were generated for coarse, leg, and arm training-fold feature rankings in `RAS` orientation. They remain representative-geometry selection-frequency maps, not atlas-localized saliency.

The tracked lightweight record is `experiments/confirmation/investigation_closeout.results.json`. The untracked closeout bundle contains `final_benchmark.{json,csv,md}`, `validation_report.json`, `artifact_manifest.json`, the fresh reproduction, and NIfTI/PNG stability-map artifacts.
