# Appendix: Leakage Checks

## Leakage Control Rules
1. Subject-wise disjoint split sets only.
2. Holdout subjects are excluded from all CV folds.
3. Validation and training sample IDs derive strictly from disjoint subject sets.
4. Split generation is deterministic and serialized to JSON.

## Validation Logic
- `scripts/make_splits.py` enforces no-overlap checks.
- `assert_no_subject_leakage` raises hard errors for any overlap.
- `scripts/check_data.py` reports per subject-run missing classes for transparency.

## Reporting Policy
- Primary claims use subject-wise metrics.
- Any alternative split policy (e.g., volume-wise) must be clearly labeled as ablation only.
