# Findings — 2026-08-12 to 2026-08-16

Additive round of publication-readiness work. **Nothing in the existing repository
is modified by this folder.** Where these findings imply revisions to
`README.md`, `docs/CURRENT_STATUS.md`, `docs/INVESTIGATION_CLOSEOUT.md`, or
`docs/PUBLICATION_PLAN.md`, the revisions are written up in
[`docs/PROPOSED_UPDATES_TO_EXISTING_DOCS.md`](docs/PROPOSED_UPDATES_TO_EXISTING_DOCS.md)
and supplied as a patch, not applied.

## Why this round

Three gaps stood between the frozen result and a Q1 neuroimaging submission: no
conventional MVPA comparator, no inter-subject spatial normalization, and a neural
negative control whose numbers were the signature of a broken training path rather
than an architecture result. All three are now closed, and the work that followed
opened several more questions and answered most of them.

## Headline results

### The comparator the central claim needed

| Decoder | Independent | Balanced |
| --- | ---: | ---: |
| Frozen hierarchy | **0.8314** | **0.8806** |
| Ensemble (inner-selected) | 0.8157 | 0.8656 |
| Nested ANOVA + linear SVM | 0.8111 | 0.8606 |
| Linear SVM | 0.8051 | 0.8441 |
| Logistic L2 | 0.8028 | 0.8456 |
| Corrected CNN (32ch, tuned) | 0.8030 | 0.8580 |
| Beta-series LSS + SVM | 0.7948 | 0.8431 |
| Correlation centroid | 0.6963 | 0.7434 |

The hierarchy's paired advantage over a linear SVM is `+0.0262`, CI95
`[+0.0107, +0.0426]`, winning on 40 of 62 subjects. Real, reliable, and modest. With
standard enhancements the gap narrows to about `0.016`.

### The dominant ingredient is preprocessing, not the decoder

| Condition | Independent |
| --- | ---: |
| No centering | 0.2860 |
| Subject-level centering (transductive) | 0.6294 |
| Subject-run centering | 0.7712 |
| Plus per-lag detrending | 0.8051 |

Unlabeled subject-run centering is worth roughly `+0.52`, twenty times the
hierarchy's advantage. Detrending adds a further `+0.034`. Prior work had already
shown the *inductive* version of centering is worth almost nothing, so the benefit is
intrinsically test-time adaptation.

### The preprocessing does not manufacture signal

200 within-run label permutations across 30 folds put every null at chance
(`0.2497`–`0.2504`), all `p < 0.005`, z of `53`–`79`. This is the direct answer to the
strongest methodological objection the project faces.

### The neural conclusion reverses

`0.2500` was a training-path artifact with three compounding causes: BatchNorm running
statistics, uncentered inputs, and global average pooling that discards the spatial
position somatotopic decoding depends on. Corrected, the same architecture reaches
`0.7913` over 30 folds — competitive with conventional MVPA, not useless.

### Surface normalization does not help, and coverage is not why

| Result | Value |
| --- | ---: |
| Surface advantage *within* subject | +0.042 |
| Surface pattern reliability advantage | +0.068 |
| Surface deficit *across* subjects | −0.059 |
| Recovered by connectivity hyperalignment | +0.033 |
| Cortex-only volumetric (ribbon-masked) | **0.8177**, above the full grid |

The surface representation is cleaner by two independent measures and still transfers
worse. The ribbon-masked control refutes the coverage explanation: restricting the
volumetric decoder to cortex costs nothing. What remains is correspondence itself —
**anatomy-based surface normalization provides no cross-subject advantage over a
bounding-box rescale on this cohort.**

### Scale and design

- **Sample size is saturated**: one more subject is worth `+0.0008`. Per unit of data
  runs are worth about `1.3x` subjects, and on cost runs win comfortably.
- **All 372 runs are palindromic.** No repetition suppression. Within-run repeats are
  `2.5x` more similar than the same class across runs, which quantifies what the
  repetition-consistency decoder exploits — and shows it is largely run-state, not
  class identity.
- **QC-60 is entirely a selection effect**: `+0.024`, against a `-0.0016` prediction
  from sample size alone. It must never be presented as a method improvement.
- **The class signal is high-dimensional**: the top 40 principal components, over a
  quarter of the variance, recover only `0.616` of `0.805`.

## Contents

```
docs/
  RESEARCH_COVERAGE_MAP.md               live ledger: tried / open, verified against all 129 commits
  STANDARD_MVPA_BASELINE.md              comparators, centering decomposition, QC-60, alignment ablations
  NEURAL_LANE_GATE.md                    eval-mode gate, three faults, corrected 30-fold result
  INTER_SUBJECT_NORMALIZATION.md         surface pipeline, hyperalignment, ribbon-masked control
  PERMUTATION_NULL.md                    label-shuffled null
  LEARNING_CURVE.md                      subject and run axes
  CEILING_AND_DESIGN_STRUCTURE.md        reliability, the invalid ceiling, palindromic design
  DECODER_IMPROVEMENTS.md                ANOVA selection, ensembling, centering requirement
  PROPOSED_UPDATES_TO_EXISTING_DOCS.md   revisions to existing docs, described but not applied
  proposed_edits_to_existing_docs.patch  the same revisions as a git patch
experiments/    seven lightweight tracked result records
figures/        surface_alignment_check.png
scripts/        20 analysis scripts
tests/          test_dual_basis_equivalence.py
```

## Running these scripts

They live outside the repository's `scripts/` directory, so each inserts the
repository's `scripts/` and `src/` onto `sys.path` at import. Run from the repository
root:

```bash
python findings_2026-08-12/scripts/run_standard_mvpa_baseline.py --help
```

```bash
python -m pytest findings_2026-08-12/tests/ -q
```

## Nothing beat the frozen decoder

Every enhancement tried lands below `0.8314`: ensemble `0.8157`, nested ANOVA
selection `0.8111`, tuned CNN `0.8030`, beta-series LSS `0.7948`. The hierarchy's
advantage over plain conventional MVPA narrows from `+0.026` to about `+0.016` once
the baseline is given standard enhancements, but it does not disappear.

## Still open

- **Permutation null for the frozen decoder itself** — only the comparators were
  permuted. Its nested pipeline would have to run inside each permutation.
- **A valid noise ceiling** — the within-subject construction was invalid, and the
  learning curve suggests a ceiling may be genuinely ill-defined here, since more
  subjects keep helping.
- **HCP leakage replication** — needs a data agreement, and is a generalisation claim
  rather than an accuracy one.
- **Ribbon-masked surface comparison at matched coverage** — the volumetric ribbon
  control is done; the reciprocal surface restriction is not.
