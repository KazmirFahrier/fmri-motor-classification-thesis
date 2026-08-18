# Findings — 2026-08-18: Interpretation

**Nothing in the existing repository is modified by this folder.** It adds the
interpretation results the manuscript was thinnest on, working the priority list in
[`../findings_2026-08-12/RESEARCH_PLAN.md`](../findings_2026-08-12/RESEARCH_PLAN.md)
Part 3.

All four analyses run under the **frozen 30-fold protocol** (6 outer folds x seeds
11/23/37/51/71) on the existing checkpoints, so every number here is directly
comparable with the previous round.

## Headline

**The two fine contrasts share a decision axis, and the plan predicted they would
not.** A decoder trained on left-versus-right leg reads forearm-versus-upper-arm at
`0.6525` on held-out subjects, 30/30 folds above chance, against a 200-draw
within-run permutation null at `0.5005` (z = 13.2, p < 0.005). The mapping was
predicted in advance from the RDM, not selected on the result.

## What closed

| Item | Result | Doc |
| --- | --- | --- |
| **P1** Temporal generalization | Code is **stationary**: off/on-diagonal ratio `0.9288` | [`TEMPORAL_GENERALIZATION.md`](docs/TEMPORAL_GENERALIZATION.md) |
| **P2** Decodability predictors | Split-half reliability `r = +0.607`, R² = `0.433` | [`DECODABILITY_PREDICTORS.md`](docs/DECODABILITY_PREDICTORS.md) |
| **P4** Representational geometry | `0.888` of errors are **within-pair**; coarse:fine `1.671` | [`REPRESENTATIONAL_GEOMETRY.md`](docs/REPRESENTATIONAL_GEOMETRY.md) |
| **P5** Cross-contrast transfer | `0.6525` / `0.6235`, 30/30 folds, null at chance | [`CROSS_CONTRAST_TRANSFER.md`](docs/CROSS_CONTRAST_TRANSFER.md) |
| **P7** Ribbon-masked surface | **Already answered** by existing numbers — see below | — |

### Three results that change what to do next

**The bottleneck is now precisely located.** `0.888` of the linear SVM's error budget
is within-pair, against `1/3` expected under random errors. Coarse leg-versus-arm is
solved. Nothing that improves coarse routing can help, which retrospectively explains
why the hierarchy gains so little over a flat classifier: its coarse stage solves a
problem the flat classifier was not failing.

**The two fine problems are not independent.** About 43% of the discriminative axis is
shared between them. The current hierarchy trains each pair's discriminator separately
and therefore cannot exploit this — a concrete, motivated architecture change, and the
first one this project has had a principled reason to try rather than a search over
variants.

**A previously unexplained null now has a mechanism.** The temporal-basis work found
that *weighting* lags does not help, and that was recorded without explanation. The
code is stationary, so different lags carry the same information at different
signal-to-noise; there is nothing for a weighting to discover. Averaging all eight lags
buys `+0.083` over the best single lag, and on a stationary code that gain is noise
reduction.

## P7 was already answered

The plan listed "ribbon-masked surface comparison at matched coverage" as open. It is
not: the volumetric ribbon control (`0.8177`) and the per-subject-valid surface
(`0.7460`) are **both cortex-only**, so the matched-coverage comparison already exists
in the previous round's numbers. The surface loses by `0.072` with coverage equalized —
slightly more than the `0.059` deficit on the full grid, not less.

The `Surface (intersected)` column (`0.6003`) is *not* the right matched-coverage
control, because it discards real measurements from every subject to accommodate the
worst-covered one. No re-run is needed. Recorded here so it is not attempted a third
time.

## Still open from Part 3

- **P3** Permutation null for the frozen decoder itself — still the one gap in the
  significance argument. Expensive because the upstream candidate JSONs regenerate per
  permutation. The Gram-caching trick used in the transfer null (below) may make it
  affordable.
- **P6** Spatial smoothing sweep — needs re-extraction, so it is gated on bandwidth
  rather than compute.

## A new item this round created

**Does transfer scale with somatotopic distance?** The twelve-class extraction supplies
graded within-limb positions (toe, ankle extending the leg pair; wrist, finger
extending the arm pair) and anatomically remote face conditions. That is the direct
test of whether the shared axis is somatotopic, amplitude-related, or a laterality
confound — the three accounts the transfer result cannot currently separate. This was
not among the five analyses planned for the twelve-class data and is now the
highest-value one.

## Method note worth reusing

The permutation null runs 200 draws x 30 folds in under an hour because the Gram matrix
`X X^T` **does not depend on the labels**. It is computed once per split and every
permutation indexes into it, so permutations change only which rows are training rows,
never the inner products between them. Verified bit-identical to the direct
computation. Any future permutation work in this project should use the same structure.

## Contents

```
docs/
  TEMPORAL_GENERALIZATION.md      P1: the code is stationary, and what that explains
  DECODABILITY_PREDICTORS.md      P2: reliability predicts accuracy, and its limit
  REPRESENTATIONAL_GEOMETRY.md    P4: within-pair error budget, RDM, shared-axis clue
  CROSS_CONTRAST_TRANSFER.md      P5: transfer succeeds, with its permutation null
scripts/
  run_temporal_generalization.py
  run_decodability_predictors.py
  run_representational_geometry.py
  run_cross_contrast_transfer.py
experiments/                      tracked result records
```

## Running these

They insert the repository's `scripts/` and the previous round's `scripts/` onto
`sys.path`. Run from the repository root:

```bash
python findings_2026-08-18_interpretation/scripts/run_cross_contrast_transfer.py --help
```
