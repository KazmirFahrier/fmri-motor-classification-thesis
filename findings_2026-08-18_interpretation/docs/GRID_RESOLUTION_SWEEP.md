# The Disclosed `24³` Grid Was Suboptimal

Added: 2026-08-18. Extraction: `../kaggle/gridsweep/`.
Script: `../scripts/run_grid_resolution_sweep.py`.
Results: `grid_resolution_sweep.results.json` and the completed Kaggle
`grid_all_results.json` confirmation.

The manuscript discloses three choices made with all 62 subjects visible: the temporal
averaging window, the covariance caps, and the `24^3` feature grid. Two have since been
quantified. The grid never had been, because unlike the others it cannot be tested
without re-extracting the whole cohort — which is exactly the work worth moving to
Kaggle, where three sharded kernels ran it concurrently.

**The `24^3` arm reproduces the frozen checkpoints bit-identically** (max difference
`0.000e+00`), so any difference between grids is the grid and nothing else.

## Result

`linear_svm`, 30 folds, no smoothing — a `3x3x3` box spans a different physical distance
on each grid, so applying it would confound resolution with smoothing extent.

| Grid | Features | Independent | Balanced |
| --- | ---: | ---: | ---: |
| `16^3` | 4096 | 0.7185 | 0.7612 |
| `24^3` (disclosed choice) | 13824 | 0.8098 | 0.8485 |
| `32^3` | 32768 | 0.8233 | 0.8684 |
| **`48^3`** | 110592 | **0.8447** | **0.8856** |
| **Nested selection across all four grids** | | **0.8447** | **0.8856** |

The original three-grid run selected `32^3` in 29 of 30 folds. The bounded follow-up
adds `48^3`, which is selected in **all 30 folds** and reproduces its fixed score under
nested selection.

## Two things follow

**The disclosed choice was not the best one.** `48^3` beats `24^3` by `+0.0349`
independent and `+0.0370` balanced. This is the first of the project's disclosed
cohort-visible choices that turns out to have been made *against* the authors' interest:
the temporal window was essentially optimal, and here the chosen value leaves accuracy on
the table. That direction is worth stating plainly, because a disclosed search that
happened to land on the best value invites the suspicion this one dispels.

**Nesting is again free, and for the same reason as smoothing.** This project now has
four measured points on the design-search spectrum:

| Choice | Fixed | Nested | Survives | Selection stability |
| --- | ---: | ---: | ---: | --- |
| Temporal window | +0.0054 | −0.0003 | 0% | split across 5 windows |
| ANOVA threshold | +0.0143 | +0.0060 | 42% | split across 4 thresholds |
| **Grid resolution, original bounded set** | **+0.0135** | **+0.0119** | **88%** | 29/30 folds agree |
| **Grid resolution, extended set** | **+0.0349** | **+0.0349** | **100%** | 30/30 folds choose `48^3` |
| Smoothing kernel | +0.0177 | +0.0177 | 100% | 30/30 folds agree |

The pattern is now unambiguous: **what nesting costs is the variance of the choice across
folds, not the fact that a search happened.** A parameter with a broad, stable optimum
costs nothing to have selected. That is a far more useful thing to tell a reviewer than a
general disclaimer, and this project can now demonstrate it with four points spanning 0%
to 100%.

## Consequence for the headline

`24^3` is the grid the frozen hierarchy runs on. A plain linear SVM on `48^3` reaches
`0.8447` against the hierarchy's `0.8314`, before native smoothing. The headline
reassessment already found the hierarchy's advantage over a
`smooth_3` baseline was `+0.0040` with an interval spanning zero; a better grid moves the
comparison further in the same direction.

Whether the two gains combine is measured separately, since smoothing and resolution
could easily be substitutes rather than complements — both trade spatial detail against
noise.

## Caveat

The tested range now extends through `48^3`, which is the bounded computational ceiling
for this confirmation. The optimum may lie beyond it, but further grid expansion is not
authorized on this cohort because every added choice consumes confirmatory credibility.
