# Smoothing at Native Resolution Is Worth Far More Than the Post-Hoc Filter

Added: 2026-08-18. Extraction: `../kaggle/nativesmooth/`.
Analysis: `../kaggle/nativesmooth_analysis/`. Result: `nativesmooth_results.json`.

The project applies spatial smoothing as a `3x3x3` box filter over the already-resampled
`24^3` grid. Standard neuroimaging practice is a Gaussian kernel at the **acquired**
resolution, before any downsampling. The distinction matters because the `24^3`
resampling already averages a large neighbourhood, so the post-hoc filter smooths
something twice while a native Gaussian smooths the data once, at the scale the noise
actually lives.

## Result

`linear_svm`, 30 folds. Sigma is in voxels of the acquired volume.

| Condition | Independent | Balanced |
| --- | ---: | ---: |
| Unsmoothed (frozen pipeline) | 0.8098 | 0.8486 |
| Post-hoc `box3` (current approach) | 0.8275 | 0.8728 |
| Native Gaussian, sigma 0.7 | 0.8545 | 0.8865 |
| Native Gaussian, sigma 1.1 | 0.8638 | 0.8943 |
| **Native Gaussian, sigma 1.4** | **0.8648** | **0.8969** |
| Native sigma 1.4 **plus** `box3` | 0.8475 | 0.8867 |

Native smoothing is worth **`+0.0550`** over no smoothing and **`+0.0373`** over the
post-hoc box filter the project currently uses. That makes it the second-largest effect
ever measured here, behind only subject-run centering (`+0.52`) and ahead of detrending
(`+0.034`), the post-hoc filter (`+0.018`), the grid (`+0.014`) and every decoder
difference (all under `+0.03`, most under `+0.01`).

## The two smoothings are substitutes, not complements

Applying the box filter **on top of** native smoothing makes things **worse** — `0.8475`
against `0.8648` for native alone. Smoothing twice over-smooths, and the second
application destroys more than it denoises. This is a useful negative: it rules out
simply stacking the two, and it means the pipeline should *replace* its post-hoc filter
rather than add to it.

## Protocol validation

Reproducing the frozen protocol inside a Kaggle kernel is the obvious place for silent
drift, since fold assignment is a seeded shuffle of the sorted subject list and any
change in ordering changes every fold. The analysis therefore carries a hard gate: its
sigma-0 arm is the frozen pipeline exactly and must reproduce `0.8098` or the run aborts.

It reproduced `0.8098` to four decimals. Every other number here rests on the same
verified code path.

## Two caveats that must travel with this

**The sigma was chosen with all 30 folds visible.** This is precisely the cohort-visible
design search the project has spent considerable effort measuring elsewhere, and it has
*not* been nested here. Based on the four measured points, the cost of nesting tracks how
stable the choice is across folds — `0%` surviving for the temporal window, `42%` for the
ANOVA threshold, `88%` for the grid, `100%` for the box kernel. Accuracy is still rising
at the largest sigma tested, so the optimum has not been bracketed and the selection could
well be unstable. **The `0.8648` figure should be treated as an upper bound until a
nested run exists**, which is queued.

**No paired test against the frozen hierarchy is possible right now.** The comparison
would be `0.8648` against the hierarchy's `0.8314`, but the hierarchy's per-subject rows
were stored outside the repository and have since been deleted along with the rest of the
`status_*` directories. Only the aggregate results of the earlier paired tests were
committed. Regenerating those rows needs the hierarchy's upstream candidate stages, which
have not been ported. Until then this is a **point-estimate comparison, not a paired
one**, and the earlier paired results stand as the rigorous evidence.

For scale: at a difference of `+0.0334`, and given the earlier paired intervals had
half-widths near `0.016`–`0.020`, a paired test would very likely exclude zero. That is a
reasonable expectation, not a result, and the manuscript should not use it as one.

## What it does to the picture

Every fair comparison so far has narrowed or reversed the hierarchy's apparent advantage,
and each time the cause was preprocessing the baseline had been denied:

| Baseline configuration | `linear_svm` independent |
| --- | ---: |
| `24^3`, unsmoothed (original comparison) | 0.8098 |
| `24^3` + post-hoc `box3` | 0.8275 |
| `32^3` + post-hoc `box3` | 0.8423 |
| **`24^3` + native Gaussian sigma 1.4** | **0.8648** |
| Frozen hierarchy | 0.8314 |

The consistent finding — that accuracy on this dataset is dominated by unlabeled
preprocessing rather than by decoder architecture — is reinforced rather than complicated.
Whether the grid and the native smoothing combine is being extracted now; they act on
different axes, but the box filter's behaviour above shows that two smoothing-like
operations can easily be substitutes.
