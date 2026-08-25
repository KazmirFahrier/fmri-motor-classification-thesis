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
| **Nested sigma selection** | **0.8639** | **0.8959** |

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

## Nested confirmation and paired inference

The queued nested run is complete. Sigma is selected only inside four training-subject
folds for each of the 30 outer folds. Sigma `1.1` is selected 10 times and sigma `1.4`
20 times, producing `0.8639` independent and `0.8959` complete-run balanced accuracy.
The nesting cost relative to the best fixed sigma is only `0.0009` independently.

The archived frozen hierarchy subject rows were also recovered. A paired 20,000 draw
subject bootstrap gives an independent gain of `+0.0326`, with interval
`[+0.0159, +0.0497]`; 41 subjects improve, one ties, and 20 regress. In the QC-60
sensitivity stratum the gain is `+0.0353`, with interval `[+0.0180, +0.0527]`.

The complete-run balanced gain is smaller at `+0.0152`, with interval
`[-0.0007, +0.0317]`. Therefore the conventional native-smoothed linear SVM is now the
primary independent internal result, while the complete-run comparison remains
inconclusive.

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
