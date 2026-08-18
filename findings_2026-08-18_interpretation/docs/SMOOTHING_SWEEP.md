# Spatial Smoothing Sweep

Added: 2026-08-18. Script: `../scripts/run_smoothing_sweep.py`.
Result: `smoothing_sweep.json`.

## It needed no re-extraction

The research plan and an earlier draft of this round's README both recorded this item
as blocked on re-extracting the cohort at several kernels. **That was wrong.**
`mean_smooth` in `scripts/run_spatial_scale_feature_sweep.py` operates on the extracted
`13824`-feature vectors, reshaping them to the `24^3` grid and box-filtering there. The
entire sweep runs on the frozen checkpoints in minutes.

## Result

`linear_svm`, 30 folds, mean over all eight lags. Kernel `1` is no smoothing.

| Kernel | Independent | Balanced |
| --- | ---: | ---: |
| 1 (none) | 0.8098 | 0.8485 |
| **3** | **0.8275** | **0.8728** |
| 5 | 0.8058 | 0.8574 |
| 7 | 0.7901 | 0.8432 |
| **Nested selection** | **0.8275** | **0.8728** |

Smoothing at `3x3x3` is worth `+0.0177` independent and `+0.0243` balanced. Larger
kernels fall away quickly, and `7x7x7` is worse than no smoothing at all.

## The first parameter in this project that survives nesting intact

This matters more than the accuracy. The project has twice measured what choosing a
parameter with the whole cohort visible costs:

| Choice | Fixed (cohort-visible) | Nested | Survives |
| --- | ---: | ---: | ---: |
| ANOVA selection threshold | +0.0143 | +0.0060 | 42% |
| Temporal averaging window | +0.0054 | −0.0003 | 0% |
| **Smoothing kernel** | **+0.0177** | **+0.0177** | **100%** |

Nesting removes **nothing**, and the reason is visible in the selections: **kernel 3 is
chosen in all 30 folds**. There is no selection variance to pay for, because every
inner-fold split agrees.

That sharpens a claim the manuscript has been making loosely. The design-search penalty
is not a tax levied for having searched; it is the variance of the choice across folds.
A parameter with a broad, stable optimum costs nothing to have selected, while one with
a fold-dependent optimum costs most or all of its apparent gain. This project now has
three measured points on that spectrum — 0%, 42%, 100% — which is a far better answer
to a reviewer than a general disclaimer about design search.

## The consequence for the headline, which needs care

The frozen hierarchy's pair specialists already use `smooth_3`; the repository records
it as "the validated spatial choice" and instructs keeping it for both specialists. The
conventional-MVPA baselines this project compares against **did not** get it.

On the same 30-fold protocol:

| Decoder | Independent |
| --- | ---: |
| Frozen hierarchy (uses `smooth_3`) | 0.8314 |
| **Linear SVM with `smooth_3`** | **0.8275** |
| Linear SVM, all 8 lags, no smoothing | 0.8098 |

The hierarchy's reported advantage over conventional MVPA was `+0.026`, narrowing to
about `+0.016` once the baseline was given standard enhancements. Giving the baseline
**the same spatial preprocessing the hierarchy itself uses** leaves a point-estimate gap
of `+0.0039`.

If that holds under a paired test, the honest statement becomes: *the hierarchy's
advantage over conventional MVPA is largely attributable to a spatial preprocessing
step the baseline was not given, rather than to the decoder.* That is a materially
different claim from the one the manuscript currently makes, and it is the single most
important thing this round found.

### The paired test has now been run, and it confirms this

The frozen decoder's stored per-fold rows were compared against a `smooth_3` linear SVM
on identical folds and subjects, 20000-iteration paired bootstrap:

| | Difference | CI95 | Excludes zero |
| --- | ---: | --- | :---: |
| Original (unsmoothed baseline) | +0.0262 | `[+0.0107, +0.0426]` | Yes |
| **Preprocessing-matched** | **+0.0040** | `[-0.0119, +0.0198]` | **No** |

**The hierarchy's advantage over a linear SVM is no longer statistically
distinguishable from zero.** Full write-up and the claims that survive:
[`HEADLINE_REASSESSMENT.md`](HEADLINE_REASSESSMENT.md).

## Why smoothing helps here, briefly

The signal is distributed — three independent lines established that — so averaging
correlated neighbours raises per-feature signal-to-noise without destroying the
structure the decoder uses. The `24^3` grid already pools a large neighbourhood, which
is why the benefit saturates immediately and reverses by `5x5x5`: beyond one grid step,
smoothing starts merging genuinely distinct somatotopic territory.
