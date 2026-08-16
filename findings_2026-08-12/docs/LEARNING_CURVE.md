# Subject-Count Learning Curve

Added: 2026-08-12.

The project reports a single estimate at full cohort size and has never asked
whether that estimate is near a ceiling. The answer determines where the remaining
effort should go, so it is worth establishing before committing months to any
particular direction.

## Design

For each of the 30 outer folds the held-out subjects are held **fixed**, and the
training subjects are subsampled at increasing counts. Only the training side
varies, so every point on the curve is evaluated against the same target. Five
independent draws are taken at each size except the largest, where only one draw
exists by construction.

Standardisation, the dual basis, and the classifier are refitted inside every draw.
`C` is fixed rather than reselected per draw, because inner selection would itself
vary with training size and confound the curve.

## Result

Mean balanced accuracy, independent rule, 30 folds:

| Training subjects | `linear_svm` | `logistic_l2` | `correlation_centroid` |
| ---: | ---: | ---: | ---: |
| 6 | 0.6003 | 0.6075 | 0.6022 |
| 10 | 0.6656 | 0.6695 | 0.6338 |
| 16 | 0.7188 | 0.7238 | 0.6620 |
| 24 | 0.7560 | 0.7593 | 0.6748 |
| 32 | 0.7780 | 0.7786 | 0.6838 |
| 40 | 0.7956 | 0.7929 | 0.6913 |
| 46 | 0.8021 | 0.7987 | 0.6933 |
| 51 | **0.8060** | **0.8017** | **0.6936** |

Across-draw standard deviation is stable at roughly `0.045` throughout, so the
trend is not an artifact of sampling variability at small `n`.

## Marginal value of one more subject

This is the number that matters, and it collapses by a factor of twenty across the
observed range:

| Interval | `linear_svm` gain per subject | `correlation_centroid` gain per subject |
| --- | ---: | ---: |
| 6 → 10 | +0.0163 | +0.0079 |
| 10 → 16 | +0.0089 | +0.0047 |
| 16 → 24 | +0.0046 | +0.0016 |
| 24 → 32 | +0.0028 | +0.0011 |
| 32 → 40 | +0.0022 | +0.0009 |
| 40 → 46 | +0.0011 | +0.0003 |
| 46 → 51 | **+0.0008** | **+0.0001** |

At the current cohort size, **one additional subject is worth about `0.0008`
balanced accuracy** to the best linear decoder. The correlation classifier is
fully saturated at `+0.0001`.

## Extrapolation, with a caution

Fitting `accuracy = A - B·n^(-c)` to the `linear_svm` curve gives predictions of
`0.8436` at `n = 100`, `0.8707` at `n = 200`, and `0.8962` at `n = 500`.

**These should be treated as indicative only.** Power-law extrapolation beyond the
observed range is unreliable, the fitted asymptote of `0.95` is not supported by
anything in the data, and the fit is driven by the small-`n` points where the curve
is steepest. The marginal-gain table above is the robust result; the extrapolation
is not.

What the extrapolation does support, weakly, is the direction: even a *doubling* of
the cohort would be expected to buy roughly `+0.04`, and that would require
recruiting 60 more subjects with an identical four-class design.

## What this means for the programme

**Sample size is not the binding constraint.** The gap between the current `0.806`
and a perfect decoder is `0.19`, and subject count at the margin addresses roughly
`0.0008` of it per subject. Effort spent enlarging the cohort would be poorly
repaid relative to effort spent on representation and cross-subject alignment.

This has three concrete consequences:

1. **It reprioritises the open items.** Inter-subject normalization and functional
   alignment are worth more than more data. That is consistent with the finding
   that unlabeled subject-run centering alone is worth `+0.52`, while second-order
   alignment and low-rank compression are both dead ends.
2. **It reframes the external-cohort search.** An external cohort remains valuable
   for *generalization claims* — demonstrating the result transfers — but it should
   not be pursued in the expectation of higher accuracy. Those are different
   arguments and the manuscript should not conflate them.
3. **It bounds what the QC-60 sensitivity analysis can do.** Removing two weak
   subjects changes `n` by two, which this curve says is worth about `0.0016`. Any
   larger QC-60 difference is about *which* subjects were removed, not how many.

## The other axis: runs per subject

`--vary runs` keeps every training subject and subsamples the runs each contributes,
drawing independently within each subject so the curve is not confounded with any
particular run ordering. Full 30-fold result, five draws per point, `linear_svm`
independent:

| Runs per training subject | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced accuracy | 0.6968 | 0.7466 | 0.7747 | 0.7884 | 0.8007 | **0.8098** |
| Increment | — | +0.0498 | +0.0280 | +0.0138 | +0.0123 | +0.0090 |

Across-draw standard deviation is `0.041`–`0.047` throughout, comparable to the
subject axis.

### Correcting the framing this result was expected to support

A one-fold preliminary suggested the run axis was "still climbing while the subject
axis had saturated", and that a follow-up study should therefore buy runs rather than
subjects because the cheap axis was also the effective one. The full run reproduces
the raw numbers — `+0.0351` from 3 to 6 runs, close to the preliminary `+0.033` — but
**the interpretation was wrong, and it was wrong in a way worth recording.**

The two axes were being compared in incommensurable units. One extra run adds `8`
events for each of roughly `51` training subjects, so `408` events. One extra subject
adds `48` events. The run step is therefore **8.5 times more data**, and comparing
`+0.0090` per run against `+0.0008` per subject compares a large increment with a
small one, not one regime with another.

Normalising by the data actually added:

| Axis | Increment | Events added | Gain per 1000 events |
| --- | ---: | ---: | ---: |
| Subjects, 46 → 51 | +0.0040 | 240 | 0.0165 |
| Runs, 5 → 6 | +0.0090 | 408 | 0.0221 |

Per unit of data the run axis is worth about **1.3×** the subject axis, not the
order of magnitude the raw increments suggest. Both axes are decelerating, and
neither is in a qualitatively different regime from the other. The earlier claim that
one had saturated while the other had not does not survive the normalisation.

### What can still be said

Two things, both narrower than the original claim:

1. **Per unit of data, runs are modestly better than subjects** — roughly `1.3×`.
   This is a real effect and it has a plausible mechanism: extra runs improve each
   subject's own centering estimate as well as adding events, whereas an extra
   subject only adds events.
2. **On cost, runs win comfortably.** One extra run across the cohort buys `+0.0090`,
   which would take about eleven additional subjects to match. A run of an already
   enrolled subject is far cheaper than eleven recruitments, so a follow-up
   acquisition should extend sessions before it extends the roster.

Point 2 is the practically useful one and it survives intact. Point 1 is the honest
version of what the preliminary appeared to show.

## Reproduction

```bash
python findings_2026-08-12/scripts/run_subject_learning_curve.py \
  --checkpoint-dir "/path/to/subject_checkpoints" \
  --out-json "/path/to/learning_curve_30fold.json" \
  --subject-seeds 11 23 37 51 71 \
  --subject-counts 6 10 16 24 32 40 46 51 \
  --draws 5
```

Roughly 28 minutes for the full 30-fold, 8-count, 5-draw grid.
