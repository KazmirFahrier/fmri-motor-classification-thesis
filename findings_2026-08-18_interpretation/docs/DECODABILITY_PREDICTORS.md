# What Predicts a Subject's Decodability

Added: 2026-08-18. Script: `../scripts/run_decodability_predictors.py`.
Result: `decodability_predictors.json`.

Per-subject accuracy runs from `0.2292` to `0.9708` — chance to near ceiling — with a
mean of `0.8100` and an sd of `0.1332`. The project had case-by-case forensics on the
weak subjects but no systematic model, which left the QC-60 exclusion resting on the
single argument that it was prespecified.

That argument is sound but thin, and the standard MVPA work already showed QC-60 is
**entirely a selection effect** worth `+0.024` against a `-0.0016` prediction from
sample size. A criterion that produces a selection effect needs to be defensible on
something other than having been written down early.

## Result

Four subject-level measures, each computed from the same preprocessed sequence the
decoders see. Accuracy is the mean over the five outer folds in which that subject is
held out, so a subject's own data never trains the decoder that scores it.

| Predictor | r with accuracy |
| --- | ---: |
| **Split-half pattern reliability** | **+0.6074** |
| Discriminability ratio | +0.4511 |
| Temporal stability | −0.2047 |
| Residual scale | −0.1210 |

Multiple regression on standardized predictors: **R² = 0.4329**.

| Predictor | Standardized β |
| --- | ---: |
| Split-half pattern reliability | +0.1193 |
| Discriminability ratio | −0.0403 |
| Residual scale | −0.0299 |
| Temporal stability | +0.0013 |

Reliability alone accounts for `r² = 0.369` of the variance; the other three add
`0.064` between them, and `discriminability_ratio` flips sign once reliability is in
the model, which marks it as largely redundant rather than independently informative.

Weakest five: `sub-52`, `sub-42`, `sub-17`, `sub-54`, `sub-65`.
Strongest five: `sub-32`, `sub-30`, `sub-29`, `sub-04`, `sub-25`.

`sub-52` and `sub-42` are the two subjects the project's earlier case-by-case forensics
had already flagged, which is an independent check that the measure is picking out the
same subjects a human investigator did.

## What this buys the manuscript

**The weak subjects are weak because their patterns are unreliable.** Split-half
reliability is computed from a subject's own runs and never sees the decoder, so this
is not a restatement of accuracy in other terms — it is a data-quality property that
predicts a modelling outcome. Roughly 37% of between-subject accuracy variance is
attributable to whether the subject's own class patterns replicate across their runs.

That converts the exclusion argument from procedural to substantive. The manuscript can
now say the excluded subjects differ in a measurable property of their data rather than
only that the threshold was fixed in advance.

## The limit of that argument, which must travel with it

**Split-half reliability uses the subject's own labels.** It is not a label-free QC
measure like motion or tSNR, so a pipeline that excludes on reliability is using label
information to choose its evaluation set. That is defensible when applied identically
to every subject and declared in advance, but it is **not** the same as excluding on
scanner-side quality metrics, and the manuscript must not imply that it is.

The honest formulation is the diagnostic one: reliability *explains* which subjects
decode poorly. Whether it should *select* them is a separate question this analysis
does not settle, and the safer presentation remains reporting the full cohort with
QC-60 as a secondary stratum — which is what the project already does.

## Also worth stating

**Most of the variance is still unexplained.** R² of `0.43` leaves `0.57` unaccounted
for. The checkpoints carry no motion parameters, no framewise displacement, and no
scanner-side quality metrics, so the obvious candidates could not be tested here.
Adding them would need the confounds files from the fmriprep derivatives, which is a
cheap extraction and is recorded as open work rather than done.
