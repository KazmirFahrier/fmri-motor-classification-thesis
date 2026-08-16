# Standard MVPA Baseline

Added: 2026-08-12.

## Why this exists

Until now the covariance hierarchy had been compared only against its own variants
and a legacy neural recipe. The central claim of this project — that published
ceilings on this task are evaluation artifacts rather than modelling results —
cannot be assessed without knowing what a *correctly evaluated conventional
method* achieves. Without that number a reader cannot tell whether `0.8314` is
strong or ordinary, which matters here because arm-versus-leg somatotopic
decoding is one of the easier problems in the field.

This document supplies that reference point.

## Protocol

The baselines are held to the frozen protocol exactly, so the only thing that
differs from the hierarchy is the classifier:

- The same `2976` events and the same `(48, 8, 13824)` frozen sequence checkpoints.
- The same time-averaged event pattern, the direct analogue of a per-event GLM beta map.
- The same `30` outer subject splits (`6` folds x seeds `11, 23, 37, 51, 71`).
- The same nested inner selection (`4` inner subject folds inside outer-training subjects).
- The same two prediction rules, `independent` (argmax) and `balanced` (design-constrained assignment).
- The same subject-level bootstrap resampling.

Feature standardisation and the regularisation constant `C` are estimated on
outer-training subjects only. The script asserts split isolation on every fold:
no held-out subject may appear in any inner selection fold.

Comparators:

| Model | Description |
| --- | --- |
| `linear_svm` | `SVC` with a precomputed linear kernel, the linear SVM conventionally used for neuroimaging MVPA. |
| `logistic_l2` | L2-regularised multinomial logistic regression. |
| `correlation_centroid` | Classic Haxby-style correlation against per-class training centroids; no hyperparameter. |

### Exact dual-basis fitting

Direct optimisation on all `13824` standardised features does not converge in
usable time on real fMRI data, which is highly correlated and near-separable.
Because features far outnumber events, every L2-penalised solution lies in the
training row space, so the models are fitted in an orthonormal basis of that
space. Writing `w = V a + w_perp` gives `Xw = XV a` while
`||w||^2 = ||a||^2 + ||w_perp||^2`, so the optimum sets `w_perp = 0` and fitting
on `Z = XV` yields identical predictions.

This is an exact reparameterisation, not dimensionality reduction. The basis is
built from training rows only, and inner folds are subsets of the outer training
rows, so their row spaces are contained in it. On these data the basis has rank
`1968` rather than `13824`. Equivalence is asserted in
`findings_2026-08-12/tests/test_dual_basis_equivalence.py`, which checks argmax agreement and
score-level agreement against full-feature fits.

## Results

All values are means over the 30 outer splits, all 62 subjects.

| Model | Rule | Balanced accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| Frozen nested temporal selector | independent | **0.8314** | 0.8306 |
| `linear_svm` | independent | 0.8051 | 0.8043 |
| `logistic_l2` | independent | 0.8028 | 0.8022 |
| `correlation_centroid` | independent | 0.6963 | 0.6947 |
| Frozen nested temporal selector | balanced | **0.8806** | 0.8806 |
| `logistic_l2` | balanced | 0.8456 | 0.8456 |
| `linear_svm` | balanced | 0.8441 | 0.8441 |
| `correlation_centroid` | balanced | 0.7434 | 0.7434 |

Subject-bootstrap 95% intervals on the baselines, `n = 62`:

| Model | Rule | Mean | CI95 |
| --- | --- | ---: | --- |
| `linear_svm` | independent | 0.8054 | `[0.7688, 0.8365]` |
| `logistic_l2` | independent | 0.8030 | `[0.7659, 0.8355]` |
| `correlation_centroid` | independent | 0.6964 | `[0.6592, 0.7311]` |

### Paired comparison

Marginal intervals do not test this difference. Both decoders run on the same
splits and subjects, so the comparison is paired on per-subject mean balanced
accuracy. Positive means the frozen hierarchy is ahead.

| Comparison | Rule | Difference | CI95 | Excludes 0 | Win/tie/loss |
| --- | --- | ---: | --- | --- | --- |
| Frozen vs `linear_svm` | independent | +0.0262 | `[+0.0107, +0.0426]` | yes | 40/1/21 |
| Frozen vs `logistic_l2` | independent | +0.0286 | `[+0.0131, +0.0445]` | yes | 41/1/20 |
| Frozen vs `correlation_centroid` | independent | +0.1352 | `[+0.1139, +0.1569]` | yes | 59/1/2 |
| Frozen vs `linear_svm` | balanced | +0.0364 | `[+0.0195, +0.0538]` | yes | 41/2/19 |
| Frozen vs `logistic_l2` | balanced | +0.0351 | `[+0.0179, +0.0524]` | yes | 41/4/17 |
| Frozen vs `correlation_centroid` | balanced | +0.1374 | `[+0.1137, +0.1632]` | yes | 60/0/2 |

## Interpretation

State this plainly in the manuscript rather than letting a reviewer derive it:

- The frozen hierarchy's advantage over a conventional linear decoder is **real
  but modest**: `+0.026` independent balanced accuracy, with a paired interval
  excluding zero. It is not a large margin, and the hierarchy **loses on 21 of 62
  subjects**.
- The advantage over the classic correlation classifier is large (`+0.135`),
  which is the more meaningful methodological contrast.
- The leakage demonstration is unaffected and remains the strongest result. A
  properly evaluated conventional SVM reaching `0.8051` while the legacy pooled
  recipe collapses from `0.8522` to `0.2629`/`0.2500` under subject-wise
  evaluation makes the artifact argument *stronger*, because it shows the
  collapse is not caused by using a weak model.
- The honest framing of the contribution is therefore: a leakage-aware benchmark
  that establishes a correctly evaluated ceiling near `0.80`–`0.83` for this task,
  plus a decoder that improves modestly and reliably on standard linear MVPA
  under that protocol.

## The strictly inductive variant: the largest effect in the study

`--preprocess raw` repeats the full 30-fold protocol without the unlabeled
subject-run centering and per-lag detrending. Complete results, all 62 subjects:

| Model | Rule | Centered | Raw | Effect of centering |
| --- | --- | ---: | ---: | ---: |
| `linear_svm` | independent | 0.8051 | **0.2860** | **+0.519** |
| `logistic_l2` | independent | 0.8028 | **0.2883** | **+0.515** |
| `correlation_centroid` | independent | 0.6963 | **0.2612** | **+0.435** |
| `linear_svm` | balanced | 0.8441 | 0.6712 | +0.173 |
| `logistic_l2` | balanced | 0.8456 | 0.7201 | +0.126 |
| `correlation_centroid` | balanced | 0.7434 | 0.5316 | +0.212 |

Subject-bootstrap intervals on the raw independent estimates are tight —
`linear_svm` `[0.2762, 0.2967]`, `logistic_l2` `[0.2776, 0.3001]` — so this is not
sampling noise. Independent prediction is at or barely above the `0.25` chance
level without centering.

### Which level of nuisance structure matters

Per-feature standardisation already removes a global training mean, so the `raw`
collapse cannot be fixed by any single global offset. Adding an intermediate
condition that removes one offset per *subject* (pooling that subject's six runs)
separates subject-level from run-level structure:

| Model | Rule | Raw | Subject-centered | Frozen (subject×run) | Subject gain | Run gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `linear_svm` | independent | 0.2860 | 0.6294 | 0.8051 | **+0.3433** | **+0.1757** |
| `logistic_l2` | independent | 0.2883 | 0.6111 | 0.8028 | +0.3229 | +0.1917 |
| `correlation_centroid` | independent | 0.2612 | 0.4091 | 0.6963 | +0.1479 | +0.2872 |
| `linear_svm` | balanced | 0.6712 | 0.7703 | 0.8441 | +0.0991 | +0.0738 |
| `logistic_l2` | balanced | 0.7201 | 0.8151 | 0.8456 | +0.0950 | +0.0306 |
| `correlation_centroid` | balanced | 0.5316 | 0.6547 | 0.7434 | +0.1230 | +0.0887 |

Subject-centered bootstrap intervals, `n = 62`: `linear_svm` independent
`[0.5995, 0.6582]`, `logistic_l2` `[0.5781, 0.6442]`.

The nuisance structure is **hierarchical and neither level alone suffices**.
Removing subject offsets recovers roughly two thirds of the gap, and removing
run offsets within subject recovers most of the remainder. That matches the
physiology and physics of the acquisition: subjects differ in anatomy and
head position, and runs differ in scanner drift and repositioning.

### Relation to the earlier train-only alignment result

This is **not** a repeat of the alignment probes recorded in
`docs/DATASET_DIAGNOSTIC_FINDINGS.md`, and the two results must be read together.

That earlier work estimated centering and standardisation statistics **from the
training split only**, falling back to training-derived statistics when a domain
key was unseen. For a held-out *subject* there is by construction no
subject-specific statistic, so it fell back to a global one. It concluded that
train-only alignment "is not enough": same-subject held-out-run cosine
nearest-centroid moved only `0.2641` → `0.3044` with train-subject centering, and
subject holdout stayed near chance at about `0.2646`.

The `subject_center` condition here is the **transductive counterpart**: each
subject is centered by their own mean, computed across that subject's own runs,
including held-out subjects. No labels are used. It reaches `0.6294`.

Together the two results make a sharper claim than either alone:

> Subject-level centering is worth roughly `+0.34` independent accuracy, but
> **only when it can be estimated on the target subject**. Estimated from training
> subjects and transferred, it is worth almost nothing. The benefit is therefore
> intrinsically test-time adaptation, not a supervised preprocessing recipe.

This is the cleanest statement of the project's central methodological finding,
and it directly answers the decision point the earlier diagnostics left open.

One caveat on attribution: the `frozen` condition applies run-level centering
**and** per-lag linear detrending together, so the "run gain" column combines the
two. Separating them would need a fourth condition. The subject-versus-run
contrast is unaffected.

This is the single largest quantitative effect measured in the project, and it
must be reported prominently rather than left implicit in a preprocessing list:

- The unlabeled subject-run centering is **not** a cosmetic detail. It is worth
  roughly `+0.52` independent accuracy — twenty times the frozen hierarchy's
  `+0.026` advantage over a standard linear decoder.
- Without it, between-subject and between-run offsets dominate the feature space
  and the decision boundary tracks subject identity rather than class, so argmax
  prediction is worthless.
- Class information is still present — the balanced rule recovers `0.53`–`0.72` —
  because forcing two events per class within a run cancels the per-run offset
  that defeats argmax.
- The same step is what makes the neural lane trainable at all; see
  [Neural Lane Gate](NEURAL_LANE_GATE.md). One preprocessing operation is
  load-bearing for both lanes.

The honest ordering of contributions is therefore: **centering first**, leakage-aware
evaluation second, the hierarchy third. The headline `0.8314` depends critically
on a step that is entirely label-free but transductive, because it uses each run's
own events. A reviewer will find this; the manuscript should state it first.

This promotes Limitation 3 in the [Publication Plan](../../docs/PUBLICATION_PLAN.md) from a
caveat into a headline methodological result.

## Where the hierarchy's advantage sits, and how much of that is ceiling

The `+0.0262` paired advantage over the linear SVM is an average, and averages hide
structure. Splitting the 62 subjects at the median of SVM accuracy:

| Measure | Hard half | Easy half | Ratio |
| --- | ---: | ---: | ---: |
| Raw gain | +0.0461 | +0.0076 | 6.11 |
| Fraction of remaining headroom captured | +0.1567 | +0.0537 | 2.92 |
| Logit-scale gain | +0.3118 | +0.1841 | 1.69 |

On the raw scale this looks like a strong effect concentrated on difficult subjects —
a six-fold difference, correlation with difficulty `-0.374`. It is tempting to report
it that way, and it would be misleading.

Easy subjects sit near `0.94`, so at most `0.06` of gain is available to them; hard
subjects sit near `0.60` with `0.40` available. Some concentration is therefore
guaranteed by arithmetic regardless of what the decoder does. Normalising for that
collapses most of it: the correlation with difficulty falls from `-0.374` to `-0.126`
on the headroom scale and `-0.153` on the logit scale, and the ratio falls from
`6.1` to between `1.7` and `2.9`.

**The honest statement** is that the hierarchy has a *modest* tendency to help more
where the linear decoder struggles, worth roughly `1.7`–`2.9×` after accounting for
ceiling, not the `6×` the raw numbers suggest. The raw figure should not be quoted
without the normalisation beside it.

One detail that supports the tempered reading: of the 21 subjects where the frozen
decoder is *worse* than the SVM, **13 are in the easy half and only 8 in the hard
half**. The pattern is consistent with a method that finds room where room exists and
occasionally costs a little where it does not — not with one that specifically
rescues difficult subjects.

The two subjects excluded by the prespecified QC-60 stratum are the hardest in the
cohort: `sub-52` at `0.204` and `sub-42` at `0.267` under the SVM. The hierarchy
gains `+0.079` on each and both remain close to chance, so neither decoder recovers
them.

## The QC-60 stratum is entirely a selection effect

The frozen protocol prespecifies a QC-60 sensitivity stratum excluding `sub-42` and
`sub-52`, and insists it is "never a replacement cohort". Running the comparators on
both cohorts tests whether that caution is warranted:

| Model | Rule | All 62 | QC-60 | Delta |
| --- | --- | ---: | ---: | ---: |
| `linear_svm` | independent | 0.8051 | 0.8260 | **+0.0210** |
| `logistic_l2` | independent | 0.8028 | 0.8283 | **+0.0255** |
| `correlation_centroid` | independent | 0.6963 | 0.7147 | +0.0185 |
| `linear_svm` | balanced | 0.8441 | 0.8680 | +0.0239 |
| `logistic_l2` | balanced | 0.8456 | 0.8717 | +0.0260 |

The learning curve gives the null this needs. Removing two subjects reduces the
training set, which by that curve is worth about `-0.0016` — a small *penalty*. The
observed effect is `+0.021` to `+0.026`, an order of magnitude larger and in the
opposite direction.

**So the entire QC-60 gain is attributable to *which* two subjects were removed, not
to how many.** `sub-52` and `sub-42` are the two hardest subjects in the cohort
(`0.204` and `0.267` under the SVM, both near the `0.25` chance level), and excluding
them raises the mean by roughly what dropping two near-chance subjects from a
62-subject average arithmetically must.

This vindicates the frozen protocol's framing and sharpens what the manuscript may
say. QC-60 is a **selection** effect, not evidence of robustness, and the `+0.024`
should never be presented as an improvement attributable to the method. The honest
sentence is that two subjects contribute near-chance data and the cohort mean rises
when they are excluded, which is why the all-subject estimate remains primary.

## Second-order alignment adds very little

Given that aligning the *first* moment per subject-run is worth roughly `+0.52`, the
obvious next question is whether aligning the *second* moment adds anything. It is
applied **on top of** the frozen pipeline, so this measures the increment rather
than a substitute.

Diagonal alignment — per-subject, per-feature centering and scaling, which has no
rank problem despite only 48 events per subject:

| Model | Rule | Frozen | Frozen + diagonal alignment | Delta |
| --- | --- | ---: | ---: | ---: |
| `linear_svm` | independent | 0.8051 | 0.8141 | **+0.0090** |
| `logistic_l2` | independent | 0.8028 | 0.8145 | **+0.0116** |
| `linear_svm` | balanced | 0.8441 | 0.8466 | +0.0025 |
| `logistic_l2` | balanced | 0.8456 | 0.8521 | +0.0065 |
| `correlation_centroid` | independent | 0.6963 | 0.5529 | **-0.1434** |
| `correlation_centroid` | balanced | 0.7434 | 0.5730 | **-0.1703** |

Two things to report:

- For the discriminative linear models the gain is **real but small**, about
  `+0.01`, an order of magnitude below the `+0.52` from first-order centering and
  well under half the frozen hierarchy's `+0.026` advantage.
- It is **badly harmful to the correlation classifier**, costing `0.14`–`0.17`.
  That is coherent rather than surprising: a nearest-centroid classifier on
  L2-normalised patterns depends on relative feature amplitude, and per-subject
  variance normalisation is precisely what removes it.

So the nuisance structure this cohort suffers from is overwhelmingly **first-order**.
Once per-subject-run means are removed, what remains of the between-subject
mismatch is not well described by a diagonal covariance difference. That is a useful
negative result and it narrows where the remaining headroom can be: not in
per-feature scaling, but in spatial correspondence or in a full functional
alignment.

### Full CORAL, with the dimensionality confound removed

Full covariance whitening (CORAL) needs each subject's covariance to be estimable
from only 48 events, so it must run inside a low-rank subspace. At 24 components
that subspace retains just `0.2034` of the variance, which would cost accuracy on
its own regardless of whether the alignment helps. The naive comparison is therefore
uninterpretable, and the control is a projection onto the **same** subspace with
**no** whitening:

| Model | Rule | Frozen (13824) | PCA-24 only | CORAL-24 | Whitening effect |
| --- | --- | ---: | ---: | ---: | ---: |
| `linear_svm` | independent | 0.8051 | 0.6114 | 0.6138 | +0.0024 |
| `logistic_l2` | independent | 0.8028 | 0.6125 | 0.6135 | +0.0010 |
| `linear_svm` | balanced | 0.8441 | 0.6375 | 0.6221 | -0.0154 |
| `logistic_l2` | balanced | 0.8456 | 0.6323 | 0.6261 | -0.0062 |
| `correlation_centroid` | independent | 0.6963 | 0.6270 | 0.6051 | -0.0219 |
| `correlation_centroid` | balanced | 0.7434 | 0.6638 | 0.6279 | -0.0359 |

**Whitening contributes nothing.** The isolated effect ranges from `+0.002` to
`-0.036`, straddling zero and trending negative. The entire `-0.19` apparent drop
was the dimensionality reduction, not the alignment. Together with the diagonal
result this closes second-order alignment as a direction: neither the diagonal nor
the full covariance form helps once the confound is removed.

### A separate finding: the signal is high-dimensional

The control carries information the CORAL question did not ask for. Compressing
13824 features to the top 24 principal components — a fifth of the total variance —
costs roughly `0.19` independent accuracy. **The discriminative information is not
concentrated in the leading principal components.** It is distributed across many
low-variance directions.

That is worth reporting in its own right. It argues against low-rank and
strong-compression approaches for this task generally, and it is consistent with
why the frozen hierarchy's comparatively generous feature caps of `1024` and `2048`
were selected over tighter ones.

Repeating both conditions at `k = 40` confirms the conclusion and sharpens the
second point:

| Model | Rule | PCA-40 | CORAL-40 | Whitening effect |
| --- | --- | ---: | ---: | ---: |
| `linear_svm` | independent | 0.6160 | 0.6202 | +0.0042 |
| `logistic_l2` | independent | 0.6006 | 0.6246 | +0.0240 |
| `linear_svm` | balanced | 0.6424 | 0.6275 | -0.0149 |
| `logistic_l2` | balanced | 0.6265 | 0.6393 | +0.0128 |
| `correlation_centroid` | independent | 0.6290 | 0.6036 | -0.0254 |
| `correlation_centroid` | balanced | 0.6655 | 0.6225 | -0.0431 |

Whitening still straddles zero with no consistent sign, at either rank.

The dimensionality curve is the striking part:

| Components | Retained variance | `linear_svm` independent |
| ---: | ---: | ---: |
| 24 | 0.2034 | 0.6114 |
| 40 | 0.2736 | 0.6160 |
| 13824 (all) | 1.0000 | **0.8051** |

Adding 16 components and a third more variance buys `+0.005`. The gap to the full
feature space is `0.19`. **The class information is essentially absent from the
leading principal components**; the top of the spectrum is dominated by structure
that does not discriminate, and the discriminative directions are numerous and
individually low-variance.

This is a substantive statement about the nature of the signal in this dataset, and
it predicts that any method resting on aggressive linear compression — PCA
pipelines, low-rank factorisations, small bottlenecks — will underperform here.

## Reproduction

```bash
python findings_2026-08-12/scripts/run_standard_mvpa_baseline.py \
  --checkpoint-dir "/path/to/continuous_event_sequence_full_cohort/subject_checkpoints" \
  --out-json "/path/to/standard_mvpa_frozen_preproc.json" \
  --preprocess frozen
```

```bash
python findings_2026-08-12/scripts/compare_frozen_vs_standard_mvpa.py \
  --frozen-json "/path/to/nested_candidate_selection_repeated.json" \
  --baseline-json "/path/to/standard_mvpa_frozen_preproc.json" \
  --out-json "/path/to/frozen_vs_standard_paired.json"
```

`--preprocess raw` repeats the comparison without the unlabeled subject-run
centering and per-lag detrending, giving a strictly inductive conventional
pipeline for readers who object to any transductive step.

The tracked lightweight record is `findings_2026-08-12/experiments/standard_mvpa.results.json`.
