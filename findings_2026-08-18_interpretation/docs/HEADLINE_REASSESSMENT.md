# The Hierarchy's Advantage Does Not Survive a Preprocessing-Matched Baseline

Added: 2026-08-18. Scripts: `../scripts/run_smoothed_baseline.py`,
`../../findings_2026-08-12/scripts/compare_frozen_vs_standard_mvpa.py`.
Result: `frozen_vs_smoothed_mvpa.json`.

**This supersedes the headline comparison in
[`../../findings_2026-08-12/docs/STANDARD_MVPA_BASELINE.md`](../../findings_2026-08-12/docs/STANDARD_MVPA_BASELINE.md).**
It does not modify that document; the numbers there remain correct for the baseline as
it was configured, which is the point.

## What was wrong with the comparison

The frozen hierarchy's pair specialists apply `smooth_3`. The repository's own status
notes record it as "the validated spatial choice" and instruct keeping it for both
specialists. The conventional-MVPA baselines built to test the hierarchy **were never
given it**.

The reported advantage of `+0.026` therefore compared a decoder *plus a spatial
preprocessing step* against a decoder *without it*, and attributed the whole difference
to the decoder.

## The corrected comparison

Paired subject-level bootstrap, 20000 iterations, on **identical folds and subjects** —
the frozen decoder's own `selected_rows` against a baseline differing only in receiving
`smooth_3`. Positive means the hierarchy is ahead.

| Rule | Baseline | Frozen | Baseline | Difference | CI95 | Excludes zero |
| --- | --- | ---: | ---: | ---: | --- | :---: |
| independent | `linear_svm` | 0.8317 | 0.8277 | **+0.0040** | `[-0.0119, +0.0198]` | **No** |
| independent | `logistic_l2` | 0.8317 | 0.8175 | +0.0142 | `[-0.0017, +0.0298]` | No |
| balanced | `linear_svm` | 0.8809 | 0.8729 | **+0.0080** | `[-0.0066, +0.0224]` | **No** |
| balanced | `logistic_l2` | 0.8809 | 0.8623 | +0.0186 | `[+0.0018, +0.0356]` | Yes |

Against the strongest baseline — a linear SVM, the conventional choice in neuroimaging
MVPA — **the difference is not statistically distinguishable from zero under either
prediction rule.**

### Before and after

| | Difference | CI95 | Excludes zero | Subject w/t/l |
| --- | ---: | --- | :---: | --- |
| Original (unsmoothed, `24^3`) | +0.0262 | `[+0.0107, +0.0426]` | Yes | 40/1/21 |
| **Preprocessing-matched** (`24^3` + `smooth_3`) | **+0.0040** | `[-0.0119, +0.0198]` | **No** | 37/0/25 |
| **Best conventional** (`32^3` + `smooth_3`) | **−0.0110** | `[-0.0297, +0.0094]` | **No** | 26/0/36 |

Roughly **85% of the hierarchy's measured advantage was a preprocessing difference**,
not a decoder difference.

### The grid sweep extends this, and flips the sign

`24^3` is the grid the hierarchy runs on, and it was one of the manuscript's disclosed
cohort-visible choices. Sweeping it found it **suboptimal**: `32^3` is selected in 29 of
30 folds and is worth `+0.0135` on its own
([`GRID_RESOLUTION_SWEEP.md`](GRID_RESOLUTION_SWEEP.md)).

Giving the baseline both improvements — the finer grid and the smoothing, each chosen by
nested selection rather than cohort-wide — puts a plain linear SVM at **`0.8423`** against
the hierarchy's `0.8314`, winning on **36 of 62 subjects**.

**That is still not a reliable win for the baseline.** The paired interval
`[-0.0297, +0.0094]` spans zero, exactly as the matched comparison did. The correct
statement is not that conventional MVPA beats the hierarchy; it is that across three
progressively fairer comparisons the difference goes from significant, to nil, to
slightly negative, and **only the first was significant** — and that one was the unfair
one.

| Configuration | `linear_svm` independent |
| --- | ---: |
| `24^3`, no smoothing (original baseline) | 0.8098 |
| `24^3` + `smooth_3` | 0.8275 |
| `32^3`, no smoothing | 0.8233 |
| **`32^3` + `smooth_3`** | **0.8423** |
| Frozen hierarchy (`24^3`, `smooth_3`, covariance caps) | 0.8314 |

The two preprocessing gains **combine rather than substitute**, which was not obvious:
both trade spatial detail against noise, so they could easily have been redundant.

## What can and cannot now be claimed

**Cannot:** that the frozen hierarchy outperforms conventional MVPA. Against a linear
SVM with matched preprocessing there is no reliable difference, and against a
better-configured one the point estimate favours the baseline. The manuscript must not
assert an advantage.

**Cannot, in the other direction either:** that conventional MVPA beats the hierarchy.
The `−0.0110` interval spans zero. Claiming a baseline win would repeat the original
error with the sign reversed.

**Can:** that the hierarchy *matches* conventional MVPA and is not distinguishable from
it in either direction. "Equivalent to a well-configured conventional baseline" is what
the data support, and it survives all three comparisons.

**Can:** that the hierarchy retains a reliable advantage over `logistic_l2` under the
balanced rule (`+0.0186`, CI excludes zero). This is a narrower claim about one
comparator and one rule, and should be presented as such rather than generalised.

**Still stands, and is strengthened:** everything about preprocessing. Unlabeled
subject-run centering is worth `+0.52`, per-lag detrending `+0.034`, and `smooth_3`
`+0.0177` — all far larger than any decoder difference. The project's most robust
finding was always that **preprocessing dominates the decoder**, and this result is the
sharpest instance of it rather than a contradiction.

## Why this is a better paper, not a worse one

The external review's first objection was that no standard MVPA baseline existed, which
made the central claim unfalsifiable. The baseline now exists, is preprocessing-matched,
and the claim it was built to test **did not survive**. That is what a real control is
for, and reporting it is the difference between a methods paper and an advertisement.

The result also reframes the contribution. This is no longer "a novel hierarchy beats
standard methods" — a claim a reviewer would rightly attack given the disclosed design
search over roughly 74 scripts. It is "on a twelve-condition somatotopic dataset,
subject-wise decoding reaches `8.2x` chance, the representational geometry recovers the
homunculus, and accuracy is dominated by transductive preprocessing rather than by
decoder architecture — including our own." The second claim is more defensible, more
useful, and better supported by the evidence assembled here.

## Caveats

`smooth_3` was itself selected on inner folds in all 30 of them, and `32^3` in 29 of 30,
so neither of the baseline's improvements is cohort-fitted. The hierarchy's own caps of
`1024` and `2048` remain choices made with full cohort visibility and are not nested
here; if they were, the hierarchy's number would likely fall somewhat further.

**But the third comparison is unfair in the opposite direction, and this must be stated.**
The baseline was given a better grid; the hierarchy was not. It has only ever been run on
`24^3`, and there is no reason to think it would fail to benefit from `32^3` as the
baseline did — the gain is a property of the representation, not of the classifier. A
symmetric comparison would re-run the hierarchy's full nested candidate-selection
pipeline on `32^3`, which has not been done.

That is why the conclusion drawn here is **equivalence**, not a baseline win. The second
comparison — `24^3` with `smooth_3`, where both sides use the same representation and
differ only in the decoder — is the one that carries the argument, and it lands at
`+0.0040` with an interval spanning zero. The third comparison shows the difference is
not robust to configuration choices the hierarchy was never given the chance to make,
which is a weaker and more careful claim than "the baseline is better".

The comparison uses the frozen decoder's stored `selected_rows` rather than a fresh run.
Those rows reproduce the closeout numbers exactly (`0.83142` independent, `0.88056`
balanced) and come from the same checkpoint directory, so they are the correct frozen
reference, but no new hierarchy run was performed.
