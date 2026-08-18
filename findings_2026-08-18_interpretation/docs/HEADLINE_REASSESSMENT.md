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
| Original (unsmoothed baseline) | +0.0262 | `[+0.0107, +0.0426]` | Yes | 40/1/21 |
| **Preprocessing-matched** | **+0.0040** | `[-0.0119, +0.0198]` | **No** | 37/0/25 |

Roughly **85% of the hierarchy's measured advantage was a preprocessing difference**,
not a decoder difference.

## What can and cannot now be claimed

**Cannot:** that the frozen hierarchy outperforms conventional MVPA. Against a linear
SVM with matched preprocessing there is no reliable difference, and the manuscript must
not assert one.

**Can:** that the hierarchy *matches* conventional MVPA. The point estimate still
favours it slightly and it wins on 37 of 62 subjects, so nothing here shows it is
worse. "Equivalent to a well-configured conventional baseline" is what the data
support.

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

`smooth_3` was itself selected on inner folds in all 30 of them, so the baseline's
smoothing is nested and not cohort-fitted. The hierarchy's own caps of `1024` and `2048`
remain choices made with full cohort visibility and are not nested here; if they were,
the hierarchy's number would likely fall somewhat further, so this comparison is if
anything generous to the hierarchy.

The comparison uses the frozen decoder's stored `selected_rows` rather than a fresh run.
Those rows reproduce the closeout numbers exactly (`0.83142` independent, `0.88056`
balanced) and come from the same checkpoint directory, so they are the correct frozen
reference, but no new hierarchy run was performed.
