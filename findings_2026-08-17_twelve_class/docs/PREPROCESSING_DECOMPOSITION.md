# Preprocessing Dominates the Decoder at Twelve Classes Too

Added: 2026-08-18. Script: `../scripts/run_twelve_class_preprocessing.py`.
Result: `twelve_class_preprocessing.results.json`.

The reassessment of the headline
([`../../findings_2026-08-18_interpretation/docs/HEADLINE_REASSESSMENT.md`](../../findings_2026-08-18_interpretation/docs/HEADLINE_REASSESSMENT.md))
leaves the paper resting on one claim: **preprocessing matters far more than decoder
architecture**. That claim had only ever been measured on four classes — and the frozen
four turn out to be an unusually confusable subset, three of them among the four
worst-decoded conditions in the dataset. A decomposition measured only there could be a
property of that subset rather than of the data.

It is not. Every stage adds one label-free step, under the frozen 30-fold protocol.

| Stage | Twelve-class | Four-class |
| --- | ---: | ---: |
| Chance | 0.0833 | 0.2500 |
| No centering | 0.0945 | 0.2860 |
| Subject-level centering | 0.5228 | 0.6294 |
| Subject-**run** centering | 0.6547 | 0.7712 |
| Plus per-lag detrending | 0.6838 | 0.8051 |
| Plus `smooth_3` | **0.7040** | **0.8275** |

| Increment | Twelve-class |
| --- | ---: |
| None → subject centering | **+0.4283** |
| Subject → subject-run centering | +0.1319 |
| Plus detrending | +0.0291 |
| Plus smoothing | +0.0202 |

## The scale of it

**Without centering, twelve-class decoding is 0.0945 against a chance of 0.0833** — very
nearly nothing. Centering in total is worth `+0.5602`, slightly *more* than the `+0.4852`
it is worth at four classes.

Set against the decoder comparison measured the same day:

| Effect | Size |
| --- | ---: |
| Unlabeled centering | +0.5602 |
| Per-lag detrending | +0.0291 |
| Spatial smoothing | +0.0202 |
| **Frozen hierarchy over a matched linear SVM** | **+0.0040** |

Twelve-class accuracy sits `0.6207` above chance. Centering accounts for **90%** of that
distance. The decoder architecture accounts for **0.6%**, and its interval includes zero.

The ordering of the preprocessing terms is preserved exactly across class counts, and
every one of them is larger than the decoder difference — smoothing, the smallest, by a
factor of five.

## A new best twelve-class number

Adding `smooth_3` lifts twelve-class accuracy from `0.6838` to **`0.7040`**, which is
**8.4x** the empirical null of `0.0832`. The smoothing benefit at twelve classes
(`+0.0202`) is close to its four-class value (`+0.0177`), so this is the same effect
rather than something specific to the harder problem.

## Why this matters for the write-up

The claim now generalizes across a threefold change in class count and a `0.6` change in
absolute accuracy, which is a much stronger basis than a single four-way problem. It also
makes the paper's structure honest: the contribution is a careful accounting of what
actually drives subject-wise decoding accuracy on this dataset, and the answer is that
almost all of it is unlabeled, test-time preprocessing rather than the classifier.

## The liability this carries, restated

Subject-run centering consults the target run. It is label-free, and the permutation
nulls at both four and twelve classes confirm it manufactures no class information —
the twelve-class null sits at `0.0832` against an analytic `0.0833`. But it is
transductive, and it is doing `90%` of the work.

Any deployment claim must therefore be stated in terms of what the method needs at test
time, not only what it achieves. The four-class analysis established that four of eight
events buy 97% of the centering benefit, which bounds that requirement usefully; the
equivalent measurement has not been made at twelve classes and is recorded as open.
