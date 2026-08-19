# Temporal Generalization: the motor code is stationary

Added: 2026-08-18. Script: `../scripts/run_temporal_generalization.py`.
Result: `temporal_generalization.json`.

Every analysis in this project collapses the eight lags by averaging. That presumes
the code is the same thing at every lag, and the presumption had never been tested.
Training at lag *i* and testing at lag *j* for all 64 pairs (King & Dehaene, 2014)
tests it directly.

## The matrix

Rows are the training lag, columns the test lag; `linear_svm`, independent rule,
30 folds. Standardization comes from the training lag and is carried over unchanged,
so the decoder is transported across time rather than refitted.

|  | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **L0** | 0.659 | 0.682 | 0.686 | 0.688 | 0.675 | 0.629 | 0.557 | 0.400 |
| **L1** | 0.669 | 0.713 | 0.716 | 0.710 | 0.701 | 0.675 | 0.614 | 0.436 |
| **L2** | 0.670 | 0.714 | **0.721** | 0.715 | 0.705 | 0.673 | 0.612 | 0.444 |
| **L3** | 0.677 | 0.719 | 0.719 | **0.722** | 0.715 | 0.676 | 0.615 | 0.457 |
| **L4** | 0.659 | 0.706 | 0.711 | 0.717 | 0.714 | 0.676 | 0.616 | 0.458 |
| **L5** | 0.625 | 0.681 | 0.694 | 0.690 | 0.687 | 0.697 | 0.661 | 0.491 |
| **L6** | 0.593 | 0.659 | 0.656 | 0.673 | 0.654 | 0.671 | 0.677 | 0.522 |
| **L7** | 0.462 | 0.537 | 0.562 | 0.566 | 0.557 | 0.585 | 0.604 | 0.529 |

| | Value |
| --- | ---: |
| Diagonal mean | 0.6791 |
| Off-diagonal mean | 0.6308 |
| **Ratio** | **0.9288** |

The ratio was `0.9293` on a single fold and `0.9288` on all thirty, so this is a stable
property of the data rather than a fold-level accident.

## What it says

**The code is stationary.** Off-diagonal generalization is 93% of on-diagonal. A
decoder trained at lag 3 reads lag 1 at `0.719` against its own diagonal of `0.722` —
a loss of `0.003`. Lags 1 through 4 form a plateau in which every lag reads every other
at `0.70`–`0.72`. The matrix is also near-symmetric: training at *i* and testing at *j*
gives about what training at *j* and testing at *i* gives, which is what a single
persistent pattern predicts and a sequence of evolving patterns does not.

The edges behave exactly as haemodynamics predicts. Lag 0 is slightly weak because the
response is still rising; lags 6 and 7 fall away as it decays, and lag 7 barely reaches
`0.529` even on its own diagonal.

## This explains a previously unexplained negative result

The temporal-basis investigation asked whether *weighting* the lags helps and found it
did not. That was recorded as a null without a mechanism.

The mechanism is here. Weighting can only help when different lags carry different
information. On a stationary code they carry the *same* information at different
signal-to-noise, so the optimal combination is close to a plain average, and there is
nothing for a learned weighting to discover. The null was not a failure of the method;
it was the correct answer to a question whose answer was already determined by the
structure of the data.

## And it justifies the window averaging

Averaging all eight lags reaches `0.8051`; the best single lag reaches about `0.722`.
Averaging buys `+0.083`, which is far more than any decoder change in this project.

On a stationary code, averaging lags is averaging repeated noisy measurements of one
pattern, and the gain is noise reduction. That is the reassuring reading: the window
mean is not a lossy shortcut that happens to work, it is close to the right estimator
for this data — which is the same conclusion the beta-series comparison reached from
the opposite direction, where LSS paid for flexibility this design does not need.

## What a reviewer gets from this

The manuscript is thinnest on interpretation, and this is an interpretation result
that costs nothing to defend: it uses the frozen protocol, the frozen data, and a
standard published design. It converts two of the project's choices — window averaging
and the negative temporal-basis result — from unexplained into explained.

## Caveats

The lags are `offset_3` onwards, so this describes the plateau and decay of the
response, not its onset. A window starting at the stimulus would be needed to say
anything about the rising phase, and that requires re-extraction.

> **Update, same day.** That re-extraction has been done — `offset 0, length 16`, the
> longest window keeping all 2976 events. **The stationarity does not extend to the full
> response.** Over lags 0-15 the ratio falls from `0.9288` to `0.6549`, and the matrix
> has two blocks whose cross-transfer lands far below chance (`0.126` training at lag 12,
> testing at lag 4). Everything on this page remains correct **for the plateau**, which is
> what it measured; the claim should be stated as "stationary across the response
> plateau" rather than without qualification. See
> [`FULL_RESPONSE_GENERALIZATION.md`](FULL_RESPONSE_GENERALIZATION.md).

The stationarity is measured at the resolution of a 2 s TR on a 16 s block design. It
does not rule out faster dynamics that this sampling cannot see, and no claim about
sub-TR dynamics should be made from it.
