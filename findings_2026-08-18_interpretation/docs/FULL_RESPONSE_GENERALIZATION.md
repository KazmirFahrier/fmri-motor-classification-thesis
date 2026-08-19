# The Full Response Has Two Phases, and They Disagree

Added: 2026-08-18. Scripts: `../scripts/run_temporal_generalization.py`,
`../scripts/run_response_phase_structure.py`.
Results: `temporal_generalization_full.json`, `temporal_generalization_perlag.json`,
`response_phase_structure.json`.

**This qualifies [`TEMPORAL_GENERALIZATION.md`](TEMPORAL_GENERALIZATION.md).** That
document concluded the motor code is stationary, off/on-diagonal ratio `0.9288`, and
flagged its own limitation: the frozen checkpoints begin at `offset_3`, so it described
the plateau and decay and never the rising phase. Extracting `offset 0, length 16` — the
longest window that keeps all 2976 events — settles what it could not.

## The stationarity does not extend to the whole response

| Window | Diagonal | Off-diagonal | Ratio |
| --- | ---: | ---: | ---: |
| Lags 3-10 (frozen) | 0.6791 | 0.6308 | **0.9288** |
| Lags 0-15 (full) | 0.5693 | 0.3729 | **0.6549** |

The matrix has two blocks rather than one. Lags 2-10 generalize broadly among
themselves, which is the block the earlier analysis measured. Lags 11-15 form a second
block with its own diagonal near `0.45` — well above the `0.25` chance level, so they
carry real class information — and transfer between the two blocks lands **far below
chance**: training at lag 12 and testing at lag 4 gives `0.126`, and lag 15 to lag 4
gives `0.109`.

Systematically-wrong prediction is not the signature of absent signal, which would sit
*at* chance. It is the signature of information the decoder is reading with the wrong
sign.

The rising phase, which was the original question, carries real but weaker information:
lag 0 reaches `0.367` on its own diagonal and lag 1 `0.477`, against `0.72` at the peak.

## What is ruled out

**A standardization artifact — ruled out.** The matrix carries the training lag's
standardization onto the test lag, and a mean offset between phases would shift every
decision value systematically. Re-running with each test lag standardized by its own
training-subject statistics gives an **identical** matrix: ratio `0.6549`, diagonal
`0.5693`, agreeing cell by cell to the third decimal. The effect is not an artifact of
how the decoder was transported.

**The covariance — ruled out.** A nearest-centroid rule, which depends only on where the
classes sit and not on any whitening, shows the same failure: `0.145` at lag 12 to lag 4
against the SVM's `0.123`. The effect lives in the class centroids.

## What is not settled

The natural explanation is the post-stimulus BOLD undershoot: the response dips below
baseline after the peak, and a sign-flipped spatial pattern would produce exactly this.
The evidence is genuinely mixed and should not be reported as resolved.

| Comparison | Same-class | Different-class |
| --- | ---: | ---: |
| Cross-subject, lag 12 vs lag 4 | **−0.095** | **+0.032** |
| Within-subject, peak (3-7) vs undershoot (11-15) | +0.118 | −0.039 |

Cross-subject — which is what a subject-wise decoder actually relies on — the same-class
centroid similarity is **negative** and the different-class similarity positive at the
extreme lag pairs. That is the inversion. Within subject, averaged over the phase blocks,
it is not: same-class similarity stays positive, and only 15 of 62 subjects go negative.

An earlier version of this analysis reported the undershoot account as **refuted** on the
strength of a block-averaged within-subject statistic. That was too coarse: lags 8-10 are
the response crossing back through baseline and correlate positively with both
neighbours, so including them averages the inversion away. The refutation was withdrawn,
but the confirmation has not been earned either — the within- and cross-subject measures
disagree, and this analysis cannot say why.

**The discrepancy replicates on independent data.** Running the same phase analysis on
the twelve-class extraction at the full response window gives peak-versus-undershoot
same-class similarity of `+0.1434` against `-0.0112` for different classes, with only 7
of 62 subjects negative — the same within-subject picture as the four-class data
(`+0.1178`, 15 of 62). Whatever the disagreement between the within- and cross-subject
measures reflects, it is a stable property of the data and not sampling noise in one
extraction.

## What can be said now

**The frozen window is well chosen, for a reason that was not previously known.** Lags
3-10 cover the coherent phase almost exactly. Extending the window to include lags 11-15
would *hurt*, despite those lags carrying real information, because averaging
anti-aligned patterns cancels signal. That is a concrete prediction, and it explains why
naively lengthening the window would have looked like a null.

**The earlier stationarity claim holds where it was measured, and only there.** Within
the plateau the code is stationary — the `0.9288` ratio is unchanged and correct. Across
the full response it is not. The manuscript should say the code is stationary *across the
response plateau*, which is the honest and still-useful version.

**There is unused signal.** The late block decodes at roughly `0.45` on its own diagonal.
A decoder that modelled the two phases separately rather than averaging across them could
in principle use it. Whether that is worth anything is untested and is recorded as open
rather than claimed.
