# Ceiling Estimates and Design Structure

Added: 2026-08-13.

Two questions with no prior answer in the project: how much accuracy the data can
support at all, and whether the two presentations of each class within a run differ.

---

## 1. The within-subject ceiling estimate, and why it is not a ceiling

**Split-half pattern reliability.** Correlating each subject's per-class mean response
from odd runs against even runs gives `r = 0.4561`, or `0.5983` after Spearman-Brown
correction to full length. Class patterns are therefore only moderately reliable at
six runs per subject.

**Centroid-count extrapolation.** Classifying each event against centroids built from
`k` of that subject's other runs:

| Runs of centroid data | 1 | 2 | 3 | 4 | 5 | extrapolated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Within-subject accuracy | 0.5731 | 0.6351 | 0.6642 | 0.6897 | 0.6969 | **0.7217** |

### The estimate does not do what it was built to do

`0.7217` was intended as an upper bound on achievable accuracy. It cannot be, because
the **cross-subject decoders already exceed it**: linear SVM reaches `0.8051` and the
frozen decoder `0.8314`.

The premise was wrong in two ways. The extrapolation is over a subject's own runs, of
which there are at most five, so it estimates the asymptote of a decoder limited to
*one subject's data* — a data limit, not a noise limit. And it uses a nearest-centroid
rule, which is the weakest decoder in the project (`0.6963` cross-subject), so the
asymptote is specific to that classifier rather than to the data.

So this is not a noise ceiling, and the manuscript must not cite `0.7217` as one.
Estimating a genuine ceiling for the cross-subject problem needs a different
construction, and remains open.

### What it does establish, which is more interesting

**Pooling across subjects beats using a subject's own data, by a wide margin.** A
decoder trained on 51 other subjects and applied to a held-out subject (`0.8051`)
substantially outperforms the asymptote of a decoder trained on that subject's own
runs (`0.7217`), and outperforms the observed within-subject leave-one-run-out result
(`0.6912`) by more than a tenth.

That is not obvious, and it reframes what the method is doing. The cross-subject
decoder is not overcoming a transfer penalty to reach parity — it is **exploiting
averaging across ~2450 training events to denoise the class templates far beyond what
40 within-subject events can support**, and that gain exceeds what imperfect
inter-subject correspondence costs.

Three earlier results line up behind this reading:

- The subject-count learning curve rises steeply at small `n`, which is what a
  denoising-limited regime looks like.
- Class-pattern reliability is only `0.5983`, so there is a great deal of noise for
  averaging to remove.
- Surface projection improved *within*-subject accuracy but hurt *cross*-subject
  accuracy — consistent with a method whose accuracy is dominated by cross-subject
  averaging, where correspondence quality matters more than per-subject signal.

---

### Surface reliability confirms the representation story independently

Repeating both estimates on the surface-projected checkpoints:

| Quantity | Volumetric | Surface |
| --- | ---: | ---: |
| Split-half class-pattern correlation | 0.4561 | **0.5359** |
| Spearman-Brown full reliability | 0.5983 | **0.6665** |
| Within-subject centroid asymptote | 0.7217 | 0.7101 |

The surface representation is **measurably more reliable**: a subject's class pattern
estimated from three runs correlates `0.67` with the same pattern from the other
three, against `0.60` volumetrically. This is an independent confirmation of the
within-subject decoding result (`0.7332` versus `0.6912`) obtained by a completely
different route — pattern reliability rather than classification — and it settles any
residual worry that the within-subject advantage was a decoder artefact.

It also sharpens the central dissociation. The surface representation is *cleaner*
by two independent measures and still transfers worse across subjects. The deficit is
not about signal quality at all; it is entirely about correspondence.

The centroid asymptote is marginally lower on the surface (`0.7101` versus `0.7217`)
despite the higher reliability, which is consistent with the nearest-centroid rule
being the one decoder that surface projection hurts even within subject.

## 2. Design structure: every run is palindromic, and repeats are highly similar

**All 372 runs are palindromic** — four classes followed by their exact mirror, with
no exceptions. This is what guarantees two events per class per run, and it means the
two presentations of a class always sit at mirrored positions, so repetition and
serial position are confounded by construction.

### No repetition suppression

| Quantity | Value |
| --- | ---: |
| Mean pattern amplitude, first presentation | 1.9452 |
| Mean pattern amplitude, second presentation | 1.9090 |
| Ratio, second over first | **1.0185** |
| CI95 | `[1.0037, 1.0333]` |

The ratio's interval excludes `1.0`, so there is a real effect, but it is **slight
enhancement rather than suppression**, and at under 2% it is too small to motivate
weighting the two presentations differently. Repetition suppression, which would have
been the expected finding, is absent.

### Within-run repeats share far more than class identity

| Comparison | Mean cosine similarity |
| --- | ---: |
| Two presentations of the same class, **same run** | **0.3713** |
| Same class, **different runs** | 0.1472 |
| Difference | **+0.2241** |

Two events of the same class in the same run are roughly **two and a half times more
similar** than two events of that class in different runs. This is measured *after*
the unlabeled subject-run centering, so it is residual structure that centering does
not remove.

This quantifies, for the first time, exactly what the repetition-consistency decoder
is exploiting when it reaches `0.8948`. Its advantage over independent prediction
comes from real and substantial within-run structure — but that structure is
**largely not class identity**, since the same class across runs shares only `0.1472`.
It is run-specific state: residual physiological, motion, and attentional structure
common to events acquired close together.

The consequence for the manuscript is a sharper description than "uses unlabeled
target-run event relationships". The honest version is that the complete-run decoder
exploits run-specific covariance that happens to be shared by same-class repeats
because of where the design places them. That is legitimate under the stated
deployment rule, and it is a further reason `0.8948` must never be presented as
comparable to independent prediction.
