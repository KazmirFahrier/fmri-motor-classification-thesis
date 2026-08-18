# Cross-Contrast Transfer: the two fine contrasts share a decision axis

Added: 2026-08-18. Script: `../scripts/run_cross_contrast_transfer.py`.
Result: `cross_contrast_transfer.json`.

The frozen four classes form two pairs that are different *kinds* of contrast. Left leg
versus right leg is a **laterality** distinction; forearm versus upper arm is a
**proximal-distal** one. The research plan recorded an expectation that a decoder
trained on one would fail on the other, on the grounds that they are conceptually
unrelated, and noted that failure would still be informative because it would explain
why the fine stage is the bottleneck.

**The expectation was wrong.** Transfer succeeds, in both directions, on held-out
subjects, against a permutation null at exact chance.

## The prediction came first

The representational geometry computed alongside this
([`REPRESENTATIONAL_GEOMETRY.md`](REPRESENTATIONAL_GEOMETRY.md)) showed that the four
cross-pair distances are not uniform: left leg sits closer to forearm (`1.2642`) than
to upper arm (`1.7644`), and right leg reverses it. That predicts a specific mapping —
`left leg -> forearm`, `right leg -> upper arm` — **before any transfer was run**.

This matters. For a binary problem, choosing whichever of the two label mappings scores
higher on held-out data guarantees a result above chance no matter what the data
contains. The mapping tested here is the one the RDM specified in advance, so the
above-chance result is a confirmed prediction rather than a selection artifact.

## Result

30 folds, binary linear SVM, held-out subjects throughout. Chance is `0.5`.

| Condition | Mean | CI95 | Folds above chance |
| --- | ---: | --- | ---: |
| Leg pair, within-contrast | 0.8385 | `[0.8281, 0.8490]` | 30/30 |
| Arm pair, within-contrast | 0.8032 | `[0.7857, 0.8206]` | 30/30 |
| **Leg → arm transfer** | **0.6525** | `[0.6331, 0.6719]` | **30/30** |
| **Arm → leg transfer** | **0.6235** | `[0.6091, 0.6379]` | **30/30** |

The swapped-mapping rows in the JSON are exactly `1 - aligned` and carry no
independent information; they are recorded for completeness only and should never be
cited as a second confirmation.

**Transfer recovers about 43% of the within-contrast margin over chance.** Leg→arm
retains `0.1525` of the leg decoder's `0.3385` (45.0%); arm→leg retains `0.1235` of
`0.3032` (40.7%).

## The control this needed, and it passes

This project's preprocessing is transductive — unlabeled subject-run centering is worth
`+0.52`, more than any decoder choice — and it forces the four class means within a run
toward summing to zero. A reviewer's first question is whether that dependency alone
manufactures apparent transfer between two unrelated contrasts.

Labels were shuffled **within each subject-run**, preserving the two-per-class
composition, and the entire analysis rerun 200 times. The preprocessing is label-free
and therefore identical under permutation, so the null holds the suspected mechanism
fixed and varies only the label mapping.

| Condition | Observed | Null mean | Null sd | z | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Leg → arm transfer | 0.6525 | 0.5005 | 0.0115 | 13.2 | < 0.005 |
| Arm → leg transfer | 0.6235 | 0.5001 | 0.0118 | 10.5 | < 0.005 |
| Leg pair, within | 0.8385 | 0.5006 | 0.0150 | 22.6 | < 0.005 |
| Arm pair, within | 0.8032 | 0.5018 | 0.0138 | 21.8 | < 0.005 |

The null lands at `0.5005` and `0.5001` — chance to within a third of its own standard
deviation. `p < 0.005` is the floor for 200 draws, meaning no permutation of 200 ever
reached the observed value. **Centering does not manufacture the transfer.**

## What it means, and what it does not

The two fine contrasts are **not** independent problems. Something like 43% of the
discriminative axis is shared between a laterality contrast and a proximal-distal one,
which is a positive result about the structure of motor cortex representation rather
than the negative one the plan anticipated.

**What that shared axis is remains open.** At least three accounts fit:

- A **somatotopic gradient**, in which both contrasts partly project onto one
  medial-lateral or proximal-distal cortical axis.
- **Movement amplitude or effort**, which could differ systematically between the
  members of each pair and would be shared without being somatotopic.
- **Laterality in the arm conditions**, if forearm and upper arm movements were
  performed with a consistent hand. The dataset names the leg conditions by side but
  not the arm conditions, so this cannot be excluded from the events files.

This analysis cannot separate them, and the manuscript should not claim a somatotopic
interpretation on this evidence alone.

## The twelve-class extraction resolves this

The dataset defines twelve conditions, and the extraction now running adds toe, ankle,
wrist, finger, jaw, lip, tongue and eye. That provides graded within-limb positions —
toe and ankle extend the leg pair, wrist and finger extend the arm pair — and face
conditions that are anatomically remote from both.

If the shared axis is somatotopic, transfer should scale with somatotopic distance and
the face conditions should sit off the axis entirely. If it is amplitude or laterality,
it should not. **This is now the highest-value analysis the twelve-class data enables**,
and it was not on the list of five planned analyses before this result.

## Why this matters beyond interpretation

The fine within-pair stage is the bottleneck: `0.888` of the linear SVM's errors are
within-pair. A shared axis means the two fine problems can **borrow strength from each
other**, which the current hierarchy — which trains each pair's discriminator
independently — does not exploit. A decoder that pools the two fine contrasts onto a
common axis is a concrete, motivated architecture change, and it is the first one this
project has had a principled reason to try rather than a search over variants.
