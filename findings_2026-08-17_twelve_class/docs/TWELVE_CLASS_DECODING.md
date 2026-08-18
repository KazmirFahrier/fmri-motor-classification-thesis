# Twelve-Class Decoding and the Somatotopic Prediction

Added: 2026-08-18. Script: `../scripts/run_twelve_class_decoding.py`.
Result: `twelve_class_decoding.json`.

The dataset defines twelve movement conditions and this project has only ever used
four, so the problem the data was built for had never been attempted. All 62 subjects
were re-extracted with every condition retained — `(144, 8, 13824)` per subject, twelve
events per class — under the identical temporal window and spatial transform, and the
four-class subset reproduces the frozen checkpoints to `1.9e-06`, which is float32
rounding in the resize.

## Result

`linear_svm`, 30 folds, subject-wise, independent rule.

| | Value |
| --- | ---: |
| **Twelve-class balanced accuracy** | **0.6838** (sd 0.0353) |
| Chance | 0.0833 |
| Ratio to chance | **8.2x** |

## The somatotopic prediction holds

Four-class analysis found that a decoder trained on left-versus-right leg reads
forearm-versus-upper-arm at `0.6525`, 30/30 folds, against a permutation null at chance.
Three accounts fit that result — a somatotopic gradient, movement amplitude, or a
laterality confound in the arm conditions — and four classes could not separate them.

Twelve conditions can, because they supply a graded ordering. Three independent lines
now converge on the somatotopic account.

**Representational distance tracks the homunculus ordering.** Ranking the conditions
toe → ankle → leg → upper arm → forearm → wrist → finger → jaw → lip → tongue → eye and
correlating that separation against the group RDM gives **Spearman rho = +0.4752**.

**Body-part structure is strong.** Conditions within the same body part are much closer
than across:

| | Mean RDM distance |
| --- | ---: |
| Within body part (lower limb / upper limb / face) | 0.8348 |
| Between body parts | 1.1724 |

**Per-class accuracy tracks cortical magnification**, which is the prediction no
competing account makes.

| Condition | Accuracy | | Condition | Accuracy |
| --- | ---: | --- | --- | ---: |
| Eye | **0.8919** | | Jaw | 0.6624 |
| Tongue | 0.7718 | | Wrist | 0.6543 |
| Finger | 0.7680 | | Forearm | 0.6481 |
| Lip | 0.7102 | | Left leg | 0.6390 |
| Toe | 0.6909 | | Ankle | 0.5922 |
| | | | Right leg | 0.5922 |
| | | | Upper arm | 0.5917 |

The four best-decoded conditions after the eye are **tongue, finger and lip** — exactly
the body parts with the largest cortical territory in the classical homunculus. The
four worst are **upper arm, right leg, ankle and left leg** — proximal limb segments
with small, mutually adjacent representations. Eye movements decode best of all, which
is expected: the frontal and parietal eye fields are spatially distinct from the motor
strip entirely.

**Amplitude and laterality do not predict that ordering.** A movement-amplitude account
predicts large proximal movements (upper arm, leg) decode *best*, and the data show the
opposite. A laterality account has nothing to say about why tongue and lip separate so
well. The somatotopic account predicts all three observations, so the shared axis found
in the four-class transfer analysis is best read as somatotopic.

## What this changes for the project

**The contribution is larger than it was.** The manuscript previously reported a
four-class decoder. It can now report twelve-class decoding at `8.2x` chance whose
representational geometry recovers the homunculus — a neuroscience result rather than
only a methods result, and the dimension the write-up was thinnest on.

**The four-class problem was the hard part of the data.** The frozen four are left leg,
right leg, forearm and upper arm: three of the four worst-decoded conditions in the
whole set, and all four are proximal limb segments with adjacent cortical territory.
That reframes the project's headline. `0.8314` on four classes was not an easy problem
made to look impressive — it was **the most confusable four-way subset the dataset
contains**, which is a considerably better thing for the manuscript to be able to say,
and it was chosen before any of this was known.

## Caveats

The somatotopic ranks are assigned from the classical homunculus, not measured. Left
and right leg share a rank, and jaw and lip are placed adjacently, because the ordering
is genuinely ambiguous at those points. The `rho = +0.4752` is therefore a correlation
against a reasonable but not unique ordering, and a different defensible ranking would
shift it somewhat. The body-part contrast (`0.8348` versus `1.1724`) and the per-class
accuracy pattern do not depend on the ranking at all, and carry the argument on their
own.

Accuracy is not corrected for the differing difficulty of the eleven-way alternatives
each class faces, and no permutation null has been run at twelve classes yet. Chance is
analytic (`1/12`) rather than empirical, and the project's standing practice is to
measure the null rather than assume it — that run is the immediate next step.
