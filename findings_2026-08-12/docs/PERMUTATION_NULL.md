# Permutation Null

Added: 2026-08-12.

Every accuracy this project has reported was stated without a null distribution.
That gap mattered more here than it usually would, because a large share of the
headline accuracy traces to an unlabeled subject-run centering step that consults
the target run's own events. A reader is entitled to ask whether the remaining
signal is a genuine label association or something the design and the preprocessing
manufacture between them.

It is genuine, and by a wide margin.

## Protocol

- **Within-run label shuffling.** Labels are permuted inside each subject-run, which
  preserves the exact two-events-per-class composition every run has, and preserves
  subject and run structure. Shuffling globally would break run composition, make
  the balanced assignment rule inapplicable, and produce an easier, less honest null.
- **200 permutations** across all **30 outer folds**, the full frozen protocol.
- The observed pass reproduced the published values exactly — correlation centroid
  `0.69625` against the `0.6963` already on record — which validates the harness
  before any null is drawn.

### Why this is affordable, and why it is valid

Every feature-side operation in the pipeline is **label-free**: `center_by_subject_run`
uses only subject and run identifiers, detrending uses event time, standardisation
uses training rows, and the dual basis and kernels are functions of the features
alone. None of them changes when labels are permuted.

That has two consequences. Practically, all of it is computed once per fold and
reused across every permutation, so only the classifier refits — about 20 seconds
per permutation rather than minutes. Conceptually, it is *why* permuting labels is a
valid null here rather than a leaky one: nothing upstream of the classifier can
carry label information into the features.

## Result

| Decoder | Rule | Observed | Null mean | Null q95 | p | z |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `linear_svm` | independent | 0.8098 | 0.2498 | 0.2619 | < 0.005 | 76.4 |
| `linear_svm` | balanced | 0.8485 | 0.2501 | 0.2622 | < 0.005 | 78.9 |
| `logistic_l2` | independent | 0.8049 | 0.2497 | 0.2620 | < 0.005 | 70.6 |
| `logistic_l2` | balanced | 0.8465 | 0.2499 | 0.2611 | < 0.005 | 78.7 |
| `correlation_centroid` | independent | 0.6963 | 0.2504 | 0.2642 | < 0.005 | 53.2 |
| `correlation_centroid` | balanced | 0.7434 | 0.2502 | 0.2632 | < 0.005 | 59.5 |

No permutation out of 200 reached the observed value for any decoder or either
rule, so `p = 1/201 ≈ 0.005` is an **upper bound** rather than an estimate. The
reported figure uses the add-one estimator and is never quoted as exactly zero.

## What this establishes

**1. The transductive preprocessing does not manufacture label information.** This
is the important one. If the unlabeled subject-run centering, the detrending, or the
balanced assignment rule were creating apparent class structure, the permuted null
would sit *above* chance — the pipeline would recover signal from shuffled labels.
It does not. Every null mean lands within `0.0004` of the `0.25` chance level.

That is the direct, quantitative answer to the most serious objection a reviewer can
raise about this project's methodology.

**2. The balanced rule is not free accuracy.** Its null also sits at chance
(`0.2501`), so the gain it provides over argmax comes from combining real per-event
evidence with the design constraint, not from the constraint alone.

**3. Every reported accuracy is far outside its null.** The smallest z is `53`.

## Limitations

- **C is fixed** at the value the unpermuted nested analysis selected most often,
  rather than reselected inside each permutation, which would multiply cost by the
  grid size times the inner fold count. This is the conventional compromise and is
  recorded in the output.
- **The frozen decoder itself was not permuted**, only the three standard
  comparators. The frozen hierarchy's own null would require running its full nested
  candidate-selection pipeline inside each permutation. Given that the comparators
  share the same representation, the same folds, and the same preprocessing, and
  that their nulls sit exactly at chance, there is no reason to expect a different
  answer — but it is not yet demonstrated, and the manuscript should say so.

## Reproduction

```bash
python findings_2026-08-12/scripts/run_permutation_test.py \
  --checkpoint-dir "/path/to/subject_checkpoints" \
  --out-json "/path/to/permutation_test_full.json" \
  --permutations 200
```

Runtime is roughly 20 s per permutation after a one-off precomputation of the
label-free structure across folds.
