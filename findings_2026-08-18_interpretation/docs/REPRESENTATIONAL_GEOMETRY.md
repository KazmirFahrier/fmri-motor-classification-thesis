# Representational Geometry and Confusion Structure

Added: 2026-08-18. Script: `../scripts/run_representational_geometry.py`.
Result: `representational_geometry.json`.

Earlier diagnostics observed, from the hierarchy alone, that coarse leg-versus-arm
routing is strong and the within-pair stage is the bottleneck. That was a property of
one decoder. Checking it against the raw representational geometry and against every
comparator built in the previous round tests whether it is a property of the **data**.

## Confusion structure: the error budget is almost entirely within-pair

Pooled over 30 folds, independent rule.

| Decoder | Accuracy | Fraction of errors that are within-pair |
| --- | ---: | ---: |
| `linear_svm` | 0.8100 | **0.888** |
| `logistic_l2` | 0.7858 | 0.850 |
| `correlation_centroid` | 0.6214 | 0.652 |

Chance for "within-pair" under random errors is `1/3`, since each class has one
pair-mate and two non-mates. Every decoder is far above that, and the ordering is
informative: the stronger the decoder, the more completely its remaining errors
collapse onto the within-pair distinction. The linear SVM has essentially **solved**
the leg-versus-arm problem and spends `0.888` of its error budget on left-versus-right
leg and forearm-versus-upper-arm.

This confirms the earlier single-method observation as a property of the data, and it
sharpens where any future gain must come from. **Nothing that improves coarse routing
can help**, because coarse routing is already at ceiling. This is a concrete
prioritisation result, and it retrospectively explains why the hierarchy — whose whole
premise is a coarse stage followed by a fine stage — gains so little over a flat
classifier: its coarse stage solves a problem the flat classifier was not failing.

## The RDM, and an unexpected structure in it

Correlation distance between class-mean patterns, averaged over 62 subjects.

| | Left leg | Right leg | Forearm | Upper arm |
| --- | ---: | ---: | ---: | ---: |
| **Left leg** | 0.0000 | 0.8456 | 1.2642 | 1.7644 |
| **Right leg** | 0.8456 | 0.0000 | 1.7406 | 1.3501 |
| **Forearm** | 1.2642 | 1.7406 | 0.0000 | 0.9852 |
| **Upper arm** | 1.7644 | 1.3501 | 0.9852 | 0.0000 |

| | Value |
| --- | ---: |
| Coarse leg-vs-arm distance | 1.5298 |
| Within leg pair | 0.8456 |
| Within arm pair | 0.9852 |
| **Coarse:fine ratio** | **1.671** |
| Inter-subject RDM agreement | 0.5999 (sd 0.3742, 1891 pairs) |

The coarse/fine split is confirmed: between-limb distances are `1.67x` within-limb
ones, which is the geometric counterpart of the confusion result.

**The cross-pair distances are not uniform, and that was not expected.** Left leg sits
much closer to forearm (`1.2642`) than to upper arm (`1.7644`), and right leg reverses
it exactly — closer to upper arm (`1.3501`) than to forearm (`1.7406`). The four
cross-pair distances split cleanly into a near pair and a far pair separated by about
`0.45`, which is larger than the entire within-leg distance.

Something therefore organises these four classes along an axis that **crosses the limb
boundary**. That prediction is tested directly in
[`CROSS_CONTRAST_TRANSFER.md`](CROSS_CONTRAST_TRANSFER.md), and it holds.

## Inter-subject agreement

Mean pairwise correlation between subjects' RDM off-diagonals is `0.5999` with an sd of
`0.3742` over 1891 subject pairs. The geometry is **substantially shared but far from
uniform**. The large sd is consistent with the decodability result: subjects whose
patterns are unreliable also have noisier RDMs, so some of this spread is measurement
noise rather than genuine individual difference, and this analysis cannot separate the
two.

That distinction matters for the alignment conclusion. The project found that surface
normalization gives no cross-subject advantage over a bounding-box rescale. A
moderately shared geometry is consistent with that: if the representational structure
were near-identical across subjects, better anatomical correspondence would have more
to work with.

## Caveats

The RDM is computed on class means after the project's transductive subject-run
centering, which forces the four class means within a run toward summing to zero and
therefore imposes a dependency between them. The **relative** structure reported here —
which cross-pair distances are large and which are small — is not something that
constraint predicts, and the transfer analysis tests it against a permutation null that
holds the centering fixed. But the absolute distances should not be read as unconstrained
correlation distances.
