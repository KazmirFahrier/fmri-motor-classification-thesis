# Cross-Contrast Transfer Is Real, Reliable, and Not a Somatotopic Gradient

Added: 2026-08-18. Script: `../scripts/run_somatotopic_transfer.py`.
Result: `somatotopic_transfer.json`.

**This qualifies a claim made earlier the same day in
[`TWELVE_CLASS_DECODING.md`](TWELVE_CLASS_DECODING.md)** — that the shared decision axis
found in the four-class transfer analysis "is best read as somatotopic." The direct
directional test does not support that reading. The correction is recorded below.

## Design

Six binary contrasts, each oriented by **somatotopic rank before any model was
fitted**: the lower-ranked (more distal or inferior) member is class 0. For a binary
problem, picking whichever mapping scores higher on held-out data guarantees an
above-chance result, so the orientation had to be fixed a priori for this to be a test
at all.

Left-versus-right leg was excluded from the directed set — its members share a rank, so
its orientation along the body axis is undefined — and reported separately.

## Result: the somatotopic prediction fails

| | Value |
| --- | ---: |
| Mean directed transfer | **0.4570** |
| Chance | 0.5000 |
| Directed pairs above chance | **7 / 20** |
| Correlation of transfer with somatotopic distance | +0.2863 |

Transfer under a-priori somatotopic orientation is **below chance on average**, and the
correlation with somatotopic separation is *positive* — transfer is slightly better
between contrasts that are further apart on the body, which is the opposite of what a
graded somatotopic axis predicts.

## But the transfer itself is real and highly reliable

This is not a null result. Of the 30 ordered contrast pairs, **24 are consistent in a
fixed direction on at least 28 of 30 folds** — 9 positive and 15 negative. Mean absolute
deviation from chance is `0.0816`. Several pairs are extremely consistent:

| Pair | Transfer | Folds |
| --- | ---: | ---: |
| `jaw_lip -> lip_tongue` | 0.1614 | 0/30 |
| `leg_left_right -> lip_tongue` | 0.7439 | 30/30 |
| `lip_tongue -> jaw_lip` | 0.3073 | 0/30 |
| `leg_left_right -> wrist_finger` | 0.5974 | 30/30 |
| `jaw_lip -> upperarm_forearm` | 0.6158 | 30/30 |

A decoder trained on one contrast predicts another far better than chance in **either**
direction. What it does not do is align with the body axis.

## The original four-class finding replicates

The four-class result was `leg_left_right -> upperarm_forearm` at `0.6525`. Here the same
pair reads `0.3655` — because this script orients the arm contrast by somatotopic rank
(class 1 = forearm) while the four-class script used the opposite order (class 1 = upper
arm). Reading it in the original orientation gives **`0.6345`**, against `0.6525`.

The residual `0.018` is expected: preprocessing here centers and detrends over all 144
events per subject-run rather than 48, so the inputs are not bit-identical even though
the underlying window and transform are.

So the effect is confirmed on independent extraction. **What has changed is only its
interpretation.**

## What this means

Two things are true at once, and the earlier document collapsed them.

**Somatotopy organises the global geometry.** Representational distance tracks the
homunculus ordering (`rho = +0.4752`), within-body-part distances are much smaller than
between (`0.8348` versus `1.1724`), and per-class accuracy follows cortical
magnification. Those results are about **where conditions sit relative to each other**
and they stand unaltered.

**The discriminative axes are not co-oriented along that ordering.** Which direction
separates toe from ankle is not the same direction, projected, that separates wrist from
finger. Transfer between contrasts is strong and reproducible, but its sign is
idiosyncratic to the particular pair rather than inherited from a single body-axis
gradient.

A useful way to hold both: somatotopy describes the **arrangement** of the
representation, not a single **direction** within it. Conditions near each other on the
homunculus have similar patterns, and yet the fine contrast between two neighbours is
carried by pair-specific structure.

## Correction to `TWELVE_CLASS_DECODING.md`

That document argued that because per-class accuracy tracks cortical magnification and
the RDM tracks the homunculus, "the shared axis found in the four-class transfer
analysis is best read as somatotopic." The evidence it cited is about distances, and
distances do not fix a direction. The directional claim needed the directional test,
which now exists and does not support it.

The three geometry findings in that document are unaffected. The sentence extending them
to the *identity of the transfer axis* over-reached and is withdrawn.

## Still open

What the shared structure actually is remains unresolved. The two candidates the
four-class work could not exclude — movement amplitude and laterality — are also not
tested by this analysis, which only rules out a graded somatotopic gradient. Testing
them needs per-condition effort or laterality information that the events files do not
carry, so it may require the dataset paper or the acquisition protocol rather than more
computation.
