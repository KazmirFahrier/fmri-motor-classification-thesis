# Twelve-Class Extension — 2026-08-17

A separate round, in its own folder. **Nothing in the existing repository or in
`findings_2026-08-12/` is modified.**

## Why this exists

`findings_2026-08-12/LITERATURE_REVIEW.md` established, by checking the dataset paper
against the raw events files, that the dataset defines **twelve** movement conditions
and the project has only ever decoded **four**.

Every run contains 24 movement blocks — each of the twelve conditions exactly twice —
of which the frozen pipeline extracts eight. Roughly two thirds of the available
movement events have never been used, and the 12-class problem the dataset was built
for has never been attempted.

## Why it is the highest-value extension available

The subject-count learning curve showed the cohort is **saturated**: one more subject
is worth `+0.0008`. The runs-per-subject curve was still climbing. Events per subject
is therefore the axis with headroom, and this triples it — 144 per subject instead of 48.

It also enables analyses the four-class design cannot support:

- **Somatotopic gradient.** Whether representational geometry recovers the classical
  toe-to-face ordering. The somatotopy literature predicts it will do so only
  partially, which is itself a result.
- **A reference ceiling.** Jaw, lip, tongue and eye are anatomically remote from limb
  representations and should separate easily, giving an upper bound the project lacks.
- **Within-limb gradients.** Toe and ankle extend the leg pair, wrist and finger the
  arm pair, testing whether the fine within-limb stage — the repeatedly identified
  bottleneck — improves or degrades with more graded classes.

## Comparability is built in and verified

The temporal window, spatial transform, and file layout are identical to the frozen
pipeline; only the set of retained events differs. Class ids `0`–`3` are the frozen
four in their original order, so any four-class analysis of this output is directly
comparable to existing results.

**Verified on `sub-01` run 1**: the four-class subset reproduces the frozen checkpoint
with identical event keys, identical labels, and a maximum feature difference of
`1.4e-06` — float32 rounding in the resize, not a pipeline difference.

## Status

Extraction running. Output `(144, 8, 13824)` per subject to
`~/Documents/New project/status_2026-08-17_twelve_class/seq/`.

Checkpointed per subject and validated on event count, so an interruption resumes
rather than silently accepting a short file.

## Planned analyses, in order

1. **Four-class replication** on the new extraction — confirms nothing changed before
   any 12-class claim is made.
2. **12-class decoding** under the same 30-fold subject protocol. Chance is `1/12`.
3. **Somatotopic RSA** — dissimilarity geometry over the twelve conditions, tested
   against the predicted homunculus ordering.
4. **Coarse groupings** — limb versus face, and within-limb gradients, to locate where
   the confusions actually lie.
5. **Does the extra data help the original four-class problem?** Train on all twelve,
   evaluate on the four. This is the direct test of whether the unused events carry
   transferable signal.
