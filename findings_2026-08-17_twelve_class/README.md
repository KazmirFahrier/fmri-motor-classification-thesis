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

---

## Results (added 2026-08-18)

Extraction complete: **62/62 subjects**, `(144, 8, 13824)` each, twelve events per
class. The four-class subset reproduces the frozen checkpoints to `1.9e-06`.

| | Value |
| --- | ---: |
| **Twelve-class balanced accuracy** | **0.6838** (sd 0.0353) |
| Chance | 0.0833 |
| Ratio to chance | **8.2x** |
| Somatotopic ordering (Spearman rho) | **+0.4752** |
| Within body part / between body part RDM distance | 0.8348 / 1.1724 |

Per-class accuracy tracks cortical magnification — eye `0.8919`, tongue `0.7718`,
finger `0.7680`, lip `0.7102` at the top; upper arm `0.5917`, leg `0.5922`, ankle
`0.5922` at the bottom. That pattern is predicted by a somatotopic account and by
neither a movement-amplitude nor a laterality account, which **resolves the ambiguity
left by the four-class cross-contrast transfer result**.

It also reframes the headline: the frozen four are three of the four worst-decoded
conditions in the set, so `0.8314` was obtained on the most confusable four-way subset
the dataset contains.

Full write-up: [`docs/TWELVE_CLASS_DECODING.md`](docs/TWELVE_CLASS_DECODING.md).
