# Proposed Updates to Existing Documents

These revisions follow from the 2026-08-12 findings. They are **described here and
supplied as `proposed_edits_to_existing_docs.patch`, and deliberately not applied**.
The existing repository files are unchanged.

To apply them later:

```bash
git apply findings_2026-08-12/docs/proposed_edits_to_existing_docs.patch
```

---

## `docs/PUBLICATION_PLAN.md`

### Add a Headline Rule section

`0.8314` is the only result directly comparable to an ordinary independent
classifier and must lead every abstract, table, and summary. `0.8948` belongs in
its own row, always adjacent to its complete-run precondition, and never as the
paper's top-line accuracy. The existing documents are scrupulous about this
distinction; a manuscript table will not be read as scrupulously, so the ordering
must enforce it.

### Add three claims

- A conventional linear SVM under the identical protocol reaches `0.8051`, and the
  classic correlation classifier `0.6963`. The frozen decoder's paired subject-level
  advantage over the SVM is `+0.0262`, CI95 `[+0.0107, +0.0426]`, winning on 40 of
  62 subjects. Reliable but modest, and it must be described that way.
- Because a correctly evaluated conventional SVM also reaches `0.8051`, the legacy
  collapse from `0.8522` to `0.2629` / `0.2500` cannot be attributed to using a weak
  model. This *strengthens* the evaluation-artifact argument.
- The unlabeled subject-run centering is worth roughly `+0.52` independent accuracy,
  about twenty times the hierarchy's advantage over standard MVPA.

### Add a Required Limitations section

1. **The design space was searched with full cohort visibility.** The `3:8` window,
   the `24³` grid, the `1024`/`2048` caps, and the hierarchy structure were selected
   across roughly 74 analysis scripts while all 62 subjects were visible. Freezing
   afterwards stops further tuning; it does not undo that history. The nested folds
   are honest at the final step, but the *architecture of the estimator* was chosen
   with knowledge of the cohort.
2. **No inter-subject spatial normalization was applied.** The representation is
   `zscore(resize(volume))` on `space-T1w` denoised BOLD, which rescales a bounding
   box of native anatomy rather than registering subjects to a template. Some part
   of the cross-subject difficulty may be unperformed alignment.
3. **The unlabeled subject-run centering is the dominant ingredient, and it is
   transductive.** Removing it collapses independent accuracy to `0.2860` against a
   `0.25` chance level. Centering and detrending use the target run's own events;
   the balanced and repetition-consistency rules additionally use the target run's
   design structure. All are label-free, none are inductive. This belongs in the
   abstract, not in a preprocessing list.
4. **The earlier neural figures are withdrawn, and the conclusion reverses.** The
   manuscript must **not** claim that neural architectures fail on this task; the
   published recipe contained an architecture-design error. Every neural number
   must be reported with its training accuracy beside it.

---

## `docs/INVESTIGATION_CLOSEOUT.md`

### Add two comparator rows to the Frozen Claims table

| Context | Result | Interpretation |
| --- | ---: | --- |
| Standard linear SVM comparator | `0.8051` independent | Conventional MVPA under the identical 30-fold nested protocol. Frozen decoder leads by `+0.0262`, CI95 `[+0.0107, +0.0426]`. |
| Classic correlation classifier | `0.6963` independent | Haxby-style nearest-centroid comparator under the same protocol. |

### Revise two legacy rows

- Full legacy pooled baseline `0.2629`: **withdrawn as architecture evidence**.
- Full legacy subject-wise holdout `0.2500`: **not a valid negative control** —
  constant single-class output from a BatchNorm train/eval mismatch.

### Add an "Amendments after freeze" section

Both additions are reporting corrections required before submission. Neither tunes
the frozen cohort, and neither alters any frozen decoder value.

---

## `docs/CURRENT_STATUS.md`

Add a dated section recording the three gaps and their state, the centering
ablation as the most consequential new finding, and the corrected ordering of
contributions: **centering first, leakage-aware evaluation second, the covariance
hierarchy third.** The manuscript currently implies the reverse.

---

## `README.md`

Update the Current Status bullets to lead with `0.8314` as the comparable number,
add the conventional comparators, note that the centering step is load-bearing,
record that the neural figures are withdrawn and the conclusion reverses, and point
to the three new documents in this folder.
