# Resume Here

Written 2026-08-12, updated 2026-08-13 after a session restart.

## What the restart cost, and what it did not

A session teardown on 2026-08-13 **wiped the entire session scratchpad** under
`/private/tmp/claude-501/.../scratchpad`. The overnight surface extraction and the
24-fold CNN run were both lost, along with every intermediate JSON and every
downloaded image.

**No findings were lost.** Every result had already been distilled into
`experiments/*.results.json` and written up in `docs/`, all of which live in the
repository. The cost was compute, not knowledge.

Two rules follow, and they are now the standing practice:

1. Long-running jobs write to a **persistent** directory following the project
   convention, `~/Documents/New project/status_<date>_<name>/`. The scratchpad is for
   smoke tests only.
2. Distil each result into a tracked repo file **as soon as it lands**, rather than at
   the end of a batch.

## Nothing is running

All queued work completed. Every intermediate output lives under
`~/Documents/New project/status_2026-08-13_surface_projection/`, and every result has
been distilled into `docs/` and `experiments/` in this folder.

Two traps to remember if any extraction is rerun:

- The cohort **skips** subject ids `15`, `19`, `28`, `40`, `41`, `64` and extends to
  `sub-68`. Always invoke `build_surface_event_sequences.py` with
  `--subjects-from <frozen checkpoint dir>`; the contiguous `--subject-count` fallback
  silently produces a 57-subject cohort that no longer matches the volumetric one.
- Raw BIDS pads the run index (`run-01`), the fmriprep derivatives do not (`run-1`).
  Getting it wrong yields a silent 404 on the events file.

## The queued sequence, in order

1. **Normalize** the surface checkpoints — matched per-volume z-score over valid
   vertices. Without this the surface/volume comparison is confounded with
   normalisation, because the extraction stores raw BOLD.

   ```bash
   python findings_2026-08-12/scripts/normalize_surface_checkpoints.py \
     --in-dir <surfseq> --out-dir <surfseq_norm>
   ```

2. **Decode on the surface** with the identical protocol, and compare against the
   volumetric `0.8051` linear SVM / `0.8314` frozen decoder.

   ```bash
   python findings_2026-08-12/scripts/run_standard_mvpa_baseline.py \
     --checkpoint-dir <surfseq_norm> --out-json <...> --preprocess frozen
   ```

3. **Hyperalign**, then decode again. Template must come from training subjects only.

   ```bash
   python findings_2026-08-12/scripts/run_connectivity_hyperalignment.py \
     --in-dir <surfseq_norm> --out-dir <surfseq_hyper> --train-subjects <fold train set>
   ```

The three-way result — volumetric, surface-aligned, hyperaligned — is the substantive
output this round has been building toward.

## Two things that must not be skipped when reading the result

**Coverage is not matched.** The volumetric bounding box contains cortex, white
matter, subcortex and cerebellum; the surface contains cortex only, minus ~7% to the
EPI field of view. A surface *loss* therefore does **not** distinguish "alignment adds
little" from "subcortex and cerebellum mattered". Separating them needs a volumetric
decoder restricted to a cortical ribbon mask.

**Valid-vertex fraction varies between subjects** — `0.932` for `sub-01` against
`0.855` for `sub-02`. If the spread holds, the intersection across all 62 may be
materially smaller than any individual subject's. `normalize_surface_checkpoints.py`
has `--intersect-valid` to force an identical feature space at the cost of coverage.
Decide that explicitly; do not let it default.

## State of the record

- Nothing in the existing repository is modified. `git status` shows only
  `findings_2026-08-12/`.
- Intended revisions to `README.md`, `docs/CURRENT_STATUS.md`,
  `docs/INVESTIGATION_CLOSEOUT.md`, and `docs/PUBLICATION_PLAN.md` are described in
  `docs/PROPOSED_UPDATES_TO_EXISTING_DOCS.md` and supplied as a patch, unapplied.
- `docs/RESEARCH_COVERAGE_MAP.md` is the live ledger. Check it before proposing
  anything as untried, and check the commit history too — several efforts left no
  distinctive identifier in code.
- 18 tests pass.

## Still open after tonight

Beta-series GLM features, searchlight, permutation null for the frozen decoder
itself, runs-per-subject learning curve, hyperparameter search for the corrected CNN,
and the HCP leakage replication.
