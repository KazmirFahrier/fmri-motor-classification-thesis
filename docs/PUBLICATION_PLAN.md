# Publication Plan

The working contribution is a leakage-aware benchmark for motor-task fMRI classification on the full public dataset. The project is being organized around what can be defended in a paper, not only what produces the highest number in a notebook.

## Core Claim Being Tested

The original pooled-split accuracy may overstate generalization. The publishable result should compare:

- Original smaller-subset pooled split.
- Full-dataset pooled split.
- Full-dataset subject-wise cross-validation and held-out subject evaluation.

## Phase 1: Honest Baseline

Phase 1 is active now. It should establish the leakage gap by running:

- Historical 9-subject pooled split: complete.
- Full 7-batch pooled split: running on Kaggle.
- Full 7-batch subject-wise 5-fold plus holdout: running on Kaggle.

The final paper should emphasize the subject-wise and held-out subject results. Pooled-split results are useful for comparison but are not enough for a generalization claim.

## Later Phases

- Phase 2: preprocessing ablation under the same subject-wise protocol.
- Phase 3: task framing comparison: volume, trial clip, and beta-map samples.
- Phase 4: architecture comparison under locked preprocessing and split protocol.
- Phase 5: subject-invariance experiments such as DANN-style adversarial subject heads.
- Phase 6: interpretability using saliency or Grad-CAM overlap with motor-region atlas masks.
- Phase 7: final statistics, manuscript tables, and submission materials.

## Repository Rule

Keep private manuscript drafts, unpublished paper PDFs, Kaggle API tokens, raw data, checkpoints, and local logs out of this public repository. Track only code, configs, lightweight result summaries, and reproducibility notes.

