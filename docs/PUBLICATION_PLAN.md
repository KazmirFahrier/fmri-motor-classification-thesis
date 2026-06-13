# Publication Plan

The working contribution is a leakage-aware benchmark for motor-task fMRI classification on the full public dataset. The project is being organized around what can be defended in a paper, not only what produces the highest number in a notebook.

## Core Claim Being Tested

The original pooled-split accuracy may overstate generalization. The publishable result should compare:

- Original smaller-subset pooled split.
- Full-dataset pooled split.
- Full-dataset subject-wise cross-validation and held-out subject evaluation.

## Phase 1: Honest Baseline

Phase 1 is complete as a baseline exercise. It established the leakage/generalization gap by running:

- Historical 9-subject pooled split: complete.
- Full 7-batch pooled split: stopped by controlled policy at epoch 25, with validation still near chance.
- Full 7-batch subject-wise 5-fold plus holdout: complete, with final holdout at chance-level.

The final paper should emphasize the subject-wise and held-out subject results. Pooled-split results are useful for comparison but are not enough for a generalization claim.

## Current Diagnostic Direction

The strongest current evidence is that the extracted data contains local motor-class signal, but raw full-cohort transfer is dominated by subject/run nuisance structure. Corrected clip-mean feature diagnostics show:

- Raw held-out-run and held-out-subject transfer are near chance.
- Transductive per-run centering plus cosine nearest centroids recovers substantial signal.
- Preliminary train-only centering does not recover most of that gain.
- Raw feature variance is overwhelmingly explained by subject and subject-run identity rather than class identity.

Publication framing should therefore avoid claiming that the transductive alignment score is a standard supervised classifier. It should be treated as evidence for domain shift unless the method section defines a legitimate test-time adaptation protocol using unlabeled target-run statistics.

## Later Phases

- Phase 2: preprocessing/domain-shift ablation under the same subject-wise protocol.
- Phase 3: task framing comparison: volume, trial clip, beta-map samples, and target-run adaptation if scientifically justified.
- Phase 4: architecture comparison under locked preprocessing and split protocol.
- Phase 5: subject/run-invariance experiments such as DANN-style adversarial domain heads or explicit run covariate removal.
- Phase 6: interpretability using saliency or Grad-CAM overlap with motor-region atlas masks.
- Phase 7: final statistics, manuscript tables, and submission materials.

## Repository Rule

Keep private manuscript drafts, unpublished paper PDFs, Kaggle API tokens, raw data, checkpoints, and local logs out of this public repository. Track only code, configs, lightweight result summaries, and reproducibility notes.
