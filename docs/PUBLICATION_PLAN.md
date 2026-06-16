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
- Dense `24³` corrected-clip features are the current practical sweet spot: they reach `0.6001` held-out-run and `0.5654` subject-fold trial accuracy under target-run centering plus cosine, while `32³` confirms saturation rather than a new subject-generalization gain.
- Several weak subjects remain unstable even after target-run centering, especially `sub-52`, `sub-42`, `sub-17`, `sub-20`, `sub-54`, and `sub-63`.
- The first event-level consistency audit suggests that at least some weak subjects are internally inconsistent across runs, not merely mismatched to the rest of the cohort. `sub-52` and `sub-42` have negative cross-run centroid margins after subject-run centering.
- Source BIDS timing is not the broad failure mode: all 372 extracted target-class run windows match OpenNeuro `ds004044` v2.0.3 `events.tsv` onset/TR starts.
- A first event-level classifier-complexity sweep found that random-projection ridge underperforms cosine nearest centroids after target-run centering. The next improvement should target representation/adaptation/QC, not only a stronger final classifier.
- A task-design-aware balanced assignment adaptation improved dense `24³` event accuracy to `0.6243` for held-out runs and `0.5826` for subject folds by enforcing two predicted events per class within each unlabeled subject-run.
- Balanced pseudo-label self-training did not improve on plain balanced assignment, so prototype updates should wait until pseudo-label quality is better.
- Balanced assignment has heterogeneous subject-level effects, so any paper table should report aggregate gains alongside weak-subject failures rather than presenting it as a universal fix.
- Simple score-penalty gating did not beat full balanced assignment, so avoiding harmful subject-level corrections likely needs a better instability/QC signal.
- Subject-level all-runs balancing reached `0.5746` subject-fold accuracy, below per-run balancing at `0.5826`, so the current adaptation baseline should keep the class-count constraint at the subject-run level.
- An independent-prediction imbalance gate is the best current adaptation variant, reaching `0.5877` subject-fold event accuracy / `0.5876` macro F1 when balancing only subject-runs with class-count L1 imbalance `>= 4`. This should be reported as exploratory until the threshold is validated separately.
- Event-error anatomy shows the adapted features are much stronger for coarse motor grouping than exact labels: leg-vs-arm accuracy is `0.8411`, while exact leg-pair and arm-pair discrimination are about `0.5860` and `0.5887`. This supports a hierarchical analysis section and suggests the main remaining bottleneck is fine-grained within-pair stability.
- Event-position effects are visible: ordinal `0` and ordinal `6` are weakest, so the methods section should include a temporal/window-sensitivity diagnostic before claiming the remaining errors are only subject variability.

Publication framing should therefore avoid claiming that the transductive alignment score is a standard supervised classifier. It should be treated as evidence for domain shift unless the method section defines a legitimate test-time adaptation protocol using unlabeled target-run statistics.

## Later Phases

- Phase 2: weak-subject and run-consistency audit, including run-pair template stability, artifact/QC checks, and anatomical/preprocessing alignment checks.
- Phase 3: preprocessing/domain-shift ablation under the same subject-wise protocol, with hierarchical leg-vs-arm versus exact four-class reporting.
- Phase 4: task framing comparison: volume, trial clip, beta-map samples, and target-run adaptation, including independent event prediction versus known-design balanced assignment.
- Phase 4a: temporal/window-sensitivity experiments around weak event positions, especially the first event and ordinal `6`.
- Phase 5: architecture comparison under locked preprocessing and split protocol.
- Phase 6: subject/run-invariance experiments such as DANN-style adversarial domain heads, explicit run covariate removal, or test-time adaptation using unlabeled target-run statistics.
- Phase 7: interpretability using saliency or Grad-CAM overlap with motor-region atlas masks.
- Phase 8: final statistics, manuscript tables, and submission materials.

## Repository Rule

Keep private manuscript drafts, unpublished paper PDFs, Kaggle API tokens, raw data, checkpoints, and local logs out of this public repository. Track only code, configs, lightweight result summaries, and reproducibility notes.
