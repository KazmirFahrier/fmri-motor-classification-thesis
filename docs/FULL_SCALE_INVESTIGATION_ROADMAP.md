# Full-Scale Investigation Roadmap

Last updated: 2026-07-09.

This document is the working memory for the full-dataset thesis investigation. The goal is not to find a quick number; the goal is to understand the dataset well enough that the final conclusion is defensible.

## Current Working Hypothesis

The full dataset contains motor-class signal, but that signal is small compared with subject, run, temporal-window, and subject-specific response geometry effects. The original high pooled-split result is not reproducible as leakage-aware full-dataset generalization with the current legacy model. The project should now focus on separating these failure modes:

- True data/QC failures in specific subjects or runs.
- Domain shift from subject/run nuisance structure.
- Temporal-window mismatch relative to the BOLD response.
- Coarse anatomical separability versus fine within-pair confusion.
- Personal subject geometry that can be recovered with labeled calibration.
- Limits of unlabeled or pseudo-labeled adaptation.

Recent update: the strongest zero-label full-cohort model is now a multi-scale hierarchy with regularized full-covariance LDA for the coarse leg-vs-arm gate and pair specialists. It reaches `0.8752` repeated nested balanced accuracy across all subjects and `0.8941` in the prespecified 60-subject QC sensitivity stratum. Coarse routing is nearly saturated, so the active modeling bottleneck is within-pair discrimination, especially forearm versus upper-arm.

## What We Have Tried

### Legacy Neural Baselines

- Original small pooled split remains the historical high result: `0.8522` accuracy.
- Full-dataset pooled legacy training was resumed through epoch 25 and stopped by controlled policy.
- Full-dataset pooled validation stayed near chance: epoch 25 accuracy `0.2629`, macro F1 `0.2501`, MCC `0.0206`.
- Full-dataset subject-wise 5-fold plus holdout completed and stayed chance-level: accuracy `0.25`, macro F1 `0.10`, MCC `0.0`.
- Corrected temporal ResNet and tiny CNN overfit probes did not produce reliable eval-mode learning on corrected clips.

Interpretation: more epochs of the same legacy model are not justified. The failure is not simply insufficient Kaggle runtime.

### Corrected Clip Feature Diagnostics

- Corrected event-window audit confirmed target-class folders contain the intended two 8-volume windows per class/run.
- Source BIDS event-window audit against OpenNeuro `ds004044` v2.0.3 found `0` timing anomalies across 62 subjects and 372 runs.
- Simple clip-mean spatial features show strong within subject-run signal.
- Raw held-out-run and held-out-subject transfer stay near chance.
- Transductive subject-run centering plus cosine recovers substantial event-level signal.
- Train-only centering does not recover most of the transductive gain.
- Raw feature variance is dominated by subject and subject-run identity, not class.

Interpretation: the data has signal, but ordinary train-only transfer is dominated by domain shift.

### Spatial Resolution

- Dense `24^3` features improved over lower-resolution features for adaptation diagnostics.
- Dense `32^3` did not materially improve subject generalization over `24^3`.

Interpretation: `24^3` is the current practical diagnostic sweet spot. Scaling spatial resolution alone is unlikely to solve the problem.

### Classifier Complexity

- Random-projection ridge and richer linear models underperformed cosine nearest centroids on aligned event features.
- A deployable two-stage centroid hierarchy did not beat flat four-class centroids.
- An oracle-coarse hierarchy reached `0.6767`, showing headroom if coarse grouping can be preserved while improving within-pair decisions.

Interpretation: the bottleneck is representation, alignment, temporal windowing, and subject stability, not simply the final linear classifier.

### Task-Design Adaptation

- Per-subject-run balanced assignment improved event accuracy by enforcing two predicted events per class.
- Subject-level all-runs balancing did not beat per-run balancing.
- Score-penalty gating did not solve harmful per-subject corrections.
- Class-count imbalance gating improved subject-fold event accuracy modestly to `0.5877`.
- Pseudo-centroid self-training from balanced assignments did not improve.

Interpretation: task-design constraints help as post-processing/test-time adaptation, but current pseudo-label quality is not sufficient for prototype updates.

### Temporal Offset And Windowing

- Averaging all three overlapping clips per event dilutes the signal.
- Offset `2` is best among the current offsets.
- Offset-2 plus task-design balancing reaches `0.6376` subject-fold event accuracy and `0.6851` held-out-run event accuracy.
- Temporal-weight mixtures over offsets `0`, `1`, and `2` do not beat pure offset `2`.

Interpretation: the useful signal is late in the current extracted event window. The next temporal work should test later or longer HRF-aligned windows from continuous BOLD, not more mixtures of the existing three offsets.

### Continuous-Window Hierarchy And Arm Residuals

- Full-cohort continuous-window extraction completed and validated `offset 3, length 8` as the fixed cohort window.
- A multi-scale hierarchy with native coarse features and smoothed pair specialists reached `0.8201`, then covariance-aware coarse and pair LDA raised the repeated nested estimate to `0.8752`.
- Pair-specific temporal controls using saved means (`2:6`, `3:6`) were negative.
- Naive multi-window concatenation and explicit reconstructed tail contrasts were negative under nested arm evaluation.
- Diagonal QDA was negative; full LDA remained selected for both leg and arm.
- Continuous temporal-basis extraction from denoised BOLD completed for all 62 subjects at fixed `3:8`.
- Nested arm screening of mean, linear, quadratic, early-vs-late, tail-vs-body, and combined basis representations is neutral overall: selected temporal-basis representations average `0.8357` balanced arm accuracy versus `0.8358` for mean-only.
- `mean_plus_tail` is the only borderline lead (`0.8452` balanced arm accuracy), but its paired interval crosses zero and independent accuracy falls.
- An end-to-end hierarchy using `mean + tail_vs_body` only in the arm branch reaches `0.8766`, only `+0.0014` over the `0.8752` baseline with uncertainty crossing zero.

Interpretation: mean-window spatial covariance is near its practical limit, and simple linear temporal basis concatenation does not unlock the remaining arm errors. If there is more recoverable arm signal, it likely needs richer sequence/HRF modeling, calibration/personalization, or subject-specific geometry rather than another handcrafted temporal coefficient.

### Exact-Class Subject Calibration

- The repeated exact-class calibration evaluator reproduces the saved fold-weighted zero-label hierarchy exactly at `0.875164` before applying any target update.
- Updating only target-subject arm class means from non-evaluation runs reaches subject-averaged balanced accuracy `0.8861`, `0.8892`, and `0.8913` with three, four, and five labeled runs.
- Five-run arm calibration improves the all-subject result by `+0.0157` (95% paired subject interval `+0.0049` to `+0.0269`) and reaches `0.9088` in the QC-60 sensitivity stratum.
- Independent exact decoding rises from `0.8250` to `0.8430`, and all five repeated subject partitions improve.
- Leg calibration is neutral or harmful, both-branch calibration is less stable, one-run arm calibration is harmful, and two runs remain inconclusive.
- Twenty-five subjects still regress under five-run validation-selected arm calibration, so this is a labeled-personalization ceiling rather than a universal adaptation rule.

Interpretation: the main residual transfer failure is target-specific forearm/upper-arm mean geometry. The next calibration experiment should gate or continuously shrink the arm update using calibration-run evidence only; held-out-run labels and post-hoc residual groups remain forbidden.

### Error Anatomy

- Coarse leg-vs-arm classification is much easier than exact four-class classification.
- Offset-2 imbalance-gated subject-fold exact accuracy is `0.6371`, while leg-vs-arm accuracy is `0.8726`.
- Residual errors concentrate within anatomical pairs: left/right leg and forearm/upper arm.
- Event ordinal `0` remains weak even after offset-2 selection; ordinal `3` is strongest.

Interpretation: the model is not randomly guessing. It sees broad motor-system structure but struggles with fine within-pair and run-position effects.

### Weak-Subject Taxonomy

- `sub-52` and `sub-42` are internally inconsistent even after offset-2 selection.
- `sub-17`, `sub-20`, `sub-26`, `sub-27`, and related cases can be internally coherent but cohort-mismatched.
- Same-subject leave-one-run accuracy correlates strongly with centroid margin.
- Removing weak subjects helps only modestly, so weak subjects are important but not the entire problem.
- Saved-feature QC shows that some failures are run-specific. `sub-52` has poor runs `1`, `3`, `4`, and `5` but usable runs `2` and `6`; `sub-42` has especially bad runs `2` and `4`; `sub-54`, `sub-63`, and `sub-20` also have very weak same-class versus different-class geometry. Stable subject `sub-62` has positive within-run geometry in all runs.

Interpretation: weak subjects should not be treated as one bucket. Some need QC/repair/exclusion analysis; others need personalization or domain adaptation.

### Subject Calibration

- Labeled subject calibration is the strongest current positive control.
- Offset-2 source/subject centroid blending plus balanced assignment reaches:
- `0.6681` with one labeled calibration run.
- `0.6939` with two labeled calibration runs.
- `0.7093` with three labeled calibration runs.
- `0.7224` with five labeled calibration runs.
- Validation-selected blending remains strong: `0.6873` with two runs, `0.7026` with three, `0.7151` with five.
- `sub-17`, `sub-20`, `sub-27`, and `sub-63` improve strongly with labeled calibration.
- `sub-52` and `sub-42` do not recover reliably.

Interpretation: many subjects contain usable personal class geometry, but the cohort model does not align to it zero-shot. Calibration is a serious modeling lane, but it must be reported separately from held-out-subject zero-shot generalization.

### Unlabeled Subject Adaptation

- Naive pseudo-labeled subject adaptation did not recover the labeled calibration gain.
- Source-only balanced assignment is `0.6344`.
- Best pseudo-subject blend is only `0.5995`.
- Calibration pseudo-labels are about `0.6344` accurate and amplify errors when used as subject prototypes.

Interpretation: simple self-training is not enough. Future semi-supervised methods need better pseudo-label confidence, coarse-to-fine structure, run-consistency filters, or a stronger representation before prototype adaptation.

## What Still Needs To Be Tried

### Raw QC And Subject Repair

- Use the saved-feature QC script as the first triage layer while raw motion/confound files are unavailable locally.
- Locate or regenerate motion/confound summaries if available.
- Compare bad runs from `sub-52`, `sub-42`, `sub-54`, `sub-63`, and `sub-20` against stable runs from `sub-62` and `sub-30` for motion, artifacts, registration quality, signal variance, and run-specific anomalies.
- Inspect whether specific classes or runs are corrupted within weak subjects.
- Test exclusion or repair policies separately from ordinary personalization.

Priority: high, because it determines whether some subjects should be modeled, repaired, or excluded with justification.

### Temporal Window Re-Extraction

- Treat linear event-time detrending and fixed `3:8` continuous windows as the current zero-label preprocessing baseline.
- The full-cohort candidate-window validation is complete; fixed `3:8` remains the defensible cohort default.
- Compact temporal-basis maps from continuous BOLD for `3:8` are extracted and tested.
- Do not spend more cycles on simple basis concatenation unless a new hypothesis changes the representation family.
- Test run-start stabilization or dropping/handling the first event separately.
- Evaluate whether ordinal `0` and ordinal `6` weaknesses are windowing, baseline, or sequence-context problems.

Priority: high, because saved mean-window temporal variants are exhausted and the arm branch is now the main residual bottleneck.

### Coarse-To-Fine Modeling

- Build stronger coarse leg-vs-arm models.
- Use coarse confidence to gate exact four-class predictions.
- Train or evaluate pair-specific classifiers for left/right leg and forearm/upper arm.
- Try coarse-first pseudo-labeling before any exact-class self-training.

Priority: high, because leg-vs-arm accuracy is high while within-pair accuracy is the main bottleneck.

### Calibration And Personalization

- Treat labeled target-subject calibration as a positive-control benchmark.
- Use linearly detrended offset-2 features as the primary calibration representation. Fixed five-run calibration reaches `0.7984`, and validation-selected calibration reaches `0.7957`.
- Validate calibration-run count and alpha selection without held-out run labels.
- Test few-shot subject calibration with fixed protocols suitable for a paper.
- Explore semi-supervised calibration only after pseudo-label quality improves.
- Compare subject-specific calibration versus cohort-only versus cohort-plus-calibration across subject groups.

Priority: high, because calibrated results are currently the strongest full-cohort event-level results.

### Domain-Invariant Representation Learning

- Test adversarial subject/run heads or domain-confusion objectives.
- Test explicit nuisance regression or residualization.
- Test subject/run normalization layers that do not rely on held-out labels.
- Compare against simple centroid baselines under identical splits.

Priority: medium-high, because raw/train-only transfer is the central generalization failure.

### Alternative Feature Targets

- Test beta-map or GLM-derived event features.
- Test ROI/motor-atlas features instead of whole dense grids.
- Test temporal summary features beyond simple clip means.
- Compare volume-level, clip-level, event-level, beta-level, and subject-calibrated protocols.

Priority: medium, because current dense clip features are informative but may not be the best scientific representation.

### Better Pseudo-Labeling

- Confidence-filter pseudo labels before prototype updates.
- Use only high-confidence coarse labels, then refine within pairs.
- Use run-consistency agreement across multiple calibration runs.
- Require pseudo-label class balance and margin thresholds.
- Compare pseudo-label quality directly before using pseudo labels for adaptation.

Priority: medium, because naive pseudo-labeling is already a clear negative.

### Architecture Revisit

- Only revisit larger neural architectures after locking preprocessing and diagnostics.
- First require small eval-mode overfit success on corrected event windows.
- Compare neural models to centroid baselines under the exact same subject/run/calibration protocols.
- Avoid claiming improvement unless it beats offset-2 centroid/balanced/calibrated baselines.

Priority: medium-low until the preprocessing and representation questions are clearer.

## Stop Conditions And Guardrails

- Do not resume the legacy full-dataset pooled run unless the decision policy is explicitly changed.
- Do not treat transductive centering or balanced assignment as ordinary supervised classification.
- Do not report labeled calibration as zero-shot held-out-subject generalization.
- Do not use pseudo-label prototypes unless pseudo-label quality is measured.
- Do not scale architecture or resolution without a diagnostic reason.
- Do not hide chance-level full-dataset results; they are scientifically important evidence.

## Current Best Benchmarks To Beat

| Protocol | Current score |
| --- | --- |
| Full legacy pooled epoch 25 | `0.2629` validation accuracy |
| Full subject-wise holdout | `0.25` accuracy |
| Offset-2 independent subject-fold event prediction | `0.6022` accuracy |
| Offset-2 balanced subject-fold event prediction | `0.6370` accuracy |
| Offset-2 imbalance-gated subject-fold event prediction | `0.6376` accuracy |
| Offset-2 held-out-run balanced event prediction | `0.6851` accuracy |
| Offset-2 + linear time detrending, subject-fold balanced | `0.7176` accuracy |
| Offset-2 + linear time detrending, held-out-run balanced | `0.7655` accuracy |
| Fixed `3:8` full-covariance hierarchy, repeated subject folds | `0.8752` balanced accuracy |
| Fixed `3:8` full-covariance hierarchy, 60-subject QC stratum | `0.8941` subject-averaged accuracy |
| Fixed `3:8` arm pair branch | `0.7955` pair accuracy |
| Fixed `3:8` temporal-basis selected arm branch | `0.8357` balanced arm accuracy versus `0.8358` mean-only |
| Fixed `3:8` mean-plus-tail hierarchy | `0.8766` balanced accuracy; not meaningfully above `0.8752` baseline |
| Fixed `3:8` hierarchy + validation-selected arm calibration, 3 runs | `0.8861` subject-averaged balanced accuracy |
| Fixed `3:8` hierarchy + validation-selected arm calibration, 5 runs | `0.8913` subject-averaged balanced accuracy; `0.9088` QC-60 |
| Fixed `3:8` hierarchy + arm calibration, independent prediction, 5 runs | `0.8430` accuracy |
| Offset-2 + linear detrending + validation-selected calibration, 3 runs | `0.7757` accuracy |
| Offset-2 + linear detrending + validation-selected calibration, 5 runs | `0.7957` accuracy |
| Offset-2 + linear detrending + fixed calibration, 5 runs | `0.7984` accuracy |
| Offset-2 validation-selected labeled calibration, 3 runs | `0.7026` accuracy |
| Offset-2 validation-selected labeled calibration, 5 runs | `0.7151` accuracy |
| Offset-2 fixed labeled calibration, 5 runs | `0.7224` accuracy |
| Best naive unlabeled pseudo-subject adaptation | `0.5995` accuracy |

## Immediate Next Experiments

1. Build and validate a calibration-only gate or adaptive shrinkage rule for the exact-class arm update, using no held-out-run labels.
2. Extract richer within-event sequences only for models that preserve temporal order; simple polynomial and contrast bases are closed.
3. Extend targeted raw/post-detrend QC to the worst run-level cases from `sub-54`, `sub-63`, and `sub-20`, and define a post-repair QC rule.
4. Compare transductive run detrending with training-only nuisance regression or domain-invariant temporal residualization.
5. Improve pseudo-labeling only after measuring whether the hierarchy raises target arm pseudo-label quality.
6. Revisit neural sequence models only after a small eval-mode overfit probe succeeds on the corrected event protocol.
