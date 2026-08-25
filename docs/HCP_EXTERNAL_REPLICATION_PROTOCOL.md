# Prespecified HCP External Replication Protocol

Protocol freeze date: 2026-08-24.

Status: frozen before HCP outcome access. This protocol tests whether the preprocessing mechanism discovered in `ds004044` transfers to a distinct motor task cohort. It is not an exact replication of the four original anatomical labels.

## Research Question

Does target run normalization followed by native spatial smoothing improve subject independent decoding of motor conditions in an external cohort?

The confirmatory analysis is deliberately limited to the mechanism supported by the internal investigation. It does not test the custom hierarchy as a novel model, and it does not use repetition consistency or any other test set assignment constraint.

## Cohort And Access

Use only the [HCP Young Adult 2025 release](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release). Do not mix it with the older S1200 release.

The motor task contains left and right finger movement, left and right toe movement, and tongue movement. Each of the two runs contains two blocks for each condition according to the [official HCP task protocol](https://hcp-db.humanconnectome.org/hcp-protocols-ya-task-fmri).

Primary sample:

1. Use the documented unrelated subject package from the 2025 release if one is available.
2. Otherwise obtain restricted family structure and keep every family entirely inside one outer fold. HCP family structure is restricted information under the [HCP restricted data terms](https://hcp-db.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage).
3. Include the first 100 eligible subjects in ascending HCP subject identifier order.
4. A subject is eligible only when both motor runs, minimally preprocessed volumetric BOLD data, motion regressors, and complete movement timing files are available.
5. Record every exclusion and its reason before model fitting.

## Frozen Outcomes

Primary outcome:

* Five class independent balanced accuracy for left finger, right finger, left toe, right toe, and tongue. Chance is `0.20`.

Secondary outcome:

* Four class independent balanced accuracy after excluding tongue blocks. The four classes are left finger, right finger, left toe, and right toe. Chance is `0.25`.

The secondary outcome is declared now and must be reported regardless of direction. It is not an exact recreation of the original forearm and upper arm task.

## Frozen Event Representation

Use minimally preprocessed volumetric BOLD in HCP standard space. Create one map for each movement block using a least squares separate first level model:

* one canonical hemodynamic response regressor for the target movement block,
* one canonical hemodynamic response regressor combining all other movement blocks,
* the supplied motion nuisance regressors and their temporal derivatives,
* run intercept and linear drift terms,
* no spatial smoothing inside the first level model.

The target block beta map is one classification observation. This produces 20 observations per complete subject, four per class. Failed or rank deficient first level fits are logged and excluded by the fixed eligibility rule rather than repaired after viewing classification outcomes.

## Frozen Preprocessing Conditions

Evaluate the following conditions using identical outer folds and classifier settings:

1. Raw target block beta maps with training subject feature standardization only.
2. Target subject centering.
3. Target run centering.
4. Target run centering plus per feature linear detrending across block order.
5. Target run centering plus native Gaussian smoothing selected inside the training data.

For condition 5, smooth each target block beta map before vectorization. The frozen Gaussian standard deviation candidates are `0`, `2.0`, `2.8`, and `4.0` millimeters. Select the candidate using inner subject grouped balanced accuracy. No spatial mask, parcel, voxel subset, timing option, or smoothing candidate may be changed after outcome access.

For every condition, fit all means, trends, standardizers, feature filters, and model parameters using training subjects only. Test run centering may use the unlabeled maps from that test run because the operation does not use class labels. Report this explicitly as test run adaptation.

## Frozen Classifier And Validation

Primary classifier: linear support vector machine with class balanced weights.

Hyperparameter grid: `C` in `0.01`, `0.1`, `1`, and `10`.

Validation:

* five outer folds grouped by subject and family,
* five fixed split seeds: `20260824`, `20260825`, `20260826`, `20260827`, and `20260828`,
* four inner folds grouped by subject and family,
* preprocessing and `C` selected using inner balanced accuracy,
* predictions pooled only after every outer test fold is complete.

No class count assignment, Hungarian matching, repetition consistency, test label calibration, or subject label personalization is permitted.

## Confirmatory Contrasts

Primary contrasts:

1. Target run centering minus the raw condition.
2. Nested native smoothing minus target run centering without smoothing.

Use paired subject bootstrap confidence intervals with `20,000` resamples and seed `20260824`. Apply Holm correction across the two primary contrasts at familywise alpha `0.05`.

Report balanced accuracy, macro F1, Matthews correlation coefficient, class recall, subject level accuracy, fold results, and seed results. Report both point estimates and confidence intervals.

## Exact Pipeline Null

Run `200` label permutations for the five class primary outcome. Shuffle labels within subject and run while preserving class counts. Each permutation must rerun inner preprocessing selection and classifier selection. The empirical p value is `(1 + null scores at least as large as observed) / 201`.

The permutation procedure must be completed for the final selected pipeline, not only for a fixed classifier fitted after model selection.

## Acceptance Rules

The preprocessing mechanism is externally confirmed only when:

* the paired confidence interval for target run centering minus raw is above zero,
* the exact pipeline permutation p value is below `0.05`,
* no family or subject crosses outer train and test partitions,
* all prespecified outcomes and exclusions are reported.

Native smoothing is considered separately confirmed when its paired confidence interval over target run centering is above zero after Holm correction. A null smoothing contrast does not invalidate a positive run normalization result.

If the external result fails, reopen only the implicated timing, registration, family structure, or domain shift diagnostic. Do not restart an unrestricted internal architecture search.

## Required Artifacts

Before evaluation, archive:

* this frozen protocol and its Git commit,
* the HCP release identifier and downloaded file manifest,
* subject eligibility and exclusion table,
* subject and family fold assignments,
* software environment lock file,
* source hashes for every analysis script.

After evaluation, archive predictions, selected parameters, fold metrics, subject metrics, permutation scores, bootstrap draws or deterministic seeds, logs, and a machine readable summary. The manuscript must label this analysis as external mechanism replication rather than exact anatomical label replication.
