# Dataset Diagnostic Findings

Last updated: 2026-06-10.

This note summarizes what the completed full-dataset runs tell us about the data and what to do next. It uses the final pooled legacy metadata from:

`/Users/USER/Documents/New project/status_2026-05-22_legacy_full_b6_v2_duration/thesis_session/thesis_legacy_full_dataset`

## What The Manifest Shows

- The extracted working set is structurally balanced: 23,808 samples, 62 subjects, 372 subject-runs, and 5,952 samples per class.
- Every subject has 6 runs.
- Every subject-run contains all 4 classes.
- Every subject-run-class block contains exactly 16 extracted 3D volumes.
- There are no filename parse failures and no missing class directories in the final manifest QC.

This means the basic folder/manifest construction is not obviously broken.

## What The Split Shows

The full-dataset pooled legacy baseline used a random sample split, not a subject/run/block split:

- All 62 subjects appear in train, validation, and test.
- All 372 subject-runs appear in train, validation, and test.
- 1,437 of 1,488 subject-run-class blocks are split across multiple splits.
- 993 blocks appear in train, validation, and test simultaneously.

So the pooled split is strongly leakage-prone. However, validation still stayed near chance. That is important: if the extracted samples carried a strong class signal or even an easy leakage shortcut, this split should have looked much better.

## Strongest Suspects

0. The subject-wise temporal run collapsed before it learned training data.

All five subject-wise CV folds ended near chance on training accuracy as well as validation accuracy. The final validation confusion matrices predicted a single class for every sample. This is not merely a subject-generalization failure; it means the current temporal training setup did not find a usable training signal.

1. The legacy run is not a clean test of the dataset.

The Kaggle legacy config used `task: volume`, `clip_length: 1`, `hrf_shift: 0`, and `normalization: none`. That asks the model to classify isolated raw 3D volumes with no temporal context and no per-volume z-scoring.

2. The legacy model/loss pairing is probably wrong.

The final Kaggle bundle uses `CrossEntropyLoss`, but the legacy model config sets `apply_output_softmax: true`. Cross-entropy expects raw logits. The model also applies DropConnect after the output activation, which perturbs class probabilities directly. This can weaken learning and makes the legacy result a poor basis for deciding whether the dataset itself is learnable.

3. The extraction should be audited against BIDS events.

Current labels come from class folders and filenames. The manifest proves the folders are balanced, but it does not prove each extracted `vol_id` was cut from the correct event window. The next high-value check is to rebuild expected windows from the original events files and compare them against the extracted filenames.

4. The original high score may reflect a different model and easier split, not the full-dataset task.

The original notebook-style model used padded 7x7x7 convolutions and no output softmax before cross-entropy. The later legacy wrapper differs from that architecture. The old 85% number should be treated as a historical subset result until reproduced with the exact original code and data subset.

## Tiny Overfit Sanity Result

The first corrected tiny-overfit check ran on Kaggle CPU using one balanced subject-run block from `thesis-batch-01`:

- selected block: `sub-01`, run `1`
- samples: 64 total, 16 volumes per class
- preprocessing: per-volume z-score, downsampled to `16 x 16 x 16`
- model/loss: small 3D CNN with raw logits and cross-entropy
- best training accuracy in 40 epochs: `0.9219`
- final training accuracy: `0.8906`

This is an important partial signal. A corrected tiny model can learn far above chance on one subject-run, so the class labels are unlikely to be pure noise. But it did not yet cleanly memorize the block, so this is not a full pass. The next diagnostic should use a smaller micro-overfit block or a stronger tiny model and should complete with a readable summary even if the threshold is not reached.

The stronger micro-overfit check then passed cleanly:

- selected block: `sub-01`, run `1`
- samples: 32 total, 8 volumes per class
- preprocessing: per-volume z-score, downsampled to `24 x 24 x 24`
- model/loss: slightly wider 3D CNN with raw logits and cross-entropy
- success target: best training accuracy at least `0.99`
- result: `1.0000` training accuracy at epoch `35`

This means the extracted class folders contain a learnable within-run signal when the model/loss/preprocessing are sane. The full-dataset failure should not be interpreted as "the data has no signal." It more likely points to the legacy wrapper configuration, temporal framing, normalization, and/or generalization split rather than totally corrupted labels.

## Feature Separability Result

Feature-separability probes were run on Kaggle to ask what separates the input volumes before training a large model.

The first probe used `thesis-batch-01`, with 12 subject-run blocks from `sub-01` and `sub-02`:

- samples analyzed: 768 total, balanced across the 4 classes
- target shape: `16 x 16 x 16`
- volumes per class per block: 16
- raw global mean/std nearest-centroid accuracy: `0.2643`
- mean within-block spatial-template accuracy: `0.7109`
- within-block range: `0.5156` to `0.8750`
- leave-one-block-out spatial-template accuracy: `0.2565`

The expanded probe used all seven batch datasets, with 42 subject-run blocks from `sub-01` through `sub-07`:

- samples analyzed: 2,688 total, balanced across the 4 classes
- target shape: `16 x 16 x 16`
- volumes per class per block: 16
- raw global mean/std nearest-centroid accuracy: `0.1678`
- mean within-block spatial-template accuracy: `0.6775`
- within-block range: `0.4844` to `0.8750`
- leave-one-block-out spatial-template accuracy: `0.2742`
- pairwise whole-slice class-template cosine similarities: all above `0.99996`

This tells us the input data does contain class-related differences inside individual subject-runs, but those differences do not transfer across runs/subjects in the current extracted-volume representation. Global intensity is not the explanation: the raw mean/std classifier is at or below chance. The class centroids are also extremely similar globally, so the usable signal is small, spatial, and probably sensitive to timing, run normalization, and subject anatomy.

Practical meaning: the next model should not simply be a bigger version of the legacy volume classifier. It should use corrected logits/loss, z-score normalization, temporal context around events, and a split strategy that measures run/subject generalization honestly. Before large training, the event-window audit should confirm that each class folder really matches the expected BIDS event timings and HRF-shifted volume windows.

## Representative Event-Window Check

For `sub-01`, run `1`, the extracted volume IDs match the original BIDS event schedule exactly for all four target classes:

- `Right leg movements`: expected and extracted `8-15` plus `216-223`
- `Upper arm movements`: expected and extracted `32-39` plus `192-199`
- `Left leg movements`: expected and extracted `72-79` plus `152-159`
- `Forearm movements`: expected and extracted `104-111` plus `120-127`

This does not prove every subject/run is correct, but it is a useful negative check: the representative run that passed the micro-overfit test is not obviously mislabeled by event timing. A full BIDS-events audit is still needed if all event files are mounted or downloaded.

## Clip HRF Window Policy Audit

A full filename-level audit across all seven extracted batch datasets found:

- subject-run-class groups: 1,488
- group length counts: every group has exactly 16 volumes
- contiguous segment counts: 2,976 segments of exactly 8 volumes
- with `clip_length=6`, `clip_stride=1`, `clip_window_stride=1`:
  - `hrf_shift=0` produces 8,928 clips
  - `hrf_shift=1` produces 5,952 clips
  - `hrf_shift=2` produces 2,976 clips
  - `hrf_shift=3` produces 0 clips

This confirms that the extracted class folders contain two 8-volume event windows per class/run. Applying a positive HRF shift inside `ClipDataset` does not create true HRF-delayed samples; it only crops within those already-extracted windows and discards valid clips. The cleaned configs now use `hrf_shift: 0` for the pre-extracted dataset. True HRF-shifted windows should be created during extraction from raw continuous 4D BIDS runs.

## Corrected Clip Baseline Result

A short corrected temporal-clip baseline was run on Kaggle with:

- `hrf_shift=0`
- per-volume z-score normalization
- no random spatial flips
- raw logits with cross-entropy
- 8 selected subjects: `sub-01` through `sub-08`
- subject-holdout split: train `sub-01` through `sub-06`, validate `sub-07` and `sub-08`
- diagnostic model: small `temporal_resnet3d`, target shape `32 x 32 x 32`, clip length `6`

Result:

- best validation accuracy: `0.25`
- best validation balanced accuracy: `0.25`
- best validation macro F1: `0.10`
- best validation MCC: `0.0`
- training accuracy stayed near chance over 5 epochs before early stopping
- final validation confusion matrix predicted only `Upper arm movements`
- Kaggle attached a Tesla P100, but the installed PyTorch build did not support P100 compute capability, so the run fell back to CPU

This means the corrected subject-wise temporal clip baseline still does not learn enough to justify scaling this exact model. Because the feature probe showed within-run separability, the next diagnostic is a same-subject run-holdout run. That will distinguish "cross-subject transfer is the main issue" from "the temporal model/training recipe is still failing."

The same corrected diagnostic was then run as a same-subject run-holdout split:

- selected subjects: `sub-01` through `sub-08`
- train runs: `1-5`
- validation run: `6`
- best validation accuracy: `0.25`
- best validation balanced accuracy: `0.25`
- best validation macro F1: `0.10`
- training accuracy stayed near chance over 5 epochs before early stopping
- final validation confusion matrix again predicted only `Upper arm movements`

This means the failure is not only cross-subject transfer. The current `temporal_resnet3d` diagnostic recipe is failing even when validating on a held-out run from the same subjects. The next required check is an intentional train=validation overfit on one subject-run using the same cleaned model path.

The train=validation overfit check on `sub-01`, run `1`, showed that the BatchNorm temporal model can fit the training batches but does not evaluate cleanly:

- train accuracy reached `1.00`
- train macro F1 reached `1.00`
- best train=validation evaluation accuracy reached only `0.50`
- best train=validation macro F1 reached only `0.375`
- evaluation repeatedly collapsed to a subset of classes despite train=validation

This points to unstable BatchNorm running statistics under tiny fMRI batches. The model code now supports `norm: group` for 3D ResNet encoders, and the corrected diagnostic configs use GroupNorm. The next overfit check should pass in evaluation mode before any broader baseline is trusted.

The GroupNorm follow-up did not fix the corrected temporal model. On the same `sub-01`, run `1`, train=validation clip split:

- best evaluation accuracy stayed at `0.25`
- best evaluation macro F1 stayed at `0.10`
- training loss converged to approximately `1.386`, consistent with uniform 4-class predictions
- the final confusion matrix predicted only `Left leg movements`

A smaller no-normalization temporal CNN probe also failed to overfit the same 24 corrected clips in eval mode, again staying at `0.25` accuracy and `0.10` macro F1. However, the same probe's non-neural nearest-centroid classifier on clip-mean spatial features reached:

- `0.75` train=eval accuracy on `sub-01`, run `1`
- `0.7083` leave-one-clip-out accuracy on `sub-01`, run `1`

This shows that corrected clips contain class-related spatial structure, but the current neural training recipes are not exploiting it reliably.

A broader corrected-clip feature-transfer diagnostic was then run on the first 8 subjects (`sub-01` through `sub-08`) using 1,152 clips and 13,824-dimensional clip-mean spatial features (`24 x 24 x 24`). Nearest-centroid results were:

- within subject-run leave-one-clip-out: accuracy `0.7613`, balanced accuracy `0.7613`, macro F1 `0.7619`, MCC `0.6824`
- same-subject run holdout, training runs `1-5` and validating run `6`: accuracy `0.2604`, balanced accuracy `0.2604`, macro F1 `0.2553`, MCC `0.0141`
- subject holdout, training `sub-01` through `sub-06` and validating `sub-07`/`sub-08`: accuracy `0.3160`, balanced accuracy `0.3160`, macro F1 `0.2981`, MCC `0.0911`

The interpretation is now sharper: class signal exists locally inside individual subject-runs, but the simple spatial signature does not transfer robustly across runs or subjects. The next modeling work should focus on run/subject nuisance control, normalization/domain alignment, and leakage-safe simple baselines before returning to larger neural models.

Domain-alignment variants were then added to the same corrected-clip feature diagnostic. These variants are transductive diagnostics because they use unlabeled validation-domain statistics, such as the mean or standard deviation of a held-out run. They should not be presented as ordinary supervised baselines, but they are useful for understanding the failure mode.

Best alignment results:

- same-subject run holdout improved from raw nearest-centroid accuracy `0.2604` / macro F1 `0.2553` to `0.5729` accuracy / `0.5634` macro F1 with per-run standardization plus cosine nearest centroids
- subject holdout improved from raw nearest-centroid accuracy `0.3160` / macro F1 `0.2981` to `0.4896` accuracy / `0.4807` macro F1 with per-run centering plus cosine nearest centroids
- all-selected train=eval improved from raw nearest-centroid accuracy `0.2917` / macro F1 `0.2844` to `0.7821` accuracy / `0.7832` macro F1 with per-run standardization plus cosine nearest centroids

This strongly suggests that run-specific baseline/scale effects are masking motor-class structure. The next practical modeling direction is to build non-transductive versions of this idea: estimate nuisance normalization from training runs only, use run/session-aware harmonization, add subject/run covariate controls, or train models with explicit domain-invariance objectives.

A reduced full-cohort version of the same diagnostic was run across all 62 subjects using lower-resolution `16 x 16 x 16` clip-mean spatial features and `clip_window_stride=8` to keep one non-overlapping clip per extracted event window. This produced 2,976 clips with 4,096 features each.

Full-cohort reduced results:

- within subject-run leave-one-clip-out was poor at this reduced/non-overlapping setting: accuracy `0.1546`, macro F1 `0.1541`, MCC `-0.1273`
- raw same-subject run holdout stayed near chance: accuracy `0.2621`, macro F1 `0.2561`, MCC `0.0164`
- raw subject holdout stayed near chance: accuracy `0.2542`, macro F1 `0.2183`, MCC `0.0059`
- same-subject run holdout improved to accuracy `0.5565`, macro F1 `0.5536`, MCC `0.4109` with per-run centering plus cosine nearest centroids
- subject holdout improved to accuracy `0.4667`, macro F1 `0.4653`, MCC `0.2902` with per-run centering plus cosine nearest centroids

This full-cohort reduced run is not directly comparable to the denser 8-subject `24 x 24 x 24` run because it uses lower spatial resolution and far fewer overlapping clips. Still, it independently confirms the main pattern: raw transfer remains chance-like, while transductive run-level alignment recovers substantial class structure.

Additional local sweeps on the downloaded full-cohort reduced feature matrix confirmed that this was not specific to the chosen run-6 or last-10-subject split:

- rotating the held-out run across all six runs gave raw nearest-centroid mean accuracy `0.2655` / mean macro F1 `0.2599`
- the same all-run sweep with per-run centering plus cosine nearest centroids gave mean accuracy `0.5228` / mean macro F1 `0.5216`
- six rotating subject folds gave raw nearest-centroid mean accuracy `0.2609` / mean macro F1 `0.2426`
- the same six subject folds with per-run centering plus cosine nearest centroids gave mean accuracy `0.4818` / mean macro F1 `0.4783`

These rotational checks strengthen the conclusion that run-level nuisance structure is systematic across the dataset, not an artifact of one chosen validation run or subject subset.

Class-wise inspection of the aligned full-cohort sweeps shows that the gain is not simply a new one-class shortcut. With per-run centering plus cosine nearest centroids, every class has above-chance recall across rotating held-out runs and subject folds. Forearm movements are consistently easiest, while left-leg and upper-arm movements are weaker and more variable. This suggests the input data carries distributed motor-class information, but it is partially masked by subject/run offsets and class-specific ambiguity.

The diagnostic script now writes rotating held-out-run and subject-fold summaries directly, including mean per-class recall and precision. Kaggle version 12 of `b6uejhvvnmiwb/thesis-corrected-clip-baseline` completed from repo commit `2a7bbbe` and reproduced the rotating full-cohort alignment result in `summary.json`. It did not include the later class-wise or train-only fields.

The script also includes train-only alignment probes. These estimate global, subject-level, run-id-level, or subject-run-level centering/standardization statistics only from the training split, with validation samples falling back to training-derived statistics when their domain key was unseen. Kaggle version 13 completed this updated code. This is the next decision point: if train-only alignment recovers a useful fraction of the transductive gain, the project can define a conventional preprocessing baseline; if it does not, the alignment result should be framed as evidence for test-time adaptation or domain-invariant modeling rather than as a standard supervised classifier.

The official version 13 results show train-only alignment is not enough. Same-subject held-out-run cosine nearest-centroid accuracy moved from `0.2641` raw to only `0.3044` with train-subject centering and `0.3185` with train-subject standardization. Subject holdout remained near chance, with the best checked cosine result around `0.2646` after train-global centering. The large `0.48-0.52` aligned accuracies therefore depend mostly on validation-domain run statistics, not on a simple supervised training-only centering recipe.

This does not make the result useless; it clarifies the result. The dataset appears to contain motor-class signal, but robust recovery across runs/subjects likely needs either an explicit test-time adaptation protocol that uses unlabeled target-run statistics, or a model trained to remove run/domain nuisance structure rather than a static train-only nearest-centroid baseline.

A local variance decomposition on the same full-cohort reduced feature matrix supports this interpretation. In the raw clip-mean feature space, class identity explained effectively `0.0000` of total variance, while subject explained about `0.9851` and subject-run explained about `0.9971`. After subject-run centering, the subject/run components were removed by construction and class variance rose to `0.0068`. That is still a small signal, but it is no longer buried under subject anatomy and run-offset structure. The diagnostic script now writes this decomposition into `summary.json` for future runs.

The denser full-cohort run using `clip_window_stride=1` at `16 x 16 x 16` completed as Kaggle version 14. The earlier full-cohort run used `clip_window_stride=8`, meaning only one non-overlapping clip per extracted 8-volume event window. The denser run shows that overlapping clips restore strong within-run structure but do not fix raw cross-run/subject transfer:

- clips analyzed: `8,928`
- within subject-run leave-one-clip-out: accuracy `0.7478`, macro F1 `0.7478`
- rotating held-out-run raw cosine: mean accuracy `0.2631`, macro F1 `0.2571`
- rotating subject-fold raw cosine: mean accuracy `0.2611`, macro F1 `0.2422`
- rotating held-out-run per-subject-run centering plus cosine: mean accuracy `0.5691`, macro F1 `0.5679`
- rotating subject-fold per-subject-run centering plus cosine: mean accuracy `0.5201`, macro F1 `0.5171`
- event-window voting over the overlapping clips improves the same adaptation protocol to `0.5833` held-out-run accuracy and `0.5312` subject-fold accuracy

This sharpens the problem statement. The extracted data has enough class information to classify clips within subject-runs, and target-run adaptation recovers a substantial fraction of that signal across domains. But supervised train-only alignment remains weak, especially for unseen subjects, so the next serious modeling work should either define a legitimate unlabeled target-run adaptation protocol or train an explicitly domain-invariant representation.

## What To Do Next

Run these checks in this order:

1. Run/subject transfer diagnostic.

Use simple feature baselines to quantify how much class structure survives each split type: within-run, held-out-run, held-out-subject, and held-out-session if available. Treat within-run success with cross-run failure as a nuisance/domain-shift warning, not as deployable classification. Use transductive run-normalization only as a diagnostic; any publishable model needs a non-transductive training-only normalization or a clearly defined test-time adaptation protocol.

Use the train-only alignment summaries to decide whether run/subject centering can be a standard supervised preprocessing step. If train-only centering remains near chance while transductive centering stays high, report the finding as a domain-shift/test-time-adaptation result and move to domain-invariant modeling.

2. Tiny overfit sanity check.

Train on one subject-run or a tiny balanced subset. A small model should reach very high training accuracy quickly. If it cannot overfit 64-256 labeled samples, the pipeline/model/loss is broken.

3. Corrected-logits sanity check.

Disable output softmax for cross-entropy and disable output-level DropConnect, or move regularization before the classifier. Repeat a short pooled run with z-score normalization. This tests whether the chance-level legacy result was caused by the model/loss bug.

4. Event-window audit.

Compare extracted filenames against original BIDS event files: class label, onset, TR, HRF shift, and expected volume window. This is the most important dataset-level check.

5. Block-level pooled split.

If a pooled baseline is still needed, split by subject-run-class block or by run, not by individual volume. Random volume splits leak adjacent volumes from the same block.

6. Temporal clip baseline.

Use clips with z-score normalization rather than isolated raw volumes. For the current pre-extracted class-folder dataset, use `hrf_shift=0`; true HRF-shifted clips require rebuilding class folders from the raw continuous 4D runs.

## Current Interpretation

We did not prove the full dataset is useless. We proved that the current extracted-volume plus legacy-wrapper setup does not produce defensible full-dataset learning. The corrected-clip feature diagnostics now suggest that motor-class signal is present inside runs but does not survive run/subject transfer under simple spatial features. The next work should be nuisance/domain-shift diagnosis and leakage-safe simple baselines, not more epochs of the same legacy run.
