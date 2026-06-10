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

## What To Do Next

Run these checks in this order:

1. Tiny overfit sanity check.

Train on one subject-run or a tiny balanced subset. A small model should reach very high training accuracy quickly. If it cannot overfit 64-256 labeled samples, the pipeline/model/loss is broken.

2. Corrected-logits sanity check.

Disable output softmax for cross-entropy and disable output-level DropConnect, or move regularization before the classifier. Repeat a short pooled run with z-score normalization. This tests whether the chance-level legacy result was caused by the model/loss bug.

3. Event-window audit.

Compare extracted filenames against original BIDS event files: class label, onset, TR, HRF shift, and expected volume window. This is the most important dataset-level check.

4. Block-level pooled split.

If a pooled baseline is still needed, split by subject-run-class block or by run, not by individual volume. Random volume splits leak adjacent volumes from the same block.

5. Temporal clip baseline.

Use clips with HRF shift and z-score normalization rather than isolated raw volumes. Motor fMRI labels are block/trial-level signals; the model should see the temporal context that defines the event.

## Current Interpretation

We did not prove the full dataset is useless. We proved that the current extracted-volume plus legacy-wrapper setup does not produce defensible full-dataset learning. The next work should be diagnosis and corrected sanity checks, not more epochs of the same legacy run.
