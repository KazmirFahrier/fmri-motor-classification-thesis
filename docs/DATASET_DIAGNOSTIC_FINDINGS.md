# Dataset Diagnostic Findings

Last updated: 2026-06-14.

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

This did not prove every subject/run was correct, but it was a useful early negative check: the representative run that passed the micro-overfit test was not obviously mislabeled by event timing. A later full source BIDS event-window audit extended this check to all extracted subjects/runs and found zero target-class timing anomalies.

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

A local shrinkage sweep on version 14 tested subtracting `alpha * subject_run_mean` before cosine nearest-centroid classification. Performance peaked sharply at full centering (`alpha=1.0`): trial-level held-out-run accuracy `0.5833` and subject-fold accuracy `0.5312`. Partial centering (`alpha=0.9`) dropped to `0.3730` / `0.3441`, and over-centering (`alpha=1.1`) similarly dropped to `0.3683` / `0.3446`. This suggests the dominant nuisance is close to an additive subject-run offset in the clip-mean feature space.

A subject-difficulty pass on the version 14 dense target-run adaptation predictions found wide subject-level variation at event-window voting level. Overall subject-fold trial accuracy was `0.5306`, but the worst subjects were far below chance or near chance (`sub-52`: `0.1875`, `sub-42`: `0.2083`, `sub-17`: `0.2708`), while the best subjects were much stronger (`sub-30`: `0.8125`, `sub-62`: `0.7292`, `sub-10`: `0.7083`). Held-out run difficulty was much less variable, ranging from `0.4819` to `0.5605`. After run centering, the remaining bottleneck appears to be subject-level robustness rather than a single bad run.

The full-cohort `24 x 24 x 24`, `clip_window_stride=1` diagnostic completed as Kaggle version 15. It keeps the dense overlapping clip policy from version 14 but restores more spatial detail. Higher resolution helps target-run adaptation but does not solve raw/train-only transfer:

- clips analyzed: `8,928`
- feature dimension: `13,824`
- within subject-run leave-one-clip-out: accuracy `0.7527`, macro F1 `0.7528`
- rotating held-out-run raw cosine: mean accuracy `0.2692`, macro F1 `0.2636`
- rotating subject-fold raw cosine: mean accuracy `0.2605`, macro F1 `0.2422`
- rotating held-out-run per-subject-run centering plus cosine: mean accuracy `0.5897`, macro F1 `0.5885`
- rotating subject-fold per-subject-run centering plus cosine: mean accuracy `0.5562`, macro F1 `0.5532`
- event-window voting over overlapping clips improves the same adaptation protocol to `0.6001` held-out-run accuracy and `0.5654` subject-fold accuracy

The higher-resolution run confirms that spatial detail matters, but only within the target-run adaptation framework. Train-only alignment remains weak: same-subject held-out-run train-subject centering reached `0.3387`, while subject holdout reached only `0.2701`. The weak-subject pattern also persists: `sub-52` remained very poor at `0.1667` subject-fold trial accuracy, while stronger subjects such as `sub-30` and `sub-62` reached `0.75`. The next data-understanding step should focus on the consistently weak subjects and whether their event timing, extracted windows, motion/artifact profile, or anatomical alignment differs from the easier subjects.

Follow-up weak-subject probes show that this is not merely cross-subject mismatch. Same-subject leave-one-run event-level classification after per-run centering averaged `0.5813` overall, but `sub-52` and `sub-42` remained at `0.1875` and `0.2083`, respectively. Their run-pair transfer matrices were inconsistent across runs, while stronger subjects showed stable cross-run mappings. A simple QC exclusion curve was modest: removing the worst 10 subjects raised dense `24³` target-run-adapted subject-fold trial accuracy from `0.5652` to only `0.6038`. So weak subjects are real and should be audited, but they are not the only reason full-cohort generalization is hard.

The full-cohort `32 x 32 x 32`, `clip_window_stride=1` diagnostic completed as Kaggle version 16. It tested whether the spatial-resolution gain from `16³` to `24³` continues when preserving still more spatial detail.

Version 16 confirmed that resolution is not the remaining bottleneck:

- clips analyzed: `8,928`
- feature dimension: `32,768`
- within subject-run leave-one-clip-out: accuracy `0.7565`, macro F1 `0.7566`
- rotating held-out-run raw cosine: mean accuracy `0.2628`, macro F1 `0.2557`
- rotating subject-fold raw cosine: mean accuracy `0.2634`, macro F1 `0.2456`
- rotating held-out-run per-subject-run centering plus cosine: mean clip accuracy `0.5939`, macro F1 `0.5927`
- rotating subject-fold per-subject-run centering plus cosine: mean clip accuracy `0.5519`, macro F1 `0.5486`
- focused event-window voting raised the same adaptation protocol to `0.6052` held-out-run trial accuracy and `0.5608` subject-fold trial accuracy

Compared with `24³`, the `32³` run gives a tiny held-out-run event-level gain (`0.6052` vs `0.6001`) but a small subject-fold drop (`0.5608` vs `0.5654`). Train-only alignment remains weak (`0.3353` same-subject held-out-run with train-subject centering; `0.2660` subject holdout with train-subject centering). The practical conclusion is that dense `24³` is the current sweet spot for fast follow-up diagnostics, while `32³` is useful confirmation that the project has hit a domain-shift/subject-robustness wall rather than a simple spatial-resolution wall.

Version 16 also reproduced the same weak-subject pattern. Event-level subject-fold target-run adaptation averaged `0.5605`, but `sub-52` stayed at `0.1875`, `sub-42` at `0.2708`, and `sub-17`, `sub-63`, and `sub-20` at `0.3333`. Stronger subjects such as `sub-62`, `sub-30`, `sub-10`, and `sub-47` reached roughly `0.71-0.77`. Because the same weak subjects were already poor at `24³`, the next data-understanding work should inspect run-to-run consistency, event timing, motion/artifact profile, and anatomical/registration effects for these subjects rather than scaling resolution or continuing the legacy model.

A focused event-level consistency audit was then run on the dense `24³` saved features. It averaged the three overlapping clips from each event window into one event feature, yielding 2,976 event windows from 8,928 clips, then centered features within each subject-run before measuring run-to-run class-template stability.

The audit found no malformed event groups and no event-count anomalies for the focus subjects, so the clearest weak-subject failures are not explained by missing extracted events. Instead, the geometry itself is unstable:

- `sub-52`: leave-one-subject adapted event accuracy `0.1667`, same-subject leave-one-run accuracy `0.1875`, mean run-pair accuracy `0.2667`, centroid margin `-0.1678`
- `sub-42`: leave-one-subject adapted event accuracy `0.3125`, same-subject leave-one-run accuracy `0.2083`, mean run-pair accuracy `0.2458`, centroid margin `-0.1641`
- `sub-54`: leave-one-subject adapted event accuracy `0.3750`, same-subject leave-one-run accuracy `0.4167`, centroid margin `-0.0384`
- `sub-30`: leave-one-subject adapted event accuracy `0.7708`, same-subject leave-one-run accuracy `0.6250`, mean run-pair accuracy `0.6000`, centroid margin `0.2168`

The centroid margin is the cosine similarity of the correct same-class template across runs minus the nearest wrong-class template. Negative margins mean a subject's run-to-run class geometry is effectively scrambled. Across subjects, same-subject leave-one-run accuracy correlated strongly with centroid margin (`r = 0.697`), while leave-one-subject adapted accuracy had moderate correlations with same-subject leave-one-run accuracy (`r = 0.470`) and centroid margin (`r = 0.384`). This supports a more precise hypothesis: the hard cases are not just hard because they differ from other subjects; several are internally inconsistent across their own runs.

A full source BIDS event-window audit was then run against OpenNeuro `ds004044` version `2.0.3`. The audit fetched only lightweight source `events.tsv` and `task-motor_bold.json` files from the OpenNeuro GitHub mirror, using the shared repetition time of `2.0` seconds. For all 62 extracted subjects and all 372 runs, each extracted target-class event window matched the source onset/TR-derived volume start exactly:

- subjects checked: `62`
- runs checked: `372`
- malformed or mismatched target-class windows: `0`
- subjects with timing anomalies: none

This rules out the broadest suspected extraction bug: the target class folders are not systematically shifted, swapped, or missing event windows relative to source BIDS timing. The weak-subject problem is therefore more likely to involve signal quality, subject/run alignment, motion/artifact structure, denoising/extraction choices, or genuine run-to-run response instability.

A fast event-level model sweep then tested whether the moderate target-run adaptation score was limited by using an overly simple nearest-centroid classifier. It was not. On dense `24³` event features, cosine nearest centroids remained best after subject-run centering:

- held-out-run event accuracy: cosine centroid `0.5981`; best random-projection ridge `0.5346`
- subject-fold event accuracy: cosine centroid `0.5669`; best random-projection ridge `0.4892`
- raw or train-global-centered subject-fold variants stayed near chance at roughly `0.26-0.27`

This is a useful negative result. It suggests the next improvement is unlikely to come from swapping the final classifier on the same feature space. The classifier is already exploiting a simple class-template geometry; the real work is to build a representation or adaptation protocol where those class templates are stable across runs and subjects.

A task-design-aware balanced assignment probe produced the first small constructive improvement on top of target-run centering. The motor paradigm contributes exactly two target events per class in each subject-run. Using that known unlabeled target-run structure, the probe keeps the same source cosine-centroid scores but assigns each 8-event subject-run exactly two predictions per class:

- held-out-run event accuracy improved from `0.5981` independent argmax to `0.6243` balanced assignment
- subject-fold event accuracy improved from `0.5669` independent argmax to `0.5826` balanced assignment

This is still a test-time adaptation result, not a standard supervised classifier. But it is more scientifically grounded than arbitrary post-processing because the class-count constraint comes from the task design. The improvement suggests that some target-run-centered predictions are close but globally inconsistent within a run; enforcing the known event balance can recover part of that lost structure.

The same balanced pseudo-labels were also tested as a way to adapt target-run class centroids. That refinement did not help. The best held-out-run pseudo-centroid variant reached `0.6240`, just below plain balanced assignment at `0.6243`, and subject-fold pseudo variants were neutral or slightly worse than the `0.5826` balanced assignment baseline. This suggests the balanced assignments are useful as a constraint, but they are not yet accurate enough to safely update class prototypes within each target run.

Per-subject effects show why this should be framed carefully. Balanced assignment improved some subjects, including `sub-54` (`0.3542` to `0.4375`), `sub-62` (`0.7083` to `0.7500`), `sub-13` (`0.5625` to `0.7083`), and `sub-04` (`0.5417` to `0.6875`). But it worsened several others, including `sub-52` (`0.1667` to `0.1458`), `sub-63` (`0.4167` to `0.3125`), and `sub-20` (`0.3958` to `0.3333`). So the known-design constraint improves the cohort average, but it does not solve the hardest weak-subject cases and can amplify wrong score geometry for unstable subjects.

A confidence-gated version was tested by applying balanced assignment only when the per-event centroid-score penalty was below a fixed threshold. This did not beat full balanced assignment. Conservative gates improved over independent argmax but peaked below or equal to full balance: held-out-run `0.6243` and subject-fold `0.5826`. Simple score-penalty gating therefore does not identify the subject-runs where balanced assignment is harmful.

A wider subject-level balanced assignment was also tested by enforcing equal class counts across all available target runs for each held-out subject. This can be viewed as asking whether strong runs can rescue weak runs when the target subject is adapted as one larger 48-event group. It did not improve the aggregate result. Subject-fold accuracy reached `0.5746`, above independent argmax at `0.5669` but below per-subject-run balancing at `0.5826`. Subject-level balancing improved over per-run balancing for 22 subjects, tied for 8, and worsened 32, so smoothing across runs is not the missing ingredient.

A subject-run QC analysis then measured unlabeled score-geometry signals to see whether they predict when balanced assignment helps or hurts. Most obvious signals were weakly correlated with balanced-assignment gain, including score penalty (`r = 0.069`), mean top-1 margin (`r = 0.081`), and score standard deviation (`r = 0.124`). The most useful simple signal was independent-prediction class-count imbalance (`r = 0.168`): subject-runs whose independent predictions visibly violate the known two-per-class run design are more likely to benefit from balancing.

Promoting that signal into a concrete gate produced the current best subject-fold event result. Applying balanced assignment only when the independent prediction counts have L1 imbalance at least `4` reached `0.5877` subject-fold accuracy / `0.5876` macro F1. This is better than independent argmax (`0.5669`) and always-balanced assignment (`0.5826`). It improved over always balancing for 27 subjects, tied for 19, and worsened 16. The same gate reduced held-out-run mean accuracy from `0.6243` to `0.6200`, so it should be treated as a subject-generalization clue rather than a universal improvement.

An event-error anatomy pass then showed that the current adapted feature space is much better at coarse anatomical grouping than exact four-way discrimination. Under the imbalance-gated subject-fold rule, exact event accuracy is `0.5874`, but leg-vs-arm accuracy is `0.8411`. Exact accuracy inside the leg pair is only `0.5860`, and exact accuracy inside the arm pair is `0.5887`. Most errors are therefore not arbitrary class confusions. They concentrate inside related motor groups:

- forearm predicted as upper arm: `203`
- right leg predicted as left leg: `202`
- left leg predicted as right leg: `175`
- upper arm predicted as forearm: `175`

This is one of the clearest current explanations of the full-dataset behavior. The data/features capture broad motor-system separation, but cross-subject/run alignment does not preserve enough fine-grained information to reliably distinguish left vs right leg or forearm vs upper arm.

The same pass found nontrivial run-position effects. The eight-event run sequence has eight observed class-order templates across the 372 subject-runs. Under the imbalance-gated subject-fold rule, event ordinal `0` is weakest (`0.5027` accuracy), ordinal `6` is also weak (`0.5269`), while ordinals `1` and `2` are strongest (`0.6317`). First occurrences of a class are slightly easier than second occurrences (`0.5981` vs `0.5766`), but that gap is smaller than the ordinal-position spread. This suggests run-start, temporal context, event-window placement, or sequence-context effects should be tested before assuming the remaining errors are purely subject anatomical variation.

The error anatomy also quantifies heterogeneity. The best subjects under the imbalance-gated rule include `sub-30` (`0.8125`), `sub-11` (`0.7917`), and `sub-46`/`sub-16`/`sub-08` (`0.7708`). The worst subjects remain `sub-52` (`0.1458`), `sub-17` (`0.2083`), `sub-27` (`0.3333`), and `sub-42` (`0.3542`). At the subject-run level, 25 subject-runs are perfect while 42 are at or below chance. This means the dataset is not uniformly noisy; it contains a mixture of very learnable subject-runs and severely unstable/corrupted or poorly aligned subject-runs.

A simple two-stage hierarchy was then tested directly. The model first predicts coarse leg-vs-arm with centroids, then predicts left-vs-right leg or forearm-vs-upper-arm using within-group centroids. This deployable hierarchy did not improve exact accuracy: subject-fold accuracy was `0.5618`, slightly below the flat four-class centroid baseline at `0.5669`. The coarse centroid stage itself reached `0.8222` subject-fold accuracy, confirming again that coarse motor grouping is much easier than exact classification.

The diagnostic oracle version is more informative. When the true leg-vs-arm group is supplied at test time and only the within-pair fine classifier is evaluated, subject-fold exact accuracy rises to `0.6767`. This is not deployable, but it estimates the headroom available if a future model can combine reliable coarse grouping with better fine within-pair alignment. The current failure is therefore not simply that a flat classifier ignores hierarchy; a naive hierarchy compounds coarse-stage errors and still inherits unstable within-pair geometry.

A clip-offset sweep then tested whether event averaging was hiding a better temporal slice. Each event currently contributes three overlapping clips with offsets `0`, `1`, and `2` relative to the eight-volume event window. Using only offset `2` was substantially better than averaging all offsets:

- independent subject-fold event accuracy: event mean `0.5669`, offset `0` `0.5059`, offset `1` `0.5640`, offset `2` `0.6022`
- independent held-out-run event accuracy: event mean `0.5981`, offset `0` `0.5353`, offset `1` `0.5974`, offset `2` `0.6425`
- coarse leg-vs-arm subject-fold accuracy: event mean `0.8262`, offset `2` `0.8600`

Combining offset `2` with the known-design class-balance constraint produced the strongest current event-level scores:

- held-out-run offset-2 balanced assignment: `0.6851` accuracy / `0.6851` macro F1
- subject-fold offset-2 balanced assignment: `0.6370` accuracy / `0.6370` macro F1
- subject-fold offset-2 imbalance-gated assignment: `0.6376` accuracy / `0.6377` macro F1
- subject-fold offset-2 balanced leg-vs-arm accuracy: `0.8811`

This is a major preprocessing clue. The earliest clip offset is weak, the middle offset roughly matches the event mean, and the latest offset is clearly strongest. Averaging all offsets therefore mixes less-informative early signal into the event representation. Future feature extraction should test later HRF-aligned windows, longer context, and learned temporal weighting rather than treating all overlapping clips equally.

A coarse temporal-weight grid then tested whether a simple mixture of offsets could beat the pure late clip. The grid used weights in `0.25` increments across offsets `0`, `1`, and `2`, evaluated with the same subject/run splits and prediction rules. No mixture beat pure `offset 2`. For subject-fold balanced assignment, pure `offset 2` reached `0.6370`, while the best nearby mixtures were lower:

- weights `(0: 0.00, 1: 0.25, 2: 0.75)`: `0.6205`
- weights `(0: 0.00, 1: 0.50, 2: 0.50)`: `0.6097`
- weights `(0: 0.25, 1: 0.00, 2: 0.75)`: `0.6088`

The same pattern held for independent predictions and held-out-run splits. This makes the temporal conclusion sharper: the useful signal is concentrated late in the extracted event window, and simple averaging or blending with earlier clips mostly dilutes it.

Repeating the event-error anatomy on offset `2` shows what the better temporal slice fixes. Under the offset-2 imbalance-gated subject-fold rule, exact event accuracy is `0.6371`, coarse leg-vs-arm accuracy is `0.8726`, exact leg-pair accuracy is `0.6492`, and exact arm-pair accuracy is `0.6250`. These are clear improvements over the event-mean anatomy (`0.5874`, `0.8411`, `0.5860`, `0.5887`), especially for within-pair discrimination. However, the qualitative error structure remains: residual mistakes are still mostly within anatomical pairs, not random cross-body confusions.

The timing/order structure also remains, though shifted upward. Event ordinal `0` is still weakest after offset-2 selection, improving from `0.5027` to `0.5457`. Ordinal `6` improves from `0.5269` to `0.5780`. Ordinal `3` becomes the strongest at `0.7070`. So offset `2` partly corrects the temporal-window issue, but first-event and sequence-position effects still need to be modeled or audited.

Weak subjects also persist after the temporal improvement. The worst offset-2 subjects under the imbalance-gated rule are still `sub-52` (`0.1667`), `sub-42` (`0.2500`), `sub-17` (`0.3125`), and `sub-27` (`0.3750`). This means the late-window improvement is real and broad, but the weak-subject problem is not merely an early-window artifact.

Repeating the subject/run consistency audit on offset `2` makes the weak-subject story more specific. The worst leave-one-subject cases are no longer all the same kind of failure:

- `sub-52`: leave-one-subject adapted accuracy `0.1667`, same-subject leave-one-run accuracy `0.2292`, centroid margin `-0.1686`
- `sub-42`: leave-one-subject adapted accuracy `0.2500`, same-subject leave-one-run accuracy `0.1667`, centroid margin `-0.1711`
- `sub-17`: leave-one-subject adapted accuracy `0.3125`, same-subject leave-one-run accuracy `0.6042`, centroid margin `0.1206`
- `sub-20`: leave-one-subject adapted accuracy `0.3958`, same-subject leave-one-run accuracy `0.6250`, centroid margin `0.0565`

So `sub-52` and `sub-42` look internally unstable even after selecting the stronger temporal slice: their own runs do not agree on class geometry. But `sub-17` and `sub-20` look more internally coherent while still transferring poorly from the rest of the cohort. Across subjects, leave-one-subject accuracy correlates moderately with same-subject leave-one-run accuracy (`r = 0.475`) and centroid margin (`r = 0.415`), while same-subject leave-one-run accuracy correlates strongly with centroid margin (`r = 0.745`). This suggests two separate next moves: audit/repair or potentially exclude internally inconsistent subjects, and test personalization/domain-adaptation methods for internally stable but cohort-mismatched subjects.

A labeled subject-calibration curve then tested that second move directly. For each target subject, it held out one run, used the other target-subject runs as labeled calibration data, blended target-subject centroids with source-cohort centroids, and evaluated the held-out run after subject-run centering. The best setting consistently used a light target-subject blend (`alpha=0.25`) plus the known two-events-per-class balanced assignment:

- source-only balanced assignment: `0.6344` accuracy / `0.6344` macro F1 / `0.8797` leg-vs-arm accuracy
- one labeled target-subject calibration run: `0.6681` accuracy / `0.6681` macro F1 / `0.9070` leg-vs-arm accuracy
- two labeled target-subject calibration runs: `0.6939` accuracy / `0.6939` macro F1 / `0.9164` leg-vs-arm accuracy
- three labeled target-subject calibration runs: `0.7093` accuracy / `0.7093` macro F1 / `0.9198` leg-vs-arm accuracy
- five labeled target-subject calibration runs: `0.7224` accuracy / `0.7224` macro F1 / `0.9227` leg-vs-arm accuracy

This is not a zero-shot full-subject result, and it must be reported as a calibration protocol. But scientifically it is a very useful positive control: many subjects contain stable personal class geometry that the cohort model does not align to by itself. The focus-subject results show the split clearly. `sub-17` improves from `0.2917` source-only balanced accuracy to `0.7500` with five calibration runs, `sub-20` from `0.4792` to `0.7083`, `sub-27` from `0.3333` to `0.7708`, and `sub-63` from `0.3958` to `0.6875`. `sub-52` and `sub-42` do not recover, reinforcing that they are likely QC/instability cases rather than ordinary personalization cases.

To reduce the most obvious overfitting concern, the calibration script also evaluates a validation-selected blend. For each held-out target run, it chooses `alpha` using only the labeled calibration runs via leave-one-calibration-run validation, then evaluates the held-out run. This keeps most of the fixed `alpha=0.25` gain:

- two calibration runs: fixed `alpha=0.25` balanced accuracy `0.6939`; validation-selected `0.6873`
- three calibration runs: fixed `0.7093`; validation-selected `0.7026`
- five calibration runs: fixed `0.7224`; validation-selected `0.7151`

The selected-alpha distribution is also reassuring: most validation splits choose `alpha=0.25`, especially for two and three calibration runs. That means light personalization is not merely a hindsight-picked parameter; it is often preferred by calibration-only validation.

The natural unlabeled follow-up was then tested by pseudo-labeling target-subject calibration runs with the source model and the known run-balance constraint, building pseudo-subject centroids, and blending them with source centroids. This did not work. Source-only balanced assignment stayed at `0.6344` accuracy, while the best unlabeled pseudo-subject blend reached only `0.5995`, even with five pseudo-labeled calibration runs. The pseudo-labels themselves were only about as accurate as the source baseline (`0.6344` under balanced assignment), and using them as subject-specific prototypes amplified their errors rather than adapting toward the labeled-calibration ceiling.

This negative result is important because it separates two claims. Labeled calibration proves that many target subjects have usable personal geometry, but simple self-training does not reveal that geometry without labels. Future semi-supervised work should not just recycle source pseudo-labels into prototypes. It likely needs stronger pseudo-label selection, coarse-to-fine constraints, confidence filtering, run-consistency gates, or a representation that makes target-subject pseudo-labels cleaner before prototype adaptation.

A saved-feature QC audit was then added as a local substitute for raw motion/confound QC, since no local confound files were found. It measures within-run same-class event consistency versus different-class separation after offset-2 subject-run centering. The result makes the weak-subject story more granular:

- `sub-52`: mean within-run leave-one-event accuracy `0.3125`, mean same-minus-different cosine `-0.0190`; runs `1`, `3`, `4`, and `5` have negative same-minus-different geometry, while runs `2` and `6` are usable.
- `sub-42`: mean within-run leave-one-event accuracy `0.2292`, mean same-minus-different cosine `0.0032`; runs `2` and `4` are the clearest failures.
- `sub-54`: mean within-run leave-one-event accuracy `0.1458`, mean same-minus-different cosine `-0.1574`; this is one of the weakest saved-feature QC subjects.
- `sub-63`: mean within-run leave-one-event accuracy `0.2500`, mean same-minus-different cosine `-0.1029`; run `1` is especially poor.
- `sub-62`: mean within-run leave-one-event accuracy `0.4375`, mean same-minus-different cosine `0.4718`; every run has positive same-minus-different geometry.

This suggests the next QC pass should not only label whole subjects as weak. It should inspect specific run failures, especially `sub-52` runs `1/3/4/5`, `sub-42` runs `2/4`, `sub-54` runs `1/4/6`, `sub-63` runs `1/4`, and `sub-20` run `3`. Raw motion, registration, and signal-quality checks should be targeted at those run-level failures first.

A targeted raw BOLD follow-up tested whether those feature failures can be explained by ordinary image-quality summaries. Five source runs were fetched directly from the public OpenNeuro S3 export and verified against the Git-annex MD5 keys: failed `sub-42/run-02` and `sub-52/run-03`, stronger within-subject controls `sub-42/run-05` and `sub-52/run-02`, and stable comparator `sub-62/run-03`.

Simple raw QC did not explain the feature failures. Median temporal SNR was similar across most runs, and the stable comparator had the largest DVARS/global-signal spike fraction (`0.0991`) despite positive class geometry. Failed `sub-52/run-03` had a much lower spike fraction (`0.0172`) but negative same-class geometry. Therefore a single tSNR, DVARS, or spike-count threshold would reject useful runs and retain failed runs.

The more important source-level effect was linear run-position drift. Offset-2 event patterns were centered within each run, then the unlabeled linear component associated with event time was removed. This substantially improved event geometry in the usable controls and partly repaired a failed run:

- `sub-42/run-05`: same-minus-different cosine `-0.070` to `0.923`; leave-one-event accuracy `0.50` to `0.875`
- `sub-52/run-02`: `0.066` to `0.502`; accuracy `0.375` to `0.625`
- `sub-52/run-03`: `-0.424` to `0.142`; accuracy `0.25` to `0.375`
- `sub-62/run-03`: `0.330` to `0.833`; accuracy `0.375` to `0.750`
- `sub-42/run-02`: `-0.584` to `-0.095`; accuracy `0.0` to `0.125`

This separates repairable temporal nuisance from a residual catastrophic-run problem. `sub-42/run-02` remains poor after detrending, while other runs reveal much cleaner motor-class structure. Exclusion should therefore be based on post-repair event consistency or a validated raw/feature QC model, not motion spikes alone.

The same linear detrending was then evaluated across all 2,976 offset-2 events. It uses only unlabeled event timestamps and features within each subject-run, independently for training and held-out runs. Under held-out-run evaluation, independent cosine-centroid accuracy increased from `0.6425` to `0.7050`, and balanced assignment increased from `0.6851` to `0.7655`. Under subject-fold evaluation, independent accuracy increased from `0.6022` to `0.6661`, while balanced assignment increased from `0.6370` to `0.7176`. Linear detrending improved independent subject accuracy for `54/62` subjects and balanced accuracy for `52/62`.

Several controls make this result harder to dismiss as generic feature removal. Five detrenders fitted to randomly permuted event timestamps lowered subject-fold balanced accuracy to `0.5808-0.5934`, below the non-detrended baseline. Quadratic true-time detrending also underperformed at `0.6355`. The useful operation is specifically removal of the linear true-time direction, which accounts for about `28.9%` of centered event-feature energy on average. Offset ordering also remains coherent after detrending: subject-fold balanced accuracy is `0.5842` at offset 0, `0.6574` at offset 1, and `0.7176` at offset 2.

This is the strongest current zero-label adaptation diagnostic, but it is still transductive preprocessing because the held-out run's unlabeled event features are used to estimate its nuisance trend. A publication must either define this as a test-time adaptation protocol or develop a training-only/domain-invariant analogue.

Labeled subject calibration was then rerun on the linearly detrended representation using the same held-out-run combinations and alpha-selection safeguards. The gains are complementary rather than redundant. Detrended source-only balanced accuracy is `0.7204`; fixed `alpha=0.25` source/subject blending reaches `0.7366` with one calibration run, `0.7650` with two, `0.7821` with three, `0.7926` with four, and `0.7984` with five. Calibration-only validation selects alphas that retain `0.7599`, `0.7757`, `0.7861`, and `0.7957` with two through five runs.

The combined experiment reinforces the weak-subject split. Detrending plus five-run validation-selected calibration raises `sub-17` to `0.8542`, `sub-27` to `0.8750`, `sub-54` to `0.6667`, and `sub-63` to `0.7500`. It does not rescue `sub-42` (`0.2083`) or `sub-52` (`0.2292`). This is evidence that temporal drift removal and personalization address separate, common failure modes, while a residual subset has class/run inconsistency that ordinary calibration cannot repair.

A follow-up run-QC policy sweep tested whether bad runs should simply be excluded after detrending. Each run was scored by label-aware within-run same-minus-different class geometry and leave-one-event accuracy, then policies dropped the lowest-scoring 5-30% of runs. Filtering low-QC runs from training only did not improve transfer: keep-all remains best for held-out-run balanced assignment (`0.7655`) and subject-fold balanced assignment (`0.7176`). Dropping the bottom 5% by within-run leave-one-event accuracy gives `0.7651` and `0.7165`. Oracle filtering of validation runs gives only a small subject-fold gain (`0.7224`) while retaining about `96.3%` of events. This means run exclusion is useful as a diagnostic/coverage label, but broad exclusion is not the main solution.

The worst post-detrend runs are not just the previously suspected `sub-42`/`sub-52` cases. By same-minus-different geometry, the weakest runs include `sub-20/run-03`, `sub-67/run-03`, `sub-54/run-06`, and `sub-23/run-01`. By within-run leave-one-event accuracy, the strict bottom-5% set includes 14 runs, with `sub-54` contributing three. This changes the interpretation: after detrending, `sub-42` and `sub-52` are not merely low within-run separability outliers. Their poor calibrated transfer likely reflects cross-subject/template mismatch or class correspondence problems that need targeted adaptation/QC, not a simple global run-exclusion threshold.

A detrended hierarchy sweep then tested whether exact errors were mainly caused by incorrect coarse leg-vs-arm routing. They were not. Predicted coarse-to-fine pairwise classification gives subject-fold accuracy `0.6659`, essentially flat independent accuracy (`0.6661`). Fused hierarchical scores with balanced assignment do not beat flat balanced assignment (`0.7176`). Even an oracle that supplies the true leg-vs-arm group for every held-out event reaches only `0.7238` exact subject-fold accuracy. Coarse routing is already strong; the remaining bottleneck is fine within-pair alignment and representation.

The next temporal extraction step was validated against the original thesis batch-generation pipeline before scaling. For five checksum-verified denoised OpenNeuro runs, continuous volumes were transformed with the original two-stage process: resize to `100³`, volume z-score, resize to `24³`, and volume z-score. Reconstructed onset+2/length-6 event means matched the saved Kaggle features almost exactly: mean cosine approximately `1.0` and relative RMSE around `6.3e-8` for every run.

An exploratory sweep over offsets `0-10` and lengths `2/4/6/8` on those five runs generated a fixed candidate shortlist. After linear detrending, offset `4`, length `2` had the best mean within-run leave-one-event accuracy (`0.85`), while offset `3`, lengths `6` and `8` had the strongest mean same-minus-different geometry. Because the sample deliberately mixes catastrophic, repairable, and stable runs, these numbers are not a performance claim. They justify a full-cohort comparison of six pre-specified windows: canonical `2:6` plus `3:6`, `3:8`, `4:2`, `5:4`, and `6:2`.

The first full-cohort streaming validation attempt exposed an event-table compatibility issue rather than a modeling result. OpenNeuro `events.tsv` files store `trial_type` as numeric task codes, while some local audit files already used text class names. The continuous-window loader now supports both forms (`3/4/5/6` map to left leg, right leg, forearm, and upper arm), and the extractor records skipped runs with reasons instead of failing on empty event sets. A quick local mapping check confirms that numeric and text versions of `sub-01/run-01` both produce the same eight target events at starts `8, 32, 72, 104, 120, 152, 192, 216`.

The fixed full-cohort run then completed all 62 subjects, 372 runs, and 2,976 events. The five-run candidate ranking generalized: offset `3`, length `8` is the strongest of the six pre-specified windows. With run centering, true-time linear detrending, and balanced assignment, it reaches `0.7960` held-out-run and `0.7421` subject-fold accuracy. The canonical offset `2`, length `6` window reaches `0.7655` and `0.7176`. Independent subject-fold accuracy rises from `0.6661` to `0.6966`, so the gain is not only an assignment artifact. Offset `3`, length `6` is close at `0.7370`, while offset `4`, length `2` falls to `0.7079`; the selected result favors a later and longer response window rather than the short-window candidate.

A leakage-safe pair-specific feature-selection experiment was then nested inside every outer run and subject split. Training labels rank voxels separately for left-versus-right leg and forearm-versus-upper-arm discrimination; inner folds choose the number of voxels and the coarse-score fusion weight. No held-out label is used for selection. On offset `3`, length `8`, the fused balanced model reaches `0.8028` held-out-run and `0.7643` subject-fold balanced accuracy, compared with the all-voxel balanced baseline at `0.7960` and `0.7421`. All six subject folds improve, by `0.0095-0.0341` absolute.

The selected feature sets are compact and pair-dependent. Subject folds choose `512-2,048` leg voxels and `1,024-2,048` arm voxels, with Jaccard overlap around `0.09-0.21`. The oracle-coarse pairwise result is `0.7743` subject-fold accuracy, only about one point above the deployable fused result. This changes the earlier hierarchy conclusion: coarse routing alone was not the answer, but pair-specific representation plus strongly weighted coarse/fine fusion is useful. The next uncertainty is stability across repeated subject partitions and anatomical consistency of the selected maps.

Five repeated shuffled subject partitions address the first uncertainty. Across 30 outer folds, with feature counts and coarse weights reselected only inside each outer training set, the pair-specific fused model averages `0.7621` balanced accuracy versus `0.7451` for the all-voxel balanced baseline. Every seed improves on average (`+0.0107` to `+0.0255`). At fold level, 22 improve, one ties, and seven regress; the worst fold delta is `-0.0688` and the best is `+0.0625`. The gain is therefore not an accident of the original deterministic partition, but the regressions motivate per-subject failure analysis and map-stability checks rather than assuming universal benefit.

The selected grids are also reproducible across training cohorts. Leg maps have mean pairwise Jaccard `0.361` and 254 voxels selected in at least 80% of outer folds. Arm maps have mean Jaccard `0.444` and 777 voxels selected in at least 80% of folds. This supports a stable pair-specific spatial signal, but the coordinates live in the resized `24³` array. They cannot be assigned to anatomical regions until the preprocessing transform is reconstructed with affine-aware spatial metadata.

Per-subject deltas explain the fold heterogeneity. `sub-12` improves by `0.150` on average (`0.775` to `0.925`), `sub-47` by `0.129`, `sub-33` by `0.117`, and `sub-39` by `0.113`. `sub-63`, previously a weak transfer subject, improves by `0.088` to `0.500`. Conversely, `sub-68` consistently falls by `0.175` (`0.600` to `0.425`), with smaller consistent regressions for `sub-25`, `sub-49`, `sub-26`, and `sub-37`. The next target is not another global feature-count sweep; it is an outer-training-validated, unlabeled signal that detects when cohort-selected pair maps mismatch a target subject.

Leave-one-subject profiles across all six fixed temporal candidates reveal a separate timing axis. Global offset `3`, length `8` improves 33 subjects, ties 12, and harms 17 relative to canonical `2:6`. If held-out labels are used to choose each subject's best window, 40/62 subjects have a better alternative and the mean oracle gain is `0.048`. The distribution is broad: `6:2` is oracle-best for 20 subjects, `3:8` for 13, `4:2` for 12, and `5:4` for nine. These counts include ties resolved by the fixed ordering and are diagnostic, not a personalization result.

Nested cohort-level selection shows that the global `3:8` choice is nevertheless robust. Across the 30 repeated outer subject folds, inner training subjects choose `3:8` in 25 folds, `3:6` in four, and `5:4` in one. The nested selected estimate is `0.7426`, compared with `0.7451` for fixed `3:8`; the five alternative inner choices all perform worse than `3:8` on their outer folds. The optimism from selecting `3:8` on the full cohort is therefore small (`0.0025`), and `3:8` should remain the fixed cohort baseline until a validated unlabeled subject-specific timing signal exists.

Timing does not explain most pair-map regressions. `sub-68` is timing-sensitive and prefers `6:2` (`0.708` balanced accuracy versus `0.625` at `3:8`). In contrast, `sub-26`, `sub-37`, and `sub-49` are best at `3:8`, while `sub-25` is effectively tied across the main windows. Those subjects remain evidence for spatial/template mismatch rather than a shared HRF-window error.

A cross-fitted unlabeled subject gate tested whether score margins, entropy, flat/pair disagreement, balance penalties, independent class-count imbalance, and pseudo-template consistency can detect pair-map mismatch. It cannot yet. A ridge gate with its threshold selected inside each outer training cohort reaches `0.7614` repeated subject-fold balanced accuracy, slightly below always-pair at `0.7621`; a fixed-zero gate reaches `0.7604`. The oracle subject choice is `0.7805`, but the learned gate chooses pair for `98.4%` of held-out observations and matches the oracle choice only `51.6%` of the time.

Retrospective signal analysis explains the failure. Subject-mean predicted and actual pair gain are negatively correlated (`r=-0.211`). Pair-minus-flat score margin is the strongest single diagnostic but remains weak (`r=0.241`, rank correlation `0.344`). `sub-68` demonstrates the central problem: its pair model has larger margins, lower entropy, and comparable pseudo-template consistency while losing `0.083-0.250` accuracy across repeats. Wrong cohort templates can therefore be confidently wrong.

A nested score-level hedge does not materially solve the problem either. Flat/pair blending reaches `0.7624`, only `0.0003` above always-pair, and 21/30 outer folds select pure pair from their inner subjects. Oracle outer-fold blend-weight selection reaches `0.7708`, still below the `0.7805` oracle per-subject choice. Simple subject gating and score mixing should not be revisited without a new subject-specific representation or alignment signal.

The validated `3:8` representation does improve labeled personalization. Fixed source/subject centroid blending at `alpha=0.25` reaches `0.7595`, `0.7887`, `0.8038`, `0.8149`, and `0.8175` balanced accuracy with one through five labeled target runs. Choosing alpha only by leave-one-calibration-run validation retains `0.7837` with two runs, `0.7968` with three, `0.8050` with four, and `0.8112` with five. These exceed the canonical `2:6` five-run results (`0.7984` fixed and `0.7957` validated), confirming that later/longer temporal extraction and subject calibration are complementary.

The calibrated weak-subject outcomes sharpen the taxonomy again. Five-run validation-selected calibration reaches `0.875` for `sub-17`, `0.792` for `sub-20`, `0.833` for `sub-26`, `0.792` for `sub-27`, `0.667` for `sub-54`, and `0.708` for `sub-63`. It still leaves `sub-42` at `0.188` and `sub-52` at `0.25`, with coarse leg-vs-arm accuracy only `0.458` and `0.333`. Those two subjects are not ordinary domain-shift cases; they require source-level class correspondence, registration, denoising, or acquisition/data-integrity investigation.

## What To Do Next

Run these checks in this order:

1. Weak-subject and run-consistency audit.

Focus on subjects that repeatedly fail despite target-run centering, especially `sub-52`, `sub-42`, `sub-17`, `sub-20`, `sub-54`, and `sub-63`. The saved-feature geometry now separates the weak cases into at least two groups. `sub-52` and `sub-42` have unstable class templates across their own runs and should get raw QC, motion/artifact, anatomical alignment, denoising, and run/class corruption checks. `sub-17` and `sub-20` look more internally stable under offset `2`, so they should be used to test subject personalization, calibration, and domain-adaptation ideas. Compare both groups against stable subjects such as `sub-30`, `sub-62`, `sub-10`, and `sub-47`.

Use the labeled calibration curve as a positive-control personalization benchmark. It shows what is reachable when target-subject labels are available, and it gives a target for future unlabeled or semi-supervised adaptation. The validation-selected blend is the safer calibrated baseline because it chooses blend strength without peeking at the held-out run labels. The first unlabeled pseudo-label adaptation is a negative control: it should be reported as evidence that naive self-training is not enough.

Use the detrended calibration curve as the positive-control personalization ceiling. The new zero-label pair-specific baseline reaches `0.7643` on offset `3`, length `8`, while the earlier validation-selected five-run labeled calibration reaches `0.7957` on the canonical window. Re-run calibration on the new window before making a direct ceiling comparison.

Do not treat broad run exclusion as a primary modeling strategy. The first post-detrend QC-policy sweep shows that removing low-scoring source runs slightly hurts or ties keep-all, while oracle validation exclusion produces only a small coverage-losing gain. Use these QC scores to prioritize run repair, subject-specific investigation, or sensitivity analyses, not to replace adaptation/modeling.

2. Run/subject transfer diagnostic.

Use simple feature baselines to quantify how much class structure survives each split type: within-run, held-out-run, held-out-subject, and held-out-session if available. Treat within-run success with cross-run failure as a nuisance/domain-shift warning, not as deployable classification. Use transductive run-normalization only as a diagnostic; any publishable model needs a non-transductive training-only normalization or a clearly defined test-time adaptation protocol.

Use the train-only alignment summaries to decide whether run/subject centering can be a standard supervised preprocessing step. If train-only centering remains near chance while transductive centering stays high, report the finding as a domain-shift/test-time-adaptation result and move to domain-invariant modeling.

Use the event-level model sweep as a guardrail before adding classifier complexity. If a richer classifier underperforms cosine centroids on the aligned features, prioritize representation learning, better target-run adaptation, or QC/preprocessing fixes instead.

Use the balanced-assignment probe to define a stronger adaptation baseline. It should be reported separately from independent per-event prediction because it uses unlabeled target-run grouping and the known balanced task design.

Avoid simple pseudo-centroid self-training for now. Both the earlier target-run pseudo-centroid sweep and the new unlabeled subject-adaptation sweep are negative. Revisit pseudo-label adaptation only if a future representation, confidence gate, or coarse-to-fine procedure produces cleaner target-subject labels than the current source model.

Analyze balanced-assignment deltas per subject whenever reporting the aggregate gain. The current result is an average improvement with heterogeneous subject-level effects, not a universal correction.

Do not rely on simple score-penalty gating to choose whether balanced assignment should be applied. A useful gate likely needs a better instability/QC signal, not just the assignment objective value.

Do not replace per-run balancing with all-runs subject-level balancing. It can reduce specific per-run overcorrections but lowers the cohort average, meaning the task balance signal is most useful at the run level where it is defined.

Use independent-prediction class-count imbalance as the next adaptation/QC lead. The `>= 4` threshold is promising but was selected after inspecting the diagnostic sweep, so a publishable variant needs separate threshold validation, nested CV, or an explicitly pre-registered threshold rule.

Add hierarchical evaluation and modeling. Report leg-vs-arm performance separately from exact four-class performance, because coarse motor grouping is already strong. Test two-stage classifiers: first leg-vs-arm, then left-vs-right leg or forearm-vs-upper-arm within the predicted group. If the second stage fails, the bottleneck is fine-grained representation/alignment rather than gross motor localization.

The first two-stage centroid test already failed to beat flat centroids, so the next hierarchy should not just wrap the same centroids. More plausible variants are multi-task training with a coarse auxiliary loss, calibrated coarse/fine score fusion, or pair-specific alignment that is validated without oracle group labels.

The detrended hierarchy result strengthens that warning: even oracle coarse routing barely improved the all-voxel representation. Nested pair-specific voxel selection now provides the first positive hierarchical follow-up, reaching `0.7643` subject-fold balanced accuracy with deployable coarse/fine fusion. Future hierarchy work should build on pair-specific representations and repeated-CV stability rather than returning to routing-only wrappers.

Investigate temporal/run-position effects. Event ordinal `0` and ordinal `6` are disproportionately weak. Test alternative event windows, longer temporal context, excluding or separately modeling first events, and run-start baseline stabilization. This should be done before another large neural run, because a window/context issue would affect any model family.

Use linear event-time detrending as the new temporal adaptation baseline. Compare every later/longer window against both centered-only and centered-plus-linear-detrended results. Keep shuffled-time and quadratic detrending as negative controls.

Use offset `3`, length `8` as the primary temporal window. It beat canonical offset `2`, length `6` across the full cohort under both independent and balanced prediction. Preserve offset `2`, length `6` as the pre-specified reference and negative-control comparison.

Start that follow-up with later shifted windows, not more mixtures of the currently extracted offsets. The coarse weight sweep suggests offset `2` itself is the strongest of the available slices, so the next question is whether even later or longer HRF-aligned windows from the continuous BOLD runs are better.

After offset `2`, keep two parallel tracks: test later/longer windows for additional temporal gain, and continue weak-subject QC because the worst subjects remain poor even with the better temporal slice.

3. Tiny overfit sanity check.

Train on one subject-run or a tiny balanced subset. A small model should reach very high training accuracy quickly. If it cannot overfit 64-256 labeled samples, the pipeline/model/loss is broken.

4. Corrected-logits sanity check.

Disable output softmax for cross-entropy and disable output-level DropConnect, or move regularization before the classifier. Repeat a short pooled run with z-score normalization. This tests whether the chance-level legacy result was caused by the model/loss bug.

5. Event-window audit.

Compare extracted filenames against original BIDS event files: class label, onset, TR, HRF shift, and expected volume window. The full target-class onset/TR check now passes with zero anomalies, but this script should be kept in the repository so future regenerated extractions can be audited automatically.

6. Block-level pooled split.

If a pooled baseline is still needed, split by subject-run-class block or by run, not by individual volume. Random volume splits leak adjacent volumes from the same block.

7. Temporal clip baseline.

Use clips with z-score normalization rather than isolated raw volumes. For the current pre-extracted class-folder dataset, use `hrf_shift=0`; true HRF-shifted clips require rebuilding class folders from the raw continuous 4D runs.

## Current Interpretation

We did not prove the full dataset is useless. We proved that the current extracted-volume plus legacy-wrapper setup does not produce defensible full-dataset learning. Corrected late-window features contain substantial motor-class signal, but run-specific offsets, linear temporal drift, and irrelevant shared voxels mask it. Full-cohort offset-3/length-8 extraction plus unlabeled run adaptation and nested pair-specific voxel selection now reaches `0.7643` subject-fold balanced accuracy and `0.8028` held-out-run balanced accuracy. The next work should test repeated subject partitions, map stability/anatomical plausibility, and the labeled-calibration curve on this stronger representation while continuing targeted investigation of residual catastrophic subjects.
