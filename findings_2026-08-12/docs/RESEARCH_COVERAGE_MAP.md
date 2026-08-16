# Research Coverage Map

Compiled 2026-08-12, for a research programme whose goal is completeness rather
than a single publishable number.

Every "tried" entry names the script or document that establishes it. Every
"untried" entry was verified absent by searching `scripts/`, `src/`, and `docs/`,
discounting incidental substring matches — `np.random.permutation`, numpy masks,
and `.pyc` byte coincidences all produce false positives and were excluded by hand.

Status values: **Done** (settled, evidence in repo) · **Partial** (started, not
settled) · **Open** (verified absent).

## How "Open" was verified against history

The working tree alone is not sufficient evidence that something was never tried,
because an approach could have been implemented and later removed. The full commit
history was therefore checked:

- **129 commits**, and `git log --all --diff-filter=D` shows **no file has ever been
  deleted**. Nothing was implemented and dropped at the file level.
- `git log -S` pickaxe searches across all commits return **zero** for `hyperalign`,
  `shared_response`, `searchlight`, `permutation_test`, `beta_series`,
  `representational_similarity`, `learning_curve`, `RidgeClassifier`,
  `RandomForest`, and `nilearn`.
- `sklearn` appears in exactly one commit — the initial import of `notebook_code.py`
  — and only for **metrics** (`accuracy_score`, `confusion_matrix`, `roc_curve`,
  `train_test_split`). `git grep -E "from sklearn\.(svm|linear_model|ensemble|discriminant)"`
  over history returns nothing: **no sklearn classifier was ever used** before
  2026-08-12.
- All 129 commit messages were read to catch work that left no distinctive
  identifier. This surfaced two overlaps, corrected below.

### Corrections the history check forced

**1. "Domain alignment" was done, but it is not hyperalignment.** Commits 45–58
implement an arc of clip domain-alignment work. Reading
`docs/DATASET_DIAGNOSTIC_FINDINGS.md` shows these estimate *centering and
standardisation statistics* at global, subject, run, and subject-run level — first
transductively, then train-only. They do not learn a functional common response
space, involve no Procrustes or SVD alignment of response patterns, and are not
hyperalignment or SRM. Those remain genuinely Open.

**2. The train-only result already exists, and it sharpens today's ablation.**
That work found train-only subject centering moved same-subject held-out-run
accuracy only `0.2641` → `0.3044`, with subject holdout near chance (`0.2646`).
Today's `subject_center` condition is the *transductive* counterpart and reaches
`0.6294`. The two are complementary: subject centering is worth a great deal, but
**only when estimated on the target subject**. See the corresponding section in
`STANDARD_MVPA_BASELINE.md`.

**3. GroupNorm was already added and already found insufficient.** Commit `8cae867`
added `norm: group`, and the diagnostics record that "the GroupNorm follow-up did
not fix the corrected temporal model", leaving an explicit gate: an eval-mode
overfit check must pass "before any broader baseline is trusted". That gate was
never completed and the `0.2500` figure was published anyway. Today's neural work
completes that specified gate rather than rediscovering the BatchNorm diagnosis.

**4. Oracle headroom analysis is already done.** An oracle supplying the true
leg-versus-arm group reaches `0.6767`/`0.7238` subject-fold exact accuracy, and
score-level flat/pair hedging was shown to be a dead end. The diagnostics state
plainly that "simple subject gating and score mixing should not be revisited
without a new subject-specific representation or alignment signal." Any new
proposal in that family should be treated as closed unless it brings exactly that.

---

## A. Representation and preprocessing

| Question | Status | Evidence / note |
| --- | --- | --- |
| Event window offset and length | Done | `sweep_continuous_bold_windows`, `run_nested_subject_window_selection`; `3:8` frozen |
| Spatial feature grid size | Done | `run_spatial_scale_feature_sweep`; `24³` chosen, `32³` tested |
| Temporal basis / within-response shape | Done | `run_continuous_temporal_basis_full_cohort`; negative |
| Learned temporal filters | Done | `run_learned_temporal_filter_hierarchy`; marginal |
| Multi-window arm representation | Done | `run_arm_multiwindow_representation` |
| Subject-run centering | Done | Now quantified: worth `+0.52` independent accuracy |
| Per-lag linear detrending | Done | Bundled with run centering; **not separately ablated** |
| Motion residualization | Done | `test_motion_residualized_classification` |
| Run-level QC filtering | Done | `run_detrended_run_qc_policy`; negative |
| **Beta-series GLM features (LSS / LSA)** | **Open** | The canonical MVPA feature is a per-trial GLM beta, not a raw window mean. Verified absent. Needs continuous BOLD, so ~138 GB of re-download. |
| Per-lag linear detrending, valued separately | **Done** | Worth `+0.034` alone — about a third of the hierarchy's total advantage over conventional MVPA, from preprocessing rather than the decoder |
| Univariate ANOVA feature selection | **Done** | `+0.006` with the threshold nested, `+0.014` with it fixed. The gap is a measured demonstration of the design-search effect. |
| Variance-based voxel masking | **Done — negative** | Hurts. High variance in this grid is edge and motion artefact, not signal. |
| **Spatial smoothing sweep** | **Open** | `smooth_3` is a fixed pair transform; smoothing kernel was never swept as a preprocessing choice |
| **Anatomical ROI restriction** | **Open** | No motor-cortex or any anatomical mask; all decoding is whole-grid |
| **High-pass / temporal filtering choices** | **Open** | Inherited from fmriprep defaults, never varied |

**Highest value here:** beta-series GLM. A reviewer will expect it, it is the
standard trial-wise estimate, and it directly tests whether the raw-window-mean
representation is costing accuracy.

---

## B. Cross-subject alignment

| Question | Status | Evidence / note |
| --- | --- | --- |
| Native bounding-box rescale | Done | `reproduce_thesis_transform`; this is what the frozen result uses |
| Subject-level centering, transductive | Done | Recovers `0.6294`; see `STANDARD_MVPA_BASELINE.md` |
| Subject-level centering, train-only | Done | Prior work: `0.2641` → `0.3044`, subject holdout near chance. Settled negative. |
| Run-level and global alignment statistics | Done | Commits 45–58; transductive and train-only variants both characterised |
| Oracle coarse-group headroom | Done | `0.6767` / `0.7238`; score-level hedging shown to be a dead end |
| Unlabeled subject adaptation | Done | `run_unlabeled_subject_adaptation`, `run_unlabeled_subject_model_gate` |
| Labeled personalization | Done | `run_hierarchy_subject_calibration`; reported separately |
| Volumetric MNI normalization | Partial | Scoped in `INTER_SUBJECT_NORMALIZATION.md`; **no MNI BOLD or transforms shipped** |
| Surface-to-volume frame offset (`c_ras`) | **Ruled out** | Suspected, investigated, **withdrawn**. Contours trace cortical folding exactly on the T1w in 3 subjects; the earlier estimate came from an objective any FOV-centring translation maximises. The ~11% missing vertices are EPI field-of-view clipping. |
| Surface projection to shared sphere | Partial | `project_bold_to_surface.py` written and verified on `sub-01`; **now unblocked**, needs out-of-FOV vertex masking and a full-cohort run |
| Classic SRM on a shared timeline | **Ruled out** | **Not applicable to this design** — see below |
| Connectivity-based hyperalignment | Partial | Implemented in `run_connectivity_hyperalignment.py`: per-parcel orthogonal Procrustes on vertex-to-parcel connectivity profiles, template from training subjects only. Needs no extra download. Awaiting the surface extraction. |
| Diagonal second-order alignment (per-subject feature z-score) | **Done** | Small consistent gain for discriminative linear models (`+0.009` SVM, `+0.012` logistic independent); **destroys** the correlation centroid (`-0.14`/`-0.17`). Applied *on top of* the frozen pipeline. |
| CORAL, full covariance in a shared subspace | **Done — negative** | Against a same-subspace no-whitening control, the isolated whitening effect is `+0.002` to `-0.036`. The apparent `-0.19` was dimensionality, not alignment. Second-order alignment is closed. |
| Low-rank compression of the feature space | **Done — negative** | Top-24 PCA (20% of variance) costs `~0.19` independent accuracy. The discriminative signal is distributed across many low-variance directions, not concentrated in leading components. |

### Why classic SRM does not apply, and what does

The Shared Response Model and Procrustes hyperalignment both require **temporal
correspondence**: subject *i* and subject *j* must be experiencing the same thing
at the same index, which is why they are normally used on movie or audiobook data.

That condition fails here. Checking the event order directly:

```
run 1: 7 distinct class orders across subjects
run 2: 7    run 3: 8    run 4: 7    run 5: 8    run 6: 8
```

Subjects receive different class orderings, so event *k* of a run means different
things for different subjects and the timelines cannot be stacked. Anyone
attempting SRM on this cohort will hit this immediately — hence recording it here.

Two observations worth keeping:

- Only 7–8 distinct orders exist across 62 subjects, so subjects fall into a small
  number of order groups that *do* share correspondence internally. Within-group
  SRM is technically possible but fragments the cohort into ~8 pieces of ~8
  subjects, which will not support the 30-fold subject protocol.
- Every run is **palindromic**: four classes followed by their mirror, e.g.
  `sub-01` run 1 is `[1,3,0,2,2,0,3,1]`. That is what guarantees exactly two events
  per class and is the structure the repetition-consistency decoder exploits.

The applicable alternatives are the ones that need **no correspondence and no
labels**:

1. **CORAL / covariance alignment.** The project has established that aligning
   *first-order* statistics (the subject-run mean) is worth `+0.52`. Aligning
   *second-order* statistics is the direct next question and is a canonical
   unsupervised domain-adaptation method. Cheap and immediately applicable.
2. **Connectivity-based hyperalignment.** Aligns subjects on voxel-to-voxel
   connectivity profiles rather than on a stimulus timeline, so it needs neither
   correspondence nor labels. This is the genuine hyperalignment option for this
   dataset, and it is the heavier of the two.

#### Connectivity hyperalignment: data requirement, resolved

The obvious objection is that connectivity profiles want continuous time series,
and the surface extraction saves only the 64 task-locked volumes per run that the
event windows need. Re-extracting all 232 volumes per run would triple the
projection cost and require a second pass of roughly 138 GB.

That is not necessary. The saved `(48, 8, V)` sequences reshape to `(384, V)` per
subject — 384 task-locked samples — which is enough for a **coarse connectivity
profile** computed entirely post hoc:

1. Partition the shared sphere into roughly 100–200 parcels and average within each,
   giving a `(384, P)` target matrix per subject.
2. Correlate every vertex with every parcel to get a `(V, P)` connectivity profile.
   The parcels are shared across subjects by construction, so these profiles live in
   a common space **without** any stimulus or label correspondence.
3. Procrustes-align each subject's profile to a template built from training subjects
   only, and apply the resulting orthogonal map to that subject's event features.

Two honest caveats to record with any result:

- These are **task-locked** samples, so the profiles are task connectivity rather
  than resting connectivity. That is a legitimate variant and arguably better suited
  to aligning task responses, but it must not be described as resting-state
  connectivity hyperalignment.
- The parcel targets are derived from the same data being aligned. The alignment uses
  no labels, so it cannot leak class information, but the template must still be
  built from training subjects only to keep the fold structure honest.

The practical consequence is that **no additional download is needed**, and the
method can be run as soon as the surface extraction completes.

**Highest value here:** hyperalignment / SRM. These are *the* canonical answers to
cross-subject decoding, and they are functional alternatives to anatomical
normalization — they learn a common response space directly from the data, which
is exactly the problem this project has been working around with centering. Their
absence is the most conspicuous gap in the whole programme, and they need no extra
data.

The history makes the case stronger rather than weaker. The earlier diagnostics
concluded, in their own words, that the project should "either define a legitimate
unlabeled target-run adaptation protocol or train an explicitly domain-invariant
representation." The first branch was taken and became the frozen decoder. **The
second branch was never taken.** Hyperalignment and SRM are precisely that second
branch, and both are label-free, so they fit the existing evaluation protocol
without weakening it.

---

## C. Decoders

| Question | Status | Evidence / note |
| --- | --- | --- |
| Covariance pair hierarchy | Done | Frozen; `0.8314` independent |
| Nested temporal candidate selection | Done | Frozen decoder |
| Repetition-consistency assignment | Done | `0.8948`, complete runs only |
| Score ensembling across pair models | Done | `run_flat_pair_score_ensemble` |
| Linear SVM / L2 logistic / correlation | Done | Added 2026-08-12; `0.8051` / `0.8028` / `0.6963` |
| Legacy 3D CNN + transformer | Done | Withdrawn — training-path artifact |
| GroupNorm encoder variant | Done | Added in commit `8cae867`; insufficient alone, confirmed again 2026-08-12 |
| Location-preserving CNN | Done (one seed) | `0.7840` independent / `0.8513` balanced over 6 folds. Ties linear MVPA on the balanced rule, `0.02`–`0.03` below on independent. Fold sd `0.050`. Needs the remaining four seeds. |
| **Regularized alternatives (ridge, RF, boosting)** | **Open** | Cheap; mainly a completeness item |
| **RSA / representational similarity** | **Open** | Verified absent. Descriptive rather than predictive, but standard in this literature |
| **Pretrained or transfer-learned encoders** | **Open** | No pretraining of any kind attempted |
| **Hyperparameter search for the working CNN** | **Open** | Current settings are the first that worked, not a searched optimum |

---

## D. Evaluation and inference

| Question | Status | Evidence / note |
| --- | --- | --- |
| Subject-wise nested CV | Done | 30 outer folds, 120 inner isolation checks |
| Split isolation assertions | Done | Enforced in code |
| Subject-level bootstrap CIs | Done | 20000 iterations |
| Paired comparisons between decoders | Done | Added 2026-08-12 |
| QC-60 sensitivity stratum | Done | Prespecified |
| Byte-identical reproduction | Done | SHA-256 verified |
| Permutation test against a label-shuffled null | **Done** | 200 permutations, 30 folds, within-run shuffling. All nulls at chance (`0.2497`–`0.2504`), all `p < 0.005`, z of `53`–`79`. Proves the transductive preprocessing does not manufacture label structure. See `PERMUTATION_NULL.md`. |
| Permutation null for the **frozen decoder itself** | **Open** | Only the three comparators were permuted; the hierarchy's own nested pipeline was not |
| Learning curve over training subjects | **Done** | Marginal value of one subject collapses twentyfold to `+0.0008` at current cohort size; correlation classifier fully saturated. **Sample size is not the binding constraint.** See `LEARNING_CURVE.md`. |
| Learning curve over runs per subject | Partial | Implemented as `--vary runs`. **Preliminary** (1 fold, 2 draws): `0.6922` → `0.7699` → `0.8030` for 1/3/6 runs, i.e. still climbing at 6 while the subject axis had saturated. Needs the full 30-fold run before it is quotable. |
| **Multiple-comparison control across the search** | **Open** | ~74 scripts of exploration, no correction or accounting |

**Highest value here:** the permutation test. It is cheap, it is expected at a Q1
venue, and given how much of the accuracy traces to a transductive centering step,
a label-shuffled null is the cleanest demonstration that the remaining signal is
real. The centering must be recomputed *inside* each permutation, or the null will
be optimistic.

---

## E. Interpretation and localization

| Question | Status | Evidence / note |
| --- | --- | --- |
| Feature-selection stability maps | Done | Explicitly not anatomical claims |
| Response topography checks | Done | `analyze_response_topography`, `audit_official_glm_topography` |
| Error anatomy | Done | `analyze_event_error_anatomy` |
| **Searchlight analysis** | **Open** | Verified absent. The standard localization method in MVPA. |
| **Anatomically-labelled results** | **Open** | Blocked until normalization is resolved (§B) |

---

## F. Generalization

| Question | Status | Evidence / note |
| --- | --- | --- |
| External four-class cohort search | Done | `EXTERNAL_CONFIRMATION.md`; none exists |
| **Leakage claim replicated on a second dataset** | **Open** | The leakage result is separately prespecifiable. HCP's hand/foot/tongue design could replicate it at coarser granularity without any post-hoc label substitution. |

---

## Status of the five candidate directions — all now run

| Direction | Outcome |
| --- | --- |
| Noise ceiling | **Ran; the estimate is invalid as a ceiling.** Cross-subject decoders exceed it. It did establish that pooling across subjects beats a subject's own data. See `CEILING_AND_DESIGN_STRUCTURE.md`. |
| Brain-masked volumetric | **Ran.** Variance-based masking *hurts*; ANOVA-based selection helps by `+0.006` nested, `+0.014` oracle. See `DECODER_IMPROVEMENTS.md`. |
| Decoder ensemble | **Ran.** `0.8157` independent, `+0.006` over best single. Contradicts the feature-concatenation null. |
| Centering data requirement | **Ran.** Four of eight events buy 97% of the benefit. Also isolated detrending at `+0.034`. |
| Within-run order effects | **Ran.** All 372 runs palindromic; no repetition suppression; within-run repeats `2.5x` more similar than across-run. |

Additionally run and decisive: the **ribbon-masked volumetric control**, which
refutes the coverage explanation for the surface deficit.

## Candidate directions generated by this round's findings

These are not from a generic methods checklist; each follows from something measured.

### 1. Noise ceiling from split-half reliability — highest value

**Follows from:** the subject-count curve saturating at `+0.0008` per subject, and the
`+0.026` hierarchy advantage looking "modest".

Every accuracy in this project is reported against a `0.25` chance floor and no
ceiling. Split-half reliability of each subject's per-class response patterns gives
the **maximum accuracy any decoder could achieve** given measurement noise. Estimate
it by correlating class patterns from odd versus even runs, then Spearman-Brown
correct.

This is potentially a reframing rather than an increment. If the ceiling is near
`0.85`, then `0.8314` is roughly `98%` of achievable and the honest description of
the frozen decoder changes from "modest gain over standard MVPA" to "close to the
limit this data supports" — and it explains the learning-curve saturation
mechanistically. If the ceiling is `0.95`, there is real headroom and the modest
framing stands. Either answer is worth more than another decimal place.

Cheap: uses existing checkpoints, no new data.

### 2. Brain-masked volumetric features

**Follows from:** the radial probe showing the outer `48%` of the grid decodes near
chance because it is skull, scalp, and air.

Half the volumetric feature space is not brain. Masking it should reduce noise, and
it makes the volumetric-versus-surface comparison substantially fairer, since the
surface is all-brain by construction. A voxel-wise reliability or variance criterion
computed on **training subjects only** would define the mask without leakage.

Cheap, and unlike the radial partition this one has a defensible criterion.

### 3. Decoder ensemble

**Follows from:** the paired comparison — the frozen decoder beats the linear SVM on
40 of 62 subjects and *loses* on 21, and the CNN sits close behind both.

Three decoders that disagree on a third of subjects may have partly decorrelated
errors. Stacking or simple score averaging, with weights selected on inner folds
only, could exceed all three. This is the cheapest remaining route to a better
headline number, and the win/loss structure is direct evidence the opportunity exists.

### 4. How many target-run events does the centering need?

**Follows from:** centering being worth `+0.52`, and being the project's main
methodological liability because it is transductive.

The centering currently uses all eight events of the target run. If four suffice, the
deployment story improves considerably; if it degrades gracefully, that is a curve
worth publishing. This directly addresses the strongest objection to the method by
quantifying exactly how much target-run data it needs.

### 5. Within-run order and habituation effects

**Follows from:** the discovery that every run is palindromic — four classes then
their mirror, which is what guarantees two events per class.

The two presentations of each class are therefore at mirrored positions in the run.
Nobody has checked whether first and second presentations differ systematically in
amplitude or pattern. If they do, that is a genuine neuroscience observation, it
bears on the repetition-consistency decoder that exploits within-run repeats, and it
may suggest weighting the two presentations differently.

Cheap, and novel rather than a methods box to tick.

## Suggested ordering, revised 2026-08-12

Four items from the original ordering are now closed, and the results reprioritise
what remains. Three independent lines of evidence now point the same way:

- Sample size is saturated (`+0.0008` per subject).
- Second-order alignment adds nothing, in either diagonal or full covariance form.
- Low-rank compression is destructive — the top 40 principal components, over a
  quarter of the variance, recover only `0.616` of the `0.805` available.

Together these say the remaining headroom is **not** in more data, not in feature
scaling, and not in the leading variance directions. It is in **spatial
correspondence** and in **functional alignment that does not compress**.

Revised order:

1. **Resolve the `c_ras` offset and complete surface normalization.** Promoted from
   fifth to first. It is the only open item that directly addresses spatial
   correspondence, and the learning curve says correspondence is where the
   remaining accuracy lives.
2. **Connectivity-based hyperalignment.** The correspondence-free functional
   alignment that classic SRM cannot provide on this design. Must be applied in the
   full feature space, not a compressed one, given the dimensionality finding.
3. **Beta-series GLM features.** The canonical representation this project skipped;
   a reviewer will expect the comparison regardless of outcome.
4. **Searchlight**, once normalization makes anatomical statements meaningful.
5. **Remaining CNN seeds**, then a hyperparameter search for the corrected
   architecture, which has never been tuned.
6. **Permutation null for the frozen decoder itself**, to close the one gap left in
   the significance testing.
7. **HCP leakage replication.** Now framed explicitly as a generalization claim
   rather than an accuracy improvement, per the learning-curve result.

Items 2, 3, 5, and 6 need no new data and no new dependencies. Item 1 needs
Connectome Workbench or a rigid-registration step. Item 7 needs an HCP data
agreement.
