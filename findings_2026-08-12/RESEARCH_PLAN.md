# Research Plan

Compiled 2026-08-17. Supersedes the ordering sections of `docs/RESEARCH_COVERAGE_MAP.md`,
which remains the per-method ledger; this document is the decision layer above it.

## How to use this

Three questions, asked in order, before any new experiment:

1. **Is it in "Settled" or "Closed"?** If so it has been done. Closed items carry the
   reason, and a proposal to revisit needs to say what has changed.
2. **Does the commit history already contain it?** The working tree is not sufficient
   evidence. Use `git log -S<term> --all` and read commit messages — several efforts
   in this repository left no distinctive identifier in code. This check has already
   caught two near-repeats.
3. **What would each possible outcome mean?** An experiment whose null and positive
   results lead to the same next action is not worth running.

---

# Part 1 — What is settled

## 1.1 The headline, and what surrounds it

| Decoder | Independent | Balanced |
| --- | ---: | ---: |
| **Frozen hierarchy** | **0.8314** | **0.8806** |
| Score ensemble, inner-selected | 0.8157 | 0.8656 |
| Hierarchical-fused SVM | 0.8143 | 0.8564 |
| Nested ANOVA + linear SVM | 0.8111 | 0.8606 |
| Linear SVM | 0.8051 | 0.8441 |
| Tuned CNN, 32 channels | 0.8030 | 0.8580 |
| Logistic L2 | 0.8028 | 0.8456 |
| Beta-series LSS + SVM | 0.7948 | 0.8431 |
| Correlation centroid | 0.6963 | 0.7434 |

Paired advantage of the hierarchy over the linear SVM: `+0.0262`, CI95
`[+0.0107, +0.0426]`, winning on 40 of 62 subjects and losing on 21. Reliable, modest,
and narrowing to about `+0.016` once the baseline gets standard enhancements.

## 1.2 Preprocessing dominates the decoder

| Condition | Independent |
| --- | ---: |
| No centering | 0.2860 |
| Subject-level centering, transductive | 0.6294 |
| Subject-run centering | 0.7712 |
| Plus per-lag detrending | 0.8051 |

Centering is worth roughly `+0.52` — twenty times the hierarchy's advantage — and
detrending a further `+0.034`. Prior work established the *inductive* version is worth
almost nothing, so the benefit is intrinsically test-time adaptation. **Four of eight
target-run events buy 97% of it**, so deployment before a run completes is viable.

## 1.3 The signal is distributed, shown three independent ways

- Top 40 principal components, over a quarter of the variance, recover only `0.616` of `0.805`.
- ANOVA selection helps only marginally, and only with many features retained.
- **Searchlight**: best 33-voxel sphere `0.6549`; median neighbourhood `0.2994`, barely
  above chance; only 1.4% of neighbourhoods exceed `0.50`.

No focal locus carries the decision. Any method resting on aggressive compression or a
single region will underperform here.

## 1.4 The methodology is sound where it was most open to attack

- **Permutation null**: 200 within-run label shuffles across 30 folds put every null at
  chance (`0.2497`–`0.2504`), all `p < 0.005`, z of `53`–`79`. The transductive
  preprocessing manufactures no class structure.
- **QC-60 is entirely a selection effect**: `+0.024` observed against `-0.0016`
  predicted from sample size. Never present it as a method gain.
- **The temporal window search bought nothing**: nested selection `-0.0003` against the
  frozen choice. A disclosed weakness measured and found harmless on that axis.

## 1.5 What the hierarchy's advantage actually is

Decomposing it against flat-SVM analogues:

| Variant | Independent |
| --- | ---: |
| Flat SVM | 0.8098 |
| Hierarchical, hard routing | 0.8140 |
| Hierarchical, score-fused | 0.8143 |
| Per-pair selection added | 0.7973 |

Two-stage **structure** recovers about a fifth of the gap; ANOVA feature selection
explains roughly another quarter; per-pair selection actively hurts. **The majority
must come from the covariance-aware scoring itself.** The method is therefore not
replaceable by an ordinary two-stage classifier — a specific claim the manuscript can
make and defend.

## 1.6 Scale, design, and the neural lane

- **Sample size is saturated**: `+0.0008` per additional subject. Runs are worth about
  `1.3x` subjects per unit of data, and far more per unit of cost.
- **All 372 runs are palindromic.** No repetition suppression (`+1.85%`, CI excludes 1
  but trivially small). Within-run same-class repeats are **2.5x** more similar than
  across runs — largely run-state, not class identity.
- **The neural conclusion reversed.** `0.2500` was a training-path artifact from
  BatchNorm statistics, uncentered inputs, and global average pooling discarding the
  spatial position somatotopic decoding depends on. Tuned, the same architecture
  reaches `0.8030` / `0.8580` — level with conventional MVPA.

---

# Part 2 — Closed. Do not retry without new grounds

| Item | Result | Why it is closed |
| --- | --- | --- |
| Surface / anatomical normalization | Transfers `-0.059` worse despite being *more* reliable and better within subject | Ribbon-masked control refutes the coverage explanation; correspondence itself is the issue |
| Classic SRM, Procrustes hyperalignment | Inapplicable | 7–8 distinct class orders per run means no shared timeline |
| Second-order alignment (CORAL, diagonal) | `+0.002` to `-0.036` against a matched control | Nuisance structure here is overwhelmingly first-order |
| Low-rank compression | Top 40 PCs give `0.616` of `0.805` | Signal lives in many low-variance directions |
| Beta-series GLM (LSS) | `-0.0103` for the linear SVM, CI excludes zero | 16 s block design; the window mean is near-optimal and LSS adds per-trial estimation noise |
| Variance-based voxel masking | Hurts | High variance in this grid is edge and motion artefact |
| Within-run PC removal | `-0.042` per component removed | At 8 events over 4 classes, the leading within-run directions *are* the class structure |
| Feature concatenation of representations | `+0.000006`, CI `[-0.008, +0.008]` | Score-level ensembling does extract something concatenation cannot — do not read this as redundancy |
| Naive subject gating / score mixing | Prior diagnostics | Explicitly closed pending "a new subject-specific representation or alignment signal"; nothing found since supplies one |
| Within-subject noise ceiling | Invalid construction | Cross-subject decoders exceed it; it measured a data limit, not a noise limit |

---

# Part 3 — Open, in priority order

Each entry states the question, the method, the cost, and — the test of whether it is
worth running — what a **positive and a null result would each change**.

### P1. Temporal generalization matrix

**Question.** Is the motor code stationary across the haemodynamic response, or does it
evolve? Every analysis so far collapses the 8 lags by averaging.

**Method.** Train at lag *i*, test at lag *j*, for all 64 pairs; the standard
temporal-generalization design (King & Dehaene). Broad off-diagonal generalization
indicates a stable code; a narrow diagonal band indicates a dynamic one.

**Cost.** Hours at most, existing data.

**Outcomes.** *Stable* justifies the window-averaging the pipeline already does and
explains why the temporal-basis work was negative. *Dynamic* means averaging discards
structure, and a lag-resolved decoder becomes worth building. Either way it is an
interpretation result, which is what the manuscript is thinnest on.

**Note.** This is not a repeat of the temporal-basis investigation, which tested whether
*weighting* lags helps. It never asked whether the code is stationary.

### P2. What predicts a subject's decodability?

**Question.** `sub-52` and `sub-42` sit near chance while others reach `0.97`. The
project has case-by-case forensics but no systematic model.

**Method.** Regress per-subject accuracy on per-subject split-half pattern reliability
(already computed), plus available quality measures. Test whether the weak subjects are
outliers in reliability or in something else.

**Cost.** Minutes, existing data.

**Outcomes.** If reliability predicts accuracy, **QC-60 becomes principled rather than
prespecified-arbitrary**, which materially strengthens a stratum the manuscript
currently has to defend on procedural grounds alone. If nothing predicts it, that is
also worth stating — it makes exclusion harder to justify and should be said plainly.

### P3. Permutation null for the frozen decoder itself

**Question.** The comparators were permuted; the hierarchy was not.

**Method.** Its nested candidate-selection pipeline inside each permutation. Expensive
because the upstream candidate JSONs must be regenerated per permutation.

**Cost.** High — the reason it has been deferred.

**Outcomes.** Expected to match the comparators, which share its representation, folds,
and preprocessing. Worth doing because it is the one remaining gap in the significance
argument, and "we permuted the baselines but not our own method" is an easy criticism.

### P4. Representational geometry and confusion structure

**Question.** Which classes are confused, and is the geometry consistent across
subjects and decoders?

**Method.** RSA-style dissimilarity matrices (Kriegeskorte) plus confusion matrices
pooled across all decoders built this round.

**Cost.** Low, existing data.

**Outcomes.** The prior diagnostics found coarse leg-versus-arm routing is strong and
the within-pair stage is the bottleneck. Confirming that across every decoder built
here would turn a single-method observation into a property of the data.

### P5. Cross-contrast transfer

**Question.** The two pairs are different *kinds* of contrast — left-versus-right leg is
laterality, forearm-versus-upper-arm is proximal-distal. Do they share geometry?

**Method.** Train on one pair, test on the other.

**Cost.** Low.

**Outcomes.** Failure is expected and still informative: it explains *why* the fine
stage is the bottleneck, since the two pairs cannot borrow strength from each other.
Success would be surprising and would suggest a shared limb-representation axis.

### P6. Spatial smoothing sweep

**Question.** `smooth_3` is a fixed transform; the kernel was never swept.

**Cost.** Low.

**Outcomes.** Given the distributed-signal finding, smoothing may help by averaging
correlated neighbours or hurt by blurring fine structure. Either result adds to the
distributed-versus-focal picture. Low value alone; worth doing for completeness.

### P7. Ribbon-masked surface comparison at matched coverage

**Question.** The volumetric ribbon control is done; the reciprocal surface restriction
is not.

**Cost.** Low, data already extracted.

**Outcomes.** Tightens the surface conclusion, which is already firm. Completeness only.

---

# Part 4 — Blocked or out of scope

| Item | Blocker |
| --- | --- |
| HCP leakage replication | Data-use agreement. Also reframed by the learning curve as a *generalisation* claim, not an accuracy one — the manuscript must not conflate them |
| A valid noise ceiling | May be genuinely ill-defined here: more subjects keep helping, so there is no clean asymptote to estimate. Report the marginal-gain curve instead of inventing a ceiling |
| Anatomically labelled results | Requires normalization the project does not have, and the surface work showed the available route does not improve decoding. Searchlight maps support extent and gross location only |
| Further cohort collection | Saturated at `+0.0008` per subject. Buy runs, not subjects |

---

# Part 5 — Standing methodological rules

Learned the hard way this round; each one cost something.

1. **Nest every selection.** Feature-selection threshold: `+0.0143` oracle became
   `+0.0060` nested. Temporal window: `+0.0054` oracle became `-0.0003`. Any tuned
   quantity reported without nesting is inflated by an unknown amount.
2. **Check the history, not just the working tree.** Two near-repeats were caught this
   way. `git log -S` plus reading commit messages.
3. **A control before a conclusion.** The CORAL result looked catastrophic until a
   same-subspace no-whitening control showed the loss was dimensionality. The radial
   partition looked meaningful until it turned out to separate brain from non-brain.
4. **Verify the pipeline before believing a negative.** "Surface alignment does not
   help" was only publishable after confirming MSMSulc was actually applied
   (`0.9887` versus `0.9786` for the unregistered sphere).
5. **Distinguish "did not learn" from "did not generalise."** Reporting validation
   accuracy without training accuracy is what allowed a total training failure to be
   published as a generalisation result for two years.
6. **Long-running outputs go to a persistent path**, never the session scratchpad, and
   every result is distilled into the repository as soon as it lands.
7. **Validate checkpoint completeness, not just existence.** An interrupted extraction
   leaves structurally valid files with too few events; skipping them silently corrupts
   the cohort.
