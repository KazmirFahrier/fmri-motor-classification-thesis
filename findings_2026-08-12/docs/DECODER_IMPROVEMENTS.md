# Attempts to Improve the Decoder

Added: 2026-08-13.

Four attempts to raise accuracy over the `0.8051` linear-SVM baseline, all under the
frozen 30-fold protocol. Two produce small reliable gains, one produces a useful
decomposition, and one is reported mainly because its honest version is so much
smaller than its optimistic version.

---

## 1. ANOVA feature selection, and what nesting costs

Univariate ANOVA selection on training rows is canonical MVPA and had never been
tried here. Sweeping the retention threshold with the threshold **fixed across folds**:

| Voxels dropped | Independent | Balanced |
| --- | ---: | ---: |
| none (13824 kept) | 0.8051 | 0.8441 |
| 50% | 0.8079 | 0.8494 |
| 80% | 0.8134 | 0.8551 |
| **90%** | **0.8194** | **0.8702** |
| 95% | 0.8128 | 0.8688 |
| 98% | 0.7797 | 0.8368 |

There is a clear optimum around keeping the top `10%` of voxels, worth `+0.014`.

**But that number is not honest**, and the reason is the project's own recurring
theme: the threshold was chosen by looking at results across the whole cohort, which
makes it a hyperparameter fitted on the test set — the same criticism the manuscript
already discloses for the `3:8` window and the covariance caps.

Selecting the threshold **jointly with `C` on the inner subject folds** gives the
quotable number:

| | Independent | Balanced |
| --- | ---: | ---: |
| No selection | 0.8051 | 0.8441 |
| **Nested selection** | **0.8111** | **0.8606** |
| Oracle fixed threshold | 0.8194 | 0.8702 |

Selected thresholds vary across folds — `0.8` in 13 folds, `0.9` in 11, `0.95` in 5,
`0.5` in 1 — so no single threshold is right for the cohort, which is exactly why the
fixed version flatters itself.

**Nesting removes more than half the apparent gain**: `+0.014` becomes `+0.006`. That
is a clean, quantitative demonstration of the design-search effect this project
worries about, measured rather than asserted, and it is worth reporting for that
reason alone.

### A connection worth stating

The frozen hierarchy already applies covariance feature **caps** of `1024` and `2048`
— which is a form of feature selection. Giving the plain baseline the same capability
narrows the hierarchy's advantage from `+0.026` to `+0.020`. Part of what the
hierarchy was buying is therefore feature selection, not the covariance structure
itself.

---

## 2. Score-level ensembling, which succeeds where concatenation failed

Six members: three decoders on each of the volumetric and surface representations.
Scores are z-scored per fold and per member, then combined; weights come from a coarse
simplex selected on inner folds only.

| Variant | Independent | Balanced |
| --- | ---: | ---: |
| **Inner-selected ensemble** | **0.8157** | **0.8656** |
| Best single (`vol linear_svm`) | 0.8098 | 0.8413 |
| Best single (`vol logistic_l2`) | 0.8049 | 0.8503 |
| Uniform average | 0.8005 | 0.8530 |

The selected ensemble beats the best single member by `+0.006` independent and
`+0.015` balanced. **Uniform averaging is worse than the best single member**, so the
weighting is doing real work — the weak correlation-centroid members drag an unweighted
average down, and an ensemble reported without that comparison would be overstating
its case.

### This contradicts the concatenation result, and that is the interesting part

Concatenating volumetric and surface **features** gave exactly nothing
(`+0.000006`, CI `[-0.008, +0.008]`). Yet the selected ensemble weights routinely
include surface members — the most common weighting is `2 x vol_logistic +
1 x surf_logistic`, and the second most common uses four members across both
representations.

So the surface does carry usable information; feature concatenation simply could not
extract it. Asking one regularised model to weigh 34308 correlated features at once is
a harder problem than letting each representation fit its own decoder and combining
their opinions afterwards. **A null from feature concatenation is not evidence that
two representations are redundant**, and the manuscript should not treat it as such.

---

## 3. How much target-run data the centering needs

The unlabeled subject-run centering is worth `+0.52` and is the project's main
deployment liability, because it consults the target run. Estimating each run's offset
from only its first `K` events quantifies the requirement:

| Events used | Independent | Balanced |
| --- | ---: | ---: |
| 2 | 0.6856 | 0.7758 |
| 4 | 0.7503 | 0.7944 |
| 6 | 0.7566 | 0.7936 |
| 8 (all) | 0.7712 | 0.7972 |

**Four events buy 97% of what eight buy.** A deployment that must predict before a run
completes is therefore viable at moderate cost, which is a materially better story
than "the method needs the whole run".

## 4. Centering and detrending, separated

The `8 events` row above uses centering **only**, with no per-lag detrending, and
reaches `0.7712`. The frozen pipeline, which centers *and* detrends, reaches `0.8051`
on the same folds.

**Per-lag linear detrending is therefore worth `+0.034` on its own** — a component
that had been bundled with centering in every previous report and never separately
valued. It is roughly a third of the frozen hierarchy's total advantage over
conventional MVPA, obtained from a preprocessing step rather than from the decoder.

---

## 5. The temporal window choice was not worth anything, which is good news

The `3:8` extraction window is one of the choices the manuscript discloses as having
been made with all 62 subjects visible. Other *offsets* cannot be tested without
re-extraction, but the checkpoints retain all eight lags, so the choice of which lags
to average can be nested and its value measured. All 28 contiguous windows were
candidates.

| Condition | `linear_svm` independent |
| --- | ---: |
| Frozen pipeline's own choice, all 8 lags | 0.8098 |
| **Nested selection on inner folds** | **0.8095** |
| Oracle best single window (`1:8`) | 0.8152 |

Selected windows across the 30 folds: `0:8` in 12, `1:8` in 11, `1:7` in 3, `0:7` in
3, `1:6` in 1.

**Nesting removes the entire apparent gain.** The oracle window is worth `+0.0054`
over the frozen choice; selecting it honestly inside the folds is worth `-0.0003`.
Compare the ANOVA result, where nesting removed slightly more than half of `+0.0143`
but left `+0.0060` standing. Here nothing survives.

### Why this is reassuring rather than disappointing

Two things follow, and both help the manuscript.

**The frozen choice is essentially optimal.** All eight lags is what inner selection
picks most often, and the nested estimate matches the frozen pipeline to `0.0003`.
The window was not a lucky pick that flatters the reported result.

**The disclosed design search bought nothing on this axis.** The project correctly
discloses that the window was chosen with full cohort visibility, and a reader is
entitled to suspect that inflates the headline. For the temporal window it
demonstrably does not: any reasonable contiguous window performs about the same, so
the choice could not have conferred an advantage worth having.

That converts a disclosed weakness into a measured non-issue for this parameter. It
says nothing about the other searched choices — the `24³` grid, the covariance caps,
the hierarchy structure — which remain disclosed and unquantified.

## 6. Selective prediction, and what it does to the design-constrained rule

Every accuracy elsewhere in this project forces a prediction on every event. The
frozen protocol also carries a deployment rule, and a deployed decoder can decline to
answer. Sweeping a confidence threshold — the margin between the top two class scores
— gives the accuracy-coverage trade-off:

| Coverage | Independent | Balanced |
| ---: | ---: | ---: |
| 1.00 | 0.8100 | 0.8488 |
| 0.90 | 0.8327 | 0.8581 |
| 0.80 | 0.8553 | 0.8672 |
| 0.70 | 0.8670 | 0.8741 |
| 0.60 | 0.8776 | 0.8806 |
| 0.50 | 0.8828 | 0.8863 |
| 0.30 | 0.8999 | 0.8974 |
| 0.20 | **0.9096** | 0.9052 |

Declining the least confident fifth of events lifts independent accuracy from
`0.8100` to `0.8553`; at 70% coverage it reaches `0.8670`.

### The interesting part: the two rules converge, then cross

The balanced rule's advantage over independent prediction is `+0.039` at full
coverage. By 60% coverage it is `+0.003`, and at 30% and 20% coverage **independent
prediction is ahead**.

This bears directly on the project's most awkward framing problem. The
design-constrained rules — balanced assignment and repetition consistency — buy their
advantage from the guarantee of two events per class per run, which is why `0.8948`
cannot be compared with independent prediction and needs its own row and its own
deployment precondition.

Selective prediction offers an alternative that needs **no design constraint at all**.
A plain independent decoder that declines its least confident events reaches
comparable accuracy without assuming anything about run composition, and is therefore
deployable on incomplete, imbalanced, or online data — precisely the regime the frozen
protocol reserves for the weaker independent decoder.

So the honest framing gains an option. Rather than presenting a high number that
requires a complete balanced run, the manuscript can present an accuracy-coverage
curve that requires nothing about the design, and note that the design-constrained
rule only helps when every event must be answered.

### Caveat that must travel with the curve

Nothing here is fitted, and that cuts both ways. The threshold is swept post hoc on
the same held-out scores the accuracy is computed from, so this is a **descriptive**
trade-off, not a tuned policy. A deployed threshold would have to be chosen on
training data and would perform slightly worse than this curve suggests.

## 7. Beta-series GLM does not beat the window mean

The canonical MVPA feature is a per-trial GLM beta, not a raw window mean, and this
project had only ever used window means. Least-Squares-Separate estimation was run
over the whole cohort — a per-trial design with that trial alone, all other trials
combined, and cubic drift — and passed through the identical spatial transform, so the
only thing differing from the frozen checkpoints is how each trial's amplitude is
estimated.

| Model | Rule | Window mean | Beta series | Paired difference | CI95 |
| --- | --- | ---: | ---: | ---: | --- |
| `linear_svm` | independent | **0.8051** | 0.7948 | **-0.0103** | `[-0.0194, -0.0011]` |
| `logistic_l2` | independent | **0.8028** | 0.7968 | -0.0061 | `[-0.0150, +0.0026]` |
| `correlation_centroid` | independent | 0.6963 | **0.7046** | +0.0084 | `[-0.0036, +0.0197]` |

The linear SVM is **reliably worse** with betas, the logistic and the centroid show no
reliable difference. Across the three, the canonical feature is at best a wash and at
worst a small loss.

### Why this is the expected answer for this design, in hindsight

LSS exists to separate overlapping haemodynamic responses in **rapid event-related**
designs, where trials are a few seconds apart and adjacent regressors are badly
collinear. This experiment is a **block design**: each event is a 16 s block, and the
response is long, large, and well sampled by the eight-volume window the frozen
pipeline already averages.

In that regime the window mean is close to an optimal estimator, while LSS pays for
its flexibility by fitting a separate GLM per trial from only 232 timepoints, adding
estimation noise for a separation problem that is not severe. The small gain for the
correlation classifier fits the same story: that decoder is the most sensitive to
per-trial amplitude scaling, which is the one thing a GLM beta does estimate better.

### Why it is still worth reporting

A reviewer will ask whether the project used trial-wise betas, and "we used window
means" is a much weaker answer than "we ran LSS across the full cohort and it was
`-0.0103` for the best decoder". It converts an apparent methodological shortcut into
a measured choice, and the reasoning generalises: for block designs of this length,
the extra machinery of trial-wise GLM estimation is not repaid.

## 8. Where this leaves the headline

| Decoder | Independent |
| --- | ---: |
| Frozen hierarchy | **0.8314** |
| Ensemble, inner-selected | 0.8157 |
| Nested ANOVA selection + SVM | 0.8111 |
| Plain linear SVM | 0.8051 |

Both improvements are real, reliable, and small. Neither overtakes the frozen
decoder, and stacking them is unlikely to close a `0.016` gap. The honest summary is
that conventional MVPA with standard enhancements lands a little closer to the
hierarchy than the original comparison suggested, narrowing the gap from `0.026` to
about `0.016`, without eliminating it.
