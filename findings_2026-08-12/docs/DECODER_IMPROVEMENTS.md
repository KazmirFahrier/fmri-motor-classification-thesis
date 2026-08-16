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

## Where this leaves the headline

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
