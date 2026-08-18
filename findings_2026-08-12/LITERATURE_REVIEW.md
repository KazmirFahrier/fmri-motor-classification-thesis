# Literature Review

Compiled 2026-08-17. Purpose: find experiments that *should* have been tried, by
checking this project's choices against the published methods literature and against
the dataset's own documentation.

One finding materially changes the project's scope. The rest either corroborate
existing results or name specific untried methods.

---

## 1. The dataset is three times larger than this project uses

**This is the most consequential finding in the review.**

The dataset paper — [An fMRI dataset for whole-body somatotopic mapping in humans](https://pmc.ncbi.nlm.nih.gov/articles/PMC9399117),
*Scientific Data* 2022 — describes **twelve** movement conditions: toe, ankle, left
leg, right leg, finger, wrist, forearm, upper arm, jaw, lip, tongue, and eyes.

This project decodes **four** of them: left leg, right leg, forearm, upper arm.

Verified directly against the raw events files rather than taken on trust. `sub-01`
run 1 has trial types:

```
0 4 8 10 6 2 12 | 0 7 3 11 9 1 5 | 0 5 1 9 11 3 7 | 0 12 2 6 10 8 4 | 0
```

Code `0` is rest, appearing five times as the paper describes. Codes `1`–`12` are the
body parts, and **each appears exactly twice per run**. So every run contains **24
movement blocks**, of which this project extracts **8** — the two repeats each of
codes `3`, `4`, `5`, `6`.

The palindromic structure discovered empirically earlier in this round is confirmed by
the paper's own description: conditions were "counterbalanced within a run", which is
exactly the mirror ordering observed in all 372 runs.

### What this unlocks

| Opportunity | Why it matters |
| --- | --- |
| **12-class decoding** | The full problem the dataset was built for. Harder, and a far more substantial claim than 4-class. |
| **~8900 movement events instead of 2976** | Three times the data per subject. The runs-per-subject curve was still climbing at `+0.033` per 3 runs, so more events per subject is the axis that was *not* saturated. |
| **Somatotopic gradient / RSA over 12 body parts** | The classic homunculus predicts an ordering from toe through leg, trunk, arm, hand, to face. Recovering that ordering from representational geometry is a **neuroscience result the 4-class design cannot produce**, and it is the kind of interpretation the manuscript is thinnest on. |
| **Face conditions as an upper bound** | Jaw, lip, tongue and eye are anatomically distant from limb representations and should be far easier to separate. They provide a sanity check and a ceiling reference the project currently lacks. |
| **Within-limb gradients** | Toe/ankle extend the leg pair; wrist/finger extend the arm pair. This tests whether the *fine* within-limb stage — repeatedly identified as the bottleneck — improves with more graded classes or degrades further. |

### The honest framing problem this creates

The choice of these four classes is a design decision made with full cohort
visibility, exactly like the `3:8` window and the covariance caps — and unlike those,
it does not appear to be disclosed anywhere in the existing documentation. Four
classes chosen from twelve is a `495`-way choice. A reviewer who reads the dataset
paper will notice, and "we used four of the twelve available conditions" needs a stated
reason.

**Recommendation.** Either extend to 12 classes, or state plainly why these four were
selected and demonstrate the result is not specific to them. The second is cheap: run
the existing pipeline on a different four-class subset drawn from the same limb
structure and check the accuracy is comparable.

---

## 2. Functional alignment: benchmarked, and one strong method untried

[An empirical evaluation of functional alignment using inter-subject decoding](https://pmc.ncbi.nlm.nih.gov/articles/PMC11653789/)
(*NeuroImage* 2021) benchmarks five methods for exactly this project's problem —
decoding across subjects.

| Their method | Status here |
| --- | --- |
| Piecewise Procrustes | **Done** — this is what the connectivity hyperalignment implements |
| Searchlight Procrustes | Not tried; the paper reports it is `25x` slower and *less* accurate than piecewise |
| Piecewise Optimal Transport | **Untried. Reported best at whole-brain scale, with little hyperparameter tuning** |
| Shared Response Modelling | Ruled out here — SRM needs a shared timeline, and this design has 7–8 distinct class orders per run |
| Intra-subject alignment | Not tried; reported weakest, sometimes harmful |

Three things follow.

**The magnitude here is consistent with the literature.** They report functional
alignment improving inter-subject decoding by `2–5%`. Connectivity hyperalignment gave
`+3.3%` on the surface representation. That is squarely in range, which is reassuring
about the implementation.

**Piecewise Optimal Transport is a concrete, well-supported untried method.** It is
reported as the best whole-brain performer and the most robust to hyperparameter
choice — attractive given how much of this round has been spent discovering that tuned
quantities inflate when not nested.

**The aggregation choice was right.** Piecewise beat searchlight in both accuracy and
speed, and piecewise is what was implemented.

### A contrast worth reporting

That paper frames the noise ceiling as the gap between within-subject and
inter-subject accuracy, averaging `8.5%` in their data, with alignment recovering more
than half of it.

**This project inverts that relationship.** Within-subject decoding here reaches
`0.6912` against cross-subject `0.8051` — cross-subject is *better*, because training
on 51 other subjects denoises class templates far beyond what one subject's ~40 events
can support. The standard framing of alignment as "recovering accuracy lost to
inter-subject variability" does not apply to a design with this few trials per subject,
and saying so is a genuine methodological contribution rather than a caveat.

---

## 3. The leakage claim is well supported by the wider literature

The project's central argument — that published ceilings on this task are evaluation
artifacts — sits in an established and serious literature.

[Inflated prediction accuracy caused by data leakage in feature selection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8042090/)
shows feature selection performed *outside* cross-validation yields considerably higher
accuracy than the same selection performed *within* it — the same effect measured
directly this round, where ANOVA selection went from `+0.0143` oracle to `+0.0060`
nested and the temporal window from `+0.0054` to `-0.0003`.

[Risk of data leakage in deep-learning-based diagnosis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10547830/)
establishes that **subject-wise** cross-validation, not trial-wise, is required when
multiple samples come from one participant — precisely the pooled-versus-subject-wise
distinction this project demonstrates with its `0.8522` → `0.2629` collapse.

[On Leakage in Machine Learning Pipelines](https://arxiv.org/pdf/2311.04179) reports
at least 294 affected papers across 17 fields, and there is at least one high-profile
retraction of a neuroimaging prediction paper on leakage grounds.

**Implication for framing.** The manuscript can position the leakage demonstration
against a documented, cross-disciplinary problem rather than as an isolated
observation. The nesting measurements taken this round are a quantitative contribution
to that literature, not merely internal housekeeping.

---

## 4. The distributed-signal finding matches the current view of somatotopy

The searchlight result — best 33-voxel neighbourhood `0.6549` against whole-grid
`0.8051`, median neighbourhood barely above chance — was interpreted here as the class
signal being distributed rather than focal.

The somatotopy literature agrees, and has moved in the same direction.
[Beyond body maps: information content of specific body parts is distributed across the somatosensory homunculus](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8938902/)
reports that information about a body part can be decoded even from non-adjacent
representations, and argues the classical map understates how distributed the
information is. Work on
[complex organization of human primary motor cortex](https://pubmed.ncbi.nlm.nih.gov/18684903/)
finds arm and finger representations interleaved rather than cleanly ordered, and
[digit action maps](https://www.sciencedirect.com/science/article/pii/S1053811919310547)
argue motor cortex is organised by coordinated actions rather than strictly by body part.

**This is a convergence worth stating explicitly in the manuscript.** An empirical
decoding result obtained from a bounding-box rescale with no anatomical registration
independently reproduces a conclusion the high-resolution somatotopy literature reached
by other means. It also explains, mechanistically, why the covariance hierarchy beats a
focal or low-rank approach: the information genuinely is spread thin across many
low-variance directions.

---

## 5. Modern subject-invariant methods: named, and mostly out of scope

A large recent literature attacks cross-subject transfer with pretraining and
adversarial objectives — [BrainLM](https://proceedings.iclr.cc/paper_files/paper/2024/file/029ce70401321de3808b3ac39e1ab167-Paper-Conference.pdf)
(pretrained on 6,700 hours), [MindLink](https://papers.miccai.org/miccai-2025/paper/5263_paper.pdf)
(3D ViT with **domain-adversarial training** for subject-agnostic features),
[fMRI-PTE](https://arxiv.org/pdf/2311.00342), and surveys of
[brain foundation models](https://arxiv.org/pdf/2503.00580).

Most of this is out of scope: it targets large-scale pretraining this project cannot
undertake, and the learning curve already shows the cohort is saturated on subjects.

**One idea is in scope and cheap: domain-adversarial training for subject invariance.**
A gradient-reversal branch predicting subject identity, trained against the class
objective, is a small addition to the existing CNN and directly targets this project's
central obstacle. Given the corrected CNN now sits level with linear MVPA, this is the
most plausible route to pushing it past. It is also a *legitimate* use of subject
labels, which are not class labels and are available for training subjects.

---

## 6. Revised priorities

The review reorders the plan. Items marked **NEW** were not in it before.

| Priority | Item | Why |
| --- | --- | --- |
| **1 (NEW)** | **Extend to the 8 unused conditions** | Three times the data, the full 12-class problem, and access to somatotopic-gradient results the current design cannot produce. Nothing else on the list changes the project's scope this much. |
| **2 (NEW)** | **Somatotopic RSA over 12 body parts** | The neuroscience result the manuscript lacks. Tests whether representational geometry recovers the homunculus ordering — and the literature predicts it will do so only partially, which is itself publishable. |
| 3 | Temporal generalization matrix | Unchanged; cheap interpretation result |
| 4 | What predicts subject decodability | Unchanged; would make QC-60 principled |
| **5 (NEW)** | **Piecewise Optimal Transport alignment** | Benchmarked best at whole-brain scale, robust to tuning; the one alignment method in the standard comparison set not yet tried |
| **6 (NEW)** | **Four-class subset robustness check** | Cheap. Establishes the headline is not specific to the four conditions chosen from twelve. |
| 7 | Permutation null for the frozen decoder | Unchanged |
| **8 (NEW)** | **Domain-adversarial subject-invariant CNN** | Directly targets cross-subject transfer; small change to a model that already works |
| 9 | RSA and confusion structure, 4-class | Partly superseded by item 2 |
| 10 | Cross-contrast transfer, smoothing sweep, ribbon-masked surface | Completeness |

---

## Sources

- [An fMRI dataset for whole-body somatotopic mapping in humans](https://pmc.ncbi.nlm.nih.gov/articles/PMC9399117) — the dataset paper; twelve conditions, block and counterbalancing structure
- [An empirical evaluation of functional alignment using inter-subject decoding](https://pmc.ncbi.nlm.nih.gov/articles/PMC11653789/) — five-method alignment benchmark
- [Inflated prediction accuracy caused by data leakage in feature selection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8042090/)
- [Risk of data leakage in deep-learning-based diagnosis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10547830/) — subject-wise versus trial-wise CV
- [On Leakage in Machine Learning Pipelines](https://arxiv.org/pdf/2311.04179)
- [Beyond body maps: information content distributed across the somatosensory homunculus](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8938902/)
- [Complex organization of human primary motor cortex](https://pubmed.ncbi.nlm.nih.gov/18684903/)
- [Sub-millimeter fMRI reveals digit action maps in human motor cortex](https://www.sciencedirect.com/science/article/pii/S1053811919310547)
- [Inter-subject pattern analysis: a scheme for group-level MVPA](https://www.biorxiv.org/content/10.1101/587899.full.pdf)
- [BrainLM: a foundation model for brain activity](https://proceedings.iclr.cc/paper_files/paper/2024/file/029ce70401321de3808b3ac39e1ab167-Paper-Conference.pdf)
- [MindLink: subject-agnostic cross-subject brain decoding](https://papers.miccai.org/miccai-2025/paper/5263_paper.pdf)
- [Brain Foundation Models: a survey](https://arxiv.org/pdf/2503.00580)
