# External Confirmation Decision

Search date: 2026-07-13. Confirmatory mechanism protocol added 2026-08-24.

## Eligibility Locked Before Evaluation

An exact external confirmation cohort must provide:

- Continuous task fMRI with event/block timing.
- Multiple subjects and subject-independent evaluation.
- Unambiguous left-leg, right-leg, forearm, and upper-arm conditions.
- Enough within-subject observations to apply the frozen `3:8` event protocol.
- Exactly two events per class within a complete run before the repetition-consistency decoder may be evaluated.

The independent decoder may be evaluated without the final composition requirement, but it still requires all four exact class definitions. Label substitutions based only on anatomical proximity are not allowed after seeing results.

## Candidate Search

| Resource | Relevant task content | Decision |
| --- | --- | --- |
| [OpenNeuro `ds004044` v2.0.3](https://openneuro.org/datasets/ds004044/versions/2.0.3) | Exact four target conditions and the repeated balanced block design. | Primary cohort; not external. |
| [HCP Young Adult motor task](https://www.humanconnectome.org/hcp-protocols-ya-task-fmri) | Left/right fingers, left/right toes, and tongue; two blocks per limb condition in each run. | Not an exact confirmation cohort. It can support a separately pre-specified coarse hand/foot or laterality study, but hand cannot be relabeled as forearm/upper arm. |
| [OpenNeuro `ds005366` v2.0.0](https://openneuro.org/datasets/ds005366/versions/2.0.0) | A 155-participant collection of heterogeneous motor/sensory tasks involving fingers, hands, arms, feet, legs, and mouth. Most runs are binary active/rest, scanner/task protocols vary, and body parts differ between runs and participants. | Not compatible with the exact four-class within-run protocol or the frozen balanced decoder. A future coarse body-region transfer study would require a new protocol. |
| [OpenNeuro `ds000114` v1.0.1](https://openneuro.org/datasets/ds000114/versions/1.0.1) | Block-design finger, foot, and lip movements. | Not compatible with the four target labels. |

The source `ds004044` data descriptor itself notes the scarcity of public whole-body somatotopic fMRI cohorts and distinguishes its broad body-part coverage from common foot/hand/tongue paradigms: [Scientific Data 9, 515 (2022)](https://www.nature.com/articles/s41597-022-01644-4).

## Decision

No compatible external four-class confirmation cohort was identified in this search. Therefore:

- No external accuracy is reported.
- The frozen `0.8948` complete-run and `0.8314` independent results remain repeated nested single-cohort estimates, not universal external-generalization estimates.
- HCP, `ds005366`, and `ds000114` must not be used with post-hoc label substitutions to manufacture an apparent replication.
- If a future cohort satisfies the locked criteria, run the independent decoder first. Run repetition-consistency only if the exact two-per-class complete-run composition is present.
- If external performance fails, reopen only timing, registration, or domain-shift diagnostics implicated by that failure. Do not restart unrestricted architecture search.

An exact label replication remains unavailable, but a separate external mechanism
replication is now prespecified in
[`HCP_EXTERNAL_REPLICATION_PROTOCOL.md`](HCP_EXTERNAL_REPLICATION_PROTOCOL.md). It uses
the five genuine HCP motor labels and tests run normalization and native smoothing
without relabeling conditions or using design-constrained assignment.

## Search Record

The search covered OpenNeuro keyword combinations for motor, somatotopy, forearm, upper arm, and left/right leg; the HCP motor protocol; recent public motor-fMRI data descriptors; and the known OpenFMRI/OpenNeuro motor benchmark. This is a documented compatibility search, not a claim that no future or access-controlled cohort can exist.
