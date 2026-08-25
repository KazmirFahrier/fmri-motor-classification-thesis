# Journal Strategy

Last reviewed: 2026-08-24.

## Primary Target

**NeuroImage** is the primary target if the internal exact pipeline null and the HCP external mechanism replication both pass. Its scope explicitly includes important developments in brain function and organization as well as advances in neuroimaging analysis methodology. The current manuscript fits as a methods and reproducibility paper rather than as a new neural architecture paper.

Submission requirements that affect the repository now:

* abstract no longer than 250 words,
* one to seven keywords,
* three to five highlights, each no longer than 85 characters,
* data and code availability statements,
* explicit author contributions, competing interests, funding, and ethics statements,
* editable source, separate publication quality figures, and a graphical abstract if used.

The current journal page lists an open access publication charge of USD 3,540 before taxes. Funding or waiver eligibility must be checked before submission.

## Alternative Targets

**Human Brain Mapping** is the first alternative. Its stated scope includes basic, technical, theoretical, and clinical work in human brain mapping. It is a strong fit if the paper emphasizes somatotopic representation and practical multivariate methodology.

**Imaging Neuroscience** is the second alternative. Its scope includes significant contributions to brain function and major advances in imaging methods. Its submission guidance strongly encourages public code and persistent research artifacts, requires data and code availability, and requests at least five potential reviewers.

Journal quartiles can change by year and category. The corresponding author must verify the current Journal Citation Reports category and institutional open access agreement immediately before submission. The repository does not label a journal as Q1 without that dated verification.

## Editorial Positioning

Proposed title:

> Preprocessing Dominates Decoder Architecture in Subject Independent Motor Task fMRI

One sentence contribution:

> A full cohort, leakage aware investigation shows that target run adaptation and native spatial smoothing determine subject independent motor decoding performance, while a custom hierarchy offers no reliable advantage over a matched linear classifier.

The paper is not positioned as:

* a novel deep learning architecture,
* universal external generalization,
* anatomical causality from classifier weights,
* ordinary independent prediction when the method uses unlabeled test run composition.

## Submission Gate

Do not submit until all items pass:

1. Joint nested preprocessing confirmation is complete and archived.
2. Exact pipeline permutation null is complete.
3. HCP external replication is complete, or the journal strategy is explicitly downgraded to a single cohort methods paper.
4. Main and supplementary tables are generated from machine readable records.
5. Every headline number is reconciled to an artifact hash.
6. Public clean clone tests pass in GitHub Actions.
7. The manuscript contains a limitations section that distinguishes induction, transduction, and personalization.
8. At least one domain expert reviews the neuroimaging methods and one independent reviewer audits leakage and statistics.

## Evidence Standard

The strongest paper is not the one with the largest accuracy. It is the one in which every comparison uses the same held out subjects, every adaptive decision is nested, every test set operation is disclosed, null distributions repeat the full selection pipeline, and negative controls remain visible.
