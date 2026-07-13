# Experiment Registry

This directory records the current experiment state in small JSON files. It is not a storage location for raw fMRI data, checkpoints, Kaggle logs, or manuscript drafts.

Each `*.results.json` file should answer:

- What question does this experiment answer?
- Which split protocol was used?
- Is the run pooled-split or subject-wise?
- Which Kaggle kernel and artifact dataset are associated with it?
- How far has it progressed?
- Which metrics are final, and which are still pending?

Status values:

- `complete`: final metrics are available.
- `running`: currently active or recently submitted on Kaggle.
- `paused`: resumable artifacts exist, but no worker is currently active.
- `failed`: the run errored and needs a fix before relaunching.
- `planned`: config exists, but execution has not started.

## Frozen Confirmation

`confirmation/frozen_protocol.json` locks the final `ds004044` preprocessing, subject splits, seeds, candidate models, consistency weights, QC-60 definition, and deployment rules. `confirmation/investigation_closeout.results.json` records the reproduced metrics, validation checks, artifact hashes, interpretability outputs, and external-confirmation decision.

Broad internal discovery is closed. Add a new modeling experiment only when external evidence identifies a specific timing, registration, or domain-shift failure; do not resume the legacy neural lane or retune the frozen cohort.
