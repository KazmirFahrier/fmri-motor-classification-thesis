# Pooled Legacy Baseline Decision Policy

Last updated: 2026-05-17.

## Current Decision

Continue the full-dataset pooled legacy baseline only as a short controlled completion to early stopping. Do not pursue the 200-epoch target unless validation meaningfully improves.

## Evidence

- Subject-wise full-dataset evaluation is complete and chance-level.
- The pooled legacy baseline is improving on training accuracy, but validation remains near chance.
- Latest completed pooled snapshot is epoch 13: validation accuracy 0.2419, balanced accuracy 0.2437, macro F1 0.2285, MCC -0.0090.
- Best visible pooled validation accuracy from the resumed metric rows is 0.2474, still below the 0.25 random baseline for four balanced classes.

## Continue, Stop, Or Extend

- Continue the current guarded Kaggle resume chain until early stopping is reached.
- Stop the legacy lane when there are 25 epochs without validation top-1 accuracy improvement, unless the extension threshold is met.
- Extend beyond early stopping only if validation accuracy is at least 0.30 and validation macro F1 is at least 0.30.
- If validation remains near chance at early stopping, stop this lane and pivot to diagnosis instead of spending more Kaggle quota.

## Post-Stop Pivot

- Audit dataset labels, event timing, class mapping, and split construction.
- Check preprocessing and normalization choices.
- Analyze why the original pooled-subset result reached 0.8522 while full-dataset subject-wise and pooled baselines remain near chance.
- Run simpler sanity-check baselines before launching larger architecture experiments.

## Operator Checklist

- After each Kaggle stop, download outputs and inspect `train/metrics.jsonl`.
- Run `python scripts/assess_pooled_legacy_policy.py <metrics.jsonl> --best-epoch 0`.
- Refresh `kazmirfahrier/thesis-legacy-full-artifacts` only if new epochs were saved.
- Relaunch the next guarded resume only when the policy decision is `continue_short_baseline` or `extend`.
