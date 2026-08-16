# Neural Lane Gate

Added: 2026-08-12.

## The problem with the reported negative control

The legacy subject-wise holdout is recorded as `0.2500` accuracy, `0.1000` macro
F1, `0.0` MCC, ROC-AUC `0.4983`, and `best_epoch: 0`. Those numbers are not a
weak result. They are the exact arithmetic signature of a model emitting a single
constant class:

- Four balanced classes, always predicting one, gives recall `[1, 0, 0, 0]` and
  precision `[0.25, 0, 0, 0]`. That class's F1 is `2 * 0.25 * 1 / 1.25 = 0.4`,
  the other three are `0`, so macro F1 is exactly `0.4 / 4 = 0.1000`.
- MCC is exactly `0.0` and ROC-AUC sits at chance.
- `best_epoch: 0` means validation never improved after initialisation.

A model that never leaves its initial state is evidence about the training path,
not about architectures. Reporting it as "the neural recipe fails under
subject-wise evaluation" invites the reply that no model was ever trained, and
the referee would be right.

## What was already known

This does not start from scratch, and the earlier diagnostics were right as far as
they went. `docs/DATASET_DIAGNOSTIC_FINDINGS.md` had already recorded that
evaluation "repeatedly collapsed to a subset of classes despite train=validation",
correctly attributed it to "unstable BatchNorm running statistics under tiny fMRI
batches", added `norm: group` to the encoder (commit `8cae867`), and then found
that **"the GroupNorm follow-up did not fix the corrected temporal model."** It
left an explicit gate: "The next overfit check should pass in evaluation mode
before any broader baseline is trusted." That check was never completed, and the
`0.2500` figure was reported anyway.

So the BatchNorm diagnosis is confirmed here, not discovered. What is new is
*why GroupNorm alone was not enough* — two further faults, one of them fatal on
its own — and a configuration that actually passes the gate the earlier work
specified.

## The gate

`findings_2026-08-12/scripts/run_eval_mode_memorization_probe.py` applies the standard sanity check:
train on a small balanced block with train == validation and require accuracy
`1.00` **in eval mode**, the mode every reported metric is computed in. Train-mode
and eval-mode accuracy are reported at every epoch, so a train/eval mismatch is
visible instead of hidden.

Inputs are 24 balanced events from `sub-01`, taken from the frozen
`(48, 8, 13824)` sequences and reshaped to `[24, 8, 1, 24, 24, 24]`. The neural
lane is therefore probed on exactly the representation that supports the frozen
linear results.

## Finding 1: BatchNorm reproduces the reported failure exactly

With `norm="batch"`, the model fits the training block perfectly in train mode and
collapses to a single class in eval mode:

| Epoch | Train-mode accuracy | Eval-mode accuracy | Distinct predicted classes (eval) |
| ---: | ---: | ---: | ---: |
| 5 | 1.0000 | 0.2500 | 1 |
| 14 | 1.0000 | 0.2500 | 1 |
| 31 | 1.0000 | 0.2500 | 1 |
| 40 | 1.0000 | 0.2500 | 1 |
| 48 | 1.0000 | 0.2500 | 1 |

Eval-mode accuracy is pinned at `0.2500` with exactly one distinct predicted
class across almost every epoch — reproducing `0.2500` accuracy and `0.1000`
macro F1 on data the model has already memorised. BatchNorm running statistics
estimated from tiny fMRI batches diverge from the batch statistics used during
training, and swapping them at eval time destroys the representation.

This also means the train-mode `1.0000` was itself partly an artifact: the model
was exploiting batch composition, which is why the accuracy vanishes the moment
running statistics replace batch statistics.

## Finding 2: without BatchNorm there is no train/eval gap at all

For every architecture without BatchNorm, train-mode and eval-mode accuracy are
**identical at every epoch**, to four decimals. This isolates BatchNorm as the
sole source of the discrepancy and rules out dropout, the evaluation code, the
metric computation, and the data loader.

## Finding 3: the data, labels, and loss path are sound

The gate **passes**. A linear decoder on the same tensors reaches `1.0000` in eval
mode at epoch 304, and an MLP at epoch 510, both with train-mode equal to
eval-mode throughout:

```
epoch=000 loss=1.5055 train_mode=0.2500 eval_mode=0.2500 eval_classes=1
epoch=100 loss=1.0402 train_mode=0.6667 eval_mode=0.6667 eval_classes=4
epoch=200 loss=0.8071 train_mode=0.7917 eval_mode=0.7917 eval_classes=4
epoch=300 loss=0.6293 train_mode=0.9583 eval_mode=0.9583 eval_classes=4
epoch=304 loss=0.6232 train_mode=1.0000 eval_mode=1.0000 eval_classes=4
```

So `0.2500` is not a property of these labels, this representation, or this task.

## Finding 4: the capacity ladder localises the break

Running the gate across a ladder of architectures, all on the same 24 events with
`lr=1e-3` at full batch, separates "cannot learn" from "cannot evaluate":

| Architecture | Epochs | Min loss | Best train-mode | Best eval-mode | Gate | Epochs with single-class eval |
| --- | ---: | ---: | ---: | ---: | :---: | ---: |
| `linear` | 304 | 0.6232 | 1.0000 | **1.0000** | PASS | 29/305 |
| `mlp` | 510 | 0.4149 | 1.0000 | **1.0000** | PASS | 33/511 |
| `shallow_cnn` | 1199 | 1.0892 | 0.6667 | 0.6667 | FAIL | 22/1200 |
| `temporal_resnet3d` (BatchNorm) | 46 | **0.0017** | **1.0000** | **0.2917** | FAIL | **45/47** |

The last row is the whole argument. The ResNet drives training loss to `0.0017`
and train-mode accuracy to `1.0000` — it has memorised the block completely — yet
its eval-mode accuracy never exceeds `0.2917`, and in 45 of 47 epochs it emits a
single constant class. The network learned; the evaluation path discarded what it
learned.

`shallow_cnn` fails differently and for an unrelated reason: it plateaus at
`0.6667` with train-mode equal to eval-mode, an ordinary capacity or optimisation
limit with no train/eval pathology.

## Finding 5: the mechanism, measured layer by layer

Feeding 24 events through an untrained encoder and measuring the **between-sample
standard deviation** at each stage separates the two failure modes:

| Stage | Spatial size | BatchNorm (eval) | GroupNorm |
| --- | :---: | ---: | ---: |
| stem | 6³ | 0.016816 | 0.032824 |
| layer1 | 6³ | 0.017090 | 0.043944 |
| layer2 | 3³ | 0.003438 | 0.029692 |
| layer3 | 2³ | 0.001350 | 0.029153 |
| layer4 | 1³ | **0.000224** | 0.029939 |
| logits | — | **0.001167** | 0.017114 |

Under BatchNorm in eval mode the between-sample signal decays by roughly two
orders of magnitude through the stack, until every sample produces effectively the
same logits — the mechanical cause of the constant-class output. Under GroupNorm
the signal is preserved. GroupNorm was therefore never a representation problem.

The second observation is that the raw input carries a between-sample standard
deviation of only `0.0142` at a given voxel, while each volume is z-scored to unit
variance overall. The discriminative variation is roughly **1.4% of total
variance**; the rest is the shared subject-and-run mean pattern. A global-average-
pooled convolutional stack cannot find a signal that small, which is why GroupNorm
settles at exactly `ln(4) = 1.3863` — the loss of a network that has given up and
emits uniform logits.

## Finding 6: the neural lane works once both faults are fixed

Applying the frozen pipeline's own unlabeled subject-run centering before the
probe, with GroupNorm, makes `temporal_resnet3d` pass the gate:

```
epoch=000 loss=1.6658 train_mode=0.2500 eval_mode=0.2500 eval_classes=1
epoch=040 loss=1.3877 train_mode=0.2500 eval_mode=0.2500 eval_classes=1
epoch=060 loss=1.3817 train_mode=0.5833 eval_mode=0.5833 eval_classes=4
epoch=080 loss=1.0517 train_mode=0.8333 eval_mode=0.8333 eval_classes=4
epoch=096 loss=0.3599 train_mode=1.0000 eval_mode=1.0000 eval_classes=4
```

Reproduced on three subjects: `sub-01` at epoch 96, `sub-07` at 165, `sub-23` at
205, with train-mode equal to eval-mode at every epoch in each case.

Two independent and individually sufficient faults are enough to explain the
memorisation failure:

1. **BatchNorm** running statistics estimated from tiny fMRI batches, which
   destroyed the representation at evaluation time.
2. **Uncentered inputs**, which left the discriminative signal at about 1.4% of
   variance — recoverable by a linear model, but not by this convolutional stack.

Notably the second fix is the same unlabeled subject-run centering that the linear
results depend on (see [Standard MVPA Baseline](STANDARD_MVPA_BASELINE.md)). One
preprocessing step is load-bearing for both lanes.

A third fault, invisible at this scale, appears as soon as the model is asked to
learn across subjects rather than memorise one block. Findings 7 and 8 isolate it.

## Finding 7: passing the gate is necessary but not sufficient

Passing a memorisation gate proves the training path works. It does **not** prove
the architecture can learn cross-subject structure, and the two must not be
conflated.

Running the corrected configuration (GroupNorm, subject-run centering) on a full
outer fold — roughly 1860 training events from ~50 subjects rather than 24 events
from one — the model does not learn at all:

```
epoch=001 loss=1.4206 train=0.2500 inner_select=0.2500
epoch=004 loss=1.3949 train=0.2500 inner_select=0.2500
epoch=006 loss=1.3936 train=0.2500 inner_select=0.2500
epoch=008 loss=1.3901 train=0.2500 inner_select=0.2500
```

**Training accuracy is at chance**, so this is underfitting, not overfitting. The
loss again descends toward `ln(4) = 1.3863` and the network emits uniform logits.
Reporting training accuracy alongside validation accuracy is what makes this
readable; the withdrawn figures reported only the validation side, which is why a
total training failure was mistaken for a generalisation result. Any future neural
number in this project must be reported with its training accuracy beside it.

### Leading hypothesis for the next experiment

The encoder downsamples a `24³` input to `1³` by layer4 and then global-average-
pools. Global pooling is translation-invariant, but somatotopic decoding is
**entirely about location** — left leg versus right leg differs in *where* the
activation sits, not whether it is present. An architecture that discards spatial
position cannot represent the discriminative variable, which would explain why it
memorises 24 samples through incidental cues yet cannot learn a rule that
transfers.

The linear decoders, which assign an independent weight to each of the 13824
voxels and so preserve location exactly, reach `0.8051` on the same data. That
contrast is consistent with the hypothesis but does not establish it.

The decisive test is a variant that retains spatial layout, evaluated under the
same protocol. It was run, and it confirms the hypothesis.

## Finding 8: global average pooling was the blocker

`SpatialCNN` keeps a coarse `6³` spatial map and flattens it instead of
global-average-pooling. Everything else is held constant — same events, same
centering, same optimiser, same splits, same seed. On the same outer fold that
left `temporal_resnet3d` at chance:

| Epoch | Loss | Train accuracy | Inner-select (held-out subjects) |
| ---: | ---: | ---: | ---: |
| 0 | 1.0767 | 0.8042 | 0.6619 |
| 1 | 0.5016 | 0.9708 | 0.8029 |
| 3 | 0.1125 | 1.0000 | 0.8141 |
| 6 | 0.0153 | 1.0000 | **0.8429** |
| 19 | 0.0014 | 1.0000 | 0.8349 |

It learns in a single epoch. On the truly held-out outer fold it reaches
**`0.7936` independent** and **`0.8636` balanced**, with a train-mode/eval-mode gap
of exactly `0.0` and all four classes predicted. For comparison on that same fold,
the linear SVM reaches `0.8030` independent and `0.8750` balanced.

The conclusion is mechanistic rather than a matter of tuning: global average
pooling is translation-invariant, and somatotopic class identity **is** a spatial
position. Pooling the final `1³` map discards precisely the discriminative
variable. The published recipe's failure is an architecture-design error about
what the task requires, not evidence that convolutional models are unsuited to
task fMRI.

This is a stronger and more interesting result than the withdrawn claim: a
correctly designed convolutional decoder is competitive with conventional linear
MVPA on this task, and the original chance-level figure reflected three
compounding faults — BatchNorm statistics, uncentered inputs, and destroyed
spatial information.

## Finding 9: the honest subject-wise number

`SpatialCNN` was then run across all six folds of seed 11 under the full protocol,
with epoch selection on an inner split of the outer-training subjects only. To keep
the comparison exact, every other decoder is restricted to those same six folds:

| Decoder | Independent | Balanced |
| --- | ---: | ---: |
| Frozen nested temporal selector | **0.8337** | **0.8838** |
| `linear_svm` | 0.8109 | 0.8508 |
| `logistic_l2` | 0.8050 | 0.8519 |
| `spatial_cnn` | 0.7840 | 0.8513 |
| `correlation_centroid` | 0.6936 | 0.7430 |

Per-fold independent accuracy: `0.7936, 0.6894, 0.8521, 0.7875, 0.8125, 0.7688`
— mean `0.7840`, standard deviation `0.0496`, range `0.163`. Selected epochs were
`6, 3, 11, 4, 6, 4`, so the inner selection is choosing early and consistently.

The train-mode minus eval-mode gap is **exactly `0.0000` in all six folds** under
the independent rule, and every fold predicts all four classes. The BatchNorm
pathology has not recurred.

### Completed across all five seeds

The remaining four seeds were run, giving the full 30-fold estimate on the same
protocol as every other decoder:

| Decoder | Independent | Balanced |
| --- | ---: | ---: |
| Frozen nested temporal selector | **0.8314** | **0.8806** |
| `linear_svm` | 0.8051 | 0.8441 |
| `logistic_l2` | 0.8028 | 0.8456 |
| `spatial_cnn` | 0.7913 | 0.8511 |
| `correlation_centroid` | 0.6963 | 0.7434 |

`spatial_cnn` over 30 folds: independent `0.7913` (sd `0.0453`, range `0.6894`–`0.8598`),
balanced `0.8511`. The single-seed estimate of `0.7840` was mildly pessimistic but
well inside fold noise.

The integrity checks hold across all 30 folds: the train-mode minus eval-mode gap is
exactly `0.0000` under the independent rule in **every** fold, and every fold predicts
all four classes. The BatchNorm pathology has not recurred anywhere.

### What this does and does not support

- The corrected convolutional decoder **trains normally and lands in the same range
  as conventional linear MVPA**, tying it under the balanced rule (`0.8513` versus
  `0.8508` and `0.8519`).
- It is **below** the linear decoders on the independent rule by roughly `0.02`–`0.03`,
  and below the frozen hierarchy by `0.05`.
- So the honest claim is *not* that a convolutional model wins. It is that the
  published chance-level figure was an artifact, and that a correctly designed
  convolutional decoder is competitive rather than useless.

Two caveats that must travel with these numbers:

1. **One seed.** This is six folds from seed 11, against 30 folds for the linear
   decoders. The comparison above controls for that by restricting everything to
   the same six folds, but the CNN estimate is correspondingly less stable.
2. **Fold variance is large.** A `0.163` range across folds means single-fold
   comparisons are not informative; only the six-fold mean should be quoted.

### A reporting flaw found and fixed

The first version of the harness computed the train/eval gap by comparing
train-mode **argmax** accuracy against the **balanced-assignment** accuracy. Those
are different prediction rules, so balanced rows showed a spurious gap of `-0.06`
to `-0.09` that had nothing to do with train/eval behaviour. The gap is now
computed by applying the same rule to both, which is what makes the exact
`0.0000` above meaningful.

## Finding 10: a second, independent optimisation failure

At the original settings (`batch_size=8`, `lr=3e-4` and `1e-3`) the loss
oscillates between `1.2` and `1.8` around `ln(4) = 1.386` and never converges;
GroupNorm stays pinned at chance regardless of learning rate. At full batch with
`lr=1e-3` the same models descend monotonically. The legacy configuration
therefore had a normalisation bug *and* an optimisation bug, and either alone is
enough to produce the reported number.

## Finding 11: the engine's gradient handling is sound

For completeness, the gradient-accumulation flush at
`src/fmri_pipeline/training/engine.py:72` was audited and is **correct** for
map-style dataloaders: after the loop `step == len(dataloader)`, so pending
gradients exist exactly when `len(dataloader) % grad_accum_steps != 0`, which is
the condition tested. It would only misbehave for an iterable dataset whose
`__len__` disagrees with its iteration count, which is not the case here.

The one real inefficiency is that `GradScaler` is constructed inside
`train_one_epoch` (line 30), so the AMP loss scale is re-calibrated from scratch
every epoch instead of persisting. This costs a few wasted steps per epoch under
CUDA AMP and is inert on CPU and MPS, where the scaler is disabled. It is not
connected to the collapse.

Recording this matters: the failure is specifically the BatchNorm train/eval
mismatch plus the optimiser configuration, and not a general defect in the
training loop.

## Required reporting change

The `0.2500` subject-wise and `0.2629` pooled figures are **withdrawn**. They are
training-path artifacts with three compounding, individually sufficient causes:
BatchNorm running statistics estimated from tiny batches, uncentered inputs, and
global average pooling that discards spatial position.

The defensible statement is:

> The originally reported chance-level neural results are training-path artifacts.
> Once GroupNorm replaces BatchNorm, the unlabeled subject-run centering is applied,
> and the encoder retains spatial layout instead of global-average-pooling, a
> convolutional decoder trains normally and reaches subject-wise accuracy
> competitive with conventional linear MVPA. The failure reflected an architecture
> design mismatched to a spatially defined task, not a limitation of convolutional
> models on task fMRI.

Every neural number reported from this project must carry its **training accuracy**
alongside its validation accuracy. Reporting only the validation side is what
allowed a total training failure to be mistaken for a generalisation result.

## Reproduction

The capacity ladder and the BatchNorm collapse:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python findings_2026-08-12/scripts/run_eval_mode_memorization_probe.py \
  --checkpoint-dir "/path/to/subject_checkpoints" \
  --out-json "/path/to/memorization_gate_ladder.json" \
  --architectures linear mlp shallow_cnn spatial_cnn temporal_resnet3d \
  --norms batch group \
  --lr 1e-3 --batch-size 24 --epochs 1200
```

The gate passing once both faults are fixed, which needs `--center-by-run`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python findings_2026-08-12/scripts/run_eval_mode_memorization_probe.py \
  --checkpoint-dir "/path/to/subject_checkpoints" \
  --out-json "/path/to/gate_centered.json" \
  --architectures temporal_resnet3d --norms group \
  --center-by-run --lr 1e-3 --batch-size 24 --epochs 300
```

Subject-wise evaluation of the corrected decoder. Swap `--architecture` between
`temporal_resnet3d` and `spatial_cnn` to reproduce the pooling contrast:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python findings_2026-08-12/scripts/run_subjectwise_neural_decoder.py \
  --checkpoint-dir "/path/to/subject_checkpoints" \
  --out-json "/path/to/subjectwise_spatial_cnn_seed11.json" \
  --architecture spatial_cnn --subject-seeds 11 --epochs 20 --lr 1e-3
```

The tracked lightweight record is
`findings_2026-08-12/experiments/neural_lane_gate.results.json`.
