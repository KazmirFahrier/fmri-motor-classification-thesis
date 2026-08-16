#!/usr/bin/env python3
"""Subject-wise evaluation of the corrected neural configuration.

The legacy neural figures are withdrawn: they came from a BatchNorm train/eval
mismatch compounded by uncentered inputs (see `docs/NEURAL_LANE_GATE.md`). This
script reports what the same architecture achieves once both faults are fixed,
under the same subject-wise protocol used for every other decoder in the project.

Corrections applied relative to the legacy recipe:

- `norm="group"`, so evaluation does not depend on running statistics estimated
  from tiny fMRI batches.
- The frozen pipeline's unlabeled subject-run centering and per-lag detrending,
  without which the discriminative variation is about 1.4% of total variance and
  the network settles at uniform logits.
- Epoch selection on an inner split of the outer-training subjects only. Held-out
  subjects never influence training or model selection.

Every reported metric is computed in eval mode, and the training-mode accuracy is
recorded alongside it so a recurrence of the original mismatch stays visible.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# This module lives in a findings folder rather than the repository's own
# scripts/ directory, so the shared primitives are resolved explicitly.
ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT / "src", ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from fmri_pipeline.models.temporal_resnet3d import TemporalResNet3D  # noqa: E402
from run_eval_mode_memorization_probe import SpatialCNN  # noqa: E402
from run_balanced_event_assignment import (  # noqa: E402
    apply_balanced_assignment,
    metrics,
)
from run_detrended_pair_feature_selection import (  # noqa: E402
    inner_splits,
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402


CLASS_COUNT = 4


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(grid: int, clip_length: int, args: argparse.Namespace) -> nn.Module:
    if args.architecture == "spatial_cnn":
        return SpatialCNN(CLASS_COUNT, channels=args.base_channels)
    return TemporalResNet3D(
        in_channels=1,
        num_classes=CLASS_COUNT,
        base_channels=args.base_channels,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        num_layers=args.transformer_layers,
        num_heads=4,
        max_clip_length=clip_length,
        norm="group",
    )


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    x: torch.Tensor,
    index: np.ndarray,
    device: torch.device,
    batch_size: int,
    training_mode: bool = False,
) -> np.ndarray:
    was_training = model.training
    model.train(training_mode)
    scores = []
    for start in range(0, len(index), batch_size):
        chunk = index[start : start + batch_size]
        scores.append(model(x[chunk].to(device)).float().cpu().numpy())
    model.train(was_training)
    return np.concatenate(scores, axis=0)


def train_fold(
    x: torch.Tensor,
    y: np.ndarray,
    train_idx: np.ndarray,
    select_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    grid: int,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(args.model_seed)
    model = build_model(grid, int(x.shape[1]), args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    y_train = torch.from_numpy(y[train_idx])

    best_state = None
    best_score = -1.0
    best_epoch = -1
    history = []
    generator = np.random.default_rng(args.model_seed)

    for epoch in range(args.epochs):
        model.train()
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            scale = (epoch + 1) / args.warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
        else:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        order = generator.permutation(len(train_idx))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            positions = order[start : start + args.batch_size]
            batch_x = x[train_idx[positions]].to(device)
            batch_y = y_train[positions].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(positions)

        select_scores = predict_scores(
            model, x, select_idx, device, args.eval_batch_size
        )
        select_accuracy = float(
            (select_scores.argmax(axis=1) == y[select_idx]).mean()
        )
        # Training accuracy in eval mode is what separates "did not learn" from
        # "learned and did not generalise". Without it a low validation score is
        # uninterpretable, which is the error that produced the withdrawn figures.
        fit_probe = train_idx[: args.train_probe_size]
        fit_scores = predict_scores(model, x, fit_probe, device, args.eval_batch_size)
        fit_accuracy = float((fit_scores.argmax(axis=1) == y[fit_probe]).mean())
        history.append(
            {
                "epoch": epoch,
                "loss": total / len(order),
                "train_accuracy": fit_accuracy,
                "inner_select_accuracy": select_accuracy,
            }
        )
        if select_accuracy > best_score:
            best_score = select_accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"    epoch={epoch:03d} loss={total / len(order):.4f} "
            f"train={fit_accuracy:.4f} inner_select={select_accuracy:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_inner_select_accuracy": best_score,
        "history": history,
    }


def subject_metrics_for(
    y: np.ndarray,
    prediction: np.ndarray,
    val_idx: np.ndarray,
    records: list[dict],
) -> dict[str, dict]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for local_pos, record_idx in enumerate(val_idx):
        grouped[str(records[int(record_idx)]["subject_id"])].append(local_pos)
    return {
        subject: metrics(y[val_idx][positions], prediction[positions])
        for subject, positions in sorted(grouped.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subject-wise evaluation of the corrected GroupNorm neural decoder."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--split-limit", type=int)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    # MPS cannot run dropout inside scaled_dot_product_attention, and the model
    # shares one dropout rate across the transformer and the classifier head.
    # Regularisation comes from weight decay and inner-split epoch selection.
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--architecture",
        choices=["temporal_resnet3d", "spatial_cnn"],
        default="temporal_resnet3d",
        help=(
            "spatial_cnn keeps a coarse spatial map and flattens it rather than "
            "global-average-pooling, testing whether location loss is what blocks "
            "the convolutional lane."
        ),
    )
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument(
        "--train-probe-size",
        type=int,
        default=480,
        help="Training events scored each epoch to distinguish underfitting from overfitting.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        help="Linear learning-rate warmup, which the flat ln(4) start benefits from.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    grid = args.grid
    if sequence.shape[2] != grid**3:
        raise ValueError(f"Feature count {sequence.shape[2]} is not {grid}^3.")
    x = torch.from_numpy(
        sequence.reshape(sequence.shape[0], sequence.shape[1], 1, grid, grid, grid)
    )
    del sequence

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    rows: list[dict] = []
    started = time.time()
    for split in splits:
        inner = inner_splits(
            records, split["train_idx"], "subject", args.inner_subject_fold_count
        )
        select_idx = inner[0]["val_idx"]
        fit_idx = inner[0]["train_idx"]

        val_subjects = {str(records[int(i)]["subject_id"]) for i in split["val_idx"]}
        for name, index in (("fit", fit_idx), ("select", select_idx)):
            overlap = {
                str(records[int(i)]["subject_id"]) for i in index
            } & val_subjects
            if overlap:
                raise RuntimeError(f"Split isolation violated in {name}: {overlap}")

        print(f"{split['split']} fit={len(fit_idx)} select={len(select_idx)} "
              f"val={len(split['val_idx'])}", flush=True)
        model, fit_info = train_fold(
            x, y, fit_idx, select_idx, args, device, grid
        )

        scores = predict_scores(
            model, x, split["val_idx"], device, args.eval_batch_size
        )
        train_mode_scores = predict_scores(
            model, x, split["val_idx"], device, args.eval_batch_size, training_mode=True
        )
        predictions = {
            "independent": scores.argmax(axis=1).astype(np.int64),
            "balanced": apply_balanced_assignment(
                scores, split["val_idx"], records
            ),
        }
        # The gap must compare like with like: apply the SAME prediction rule to
        # train-mode scores. Comparing train-mode argmax against balanced-assignment
        # accuracy mixes two rules and produces a spurious non-zero gap.
        train_mode_predictions = {
            "independent": train_mode_scores.argmax(axis=1).astype(np.int64),
            "balanced": apply_balanced_assignment(
                train_mode_scores, split["val_idx"], records
            ),
        }
        for rule, prediction in predictions.items():
            row_metrics = metrics(y[split["val_idx"]], prediction)
            train_mode_accuracy = float(
                (train_mode_predictions[rule] == y[split["val_idx"]]).mean()
            )
            rows.append(
                {
                    "split": split["split"],
                    "subject_seed": split["subject_seed"],
                    # Named "model" so these rows feed compare_frozen_vs_standard_mvpa.py
                    # unchanged, alongside the linear comparators.
                    "model": args.architecture,
                    "prediction_rule": rule,
                    "best_epoch": fit_info["best_epoch"],
                    "best_inner_select_accuracy": fit_info["best_inner_select_accuracy"],
                    "eval_mode_train_mode_gap": train_mode_accuracy
                    - row_metrics["accuracy"],
                    "distinct_predicted_classes": int(len(set(prediction.tolist()))),
                    "metrics": row_metrics,
                    "subject_metrics": subject_metrics_for(
                        y, prediction, split["val_idx"], records
                    ),
                }
            )
        print(
            f"  {split['split']} independent="
            f"{[r for r in rows if r['split'] == split['split'] and r['prediction_rule'] == 'independent'][0]['metrics']['accuracy']:.4f} "
            f"balanced="
            f"{[r for r in rows if r['split'] == split['split'] and r['prediction_rule'] == 'balanced'][0]['metrics']['balanced_accuracy']:.4f} "
            f"best_epoch={fit_info['best_epoch']} [{time.time() - started:.0f}s]",
            flush=True,
        )

    summary = []
    for rule in ("independent", "balanced"):
        values = [row for row in rows if row["prediction_rule"] == rule]
        if not values:
            continue
        summary.append(
            {
                "prediction_rule": rule,
                "split_count": len(values),
                "mean_accuracy": float(
                    np.mean([row["metrics"]["accuracy"] for row in values])
                ),
                "mean_balanced_accuracy": float(
                    np.mean([row["metrics"]["balanced_accuracy"] for row in values])
                ),
                "mean_macro_f1": float(
                    np.mean([row["metrics"]["macro_f1"] for row in values])
                ),
                "min_distinct_predicted_classes": int(
                    min(row["distinct_predicted_classes"] for row in values)
                ),
            }
        )

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "architecture": args.architecture,
        "norm": "group",
        "preprocess": "frozen (unlabeled subject-run centering and per-lag detrending)",
        "outer_split_count": len(splits),
        "subject_seeds": args.subject_seeds,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "model_selection": "inner split of outer-training subjects only",
        "device": device.type,
        "rows": rows,
        "summary": summary,
        "note": (
            "Corrected configuration replacing the withdrawn legacy neural figures. "
            "GroupNorm removes the BatchNorm train/eval mismatch and the unlabeled "
            "subject-run centering restores a learnable signal. min_distinct_predicted_"
            "classes above 1 confirms the constant-class collapse has not recurred."
        ),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
