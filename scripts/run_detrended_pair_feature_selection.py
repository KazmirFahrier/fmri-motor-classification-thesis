#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    apply_balanced_assignment,
    center_by_subject_run,
    centroid_matrix,
    metrics,
    score_with_centroids,
    split_indices,
)
from run_clip_offset_event_sweep import aggregate_events_for_offset, coarse_metrics
from run_detrended_hierarchy_sweep import (
    centroid_matrix_for_classes,
    exact_scores_from_hierarchy,
    score,
)
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


PAIR_CLASSES = {
    "leg": (0, 1),
    "arm": (2, 3),
}


def load_checkpoints(
    checkpoint_dir: Path,
    window_names: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict]]:
    features = {name: [] for name in window_names}
    labels = []
    records = []
    for path in sorted(checkpoint_dir.glob("sub-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for name in window_names:
                features[name].append(data[name].astype(np.float32))
            labels.append(data["labels"].astype(np.int64))
            records.extend(json.loads(str(data["records_json"])))
    if not labels:
        raise ValueError(f"No subject checkpoints found in {checkpoint_dir}.")
    return (
        {name: np.concatenate(values, axis=0) for name, values in features.items()},
        np.concatenate(labels, axis=0),
        records,
    )


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_features(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict], str]:
    if args.checkpoint_dir:
        features, labels, records = load_checkpoints(
            Path(args.checkpoint_dir),
            [args.window_name],
        )
        return features[args.window_name], labels, records, f"{args.checkpoint_dir}:{args.window_name}"

    feature_dir = Path(args.feature_dir)
    clip_x = np.load(feature_dir / "features.npy").astype(np.float32)
    clip_y = np.load(feature_dir / "labels.npy").astype(np.int64)
    clip_records = json.loads((feature_dir / "records.json").read_text())
    event_x, event_y, event_records = aggregate_events_for_offset(
        clip_x,
        clip_y,
        clip_records,
        args.clip_offset,
    )
    return event_x, event_y, event_records, f"{feature_dir}:clip_offset_{args.clip_offset}"


def pair_indices(indices: np.ndarray, y: np.ndarray, classes: tuple[int, int]) -> np.ndarray:
    return indices[np.isin(y[indices], classes)]


def rank_pair_features(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    classes: tuple[int, int],
) -> np.ndarray:
    first = x[train_idx[y[train_idx] == classes[0]]].astype(np.float64)
    second = x[train_idx[y[train_idx] == classes[1]]].astype(np.float64)
    pooled_variance = 0.5 * (first.var(axis=0) + second.var(axis=0))
    positive = pooled_variance[pooled_variance > 0]
    variance_floor = float(np.median(positive) * 1e-3) if len(positive) else 1e-8
    standardized_difference = np.abs(first.mean(axis=0) - second.mean(axis=0)) / np.sqrt(
        pooled_variance + max(variance_floor, 1e-12)
    )
    return np.argsort(-standardized_difference, kind="stable")


def pair_score_matrix(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    classes: tuple[int, int],
    selected_features: np.ndarray,
) -> np.ndarray:
    pair_train = pair_indices(train_idx, y, classes)
    local_y = (y[pair_train] == classes[1]).astype(np.int64)
    centroids = centroid_matrix_for_classes(
        x[pair_train][:, selected_features],
        local_y,
        [0, 1],
    )
    return score(x[val_idx][:, selected_features], centroids)


def pair_accuracy(
    scores: np.ndarray,
    y: np.ndarray,
    val_idx: np.ndarray,
    classes: tuple[int, int],
) -> float:
    mask = np.isin(y[val_idx], classes)
    true_local = (y[val_idx][mask] == classes[1]).astype(np.int64)
    return float(np.mean(scores[mask].argmax(axis=1) == true_local))


def inner_splits(
    records: list[dict],
    outer_train_idx: np.ndarray,
    family: str,
    subject_fold_count: int,
) -> list[dict]:
    subjects = np.asarray([str(record["subject_id"]) for record in records])
    runs = np.asarray([int(record["run_id"]) for record in records])
    splits = []
    if family == "run":
        for run_id in sorted(set(runs[outer_train_idx].tolist())):
            val_idx = outer_train_idx[runs[outer_train_idx] == run_id]
            train_idx = outer_train_idx[runs[outer_train_idx] != run_id]
            splits.append({"split": f"inner_run_{run_id}", "train_idx": train_idx, "val_idx": val_idx})
        return splits

    subject_list = sorted(set(subjects[outer_train_idx].tolist()))
    fold_count = min(subject_fold_count, len(subject_list))
    for fold_idx in range(fold_count):
        held_out = subject_list[fold_idx::fold_count]
        val_mask = np.isin(subjects[outer_train_idx], held_out)
        splits.append(
            {
                "split": f"inner_subject_{fold_idx}",
                "train_idx": outer_train_idx[~val_mask],
                "val_idx": outer_train_idx[val_mask],
            }
        )
    return splits


def outer_splits(
    records: list[dict],
    split_family: str,
    subject_fold_count: int,
    subject_seeds: list[int],
) -> list[dict]:
    splits = []
    if split_family in ("all", "run"):
        splits.extend(split_indices(records, "run", subject_fold_count))
    if split_family not in ("all", "subject"):
        return splits
    if not subject_seeds:
        splits.extend(split_indices(records, "subject", subject_fold_count))
        return splits

    all_idx = np.arange(len(records), dtype=np.int64)
    subjects = np.asarray([str(record["subject_id"]) for record in records])
    subject_list = np.asarray(sorted(set(subjects.tolist())))
    for seed in subject_seeds:
        shuffled = subject_list.copy()
        np.random.default_rng(seed).shuffle(shuffled)
        for fold_idx in range(subject_fold_count):
            held_out = shuffled[fold_idx::subject_fold_count].tolist()
            val_mask = np.isin(subjects, held_out)
            splits.append(
                {
                    "split": f"subject_seed_{seed}_fold_{fold_idx}",
                    "family": "subject",
                    "subject_seed": int(seed),
                    "val_subjects": held_out,
                    "train_idx": all_idx[~val_mask],
                    "val_idx": all_idx[val_mask],
                }
            )
    return splits


def choose_feature_counts(
    x: np.ndarray,
    y: np.ndarray,
    inner: list[dict],
    feature_counts: list[int],
) -> tuple[dict[str, int], dict[str, list[dict]]]:
    selected = {}
    diagnostics = {}
    for pair_name, classes in PAIR_CLASSES.items():
        rows = []
        for split in inner:
            ranking = rank_pair_features(x, y, split["train_idx"], classes)
            for feature_count in feature_counts:
                count = min(feature_count, x.shape[1])
                scores = pair_score_matrix(
                    x,
                    y,
                    split["train_idx"],
                    split["val_idx"],
                    classes,
                    ranking[:count],
                )
                rows.append(
                    {
                        "split": split["split"],
                        "feature_count": int(count),
                        "accuracy": pair_accuracy(scores, y, split["val_idx"], classes),
                    }
                )
        means = {
            count: float(np.mean([row["accuracy"] for row in rows if row["feature_count"] == count]))
            for count in sorted(set(row["feature_count"] for row in rows))
        }
        best_count = min(means, key=lambda count: (-means[count], count))
        selected[pair_name] = int(best_count)
        diagnostics[pair_name] = [
            {"feature_count": int(count), "mean_inner_accuracy": accuracy}
            for count, accuracy in means.items()
        ]
    return selected, diagnostics


def selected_pair_scores(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    selected_counts: dict[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scores = {}
    rankings = {}
    for pair_name, classes in PAIR_CLASSES.items():
        ranking = rank_pair_features(x, y, train_idx, classes)
        selected_features = ranking[: selected_counts[pair_name]]
        rankings[pair_name] = selected_features
        scores[pair_name] = pair_score_matrix(
            x,
            y,
            train_idx,
            val_idx,
            classes,
            selected_features,
        )
    return scores, rankings


def coarse_score_matrix(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> np.ndarray:
    coarse_y = (y >= 2).astype(np.int64)
    centroids = centroid_matrix_for_classes(x[train_idx], coarse_y[train_idx], [0, 1])
    return score(x[val_idx], centroids)


def choose_coarse_weight(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    inner: list[dict],
    selected_counts: dict[str, int],
    coarse_weights: list[float],
) -> tuple[float, list[dict]]:
    rows = []
    for split in inner:
        pair_scores, _ = selected_pair_scores(
            x,
            y,
            split["train_idx"],
            split["val_idx"],
            selected_counts,
        )
        coarse_scores = coarse_score_matrix(x, y, split["train_idx"], split["val_idx"])
        for weight in coarse_weights:
            exact_scores = exact_scores_from_hierarchy(
                coarse_scores,
                pair_scores["leg"],
                pair_scores["arm"],
                weight,
            )
            pred = apply_balanced_assignment(exact_scores, split["val_idx"], records)
            rows.append(
                {
                    "split": split["split"],
                    "coarse_weight": float(weight),
                    "balanced_accuracy": metrics(y[split["val_idx"]], pred)["balanced_accuracy"],
                }
            )
    means = {
        weight: float(
            np.mean([row["balanced_accuracy"] for row in rows if row["coarse_weight"] == weight])
        )
        for weight in coarse_weights
    }
    best_weight = min(means, key=lambda weight: (-means[weight], weight))
    diagnostics = [
        {"coarse_weight": float(weight), "mean_inner_balanced_accuracy": accuracy}
        for weight, accuracy in means.items()
    ]
    return float(best_weight), diagnostics


def evaluate_outer_split(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    split: dict,
    feature_counts: list[int],
    coarse_weights: list[float],
    inner_subject_fold_count: int,
) -> tuple[list[dict], dict]:
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    inner = inner_splits(records, train_idx, split["family"], inner_subject_fold_count)
    selected_counts, count_diagnostics = choose_feature_counts(
        x,
        y,
        inner,
        feature_counts,
    )
    selected_weight, weight_diagnostics = choose_coarse_weight(
        x,
        y,
        records,
        inner,
        selected_counts,
        coarse_weights,
    )

    pair_scores, rankings = selected_pair_scores(x, y, train_idx, val_idx, selected_counts)
    coarse_scores = coarse_score_matrix(x, y, train_idx, val_idx)
    fused_scores = exact_scores_from_hierarchy(
        coarse_scores,
        pair_scores["leg"],
        pair_scores["arm"],
        selected_weight,
    )
    coarse_pred = coarse_scores.argmax(axis=1).astype(np.int64)
    leg_pred = pair_scores["leg"].argmax(axis=1).astype(np.int64)
    arm_pred = pair_scores["arm"].argmax(axis=1).astype(np.int64) + 2
    true_coarse = (y[val_idx] >= 2).astype(np.int64)

    flat_centroids = centroid_matrix(x[train_idx], y[train_idx])
    flat_scores = score_with_centroids(x[val_idx], flat_centroids)
    predictions = [
        ("flat_all_features_independent", flat_scores.argmax(axis=1).astype(np.int64)),
        ("flat_all_features_balanced", apply_balanced_assignment(flat_scores, val_idx, records)),
        ("selected_pair_predicted_coarse", np.where(coarse_pred == 0, leg_pred, arm_pred)),
        ("selected_pair_oracle_coarse", np.where(true_coarse == 0, leg_pred, arm_pred)),
        ("selected_pair_fused_independent", fused_scores.argmax(axis=1).astype(np.int64)),
        ("selected_pair_fused_balanced", apply_balanced_assignment(fused_scores, val_idx, records)),
    ]
    rows = []
    val_subjects = np.asarray([str(records[int(idx)]["subject_id"]) for idx in val_idx])
    for rule, pred in predictions:
        subject_metrics = {}
        for subject in sorted(set(val_subjects.tolist())):
            subject_mask = val_subjects == subject
            subject_metrics[subject] = metrics(y[val_idx][subject_mask], pred[subject_mask])
        rows.append(
            {
                "split": split["split"],
                "family": split["family"],
                "prediction_rule": rule,
                "metrics": metrics(y[val_idx], pred),
                "coarse_metrics": coarse_metrics(y[val_idx], pred),
                "subject_metrics": subject_metrics,
            }
        )

    overlap = len(set(rankings["leg"].tolist()) & set(rankings["arm"].tolist()))
    hyperparameters = {
        "split": split["split"],
        "family": split["family"],
        "selected_feature_counts": selected_counts,
        "selected_coarse_weight": selected_weight,
        "inner_feature_count_diagnostics": count_diagnostics,
        "inner_coarse_weight_diagnostics": weight_diagnostics,
        "outer_pair_accuracy": {
            pair_name: pair_accuracy(pair_scores[pair_name], y, val_idx, classes)
            for pair_name, classes in PAIR_CLASSES.items()
        },
        "selected_feature_overlap": int(overlap),
        "selected_feature_jaccard": float(
            overlap
            / max(
                1,
                len(set(rankings["leg"].tolist()) | set(rankings["arm"].tolist())),
            )
        ),
        "top_feature_indices": {
            pair_name: ranking[:20].tolist()
            for pair_name, ranking in rankings.items()
        },
        "selected_feature_indices": {
            pair_name: ranking.tolist()
            for pair_name, ranking in rankings.items()
        },
    }
    return rows, hyperparameters


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["prediction_rule"])].append(row)
    summary = []
    for (family, rule), group in sorted(grouped.items()):
        summary.append(
            {
                "family": family,
                "prediction_rule": rule,
                "split_count": len(group),
                "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in group])),
                "mean_balanced_accuracy": float(
                    np.mean([row["metrics"]["balanced_accuracy"] for row in group])
                ),
                "mean_macro_f1": float(np.mean([row["metrics"]["macro_f1"] for row in group])),
                "mean_leg_vs_arm_accuracy": float(
                    np.mean([row["coarse_metrics"]["leg_vs_arm_accuracy"] for row in group])
                ),
            }
        )
    return sorted(summary, key=lambda row: (row["family"], -row["mean_balanced_accuracy"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use nested training-only voxel selection to improve detrended within-leg and "
            "within-arm event discrimination."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--feature-dir")
    inputs.add_argument("--checkpoint-dir")
    parser.add_argument("--window-name", default="offset_3_length_8")
    parser.add_argument("--clip-offset", type=int, default=2)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--split-family", choices=["all", "run", "subject"], default="all")
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument(
        "--subject-seeds",
        nargs="*",
        type=int,
        default=[],
        help="Use repeated shuffled subject folds for these seeds instead of the fixed subject partition.",
    )
    parser.add_argument(
        "--feature-counts",
        nargs="*",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 13824],
    )
    parser.add_argument(
        "--coarse-weights",
        nargs="*",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    args = parser.parse_args()

    event_x, event_y, event_records, source = load_features(args)
    centered = center_by_subject_run(event_x, event_records)
    detrended, group_rows = temporal_detrend_by_subject_run(centered, event_records, degree=1)

    rows = []
    hyperparameters = []
    for split in outer_splits(
        event_records,
        args.split_family,
        args.subject_fold_count,
        args.subject_seeds,
    ):
        print(f'evaluating {split["split"]}', flush=True)
        split_rows, split_hyperparameters = evaluate_outer_split(
            detrended,
            event_y,
            event_records,
            split,
            sorted(set(args.feature_counts)),
            sorted(set(args.coarse_weights)),
            args.inner_subject_fold_count,
        )
        rows.extend(split_rows)
        hyperparameters.append(split_hyperparameters)

    result = {
        "source": source,
        "event_feature_shape": list(event_x.shape),
        "feature_counts": sorted(set(args.feature_counts)),
        "coarse_weights": sorted(set(args.coarse_weights)),
        "subject_seeds": args.subject_seeds,
        "mean_linear_time_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in group_rows])
        ),
        "rows": rows,
        "hyperparameters": hyperparameters,
        "summary": summarize(rows),
        "note": (
            "All voxel rankings, feature-count choices, and coarse-weight choices are learned only "
            "inside each outer training split. Oracle-coarse results use held-out coarse labels and "
            "are diagnostic upper bounds; predicted/fused results do not use held-out labels."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(json.dumps({"out_json": args.out_json, "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
