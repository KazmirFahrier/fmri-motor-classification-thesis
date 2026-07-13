#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(
            f"{label} mismatch: actual={actual:.16g} expected={expected:.16g} "
            f"tolerance={tolerance:.3g}"
        )


def validate_checkpoints(checkpoint_dir: Path, protocol: dict) -> tuple[dict, list[dict]]:
    dataset = protocol["dataset"]
    representation = protocol["representation"]
    sequence_key = representation["sequence_key"]
    expected_subject_shape = tuple(representation["sequence_shape_per_subject"])
    paths = sorted(checkpoint_dir.glob("sub-*.npz"))
    if len(paths) != dataset["subject_count"]:
        raise ValueError(
            f"Expected {dataset['subject_count']} subject checkpoints, found {len(paths)}."
        )

    all_records = []
    manifest_rows = []
    class_counts: Counter[int] = Counter()
    run_count = 0
    for path in paths:
        file_hash = sha256_file(path)
        with np.load(path, allow_pickle=False) as payload:
            required = {sequence_key, "labels", "records_json"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"{path.name} is missing keys: {missing}")
            sequence = payload[sequence_key]
            labels = payload["labels"].astype(np.int64)
            records = json.loads(str(payload["records_json"]))
            if tuple(sequence.shape) != expected_subject_shape:
                raise ValueError(
                    f"{path.name} sequence shape {sequence.shape} != {expected_subject_shape}."
                )
            if labels.shape != (expected_subject_shape[0],) or len(records) != len(labels):
                raise ValueError(f"{path.name} labels/records do not match event count.")
            if not np.all(np.isfinite(sequence)):
                raise ValueError(f"{path.name} contains non-finite sequence values.")

        expected_subject = path.stem
        grouped: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        for label, record in zip(labels.tolist(), records):
            if str(record["subject_id"]) != expected_subject:
                raise ValueError(
                    f"{path.name} contains record for {record['subject_id']}."
                )
            if int(record["class_id"]) != int(label):
                raise ValueError(f"{path.name} has a record/label class mismatch.")
            grouped[int(record["run_id"])].append((int(label), record))
            class_counts[int(label)] += 1
            all_records.append(record)
        if len(grouped) != dataset["runs_per_subject"]:
            raise ValueError(
                f"{path.name} has {len(grouped)} runs, expected {dataset['runs_per_subject']}."
            )
        for run_id, rows in grouped.items():
            if len(rows) != dataset["events_per_run"]:
                raise ValueError(
                    f"{path.name} run {run_id} has {len(rows)} events, "
                    f"expected {dataset['events_per_run']}."
                )
            counts = np.bincount(
                np.asarray([label for label, _ in rows]),
                minlength=len(dataset["classes"]),
            )
            expected = np.full(
                len(dataset["classes"]),
                dataset["events_per_class_per_run"],
                dtype=np.int64,
            )
            if not np.array_equal(counts, expected):
                raise ValueError(
                    f"{path.name} run {run_id} class counts {counts.tolist()} != "
                    f"{expected.tolist()}."
                )
            starts = [int(record["event_start"]) for _, record in rows]
            if len(starts) != len(set(starts)):
                raise ValueError(f"{path.name} run {run_id} has duplicate event starts.")
            run_count += 1

        manifest_rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_hash,
                "event_count": len(labels),
                "run_count": len(grouped),
                "sequence_shape": list(expected_subject_shape),
            }
        )

    if len(all_records) != dataset["event_count"] or run_count != dataset["run_count"]:
        raise ValueError(
            f"Global counts are events={len(all_records)}, runs={run_count}; expected "
            f"events={dataset['event_count']}, runs={dataset['run_count']}."
        )
    expected_class_count = dataset["event_count"] // len(dataset["classes"])
    if class_counts != Counter({class_id: expected_class_count for class_id in range(4)}):
        raise ValueError(f"Unexpected global class counts: {dict(class_counts)}")

    combined = hashlib.sha256()
    for row in manifest_rows:
        combined.update(f"{row['sha256']}  {row['file']}\n".encode())
    summary = {
        "status": "pass",
        "checkpoint_count": len(paths),
        "subject_count": len(paths),
        "run_count": run_count,
        "event_count": len(all_records),
        "class_counts": {
            dataset["classes"][class_id]: class_counts[class_id]
            for class_id in range(len(dataset["classes"]))
        },
        "sequence_shape_per_subject": list(expected_subject_shape),
        "total_checkpoint_bytes": int(sum(row["bytes"] for row in manifest_rows)),
        "combined_checkpoint_sha256": combined.hexdigest(),
    }
    return summary, manifest_rows


def reconstruct_outer_splits(records: list[dict], protocol: dict) -> dict[str, dict]:
    validation = protocol["validation"]
    subjects = np.asarray([str(record["subject_id"]) for record in records])
    subject_list = np.asarray(sorted(set(subjects.tolist())))
    all_indices = np.arange(len(records), dtype=np.int64)
    result = {}
    for seed in validation["outer_seeds"]:
        shuffled = subject_list.copy()
        np.random.default_rng(seed).shuffle(shuffled)
        for fold_index in range(validation["outer_fold_count"]):
            split = f"subject_seed_{seed}_fold_{fold_index}"
            val_subjects = set(shuffled[fold_index :: validation["outer_fold_count"]])
            val_mask = np.isin(subjects, sorted(val_subjects))
            result[split] = {
                "train_idx": all_indices[~val_mask],
                "val_idx": all_indices[val_mask],
                "train_subjects": set(subjects[~val_mask]),
                "val_subjects": val_subjects,
            }
    return result


def validate_split_isolation(
    records: list[dict], protocol: dict, expected_split_names: set[str]
) -> dict:
    reconstructed = reconstruct_outer_splits(records, protocol)
    if set(reconstructed) != expected_split_names:
        missing = sorted(expected_split_names - set(reconstructed))
        extra = sorted(set(reconstructed) - expected_split_names)
        raise ValueError(f"Split-name mismatch; missing={missing}, extra={extra}")

    validation = protocol["validation"]
    inner_checks = 0
    per_seed_counts: dict[int, Counter[str]] = {}
    for seed in validation["outer_seeds"]:
        per_seed_counts[seed] = Counter()
    for split_name, split in reconstructed.items():
        if split["train_subjects"] & split["val_subjects"]:
            raise ValueError(f"Outer split {split_name} leaks validation subjects.")
        seed = int(split_name.split("_")[2])
        per_seed_counts[seed].update(split["val_subjects"])
        train_subjects = sorted(split["train_subjects"])
        for inner_fold in range(validation["inner_subject_fold_count"]):
            inner_val = set(
                train_subjects[inner_fold :: validation["inner_subject_fold_count"]]
            )
            inner_train = set(train_subjects) - inner_val
            if inner_train & inner_val or inner_val & split["val_subjects"]:
                raise ValueError(f"Inner split leakage in {split_name}, fold {inner_fold}.")
            if inner_train | inner_val != split["train_subjects"]:
                raise ValueError(f"Inner split coverage failure in {split_name}.")
            inner_checks += 1
    for seed, counts in per_seed_counts.items():
        if set(counts.values()) != {1}:
            raise ValueError(f"Seed {seed} does not hold out every subject exactly once.")
    return {
        "status": "pass",
        "outer_split_count": len(reconstructed),
        "inner_split_isolation_checks": inner_checks,
        "subjects_held_out_once_per_seed": True,
        "outer_validation_subjects_absent_from_inner_selection": True,
    }


def metric_by_rule(rows: list[dict], rule: str) -> dict:
    matches = [row for row in rows if row["prediction_rule"] == rule]
    if len(matches) != 1:
        raise ValueError(f"Expected one summary row for {rule}, found {len(matches)}.")
    return matches[0]


def subject_average(selected_rows: list[dict], rule: str, excluded: set[str]) -> float:
    values: dict[str, list[float]] = defaultdict(list)
    for row in selected_rows:
        if row["prediction_rule"] != rule:
            continue
        for subject, metrics in row["subject_metrics"].items():
            if subject not in excluded:
                values[subject].append(float(metrics["balanced_accuracy"]))
    return float(np.mean([np.mean(subject_values) for subject_values in values.values()]))


def repetition_subject_average(rows: list[dict], selected: bool, excluded: set[str]) -> float:
    key = "selected_subject_metrics" if selected else "baseline_subject_metrics"
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for subject, metrics in row[key].items():
            if subject not in excluded:
                values[subject].append(float(metrics["balanced_accuracy"]))
    return float(np.mean([np.mean(subject_values) for subject_values in values.values()]))


def aggregate_repetition_subjects(rows: list[dict]) -> list[dict]:
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"baseline": [], "selected": []}
    )
    for row in rows:
        for subject, metrics in row["baseline_subject_metrics"].items():
            values[subject]["baseline"].append(float(metrics["balanced_accuracy"]))
        for subject, metrics in row["selected_subject_metrics"].items():
            values[subject]["selected"].append(float(metrics["balanced_accuracy"]))
    result = []
    for subject in sorted(values):
        baseline = float(np.mean(values[subject]["baseline"]))
        selected = float(np.mean(values[subject]["selected"]))
        result.append(
            {
                "subject": subject,
                "repeat_count": len(values[subject]["selected"]),
                "baseline_balanced_accuracy": baseline,
                "selected_balanced_accuracy": selected,
                "difference": selected - baseline,
            }
        )
    return result


def aggregate_temporal_subjects(selected_rows: list[dict], rule: str) -> list[dict]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in selected_rows:
        if row["prediction_rule"] != rule:
            continue
        for subject, metrics in row["subject_metrics"].items():
            values[subject].append(float(metrics["balanced_accuracy"]))
    return [
        {
            "subject": subject,
            "repeat_count": len(values[subject]),
            "balanced_accuracy": float(np.mean(values[subject])),
        }
        for subject in sorted(values)
    ]


def compare_repetition_results(reference: dict, reproduction: dict, tolerance: float) -> dict:
    scalar_keys = [
        "assignment_count_per_run",
        "mean_baseline_balanced_accuracy",
        "mean_selected_balanced_accuracy",
    ]
    max_difference = 0.0
    for key in scalar_keys:
        difference = abs(float(reference[key]) - float(reproduction[key]))
        max_difference = max(max_difference, difference)
        if difference > tolerance:
            raise ValueError(f"Reproduction mismatch for {key}: {difference}")
    reference_rows = {row["split"]: row for row in reference["rows"]}
    reproduction_rows = {row["split"]: row for row in reproduction["rows"]}
    if set(reference_rows) != set(reproduction_rows):
        raise ValueError("Reproduction split set differs from the reference result.")
    for split, expected in reference_rows.items():
        actual = reproduction_rows[split]
        for key in ("candidate", "selected_weight"):
            if actual[key] != expected[key]:
                raise ValueError(f"{split} reproduction differs for {key}.")
        for section in ("baseline_metrics", "selected_metrics"):
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                difference = abs(float(actual[section][metric]) - float(expected[section][metric]))
                max_difference = max(max_difference, difference)
                if difference > tolerance:
                    raise ValueError(f"{split} {section}.{metric} mismatch: {difference}")
    return {
        "status": "pass",
        "split_count": len(reference_rows),
        "maximum_metric_difference": max_difference,
        "selected_weights_match": True,
        "selected_candidates_match": True,
    }


def validate_lambda_zero(
    temporal: dict, repetition: dict, tolerance: float
) -> dict:
    balanced_rows = {
        row["split"]: row
        for row in temporal["selected_rows"]
        if row["prediction_rule"] == "balanced"
    }
    max_difference = 0.0
    for row in repetition["rows"]:
        baseline = balanced_rows[row["split"]]["metrics"]
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            difference = abs(float(row["baseline_metrics"][metric]) - float(baseline[metric]))
            max_difference = max(max_difference, difference)
            if difference > tolerance:
                raise ValueError(
                    f"Lambda-zero reconstruction failed for {row['split']} {metric}."
                )
    return {
        "status": "pass",
        "reconstructed_split_count": len(repetition["rows"]),
        "maximum_metric_difference": max_difference,
    }


def calibration_summary(calibration: dict, rule: str) -> tuple[dict, dict]:
    row = next(
        item
        for item in calibration["summary"]
        if item["branch_mode"] == "arm"
        and item["protocol"] == "calibration_loo_alpha"
        and item["calibration_run_count"] == 5
        and item["prediction_rule"] == rule
    )
    bootstrap = next(
        item
        for item in calibration["paired_subject_bootstrap"]
        if item["branch_mode"] == "arm"
        and item["protocol"] == "calibration_loo_alpha"
        and item["calibration_run_count"] == 5
        and item["stratum"] == "qc"
    )
    return row, bootstrap


def result_row(
    result_id: str,
    model: str,
    protocol: str,
    prediction_context: str,
    label_requirement: str,
    accuracy: float | None,
    balanced_accuracy: float | None,
    macro_f1: float | None,
    mcc: float | None = None,
    subject_weighted_accuracy: float | None = None,
    qc60_subject_weighted_accuracy: float | None = None,
    ci95: list[float] | None = None,
    claim_status: str = "supporting",
) -> dict:
    return {
        "result_id": result_id,
        "model": model,
        "protocol": protocol,
        "prediction_context": prediction_context,
        "label_requirement": label_requirement,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "subject_weighted_accuracy": subject_weighted_accuracy,
        "qc60_subject_weighted_accuracy": qc60_subject_weighted_accuracy,
        "paired_gain_ci95": ci95,
        "claim_status": claim_status,
    }


def build_benchmark(
    protocol: dict,
    legacy_original: dict,
    legacy_full: dict,
    legacy_subjectwise: dict,
    mean_hierarchy: dict,
    temporal: dict,
    repetition: dict,
    calibration: dict,
) -> tuple[list[dict], dict]:
    qc_excluded = set(protocol["dataset"]["qc60_excluded_subjects"])
    mean_balanced = metric_by_rule(mean_hierarchy["summary"], "hybrid_fused_balanced")
    mean_independent = metric_by_rule(mean_hierarchy["summary"], "hybrid_fused_independent")
    temporal_balanced = metric_by_rule(temporal["summary"], "balanced")
    temporal_independent = metric_by_rule(temporal["summary"], "independent")
    calibration_balanced, calibration_qc = calibration_summary(calibration, "balanced")
    calibration_independent, _ = calibration_summary(calibration, "independent")
    mean_reproduction = calibration["baseline_reproduction"]

    rows = [
        result_row(
            "historical_original_subset",
            legacy_original["model"]["name"],
            legacy_original["split"]["protocol"],
            "pooled historical comparison",
            "supervised labels",
            legacy_original["metrics"]["accuracy"],
            legacy_original["metrics"]["balanced_accuracy"],
            legacy_original["metrics"]["macro_f1"],
            legacy_original["metrics"]["mcc"],
            claim_status="historical_high_leakage_risk",
        ),
        result_row(
            "legacy_full_pooled_epoch25",
            legacy_full["model"]["name"],
            legacy_full["split"]["protocol"],
            "pooled full-dataset validation",
            "supervised labels",
            legacy_full["metrics"]["accuracy"],
            legacy_full["metrics"]["balanced_accuracy"],
            legacy_full["metrics"]["macro_f1"],
            legacy_full["metrics"]["mcc"],
            claim_status="negative_control",
        ),
        result_row(
            "legacy_subjectwise_holdout",
            legacy_subjectwise["model"]["name"],
            legacy_subjectwise["split"]["protocol"],
            "held-out subjects",
            "supervised labels",
            legacy_subjectwise["metrics"]["accuracy"],
            legacy_subjectwise["metrics"]["balanced_accuracy"],
            legacy_subjectwise["metrics"]["macro_f1"],
            legacy_subjectwise["metrics"]["mcc"],
            claim_status="negative_control",
        ),
        result_row(
            "mean_hierarchy_balanced",
            "fixed_3x8_full_covariance_hierarchy",
            "30 repeated nested subject folds",
            "complete balanced runs",
            "zero target labels; known two-per-class composition",
            mean_balanced["mean_accuracy"],
            mean_balanced["mean_balanced_accuracy"],
            mean_balanced["mean_macro_f1"],
            subject_weighted_accuracy=mean_reproduction[
                "reconstructed_subject_weighted_balanced_accuracy"
            ],
            qc60_subject_weighted_accuracy=calibration_qc["source_balanced_accuracy"],
            claim_status="conservative_primary_baseline",
        ),
        result_row(
            "mean_hierarchy_independent",
            "fixed_3x8_full_covariance_hierarchy",
            "30 repeated nested subject folds",
            "independent event prediction",
            "zero target labels",
            mean_independent["mean_accuracy"],
            mean_independent["mean_balanced_accuracy"],
            mean_independent["mean_macro_f1"],
            claim_status="baseline",
        ),
        result_row(
            "temporal_selector_balanced",
            "nested_temporal_candidate_selector",
            "30 repeated nested subject folds",
            "complete balanced runs",
            "zero target labels; known two-per-class composition",
            temporal_balanced["mean_balanced_accuracy"],
            temporal_balanced["mean_balanced_accuracy"],
            temporal_balanced["mean_macro_f1"],
            subject_weighted_accuracy=subject_average(
                temporal["selected_rows"], "balanced", set()
            ),
            qc60_subject_weighted_accuracy=subject_average(
                temporal["selected_rows"], "balanced", qc_excluded
            ),
            ci95=temporal["bootstrap_vs_mean_cap1024"]["balanced"]["fold_ci95"],
            claim_status="supporting",
        ),
        result_row(
            "temporal_selector_independent",
            "nested_temporal_candidate_selector",
            "30 repeated nested subject folds",
            "independent, incomplete, imbalanced, or online events",
            "zero target labels",
            temporal_independent["mean_balanced_accuracy"],
            temporal_independent["mean_balanced_accuracy"],
            temporal_independent["mean_macro_f1"],
            subject_weighted_accuracy=subject_average(
                temporal["selected_rows"], "independent", set()
            ),
            qc60_subject_weighted_accuracy=subject_average(
                temporal["selected_rows"], "independent", qc_excluded
            ),
            ci95=temporal["bootstrap_vs_mean_cap1024"]["independent"]["subject_ci95"],
            claim_status="primary_independent",
        ),
        result_row(
            "repetition_consistency_balanced",
            "nested_repetition_consistency_assignment",
            "30 repeated nested subject folds",
            "complete balanced runs only",
            "zero target labels; known two-per-class composition",
            repetition["mean_selected_balanced_accuracy"],
            repetition["mean_selected_balanced_accuracy"],
            repetition["mean_selected_balanced_accuracy"],
            subject_weighted_accuracy=repetition_subject_average(
                repetition["rows"], True, set()
            ),
            qc60_subject_weighted_accuracy=repetition_subject_average(
                repetition["rows"], True, qc_excluded
            ),
            ci95=repetition["paired_bootstrap"]["subject_ci95"],
            claim_status="primary_complete_run",
        ),
        result_row(
            "five_run_arm_calibration_balanced",
            "validation_selected_arm_calibration",
            "leave-one-evaluation-run-out across repeated subject folds",
            "complete balanced evaluation runs",
            "five labeled target-subject calibration runs",
            calibration_balanced["accuracy"],
            calibration_balanced["balanced_accuracy"],
            calibration_balanced["macro_f1"],
            subject_weighted_accuracy=calibration_balanced["balanced_accuracy"],
            qc60_subject_weighted_accuracy=calibration_qc["calibrated_balanced_accuracy"],
            ci95=calibration_qc["ci95"],
            claim_status="labeled_personalization",
        ),
        result_row(
            "five_run_arm_calibration_independent",
            "validation_selected_arm_calibration",
            "leave-one-evaluation-run-out across repeated subject folds",
            "independent event prediction",
            "five labeled target-subject calibration runs",
            calibration_independent["accuracy"],
            calibration_independent["balanced_accuracy"],
            calibration_independent["macro_f1"],
            claim_status="labeled_personalization",
        ),
    ]

    class_names = protocol["dataset"]["classes"]
    confusion = np.sum(
        [np.asarray(row["selected_metrics"]["confusion_matrix"]) for row in repetition["rows"]],
        axis=0,
    )
    recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    seed_differences = {}
    for seed in protocol["validation"]["outer_seeds"]:
        seed_rows = [
            row for row in repetition["rows"] if row["split"].startswith(f"subject_seed_{seed}_")
        ]
        seed_differences[str(seed)] = float(
            np.mean(
                [
                    row["selected_metrics"]["balanced_accuracy"]
                    - row["baseline_metrics"]["balanced_accuracy"]
                    for row in seed_rows
                ]
            )
        )
    supporting = {
        "complete_run_per_class_recall": dict(zip(class_names, recall.tolist())),
        "complete_run_selected_weight_counts": repetition["selected_weight_counts"],
        "complete_run_seed_differences": seed_differences,
        "complete_run_uncertainty": repetition["paired_bootstrap"],
        "complete_run_fold_results": [
            {
                "split": row["split"],
                "candidate": row["candidate"],
                "selected_weight": row["selected_weight"],
                "baseline_balanced_accuracy": row["baseline_metrics"][
                    "balanced_accuracy"
                ],
                "selected_balanced_accuracy": row["selected_metrics"][
                    "balanced_accuracy"
                ],
                "difference": row["selected_metrics"]["balanced_accuracy"]
                - row["baseline_metrics"]["balanced_accuracy"],
            }
            for row in repetition["rows"]
        ],
        "complete_run_subject_results": aggregate_repetition_subjects(
            repetition["rows"]
        ),
        "complete_run_fold_outcomes": {
            "wins": int(
                sum(
                    row["selected_metrics"]["balanced_accuracy"]
                    > row["baseline_metrics"]["balanced_accuracy"]
                    for row in repetition["rows"]
                )
            ),
            "ties": int(
                sum(
                    row["selected_metrics"]["balanced_accuracy"]
                    == row["baseline_metrics"]["balanced_accuracy"]
                    for row in repetition["rows"]
                )
            ),
        },
        "temporal_candidate_counts": temporal["candidate_counts"],
        "independent_uncertainty": temporal["bootstrap_vs_mean_cap1024"][
            "independent"
        ],
        "independent_subject_results": aggregate_temporal_subjects(
            temporal["selected_rows"], "independent"
        ),
        "mean_hierarchy_baseline_reproduction": mean_reproduction,
    }
    supporting["complete_run_fold_outcomes"]["losses"] = (
        len(repetition["rows"])
        - supporting["complete_run_fold_outcomes"]["wins"]
        - supporting["complete_run_fold_outcomes"]["ties"]
    )
    return rows, supporting


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def format_metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Frozen Final Benchmark",
        "",
        "| Result | Context | Target labels | Balanced accuracy | Subject-weighted | QC-60 | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {result_id} | {prediction_context} | {label_requirement} | {balanced} | "
            "{subject} | {qc} | {status} |".format(
                result_id=row["result_id"],
                prediction_context=row["prediction_context"],
                label_requirement=row["label_requirement"],
                balanced=format_metric(row["balanced_accuracy"]),
                subject=format_metric(row["subject_weighted_accuracy"]),
                qc=format_metric(row["qc60_subject_weighted_accuracy"]),
                status=row["claim_status"],
            )
        )
    lines.extend(
        [
            "",
            "The complete-run decoder is transductive and requires exactly two events per class. "
            "Incomplete, imbalanced, or online events must use the independent decoder. Labeled "
            "calibration is a separate personalization protocol.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the final ds004044 investigation benchmark."
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--sequence-summary", required=True)
    parser.add_argument("--legacy-original", required=True)
    parser.add_argument("--legacy-full", required=True)
    parser.add_argument("--legacy-subjectwise", required=True)
    parser.add_argument("--mean-hierarchy", required=True)
    parser.add_argument("--temporal-selection", required=True)
    parser.add_argument("--repetition-reference", required=True)
    parser.add_argument("--repetition-reproduction", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    paths = {key: Path(value) for key, value in vars(args).items() if key != "out_dir"}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = read_json(paths["protocol"])
    sequence_summary = read_json(paths["sequence_summary"])
    legacy_original = read_json(paths["legacy_original"])
    legacy_full = read_json(paths["legacy_full"])
    legacy_subjectwise = read_json(paths["legacy_subjectwise"])
    mean_hierarchy = read_json(paths["mean_hierarchy"])
    temporal = read_json(paths["temporal_selection"])
    repetition_reference = read_json(paths["repetition_reference"])
    repetition_reproduction = read_json(paths["repetition_reproduction"])
    calibration = read_json(paths["calibration"])
    tolerance = float(protocol["validation"]["metric_tolerance"])

    checkpoint_summary, checkpoint_manifest = validate_checkpoints(
        paths["checkpoint_dir"], protocol
    )
    if int(sequence_summary["subject_count"]) != checkpoint_summary["subject_count"]:
        raise ValueError("Sequence summary and checkpoint subject counts differ.")
    if list(sequence_summary["feature_shape"]) != protocol["representation"]["feature_shape"]:
        raise ValueError("Sequence summary feature shape differs from the frozen protocol.")

    _, _, records = load_lightweight_records(paths["checkpoint_dir"])
    expected_splits = {row["split"] for row in repetition_reproduction["rows"]}
    split_validation = validate_split_isolation(records, protocol, expected_splits)
    reproduction_validation = compare_repetition_results(
        repetition_reference, repetition_reproduction, tolerance
    )
    lambda_zero_validation = validate_lambda_zero(
        temporal, repetition_reproduction, tolerance
    )

    assert_close(
        repetition_reproduction["mean_selected_balanced_accuracy"],
        protocol["models"]["complete_balanced_run_decoder"]["balanced_accuracy"],
        tolerance,
        "complete-run balanced accuracy",
    )
    temporal_independent = metric_by_rule(temporal["summary"], "independent")
    assert_close(
        temporal_independent["mean_balanced_accuracy"],
        protocol["models"]["independent_decoder"]["independent_accuracy"],
        tolerance,
        "independent accuracy",
    )

    benchmark, supporting = build_benchmark(
        protocol,
        legacy_original,
        legacy_full,
        legacy_subjectwise,
        mean_hierarchy,
        temporal,
        repetition_reproduction,
        calibration,
    )
    source_manifest = []
    for name, path in sorted(paths.items()):
        if name == "checkpoint_dir":
            continue
        source_manifest.append(
            {
                "role": name,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    artifact_manifest = {
        "checkpoint_summary": checkpoint_summary,
        "subject_checkpoints": checkpoint_manifest,
        "source_files": source_manifest,
    }
    validation_report = {
        "status": "pass",
        "protocol_id": protocol["protocol_id"],
        "checkpoint_integrity": checkpoint_summary,
        "split_isolation": split_validation,
        "repetition_reproduction": reproduction_validation,
        "lambda_zero_reconstruction": lambda_zero_validation,
        "guardrails": protocol["decision_guardrails"],
    }
    final = {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(paths["protocol"]),
        "validation": validation_report,
        "benchmark": benchmark,
        "supporting_statistics": supporting,
        "artifact_manifest": {
            "combined_checkpoint_sha256": checkpoint_summary[
                "combined_checkpoint_sha256"
            ],
            "source_files": source_manifest,
        },
        "deployment_policy": protocol["deployment_policy"],
    }
    (out_dir / "validation_report.json").write_text(json.dumps(validation_report, indent=2))
    (out_dir / "artifact_manifest.json").write_text(json.dumps(artifact_manifest, indent=2))
    (out_dir / "final_benchmark.json").write_text(json.dumps(final, indent=2))
    write_csv(out_dir / "final_benchmark.csv", benchmark)
    write_markdown(out_dir / "final_benchmark.md", benchmark)
    print(
        json.dumps(
            {
                "status": "complete",
                "out_dir": str(out_dir),
                "checkpoint_integrity": checkpoint_summary,
                "split_isolation": split_validation,
                "reproduction": reproduction_validation,
                "lambda_zero": lambda_zero_validation,
                "primary_complete_run": protocol["models"][
                    "complete_balanced_run_decoder"
                ]["balanced_accuracy"],
                "primary_independent": protocol["models"]["independent_decoder"][
                    "independent_accuracy"
                ],
            },
            indent=2,
        )
    )


def load_lightweight_records(checkpoint_dir: Path) -> tuple[list[str], np.ndarray, list[dict]]:
    subjects = []
    labels = []
    records = []
    for path in sorted(checkpoint_dir.glob("sub-*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            subject_labels = payload["labels"].astype(np.int64)
            subject_records = json.loads(str(payload["records_json"]))
        subjects.append(path.stem)
        labels.append(subject_labels)
        records.extend(subject_records)
    return subjects, np.concatenate(labels), records


if __name__ == "__main__":
    main()
