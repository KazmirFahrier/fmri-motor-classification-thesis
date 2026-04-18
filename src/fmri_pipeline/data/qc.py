from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd


def _to_set(values: Iterable[str]) -> set[str]:
    return {str(v) for v in values}


def run_manifest_checks(
    df: pd.DataFrame,
    expected_class_names: Sequence[str],
    parse_failure_count: int = 0,
) -> Tuple[Dict[str, object], List[str]]:
    failures: List[str] = []

    expected_classes = _to_set(expected_class_names)
    present_classes = _to_set(df["class_name"].unique().tolist()) if len(df) else set()

    missing_classes = sorted(expected_classes.difference(present_classes))
    unexpected_classes = sorted(present_classes.difference(expected_classes))
    if missing_classes:
        failures.append(f"Missing expected classes: {missing_classes}")
    if unexpected_classes:
        failures.append(f"Unexpected classes found: {unexpected_classes}")

    if parse_failure_count > 0:
        failures.append(f"Filename parsing failures detected: {parse_failure_count}")

    missing_files = int((~df["exists"]).sum()) if len(df) else 0
    if missing_files > 0:
        failures.append(f"Missing files on disk: {missing_files}")

    subject_run_missing_classes: Dict[str, List[str]] = {}
    if len(df):
        grouped = df.groupby(["subject_id", "run_id"])
        for (subject_id, run_id), g in grouped:
            classes_here = _to_set(g["class_name"].unique())
            missing_here = sorted(expected_classes.difference(classes_here))
            if missing_here:
                key = f"{subject_id}_run-{run_id}"
                subject_run_missing_classes[key] = missing_here

    if subject_run_missing_classes:
        failures.append(
            f"Subject-run groups missing one or more classes: {len(subject_run_missing_classes)}"
        )

    class_counts = (
        df["class_name"].value_counts().sort_index().to_dict() if len(df) else {}
    )
    class_count_values = np.array(list(class_counts.values()), dtype=np.float32) if class_counts else np.array([])
    class_imbalance_ratio = float(class_count_values.max() / class_count_values.min()) if len(class_count_values) else float("nan")

    report: Dict[str, object] = {
        "num_samples": int(len(df)),
        "num_subjects": int(df["subject_id"].nunique()) if len(df) else 0,
        "num_subject_runs": int(df[["subject_id", "run_id"]].drop_duplicates().shape[0]) if len(df) else 0,
        "present_classes": sorted(present_classes),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "class_imbalance_ratio": class_imbalance_ratio,
        "missing_files": missing_files,
        "subject_run_missing_classes": subject_run_missing_classes,
        "parse_failure_count": int(parse_failure_count),
        "failure_count": len(failures),
        "failures": failures,
    }
    return report, failures


def check_nifti_integrity(
    filepaths: Sequence[str],
    max_files: int | None = None,
) -> Dict[str, object]:
    checked = 0
    corrupted: List[str] = []

    iterable = filepaths[:max_files] if max_files is not None else filepaths

    for fp in iterable:
        checked += 1
        try:
            img = nib.load(fp)
            data = img.get_fdata(dtype=np.float32)
            if data.ndim not in (3, 4):
                corrupted.append(fp)
        except Exception:
            corrupted.append(fp)

    return {
        "files_checked": checked,
        "corrupted_count": len(corrupted),
        "corrupted_files": corrupted,
    }
