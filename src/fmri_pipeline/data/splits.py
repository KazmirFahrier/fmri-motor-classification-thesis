from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class SplitSpec:
    seed: int
    holdout_subject_count: int
    cv_folds: int


def _subject_list(df: pd.DataFrame) -> List[str]:
    return sorted(df["subject_id"].unique().tolist())


def resolve_sample_ids(df: pd.DataFrame, subjects: Sequence[str]) -> List[int]:
    subject_set = set(subjects)
    sample_ids = df.loc[df["subject_id"].isin(subject_set), "sample_id"].tolist()
    return [int(x) for x in sample_ids]


def create_subjectwise_splits(
    df: pd.DataFrame,
    seed: int,
    holdout_subject_count: int,
    cv_folds: int,
) -> Dict[str, object]:
    subjects = _subject_list(df)
    if len(subjects) <= holdout_subject_count:
        raise ValueError(
            f"holdout_subject_count ({holdout_subject_count}) must be smaller than number of subjects ({len(subjects)})"
        )

    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)

    holdout_subjects = sorted(shuffled[:holdout_subject_count])
    cv_subjects = sorted(shuffled[holdout_subject_count:])

    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2")
    if cv_folds > len(cv_subjects):
        raise ValueError(
            f"cv_folds ({cv_folds}) cannot exceed number of cv subjects ({len(cv_subjects)})"
        )

    fold_defs: List[Dict[str, object]] = []
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_array = np.array(cv_subjects)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(cv_array)):
        train_subjects = sorted(cv_array[train_idx].tolist())
        val_subjects = sorted(cv_array[val_idx].tolist())

        fold_defs.append(
            {
                "fold": int(fold_idx),
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "train_sample_ids": resolve_sample_ids(df, train_subjects),
                "val_sample_ids": resolve_sample_ids(df, val_subjects),
            }
        )

    output = {
        "seed": int(seed),
        "num_subjects": len(subjects),
        "subject_ids": subjects,
        "holdout_subjects": holdout_subjects,
        "holdout_sample_ids": resolve_sample_ids(df, holdout_subjects),
        "cv_subjects": cv_subjects,
        "cv_folds": int(cv_folds),
        "folds": fold_defs,
    }
    return output


def assert_no_subject_leakage(split_data: Dict[str, object]) -> None:
    holdout = set(split_data["holdout_subjects"])
    for fold in split_data["folds"]:
        train = set(fold["train_subjects"])
        val = set(fold["val_subjects"])

        if train.intersection(val):
            raise ValueError(f"Leakage: train/val overlap in fold {fold['fold']}")
        if train.intersection(holdout):
            raise ValueError(f"Leakage: train/holdout overlap in fold {fold['fold']}")
        if val.intersection(holdout):
            raise ValueError(f"Leakage: val/holdout overlap in fold {fold['fold']}")


def make_internal_train_val_subject_split(
    subjects: Sequence[str],
    val_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")

    unique_subjects = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    arr = np.array(unique_subjects)
    rng.shuffle(arr)
    n_val = max(1, int(round(len(arr) * val_fraction)))
    val_subjects = sorted(arr[:n_val].tolist())
    train_subjects = sorted(arr[n_val:].tolist())
    if not train_subjects:
        raise ValueError("Internal split created empty train subject set")
    return train_subjects, val_subjects
