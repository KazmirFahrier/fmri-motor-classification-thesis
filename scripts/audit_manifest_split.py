#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: pandas. Install pandas plus a parquet engine, e.g. "
        "`python -m pip install pandas pyarrow`, then rerun this audit."
    ) from exc


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _count_rows(df: pd.DataFrame, group_cols: list[str]) -> Dict[str, int]:
    counts = df.groupby(group_cols).size()
    return {str(k): int(v) for k, v in counts.items()}


def _split_ids(split_data: Dict[str, Any]) -> Dict[str, set[int]]:
    names = ["train", "val", "test", "holdout"]
    ids: Dict[str, set[int]] = {}
    for name in names:
        key = f"{name}_sample_ids"
        if key in split_data:
            ids[name] = {int(x) for x in split_data[key]}

    if "folds" in split_data:
        for fold in split_data["folds"]:
            fold_idx = int(fold["fold"])
            ids[f"fold_{fold_idx:02d}_train"] = {int(x) for x in fold["train_sample_ids"]}
            ids[f"fold_{fold_idx:02d}_val"] = {int(x) for x in fold["val_sample_ids"]}

    return ids


def _split_summary(df: pd.DataFrame, split_ids: Dict[str, set[int]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for name, ids in split_ids.items():
        subset = df.loc[df["sample_id"].isin(ids)]
        summary[name] = {
            "num_samples": int(len(subset)),
            "num_subjects": int(subset["subject_id"].nunique()),
            "num_subject_runs": int(subset[["subject_id", "run_id"]].drop_duplicates().shape[0]),
            "class_counts": {
                str(k): int(v)
                for k, v in subset["class_name"].value_counts().sort_index().to_dict().items()
            },
        }
    return summary


def _membership_leakage(
    df: pd.DataFrame,
    split_ids: Dict[str, set[int]],
    group_cols: list[str],
) -> Dict[str, Any]:
    membership: Dict[tuple[Any, ...], set[str]] = {}
    for split_name, ids in split_ids.items():
        subset = df.loc[df["sample_id"].isin(ids)]
        group_key: str | list[str] = group_cols[0] if len(group_cols) == 1 else group_cols
        for key in subset.groupby(group_key).groups:
            if not isinstance(key, tuple):
                key = (key,)
            membership.setdefault(key, set()).add(split_name)

    leaked = {key: names for key, names in membership.items() if len(names) > 1}
    pattern_counts = Counter(tuple(sorted(names)) for names in leaked.values())
    return {
        "group_columns": group_cols,
        "num_groups": len(membership),
        "num_groups_in_multiple_splits": len(leaked),
        "multi_split_fraction": float(len(leaked) / len(membership)) if membership else 0.0,
        "pattern_counts": {str(k): int(v) for k, v in pattern_counts.items()},
        "examples": [
            {"group": [str(x) for x in key], "splits": sorted(names)}
            for key, names in list(leaked.items())[:10]
        ],
    }


def _vol_id_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for class_name, class_df in df.groupby("class_name"):
        group_sets = (
            class_df.groupby(["subject_id", "run_id"])["vol_id"]
            .apply(lambda s: tuple(sorted(int(x) for x in s)))
            .value_counts()
        )
        result[str(class_name)] = {
            "min_vol_id": int(class_df["vol_id"].min()),
            "max_vol_id": int(class_df["vol_id"].max()),
            "num_unique_vol_ids": int(class_df["vol_id"].nunique()),
            "num_unique_subject_run_vol_sets": int(len(group_sets)),
            "top_subject_run_vol_sets": {
                str(k): int(v)
                for k, v in group_sets.head(5).to_dict().items()
            },
        }
    return result


def audit(index_path: str | Path, split_path: str | Path | None = None) -> Dict[str, Any]:
    df = pd.read_parquet(index_path)
    report: Dict[str, Any] = {
        "index_path": str(index_path),
        "num_samples": int(len(df)),
        "num_subjects": int(df["subject_id"].nunique()),
        "num_subject_runs": int(df[["subject_id", "run_id"]].drop_duplicates().shape[0]),
        "class_counts": {
            str(k): int(v)
            for k, v in df["class_name"].value_counts().sort_index().to_dict().items()
        },
        "batch_counts": {
            str(k): int(v)
            for k, v in df["batch_id"].value_counts().sort_index().to_dict().items()
        },
        "runs_per_subject": {
            str(k): int(v)
            for k, v in (
                df[["subject_id", "run_id"]]
                .drop_duplicates()
                .groupby("subject_id")
                .size()
                .value_counts()
                .sort_index()
                .to_dict()
            ).items()
        },
        "samples_per_subject_run_class": {
            str(k): int(v)
            for k, v in (
                df.groupby(["subject_id", "run_id", "class_name"])
                .size()
                .value_counts()
                .sort_index()
                .to_dict()
            ).items()
        },
        "vol_id_patterns": _vol_id_patterns(df),
    }

    if split_path:
        split_data = _load_json(split_path)
        ids = _split_ids(split_data)
        report["split_path"] = str(split_path)
        report["split_summary"] = _split_summary(df, ids)
        report["subject_leakage"] = _membership_leakage(df, ids, ["subject_id"])
        report["subject_run_leakage"] = _membership_leakage(df, ids, ["subject_id", "run_id"])
        report["subject_run_class_block_leakage"] = _membership_leakage(
            df,
            ids,
            ["subject_id", "run_id", "class_name"],
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit thesis manifest and split structure.")
    parser.add_argument("--index", required=True, help="Manifest parquet path.")
    parser.add_argument("--split", default=None, help="Optional pooled or subject-wise split JSON path.")
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    report = audit(args.index, args.split)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
