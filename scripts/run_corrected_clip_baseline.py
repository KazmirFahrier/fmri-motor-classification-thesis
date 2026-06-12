#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fmri_pipeline.config import load_config
from fmri_pipeline.data.manifest import build_manifest, save_manifest_parquet
from fmri_pipeline.training.pipeline import train_fold
from fmri_pipeline.utils.io import write_json
from fmri_pipeline.utils.log_utils import setup_logger
from fmri_pipeline.utils.seed import seed_everything


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]


def find_dataset_root(slug: str) -> Path | None:
    direct = Path("/kaggle/input") / slug
    if direct.exists():
        return direct
    datasets_root = Path("/kaggle/input/datasets")
    if datasets_root.exists():
        matches = sorted(p for p in datasets_root.rglob(slug) if p.is_dir())
        if matches:
            return matches[0]
    matches = sorted(p for p in Path("/kaggle/input").rglob(slug) if p.is_dir())
    return matches[0] if matches else None


def resolve_batch_roots(batch_slugs: List[str]) -> List[Path]:
    roots: List[Path] = []
    missing: List[str] = []
    for slug in batch_slugs:
        root = find_dataset_root(slug)
        if root is None:
            missing.append(slug)
        else:
            roots.append(root)
    if missing:
        raise FileNotFoundError(f"Missing mounted batch datasets: {missing}")
    return roots


def choose_subjects(subjects: List[str], max_subjects: int, val_subject_count: int) -> tuple[List[str], List[str]]:
    selected = sorted(subjects)[:max_subjects]
    if len(selected) <= val_subject_count:
        raise ValueError("Need more selected subjects than validation subjects")
    val_subjects = selected[-val_subject_count:]
    train_subjects = selected[:-val_subject_count]
    return train_subjects, val_subjects


def make_split(
    manifest_df,
    split_mode: str,
    max_subjects: int,
    val_subject_count: int,
    val_run_id: int,
    overfit_subject_id: str,
    overfit_run_id: int,
) -> tuple[List[int], List[int], dict]:
    subjects = sorted(manifest_df["subject_id"].unique().tolist())
    if split_mode == "subject_holdout":
        train_subjects, val_subjects = choose_subjects(subjects, max_subjects, val_subject_count)
        train_ids = manifest_df.loc[manifest_df["subject_id"].isin(train_subjects), "sample_id"].astype(int).tolist()
        val_ids = manifest_df.loc[manifest_df["subject_id"].isin(val_subjects), "sample_id"].astype(int).tolist()
        split_summary = {
            "split_mode": split_mode,
            "all_subject_count": len(subjects),
            "selected_subjects": train_subjects + val_subjects,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "train_sample_count": len(train_ids),
            "val_sample_count": len(val_ids),
        }
        return train_ids, val_ids, split_summary

    if split_mode == "run_holdout":
        selected_subjects = sorted(subjects)[:max_subjects]
        selected = manifest_df.loc[manifest_df["subject_id"].isin(selected_subjects)]
        train_df = selected.loc[selected["run_id"].astype(int) != int(val_run_id)]
        val_df = selected.loc[selected["run_id"].astype(int) == int(val_run_id)]
        if train_df.empty or val_df.empty:
            raise ValueError(f"Run-holdout split produced empty train or val set for val_run_id={val_run_id}")
        train_ids = train_df["sample_id"].astype(int).tolist()
        val_ids = val_df["sample_id"].astype(int).tolist()
        split_summary = {
            "split_mode": split_mode,
            "all_subject_count": len(subjects),
            "selected_subjects": selected_subjects,
            "train_subjects": selected_subjects,
            "val_subjects": selected_subjects,
            "train_run_ids": sorted(int(v) for v in train_df["run_id"].unique().tolist()),
            "val_run_ids": sorted(int(v) for v in val_df["run_id"].unique().tolist()),
            "train_sample_count": len(train_ids),
            "val_sample_count": len(val_ids),
        }
        return train_ids, val_ids, split_summary

    if split_mode == "overfit_subject_run":
        selected = manifest_df.loc[
            (manifest_df["subject_id"] == overfit_subject_id)
            & (manifest_df["run_id"].astype(int) == int(overfit_run_id))
        ]
        if selected.empty:
            raise ValueError(f"No samples found for {overfit_subject_id} run {overfit_run_id}")
        ids = selected["sample_id"].astype(int).tolist()
        split_summary = {
            "split_mode": split_mode,
            "all_subject_count": len(subjects),
            "selected_subjects": [overfit_subject_id],
            "train_subjects": [overfit_subject_id],
            "val_subjects": [overfit_subject_id],
            "train_run_ids": [int(overfit_run_id)],
            "val_run_ids": [int(overfit_run_id)],
            "train_sample_count": len(ids),
            "val_sample_count": len(ids),
            "note": "Intentional train=val overfit sanity check for the cleaned temporal model path.",
        }
        return ids, ids, split_summary

    raise ValueError(f"Unsupported split_mode: {split_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short corrected temporal-clip baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-slugs", nargs="+", default=[f"thesis-batch-{i:02d}" for i in range(1, 8)])
    parser.add_argument("--max-subjects", type=int, default=8)
    parser.add_argument("--val-subject-count", type=int, default=2)
    parser.add_argument(
        "--split-mode",
        choices=["subject_holdout", "run_holdout", "overfit_subject_run"],
        default="subject_holdout",
    )
    parser.add_argument("--val-run-id", type=int, default=6)
    parser.add_argument("--overfit-subject-id", default="sub-01")
    parser.add_argument("--overfit-run-id", type=int, default=1)
    parser.add_argument("--run-name", default="corrected_clip_diagnostic")
    parser.add_argument("--out-dir", default="/kaggle/working/corrected_clip_baseline")
    args = parser.parse_args()

    loaded = load_config(args.config)
    cfg = loaded.data
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["training"]["deterministic"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("corrected_clip_baseline", out_dir / "run.log")

    roots = resolve_batch_roots(args.batch_slugs)
    manifest_df, qc = build_manifest(roots, CLASS_NAMES)
    index_path = out_dir / "index.parquet"
    save_manifest_parquet(manifest_df, index_path)
    write_json(qc, out_dir / "index_qc.json")

    train_ids, val_ids, split_summary = make_split(
        manifest_df,
        split_mode=args.split_mode,
        max_subjects=args.max_subjects,
        val_subject_count=args.val_subject_count,
        val_run_id=args.val_run_id,
        overfit_subject_id=args.overfit_subject_id,
        overfit_run_id=args.overfit_run_id,
    )
    write_json(split_summary, out_dir / "split_summary.json")
    logger.info("Diagnostic split: %s", split_summary)

    config_copy = out_dir / "config.yaml"
    shutil.copyfile(loaded.path, config_copy)

    summary = train_fold(
        config=cfg,
        manifest_df=manifest_df,
        train_sample_ids=train_ids,
        val_sample_ids=val_ids,
        run_dir=out_dir / "train_fold",
        run_name=args.run_name,
        fold_name="diagnostic_subject_holdout",
        config_path=str(config_copy),
        logger=logger,
    )
    write_json(summary, out_dir / "summary.json")

    decision = {
        "threshold_accuracy": 0.30,
        "threshold_macro_f1": 0.30,
        "passes_threshold": bool(
            float(summary["val_metrics"]["top1_accuracy"]) >= 0.30
            and float(summary["val_metrics"]["macro_f1"]) >= 0.30
        ),
        "summary": summary,
    }
    write_json(decision, out_dir / "decision.json")
    print("SUMMARY", json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
