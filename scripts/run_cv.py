#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Notebook execution fallback (__file__ is undefined in cells).
    ROOT = Path.cwd().resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fmri_pipeline.config import load_config
from fmri_pipeline.data.manifest import read_manifest
from fmri_pipeline.data.splits import (
    make_internal_train_val_subject_split,
    resolve_sample_ids,
)
from fmri_pipeline.report import aggregate_cv, export_publication_bundle, summaries_to_dataframe
from fmri_pipeline.training import evaluate_checkpoint, train_fold
from fmri_pipeline.utils.io import read_json, write_json
from fmri_pipeline.utils.log_utils import setup_logger
from fmri_pipeline.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run subject-wise 5-fold CV and final holdout evaluation.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--index", required=True, help="Parquet manifest path.")
    parser.add_argument("--splits", required=True, help="Splits JSON path.")
    parser.add_argument("--out-dir", required=True, help="Output root directory.")
    parser.add_argument("--run-name", required=True, help="Run name.")
    parser.add_argument("--holdout-val-fraction", type=float, default=0.1, help="Internal validation fraction when fitting final holdout model.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional truncation for smoke runs.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional truncation for smoke runs.")
    args = parser.parse_args()

    loaded = load_config(args.config)
    cfg = loaded.data

    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["training"]["deterministic"]))

    manifest_df = read_manifest(args.index)
    split_data = read_json(args.splits)

    run_root = Path(args.out_dir).expanduser().resolve() / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("run_cv", run_root / "run_cv.log")

    fold_summaries = []
    for fold in split_data["folds"]:
        fold_idx = int(fold["fold"])
        fold_dir = run_root / "cv" / f"fold_{fold_idx:02d}"
        logger.info("Training CV fold %d", fold_idx)
        summary = train_fold(
            config=cfg,
            manifest_df=manifest_df,
            train_sample_ids=fold["train_sample_ids"],
            val_sample_ids=fold["val_sample_ids"],
            run_dir=fold_dir,
            run_name=args.run_name,
            fold_name=f"cv_fold_{fold_idx:02d}",
            config_path=str(loaded.path),
            logger=logger,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        )
        fold_summaries.append(summary)

    write_json(fold_summaries, run_root / "cv_fold_summaries.json")

    fold_df = summaries_to_dataframe(fold_summaries)
    cv_stats = aggregate_cv(fold_df)

    # Train final model on non-holdout subjects and evaluate on holdout once.
    non_holdout_subjects = split_data["cv_subjects"]
    train_subj, val_subj = make_internal_train_val_subject_split(
        subjects=non_holdout_subjects,
        val_fraction=float(args.holdout_val_fraction),
        seed=int(cfg["seed"]) + 999,
    )

    holdout_model_dir = run_root / "holdout_model"
    holdout_train_summary = train_fold(
        config=cfg,
        manifest_df=manifest_df,
        train_sample_ids=resolve_sample_ids(manifest_df, train_subj),
        val_sample_ids=resolve_sample_ids(manifest_df, val_subj),
        run_dir=holdout_model_dir,
        run_name=args.run_name,
        fold_name="holdout_model_training",
        config_path=str(loaded.path),
        logger=logger,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    write_json(holdout_train_summary, run_root / "holdout_model" / "training_summary.json")

    holdout_eval = evaluate_checkpoint(
        checkpoint_path=holdout_train_summary["checkpoint"],
        manifest_df=manifest_df,
        sample_ids=split_data["holdout_sample_ids"],
        out_dir=run_root / "holdout_eval",
        split_name="holdout",
    )
    write_json(holdout_eval, run_root / "holdout_eval" / "holdout_summary.json")

    report_dir = run_root / "publication_bundle"
    export_publication_bundle(
        fold_df=fold_df,
        cv_stats_df=cv_stats,
        holdout_summary=holdout_eval,
        out_dir=report_dir,
    )

    logger.info("CV + holdout complete. Report bundle: %s", report_dir)


if __name__ == "__main__":
    main()
