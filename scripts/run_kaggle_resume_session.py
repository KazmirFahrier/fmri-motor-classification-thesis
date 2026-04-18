#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fmri_pipeline.config import load_config
from fmri_pipeline.data.manifest import build_manifest, read_manifest, save_manifest_parquet
from fmri_pipeline.data.splits import create_subjectwise_splits, make_internal_train_val_subject_split, resolve_sample_ids
from fmri_pipeline.report import aggregate_cv, export_publication_bundle, summaries_to_dataframe
from fmri_pipeline.training import evaluate_checkpoint, train_fold
from fmri_pipeline.utils.io import read_json, write_json
from fmri_pipeline.utils.log_utils import setup_logger
from fmri_pipeline.utils.seed import seed_everything


DEFAULT_BATCH_SLUGS = [
    "thesis-batch-01",
    "thesis-batch-02",
    "thesis-batch-03",
    "thesis-batch-04",
    "thesis-batch-05",
    "thesis-batch-06",
    "thesis-batch-07",
]

DEFAULT_RESUME_SLUGS = [
    "thesis-7batch-artifacts",
    "thesis-7batch-resume",
]


def _parse_class_names(raw: str) -> List[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if not names:
        raise ValueError("Expected at least one class name")
    return names


def _find_mounted_dataset_dir(slug: str) -> Path | None:
    direct = Path("/kaggle/input") / slug
    if direct.exists():
        return direct

    datasets_root = Path("/kaggle/input/datasets")
    if datasets_root.exists():
        matches = sorted(
            p for p in datasets_root.rglob(slug) if p.is_dir()
        )
        if matches:
            return matches[0]

    generic_matches = sorted(
        p for p in Path("/kaggle/input").rglob(slug) if p.is_dir()
    )
    if generic_matches:
        return generic_matches[0]

    return None


def _discover_batch_roots(batch_slugs: List[str]) -> List[Path]:
    roots: List[Path] = []
    missing: List[str] = []
    for slug in batch_slugs:
        candidate = _find_mounted_dataset_dir(slug)
        if candidate is None:
            missing.append(slug)
        else:
            roots.append(candidate)

    if missing:
        raise FileNotFoundError(f"Missing Kaggle dataset mounts: {missing}")
    return roots


def _sync_resume_tree(resume_root: Path, work_root: Path, logger) -> None:
    if not resume_root.exists():
        logger.info("Resume root does not exist, starting fresh: %s", resume_root)
        return

    logger.info("Syncing prior artifacts from %s into %s", resume_root, work_root)
    shutil.copytree(resume_root, work_root, dirs_exist_ok=True)


def _auto_resume_root(run_name: str) -> Path | None:
    for slug in DEFAULT_RESUME_SLUGS:
        candidate = _find_mounted_dataset_dir(slug)
        if candidate is None:
            continue
        nested = candidate / run_name
        if nested.exists():
            return nested
        return candidate
    return None


def _stage_cv_summary_path(run_root: Path, fold_idx: int) -> Path:
    return run_root / "cv" / f"fold_{fold_idx:02d}" / "summary.json"


def _stage_cv_checkpoint_path(run_root: Path, fold_idx: int) -> Path:
    return run_root / "cv" / f"fold_{fold_idx:02d}" / "last_checkpoint.pt"


def _holdout_summary_path(run_root: Path) -> Path:
    return run_root / "holdout_model" / "summary.json"


def _holdout_checkpoint_path(run_root: Path) -> Path:
    return run_root / "holdout_model" / "last_checkpoint.pt"


def _holdout_eval_path(run_root: Path) -> Path:
    return run_root / "holdout_eval" / "holdout_summary.json"


def _publication_report_path(run_root: Path) -> Path:
    return run_root / "publication_bundle" / "publication_report.md"


def _next_stage(run_root: Path, cv_folds: int) -> Dict[str, Any]:
    for fold_idx in range(cv_folds):
        if not _stage_cv_summary_path(run_root, fold_idx).exists():
            return {"kind": "cv", "fold": fold_idx}

    if not _holdout_summary_path(run_root).exists():
        return {"kind": "holdout_model"}

    if not _holdout_eval_path(run_root).exists() or not _publication_report_path(run_root).exists():
        return {"kind": "holdout_eval"}

    return {"kind": "complete"}


def _write_partial_cv_summaries(run_root: Path, cv_folds: int) -> None:
    summaries = []
    for fold_idx in range(cv_folds):
        summary_path = _stage_cv_summary_path(run_root, fold_idx)
        if summary_path.exists():
            summaries.append(read_json(summary_path))

    if summaries:
        write_json(summaries, run_root / "cv_fold_summaries.json")


def _export_if_ready(run_root: Path) -> None:
    fold_summaries_path = run_root / "cv_fold_summaries.json"
    holdout_summary_path = _holdout_eval_path(run_root)
    if not fold_summaries_path.exists() or not holdout_summary_path.exists():
        return

    fold_summaries = read_json(fold_summaries_path)
    fold_df = summaries_to_dataframe(fold_summaries)
    cv_stats = aggregate_cv(fold_df)
    export_publication_bundle(
        fold_df=fold_df,
        cv_stats_df=cv_stats,
        holdout_summary=read_json(holdout_summary_path),
        out_dir=run_root / "publication_bundle",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one resumable Kaggle training session across all 7 thesis batches."
    )
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--run-name", required=True, help="Experiment run name.")
    parser.add_argument(
        "--class-names",
        default="Left leg movements,Right leg movements,Forearm movements,Upper arm movements",
        help="Comma-separated class names.",
    )
    parser.add_argument(
        "--batch-slugs",
        nargs="+",
        default=DEFAULT_BATCH_SLUGS,
        help="Kaggle dataset slugs mounted under /kaggle/input.",
    )
    parser.add_argument(
        "--work-root",
        default="/kaggle/working/thesis_session",
        help="Writable session artifact root.",
    )
    parser.add_argument(
        "--resume-root",
        default=None,
        help="Optional mounted artifact root from a prior Kaggle session.",
    )
    parser.add_argument(
        "--holdout-val-fraction",
        type=float,
        default=0.1,
        help="Internal validation fraction for final holdout model training.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional truncation for smoke testing.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional truncation for smoke testing.",
    )
    args = parser.parse_args()

    loaded = load_config(args.config)
    cfg = loaded.data
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["training"]["deterministic"]))

    class_names = _parse_class_names(args.class_names)
    work_root = Path(args.work_root).expanduser().resolve()
    run_root = work_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("kaggle_resume_session", run_root / "session.log")

    resume_root: Path | None = None
    if args.resume_root:
        resume_root = Path(args.resume_root).expanduser().resolve()
        if (resume_root / args.run_name).exists():
            resume_root = resume_root / args.run_name
    else:
        resume_root = _auto_resume_root(args.run_name)

    if resume_root is not None:
        _sync_resume_tree(resume_root, run_root, logger)

    metadata_dir = run_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    index_path = metadata_dir / "index.parquet"
    manifest_qc_path = metadata_dir / "index_qc.json"
    splits_path = metadata_dir / "splits_subjectwise.json"

    batch_roots = _discover_batch_roots(list(args.batch_slugs))
    logger.info("Using batch roots: %s", [str(p) for p in batch_roots])

    if not index_path.exists():
        manifest_df, qc = build_manifest(batch_roots, class_names)
        save_manifest_parquet(manifest_df, index_path)
        write_json(qc, manifest_qc_path)
        logger.info(
            "Built manifest: samples=%d subjects=%d runs=%d",
            int(qc["num_samples"]),
            int(qc["num_subjects"]),
            int(qc["num_runs"]),
        )

    manifest_df = read_manifest(index_path)

    if not splits_path.exists():
        split_data = create_subjectwise_splits(
            manifest_df,
            seed=int(cfg["seed"]),
            holdout_subject_count=8,
            cv_folds=5,
        )
        split_data["class_names"] = sorted(manifest_df["class_name"].unique().tolist())
        write_json(split_data, splits_path)

    split_data = read_json(splits_path)
    stage = _next_stage(run_root, int(split_data["cv_folds"]))
    logger.info("Selected stage: %s", stage)

    summary: Dict[str, Any] = {"stage": stage, "run_root": str(run_root)}

    if stage["kind"] == "cv":
        fold_idx = int(stage["fold"])
        fold = split_data["folds"][fold_idx]
        fold_dir = run_root / "cv" / f"fold_{fold_idx:02d}"
        resume_path = _stage_cv_checkpoint_path(run_root, fold_idx)
        summary["result"] = train_fold(
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
            resume_from=resume_path if resume_path.exists() else None,
        )
        _write_partial_cv_summaries(run_root, int(split_data["cv_folds"]))

    elif stage["kind"] == "holdout_model":
        non_holdout_subjects = split_data["cv_subjects"]
        train_subj, val_subj = make_internal_train_val_subject_split(
            subjects=non_holdout_subjects,
            val_fraction=float(args.holdout_val_fraction),
            seed=int(cfg["seed"]) + 999,
        )
        holdout_model_dir = run_root / "holdout_model"
        resume_path = _holdout_checkpoint_path(run_root)
        summary["result"] = train_fold(
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
            resume_from=resume_path if resume_path.exists() else None,
        )

    elif stage["kind"] == "holdout_eval":
        holdout_train_summary = read_json(_holdout_summary_path(run_root))
        holdout_eval = evaluate_checkpoint(
            checkpoint_path=holdout_train_summary["checkpoint"],
            manifest_df=manifest_df,
            sample_ids=split_data["holdout_sample_ids"],
            out_dir=run_root / "holdout_eval",
            split_name="holdout",
        )
        write_json(holdout_eval, _holdout_eval_path(run_root))
        _write_partial_cv_summaries(run_root, int(split_data["cv_folds"]))
        _export_if_ready(run_root)
        summary["result"] = holdout_eval

    elif stage["kind"] == "complete":
        logger.info("Run already complete: %s", run_root)
        summary["result"] = {"status": "complete"}

    else:
        raise ValueError(f"Unsupported stage kind: {stage['kind']}")

    next_stage = _next_stage(run_root, int(split_data["cv_folds"]))
    summary["next_stage"] = next_stage
    write_json(summary, run_root / "session_state.json")
    logger.info("Session complete. Next stage: %s", next_stage)


if __name__ == "__main__":
    main()
