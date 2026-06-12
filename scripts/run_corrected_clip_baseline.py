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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short corrected temporal-clip baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-slugs", nargs="+", default=[f"thesis-batch-{i:02d}" for i in range(1, 8)])
    parser.add_argument("--max-subjects", type=int, default=8)
    parser.add_argument("--val-subject-count", type=int, default=2)
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

    subjects = sorted(manifest_df["subject_id"].unique().tolist())
    train_subjects, val_subjects = choose_subjects(subjects, args.max_subjects, args.val_subject_count)
    train_ids = manifest_df.loc[manifest_df["subject_id"].isin(train_subjects), "sample_id"].astype(int).tolist()
    val_ids = manifest_df.loc[manifest_df["subject_id"].isin(val_subjects), "sample_id"].astype(int).tolist()
    split_summary = {
        "all_subject_count": len(subjects),
        "selected_subjects": train_subjects + val_subjects,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "train_sample_count": len(train_ids),
        "val_sample_count": len(val_ids),
    }
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
