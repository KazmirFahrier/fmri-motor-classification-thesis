#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict

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
from fmri_pipeline.training.pipeline import train_fold
from fmri_pipeline.utils.io import read_json, write_json
from fmri_pipeline.utils.log_utils import setup_logger
from fmri_pipeline.utils.seed import seed_everything


def _get_fold(split_data: Dict[str, Any], fold_idx: int) -> Dict[str, Any]:
    for fold in split_data["folds"]:
        if int(fold["fold"]) == int(fold_idx):
            return fold
    raise ValueError(f"Fold {fold_idx} not found in splits file")


def _apply_optuna_params(cfg: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    if "lr" in params:
        out["optimizer"]["lr"] = float(params["lr"])
    if "weight_decay" in params:
        out["optimizer"]["weight_decay"] = float(params["weight_decay"])
    if "dropout" in params:
        out["model"]["dropout"] = float(params["dropout"])
    if "clip_length" in params:
        out["loader"]["clip_length"] = int(params["clip_length"])
    if "label_smoothing" in params:
        out["training"]["label_smoothing"] = float(params["label_smoothing"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one fold with strict subject-wise split.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--index", required=True, help="Parquet manifest path.")
    parser.add_argument("--splits", required=True, help="Splits JSON path.")
    parser.add_argument("--fold", type=int, required=True, help="Fold index to train.")
    parser.add_argument("--run-name", required=True, help="Run name prefix.")
    parser.add_argument("--out-dir", required=True, help="Output artifact directory.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional truncation for smoke tests.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional truncation for smoke tests.")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Optional Optuna trial count for fold-local tuning.")
    parser.add_argument("--resume", default=None, help="Optional path to a saved last_checkpoint.pt file.")
    args = parser.parse_args()

    loaded = load_config(args.config)
    cfg = loaded.data

    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["training"]["deterministic"]))

    manifest_df = read_manifest(args.index)
    split_data = read_json(args.splits)
    fold_spec = _get_fold(split_data, args.fold)

    run_root = Path(args.out_dir).expanduser().resolve() / args.run_name / f"fold_{args.fold:02d}"
    run_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train", run_root / "train.log")

    final_cfg = copy.deepcopy(cfg)

    if int(args.optuna_trials) > 0:
        try:
            import optuna
        except Exception as exc:
            raise SystemExit(f"Optuna requested but unavailable: {exc}") from exc

        direction = "maximize" if cfg["training"]["monitor_mode"] == "max" else "minimize"
        sweep_epochs = int(cfg["training"].get("sweep_epochs", max(5, int(cfg["training"]["epochs"]) // 4)))

        logger.info("Starting Optuna sweep: trials=%d direction=%s", args.optuna_trials, direction)

        def objective(trial):
            trial_params = {
                "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
            }
            if cfg["task"] == "clip":
                trial_params["clip_length"] = trial.suggest_categorical(
                    "clip_length", [int(x) for x in cfg["loader"]["clip_lengths"]]
                )

            trial_cfg = _apply_optuna_params(cfg, trial_params)
            trial_cfg["training"]["epochs"] = sweep_epochs

            trial_dir = run_root / "optuna_trials" / f"trial_{trial.number:03d}"
            summary = train_fold(
                config=trial_cfg,
                manifest_df=manifest_df,
                train_sample_ids=fold_spec["train_sample_ids"],
                val_sample_ids=fold_spec["val_sample_ids"],
                run_dir=trial_dir,
                run_name=f"{args.run_name}_trial_{trial.number:03d}",
                fold_name=f"fold_{args.fold:02d}_trial_{trial.number:03d}",
                config_path=str(loaded.path),
                logger=logger,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                resume_from=None,
            )
            return float(summary["best_monitor_value"])

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=int(args.optuna_trials))

        best_params = study.best_params
        write_json(
            {
                "direction": direction,
                "best_value": float(study.best_value),
                "best_params": best_params,
                "n_trials": int(args.optuna_trials),
            },
            run_root / "optuna_best.json",
        )

        final_cfg = _apply_optuna_params(cfg, best_params)
        logger.info("Optuna best params: %s", best_params)

    summary = train_fold(
        config=final_cfg,
        manifest_df=manifest_df,
        train_sample_ids=fold_spec["train_sample_ids"],
        val_sample_ids=fold_spec["val_sample_ids"],
        run_dir=run_root / "final",
        run_name=args.run_name,
        fold_name=f"fold_{args.fold:02d}",
        config_path=str(loaded.path),
        logger=logger,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        resume_from=args.resume,
    )

    write_json(summary, run_root / "final_summary.json")
    logger.info("Completed fold training. Summary: %s", summary)


if __name__ == "__main__":
    main()
