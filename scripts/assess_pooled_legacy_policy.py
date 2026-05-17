#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess whether the full-dataset pooled legacy baseline should continue."
    )
    parser.add_argument("metrics_jsonl", type=Path, help="Path to train/metrics.jsonl.")
    parser.add_argument("--best-epoch", type=int, default=None, help="Known best epoch from checkpoint/logs.")
    parser.add_argument("--patience", type=int, default=25, help="Early-stopping patience in epochs.")
    parser.add_argument("--extend-accuracy", type=float, default=0.30, help="Validation accuracy needed to extend.")
    parser.add_argument("--extend-macro-f1", type=float, default=0.30, help="Validation macro F1 needed to extend.")
    args = parser.parse_args()

    rows = _load_rows(args.metrics_jsonl)
    if not rows:
        raise SystemExit("No metric rows found.")

    latest = rows[-1]
    best_visible = max(rows, key=lambda row: row["val"]["top1_accuracy"])
    best_epoch = args.best_epoch if args.best_epoch is not None else int(best_visible["epoch"])
    latest_epoch = int(latest["epoch"])
    epochs_since_best = max(0, latest_epoch - best_epoch)

    latest_val = latest["val"]
    should_extend = (
        float(latest_val["top1_accuracy"]) >= args.extend_accuracy
        and float(latest_val["macro_f1"]) >= args.extend_macro_f1
    )
    should_stop = epochs_since_best >= args.patience and not should_extend

    decision = "extend" if should_extend else "stop" if should_stop else "continue_short_baseline"
    remaining_to_patience = max(0, args.patience - epochs_since_best)

    print(
        json.dumps(
            {
                "decision": decision,
                "latest_epoch": latest_epoch,
                "best_epoch_for_patience": best_epoch,
                "epochs_since_best": epochs_since_best,
                "remaining_epochs_to_patience": remaining_to_patience,
                "latest_validation": {
                    "accuracy": latest_val["top1_accuracy"],
                    "balanced_accuracy": latest_val["balanced_accuracy"],
                    "macro_f1": latest_val["macro_f1"],
                    "mcc": latest_val["mcc"],
                    "roc_auc": latest_val.get("roc_auc_ovr_macro"),
                    "pr_auc": latest_val.get("pr_auc_macro"),
                },
                "extend_threshold": {
                    "accuracy": args.extend_accuracy,
                    "macro_f1": args.extend_macro_f1,
                },
                "stop_rule": f"stop if epochs_since_best >= {args.patience} unless extension threshold is met",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
