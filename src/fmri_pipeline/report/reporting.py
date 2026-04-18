from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fmri_pipeline.utils.io import write_json


KEY_METRICS = [
    "top1_accuracy",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "roc_auc_ovr_macro",
    "pr_auc_macro",
]


def summaries_to_dataframe(summaries: Sequence[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for s in summaries:
        row = {
            "fold_name": str(s["fold_name"]),
            "best_epoch": int(s["best_epoch"]),
            "best_monitor_value": float(s["best_monitor_value"]),
        }
        metrics = s["val_metrics"]
        for key in KEY_METRICS:
            row[key] = float(metrics.get(key, float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_cv(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in KEY_METRICS:
        values = df[key].astype(float).to_numpy()
        rows.append(
            {
                "metric": key,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
        )
    return pd.DataFrame(rows)


def save_cv_plots(cv_stats_df: pd.DataFrame, out_pdf: str | Path) -> None:
    out_path = Path(out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(cv_stats_df))
    means = cv_stats_df["mean"].to_numpy()
    stds = cv_stats_df["std"].to_numpy()

    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, cv_stats_df["metric"], rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Cross-Validation Metrics (mean ± std)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def export_publication_bundle(
    *,
    fold_df: pd.DataFrame,
    cv_stats_df: pd.DataFrame,
    holdout_summary: Dict[str, object] | None,
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fold_df.to_csv(out / "cv_fold_metrics.csv", index=False)
    cv_stats_df.to_csv(out / "cv_summary_metrics.csv", index=False)
    write_json(fold_df.to_dict(orient="records"), out / "cv_fold_metrics.json")
    write_json(cv_stats_df.to_dict(orient="records"), out / "cv_summary_metrics.json")

    if holdout_summary is not None:
        write_json(holdout_summary, out / "holdout_summary.json")

    save_cv_plots(cv_stats_df, out / "cv_summary_plot.pdf")

    lines = [
        "# Publication Metrics Summary",
        "",
        "## Cross-Validation (Subject-wise)",
        "",
    ]
    for row in cv_stats_df.to_dict(orient="records"):
        lines.append(
            f"- {row['metric']}: {row['mean']:.4f} ± {row['std']:.4f} (min={row['min']:.4f}, max={row['max']:.4f})"
        )

    if holdout_summary is not None:
        lines += ["", "## Holdout", ""]
        metrics = holdout_summary.get("metrics", {})
        for key in KEY_METRICS:
            if key in metrics:
                lines.append(f"- {key}: {float(metrics[key]):.4f}")

        bootstrap = holdout_summary.get("bootstrap_ci", {})
        if bootstrap:
            lines += ["", "### Holdout Bootstrap CI"]
            for key, ci in bootstrap.items():
                lines.append(
                    f"- {key}: mean={float(ci['mean']):.4f}, 95% CI [{float(ci['low']):.4f}, {float(ci['high']):.4f}]"
                )

    (out / "publication_report.md").write_text("\n".join(lines), encoding="utf-8")
