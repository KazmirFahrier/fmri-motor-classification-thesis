#!/usr/bin/env python3
"""Generate the main protocol separated performance figure from frozen records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#17324D"
BLUE = "#2A6F97"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
ORANGE = "#E76F51"
GRID = "#D7E0E8"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def label_bars(axis, bars) -> None:
    for bar in bars:
        axis.text(
            bar.get_width() + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            ha="left",
            va="center",
            fontsize=9,
            color=INK,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closeout",
        type=Path,
        default=Path("experiments/confirmation/investigation_closeout.results.json"),
    )
    parser.add_argument(
        "--confirmation",
        type=Path,
        default=Path(
            "findings_2026-08-18_interpretation/experiments/q1_confirmation.results.json"
        ),
    )
    parser.add_argument(
        "--paired",
        type=Path,
        default=Path(
            "findings_2026-08-18_interpretation/experiments/"
            "frozen_vs_nested_native_smoothing.results.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("manuscript/figures")
    )
    args = parser.parse_args()

    closeout = load(args.closeout)
    confirmation = load(args.confirmation)
    paired = load(args.paired)
    frozen = closeout["frozen_results"]
    evidence = confirmation["completed_evidence"]
    comparisons = paired["comparisons"]

    independent_names = [
        "Legacy neural holdout",
        "Frozen temporal hierarchy",
        "Nested grid linear SVM",
        "Nested smoothing linear SVM",
    ]
    independent_values = [
        frozen["legacy_subjectwise_holdout"]["balanced_accuracy"],
        comparisons["independent|all62"]["reference_mean"],
        evidence["nested_spatial_grid"]["independent_balanced_accuracy"],
        comparisons["independent|all62"]["comparison_mean"],
    ]

    constrained_names = [
        "Mean window hierarchy",
        "Repetition consistency",
        "Smoothing plus assignment",
    ]
    constrained_values = [
        frozen["conservative_mean_hierarchy"]["balanced_accuracy"],
        frozen["complete_balanced_run"]["subject_weighted_accuracy"],
        comparisons["balanced|all62"]["comparison_mean"],
    ]

    contrast_keys = [
        "independent|all62",
        "independent|qc60",
        "balanced|all62",
        "balanced|qc60",
    ]
    contrast_names = [
        "Independent, all 62",
        "Independent, QC60",
        "Assignment, all 62",
        "Assignment, QC60",
    ]
    differences = np.asarray(
        [comparisons[key]["comparison_minus_reference"] for key in contrast_keys]
    )
    intervals = np.asarray([comparisons[key]["ci95"] for key in contrast_keys])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
        }
    )
    figure = plt.figure(figsize=(13.2, 4.8), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.25])

    first = figure.add_subplot(grid[0, 0])
    bars = first.barh(
        np.arange(len(independent_values)),
        independent_values,
        color=[ORANGE, GOLD, BLUE, TEAL],
        height=0.68,
    )
    first.axvline(0.25, color=INK, linestyle="--", linewidth=1, label="chance")
    first.set_xlim(0, 1.0)
    first.set_xlabel("Balanced accuracy")
    first.set_yticks(np.arange(len(independent_names)), independent_names)
    first.invert_yaxis()
    first.set_title("A  Independent subject decoding", loc="left", fontweight="bold")
    first.grid(axis="x", color=GRID, linewidth=0.8)
    first.set_axisbelow(True)
    label_bars(first, bars)

    second = figure.add_subplot(grid[0, 1])
    bars = second.barh(
        np.arange(len(constrained_values)),
        constrained_values,
        color=[GOLD, BLUE, TEAL],
        height=0.62,
    )
    second.axvline(0.25, color=INK, linestyle="--", linewidth=1)
    second.set_xlim(0, 1.0)
    second.set_xlabel("Balanced accuracy")
    second.set_yticks(np.arange(len(constrained_names)), constrained_names)
    second.invert_yaxis()
    second.set_title("B  Design constrained decoding", loc="left", fontweight="bold")
    second.grid(axis="x", color=GRID, linewidth=0.8)
    second.set_axisbelow(True)
    label_bars(second, bars)

    third = figure.add_subplot(grid[0, 2])
    positions = np.arange(len(contrast_names))[::-1]
    colors = [TEAL, TEAL, BLUE, BLUE]
    for position, value, interval, color in zip(
        positions, differences, intervals, colors, strict=True
    ):
        third.errorbar(
            value,
            position,
            xerr=[[value - interval[0]], [interval[1] - value]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=7,
            capsize=4,
            linewidth=2,
        )
    third.axvline(0, color=INK, linestyle="--", linewidth=1)
    third.set_yticks(positions, contrast_names)
    third.set_xlabel("Nested smoothing minus frozen hierarchy")
    third.set_title("C  Paired subject differences", loc="left", fontweight="bold")
    third.grid(axis="x", color=GRID, linewidth=0.8)
    third.set_axisbelow(True)
    third.set_xlim(-0.01, 0.06)

    figure.suptitle(
        "Prediction protocol changes both performance and interpretation",
        fontsize=15,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "protocol_separated_performance.png", dpi=300)
    figure.savefig(args.output_dir / "protocol_separated_performance.pdf")
    plt.close(figure)
    print(f"wrote protocol figure to {args.output_dir}")


if __name__ == "__main__":
    main()
