#!/usr/bin/env python3
"""Conventional MVPA comparators under the frozen closeout protocol.

The covariance hierarchy has only ever been compared against its own variants and
a legacy neural recipe. This script supplies the missing reference point: standard
linear decoders -- L2-regularised multinomial logistic regression, a linear SVM,
and the classic correlation/nearest-centroid classifier -- evaluated on the exact
same events, the exact same representation, and the exact same 30 outer subject
splits with nested inner-subject hyperparameter selection.

Nothing here is allowed to see held-out subjects. Feature standardisation and the
regularisation constant are fitted on outer-training subjects only, and the inner
selection folds partition those training subjects further.

Both prediction rules from the frozen protocol are reported:

``independent``
    Ordinary argmax over class scores. This is the comparable number.
``balanced``
    Design-constrained assignment forcing exactly two events per class per run.
    Reported so the hierarchy and the standard baselines are compared under the
    same rule, never to imply the baselines are ordinary inductive predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# This module lives in a findings folder rather than the repository's own
# scripts/ directory, so the shared protocol primitives are resolved explicitly.
REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from run_balanced_event_assignment import apply_balanced_assignment, metrics
from run_detrended_pair_feature_selection import (
    inner_splits,
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence


DEFAULT_C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)
MODELS = ("logistic_l2", "linear_svm", "correlation_centroid")


def standardize(
    x: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-feature mean/scale estimated on training rows only."""
    mean = x[train_idx].mean(axis=0, dtype=np.float64)
    scale = x[train_idx].std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def dual_basis(
    x_train: np.ndarray,
    x_val: np.ndarray,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Project onto the training row space. Exact for L2-penalised linear models.

    With 13824 features and roughly 2480 training events, direct optimisation on
    the full feature matrix does not converge in usable time on real fMRI data,
    which is highly correlated and near-separable.

    For any objective of the form ``L(Xw) + lambda * ||w||^2`` the optimum lies in
    the row space of ``X``: writing ``w = V a + w_perp`` with ``V`` an orthonormal
    basis of that row space gives ``Xw = XV a`` while ``||w||^2 = ||a||^2 +
    ||w_perp||^2``, so the optimum sets ``w_perp = 0``. Fitting on ``Z = XV``
    therefore yields identical predictions, since ``Z_val a = X_val V a = X_val w``.
    This is an exact reparameterisation, not a dimensionality-reduction
    approximation, and it holds for L2-penalised logistic regression and for
    LinearSVC with ``dual=True``.

    The basis is built from training rows only, so held-out subjects never
    influence it. Inner selection folds are subsets of the outer training rows,
    so their row spaces are contained in this basis and the reparameterisation
    stays exact for them too.

    ``V`` is never formed explicitly. Using the Gram matrix ``G = X X^T = U S^2 U^T``
    we get ``Z_train = U S`` and ``Z_val = (X_val X_train^T) U / S``.
    """
    gram = (x_train @ x_train.T).astype(np.float64)
    cross = (x_val @ x_train.T).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = eigenvalues[::-1]
    vectors = eigenvectors[:, ::-1]
    keep = order > max(tol, float(order[0]) * tol)
    singular = np.sqrt(order[keep])
    vectors = vectors[:, keep]
    z_train = vectors * singular
    z_val = (cross @ vectors) / singular
    return z_train.astype(np.float64), z_val.astype(np.float64)


def subject_zscore(mean_x: np.ndarray, records: list[dict]) -> np.ndarray:
    """Diagonal second-order alignment: per-subject per-feature centering and scaling.

    The project has established that aligning the *first* moment per subject-run is
    worth roughly +0.52 independent accuracy. This aligns the *second* moment as
    well, one feature at a time. It uses no labels, and it is diagonal so it has no
    rank problem despite each subject contributing only 48 events.
    """
    subjects = np.asarray([str(record["subject_id"]) for record in records])
    out = mean_x.copy()
    for subject in sorted(set(subjects.tolist())):
        mask = subjects == subject
        block = out[mask]
        block -= block.mean(axis=0)
        scale = block.std(axis=0)
        scale[scale < 1e-8] = 1.0
        out[mask] = block / scale
    return out


def coral_whiten(
    mean_x: np.ndarray,
    records: list[dict],
    component_count: int,
    shrinkage: float,
    whiten: bool = True,
) -> tuple[np.ndarray, dict]:
    """Full second-order alignment (CORAL) inside a shared low-rank subspace.

    Each subject is whitened by their own covariance, estimated in a common
    subspace. For a *linear* decoder the subsequent recolouring to a shared target
    covariance is absorbed into the weights, so it is omitted: per-subject
    whitening is the operative step.

    The subspace is a label-free PCA of all events. That is transductive in exactly
    the same sense as the subject-run centering the frozen pipeline already uses --
    it consults held-out subjects' features but never their labels.

    Each subject supplies only 48 events, so the covariance is regularised toward
    its diagonal by ``shrinkage`` and ``component_count`` must stay well below 48.
    """
    centered = mean_x - mean_x.mean(axis=0, keepdims=True)
    # Economy SVD via the Gram matrix; events are far fewer than features.
    gram = (centered @ centered.T).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:component_count]
    singular = np.sqrt(np.maximum(eigenvalues[order], 1e-12))
    scores = eigenvectors[:, order] * singular

    subjects = np.asarray([str(record["subject_id"]) for record in records])
    if not whiten:
        # Control condition: the same subspace and the same per-subject centering,
        # but no covariance whitening. Any difference between this and the whitened
        # version is attributable to the alignment rather than to the dimensionality
        # reduction the subspace imposes.
        out = np.zeros_like(scores)
        for subject in sorted(set(subjects.tolist())):
            mask = subjects == subject
            out[mask] = scores[mask] - scores[mask].mean(axis=0)
        return out.astype(np.float32), {
            "component_count": int(component_count),
            "whiten": False,
            "retained_variance_fraction": float(
                np.sum(eigenvalues[order]) / max(np.sum(np.maximum(eigenvalues, 0)), 1e-12)
            ),
        }
    out = np.zeros_like(scores)
    for subject in sorted(set(subjects.tolist())):
        mask = subjects == subject
        block = scores[mask]
        block = block - block.mean(axis=0)
        covariance = np.cov(block, rowvar=False)
        covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(
            np.diag(covariance)
        )
        covariance += 1e-8 * np.eye(covariance.shape[0])
        values, vectors = np.linalg.eigh(covariance)
        whitener = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, 1e-12))) @ vectors.T
        out[mask] = block @ whitener
    info = {
        "component_count": int(component_count),
        "shrinkage": float(shrinkage),
        "events_per_subject": int(np.sum(subjects == subjects[0])),
        "retained_variance_fraction": float(
            np.sum(eigenvalues[order]) / max(np.sum(np.maximum(eigenvalues, 0)), 1e-12)
        ),
    }
    return out.astype(np.float32), info


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return np.nan_to_num(x / denom, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def correlation_centroid_scores(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    class_count: int,
) -> np.ndarray:
    """Haxby-style correlation classifier against per-class training centroids."""
    train_normed = l2_normalize(x[train_idx].astype(np.float32))
    centroids = np.zeros((class_count, x.shape[1]), dtype=np.float32)
    for class_idx in range(class_count):
        mask = y[train_idx] == class_idx
        if not np.any(mask):
            continue
        centroids[class_idx] = train_normed[mask].mean(axis=0)
    centroids = l2_normalize(centroids)
    return l2_normalize(x[val_idx].astype(np.float32)) @ centroids.T


def fit_projected(
    model_name: str,
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    kernel_train: np.ndarray,
    kernel_val: np.ndarray,
    c_value: float,
    seed: int,
) -> np.ndarray:
    """Fit a standard linear decoder and return validation class scores.

    ``linear_svm`` uses ``SVC`` with a precomputed linear kernel, which is the
    linear SVM conventionally used for neuroimaging MVPA. ``logistic_l2`` is
    fitted in the exact dual basis. Both are equivalent to fitting on all 13824
    standardised features.
    """
    if model_name == "logistic_l2":
        model = LogisticRegression(
            C=c_value,
            max_iter=5000,
            tol=1e-5,
            random_state=seed,
        )
        model.fit(z_train, y_train)
        return model.decision_function(z_val).astype(np.float64)

    if model_name == "linear_svm":
        model = SVC(
            C=c_value,
            kernel="precomputed",
            decision_function_shape="ovr",
            random_state=seed,
        )
        model.fit(kernel_train, y_train)
        return model.decision_function(kernel_val).astype(np.float64)

    raise ValueError(f"Unknown model {model_name}.")


def subject_metrics_for(
    y: np.ndarray,
    prediction: np.ndarray,
    val_idx: np.ndarray,
    records: list[dict],
) -> dict[str, dict]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for local_pos, record_idx in enumerate(val_idx):
        grouped[str(records[int(record_idx)]["subject_id"])].append(local_pos)
    return {
        subject: metrics(y[val_idx][positions], prediction[positions])
        for subject, positions in sorted(grouped.items())
    }


def evaluate_split(
    scores: np.ndarray,
    y: np.ndarray,
    val_idx: np.ndarray,
    records: list[dict],
) -> dict[str, dict]:
    predictions = {
        "independent": scores.argmax(axis=1).astype(np.int64),
        "balanced": apply_balanced_assignment(scores, val_idx, records),
    }
    return {
        rule: {
            "metrics": metrics(y[val_idx], prediction),
            "subject_metrics": subject_metrics_for(y, prediction, val_idx, records),
        }
        for rule, prediction in predictions.items()
    }


def select_c(
    model_name: str,
    z_train: np.ndarray,
    kernel_train: np.ndarray,
    y: np.ndarray,
    inner_local: list[dict],
    c_grid: tuple[float, ...],
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Choose C on inner balanced accuracy, matching the frozen selection metric.

    Inner folds are subsets of the outer training rows, so they index directly
    into the already-projected training block.
    """
    inner_means: dict[str, float] = {}
    for c_value in c_grid:
        fold_values = []
        for inner_split in inner_local:
            train_pos = inner_split["train_pos"]
            val_pos = inner_split["val_pos"]
            scores = fit_projected(
                model_name,
                z_train[train_pos],
                y[inner_split["train_idx"]],
                z_train[val_pos],
                kernel_train[np.ix_(train_pos, train_pos)],
                kernel_train[np.ix_(val_pos, train_pos)],
                c_value,
                seed,
            )
            prediction = scores.argmax(axis=1).astype(np.int64)
            fold_values.append(
                metrics(y[inner_split["val_idx"]], prediction)["balanced_accuracy"]
            )
        inner_means[str(c_value)] = float(np.mean(fold_values))
    winner = max(c_grid, key=lambda value: (inner_means[str(value)], -c_grid.index(value)))
    return float(winner), inner_means


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for model_name in sorted({row["model"] for row in rows}):
        for rule in ("independent", "balanced"):
            values = [row for row in rows if row["model"] == model_name and row["prediction_rule"] == rule]
            if not values:
                continue
            output.append(
                {
                    "model": model_name,
                    "prediction_rule": rule,
                    "split_count": len(values),
                    "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in values])),
                    "mean_balanced_accuracy": float(
                        np.mean([row["metrics"]["balanced_accuracy"] for row in values])
                    ),
                    "std_balanced_accuracy": float(
                        np.std([row["metrics"]["balanced_accuracy"] for row in values])
                    ),
                    "mean_macro_f1": float(np.mean([row["metrics"]["macro_f1"] for row in values])),
                    "mean_per_class_recall": {
                        class_name: float(
                            np.mean([row["metrics"]["per_class_recall"][class_name] for row in values])
                        )
                        for class_name in values[0]["metrics"]["per_class_recall"]
                    },
                }
            )
    return output


def subject_bootstrap(
    rows: list[dict],
    iterations: int,
    seed: int,
) -> dict:
    """Subject-level bootstrap CI on each model/rule mean balanced accuracy."""
    output = {}
    rng = np.random.default_rng(seed)
    for model_name in sorted({row["model"] for row in rows}):
        for rule in ("independent", "balanced"):
            values = [
                row for row in rows if row["model"] == model_name and row["prediction_rule"] == rule
            ]
            if not values:
                continue
            by_subject: dict[str, list[float]] = defaultdict(list)
            for row in values:
                for subject, metric_row in row["subject_metrics"].items():
                    by_subject[subject].append(metric_row["balanced_accuracy"])
            subjects = sorted(by_subject)
            per_subject = np.asarray([np.mean(by_subject[subject]) for subject in subjects])
            samples = rng.choice(
                per_subject, size=(iterations, len(per_subject)), replace=True
            ).mean(axis=1)
            output[f"{model_name}|{rule}"] = {
                "subject_count": len(subjects),
                "subject_mean_balanced_accuracy": float(np.mean(per_subject)),
                "subject_ci95": [float(v) for v in np.quantile(samples, [0.025, 0.975])],
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standard MVPA comparators under the frozen 30-fold subject protocol."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument(
        "--preprocess",
        choices=["frozen", "raw", "subject_center"],
        default="frozen",
        help=(
            "frozen: the unlabeled subject-run centering and per-lag detrending used by "
            "the frozen hierarchy, i.e. one offset removed per subject AND run. "
            "raw: no centering; note that per-feature standardisation already removes a "
            "global training mean, so this isolates the effect of nuisance structure that "
            "a single global offset cannot absorb. "
            "subject_center: one offset removed per subject, pooling that subject's runs, "
            "which separates subject-level from run-level nuisance."
        ),
    )
    parser.add_argument(
        "--align",
        choices=["none", "subject_zscore", "coral", "pca_only"],
        default="none",
        help=(
            "Second-order per-subject alignment applied after --preprocess. "
            "subject_zscore is diagonal; coral whitens the full covariance in a "
            "shared low-rank subspace. Both are label-free."
        ),
    )
    parser.add_argument("--coral-components", type=int, default=24,
                        help="Subspace rank for CORAL. Must stay well below 48 events per subject.")
    parser.add_argument("--coral-shrinkage", type=float, default=0.3,
                        help="Shrinkage of each subject covariance toward its diagonal.")
    parser.add_argument(
        "--brain-mask-quantile",
        type=float,
        help=(
            "Keep only voxels whose across-event standard deviation on TRAINING rows "
            "exceeds this quantile. The radial probe showed roughly half the 24^3 grid "
            "is skull, scalp, and air, contributing nothing. Computed per fold from "
            "training rows only, so it introduces no leakage."
        ),
    )
    parser.add_argument(
        "--voxel-mask-npz",
        help=(
            "NPZ holding a `frequency` array over the feature grid, from "
            "build_cortical_ribbon_mask.py. Restricts decoding to voxels whose ribbon "
            "frequency across subjects meets --voxel-mask-threshold. Derived from "
            "anatomy alone, so it consults no labels and no functional data."
        ),
    )
    parser.add_argument("--voxel-mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--mask-criterion",
        choices=["anova", "variance"],
        default="anova",
        help=(
            "How --brain-mask-quantile ranks voxels. anova uses a one-way F across "
            "classes on training rows, the canonical MVPA selection. variance ranks by "
            "raw spread and performs poorly here, because high variance in this grid "
            "reflects edge and motion artefact rather than signal."
        ),
    )
    parser.add_argument(
        "--center-events",
        type=int,
        help=(
            "Center each subject-run using only its first K events rather than all "
            "eight. Quantifies how much target-run data the centering actually needs, "
            "which is the project's main deployment liability."
        ),
    )
    parser.add_argument(
        "--exclude-subjects",
        nargs="+",
        default=[],
        help=(
            "Drop these subjects before splitting. The frozen protocol's prespecified "
            "QC-60 stratum is --exclude-subjects sub-42 sub-52."
        ),
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--split-limit", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    parser.add_argument("--model-seed", type=int, default=0)
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence = feature_dict.pop(args.sequence_key)
    if args.exclude_subjects:
        excluded = set(args.exclude_subjects)
        keep = np.asarray(
            [str(record["subject_id"]) not in excluded for record in records]
        )
        if not keep.any():
            raise SystemExit("--exclude-subjects removed every event.")
        # Excluding before preprocessing matters: subject-run centering and
        # detrending are per-run, but the dropped subjects would otherwise still
        # influence the outer split construction.
        sequence = sequence[keep]
        y = y[keep]
        records = [record for record, flag in zip(records, keep) if flag]
        print(
            f"excluded {sorted(excluded)}: {int(keep.sum())} events from "
            f"{len({r['subject_id'] for r in records})} subjects remain",
            flush=True,
        )
    detrend_rows: list[dict] = []
    if args.center_events is not None:
        # Partial centering: the offset for each run is estimated from its first K
        # events only, then applied to all of them. This is what a deployment that
        # must predict before the run ends would actually have available.
        order = sorted(
            range(len(records)),
            key=lambda i: (
                str(records[i]["subject_id"]),
                int(records[i]["run_id"]),
                int(records[i]["event_start"]),
            ),
        )
        groups: dict[tuple[str, int], list[int]] = {}
        for i in order:
            groups.setdefault(
                (str(records[i]["subject_id"]), int(records[i]["run_id"])), []
            ).append(i)
        for indices in groups.values():
            head = indices[: args.center_events]
            for lag in range(sequence.shape[1]):
                sequence[np.asarray(indices), lag] -= sequence[
                    np.asarray(head), lag
                ].mean(axis=0)
        print(
            f"centered each run using its first {args.center_events} of "
            f"{len(next(iter(groups.values())))} events",
            flush=True,
        )
    elif args.preprocess == "frozen":
        sequence, detrend_rows = preprocess_sequence(sequence, records)
    mean_x = sequence.mean(axis=1, dtype=np.float32)
    del sequence
    ribbon_info: dict = {}
    if args.voxel_mask_npz:
        with np.load(args.voxel_mask_npz) as data:
            frequency = data["frequency"]
        if len(frequency) != mean_x.shape[1]:
            raise SystemExit(
                f"mask has {len(frequency)} voxels, features have {mean_x.shape[1]}"
            )
        keep = frequency >= args.voxel_mask_threshold
        ribbon_info = {
            "threshold": args.voxel_mask_threshold,
            "voxels_kept": int(keep.sum()),
            "voxels_total": int(len(frequency)),
        }
        mean_x = mean_x[:, keep]
        print(f"ribbon mask: {ribbon_info}", flush=True)

    alignment_info: dict = {}
    if args.preprocess == "subject_center":
        subjects = np.asarray([str(record["subject_id"]) for record in records])
        for subject in sorted(set(subjects.tolist())):
            mask = subjects == subject
            mean_x[mask] -= mean_x[mask].mean(axis=0)

    # Second-order alignment is applied *after* the chosen centering, so it tests
    # whether aligning subject covariances adds anything to the first-order
    # alignment the frozen pipeline already performs.
    if args.align == "subject_zscore":
        mean_x = subject_zscore(mean_x, records)
        alignment_info = {"method": "subject_zscore", "diagonal": True}
    elif args.align in ("coral", "pca_only"):
        mean_x, alignment_info = coral_whiten(
            mean_x,
            records,
            args.coral_components,
            args.coral_shrinkage,
            whiten=(args.align == "coral"),
        )
        alignment_info["method"] = args.align
        print(f"{args.align}: {alignment_info}", flush=True)
    class_count = int(y.max()) + 1

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    rows: list[dict] = []
    selection_rows: list[dict] = []
    started = time.time()
    for split in splits:
        inner = inner_splits(
            records, split["train_idx"], "subject", args.inner_subject_fold_count
        )
        # Split isolation: no held-out subject may appear in any inner selection fold.
        val_subjects = {str(records[int(i)]["subject_id"]) for i in split["val_idx"]}
        for inner_split in inner:
            for key in ("train_idx", "val_idx"):
                inner_subjects = {
                    str(records[int(i)]["subject_id"]) for i in inner_split[key]
                }
                if inner_subjects & val_subjects:
                    raise RuntimeError(
                        f"Split isolation violated in {split['split']}/{inner_split['split']}."
                    )

        needs_projection = any(name != "correlation_centroid" for name in args.models)
        z_train = z_val = kernel_train = kernel_val = None
        inner_local: list[dict] = []
        voxel_mask = None
        if args.brain_mask_quantile is not None:
            train_idx = split["train_idx"]
            if args.mask_criterion == "variance":
                statistic = mean_x[train_idx].std(axis=0)
            else:
                # One-way ANOVA F across the four classes, computed on training rows
                # with training labels only. This is the canonical univariate feature
                # selection used in MVPA, and unlike variance it targets voxels that
                # discriminate rather than voxels that merely move -- which matters
                # here, since the discriminative signal sits in low-variance directions.
                block = mean_x[train_idx]
                labels = y[train_idx]
                grand = block.mean(axis=0)
                between = np.zeros(block.shape[1], dtype=np.float64)
                within = np.zeros(block.shape[1], dtype=np.float64)
                for class_id in range(class_count):
                    class_rows = block[labels == class_id]
                    if len(class_rows) < 2:
                        continue
                    centre = class_rows.mean(axis=0)
                    between += len(class_rows) * (centre - grand) ** 2
                    within += ((class_rows - centre) ** 2).sum(axis=0)
                statistic = between / np.maximum(within, 1e-12)
            voxel_mask = statistic > np.quantile(statistic, args.brain_mask_quantile)
        fold_x = mean_x[:, voxel_mask] if voxel_mask is not None else mean_x

        if needs_projection:
            mean, scale = standardize(fold_x, split["train_idx"])
            z_train, z_val = dual_basis(
                (fold_x[split["train_idx"]] - mean) / scale,
                (fold_x[split["val_idx"]] - mean) / scale,
            )
            kernel_train = z_train @ z_train.T
            kernel_val = z_val @ z_train.T
            position = {
                int(global_idx): local_pos
                for local_pos, global_idx in enumerate(split["train_idx"])
            }
            inner_local = [
                {
                    **inner_split,
                    "train_pos": np.asarray(
                        [position[int(i)] for i in inner_split["train_idx"]], dtype=np.int64
                    ),
                    "val_pos": np.asarray(
                        [position[int(i)] for i in inner_split["val_idx"]], dtype=np.int64
                    ),
                }
                for inner_split in inner
            ]

        for model_name in args.models:
            if model_name == "correlation_centroid":
                chosen_c, inner_means = 0.0, {}
                scores = correlation_centroid_scores(
                    fold_x, y, split["train_idx"], split["val_idx"], class_count
                )
            else:
                chosen_c, inner_means = select_c(
                    model_name,
                    z_train,
                    kernel_train,
                    y,
                    inner_local,
                    tuple(args.c_grid),
                    args.model_seed,
                )
                scores = fit_projected(
                    model_name,
                    z_train,
                    y[split["train_idx"]],
                    z_val,
                    kernel_train,
                    kernel_val,
                    chosen_c,
                    args.model_seed,
                )
            evaluated = evaluate_split(scores, y, split["val_idx"], records)
            for rule, payload in evaluated.items():
                rows.append(
                    {
                        "split": split["split"],
                        "subject_seed": split["subject_seed"],
                        "model": model_name,
                        "prediction_rule": rule,
                        "selected_c": chosen_c,
                        "val_subject_count": len(val_subjects),
                        "val_event_count": int(len(split["val_idx"])),
                        "metrics": payload["metrics"],
                        "subject_metrics": payload["subject_metrics"],
                    }
                )
            selection_rows.append(
                {
                    "split": split["split"],
                    "model": model_name,
                    "selected_c": chosen_c,
                    "dual_rank": int(z_train.shape[1]) if z_train is not None else None,
                    "voxels_kept": int(fold_x.shape[1]),
                    "inner_balanced_accuracy": inner_means,
                }
            )
            print(
                f"{split['split']} {model_name} C={chosen_c} "
                f"independent={evaluated['independent']['metrics']['accuracy']:.4f} "
                f"balanced={evaluated['balanced']['metrics']['balanced_accuracy']:.4f} "
                f"[{time.time() - started:.0f}s]",
                flush=True,
            )

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "sequence_key": args.sequence_key,
        "preprocess": args.preprocess,
        "align": args.align,
        "feature_count": int(mean_x.shape[1]),
        "event_count": int(mean_x.shape[0]),
        "class_count": class_count,
        "models": list(args.models),
        "c_grid": list(args.c_grid),
        "outer_split_count": len(splits),
        "inner_subject_fold_count": args.inner_subject_fold_count,
        "selection_metric": "inner-subject-fold balanced accuracy, argmax rule",
        "alignment_info": alignment_info,
        "ribbon_mask": ribbon_info,
        "detrend_by_lag": detrend_rows,
        "selected_c_counts": {
            model_name: dict(
                Counter(
                    row["selected_c"]
                    for row in selection_rows
                    if row["model"] == model_name
                )
            )
            for model_name in args.models
        },
        "selection_rows": selection_rows,
        "rows": rows,
        "summary": summarize(rows),
        "subject_bootstrap": subject_bootstrap(
            rows, args.bootstrap_iterations, args.bootstrap_seed
        ),
        "feature_basis": (
            "L2-penalised models are fitted in an exact dual basis spanning the outer "
            "training row space, which yields identical predictions to fitting on all "
            "13824 features while remaining tractable. The basis uses training rows "
            "only. The correlation centroid is computed on the full feature space."
        ),
        "note": (
            "Conventional linear MVPA comparators on the same events, representation, "
            "and outer subject splits as the frozen hierarchy. Standardisation and C "
            "selection use outer-training subjects only; inner folds partition those "
            "training subjects. The independent rule is the comparable number. The "
            "balanced rule is the design-constrained assignment and is reported only "
            "so both approaches are compared under an identical rule."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps({"out_json": args.out_json, "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
