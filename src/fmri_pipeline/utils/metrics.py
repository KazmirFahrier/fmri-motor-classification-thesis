from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_prob: np.ndarray | None,
    class_names: List[str],
) -> Tuple[Dict[str, Any], np.ndarray]:
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.int64)
    num_classes = len(class_names)

    metrics: Dict[str, Any] = {
        "top1_accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
    }

    per_class_recall = recall_score(
        y_true_arr,
        y_pred_arr,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    metrics["per_class_recall"] = {
        class_names[i]: float(per_class_recall[i]) for i in range(num_classes)
    }

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(num_classes)))

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=np.float32)
        y_true_onehot = _one_hot(y_true_arr, num_classes)
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true_onehot,
                    y_prob_arr,
                    average="macro",
                    multi_class="ovr",
                )
            )
        except Exception:
            metrics["roc_auc_ovr_macro"] = float("nan")

        try:
            metrics["pr_auc_macro"] = float(
                average_precision_score(y_true_onehot, y_prob_arr, average="macro")
            )
        except Exception:
            metrics["pr_auc_macro"] = float("nan")
    else:
        metrics["roc_auc_ovr_macro"] = float("nan")
        metrics["pr_auc_macro"] = float("nan")

    return metrics, cm


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:
            continue

    if not scores:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}

    scores_arr = np.asarray(scores, dtype=np.float32)
    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    return {
        "mean": float(scores_arr.mean()),
        "low": float(np.quantile(scores_arr, lower_q)),
        "high": float(np.quantile(scores_arr, upper_q)),
    }


def compute_bootstrap_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

    bundle: Dict[str, Dict[str, float]] = {}
    bundle["top1_accuracy"] = bootstrap_metric_ci(
        y_true,
        y_pred,
        metric_fn=lambda a, b: accuracy_score(a, b),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed,
    )
    bundle["balanced_accuracy"] = bootstrap_metric_ci(
        y_true,
        y_pred,
        metric_fn=lambda a, b: balanced_accuracy_score(a, b),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed + 1,
    )
    bundle["macro_f1"] = bootstrap_metric_ci(
        y_true,
        y_pred,
        metric_fn=lambda a, b: f1_score(a, b, average="macro", zero_division=0),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed + 2,
    )
    bundle["mcc"] = bootstrap_metric_ci(
        y_true,
        y_pred,
        metric_fn=lambda a, b: matthews_corrcoef(a, b),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed + 3,
    )
    return bundle
