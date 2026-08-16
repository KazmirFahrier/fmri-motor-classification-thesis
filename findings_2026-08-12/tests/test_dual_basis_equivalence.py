from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The module under test lives in this findings folder, not the repository's scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_standard_mvpa_baseline import (  # noqa: E402
    correlation_centroid_scores,
    dual_basis,
    fit_projected,
)

pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.svm import SVC  # noqa: E402


def make_problem(
    n_train: int = 150,
    n_val: int = 50,
    n_features: int = 800,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A wide problem, matching the regime where features far outnumber events."""
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal((n_features, 4))
    x_train = rng.standard_normal((n_train, n_features))
    x_val = rng.standard_normal((n_val, n_features))
    y_train = (x_train @ weights).argmax(axis=1)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (x_train - mean) / scale, (x_val - mean) / scale, y_train


def test_dual_basis_rank_is_bounded_by_sample_count():
    x_train, x_val, _ = make_problem()
    z_train, z_val = dual_basis(x_train, x_val)
    assert z_train.shape[0] == x_train.shape[0]
    assert z_val.shape[0] == x_val.shape[0]
    # The row space cannot exceed the number of training rows, so the basis is
    # far smaller than the 800 raw features.
    assert z_train.shape[1] <= x_train.shape[0]
    assert z_train.shape[1] < x_train.shape[1]


def test_dual_basis_preserves_inner_products():
    x_train, x_val, _ = make_problem()
    z_train, z_val = dual_basis(x_train, x_val)
    np.testing.assert_allclose(z_train @ z_train.T, x_train @ x_train.T, atol=1e-6)
    np.testing.assert_allclose(z_val @ z_train.T, x_val @ x_train.T, atol=1e-6)


@pytest.mark.parametrize("c_value", [1e-3, 1e-2, 1e-1])
def test_projected_logistic_matches_full_feature_fit(c_value):
    """The dual basis is an exact reparameterisation, not an approximation."""
    x_train, x_val, y_train = make_problem()
    z_train, z_val = dual_basis(x_train, x_val)

    full = (
        LogisticRegression(C=c_value, max_iter=20000, tol=1e-10, random_state=0)
        .fit(x_train, y_train)
        .decision_function(x_val)
    )
    projected = fit_projected(
        "logistic_l2", z_train, y_train, z_val, None, None, c_value, 0
    )
    assert np.array_equal(full.argmax(axis=1), projected.argmax(axis=1))
    assert np.abs(full - projected).max() < 1e-2


@pytest.mark.parametrize("c_value", [1e-3, 1e-1, 1.0])
def test_precomputed_kernel_svm_matches_linear_kernel_svm(c_value):
    x_train, x_val, y_train = make_problem()
    z_train, z_val = dual_basis(x_train, x_val)
    kernel_train = z_train @ z_train.T
    kernel_val = z_val @ z_train.T

    full = (
        SVC(C=c_value, kernel="linear", decision_function_shape="ovr", random_state=0)
        .fit(x_train, y_train)
        .decision_function(x_val)
    )
    projected = fit_projected(
        "linear_svm", z_train, y_train, z_val, kernel_train, kernel_val, c_value, 0
    )
    np.testing.assert_allclose(full, projected, atol=1e-4)


def test_correlation_centroid_predicts_training_structure():
    x_train, x_val, y_train = make_problem()
    x = np.concatenate([x_train, x_val], axis=0)
    train_idx = np.arange(len(x_train))
    scores = correlation_centroid_scores(
        x, np.concatenate([y_train, np.zeros(len(x_val), dtype=np.int64)]),
        train_idx, train_idx, 4,
    )
    # Evaluated on its own training rows the centroid classifier must beat chance.
    assert (scores.argmax(axis=1) == y_train).mean() > 0.4
