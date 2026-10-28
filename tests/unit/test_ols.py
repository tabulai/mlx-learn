"""Host-side OLS pieces that do not need MLX."""

from __future__ import annotations

import re

import numpy as np
import pytest

from mlxlearn._solvers.ols import OLSConfig, _solve_gram, reference_objective, solve_ols


def test_reference_objective_is_zero_on_a_perfect_line():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert reference_objective(np.array([1.0]), 0.0, X, y) == pytest.approx(0.0)


def test_reference_objective_matches_mean_squared_error_scale():
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 0.0])
    # Predictions are 0 and 1; MSE is 0.5; the objective is MSE / 2 = 0.25.
    assert reference_objective(np.array([1.0]), 0.0, X, y) == pytest.approx(0.25)


def test_solve_gram_recovers_a_full_rank_system():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 3))
    gram = A.T @ A
    coef = rng.normal(size=(3, 2))
    rhs = gram @ coef
    solved, rank, singular = _solve_gram(gram, rhs, rcond=1e-6)
    assert rank == 3
    assert singular.shape == (3,)
    np.testing.assert_allclose(solved, coef, atol=1e-10)


def test_all_zero_sample_weight_error_is_recognizable():
    X = np.zeros((6, 3), dtype=np.float32)
    y = np.arange(6, dtype=np.float64)
    with pytest.raises(ValueError) as excinfo:
        solve_ols(X, y, sample_weight=np.zeros(6), config=OLSConfig())
    assert re.search(r"(weight.*zero)|(zero.*weight)", str(excinfo.value), re.IGNORECASE)


def test_rcond_rejects_a_negative_cutoff():
    with pytest.raises(ValueError, match="rcond"):
        OLSConfig(rcond=-1.0)


def test_gram_path_matches_sklearn_lstsq_in_float64():
    """The host factorization is the same min-norm solution as lstsq.

    This does not exercise MLX; it locks the Gram reduction against
    scikit-learn so a device-product bug cannot hide behind a wrong solver.
    """
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    X, y = make_regression(n_samples=200, n_features=8, noise=0.05, random_state=0)
    x_off = X.mean(axis=0)
    y_off = float(np.mean(y))
    Xc = X - x_off
    yc = y - y_off
    coef_T, rank, _ = _solve_gram(Xc.T @ Xc, Xc.T @ yc.reshape(-1, 1), rcond=1e-6)
    coef = coef_T.ravel()
    intercept = y_off - x_off @ coef
    ref = SkLinearRegression().fit(X, y)
    np.testing.assert_allclose(coef, ref.coef_, atol=1e-10)
    np.testing.assert_allclose(intercept, ref.intercept_, atol=1e-10)
    assert rank == ref.rank_
