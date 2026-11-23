"""Tall least squares when feature units account for the weak spectrum."""

import numpy as np
import pytest
from scipy import linalg

from mlxlearn._solvers.ols import OLSConfig, solve_ols


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("repeated_column", [False, True])
def test_graded_columns_preserve_original_units(weighted, repeated_column):
    rng = np.random.default_rng(2026)
    n, d = 4096, 64
    X = np.asfortranarray(rng.normal(size=(n, d)) * rng.permutation(np.geomspace(1, 1e-9, d)))
    if repeated_column:
        X[:, -1] = 0.75 * X[:, 0]
    y = X @ rng.normal(size=(d, 3)) + 0.05 * rng.normal(size=(n, 3))
    w = rng.uniform(0.5, 2, n) if weighted else None
    if weighted:
        w[::13] = 0
    sol = solve_ols(X, y, sample_weight=w, config=OLSConfig(rcond=1e-6))

    Xc = X - np.average(X, axis=0, weights=w)
    yc = y - np.average(y, axis=0, weights=w)
    if weighted:
        Xc *= np.sqrt(w)[:, None]
        yc *= np.sqrt(w)[:, None]
    coef, _, rank, singular = linalg.lstsq(Xc, yc, cond=1e-6)
    assert sol.method == ("lstsq" if repeated_column else "cholqr")
    assert sol.rank == rank < d
    np.testing.assert_allclose(sol.singular, singular, atol=1e-12 * singular[0], rtol=1e-10)
    np.testing.assert_allclose(sol.coef, coef.T, atol=1e-10 * np.abs(coef).max(), rtol=1e-10)


def test_graded_column_at_cutoff_uses_stock_arithmetic():
    d = 64
    scales = np.geomspace(1, 1e-9, d)
    scales[42] = 1e-6
    X = np.tile(np.diag(scales), (64, 1))
    y = np.random.default_rng(2026).normal(size=len(X))
    sol = solve_ols(X, y, config=OLSConfig(fit_intercept=False, rcond=1e-6))
    coef, _, rank, singular = linalg.lstsq(X, y, cond=1e-6)
    assert sol.method == "lstsq"
    assert sol.rank == rank
    np.testing.assert_array_equal(sol.singular, singular)
    np.testing.assert_array_equal(sol.coef[0], coef)


def test_orthogonal_target_among_multiple_outputs_preserves_stock_rounding():
    rng = np.random.default_rng(2027)
    d = 64
    half = rng.normal(size=(2048, d)) * np.geomspace(1, 1e-9, d)
    X = np.vstack((half, half))
    response = rng.normal(size=len(half))
    y = np.column_stack((X @ rng.normal(size=d), np.concatenate((response, -response))))
    sol = solve_ols(X, y, config=OLSConfig(fit_intercept=False, rcond=1e-6))
    ref, _, rank, singular = linalg.lstsq(X, y, cond=1e-6)
    assert sol.method == "lstsq"
    assert sol.rank == rank
    np.testing.assert_array_equal(sol.singular, singular)
    np.testing.assert_array_equal(sol.coef, ref.T)
