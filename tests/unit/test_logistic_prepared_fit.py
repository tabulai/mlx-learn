"""Estimator integration for fused centering, initialization and buffer lifetime."""

from __future__ import annotations

import gc
import weakref
from contextlib import contextmanager

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import expit

from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn.exceptions import BackendExecutionError
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.linear_model import _logistic as estimator

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def fused_route():
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    with tuning_context(force_backend="mlx"):
        yield


def data(dtype, width=519):
    rng = np.random.default_rng(371)
    X = rng.normal(scale=0.1, size=(259, width)).astype(dtype) + dtype(0.125)
    y = rng.integers(0, 2, len(X))
    return X, y


@pytest.mark.parametrize("width", [7, 519])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("weighted", [False, True])
def test_prepared_fit_owns_input_and_preserves_true_gradient(monkeypatch, dtype, weighted, width):
    X, y = data(dtype, width)
    X = np.asfortranarray(X)
    X.flags.writeable = False
    before = X.copy()
    weights = np.geomspace(0.1, 10.0, len(X)) if weighted else None
    if weighted:
        weights[::13] = 0
    original = estimator._prepare_centered_logistic_input
    seen = []

    @contextmanager
    def observe(array, labels, sample_weights, *, native, center):
        with original(array, labels, sample_weights, native=native, center=center) as prepared:
            owned, shift, finite, parts = prepared
            assert finite and native and center and parts is not None
            assert not np.shares_memory(np.asarray(owned), X)
            np.testing.assert_array_equal(np.asarray(owned), array - np.asarray(shift))
            seen.append(True)
            yield prepared

    monkeypatch.setattr(estimator, "_prepare_centered_logistic_input", observe)
    model = LogisticRegression(C=0.1, tol=1e-5, max_iter=200).fit(X, y, sample_weight=weights)
    assert seen == [True]
    assert model.execution_backend_ == "mlx"
    np.testing.assert_array_equal(X, before)
    assert not X.flags.writeable
    raw = np.asarray(X, dtype=np.float64) @ model.coef_[0] + model.intercept_[0]
    residual = expit(raw) - y
    denominator = len(X)
    if weighted:
        residual *= weights
        denominator = weights.sum()
    gradient = residual @ X / denominator + model.coef_[0] / (0.1 * denominator)
    gradient = np.append(gradient, residual.sum() / denominator)
    assert np.max(np.abs(gradient)) <= 1e-5
    prediction = model.predict_proba(before[:10])
    X.flags.writeable = True
    X.fill(0)
    np.testing.assert_array_equal(model.predict_proba(before[:10]), prediction)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_prepared_fit_reports_original_nonfinite_input(dtype, value):
    X, y = data(dtype)
    X[-1, -1] = value
    with pytest.raises(ValueError) as error:
        LogisticRegression().fit(X, y)
    assert not isinstance(error.value, BackendExecutionError)
    assert "NaN" in str(error.value) or "infinity" in str(error.value)


@pytest.mark.parametrize("width", [7, 519])
def test_nonintercept_fit_prepares_an_uncentered_design_without_intercept_parts(monkeypatch, width):
    """Without an intercept nothing can absorb a shift, so the owned copy is the input."""
    X, y = data(np.float32, width)
    original = estimator._prepare_centered_logistic_input
    seen = []

    @contextmanager
    def observe(array, labels, sample_weights, *, native, center):
        with original(array, labels, sample_weights, native=native, center=center) as prepared:
            owned, shift, finite, parts = prepared
            seen.append((center, shift, finite, native, parts.shape))
            np.testing.assert_array_equal(np.asarray(owned), array)
            yield prepared

    monkeypatch.setattr(estimator, "_prepare_centered_logistic_input", observe)
    model = LogisticRegression(fit_intercept=False, C=0.1, max_iter=200).fit(X, y)
    assert seen == [(False, None, True, True, (2, 2, width))]
    np.testing.assert_array_equal(model.intercept_, [0.0])
    raw = np.asarray(X, dtype=np.float64) @ model.coef_[0]
    gradient = (expit(raw) - y) @ X / len(X) + model.coef_[0] / (0.1 * len(X))
    assert np.max(np.abs(gradient)) <= 1e-4


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_scale_skew_threshold_reads_parts_only_for_centered_designs(monkeypatch, fit_intercept):
    """Fits without an intercept keep the stopping threshold they had before
    they carried prepared statistics."""
    from mlxlearn._solvers import native_logistic

    X, y = data(np.float32, 519)
    X *= np.geomspace(0.01, 100.0, X.shape[1], dtype=np.float32)
    original = native_logistic._conservative_config_for_skewed_curvature
    seen = []

    def observe(parts, *args):
        seen.append(parts)
        return original(parts, *args)

    monkeypatch.setattr(native_logistic, "_conservative_config_for_skewed_curvature", observe)
    LogisticRegression(fit_intercept=fit_intercept, max_iter=5).fit(X, y)
    assert len(seen) == 1
    if fit_intercept:
        assert seen[0] is not None and seen[0].shape == (2, 2, X.shape[1] + 1)
    else:
        assert seen[0] is None


def test_private_converted_input_is_released_after_prepared_fit(monkeypatch):
    X, y = data(np.float64)
    original = estimator._prepare_centered_logistic_input
    owners = []

    @contextmanager
    def observe(array, *args, **kwargs):
        owners.append(weakref.ref(array))
        with original(array, *args, **kwargs) as prepared:
            yield prepared

    monkeypatch.setattr(estimator, "_prepare_centered_logistic_input", observe)
    model = LogisticRegression(C=0.1, max_iter=200).fit(X, y)
    gc.collect()
    assert owners and all(owner() is None for owner in owners)
    assert np.isfinite(model.predict_proba(X[:10])).all()


def test_gpu_input_context_drains_on_solver_failure(monkeypatch):
    X, y = data(np.float32)
    original_sync = mx.synchronize
    drains = []

    def synchronize(*args, **kwargs):
        drains.append(True)
        return original_sync(*args, **kwargs)

    def broken_solver(*args, **kwargs):
        assert kwargs["X_center"] is not None
        assert kwargs["initial_parts"] is not None
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr(mx, "synchronize", synchronize)
    monkeypatch.setattr(estimator, "solve_logistic", broken_solver)
    with pytest.raises(BackendExecutionError, match="injected solver failure"):
        LogisticRegression().fit(X, y)
    assert drains == [True]
