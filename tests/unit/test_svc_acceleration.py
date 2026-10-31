"""Estimator-level checks for fused/native SVC and its portable fallback."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC as SkSVC

from mlxlearn._common.config import tuning_context
from mlxlearn._kernels import svc_cpu
from mlxlearn._kernels.kernels import KernelSpec, kernel_combine, kernel_diagonal
from mlxlearn._solvers import native_smo
from mlxlearn.svm import SVC
from mlxlearn.svm import _classes as estimator


def _require_backend(backend):
    if backend == "mlx" and not mx.metal.is_available():
        pytest.skip("requires Metal")


def _data():
    rng = np.random.default_rng(137)
    codes = np.tile(np.arange(3), 24)
    centers = np.array([[-2, 0, 1, 0, 0], [1, 2, -1, 0, 0], [1, -2, 0, 1, 0]])
    x = (centers[codes] + rng.normal(scale=0.6, size=(len(codes), 5))).astype(np.float32)
    labels = np.array(["zebra", "ant", "moth"])[codes]
    weights = rng.uniform(0.3, 1.8, len(codes))
    weights[::11] = 0
    query = np.vstack([x[:15], rng.normal(size=(8, 5))]).astype(np.float32)
    return x, labels, weights, query


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize("cache_size", [0.0001, 200])
def test_estimator_uses_native_optimizer_for_weighted_multiclass_cache_modes(
    monkeypatch, backend, cache_size,
):
    _require_backend(backend)
    if not native_smo.available():
        pytest.skip("optional native SMO extension is not built")
    x, y, weights, query = _data()
    calls = []
    original_native = native_smo.solve_c_svc

    def native(*args, **kwargs):
        calls.append(kwargs["gram"] is not None)
        return original_native(*args, **kwargs)

    def forbidden_portable(*args, **kwargs):
        raise AssertionError("native-enabled estimator invoked the portable optimizer")

    monkeypatch.setattr(native_smo, "solve_c_svc", native)
    monkeypatch.setattr(estimator, "solve_c_svc", forbidden_portable)
    kwargs = {"kernel": "rbf", "gamma": 0.2, "tol": 1e-5, "decision_function_shape": "ovo",
                  "cache_size": cache_size, "class_weight": {"ant": 1.5, "moth": 0.7, "zebra": 1.0}}
    with tuning_context(force_backend=backend):
        model = SVC(**kwargs).fit(x, y, sample_weight=weights)
    reference = SkSVC(**kwargs).fit(x, y, sample_weight=weights)
    assert model.execution_backend_ == backend
    assert calls == [cache_size == 200] * 3
    assert model.fit_status_ == 0
    assert not np.intersect1d(model.support_, np.flatnonzero(weights == 0)).size
    np.testing.assert_array_equal(model.classes_, reference.classes_)
    np.testing.assert_array_equal(model.predict(query), reference.predict(query))
    np.testing.assert_allclose(model.decision_function(query), reference.decision_function(query),
                               atol=8e-5, rtol=1e-4)


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize("missing", ["solver", "arithmetic", "both"])
def test_estimator_still_fits_and_predicts_when_optional_extensions_are_absent(
    monkeypatch, backend, missing,
):
    _require_backend(backend)
    x, y, weights, query = _data()
    kwargs = {"gamma": 0.2, "tol": 1e-5, "decision_function_shape": "ovr", "break_ties": True}
    with tuning_context(force_backend=backend):
        reference = SVC(**kwargs).fit(x, y, sample_weight=weights)
    expected_decisions = reference.decision_function(query)
    expected_predictions = reference.predict(query)
    calls = []
    portable = estimator.solve_c_svc

    def counted_portable(*args, **kwargs):
        calls.append(True)
        return portable(*args, **kwargs)

    monkeypatch.setattr(estimator, "solve_c_svc", counted_portable)
    if missing in ("solver", "both"):
        monkeypatch.setattr(native_smo, "_smo_native", None)
    if missing in ("arithmetic", "both"):
        monkeypatch.setattr(svc_cpu, "_native", None)
    with tuning_context(force_backend=backend):
        fallback = SVC(**kwargs).fit(x, y, sample_weight=weights)
    if missing in ("solver", "both"):
        assert len(calls) == 3
    assert fallback.execution_backend_ == backend
    assert fallback.fit_status_ == 0
    np.testing.assert_array_equal(fallback.predict(query), expected_predictions)
    np.testing.assert_allclose(fallback.decision_function(query), expected_decisions,
                               atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("shape", ["ovo", "ovr"])
@pytest.mark.parametrize("break_ties", [False, True])
@pytest.mark.parametrize("pair_values", [[0.0, 0.0, 0.0], [0.2, -0.9, 0.4]])
def test_zero_support_bias_and_multiclass_boundary_rules(
    monkeypatch, portable, shape, break_ties, pair_values,
):
    if portable:
        monkeypatch.setattr(svc_cpu, "_native", None)
    x = np.array([[-1.0, 0], [0.0, 1], [1.0, 0]])
    y = np.array(["ant", "moth", "zebra"])
    with tuning_context(force_backend="cpu"), pytest.warns(ConvergenceWarning):
        model = SVC(max_iter=0, break_ties=break_ties, decision_function_shape=shape).fit(x, y)
    assert model.support_.size == 0
    # A zero-budget fit gives a valid empty support expansion. Set its three
    # biases to isolate the intercept and exact-zero voting rules end to end.
    decision = np.tile(pair_values, (len(x), 1))
    model._pair_rho_ = -np.asarray(pair_values)
    model._intercept_ = model.intercept_ = np.asarray(pair_values)
    expected_ovr = estimator.ovr_decision_function(decision < 0, -decision, 3)
    expected_decision = decision if shape == "ovo" else expected_ovr
    np.testing.assert_array_equal(model.decision_function(x), expected_decision)
    if break_ties and shape == "ovo":
        with pytest.raises(ValueError, match="break_ties must be False"):
            model.predict(x)
        return
    if break_ties:
        winner = 0 if not any(pair_values) else 2
    else:
        winner = 2 if not any(pair_values) else 0
    np.testing.assert_array_equal(model.predict(x), np.repeat(model.classes_[winner], len(x)))


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_kernel_combine_is_independent_of_query_block_size(backend, kernel):
    _require_backend(backend)
    rng = np.random.default_rng(724)
    # Dyadic coordinates make the small dot products and centering exact, so
    # this tests the contraction's blocking independently of GEMM roundoff.
    a = rng.integers(-5, 6, size=(17, 7)).astype(np.float32) / 4
    b = rng.integers(-5, 6, size=(23, 7)).astype(np.float32) / 4
    weights = rng.normal(size=(23, 6))
    bias = rng.normal(size=6)
    spec = KernelSpec(kernel, gamma=0.125, coef0=-0.25, degree=3)
    kwargs = {"backend": backend, "center": np.full(7, 0.5, np.float32), "bias": bias}
    expected = kernel_combine(spec, a, b, weights, block_rows=len(a), **kwargs)
    for rows in (1, 4):
        actual = kernel_combine(spec, a, b, weights, block_rows=rows, **kwargs)
        if backend == "cpu" and kernel == "sigmoid":
            # MLX CPU tanh chooses scalar/vector approximations by tile shape.
            # Bound two float32 ulps per entry propagated through the weights.
            atol = 2 * np.finfo(np.float32).eps * np.abs(weights).sum(axis=0).max()
        else:
            atol = 2e-14
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=2e-14)


@pytest.mark.parametrize("kernel", ["linear", "poly", "sigmoid"])
def test_kernel_diagonal_preserves_original_float64_coordinates(kernel):
    x = np.array([[1 + 2**-30, 2**-32], [1 + 2**-29, -0.4]], dtype=np.float64)
    spec = KernelSpec(kernel, gamma=0.75, coef0=0.25, degree=3)
    squared = np.sum(x * x, axis=1)
    expected = squared if kernel == "linear" else (
        np.tanh(spec.gamma * squared + spec.coef0) if kernel == "sigmoid"
        else (spec.gamma * squared + spec.coef0) ** spec.degree
    )
    actual = kernel_diagonal(spec, x)
    np.testing.assert_allclose(actual, expected, rtol=3e-16, atol=0)
    assert actual[0] != kernel_diagonal(spec, x.astype(np.float32))[0]
