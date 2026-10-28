"""Independent linear-system and rank checks for the native Newton CG loop."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import _native_cg


def _arrays():
    with mx.stream(mx.cpu):
        return mx.zeros((8, 3)), mx.ones((8,)), mx.zeros((4,)), mx.ones((4,))


@pytest.mark.parametrize("index,value", [
    (0, None), (1, None), (2, None), (3, None),
])
def test_input_types_are_checked_before_export(monkeypatch, index, value):
    monkeypatch.setattr(_native_cg, "metal_loop", SimpleNamespace(run_cg=lambda *a: pytest.fail("native called")))
    arrays = list(_arrays())
    arrays[index] = value
    with pytest.raises(ValueError, match="float32"):
        _native_cg.solve_cg(*arrays)


@pytest.mark.parametrize("forcing", [True, 0.0, -0.1, 1e-5, 0.5001, float("nan"), float("inf")])
def test_invalid_forcing_is_rejected_before_export(monkeypatch, forcing):
    monkeypatch.setattr(_native_cg, "metal_loop", SimpleNamespace(run_cg=lambda *a: pytest.fail("native called")))
    with pytest.raises(ValueError, match="forcing"):
        _native_cg.solve_cg(*_arrays(), forcing=forcing)


@pytest.mark.parametrize("budget", [True, 0, -1, 513, 1.0])
def test_invalid_iteration_budget_is_rejected_before_export(monkeypatch, budget):
    monkeypatch.setattr(_native_cg, "metal_loop", SimpleNamespace(run_cg=lambda *a: pytest.fail("native called")))
    with pytest.raises(ValueError, match="iteration budget"):
        _native_cg.solve_cg(*_arrays(), max_iter=budget)


def test_unavailable_extension_is_explicit(monkeypatch):
    monkeypatch.setattr(_native_cg, "metal_loop", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        _native_cg.solve_cg(*_arrays())


def _require_native():
    if (not mx.metal.is_available() or _native_cg.metal_loop is None
            or not hasattr(_native_cg.metal_loop, "run_cg")):
        pytest.skip("requires native Metal CG extension")


def _hessian(X, curvature, regularization):
    A = np.column_stack((np.asarray(X, np.float64), np.ones(len(X))))
    return A.T @ (np.asarray(curvature, np.float64)[:, None] * A) + np.diag(regularization.astype(np.float64))


@pytest.mark.gpu
@pytest.mark.parametrize("d", [13, 65])
def test_cg_matches_independent_float64_hessian(d):
    _require_native()
    rng = np.random.default_rng(448 + d)
    X = rng.normal(size=(257, d)).astype(np.float32)
    curvature = rng.uniform(0.1, 0.4, len(X)).astype(np.float32) / len(X)
    gradient = rng.normal(scale=0.1, size=d + 1).astype(np.float32)
    regularization = np.r_[np.full(d, 0.03), 0].astype(np.float32)
    hessian = _hessian(X, curvature, regularization)
    expected = np.linalg.solve(hessian, -gradient)
    with mx.stream(mx.gpu):
        result, metadata = _native_cg.solve_cg(
            mx.array(X), mx.array(curvature), mx.array(gradient), mx.array(regularization),
            forcing=1e-4, max_iter=128,
        )
        actual = np.array(result, dtype=np.float64)
    assert metadata[1] == 1
    assert 0 < metadata[0] <= 128
    assert np.linalg.norm(hessian @ actual + gradient) / np.linalg.norm(gradient) < 2e-4
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 3e-4


@pytest.mark.gpu
def test_cg_retains_identifiable_slow_mode_after_a_large_rayleigh_quotient():
    """A diagonal SPD system remains identifiable at a 1e10 curvature spread."""
    _require_native()
    X = np.array([[1e5, 0], [-1e5, 0], [0, 1], [0, -1]], dtype=np.float32)
    curvature = np.full(4, 0.0625, dtype=np.float32)
    gradient = np.array([1, 1, 0], dtype=np.float32)
    regularization = np.array([1e-4, 1e-4, 0], dtype=np.float32)
    # The columns and intercept are exactly orthogonal, so the low-curvature
    # direction is resolvable without subtracting large nearby quantities.
    diagonal = np.array([1.25e9, 0.125, 0.25]) + regularization
    expected = -gradient / diagonal
    with mx.stream(mx.gpu):
        result, metadata = _native_cg.solve_cg(
            mx.array(X), mx.array(curvature), mx.array(gradient), mx.array(regularization),
            forcing=1e-4, max_iter=128,
        )
        actual = np.asarray(result, dtype=np.float64)
    assert metadata[1] == 1, metadata
    assert 1 < metadata[0] <= 128
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=1e-12)
    assert np.linalg.norm(diagonal * actual + gradient) / np.linalg.norm(gradient) < 3e-4


@pytest.mark.gpu
def test_cg_preserves_general_exact_dependency_at_weak_regularization():
    _require_native()
    rng = np.random.default_rng(633)
    n, d = 512, 129
    X = rng.integers(-2, 3, (n, d)).astype(np.float32)
    X[:, -1] = X[:, 1] + X[:, 67]
    curvature = np.full(n, 0.25 / n, np.float32)
    residual = rng.integers(-8, 9, n).astype(np.float64) / 4096
    gradient = np.r_[X.astype(np.float64).T @ residual, residual.sum()].astype(np.float32)
    regularization = np.r_[np.full(d, 1e-10), 0].astype(np.float32)
    hessian = _hessian(X, curvature, regularization)
    # The tiny regularizer is deliberately below the float32 operator floor.
    # The reference retains the identifiable row-space solution only.
    expected = -np.linalg.pinv(hessian, rcond=1e-7) @ gradient
    with mx.stream(mx.gpu):
        result, metadata = _native_cg.solve_cg(
            mx.array(X), mx.array(curvature), mx.array(gradient), mx.array(regularization),
            forcing=1e-4, max_iter=128,
        )
        actual = np.array(result, dtype=np.float64)
    assert metadata[1] in (1, 2)
    assert np.all(np.isfinite(actual))
    null_ratio = abs(actual[-2] - actual[1] - actual[67]) / (np.sqrt(3) * np.linalg.norm(actual[:-1]))
    assert null_ratio < 2e-5
    assert np.linalg.norm(hessian @ actual + gradient) / np.linalg.norm(gradient) < 3e-4
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-4


@pytest.mark.gpu
def test_weak_regularization_does_not_amplify_roundoff_in_a_null_direction():
    """A small near-stationary RHS must not produce a huge null-space step.

    The identical columns have one identifiable direction. The 3e-9 component
    along their difference represents absolute gradient roundoff near an
    optimum; resolving it against a 1e-10 regularizer would move coefficients
    by about 30 while barely changing the predictions.
    """
    _require_native()
    with mx.stream(mx.gpu):
        result, metadata = _native_cg.solve_cg(
            mx.array([[1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]]),
            mx.full((4,), 0.25),
            mx.array([1.03e-7, 0.97e-7, 0.0]),
            mx.array([1e-10, 1e-10, 0.0]),
            forcing=1e-4, max_iter=128,
        )
        actual = np.asarray(result, dtype=np.float64)
    assert metadata[1] == 2, metadata  # Truncated unresolved curvature.
    assert np.all(np.isfinite(actual))
    assert np.linalg.norm(actual) < 1e-6
    # The useful row-space step is retained despite the unresolved component.
    np.testing.assert_allclose(actual[0] + actual[1], -1e-7, rtol=2e-3, atol=0)
    assert actual[2] == 0.0


@pytest.mark.gpu
def test_zero_rhs_returns_zero_without_cg_updates():
    _require_native()
    with mx.stream(mx.gpu):
        result, metadata = _native_cg.solve_cg(
            mx.ones((8, 3)), mx.full((8,), 0.1), mx.zeros((4,)), mx.array([0.1, 0.1, 0.1, 0]),
        )
        actual = np.array(result)
    assert metadata[0] == 0 and metadata[1] == 1
    np.testing.assert_array_equal(actual, np.zeros(4))


@pytest.mark.gpu
def test_returned_output_survives_temporary_owner_release_and_reuse():
    _require_native()
    rng = np.random.default_rng(823)
    X = rng.normal(size=(64, 9)).astype(np.float32)
    gradient = rng.normal(size=10).astype(np.float32)
    regularization = np.r_[np.full(9, 0.2), 0].astype(np.float32)
    curvature = np.full(64, 1 / 64, np.float32)
    expected = np.linalg.solve(_hessian(X, curvature, regularization), -gradient)
    with mx.stream(mx.gpu):
        outputs = [
            _native_cg.solve_cg(mx.array(X), mx.array(curvature), mx.array(gradient),
                                mx.array(regularization), forcing=1e-4)[0]
            for _ in range(3)
        ]
        gc.collect()
        churn = mx.arange(8192, dtype=mx.float32) ** 2
        mx.eval(churn)
        for output in outputs:
            np.testing.assert_allclose(np.array(output), expected, atol=2e-4, rtol=3e-4)


def test_native_rejects_source_specialization_before_touching_capsules():
    if _native_cg.metal_loop is None or not hasattr(_native_cg.metal_loop, "run_cg"):
        pytest.skip("requires native Metal CG extension")
    capsules = (None,) * 5
    owners = (capsules,) + (None,) * 5
    with pytest.raises(ValueError, match="specialization mismatch"):
        _native_cg.metal_loop.run_cg("#define N 99\n", capsules, 8, 3, 4, 1, 2, 8, 0.01, owners)
