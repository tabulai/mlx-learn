"""Fused centering computes initial statistics from its actual owned output."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._kernels.logistic_input import prepare_centered
from mlxlearn._solvers.lbfgs import LBFGSConfig
from mlxlearn._solvers.logistic import _native_binary_objective
from mlxlearn._solvers.native_logistic import initial, initial_from_parts, minimize_binary

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _gpu_stream():
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    with use_device("gpu"):
        yield


def _reference_initial(design, y, weights, denominator, intercept):
    """Float64 initial gradient and mean curvature diagonal of the solver's metric."""
    design = np.asarray(design, dtype=np.float64)
    residual = (0.5 - np.asarray(y, dtype=np.float64)) * weights
    gradient = residual @ design / denominator
    diagonal = 0.25 * (weights @ (design * design)) / denominator
    if intercept:
        gradient = np.append(gradient, residual.sum() / denominator)
        diagonal = np.append(diagonal, 0.25 * weights.sum() / denominator)
    return gradient, diagonal.mean()


@pytest.mark.parametrize("width", [1, 31, 32, 33, 1024])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("center", [True, False])
def test_prepared_initial_parts_match_a_float64_reference(width, weighted, center):
    rng = np.random.default_rng(548)
    n = 259  # Tail rows and feature tiles must also be initialized.
    X = rng.normal(size=(n, width)).astype(np.float32)
    X += np.linspace(-1e4, 1e4, width, dtype=np.float32) if center else np.float32(0.5)
    y = mx.array(rng.integers(0, 2, n).astype(np.float32))
    weights = np.geomspace(1e-3, 1e3, n).astype(np.float32)
    weights[::7] = 0
    sw = mx.array(weights) if weighted else None
    owned, shift, finite, parts = prepare_centered(mx.array(X), y, sw, center=center)
    p = width + int(center)
    mx.eval(owned, finite, parts, *([shift] if center else []))
    assert bool(finite)
    assert parts.shape == (2, 2, p)
    if center:
        np.testing.assert_array_equal(np.asarray(owned), X - np.asarray(shift))
    else:
        assert shift is None
        np.testing.assert_array_equal(np.asarray(owned), X)
    # Verify independently, including the implicit intercept and zero weights.
    denominator = float(weights.sum(dtype=np.float64)) if weighted else float(n)
    gradient, metric = initial_from_parts(parts, mx.zeros(p), mx.array(denominator))
    expected_gradient, expected_diagonal = _reference_initial(
        owned, y, weights if weighted else np.ones(n), denominator, center
    )
    np.testing.assert_allclose(np.asarray(gradient), expected_gradient, rtol=3e-5, atol=2e-7)
    np.testing.assert_allclose(1.0 / float(metric[0]), expected_diagonal, rtol=3e-5)


@pytest.mark.parametrize("width", [7, 37, 519])
def test_uncentered_parts_of_a_centered_design_match_centered_parts_bitwise(width):
    """The solver-side fallback and the prepared route use one statistics kernel."""
    rng = np.random.default_rng(91)
    n = 300
    X = (rng.normal(size=(n, width)) + 3.0).astype(np.float32)
    y = mx.array(rng.integers(0, 2, n).astype(np.float32))
    owned, shift, finite, parts = prepare_centered(mx.array(X), y)
    mx.eval(owned, shift, finite, parts)
    host_centered = mx.array(X - np.asarray(shift))
    _, none, _, direct = prepare_centered(host_centered, y, center=False, intercept=True)
    mx.eval(direct)
    assert none is None
    np.testing.assert_array_equal(np.asarray(owned), np.asarray(host_centered))
    np.testing.assert_array_equal(np.asarray(parts), np.asarray(direct))
    reg, denominator = mx.zeros(width + 1), mx.array(float(n))
    expected = initial_from_parts(parts, reg, denominator)
    fallback = initial(owned, y, mx.ones(n), width + 1, reg, denominator)
    mx.eval(expected, fallback)
    np.testing.assert_array_equal(np.asarray(fallback[0]), np.asarray(expected[0]))
    np.testing.assert_array_equal(np.asarray(fallback[1]), np.asarray(expected[1]))


@pytest.mark.parametrize("binary", [False, True])
def test_centering_finite_constants_and_readonly_input(binary):
    limit = np.finfo(np.float32).max
    values = np.array([limit, -limit, 1e4, -1e4, 0.0], dtype=np.float32)
    X = np.broadcast_to(values, (261, len(values))).copy()
    X.flags.writeable = False
    borrowed = mx.asarray(X) if hasattr(mx, "asarray") else mx.array(X)
    y = mx.zeros(len(X)) if binary else None
    owned, center, finite, parts = prepare_centered(borrowed, y)
    arrays = [owned, center, finite] + ([] if parts is None else [parts])
    mx.eval(*arrays)
    assert bool(finite)
    assert (parts is not None) is binary
    np.testing.assert_array_equal(np.asarray(center), values)
    np.testing.assert_array_equal(np.asarray(owned), np.zeros_like(X))
    assert not np.shares_memory(np.asarray(owned), X)
    np.testing.assert_array_equal(X[0], values)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("row", [0, 128, 258])
def test_centering_checks_every_original_input_value(bad, row):
    X = np.zeros((259, 33), dtype=np.float32)
    X[row, -1] = bad
    owned, center, finite, parts = prepare_centered(mx.array(X))
    mx.eval(owned, center, finite)
    assert parts is None
    assert not bool(finite)


def test_precomputed_initial_parts_preserve_complete_native_fit():
    rng = np.random.default_rng(649)
    n, d = 273, 37
    X = mx.array(rng.normal(size=(n, d)).astype(np.float32) + 1e4)
    y = mx.array(rng.integers(0, 2, n).astype(np.float32))
    weights = np.geomspace(1e-2, 1e2, n).astype(np.float32)
    weights[::13] = 0
    sw = mx.array(weights)
    owned, center, finite, parts = prepare_centered(X, y, sw)
    mx.eval(owned, center, finite, parts)
    denominator = float(weights.sum(dtype=np.float64))
    fun = _native_binary_objective(owned, y, sw, denominator, 0.1, True)
    arguments = (owned, y, sw, denominator, 0.1, True, LBFGSConfig(tol=1e-4), fun)
    expected = minimize_binary(*arguments)
    prepared = minimize_binary(*arguments, initial_parts=parts)
    assert expected.converged and prepared.converged
    assert expected.n_iter == prepared.n_iter
    assert expected.objective == prepared.objective
    np.testing.assert_array_equal(np.asarray(expected.x), np.asarray(prepared.x))
