"""Numerical and state contracts for the native Metal logistic optimizer."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._kernels.logistic_input import prepare_centered
from mlxlearn._solvers import native_logistic
from mlxlearn._solvers.lbfgs import LBFGSConfig
from mlxlearn._solvers.logistic import (
    _binary_objective,
    _native_binary_objective,
    reference_objective,
)
from mlxlearn._solvers.native_logistic import minimize_binary

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _gpu_stream():
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    with use_device("gpu"):
        yield


def _data(n_samples=259, n_features=37):
    rng = np.random.default_rng(152)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    logits = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3
    y = (rng.uniform(size=n_samples) < expit(logits)).astype(np.float32)
    return X, y


def _problem(X, y, *, weights=None, C=1.0, fit_intercept=True):
    Xm, ym = mx.array(X), mx.array(y)
    sw = None if weights is None else mx.array(weights, dtype=mx.float32)
    sw_sum = float(len(X) if weights is None else np.sum(weights, dtype=np.float64))
    l2 = 0.0 if math.isinf(C) else 1.0 / (C * sw_sum)
    fun = _binary_objective(
        mx, Xm, ym, sw, sw_sum, l2, fit_intercept, X.shape[1]
    )
    return Xm, ym, sw, sw_sum, l2, fit_intercept, fun


def _run(problem, config):
    X, y, sw, sw_sum, l2, fit_intercept, fun = problem
    return minimize_binary(X, y, sw, sw_sum, l2, fit_intercept, config, fun)


def _float64_gradient(theta, X, y, weights, l2, fit_intercept):
    coefficients = np.asarray(theta, dtype=np.float64)[0]
    w = coefficients[: X.shape[1]]
    raw = np.asarray(X, dtype=np.float64) @ w
    if fit_intercept:
        raw += coefficients[-1]
    residual = expit(raw) - y
    if weights is not None:
        residual *= weights
    sw_sum = len(X) if weights is None else np.sum(weights, dtype=np.float64)
    grad = residual @ X / sw_sum + l2 * w
    if fit_intercept:
        grad = np.append(grad, residual.sum() / sw_sum)
    return grad


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("C", [1.0, math.inf])
def test_native_tail_blocks_match_direct_gradient_and_sklearn(weighted, fit_intercept, C):
    """Non-aligned rows/features exercise every partial native thread block."""
    X, y = _data()
    weights = np.linspace(0.0, 2.0, len(X), dtype=np.float32) if weighted else None
    problem = _problem(X, y, weights=weights, C=C, fit_intercept=fit_intercept)
    original_X = np.asarray(problem[0]).copy()
    original_y = np.asarray(problem[1]).copy()
    config = LBFGSConfig(max_iter=120, tol=1e-4, history_size=3)

    state = _run(problem, config)

    assert state.converged, state
    assert state.stop_reason == "gtol"
    assert config.history_size < state.n_iter <= config.max_iter
    assert state.x.shape == (1, X.shape[1] + int(fit_intercept))
    value, gradient = problem[-1](state.x)
    assert state.objective == pytest.approx(value, abs=1e-10)
    assert state.grad_max_abs == pytest.approx(float(mx.max(mx.abs(gradient))), abs=1e-7)
    assert state.grad_max_abs <= config.tol
    true_gradient = _float64_gradient(state.x, X, y, weights, problem[4], fit_intercept)
    assert np.max(np.abs(true_gradient)) <= config.tol * 1.05
    np.testing.assert_array_equal(np.asarray(problem[0]), original_X)
    np.testing.assert_array_equal(np.asarray(problem[1]), original_y)

    reference = SkLogisticRegression(
        C=C, fit_intercept=fit_intercept, max_iter=500, tol=1e-10
    ).fit(X, y, sample_weight=weights)
    theta = np.asarray(state.x, dtype=np.float64)
    intercept = theta[:, -1] if fit_intercept else np.zeros(1)
    actual_objective = reference_objective(
        theta[:, : X.shape[1]], intercept, X, y.astype(int),
        n_classes=2, C=C, sample_weight=weights,
    )
    optimum = reference_objective(
        reference.coef_, reference.intercept_, X, y.astype(int),
        n_classes=2, C=C, sample_weight=weights,
    )
    assert actual_objective <= optimum + 2e-6


def test_zero_budget_reports_the_true_start_without_updates():
    X, y = _data(n_samples=17, n_features=3)
    problem = _problem(X, y)

    state = _run(problem, LBFGSConfig(max_iter=0, tol=1e-6))

    assert state.n_iter == 0
    assert state.n_fev >= 1
    assert not state.converged
    assert state.stop_reason == "max_iter"
    np.testing.assert_array_equal(np.asarray(state.x), np.zeros((1, 4)))
    assert state.objective == pytest.approx(np.log(2), rel=1e-6)
    _, gradient = problem[-1](state.x)
    assert state.grad_max_abs == pytest.approx(float(mx.max(mx.abs(gradient))), abs=1e-7)


@pytest.mark.parametrize("budget", [0, 4])
def test_initial_optimum_converges_without_an_accepted_step(budget):
    X = np.zeros((18, 3), dtype=np.float32)
    y = np.tile(np.array([0, 1], dtype=np.float32), 9)
    problem = _problem(X, y)

    state = _run(problem, LBFGSConfig(max_iter=budget))

    assert state.n_iter == 0
    assert state.converged
    assert state.stop_reason == "gtol"
    assert state.grad_max_abs == 0.0
    assert state.objective == pytest.approx(np.log(2), rel=1e-6)
    np.testing.assert_array_equal(np.asarray(state.x), np.zeros((1, 4)))


def test_one_iteration_budget_keeps_finite_unconverged_progress():
    X, y = _data()
    problem = _problem(X, y)
    config = LBFGSConfig(max_iter=1, tol=1e-8)

    state = _run(problem, config)

    assert state.n_iter == 1
    assert not state.converged
    assert state.stop_reason == "max_iter"
    assert 0.0 < state.objective < np.log(2)
    assert state.grad_max_abs > config.tol
    assert np.isfinite(np.asarray(state.x)).all()


@pytest.mark.parametrize("budget", [0, 4])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_initial_data_never_claims_convergence(budget, bad_value):
    X, y = _data(n_samples=17, n_features=3)
    X[0, 0] = bad_value
    problem = _problem(X, y)

    state = _run(problem, LBFGSConfig(max_iter=budget))

    assert not state.converged
    assert state.stop_reason == "non-finite-objective"
    assert state.n_iter == 0
    np.testing.assert_array_equal(np.asarray(state.x), np.zeros((1, 4)))


def test_final_gradient_is_obtained_from_the_supplied_true_objective():
    """A cached-logit gradient must not substitute for the final direct check."""
    X, y = _data()
    problem = _problem(X, y)
    original_fun = problem[-1]
    calls = []

    def checked_objective(theta):
        value, gradient = original_fun(theta)
        calls.append(np.asarray(theta).copy())
        return value, gradient

    state = _run((*problem[:-1], checked_objective), LBFGSConfig(max_iter=1))

    assert calls
    np.testing.assert_array_equal(calls[-1], np.asarray(state.x))
    value, gradient = original_fun(state.x)
    assert state.objective == pytest.approx(value, abs=1e-10)
    assert state.grad_max_abs == pytest.approx(float(mx.max(mx.abs(gradient))), abs=1e-7)


def test_true_objective_exception_propagates():
    X, y = _data(n_samples=17, n_features=3)
    problem = _problem(X, y)
    error = RuntimeError("direct objective evaluation failed")

    def broken_objective(theta):
        raise error

    with pytest.raises(RuntimeError, match="direct objective evaluation failed") as caught:
        _run((*problem[:-1], broken_objective), LBFGSConfig(max_iter=0))
    assert caught.value is error


def test_device_tolerance_never_rounds_above_the_requested_tolerance(monkeypatch):
    """A threshold rounded upward can produce a zero-step reanchor loop."""
    tolerance = 1e-3
    rounded = np.float32(tolerance)
    assert float(rounded) > tolerance
    X = np.array([[2 * rounded]], dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    problem = _problem(X, y, C=math.inf, fit_intercept=False)
    original_step = native_logistic.step
    observed_tolerances = []

    def record_tolerance(*args):
        observed_tolerances.append(float(args[5]))
        return original_step(*args)

    monkeypatch.setattr(native_logistic, "step", record_tolerance)
    state = _run(problem, LBFGSConfig(max_iter=1, tol=tolerance))

    assert observed_tolerances
    assert all(value <= tolerance for value in observed_tolerances)
    assert state.n_iter == 1
    assert state.converged
    assert state.grad_max_abs <= tolerance


def test_large_finite_device_tolerance_does_not_overflow(monkeypatch):
    X, y = _data(n_samples=17, n_features=3)
    problem = _problem(X, y)
    original_step = native_logistic.step
    observed_tolerances = []

    def record_tolerance(*args):
        observed_tolerances.append(float(args[5]))
        return original_step(*args)

    monkeypatch.setattr(native_logistic, "step", record_tolerance)
    with np.errstate(over="raise"):
        state = _run(problem, LBFGSConfig(max_iter=1, tol=1e40))
    assert observed_tolerances
    assert all(value == float(np.finfo(np.float32).max) for value in observed_tolerances)
    assert state.converged and state.n_iter == 0


def test_repeated_reanchor_without_an_accepted_step_reports_failure(monkeypatch):
    """A stale device convergence signal cannot keep retrying indefinitely."""
    X, y = _data(n_samples=17, n_features=3)
    problem = _problem(X, y)
    calls = 0

    def direct_objective(theta):
        nonlocal calls
        calls += 1
        # Bound the test even if the stagnation guard regresses.
        assert calls <= 2, "reanchoring repeated without any accepted step"
        return 1.0, mx.full(theta.shape, 0.25)

    def stale_convergence(*args):
        state = args[-9:]
        metadata = mx.concatenate([state[7][:1], mx.ones(1), state[7][2:]])
        return (*state[:7], metadata, state[8])

    monkeypatch.setattr(native_logistic, "step", stale_convergence)
    state = _run((*problem[:-1], direct_objective), LBFGSConfig(max_iter=10, tol=0.1))

    assert calls == 2
    assert state.n_iter == 0
    assert not state.converged
    assert state.stop_reason == "line-search-failure"
    assert state.objective == 1.0
    assert state.grad_max_abs == 0.25
    np.testing.assert_array_equal(np.asarray(state.x), np.zeros((1, 4)))


def test_rejected_nonfinite_projection_keeps_the_last_finite_iterate():
    """Zero times an infinite direction must never corrupt a rejected step."""
    w = mx.array([1.0, 2.0])
    raw = mx.array([3.0, 4.0, 5.0])
    stats = mx.array([7.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    neww, newraw, residual, metadata = native_logistic._accept(
        inputs=[
            w, raw, mx.full((2,), np.inf), mx.full((3,), np.inf),
            mx.zeros(3), mx.ones(3), stats, mx.array([np.inf]), mx.array(1e-4),
        ],
        template=[("N", 3), ("P", 2), ("CAND", 1)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(2,), (3,), (3,), (12,)],
        output_dtypes=[mx.float32] * 4,
    )
    mx.eval(neww, newraw, residual, metadata)

    np.testing.assert_array_equal(np.asarray(neww), np.asarray(w))
    np.testing.assert_array_equal(np.asarray(newraw), np.asarray(raw))
    assert np.isfinite(np.asarray(residual)).all()
    assert float(metadata[0]) == 7.0
    assert float(metadata[1]) == 2.0


@pytest.mark.parametrize("n_samples", [1, 3, 259])
def test_gpu_centering_preserves_extreme_finite_constant_columns(n_samples):
    limit = np.finfo(np.float32).max
    row = np.array([limit, -limit, 0.0, 1e30, -1e30], dtype=np.float32)
    X = np.tile(row, (n_samples, 1))
    Xm = mx.array(X)

    centered, shift, finite, _ = prepare_centered(Xm)
    mx.eval(centered, shift, finite)

    assert bool(finite)
    assert np.isfinite(np.asarray(centered)).all()
    assert np.isfinite(np.asarray(shift)).all()
    np.testing.assert_array_equal(np.asarray(centered), np.zeros_like(X))
    np.testing.assert_array_equal(np.asarray(shift), row)
    np.testing.assert_array_equal(np.asarray(Xm), X)


@pytest.mark.parametrize("offset", [0.0, 1e5, -1e5])
def test_gpu_centering_matches_mean_and_applied_shift_without_mutation(offset):
    X, _ = _data(n_samples=259, n_features=37)
    X += np.float32(offset)
    Xm = mx.array(X)
    expected_mean = X.mean(axis=0, dtype=np.float64)

    centered, shift, _, _ = prepare_centered(Xm)
    mx.eval(centered, shift)
    actual_shift = np.asarray(shift)
    actual_centered = np.asarray(centered)

    if offset:
        # A rounded float32 mean cannot be closer than half its local spacing.
        tolerance = np.max(np.abs(np.spacing(expected_mean.astype(np.float32))))
    else:
        tolerance = 8 * np.finfo(np.float32).eps * np.max(np.abs(X))
    np.testing.assert_allclose(actual_shift, expected_mean, atol=tolerance, rtol=0)
    np.testing.assert_array_equal(actual_centered, X - actual_shift)
    np.testing.assert_array_equal(np.asarray(Xm), X)
    assert np.max(np.abs(actual_centered.mean(axis=0, dtype=np.float64))) <= tolerance


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("C", [1.0, math.inf])
def test_native_direct_objective_matches_generic_for_tail_blocks(weighted, fit_intercept, C):
    X, y = _data(n_samples=259, n_features=37)
    weights = np.linspace(0.0, 2.0, len(X), dtype=np.float32) if weighted else None
    problem = _problem(X, y, weights=weights, C=C, fit_intercept=fit_intercept)
    Xm, ym, sw, sw_sum, l2, _, generic = problem
    native = _native_binary_objective(Xm, ym, sw, sw_sum, l2, fit_intercept)
    rng = np.random.default_rng(83)
    theta = mx.array(
        rng.normal(scale=0.2, size=(1, X.shape[1] + int(fit_intercept))).astype(np.float32)
    )

    value, gradient = native(theta)
    expected_value, expected_gradient = generic(theta)

    # Two independent float32 reductions can differ by more than either one's
    # rounding error, especially across GPU generations. Check both against
    # the represented input and parameter values in float64. Two float32 ULPs
    # at the loss magnitude permit the observed reduction rounding while still
    # detecting an incorrect objective or penalty.
    represented = np.asarray(theta, dtype=np.float64)
    coef = represented[:, :X.shape[1]]
    intercept = represented[:, -1] if fit_intercept else np.zeros(1)
    exact = reference_objective(
        coef, intercept, X, y.astype(int), n_classes=2, C=C, sample_weight=weights,
    )
    objective_atol = 2 * abs(float(np.spacing(np.float32(exact))))
    assert value == pytest.approx(exact, abs=objective_atol, rel=0)
    assert expected_value == pytest.approx(exact, abs=objective_atol, rel=0)
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(expected_gradient), atol=1e-7, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(gradient)[0], _float64_gradient(theta, X, y, weights, l2, fit_intercept),
        atol=1e-7, rtol=1e-6,
    )


def test_native_direct_unpenalized_large_coefficients_do_not_square_to_nan():
    fun = _native_binary_objective(mx.zeros((4, 1)), mx.zeros(4), None, 4.0, 0.0, False)

    value, gradient = fun(mx.full((1, 1), 1e20))

    assert value == pytest.approx(np.log(2), rel=1e-6)
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros((1, 1)))


def test_native_direct_positive_tail_retains_small_loss_and_gradient():
    fun = _native_binary_objective(mx.ones((1, 1)), mx.ones(1), None, 1.0, 0.0, False)

    value, gradient = fun(mx.array([[20.0]]))

    assert value == pytest.approx(np.logaddexp(0.0, -20.0), rel=1e-5)
    assert float(gradient[0, 0]) == pytest.approx(-expit(-20.0), rel=1e-5)


def test_adaptive_line_search_uses_the_remaining_candidate_budget():
    """A rank-one design requires smaller steps than the first eight trials."""
    rng = np.random.default_rng(35)
    column = rng.normal(size=259).astype(np.float32)
    X = np.tile(column[:, None], (1, 1024))
    y = rng.integers(0, 2, size=259).astype(np.float32)
    problem = _problem(X, y)

    limited = _run(problem, LBFGSConfig(max_iter=1, max_line_search=8))
    continued = _run(problem, LBFGSConfig(max_iter=1, max_line_search=32))

    assert limited.n_iter == 0
    assert limited.n_fev == 9
    assert limited.stop_reason == "line-search-failure"
    assert continued.n_iter == 1
    assert continued.n_fev == 33
    assert continued.objective < limited.objective
    assert np.isfinite(np.asarray(continued.x)).all()


def test_convergence_at_chunk_boundary_does_not_schedule_an_extra_chunk(monkeypatch):
    X, y = _data(n_samples=17, n_features=3)
    problem = _problem(X, y)
    step_calls = 0

    def counted_step(*args):
        nonlocal step_calls
        step_calls += 1
        state = args[-9:]
        # Emulate a gradient reaching tolerance on the chunk's last update,
        # before another direction kernel has marked the device status.
        gradient = mx.zeros_like(state[1]) if step_calls >= 4 else mx.ones_like(state[1])
        metadata = mx.concatenate([
            mx.array([float(min(step_calls, 4)), float(step_calls > 4)]), state[7][2:]
        ])
        return (state[0], gradient, *state[2:7], metadata, state[8])

    def direct_objective(theta):
        return 0.5, mx.zeros_like(theta)

    monkeypatch.setattr(native_logistic, "step", counted_step)
    state = _run((*problem[:-1], direct_objective), LBFGSConfig(max_iter=100))

    assert step_calls == 4
    assert state.n_iter == 4
    assert state.converged
    assert state.stop_reason == "gtol"
