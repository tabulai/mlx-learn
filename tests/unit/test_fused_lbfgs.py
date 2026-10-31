"""GPU optimizer contracts that estimator-level parity does not exercise."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers.fused_lbfgs import minimize
from mlxlearn._solvers.lbfgs import LBFGSConfig

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _gpu_stream():
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    with use_device("gpu"):
        yield


def _quadratic(x):
    return float(mx.sum(x * x)), 2.0 * x


def test_zero_budget_evaluates_start_without_taking_a_step():
    """An evaluation budget of zero must never perform an optimizer update."""
    x0 = mx.array([[1.0, -2.0], [3.0, -4.0]], dtype=mx.float32)
    calls = []

    def objective(x):
        calls.append(np.asarray(x).copy())
        return _quadratic(x)

    state = minimize(objective, x0, LBFGSConfig(max_iter=0))

    assert len(calls) == state.n_fev == 1
    assert state.n_iter == 0
    assert not state.converged
    assert state.stop_reason == "max_iter"
    assert state.objective == pytest.approx(30.0)
    assert state.grad_max_abs == pytest.approx(8.0)
    np.testing.assert_array_equal(np.asarray(state.x), np.asarray(x0))


@pytest.mark.parametrize("budget", [0, 10])
def test_initial_optimum_is_converged_without_a_step(budget):
    state = minimize(_quadratic, mx.zeros((2, 3)), LBFGSConfig(max_iter=budget))

    assert state.converged
    assert state.stop_reason == "gtol"
    assert state.n_iter == 0
    assert state.n_fev == 1
    assert state.grad_max_abs == 0.0
    assert state.objective == 0.0


def test_matrix_parameter_layout_and_input_survive_optimizer():
    """Compiled updates must preserve the class/feature axes and caller's input."""
    target_np = np.arange(12, dtype=np.float32).reshape(3, 4) / 7.0
    target = mx.array(target_np)
    x0 = mx.full((3, 4), -2.0, dtype=mx.float32)
    initial = np.asarray(x0).copy()
    seen_shapes = []

    def objective(x):
        seen_shapes.append(x.shape)
        delta = x - target
        return float(mx.sum(delta * delta)), 2.0 * delta

    state = minimize(objective, x0, LBFGSConfig(tol=1e-5))

    assert state.converged
    assert state.stop_reason == "gtol"
    assert state.grad_max_abs <= 1e-5
    assert state.x.shape == (3, 4)
    assert set(seen_shapes) == {(3, 4)}
    np.testing.assert_allclose(np.asarray(state.x), target_np, atol=5e-6, rtol=0)
    np.testing.assert_array_equal(np.asarray(x0), initial)
    value, gradient = objective(state.x)
    assert state.objective == pytest.approx(value, abs=1e-12)
    assert state.grad_max_abs == pytest.approx(float(mx.max(mx.abs(gradient))), abs=1e-7)


def test_iteration_budget_returns_finite_unconverged_progress():
    scale = mx.array([1.0, 100.0])
    x0 = mx.array([4.0, 4.0])

    def objective(x):
        return float(mx.sum(scale * x * x)), 2.0 * scale * x

    state = minimize(objective, x0, LBFGSConfig(max_iter=1, tol=1e-8))

    assert state.n_iter == 1
    assert state.n_fev >= 2
    assert not state.converged
    assert state.stop_reason == "max_iter"
    assert state.grad_max_abs > 1e-8
    assert 0.0 < state.objective < objective(x0)[0]
    assert np.isfinite(np.asarray(state.x)).all()


def test_curvature_history_rollover_reaches_anisotropic_minimum():
    """Repeated compiled updates must retain the correct order of history pairs."""
    weights_np = np.geomspace(1.0, 100.0, 24).astype(np.float32).reshape(3, 8)
    weights = mx.array(weights_np)
    config = LBFGSConfig(max_iter=300, tol=1e-4, history_size=4)

    def objective(x):
        # Preserve loss resolution while exercising the GPU gradient/update path.
        terms = weights * x * x
        gradient = 2.0 * weights * x
        mx.eval(terms, gradient)
        return float(np.asarray(terms).astype(np.float64).sum()), gradient

    state = minimize(objective, mx.ones((3, 8)), config)

    assert state.n_iter > config.history_size
    assert state.converged
    assert state.stop_reason == "gtol"
    assert state.grad_max_abs <= config.tol
    np.testing.assert_allclose(np.asarray(state.x), 0.0, atol=5e-5, rtol=0)


@pytest.mark.parametrize("module_name", ["fused_lbfgs", "lbfgs"])
def test_failed_quasi_newton_search_retries_along_the_gradient(monkeypatch, module_name):
    """One failed search with a history clears it and retries steepest descent;
    only a second failure at the same point ends the run."""
    from mlxlearn._solvers import fused_lbfgs, lbfgs

    module = fused_lbfgs if module_name == "fused_lbfgs" else lbfgs
    original = lbfgs._LineSearch
    searches = []
    budgets = []

    class FailingOnce(original):
        def __init__(self, probe, f0, gtd0, config):
            super().__init__(probe, f0, gtd0, config)
            searches.append(gtd0)
            budgets.append(config.max_line_search)

        def run(self, t_init):
            # The fourth search of the run follows three accepted steps, so it
            # carries a quasi-Newton direction. Spend one evaluation before
            # failing, then let every later search behave normally.
            if len(searches) == 4:
                x_t, f_t, g_t, _ = self._evaluate(0.0)
                return False, 0.0, x_t, f_t, g_t, False
            return super().run(t_init)

    monkeypatch.setattr(module, "_LineSearch", FailingOnce)
    scale = mx.array([1.0, 25.0, 0.04, 9.0], dtype=mx.float32)

    def objective(x):
        return float(mx.sum(scale * x * x)), 2.0 * scale * x

    config = LBFGSConfig(max_iter=60, tol=1e-5, history_size=3)
    state = module.minimize(objective, mx.array([1.0, -1.0, 2.0, 0.5]), config)

    assert state.converged and state.stop_reason == "gtol"
    # Three accepted steps, one rejected search, its gradient retry, and the
    # searches of the remaining accepted steps: exactly one extra construction.
    assert len(searches) == state.n_iter + 1
    assert searches[4] < 0.0  # the retry descends along the gradient
    assert budgets[4] == config.max_line_search - 1
    assert all(budget == config.max_line_search for i, budget in enumerate(budgets) if i != 4)
    np.testing.assert_allclose(np.asarray(state.x), 0.0, atol=1e-3, rtol=0)


@pytest.mark.parametrize("module_name", ["fused_lbfgs", "lbfgs"])
def test_failed_search_cannot_exceed_the_iteration_evaluation_budget(module_name):
    """A real failed quasi-Newton search must not buy a second trial allowance."""
    from mlxlearn._solvers import fused_lbfgs, lbfgs

    module = fused_lbfgs if module_name == "fused_lbfgs" else lbfgs
    rng = np.random.default_rng(86)
    matrix = rng.normal(size=(4, 4))
    hessian = mx.array((100.0 * (matrix.T @ matrix + 1e-5 * np.eye(4))).astype(np.float32))
    x0 = mx.array(rng.normal(size=4).astype(np.float32))
    calls = 0

    def objective(x):
        nonlocal calls
        calls += 1
        gradient = hessian @ x
        return float(0.5 * mx.sum(x * gradient)), gradient

    config = LBFGSConfig(max_iter=2, max_line_search=1, tol=0)
    state = module.minimize(objective, x0, config)

    assert state.n_fev == calls
    assert calls <= 1 + config.max_iter * config.max_line_search
    assert state.stop_reason == "line-search-failure"
    assert state.n_iter == 1
    assert not state.converged


def test_float32_function_plateau_is_not_reported_as_convergence():
    """Equal rounded losses cannot establish gradient convergence."""

    def objective(x):
        value = mx.array(1e8, dtype=mx.float32) + mx.sum(x * x)
        return float(value), 2.0 * x

    state = minimize(
        objective,
        mx.array([0.25, 0.5]),
        LBFGSConfig(max_iter=10, max_line_search=4, tol=1e-6),
    )

    assert not state.converged
    assert state.stop_reason == "line-search-failure"
    assert state.grad_max_abs > 1e-6
    assert np.isfinite(np.asarray(state.x)).all()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_initial_objective_stops_before_optimization(bad_value):
    state = minimize(
        lambda x: (bad_value, 2.0 * x),
        mx.array([1.0]),
        LBFGSConfig(),
    )

    assert not state.converged
    assert state.stop_reason == "non-finite-objective"
    assert state.n_iter == 0
    assert state.n_fev == 1


def test_nonfinite_initial_gradient_cannot_claim_convergence():
    state = minimize(
        lambda x: (1.0, mx.full(x.shape, float("nan"))),
        mx.array([1.0]),
        LBFGSConfig(),
    )

    assert not state.converged
    assert state.stop_reason == "non-finite-objective"
    assert state.n_iter == 0
    assert state.n_fev == 1


@pytest.mark.parametrize("raise_at", [1, 2])
def test_objective_exception_propagates_without_another_optimizer(raise_at):
    error = RuntimeError("device evaluation failed")
    calls = 0

    def objective(x):
        nonlocal calls
        calls += 1
        if calls == raise_at:
            raise error
        return _quadratic(x)

    with pytest.raises(RuntimeError, match="device evaluation failed") as caught:
        minimize(objective, mx.array([1.0]), LBFGSConfig())
    assert caught.value is error
