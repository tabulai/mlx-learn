"""Shrinking preserves the full C-SVC problem and honest stopping diagnostics."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mlxlearn._solvers import native_smo
from mlxlearn._solvers.smo import SMOConfig

native = pytest.importorskip("mlxlearn._solvers._smo_native")


def _problem(kernel="linear", seed=19, n=121):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6)).astype(np.float32).astype(np.float64)
    # More bound points than free margin points gives shrinking real work.
    y = np.where(x[:, 0] + 0.5 * x[:, 1] > 0, 1.0, -1.0)
    x[-4:] = x[:4]
    y[-4:] = y[:4]
    bounds = rng.uniform(0.05, 3.0, n)
    dot = x @ x.T
    if kernel == "linear":
        raw = dot
    elif kernel == "rbf":
        squared = np.sum(x * x, axis=1)
        raw = np.exp(-0.13 * np.maximum(squared[:, None] + squared[None, :] - 2 * dot, 0))
    elif kernel == "poly":
        raw = (0.15 * dot - 0.2) ** 3
    else:
        raw = np.tanh(0.2 * dot + 0.3)
    return y, bounds, np.diag(raw).copy(), raw.astype(np.float32)


def _solve(problem, *, shrinking=True, max_iter=100000, cache_rows=None, tol=1e-6):
    y, bounds, diagonal, gram = problem
    config = SMOConfig(tol=tol, max_iter=max_iter,
                       cache_rows=len(y) if cache_rows is None else cache_rows,
                       shrinking=shrinking)
    return native_smo.solve_c_svc(y, bounds, diagonal, gram.__getitem__, config,
                                  gram=gram if cache_rows is None else None)


def _check_full_result(state, problem, *, tol=None):
    y, bounds, _, gram = problem
    alpha = state.alpha
    np.testing.assert_array_less(-2e-13, alpha)
    np.testing.assert_array_less(alpha - bounds, 2e-13)
    assert abs(alpha @ y) < 2e-11
    # Row orientation is intentional: updates add row_i to the whole gradient.
    yg = gram.astype(np.float64).T @ (alpha * y) - y
    upper = alpha >= bounds
    lower = ~upper & (alpha <= 0)
    up = np.where(y > 0, ~upper, ~lower)
    low = np.where(y > 0, ~lower, ~upper)
    gap = -yg[up].min() + yg[low].max() if up.any() and low.any() else 0.0
    assert state.max_kkt_violation == pytest.approx(gap, abs=2e-11)
    free = ~upper & ~lower
    if free.any():
        rho = yg[free].mean()
    else:
        upper_side = (upper & (y < 0)) | (lower & (y > 0))
        lower_side = (upper & (y > 0)) | (lower & (y < 0))
        ub = yg[upper_side].min() if upper_side.any() else np.inf
        lb = yg[lower_side].max() if lower_side.any() else -np.inf
        rho = (ub + lb) / 2
    assert state.rho == pytest.approx(rho, abs=2e-11)
    if tol is not None:
        assert state.converged
        assert gap < tol + 2e-11
    signed = alpha * y
    return 0.5 * signed @ gram.astype(np.float64) @ signed - alpha.sum()


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
@pytest.mark.parametrize("seed", [19, 37])
def test_shrinking_checks_full_gradient_and_weighted_dual_objective(kernel, seed):
    problem = _problem(kernel, seed)
    shrunk = _solve(problem)
    plain = _solve(problem, shrinking=False)
    value = _check_full_result(shrunk, problem, tol=1e-6)
    reference = _check_full_result(plain, problem, tol=1e-6)
    if kernel in ("linear", "rbf"):
        assert value == pytest.approx(reference, abs=5e-6, rel=3e-7)
    else:
        # Indefinite kernels need not share a local optimum; both states must
        # satisfy full KKT and improve the zero dual objective.
        assert value <= 0
        assert reference <= 0


@pytest.mark.parametrize("budget", [0, 1, 121, 122, 200])
def test_budget_exhaustion_reconstructs_every_coordinate(budget):
    problem = _problem()
    state = _solve(problem, max_iter=budget, tol=1e-12)
    assert state.n_iter == budget
    assert not state.converged
    _check_full_result(state, problem)
    if budget == 0:
        assert state.max_kkt_violation == 2.0
        np.testing.assert_array_equal(state.alpha, 0)


@pytest.mark.parametrize("cache_rows", [2, 9])
def test_reconstruction_with_bounded_callback_cache_matches_dense(cache_rows):
    problem = _problem("linear", n=61)
    dense = _solve(problem, tol=1e-5)
    bounded = _solve(problem, cache_rows=cache_rows, tol=1e-5)
    np.testing.assert_array_equal(bounded.alpha, dense.alpha)
    assert bounded.rho == dense.rho
    assert bounded.n_iter == dense.n_iter
    _check_full_result(bounded, problem, tol=1e-5)


def test_last_budget_reconstruction_respects_asymmetric_stored_rows():
    y, bounds, diagonal, gram = _problem(n=61)
    gram[0, 1] = np.nextafter(gram[0, 1], np.float32(np.inf))
    problem = y, bounds, diagonal, gram
    state = _solve(problem, max_iter=200, tol=1e-12)
    assert state.n_iter == 200
    _check_full_result(state, problem)


def test_shrinking_keeps_zero_box_upper_bound_precedence():
    y, bounds, diagonal, gram = _problem(n=61)
    bounds[0] = bounds[-1] = 0
    problem = y, bounds, diagonal, gram
    state = _solve(problem, max_iter=130, tol=1e-12)
    _check_full_result(state, problem)
    assert state.alpha[0] == state.alpha[-1] == 0


def test_old_raw_signature_and_explicit_false_are_identical():
    y, bounds, diagonal, gram = _problem(n=31)
    alpha_implicit = np.empty_like(y)
    alpha_false = np.empty_like(y)
    implicit = native.solve(y, bounds, diagonal, alpha_implicit, gram, None,
                            1e-6, 10000, 31, 1e-12)
    explicit = native.solve(y, bounds, diagonal, alpha_false, gram, None,
                            1e-6, 10000, 31, 1e-12, False)
    assert explicit == implicit
    np.testing.assert_array_equal(alpha_implicit, alpha_false)


def test_wrapper_accepts_config_without_shrinking_field():
    y, bounds, diagonal, gram = _problem(n=31)
    config = SimpleNamespace(tol=1e-6, max_iter=10000, cache_rows=31, tau=1e-12)
    actual = native_smo.solve_c_svc(y, bounds, diagonal, None, config, gram=gram)
    expected = _solve((y, bounds, diagonal, gram), shrinking=False, max_iter=10000)
    np.testing.assert_array_equal(actual.alpha, expected.alpha)
    assert actual.rho == expected.rho


def test_small_budget_before_first_screen_retains_pair_updates():
    problem = _problem(n=61)
    config = SMOConfig(tol=1e-8, max_iter=10, cache_rows=61)
    y, bounds, diagonal, gram = problem
    plain = native_smo.solve_c_svc(y, bounds, diagonal, None, config, gram=gram)
    shrunk = native_smo.solve_c_svc(y, bounds, diagonal, None,
                                   replace(config, shrinking=True), gram=gram)
    np.testing.assert_array_equal(plain.alpha, shrunk.alpha)
    assert plain.n_iter == shrunk.n_iter == 10
    _check_full_result(shrunk, problem)
