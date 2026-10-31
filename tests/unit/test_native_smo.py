"""Native SMO correctness at the precision and buffer boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from mlxlearn._solvers import native_smo
from mlxlearn._solvers.smo import SMOConfig
from mlxlearn._solvers.smo import solve_c_svc as portable_solve

native = pytest.importorskip("mlxlearn._solvers._smo_native")


def problem(kernel="rbf", n=41, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 7)).astype(np.float32).astype(np.float64)
    X[-2:] = X[:2]  # Equal rows exercise zero/negative curvature handling.
    dot = X @ X.T
    if kernel == "linear":
        K = dot
    elif kernel == "rbf":
        sq = np.sum(X * X, axis=1)
        K = np.exp(-0.11 * np.maximum(sq[:, None] + sq[None, :] - 2 * dot, 0))
    elif kernel == "poly":
        K = (0.09 * dot - 0.1) ** 3
    else:
        K = np.tanh(0.2 * dot + 0.3)
    y = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    bounds = rng.uniform(0.1, 2.0, n)
    return y, bounds, np.diag(K).copy(), K.astype(np.float32)


def gap_and_feasibility(state, y, bounds, gram):
    alpha = state.alpha
    assert np.min(alpha) >= -2e-14
    assert np.max(alpha - bounds) <= 2e-14
    assert abs(alpha @ y) < 2e-12
    upper = alpha >= bounds
    lower = ~upper & (alpha <= 0)
    up = np.where(y > 0, ~upper, ~lower)
    low = np.where(y > 0, ~lower, ~upper)
    yg = gram.astype(np.float64) @ (alpha * y) - y
    gap = np.max(-yg[up]) + np.max(yg[low]) if up.any() and low.any() else 0.0
    assert state.max_kkt_violation == pytest.approx(gap, abs=3e-12)
    return gap


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
@pytest.mark.parametrize("cache_rows", [2, 9, 41])
def test_native_matches_portable_and_independent_kkt(kernel, cache_rows):
    y, bounds, diagonal, gram = problem(kernel)
    config = SMOConfig(tol=1e-7, max_iter=40000, cache_rows=cache_rows)
    reference = portable_solve(y, bounds, diagonal, gram.__getitem__, config)
    state = native_smo.solve_c_svc(
        y, bounds, diagonal, gram.__getitem__, config,
        gram=gram if cache_rows == len(y) else None,
    )
    assert state.n_iter == reference.n_iter
    assert state.converged == reference.converged
    np.testing.assert_allclose(state.alpha, reference.alpha, rtol=0, atol=2e-13)
    assert state.rho == pytest.approx(reference.rho, abs=1e-13)
    gap = gap_and_feasibility(state, y, bounds, gram)
    if state.converged:
        assert gap < config.tol + 3e-12


@pytest.mark.parametrize("budget", [0, 1, 7])
def test_budget_reports_current_gap_including_zero(budget):
    y, bounds, diagonal, gram = problem()
    config = SMOConfig(tol=1e-12, max_iter=budget, cache_rows=2)
    state = native_smo.solve_c_svc(y, bounds, diagonal, None, config, gram=gram)
    assert state.n_iter == budget
    assert not state.converged
    gap_and_feasibility(state, y, bounds, gram)
    if budget == 0:
        np.testing.assert_array_equal(state.alpha, 0)
        assert state.max_kkt_violation == 2.0


def test_native_lru_owns_rows_even_when_callback_reuses_scratch():
    y, bounds, diagonal, gram = problem(n=31)
    scratch = np.empty(31, np.float32)
    calls = []

    def fetch(i):
        calls.append(i)
        scratch[:] = gram[i]
        return scratch

    config = SMOConfig(tol=1e-8, max_iter=10000, cache_rows=2)
    actual = native_smo.solve_c_svc(y, bounds, diagonal, fetch, config)
    expected = native_smo.solve_c_svc(y, bounds, diagonal, None, config, gram=gram)
    np.testing.assert_array_equal(actual.alpha, expected.alpha)
    assert actual.rho == expected.rho
    assert calls
    assert len(calls) <= 2 * actual.n_iter


def test_zero_bounds_retain_upper_bound_precedence():
    y, bounds, diagonal, gram = problem(n=17)
    bounds[0] = bounds[-1] = 0
    config = SMOConfig(tol=1e-9, max_iter=25, cache_rows=2)
    expected = portable_solve(y, bounds, diagonal, gram.__getitem__, config)
    actual = native_smo.solve_c_svc(y, bounds, diagonal, None, config, gram=gram)
    np.testing.assert_allclose(actual.alpha, expected.alpha, atol=2e-14, rtol=0)
    assert actual.n_iter == expected.n_iter
    gap_and_feasibility(actual, y, bounds, gram)


def raw_args():
    y, bounds, diagonal, gram = problem(n=9)
    return [y, bounds, diagonal, np.empty_like(y), gram, None, 1e-3, 1000, 2, 1e-12]


@pytest.mark.parametrize("argument,value", [(6, 0.0), (6, np.inf), (6, np.nan), (7, -1), (8, 1), (9, 0), (9, np.inf)])
def test_rejects_invalid_config(argument, value):
    args = raw_args()
    args[argument] = value
    with pytest.raises(ValueError):
        native.solve(*args)


@pytest.mark.parametrize("argument,value", [(0, 0.0), (0, np.nan), (1, -1.0), (1, np.inf), (2, np.nan), (4, np.inf)])
def test_rejects_invalid_numeric_inputs_before_writing(argument, value):
    args = raw_args()
    args[3].fill(123.0)
    args[argument].flat[0] = value
    with pytest.raises(ValueError):
        native.solve(*args)
    np.testing.assert_array_equal(args[3], 123.0)


def test_rejects_output_alias():
    args = raw_args()
    args[3] = args[0]
    with pytest.raises(ValueError, match="overlap"):
        native.solve(*args)


@pytest.mark.parametrize("mode", ["readonly", "strided", "wrong_dtype", "unaligned", "wrong_shape", "wrong_endian"])
def test_rejects_invalid_buffers(mode):
    args = raw_args()
    if mode == "readonly":
        args[3].flags.writeable = False
    elif mode == "strided":
        args[0] = np.repeat(args[0], 2)[::2]
    elif mode == "wrong_dtype":
        args[0] = args[0].astype(np.float32)
    elif mode == "unaligned":
        args[0] = np.ndarray((9,), buffer=bytearray(9 * 8 + 1), offset=1, dtype=np.float64)
    elif mode == "wrong_shape":
        args[0] = args[0][:, None]
    else:
        args[0] = args[0].astype(">f8")
    with pytest.raises((ValueError, BufferError)):
        native.solve(*args)


@pytest.mark.parametrize("mode", ["short", "nan", "dtype", "strided", "raises"])
def test_callback_failure_propagates(mode):
    args = raw_args()
    gram = args[4]
    args[4] = None

    def fetch(i):
        row = gram[i].copy()
        if mode == "short":
            return row[:-1]
        if mode == "nan":
            row[0] = np.nan
        if mode == "dtype":
            return row.astype(np.float64)
        if mode == "strided":
            return np.repeat(row, 2)[::2]
        if mode == "raises":
            raise RuntimeError("row source failed")
        return row

    args[5] = fetch
    with pytest.raises((ValueError, BufferError, RuntimeError)):
        native.solve(*args)


def test_missing_extension_signals_portable_fallback(monkeypatch):
    monkeypatch.setattr(native_smo, "_smo_native", None)
    assert not native_smo.available()
    assert native_smo.solve_c_svc(None, None, None, None, SMOConfig()) is None


def test_small_positive_curvature_is_not_replaced_by_tau():
    # Positive eta is meaningful even below TAU: LIBSVM substitutes TAU only
    # for non-positive curvature. Replacing this eta changes the step 100-fold.
    y = np.array([1.0, -1.0])
    diagonal = np.full(2, 5e-15)
    gram = np.diag(diagonal).astype(np.float32)
    state = native_smo.solve_c_svc(
        y, np.full(2, 1e15), diagonal, None,
        SMOConfig(max_iter=1, tau=1e-12), gram=gram,
    )
    np.testing.assert_allclose(state.alpha, 2e14, rtol=1e-15)
    gap_and_feasibility(state, y, np.full(2, 1e15), gram)


def test_last_tied_pair_is_selected():
    y = np.array([1.0, -1.0, 1.0, -1.0])
    state = native_smo.solve_c_svc(
        y, np.full(4, 2.0), np.ones(4), None,
        SMOConfig(max_iter=1), gram=np.eye(4, dtype=np.float32),
    )
    np.testing.assert_array_equal(state.alpha, [0, 0, 1, 1])
