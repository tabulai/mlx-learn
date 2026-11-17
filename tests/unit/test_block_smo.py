"""Independent dual checks for the hierarchical native SVC optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from mlxlearn._solvers import block_smo
from mlxlearn._solvers.smo import SMOConfig
from mlxlearn._solvers.smo import solve_c_svc as reference_solve

native = pytest.importorskip("mlxlearn._solvers._smo_block_native")


def problem(kernel="rbf", n=71, seed=102):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 9)).astype(np.float32).astype(np.float64)
    X[-2:] = X[:2]
    dot = X @ X.T
    if kernel == "linear":
        K = dot
    elif kernel == "poly":
        K = (0.08 * dot + 0.2) ** 3
    elif kernel == "sigmoid":
        K = np.tanh(0.2 * dot + 0.3)
    else:
        sq = np.sum(X * X, axis=1)
        K = np.exp(-0.1 * np.maximum(sq[:, None] + sq[None, :] - 2 * dot, 0))
    y = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    bounds = rng.uniform(0.03, 2.0, n)
    return y, bounds, np.diag(K).copy(), K.astype(np.float32)


def diagnostics(state, y, bounds, gram):
    alpha = state.alpha
    assert np.min(alpha) >= -1e-13
    assert np.max(alpha - bounds) <= 1e-13
    assert abs(alpha @ y) < 3e-11
    movable = bounds > 0
    upper = alpha >= bounds
    lower = ~upper & (alpha <= 0)
    up = movable & np.where(y > 0, ~upper, ~lower)
    low = movable & np.where(y > 0, ~lower, ~upper)
    product = gram.astype(np.float64) @ (alpha * y)
    yg = product - y
    gap = float(np.max(-yg[up]) + np.max(yg[low])) if up.any() and low.any() else 0.0
    assert state.max_kkt_violation == pytest.approx(gap, abs=2e-11)
    free = (alpha > 0) & (alpha < bounds)
    if free.any():
        assert state.rho == pytest.approx(np.mean(yg[free]), abs=2e-11)
    return 0.5 * (alpha * y) @ product - alpha.sum(), gap


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
@pytest.mark.parametrize("width", [2, 8, 32, 128])
def test_matches_reference_objective_and_full_kkt(kernel, width):
    y, bounds, diagonal, gram = problem(kernel)
    config = SMOConfig(tol=1e-5, max_iter=100000)
    reference = reference_solve(y, bounds, diagonal, gram.__getitem__, config)
    state = block_smo.solve_c_svc(y, bounds, diagonal, None, config, gram=gram, working_set_size=width)
    assert state.converged
    assert state.n_iter <= config.max_iter
    objective, gap = diagnostics(state, y, bounds, gram)
    reference_objective = 0.5 * (reference.alpha * y) @ gram.astype(np.float64) @ (reference.alpha * y) - reference.alpha.sum()
    assert objective == pytest.approx(reference_objective, abs=3e-7, rel=3e-8)
    assert gap < config.tol + 2e-11


@pytest.mark.parametrize("budget", [0, 1, 3, 19])
@pytest.mark.parametrize("callback", [False, True])
def test_budget_counts_pair_updates_and_certifies_returned_state(budget, callback):
    y, bounds, diagonal, gram = problem(n=37)
    called = []

    def fetch(indices):
        called.append(list(indices))
        return gram[indices]

    state = block_smo.solve_c_svc(
        y, bounds, diagonal, fetch, SMOConfig(tol=1e-12, max_iter=budget),
        gram=None if callback else gram, working_set_size=8,
    )
    assert state.n_iter == budget
    assert not state.converged
    diagnostics(state, y, bounds, gram)
    if budget == 0:
        assert not called
        assert state.max_kkt_violation == 2
        assert state.rho == 0


def test_extra_cache_reuses_rows_without_changing_the_working_set_solution():
    y, bounds, diagonal, gram = problem(n=71)
    width = 8
    scratch = np.empty((width, len(y)), np.float32)
    config = SMOConfig(tol=1e-6, max_iter=100000)
    expected = block_smo.solve_c_svc(y, bounds, diagonal, None, config, gram=gram, working_set_size=width)
    assert expected.converged
    _, gap = diagnostics(expected, y, bounds, gram)
    assert gap < config.tol

    def solve_with_retained_rows(capacity):
        calls = []

        def fetch(indices):
            assert len(indices) <= width
            assert len(set(indices)) == len(indices)
            calls.append(list(indices))
            # The callback reuses its scratch. The solver must own the rows
            # if it will read them after a subsequent callback.
            scratch[:len(indices)] = gram[indices]
            return scratch[:len(indices)]

        config = SMOConfig(tol=1e-6, max_iter=100000, cache_rows=width + capacity)
        actual = block_smo.solve_c_svc(y, bounds, diagonal, fetch, config, working_set_size=width)
        assert actual.converged
        np.testing.assert_array_equal(actual.alpha, expected.alpha)
        assert actual.rho == expected.rho
        assert actual.n_iter == expected.n_iter
        assert actual.max_kkt_violation == expected.max_kkt_violation
        assert any(len(batch) > 1 for batch in calls)
        return [row for batch in calls for row in batch]

    small = solve_with_retained_rows(width)
    intermediate = solve_with_retained_rows(3 * width)
    complete = solve_with_retained_rows(len(y))
    assert set(small) == set(intermediate) == set(complete)
    assert len(complete) == len(set(complete))
    assert len(complete) < len(intermediate) < len(small)


def test_zero_boxes_are_excluded_from_feasible_directions():
    y, bounds, diagonal, gram = problem(n=41)
    bounds[::5] = 0
    state = block_smo.solve_c_svc(
        y, bounds, diagonal, None, SMOConfig(tol=1e-6, max_iter=100000),
        gram=gram, working_set_size=8,
    )
    assert state.converged
    np.testing.assert_array_equal(state.alpha[bounds == 0], 0)
    _, gap = diagnostics(state, y, bounds, gram)
    assert gap < 1e-6


def test_indefinite_kernel_retains_feasibility_and_actual_kkt():
    y, bounds, diagonal, gram = problem("sigmoid", n=41)
    state = block_smo.solve_c_svc(
        y, bounds, diagonal, None, SMOConfig(tol=1e-6, max_iter=20000),
        gram=gram, working_set_size=8,
    )
    _, gap = diagnostics(state, y, bounds, gram)
    if state.converged:
        assert gap < 1e-6


def test_tiny_positive_curvature_is_preserved():
    y = np.array([1., -1.])
    diagonal = np.full(2, 5e-15)
    gram = np.diag(diagonal).astype(np.float32)
    state = block_smo.solve_c_svc(
        y, np.full(2, 1e15), diagonal, None, SMOConfig(max_iter=1),
        gram=gram, working_set_size=2,
    )
    np.testing.assert_allclose(state.alpha, 2e14, rtol=1e-15)


def raw_args():
    y, bounds, diagonal, gram = problem(n=11)
    return [y, bounds, diagonal, np.empty_like(y), gram, None, 1e-3, 100, 4, 1e-12]


@pytest.mark.parametrize("argument,value", [(6, 0), (6, np.inf), (6, np.nan), (7, -1), (8, 1), (9, 0), (9, np.inf)])
def test_invalid_config(argument, value):
    args = raw_args()
    args[argument] = value
    with pytest.raises(ValueError):
        native.solve(*args)


@pytest.mark.parametrize("capacity", [-1, 3, 12])
def test_invalid_explicit_cache_capacity(capacity):
    args = raw_args()  # width 4, 11 samples
    args[5] = args[4].__getitem__
    args[4] = None
    with pytest.raises(ValueError):
        native.solve(*args, capacity)


@pytest.mark.parametrize("argument,value", [(0, 0), (0, np.nan), (1, -1), (1, np.inf), (2, np.nan), (4, np.inf)])
def test_invalid_data_before_output_write(argument, value):
    args = raw_args()
    args[3].fill(123)
    args[argument].flat[0] = value
    with pytest.raises(ValueError):
        native.solve(*args)
    np.testing.assert_array_equal(args[3], 123)


@pytest.mark.parametrize("mode", ["alias", "readonly", "unaligned", "strided", "wrong_dtype", "wrong_shape", "endian"])
def test_invalid_buffers(mode):
    args = raw_args()
    if mode == "alias":
        args[3] = args[0]
    elif mode == "readonly":
        args[3].flags.writeable = False
    elif mode == "unaligned":
        args[0] = np.ndarray((11,), buffer=bytearray(11 * 8 + 1), offset=1, dtype=np.float64)
    elif mode == "strided":
        args[0] = np.repeat(args[0], 2)[::2]
    elif mode == "wrong_dtype":
        args[0] = args[0].astype(np.float32)
    elif mode == "wrong_shape":
        args[0] = args[0][:, None]
    else:
        args[0] = args[0].astype(">f8")
    with pytest.raises((ValueError, BufferError)):
        native.solve(*args)


@pytest.mark.parametrize("mode", ["rows", "columns", "nan", "dtype", "strided", "raises"])
def test_bad_callback(mode):
    args = raw_args()
    gram = args[4]
    args[4] = None

    def fetch(indices):
        result = gram[indices]
        if mode == "rows":
            return result[:-1]
        if mode == "columns":
            return result[:, :-1].copy()
        if mode == "nan":
            result[0, 0] = np.nan
        if mode == "dtype":
            return result.astype(np.float64)
        if mode == "strided":
            return np.repeat(result, 2, axis=0)[::2]
        if mode == "raises":
            raise RuntimeError("source failed")
        return result

    args[5] = fetch
    with pytest.raises((ValueError, BufferError, RuntimeError)):
        native.solve(*args)


def test_optional_extension(monkeypatch):
    monkeypatch.setattr(block_smo, "_smo_block_native", None)
    assert not block_smo.available()
    assert block_smo.solve_c_svc(None, None, None, None, SMOConfig()) is None


def test_wrapper_can_use_an_older_working_set_extension(monkeypatch):
    class PreviousExtension:
        @staticmethod
        def solve(y, bounds, diag, alpha, gram, fetch, tol, budget, width, tau):
            return native.solve(y, bounds, diag, alpha, gram, fetch, tol, budget, width, tau)

    y, bounds, diagonal, gram = problem()
    config = SMOConfig(tol=1e-6, cache_rows=60)
    monkeypatch.setattr(block_smo, "_smo_block_native", PreviousExtension)
    result = block_smo.solve_c_svc(y, bounds, diagonal, gram.__getitem__, config, working_set_size=8)
    assert result.converged
    _, gap = diagnostics(result, y, bounds, gram)
    assert gap < config.tol


def test_callback_type_error_propagates_without_retry():
    y, bounds, diagonal, _ = problem()
    calls = 0

    def fetch(_indices):
        nonlocal calls
        calls += 1
        raise TypeError("kernel callback failed")

    with pytest.raises(TypeError, match="kernel callback failed"):
        block_smo.solve_c_svc(y, bounds, diagonal, fetch, SMOConfig(), working_set_size=8)
    assert calls == 1
