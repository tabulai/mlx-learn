"""Native attempts preserve iteration budgets and restart only from their state."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._solvers import native_command, native_logistic
from mlxlearn._solvers.lbfgs import LBFGSConfig, LBFGSState


@pytest.mark.parametrize(
    ("status", "n_iter", "grad", "reason"),
    [(0, 5, 2e-4, "max_iter"), (2, 2, 0.0, "line-search-failure"),
     (3, 2, 0.0, "non-finite-objective"), (1, 2, 2e-4, "line-search-failure")],
)
def test_attempt_returns_directly_scored_native_failure(monkeypatch, status, n_iter, grad, reason):
    with mx.stream(mx.cpu):
        n, d = 5, 512
        X, y = mx.ones((n, d), mx.float32), mx.zeros((n,), mx.float32)
        parts = mx.zeros((2, 1, d + 1), mx.float32)
        calls = []

        def run(*args):
            calls.append(args)
            return np.zeros(d + 1, dtype=np.float32).tobytes(), n_iter, status, 16, 0.0, 0.0

        def direct(theta):
            return 0.2, mx.full(theta.shape, grad, dtype=mx.float32)

        monkeypatch.setattr(native_command, "_supports_metal_dlpack", lambda: True)
        monkeypatch.setattr(native_command, "_source", lambda *args: "fake Metal source")
        monkeypatch.setattr(native_command, "_metal_loop", SimpleNamespace(run=run))
        attempt = native_command.run_native_loop(
            X, y, None, float(n), 0.1, True, LBFGSConfig(max_iter=5, tol=1e-4), direct,
            initial_parts=parts, return_attempt=True,
        )
        assert len(calls) == 1
        assert isinstance(attempt, native_command.NativeAttempt)
        assert attempt.status == status and not attempt.state.converged
        assert attempt.state.n_iter == n_iter and attempt.state.n_fev == 17
        assert attempt.state.stop_reason == reason
        np.testing.assert_allclose(attempt.gradient, grad)


@pytest.mark.parametrize(("status", "reason"), [(0, "max_iter"), (3, "non-finite-objective")])
def test_exhausted_or_nonfinite_native_attempt_does_not_restart(monkeypatch, status, reason):
    with mx.stream(mx.cpu):
        X, y = mx.ones((3, 2), mx.float32), mx.array([0, 1, 0], mx.float32)
        parts = mx.zeros((2, 1, 3), mx.float32)
        state = LBFGSState(mx.zeros((1, 3)), 0.2, 2e-4, 5 if status == 0 else 2, 20, False, reason)
        attempt = native_command.NativeAttempt(state, mx.ones((1, 3)), status)
        monkeypatch.setattr(native_command, "run_native_loop", lambda *a, **kw: attempt)
        monkeypatch.setattr(native_logistic, "initial_from_parts", lambda *a: pytest.fail("restart"))
        result = native_logistic.minimize_binary(
            X, y, None, 3.0, 0.1, True, LBFGSConfig(max_iter=5),
            lambda theta: (0.2, mx.ones(theta.shape)), initial_parts=parts,
        )
    assert result is state


@pytest.mark.parametrize("status", [1, 2])
def test_resumed_attempt_preserves_parameters_budget_and_evaluation_count(monkeypatch, status):
    """Reanchor only from accepted native coefficients, with no cached history."""
    with mx.stream(mx.cpu):
        X = mx.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], mx.float32)
        y = mx.array([0, 1, 0], mx.float32)
        parts = mx.zeros((2, 1, 3), mx.float32)
        accepted = mx.array([[0.25, 0.5, 0.7]], mx.float32)
        direct_gradient = mx.array([[0.02, 0.04, 0.01]], mx.float32)
        native_state = LBFGSState(accepted, 0.5, 0.04, 3, 25, False, "line-search-failure")
        native_attempt = native_command.NativeAttempt(native_state, direct_gradient, status)
        monkeypatch.setattr(native_command, "run_native_loop", lambda *a, **kw: native_attempt)
        monkeypatch.setattr(
            native_logistic, "initial_from_parts",
            lambda *a: (mx.zeros((3,), mx.float32), mx.ones((3,), mx.float32)),
        )
        calls = []

        def fake_step(*args):
            w, g, oldw, oldg, hs, hy, hc, meta, raw = args[12:]
            calls.append((np.asarray(w).copy(), np.asarray(g).copy(), np.asarray(oldw).copy(),
                          np.asarray(oldg).copy(), np.asarray(hs).copy(), np.asarray(hy).copy(),
                          np.asarray(hc).copy(), np.asarray(meta).copy(), np.asarray(raw).copy()))
            next_meta = np.asarray(meta).copy()
            next_meta[0] += 1
            next_meta[9] += 8
            next_meta[10] = -0.01
            next_meta[11] = 0.49
            return w, g, oldw, oldg, hs, hy, hc, mx.array(next_meta), raw

        def direct(theta):
            return 0.49, mx.full(theta.shape, 0.01, dtype=mx.float32)

        monkeypatch.setattr(native_logistic, "step", fake_step)
        result = native_logistic.minimize_binary(
            X, y, None, 3.0, 0.1, True, LBFGSConfig(max_iter=5, max_line_search=8),
            direct, initial_parts=parts,
        )
    assert len(calls) == 2  # 3 native accepted iterations leave exactly two.
    w, g, oldw, oldg, hs, hy, hc, meta, raw = calls[0]
    np.testing.assert_array_equal(w, np.asarray(accepted).ravel())
    np.testing.assert_array_equal(g, np.asarray(direct_gradient).ravel())
    np.testing.assert_array_equal(w, oldw)
    np.testing.assert_array_equal(g, oldg)
    assert not np.any(hs) and not np.any(hy) and not np.any(hc)
    assert meta[0] == 3 and meta[9] == 24 and np.isinf(meta[4]) and np.isinf(meta[10])
    np.testing.assert_allclose(raw, np.asarray(X) @ w[:-1] + w[-1])
    assert result.n_iter == 5 and result.n_fev == 42
    assert not result.converged and result.stop_reason == "max_iter"


def test_explicit_skip_never_calls_native_command(monkeypatch):
    with mx.stream(mx.cpu):
        X, y = mx.ones((3, 2), mx.float32), mx.array([0, 1, 0], mx.float32)
        parts = mx.zeros((2, 1, 3), mx.float32)
        monkeypatch.setattr(native_command, "run_native_loop", lambda *a, **kw: pytest.fail("native"))
        monkeypatch.setattr(
            native_logistic, "initial_from_parts",
            lambda *a: (mx.zeros((3,), mx.float32), mx.ones((3,), mx.float32)),
        )
        def fake_step(*args):
            w, g, oldw, oldg, hs, hy, hc, meta, raw = args[12:]
            metadata = np.asarray(meta).copy()
            metadata[0] = 1
            return w, g, oldw, oldg, hs, hy, hc, mx.array(metadata), raw
        monkeypatch.setattr(native_logistic, "step", fake_step)
        result = native_logistic.minimize_binary(
            X, y, None, 3.0, 0.1, True, LBFGSConfig(max_iter=1),
            lambda theta: (0.2, mx.ones(theta.shape)), initial_parts=parts, use_native_loop=False,
        )
    assert result.n_iter == 1 and not result.converged


def test_failed_native_search_with_small_gradient_is_not_relabelled_converged(monkeypatch):
    with mx.stream(mx.cpu):
        X, y = mx.ones((3, 2), mx.float32), mx.array([0, 1, 0], mx.float32)
        parts = mx.zeros((2, 1, 3), mx.float32)
        native_state = LBFGSState(mx.zeros((1, 3)), 0.2, 0.0, 2, 20, False, "line-search-failure")
        attempt = native_command.NativeAttempt(native_state, mx.zeros((1, 3)), 2)
        monkeypatch.setattr(native_command, "run_native_loop", lambda *a, **kw: attempt)
        monkeypatch.setattr(native_logistic, "initial_from_parts", lambda *a: pytest.fail("zero-step resume"))
        result = native_logistic.minimize_binary(
            X, y, None, 3.0, 0.1, True, LBFGSConfig(max_iter=5),
            lambda theta: (0.2, mx.zeros(theta.shape)), initial_parts=parts,
        )
    assert result is native_state and not result.converged
    assert result.stop_reason == "line-search-failure"


def test_worse_resumed_objective_keeps_native_parameters_and_counts(monkeypatch):
    with mx.stream(mx.cpu):
        X, y = mx.ones((3, 2), mx.float32), mx.array([0, 1, 0], mx.float32)
        parts = mx.zeros((2, 1, 3), mx.float32)
        w = mx.array([[0.25, 0.5, 0.7]], mx.float32)
        gradient = mx.array([[0.02, 0.04, 0.01]], mx.float32)
        native_state = LBFGSState(w, 0.5, 0.04, 3, 25, False, "line-search-failure")
        attempt = native_command.NativeAttempt(native_state, gradient, 2)
        monkeypatch.setattr(native_command, "run_native_loop", lambda *a, **kw: attempt)
        monkeypatch.setattr(
            native_logistic, "initial_from_parts",
            lambda *a: (mx.zeros((3,), mx.float32), mx.ones((3,), mx.float32)),
        )

        def fake_step(*args):
            w, g, oldw, oldg, hs, hy, hc, meta, raw = args[12:]
            metadata = np.asarray(meta).copy()
            metadata[0] += 1
            metadata[9] += 8
            metadata[10] = -0.01
            return w + 0.1, g, oldw, oldg, hs, hy, hc, mx.array(metadata), raw

        monkeypatch.setattr(native_logistic, "step", fake_step)
        result = native_logistic.minimize_binary(
            X, y, None, 3.0, 0.1, True, LBFGSConfig(max_iter=4, max_line_search=8),
            lambda theta: (0.6, mx.full(theta.shape, 0.01)), initial_parts=parts,
        )
    np.testing.assert_array_equal(result.x, w)
    assert result.objective == 0.5 and result.grad_max_abs == 0.04
    assert not result.converged and result.stop_reason == "line-search-failure"
    assert result.n_iter == 4 and result.n_fev == 34

@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("max_line_search", [8, 16])
def test_managed_failed_search_is_not_promoted_by_small_direct_gradient(monkeypatch, weighted, max_line_search):
    """Even a small direct gradient cannot certify a rejected search direction."""
    with mx.stream(mx.cpu):
        X = mx.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], mx.float32)
        y = mx.array([0, 1, 0], mx.float32)
        weights = mx.array([0.5, 1.0, 1.5], mx.float32) if weighted else None
        monkeypatch.setattr(native_logistic, "initial", lambda *args: (mx.zeros((3,)), mx.ones((3,))))
        calls = []

        def failed_step(*args, **kwargs):
            w, g, oldw, oldg, hs, hy, hc, meta, raw = args[12:]
            values = np.asarray(meta).copy()
            values[0] = 2
            values[1] = 2  # No candidate in this line-search batch was accepted.
            values[4] = values[10] = -1e-10
            values[11] = 0.2
            calls.append(float(np.asarray(args[10])))
            return w, g, oldw, oldg, hs, hy, hc, mx.array(values), raw

        monkeypatch.setattr(native_logistic, "step", failed_step)
        result = native_logistic.minimize_binary(
            X, y, weights, 3.0, 0.1, True,
            LBFGSConfig(max_iter=8, tol=1e-4, max_line_search=max_line_search),
            lambda theta: (0.2, mx.full(theta.shape, 1e-6)), use_native_loop=False,
        )
    assert len(calls) == (4 if max_line_search == 8 else 5)
    assert not result.converged and result.stop_reason == "line-search-failure"
    assert result.grad_max_abs < 1e-4 and result.n_iter == 2


def _resume_fixture(monkeypatch, status=2):
    """A direct-scored native attempt at objective 0.5 whose gradient is above tol."""
    X, y = mx.ones((3, 2), mx.float32), mx.array([0, 1, 0], mx.float32)
    parts = mx.zeros((2, 1, 3), mx.float32)
    w = mx.array([[0.25, 0.5, 0.7]], mx.float32)
    gradient = mx.array([[0.02, 0.04, 0.01]], mx.float32)
    native_state = LBFGSState(w, 0.5, 0.04, 3, 25, False, "line-search-failure")
    monkeypatch.setattr(
        native_command, "run_native_loop",
        lambda *a, **kw: native_command.NativeAttempt(native_state, gradient, status),
    )
    monkeypatch.setattr(
        native_logistic, "initial_from_parts",
        lambda *a: (mx.zeros((3,), mx.float32), mx.ones((3,), mx.float32)),
    )

    def accepted_step(*args):
        w, g, oldw, oldg, hs, hy, hc, meta, raw = args[12:]
        metadata = np.asarray(meta).copy()
        metadata[0] += 1
        metadata[9] += 8
        metadata[10] = -0.01
        return w + 0.1, g, oldw, oldg, hs, hy, hc, mx.array(metadata), raw + 0.1

    monkeypatch.setattr(native_logistic, "step", accepted_step)
    return X, y, parts, w, native_state


def test_certified_continuation_is_returned_despite_objective_noise(monkeypatch):
    """A continuation the direct check certifies is never discarded."""
    with mx.stream(mx.cpu):
        X, y, parts, w, _ = _resume_fixture(monkeypatch)
        noisy = 0.5 + 1e-9  # Higher than the native objective by float32 noise.
        result = native_logistic.minimize_binary(
            X, y, None, 3.0, 0.1, True, LBFGSConfig(max_iter=4, tol=1e-4, max_line_search=8),
            lambda theta: (noisy, mx.full(theta.shape, 5e-5)), initial_parts=parts,
        )
    np.testing.assert_allclose(result.x, np.asarray(w) + 0.1)
    assert result.converged and result.stop_reason == "gtol"
    assert result.objective == noisy and result.grad_max_abs == pytest.approx(5e-5)
    assert result.n_iter == 4 and result.n_fev == 34


@pytest.mark.parametrize(
    ("objective", "grad", "keeps_native"),
    [
        (0.5 + 1e-9, 0.03, False),  # Inside the noise band with a smaller gradient.
        (0.5 - 1e-9, 0.05, True),  # Inside the band with a larger gradient.
        (0.5 + 1e-9, 0.04, False),  # Inside the band with an equal gradient.
        (0.5 + 1e-6, 0.01, True),  # Clearly higher: the former behaviour stands.
        (0.49, 0.05, False),  # Clearly lower keeps the continuation.
    ],
)
def test_uncertified_continuation_objective_guard_uses_a_noise_band(
    monkeypatch, objective, grad, keeps_native
):
    with mx.stream(mx.cpu):
        X, y, parts, w, native_state = _resume_fixture(monkeypatch)
        result = native_logistic.minimize_binary(
            X, y, None, 3.0, 0.1, True, LBFGSConfig(max_iter=4, tol=1e-4, max_line_search=8),
            lambda theta: (objective, mx.full(theta.shape, grad)), initial_parts=parts,
        )
    assert not result.converged
    assert result.n_iter == 4 and result.n_fev == 34
    if keeps_native:
        np.testing.assert_array_equal(result.x, w)
        assert result.objective == 0.5 and result.grad_max_abs == 0.04
        assert result.stop_reason == "line-search-failure"
    else:
        np.testing.assert_allclose(result.x, np.asarray(w) + 0.1)
        assert result.objective == objective and result.grad_max_abs == pytest.approx(grad)
        assert result.stop_reason == "max_iter"


@pytest.mark.parametrize("band_test", [True])
def test_resume_noise_band_scales_with_the_objective(band_test):
    previous = LBFGSState(mx.zeros((1, 3)), 2.0, 0.04, 3, 25, False, "line-search-failure")
    band = native_logistic._RESUME_OBJECTIVE_BAND * 2.0
    keeps = native_logistic._resume_keeps_native_iterate
    assert keeps(previous, 2.0 + 2 * band, 0.01, True)
    assert not keeps(previous, 2.0 + 0.5 * band, 0.03, True)
    assert keeps(previous, 2.0 + 0.5 * band, 0.05, True)
    assert not keeps(previous, 2.0 - 2 * band, 0.05, True)
    assert keeps(previous, 2.0, 0.01, False)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("seed", "tol", "estimator_converges"),
    [(103, 1e-8, True), (100, 3e-9, False)],
)
def test_wide_fit_keeps_certified_native_or_resumed_result(seed, tol, estimator_converges):
    """A tight-tolerance wide fit keeps its certified native or resumed result.

    Some GPUs converge in the native loop; others need an MLX continuation.
    A continuation with an objective higher by float32 noise used to be discarded
    and reported as ``line-search-failure`` with its steps still counted. The
    deterministic tests above exercise that guard regardless of GPU rounding.
    At ``3e-9`` the float64 certificate
    may still reject the float32-converged model; that verdict is independent
    of the resume guard, so only the solver's result is asserted there.
    """
    from sklearn.exceptions import ConvergenceWarning

    from mlxlearn._common.config import tuning_context
    from mlxlearn._common.device import gpu_available
    from mlxlearn.linear_model import LogisticRegression

    if (not gpu_available() or native_command._metal_loop is None
            or not native_command._supports_metal_dlpack()):
        pytest.skip("requires the native Metal loop on a supported MLX")
    n, d = 65_536, 512
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d) / np.sqrt(d)
    y = (X @ w + 0.3 * rng.standard_normal(n) > 0).astype(int)
    attempts, results = [], []
    original_loop, original_minimize = native_command.run_native_loop, native_logistic.minimize_binary

    def observe_loop(*args, **kwargs):
        attempt = original_loop(*args, **kwargs)
        attempts.append(attempt)
        return attempt

    def observe_minimize(*args, **kwargs):
        result = original_minimize(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(native_command, "run_native_loop", observe_loop)
    monkeypatch.setattr(native_logistic, "minimize_binary", observe_minimize)
    try:
        with tuning_context(force_backend="mlx"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = LogisticRegression(C=1.0, tol=tol, max_iter=200).fit(X, y)
    finally:
        monkeypatch.undo()
    (attempt,), (state,) = attempts, results
    assert isinstance(attempt, native_command.NativeAttempt)
    assert state.converged and state.stop_reason == "gtol"
    assert state.grad_max_abs <= tol
    if attempt.state.converged:
        # A certified native result should be returned without another solve.
        assert state is attempt.state
    else:
        # The failed native attempt must be continued within the same budget.
        assert attempt.state.n_iter < state.n_iter <= 200
    if estimator_converges:
        assert not [c for c in caught if issubclass(c.category, ConvergenceWarning)]
        assert model.n_iter_[0] == state.n_iter
    else:
        assert model.n_iter_[0] >= state.n_iter
