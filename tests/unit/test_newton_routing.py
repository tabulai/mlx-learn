"""Hybrid selection, total iteration budgets, and failed-refinement handling."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers import logistic, native_logistic
from mlxlearn._solvers.lbfgs import LBFGSState
from mlxlearn_device import _native_cg, _newton


@pytest.mark.parametrize("n,d,l2,ratio,expected", [
    (4096, 2048, 1e-5, 1000, True),
    (20_000, 4096, 1e-5, 1000, True),
    (20_000, 4096, 5e-11, 1000, False),
    (20_000, 4096, 0.999e-7, 1000, False),
    (20_000, 4096, 1e-5, 100, False),
    (20_000, 4096, 1e-5, 1, False),
    (4095, 4096, 1e-5, 1000, False),
    (20_000, 1024, 1e-5, 1000, False),
    (20_000, 1024, 1e-5, 2**24, True),
    (20_000, 512, 1e-5, 2**24, True),
    (20_000, 511, 1e-5, 2**24, False),
    (20_000, 1024, 1e-5, 2**22, False),
    (20_000, 1024, 5e-11, 2**24, False),
    (4095, 1024, 1e-5, 2**24, False),
])
def test_hybrid_selection_keeps_weak_regularization_on_existing_solver(monkeypatch, n, d, l2, ratio, expected):
    monkeypatch.setattr(_native_cg, "metal_loop", SimpleNamespace(run_cg=object()))
    with mx.stream(mx.cpu):
        diagonal = mx.concatenate((mx.ones((d // 2,)), mx.full((d - d // 2,), ratio), mx.ones((1,))))
        parts = mx.stack((mx.zeros((1, d + 1)), diagonal[None, :]))
        chosen = _newton.use_newton_for_curvature(parts, n, d, 1.0, l2)
    assert chosen is expected


@pytest.mark.parametrize("d", [512, 1024, 2048, 3072, 4096])
def test_hybrid_selection_respects_native_cg_workspace_boundary(monkeypatch, d):
    """An unavailable refinement must not consume the budget on its warm start."""
    from mlxlearn_device._hessian_vector import _geometry

    monkeypatch.setattr(_native_cg, "metal_loop", SimpleNamespace(run_cg=object()))
    groups, feature_groups, rows, _ = _geometry(1_000_000, d)
    rows_per_block = groups // feature_groups * rows
    max_blocks = (64 * 1024 * 1024) // ((d + 1) * 4)
    last_supported_n = max_blocks * rows_per_block
    # Only the small statistics vector is needed to check routing for millions
    # of samples; no full design or oversized workspace is allocated.
    with mx.stream(mx.cpu):
        diagonal = mx.concatenate((mx.ones((d - 1,)), mx.array([1e10, 1.0])))
        parts = mx.stack((mx.zeros((1, d + 1)), diagonal[None, :]))
        assert _native_cg._workspace_supported(last_supported_n, d)
        assert _newton.use_newton_for_curvature(parts, last_supported_n, d, 1.0, 1e-5)
        assert not _native_cg._workspace_supported(last_supported_n + 1, d)
        assert not _newton.use_newton_for_curvature(parts, last_supported_n + 1, d, 1.0, 1e-5)


@pytest.fixture
def mocked_host_hybrid(monkeypatch):
    @contextmanager
    def pretend_gpu(_device):
        with use_device("cpu"):
            yield "gpu"

    monkeypatch.setattr(logistic, "use_device", pretend_gpu)
    monkeypatch.setattr(logistic, "_use_native_solver", lambda *_args: True)
    monkeypatch.setattr(_newton, "use_newton_for_curvature", lambda *_args: True)
    monkeypatch.setattr(logistic, "_needs_float64_certificate", lambda *_args: True)
    monkeypatch.setattr(logistic, "_original_float64_certificate", lambda *_args: (0.4, 0.0))
    monkeypatch.setattr(logistic, "_polish_binary_intercept", lambda *_a, **_kw: pytest.fail("hybrid must not bias-polish"))
    calls = []

    def run(budget, *, warm_used=None, refined=True, warm_reason="gtol", warm_objective=0.5):
        def warm(X, y, sw, sw_sum, l2, fit_intercept, config, fun, **_kwargs):
            calls.append(("warm", config.max_iter))
            used = config.max_iter if warm_used is None else warm_used
            return LBFGSState(mx.zeros((1, X.shape[1] + 1)), warm_objective, 0.0,
                              used, 2, warm_reason == "gtol", warm_reason)

        def newton(original, centered, center, y, sw, denom, l2, **kwargs):
            calls.append(("newton", kwargs["max_iter"]))
            used = min(3, kwargs["max_iter"])
            return _newton.NewtonResult(kwargs["initial"], used, 5, refined,
                                        "duality-gap" if refined else "line-search-failure",
                                        0.4, 0.0, 1e-6 if refined else 1e-2, 0.0, 4, True)

        monkeypatch.setattr(native_logistic, "minimize_binary", warm)
        monkeypatch.setattr(_newton, "minimize_newton", newton)
        with use_device("cpu"):
            X = np.zeros((3, 512), dtype=np.float32)
            result = logistic.solve_logistic(
                X, np.array([0, 1, 0]), n_classes=2,
                config=logistic.LogisticConfig(device="gpu", max_iter=budget),
                X_device=mx.zeros((3, 512)), X_center=mx.zeros((512,)),
                initial_parts=mx.ones((2, 1, 513)),
            )
        return result, calls

    return run


@pytest.mark.parametrize("budget", [1, 7, 32, 33, 40, 200])
def test_host_hybrid_shares_accepted_iteration_budget(mocked_host_hybrid, budget):
    result, calls = mocked_host_hybrid(budget)
    assert calls[0] == ("warm", min(32, budget))
    assert result.n_iter <= budget
    if budget <= 32:
        assert len(calls) == 1
        assert not result.converged and result.stop_reason == "max_iter"
    else:
        assert calls[1] == ("newton", budget - 32)
        assert result.n_iter == 32 + min(3, budget - 32)
        assert result.converged


def test_early_warm_stop_leaves_the_actual_remaining_budget(mocked_host_hybrid):
    result, calls = mocked_host_hybrid(200, warm_used=3)
    assert calls == [("warm", 32), ("newton", 197)]
    assert result.n_iter == 6 and result.converged


def test_failed_newton_cannot_be_promoted_by_perfect_float64_gradient(mocked_host_hybrid):
    result, calls = mocked_host_hybrid(200, refined=False)
    assert calls == [("warm", 32), ("newton", 168)]
    assert not result.converged and result.stop_reason == "line-search-failure"
    assert result.grad_max_abs == 0
    assert result.n_iter == 35


@pytest.mark.parametrize("reason,objective", [
    ("non-finite-objective", 0.5), ("max_iter", float("nan")), ("max_iter", float("inf")),
])
def test_nonfinite_warm_failure_never_enters_newton_or_passes_certificate(mocked_host_hybrid, reason, objective):
    result, calls = mocked_host_hybrid(200, warm_reason=reason, warm_objective=objective)
    assert calls == [("warm", 32)]
    assert not result.converged and result.stop_reason == "non-finite-objective"


@pytest.mark.parametrize("version", [None, 123, "0.31.0", "0.33.0", "0.32.2rc1"])
def test_hybrid_does_not_select_unsupported_dlpack_version(monkeypatch, version):
    monkeypatch.setattr(mx, "__version__", version)
    monkeypatch.setattr(_native_cg, "metal_loop", SimpleNamespace(run_cg=object()))
    with mx.stream(mx.cpu):
        parts = mx.ones((2, 1, 2049))
        assert not _newton.use_newton_for_curvature(parts, 20_000, 2048, 1.0, 1e-5)


@pytest.mark.parametrize(
    ("tol", "expect_newton"),
    [(0.0, False), (1e-39, False), (0.5 * float(np.finfo(np.float32).tiny), False),
     (float(np.finfo(np.float32).tiny), True),
     (float(np.finfo(np.float32).max), True), (1e40, False)],
)
def test_newton_requires_representable_positive_tolerance(monkeypatch, tol, expect_newton):
    """``minimize_newton`` validates a normal float32 tolerance; L-BFGS accepts any."""
    @contextmanager
    def pretend_gpu(_device):
        with use_device("cpu"):
            yield "gpu"

    monkeypatch.setattr(logistic, "use_device", pretend_gpu)
    monkeypatch.setattr(logistic, "_use_native_solver", lambda *_args: True)
    monkeypatch.setattr(_newton, "use_newton_for_curvature", lambda *_args: True)
    monkeypatch.setattr(logistic, "_needs_float64_certificate", lambda *_args: False)
    calls = []

    def warm(X, y, sw, sw_sum, l2, fit_intercept, config, fun, **_kwargs):
        calls.append(("warm", config.max_iter))
        return LBFGSState(mx.zeros((1, X.shape[1] + 1)), 0.5, 1e-3, config.max_iter, 2,
                          False, "max_iter")

    def newton(original, centered, center, y, sw, denom, l2, **kwargs):
        calls.append(("newton", kwargs["tol"]))
        return _newton.NewtonResult(kwargs["initial"], 3, 5, True, "duality-gap",
                                    0.4, 0.0, 1e-6, 0.0, 4, True)

    monkeypatch.setattr(native_logistic, "minimize_binary", warm)
    monkeypatch.setattr(_newton, "minimize_newton", newton)
    with use_device("cpu"):
        result = logistic.solve_logistic(
            np.zeros((3, 512), dtype=np.float32), np.array([0, 1, 0]), n_classes=2,
            config=logistic.LogisticConfig(device="gpu", max_iter=200, tol=tol),
            X_device=mx.zeros((3, 512)), X_center=mx.zeros((512,)),
            initial_parts=mx.ones((2, 1, 513)),
        )
    if expect_newton:
        assert calls == [("warm", 32), ("newton", tol)]
        assert result.converged and result.n_iter == 35
    else:
        # The whole budget goes to L-BFGS instead of a 32-step Newton warm start.
        assert calls == [("warm", 200)]
        assert not result.converged and result.n_iter == 200


@pytest.mark.gpu
def test_newton_eligible_fit_accepts_zero_tiny_and_large_tolerances(monkeypatch):
    """The curvature gate must preserve valid tolerances outside Newton's range."""
    from sklearn.exceptions import ConvergenceWarning

    from mlxlearn._common.config import tuning_context
    from mlxlearn._solvers import native_command
    from mlxlearn.linear_model import LogisticRegression

    if (not gpu_available() or native_command._metal_loop is None
            or not hasattr(native_command._metal_loop, "run_cg")
            or not native_command._supports_metal_dlpack()):
        pytest.skip("requires the native Metal CG extension on a supported MLX")
    rng = np.random.default_rng(0)
    n, d = 4096, 2048
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, 0] *= 100.0  # Curvature spread above 100 selects the Newton route.
    w = rng.normal(size=d) / np.sqrt(d)
    y = ((X @ w + rng.normal(scale=0.5, size=n)) > 0).astype(int)
    newton_tolerances = []
    original = _newton.minimize_newton

    def observe(*args, **kwargs):
        newton_tolerances.append(kwargs["tol"])
        return original(*args, **kwargs)

    monkeypatch.setattr(_newton, "minimize_newton", observe)
    with tuning_context(force_backend="mlx"), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("error", RuntimeWarning)
        LogisticRegression(C=1.0, tol=1e-4, max_iter=100).fit(X, y)
        assert newton_tolerances == [pytest.approx(5e-5)]  # Control: Newton route taken.
        for tol in (0.0, 1e-39, 1e40):
            # Exceed the 32-step warm start: a shorter budget can hide the old
            # invalid-tolerance crash by leaving no iterations for Newton.
            model = LogisticRegression(C=1.0, tol=tol, max_iter=100).fit(X, y)
            assert model._execution_backend_ == "mlx"
            assert 0 <= model.n_iter_[0] <= 100
    assert len(newton_tolerances) == 1


@pytest.mark.gpu
def test_extreme_skew_newton_preserves_rank_deficient_feature_symmetry(monkeypatch):
    """The narrower Newton route must not introduce a diagonal-metric null component."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression as SkLogisticRegression
    from threadpoolctl import threadpool_limits

    from mlxlearn._common.config import tuning_context
    from mlxlearn._solvers import native_command
    from mlxlearn.linear_model import LogisticRegression

    if (not gpu_available() or native_command._metal_loop is None
            or not hasattr(native_command._metal_loop, "run_cg")
            or not native_command._supports_metal_dlpack()):
        pytest.skip("requires the native Metal CG extension on a supported MLX")
    rng = np.random.default_rng(621)
    n, d = 4096, 512
    latent = rng.normal(size=(n, 8))
    # Eight independent columns repeated 64 times give an explicit null space.
    X = np.tile(latent, (1, d // 8))
    y = (latent @ rng.normal(size=8) + rng.normal(size=n) > 0).astype(int)
    X[:, 3] *= 1e5
    calls = []
    original = _newton.minimize_newton

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((kwargs["max_iter"], result.n_iter))
        return result

    monkeypatch.setattr(_newton, "minimize_newton", observe)
    params = {"C": 1.0, "tol": 1e-4, "max_iter": 100}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        with tuning_context(force_backend="mlx"):
            model = LogisticRegression(**params).fit(X, y)
        reference = SkLogisticRegression(**params).fit(X, y)
    assert len(calls) == 1
    remaining, used = calls[0]
    assert used <= remaining <= params["max_iter"]
    assert model.n_iter_[0] <= params["max_iter"]
    # Equal columns have equal minimum-norm coefficients. Check every group
    # without the rescaled column, including distant feature tiles.
    for group in (0, 1, 2, 4, 5, 6, 7):
        coefficients = model.coef_[0, np.arange(d) % 8 == group]
        np.testing.assert_allclose(coefficients, coefficients[0], rtol=1e-5, atol=1e-7)
    objective = logistic.reference_objective(model.coef_, model.intercept_, X, y, n_classes=2)
    reference_value = logistic.reference_objective(
        reference.coef_, reference.intercept_, X, y, n_classes=2
    )
    assert (objective - reference_value) / reference_value <= 0.01
