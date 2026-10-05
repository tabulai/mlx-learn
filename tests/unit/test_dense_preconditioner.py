"""LogisticRegression's dense L-BFGS preconditioner: its gate, switch and bookkeeping.

What the plan in ``_solvers/logistic.py`` promises (``docs/fp32_policy.md`` §2.3): the
gate fires on correlated features and turns away uncorrelated data, nearly separable
classes and widths it is not built for; a fit it turns away, and every fit with the
switch off, is the diagonal path bit for bit; its host work runs once, while the
device computes the first evaluation, on one BLAS thread; and the fit's one event
records what it decided. Parity of the answers it produces is
``tests/parity/test_logistic_preconditioner.py``'s.
"""

from __future__ import annotations

import math
import threading
import types
import warnings

import numpy as np
import pytest

import mlxlearn
from mlxlearn._common.config import Tuning, _tuning_from_env, tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._solvers import logistic
from mlxlearn._solvers.lbfgs import DensePreconditioner
from mlxlearn._solvers.logistic import LogisticConfig, solve_logistic
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.patching._estimators import LogisticRegression as PatchedLogisticRegression
from tests._helpers import join_or_fail, logistic_data

DEVICES = ["gpu", "cpu"]


def _device(device):
    if device == "gpu" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    return device


@pytest.fixture
def plans(monkeypatch):
    """Every ``_DensePlan`` the fits in a test build, in order."""
    built = []

    class Recorded(logistic._DensePlan):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(logistic, "_DensePlan", Recorded)
    return built


def _solve(X, y, n_classes, **config):
    return solve_logistic(X, y, n_classes=n_classes, config=LogisticConfig(**config))


# ---------------------------------------------------------------------- the gate


@pytest.mark.parametrize(
    "law, n, d, K, reason",
    [
        ("ar99", 20000, 16, 2, "correlated"),
        ("lowrank5", 20000, 32, 3, "correlated"),
        ("onehot", 20000, 64, 2, "correlated"),
        ("colscale", 20000, 16, 2, "correlated"),  # scales the diagonal leaves apart
        ("random", 20000, 32, 5, "correlated"),  # correlated features, no signal
        ("breast_cancer", 569, 30, 2, "correlated"),
        ("digits", 1797, 64, 3, "correlated"),
        ("iso", 20000, 32, 2, "uncorrelated"),
        ("blobs", 20000, 16, 3, "uncorrelated"),  # separable, but nothing to precondition
        ("blobsar", 20000, 16, 3, "separated"),
        ("digits", 1797, 64, 10, "separated"),
        ("digits", 20000, 64, 10, "separated"),
    ],
)
def test_the_gate_decides_by_data_law(law, n, d, K, reason, plans):
    X, y = logistic_data(law, n, d, K, seed=3)
    solution = _solve(X, y, K, max_iter=1, device="cpu")

    assert solution.preconditioner_reason == reason, plans[0].stats
    assert solution.preconditioner == ("dense" if reason == "correlated" else "diagonal")
    stats = plans[0].stats
    if reason == "uncorrelated":
        assert stats["correlation"] <= logistic._GATE_CORRELATION
        assert stats["condition"] <= logistic._GATE_CONDITION
        assert not stats["spread"] > logistic._GATE_SPREAD
        assert "separation" not in stats  # no model sample was taken
    elif reason == "separated":
        assert stats["separation"] >= logistic._SEPARATION_LIMIT
    else:
        assert stats["separation"] < logistic._SEPARATION_LIMIT


@pytest.mark.parametrize("lam", [2.0, 3.5, 8.0, 16.0])
def test_the_separation_is_the_largest_eigenvalue_of_within_over_between(lam, plans):
    """Two classes, means ``±Δ/2``, AR(0.9) noise ``W``: ``λ = ΔᵀW⁻¹Δ/4``, set exactly.

    The model reads it from 8 192 of 40 000 rows, against its own curvature, at
    ``C = inf`` so that no penalty discounts it.
    """
    rng = np.random.default_rng(0)
    n, d = 40000, 16
    lags = np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    W = 0.9**lags
    direction = rng.normal(size=d)
    delta = direction * math.sqrt(4.0 * lam / (direction @ np.linalg.solve(W, direction)))
    y = rng.integers(0, 2, size=n)
    noise = rng.multivariate_normal(np.zeros(d), W, size=n)
    X = noise + np.where(y[:, None] == 1, 0.5, -0.5) * delta

    _solve(X, y, 2, C=math.inf, max_iter=1, device="cpu")

    assert plans[0].stats["separation"] == pytest.approx(lam, rel=0.1)
    expected = "separated" if lam >= logistic._SEPARATION_LIMIT else "correlated"
    assert plans[0].reason == expected


def test_a_strong_penalty_reads_as_less_separable(plans):
    """The penalty enters through the model's curvature: a fit that cannot become
    confident is not turned away for the separation its data would allow (ten-class
    digits: λ ≈ 8 at C = 1, 5.4 at 1e-4, 1.9 at 1e-5)."""
    X, y = logistic_data("digits", 20000, 64, 10, seed=3)
    assert _solve(X, y, 10, C=1.0, max_iter=1, device="cpu").preconditioner_reason == "separated"
    assert _solve(X, y, 10, C=1e-5, max_iter=1, device="cpu").preconditioner_reason == "correlated"
    assert plans[1].stats["separation"] < plans[0].stats["separation"] / 2


@pytest.mark.parametrize("d", [1, 1025])
def test_widths_the_model_is_not_built_for_take_the_diagonal(d, plans):
    X, y = logistic_data("ar90", 2048, d, 2, seed=1)
    solution = _solve(X, y, 2, max_iter=2, device="cpu")
    assert (solution.preconditioner, solution.preconditioner_reason) == ("diagonal", "width")
    assert not plans


def test_a_non_finite_sample_is_declined_not_raised():
    X = np.ones((600, 4), dtype=np.float32)
    X[::7, 2] = np.inf
    plan = logistic._DensePlan(
        X, np.arange(600) % 2, None, n_classes=2, fit_intercept=True, l2=1e-3,
        diagonal=np.ones((1, 5)),
    )
    plan.run()
    assert plan.reason == "non-finite" and plan.resolve() is None


@pytest.mark.parametrize("n", [1, 2, 3, 40])
def test_tiny_and_degenerate_samples_do_not_break_the_plan(n):
    """A handful of rows, a constant column, a column equal to another: the plan
    decides something, and the fit runs."""
    rng = np.random.default_rng(n)
    X = rng.normal(size=(n, 5))
    X[:, 1] = 3.0
    X[:, 4] = X[:, 0]
    y = np.arange(n) % 2 if n > 1 else np.array([1])
    solution = _solve(X, y, 2, device="cpu")
    assert solution.preconditioner_reason in {
        "correlated", "uncorrelated", "separated", "no-step"
    }
    assert np.all(np.isfinite(solution.coef))


def test_null_directions_get_neutral_curvature(plans):
    """Three one-hot blocks beside an intercept are three exact dependencies. Without a
    penalty the model's curvature along them is only its ridge, and its inverse would
    amplify the gradient's rounding there by 1/ridge (‖coef‖ 1 260 against 16.7 when
    it did); they get the model's mean curvature instead. Every level is common, so the
    sample reads each as the whole design does and no floor disturbs the dependencies."""
    X, y = logistic_data("onehot", 10000, 68, 2, seed=5)
    dense = _solve(X, y, 2, C=math.inf, tol=1e-6, device="cpu")
    diagonal = _solve(X, y, 2, C=math.inf, tol=1e-6, device="cpu", dense_preconditioner=False)

    assert dense.preconditioner == "dense" and plans[0].stats["null_directions"] == 3
    assert plans[0].stats["floored"] == 0
    assert np.linalg.norm(dense.coef) <= 1.5 * np.linalg.norm(diagonal.coef)
    assert dense.objective <= diagonal.objective * (1 + 1e-6)


# ------------------------------------------------- what a sample of the rows misses


@pytest.mark.parametrize("n, rows", [(8192, 8192), (16383, 8192), (65535, 8192), (1000, 999),
                                     (600, 512)])
def test_the_sample_spreads_over_every_row(n, rows):
    """A strided prefix never saw the last ``n − stride·rows`` rows: 8 191 of 16 383."""
    index = logistic._sample_rows(n, rows)
    assert index.shape == (rows,) and index[0] == 0
    assert np.all(np.diff(index) >= 1) and index[-1] < n
    assert index[-1] >= n - math.ceil(n / rows)


def test_weighted_rows_are_drawn_in_proportion_to_their_weight():
    """Ten rows of 60 000 carrying 90% of the weight are 90% of the sample, rows of zero
    weight are never drawn, and unit weights draw exactly the unweighted rows."""
    n, rows = 60000, 8192
    weights = np.ones(n)
    weights[1::7] = 0.0
    heavy = np.arange(5, n, n // 10)[:10]
    weights[heavy] = 9.0 * (n - 10) / 10
    index = logistic._sample_rows(n, rows, weights)

    assert np.all(np.diff(index) >= 0) and 0 <= index[0] and index[-1] < n
    # Systematic sampling draws each row ⌊share·rows⌋ or ⌈share·rows⌉ times.
    share = weights[heavy].sum() / weights.sum()
    assert np.mean(np.isin(index, heavy)) == pytest.approx(share, abs=len(heavy) / rows)
    assert not np.any(weights[index] == 0.0)
    np.testing.assert_array_equal(
        logistic._sample_rows(16383, rows, np.ones(16383)), logistic._sample_rows(16383, rows)
    )


@pytest.mark.parametrize("law, d, K, reason", [
    ("blobsar", 32, 2, "separated"),  # λ ≈ 39: read as 0.04 from the head of sorted rows
    ("digits", 64, 3, "correlated"),
    ("twosource", 16, 3, "correlated"),  # AR(0.99) rows, then as many N(0, I) rows
])
def test_the_order_of_the_rows_does_not_decide(law, d, K, reason, plans):
    X, y = logistic_data(law, 16383, d, K, seed=3)
    order = np.argsort(y, kind="stable")
    shuffled = _solve(X, y, K, max_iter=1, device="cpu")
    ordered = _solve(X[order], y[order], K, max_iter=1, device="cpu")

    assert shuffled.preconditioner_reason == ordered.preconditioner_reason == reason
    assert plans[1].stats["separation"] == pytest.approx(plans[0].stats["separation"], rel=0.05)


def test_features_the_sample_misses_get_the_whole_designs_curvature(plans):
    """Two hundred Zipf levels in 20 000 rows: the 8 192-row sample misses the rarest,
    whose curvature it reads as c·p², the centering's, where the design's is c·p. Without
    the floor the model's step along them was ~1e4 times too long, and the fit, falling
    back, stopped on a line-search failure 9% above the optimum after 586 evaluations."""
    X, y = logistic_data("rare", 20000, 216, 3, seed=3)
    solution = _solve(X, y, 3, C=math.inf, max_iter=1000, device="cpu")

    assert solution.preconditioner_reason == "correlated" and plans[0].stats["floored"] > 0
    assert solution.converged and solution.n_fev < 60


def test_the_floor_is_the_whole_designs_clipped_weighted_second_moment():
    rng = np.random.default_rng(0)
    X = rng.standard_t(2.0, size=(30000, 7)).astype(np.float32)
    weights = rng.exponential(size=30000)
    cap = 2.5
    norms = np.sum(X.astype(np.float64) ** 2, axis=1)
    w = weights * np.minimum(1.0, cap * cap / norms)
    expected = (w @ X.astype(np.float64) ** 2) / w.sum()

    np.testing.assert_allclose(logistic._column_moments(X, weights, cap), expected, rtol=1e-12)
    np.testing.assert_allclose(
        logistic._column_moments(X, None, 0.0), np.mean(X.astype(np.float64) ** 2, axis=0),
        rtol=1e-12,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_the_floors_pass_raises_no_floating_point_warnings(device, plans):
    """The floor's pass over the rows runs on the host pool's threads, which the plan's
    errstate — its own thread's — does not reach. Without an intercept an all-zero row
    stays zero, and beside rows of norm ~1e5 the clip's cap² over its norm overflowed
    there, to the shrink of 1 it gets anyway: the fit warned, or raised under an error
    filter, where the diagonal path says nothing."""
    device = _device(device)
    X, y = logistic_data("ar90", 60000, 64, 3, seed=3)
    X = X * 1e4
    X[::1000] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _solve(X, y, 3, fit_intercept=False, device=device)
    # The model was built from a sample, so the floor's pass ran.
    assert plans[0].reason == "correlated" and plans[0].stats["rows"] < len(X)


@pytest.mark.parametrize("scale", [1e4, 1e8])
def test_the_model_does_not_depend_on_a_columns_units(scale, plans):
    """λ of W⁻¹B is invariant to a column's units, and so is the model's reading of it
    now that each feature's ridge is relative to its own curvature. Relative to the mean
    curvature, a column in units of 1e8 swamped every other one: the separation read
    0.005 where it is 1.05, and the dense fit stopped on ``max_iter`` 1.4% above the
    optimum (200 000 x 16, two classes)."""
    X, y = logistic_data("ar90", 20000, 16, 3, seed=3)
    scaled = X.copy()
    scaled[:, 0] *= scale
    _solve(X, y, 3, max_iter=1, device="cpu")
    _solve(scaled, y, 3, max_iter=1, device="cpu")

    assert plans[0].reason == plans[1].reason == "correlated"
    assert plans[1].stats["separation"] == pytest.approx(plans[0].stats["separation"], rel=1e-3)


def test_a_constant_column_gets_no_step_of_its_own():
    """Centering a constant column of 0.3 in float64 leaves a residue of order 1e-13 in
    every row, collinear with the intercept. A ridge relative to that column's own
    curvature alone would make the model's step along it 1e26 times the gradient's; the
    geometric mean of the curvatures floors it (the diagonal path puts -5.8e8 there)."""
    X, y = logistic_data("ar90", 20000, 16, 2, seed=1)
    X = np.hstack([X, np.full((20000, 1), 0.3)])
    solution = _solve(X, y, 2, C=math.inf, device="cpu")

    assert solution.preconditioner == "dense" and solution.converged
    assert abs(solution.coef[0, -1]) < 1.0


# ----------------------------------------------------- the switch, bit for bit


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("decline", ["switch-off", "gate", "model"])
def test_a_declined_plan_is_the_diagonal_path_bit_for_bit(decline, device, monkeypatch):
    """The overlap, the sample and the gate change nothing a declined fit computes:
    whether the switch is off, the gate turns the fit away or the model is refused,
    every bit of the answer and every count is the diagonal path's."""
    device = _device(device)
    X, y = logistic_data("ar99", 20000, 24, 3, seed=11)
    off = _solve(X, y, 3, device=device, dense_preconditioner=False)
    if decline == "gate":
        monkeypatch.setattr(logistic._DensePlan, "_gate", lambda self: False)
    elif decline == "model":
        monkeypatch.setattr(logistic._DensePlan, "_model", lambda self: None)
    declined = _solve(X, y, 3, device=device, dense_preconditioner=decline != "switch-off")

    assert declined.preconditioner == off.preconditioner == "diagonal"
    np.testing.assert_array_equal(declined.coef, off.coef)
    np.testing.assert_array_equal(declined.intercept, off.intercept)
    assert (declined.n_iter, declined.n_fev, declined.objective, declined.grad_max_abs) == (
        off.n_iter, off.n_fev, off.objective, off.grad_max_abs
    )


def test_the_switch_off_takes_no_sample(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a plan was built with the switch off")

    monkeypatch.setattr(logistic, "_DensePlan", forbidden)
    monkeypatch.setattr(logistic, "_sample_moments", forbidden)
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    solution = _solve(X, y, 2, device="cpu", dense_preconditioner=False)
    assert solution.preconditioner_reason == "disabled"


@pytest.mark.parametrize("device", DEVICES)
def test_a_preconditioned_fit_is_deterministic(device):
    device = _device(device)
    X, y = logistic_data("lowrank20", 30000, 48, 4, seed=4)
    first, second = (_solve(X, y, 4, device=device, tol=1e-6) for _ in range(2))
    assert first.preconditioner == "dense"
    np.testing.assert_array_equal(first.coef, second.coef)
    np.testing.assert_array_equal(first.intercept, second.intercept)
    assert (first.n_fev, first.objective) == (second.n_fev, second.objective)


@pytest.mark.parametrize(
    "config, reason",
    [({"tol": 1e9}, "no-step"), ({"max_iter": 0}, "no-step")],
    ids=["optimal-start", "no-budget"],
)
def test_a_fit_that_takes_no_step_reports_the_diagonal(config, reason):
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    solution = _solve(X, y, 2, device="cpu", **config)
    assert (solution.n_iter, solution.preconditioner) == (0, "diagonal")
    assert solution.preconditioner_reason == reason


@pytest.mark.parametrize("device", DEVICES)
def test_a_wrong_model_falls_back_to_the_diagonal_and_says_so(device, monkeypatch):
    """A model whose steps are 1e30 times too long, as a sample that missed what is rare
    once made them along some coordinates: the run leaves it where it would have
    stopped short, converges on the diagonal path from there, and records the fallback
    — on the solution and on the estimator's one fit event."""
    device = _device(device)

    def wrong(self):
        self.reason = "correlated"
        return DensePreconditioner(lambda v: 1e30 * v)

    monkeypatch.setattr(logistic._DensePlan, "_model", wrong)
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    off = _solve(X, y, 2, device=device, dense_preconditioner=False)
    solution = _solve(X, y, 2, device=device)

    assert (solution.preconditioner, solution.preconditioner_reason) == ("diagonal", "fallback")
    assert solution.converged
    assert solution.objective == pytest.approx(off.objective, rel=1e-6)

    backend = "mlx" if device == "gpu" else "cpu"
    with tuning_context(force_backend=backend):
        model = LogisticRegression().fit(X, y)
    extra = model.fit_event_.extra
    assert (extra["preconditioner"], extra["preconditioner_reason"]) == ("diagonal", "fallback")
    assert "stop_reason" not in extra


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize(
    "law, n_classes", [("ar99", 2), ("lowrank5", 5)], ids=["binary", "multinomial"]
)
def test_a_wide_dense_fit_raises_no_floating_point_warnings(law, n_classes, backend):
    """NumPy 2.2's BLAS on macOS left floating-point flags set on finite products — the
    operator's from 65 parameters, L-BFGS's inner products from about 900 — and a fit
    on the dense path warned on every iteration. The plan's own products run under its
    errstate, but L-BFGS applies the operator after the plan has returned, and takes
    ``yᵀPy`` for its scaling (``lbfgs._dot``)."""
    if backend == "mlx" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    X, y = logistic_data(law, 3000, 1000, n_classes, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with tuning_context(force_backend=backend):
            model = LogisticRegression().fit(X, y)
    extra = model.fit_event_.extra
    assert (extra["preconditioner"], extra["preconditioner_reason"]) == ("dense", "correlated")
    assert model.n_iter_[0] > 1  # curvature pairs, so the scaling's yᵀPy ran too


# ----------------------------------------------------- overlap and BLAS threads


class _RecordingMLX:
    def __init__(self, log):
        self.log = log

    def async_eval(self, *arrays):
        self.log.append("submit")

    def eval(self, *arrays):
        self.log.append("wait")


def test_the_host_work_runs_between_submitting_and_waiting_for_the_first_evaluation():
    log = []
    first = logistic._FirstEvaluation(lambda: log.append("host"))
    mx = _RecordingMLX(log)

    logistic._evaluate(mx, ("value", "grad"), first)
    logistic._evaluate(mx, ("value", "grad"), first)
    logistic._evaluate(mx, ("value", "grad"), None)

    assert log == ["submit", "host", "wait", "wait", "wait"]


def test_the_plan_runs_once_per_fit_before_it_is_resolved(monkeypatch):
    calls = []
    run, resolve = logistic._DensePlan.run, logistic._DensePlan.resolve
    monkeypatch.setattr(logistic._DensePlan, "run", lambda self: (calls.append("run"), run(self))[1])
    monkeypatch.setattr(
        logistic._DensePlan, "resolve", lambda self: (calls.append("resolve"), resolve(self))[1]
    )
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    assert _solve(X, y, 2, device="cpu").preconditioner == "dense"
    assert calls == ["run", "resolve"]


class _FakePools:
    """threadpoolctl's ``limit`` on a controller with a BLAS pool, recorded."""

    def __init__(self, log, threads=16):
        self.log = log
        self.lib_controllers = [types.SimpleNamespace(num_threads=threads)]

    def limit(self, *, limits):
        log = self.log

        class Limit:
            def __enter__(self):
                assert logistic._BLAS_LIMIT_LOCK.locked()
                log.append(("limit", limits))

            def __exit__(self, *exc):
                log.append("restore")

        return Limit()


def test_the_model_runs_on_one_blas_thread_under_the_lock(monkeypatch):
    log = []
    monkeypatch.setattr(logistic, "_blas_pools", lambda: _FakePools(log))
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    assert _solve(X, y, 2, device="cpu").preconditioner == "dense"
    assert log == [("limit", 1), "restore"]
    assert not logistic._BLAS_LIMIT_LOCK.locked()


def test_a_fit_the_gate_turns_away_leaves_the_blas_alone(monkeypatch):
    """The limit is process-wide; the gate, which every fit runs, never takes it."""
    log = []
    monkeypatch.setattr(logistic, "_blas_pools", lambda: _FakePools(log))
    X, y = logistic_data("iso", 5000, 16, 2, seed=2)
    assert _solve(X, y, 2, device="cpu").preconditioner_reason == "uncorrelated"
    assert log == []


def test_pools_already_on_one_thread_are_left_alone(monkeypatch):
    log = []
    monkeypatch.setattr(logistic, "_blas_pools", lambda: _FakePools(log, threads=1))
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    assert _solve(X, y, 2, device="cpu").preconditioner == "dense"
    assert log == []
    assert not logistic._BLAS_LIMIT_LOCK.locked()


def test_without_a_blas_pool_there_is_nothing_to_limit(monkeypatch):
    monkeypatch.setattr(logistic, "_blas_pools", lambda: None)
    with logistic._one_blas_thread():
        assert not logistic._BLAS_LIMIT_LOCK.locked()


def _blas_threads():
    from threadpoolctl import threadpool_info

    return [(lib["filepath"], lib["num_threads"]) for lib in threadpool_info()
            if lib["user_api"] == "blas"]


def test_an_outside_limit_that_overlaps_the_gate_is_restored_as_it_was_found(monkeypatch):
    """Another thread's threadpoolctl limit — scikit-learn's ``KMeans`` sets one — that
    begins while a fit's gate runs and ends after the fit. Had the gate held the plan's
    limit, the outside one would have saved the plan's single thread and restored it
    when it ended, leaving the process on one BLAS thread for good. A no-op where the
    BLAS has no pool (Accelerate), and the test that matters on an OpenBLAS stack."""
    from threadpoolctl import threadpool_limits

    before = _blas_threads()
    in_gate, entered, fitted = threading.Event(), threading.Event(), threading.Event()
    gate = logistic._DensePlan._gate

    def paused_gate(self):
        in_gate.set()
        assert entered.wait(30)
        return gate(self)

    monkeypatch.setattr(logistic._DensePlan, "_gate", paused_gate)
    errors = []

    def outside():
        try:
            assert in_gate.wait(30)
            with threadpool_limits(limits=2, user_api="blas"):
                entered.set()
                assert fitted.wait(30)
        except BaseException as exc:  # reported below, by the main thread
            errors.append(exc)
            entered.set()

    thread = threading.Thread(target=outside, daemon=True)
    thread.start()
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    try:
        assert _solve(X, y, 2, device="cpu").preconditioner == "dense"
    finally:
        fitted.set()
        join_or_fail([thread])
    assert not errors, errors
    assert _blas_threads() == before


def test_concurrent_fits_leave_the_blas_thread_count_as_they_found_it():
    """The limit is process-wide BLAS state; fits on several threads must not save and
    restore it out of order and leave the process on one thread. A no-op where the
    BLAS has no pool (Accelerate), and the test that matters on an OpenBLAS stack."""
    before = _blas_threads()
    X, y = logistic_data("ar99", 8000, 32, 3, seed=7)
    errors = []

    def fit():
        try:
            for _ in range(3):
                assert _solve(X, y, 3, device="cpu").preconditioner == "dense"
        except BaseException as exc:  # reported below, by the main thread
            errors.append(exc)

    threads = [threading.Thread(target=fit, daemon=True) for _ in range(4)]
    for thread in threads:
        thread.start()
    join_or_fail(threads)
    assert not errors, errors
    assert _blas_threads() == before


# ------------------------------------------------------- the estimator's record


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_the_fit_event_records_the_preconditioner_and_the_evaluations(backend):
    if backend == "mlx" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    with tuning_context(force_backend=backend):
        model = LogisticRegression().fit(X, y)
    events = [e for e in mlxlearn.get_backend_diagnostics() if e.operation == "fit"]
    assert events == [model.fit_event_]
    extra = model.fit_event_.extra
    assert (extra["preconditioner"], extra["preconditioner_reason"]) == ("dense", "correlated")
    assert extra["n_fev"] >= model.n_iter_[0] + 1
    assert "stop_reason" not in extra  # converged: nothing else to record


def test_the_patched_estimator_records_it_too():
    X, y = logistic_data("iso", 5000, 16, 3, seed=2)
    with tuning_context(force_backend="cpu"):
        model = PatchedLogisticRegression().fit(X, y)
    assert model.fit_event_.extra["preconditioner_reason"] == "uncorrelated"
    assert model.fit_event_.extra["preconditioner"] == "diagonal"


def test_tuning_turns_it_off_for_the_estimator():
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    with tuning_context(force_backend="cpu", lbfgs_dense_preconditioner=False):
        off = LogisticRegression().fit(X, y)
    with tuning_context(force_backend="cpu"):
        on = LogisticRegression().fit(X, y)
    assert off.fit_event_.extra["preconditioner_reason"] == "disabled"
    assert on.fit_event_.extra["n_fev"] < off.fit_event_.extra["n_fev"]


@pytest.mark.parametrize("raw, value", [("0", False), ("off", False), ("1", True), ("TRUE", True)])
def test_the_switch_reads_the_environment(monkeypatch, raw, value):
    monkeypatch.setenv("MLXLEARN_LBFGS_DENSE_PRECONDITIONER", raw)
    assert _tuning_from_env(Tuning()).lbfgs_dense_preconditioner is value


def test_a_malformed_switch_is_an_error(monkeypatch):
    monkeypatch.setenv("MLXLEARN_LBFGS_DENSE_PRECONDITIONER", "sometimes")
    with pytest.raises(ValueError, match="MLXLEARN_LBFGS_DENSE_PRECONDITIONER"):
        _tuning_from_env(Tuning())


@pytest.mark.parametrize("value", [1, "yes", None])
def test_the_switch_is_a_bool(value):
    with pytest.raises(TypeError, match="lbfgs_dense_preconditioner must be a bool"):
        Tuning(lbfgs_dense_preconditioner=value)


def test_the_switch_is_on_by_default():
    assert Tuning().lbfgs_dense_preconditioner is True
    assert LogisticConfig().dense_preconditioner is True
    assert Tuning(lbfgs_dense_preconditioner=np.bool_(False)).lbfgs_dense_preconditioner is False
