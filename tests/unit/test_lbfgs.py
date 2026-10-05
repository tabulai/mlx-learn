"""The L-BFGS solver, tested directly rather than through an estimator.

``docs/development.md`` shapes every solver as "a frozen config and a callable in, a
frozen state out" precisely so that its numerics can be tested without constructing
an estimator. Most of what this module checks is therefore the *contract* the module
docstring of ``_solvers/lbfgs.py`` promises — which stop reason is reported when,
what ``n_iter`` counts, that a line-search failure returns instead of raising — on
synthetic objectives whose answers are known in closed form.

The pinned logistic trajectories at the bottom are the regression net for changes
to the solver's internals: on well-conditioned problems the iteration count must not
grow and the objective must not get worse.
"""

from __future__ import annotations

import math
import warnings

import mlx.core as mx
import numpy as np
import pytest
from sklearn.datasets import make_classification

from mlxlearn._solvers.lbfgs import DensePreconditioner, LBFGSConfig, LBFGSState, minimize
from mlxlearn._solvers.logistic import LogisticConfig, reference_objective, solve_logistic


def _quadratic(scale):
    """``f(x) = 0.5 * Σ scale_i x_i²``, minimized at the origin."""
    a = mx.array(np.asarray(scale, dtype=np.float32))

    def fun(x):
        return float(0.5 * mx.sum(a * x * x)), a * x

    return fun


def _rosenbrock(x):
    """The n-dimensional Rosenbrock function, minimized at ``x = 1``."""
    head, tail = x[:-1], x[1:]
    value = mx.sum(100.0 * (tail - head * head) ** 2 + (1.0 - head) ** 2)
    grad_head = -400.0 * head * (tail - head * head) - 2.0 * (1.0 - head)
    grad_tail = 200.0 * (tail - head * head)
    zero = mx.zeros((1,), dtype=x.dtype)
    grad = mx.concatenate([grad_head, zero]) + mx.concatenate([zero, grad_tail])
    return float(value), grad


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": -1},
        {"tol": -1e-4},
        {"tol": math.inf},
        {"tol": math.nan},
        {"history_size": 0},
        {"max_line_search": 0},
        {"c1": 0.0},
        {"c1": 0.95, "c2": 0.9},
        {"c2": 1.0},
        {"min_curvature": -1.0},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LBFGSConfig(**kwargs)


# ------------------------------------------------------------------ stop reasons


def test_quadratic_converges_on_gtol():
    fun = _quadratic([1.0, 10.0, 100.0])
    state = minimize(fun, mx.array([3.0, -4.0, 0.5]), LBFGSConfig(tol=1e-6))

    assert isinstance(state, LBFGSState)
    assert state.converged and state.stop_reason == "gtol"
    assert state.grad_max_abs <= 1e-6
    assert state.n_fev >= state.n_iter + 1
    np.testing.assert_allclose(np.asarray(state.x), 0.0, atol=1e-6)


def test_the_parameter_keeps_its_shape_and_type():
    """The solver treats ``x`` as a vector space and never reshapes the caller's layout."""
    fun = _quadratic(np.arange(1.0, 7.0).reshape(2, 3))
    state = minimize(fun, mx.ones((2, 3)), LBFGSConfig(tol=1e-6))
    assert isinstance(state.x, mx.array)
    assert tuple(state.x.shape) == (2, 3)
    assert state.converged


def test_an_optimal_starting_point_takes_no_steps():
    """``n_iter`` counts accepted steps, so a converged start reports zero."""
    state = minimize(_quadratic([1.0, 2.0]), mx.zeros((2,)), LBFGSConfig())
    assert (state.n_iter, state.n_fev, state.stop_reason, state.converged) == (0, 1, "gtol", True)


def test_max_iter_zero_evaluates_once_and_does_not_claim_convergence():
    state = minimize(_quadratic([1.0, 2.0]), mx.array([1.0, 1.0]), LBFGSConfig(max_iter=0))
    assert (state.n_iter, state.n_fev, state.stop_reason, state.converged) == (
        0, 1, "max_iter", False,
    )
    assert state.objective == pytest.approx(1.5)


def test_max_iter_is_reported_when_the_budget_runs_out():
    fun = _quadratic(np.logspace(0, 4, 20))
    state = minimize(fun, mx.ones((20,)), LBFGSConfig(max_iter=2, tol=1e-12))
    assert state.stop_reason == "max_iter" and not state.converged
    assert state.n_iter == 2


def test_a_non_finite_start_is_a_stop_reason_not_an_exception():
    def fun(x):
        return math.inf, mx.zeros_like(x)

    state = minimize(fun, mx.array([1.0]), LBFGSConfig())
    assert state.stop_reason == "non-finite-objective"
    assert not state.converged and state.n_iter == 0


def test_an_inconsistent_gradient_ends_in_line_search_failure():
    """A gradient that points uphill can never meet sufficient decrease.

    That is the ancestor's clipped-objective/unclipped-gradient bug in miniature.
    It must come back as a stop reason with ``converged=False`` — not raise, and not
    take null steps until ``max_iter``.
    """
    base = _quadratic([1.0, 3.0])

    def fun(x):
        value, grad = base(x)
        return value, -grad

    state = minimize(fun, mx.array([2.0, -1.0]), LBFGSConfig(max_iter=50))
    assert state.stop_reason == "line-search-failure"
    assert not state.converged
    assert state.n_iter == 0
    np.testing.assert_array_equal(np.asarray(state.x), [2.0, -1.0])


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_gradient_that_turns_non_finite_never_reports_convergence(bad):
    """The solver's inner products report no floating-point flags (``lbfgs._dot``), so
    a gradient that is not finite has to be found by the solver itself.

    A quadratic in 1 000 variables whose gradient turns ``bad`` in one coordinate
    past ``x₀ = −0.5``, the value still finite: the run stops unconverged with that
    ``max|g|``, on an iterate whose objective was finite. A NaN raises no flag of its
    own, so it is answered without a single warning; an infinity still warns where
    the iterate's own arithmetic meets it.
    """
    target = np.linspace(-1.0, 1.0, 1000)

    def fun(x):
        v = np.asarray(x, dtype=np.float64)
        grad = 2.0 * (v - target)
        if v[0] < -0.5:
            grad[0] = bad
        return float(np.sum((v - target) ** 2)), grad

    with warnings.catch_warnings():
        warnings.simplefilter("error" if math.isnan(bad) else "ignore", RuntimeWarning)
        state = minimize(fun, mx.zeros((1000,)), LBFGSConfig())
    assert not state.converged and state.stop_reason != "gtol"
    assert not math.isfinite(state.grad_max_abs)
    assert np.isfinite(np.asarray(state.x)).all()


def test_rosenbrock():
    """A classic non-convex valley. scipy's L-BFGS-B needs 43 iterations on this one."""
    x0 = mx.array([-1.2, 1.0, -1.2, 1.0])
    state = minimize(_rosenbrock, x0, LBFGSConfig(max_iter=500, tol=1e-5))
    assert state.converged, state.stop_reason
    np.testing.assert_allclose(np.asarray(state.x), 1.0, atol=1e-3)
    assert state.n_iter <= 100


# ------------------------------------------------------------------- line search


def test_a_trial_step_into_an_infinite_region_contracts():
    """``-log(x) - log(1 - x)``, infinite outside ``(0, 1)``.

    From ``x = 0.01`` the steepest-descent step overshoots past 1, where the
    objective is infinite. An infinite value carries no slope to interpolate on, so
    the search must contract toward the last finite point rather than give up.
    """

    def fun(x):
        v = np.asarray(x, dtype=np.float64)
        if np.any(v <= 0.0) or np.any(v >= 1.0):
            return math.inf, mx.zeros_like(x)
        value = float(np.sum(-np.log(v) - np.log1p(-v)))
        return value, mx.array((-1.0 / v + 1.0 / (1.0 - v)).astype(np.float32))

    state = minimize(fun, mx.array([0.01]), LBFGSConfig(tol=1e-5))
    assert state.converged, state.stop_reason
    np.testing.assert_allclose(np.asarray(state.x), 0.5, atol=1e-4)


def test_an_exhausted_budget_still_takes_the_best_sufficient_decrease_step():
    """Running out of evaluations is "could not certify this step", not "could not move".

    With a one-evaluation budget, the first trial of this shallow quadratic meets the
    Armijo condition but not the curvature condition. The search must accept it —
    the caller's curvature test protects the Hessian approximation — instead of
    reporting a failure.
    """
    fun = _quadratic([1e-4])
    config = LBFGSConfig(max_iter=1, max_line_search=1)
    state = minimize(fun, mx.array([100.0]), config)
    assert state.n_iter == 1
    assert state.objective < 0.5 * 1e-4 * 100.0**2


def test_the_zoom_has_no_absolute_step_floor():
    """``f(x) = ½s²(x − 1/s)²`` at ``s = 1e8``, from ``x = 0``.

    The first trial ``t = 1/‖g‖₁ = 1/s`` overshoots by a factor of ``s``, so the
    acceptable step is about ``1/s² = 1e-16``. The zoom used to stop once the bracket
    was narrower than 1e-12 *absolute*, returning a line-search failure at
    ``n_iter=0`` — which is how LogisticRegression returned the null model for
    features around 1e8.
    """
    s = 1e8

    def fun(x):
        v = np.asarray(x, dtype=np.float64)
        return float(0.5 * s * s * np.sum((v - 1.0 / s) ** 2)), mx.array(
            (s * s * (v - 1.0 / s)).astype(np.float32)
        )

    state = minimize(fun, mx.array([0.0]), LBFGSConfig())
    assert state.n_iter >= 1, state.stop_reason
    assert state.objective < 1e-6 * 0.5
    np.testing.assert_allclose(np.asarray(state.x), 1.0 / s, rtol=1e-5)


def test_the_preconditioner_reaches_coordinates_on_a_different_scale():
    """Curvatures 1e12 apart: one coordinate the gradient is dominated by, one it is not.

    The diagonal preconditioner is what lets the small-curvature coordinate move at
    all before the budget runs out; without it every step is sized for the large
    curvature. ``tol`` still applies to the unscaled gradient.
    """
    scale = np.array([1e12, 1.0])
    fun = _quadratic(scale)
    x0 = mx.array([1.0, 1.0])
    config = LBFGSConfig(max_iter=20, tol=1e-3)

    plain = minimize(fun, x0, config)
    conditioned = minimize(fun, x0, config, preconditioner=1.0 / scale)

    assert conditioned.converged and conditioned.stop_reason == "gtol"
    assert conditioned.n_iter < plain.n_iter or not plain.converged
    np.testing.assert_allclose(np.asarray(conditioned.x), 0.0, atol=1e-3)


@pytest.mark.parametrize(
    "preconditioner", [np.zeros(2), -np.ones(2), np.array([1.0, np.inf]), np.ones(3)]
)
def test_an_invalid_preconditioner_is_rejected(preconditioner):
    with pytest.raises(ValueError, match="preconditioner"):
        minimize(_quadratic([1.0, 2.0]), mx.ones((2,)), LBFGSConfig(), preconditioner=preconditioner)


# -------------------------------------------------------- dense initial inverse Hessian


def _correlated_quadratic(n=16, rho=0.99):
    """``½xᵀAx − bᵀx`` with ``A`` the AR(ρ) correlation matrix: condition ~4e4."""
    lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    A = rho**lags
    b = np.linspace(-1.0, 1.0, n)
    A32, b32 = mx.array(A.astype(np.float32)), mx.array(b.astype(np.float32))

    def fun(x):
        Ax = A32 @ x
        return float(0.5 * mx.sum(x * Ax) - mx.sum(b32 * x)), Ax - b32

    return fun, A, b


def test_an_exact_dense_preconditioner_takes_one_newton_step():
    """With ``P = A⁻¹`` the first direction ``−Pg`` is the Newton step, and its unit
    trial lands on the minimizer: one iteration, one line-search evaluation. From the
    identity, the same quadratic stalls on the float32 floor after dozens."""
    fun, A, b = _correlated_quadratic()
    P = np.linalg.inv(A)
    config = LBFGSConfig(tol=1e-4, max_iter=500)

    dense = minimize(fun, mx.zeros((16,)), config, dense_preconditioner=DensePreconditioner(
        lambda v: P @ v
    ))
    plain = minimize(fun, mx.zeros((16,)), config)

    assert dense.converged and (dense.n_iter, dense.n_fev) == (1, 2)
    assert dense.preconditioner == "dense"
    assert plain.n_iter > 10 * dense.n_iter
    np.testing.assert_allclose(np.asarray(dense.x), np.linalg.solve(A, b), atol=1e-3)


def test_a_deferred_dense_preconditioner_is_resolved_once_after_the_first_evaluation():
    fun, A, _ = _correlated_quadratic()
    events = []

    def counted(x):
        events.append("evaluate")
        return fun(x)

    def deferred():
        events.append("resolve")
        return DensePreconditioner(lambda v: np.linalg.solve(A, v))

    state = minimize(counted, mx.zeros((16,)), LBFGSConfig(tol=1e-4), dense_preconditioner=deferred)

    assert events[:2] == ["evaluate", "resolve"] and events.count("resolve") == 1
    assert state.preconditioner == "dense" and state.converged


@pytest.mark.parametrize("config", [LBFGSConfig(tol=1e3), LBFGSConfig(max_iter=0)],
                         ids=["optimal-start", "no-budget"])
def test_a_deferred_dense_preconditioner_is_not_resolved_without_a_step(config):
    fun, _, _ = _correlated_quadratic()

    def deferred():
        raise AssertionError("resolved for a run that takes no step")

    state = minimize(fun, mx.zeros((16,)), config, dense_preconditioner=deferred)
    assert state.n_iter == 0 and state.preconditioner == "identity"


def test_a_deferred_none_keeps_the_diagonal():
    """``None`` from the deferred callable is the caller declining: the run is the
    diagonal-preconditioned one, bit for bit."""
    scale = np.array([1e6, 1.0, 1.0, 30.0])
    fun = _quadratic(scale)
    x0 = mx.array([1.0, 1.0, -2.0, 0.5])
    config = LBFGSConfig(tol=1e-6)

    declined = minimize(fun, x0, config, preconditioner=1.0 / scale, dense_preconditioner=lambda: None)
    diagonal = minimize(fun, x0, config, preconditioner=1.0 / scale)

    assert declined.preconditioner == diagonal.preconditioner == "diagonal"
    assert (declined.n_iter, declined.n_fev, declined.objective) == (
        diagonal.n_iter, diagonal.n_fev, diagonal.objective
    )
    np.testing.assert_array_equal(np.asarray(declined.x), np.asarray(diagonal.x))


def test_a_deferred_dense_preconditioner_of_the_wrong_type_is_an_error():
    fun, _, _ = _correlated_quadratic()
    with pytest.raises(TypeError, match="DensePreconditioner"):
        minimize(fun, mx.zeros((16,)), LBFGSConfig(), dense_preconditioner=lambda: np.ones(16))


def test_a_dense_operator_whose_search_fails_falls_back_to_the_diagonal():
    """An operator whose Newton step is 1e30 times too long: the line search, 32
    evaluations long, cannot shrink it to any decrease, and the run would stop on a
    line-search failure far from the minimizer. It leaves the operator instead, starts
    again from the same point as the diagonal path would, and converges there."""
    scale = np.array([4.0, 1.0, 1.0, 9.0])
    fun = _quadratic(scale)
    x0 = mx.array([1.0, 1.0, -2.0, 0.5])
    config = LBFGSConfig(tol=1e-6)
    wrong = DensePreconditioner(lambda v: 1e30 * v)

    state = minimize(fun, x0, config, preconditioner=1.0 / scale, dense_preconditioner=wrong)

    assert state.fallback and state.converged and state.stop_reason == "gtol"
    assert state.preconditioner == "diagonal"
    np.testing.assert_allclose(np.asarray(state.x), np.zeros(4), atol=1e-5)


def test_a_dense_run_that_converges_never_falls_back():
    fun, A, _ = _correlated_quadratic()
    state = minimize(fun, mx.zeros((16,)), LBFGSConfig(tol=1e-4), dense_preconditioner=(
        DensePreconditioner(lambda v: np.linalg.solve(A, v))
    ))
    assert state.converged and state.preconditioner == "dense" and not state.fallback


def test_the_callback_sees_every_accepted_iterate():
    seen = []
    state = minimize(
        _quadratic([1.0, 10.0]),
        mx.array([3.0, 1.0]),
        LBFGSConfig(tol=1e-6),
        callback=lambda x: seen.append(x.copy()),
    )
    assert len(seen) == state.n_iter
    np.testing.assert_array_equal(seen[-1], np.asarray(state.x, dtype=np.float64))
    assert all(x.dtype == np.float64 and x.shape == (2,) for x in seen)


# -------------------------------------------------------------- host bookkeeping


def test_the_bookkeeping_never_touches_the_device(monkeypatch):
    """One synchronization per evaluation of ``fun``, and none of the solver's own.

    Every inner product used to be ``float(mx.sum(a * b))`` — a blocking device
    round trip of ~120 µs, 12 to 33 of them per iteration — which is what made an
    iteration cost milliseconds. The two-loop recursion, the curvature test and
    ``max|g|`` are host NumPy now; this pins that down by making the MLX reductions
    unusable while the solver runs, and counting evaluations.
    """
    a = np.array([1.0, 4.0, 9.0, 25.0], dtype=np.float32)
    real_eval = mx.eval
    evals = []

    def fun(x):
        grad = mx.array(a) * x
        real_eval(grad)
        evals.append(1)
        v = np.asarray(x, dtype=np.float64)
        return float(0.5 * np.sum(a * v * v)), grad

    def forbidden(*args, **kwargs):
        raise AssertionError("L-BFGS bookkeeping must not run on the device")

    for name in ("sum", "max", "abs", "eval"):
        monkeypatch.setattr(mx, name, forbidden)
    state = minimize(fun, mx.array([1.0, -2.0, 0.5, 3.0]), LBFGSConfig(tol=1e-6))
    monkeypatch.undo()

    assert state.converged
    assert len(evals) == state.n_fev


def test_the_inner_products_share_their_errstates(monkeypatch):
    """The solver's products report no floating-point flags (``lbfgs._dot``), but not
    through one ``np.errstate`` per product: entering one costs about half a
    microsecond on NumPy 2 and two on NumPy 1.23, and one per product — ``2m + 1`` in
    the two-loop recursion alone — added 8-17 µs to every iteration, 2-11% of a
    directly imported ``LogisticRegression`` fit of 3-10 ms. One covers the recursion
    and its slope, one the curvature test, and one each line-search slope, so an
    iteration with a full history of curvature pairs enters no more of them than the
    first one does.
    """
    real = np.errstate
    entered = []

    def counting(*args, **kwargs):
        entered.append(1)
        return real(*args, **kwargs)

    fun = _quadratic(np.logspace(0.0, 2.0, 40))
    monkeypatch.setattr(np, "errstate", counting)
    state = minimize(fun, mx.array(np.ones(40, dtype=np.float32)), LBFGSConfig(tol=1e-3))
    monkeypatch.undo()

    assert state.converged and state.n_iter > LBFGSConfig().history_size  # a full history
    # Per iteration: the direction and the curvature test; per evaluation after the
    # first, its slope; once, the first step's length.
    assert len(entered) <= 2 * state.n_iter + state.n_fev


# ----------------------------------------------------------------- float32 floor


def _float32_valued(scale, center=0.0):
    """``1 + ½Σ a(x − c)²`` whose *value* is rounded to float32, as an MLX sum is.

    Near the optimum the decrease a step buys drops below float32's spacing at 1.0
    (1.2e-7) while the float32 gradient is still accurate — the situation a
    float32-accumulated log-loss is in when ``tol`` is small.
    """
    a = np.asarray(scale, dtype=np.float64)

    def fun(x):
        v = np.asarray(x, dtype=np.float64)
        value = float(np.float32(1.0 + 0.5 * np.sum(a * (v - center) ** 2)))
        return value, mx.array((a * (v - center)).astype(np.float32))

    return fun


def test_approximate_wolfe_reaches_a_tol_the_objective_cannot_resolve():
    """The objective cannot see the last steps; the slope can.

    Accepting only on a decrease of the float32 value stalled at ``max|g|`` about 50x
    above the float32 gradient floor, with a line-search failure and a
    ConvergenceWarning where scikit-learn converges silently.
    """
    fun = _float32_valued(np.linspace(1.0, 100.0, 8))
    x0 = mx.full((8,), 0.1)

    state = minimize(fun, x0, LBFGSConfig(tol=1e-6, max_iter=500))
    assert state.converged and state.stop_reason == "gtol", state.stop_reason

    exact = minimize(fun, x0, LBFGSConfig(tol=1e-6, max_iter=500, objective_noise=0.0))
    assert not exact.converged and exact.stop_reason == "line-search-failure"


def test_a_tol_below_the_float32_floor_stops_on_the_floor():
    """``tol=0`` is honored as "run to the float32 floor", not as "run to max_iter".

    Centered away from zero, so the iterate itself runs out of float32 resolution.
    """
    fun = _float32_valued(np.linspace(1.0, 100.0, 8), center=1.0)
    state = minimize(fun, mx.zeros((8,)), LBFGSConfig(tol=0.0, max_iter=1000))

    assert state.stop_reason == "float32-floor"
    assert not state.converged
    assert state.n_iter < 200
    assert state.at_resolution_floor
    assert state.initial_grad_max_abs == pytest.approx(100.0)


# --------------------------------------------------------- pinned logistic trajectories

#: ``(n_classes, C, fit_intercept) -> (n_iter, objective)`` measured on
#: ``make_classification(5000, 20, n_informative=12, n_redundant=4, random_state=0)``,
#: on an M4 Max: the same iteration counts on the GPU and CPU devices, objectives up to
#: 4.9e-9 apart. Re-pinned when the first step became L-BFGS-B's 1/‖d‖₂: the binary
#: runs take two or three more iterations than the 16-17 of 0.1.0a2 (scikit-learn: 18-24),
#: the multinomial ones one fewer or one more than 30-47 (scikit-learn: 29-43), and every
#: objective is lower than it was — while fits scikit-learn finishes in one iteration
#: now do too. The GPU objectives' batched gradient, spelled-out softmax and float64
#: finish then left every count unchanged, on MLX 0.31.2 and 0.32.2, and moved the GPU
#: objectives by at most 8e-10 relative, both ways. Changes to the solver's internals may
#: reach the same point in fewer iterations or at a lower objective, never the reverse.
#: These are the diagonal path's, pinned with the dense preconditioner off.
_PINNED = {
    (2, 0.01, True): (18, 0.5293678799),
    (2, 1.0, True): (19, 0.5263766441),
    (2, 1.0, False): (18, 0.5294598445),
    (2, 100.0, True): (19, 0.5263454056),
    (5, 0.01, True): (43, 1.0993983064),
    (5, 1.0, True): (46, 1.0847425512),
    (5, 1.0, False): (29, 1.0866184672),
    (5, 100.0, True): (46, 1.0845855493),
}

#: The same problems as the solver takes them by default: four of the twenty features
#: are linear combinations of others, so the dense preconditioner's gate fires
#: (``docs/fp32_policy.md`` §2.3) and the runs take 6 iterations where the diagonal
#: path takes 18-19, and 8-12 where it takes 29-46, to a lower objective in every
#: case. Measured with MLX 0.32.2 on an M4 Max: the same counts on both devices,
#: objectives up to 1.4e-10 apart.
_PINNED_DENSE = {
    (2, 0.01, True): (6, 0.5293678737),
    (2, 1.0, True): (6, 0.5263766419),
    (2, 1.0, False): (6, 0.5294598380),
    (2, 100.0, True): (6, 0.5263454034),
    (5, 0.01, True): (11, 1.0993982091),
    (5, 1.0, True): (12, 1.0847424806),
    (5, 1.0, False): (8, 1.0866183786),
    (5, 100.0, True): (12, 1.0845854610),
}


@pytest.fixture(scope="module")
def _pinned_data():
    return {
        k: make_classification(
            n_samples=5000, n_features=20, n_informative=12, n_redundant=4,
            n_classes=k, random_state=0,
        )
        for k in (2, 5)
    }


@pytest.mark.parametrize("dense", [False, True], ids=["diagonal", "dense"])
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("key", sorted(_PINNED))
def test_logistic_trajectories_do_not_regress(key, device, dense, _pinned_data):
    from mlxlearn._common.device import gpu_available

    if device == "gpu" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    n_classes, C, fit_intercept = key
    pinned_iter, pinned_objective = (_PINNED_DENSE if dense else _PINNED)[key]
    X, y = _pinned_data[n_classes]

    solution = solve_logistic(
        X, y, n_classes=n_classes,
        config=LogisticConfig(
            C=C, fit_intercept=fit_intercept, device=device, dense_preconditioner=dense
        ),
    )
    objective = reference_objective(
        solution.coef, solution.intercept, X, y, n_classes=n_classes, C=C
    )

    assert solution.preconditioner == ("dense" if dense else "diagonal")
    assert solution.converged and solution.stop_reason == "gtol"
    # Headroom for another chip's float32 accumulation order, which is all that
    # differs between Apple GPUs and between Accelerate builds. One iteration: the
    # (2, 1.0, False) run stops at max|g| = 0.995 tol, so a wobble of half a percent
    # takes one more step. And 1e-8 relative on the objective: reordering only the
    # rows of the data moves it by up to about 4e-9 on an objective near 1.1 — past
    # the 2e-9 absolute this used to allow in 22 of 640 orders — while an actual
    # regression shows up orders larger, and the parity budget is 1e-5.
    assert solution.n_iter <= pinned_iter + 1
    assert objective <= pinned_objective * (1 + 1e-8)
