"""Host OLS solver: paths, certification and the tiled QR, against LAPACK gelsd."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import linalg

from mlxlearn._solvers import ols
from mlxlearn._solvers.ols import OLSConfig, reference_objective, solve_ols


def _gelsd(X, y, *, rcond, fit_intercept=True, sample_weight=None):
    """scikit-learn's dense computation, spelled out: centre in X's dtype, solve in float64."""
    X = np.asarray(X)
    if X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float64)
    y2 = np.asarray(y, dtype=X.dtype).reshape(len(X), -1)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=X.dtype)
    if fit_intercept:
        xo = np.average(X, axis=0, weights=w).astype(X.dtype)
        yo = np.average(y2, axis=0, weights=w)
    else:
        xo, yo = np.zeros(X.shape[1], X.dtype), np.zeros(y2.shape[1], X.dtype)
    Xc, yc = X - xo, y2 - yo
    if w is not None:
        root = np.sqrt(w)[:, None]
        Xc, yc = Xc * root, yc * root
    coef, _, rank, singular = linalg.lstsq(Xc.astype(np.float64), yc.astype(np.float64), cond=rcond)
    return coef.T, yo.astype(np.float64) - coef.T @ xo.astype(np.float64), int(rank), singular


def _spectrum(n, d, singular, seed=0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    X = (U * singular) @ V.T
    return X, X @ rng.standard_normal(d) + 1e-3 * rng.standard_normal(n)


def _check(solution, X, y, rcond, *, fit_intercept=True, sample_weight=None, coef_rtol=1e-10):
    coef, intercept, rank, singular = _gelsd(
        X, y, rcond=rcond, fit_intercept=fit_intercept, sample_weight=sample_weight
    )
    assert solution.rank == rank
    np.testing.assert_allclose(solution.singular, singular, rtol=1e-8, atol=1e-11 * singular[0])
    scale = float(np.max(np.abs(coef)))
    np.testing.assert_allclose(solution.coef, coef, rtol=coef_rtol, atol=coef_rtol * scale)
    np.testing.assert_allclose(solution.intercept, intercept, rtol=coef_rtol,
                               atol=coef_rtol * (scale + float(np.max(np.abs(intercept)))) + 1e-14)


def test_reference_objective_is_zero_on_a_perfect_line():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert reference_objective(np.array([1.0]), 0.0, X, y) == pytest.approx(0.0)


def test_reference_objective_matches_mean_squared_error_scale():
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 0.0])
    assert reference_objective(np.array([1.0]), 0.0, X, y) == pytest.approx(0.25)


def test_config_rejects_a_negative_cutoff_and_threads():
    with pytest.raises(ValueError, match="rcond"):
        OLSConfig(rcond=-1.0)
    with pytest.raises(ValueError, match="rcond"):
        OLSConfig(rcond=float("nan"))
    with pytest.raises(ValueError, match="threads"):
        OLSConfig(threads=-1)


def test_zero_weight_sum_with_an_intercept_is_undefined():
    X = np.zeros((6, 3))
    y = np.arange(6, dtype=np.float64)
    with pytest.raises(ZeroDivisionError, match="Weights sum to zero"):
        solve_ols(X, y, sample_weight=np.zeros(6), config=OLSConfig())


def test_zero_weights_without_an_intercept_give_the_empty_model():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(50, 4)), rng.normal(size=50)
    sol = solve_ols(X, y, sample_weight=np.zeros(50), config=OLSConfig(fit_intercept=False))
    assert sol.rank == 0
    np.testing.assert_array_equal(sol.coef, 0.0)


def test_well_conditioned_designs_take_the_gram_path():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6000, 40))
    y = X @ rng.normal(size=40) + rng.normal(size=6000)
    sol = solve_ols(X, y, config=OLSConfig())
    assert sol.method == "gram"
    _check(sol, X, y, 1e-6, coef_rtol=1e-12)


@pytest.mark.parametrize("rcond", [1e-6, 6000 * np.finfo(float).eps])
def test_exact_rank_loss_takes_the_ritz_path(rcond):
    rng = np.random.default_rng(2)
    X = np.zeros((6000, 20))
    X[np.arange(6000), rng.integers(0, 20, 6000)] = 1.0  # one-hot, collinear with the intercept
    X = np.hstack([X, rng.normal(size=(6000, 10))])
    y = X @ rng.normal(size=30) + rng.normal(size=6000)
    sol = solve_ols(X, y, config=OLSConfig(rcond=rcond))
    assert sol.method == "ritz"
    _check(sol, X, y, rcond, coef_rtol=1e-11)


def test_small_singular_values_are_measured_on_the_data():
    """Squared, sigma = 1e-7 sigma_max is lost in a Gram; measured on X it is exact."""
    singular = np.concatenate([np.linspace(1.0, 0.5, 14), [1e-7, 3e-7]])
    X, y = _spectrum(5000, 16, singular, seed=3)
    sol = solve_ols(X, y, config=OLSConfig(fit_intercept=False, rcond=5000 * np.finfo(float).eps))
    assert sol.method == "ritz"
    assert sol.rank == 16
    np.testing.assert_allclose(np.sort(sol.singular), np.sort(singular), rtol=1e-6)
    _check(sol, X, y, 5000 * np.finfo(float).eps, fit_intercept=False, coef_rtol=1e-7)


def test_a_singular_value_at_the_cutoff_takes_the_exact_path():
    singular = np.logspace(0, -9, 16)
    assert singular[10] == pytest.approx(1e-6)
    singular[10] = 1e-6 * (1 + 1e-10)  # the only value near the cutoff
    X, y = _spectrum(8192, 16, singular, seed=4)
    sol = solve_ols(X, y, config=OLSConfig(fit_intercept=False, rcond=1e-6))
    # Inside scikit-learn's rounding band: its own arithmetic decides, bit for bit.
    assert sol.method == "lstsq"
    _check(sol, X, y, 1e-6, fit_intercept=False, coef_rtol=1e-12)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("shape", [(40, 200), (300, 1200)])
def test_wide_designs_use_the_dual_gram(weighted, shape):
    """With an intercept the constant direction is deflated (method "dual")."""
    rng = np.random.default_rng(5)
    X = 5 + rng.normal(size=shape)
    y = rng.normal(size=(shape[0], 2))
    w = rng.uniform(0.5, 1.5, shape[0]) if weighted else None
    sol = solve_ols(X, y, sample_weight=w, config=OLSConfig())
    assert sol.method == "dual"  # the constant direction is deflated and certified null
    _check(sol, X, y, 1e-6, sample_weight=w, coef_rtol=1e-9)
    sol = solve_ols(X, y, sample_weight=w, config=OLSConfig(fit_intercept=False))
    assert sol.method == "gram"
    _check(sol, X, y, 1e-6, fit_intercept=False, sample_weight=w, coef_rtol=1e-9)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_float32_input_is_widened_exactly(dtype):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(20000, 12)).astype(dtype)
    y = rng.normal(size=20000)
    sol = solve_ols(X, y, config=OLSConfig())
    # y is cast to X's dtype first, and X and y are centred in it, as scikit-learn does;
    # that matrix is then solved in float64.
    _check(sol, X, y, 1e-6, coef_rtol=1e-12)


@pytest.mark.parametrize("shape", [(602, 301), (5000, 300), (20000, 33), (8192, 65), (3000, 129)])
def test_tiled_qr_is_an_orthogonal_reduction(shape):
    rng = np.random.default_rng(7)
    A = rng.normal(size=shape)
    for threads in (1, 4, 16):
        S = ols._tiled_r(A, threads)
        assert S.shape == (shape[1], shape[1])
        np.testing.assert_allclose(np.triu(S), S)
        s = np.linalg.svd(S, compute_uv=False)
        ref = np.linalg.svd(A, compute_uv=False)
        np.testing.assert_allclose(s, ref, rtol=1e-12, atol=1e-13 * ref[0])


def test_exact_tall_path_matches_gelsd_on_a_rank_deficient_design():
    rng = np.random.default_rng(8)
    A = rng.normal(size=(9000, 10))
    X = np.hstack([A, A @ rng.normal(size=(10, 5))])
    y = X @ rng.normal(size=15) + rng.normal(size=9000)
    Ac = np.hstack([X - X.mean(0), (y - y.mean())[:, None]])
    coef, rank, singular, method = ols._qr_tall(Ac, 15, 1e-6, lambda smax: 1e-13 * smax, 8)
    ref, _, ref_rank, ref_singular = linalg.lstsq(Ac[:, :15], Ac[:, 15:], cond=1e-6)
    assert method == "qr" and rank == ref_rank == 10
    np.testing.assert_allclose(singular, ref_singular, rtol=1e-10, atol=1e-12 * ref_singular[0])
    np.testing.assert_allclose(coef, ref, rtol=1e-10, atol=1e-12)


def test_overflowing_gram_falls_back_to_the_exact_path():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(20000, 6)) * 1e160  # above the tiny tier: the Gram overflows
    assert X.shape[0] * (X.shape[1] + 1) ** 2 > ols._SMALL_WORK
    y = X @ np.ones(6) / 1e160
    sol = solve_ols(X, y, config=OLSConfig())
    assert sol.method in ("qr", "lstsq")
    _check(sol, X, y, 1e-6, coef_rtol=1e-10)


def test_thread_count_does_not_change_the_rank_or_path():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(50000, 24)) * rng.uniform(0.01, 100, 24)
    y = X @ rng.normal(size=24) + rng.normal(size=50000)
    one = solve_ols(X, y, config=OLSConfig(threads=1))
    many = solve_ols(X, y, config=OLSConfig(threads=12))
    assert one.method == many.method and one.rank == many.rank
    np.testing.assert_array_equal(one.coef, many.coef)
    np.testing.assert_array_equal(one.singular, many.singular)


def test_the_exact_fallback_is_stock_to_the_bit():
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(11)
    X = np.asfortranarray(100 + rng.normal(size=(32, 120)))
    y = rng.normal(size=32)
    w = rng.uniform(0.5, 1.5, 32)
    _check_stock_bits(X[:, :1].copy(), y, w)  # d = 1, weighted
    for dtype in (np.float64, np.float32):
        Xd, yd, wd = X.astype(dtype), y.astype(dtype), w.astype(dtype)
        ref = SkLinearRegression().fit(Xd, yd, sample_weight=wd)
        sol = ols._stock_lstsq(Xd, yd, wd, True, ref.tol if hasattr(ref, "tol") else 1e-6)
        if not ols_cutoff_is_tol():
            sol = ols._stock_lstsq(Xd, yd, wd, True, max(Xd.shape) * np.finfo(dtype).eps)
        assert sol.rank == ref.rank_
        np.testing.assert_array_equal(sol.coef.ravel(), np.asarray(ref.coef_, dtype=np.float64))
        np.testing.assert_array_equal(sol.singular, np.asarray(ref.singular_, dtype=np.float64))
        assert sol.intercept[0] == np.float64(ref.intercept_)


def ols_cutoff_is_tol():
    import sklearn

    return tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 9)


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_gets_a_working_thread_pool():
    import os
    import sys

    if not hasattr(os, "fork") or sys.platform == "win32":
        pytest.skip("no fork")
    rng = np.random.default_rng(13)
    X, y = rng.normal(size=(60000, 16)), rng.normal(size=60000)
    solve_ols(X, y, config=OLSConfig(threads=4))  # the parent has a live pool
    pid = os.fork()
    if pid == 0:  # child: must not deadlock on the inherited executor
        try:
            sol = solve_ols(X, y, config=OLSConfig(threads=4))
            os._exit(0 if sol.rank == 16 else 1)
        except BaseException:
            os._exit(2)
    import signal
    import time

    deadline = time.monotonic() + 120
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail("the forked child hung on the inherited thread pool")
        time.sleep(0.05)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def test_omp_num_threads_caps_the_thread_budget(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert ols._thread_budget() == min(2, ols._HOST_THREADS)
    assert ols._thread_count(1 << 26, 0) <= 2
    monkeypatch.setenv("OMP_NUM_THREADS", "not-a-number")
    assert ols._thread_budget() == ols._HOST_THREADS
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert ols._thread_budget() == ols._HOST_THREADS


def _check_stock_bits(X, y, w):
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    ref = SkLinearRegression().fit(X, y, sample_weight=w)
    rcond = 1e-6 if ols_cutoff_is_tol() else max(X.shape) * np.finfo(X.dtype).eps
    sol = ols._stock_lstsq(X, y, w, True, rcond)
    assert sol.rank == ref.rank_
    np.testing.assert_array_equal(sol.coef.ravel(), np.asarray(ref.coef_, dtype=np.float64))
    assert sol.intercept[0] == np.float64(ref.intercept_)


def test_hard_paths_are_bit_identical_across_thread_budgets():
    rng = np.random.default_rng(17)
    U, _ = np.linalg.qr(rng.normal(size=(40000, 24)))
    X = (U * np.logspace(0, -9, 24)) @ rng.normal(size=(24, 24))  # graded: tiled-QR path
    y = X @ rng.normal(size=24) + 1e-3 * rng.normal(size=40000)
    sols = [solve_ols(X, y, config=OLSConfig(fit_intercept=False, threads=t)) for t in (1, 3, 8)]
    for sol in sols[1:]:
        assert sol.method == sols[0].method and sol.rank == sols[0].rank
        np.testing.assert_array_equal(sol.coef, sols[0].coef)
        np.testing.assert_array_equal(sol.singular, sols[0].singular)


def test_overflowing_centring_raises_what_stock_raises():
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    from mlxlearn.linear_model import LinearRegression

    X = np.zeros((3000, 3))
    X[:, 0] = np.tile([1.7e308, 1.7e308, -1.7e308], 1000)  # the column sum overflows
    X[:, 1:] = np.random.default_rng(0).normal(size=(3000, 2))
    y = np.arange(3000.0)

    def outcome(cls):
        try:
            cls().fit(X, y)
        except Exception as exc:  # noqa: BLE001 - compared below
            return type(exc).__name__
        return "ok"

    with np.errstate(over="ignore", invalid="ignore"):
        stock = outcome(SkLinearRegression)  # stock also warns about the overflow
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ours = outcome(LinearRegression)
    assert ours == stock == "ValueError"


def test_wide_zero_weight_rows_are_dropped_and_reported_as_zero_singular_values():
    rng = np.random.default_rng(18)
    X = rng.normal(size=(120, 600)) + 3.0
    y = rng.normal(size=120)
    w = rng.uniform(0.5, 1.5, 120)
    w[::7] = 0.0
    sol = solve_ols(X, y, sample_weight=w, config=OLSConfig())
    assert sol.method == "dual"
    assert sol.singular.shape == (120,)
    assert np.count_nonzero(sol.singular == 0.0) >= int(np.sum(w == 0))
    _check(sol, X, y, 1e-6, sample_weight=w, coef_rtol=1e-9)


def test_a_gelsd_that_fails_silently_is_recomputed_with_gelss(monkeypatch):
    """macOS 27.2's dgelsd can return garbage (rank ~4.6e18, unsorted values)."""
    rng = np.random.default_rng(19)
    X = rng.normal(size=(300, 8))
    y = X @ rng.normal(size=8) + 0.1 * rng.normal(size=300)
    original = linalg.lstsq

    def broken(a, b, *args, lapack_driver=None, **kwargs):
        out = original(a, b, *args, lapack_driver=lapack_driver, **kwargs)
        if lapack_driver in (None, "gelsd"):
            coef, res, rank, s = out
            return np.zeros_like(coef) + 1e3, res, 4607182418800017408, s[::-1]
        return out

    monkeypatch.setattr(linalg, "lstsq", broken)
    sol = solve_ols(X, y, config=OLSConfig(rcond=1e-6))  # tiny: the exact path
    assert sol.method == "gelss"
    monkeypatch.setattr(linalg, "lstsq", original)
    _check(sol, X, y, 1e-6, coef_rtol=1e-10)


@pytest.mark.parametrize("m0, skipped", [(40, True), (10, False)])
def test_an_expensive_measurement_that_would_fail_is_skipped(m0, skipped):
    """Many likely-null directions under a cutoff at the product's rounding: no
    eigh, no product. A few are always measured (cheap, and they may be kept)."""
    d = 500
    lam = np.r_[np.zeros(m0), np.linspace(1.0, 2.0, d - m0)]
    tr = float(lam.sum())
    E = 1e-14 * tr

    def measure(mask):
        raise AssertionError("measured")

    floor = ols._gamma(d + 2) * np.sqrt(tr) * np.sqrt(m0)
    tiny = 0.5 * floor / np.sqrt(2.0)
    if skipped:
        assert ols._certify(lam, E, tr, tiny, measure, d, lambda s: 0.0) is None
    else:
        with pytest.raises(AssertionError, match="measured"):
            ols._certify(lam, E, tr, tiny, measure, d, lambda s: 0.0)
    # With a cutoff well above that rounding the same spectrum is measured.
    with pytest.raises(AssertionError, match="measured"):
        ols._certify(lam, E, tr, 1e-6, measure, d, lambda s: 0.0)


def test_small_kept_directions_are_measured_under_the_old_cutoff():
    """Review round 2: columns in very different units (no collinearity) under the
    1.7/1.8 cutoff are certified on the Ritz path, not left to the exact path."""
    rng = np.random.default_rng(36)
    n, d = 4096, 256
    X = rng.normal(size=(n, d))
    X[:, :16] *= 1e-6
    y = X @ rng.normal(size=d) + rng.normal(size=n)
    rcond = max(n, d) * np.finfo(np.float64).eps
    sol = solve_ols(X, y, config=OLSConfig(rcond=rcond))
    assert sol.method == "ritz"
    _check(sol, X, y, rcond, coef_rtol=1e-8)


def test_ritz_measures_a_few_small_directions_past_1024_features():
    rng = np.random.default_rng(21)
    n, d = 2400, 1100
    X = rng.normal(size=(n, d))
    X[:, 7] = X[:, 3]  # three exact null directions
    X[:, 500] = 2.0 * X[:, 11]
    X[:, 1000] = X[:, 20] - X[:, 21]
    y = X @ rng.normal(size=d) + rng.normal(size=n)
    sol = solve_ols(X, y, config=OLSConfig())
    assert sol.method == "ritz"
    _check(sol, X, y, 1e-6, coef_rtol=1e-9)


@pytest.mark.parametrize("weighted", [False, True])
def test_wide_intercept_direction_is_certified_under_the_old_cutoff(weighted):
    """1.7/1.8 cut at max(n, d) eps: the bound on ||Xc^T q|| must sit well below it."""
    rng = np.random.default_rng(22)
    n, d = 600, 2000
    X = rng.normal(size=(n, d)) + 3.0
    y = X @ rng.normal(size=d) + rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, n) if weighted else None
    rcond = max(n, d) * np.finfo(np.float64).eps
    sol = solve_ols(X, y, sample_weight=w, config=OLSConfig(rcond=rcond))
    assert sol.method == "dual"
    _check(sol, X, y, rcond, sample_weight=w, coef_rtol=1e-8)


def _mixed_scales(n, d, seed):
    """Numeric columns of scale 0.1-100 next to a one-hot block collinear with the intercept."""
    rng = np.random.default_rng(seed)
    h = d // 2
    onehot = np.zeros((n, d - h))
    onehot[np.arange(n), rng.integers(0, d - h, n)] = 1.0
    X = np.hstack([rng.normal(size=(n, h)) * rng.uniform(0.1, 100, h), onehot])
    return X, X @ rng.normal(size=d) + 0.1 * rng.normal(size=n)


def test_long_designs_measure_half_their_directions():
    """Half the directions sit below the split; on a long design the Ritz path takes them."""
    X, y = _mixed_scales(4096, 128, seed=23)
    assert ols._ritz_fraction(*X.shape) == ols._RITZ_FRACTION_LONG
    sol = solve_ols(X, y, config=OLSConfig())
    assert sol.method == "ritz"
    _check(sol, X, y, 1e-6, coef_rtol=1e-9)


def test_shorter_designs_leave_that_many_directions_to_the_exact_path(monkeypatch):
    X, y = _mixed_scales(1500, 128, seed=24)
    assert ols._ritz_fraction(*X.shape) == ols._RITZ_FRACTION

    def no_eigensolver(G):
        raise AssertionError("the screen should have stopped this fit")

    monkeypatch.setattr(ols, "_eigvalsh", no_eigensolver)
    sol = solve_ols(X, y, config=OLSConfig())
    assert sol.method == "lstsq"
    _check(sol, X, y, 1e-6)


def test_the_screen_refuses_small_directions_the_ritz_path_would_not_measure():
    rng = np.random.default_rng(25)
    for d, small, refused in [(1100, 3, False), (1100, 80, True), (2100, 1, True), (2100, 0, False)]:
        X = rng.normal(size=(d + 200, d))
        X[:, :small] = X[:, small:2 * small] if small else X[:, :0]
        G = X.T @ X
        assert ols._too_many_small(G, float(np.trace(G)), d + 300, ols._RITZ_FRACTION) is refused, (d, small)


@pytest.mark.parametrize("d", [1536, 2048])
def test_padded_eigensolver_matches_the_plain_one(d):
    """Sizes Accelerate's dsyevd handles slowly get a decoupled pad; it must vanish."""
    rng = np.random.default_rng(26)
    A = rng.normal(size=(d + 64, d))
    G = A.T @ A
    assert ols._padded(G) is not None
    lam = ols._eigvalsh(G)
    ref = linalg.eigh(G, eigvals_only=True)
    np.testing.assert_allclose(lam, ref, rtol=1e-10, atol=1e-12 * ref[-1])
    lam2, V = ols._eigh(G)
    np.testing.assert_allclose(lam2, ref, rtol=1e-10, atol=1e-12 * ref[-1])
    assert np.abs(V.T @ V - np.eye(d)).max() < 1e-12
    assert np.abs(G @ V - V * lam2).max() < 1e-11 * ref[-1]


def test_a_pad_equal_to_an_eigenvalue_stays_decoupled():
    """A diagonal Gram (one-hot design, no intercept): the pad equals lambda_max."""
    rng = np.random.default_rng(27)
    D = np.diag(rng.uniform(1.0, 5.0, 1536))
    D[7, 7] = 5.0
    lam, V = ols._eigh(D)
    np.testing.assert_allclose(np.sort(lam), np.sort(np.diag(D)), rtol=1e-14)
    assert np.abs(V.T @ V - np.eye(1536)).max() < 1e-13
    assert np.abs(D @ V - V * lam).max() < 1e-13


@pytest.mark.parametrize("old_cutoff", [False, True], ids=["tol", "max(n,d)*eps"])
@pytest.mark.parametrize("where", ["at the cutoff", "inside the band", "clear"])
def test_certification_margins_decide_a_measured_direction(monkeypatch, old_cutoff, where):
    """Two small directions, well under the Ritz limit, are measured; one sits near the cutoff.

    At the cutoff mlxlearn's own measurement bound refuses. 3e-14 sigma_max above
    it, that bound would accept but scikit-learn's rounding band (about 6e-14
    here) does not: only stock's arithmetic can decide. A clear margin
    certifies. Under the 1.7/1.8 cutoff a kept value must also stand well above
    the Gram's error budget for the refinement to be trusted, hence the larger
    margin there.
    """
    n, d = 4096, 20
    rcond = n * np.finfo(np.float64).eps if old_cutoff else 1e-6
    singular = np.ones(d)
    if where == "at the cutoff":
        singular[-2] = rcond * (1 + 1e-13)
    elif where == "inside the band":
        singular[-2] = rcond + 3e-14
    else:
        singular[-2] = rcond * (200.0 if old_cutoff else 1.5)
    at_the_cutoff = where != "clear"
    singular[-1] = 1e-9 * rcond
    X, y = _spectrum(n, d, singular, seed=28)
    measured, exact = [], []
    for name, log in (("_tall_svd", measured), ("_stock_lstsq", exact)):
        original = getattr(ols, name)
        monkeypatch.setattr(ols, name, lambda *a, _f=original, _l=log, **k: _l.append(1) or _f(*a, **k))
    sol = solve_ols(X, y, config=OLSConfig(rcond=rcond, fit_intercept=False))
    assert measured, "the small directions were never measured"
    assert bool(exact) is at_the_cutoff
    assert sol.method == ("lstsq" if at_the_cutoff else "ritz")
    kappa = 1.0 / singular[-2]
    _check(sol, X, y, rcond, fit_intercept=False, coef_rtol=max(1e-9, 1e3 * kappa * np.finfo(float).eps))


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("weighted", [False, True])
def test_the_exact_fallback_is_stock_to_the_bit_with_many_targets(threads, layout, weighted):
    """Threaded Fortran-ordered copies of X (several row blocks) and stock-layout
    copies of several targets round as stock's."""
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(29)
    n, d, k = 8000, 40, 6  # n > 2 * 2**17 / d: the copy runs in several blocks
    X = 50 + rng.normal(size=(n, d)) * rng.uniform(0.1, 10, d)
    Y = X @ rng.normal(size=(d, k)) + 1e3 + rng.normal(size=(n, k))
    if layout == "F":
        X, Y = np.asfortranarray(X), np.asfortranarray(Y)
    w = rng.uniform(0.5, 1.5, n) if weighted else None
    ref = SkLinearRegression().fit(X, Y, sample_weight=w)
    rcond = 1e-6 if ols_cutoff_is_tol() else max(n, d) * np.finfo(np.float64).eps
    sol = ols._stock_lstsq(X, Y, w, True, rcond, threads)
    assert sol.rank == ref.rank_
    np.testing.assert_array_equal(sol.coef, ref.coef_)
    np.testing.assert_array_equal(sol.intercept, ref.intercept_)
    np.testing.assert_array_equal(sol.singular, ref.singular_)


def test_the_exact_fallback_is_stock_to_the_bit_across_shapes():
    """Random tall and wide shapes, several targets, weights and layouts.

    scipy's batched lstsq rounds a wide matrix differently in the last bit when
    handed an in-place Fortran copy, so wide designs must keep stock's route.
    """
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(30)
    for _ in range(60):
        if rng.random() < 0.5:
            m = int(rng.integers(2, 300))
            n = int(rng.integers(m + 1, 3 * m + 2))
        else:
            n = int(rng.integers(1, 120))
            m = int(rng.integers(n + 1, 20 * n + 50))
        k = int(rng.choice([1, 2, 5]))
        X = rng.normal(size=(m, n)) * rng.uniform(0.01, 100, n) + 10 * rng.normal(size=n)
        Y = rng.normal(size=(m, k)) if k > 1 else rng.normal(size=m)
        if rng.random() < 0.3:
            X = np.asfortranarray(X)
        w = rng.uniform(0.5, 1.5, m) if rng.random() < 0.4 else None
        ref = SkLinearRegression().fit(X, Y, sample_weight=w)
        rcond = 1e-6 if ols_cutoff_is_tol() else max(m, n) * np.finfo(np.float64).eps
        sol = ols._stock_lstsq(X, Y, w, True, rcond, 4)
        shape = (m, n, k, w is not None, X.flags.f_contiguous)
        assert sol.rank == ref.rank_, shape
        np.testing.assert_array_equal(sol.coef, np.reshape(ref.coef_, sol.coef.shape), err_msg=str(shape))
        np.testing.assert_array_equal(sol.singular, ref.singular_, err_msg=str(shape))


def _aligned_wide_design(n, d, offset, s, scale, seed=0):
    """Column 0 = offset + s u; the rest B (scale C0) with B orthogonal to {1, u}.

    Stock's centred column 0 is s u + e 1 (e = rounding of its mean), so the
    design has a singular value sqrt(s^2 + nu^2) where the deflated matrix has s.
    """
    rng = np.random.default_rng(seed)
    one = np.ones(n) / np.sqrt(n)
    Q, _ = np.linalg.qr(np.column_stack([one, rng.standard_normal((n, n - 1))]))  # n <= 400
    u, B = Q[:, 1], Q[:, 2:]
    P, _ = np.linalg.qr(rng.standard_normal((n - 2, n - 2)))
    Q2, _ = np.linalg.qr(rng.standard_normal((d - 1, n - 2)))  # thin
    C0 = (P * np.linspace(1.0, 0.5, n - 2)) @ Q2.T
    X = np.empty((n, d))
    X[:, 1:] = B @ (scale * C0)
    X[:, 0] = offset + s * u
    return X, rng.standard_normal(n)


@pytest.mark.parametrize("place", [0.2, 0.5, 0.8])
def test_wide_drops_clear_the_deflated_rank_one_part(place):
    """Review round 2 (critical): a measured direction just under the cutoff in the
    deflated matrix may sit above it in the design; the drop must account for it."""
    n, d, rcond, s = 200, 400, 1e-6, 1e-3
    X, y = _aligned_wide_design(n, d, 1e9, s, 1e3)
    Xc = X - np.average(X, axis=0)
    q = np.ones(n) / np.sqrt(n)
    sd = np.linalg.svd(Xc, compute_uv=False)
    sz = np.linalg.svd(Xc - np.outer(q, q @ Xc), compute_uv=False)
    target = sz[-2] + place * (sd[-2] - sz[-2])  # put the cutoff between the two
    X, y = _aligned_wide_design(n, d, 1e9, s, 1e3 * target / (rcond * sd[0]))
    sol = solve_ols(X, y, config=OLSConfig(rcond=rcond))
    coef, _, rank, singular = _gelsd(X, y, rcond=rcond)
    assert sol.rank == rank
    scale = float(np.max(np.abs(coef)))
    np.testing.assert_allclose(sol.coef.ravel(), coef.ravel(), rtol=1e-6, atol=1e-6 * scale)


@pytest.mark.parametrize("offset", [0.0, 1e4, 1e6, 1e7])
@pytest.mark.parametrize("weighted", [False, True])
def test_wide_intercept_reports_the_design_singular_value(offset, weighted):
    """The deflated constant direction's singular value is measured, not its bound."""
    rng = np.random.default_rng(34)
    X = rng.standard_normal((200, 400)) + offset
    y = rng.standard_normal(200)
    w = rng.uniform(0.5, 2.0, 200) if weighted else None
    sol = solve_ols(X, y, sample_weight=w, config=OLSConfig(rcond=1e-6))
    _, _, rank, singular = _gelsd(X, y, rcond=1e-6, sample_weight=w)
    assert sol.rank == rank
    np.testing.assert_allclose(sol.singular, singular, rtol=1e-8, atol=1e-10 * singular[0])


def _noise_kept_case(seed):
    """Review round 2's draws: float32, exact rank loss, a cutoff below float32 noise."""
    rng = np.random.default_rng(seed)
    n = int(rng.choice([2000, 4000, 9000]))
    d = int(rng.choice([24, 40, 64, 100]))
    h = d - max(1, d // int(rng.choice([4, 8])))
    A = rng.standard_normal((n, h))
    X = np.hstack([A, A @ rng.standard_normal((h, d - h))]).astype(np.float32)
    k = int(rng.choice([1, 3, 12]))
    Y = X.astype(np.float64) @ rng.standard_normal((d, k)) + float(rng.choice([0.1, 10.0, 1e3])) * rng.standard_normal((n, k))
    Y = (Y[:, 0] if k == 1 else Y).astype(np.float32)  # the estimator casts y to X's dtype
    w = rng.uniform(0.2, 3.0, n).astype(np.float32) if rng.random() < 0.4 else None
    tol = float(rng.choice([1e-7, 3e-8, 1e-8, 1e-9]))
    return X, Y, w, tol


@pytest.mark.parametrize("seed", [1012, 1001])
def test_a_cutoff_that_keeps_noise_directions_stays_on_gelsd(seed, monkeypatch):
    """Review round 2: gelsd's backward error is legitimate residual, not a failure."""
    X, Y, w, tol = _noise_kept_case(seed)
    sol = ols._stock_lstsq(X, Y, w, True, tol)
    assert sol.method == "lstsq"
    if ols_cutoff_is_tol():
        from sklearn.linear_model import LinearRegression as SkLinearRegression

        ref = SkLinearRegression(tol=tol).fit(X, Y, sample_weight=w)
        assert sol.rank == ref.rank_
        np.testing.assert_array_equal(sol.coef.reshape(np.shape(ref.coef_)), ref.coef_)
    # Without the backward-error term the same result is (wrongly) rejected.
    monkeypatch.setattr(ols, "_GELSD_BACKWARD", 0.0)
    assert ols._stock_lstsq(X, Y, w, True, tol).method == "gelss"

def test_offset_near_duplicates_keep_stock_bits(monkeypatch):
    """Review round 2: X b - x_off b cancels for huge b on offset data; the
    plausibility test must work on the centred matrix and not reject gelsd
    (checked without the backward-error slack, which would also mask it)."""
    monkeypatch.setattr(ols, "_GELSD_BACKWARD", 0.0)
    rng = np.random.default_rng(10)
    gelss = 0
    for _ in range(120):
        n = int(rng.integers(200, 800))
        x1 = 1e4 + rng.normal(size=n)
        x2 = x1.copy()
        rows = rng.choice(n, int(rng.integers(1, 12)), replace=False)
        x2[rows] = np.nextafter(x2[rows], np.inf)
        X = np.c_[x1, x2, rng.normal(size=n)]
        y = rng.normal(size=n)
        rcond = max(n, 3) * np.finfo(float).eps
        sol = solve_ols(X, y, config=OLSConfig(rcond=rcond))
        gelss += sol.method == "gelss"
    assert gelss == 0


def test_map_runs_each_item_once_when_submission_fails_midway(monkeypatch):
    """At interpreter shutdown the pool refuses new futures; submitted units
    must not run again serially (a chunk scaled twice would corrupt A)."""
    calls = []

    class Refusing:
        def __init__(self):
            self.n = 0

        def submit(self, fn, *args):
            self.n += 1
            if self.n > 2:
                raise RuntimeError("cannot schedule new futures after shutdown")
            return ols._Ready(fn(*args))

    monkeypatch.setattr(ols, "_pool", lambda: Refusing())
    out = ols._map(lambda i: calls.append(i) or i * i, list(range(10)), 4)
    assert out == [i * i for i in range(10)]
    assert sorted(calls) == list(range(10))


def test_garbage_coefficients_with_a_sane_spectrum_are_caught_on_larger_fits(monkeypatch):
    """The residual part of the gelsd plausibility test (problems past its floor)."""
    rng = np.random.default_rng(37)
    X = rng.normal(size=(40000, 8))
    y = X @ rng.normal(size=8) + 0.1 * rng.normal(size=40000)
    assert X.size >= ols._PLAUSIBLE_RESIDUAL_WORK
    monkeypatch.setattr(ols, "_SMALL_WORK", 1 << 40)  # the exact path at this size
    original = linalg.lstsq

    def broken(a, b, *args, lapack_driver=None, **kwargs):
        out = original(a, b, *args, lapack_driver=lapack_driver, **kwargs)
        if lapack_driver in (None, "gelsd"):
            coef, res, rank, s = out
            return np.zeros_like(coef) + 1e3, res, rank, s  # rank and spectrum intact
        return out

    monkeypatch.setattr(linalg, "lstsq", broken)
    sol = solve_ols(X, y, config=OLSConfig(rcond=1e-6))
    monkeypatch.setattr(linalg, "lstsq", original)
    assert sol.method == "gelss"
    _check(sol, X, y, 1e-6, coef_rtol=1e-10)



@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_plausibility_accepts_stock_answers_on_near_duplicates(dtype):
    """Review round 2: near-duplicate columns (one ulp on a few rows), an offset
    and a noise-dominated target give stock large coefficients; a narrow-dtype
    product made the plausibility test reject stock's correct gelsd answer."""
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(5)
    off = 1e3 if dtype == np.float32 else 100.0
    for _ in range(25):
        n, d = 500, 6
        Z = rng.standard_normal((n, d))
        X = (off + Z).astype(dtype)
        nudge = rng.random(n) < 0.02
        X[:, 1] = X[:, 0]
        X[nudge, 1] = np.nextafter(X[nudge, 0], dtype(np.inf))
        y = (0.3 * Z[:, 2:] @ rng.standard_normal(d - 2) + 3.0 * rng.standard_normal(n)).astype(dtype)
        ref = SkLinearRegression().fit(X, y)
        rcond = 1e-6 if ols_cutoff_is_tol() else max(n, d) * np.finfo(dtype).eps
        sol = solve_ols(X, y, config=OLSConfig(rcond=rcond))
        assert sol.method == "lstsq"
        np.testing.assert_array_equal(sol.coef.ravel(), np.asarray(ref.coef_, dtype=np.float64))


@pytest.mark.skipif(not ols_cutoff_is_tol(), reason="tol is the dense cutoff from scikit-learn 1.9")
@pytest.mark.parametrize("tol", [0.3, 0.5])
def test_tol_truncates_a_well_conditioned_design(tol):
    """The m0 == 0 refusal: a cutoff among well-resolved eigenvalues is not certified."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20000, 10)) * np.linspace(1.0, 10.0, 10)
    y = X @ rng.standard_normal(10) + rng.standard_normal(20000)
    sol = solve_ols(X, y, config=OLSConfig(rcond=tol))
    assert sol.method in ("qr", "lstsq")
    _check(sol, X, y, tol, coef_rtol=1e-9)


@pytest.mark.parametrize("kind", ["gauss", "duplicates"])
def test_many_targets_use_separate_products(kind):
    """k > max(8, d // 4): X^T X and X^T Y are formed separately."""
    rng = np.random.default_rng(38)
    n, d, k = 20000, 16, 20
    X = rng.standard_normal((n, d))
    if kind == "duplicates":
        X[:, 5] = X[:, 2]
    Y = X @ rng.standard_normal((d, k)) + 0.1 * rng.standard_normal((n, k))
    assert k > max(8, d // 4)
    sol = solve_ols(X, Y, config=OLSConfig())
    assert sol.method == ("gram" if kind == "gauss" else "ritz")
    _check(sol, X, Y, 1e-6, coef_rtol=1e-10)


def test_a_tiny_second_target_sends_the_fit_to_the_exact_path():
    """Per-target magnitudes: X ~ 1e-120 times y ~ 1e-205 underflows in X^T y."""
    rng = np.random.default_rng(9)
    n, d = 3000, 12
    Z = rng.standard_normal((n, d))
    b = rng.standard_normal(d)
    X = Z * 1e-120
    tiny = (Z @ b + 0.1 * rng.standard_normal(n)) * 1e-205
    Y = np.column_stack([Z @ b, tiny])
    assert n * (d + 2) ** 2 > ols._SMALL_WORK
    sol = solve_ols(X, Y, config=OLSConfig())
    assert sol.method in ("lstsq", "qr")
    coef, _, rank, _ = _gelsd(X, Y, rcond=1e-6)
    assert sol.rank == rank
    np.testing.assert_allclose(sol.coef[1], coef[1], rtol=1e-8)
    assert np.abs(sol.coef[1]).max() > 0


def test_null_direction_bound_is_rigorous_and_tight():
    """||Xc^T q|| against an exact (rational) evaluation of the same products."""
    from fractions import Fraction

    rng = np.random.default_rng(39)
    n, d = 150, 12
    X = 1e4 + rng.standard_normal((n, d)) * np.logspace(0, 3, d)
    Xc = X - np.average(X, axis=0)
    q = rng.uniform(0.5, 2.0, n)
    q /= ols._nrm2(q)
    tr = float(np.einsum("ij,ij->", Xc, Xc))
    bound, z = ols._null_direction_bound(Xc, q, tr)
    exact_sq = sum((sum(Fraction(float(Xc[i, j])) * Fraction(float(q[i])) for i in range(n))) ** 2 for j in range(d))
    exact = float(exact_sq) ** 0.5
    assert bound >= exact
    assert bound <= exact + 2 * ols._gamma(ols._NULL_BLOCK + 8) * tr ** 0.5
    np.testing.assert_allclose(z, Xc.T @ q, rtol=1e-12, atol=1e-12 * tr ** 0.5)


def test_wide_fail_fast_skips_the_eigensolver(monkeypatch):
    """A constant direction too large for the cutoff is refused before eigvalsh."""
    rng = np.random.default_rng(40)
    X = rng.standard_normal((200, 400)) + 1e12
    y = rng.standard_normal(200)

    def no_eigensolver(G):
        raise AssertionError("the fail-fast check should have refused")

    monkeypatch.setattr(ols, "_eigvalsh", no_eigensolver)
    rcond = max(X.shape) * np.finfo(np.float64).eps
    sol = solve_ols(X, y, config=OLSConfig(rcond=rcond))
    assert sol.method == "lstsq"


@pytest.mark.parametrize("m, c, blocked", [(5000, 1, False), (5000, 3, False), (5000, 4, True),
                                           (2100, 60, False), (8000, 1281, False), (8000, 1280, True)])
def test_leaf_qr_uses_blocked_dgeqrf_only_on_swept_shapes(monkeypatch, m, c, blocked):
    from scipy.linalg import lapack

    seen = []

    def spy(a, lwork=None, overwrite_a=0):
        seen.append(lwork)
        return a, np.zeros(min(a.shape)), None, 0

    monkeypatch.setattr(lapack, "dgeqrf", spy)
    ols._leaf_r(np.zeros((m, c), order="F"))
    assert (seen[0] is not None) is blocked


@pytest.mark.parametrize("d, small, fraction, refused", [
    (200, 69, ols._RITZ_FRACTION, False), (200, 71, ols._RITZ_FRACTION, True),
    (200, 129, ols._RITZ_FRACTION_LONG, False), (200, 131, ols._RITZ_FRACTION_LONG, True),
    (1100, 64, ols._RITZ_FRACTION, False), (1100, 65, ols._RITZ_FRACTION, True),
])
def test_screen_boundaries(d, small, fraction, refused):
    """The screen refuses just above the Ritz limit plus its margin (exact null directions)."""
    rng = np.random.default_rng(41)
    A = rng.standard_normal((d + 300, d - small))
    X = np.hstack([A, A[:, np.arange(small) % (d - small)]])  # `small` exact null directions
    G = X.T @ X
    assert ols._too_many_small(G, float(np.trace(G)), d + 400, fraction) is refused


def test_near_square_designs_run_stock_arithmetic():
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(42)
    X = rng.standard_normal((600, 400)) + 2.0
    y = X @ rng.standard_normal(400) + rng.standard_normal(600)
    assert X.shape[0] * 401 ** 2 > ols._SMALL_WORK and 600 < ols._SQUARE_RATIO * 400
    rcond = 1e-6 if ols_cutoff_is_tol() else 600 * np.finfo(np.float64).eps
    sol = solve_ols(X, y, config=OLSConfig(rcond=rcond))
    assert sol.method == "lstsq"
    ref = SkLinearRegression().fit(X, y)
    np.testing.assert_array_equal(sol.coef.ravel(), ref.coef_)
    np.testing.assert_array_equal(sol.singular, ref.singular_)


def test_the_qr_path_retries_a_failed_gelsd_with_gelss(monkeypatch):
    X, y = _spectrum(20000, 30, np.logspace(0, -10, 30), seed=43)  # graded: the tiled QR
    original = linalg.lstsq

    def broken(a, b, *args, lapack_driver=None, **kwargs):
        out = original(a, b, *args, lapack_driver=lapack_driver, **kwargs)
        if lapack_driver in (None, "gelsd"):
            coef, res, rank, s = out
            return coef, res, 4607182418800017408, s[::-1]
        return out

    monkeypatch.setattr(linalg, "lstsq", broken)
    sol = solve_ols(X, y, config=OLSConfig(fit_intercept=False))
    monkeypatch.setattr(linalg, "lstsq", original)
    assert sol.method == "qr"
    _check(sol, X, y, 1e-6, fit_intercept=False, coef_rtol=1e-6)


def test_weighted_float32_on_a_certified_path():
    """float32 rows centred and weighted in float32, as stock does, then widened."""
    rng = np.random.default_rng(44)
    X = (3.0 + rng.standard_normal((20000, 12))).astype(np.float32)
    y = (X.astype(np.float64) @ rng.standard_normal(12) + rng.standard_normal(20000)).astype(np.float32)
    w = rng.uniform(0.5, 2.0, 20000).astype(np.float32)
    sol = solve_ols(X, y, sample_weight=w, config=OLSConfig())
    assert sol.method == "gram"
    _check(sol, X, y, 1e-6, sample_weight=w, coef_rtol=1e-9)


def test_wide_designs_with_many_repeated_rows_leave_the_dual_path():
    """Half the rows repeated: the screen refuses before the eigensolver."""
    rng = np.random.default_rng(45)
    base = rng.standard_normal((100, 800))
    X = np.vstack([base, base])
    y = rng.standard_normal(200)
    sol = solve_ols(X, y, config=OLSConfig(rcond=1e-6))
    assert sol.method == "lstsq"
    _check(sol, X, y, 1e-6)


@pytest.mark.parametrize("weighted", [False, True])
def test_wide_intercept_with_a_few_repeated_rows_measures_them(weighted):
    """Deflated dual path with exact null directions besides the constant one."""
    rng = np.random.default_rng(46)
    X = rng.standard_normal((200, 600)) + 3.0
    X[197:200] = X[10:13]  # three repeated rows: three more null directions
    y = rng.standard_normal(200)
    w = rng.uniform(0.5, 2.0, 200) if weighted else None
    sol = solve_ols(X, y, sample_weight=w, config=OLSConfig(rcond=1e-6))
    assert sol.method == "dual"
    _check(sol, X, y, 1e-6, sample_weight=w, coef_rtol=1e-8)


@pytest.mark.skipif(not ols_cutoff_is_tol(), reason="tol is the dense cutoff from scikit-learn 1.9")
@pytest.mark.parametrize("dtype, tol", [(np.float32, 1e-6), (np.float64, 1e-15)])
def test_wide_zero_weight_rows_respect_stocks_band(dtype, tol):
    """Review round 2: stock's gelsd sees dropped zero rows as rounding noise and
    can keep one when the cutoff sits inside its band; that rank is stock's call."""
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    for seed in range(3):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((120, 400)).astype(dtype)
        y = rng.standard_normal(120).astype(dtype)
        w = rng.uniform(0.5, 2.0, 120).astype(dtype)
        w[rng.choice(120, 20, replace=False)] = 0
        ref = SkLinearRegression(fit_intercept=False, tol=tol).fit(X, y, sample_weight=w)
        sol = solve_ols(X, y, sample_weight=w, config=OLSConfig(fit_intercept=False, rcond=tol))
        assert sol.rank == ref.rank_


@pytest.mark.parametrize("rcond, dtype, n", [(1.0, np.float64, 300), (0.0, np.float64, 300),
                                             (1500 * float(np.finfo(np.float16).eps), np.float16, 1500)])
def test_the_gelss_retry_uses_gelsds_effective_cutoff(monkeypatch, rcond, dtype, n):
    """Review round 2: gelsd maps a cutoff outside (0, 1) to machine epsilon; gelss
    takes it literally (rank 0 at 1), so the retry must be given gelsd's cutoff."""
    rng = np.random.default_rng(47)
    X = (rng.standard_normal((n, 4)) * np.array([1.0, 1.0, 0.5, 0.25])).astype(dtype)
    y = (X.astype(np.float64) @ np.array([1.0, -2.0, 0.5, 3.0]) + 0.01 * rng.standard_normal(n)).astype(dtype)
    healthy = solve_ols(X, y, config=OLSConfig(rcond=rcond))
    original = linalg.lstsq

    def broken(a, b, *args, lapack_driver=None, **kwargs):
        out = original(a, b, *args, lapack_driver=lapack_driver, **kwargs)
        if lapack_driver in (None, "gelsd"):
            coef, res, rank, s = out
            return coef, res, 4607182418800017408, s[::-1]
        return out

    monkeypatch.setattr(linalg, "lstsq", broken)
    sol = solve_ols(X, y, config=OLSConfig(rcond=rcond))
    monkeypatch.setattr(linalg, "lstsq", original)
    assert sol.method == "gelss"
    assert sol.rank == healthy.rank == 4
    np.testing.assert_allclose(sol.coef, healthy.coef, rtol=1e-4 if dtype == np.float16 else 1e-10)


def test_a_measured_knife_edge_skips_the_tiled_qr(monkeypatch):
    """A measured value at the cutoff: the QR would defer too, so stock decides at once."""
    n, d = 16384, 20  # tall enough for the tiled QR
    rcond = 1e-6
    singular = np.ones(d)
    singular[-2] = rcond * (1 + 1e-13)
    singular[-1] = 1e-9 * rcond
    X, y = _spectrum(n, d, singular, seed=49)
    calls = []
    for name in ("_tall_svd", "_qr_tall", "_stock_lstsq"):
        original = getattr(ols, name)
        monkeypatch.setattr(ols, name, lambda *a, _f=original, _n=name, **k: calls.append(_n) or _f(*a, **k))
    sol = solve_ols(X, y, config=OLSConfig(rcond=rcond, fit_intercept=False))
    assert "_tall_svd" in calls and "_qr_tall" not in calls and calls[-1] == "_stock_lstsq"
    assert sol.method == "lstsq"
    _check(sol, X, y, rcond, fit_intercept=False, coef_rtol=1e-6)
