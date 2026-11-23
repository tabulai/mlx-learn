"""LinearRegression parity against stock scikit-learn.

The load-bearing assertion is the **objective value**: two correct
factorizations of a full-rank design land in the same place, and both
coefficient sets are scored by one float64 function. Rank, singular values and
coefficients are compared too, because they are part of the fitted model:
``rank_`` must equal stock's under the cutoff of the *installed* scikit-learn
(``tol`` on 1.9, ``max(n, d) * eps`` on 1.7 and 1.8), and the coefficients must
be stock's minimum-norm solution, not merely another minimizer.

Tolerances come from ``docs/fp32_policy.md`` §3.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import sklearn
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SkLinearRegression

from mlxlearn import config_context, get_backend_diagnostics, gpu_available
from mlxlearn._common.config import tuning_context
from mlxlearn._solvers.ols import reference_objective
from mlxlearn.exceptions import SparseInputError, UnsupportedParameterError
from mlxlearn.linear_model import LinearRegression

OBJECTIVE_RTOL = 1e-5
COEF_ATOL, COEF_RTOL = 2e-3, 5e-3
SKLEARN_19 = tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 9)


def _data(n_samples=4000, n_features=16, n_targets=1, seed=0, noise=0.1):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(1, n_features // 2),
        n_targets=n_targets,
        noise=noise,
        random_state=seed,
    )
    return X, y


def _design(kind, n, d, seed=0):
    """Designs that exercise every solver path: conditioning, exact rank loss, offsets."""
    rng = np.random.default_rng(seed)
    if kind == "gauss":
        X = rng.standard_normal((n, d))
    elif kind == "offset":
        X = rng.standard_normal((n, d)) + 1e3
    elif kind.startswith("cond"):
        r = min(n, d)
        U, _ = np.linalg.qr(rng.standard_normal((n, r)))
        V, _ = np.linalg.qr(rng.standard_normal((d, r)))
        X = (U * np.logspace(0, -float(kind[4:]), r)) @ V.T * np.sqrt(n)
    elif kind == "rankdef":
        h = max(1, d // 2)
        A = rng.standard_normal((n, h))
        X = np.hstack([A, A @ rng.standard_normal((h, d - h))])
    elif kind == "onehot":
        X = np.zeros((n, d))
        X[np.arange(n), rng.integers(0, d, n)] = 1.0
    elif kind == "mixed":
        h = max(1, d // 2)
        oh = np.zeros((n, d - h))
        oh[np.arange(n), rng.integers(0, d - h, n)] = 1.0
        X = np.hstack([rng.standard_normal((n, h)) * rng.uniform(0.1, 100, h), oh])
    elif kind == "unscaled":
        # The review's example: columns spanning more than six decades.
        X = rng.standard_normal((n, d)) * np.logspace(3, -5, d)
    else:
        raise ValueError(kind)
    y = X @ rng.standard_normal(d) + 0.1 * rng.standard_normal(n)
    return X, y


def _assert_parity(mlx, ref, X, y, *, sample_weight=None):
    j_mlx = reference_objective(mlx.coef_, mlx.intercept_, X, y, sample_weight=sample_weight)
    j_ref = reference_objective(ref.coef_, ref.intercept_, X, y, sample_weight=sample_weight)
    assert j_mlx == pytest.approx(j_ref, rel=OBJECTIVE_RTOL, abs=1e-8), (
        f"objective differs: mlxlearn {j_mlx:.10f} vs scikit-learn {j_ref:.10f}"
    )
    np.testing.assert_allclose(mlx.coef_, ref.coef_, atol=COEF_ATOL, rtol=COEF_RTOL)
    np.testing.assert_allclose(np.asarray(mlx.intercept_), np.asarray(ref.intercept_), atol=COEF_ATOL, rtol=COEF_RTOL)
    np.testing.assert_allclose(mlx.predict(X), ref.predict(X), atol=COEF_ATOL, rtol=COEF_RTOL)


def _assert_same_model(mlx, ref, X, y, *, coef_rtol=1e-7, sample_weight=None):
    """Stock's rank, singular values and minimum-norm coefficients, not just its objective.

    ``coef_rtol`` also bounds the objective: two backward-stable solutions of an
    ill-conditioned problem differ by about ``kappa * u`` in both.
    """
    assert mlx.rank_ == ref.rank_
    s, s_ref = mlx.singular_, ref.singular_
    assert s.shape == s_ref.shape
    smax = float(s_ref[0]) if s_ref.size else 1.0
    np.testing.assert_allclose(s, s_ref, rtol=1e-8, atol=1e-10 * smax)
    scale = max(float(np.max(np.abs(ref.coef_))), 1e-300)
    np.testing.assert_allclose(mlx.coef_, ref.coef_, rtol=coef_rtol, atol=coef_rtol * scale)
    j_mlx = reference_objective(mlx.coef_, mlx.intercept_, X, y, sample_weight=sample_weight)
    j_ref = reference_objective(ref.coef_, ref.intercept_, X, y, sample_weight=sample_weight)
    assert j_mlx <= j_ref * (1 + max(1e-9, coef_rtol)) + 1e-12 * float(np.mean(np.asarray(y) ** 2))


@pytest.fixture
def no_small_tier(monkeypatch):
    """Send small fits through the certified paths instead of the exact small tier."""
    from mlxlearn._solvers import ols

    monkeypatch.setattr(ols, "_SMALL_WORK", 0)


def _last_method():
    """The solver path of the most recent fit, from its ``ols-<method>`` diagnostic."""
    reasons = [e.reason for e in get_backend_diagnostics() if e.operation == "fit" and e.reason.startswith("ols-")]
    return reasons[-1][4:] if reasons else None


# ----------------------------------------------------------------------------- parity


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("n_targets", [1, 3])
def test_parity_across_intercept_and_outputs(fit_intercept, n_targets):
    X, y = _data(n_targets=n_targets)
    mlx = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    ref = SkLinearRegression(fit_intercept=fit_intercept).fit(X, y)
    assert mlx.execution_backend_ == "cpu"
    _assert_parity(mlx, ref, X, y)
    _assert_same_model(mlx, ref, X, y, coef_rtol=1e-10)

    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 40))
    y = X @ rng.normal(size=40) + rng.normal(scale=0.05, size=20)
    mlx = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    ref = SkLinearRegression(fit_intercept=fit_intercept).fit(X, y)
    _assert_parity(mlx, ref, X, y)
    _assert_same_model(mlx, ref, X, y, coef_rtol=1e-9)


@pytest.mark.parametrize(
    "kind", ["gauss", "offset", "cond6", "cond10", "cond14", "rankdef", "onehot", "mixed", "unscaled"]
)
@pytest.mark.parametrize("shape", [(3000, 24), (400, 60), (5000, 300)])
def test_rank_singular_values_and_coefficients_match_stock(kind, shape):
    X, y = _design(kind, *shape)
    mlx = LinearRegression().fit(X, y)
    ref = SkLinearRegression().fit(X, y)
    # Ill-conditioned full-rank problems carry kappa * u forward error in any
    # backward-stable solver, so their coefficients agree only that far.
    s = ref.singular_
    kept = s[s > (ref.tol if SKLEARN_19 else max(X.shape) * np.finfo(float).eps) * s[0]]
    kappa = float(kept[0] / kept[-1])
    _assert_same_model(mlx, ref, X, y, coef_rtol=max(1e-9, 64 * kappa * np.finfo(float).eps))


@pytest.mark.parametrize("kind", ["onehot", "mixed", "cond10"])
def test_weighted_multi_target_hard_designs(kind):
    X, y = _design(kind, 2500, 40, seed=3)
    rng = np.random.default_rng(4)
    Y = np.column_stack([y, rng.standard_normal(len(y)), y - X[:, 0]])
    w = rng.uniform(0.0, 2.0, len(y))
    w[rng.random(len(y)) < 0.05] = 0.0
    mlx = LinearRegression().fit(X, Y, sample_weight=w)
    ref = SkLinearRegression().fit(X, Y, sample_weight=w)
    s = ref.singular_
    kept = s[s > (ref.tol if SKLEARN_19 else max(X.shape) * np.finfo(float).eps) * s[0]]
    _assert_same_model(mlx, ref, X, Y, sample_weight=w,
                       coef_rtol=max(1e-9, 64 * float(kept[0] / kept[-1]) * np.finfo(float).eps))


@pytest.mark.usefixtures("no_small_tier")
def test_the_cutoff_follows_the_installed_scikit_learn():
    """1.9 truncates at tol; 1.7 and 1.8 keep every direction above max(n,d)*eps."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5000, 4)) * np.array([1e3, 1.0, 1e-4, 3e-5])
    y = X @ np.array([1.0, 2.0, 3e3, -4e3]) + 0.01 * rng.standard_normal(5000)
    mlx = LinearRegression().fit(X, y)
    assert _last_method() != "lstsq"
    ref = SkLinearRegression().fit(X, y)
    assert mlx.rank_ == ref.rank_ == (2 if SKLEARN_19 else 4)
    _assert_same_model(mlx, ref, X, y, coef_rtol=1e-6)


@pytest.mark.usefixtures("no_small_tier")
@pytest.mark.skipif(not SKLEARN_19, reason="tol is the dense cutoff from scikit-learn 1.9")
@pytest.mark.parametrize("kind", ["cond6", "cond10"])
@pytest.mark.parametrize("tol", [0.0, 1e-12, 1e-3, 0.3, 1.0, 5.0])
def test_tol_is_the_dense_cutoff(tol, kind):
    X, y = _design(kind, 3000, 30, seed=5)
    mlx = LinearRegression(tol=tol).fit(X, y)
    ref = SkLinearRegression(tol=tol).fit(X, y)
    assert mlx.rank_ == ref.rank_
    if kind == "cond6" and tol in (1e-12, 1e-3):
        assert _last_method() == "ritz"  # half the directions measured on a long design
    if not 0 < tol < 1:
        # gelsd replaces such cutoffs with machine epsilon; stock is reproduced exactly.
        np.testing.assert_array_equal(mlx.coef_, ref.coef_)
        return
    s = ref.singular_
    kept = s[s > tol * s[0]]
    _assert_same_model(mlx, ref, X, y, coef_rtol=max(1e-9, 64 * float(kept[0] / kept[-1]) * 2.2e-16))


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize("weighted", [False, True])
def test_centered_wide_fit_resolves_the_nonconstant_row_space(backend, weighted, monkeypatch):
    from mlxlearn._solvers import ols

    rng = np.random.default_rng(24)
    X = 100 + rng.normal(size=(64, 300))  # large enough for the dual Gram path
    y = rng.normal(size=(64, 2))
    weights = rng.uniform(0.8, 1.2, size=len(X)) if weighted else None
    ref = SkLinearRegression().fit(X, y, sample_weight=weights)

    def unexpected(*args, **kwargs):
        pytest.fail("the centred design has only the known constant null direction")

    if SKLEARN_19:
        # Under 1.7/1.8 the cutoff (max(n, d) * eps) sits below the product
        # rounding bound, so gelsd decides; under 1.9 the Gram path must.
        monkeypatch.setattr(ols, "_stock_lstsq", unexpected)
    with tuning_context(force_backend=backend):
        model = LinearRegression().fit(X, y, sample_weight=weights)
    assert model.execution_backend_ == "cpu"
    assert model.rank_ == ref.rank_
    if SKLEARN_19:
        assert model.rank_ == len(X) - 1
    # Under 1.7/1.8 stock may keep the rounding-level null direction (its rank is
    # then len(X)); the exact fallback reproduces that bit for bit.
    np.testing.assert_allclose(model.singular_, ref.singular_, atol=1e-10, rtol=1e-5)
    np.testing.assert_allclose(model.coef_, ref.coef_, atol=1e-8)
    np.testing.assert_allclose(model.predict(X), ref.predict(X), atol=1e-8)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("shape", [(64, 300), (200, 1500)])
def test_wide_designs_match_stock(fit_intercept, shape):
    X, y = _design("gauss", *shape, seed=7)
    mlx = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    ref = SkLinearRegression(fit_intercept=fit_intercept).fit(X, y)
    _assert_same_model(mlx, ref, X, y, coef_rtol=1e-9)


@pytest.mark.usefixtures("no_small_tier")
def test_rank_deficient_objective_and_min_norm_coefficients():
    """Duplicate columns: stock returns the minimum-norm solution, and so must mlxlearn."""
    X, y = _data(n_features=8, seed=4)
    X = np.hstack([X, X[:, :1]])
    mlx = LinearRegression().fit(X, y)
    assert _last_method() == "ritz"
    ref = SkLinearRegression().fit(X, y)
    _assert_same_model(mlx, ref, X, y, coef_rtol=1e-9)


@pytest.mark.usefixtures("no_small_tier")
@pytest.mark.parametrize("offset", [0.0, 1e2, 1e4])
def test_centering_absorbs_a_constant_offset(offset):
    X, y = _data(n_samples=2000, n_features=8, seed=0)
    X = X + offset
    mlx = LinearRegression().fit(X, y)
    ref = SkLinearRegression().fit(X, y)
    _assert_parity(mlx, ref, X, y)
    np.testing.assert_allclose(mlx.coef_, ref.coef_, rtol=1e-9)


@pytest.mark.usefixtures("no_small_tier")
@pytest.mark.parametrize("n_samples, n_features", [(2000, 12), (40, 120)])
def test_refinement_survives_an_ill_conditioned_gram(n_samples, n_features):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n_samples, n_features))
    X[:, 1] = X[:, 0] + 1e-4 * X[:, 1]
    y = X @ rng.normal(size=n_features) + rng.normal(scale=0.1, size=n_samples)
    mlx = LinearRegression().fit(X, y)
    if SKLEARN_19 or n_samples > n_features:
        # 1.7/1.8 cut a 40 x 120 design at 120 eps, inside the rounding of the
        # bound on its constant direction: there gelsd decides.
        assert _last_method() in ("ritz", "dual")
    ref = SkLinearRegression().fit(X, y)
    _assert_same_model(mlx, ref, X, y, coef_rtol=1e-7)


@pytest.mark.usefixtures("no_small_tier")
@pytest.mark.parametrize("repeat", [1, 16])
@pytest.mark.parametrize("transpose", [False, True], ids=["tall", "wide"])
def test_rank_survives_a_singular_direction_lost_in_the_gram(transpose, repeat):
    """sigma_min^2 = 2^-32 sigma_max^2 vanishes in the Gram; it must be measured (or left to gelsd)."""
    u = np.array([1.0, 1.0, -1.0, -1.0])
    v = np.array([1.0, -1.0, 1.0, -1.0])
    X = np.tile(np.column_stack((u + 2**-16 * v, u - 2**-16 * v)), (repeat, 1))
    if transpose:
        X = X.T
    y = X[:, 0]
    model = LinearRegression(fit_intercept=False).fit(X, y)
    if repeat > 1:
        assert _last_method() == "ritz"
    ref = SkLinearRegression(fit_intercept=False).fit(X, y)
    assert model.rank_ == ref.rank_ == 2
    np.testing.assert_allclose(model.singular_, ref.singular_, rtol=1e-8)
    np.testing.assert_allclose(model.coef_, ref.coef_, atol=1e-10)


@pytest.mark.usefixtures("no_small_tier")
def test_no_centering_without_an_intercept():
    X, y = _data(n_samples=2000, n_features=8)
    X = X + 5.0
    mlx = LinearRegression(fit_intercept=False).fit(X, y)
    ref = SkLinearRegression(fit_intercept=False).fit(X, y)
    _assert_parity(mlx, ref, X, y)
    assert mlx.intercept_ == 0.0


# ------------------------------------------------------------------------- precision


def test_small_float32_problems_reproduce_stock_bit_for_bit():
    X, y = _data(n_samples=500, n_features=6)
    X32, y32 = X.astype(np.float32), y.astype(np.float32)
    mlx = LinearRegression().fit(X32, y32)
    ref = SkLinearRegression().fit(X32, y32)
    assert mlx.coef_.dtype == np.float64
    np.testing.assert_array_equal(mlx.coef_, ref.coef_.astype(np.float64))
    np.testing.assert_array_equal(mlx.singular_, ref.singular_.astype(np.float64))
    assert mlx.rank_ == ref.rank_


def test_float32_input_is_solved_in_float64():
    X, y = _data(n_samples=20000, n_features=12)
    X32, y32 = X.astype(np.float32), y.astype(np.float32)
    mlx = LinearRegression().fit(X32, y32)
    ref = SkLinearRegression().fit(X32, y32)
    assert mlx.coef_.dtype == np.float64
    assert mlx.singular_.dtype == np.float64
    assert mlx.rank_ == ref.rank_
    # The exact float64 solution of stock's own float32-centred design (stock
    # solves that same matrix in float32).
    Xc = (X32 - np.average(X32, axis=0).astype(np.float32)).astype(np.float64)
    yc = (y32 - np.average(y32)).astype(np.float64)
    ref64 = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    np.testing.assert_allclose(mlx.coef_, ref64, rtol=1e-10, atol=1e-12 * np.abs(ref64).max())
    j_mlx = reference_objective(mlx.coef_, mlx.intercept_, X32, y32)
    j_ref = reference_objective(ref.coef_, ref.intercept_, X32, y32)
    assert j_mlx <= j_ref * (1 + 1e-7)


@pytest.mark.parametrize("n_targets", [1, 3])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_float16_input_reproduces_stock(weighted, fit_intercept, n_targets):
    """Stock keeps float16 (and, on 1.7/1.8, float16's eps in the cutoff).

    Without an intercept stock's coef_ stays scipy's float32 solution; with one,
    _set_intercept rounds it to float16 first.
    """
    rng = np.random.default_rng(15)
    X = (rng.normal(size=(300, 5)) * np.array([1, 1, 1, 0.05, 0.02])).astype(np.float16)
    Y = X.astype(float) @ rng.normal(size=(5, n_targets)) + 0.01 * rng.normal(size=(300, n_targets))
    y = (Y[:, 0] if n_targets == 1 else Y).astype(np.float16)
    w = rng.uniform(0.5, 1.5, 300).astype(np.float16) if weighted else None
    mlx = LinearRegression(fit_intercept=fit_intercept).fit(X, y, sample_weight=w)
    assert _last_method() == "lstsq"
    ref = SkLinearRegression(fit_intercept=fit_intercept).fit(X, y, sample_weight=w)
    assert mlx.rank_ == ref.rank_
    np.testing.assert_array_equal(mlx.coef_, np.asarray(ref.coef_, dtype=np.float64))
    np.testing.assert_array_equal(mlx.singular_, np.asarray(ref.singular_, dtype=np.float64))
    np.testing.assert_array_equal(np.asarray(mlx.intercept_), np.asarray(ref.intercept_, dtype=np.float64))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_offset_data_in_narrow_dtypes_stays_on_gelsd(dtype):
    """The gelsd plausibility check must not misread a narrow-dtype product as garbage."""
    rng = np.random.default_rng(3)
    X = (2000 + rng.normal(size=(5000, 8))).astype(dtype)
    beta = rng.normal(size=8)
    y = ((X.astype(np.float64) - 2000) @ beta + 1e-3 * rng.normal(size=5000)).astype(dtype)
    for seed in range(4):
        Xs = X[np.random.default_rng(seed).permutation(5000)]
        mlx = LinearRegression().fit(Xs, y)
        assert _last_method() in ("lstsq", "gram", "ritz", "qr")
        ref = SkLinearRegression().fit(Xs, y)
        if _last_method() == "lstsq":
            np.testing.assert_array_equal(mlx.singular_, np.asarray(ref.singular_, dtype=np.float64))


@pytest.mark.usefixtures("no_small_tier")
@pytest.mark.parametrize("offset", [1e12, 1e15])
def test_count_data_with_a_huge_target_offset(offset):
    rng = np.random.default_rng(16)
    X = rng.poisson(3.0, size=(20000, 12)).astype(np.float64)
    y = offset + X @ rng.normal(size=12) + rng.normal(size=20000)
    mlx = LinearRegression().fit(X, y)
    ref = SkLinearRegression().fit(X, y)
    assert mlx.rank_ == ref.rank_
    np.testing.assert_allclose(mlx.coef_, ref.coef_, rtol=1e-6, atol=1e-6)
    assert abs(mlx.intercept_ - ref.intercept_) <= 1e-9 * offset


@pytest.mark.usefixtures("no_small_tier")
@pytest.mark.parametrize("scale", [1e-160, 1e-100, 1e100, 1e150])
def test_extreme_magnitudes_match_stock_without_warnings(scale):
    X, y = _design("gauss", 2000, 10, seed=8)
    Xs, ys = X * scale, y * scale
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mlx = LinearRegression().fit(Xs, ys)
    ref = SkLinearRegression().fit(Xs, ys)
    assert mlx.rank_ == ref.rank_
    np.testing.assert_allclose(mlx.coef_, ref.coef_, rtol=1e-9)
    np.testing.assert_allclose(mlx.singular_, ref.singular_, rtol=1e-8)


@pytest.mark.usefixtures("no_small_tier")
@pytest.mark.parametrize("scale", [1e-200, 1e200])
@pytest.mark.parametrize("kind", ["gauss", "onehot"])
def test_extreme_targets_on_an_ordinary_design(scale, kind):
    X, y = _design(kind, 3000, 12, seed=9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mlx = LinearRegression().fit(X, y * scale)
    ref = SkLinearRegression().fit(X, y * scale)
    assert mlx.rank_ == ref.rank_
    np.testing.assert_allclose(mlx.coef_, ref.coef_, rtol=1e-9, atol=1e-12 * np.abs(ref.coef_).max())


@pytest.mark.parametrize("n_samples", [1500, 40000])  # the second runs threaded
def test_fits_raise_no_runtime_warnings_on_hard_designs(n_samples):
    for kind in ("cond14", "rankdef", "onehot", "mixed"):
        X, y = _design(kind, n_samples, 30)
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            LinearRegression().fit(X, y)
            LinearRegression().fit(X[:20], y[:20])


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_float32_cutoff_follows_the_installed_scikit_learn(fit_intercept):
    """1.7/1.8 cut float32 fits at max(n, d) * eps(float32), about 2e-3 here."""
    rng = np.random.default_rng(31)
    X = (rng.standard_normal((20000, 6)) * np.array([1.0, 1.0, 1.0, 1e-2, 3e-4, 1e-4])).astype(np.float32)
    y = (X.astype(np.float64) @ np.array([1.0, -2.0, 0.5, 30.0, 1e3, -2e3]) + 0.01 * rng.standard_normal(20000))
    y = y.astype(np.float32)
    mlx = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    assert _last_method() != "lstsq"
    ref = SkLinearRegression(fit_intercept=fit_intercept).fit(X, y)
    assert mlx.rank_ == ref.rank_ == (6 if SKLEARN_19 else 4)
    np.testing.assert_allclose(mlx.singular_, ref.singular_, rtol=1e-4)
    j_mlx = reference_objective(mlx.coef_, mlx.intercept_, X, y)
    j_ref = reference_objective(ref.coef_, ref.intercept_, X, y)
    assert j_mlx <= j_ref * (1 + 1e-5)


def test_timestamps_with_a_huge_offset():
    """Epoch seconds as features: offsets of 1.7e9 over ranges of days, n = 200k."""
    rng = np.random.default_rng(32)
    n = 200_000
    start = 1.7e9 + np.sort(rng.uniform(0, 90 * 86400, n))
    duration = rng.exponential(3600.0, n)
    end = start + duration
    X = np.column_stack([start, end, duration, rng.normal(size=n)])
    y = 1e-6 * (start - 1.7e9) + 0.01 * duration + rng.normal(size=n)
    mlx = LinearRegression().fit(X, y)
    ref = SkLinearRegression().fit(X, y)
    # end = start + duration exactly up to rounding: a null direction of the
    # centred design, which every version's cutoff drops.
    assert mlx.rank_ == ref.rank_
    np.testing.assert_allclose(mlx.singular_, ref.singular_, rtol=1e-8, atol=1e-10 * ref.singular_[0])
    np.testing.assert_allclose(mlx.predict(X[:1000]), ref.predict(X[:1000]), rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("shape", [(3000, 20), (40000, 300)])
def test_a_residual_that_dwarfs_the_signal_keeps_the_coefficients(shape):
    """Review C01: the old certificate accepted coef_ = 0 when ||r|| >> ||X b||."""
    rng = np.random.default_rng(12)
    X = rng.standard_normal(shape)
    b = rng.standard_normal(shape[1]) * 1e-3
    y = X @ b + 1e6 * rng.standard_normal(shape[0])
    for multi in (False, True):
        Y = np.column_stack([y, X @ b]) if multi else y
        mlx = LinearRegression().fit(X, Y)
        ref = SkLinearRegression().fit(X, Y)
        np.testing.assert_allclose(mlx.coef_, ref.coef_, rtol=1e-9, atol=1e-12 * np.abs(ref.coef_).max())


# --------------------------------------------------------------------------- solver


def test_exact_path_decides_an_ambiguous_rank(monkeypatch):
    """A singular value at the cutoff: the tiled QR sees it inside stock's band and
    defers, and stock's own arithmetic decides."""
    from mlxlearn._solvers import ols

    calls = []
    for name in ("_qr_tall", "_stock_lstsq"):
        original = getattr(ols, name)
        monkeypatch.setattr(ols, name, lambda *a, _f=original, _n=name, **k: calls.append(_n) or _f(*a, **k))
    rng = np.random.default_rng(9)
    n, d = 16384, 20  # tall enough for the tiled QR
    U, _ = np.linalg.qr(rng.standard_normal((n, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.logspace(0, -9, d)
    if SKLEARN_19:
        s[-3] = 1e-6 * (1 + 1e-9)  # tol
    else:
        s[-1] = n * np.finfo(np.float64).eps * (1 + 1e-9)  # max(n, d) * eps
    X = (U * s) @ V.T
    y = X @ rng.standard_normal(d)
    mlx = LinearRegression(fit_intercept=False).fit(X, y)
    ref = SkLinearRegression(fit_intercept=False).fit(X, y)
    assert calls == ["_qr_tall", "_stock_lstsq"], calls
    assert mlx.rank_ == ref.rank_
    np.testing.assert_allclose(mlx.singular_, ref.singular_, rtol=1e-6, atol=1e-13)


def test_blocked_householder_qr_only_runs_on_tall_leaves(monkeypatch):
    """Accelerate's blocked dgeqrf overruns the heap for some squarish shapes.

    numpy.linalg.qr and scipy.linalg.qr (blocked, any shape) must never run;
    dgeqrf with a real workspace only on leaves 2048+ rows taller than wide.
    """
    import numpy.linalg
    import scipy.linalg
    from scipy.linalg import lapack


    designs = [_design(kind, n, d) for kind, n, d in
               (("cond10", 20000, 30), ("cond14", 9000, 301), ("mixed", 5000, 300), ("cond6", 3000, 60),
                ("onehot", 1500, 40))]  # the last measures W = A V0 on a short leaf

    def forbidden(*args, **kwargs):
        raise AssertionError("numpy/scipy QR must not be called")

    calls = []
    original = lapack.dgeqrf

    def checked(a, lwork=None, overwrite_a=0):
        m, c = a.shape
        calls.append((m, c, lwork))
        if lwork is not None:
            # The guard-malloc sweep cleared 4-1280 columns at 2048-8192 extra
            # rows; these literals pin that range against edits to the constants.
            assert m >= c + 2048 and c <= 1280, f"blocked dgeqrf on a {m}x{c} block"
            return original(a, lwork=lwork, overwrite_a=overwrite_a)
        return original(a, overwrite_a=overwrite_a)

    monkeypatch.setattr(numpy.linalg, "qr", forbidden)
    monkeypatch.setattr(scipy.linalg, "qr", forbidden)
    monkeypatch.setattr(lapack, "dgeqrf", checked)
    for X, y in designs:
        LinearRegression().fit(X, y)
    assert any(lwork is not None for _, _, lwork in calls), "no blocked leaf ran"
    assert any(lwork is None for _, _, lwork in calls), "no short leaf ran"


def test_threads_do_not_change_the_model():
    X, y = _design("mixed", 40000, 33, seed=11)
    with tuning_context(linreg_threads=1):
        one = LinearRegression().fit(X, y)
    with tuning_context(linreg_threads=8):
        many = LinearRegression().fit(X, y)
    # The chunk grouping depends on the machine, not on the budget, and partials
    # are summed in a fixed order: the model is bit-identical.
    assert one.rank_ == many.rank_
    np.testing.assert_array_equal(one.coef_, many.coef_)
    np.testing.assert_array_equal(one.singular_, many.singular_)


# ------------------------------------------------------------------- sample weights


def _outcome(estimator, X, y, sample_weight=None, **params):
    """What a fit returns or raises, comparable across implementations."""
    try:
        model = estimator(**params).fit(X, y, sample_weight=sample_weight)
    except Exception as exc:  # noqa: BLE001 - compared by the caller
        return type(exc), str(exc)
    return "ok", (int(model.rank_), np.asarray(model.coef_, dtype=np.float64).round(8).tolist())


_WEIGHTS = {
    "negative": lambda n: np.r_[-1.0, np.ones(n - 1)],
    "zero": lambda n: np.zeros(n),
    "two-dimensional": lambda n: np.ones((n, 1)),
    "short": lambda n: np.ones(n - 1),
    "nan": lambda n: np.r_[np.nan, np.ones(n - 1)],
    "scalar": lambda n: 2.0,
    "integer": lambda n: np.arange(1, n + 1),
    "float32": lambda n: np.ones(n, dtype=np.float32),
    "list": lambda n: [1.0] * n,
    "nan scalar": lambda n: float("nan"),
    "inf scalar": lambda n: float("inf"),
}


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("weights", sorted(_WEIGHTS))
def test_sample_weight_validation_matches_stock(fit_intercept, weights):
    X, y = _data(n_samples=200, n_features=4)
    w = _WEIGHTS[weights](len(y))
    ours = _outcome(LinearRegression, X, y, w, fit_intercept=fit_intercept)
    assert ours == _outcome(SkLinearRegression, X, y, w, fit_intercept=fit_intercept)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int64])
def test_sample_weight_dtype_follows_stock(dtype):
    """float64 weights unless X is float32/float64, as in _check_sample_weight."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4)).astype(dtype) if dtype != np.int64 else rng.integers(-5, 5, (300, 4))
    y = X.astype(np.float64) @ np.arange(1.0, 5.0) + rng.normal(size=300)
    w = rng.uniform(0.5, 1.5, 300)
    ours = _outcome(LinearRegression, X, y, w)
    assert ours == _outcome(SkLinearRegression, X, y, w)


_BAD_INPUTS = {
    "strings": lambda X, y: (X.astype(str), y),
    "complex": lambda X, y: (X.astype(complex), y),
    "nan in X": lambda X, y: (np.where(np.arange(X.size).reshape(X.shape) == 3, np.nan, X), y),
    "inf in y": lambda X, y: (X, np.r_[np.inf, y[1:]]),
    "y beyond float32": lambda X, y: (X.astype(np.float32), y * 1e300),
    "y rows": lambda X, y: (X, y[:-1]),
    "3-d X": lambda X, y: (X[:, :, None], y),
    "one sample": lambda X, y: (X[:1], y[:1]),
    "no features": lambda X, y: (X[:, :0], y),
    "boolean X": lambda X, y: (X > 0, y),
    "integer X": lambda X, y: (np.round(X * 10).astype(np.int32), y),
    "string y": lambda X, y: (X, y.astype(str)),
    "object y with None": lambda X, y: (X, np.array([None] + list(y[1:]), dtype=object)),
    "string y with nan": lambda X, y: (X, np.array(["nan"] + [str(v) for v in y[1:]])),
    "string y with inf": lambda X, y: (X, np.array(["inf"] + [str(v) for v in y[1:]])),
    "list X with None": lambda X, y: ([[None] + list(X[0, 1:])] + X[1:].tolist(), y),
    "list X of strings": lambda X, y: ([[str(v) for v in row] for row in X.tolist()], y),
    "sparse y": lambda X, y: (X, __import__("scipy.sparse").sparse.csr_matrix(y[:, None])),
    "sparse 2-target y": lambda X, y: (X, __import__("scipy.sparse").sparse.csr_matrix(np.c_[y, y])),
}


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("problem", ["sparse y", "y beyond float32", "list X with None, short weights"])
def test_zero_weights_raise_where_stock_raises(problem, dtype):
    """Stock checks X and y before its weighted mean meets all-zero weights (1.7/1.8)."""
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _data(n_samples=200, n_features=4)
    X, y = np.asarray(X, dtype=dtype), np.asarray(y)
    w = np.zeros(len(y))
    if problem == "sparse y":
        y = sparse.csr_matrix(y[:, None])
    elif problem == "y beyond float32":
        y = y * 1e300
    else:
        X = [[None] + list(X[0, 1:])] + X[1:].tolist()
        w = np.ones(len(y) + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _outcome(LinearRegression, X, y, w) == _outcome(SkLinearRegression, X, y, w)


def test_nan_weights_under_assume_finite_raise_where_stock_raises():
    """A NaN weight is not a zero weight: the wide path must not drop the row."""
    from sklearn import config_context as sk_config_context

    rng = np.random.default_rng(33)
    X, y = rng.normal(size=(300, 2000)), rng.normal(size=300)
    w = np.ones(300)
    w[3] = np.nan
    with sk_config_context(assume_finite=True):
        for fit_intercept in (False, True):
            ours = _outcome(LinearRegression, X, y, w, fit_intercept=fit_intercept)
            assert ours == _outcome(SkLinearRegression, X, y, w, fit_intercept=fit_intercept)


@pytest.mark.parametrize("case", sorted(_BAD_INPUTS))
def test_input_validation_matches_stock(case):
    X, y = _data(n_samples=200, n_features=4)
    X, y = _BAD_INPUTS[case](np.asarray(X), np.asarray(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # stock's own overflow warnings
        assert _outcome(LinearRegression, X, y) == _outcome(SkLinearRegression, X, y)
        if not isinstance(X, list):
            return
        # Patched mode, both below the crossover (stock) and above it.
        from mlxlearn.patching._estimators import LinearRegression as Patched

        assert _outcome(Patched, X, y) == _outcome(SkLinearRegression, X, y)


@pytest.mark.usefixtures("no_small_tier")
def test_sample_weight_zeros_drop_rows():
    X, y = _data(n_samples=500, n_features=6, seed=5)
    w = np.ones(len(y))
    w[::3] = 0.0
    weighted = LinearRegression().fit(X, y, sample_weight=w)
    plain = LinearRegression().fit(X[w > 0], y[w > 0])
    np.testing.assert_allclose(weighted.coef_, plain.coef_, rtol=1e-10, atol=1e-12)


@pytest.mark.usefixtures("no_small_tier")
def test_integer_sample_weights_repeat_rows():
    X, y = _data(n_samples=300, n_features=5, seed=6)
    w = np.random.default_rng(1).integers(1, 4, len(y))
    weighted = LinearRegression().fit(X, y, sample_weight=w)
    repeated = LinearRegression().fit(np.repeat(X, w, axis=0), np.repeat(y, w))
    np.testing.assert_allclose(weighted.coef_, repeated.coef_, rtol=1e-9, atol=1e-11)


# ----------------------------------------------------------------- estimator contract


def test_fitted_attribute_shapes_and_dtypes():
    X, y = _data(n_targets=1)
    model = LinearRegression().fit(X, y)
    assert model.coef_.shape == (X.shape[1],)
    assert model.coef_.dtype == np.float64
    assert np.ndim(model.intercept_) == 0
    assert model.singular_.dtype == np.float64
    assert model.singular_.shape == (min(X.shape),)
    assert model.rank_ == X.shape[1]

    X2, y2 = _data(n_targets=3, seed=1)
    multi = LinearRegression().fit(X2, y2)
    assert multi.coef_.shape == (3, X2.shape[1])
    assert multi.intercept_.shape == (3,)
    column = LinearRegression().fit(X2, y2[:, :1])
    assert column.coef_.shape == (1, X2.shape[1])
    assert column.intercept_.shape == (1,)


def test_inference_reuses_the_fitted_backend():
    X, y = _data()
    model = LinearRegression().fit(X, y)
    fitted = model.execution_backend_
    model.predict(X[:8])
    inference = [e for e in get_backend_diagnostics() if e.operation == "predict"]
    assert inference
    assert all(e.backend == fitted for e in inference)
    assert all(e.reason == "sticky" for e in inference)


def test_small_problems_use_the_host_path(small_classification_data):
    X, y = small_classification_data
    model = LinearRegression().fit(X, y.astype(float))
    assert model.execution_backend_ == "cpu"


@pytest.mark.parametrize("request_gpu", ["force_backend", "device"])
def test_a_gpu_request_is_solved_on_the_host(request_gpu, small_classification_data):
    X, y = small_classification_data
    if request_gpu == "force_backend":
        with tuning_context(force_backend="mlx"):
            model = LinearRegression().fit(X, y.astype(float))
    else:
        if not gpu_available():
            pytest.skip("no Metal device")
        with config_context(device="gpu"):
            model = LinearRegression().fit(X, y.astype(float))
    assert model.execution_backend_ == "cpu"
    events = [e for e in get_backend_diagnostics() if e.operation == "fit"]
    assert any(e.reason == "host-solve" for e in events)
    _assert_same_model(model, SkLinearRegression().fit(X, y), X, y)


@pytest.mark.parametrize("kind, method", [("gauss", "ols-gram"), ("onehot", "ols-ritz")])
def test_the_solver_path_is_recorded(kind, method):
    X, y = _design(kind, 20000, 24)
    LinearRegression().fit(X, y)
    reasons = [e.reason for e in get_backend_diagnostics() if e.operation == "fit"]
    assert method in reasons


def test_ragged_lists_raise_what_stock_raises():
    X = [[1.0, 2.0], [3.0]]
    y = [1.0, 2.0]
    with pytest.raises(ValueError) as ours:
        LinearRegression().fit(X, y)
    with pytest.raises(ValueError) as stock:
        SkLinearRegression().fit(X, y)
    assert type(ours.value) is type(stock.value)


def test_ragged_targets_do_not_preempt_design_validation():
    from mlxlearn.patching._estimators import LinearRegression as Patched

    X = np.empty((0, 2))
    y = [[1.0], [2.0, 3.0]]
    with pytest.raises(ValueError) as stock:
        SkLinearRegression().fit(X, y)
    for estimator in (LinearRegression, Patched):
        with pytest.raises(ValueError) as ours:
            estimator().fit(X, y)
        assert str(ours.value) == str(stock.value)


def test_positive_raises():
    X, y = _data(n_samples=200, n_features=4)
    with pytest.raises(UnsupportedParameterError) as excinfo:
        LinearRegression(positive=True).fit(X, y)
    assert excinfo.value.parameter == "positive"


def test_sparse_input_raises():
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _data(n_samples=200, n_features=4)
    with pytest.raises(SparseInputError) as excinfo:
        LinearRegression().fit(sparse.csr_matrix(X), y)
    assert "sparse" in str(excinfo.value).lower()


@pytest.mark.usefixtures("no_small_tier")
@pytest.mark.parametrize("columns", ["one", "all"])
def test_values_beyond_float32_range_are_fitted_in_float64(columns):
    X, y = _data(n_samples=200, n_features=4)
    X = np.array(X)
    if columns == "one":
        X[:, 0] *= 1e100  # a 1e100 condition number: gelsd decides
    else:
        X *= 1e100
    y = y * 1e100
    mlx = LinearRegression().fit(X, y)
    if columns == "all":
        assert _last_method() == "gram"
    ref = SkLinearRegression().fit(X, y)
    assert mlx.execution_backend_ == "cpu"
    _assert_same_model(mlx, ref, X, y, coef_rtol=1e-9)


@pytest.mark.parametrize("n_samples", [500, 20000])
def test_mlx_arrays_are_fitted_like_numpy_float32(n_samples):
    """Stock validates MLX float32 arrays to float32 NumPy; so does mlxlearn."""
    mx = pytest.importorskip("mlx.core")
    X, y = _data(n_samples=n_samples, n_features=6)
    X32, y32 = X.astype(np.float32), y.astype(np.float32)
    mlx = LinearRegression().fit(mx.array(X32), mx.array(y32))
    same = LinearRegression().fit(X32, y32)
    np.testing.assert_array_equal(mlx.coef_, same.coef_)
    np.testing.assert_array_equal(mlx.singular_, same.singular_)
    assert mlx.intercept_ == same.intercept_
    ref = SkLinearRegression().fit(mx.array(X32), mx.array(y32))
    assert mlx.rank_ == ref.rank_
    if n_samples == 500:  # the small tier runs stock's arithmetic
        np.testing.assert_array_equal(mlx.coef_, ref.coef_.astype(np.float64))


@pytest.mark.usefixtures("no_small_tier")
def test_copy_x_false_leaves_the_input_untouched():
    X, y = _data(n_samples=1000, n_features=6)
    X = np.array(X) + 3.0
    before = X.copy()
    LinearRegression(copy_X=False).fit(X, y)
    np.testing.assert_array_equal(X, before)


def test_fit_is_a_pure_function_of_its_arguments():
    X_a, y_a = _data(seed=1)
    X_b, y_b = _data(seed=2)
    sequential = LinearRegression().fit(X_a, y_a).fit(X_b, y_b)
    fresh = LinearRegression().fit(X_b, y_b)
    np.testing.assert_array_equal(sequential.coef_, fresh.coef_)


def test_mutating_the_input_after_fit_does_not_change_predictions():
    X, y = _data()
    X = np.array(X)
    query = X[:50].copy()
    model = LinearRegression().fit(X, y)
    before = model.predict(query)
    X[:] = 0.0
    np.testing.assert_array_equal(before, model.predict(query))


def test_list_input_is_validated_once_and_routed_by_shape():
    X, y = _data(n_samples=50, n_features=3)
    from mlxlearn.patching._estimators import LinearRegression as Patched

    as_list = Patched().fit(X.tolist(), y.tolist())
    as_array = Patched().fit(X, y)
    assert as_list.execution_backend_ == as_array.execution_backend_
    np.testing.assert_allclose(as_list.coef_, as_array.coef_, rtol=1e-12)


# ---------------------------------------------------------------------- patched layer


def test_patched_positive_falls_back():
    import sklearn.linear_model

    import mlxlearn

    X, y = _data(n_samples=200, n_features=4)
    mlxlearn.patch_sklearn()
    model = sklearn.linear_model.LinearRegression(positive=True).fit(X, y)
    assert model.execution_backend_ == "sklearn"
    events = [e for e in mlxlearn.get_backend_diagnostics() if e.fallback]
    assert any("positive" in e.reason for e in events)


@pytest.mark.parametrize(
    "shape, backend",
    [
        ((200, 4), "sklearn"),
        ((4096, 256), "cpu"),
        ((8000, 64), "cpu"),
        ((20000, 64), "cpu"),
        ((20000, 2000), "sklearn"),
        ((512, 2048), "cpu"),
        ((512, 1024), "sklearn"),
    ],
)
def test_patched_crossover_routes_by_shape(shape, backend):
    from mlxlearn.patching._estimators import LinearRegression as Patched

    X, y = _design("gauss", *shape)
    assert Patched().fit(X, y).execution_backend_ == backend


def test_patched_wide_float32_preserves_stock_rank():
    from mlxlearn.patching._estimators import LinearRegression as Patched

    X, y = _design("gauss", 256, 1024)
    X, y = X.astype(np.float32), y.astype(np.float32)
    weights = np.linspace(0.5, 1.5, len(X), dtype=np.float32)
    weights[::20] = 0
    ref = SkLinearRegression().fit(X, y, sample_weight=weights)
    patched = Patched().fit(X, y, sample_weight=weights)
    assert patched.execution_backend_ == "sklearn"
    assert patched.rank_ == ref.rank_
    np.testing.assert_array_equal(patched.coef_, ref.coef_)


@pytest.mark.parametrize("n_targets, backend", [(16, "cpu"), (64, "cpu"), (65, "sklearn")])
def test_patched_crossover_leaves_more_targets_than_features_to_stock(n_targets, backend):
    from mlxlearn.patching._estimators import LinearRegression as Patched

    rng = np.random.default_rng(48)
    X = rng.standard_normal((20000, 64))
    Y = X @ rng.standard_normal((64, n_targets)) + rng.standard_normal((20000, n_targets))
    assert Patched().fit(X, Y).execution_backend_ == backend
    assert Patched().fit(X, Y.tolist() if n_targets == 65 else Y).execution_backend_ == backend


def test_patched_predict_accepts_hand_set_coefficients():
    import sklearn.linear_model

    import mlxlearn

    X, y = _data(n_samples=100, n_features=3)
    stock = SkLinearRegression().fit(X, y)
    mlxlearn.patch_sklearn()
    model = sklearn.linear_model.LinearRegression()
    model.coef_, model.intercept_ = stock.coef_, stock.intercept_
    model.n_features_in_ = X.shape[1]
    np.testing.assert_allclose(model.predict(X), stock.predict(X))


def test_scalar_sample_weight_for_float32_is_float32_like_stock():
    rng = np.random.default_rng(0)
    for _ in range(5):
        X = (3 + rng.normal(size=(300, 5))).astype(np.float32)
        y = (X.astype(np.float64) @ rng.normal(size=5) + rng.normal(size=300)).astype(np.float32)
        w = float(rng.uniform(0.3, 3.0))
        mlx = LinearRegression().fit(X, y, sample_weight=w)
        ref = SkLinearRegression().fit(X, y, sample_weight=w)
        np.testing.assert_array_equal(mlx.coef_, ref.coef_.astype(np.float64))
        np.testing.assert_array_equal(mlx.singular_, ref.singular_.astype(np.float64))
