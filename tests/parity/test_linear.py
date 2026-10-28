"""LinearRegression parity against stock scikit-learn.

The load-bearing assertion is the **objective value**, not the coefficients.
Ordinary least squares is convex, so two correct factorizations of a
full-rank design land in the same place; a rank-deficient design has a
min-norm family, and coefficient-only comparison then fails for a correct
implementation. Both coefficient sets are scored by one float64 function.

Tolerances come from ``docs/fp32_policy.md`` §3.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SkLinearRegression

from mlxlearn._common.config import tuning_context
from mlxlearn._solvers.ols import reference_objective
from mlxlearn.exceptions import SparseInputError, UnsupportedInputError, UnsupportedParameterError
from mlxlearn.linear_model import LinearRegression

OBJECTIVE_RTOL = 1e-5
COEF_ATOL, COEF_RTOL = 2e-3, 5e-3


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


def _assert_parity(mlx, ref, X, y, *, sample_weight=None):
    j_mlx = reference_objective(mlx.coef_, mlx.intercept_, X, y, sample_weight=sample_weight)
    j_ref = reference_objective(ref.coef_, ref.intercept_, X, y, sample_weight=sample_weight)
    assert j_mlx == pytest.approx(j_ref, rel=OBJECTIVE_RTOL, abs=1e-8), (
        f"objective differs: mlxlearn {j_mlx:.10f} vs scikit-learn {j_ref:.10f}"
    )
    np.testing.assert_allclose(mlx.coef_, ref.coef_, atol=COEF_ATOL, rtol=COEF_RTOL)
    np.testing.assert_allclose(np.asarray(mlx.intercept_), np.asarray(ref.intercept_), atol=COEF_ATOL, rtol=COEF_RTOL)
    np.testing.assert_allclose(mlx.predict(X), ref.predict(X), atol=COEF_ATOL, rtol=COEF_RTOL)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("n_targets", [1, 3])
def test_parity_across_intercept_and_outputs(fit_intercept, n_targets):
    X, y = _data(n_targets=n_targets)
    with tuning_context(force_backend="mlx"):
        mlx = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    ref = SkLinearRegression(fit_intercept=fit_intercept).fit(X, y)
    assert mlx.execution_backend_ == "mlx"
    _assert_parity(mlx, ref, X, y)


def test_parity_with_sample_weight():
    X, y = _data()
    rng = np.random.default_rng(0)
    sample_weight = rng.uniform(0.5, 3.0, size=len(y))
    with tuning_context(force_backend="mlx"):
        mlx = LinearRegression().fit(X, y, sample_weight=sample_weight)
    ref = SkLinearRegression().fit(X, y, sample_weight=sample_weight)
    _assert_parity(mlx, ref, X, y, sample_weight=sample_weight)


def test_unit_sample_weight_is_the_unweighted_fit():
    X, y = _data()
    with tuning_context(force_backend="mlx"):
        weighted = LinearRegression().fit(X, y, sample_weight=np.ones(len(y)))
        plain = LinearRegression().fit(X, y)
    np.testing.assert_allclose(weighted.coef_, plain.coef_, atol=1e-6)


def test_sample_weight_equivalence_at_float32_tolerance():
    """Repeating a row ``w`` times equals a weight of ``w``, at float32 scale."""
    X, y = _data(n_samples=800, n_features=8, seed=1)
    weights = np.random.default_rng(0).integers(1, 4, size=len(y)).astype(float)
    with tuning_context(force_backend="mlx"):
        weighted = LinearRegression().fit(X, y, sample_weight=weights)
        repeated = LinearRegression().fit(
            np.repeat(X, weights.astype(int), axis=0),
            np.repeat(y, weights.astype(int)),
        )
    np.testing.assert_allclose(weighted.coef_, repeated.coef_, atol=1e-3, rtol=2e-3)


def test_underdetermined_min_norm_matches_sklearn():
    """More features than rows: both return the min-norm least-squares solution."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 40))
    y = X @ rng.normal(size=40) + rng.normal(scale=0.05, size=20)
    with tuning_context(force_backend="mlx"):
        mlx = LinearRegression(fit_intercept=False).fit(X, y)
    ref = SkLinearRegression(fit_intercept=False).fit(X, y)
    _assert_parity(mlx, ref, X, y)


def test_rank_deficient_objective_still_matches():
    """Duplicate columns: coefficients may differ, the objective must not."""
    X, y = _data(n_features=8, seed=4)
    X = np.hstack([X, X[:, :1]])
    with tuning_context(force_backend="mlx"):
        mlx = LinearRegression().fit(X, y)
    ref = SkLinearRegression().fit(X, y)
    j_mlx = reference_objective(mlx.coef_, mlx.intercept_, X, y)
    j_ref = reference_objective(ref.coef_, ref.intercept_, X, y)
    assert j_mlx == pytest.approx(j_ref, rel=OBJECTIVE_RTOL, abs=1e-8)
    np.testing.assert_allclose(mlx.predict(X), ref.predict(X), atol=COEF_ATOL, rtol=COEF_RTOL)


@pytest.mark.parametrize("offset", [0.0, 1e2, 1e4])
def test_centering_absorbs_a_constant_offset(offset):
    """A constant feature offset must not collapse predictions.

    ``coef_`` error of order ``1e-5`` times an offset of ``1e4`` moves the
    intercept by tenths, so intercepts are not compared here. The objective
    and the predictions are the restored model and are.
    """
    X, y = _data(n_samples=2000, n_features=8, seed=0)
    X = X + offset
    with tuning_context(force_backend="mlx"):
        mlx = LinearRegression().fit(X, y)
    ref = SkLinearRegression().fit(X, y)
    j_mlx = reference_objective(mlx.coef_, mlx.intercept_, X, y)
    j_ref = reference_objective(ref.coef_, ref.intercept_, X, y)
    assert j_mlx == pytest.approx(j_ref, rel=OBJECTIVE_RTOL, abs=1e-8)
    np.testing.assert_allclose(mlx.predict(X), ref.predict(X), atol=COEF_ATOL, rtol=COEF_RTOL)


def test_no_centering_without_an_intercept():
    X, y = _data(n_samples=2000, n_features=8)
    X = X + 5.0
    with tuning_context(force_backend="mlx"):
        mlx = LinearRegression(fit_intercept=False).fit(X, y)
    ref = SkLinearRegression(fit_intercept=False).fit(X, y)
    _assert_parity(mlx, ref, X, y)
    assert mlx.intercept_ == 0.0


def test_fitted_attribute_shapes_and_dtypes():
    X, y = _data(n_targets=1)
    with tuning_context(force_backend="mlx"):
        model = LinearRegression().fit(X, y)
    assert model.coef_.shape == (X.shape[1],)
    assert model.coef_.dtype == np.float64
    assert np.ndim(model.intercept_) == 0
    assert model.singular_.dtype == np.float64
    assert model.rank_ >= 1

    X2, y2 = _data(n_targets=3, seed=1)
    with tuning_context(force_backend="mlx"):
        multi = LinearRegression().fit(X2, y2)
    assert multi.coef_.shape == (3, X2.shape[1])
    assert multi.intercept_.shape == (3,)


def test_inference_reuses_the_fitted_backend():
    from mlxlearn import get_backend_diagnostics

    X, y = _data()
    with tuning_context(force_backend="mlx"):
        model = LinearRegression().fit(X, y)
    fitted = model.execution_backend_
    model.predict(X[:8])
    inference = [e for e in get_backend_diagnostics() if e.operation == "predict"]
    assert inference
    assert all(e.backend == fitted for e in inference)
    assert all(e.reason == "sticky" for e in inference)


def test_small_problems_use_the_cpu_path(small_classification_data):
    X, y = small_classification_data
    model = LinearRegression().fit(X, y.astype(float))
    assert model.execution_backend_ == "cpu"


def test_forced_backend_overrides_crossover(small_classification_data):
    X, y = small_classification_data
    with tuning_context(force_backend="mlx"):
        model = LinearRegression().fit(X, y.astype(float))
    assert model.execution_backend_ == "mlx"


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


def test_float32_overflow_is_a_capability_error():
    X, y = _data(n_samples=200, n_features=4)
    X = np.array(X)
    X[0, 0] = 1e300
    with pytest.raises(UnsupportedInputError) as excinfo:
        LinearRegression().fit(X, y)
    assert excinfo.value.reason == "float32-overflow"


def test_negative_sample_weight_is_a_capability_error():
    X, y = _data(n_samples=200, n_features=4)
    weights = np.ones(len(y))
    weights[0] = -1.0
    with pytest.raises(UnsupportedInputError) as excinfo:
        LinearRegression().fit(X, y, sample_weight=weights)
    assert excinfo.value.reason == "negative-sample-weight"


def test_fit_is_a_pure_function_of_its_arguments():
    X_a, y_a = _data(seed=1)
    X_b, y_b = _data(seed=2)
    with tuning_context(force_backend="mlx"):
        sequential = LinearRegression().fit(X_a, y_a).fit(X_b, y_b)
        fresh = LinearRegression().fit(X_b, y_b)
    np.testing.assert_array_equal(sequential.coef_, fresh.coef_)


def test_mutating_the_input_after_fit_does_not_change_predictions():
    X, y = _data()
    X = np.array(X)
    query = X[:50].copy()
    with tuning_context(force_backend="mlx"):
        model = LinearRegression().fit(X, y)
    before = model.predict(query)
    X[:] = 0.0
    np.testing.assert_array_equal(before, model.predict(query))


@pytest.mark.parametrize("backend", ["mlx", "cpu"])
def test_backends_agree(backend):
    X, y = _data(n_samples=3000)
    with tuning_context(force_backend=backend):
        model = LinearRegression().fit(X, y)
    assert model.execution_backend_ == backend
    with tuning_context(force_backend="mlx"):
        reference = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-5)


def test_patched_positive_falls_back():
    import sklearn.linear_model

    import mlxlearn

    X, y = _data(n_samples=200, n_features=4)
    mlxlearn.patch_sklearn()
    model = sklearn.linear_model.LinearRegression(positive=True).fit(X, y)
    assert model.execution_backend_ == "sklearn"
    events = [e for e in mlxlearn.get_backend_diagnostics() if e.fallback]
    assert any("positive" in e.reason for e in events)
