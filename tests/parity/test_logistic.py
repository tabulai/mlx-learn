"""LogisticRegression parity against stock scikit-learn.

The load-bearing assertion here is the **objective value**, not the coefficients.
Two correct solvers stopping at slightly different points land in measurably
different places in coefficient space while sitting at essentially the same
objective; a test that only compares `coef_` therefore fails for a correct
implementation and passes for a subtly wrong one. Every coefficient comparison
below is paired with an objective comparison computed by one float64 function
applied to *both* solutions.

Tolerances come from ``docs/fp32_policy.md`` §3.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from mlxlearn._common.config import tuning_context
from mlxlearn._solvers.logistic import reference_objective
from mlxlearn.exceptions import SparseInputError, UnsupportedParameterError
from mlxlearn.linear_model import LogisticRegression

# docs/fp32_policy.md §3
OBJECTIVE_RTOL = 1e-5
COEF_ATOL, COEF_RTOL = 1e-3, 2e-3
PROBA_ATOL = 2e-3
MIN_AGREEMENT = 0.995


def _data(n_classes=2, n_samples=5000, n_features=20, seed=0):
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(12, n_features - 4),
        n_redundant=4,
        n_classes=n_classes,
        random_state=seed,
    )


def _assert_parity(mlx, ref, X, y, *, C, sample_weight=None):
    """Compare two fitted logistic models on every axis the policy names."""
    n_classes = len(ref.classes_)
    encoded = np.searchsorted(ref.classes_, y)

    def objective(model):
        return reference_objective(
            model.coef_, model.intercept_, X, encoded,
            n_classes=n_classes, C=C, sample_weight=sample_weight,
        )

    j_mlx, j_ref = objective(mlx), objective(ref)
    assert j_mlx == pytest.approx(j_ref, rel=OBJECTIVE_RTOL), (
        f"objective differs: mlxlearn {j_mlx:.10f} vs scikit-learn {j_ref:.10f}"
    )

    np.testing.assert_allclose(mlx.coef_, ref.coef_, atol=COEF_ATOL, rtol=COEF_RTOL)
    np.testing.assert_allclose(mlx.intercept_, ref.intercept_, atol=COEF_ATOL, rtol=COEF_RTOL)

    agreement = np.mean(mlx.predict(X) == ref.predict(X))
    assert agreement >= MIN_AGREEMENT, f"label agreement {agreement:.4f}"
    np.testing.assert_allclose(mlx.predict_proba(X), ref.predict_proba(X), atol=PROBA_ATOL)


@pytest.mark.parametrize("C", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("n_classes", [2, 5])
def test_parity_across_regularization(C, n_classes):
    X, y = _data(n_classes=n_classes)
    # Forced onto MLX: this dataset sits far below the measured logistic
    # crossover, so the default dispatch would hand it to scikit-learn and the
    # test would compare scikit-learn against itself.
    with tuning_context(force_backend="mlx"):
        mlx = LogisticRegression(C=C).fit(X, y)
    ref = SkLogisticRegression(C=C, solver="lbfgs").fit(X, y)
    assert mlx.execution_backend_ == "mlx"
    _assert_parity(mlx, ref, X, y, C=C)


@pytest.mark.parametrize("n_classes", [2, 5])
def test_parity_without_intercept(n_classes):
    X, y = _data(n_classes=n_classes)
    mlx = LogisticRegression(fit_intercept=False).fit(X, y)
    ref = SkLogisticRegression(fit_intercept=False, solver="lbfgs").fit(X, y)
    _assert_parity(mlx, ref, X, y, C=1.0)


@pytest.mark.parametrize("n_classes", [2, 5])
def test_parity_with_balanced_class_weight(n_classes):
    X, y = _data(n_classes=n_classes, seed=3)
    keep = np.concatenate([np.where(y == 0)[0], np.where(y != 0)[0][:400]])
    X, y = X[keep], y[keep]

    mlx = LogisticRegression(class_weight="balanced").fit(X, y)
    ref = SkLogisticRegression(class_weight="balanced", solver="lbfgs").fit(X, y)

    # The objective is compared through the equivalent sample weights, since
    # `balanced` is defined as a particular reweighting and comparing the
    # unweighted objective would be comparing the wrong function.
    from sklearn.utils.class_weight import compute_class_weight

    per_class = compute_class_weight("balanced", classes=ref.classes_, y=y)
    weights = per_class[np.searchsorted(ref.classes_, y)]
    _assert_parity(mlx, ref, X, y, C=1.0, sample_weight=weights)


def test_parity_with_sample_weight():
    X, y = _data()
    sample_weight = np.where(y == 1, 10.0, 1.0)
    mlx = LogisticRegression().fit(X, y, sample_weight=sample_weight)
    ref = SkLogisticRegression(solver="lbfgs").fit(X, y, sample_weight=sample_weight)
    _assert_parity(mlx, ref, X, y, C=1.0, sample_weight=sample_weight)


def test_class_weight_equals_the_equivalent_sample_weight():
    """The two paths must produce the same model, because they are the same model."""
    X, y = _data()
    by_class = LogisticRegression(class_weight={0: 1.0, 1: 3.0}).fit(X, y)
    by_sample = LogisticRegression().fit(X, y, sample_weight=np.where(y == 1, 3.0, 1.0))
    np.testing.assert_allclose(by_class.coef_, by_sample.coef_, atol=1e-10)


def test_unit_sample_weight_is_the_unweighted_fit():
    X, y = _data()
    weighted = LogisticRegression().fit(X, y, sample_weight=np.ones(len(y)))
    plain = LogisticRegression().fit(X, y)
    np.testing.assert_allclose(weighted.coef_, plain.coef_, atol=1e-10)


def test_unpenalized_parity():
    X, y = _data()
    mlx = LogisticRegression(penalty=None).fit(X, y)
    ref = SkLogisticRegression(penalty=None, solver="lbfgs").fit(X, y)
    assert np.mean(mlx.predict(X) == ref.predict(X)) >= MIN_AGREEMENT


@pytest.mark.parametrize("C", [1.0, 1e4, 1e6])
def test_separable_data_terminates(C):
    """The minimizer is at infinity; both must stop, finitely, and agree on labels.

    Relative objective parity is deliberately *not* asserted here — see
    docs/fp32_policy.md §3.2. Comparing two numbers near zero relatively says
    nothing; at C=1e6 the measured relative gap is 0.114 from an absolute gap of
    2.8e-06, with 100% label agreement.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 10))
    y = (X[:, 0] > 0).astype(int)

    mlx = LogisticRegression(C=C, max_iter=200).fit(X, y)
    ref = SkLogisticRegression(C=C, max_iter=200, solver="lbfgs").fit(X, y)

    assert np.all(np.isfinite(mlx.coef_))
    assert np.all(np.isfinite(mlx.predict_proba(X)))
    np.testing.assert_array_equal(mlx.predict(X), ref.predict(X))


def test_fit_is_a_pure_function_of_its_arguments():
    """No warm start, no continuation cache.

    The ancestor cached solutions keyed on a raw array address and warm-started
    from them by default, so ``fit(A); fit(B)`` gave a different model than
    ``fit(B)`` alone on unrelated data of the same shape.
    """
    X_a, y_a = _data(seed=1)
    X_b, y_b = _data(seed=2)

    sequential = LogisticRegression().fit(X_a, y_a).fit(X_b, y_b)
    fresh = LogisticRegression().fit(X_b, y_b)
    np.testing.assert_array_equal(sequential.coef_, fresh.coef_)


def test_mutating_the_input_after_fit_does_not_change_predictions():
    X, y = _data()
    X = np.array(X)
    query = X[:50].copy()
    model = LogisticRegression().fit(X, y)
    before = model.predict_proba(query)
    X[:] = 0.0
    np.testing.assert_array_equal(before, model.predict_proba(query))


@pytest.mark.parametrize("backend", ["mlx", "cpu"])
def test_backends_agree(backend):
    X, y = _data(n_samples=3000)
    with tuning_context(force_backend=backend):
        model = LogisticRegression().fit(X, y)
    assert model.execution_backend_ == backend

    with tuning_context(force_backend="mlx"):
        reference = LogisticRegression().fit(X, y)
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-5)


def test_fitted_attribute_shapes_and_dtypes():
    X, y = _data(n_classes=5)
    model = LogisticRegression().fit(X, y)
    assert model.coef_.shape == (5, X.shape[1])
    assert model.intercept_.shape == (5,)
    assert model.coef_.dtype == np.float64
    assert model.n_iter_.shape == (1,)

    X2, y2 = _data(n_classes=2)
    binary = LogisticRegression().fit(X2, y2)
    assert binary.coef_.shape == (1, X2.shape[1])
    assert binary.intercept_.shape == (1,)


@pytest.mark.parametrize(
    "kwargs, parameter",
    [
        ({"solver": "liblinear"}, "solver"),
        ({"solver": "saga"}, "solver"),
        ({"penalty": "l1", "solver": "saga"}, "solver"),
        ({"dual": True, "solver": "liblinear", "penalty": "l2"}, "solver"),
        ({"warm_start": True}, "warm_start"),
    ],
)
def test_unsupported_parameters_raise(kwargs, parameter):
    X, y = _data(n_samples=300)
    with pytest.raises(UnsupportedParameterError):
        LogisticRegression(**kwargs).fit(X, y)


def test_sparse_input_raises_type_error():
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _data(n_samples=300)
    with pytest.raises(SparseInputError) as excinfo:
        LogisticRegression().fit(sparse.csr_matrix(X), y)
    assert isinstance(excinfo.value, TypeError)


def test_non_convergence_warns():
    from sklearn.exceptions import ConvergenceWarning

    X, y = _data()
    with pytest.warns(ConvergenceWarning):
        LogisticRegression(max_iter=1).fit(X, y)


def test_sample_weight_equivalence_at_float32_tolerance():
    """scikit-learn's own check runs at float64 tolerances and is xfailed (Rule A).

    The property is real and is asserted here at the tolerance float32 actually
    supports, so the xfail does not mean the behavior goes untested.
    """
    X, y = _data(n_samples=800, n_features=10)
    weights = np.random.default_rng(0).integers(1, 4, size=len(y)).astype(float)

    weighted = LogisticRegression().fit(X, y, sample_weight=weights)
    repeated = LogisticRegression().fit(
        np.repeat(X, weights.astype(int), axis=0),
        np.repeat(y, weights.astype(int)),
    )
    np.testing.assert_allclose(weighted.coef_, repeated.coef_, atol=1e-3, rtol=2e-3)
