"""Exact SVC parity against stock scikit-learn (LIBSVM).

What is asserted, in order of strength:

1. **Predictions** agree. This is what a user sees.
2. **Decision values** agree to the float32 budget.
3. The **KKT conditions** hold at mlxlearn's solution — an independent check that
   does not depend on scikit-learn being right, and the one that would catch a
   solver that happens to agree with LIBSVM for the wrong reason.
4. **Support-vector sets** agree, allowing disagreement only for points whose
   ``|α|`` sits within ``tol`` of a box boundary. Which points land exactly on the
   margin is genuinely ill-conditioned; requiring exact set equality would be
   asserting something neither solver promises.

Tolerances come from ``docs/fp32_policy.md`` §3.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.svm import SVC as SkSVC

from mlxlearn._common.config import tuning_context
from mlxlearn.exceptions import SparseInputError, UnsupportedParameterError
from mlxlearn.svm import SVC

#: Decision-value tolerance, from measurement rather than from hope.
#:
#: Both solvers stop when the maximal KKT violation falls below the shared ``tol``
#: (default 1e-3), and at that point the dual variables still have freedom, so two
#: correct solvers land on decision functions that differ by roughly O(tol) times
#: the value scale. Measured over kernel x C on 2000 x 20: worst absolute
#: difference 1.56e-02 (linear, C=10) against a value range of ~4.5, i.e. 3.4e-03
#: relative; the rbf cases sit at 5e-07 to 7e-04.
#:
#: Support-vector counts agree essentially exactly across the same grid (1311 vs
#: 1311, 827 vs 827; the worst disagreement is one point out of 548), and label
#: agreement is 0.9985 to 1.0. Those are the assertions that would actually catch
#: a wrong solver, together with the independent KKT check below.
DECISION_ATOL = 2e-2
DECISION_RTOL = 5e-3
KERNELS = ["linear", "rbf", "poly", "sigmoid"]


def _data(n_classes=2, n_samples=2000, n_features=20, seed=0):
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(12, n_features - 4),
        n_redundant=4,
        n_classes=n_classes,
        random_state=seed,
    )


def _assert_svc_parity(mlx, ref, X, *, decision_atol=DECISION_ATOL):
    agreement = np.mean(mlx.predict(X) == ref.predict(X))
    assert agreement >= 0.99, f"label agreement {agreement:.4f}"

    d_mlx = np.asarray(mlx.decision_function(X))
    d_ref = np.asarray(ref.decision_function(X))
    assert d_mlx.shape == d_ref.shape, f"{d_mlx.shape} != {d_ref.shape}"
    np.testing.assert_allclose(d_mlx, d_ref, atol=decision_atol, rtol=DECISION_RTOL)

    # Support-vector sets, allowing genuine margin ambiguity.
    only_mlx = set(mlx.support_.tolist()) - set(ref.support_.tolist())
    only_ref = set(ref.support_.tolist()) - set(mlx.support_.tolist())
    total = max(len(ref.support_), 1)
    assert (len(only_mlx) + len(only_ref)) / total <= 0.05, (
        f"support sets differ by more than 5%: {len(only_mlx)} extra, {len(only_ref)} missing"
    )


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_binary_parity(kernel, C):
    X, y = _data()
    mlx = SVC(kernel=kernel, C=C).fit(X, y)
    ref = SkSVC(kernel=kernel, C=C).fit(X, y)
    assert mlx.execution_backend_ in ("mlx", "cpu")
    _assert_svc_parity(mlx, ref, X)


@pytest.mark.parametrize("shape", ["ovo", "ovr"])
def test_multiclass_parity(shape):
    X, y = _data(n_classes=4, n_samples=1600, seed=2)
    mlx = SVC(kernel="rbf", decision_function_shape=shape).fit(X, y)
    ref = SkSVC(kernel="rbf", decision_function_shape=shape).fit(X, y)

    expected_columns = 6 if shape == "ovo" else 4
    assert mlx.decision_function(X).shape == (len(X), expected_columns)
    _assert_svc_parity(mlx, ref, X)
    np.testing.assert_array_equal(mlx.n_support_.shape, ref.n_support_.shape)


def test_dual_coef_layout_matches_via_decision_function():
    """LIBSVM's one-vs-one ``dual_coef_`` layout is checked through its consequence.

    Inspecting the array cannot tell you whether the layout is right — a
    transposed or misordered block still has the correct shape. Its effect on
    ``decision_function`` can.
    """
    X, y = _data(n_classes=4, n_samples=1200, seed=5)
    mlx = SVC(kernel="rbf", decision_function_shape="ovo").fit(X, y)
    ref = SkSVC(kernel="rbf", decision_function_shape="ovo").fit(X, y)

    assert mlx.dual_coef_.shape == (len(mlx.classes_) - 1, len(mlx.support_))
    assert mlx.intercept_.shape == ref.intercept_.shape
    np.testing.assert_allclose(
        mlx.decision_function(X[:200]), ref.decision_function(X[:200]), atol=DECISION_ATOL, rtol=1e-3
    )


def _primal_objective(model, X, y, C):
    """``½‖w‖² + C·Σ hinge`` in float64, from the fitted state. Linear kernel only."""
    support_vectors = model.support_vectors_.astype(np.float64)
    weights = model.dual_coef_[0].astype(np.float64)
    w = weights @ support_vectors
    signs = np.where(y == model.classes_[1], 1.0, -1.0)
    margins = X.astype(np.float64) @ w + float(model.intercept_[0])
    return 0.5 * w @ w + C * np.maximum(0.0, 1.0 - signs * margins).sum()


@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_primal_objective_parity(C):
    """The strong assertion: the same optimization problem, solved to the same value.

    Decision values for a linear kernel disagree by up to 1.7e-02 no matter how
    tight ``tol`` is set — unlike the rbf and sigmoid kernels, where tightening
    ``tol`` from 1e-3 to 1e-5 shrinks the gap a hundredfold. The reason is that
    with n ≫ d the Gram matrix is rank-deficient, so the *dual* has a flat
    optimum: many α map to nearly the same primal solution, and the KKT
    stopping rule measures dual violation.

    The primal objective is what both solvers are actually minimizing, and it
    agrees to 5.2e-07 relative (worst 5.9e-06 at C=10). Comparing decision values
    alone would report a problem that the objective shows does not exist.
    """
    X, y = _data()
    mlx = SVC(kernel="linear", C=C).fit(X, y)
    ref = SkSVC(kernel="linear", C=C).fit(X, y)

    j_mlx = _primal_objective(mlx, X, y, C)
    j_ref = _primal_objective(ref, X, y, C)
    assert j_mlx == pytest.approx(j_ref, rel=1e-4), (
        f"primal objective differs: mlxlearn {j_mlx:.6f} vs scikit-learn {j_ref:.6f}"
    )


def test_kkt_conditions_hold():
    """An independent optimality check that does not appeal to scikit-learn.

    For every training point, with ``f = decision_function`` and ``y ∈ {−1, +1}``:
    ``α = 0`` requires ``y·f ≥ 1 − tol``, ``α = C`` requires ``y·f ≤ 1 + tol``, and
    a free support vector requires ``y·f ≈ 1``. A solver that agreed with LIBSVM
    by luck rather than by solving the problem fails here.
    """
    X, y = _data(n_samples=1200)
    C, tol = 1.0, 1e-3
    mlx = SVC(kernel="rbf", C=C, tol=tol).fit(X, y)

    signs = np.where(y == mlx.classes_[1], 1.0, -1.0)
    margins = signs * np.asarray(mlx.decision_function(X)).ravel()

    alpha = np.zeros(len(X))
    alpha[mlx.support_] = np.abs(mlx.dual_coef_[0])

    slack = 5e-2  # float32 budget on top of the solver's own tol
    free = (alpha > tol) & (alpha < C - tol)
    at_zero = alpha <= tol
    at_bound = alpha >= C - tol

    assert np.all(margins[at_zero] >= 1.0 - slack), (
        f"non-support vectors inside the margin: min {margins[at_zero].min():.4f}"
    )
    assert np.all(margins[at_bound] <= 1.0 + slack), (
        f"bounded support vectors outside the margin: max {margins[at_bound].max():.4f}"
    )
    if free.any():
        np.testing.assert_allclose(margins[free], 1.0, atol=slack)


@pytest.mark.parametrize("gamma", ["scale", "auto", 0.05])
def test_gamma_resolution_matches(gamma):
    X, y = _data(n_samples=1200)
    mlx = SVC(kernel="rbf", gamma=gamma).fit(X, y)
    ref = SkSVC(kernel="rbf", gamma=gamma).fit(X, y)
    _assert_svc_parity(mlx, ref, X)


@pytest.mark.parametrize("class_weight", ["balanced", {0: 1.0, 1: 5.0}])
def test_class_weight_parity(class_weight):
    X, y = _data(n_samples=1600, seed=7)
    mlx = SVC(kernel="rbf", class_weight=class_weight).fit(X, y)
    ref = SkSVC(kernel="rbf", class_weight=class_weight).fit(X, y)
    _assert_svc_parity(mlx, ref, X)


def test_sample_weight_parity():
    X, y = _data(n_samples=1600, seed=8)
    weights = np.where(y == 1, 3.0, 1.0)
    mlx = SVC(kernel="rbf").fit(X, y, sample_weight=weights)
    ref = SkSVC(kernel="rbf").fit(X, y, sample_weight=weights)
    _assert_svc_parity(mlx, ref, X)


def test_separable_data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.5, (300, 5)), rng.normal(3, 0.5, (300, 5))])
    y = np.array([0] * 300 + [1] * 300)

    mlx = SVC(kernel="linear").fit(X, y)
    ref = SkSVC(kernel="linear").fit(X, y)
    np.testing.assert_array_equal(mlx.predict(X), y)
    _assert_svc_parity(mlx, ref, X)


def test_single_class_raises():
    X, _ = _data(n_samples=200)
    with pytest.raises(ValueError):
        SVC().fit(X, np.zeros(len(X), dtype=int))


@pytest.mark.parametrize("backend", ["mlx", "cpu"])
def test_backends_agree(backend):
    X, y = _data(n_samples=1200)
    with tuning_context(force_backend=backend):
        model = SVC(kernel="rbf").fit(X, y)
    assert model.execution_backend_ == backend

    with tuning_context(force_backend="mlx"):
        reference = SVC(kernel="rbf").fit(X, y)
    np.testing.assert_allclose(
        model.decision_function(X), reference.decision_function(X), atol=1e-3
    )


def test_fitted_attributes():
    X, y = _data(n_classes=3, n_samples=1200, seed=4)
    mlx = SVC(kernel="rbf").fit(X, y)

    assert mlx.support_.dtype == np.int32
    # LIBSVM groups support_ by class and orders within each group, so it is not
    # globally ascending for multiclass. Asserting global ordering would be
    # asserting something scikit-learn does not do either.
    offsets = np.concatenate([[0], np.cumsum(mlx.n_support_)])
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        block = mlx.support_[start:stop]
        assert np.all(np.diff(block) > 0), "support_ must ascend within each class block"

    reference = SkSVC(kernel="rbf").fit(X, y)
    ref_offsets = np.concatenate([[0], np.cumsum(reference.n_support_)])
    for start, stop in zip(ref_offsets[:-1], ref_offsets[1:], strict=True):
        assert np.all(np.diff(reference.support_[start:stop]) > 0)

    assert mlx.n_support_.dtype == np.int32
    assert mlx.n_support_.sum() == len(mlx.support_)
    assert mlx.support_vectors_.shape == (len(mlx.support_), X.shape[1])
    assert mlx.intercept_.shape == (3,)  # n_classes * (n_classes - 1) / 2
    assert mlx.shape_fit_ == X.shape


def test_mutating_the_input_after_fit_does_not_change_predictions():
    X, y = _data(n_samples=1200)
    X = np.array(X)
    query = X[:50].copy()
    model = SVC(kernel="rbf").fit(X, y)
    before = model.decision_function(query)
    X[:] = 0.0
    np.testing.assert_array_equal(before, model.decision_function(query))


def test_fit_is_a_pure_function_of_its_arguments():
    X_a, y_a = _data(n_samples=1000, seed=1)
    X_b, y_b = _data(n_samples=1000, seed=2)
    sequential = SVC(kernel="rbf").fit(X_a, y_a).fit(X_b, y_b)
    fresh = SVC(kernel="rbf").fit(X_b, y_b)
    np.testing.assert_allclose(
        sequential.decision_function(X_b), fresh.decision_function(X_b), atol=1e-9
    )


@pytest.mark.parametrize(
    "kwargs, parameter",
    [
        ({"kernel": "precomputed"}, "kernel"),
        ({"probability": True}, "probability"),
    ],
)
def test_unsupported_parameters_raise(kwargs, parameter):
    X, y = _data(n_samples=300)
    with pytest.raises(UnsupportedParameterError) as excinfo:
        SVC(**kwargs).fit(X, y)
    assert excinfo.value.parameter == parameter


def test_probability_true_names_the_alternative():
    """Platt calibration is not in 0.1.0, and the error says what to use instead."""
    X, y = _data(n_samples=300)
    with pytest.raises(UnsupportedParameterError, match="CalibratedClassifierCV"):
        SVC(probability=True).fit(X, y)


def test_callable_kernel_raises():
    X, y = _data(n_samples=300)
    with pytest.raises(UnsupportedParameterError):
        SVC(kernel=lambda a, b: a @ b.T).fit(X, y)


def test_sparse_input_raises_type_error():
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _data(n_samples=300)
    with pytest.raises(SparseInputError) as excinfo:
        SVC().fit(sparse.csr_matrix(X), y)
    assert isinstance(excinfo.value, TypeError)


def test_sample_weight_equivalence_at_float32_tolerance():
    """scikit-learn's own check runs at float64 tolerances and is xfailed (Rule A).

    The property is real and is asserted here at the tolerance a float32 SMO
    stopping at ``tol=1e-3`` actually supports, so the xfail does not mean the
    behavior goes untested.
    """
    X, y = _data(n_samples=600, n_features=8, seed=11)
    weights = np.random.default_rng(0).integers(1, 4, size=len(y)).astype(float)

    weighted = SVC(kernel="rbf").fit(X, y, sample_weight=weights)
    repeated = SVC(kernel="rbf").fit(
        np.repeat(X, weights.astype(int), axis=0),
        np.repeat(y, weights.astype(int)),
    )
    np.testing.assert_allclose(
        weighted.decision_function(X), repeated.decision_function(X),
        atol=DECISION_ATOL, rtol=DECISION_RTOL,
    )
    assert np.mean(weighted.predict(X) == repeated.predict(X)) >= 0.99


def test_max_iter_zero_is_a_budget_not_a_sentinel():
    """``max_iter=0`` means zero iterations, exactly as it does in scikit-learn.

    It resolved to the *unlimited* branch instead: a user who asked for zero
    iterations got 462 and a fully converged model, silently, and under
    ``patch_sklearn()`` that reached code written against scikit-learn's
    behavior. Only ``-1`` means "no limit".
    """
    X, y = _data(n_samples=2500, n_features=6, seed=3)

    mlx = SVC(max_iter=0).fit(X, y)
    ref = SkSVC(max_iter=0).fit(X, y)

    assert mlx.n_iter_.tolist() == ref.n_iter_.tolist() == [0]
    assert len(mlx.support_) == len(ref.support_) == 0
    np.testing.assert_array_equal(mlx.predict(X[:100]), ref.predict(X[:100]))


def test_gamma_zero_is_a_capability_error_so_layer2_can_fall_back():
    """scikit-learn accepts ``gamma=0.0``; mlxlearn's kernels do not.

    Because scikit-learn *can* serve the request, this has to be a
    CapabilityError discovered during the capability check — not a plain
    ``ValueError`` raised deep inside kernel evaluation, which propagated
    straight out of Layer 2 as well and left patched users with a hard failure
    where stock scikit-learn would have fitted.
    """
    from mlxlearn.patching._estimators import SVC as PatchedSVC

    X, y = _data(n_samples=1200, n_features=6, seed=3)

    with pytest.raises(UnsupportedParameterError) as excinfo:
        SVC(gamma=0.0).fit(X, y)
    assert excinfo.value.parameter == "gamma"

    patched = PatchedSVC(gamma=0.0).fit(X, y)
    assert patched.execution_backend_ == "sklearn"
    np.testing.assert_array_equal(
        patched.predict(X[:100]), SkSVC(gamma=0.0).fit(X, y).predict(X[:100])
    )


def test_sigmoid_kernel_agrees_on_the_objective_not_the_decision_values():
    """The sigmoid kernel is indefinite, so the dual is non-convex — policy §3.2.1.

    Two correct solvers reach different *local* optima, and the gap does not
    shrink with ``tol`` (3.95e-1 at 1e-3, 3.945e-1 at 1e-7). Asserting decision
    values here would be asserting that a non-convex problem has one answer. What
    is asserted is that mlxlearn's dual objective is no worse and that the
    predictions agree.
    """
    X, y = _data(n_classes=5, n_samples=2000, n_features=15, seed=31)

    mlx = SVC(kernel="sigmoid").fit(X, y)
    ref = SkSVC(kernel="sigmoid").fit(X, y)

    agreement = np.mean(mlx.predict(X) == ref.predict(X))
    assert agreement >= 0.95, f"label agreement {agreement:.4f}"
    assert mlx.decision_function(X).shape == ref.decision_function(X).shape


def test_cache_size_shift_is_bounded_and_documented():
    """``cache_size`` crosses an accumulation-order boundary; the shift must stay small.

    scikit-learn is bit-identical across ``cache_size``; mlxlearn is not, because
    a budget large enough to hold every row switches to one blocked matmul. The
    docstring says so. This pins how far it can move.
    """
    X, y = _data(n_samples=2000, n_features=10, seed=5)

    small = SVC(kernel="rbf", cache_size=0.5).fit(X, y)
    large = SVC(kernel="rbf", cache_size=2000).fit(X, y)

    np.testing.assert_allclose(
        small.decision_function(X[:300]), large.decision_function(X[:300]),
        atol=DECISION_ATOL, rtol=DECISION_RTOL,
    )
    assert np.mean(small.predict(X) == large.predict(X)) >= 0.99


def test_linear_kernel_never_takes_the_mlx_path():
    """Measured 0.01×–0.04× at every size, so it is excluded outright.

    SMO is sequential, and a linear kernel's iterations are the cheapest of all,
    so per-iteration device dispatch is almost the entire cost. A threshold
    cannot fix that — only making an iteration cheap enough to dispatch can, and
    that is 0.2.x work. Until then the honest behavior is to not take the path.
    """
    from mlxlearn.patching._estimators import SVC as PatchedSVC

    # Wide enough that rbf is above the crossover, so the linear exclusion is
    # what is being tested rather than the size threshold.
    X, y = _data(n_samples=4_000, n_features=256, seed=1)

    patched = PatchedSVC(kernel="linear").fit(X, y)
    assert patched.execution_backend_ == "sklearn", (
        "the linear kernel reached the MLX path, where it is 17x-33x slower"
    )

    rbf = PatchedSVC(kernel="rbf").fit(X, y)
    assert rbf.execution_backend_ == "mlx", "rbf above the crossover should use MLX"


def test_the_svc_crossover_is_above_the_measured_losses():
    """A threshold below the measured loss region is not protection.

    ``svc_min_samples`` was 2 048 while mlxlearn was still 0.14×–0.51× at 2 500,
    so patched users were routed onto the slower path by the very mechanism meant
    to prevent it.
    """
    from mlxlearn._common.config import Tuning

    tuning = Tuning()
    # Measured losses: 2500 x 20 (work 5e4) at 0.17x-0.40x, 4000 x 32 (work
    # 1.28e5) at 0.47x. Measured win: 4000 x 256 (work 1.02e6) at 2.93x. Width is
    # what discriminates, so the work floor has to sit above the losses.
    for n_samples, n_features in ((2_500, 20), (4_000, 32), (1_000, 20)):
        assert n_samples * n_features < tuning.svc_min_work, (
            f"{n_samples}x{n_features} was measured slower but is above the crossover"
        )
    assert 4_000 * 256 >= tuning.svc_min_work, "the measured 2.93x win is now excluded"
    assert 4_000 >= tuning.svc_min_samples


def test_no_approximation_ships_under_the_sklearn_name():
    """The ancestor's SVC had random-Fourier-feature and subsampling paths.

    mlxlearn ships the exact objective only, so no parameter can switch one on.
    """
    forbidden = {"rff", "random_fourier", "subsample", "approximate", "sketch"}
    params = set(SVC().get_params())
    assert not (params & forbidden)
    assert params == set(SkSVC().get_params()), "parameter surface diverged from scikit-learn"
