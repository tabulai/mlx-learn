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

Parity is asserted on **both** MLX devices. Everything here is below the SVC
crossover, so Layer 1 would silently run all of it on the MLX CPU device and never
compare the Metal path — the one users above the crossover actually get — with
LIBSVM at all. The non-linear cases are therefore parametrized over
``force_backend`` and assert the backend they ran on. The linear kernel is not: it
is routed off the GPU at every size, and forcing it there tests a path no
configuration reaches.

Tolerances come from ``docs/fp32_policy.md`` §3.
"""

from __future__ import annotations

import copy
import pickle
import sys
import threading
import time
import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import SVC as SkSVC

from mlxlearn import get_backend_diagnostics
from mlxlearn._common import device as device_module
from mlxlearn._common.config import config_context, get_tuning, tuning_context
from mlxlearn.exceptions import (
    CapabilityError,
    DeviceError,
    SparseInputError,
    UnsupportedInputError,
    UnsupportedParameterError,
)
from mlxlearn.patching._estimators import SVC as PatchedSVC
from mlxlearn.svm import SVC
from tests._helpers import join_or_fail

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
NONLINEAR = ["rbf", "poly", "sigmoid"]
BACKENDS = ["cpu", "mlx"]

#: The mlxlearn-only fitted state; a refit that ends on scikit-learn must drop it.
MLX_ATTRS = SVC._mlxlearn_mlx_attrs


def _data(n_classes=2, n_samples=2000, n_features=20, seed=0):
    n_informative = max(2, min(12, n_features - 4))
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=min(4, n_features - n_informative),
        n_classes=n_classes,
        random_state=seed,
    )


def _fit(backend, X, y, sample_weight=None, **params):
    """Layer 1 fit on a forced backend, asserting it really ran there."""
    with tuning_context(force_backend=backend):
        model = SVC(**params).fit(X, y, sample_weight=sample_weight)
    assert model.execution_backend_ == backend
    return model


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


@pytest.fixture(scope="module")
def binary_fits():
    """One fit per ``(kernel, C, backend)`` on the 2000 × 20 calibration set.

    ``test_binary_parity`` and ``test_primal_objective_parity`` assert different
    things about the *same* fit, and each fitted it separately — about 52 s apiece
    for linear at C=10, most of the suite's wall time. The models are only read
    afterwards, so sharing them loses nothing. The cache lives and dies with this
    module's fixture scope.
    """
    X, y = _data()
    references: dict = {}
    models: dict = {}

    def get(kernel, C, backend="cpu"):
        if (kernel, C) not in references:
            references[kernel, C] = SkSVC(kernel=kernel, C=C).fit(X, y)
        if (kernel, C, backend) not in models:
            models[kernel, C, backend] = _fit(backend, X, y, kernel=kernel, C=C)
        return X, y, models[kernel, C, backend], references[kernel, C]

    return get


def _binary_cases():
    for C in (0.1, 1.0, 10.0):
        for kernel in NONLINEAR:
            for backend in BACKENDS:
                yield pytest.param(kernel, C, backend, id=f"{kernel}-{C}-{backend}")
        # ~50 s: 3.9M SMO iterations of a rank-deficient dual. Shrinking, the
        # default, needs 1.7M more than the plain loop here, but runs them over
        # the active set only, in about the same time. It is the case the decision
        # tolerance above was calibrated on, so it stays, but only in the slow
        # run; the fast primal case below covers C=10 linear on small data.
        marks = [pytest.mark.slow] if C == 10.0 else []
        yield pytest.param("linear", C, "cpu", id=f"linear-{C}-cpu", marks=marks)


@pytest.mark.parametrize("kernel, C, backend", list(_binary_cases()))
def test_binary_parity(binary_fits, kernel, C, backend):
    X, _, mlx, ref = binary_fits(kernel, C, backend)
    _assert_svc_parity(mlx, ref, X)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", ["ovo", "ovr"])
def test_multiclass_parity(shape, backend):
    X, y = _data(n_classes=4, n_samples=1600, seed=2)
    mlx = _fit(backend, X, y, kernel="rbf", decision_function_shape=shape)
    ref = SkSVC(kernel="rbf", decision_function_shape=shape).fit(X, y)

    expected_columns = 6 if shape == "ovo" else 4
    assert mlx.decision_function(X).shape == (len(X), expected_columns)
    _assert_svc_parity(mlx, ref, X)
    np.testing.assert_array_equal(mlx.n_support_.shape, ref.n_support_.shape)


# ----------------------------------------------------------------- dual_coef_ layout


def _ovo_from_public_attributes(model, X):
    """One-vs-one decision values rebuilt from ``dual_coef_`` & co. alone.

    Pair ``(i, j)`` is ``K[:, sv_i] @ dual_coef_[j-1, sv_i] + K[:, sv_j] @
    dual_coef_[i, sv_j] + intercept_[k]``. Nothing mlxlearn computes internally
    is read, which is the point: its own ``decision_function`` uses
    ``_pair_weights_`` and cannot see this layout at all.
    """
    K = pairwise_kernels(
        X,
        model.support_vectors_,
        metric=model.kernel,
        filter_params=True,
        gamma=model._gamma,
        degree=model.degree,
        coef0=model.coef0,
    )
    offsets = np.concatenate([[0], np.cumsum(model.n_support_)])
    n_classes = len(model.classes_)
    columns = []
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            sv_i = slice(offsets[i], offsets[i + 1])
            sv_j = slice(offsets[j], offsets[j + 1])
            columns.append(
                K[:, sv_i] @ model.dual_coef_[j - 1, sv_i]
                + K[:, sv_j] @ model.dual_coef_[i, sv_j]
                + model.intercept_[k]
            )
            k += 1
    return np.stack(columns, axis=1)


@pytest.mark.parametrize(
    "kernel, n_classes", [("rbf", 4), ("poly", 3), ("linear", 3), ("rbf", 2), ("poly", 2)]
)
def test_dual_coef_layout_is_what_the_public_state_evaluates_to(kernel, n_classes):
    """LIBSVM's one-vs-one ``dual_coef_`` layout, checked through the public state.

    Comparing ``decision_function`` with scikit-learn's cannot detect a wrong
    layout: mlxlearn's inference reads ``_pair_weights_``, so a transposed or
    swapped ``dual_coef_`` leaves it untouched — a mutant with the class halves
    swapped passed every test in the suite. What does see the layout:

    * stock LIBSVM's ``decision_function`` run on mlxlearn's fitted state, which is
      what any consumer of the scikit-learn attributes gets;
    * the ovo decision rebuilt from ``dual_coef_``/``support_vectors_``/``n_support_``
      /``intercept_`` alone;
    * ``coef_`` for the linear kernel, which scikit-learn derives from the layout.

    All three are compared against mlxlearn's own decision values, which carry no
    solver tolerance at all — only float32 kernel rounding.
    """
    X, y = _data(n_classes=n_classes, n_samples=600, n_features=10, seed=5)
    mlx = SVC(kernel=kernel, decision_function_shape="ovo").fit(X, y)
    own = mlx.decision_function(X)

    assert mlx.dual_coef_.shape == (n_classes - 1, len(mlx.support_))
    libsvm = SkSVC.decision_function(mlx, X)
    np.testing.assert_allclose(libsvm, own, atol=1e-3, rtol=1e-4)

    public = _ovo_from_public_attributes(mlx, X)
    np.testing.assert_allclose(public if n_classes > 2 else public[:, 0], own, atol=1e-3, rtol=1e-4)

    if kernel == "linear":
        linear = X @ mlx.coef_.T + mlx.intercept_
        np.testing.assert_allclose(linear if n_classes > 2 else linear[:, 0], own, atol=1e-3)
        ref = SkSVC(kernel=kernel, decision_function_shape="ovo").fit(X, y)
        np.testing.assert_allclose(mlx.coef_, ref.coef_, atol=DECISION_ATOL)


def test_binary_sign_flip_is_applied_to_the_public_attributes_only():
    X, y = _data(n_samples=400, n_features=8, seed=3)
    mlx = SVC(kernel="rbf").fit(X, y)
    np.testing.assert_array_equal(mlx.dual_coef_, -mlx._dual_coef_)
    np.testing.assert_array_equal(mlx.intercept_, -mlx._intercept_)


# ----------------------------------------------------------------- objectives and KKT


def _primal_objective(model, X, y, C):
    """``½‖w‖² + C·Σ hinge`` in float64, from the fitted state. Linear kernel only."""
    support_vectors = model.support_vectors_.astype(np.float64)
    weights = model.dual_coef_[0].astype(np.float64)
    w = weights @ support_vectors
    signs = np.where(y == model.classes_[1], 1.0, -1.0)
    margins = X.astype(np.float64) @ w + float(model.intercept_[0])
    return 0.5 * w @ w + C * np.maximum(0.0, 1.0 - signs * margins).sum()


def _assert_primal_parity(mlx, ref, X, y, C):
    j_mlx = _primal_objective(mlx, X, y, C)
    j_ref = _primal_objective(ref, X, y, C)
    assert j_mlx == pytest.approx(j_ref, rel=1e-4), (
        f"primal objective differs: mlxlearn {j_mlx:.6f} vs scikit-learn {j_ref:.6f}"
    )


@pytest.mark.parametrize("C", [0.1, 1.0, pytest.param(10.0, marks=pytest.mark.slow)])
def test_primal_objective_parity(binary_fits, C):
    """The strong assertion: the same optimization problem, solved to the same value.

    Decision values for a linear kernel disagree by up to 1.7e-02 no matter how
    tight ``tol`` is set — unlike the rbf and sigmoid kernels, where tightening
    ``tol`` from 1e-3 to 1e-5 shrinks the gap a hundredfold. The reason is that
    with n ≫ d the Gram matrix is rank-deficient, so the *dual* has a flat
    optimum: many α map to nearly the same primal solution, and the KKT
    stopping rule measures dual violation.

    The primal objective is what both solvers are actually minimizing, and it
    agrees to 5.2e-07 relative (worst 6.2e-06 at C=10). Comparing decision values
    alone would report a problem that the objective shows does not exist.
    """
    X, y, mlx, ref = binary_fits("linear", C)
    _assert_primal_parity(mlx, ref, X, y, C)


def test_primal_objective_parity_at_large_C_on_small_data():
    """C=10 linear in about a second, so the fast suite still covers it.

    600 × 8 at seed 11 needs 45k SMO iterations with the default shrinking — a
    real workout of the flat dual — against 3.9M for the 2000 × 20 calibration
    set. Measured: primal relative difference 4.1e-06, decision values within
    5.0e-03, identical support counts.
    """
    X, y = _data(n_samples=600, n_features=8, seed=11)
    mlx = SVC(kernel="linear", C=10.0).fit(X, y)
    ref = SkSVC(kernel="linear", C=10.0).fit(X, y)
    _assert_primal_parity(mlx, ref, X, y, 10.0)
    _assert_svc_parity(mlx, ref, X)


@pytest.mark.parametrize("backend", BACKENDS)
def test_kkt_conditions_hold(backend):
    """An independent optimality check that does not appeal to scikit-learn.

    For every training point, with ``f = decision_function`` and ``y ∈ {−1, +1}``:
    ``α = 0`` requires ``y·f ≥ 1 − tol``, ``α = C`` requires ``y·f ≤ 1 + tol``, and
    a free support vector requires ``y·f ≈ 1``. A solver that agreed with LIBSVM
    by luck rather than by solving the problem fails here.

    The stopping rule bounds the maximal violating pair by ``tol`` and ``rho`` sits
    inside that interval, so in exact arithmetic no margin is off by more than
    ``tol``; the second ``tol`` is the float32 kernel's budget. Measured worst:
    4.9e-4 on the CPU device, 5.2e-4 on the GPU, 7.9e-4 over a 72-configuration
    seed × kernel × C sweep. The slack was 5e-2, which let a bias error of 0.045
    — or a solver stopped at ``tol=0.1`` — pass.
    """
    X, y = _data(n_samples=1200)
    C, tol = 1.0, 1e-3
    mlx = _fit(backend, X, y, kernel="rbf", C=C, tol=tol)

    signs = np.where(y == mlx.classes_[1], 1.0, -1.0)
    margins = signs * np.asarray(mlx.decision_function(X)).ravel()

    alpha = np.zeros(len(X))
    alpha[mlx.support_] = np.abs(mlx.dual_coef_[0])

    slack = 2 * tol
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


def _pair_duals_and_kkt(model, X, y, C):
    """Per one-vs-one pair: the LIBSVM dual objective and the worst KKT violation.

    The dual is ``½ cᵀKc − Σ|c|`` with ``c = y·α`` read from ``dual_coef_``, and the
    kernel evaluated in float64 from ``_gamma``. The KKT violation is measured on
    the pair's own training rows against its ovo decision value.
    """
    encoded = np.searchsorted(model.classes_, y)
    decision = model.decision_function(X)
    K_sv = pairwise_kernels(
        model.support_vectors_, metric=model.kernel, filter_params=True,
        gamma=model._gamma, degree=model.degree, coef0=model.coef0,
    )
    offsets = np.concatenate([[0], np.cumsum(model.n_support_)])
    n_classes = len(model.classes_)
    duals, violations = [], []
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            coef = np.zeros(len(model.support_))
            sv_i = slice(offsets[i], offsets[i + 1])
            sv_j = slice(offsets[j], offsets[j + 1])
            coef[sv_i] = model.dual_coef_[j - 1, sv_i]
            coef[sv_j] = model.dual_coef_[i, sv_j]
            duals.append(0.5 * coef @ K_sv @ coef - np.abs(coef).sum())

            rows = np.flatnonzero((encoded == i) | (encoded == j))
            labels = np.where(encoded[rows] == i, 1.0, -1.0)
            alpha = np.zeros(len(X))
            alpha[model.support_] = np.abs(coef)
            alpha = alpha[rows]
            margins = labels * decision[rows, k]
            violation = np.concatenate([
                (1.0 - margins)[alpha <= 0.0],
                (margins - 1.0)[alpha >= C],
                np.abs(margins - 1.0)[(alpha > 0.0) & (alpha < C)],
            ])
            violations.append(max(0.0, float(violation.max())))
            k += 1
    return np.array(duals), np.array(violations)


def test_sigmoid_kernel_reaches_a_kkt_point_near_libsvms():
    """The sigmoid kernel is indefinite, so the dual is non-convex — policy §3.2.1.

    Two correct solvers can reach different *local* optima, and on this data they
    do: the worst ovo pair's decision values differ by 0.63 and the gap does not
    shrink with ``tol``. Asserting decision values would be asserting that a
    non-convex problem has one answer. What is asserted instead:

    * **KKT at mlxlearn's own solution**, pair by pair — any stationary point is a
      legitimate answer to a non-convex problem, and this is the check that it is
      one. Measured worst 6.7e-4, against ``tol=1e-3``.
    * **Each pair's dual objective within 1e-3 of LIBSVM's**, two-sided. mlxlearn
      is not always the lower one: on pair (1, 2) here it is 2.3e-4 *higher*
      (−581.537 against −581.672), a different local optimum, not a worse solve.
      This pins today's behavior on this dataset; it is not a general guarantee.
    * Predictions agree.
    """
    X, y = _data(n_classes=5, n_samples=2000, n_features=15, seed=31)
    C, tol = 1.0, 1e-3

    mlx = SVC(kernel="sigmoid", C=C, tol=tol, decision_function_shape="ovo").fit(X, y)
    ref = SkSVC(kernel="sigmoid", C=C, tol=tol, decision_function_shape="ovo").fit(X, y)

    agreement = np.mean(mlx.predict(X) == ref.predict(X))
    assert agreement >= 0.95, f"label agreement {agreement:.4f}"

    dual_mlx, violation_mlx = _pair_duals_and_kkt(mlx, X, y, C)
    dual_ref, _ = _pair_duals_and_kkt(ref, X, y, C)
    assert violation_mlx.max() <= 2 * tol, f"KKT violation {violation_mlx.max():.2e}"
    np.testing.assert_allclose(dual_mlx, dual_ref, rtol=1e-3)


# ----------------------------------------------------------------- parameters


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("gamma", ["scale", "auto", 0.05])
def test_gamma_resolution_matches(gamma, backend):
    X, y = _data(n_samples=1200)
    mlx = _fit(backend, X, y, kernel="rbf", gamma=gamma)
    ref = SkSVC(kernel="rbf", gamma=gamma).fit(X, y)
    assert mlx._gamma == ref._gamma
    _assert_svc_parity(mlx, ref, X)


def test_gamma_scale_matches_scikit_learn_on_small_scale_data():
    """``gamma="scale"`` falls back to 1 only for an exactly-zero variance.

    It fell back whenever the variance was at most 1e-12 — features around 1e-6
    and below, such as raw SI units. The correct gamma there is 1e12 and up; with
    gamma=1 every kernel entry is 1, every point a support vector, and the model
    at chance (0.50 against 0.93), with no warning.
    """
    X, y = _data(n_samples=600, n_features=10, seed=0)
    X = X * 1e-7
    mlx = SVC().fit(X, y)
    ref = SkSVC().fit(X, y)

    assert mlx._gamma == pytest.approx(ref._gamma, rel=1e-12)
    _assert_svc_parity(mlx, ref, X)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "kernel, scale",
    [("rbf", 1e-20), ("rbf", 1e19), ("rbf", 1e20), ("poly", 1e20), ("sigmoid", 1e19)],
)
def test_extreme_magnitudes_are_served_exactly(kernel, scale, backend):
    """The float32 frame's power-of-two scale keeps squares inside float32.

    At 1e19 the squared norms overflow float32 (the model came back with a NaN
    intercept and no support vectors); at 1e-20 ``gamma="scale"`` is 3e38, past
    float32's range. Folding the exact power-of-two scale into gamma serves both
    with the decision values of unit-scale data — measured max 2e-6 against
    LIBSVM for rbf.
    """
    X, y = _data(n_samples=600, n_features=10, seed=0)
    X = X * scale
    mlx = _fit(backend, X, y, kernel=kernel)
    ref = SkSVC(kernel=kernel).fit(X, y)
    _assert_svc_parity(mlx, ref, X)


def test_scaling_is_bit_identical_for_in_range_data():
    """A power-of-two scale commutes with float32 rounding.

    So scaling the input by any power of two — which changes the frame's scale by
    the inverse — must reproduce the decision values bit for bit. Anything else
    would mean the frame is an approximation, not a change of units.
    """
    X, y = _data(n_samples=400, n_features=8, seed=4)
    for kernel in NONLINEAR:
        base = SVC(kernel=kernel).fit(X, y)
        scaled = SVC(kernel=kernel).fit(X * 2.0**40, y)
        np.testing.assert_array_equal(
            base.decision_function(X), scaled.decision_function(X * 2.0**40)
        )


def test_large_offset_is_removed_before_the_float32_cast():
    """The rbf translation runs in float64, before float32 rounds the offset away.

    Coordinates of 1e4 with a 1e-2 spread keep only about two significant digits
    of the spread once cast to float32; translating afterwards cannot recover
    them. Measured before: label agreement 0.989, decision values off by 0.19.
    """
    rng = np.random.default_rng(0)
    X = 1e4 + 1e-2 * rng.normal(size=(1500, 4))
    y = (X[:, 0] - 1e4 + 5e-3 * rng.normal(size=1500) > 0).astype(int)
    mlx = SVC().fit(X, y)
    ref = SkSVC().fit(X, y)
    _assert_svc_parity(mlx, ref, X)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rbf_on_tight_clusters_far_apart_matches_scikit_learn(backend):
    """Rows close to each other but far from the mean: centering cannot help.

    The Gram identity's float32 error grows with ``‖a‖² + ‖b‖²``, so with
    ``gamma`` matched to clusters 2 000 apart the kernel inside a cluster was
    wrong by up to 0.58: decision values off by 6.1 against a range of ±3, 87.5%
    of labels agreeing, ``fit_status_=0`` and no warning. The entries the
    identity may have got wrong are now recomputed directly — here and at 20
    clusters in 64 dimensions, measured 1.7 off before — and where the true
    kernel is the identity matrix, the dual is LIBSVM's exactly.
    """
    rng = np.random.default_rng(0)
    near = rng.standard_normal((300, 3))
    far = 2000.0 + rng.standard_normal((300, 3))
    X = np.r_[near, far]
    y = np.r_[near[:, 0] * near[:, 1] > 0, (far[:, 0] - 2000) * (far[:, 1] - 2000) > 0]
    mlx = _fit(backend, X, y.astype(int), gamma=1.0)
    _assert_svc_parity(mlx, SkSVC(gamma=1.0).fit(X, y.astype(int)), X)

    centers = rng.uniform(-100.0, 100.0, size=(20, 64))
    cluster = rng.integers(0, 20, 1200)
    X = centers[cluster] + 0.05 * rng.standard_normal((1200, 64))
    y = (cluster % 2) ^ (rng.random(1200) < 0.2)
    gamma = 1.0 / (2 * 64 * 0.05**2)
    mlx = _fit(backend, X, y, gamma=gamma)
    _assert_svc_parity(mlx, SkSVC(gamma=gamma).fit(X, y), X)

    X = rng.uniform(size=(100, 30)) * 1e10  # every pair is ~1e10 apart: K = I
    y = np.r_[np.ones(50, int), np.zeros(50, int)]
    mlx = _fit(backend, X, y, C=100.0, gamma="auto")
    ref = SkSVC(C=100.0, gamma="auto").fit(X, y)
    np.testing.assert_array_equal(mlx.dual_coef_, ref.dual_coef_)
    np.testing.assert_array_equal(mlx.predict(X), ref.predict(X))


@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_and_inference_raise_no_floating_point_warnings(backend, monkeypatch):
    """NumPy 2.2's BLAS on macOS left floating-point flags set on finite products, and
    the rbf refinement, the host's contraction of each device tile and the host tiles
    of a small batch warned on every fit and every call.

    The clusters of the test above, whose entries are refined at fit and at
    inference alike; a count, not a record of the tiles, says the refinement ran.
    """
    import mlxlearn._kernels.kernels as kernels_module

    refine = kernels_module._RbfRefiner.__call__
    refinements = 0

    def counted(self, tile, left):
        nonlocal refinements
        refinements += 1
        return refine(self, tile, left)

    monkeypatch.setattr(kernels_module._RbfRefiner, "__call__", counted)
    rng = np.random.default_rng(0)
    centers = rng.uniform(-100.0, 100.0, size=(20, 64))
    cluster = rng.integers(0, 20, 1200)
    X = centers[cluster] + 0.05 * rng.standard_normal((1200, 64))
    y = (cluster % 2) ^ (rng.random(1200) < 0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = _fit(backend, X, y, gamma=1.0 / (2 * 64 * 0.05**2))
        assert refinements > 0
        # Device tiles, then the host's: svc_host_inference_work picks the path.
        for work, compute in ((0, None), (2**62, "numpy-float32")):
            refinements = 0
            with tuning_context(svc_host_inference_work=work):
                model.predict(X[:300])
                model.decision_function(X[:300])
            assert get_backend_diagnostics()[-1].extra.get("compute") == compute
            assert refinements > 0


POLY_OFFSET = {"kernel": "poly", "degree": 3, "coef0": 1.0, "gamma": "auto", "max_iter": 20_000}


def _offset_data():
    """The reported case: an offset of 20 with a spread of about 0.01."""
    X, y = make_classification(400, 10, random_state=0)
    return X * 0.01 + 20.0, y


def test_poly_on_data_float32_cannot_resolve_is_a_capability_error():
    """The poly kernel reads the raw inner product, which the frame cannot centre.

    Its float32 error, about 40 near 6.5e7, exceeded the curvature the solver
    learns from (3–100): with the default ``max_iter`` the fit circled 10⁷
    iterations (376 s against scikit-learn's 0.001 s) and returned a model 13
    points less accurate; at a spread of 0.03 it converged, silently, to one 12
    points worse. Layer 1 now refuses it with a precise reason, and patched mode
    hands it to scikit-learn.
    """
    X, y = _offset_data()
    with pytest.raises(UnsupportedInputError) as excinfo:
        SVC(**POLY_OFFSET).fit(X, y)
    assert excinfo.value.reason == "float32-cancellation"
    for spread in (0.03, 0.1):
        with pytest.raises(UnsupportedInputError):
            SVC(**POLY_OFFSET).fit((X - 20.0) * spread / 0.01 + 20.0, y)

    with tuning_context(force_backend="mlx"):
        patched = PatchedSVC(**POLY_OFFSET).fit(X, y)
    assert patched.execution_backend_ == "sklearn"
    np.testing.assert_array_equal(patched.predict(X), SkSVC(**POLY_OFFSET).fit(X, y).predict(X))


def test_poly_self_entries_keep_the_solver_from_circling(monkeypatch):
    """Even on data the precision check refuses, the solver must not spin.

    It took its curvature from the float64 closed-form diagonal and its gradient
    updates from float32 rows whose self-entries differed from it by up to 31:
    each step solved one quadratic and applied another, and the fit ran its whole
    budget without progress. With the self-entries written from the closed form,
    the same kernel converges in a few hundred iterations.
    """
    import mlxlearn.svm._classes as svc_module

    monkeypatch.setattr(svc_module, "check_frame_resolution", lambda spec, X32: None)
    X, y = _offset_data()
    for backend in BACKENDS:
        model = _fit(backend, X, y, **POLY_OFFSET)
        assert model.fit_status_ == 0
        assert model.n_iter_.max() < 2_000


def test_gamma_that_float32_cannot_represent_is_a_capability_error():
    """Folded into the frame, gamma=1e40 is still ~1e42: not a float32.

    It used to reach the device as ``inf`` and produce a NaN model. scikit-learn
    evaluates the kernel in float64 and serves it, so patched mode falls back.
    """
    X, y = _data(n_samples=300, n_features=6, seed=3)
    with pytest.raises(UnsupportedParameterError) as excinfo:
        SVC(gamma=1e40).fit(X, y)
    assert excinfo.value.parameter == "gamma"

    with tuning_context(force_backend="mlx"):
        patched = PatchedSVC(gamma=1e40).fit(X, y)
    assert patched.execution_backend_ == "sklearn"
    np.testing.assert_array_equal(patched.predict(X), SkSVC(gamma=1e40).fit(X, y).predict(X))


def test_non_finite_model_raises_like_scikit_learn():
    """A kernel that overflows float32 must not return a NaN model.

    poly with ``gamma=1`` on 1e8-scale data has kernel values near 1e51. The solver
    saw NaN gradients, found no eligible working pair and reported convergence
    after one iteration; the fit returned ``intercept_=[nan]`` with
    ``fit_status_=0``. LIBSVM caches kernel rows in float too, and scikit-learn
    raises — mlxlearn now raises the same error.
    """
    X, y = _data(n_samples=200, n_features=4, seed=0)
    X = X * 1e8
    with pytest.raises(ValueError, match="dual coefficients or intercepts are not finite"):
        SkSVC(kernel="poly", gamma=1.0).fit(X, y)
    model = SVC(kernel="poly", gamma=1.0)
    with pytest.raises(ValueError, match="dual coefficients or intercepts are not finite"):
        model.fit(X, y)
    assert not hasattr(model, "support_")


def test_linear_kernel_overflow_is_a_capability_error():
    """The linear kernel has no gamma to fold a scale into, so it is range-guarded.

    At 1e20 its inner products overflow float32. LIBSVM fails on the same input,
    so the precise error in Layer 1 and scikit-learn's own error in Layer 2 are
    both the right outcome — a NaN model is not.
    """
    X, y = _data(n_samples=200, n_features=4, seed=0)
    X = X * 1e20
    with pytest.raises(UnsupportedInputError) as excinfo:
        SVC(kernel="linear").fit(X, y)
    assert excinfo.value.reason == "float32-overflow"

    with tuning_context(force_backend="cpu"):
        with pytest.raises(ValueError, match="not finite") as excinfo:
            PatchedSVC(kernel="linear").fit(X, y)
    # scikit-learn's own error — a plain ValueError — so the range guard handed the
    # fit over rather than the MLX path failing (its error is an InvalidInputError).
    # Both fits failed, and both decisions are on the record: Layer 1's refusal,
    # then the patched handoff.
    assert type(excinfo.value) is ValueError
    assert [(e.backend, e.reason, e.fallback) for e in get_backend_diagnostics()] == [
        ("cpu", "float32-overflow", False),
        ("sklearn", "float32-overflow", True),
    ]


def test_linear_kernel_accepts_gamma_zero():
    """The linear kernel never reads gamma; scikit-learn stores 0.0 and fits.

    The capability check exempted linear from the gamma > 0 rule, but the fit
    resolved gamma unconditionally and raised anyway — in Layer 1 always, and in
    patched mode whenever a backend was forced.
    """
    X, y = _data(n_classes=3, n_samples=300, n_features=8, seed=3)
    mlx = SVC(kernel="linear", gamma=0.0).fit(X, y)
    ref = SkSVC(kernel="linear", gamma=0.0).fit(X, y)
    assert mlx._gamma == ref._gamma == 0.0
    _assert_svc_parity(mlx, ref, X)

    with tuning_context(force_backend="cpu"):
        patched = PatchedSVC(kernel="linear", gamma=0.0).fit(X, y)
    assert patched.execution_backend_ == "cpu"


@pytest.mark.parametrize("kernel, params", [
    ("poly", {"degree": 2, "coef0": 1.5}),
    ("poly", {"degree": 4, "coef0": -0.5, "gamma": 0.05}),
    ("sigmoid", {"coef0": -1.0, "gamma": 0.01}),
])
def test_non_default_degree_and_coef0(kernel, params):
    """``coef0`` was only ever exercised at 0.0, so dropping it would go unnoticed."""
    X, y = _data(n_samples=800, n_features=10, seed=6)
    mlx = SVC(kernel=kernel, **params).fit(X, y)
    ref = SkSVC(kernel=kernel, **params).fit(X, y)
    _assert_svc_parity(mlx, ref, X)
    # Which point sits exactly on the margin is ill-conditioned (module docstring).
    assert np.abs(mlx.n_support_ - ref.n_support_).max() <= 1


def test_break_ties_parity():
    X, y = _data(n_classes=3, n_samples=600, n_features=10, seed=9)
    mlx = SVC(break_ties=True).fit(X, y)
    ref = SkSVC(break_ties=True).fit(X, y)
    assert np.mean(mlx.predict(X) == ref.predict(X)) >= 0.99

    for model in (SVC(break_ties=True, decision_function_shape="ovo"),
                  SkSVC(break_ties=True, decision_function_shape="ovo")):
        model.fit(X, y)
        with pytest.raises(ValueError, match="break_ties must be False"):
            model.predict(X)


@pytest.mark.parametrize("class_weight", ["balanced", {0: 1.0, 1: 5.0}])
@pytest.mark.parametrize("backend", BACKENDS)
def test_class_weight_parity(class_weight, backend):
    X, y = _data(n_samples=1600, seed=7)
    mlx = _fit(backend, X, y, kernel="rbf", class_weight=class_weight)
    ref = SkSVC(kernel="rbf", class_weight=class_weight).fit(X, y)
    _assert_svc_parity(mlx, ref, X)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_weight_parity(backend):
    X, y = _data(n_samples=1600, seed=8)
    weights = np.where(y == 1, 3.0, 1.0)
    mlx = _fit(backend, X, y, sample_weight=weights, kernel="rbf")
    ref = SkSVC(kernel="rbf").fit(X, y, sample_weight=weights)
    _assert_svc_parity(mlx, ref, X)


def test_zero_weight_rows_are_dropped_and_support_indexes_the_callers_rows():
    """The one deliberate ``support_`` deviation, which the docstring promises.

    scikit-learn reports indices into its *filtered* problem, so for it
    ``support_vectors_ != X[support_]``; mlxlearn's indices point at the caller's
    rows, and never at a dropped one.
    """
    X, y = _data(n_samples=500, n_features=8, seed=12)
    weights = np.ones(len(y))
    weights[::7] = 0.0
    mlx = SVC().fit(X, y, sample_weight=weights)
    ref = SkSVC().fit(X, y, sample_weight=weights)

    np.testing.assert_array_equal(mlx.support_vectors_, X[mlx.support_])
    assert not np.any(weights[mlx.support_] == 0.0)
    np.testing.assert_array_equal(mlx.n_support_, ref.n_support_)
    _assert_svc_parity_without_support_sets(mlx, ref, X)


def _assert_svc_parity_without_support_sets(mlx, ref, X):
    assert np.mean(mlx.predict(X) == ref.predict(X)) >= 0.99
    np.testing.assert_allclose(
        mlx.decision_function(X), ref.decision_function(X),
        atol=DECISION_ATOL, rtol=DECISION_RTOL,
    )


def test_negative_sample_weight_is_a_zero_weight():
    """LIBSVM drops every row whose weight is not positive, negative ones included.

    So stock scikit-learn returns the *same* model for a negative weight as for a
    zero one. mlxlearn rejected them with a CapabilityError raised after backend
    selection, where patched mode could not act on it: ``patch_sklearn()`` turned
    code stock scikit-learn runs into a crash, at every size and every kernel.
    """
    X, y = _data(n_samples=400, n_features=8, seed=13)
    weights = np.ones(len(y))
    weights[::9] = -1.0
    weights[5] = -0.5
    zeroed = np.clip(weights, 0.0, None)

    negative = SVC().fit(X, y, sample_weight=weights)
    zero = SVC().fit(X, y, sample_weight=zeroed)
    np.testing.assert_array_equal(negative.decision_function(X), zero.decision_function(X))
    _assert_svc_parity_without_support_sets(negative, SkSVC().fit(X, y, sample_weight=weights), X)

    assert PatchedSVC().fit(X, y, sample_weight=weights).execution_backend_ == "sklearn"
    with tuning_context(force_backend="mlx"):
        assert PatchedSVC().fit(X, y, sample_weight=weights).execution_backend_ == "mlx"


def test_a_class_with_no_positive_weight_is_handed_to_scikit_learn():
    """A class whose every sample is dropped is a capability limit, not bad input.

    scikit-learn's LIBSVM drops the rows and fits the remaining classes, into a
    model whose ``decision_function`` then fails. mlxlearn builds no such model,
    but it raised ``InvalidInputError`` from inside the fit, where patched mode
    could not act on it: ``patch_sklearn()`` turned code stock scikit-learn runs
    into a crash above the crossover. It is now found before any device work, so
    Layer 1 raises precisely and patched mode falls back.
    """
    X, y = _data(n_classes=3, n_samples=600, n_features=8, seed=3)
    weights = np.where(y == 2, 0.0, 1.0)
    with pytest.raises(UnsupportedInputError) as excinfo:
        SVC().fit(X, y, sample_weight=weights)
    assert excinfo.value.reason == "empty-class"

    with tuning_context(force_backend="mlx"):
        patched = PatchedSVC().fit(X, y, sample_weight=weights)
    assert patched.execution_backend_ == "sklearn"
    stock = SkSVC().fit(X, y, sample_weight=weights)
    np.testing.assert_array_equal(patched.predict(X), stock.predict(X))


def test_sample_weight_equivalence_at_float32_tolerance():
    """scikit-learn's own float64 equivalence check is an inherited exemption.

    scikit-learn declares ``check_sample_weight_equivalence_on_{dense,sparse}_data``
    an expected failure for its own ``SVC``, and mlxlearn inherits that exemption
    (``INHERITED_FAILURES`` in ``tests/test_sklearn_compliance.py``; fp32_policy
    §4) — it is not one of mlxlearn's own xfails, of which there are none. The
    property is real and is asserted here at the tolerance a float32 SMO stopping
    at ``tol=1e-3`` supports, so the exemption does not mean the behavior goes
    untested.
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


@pytest.mark.parametrize("kernel", NONLINEAR)
def test_backends_agree(kernel):
    """The MLX CPU and GPU devices run the same code and must agree closely."""
    X, y = _data(n_samples=1200)
    on_cpu = _fit("cpu", X, y, kernel=kernel)
    on_gpu = _fit("mlx", X, y, kernel=kernel)
    np.testing.assert_allclose(
        on_cpu.decision_function(X), on_gpu.decision_function(X), atol=5e-3
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
    assert mlx.fit_status_ == 0


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


def _future_warnings(fit):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit()
    return [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]


@pytest.mark.skipif(
    SkSVC().get_params()["probability"] != "deprecated",
    reason="scikit-learn deprecated `probability` in 1.9",
)
def test_probability_deprecation_warning_matches_scikit_learn():
    """The MLX path used to swallow scikit-learn's advance notice of a removal.

    Layer 1 never hands a fit to scikit-learn, so its users never saw it at all.
    Patched mode must still warn exactly once when scikit-learn does the fit.
    """
    X, y = _data(n_samples=300, n_features=6)
    stock = _future_warnings(lambda: SkSVC(probability=False).fit(X, y))
    ours = _future_warnings(lambda: SVC(probability=False).fit(X, y))
    assert len(stock) == len(ours) == 1
    assert "probability" in ours[0]
    assert _future_warnings(lambda: SVC().fit(X, y)) == []
    assert len(_future_warnings(lambda: PatchedSVC(probability=False).fit(X, y))) == 1


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


@pytest.mark.parametrize("backend", BACKENDS)
def test_sparse_query_on_a_dense_fit_raises_scikit_learns_error(backend):
    """scikit-learn raises ValueError for this, not the TypeError validation gave."""
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _data(n_samples=300)
    model = _fit(backend, X, y)
    for method in (model.predict, model.decision_function):
        with pytest.raises(ValueError, match="cannot use sparse input in 'SVC' trained on dense"):
            method(sparse.csr_matrix(X[:5]))


def test_max_iter_zero_is_a_budget_not_a_sentinel():
    """``max_iter=0`` means zero iterations, exactly as it does in scikit-learn.

    It resolved to the *unlimited* branch instead: a user who asked for zero
    iterations got 462 and a fully converged model, silently, and under
    ``patch_sklearn()`` that reached code written against scikit-learn's
    behavior. Only ``-1`` means "no limit".
    """
    X, y = _data(n_samples=2500, n_features=6, seed=3)

    with pytest.warns(ConvergenceWarning):
        mlx = SVC(max_iter=0).fit(X, y)
    with pytest.warns(ConvergenceWarning):
        ref = SkSVC(max_iter=0).fit(X, y)

    assert mlx.n_iter_.tolist() == ref.n_iter_.tolist() == [0]
    assert len(mlx.support_) == len(ref.support_) == 0
    np.testing.assert_array_equal(mlx.predict(X[:100]), ref.predict(X[:100]))


def test_fit_status_and_convergence_warning_match_scikit_learn():
    """``fit_status_=1`` comes with a ConvergenceWarning, in both libraries."""
    X, y = _data(n_samples=400, n_features=8, seed=0)
    with pytest.warns(ConvergenceWarning, match=r"Solver terminated early \(max_iter=5\)"):
        mlx = SVC(max_iter=5).fit(X, y)
    with pytest.warns(ConvergenceWarning, match=r"Solver terminated early \(max_iter=5\)"):
        ref = SkSVC(max_iter=5).fit(X, y)
    assert mlx.fit_status_ == ref.fit_status_ == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        assert SVC().fit(X, y).fit_status_ == 0


def test_max_iter_minus_one_names_the_budget_it_resolved_to():
    """``-1`` is capped in mlxlearn, unlimited in scikit-learn, and the warning says so.

    It used to read "Solver terminated early (max_iter=-1)" — self-contradictory,
    and with no hint that a larger explicit ``max_iter`` would keep going.
    """
    X, y = _data(n_samples=400, n_features=8, seed=0)
    with tuning_context(smo_min_iter_budget=50, smo_max_iter_factor=0):
        with pytest.warns(ConvergenceWarning, match=r"Solver terminated early \(max_iter=-1") as rec:
            model = SVC().fit(X, y)
    [message] = [str(w.message) for w in rec if issubclass(w.category, ConvergenceWarning)]
    assert "budget of 50 iterations" in message
    assert "larger explicit max_iter" in message
    assert model.fit_status_ == 1
    assert model.n_iter_.tolist() == [50]


# ----------------------------------------------------------------- dispatch


def test_gamma_zero_is_a_capability_error_so_layer2_can_fall_back():
    """scikit-learn accepts ``gamma=0.0``; mlxlearn's kernels do not.

    Because scikit-learn *can* serve the request, this has to be a
    CapabilityError discovered during the capability check — not a plain
    ``ValueError`` raised deep inside kernel evaluation, which propagated
    straight out of Layer 2 as well and left patched users with a hard failure
    where stock scikit-learn would have fitted.
    """
    X, y = _data(n_samples=1200, n_features=6, seed=3)

    with pytest.raises(UnsupportedParameterError) as excinfo:
        SVC(gamma=0.0).fit(X, y)
    assert excinfo.value.parameter == "gamma"

    patched = PatchedSVC(gamma=0.0).fit(X, y)
    assert patched.execution_backend_ == "sklearn"
    np.testing.assert_array_equal(
        patched.predict(X[:100]), SkSVC(gamma=0.0).fit(X, y).predict(X[:100])
    )


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


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["linear", "sigmoid"])
def test_linear_and_sigmoid_kernels_never_take_the_mlx_path(kernel):
    """Measured slower on MLX at sizes where rbf and poly win, so they are excluded.

    The linear kernel ran 0.07×–0.10× forced onto MLX (2 000 × 20 and × 256): SMO
    is sequential, and its iterations are the cheapest of all, so per-iteration
    overhead is the entire cost. The sigmoid kernel lost at C = 10 and 100 from 128
    features, and at C = 1 at 1 000 × 256, while it won at C = 0.1; the rule cannot
    see C. A threshold cannot fix either. The exclusion does not depend on size, so
    it is checked with every size floor at zero — otherwise it would be the floor,
    not the exclusion, routing the fit. The rbf half keeps the real thresholds: it
    is the one end-to-end check that they send a measured win to MLX.
    """
    X_small, y_small = _data(n_samples=300, n_features=8, seed=1)
    no_floor = {
        "svc_min_features": 0, "svc_min_samples": 0, "svc_min_work": 0,
        "svc_min_pair_work": 0,
    }
    with tuning_context(**no_floor):
        patched = PatchedSVC(kernel=kernel).fit(X_small, y_small)
        assert patched.execution_backend_ == "sklearn", (
            f"the {kernel} kernel reached the MLX path"
        )
        assert f"the {kernel} kernel" in patched.fit_event_.detail
        assert PatchedSVC(kernel="rbf").fit(X_small, y_small).execution_backend_ == "mlx"

    X, y = _data(n_samples=4_000, n_features=256, seed=1)
    rbf = PatchedSVC(kernel="rbf").fit(X, y)
    assert rbf.execution_backend_ == "mlx", "rbf above the crossover should use MLX"


def test_the_svc_crossover_is_above_the_measured_losses():
    """A threshold below the measured loss region is not protection.

    Width decides: per SMO iteration mlxlearn's cost is nearly flat in the number
    of features and LIBSVM's grows with it. Measured on scikit-learn 1.9.0 and
    1.7.2 (docs/benchmarks.md, the ``svc``, ``svc-hard`` and ``svc-narrow``
    tables), rbf and poly at C = 10 lost at 4 000 x 16..64, and past the kernel
    cache narrow data lost even at C = 1. At 128 features the benchmark's target won
    from 3 000 rows, but labels with little or no signal — where nearly every row
    becomes a support vector — lost at C = 10 and 100 up to 6 000 rows, and
    one-vs-one pairs of 3 000 x 128 did too; from 256 features every one of them
    won, from 1 500 x 256 and 1 024 x 512. The first release's work floor,
    ``n·d >= 10⁶``, could not tell 16 000 x 64 from 4 000 x 256.
    """
    model = SVC()
    losses = (
        (4_000, 16), (4_000, 64), (16_000, 8), (16_000, 64), (2_500, 20),
        (3_000, 128), (6_000, 128), (2_000, 128),
    )
    for n_samples, n_features in losses:
        below, detail = model._below_crossover(n_samples, n_features)
        assert below, f"{n_samples}x{n_features} was measured slower but is above the crossover"
        assert "256 features" in detail
    # Measured wins at every kernel, C and label law tried, and the smallest ones kept.
    for n_samples, n_features in ((1_500, 256), (1_024, 512), (4_000, 256)):
        assert not model._below_crossover(n_samples, n_features)[0]
    # 1000 x 256 won by only 1.04x (poly, C=10, on 1.7.2); the work floor leaves it.
    below, detail = model._below_crossover(1_000, 256)
    assert below and "384000 element-ops" in detail


def test_the_svc_crossover_accounts_for_the_number_of_classes():
    """One-vs-one pays per pair, so many small classes lose despite a large total.

    The mean pair must clear the binary work floor. On balanced separated blobs
    (rbf, C = 1) pairs of 256 x 512 (2048 x 512, K=16) and 500 x 256 (8000 x 256,
    K=32) lost, and on random labels at C = 10 and 100 pairs of 1 500 x 256 and
    1 024 x 512 won. The total size clears every floor in each case, so only the
    class count separates them.
    """
    model = SVC()
    for n_classes in (2, 4):
        assert not model._below_crossover(2048, 512, n_classes=n_classes)[0]
    for n_classes in (8, 16, 64, 128):
        below, detail = model._below_crossover(2048, 512, n_classes=n_classes)
        assert below and f"{n_classes} classes" in detail
    assert not model._below_crossover(8000, 256, n_classes=8)[0]
    assert model._below_crossover(8000, 256, n_classes=16)[0]
    # Pairs of 1 500 x 256 against 1 000 x 256; pairs of 3 000 x 128 are below the
    # width floor, as the binary fits of that size are.
    assert not model._below_crossover(2250, 256, n_classes=3)[0]
    assert model._below_crossover(1500, 256, n_classes=3)[0]
    assert model._below_crossover(4500, 128, n_classes=3)[0]


@pytest.mark.gpu
def test_patched_dispatch_sends_only_rbf_with_a_box_of_one_and_the_default_cache():
    """How many SMO iterations a fit takes per row decides it, and C lets them
    multiply: on data of low intrinsic dimension rbf fits lost at C = 10 whatever the
    width, and won at C <= 1 at every gamma measured. So a patched fit needs a mean
    box, C x class weight x sample weight, of at most 1. The poly kernel lost even at
    C <= 1 on data outside the Gaussian grids and goes to LIBSVM whatever its
    parameters, as linear and sigmoid do; so does a cache_size below scikit-learn's
    default. Patched only: Layer 1 keeps them on MLX."""
    X, y = _data(n_samples=300, n_features=8, seed=1)
    tuning = get_tuning()
    above = 10.0 * tuning.svc_max_c
    no_floor = {
        "svc_min_features": 0, "svc_min_samples": 0, "svc_min_work": 0,
        "svc_min_pair_work": 0,
    }
    scale = 1.0 / (X.shape[1] * X.var())
    cases = [
        ({"C": above}, True, "box constraint"),
        ({"C": tuning.svc_max_c}, False, ""),
        ({"C": tuning.svc_max_c, "gamma": 0.01 * scale}, False, ""),
        ({"C": tuning.svc_max_c, "gamma": 1_000.0 * scale}, False, ""),
        ({"kernel": "poly"}, True, "only the rbf kernel"),
        ({"kernel": "poly", "C": 0.1}, True, "only the rbf kernel"),
        ({"kernel": "poly", "degree": 2, "coef0": 1.0}, True, "only the rbf kernel"),
        ({"kernel": "poly", "C": above}, True, "only the rbf kernel"),
        ({"kernel": "sigmoid"}, True, "always handed to scikit-learn"),
        ({"kernel": "linear"}, True, "always handed to scikit-learn"),
        ({"cache_size": 1.0}, True, "cache_size=1 is below"),
        ({"cache_size": 199.0}, True, "cache_size=199 is below"),
        ({"cache_size": 200.0}, False, ""),
        ({"cache_size": 1_000.0}, False, ""),
        # Balanced class weights average to one, so the mean box is C.
        ({"C": tuning.svc_max_c, "class_weight": "balanced"}, False, ""),
        ({"C": tuning.svc_max_c, "class_weight": {0: 1.0, 1: 5.0}}, True, "mean class and"),
    ]
    with tuning_context(**no_floor):
        for params, handed, why in cases:
            model = PatchedSVC(**params).fit(X, y)
            assert (model.execution_backend_ == "sklearn") == handed, params
            if handed:
                assert model.fit_event_.reason == "below-crossover"
                assert why in model.fit_event_.detail, (params, model.fit_event_.detail)
                np.testing.assert_array_equal(
                    model.predict(X), SkSVC(**params).fit(X, y).predict(X)
                )
        # Sample weights scale the box as C does: all 5 is C = 5, all 0.05 at C = 10 is
        # C = 0.5, and a zero weight drops its sample rather than lowering the mean.
        fives = np.full(len(y), 5.0)
        assert PatchedSVC().fit(X, y, sample_weight=fives).execution_backend_ == "sklearn"
        halves = np.where(np.arange(len(y)) % 2 == 0, 0.0, 1.0)
        assert PatchedSVC().fit(X, y, sample_weight=halves).execution_backend_ != "sklearn"
        small = np.full(len(y), 0.05)
        assert PatchedSVC(C=above).fit(X, y, sample_weight=small).execution_backend_ != "sklearn"
        # Layer 1 has no LIBSVM to hand anything to: poly still runs, on the GPU.
        assert SVC(C=above).fit(X, y).execution_backend_ == "mlx"
        assert SVC(kernel="poly").fit(X, y).execution_backend_ == "mlx"
        assert SVC(kernel="poly", degree=2).fit(X, y).execution_backend_ == "mlx"
        assert SVC(cache_size=1.0).fit(X, y).execution_backend_ == "mlx"


@pytest.mark.gpu
@pytest.mark.parametrize("weighting", ["balanced", "lognormal"])
def test_weights_that_average_to_one_leave_the_box_at_c_despite_rounding(weighting):
    """Balanced class weights, and sample weights normalized to mean 1, average to
    one only up to rounding: at 2% positives the balanced weights' mean over the
    samples is 1.0000000000000002 in float64, and so is the mean of these lognormal
    weights. The mean box they give at C = 1 must still read as C, so the fit stays
    on MLX; a strict comparison sent it to LIBSVM."""
    from sklearn.utils.class_weight import compute_class_weight

    rng = np.random.default_rng(37)
    n = 1_000
    X = rng.normal(size=(n, 8))
    y = (rng.random(n) < 0.02).astype(int)
    assert 0.01 < y.mean() < 0.03
    weights = rng.lognormal(0.0, 1.0, size=n)
    weights = weights / weights.mean()
    if weighting == "balanced":
        params, fit_kw = {"class_weight": "balanced"}, {}
        naive = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)[y]
    else:
        params, fit_kw = {}, {"sample_weight": weights}
        naive = weights
    # The precondition that makes this a test of rounding: the plain mean is above 1.
    assert float(np.mean(1.0 * naive)) > 1.0
    no_floor = {"svc_min_features": 0, "svc_min_samples": 0, "svc_min_work": 0}
    with tuning_context(**no_floor):
        model = PatchedSVC(C=1.0, **params).fit(X, y, **fit_kw)
        assert model.execution_backend_ != "sklearn", model.fit_event_
        assert model.fit_event_.reason == "selected"
        assert model._kernel_handoff(model._mean_box(np.asarray(y), fit_kw.get("sample_weight"))) is None
        # A box measurably above one still goes to LIBSVM.
        heavier = PatchedSVC(C=1.0 + 1e-6, **params).fit(X, y, **fit_kw)
        assert heavier.execution_backend_ == "sklearn"
        assert "box constraint" in heavier.fit_event_.detail


@pytest.mark.parametrize(
    "params",
    [
        {"kernel": "poly"},
        {"kernel": "poly", "degree": 2},
        {"C": 10.0},
        {"C": 10.0, "class_weight": "balanced"},
        {"cache_size": 100.0},
    ],
)
def test_the_kernel_c_and_cache_rules_are_decided_before_the_data_are_read(params, monkeypatch):
    """They need only the parameters, so a handed-off fit pays for no validation."""
    import mlxlearn.svm._classes as svc_module

    def forbidden(*args, **kwargs):
        raise AssertionError("a handed-off patched fit scanned the data")

    monkeypatch.setattr(svc_module, "validate_data", forbidden)
    monkeypatch.setattr(svc_module, "check_float32_range", forbidden)
    X, y = _data(n_samples=300, n_features=8, seed=1)
    with tuning_context(svc_min_features=0, svc_min_samples=0, svc_min_work=0):
        model = PatchedSVC(**params).fit(X, y)
    assert model.execution_backend_ == "sklearn"


@pytest.mark.gpu
def test_patched_dispatch_uses_the_class_count():
    X, y2 = _data(n_samples=400, n_features=8, seed=2)
    y8 = np.arange(len(X)) % 8
    # Mean pair: 400 x 8 = 3200 for two classes, 100 x 8 = 800 for eight.
    with tuning_context(
        svc_min_features=0, svc_min_samples=0, svc_min_work=0, svc_min_pair_work=1_000
    ):
        assert PatchedSVC().fit(X, y2).execution_backend_ == "mlx"
        assert PatchedSVC().fit(X, y8).execution_backend_ == "sklearn"
        # Layer 1 keeps the fit, on the MLX CPU device, as for any small problem.
        assert SVC().fit(X, y8).execution_backend_ == "cpu"


def test_patched_fit_below_the_crossover_skips_the_mlx_preamble(monkeypatch):
    """Work patched mode hands to scikit-learn must not be validated twice.

    Everything below the crossover used to be validated, range-scanned and
    target-checked by mlxlearn first, only for scikit-learn to repeat it —
    measured 1.4x-1.6x stock at 200 × 5. It is now decided from the raw shape,
    after only the parameter and sparsity checks, and recorded as one event.
    """
    import mlxlearn.svm._classes as svc_module

    def forbidden(*args, **kwargs):
        raise AssertionError("a below-crossover patched fit scanned the data")

    monkeypatch.setattr(svc_module, "validate_data", forbidden)
    monkeypatch.setattr(svc_module, "check_float32_range", forbidden)
    monkeypatch.setattr(svc_module, "validate_float32_input", forbidden)

    X, y = _data(n_samples=200, n_features=5, seed=0)
    patched = PatchedSVC().fit(X, y)
    assert patched.execution_backend_ == "sklearn"
    np.testing.assert_array_equal(patched.predict(X), SkSVC().fit(X, y).predict(X))

    fit_events = [e for e in get_backend_diagnostics() if e.operation == "fit"]
    assert [(e.backend, e.reason) for e in fit_events] == [("sklearn", "below-crossover")]

    # A parameter mismatch is still reported as one, not as a size decision.
    with config_context(fallback_policy="raise"):
        with pytest.raises(CapabilityError, match="precomputed"):
            PatchedSVC(kernel="precomputed").fit(X @ X.T, y)


@pytest.mark.parametrize("spelling", ["int", "object"])
def test_patched_many_class_handoff_skips_the_mlx_preamble(monkeypatch, spelling):
    """The many-class rule is decided from the raw labels, before any scan of X.

    It needs the class count, so it was applied only after mlxlearn had copied X
    to float64 and scanned it — work scikit-learn then repeats, on a fit patched
    mode hands it by design. Labels given as Python objects (class names in a
    pandas ``Series``, say) are counted too; they used to wait for validation.
    """
    import mlxlearn.svm._classes as svc_module

    def forbidden(*args, **kwargs):
        raise AssertionError("a below-crossover patched fit scanned the data")

    monkeypatch.setattr(svc_module, "validate_data", forbidden)
    monkeypatch.setattr(svc_module, "check_float32_range", forbidden)

    X, _ = _data(n_samples=400, n_features=8, seed=2)
    y8 = np.arange(len(X)) % 8
    if spelling == "object":
        y8 = np.array([f"class {k}" for k in y8], dtype=object)
    with tuning_context(
        svc_min_features=0, svc_min_samples=0, svc_min_work=0, svc_min_pair_work=1_000
    ):
        patched = PatchedSVC().fit(X.astype(np.float32), y8)
    assert patched.execution_backend_ == "sklearn"
    [event] = [e for e in get_backend_diagnostics() if e.operation == "fit"]
    assert event.reason == "below-crossover" and "8 classes" in event.detail


def test_patched_probability_calls_record_one_event_each():
    """``probability=True`` is scikit-learn's to fit; its inference still records.

    scikit-learn's ``predict_proba`` and ``predict_log_proba`` are ``available_if``
    descriptors, which mlxlearn leaves unwrapped so that they keep not existing
    when ``probability=False``, and so they recorded no event at all — though
    every other public inference call records one. ``predict_log_proba`` calls
    ``predict_proba``, and still records one.
    """
    X, y = _data(n_samples=300, n_features=6, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # deprecated in scikit-learn 1.9
        patched = PatchedSVC(probability=True, random_state=0).fit(X, y)
        stock = SkSVC(probability=True, random_state=0).fit(X, y)
    assert patched.execution_backend_ == "sklearn"
    for method in ("predict_proba", "predict_log_proba"):
        before = len(get_backend_diagnostics())
        getattr(patched, method)(X[:5])
        events = get_backend_diagnostics()[before:]
        assert [(e.operation, e.reason) for e in events] == [(method, "sticky")]
    np.testing.assert_array_equal(patched.predict_proba(X[:5]), stock.predict_proba(X[:5]))
    for model in (SVC(), PatchedSVC(), SVC().fit(X, y)):
        assert not hasattr(model, "predict_proba")
        assert not hasattr(model, "predict_log_proba")


def test_refit_on_scikit_learn_clears_the_mlx_state():
    """A refit that falls back must not keep the previous fit's MLX state around.

    ``_clear_backend`` deleted a ``_mlx_state_`` nothing ever set, so the pair
    weights, the kernel frame and the device-resident support vectors of the
    previous fit survived in memory and in every pickle of the new one.
    """
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _data(n_samples=600, n_features=10, seed=0)
    patched = PatchedSVC()
    with tuning_context(force_backend="mlx"):
        patched.fit(X, y)
        patched.predict(X[:5])  # builds the device copy of the support vectors
    assert all(hasattr(patched, name) for name in MLX_ATTRS)

    patched.fit(sparse.csr_matrix(X[:200]), y[:200])
    assert patched.execution_backend_ == "sklearn"
    assert [name for name in MLX_ATTRS if hasattr(patched, name)] == []


# ----------------------------------------------------------------- inference


@pytest.mark.parametrize("value", [1e39, -1e300])
@pytest.mark.parametrize("layer", ["layer1", "patched"])
def test_out_of_range_queries_raise_instead_of_flipping_labels(value, layer):
    """fp32_policy §1: a value float32 cannot hold is rejected, not made infinite.

    Only ``fit`` checked. A 1e39 query became ``inf`` in float32, its decision
    values NaN, and the one-vs-one vote — for which ``NaN > 0`` is false — sent it
    silently to each pair's second class. Inference cannot fall back (the backend
    is sticky), so both layers raise.
    """
    X, y = _data(n_samples=300, n_features=4, seed=0)
    if layer == "layer1":
        model = SVC().fit(X, y)
    else:
        with tuning_context(force_backend="mlx"):
            model = PatchedSVC().fit(X, y)
    query = X[:4].copy()
    query[:, 0] = value
    for method in (model.predict, model.decision_function):
        with pytest.raises(UnsupportedInputError) as excinfo:
            method(query)
        assert excinfo.value.reason == "float32-overflow"


@pytest.mark.parametrize("kernel", ["rbf", "sigmoid", "linear"])
@pytest.mark.parametrize("layer", ["layer1", "patched"])
def test_far_queries_are_answered_like_scikit_learn(layer, kernel):
    """A CMIP ``1e20`` or netCDF ``9.97e36`` fill value in a query row is not an error.

    The query check bounded ``‖q‖²`` as if the query were a Gram row, so one such
    value failed the whole ``predict`` — in both layers, since inference cannot
    fall back — where scikit-learn and mlxlearn 0.1.0a2 answered. Only the inner
    product with the support vectors has to stay in float32: the rbf entry of a
    query that far out is ``exp(−inf) = 0``, and ``tanh`` saturates.
    """
    X, y = _data(n_samples=300, n_features=8, seed=0)
    if layer == "layer1":
        model = SVC(kernel=kernel).fit(X, y)
    else:
        with tuning_context(force_backend="mlx"):
            model = PatchedSVC(kernel=kernel).fit(X, y)
    ref = SkSVC(kernel=kernel).fit(X, y)
    query = X[:6].copy()
    query[1, 0] = 1e20
    query[3, 2] = 9.96921e36
    query[5, 1] = -1e30
    np.testing.assert_array_equal(model.predict(query), ref.predict(query))
    np.testing.assert_allclose(
        model.decision_function(query), ref.decision_function(query),
        atol=DECISION_ATOL, rtol=DECISION_RTOL,
    )


def test_query_beyond_the_training_frame_raises():
    """In range for float32, but not for the scale the training data set.

    Trained at 1e-20, the frame scales by about 2⁶⁶; a query at 1e19 then has
    squared norms far past float32 inside the frame.
    """
    X, y = _data(n_samples=300, n_features=4, seed=0)
    model = SVC().fit(X * 1e-20, y)
    with pytest.raises(UnsupportedInputError, match="float32"):
        model.predict(X[:3] * 1e19)


def test_object_dtype_input_is_range_checked():
    """The raw-input range scan skipped non-float dtypes; the validated one does not."""
    X, y = _data(n_samples=300, n_features=4, seed=0)
    X_object = X.astype(object)
    X_object[0, 0] = 1e300
    with pytest.raises(UnsupportedInputError) as excinfo:
        SVC().fit(X_object, y)
    assert excinfo.value.reason == "float32-overflow"


def test_float32_queries_match_float64_queries_exactly():
    X, y = _data(n_samples=500, n_features=8, seed=1)
    for kernel in KERNELS:
        model = SVC(kernel=kernel).fit(X, y)
        query = X[:100].astype(np.float32)
        np.testing.assert_array_equal(
            model.decision_function(query), model.decision_function(query.astype(np.float64))
        )


def test_support_vector_basis_survives_pickle_and_copy_without_device_state():
    """The support vectors' device copy is built once, lazily, and never pickled."""
    X, y = _data(n_classes=3, n_samples=600, n_features=10, seed=2)
    model = _fit("mlx", X, y)
    before = model.decision_function(X)
    assert model._kernel_basis_._operand is not None

    assert set(model._kernel_basis_.__getstate__()) == {"spec", "center", "scale"}
    for restored in (pickle.loads(pickle.dumps(model)), copy.deepcopy(model)):
        assert restored._kernel_basis_._operand is None
        np.testing.assert_array_equal(restored.decision_function(X), before)


def _first_predictions_in_a_burst(blob, X, n_threads=12):
    """Unpickle a fresh model, then have ``n_threads`` threads predict at once."""
    model = pickle.loads(blob)  # nothing built yet
    barrier = threading.Barrier(n_threads, timeout=60)
    results: list = [None] * n_threads
    errors: list[BaseException] = []

    def worker(slot):
        try:
            barrier.wait()
            results[slot] = model.decision_function(X)
        except BaseException as exc:  # noqa: BLE001 - reported by the caller
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    join_or_fail(threads)
    return results, errors


def test_predictions_from_many_threads_match():
    """Concurrent first predictions on an unpickled model: the operand is published whole.

    The support vectors' device copy is built lazily. Published before it was
    evaluated, another thread would evaluate its graph on a stream it does not
    own: "There is no Stream(gpu, N) in current thread", or a segfault. Evaluating
    before publishing is what makes concurrent predictions agree. Six threads
    started one after another never raced — the first built the operand before
    the second ran, and the test passed with the evaluation removed — so the
    threads are released together by a barrier, with the switch interval
    shortened to interleave them, over several fresh models.
    """
    X, y = _data(n_samples=600, n_features=10, seed=3)
    model = _fit("mlx", X, y)
    expected = model.decision_function(X)
    blob = pickle.dumps(model)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(3):
            results, errors = _first_predictions_in_a_burst(blob, X)
            assert not errors, errors[:2]
            for result in results:
                np.testing.assert_array_equal(result, expected)
    finally:
        sys.setswitchinterval(interval)


def test_the_support_vector_operand_is_built_once(monkeypatch):
    """The lock makes concurrent first predictions build the device operand once."""
    from mlxlearn._kernels.kernels import KernelBasis

    X, y = _data(n_samples=600, n_features=10, seed=3)
    blob = pickle.dumps(_fit("mlx", X, y))
    builds = []
    prepare = KernelBasis.prepare

    def slow_prepare(self, rows, **kwargs):
        if kwargs.get("name") == "support_vectors_":
            builds.append(threading.get_ident())
            time.sleep(0.02)
        return prepare(self, rows, **kwargs)

    monkeypatch.setattr(KernelBasis, "prepare", slow_prepare)
    _, errors = _first_predictions_in_a_burst(blob, X)
    assert not errors, errors[:2]
    assert len(builds) == 1, f"the operand was built {len(builds)} times"


def test_model_pickled_before_the_compute_frame_still_predicts():
    """0.1.0a2 pickles carry a float32 centre and no scale or basis."""
    X, y = _data(n_samples=500, n_features=8, seed=5)
    model = SVC().fit(X, y)
    expected = model.decision_function(X)
    legacy = copy.deepcopy(model)
    del legacy.__dict__["_kernel_basis_"]
    legacy._kernel_center_ = legacy._kernel_center_.astype(np.float32)
    np.testing.assert_allclose(legacy.decision_function(X), expected, atol=1e-4)


def test_gpu_fitted_model_predicts_on_a_host_without_a_gpu(monkeypatch):
    """The same MLX code runs on the MLX CPU device; only an explicit gpu refuses.

    A model fitted on the GPU and loaded where none is usable raised "device='gpu'
    was requested" under every configuration, including the ``"auto"`` and
    ``"cpu"`` that ask for exactly this degradation.
    """
    X, y = _data(n_samples=600, n_features=10, seed=6)
    model = _fit("mlx", X, y)
    expected_labels = model.predict(X)
    expected = model.decision_function(X)
    restored = pickle.loads(pickle.dumps(model))

    monkeypatch.setattr(device_module, "gpu_available", lambda: False)
    # 600 rows are a batch above svc_host_inference_work only with it at 0: this is
    # about the device path, which a smaller batch does not take.
    with tuning_context(svc_host_inference_work=0):
        for device in ("auto", "cpu"):
            with config_context(device=device):
                np.testing.assert_array_equal(restored.predict(X), expected_labels)
                np.testing.assert_allclose(restored.decision_function(X), expected, atol=1e-4)
        assert get_backend_diagnostics()[-1].reason == "gpu-unavailable"
        assert restored.execution_backend_ == "mlx"

        with config_context(device="gpu"):
            with pytest.raises(DeviceError, match="fitted on the MLX GPU"):
                restored.predict(X)


def test_no_approximation_ships_under_the_sklearn_name():
    """The ancestor's SVC had random-Fourier-feature and subsampling paths.

    mlxlearn ships the exact objective only, so no parameter can switch one on.
    """
    forbidden = {"rff", "random_fourier", "subsample", "approximate", "sketch"}
    params = set(SVC().get_params())
    assert not (params & forbidden)
    assert params == set(SkSVC().get_params()), "parameter surface diverged from scikit-learn"
