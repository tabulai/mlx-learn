"""LogisticRegression parity against stock scikit-learn.

The load-bearing assertion here is the **objective value**, not the coefficients.
Two correct solvers stopping at slightly different points land in measurably
different places in coefficient space while sitting at essentially the same
objective; a test that only compares `coef_` therefore fails for a correct
implementation and passes for a subtly wrong one. Every coefficient comparison
against scikit-learn below is paired with an objective comparison computed by one
float64 function applied to *both* solutions — and that function is itself checked
against an independent formula (``test_reference_objective_matches_an_independent_formula``).

Tolerances come from ``docs/fp32_policy.md`` §3.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

import mlxlearn
from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._solvers.logistic import reference_objective
from mlxlearn.exceptions import (
    SparseInputError,
    UnsupportedInputError,
    UnsupportedParameterError,
)
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.patching._estimators import LogisticRegression as PatchedLogisticRegression
from tests._helpers import logistic_objective

# docs/fp32_policy.md §3
OBJECTIVE_RTOL = 1e-5
COEF_ATOL, COEF_RTOL = 2e-3, 5e-3
PROBA_ATOL = 2e-3
MIN_AGREEMENT = 0.995

BACKENDS = ["cpu", "mlx"]

#: Everything MLX-side is in play: no crossover can hand a problem to scikit-learn.
_NO_CROSSOVER = {
    "logreg_min_samples": 0,
    "logreg_multinomial_min_features": 0,
    "logreg_min_max_iter": 0,
    "logreg_binary_min_samples": 0,
    "logreg_binary_max_features": 2**31,
}

#: scikit-learn 1.7 still has ``multi_class``; 1.8 removed it.
_HAS_MULTI_CLASS = hasattr(SkLogisticRegression(), "multi_class")
#: scikit-learn 1.9 added the callback API.
_HAS_CALLBACKS = hasattr(SkLogisticRegression(), "set_callbacks")


def _data(n_classes=2, n_samples=5000, n_features=20, seed=0):
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(12, n_features - 4),
        n_redundant=4,
        n_classes=n_classes,
        random_state=seed,
    )


def _noisy_full_rank(n_classes=2, seed=0):
    """Small, noisy, and full rank: the penalty — or its absence — moves the answer.

    On ``_data()`` (5000 rows) the ``1/(C·n)`` penalty is 2e-4, too small to show:
    a fit that silently kept ``C=1`` when asked for no penalty agreed with
    scikit-learn's unpenalized fit to every tolerance. Here the same mistake costs
    a relative objective gap of 9e-5 (three classes) to 1.1e-4 (binary).
    """
    return make_classification(
        n_samples=200, n_features=10, n_informative=6, n_redundant=0,
        n_classes=n_classes, flip_y=0.2, random_state=seed,
    )


def _objective(model, X, y, *, C, sample_weight=None):
    classes = np.unique(y)
    return reference_objective(
        model.coef_, model.intercept_, X, np.searchsorted(classes, y),
        n_classes=len(classes), C=C, sample_weight=sample_weight,
    )


def _assert_parity(mlx, ref, X, y, *, C, sample_weight=None):
    """Compare two fitted logistic models on every axis the policy names."""
    j_mlx = _objective(mlx, X, y, C=C, sample_weight=sample_weight)
    j_ref = _objective(ref, X, y, C=C, sample_weight=sample_weight)
    assert j_mlx == pytest.approx(j_ref, rel=OBJECTIVE_RTOL), (
        f"objective differs: mlxlearn {j_mlx:.10f} vs scikit-learn {j_ref:.10f}"
    )

    np.testing.assert_allclose(mlx.coef_, ref.coef_, atol=COEF_ATOL, rtol=COEF_RTOL)
    np.testing.assert_allclose(mlx.intercept_, ref.intercept_, atol=COEF_ATOL, rtol=COEF_RTOL)

    agreement = np.mean(mlx.predict(X) == ref.predict(X))
    assert agreement >= MIN_AGREEMENT, f"label agreement {agreement:.4f}"
    np.testing.assert_allclose(mlx.predict_proba(X), ref.predict_proba(X), atol=PROBA_ATOL)


def _fit(backend, estimator, X, y, **fit_kwargs):
    """Fit on a forced backend and prove it ran there."""
    if backend == "mlx" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    with tuning_context(force_backend=backend):
        estimator.fit(X, y, **fit_kwargs)
    assert estimator.execution_backend_ == backend
    return estimator


# ------------------------------------------------------------------------ parity


@pytest.mark.parametrize("C", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("n_classes", [2, 5])
def test_parity_across_regularization(C, n_classes):
    X, y = _data(n_classes=n_classes)
    # Forced onto MLX: this dataset sits far below the measured logistic
    # crossover, so the default dispatch would hand it to scikit-learn and the
    # test would compare scikit-learn against itself.
    mlx = _fit("mlx", LogisticRegression(C=C), X, y)
    ref = SkLogisticRegression(C=C, solver="lbfgs").fit(X, y)
    _assert_parity(mlx, ref, X, y, C=C)


def test_a_wide_problem_agrees_on_the_objective_then_on_probabilities_at_tight_tol():
    """docs/fp32_policy.md §3: the probability tolerance at default ``tol`` is the grid's.

    On a wide problem on the MLX path, where lbfgs stops moves the probabilities
    by up to 5e-3 at default ``tol`` — scikit-learn's own lbfgs and newton-cg
    differ as much — so the objective is what agrees there; at ``tol=1e-6`` every
    axis does. Patched dispatch hands a binary fit this wide to scikit-learn, so
    the crossover is lifted, and the fit still takes the ordinary route to MLX.
    """
    X, y = make_classification(25_000, 1_000, random_state=2)
    for tol in (1e-4, 1e-6):
        with tuning_context(**_NO_CROSSOVER):
            mlx = LogisticRegression(tol=tol, max_iter=1000).fit(X, y)
        ref = SkLogisticRegression(tol=tol, max_iter=1000).fit(X, y)
        assert _objective(mlx, X, y, C=1.0) == pytest.approx(
            _objective(ref, X, y, C=1.0), rel=OBJECTIVE_RTOL
        )
    assert mlx.fit_event_.reason == "selected"
    _assert_parity(mlx, ref, X, y, C=1.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "n_classes, units", [(2, 1.0), (5, 100.0)], ids=["binary", "multinomial-mixed-units"]
)
def test_a_wide_fit_raises_no_floating_point_warnings(n_classes, units, backend):
    """NumPy 2.2's BLAS on macOS left floating-point flags set on finite products of
    about 900 entries and more, and a wide fit warned on every L-BFGS iteration —
    each inner product, the preconditioned step's included — then on undoing the
    centering, and the reference objective on scoring the result.

    Inference is scikit-learn's own float64 code, and its warnings are its own.
    """
    X, y = _data(n_classes=n_classes, n_samples=2000, n_features=1000)
    X[:, ::3] *= units  # a third of the columns in other units: a preconditioned step
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        mlx = _fit(backend, LogisticRegression(max_iter=1000), X, y)
        objective = reference_objective(mlx.coef_, mlx.intercept_, X, y, n_classes=n_classes)
    assert math.isfinite(objective)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_classes", [2, 5])
def test_parity_without_intercept(n_classes, backend):
    X, y = _data(n_classes=n_classes)
    mlx = _fit(backend, LogisticRegression(fit_intercept=False), X, y)
    ref = SkLogisticRegression(fit_intercept=False, solver="lbfgs").fit(X, y)
    _assert_parity(mlx, ref, X, y, C=1.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_classes", [2, 5])
def test_parity_with_balanced_class_weight(n_classes, backend):
    X, y = _data(n_classes=n_classes, seed=3)
    keep = np.concatenate([np.where(y == 0)[0], np.where(y != 0)[0][:400]])
    X, y = X[keep], y[keep]

    mlx = _fit(backend, LogisticRegression(class_weight="balanced"), X, y)
    ref = SkLogisticRegression(class_weight="balanced", solver="lbfgs").fit(X, y)

    # The objective is compared through the equivalent sample weights, since
    # `balanced` is defined as a particular reweighting and comparing the
    # unweighted objective would be comparing the wrong function.
    from sklearn.utils.class_weight import compute_class_weight

    per_class = compute_class_weight("balanced", classes=ref.classes_, y=y)
    weights = per_class[np.searchsorted(ref.classes_, y)]
    _assert_parity(mlx, ref, X, y, C=1.0, sample_weight=weights)


@pytest.mark.parametrize("backend", BACKENDS)
def test_parity_with_sample_weight(backend):
    X, y = _data()
    sample_weight = np.where(y == 1, 10.0, 1.0)
    mlx = _fit(backend, LogisticRegression(), X, y, sample_weight=sample_weight)
    ref = SkLogisticRegression(solver="lbfgs").fit(X, y, sample_weight=sample_weight)
    _assert_parity(mlx, ref, X, y, C=1.0, sample_weight=sample_weight)


def test_class_weight_equals_the_equivalent_sample_weight():
    """The two paths must produce the same model, because they are the same model."""
    X, y = _data()
    by_class = LogisticRegression(class_weight={0: 1.0, 1: 3.0}).fit(X, y)
    by_sample = LogisticRegression().fit(X, y, sample_weight=np.where(y == 1, 3.0, 1.0))
    np.testing.assert_allclose(by_class.coef_, by_sample.coef_, atol=1e-10)
    np.testing.assert_allclose(by_class.intercept_, by_sample.intercept_, atol=1e-10)


def test_unit_sample_weight_is_the_unweighted_fit():
    X, y = _data()
    weighted = LogisticRegression().fit(X, y, sample_weight=np.ones(len(y)))
    plain = LogisticRegression().fit(X, y)
    np.testing.assert_allclose(weighted.coef_, plain.coef_, atol=1e-10)
    np.testing.assert_allclose(weighted.intercept_, plain.intercept_, atol=1e-10)


@pytest.mark.filterwarnings("ignore:'penalty' was deprecated:FutureWarning")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("spelling", [{"penalty": None}, {"C": np.inf}], ids=["penalty=None", "C=inf"])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_unpenalized_parity(n_classes, spelling, backend):
    """No penalty means no penalty — on the objective, not only on the labels.

    Both spellings are valid across the supported range (``penalty=None`` is
    deprecated from 1.8, ``C=np.inf`` is its replacement). The data is small and
    noisy enough that a fit which quietly kept ``C=1`` fails the objective
    assertion; the last lines prove that it would.
    """
    X, y = _noisy_full_rank(n_classes=n_classes)
    mlx = _fit(backend, LogisticRegression(**spelling), X, y)
    ref = SkLogisticRegression(**spelling, solver="lbfgs").fit(X, y)
    _assert_parity(mlx, ref, X, y, C=math.inf)

    penalized = LogisticRegression(C=1.0).fit(X, y)
    j_mlx = _objective(mlx, X, y, C=math.inf)
    assert _objective(penalized, X, y, C=math.inf) > j_mlx * (1 + 4 * OBJECTIVE_RTOL)


def test_reference_objective_matches_an_independent_formula():
    """The parity oracle is mlxlearn's own function, so it is checked against another.

    ``tests/_helpers.logistic_objective`` is written differently — signed margins
    for binary, an explicit log-softmax for multiclass — and scaled by ``C·sw_sum``.
    """
    rng = np.random.default_rng(7)
    for n_classes in (2, 4):
        X, y = _data(n_classes=n_classes, n_samples=300, n_features=8, seed=5)
        n_targets = 1 if n_classes == 2 else n_classes
        coef = rng.normal(size=(n_targets, 8))
        intercept = rng.normal(size=n_targets)
        for C, weights in ((0.1, None), (3.0, rng.uniform(0.5, 2.0, size=len(y)))):
            sw_sum = len(y) if weights is None else weights.sum()
            ours = reference_objective(
                coef, intercept, X, y, n_classes=n_classes, C=C, sample_weight=weights
            )
            theirs = logistic_objective(
                coef, intercept, X, y, C=C, classes=np.arange(n_classes), sample_weight=weights
            )
            assert ours == pytest.approx(theirs / (C * sw_sum), rel=1e-12)


def test_the_preconditioner_leaves_ordinary_scales_alone():
    """Exactly the identity within 4⁴ of the intercept's curvature, shrunk beyond it.

    Standardized and ordinarily scaled data must see the unpreconditioned solver —
    the pinned iteration counts in tests/unit/test_lbfgs.py depend on it — and a
    column that could not be measured must not poison the rest.
    """
    from mlxlearn._solvers.logistic import _preconditioner

    scale = np.array([0.1, 1.0, 20.0, 2.0**10, 2.0**-10, np.inf, 0.0])
    D = _preconditioner(scale, 0.0, 2, True)
    assert D.shape == (2, 8)
    np.testing.assert_array_equal(D[:, :3], 1.0)
    assert D[0, 3] == 4.0 ** -(10 - 4)
    assert D[0, 4] == 4.0 ** (10 - 4)
    np.testing.assert_array_equal(D[:, 5:], 1.0)  # unmeasurable, constant, intercept
    # The penalty's own curvature bounds a tiny column's from below.
    assert _preconditioner(np.array([2.0**-10]), 0.25, 1, False)[0, 0] == 1.0


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_binary_objective_matches_float64_in_both_forms(fit_intercept, weighted, device):
    """The value and gradient the solver sees, against a float64 NumPy computation.

    The GPU's form sums the gradient as a batched product over row blocks and
    finishes on the host, because MLX's ``(n,) @ (n, d)`` product took a path up
    to 38x slower than the forward pass; both forms must describe the same function
    on either device, the rows past the last whole block included.
    """
    import mlx.core as mx

    from mlxlearn._common.device import use_device
    from mlxlearn._solvers.logistic import (
        _binary_objective_cpu,
        _binary_objective_gpu,
        _row_blocks,
    )

    if device == "gpu" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    rng = np.random.default_rng(1)
    n, d = 3000, 12
    assert _row_blocks(n)[2] > 0, "precondition: rows past the last whole block"
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (rng.uniform(size=n) < 0.4).astype(np.float32)
    sw = rng.uniform(0.5, 2.0, size=n).astype(np.float32) if weighted else None
    theta = rng.normal(scale=0.3, size=(1, d + fit_intercept)).astype(np.float32)
    l2 = 1e-3
    sw_sum = float(n if sw is None else sw.astype(np.float64).sum())

    Xd, th = X.astype(np.float64), theta.astype(np.float64)
    raw = Xd @ th[0, :d] + (th[0, d] if fit_intercept else 0.0)
    w64 = np.ones(n) if sw is None else sw.astype(np.float64)
    residual = w64 * (1.0 / (1.0 + np.exp(-raw)) - y)
    value = (w64 * (np.logaddexp(0.0, raw) - y * raw)).sum() / sw_sum
    value += 0.5 * l2 * (th[0, :d] ** 2).sum()
    grad = [Xd.T @ residual / sw_sum + l2 * th[0, :d]]
    if fit_intercept:
        grad.append([residual.sum() / sw_sum])
    grad = np.concatenate(grad)

    with use_device(device):
        for objective in (_binary_objective_cpu, _binary_objective_gpu):
            fun = objective(
                mx, mx.array(X), mx.array(y), None if sw is None else mx.array(sw),
                sw_sum, l2, fit_intercept, d,
            )
            got_value, got_grad = fun(mx.array(theta))
            assert got_value == pytest.approx(value, rel=1e-6)
            assert np.shape(got_grad) == theta.shape
            np.testing.assert_allclose(np.asarray(got_grad)[0], grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("n_features", [1, 6])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_multinomial_objective_matches_float64_in_both_forms(
    fit_intercept, weighted, n_features, device
):
    """The same for the softmax: MLX's ``logsumexp`` in one form, the GPU's
    spelled-out one in the other, and at one feature the GPU's batched gradient.

    Logits in the hundreds too, where most probabilities underflow and only the
    shifted identity keeps the loss exact. There the gradient is held to float32's
    resolution of it (``docs/fp32_policy.md`` §2.3), since rounding a logit of 300
    alone moves a probability by about 2e-5 of itself.
    """
    import mlx.core as mx

    from mlxlearn._common.device import use_device
    from mlxlearn._solvers.lbfgs import GRADIENT_RESOLUTION
    from mlxlearn._solvers.logistic import (
        _multinomial_objective_cpu,
        _multinomial_objective_gpu,
        _row_blocks,
    )

    if device == "gpu" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    rng = np.random.default_rng(2)
    n, d, K = 3000, n_features, 4
    assert _row_blocks(n)[2] > 0, "precondition: rows past the last whole block"
    codes = rng.integers(0, K, size=n)
    X = (rng.normal(size=(K, d))[codes] + rng.normal(size=(n, d))).astype(np.float32)
    sw = rng.uniform(0.5, 2.0, size=n).astype(np.float32) if weighted else None
    l2 = 1e-3
    sw_sum = float(n if sw is None else sw.astype(np.float64).sum())
    Xd = X.astype(np.float64)
    w64 = np.ones(n) if sw is None else sw.astype(np.float64)
    onehot = np.eye(K)[codes]

    for scale in (0.3, 100.0):
        theta = rng.normal(scale=scale, size=(K, d + fit_intercept)).astype(np.float32)
        th = theta.astype(np.float64)
        raw = Xd @ th[:, :d].T + (th[:, d] if fit_intercept else 0.0)
        shift = raw.max(axis=1, keepdims=True)
        log_norm = shift[:, 0] + np.log(np.exp(raw - shift).sum(axis=1))
        residual = w64[:, None] * (np.exp(raw - log_norm[:, None]) - onehot)
        value = (w64 * (log_norm - raw[np.arange(n), codes])).sum() / sw_sum
        value += 0.5 * l2 * (th[:, :d] ** 2).sum()
        grad = residual.T @ Xd / sw_sum + l2 * th[:, :d]
        if fit_intercept:
            grad = np.concatenate([grad, residual.sum(axis=0)[:, None] / sw_sum], axis=1)

        with use_device(device):
            for objective in (_multinomial_objective_cpu, _multinomial_objective_gpu):
                fun = objective(
                    mx, mx.array(X), mx.array(codes.reshape(-1, 1).astype(np.int32)),
                    mx.array(onehot.astype(np.float32)), None if sw is None else mx.array(sw),
                    sw_sum, l2, fit_intercept, d,
                )
                got_value, got_grad = fun(mx.array(theta))
                assert got_value == pytest.approx(value, rel=1e-6)
                assert np.shape(got_grad) == theta.shape
                atol = 1e-6 if scale < 1 else GRADIENT_RESOLUTION * np.abs(grad).max()
                np.testing.assert_allclose(np.asarray(got_grad), grad, rtol=1e-4, atol=atol)


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, (1, 0, 0)),
        (31, (1, 31, 0)),
        (32, (16, 2, 0)),
        (3000, (16, 187, 8)),
        (65_536, (16, 4096, 0)),
        (65_537, (16, 4096, 1)),
        (1_000_000, (244, 4098, 88)),
        (10**7 + 3, (256, 39_062, 131)),
    ],
)
def test_row_blocks_cover_every_row_once(n, expected):
    """The batched gradient's ``(blocks, rows, tail)``: every row in exactly one product.

    Pinned, because they fix the order the GPU gradient is summed in, and with it
    every bit of the result: a function of the row count alone, the same on every
    machine.
    """
    from mlxlearn._solvers.logistic import _row_blocks

    blocks, rows, tail = _row_blocks(n)
    assert (blocks, rows, tail) == expected
    assert blocks * rows + tail == n and 0 <= tail < blocks


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_gpu_fit_runs_the_gpu_objectives_on_the_row_major_design(backend, monkeypatch):
    """The objective is chosen from the device the fit actually runs on.

    The gate compared what ``use_device`` yields with the string ``"gpu"``; once it
    yielded an ``mx.Stream`` that never matched, and every GPU binary fit silently
    took the slow ``(n,) @ (n, d)`` gradient, 2.5-2.9x slower on tall, narrow data.
    Both devices get the design row-major, as it is: the GPU's batched gradient
    reads blocks of it in place, where a transposed copy used to be built.
    """
    from mlxlearn._solvers import logistic

    seen = []

    def spy(name):
        original = getattr(logistic, name)

        def build(mx, X, *args):
            seen.append((name, tuple(X.shape)))
            return original(mx, X, *args)

        return build

    for form in ("binary", "multinomial"):
        for device in ("cpu", "gpu"):
            name = f"_{form}_objective_{device}"
            monkeypatch.setattr(logistic, name, spy(name))
    for n_classes in (2, 3):
        X, y = _data(n_classes=n_classes, n_samples=400, n_features=8)
        _fit(backend, LogisticRegression(), X, y)
    device = "gpu" if backend == "mlx" else "cpu"
    assert seen == [
        (f"_binary_objective_{device}", (400, 8)),
        (f"_multinomial_objective_{device}", (400, 8)),
    ]


# ------------------------------------------------------------- hard numerics


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


@pytest.mark.parametrize("offset", [0.0, 1e2, 1e3, 1e4, 1e5])
def test_constant_feature_offset_does_not_destroy_the_fit(offset):
    """Features carrying a large constant offset must not collapse the model.

    Without centering, at an offset of 1e4 the solver stopped after 3 iterations
    with an all-zero coefficient vector, an objective of exactly log 2 — the null
    model — and 0.50 accuracy where scikit-learn reached 0.62. The gradient of an
    offset-dominated design is numerically parallel to the constant direction, and
    at 1e4 float32's spacing is ~1e-3, which quantizes the informative signal away.

    Centering is exact when there is an intercept to absorb the shift, so this is
    a correctness fix rather than a heuristic. mlxlearn now also converges in
    fewer iterations than scikit-learn on offset data.
    """
    X, y = _data(n_samples=2000, n_features=8, seed=0)
    X = X + offset

    with tuning_context(force_backend="mlx"):
        mlx = LogisticRegression(max_iter=1000).fit(X, y)
    ref = SkLogisticRegression(max_iter=1000, solver="lbfgs").fit(X, y)

    assert np.abs(mlx.coef_).max() > 1e-6, "coefficients collapsed to zero"
    assert mlx.score(X, y) >= ref.score(X, y) - 0.01

    # This fixture has 4 informative and 4 redundant features, so the design is
    # rank-deficient and the coefficients are only pinned down by the penalty. The
    # objective is what both solvers minimize, so that is what is compared — and
    # one-sidedly, because on offset data mlxlearn reaches a *lower* objective than
    # scikit-learn (0.6104452 against 0.6104529 at an offset of 1e5, converging in
    # 113 iterations against 175). A two-sided tolerance would fail mlxlearn for
    # winning.
    j_mlx = _objective(mlx, X, y, C=1.0)
    j_ref = _objective(ref, X, y, C=1.0)
    assert j_mlx <= j_ref * (1 + OBJECTIVE_RTOL), (
        f"mlxlearn's objective is worse at offset {offset}: {j_mlx:.10f} vs {j_ref:.10f}"
    )


def test_float64_offsets_are_centered_before_rounding():
    """Two Unix-timestamp columns, an hour wide, next to ordinary features.

    Rounded to float32 first, a timestamp keeps a resolution of 128 seconds, so the
    informative spread is quantized before centering can protect it: the fitted
    timestamp coefficients came out about 4x attenuated. Centered in float64 first,
    the fit agrees with scikit-learn fitted on the centered data — which is the
    well-conditioned version of the same problem.
    """
    rng = np.random.default_rng(3)
    n = 4000
    Z = rng.normal(size=(n, 4))
    logit = 1.5 * Z[:, 0] - 1.0 * Z[:, 1] + 0.8 * Z[:, 2] - 0.5 * Z[:, 3]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    X = np.column_stack([1.7e9 + 600.0 * Z[:, 0], 1.7e9 + 600.0 * Z[:, 1], Z[:, 2], Z[:, 3]])

    mlx = _fit("mlx", LogisticRegression(), X, y)
    ref = SkLogisticRegression().fit(X - X.mean(axis=0), y)

    np.testing.assert_allclose(mlx.coef_, ref.coef_, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(mlx.predict_proba(X), ref.predict_proba(X - X.mean(axis=0)), atol=PROBA_ATOL)


def test_centering_is_exact_not_approximate():
    """Shifting the data must not change the model, only the intercept.

    ``w·(x − μ) + b′ ≡ w·x + (b′ − w·μ)``. If the correction were wrong, the
    coefficients would still look fine and every prediction would be shifted.
    """
    X, y = _data(n_samples=3000, n_features=10)
    with tuning_context(force_backend="mlx"):
        plain = LogisticRegression(max_iter=1000).fit(X, y)
        shifted = LogisticRegression(max_iter=1000).fit(X + 500.0, y)

    np.testing.assert_allclose(shifted.coef_, plain.coef_, atol=1e-3, rtol=2e-3)
    np.testing.assert_allclose(
        shifted.predict_proba(X + 500.0), plain.predict_proba(X), atol=PROBA_ATOL
    )


def test_no_centering_without_an_intercept():
    """With ``fit_intercept=False`` the shifted problem is a different problem.

    Centering anyway would silently solve something the caller did not ask for.
    """
    X, y = _data(n_samples=2000, n_features=8)
    with tuning_context(force_backend="mlx"):
        mlx = LogisticRegression(fit_intercept=False, max_iter=1000).fit(X + 5.0, y)
    ref = SkLogisticRegression(fit_intercept=False, max_iter=1000, solver="lbfgs").fit(X + 5.0, y)
    np.testing.assert_allclose(mlx.coef_, ref.coef_, atol=COEF_ATOL, rtol=COEF_RTOL)
    assert _objective(mlx, X + 5.0, y, C=1.0) == pytest.approx(
        _objective(ref, X + 5.0, y, C=1.0), rel=OBJECTIVE_RTOL
    )
    np.testing.assert_array_equal(mlx.intercept_, np.zeros_like(mlx.intercept_))


@pytest.mark.parametrize("scale", [1e8, 1e10])
def test_features_in_large_units_reach_the_same_optimum(scale):
    """Features around 1e8 used to return the null model at ``n_iter=0``.

    The line search's zoom stopped on an absolute step floor of 1e-12 while the
    acceptable first step at this scale is ~1e-16; with that fixed, the intercept
    still never moved until the preconditioner told the solver its curvature is
    1e16 times smaller. Rescaling the features is an exact reparametrization, and at
    ``coef_ ~ 1/scale`` the C=1 penalty is negligible, so the answer must be the
    unit-scale unpenalized one: same objective, ``coef_ · scale`` the same
    coefficients (the design is full rank, so they are unique).
    """
    X, y = make_classification(3000, 6, n_informative=4, n_redundant=0, random_state=7)
    unit = _fit("mlx", LogisticRegression(C=np.inf, max_iter=1000), X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scaled = _fit("mlx", LogisticRegression(max_iter=1000), X * scale, y)

    assert scaled.n_iter_[0] > 0
    assert _objective(scaled, X * scale, y, C=math.inf) == pytest.approx(
        _objective(unit, X, y, C=math.inf), rel=OBJECTIVE_RTOL
    )
    np.testing.assert_allclose(scaled.coef_ * scale, unit.coef_, atol=COEF_ATOL, rtol=COEF_RTOL)
    np.testing.assert_allclose(scaled.intercept_, unit.intercept_, atol=COEF_ATOL, rtol=COEF_RTOL)


def test_one_feature_in_large_units_among_ordinary_ones():
    """A mixed-scale design, which scikit-learn's own lbfgs also handles.

    With one column 1e6 times the others the curvature along its coefficient is
    1e12 times the rest; the diagonal preconditioner is what lets the ordinary
    coefficients and the intercept move at all.
    """
    X, y = make_classification(3000, 6, n_informative=4, random_state=7)
    X[:, 0] *= 1e6
    mlx = _fit("mlx", LogisticRegression(max_iter=1000), X, y)
    ref = SkLogisticRegression(max_iter=10000).fit(X, y)
    assert _objective(mlx, X, y, C=1.0) <= _objective(ref, X, y, C=1.0) * (1 + OBJECTIVE_RTOL)


@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize("tol", [1e-6, 1e-8])
def test_a_small_tol_converges_without_a_warning(tol, n_classes):
    """scikit-learn reaches these tolerances silently, so mlxlearn must too.

    Accepting a step only when the float32 objective visibly decreased stalled at
    ``max|g|`` about 8e-6 (binary) and 4e-5 (five classes) — a line-search failure
    and a ConvergenceWarning, 50x above the float32 gradient floor. The line search
    now also accepts approximate-Wolfe steps, and a ``tol`` below what float32 can
    resolve is honored as "run to the floor" rather than warned about.
    """
    X, y = _data(n_classes=n_classes)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        mlx = _fit("mlx", LogisticRegression(tol=tol, max_iter=1000), X, y)
    ref = SkLogisticRegression(tol=tol, max_iter=1000).fit(X, y)
    assert _objective(mlx, X, y, C=1.0) <= _objective(ref, X, y, C=1.0) * (1 + 1e-7)


@pytest.mark.parametrize("scale", [100.0, 1000.0])
def test_scaled_features_converge_without_a_warning(scale):
    """Standardized features times 100 warned on 8 of 16 seeds; scikit-learn on none."""
    for seed in range(4):
        X, y = make_classification(2000, 4, n_informative=3, n_redundant=1, random_state=seed)
        X = (X - X.mean(axis=0)) / X.std(axis=0) * scale
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            mlx = _fit("cpu", LogisticRegression(), X, y)
        ref = SkLogisticRegression().fit(X, y)
        _assert_parity(mlx, ref, X, y, C=1.0)


def test_tol_zero_runs_to_the_float32_floor_and_records_it():
    """``tol=0`` is "as far as float32 goes", which is a stop, not a failure.

    docs/fp32_policy.md §2.3: the run ends when the objective stops resolving
    progress, with no ConvergenceWarning once ``max|g|`` is below what float32
    resolves, and the stop is recorded in the diagnostics.
    """
    X, y = _data()
    mlxlearn.clear_backend_diagnostics()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = _fit("mlx", LogisticRegression(tol=0.0, max_iter=1000), X, y)

    assert model.n_iter_[0] < 200
    assert mlxlearn.get_backend_diagnostics() == [model.fit_event_]
    assert model.fit_event_.extra["stop_reason"] == "float32-floor"
    assert model.fit_event_.extra["at_resolution_floor"] is True
    assert "float32 floor" in model.fit_event_.detail


def test_non_convergence_warns_and_is_recorded():
    """On the fit's one event: a second, solver event became
    ``get_last_backend_event()`` in the fit's place and counted as more work.
    """
    X, y = _data()
    mlxlearn.clear_backend_diagnostics()
    with pytest.warns(ConvergenceWarning, match=r"\(max_iter\).*Increase max_iter"):
        model = LogisticRegression(max_iter=1).fit(X, y)
    assert mlxlearn.get_backend_diagnostics() == [model.fit_event_]
    assert mlxlearn.get_last_backend_event() is model.fit_event_
    event = model.fit_event_
    assert (event.operation, event.reason) == ("fit", "below-crossover")
    assert event.extra["stop_reason"] == "max_iter" and event.extra["n_iter"] == 1
    # The crossover's explanation first, then the solver's.
    assert event.detail.index("crossover") < event.detail.index("failed to converge")
    assert mlxlearn.diagnostics_summary() == {"LogisticRegression.fit -> cpu": 1}


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_convergence_warning_starts_as_scikit_learns_does(backend):
    """Filters on "lbfgs failed to converge" — scikit-learn's own tests use them —
    must match wherever the fit ran. The message began "LogisticRegression failed
    to converge", so under patch_sklearn() such a filter silenced the small fits it
    handed to scikit-learn and none of the large ones it sent to MLX.
    """
    X, y = _data()
    prefix = r"^lbfgs failed to converge after 1 iteration\(s\) \("
    with pytest.warns(ConvergenceWarning, match=prefix):
        SkLogisticRegression(max_iter=1).fit(X, y)
    with pytest.warns(ConvergenceWarning, match=prefix):
        _fit(backend, LogisticRegression(max_iter=1), X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.filterwarnings("ignore", message="lbfgs failed to converge")
        _fit(backend, LogisticRegression(max_iter=1), X, y)


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_escalated_convergence_warning_leaves_the_fit_incomplete(backend):
    """A fit that raises does not commit, whatever raised — here, an "error" filter.

    The warning came after the commit point, so the fit raised with the model
    fitted, predicting and recorded as a completed fit, where the patched class's
    scikit-learn path and SVC both end unfitted.
    """
    if backend == "mlx" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    X, y = _data()
    model = _fit(backend, LogisticRegression(), X, y)
    model.set_params(max_iter=1)
    mlxlearn.clear_backend_diagnostics()
    with warnings.catch_warnings(), tuning_context(force_backend=backend):
        warnings.simplefilter("error", ConvergenceWarning)
        with pytest.raises(ConvergenceWarning):
            model.fit(X, y)

    # Not the previous fit's coefficients either.
    with pytest.raises(NotFittedError):
        model.predict(X)
    with pytest.raises(AttributeError):
        model.execution_backend_
    # On the record all the same, as every fit's decision is, with the stop that
    # raised: one event, the fit's.
    (event,) = mlxlearn.get_backend_diagnostics()
    assert (event.operation, event.reason) == ("fit", "forced")
    assert event.extra["stop_reason"] == "max_iter"


@pytest.mark.parametrize("budget", [5, 10, 20])
def test_objective_is_no_worse_at_an_equal_truncated_budget(budget):
    """The fair comparison when neither solver has converged.

    On a hard problem at a small ``max_iter`` both solvers stop on the iteration
    limit and both warn. Comparing their coefficients then compares two arbitrary
    intermediate iterates, and the observed gaps are large. What can be asserted —
    and what actually matters — is that mlxlearn has not made *less* progress.

    The budget was 100 until the diagonal preconditioner, with which mlxlearn
    converged on this problem in 91 iterations (scikit-learn needs 251), and then
    50. With the dense preconditioner (``docs/fp32_policy.md`` §2.3) — the features
    are correlated, one of them redundant — it converges in 23, so the budgets are
    5 to 20, where both are still truncated. The claim holds at each: the objective
    was 2.3-3.6% below scikit-learn's at 5-20 iterations; on the diagonal path it
    holds from 5 (0.1-2.0% below) and failed below that.
    """
    from sklearn.utils.class_weight import compute_class_weight

    X, y = make_classification(
        n_samples=7785, n_features=18, n_informative=9, n_redundant=1,
        n_classes=6, n_clusters_per_class=1, random_state=42,
    )
    X = X * 18.7
    C = 39.0

    with tuning_context(force_backend="mlx"):
        mlx = LogisticRegression(C=C, class_weight="balanced", max_iter=budget).fit(X, y)
    ref = SkLogisticRegression(
        C=C, class_weight="balanced", max_iter=budget, solver="lbfgs"
    ).fit(X, y)

    assert mlx.fit_event_.extra["preconditioner"] == "dense"
    assert mlx.n_iter_[0] == budget and ref.n_iter_[0] == budget, (
        "precondition: neither converged"
    )

    weights = compute_class_weight("balanced", classes=ref.classes_, y=y)[
        np.searchsorted(ref.classes_, y)
    ]
    j_mlx = _objective(mlx, X, y, C=C, sample_weight=weights)
    j_ref = _objective(ref, X, y, C=C, sample_weight=weights)

    assert j_mlx <= j_ref * (1 + 1e-4), (
        f"mlxlearn made less progress in the same budget: {j_mlx:.8f} vs {j_ref:.8f}"
    )


# ---------------------------------------------------------------- sample weights


def test_sample_weight_scale_is_exact():
    """Multiplying every weight by 2**k is the same problem as multiplying C by 2**k.

    ``l2 = 1/(C·Σw)``, so the two agree term for term; and because the solver
    normalizes weights by an exact power of two, bit for bit.
    """
    X, y = _data(n_samples=2000, n_features=8)
    weights = np.random.default_rng(0).uniform(0.5, 2.0, size=len(y))
    scaled = LogisticRegression().fit(X, y, sample_weight=weights * 2.0**40)
    equivalent = LogisticRegression(C=2.0**40).fit(X, y, sample_weight=weights)
    np.testing.assert_array_equal(scaled.coef_, equivalent.coef_)
    np.testing.assert_array_equal(scaled.intercept_, equivalent.intercept_)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("scale", [1e36, 1e39])
def test_huge_sample_weights_do_not_overflow_float32(scale, n_classes):
    """Weights of 1e36 are legal, and scikit-learn fits them.

    Cast straight to float32, weights above 3.4e38 became infinite; below that, the
    float32 reduction ``Σ w·x·r`` overflowed from about 1e36 at 2000 rows. Either way
    the fit returned an all-zero model. The same problem stated with unit-scale
    weights and ``C`` scaled to match must give the same model.
    """
    X, y = _data(n_classes=n_classes, n_samples=2000, n_features=8)
    weights = np.random.default_rng(0).uniform(0.5, 2.0, size=len(y))
    huge = LogisticRegression().fit(X, y, sample_weight=weights * scale)
    equivalent = LogisticRegression(C=scale).fit(X, y, sample_weight=weights)

    assert np.abs(huge.coef_).max() > 0.1, "collapsed to the null model"
    np.testing.assert_allclose(huge.coef_, equivalent.coef_, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(huge.intercept_, equivalent.intercept_, rtol=1e-4, atol=1e-5)


def test_sample_weight_equivalence_at_float32_tolerance():
    """The property scikit-learn's own check asserts, at float32's tolerance.

    ``check_sample_weight_equivalence_on_dense_data`` is one of scikit-learn's
    declared expected failures for its own LogisticRegression, so the compliance
    suite skips it for mlxlearn's too (``INHERITED_FAILURES`` in
    ``tests/test_sklearn_compliance.py``; nothing is xfailed under a Rule). The
    property is real, so it is asserted here at the tolerance float32 actually
    supports rather than left untested.
    """
    X, y = _data(n_samples=800, n_features=10)
    weights = np.random.default_rng(0).integers(1, 4, size=len(y)).astype(float)

    # Solved to tol=1e-6: at the default 1e-4 each fit stops wherever max|g| first
    # drops below it, and in float32 the two problems' rounding sends the solver down
    # different paths — their intercepts stopped 2.2e-3 apart, each within 2.5e-3 of
    # the optimum. That measures the stopping rule, not the weighting.
    weighted = LogisticRegression(tol=1e-6).fit(X, y, sample_weight=weights)
    repeated = LogisticRegression(tol=1e-6).fit(
        np.repeat(X, weights.astype(int), axis=0),
        np.repeat(y, weights.astype(int)),
    )
    np.testing.assert_allclose(weighted.coef_, repeated.coef_, atol=1e-3, rtol=2e-3)
    np.testing.assert_allclose(weighted.intercept_, repeated.intercept_, atol=1e-3, rtol=2e-3)


# ------------------------------------------------------------ capability gates


def test_float32_overflow_is_a_capability_error():
    """A float64 value that becomes infinite in float32 must be a *capability* limit.

    scikit-learn's float64 path handles it fine, so patched mode has somewhere to
    fall back to — but only if the error is in the CapabilityError family. Raised
    before ``validate_data``, which would otherwise convert first and produce a
    plain ``ValueError``.
    """
    X, y = _data(n_samples=300, n_features=6)
    X = np.array(X)
    X[0, 0] = 1e300

    with pytest.raises(UnsupportedInputError) as excinfo:
        LogisticRegression().fit(X, y)
    assert excinfo.value.reason == "float32-overflow"


def test_negative_sample_weight_is_a_capability_error():
    """scikit-learn accepts negative sample weights; mlxlearn's solvers do not.

    Since scikit-learn can serve the request, patched mode must be able to fall
    back — which requires this to be a CapabilityError, not a bare ValueError.
    """
    X, y = _data(n_samples=300, n_features=6)
    weights = np.ones(len(y))
    weights[0] = -1.0

    with pytest.raises(UnsupportedInputError) as excinfo:
        LogisticRegression().fit(X, y, sample_weight=weights)
    assert excinfo.value.reason == "negative-sample-weight"


@pytest.mark.parametrize("tuning", [{}, _NO_CROSSOVER, {"force_backend": "mlx"}], ids=["below", "above", "forced"])
def test_negative_sample_weight_falls_back_under_patch(tuning, recwarn):
    """The error message promised a fallback; above the crossover it used to raise.

    The weight check ran after backend selection, outside the capability check, so
    only a problem small enough to be handed to scikit-learn anyway "fell back".
    """
    X, y = _data(n_samples=300, n_features=6)
    weights = np.ones(len(y))
    weights[:5] = -1.0
    with tuning_context(**tuning):
        model = PatchedLogisticRegression().fit(X, y, sample_weight=weights)
    ref = SkLogisticRegression().fit(X, y, sample_weight=weights)
    assert model.execution_backend_ == "sklearn"
    np.testing.assert_array_equal(model.coef_, ref.coef_)


_GATES = [
    ({"solver": "liblinear"}, "solver"),
    ({"solver": "saga"}, "solver"),
    ({"solver": "newton-cg"}, "solver"),
    ({"penalty": "l1", "solver": "saga"}, "solver"),
    ({"penalty": "l1"}, "penalty"),
    ({"dual": True}, "dual"),
    ({"dual": True, "solver": "liblinear", "penalty": "l2"}, "solver"),
    ({"warm_start": True}, "warm_start"),
    ({"intercept_scaling": 2.0}, "intercept_scaling"),
]
if _HAS_MULTI_CLASS:
    _GATES += [({"multi_class": "ovr"}, "multi_class"), ({"multi_class": "multinomial"}, "multi_class")]


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("kwargs, parameter", _GATES)
def test_unsupported_parameters_raise_naming_the_parameter(kwargs, parameter):
    """Each gate on its own, so removing any one of them fails a test.

    ``multi_class='ovr'`` is the ancestor's headline bug — accepted, and a
    multinomial model fitted instead — and exists only on scikit-learn 1.7.
    """
    X, y = _data(n_samples=300)
    with pytest.raises(UnsupportedParameterError) as excinfo:
        LogisticRegression(**kwargs).fit(X, y)
    assert excinfo.value.parameter == parameter


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("kwargs, parameter", _GATES)
def test_unsupported_parameters_fall_back_under_patch(kwargs, parameter, recwarn):
    """Where Layer 1 raises, Layer 2 hands the request to scikit-learn — recorded.

    Some of these combinations scikit-learn rejects too (``dual=True`` with lbfgs);
    then the patched class must fail exactly as scikit-learn does, not differently.
    """
    X, y = _data(n_samples=300)
    try:
        SkLogisticRegression(**kwargs).fit(X, y)
    except ValueError as exc:
        with tuning_context(**_NO_CROSSOVER), pytest.raises(type(exc), match="."):
            PatchedLogisticRegression(**kwargs).fit(X, y)
        return

    mlxlearn.clear_backend_diagnostics()
    with tuning_context(**_NO_CROSSOVER):
        model = PatchedLogisticRegression(**kwargs).fit(X, y)
    assert model.execution_backend_ == "sklearn"
    fit_events = [e for e in mlxlearn.get_backend_diagnostics() if e.operation == "fit"]
    assert [e.reason for e in fit_events] == [f"unsupported-parameter:{parameter}"]


def test_intercept_scaling_is_irrelevant_without_an_intercept():
    X, y = _data(n_samples=300)
    model = LogisticRegression(intercept_scaling=2.0, fit_intercept=False).fit(X, y)
    assert model.execution_backend_ == "cpu"


def test_multi_label_target_is_a_capability_error():
    X, y = _data(n_samples=300)
    Y = np.column_stack([y, 1 - y])
    with pytest.raises(UnsupportedInputError) as excinfo:
        LogisticRegression().fit(X, Y)
    assert excinfo.value.reason == "multi-label-target"


def test_regression_target_gets_scikit_learns_message():
    X, _ = _data(n_samples=300)
    y = np.random.default_rng(0).normal(size=300)
    with pytest.raises(ValueError, match=r"Unknown label type: continuous\. Maybe you are"):
        LogisticRegression().fit(X, y)


def test_sparse_input_raises_type_error():
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _data(n_samples=300)
    with pytest.raises(SparseInputError) as excinfo:
        LogisticRegression().fit(sparse.csr_matrix(X), y)
    assert isinstance(excinfo.value, TypeError)


# -------------------------------------------------------------------- dispatch


def test_patched_fit_below_the_crossover_skips_the_mlx_preparation(monkeypatch):
    """Work handed to scikit-learn must not first be validated for MLX.

    Scanning ``X`` for float32 overflow and converting it to float32 — all of which
    ``super().fit`` then repeats in float64 — made a handed-off fit up to 2x slower
    than stock. The decision now comes from the raw shape after the parameter
    checks alone, and is recorded once.
    """
    import mlxlearn.linear_model._logistic as module

    def forbidden(*args, **kwargs):
        raise AssertionError("MLX-path preparation ran for a problem sent to scikit-learn")

    monkeypatch.setattr(module, "check_float32_range", forbidden)
    monkeypatch.setattr(module, "validate_data", forbidden)

    X, y = _data(n_samples=500)
    mlxlearn.clear_backend_diagnostics()
    model = PatchedLogisticRegression().fit(X, y)
    ref = SkLogisticRegression().fit(X, y)

    assert model.execution_backend_ == "sklearn"
    np.testing.assert_array_equal(model.coef_, ref.coef_)
    fit_events = [e for e in mlxlearn.get_backend_diagnostics() if e.operation == "fit"]
    assert [(e.backend, e.reason) for e in fit_events] == [("sklearn", "below-crossover")]


def _narrow(n_classes, n_samples, n_features, seed=0):
    """Few features — fewer than ``_data`` can make — and labels cut from the first."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    score = X[:, 0] + rng.normal(scale=0.5, size=n_samples)
    return X, np.digitize(score, np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1]))


def test_a_multinomial_fit_needs_two_features():
    """MLX's multinomial path costs about 10 ms at 60 000 rows and one feature,
    whatever the class count, and one-feature fits of labels without signal lost or
    tied against stock up to nine classes. A binary fit has its own rules — rows and
    width — and this one does not bind it: binary fits won from 200 000 rows at one
    feature."""
    n = 1000
    with tuning_context(logreg_min_samples=n, logreg_binary_min_samples=n):
        binary = PatchedLogisticRegression().fit(*_narrow(2, n, 1))
        narrow = PatchedLogisticRegression().fit(*_narrow(3, n, 1))
        wide = PatchedLogisticRegression().fit(*_narrow(3, n, 2))
    assert binary.execution_backend_ != "sklearn", binary.fit_event_
    assert narrow.execution_backend_ == "sklearn"
    assert "multinomial" in narrow.fit_event_.detail and "2 features" in narrow.fit_event_.detail
    assert wide.execution_backend_ in ("mlx", "cpu"), wide.fit_event_


@pytest.mark.parametrize(
    "params, handed",
    [
        ({}, False),
        ({"tol": 1e-4}, False),
        ({"tol": 1e-5}, False),
        ({"tol": np.nextafter(1e-4, 1.0)}, True),
        ({"tol": 5e-4}, True),
        ({"tol": 1e-3}, True),
        ({"tol": 1e-2}, True),
        ({"max_iter": 3}, False),
        ({"max_iter": 2}, True),
        ({"max_iter": 0}, True),
    ],
)
def test_patched_dispatch_hands_short_fits_to_scikit_learn(params, handed):
    """A looser ``tol`` or a smaller ``max_iter`` stops a fit within an iteration or
    two, where it is mostly MLX's fixed cost: a ``tol`` above scikit-learn's default
    (1e-4, where the size rules were measured) and ``max_iter`` of 1-2 go to
    scikit-learn. Patched only, and decided before the data are read."""
    n = 1000
    X, y = _narrow(2, n, 4)
    lifted = {"logreg_min_samples": n, "logreg_binary_min_samples": n}
    with tuning_context(**lifted), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = PatchedLogisticRegression(**params).fit(X, y)
        direct = LogisticRegression(**params).fit(X, y)
    assert (model.execution_backend_ == "sklearn") == handed, model.fit_event_
    assert direct.execution_backend_ != "sklearn"
    if handed:
        assert model.fit_event_.reason == "below-crossover"
        why = "scikit-learn's default of 0.0001" if "tol" in params else "solver parameters"
        assert why in model.fit_event_.detail, model.fit_event_.detail
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            ref = SkLogisticRegression(**params).fit(X, y)
        np.testing.assert_array_equal(model.coef_, ref.coef_)


@pytest.mark.parametrize("spelling", ["int", "str", "object", "series", "category"])
def test_the_crossover_counts_classes_however_the_labels_are_spelled(spelling):
    """Where a fit runs does not depend on the labels' dtype.

    A pandas ``Series`` of class names is an object array, which the class count
    used to read as binary: a five-class fit then went to scikit-learn under the
    binary rule while the same labels as integers went to MLX.
    """
    n, d = 1000, 20
    X5, y5 = _data(n_classes=5, n_samples=n, n_features=d)
    X2, y2 = _data(n_classes=2, n_samples=n, n_features=d)
    names = np.array(["ant", "bee", "cat", "dog", "eel"], dtype=object)

    def spell(y):
        if spelling == "int":
            return y
        if spelling == "str":
            return names[y].astype(str)
        if spelling == "object":
            return names[y]
        pd = pytest.importorskip("pandas")
        return pd.Series(names[y], dtype="category" if spelling == "category" else object)

    # Thresholds that send a multinomial fit of this size to MLX and a binary one
    # to scikit-learn, as the binary row floor does at 60 000-100 000 rows.
    with tuning_context(
        logreg_min_samples=n, logreg_binary_min_samples=n + 1
    ):
        multi = PatchedLogisticRegression().fit(X5, spell(y5))
        binary = PatchedLogisticRegression().fit(X2, spell(y2))
    assert multi.execution_backend_ != "sklearn", multi.fit_event_
    assert binary.execution_backend_ == "sklearn"
    assert "binary" in binary.fit_event_.detail
    np.testing.assert_array_equal(multi.classes_, np.unique(spell(y5)))


def test_backends_agree():
    """The CPU and GPU devices run the same code on different streams."""
    X, y = _data(n_samples=3000)
    cpu = _fit("cpu", LogisticRegression(), X, y)
    gpu = _fit("mlx", LogisticRegression(), X, y)
    np.testing.assert_allclose(cpu.coef_, gpu.coef_, atol=1e-5)
    np.testing.assert_allclose(cpu.intercept_, gpu.intercept_, atol=1e-5)
    assert _objective(cpu, X, y, C=1.0) == pytest.approx(_objective(gpu, X, y, C=1.0), rel=1e-8)


def test_a_forced_mlx_fit_without_a_gpu_runs_on_mlxs_cpu(monkeypatch):
    """As the other estimators' forced fits do, unless ``device="gpu"`` was named.

    The forced backend was mapped straight to ``device="gpu"``, so on a host
    with no GPU the fit raised "device='gpu' was requested ... use device='auto'"
    under a configured ``"auto"`` or ``"cpu"`` that asked for nothing of the kind.
    """
    from mlxlearn.exceptions import DeviceError
    from mlxlearn.linear_model import _logistic

    devices = []
    original = _logistic.solve_logistic

    def spy(*args, config, **kwargs):
        devices.append(config.device)
        return original(*args, config=config, **kwargs)

    monkeypatch.setattr(_logistic, "solve_logistic", spy)
    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    X, y = _data(n_samples=400)
    for device in ("auto", "cpu"):
        with mlxlearn.config_context(device=device), tuning_context(force_backend="mlx"):
            try:
                model = LogisticRegression().fit(X, y)
            except DeviceError as exc:  # which tests/_gpu.py would report as a skip
                pytest.fail(f"device={device!r}: {exc}")
        assert (model.execution_backend_, model.fit_event_.reason) == ("mlx", "forced")
    assert devices == ["cpu", "cpu"]

    with mlxlearn.config_context(device="gpu"), tuning_context(force_backend="mlx"):
        with pytest.raises(DeviceError, match="device='gpu' was requested"):
            LogisticRegression().fit(X, y)


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


# ------------------------------------------------------------------- inference


@pytest.mark.parametrize("cls", [LogisticRegression, PatchedLogisticRegression])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_assigned_coefficients_predict_like_scikit_learn(n_classes, cls):
    """Federated averaging and model import build a model by assigning its state.

    Inference is scikit-learn's float64 code on ``coef_``, ``intercept_`` and
    ``classes_``, so there is nothing an MLX fit would have added; refusing to
    predict broke code that works on stock scikit-learn, patched or not.
    """
    X, y = _data(n_classes=n_classes, n_samples=400)
    donor = SkLogisticRegression().fit(X, y)
    labels = np.array(["a", "b", "c"][:n_classes])

    ours, stock = cls(), SkLogisticRegression()
    for model in (ours, stock):
        model.coef_ = donor.coef_.copy()
        model.intercept_ = donor.intercept_.copy()
        model.classes_ = labels

    mlxlearn.clear_backend_diagnostics()
    for method in ("predict", "predict_proba", "predict_log_proba", "decision_function"):
        np.testing.assert_array_equal(getattr(ours, method)(X), getattr(stock, method)(X))
    assert ours.score(X, labels[y]) == stock.score(X, labels[y])

    events = mlxlearn.get_backend_diagnostics()
    assert {(e.backend, e.reason) for e in events} == {("sklearn", "external-state")}
    assert not any(e.fallback for e in events)


def test_an_unfitted_model_still_raises_not_fitted():
    with pytest.raises(NotFittedError):
        LogisticRegression().predict(np.zeros((2, 3)))


def test_a_refit_that_failed_partway_does_not_predict_from_mixed_state():
    """New ``classes_`` beside the previous ``coef_`` must not produce labels."""
    X, y = _data(n_samples=400)
    model = LogisticRegression().fit(X, y)
    with pytest.raises(ValueError, match="at least 2 classes"):
        model.fit(X, np.full(len(y), 7))
    # An incomplete fit reports the estimator unfitted (BackendMixin.__sklearn_is_fitted__),
    # so this is scikit-learn's NotFittedError — which handlers written against
    # scikit-learn catch — rather than a bare MLXLearnError.
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_a_patched_refit_that_failed_in_scikit_learn_does_not_predict_either():
    """The below-crossover path lifts the in-progress marker for scikit-learn's fit."""
    X, y = _data(n_samples=400)
    model = PatchedLogisticRegression().fit(X, y)
    assert model.execution_backend_ == "sklearn"
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.fit(X_bad, y)
    with pytest.raises(NotFittedError):
        model.predict(X)


@pytest.mark.parametrize(
    "method", ["predict", "predict_proba", "predict_log_proba", "decision_function", "score"]
)
def test_one_diagnostics_event_per_public_call(method):
    """scikit-learn's inference calls back into the overridden methods.

    ``predict`` called ``decision_function``, ``predict_log_proba`` called
    ``predict_proba``, ``score`` called ``predict``, and each recorded its own
    event — two or three per call, labelled with whichever ran last. The event says
    which backend owns the model and that the arithmetic was NumPy float64.
    """
    X, y = _data(n_samples=400)
    model = _fit("cpu", LogisticRegression(), X, y)
    mlxlearn.clear_backend_diagnostics()
    args = (X, y) if method == "score" else (X,)
    getattr(model, method)(*args)

    events = mlxlearn.get_backend_diagnostics()
    assert [(e.operation, e.backend, e.reason) for e in events] == [(method, "cpu", "sticky")]
    # fit_reason: why the model was fitted where it was, as every sticky event says.
    assert events[0].extra == {"compute": "numpy-float64", "fit_reason": model.fit_event_.reason}


# ------------------------------------------------ scikit-learn's fit-time warnings


def _warning_grid():
    grid = [
        {},
        {"penalty": "l2"},
        {"penalty": None},
        {"penalty": None, "C": 2.0},
        {"C": np.inf},
        {"n_jobs": 2},
        {"l1_ratio": None},
        {"l1_ratio": 0.0},
        {"penalty": "l2", "l1_ratio": 0.5},
    ]
    if _HAS_MULTI_CLASS:
        grid.append({"multi_class": "auto"})
    return grid


def _fit_warnings(estimator, X, y):
    # The whole message: a 40-character prefix hid the removal version, which
    # scikit-learn 1.7.2 changed from 1.7 to 1.8.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X, y)
    return Counter(
        (w.category.__name__, str(w.message))
        for w in caught
        if w.category in (FutureWarning, UserWarning)
    )


@pytest.mark.parametrize("kwargs", _warning_grid(), ids=repr)
def test_fit_warns_exactly_as_scikit_learn_does(kwargs):
    """The only advance notice of a removal must not depend on which backend ran.

    scikit-learn raises its deprecation warnings from inside its own ``fit``, which
    the MLX path never calls — and a Layer 1 estimator never reaches it at all.
    Compared against the installed scikit-learn, so the test follows it across the
    supported range.
    """
    X, y = _data(n_samples=300)
    expected = _fit_warnings(SkLogisticRegression(**kwargs), X, y)
    assert _fit_warnings(LogisticRegression(**kwargs), X, y) == expected
    # The fallback path is scikit-learn's own fit: no second copy of any of them.
    assert _fit_warnings(PatchedLogisticRegression(**kwargs), X, y) == expected


# ------------------------------------------------------------------- callbacks


class _Recorder:
    """A FitCallback that counts hooks and uses every iterate it is given."""

    def __init__(self):
        self.hooks = Counter()
        self.predictions = 0

    def setup(self, estimator, context):
        self.hooks["setup"] += 1

    def on_fit_task_begin(self, estimator, context, *, X=None, y=None, metadata=None, fitted_estimator=None):
        self.hooks["begin"] += 1

    def on_fit_task_end(self, estimator, context, *, X=None, y=None, metadata=None, fitted_estimator=None):
        self.hooks["end"] += 1
        if fitted_estimator is not None:
            assert fitted_estimator.predict(X[:5]).shape == (5,)
            self.predictions += 1

    def teardown(self, estimator, context):
        self.hooks["teardown"] += 1


@pytest.mark.skipif(not _HAS_CALLBACKS, reason="scikit-learn < 1.9 has no callback API")
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("backend", BACKENDS)
def test_callbacks_see_every_iteration(backend, n_classes):
    """scikit-learn 1.9's callbacks were accepted and never called on the MLX path.

    The task tree matches scikit-learn's lbfgs path: the fit, one subtask per
    accepted iteration, and the trailing empty subtask its bridge always leaves —
    so ``n_iter_ + 2`` begin/end pairs, every iterate usable as a fitted model.
    """
    X, y = _data(n_classes=n_classes, n_samples=500, n_features=8)
    recorder = _Recorder()
    model = _fit(backend, LogisticRegression().set_callbacks(recorder), X, y)

    n_iter = int(model.n_iter_[0])
    assert recorder.hooks == Counter(setup=1, begin=n_iter + 2, end=n_iter + 2, teardown=1)
    assert recorder.predictions == n_iter + 1
    assert not hasattr(model, "_callback_fit_ctx")

    reference = _Recorder()
    stock = SkLogisticRegression().set_callbacks(reference).fit(X, y)
    assert reference.hooks["begin"] == int(stock.n_iter_[0]) + 2


class _FailingRecorder(_Recorder):
    """Raises from the per-iteration hook, which runs inside the guarded solver."""

    def __init__(self, error):
        super().__init__()
        self.error = error

    def on_fit_task_end(self, estimator, context, **kwargs):
        super().on_fit_task_end(estimator, context, **kwargs)
        if self.hooks["end"] == 2:
            raise self.error


@pytest.mark.skipif(not _HAS_CALLBACKS, reason="scikit-learn < 1.9 has no callback API")
@pytest.mark.parametrize("error", [ValueError("from the callback"), KeyError("callback")])
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_callback_error_is_raised_as_itself(backend, error):
    """Not as a BackendExecutionError: the callback is the caller's code, not MLX."""
    if backend == "mlx" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    X, y = _data(n_samples=500, n_features=8)
    model = LogisticRegression().set_callbacks(_FailingRecorder(error))
    with tuning_context(force_backend=backend), pytest.raises(type(error)) as excinfo:
        model.fit(X, y)
    assert excinfo.value is error


class _CallbackFailure(Exception):
    pass


class _ChainingRecorder(_Recorder):
    """Raises from the per-iteration hook with an explicit cause, or inside a handler."""

    def __init__(self, chain):
        super().__init__()
        self.chain = chain

    def on_fit_task_end(self, estimator, context, **kwargs):
        super().on_fit_task_end(estimator, context, **kwargs)
        if self.hooks["end"] != 2:
            return
        if self.chain == "cause":
            raise _CallbackFailure("from the callback") from OSError("root cause")
        try:
            {}["missing"]
        except KeyError:
            raise _CallbackFailure("from the callback")  # noqa: B904


@pytest.mark.skipif(not _HAS_CALLBACKS, reason="scikit-learn < 1.9 has no callback API")
@pytest.mark.parametrize("chain", ["cause", "context"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_callback_error_keeps_its_chain_and_lets_the_fit_go(backend, chain):
    """Its own ``__cause__`` and ``__context__`` survive, as with scikit-learn, and
    the failed fit's MLX arrays are freed without waiting for the garbage collector.

    It was re-raised inside the handler for the ``BackendExecutionError`` around it,
    which replaced both with that error; and its traceback ran through
    ``mlx_guard``'s generator frame, whose caller's frame held the error itself, a
    cycle that kept the solver's frame and its device copy of ``X`` alive.
    """
    import gc

    import mlx.core as mx

    if backend == "mlx" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    active_memory = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50_000, 16), dtype=np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    model = LogisticRegression().set_callbacks(_ChainingRecorder(chain))

    gc.collect()
    gc.disable()
    try:
        before = active_memory()
        with tuning_context(force_backend=backend):
            try:
                model.fit(X, y)
            except _CallbackFailure as error:
                # Types only: a raised context's traceback holds the fit's frames
                # for as long as it is referenced, cycle or not.
                chained = (
                    type(error.__cause__), type(error.__context__), error.__suppress_context__
                )
        held = active_memory() - before
    finally:
        gc.enable()

    assert held < X.nbytes // 2, f"{held} bytes of MLX memory outlived the failed fit"
    if chain == "cause":
        assert chained == (OSError, type(None), True)
    else:
        assert chained == (type(None), KeyError, False)


@pytest.mark.skipif(not _HAS_CALLBACKS, reason="scikit-learn < 1.9 has no callback API")
@pytest.mark.parametrize("tuning", [{}, _NO_CROSSOVER], ids=["below", "above"])
def test_patched_callbacks_fire_on_both_sides_of_the_crossover(tuning):
    X, y = _data(n_samples=500, n_features=6)
    recorder = _Recorder()
    with tuning_context(**tuning):
        model = PatchedLogisticRegression().set_callbacks(recorder).fit(X, y)
    assert recorder.hooks["setup"] == recorder.hooks["teardown"] == 1
    assert recorder.hooks["begin"] == int(model.n_iter_[0]) + 2
    assert recorder.predictions == int(model.n_iter_[0]) + 1


@pytest.mark.parametrize(
    "y",
    [
        np.array([3, 1, 3, 3, 7, 1]),
        np.array([2.5, -1.0, 2.5, 0.0, -0.0]),
        np.array([True, False, False, True]),
        np.array(["b", "a", "c", "a"]),
        np.array(["b", "a", "b"], dtype=object),
    ],
    ids=["int", "float", "bool", "str", "object"],
)
def test_labels_are_encoded_as_numpy_unique_encodes_them(y):
    from mlxlearn.linear_model._logistic import _encode_labels

    classes, codes = _encode_labels(y)
    expected_classes, expected_codes = np.unique(y, return_inverse=True)
    np.testing.assert_array_equal(classes, expected_classes)
    np.testing.assert_array_equal(codes, expected_codes)


def test_separable_classes_converge_in_scikit_learns_one_iteration():
    """Well-separated blobs, benchmarks/run.py's multiclass data: scikit-learn's first
    L-BFGS-B step, a unit move, already meets its tolerance. The solver's first trial
    step is that one now; the 1/‖d‖₁ it replaced started √p times shorter and took
    11-15 iterations to get there."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, size=6000)
    X = rng.normal(scale=2.0, size=(5, 256))[y] + rng.normal(size=(6000, 256))
    with tuning_context(force_backend="mlx"):
        mlx = LogisticRegression().fit(X, y)
    stock = SkLogisticRegression().fit(X, y)
    assert int(mlx.n_iter_[0]) <= int(stock.n_iter_[0]) + 1 <= 3

    def objective(model):
        return reference_objective(model.coef_, model.intercept_, X, y, n_classes=5, C=1.0)

    assert objective(mlx) <= objective(stock) * (1 + 1e-3)


@pytest.mark.skipif(not _HAS_CALLBACKS, reason="scikit-learn < 1.9 has no callback API")
def test_scoring_monitor_gets_a_score_per_iteration():
    from sklearn.callback import ScoringMonitor

    X, y = _data(n_samples=500, n_features=6)
    monitor = ScoringMonitor(scoring="accuracy")
    model = LogisticRegression().set_callbacks(monitor).fit(X, y)
    assert model.execution_backend_ == "cpu"
    assert monitor.get_logs() is not None
