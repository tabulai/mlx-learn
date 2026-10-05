"""The dense L-BFGS preconditioner against stock scikit-learn and a float64 oracle.

``docs/fp32_policy.md`` §2.3 describes the preconditioner: where a gate on a sample of
the rows finds correlated features and no nearly separable classes, L-BFGS starts every
iteration from the inverse of the loss's curvature at the zero start. It changes the
search direction and nothing else, so every fit here is held to §3 exactly as any other
fit is — converged fits only, the objective as the strong assertion — with the
diagonal path (``Tuning.lbfgs_dense_preconditioner`` off) beside it.

One allowance is §3's own, made explicit. On an ill-conditioned problem at the default
``tol`` a fit stops wherever ``max|g| ≤ tol`` first holds, and two correct solvers stop
at measurably different objectives; §3 states its tolerances for converged fits. Where
a strict comparison with scikit-learn misses, the fit passes only if it is at least as
close to the float64 optimum (:func:`tests._helpers.logistic_newton_oracle`) as both
scikit-learn and the diagonal path are — the arbiter neither solver can grade itself
by. Every case also asserts which path ran.

The grid includes the data a sample of the rows reads wrong — rare one-hot levels,
rows ordered by source or by label, one column on a scale of its own, a few rows
carrying most of the weight — where the model, built from at most 8 192 rows, once
stepped orders of magnitude too far or read the separation of the data's head alone.
No case here is separable in §3.3's sense: separable classes never reach the dense
path, because the separation limit turns them away, and those cases assert that the
diagonal path ran, bit for bit.

The default run takes a fast subset; ``pytest -m slow`` the whole grid, and the
routed-size cells, among them the ten-class digits the gate turns away and the laws
a sample reads wrong at ``C`` from 1 to ``inf``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._solvers.logistic import reference_objective
from mlxlearn.linear_model import LogisticRegression
from tests._helpers import logistic_data, logistic_newton_oracle

# docs/fp32_policy.md §3
OBJECTIVE_RTOL = 1e-5
COEF_ATOL, COEF_RTOL = 2e-3, 5e-3
PROBA_ATOL = 2e-3
MIN_AGREEMENT = 0.995


@dataclass(frozen=True)
class Case:
    name: str
    law: str
    n: int
    d: int
    K: int = 2
    C: float = 1.0
    tol: float = 1e-4
    fit_intercept: bool = True
    weights: str = ""  # "", "sample", "class", "heavy" or "heavy-many"
    order: str = ""  # "" or "label": the rows sorted by class
    expect: str = "correlated"  # the preconditioner_reason the fit must report

    def __str__(self) -> str:
        return self.name


FAST = [
    Case("ar90-binary", "ar90", 10000, 32),
    Case("ar99-binary-Cinf-tol6", "ar99", 10000, 16, C=math.inf, tol=1e-6),
    Case("ar99-K5-tol6", "ar99", 10000, 24, K=5, tol=1e-6),
    Case("onehot-binary-Cinf", "onehot", 10000, 68, C=math.inf),
    Case("ar90-binary-sample-weights", "ar90", 10000, 32, weights="sample"),
    Case("ar90-K3-class-weights", "ar90", 10000, 32, K=3, weights="class"),
    Case("ar90-binary-no-intercept", "ar90", 10000, 32, fit_intercept=False),
    Case("lowrank20-K20", "lowrank20", 20000, 32, K=20),
    Case("colscale-K3-tol6", "colscale", 10000, 16, K=3, tol=1e-6),
    Case("breast_cancer", "breast_cancer", 569, 30),
    Case("digits-K3", "digits", 1797, 64, K=3),
    Case("digits-K10", "digits", 1797, 64, K=10, expect="separated"),
    Case("blobs-K3", "blobs", 10000, 16, K=3, expect="uncorrelated"),
    Case("blobsar-K3", "blobsar", 10000, 16, K=3, expect="separated"),
    Case("iso-binary", "iso", 10000, 32, expect="uncorrelated"),
    # Data a sample of the rows reads wrong, at sizes where it is a sample.
    Case("rare-binary-C1e4", "rare", 12000, 116, C=1e4),
    Case("twosource-K3-16383", "twosource", 16383, 16, K=3),
    Case("digits-K3-label-sorted-16383", "digits", 16383, 64, K=3, order="label"),
    Case("blobsar-binary-label-sorted-16383", "blobsar", 16383, 32, order="label",
         expect="separated"),
    Case("dominant-K3", "dominant", 10000, 16, K=3),
]

FULL = [
    Case("ar90-binary-tol6", "ar90", 20000, 32, tol=1e-6),
    Case("ar99-binary-C1e-4", "ar99", 10000, 16, C=1e-4),
    Case("ar99-binary-C1e4", "ar99", 10000, 16, C=1e4),
    Case("ar99-binary-Cinf", "ar99", 10000, 16, C=math.inf),
    Case("lowrank5-binary", "lowrank5", 10000, 32),
    Case("lowrank20-binary-tol6", "lowrank20", 10000, 48, tol=1e-6),
    Case("ar90-K3", "ar90", 10000, 32, K=3),
    Case("lowrank5-K5", "lowrank5", 10000, 32, K=5),
    Case("colscale-binary", "colscale", 10000, 16),
    Case("colscale-binary-C100-tol6", "colscale", 10000, 16, C=100.0, tol=1e-6),
    Case("onehot-binary", "onehot", 10000, 68),
    Case("onehot-K3-tol6", "onehot", 10000, 68, K=3, tol=1e-6),
    Case("onehot-binary-C1e4-tol6", "onehot", 10000, 68, C=1e4, tol=1e-6),
    Case("onehot-binary-Cinf-tol6", "onehot", 10000, 68, C=math.inf, tol=1e-6),
    Case("ar99-K5-C1e-4", "ar99", 10000, 24, K=5, C=1e-4),
    Case("ar99-K5-C1e4", "ar99", 10000, 24, K=5, C=1e4),
    Case("ar99-K3-Cinf-tol6", "ar99", 10000, 24, K=3, C=math.inf, tol=1e-6),
    Case("ar90-K5-sample-weights-no-intercept-tol6", "ar90", 10000, 32, K=5,
         weights="sample", fit_intercept=False, tol=1e-6),
    Case("ar99-binary-class-weights-Cinf-tol6", "ar99", 10000, 16, weights="class",
         C=math.inf, tol=1e-6),
    Case("ar90-binary-sample-weights-tol6", "ar90", 10000, 32, weights="sample", tol=1e-6),
    Case("ar99-K3-no-intercept-tol6", "ar99", 10000, 16, K=3, fit_intercept=False, tol=1e-6),
    Case("random-binary", "random", 20000, 32),
    Case("random-K5", "random", 20000, 32, K=5),
    Case("lowrank20-K20-tol6", "lowrank20", 20000, 32, K=20, tol=1e-6),
    Case("breast_cancer-upsampled", "breast_cancer", 20000, 30),
    Case("breast_cancer-tol6", "breast_cancer", 569, 30, tol=1e-6),
    Case("digits-binary-upsampled", "digits", 20000, 64),
    Case("digits-K3-upsampled-tol6", "digits", 20000, 64, K=3, tol=1e-6),
    Case("blobs-K10-tol6", "blobs", 10000, 16, K=10, tol=1e-6, expect="uncorrelated"),
    Case("blobsar-K10-tol6", "blobsar", 10000, 16, K=10, tol=1e-6, expect="separated"),
    Case("blobsar-K5-C1e4", "blobsar", 10000, 32, K=5, C=1e4, expect="separated"),
    Case("blobsar-binary", "blobsar", 10000, 32, expect="separated"),
    Case("iso-K3-tol6", "iso", 10000, 16, K=3, tol=1e-6, expect="uncorrelated"),
    Case("twosource-binary-12000", "twosource", 12000, 16),
    Case("dominant-binary-Cinf", "dominant", 10000, 16, C=math.inf),
    Case("ar90-K3-heavy-weights", "ar90", 10000, 16, K=3, weights="heavy", expect="separated"),
    Case("ar90-K3-heavy-many-weights", "ar90", 10000, 16, K=3, weights="heavy-many"),
]

#: Sizes patched dispatch sends to MLX: multinomial from 60 000 rows, binary from
#: 200 000 at up to 64 features. The ten-class digits are the cells the gate's
#: separation limit was set for (35-42 evaluations became 65-116 without it).
ROUTED = [
    Case("digits-K10-60000", "digits", 60000, 64, K=10, expect="separated"),
    Case("digits-K10-200000", "digits", 200000, 64, K=10, expect="separated"),
    Case("digits-K3-60000", "digits", 60000, 64, K=3),
    Case("ar90-K10-60000", "ar90", 60000, 64, K=10),
    Case("lowrank20-K3-60000", "lowrank20", 60000, 64, K=3),
    Case("onehot-K3-60000", "onehot", 60000, 64, K=3),
    Case("ar99-binary-200000", "ar99", 200000, 16),
    Case("colscale-binary-200000", "colscale", 200000, 64),
    Case("breast_cancer-200000", "breast_cancer", 200000, 30),
    Case("twosource-K3-65535", "twosource", 65535, 16, K=3),
    Case("digits-K3-label-sorted-65535", "digits", 65535, 64, K=3, order="label"),
]

#: The laws a sample of the rows reads wrong, at routed sizes and C from 1 to inf: rare
#: one-hot levels (16 numerics, 200 Zipf levels), one column in units of 1e8, and a few
#: rows of 60 000 carrying most of the weight.
ROUTED_SAMPLED = [
    Case(f"{name}-C{C:g}", law, 60000, d, K=3, C=C, weights=weights, expect=expect)
    for name, law, d, weights, expect in (
        ("rare-K3-60000", "rare", 216, "", "correlated"),
        ("dominant-K3-60000", "dominant", 16, "", "correlated"),
        ("ar90-K3-60000-heavy-weights", "ar90", 16, "heavy", "separated"),
        ("ar90-K3-60000-heavy-many-weights", "ar90", 16, "heavy-many", "correlated"),
    )
    for C in (1.0, 100.0, 1e4, math.inf)
]


def _data(case: Case):
    X, y = logistic_data(case.law, case.n, case.d, case.K, seed=sum(map(ord, case.name)) % 1000)
    sw = cw = None
    if case.weights == "sample":
        sw = np.random.default_rng(1).exponential(size=len(y)) + 0.05
    elif case.weights == "class":
        cw = "balanced"
    elif case.weights.startswith("heavy"):
        # A few rows carry most of the weight, and the curvature is mostly theirs: a
        # sample of 8 192 spread over the rows would mostly miss them, so it is spread
        # over the weight. Ten of them, unlike the rest, with 90% of it make the
        # weighted classes nearly separable — ten points in sixteen dimensions — and
        # the separation limit turns that fit away; three hundred of the data's own
        # rows, three times as far out, with 75% of it do not.
        count, share = (10, 0.9) if case.weights == "heavy" else (300, 0.75)
        rng = np.random.default_rng(2)
        heavy = rng.choice(len(y), size=count, replace=False)
        X = X.copy()
        if case.weights == "heavy":
            X[heavy] = 3.0 * rng.normal(size=(count, X.shape[1]))
        else:
            X[heavy] *= 3.0
        sw = np.ones(len(y))
        sw[heavy] = share / (1.0 - share) * (len(y) - count) / count
    if case.order == "label":
        order = np.argsort(y, kind="stable")
        X, y = X[order], y[order]
        sw = None if sw is None else sw[order]
    return X, y, sw, cw


def _effective_weights(y, sw, cw):
    if cw is None:
        return sw
    from sklearn.utils.class_weight import compute_class_weight

    classes, codes = np.unique(y, return_inverse=True)
    base = np.ones(len(y)) if sw is None else sw
    return base * compute_class_weight(cw, classes=classes, y=y, sample_weight=base)[codes]


def _fit(case, X, y, sw, cw, *, dense, device="gpu"):
    if device == "gpu" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    backend = "mlx" if device == "gpu" else "cpu"
    with tuning_context(force_backend=backend, lbfgs_dense_preconditioner=dense), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LogisticRegression(
            C=case.C, tol=case.tol, max_iter=2000, fit_intercept=case.fit_intercept,
            class_weight=cw,
        ).fit(X, y, sample_weight=sw)
    assert model.execution_backend_ == backend
    return model


def _stock(case, X, y, sw, cw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SkLogisticRegression(
            C=case.C, tol=case.tol, max_iter=20000, fit_intercept=case.fit_intercept,
            class_weight=cw,
        ).fit(X, y, sample_weight=sw)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


def _converged(model) -> bool:
    """§3's precondition, as the fit records it: tol reached, or the float32 floor."""
    extra = model.fit_event_.extra
    return "stop_reason" not in extra or (
        extra["stop_reason"] == "float32-floor" and extra["at_resolution_floor"]
    )


def _proba(coef, intercept, X, K):
    raw = np.asarray(X, dtype=np.float64) @ np.atleast_2d(coef).T + np.ravel(intercept)
    if K == 2:
        p1 = 0.5 * (1.0 + np.tanh(0.5 * raw[:, 0]))
        return np.column_stack([1.0 - p1, p1])
    raw -= raw.max(axis=1, keepdims=True)
    e = np.exp(raw)
    return e / e.sum(axis=1, keepdims=True)


def _assert_policy(case, X, y, sw, cw, *, device="gpu"):
    """One case against §3: the preconditioned fit, the diagonal one, stock, the oracle."""
    K = len(np.unique(y))
    codes = np.searchsorted(np.unique(y), y)
    swf = _effective_weights(y, sw, cw)

    model = _fit(case, X, y, sw, cw, dense=True, device=device)
    diagonal = _fit(case, X, y, sw, cw, dense=False, device=device)
    extra = model.fit_event_.extra
    assert extra["preconditioner_reason"] == case.expect, extra
    assert extra["preconditioner"] == ("dense" if case.expect == "correlated" else "diagonal")
    assert diagonal.fit_event_.extra["preconditioner_reason"] == "disabled"
    if case.expect not in ("correlated", "fallback"):
        # The gate turned it away: the diagonal path, bit for bit.
        np.testing.assert_array_equal(model.coef_, diagonal.coef_)
        np.testing.assert_array_equal(model.intercept_, diagonal.intercept_)
        assert extra["n_fev"] == diagonal.fit_event_.extra["n_fev"]
        return
    assert _converged(model), extra

    def objective(m):
        return reference_objective(m.coef_, m.intercept_, X, codes, n_classes=K, C=case.C,
                                   sample_weight=swf)

    stock, stock_converged = _stock(case, X, y, sw, cw)
    coef_star, intercept_star = logistic_newton_oracle(
        X, codes, K, C=case.C, sample_weight=swf, fit_intercept=case.fit_intercept
    )
    j, j_diag, j_sk = objective(model), objective(diagonal), objective(stock)
    j_star = reference_objective(coef_star, intercept_star, X, codes, n_classes=K, C=case.C,
                                 sample_weight=swf)
    j_star = min(j_star, j, j_diag, j_sk)  # the oracle cannot be beaten, only matched

    def gap(value):
        return (value - j_star) / abs(j_star)

    # Never further from the optimum than both scikit-learn and the diagonal path.
    explained = gap(j) <= max(gap(j_sk), gap(j_diag)) + 1e-6
    detail = (f"J {j:.10g}, stock {j_sk:.10g}, diagonal {j_diag:.10g}, optimum {j_star:.10g}; "
              f"n_fev {extra['n_fev']} (diagonal {diagonal.fit_event_.extra['n_fev']})")
    if stock_converged:
        assert j <= j_sk * (1 + OBJECTIVE_RTOL) or explained, detail
    else:
        assert explained, detail

    P_star = _proba(coef_star, intercept_star, X, K)
    P, P_sk = model.predict_proba(X), stock.predict_proba(X)
    agreement = np.mean(model.predict(X) == stock.predict(X))
    oracle_agreement = np.mean(P.argmax(1) == P_star.argmax(1))
    assert agreement >= MIN_AGREEMENT or oracle_agreement >= min(
        MIN_AGREEMENT, np.mean(P_sk.argmax(1) == P_star.argmax(1))
    ), f"labels {agreement:.4f} vs stock, {oracle_agreement:.4f} vs the optimum"

    def further(estimate, reference):
        return np.max(np.abs(estimate - P_star)) > np.max(np.abs(reference - P_star))

    # Probabilities and coefficients at tol=1e-6, where §3 states them for any
    # converged problem; at the default tol it is the objective that judges an
    # ill-conditioned one — there the three solvers' coefficients were measured up to
    # 2.6e-2 from the optimum and from each other (one-hot, 60 000 rows, 3 classes).
    if case.tol <= 1e-6 and stock_converged:
        dp = np.max(np.abs(P - P_sk))
        assert dp <= PROBA_ATOL or not further(P, P_sk), f"max|Δp| {dp:.2e} vs stock"
        # Without a penalty a rank-deficient design (one-hot blocks) has no unique
        # coefficients, only unique probabilities: each solver keeps the null-space
        # component its own path gives it.
        if math.isfinite(case.C) or case.law != "onehot":
            coef_ok = np.allclose(model.coef_, stock.coef_, atol=COEF_ATOL, rtol=COEF_RTOL)
            closer = np.max(np.abs(model.coef_ - coef_star)) <= max(
                np.max(np.abs(stock.coef_ - coef_star)),
                np.max(np.abs(diagonal.coef_ - coef_star)),
            )
            assert coef_ok or closer, (
                f"coef max|Δ| {np.max(np.abs(model.coef_ - stock.coef_)):.2e} vs stock"
            )


@pytest.mark.parametrize("case", FAST, ids=str)
def test_preconditioned_fits_meet_the_policy(case):
    _assert_policy(case, *_data(case))


@pytest.mark.parametrize(
    "case", [c for c in FAST if c.name in ("ar99-K5-tol6", "ar90-binary-sample-weights",
                                           "breast_cancer", "digits-K10")], ids=str
)
def test_preconditioned_fits_meet_the_policy_on_the_cpu_device(case):
    _assert_policy(case, *_data(case), device="cpu")


@pytest.mark.slow
@pytest.mark.parametrize("case", FULL, ids=str)
def test_preconditioned_fits_meet_the_policy_full_grid(case):
    _assert_policy(case, *_data(case))


@pytest.mark.slow
@pytest.mark.parametrize("case", ROUTED, ids=str)
def test_preconditioned_fits_meet_the_policy_at_routed_sizes(case):
    _assert_policy(case, *_data(case))


@pytest.mark.slow
@pytest.mark.parametrize("case", ROUTED_SAMPLED, ids=str)
def test_fits_a_sample_reads_wrong_meet_the_policy_at_routed_sizes(case):
    _assert_policy(case, *_data(case))


def test_float32_input_takes_the_same_path_to_the_same_answer():
    """float32 input is not upcast; it is centered in float32 (``_prepare_design``) and
    its model sample comes from the same float32 design, so the decision is the same
    and the answer agrees with the float64 input's to §3."""
    case = Case("ar99-binary", "ar99", 10000, 16)
    X, y, _, _ = _data(case)
    m64 = _fit(case, X, y, None, None, dense=True)
    m32 = _fit(case, X.astype(np.float32), y, None, None, dense=True)
    assert m32.fit_event_.extra["preconditioner"] == "dense"
    j64 = reference_objective(m64.coef_, m64.intercept_, X, y, n_classes=2)
    j32 = reference_objective(m32.coef_, m32.intercept_, X, y, n_classes=2)
    assert j32 == pytest.approx(j64, rel=OBJECTIVE_RTOL)
    np.testing.assert_allclose(m32.coef_, m64.coef_, atol=COEF_ATOL, rtol=COEF_RTOL)
    assert np.mean(m32.predict(X) == m64.predict(X)) >= MIN_AGREEMENT
