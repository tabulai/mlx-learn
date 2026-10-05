"""Multiclass rbf ``SVC`` on the GPU one-vs-one kernel, against stock scikit-learn.

Every case here runs every one-vs-one pair in the kernel of ``_kernels/_ovo.py`` — the
test asserts it from the fit's event — and holds the model to ``docs/fp32_policy.md``
§3: labels, one-vs-one decision values, one-vs-rest values where no vote flips,
support-vector sets, each pair's dual objective, and the KKT conditions at
mlxlearn's own solution in float64, which appeals to nobody. Classes from 3 to 26,
boxes from 0.1 to 100, ``gamma`` resolved four ways, class and sample weights (zero
weights included), imbalanced classes, both decision shapes and ``break_ties``.

``test_a_soak_of_random_fits`` (slow) draws 200 more and holds each to the host
path's model, the KKT conditions and determinism, and to scikit-learn on every tenth.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC as SkSVC

from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._kernels import _metal
from mlxlearn.exceptions import UnsupportedParameterError
from mlxlearn.patching._estimators import SVC as PatchedSVC
from mlxlearn.svm import SVC

DECISION_ATOL = 2e-2
DECISION_RTOL = 5e-3
LABELS = 0.995
SV_SYMMETRIC_DIFFERENCE = 0.05
DUAL_RTOL = 1e-5
TOL = 1e-3

KERNEL = gpu_available() and _metal.mlx_version()[:2] >= _metal.MIN_MLX
needs_kernel = pytest.mark.skipif(
    not KERNEL, reason="the one-vs-one kernel needs the GPU and MLX 0.32"
)
pytestmark = pytest.mark.gpu


def _data(kind="blobs", n=1200, d=8, K=4, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "random":
        return rng.normal(size=(n, d)), rng.integers(0, K, size=n)
    if kind == "imbalanced":  # 10:1 between the largest class and the smallest
        w = np.geomspace(10.0, 1.0, K)
        y = rng.choice(K, size=n, p=w / w.sum())
    else:
        y = rng.integers(0, K, size=n)
    if kind == "lowrank":  # five latent directions, classes cut from the first
        Z = rng.normal(size=(n, 5))
        X = Z @ rng.normal(size=(5, d)) + 0.3 * rng.normal(size=(n, d))
        s = Z[:, 0] + rng.normal(size=n)
        return X, np.digitize(s, np.quantile(s, np.linspace(0, 1, K + 1)[1:-1]))
    centers = rng.normal(scale=2.0, size=(K, d)) / np.sqrt(d) * 3
    return centers[y] + rng.normal(size=(n, d)), y


def _fits(X, y, sample_weight=None, **params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        with tuning_context(force_backend="mlx"):
            mine = SVC(**params).fit(X, y, sample_weight=sample_weight)
        theirs = SkSVC(**params).fit(X, y, sample_weight=sample_weight)
    return mine, theirs


def _values(model, X, shape):
    saved = model.decision_function_shape
    model.decision_function_shape = shape
    try:
        return np.asarray(model.decision_function(X), dtype=np.float64)
    finally:
        model.decision_function_shape = saved


def _pairs(model, X, y, support):
    """Per pair: ``(coef, labels, rows)`` from the public ``dual_coef_`` block layout,
    with ``support`` the support vectors' rows in ``X``."""
    encoded = np.searchsorted(model.classes_, y)
    offsets = np.concatenate([[0], np.cumsum(model.n_support_)])
    K = len(model.classes_)
    out = []
    for i in range(K):
        for j in range(i + 1, K):
            rows = np.concatenate([np.flatnonzero(encoded == i), np.flatnonzero(encoded == j)])
            coef = np.zeros(len(X))
            si, sj = slice(offsets[i], offsets[i + 1]), slice(offsets[j], offsets[j + 1])
            coef[support[si]] = model.dual_coef_[j - 1, si]
            coef[support[sj]] = model.dual_coef_[i, sj]
            out.append((coef[rows], np.where(encoded[rows] == i, 1.0, -1.0), rows))
    return out


def _box(model, y, sample_weight):
    box = model.C * model.class_weight_[np.searchsorted(model.classes_, y)]
    return box if sample_weight is None else box * np.maximum(sample_weight, 0.0)


def assert_ovo_parity(mine, theirs, X, y, sample_weight=None, tol=TOL, curated=True):
    """``docs/fp32_policy.md`` §3, pair by pair, for a model the kernel fitted.

    ``curated`` adds the support-vector sets and each pair's dual objective, which the
    curated cases hold to 5% and 1e-5; the soak's draws include problems — two
    features at a large box — where today's host path itself is 30% and 3.7e-5 from
    LIBSVM there, so it holds those to the host path instead.
    """
    pairs = len(mine.classes_) * (len(mine.classes_) - 1) // 2
    extra = mine.fit_event_.extra
    assert extra["ovo_kernel_pairs"] == pairs, extra
    assert extra["ovo_resolved_non_finite"] == 0, extra

    assert np.mean(mine.predict(X) == theirs.predict(X)) >= LABELS

    ovo_mine, ovo_theirs = _values(mine, X, "ovo"), _values(theirs, X, "ovo")
    np.testing.assert_allclose(ovo_mine, ovo_theirs, atol=DECISION_ATOL, rtol=DECISION_RTOL)
    # One-vs-rest values count votes: where a vote flips — an ovo value inside the
    # tolerance of zero — they differ by one, legitimately. Elsewhere they are held to
    # the ovo tolerance, which their confidence term (1/3-Lipschitz) inherits.
    flipped = ((ovo_mine > 0) != (ovo_theirs > 0)).any(axis=1)
    ovr_mine, ovr_theirs = _values(mine, X, "ovr"), _values(theirs, X, "ovr")
    np.testing.assert_allclose(
        ovr_mine[~flipped], ovr_theirs[~flipped], atol=DECISION_ATOL, rtol=DECISION_RTOL
    )
    assert flipped.mean() <= 1 - LABELS

    if curated:
        # scikit-learn's support_ indexes the rows a zero weight leaves; mlxlearn's, X.
        kept = np.arange(len(X)) if sample_weight is None else np.flatnonzero(sample_weight > 0)
        support_theirs = kept[theirs.support_]
        only = set(mine.support_.tolist()) ^ set(support_theirs.tolist())
        assert len(only) <= SV_SYMMETRIC_DIFFERENCE * max(len(support_theirs), 1)
        gamma = mine._gamma
        for (c_mine, _, rows), (c_theirs, _, _) in zip(
            _pairs(mine, X, y, mine.support_), _pairs(theirs, X, y, support_theirs), strict=True
        ):
            K = rbf_kernel(X[rows], gamma=gamma)
            dual = [0.5 * c @ K @ c - np.abs(c).sum() for c in (c_mine, c_theirs)]
            np.testing.assert_allclose(dual[0], dual[1], rtol=DUAL_RTOL)
    violation = _kkt_violation(mine, X, y, sample_weight, ovo_mine)
    assert violation <= 2 * tol, f"KKT violation {violation:.2e}"


def _kkt_violation(model, X, y, sample_weight, decision):
    """The largest KKT violation of any pair, measured as ``docs/fp32_policy.md`` §3
    states it: each training row's margin ``y·f`` on its pair's decision value —
    ``α = 0`` needs ``≥ 1``, ``α = C`` needs ``≤ 1``, a free ``α`` needs ``= 1`` — and
    the box and equality constraints on the way."""
    box = _box(model, y, sample_weight)
    worst = 0.0
    for k, (c, labels, rows) in enumerate(_pairs(model, X, y, model.support_)):
        alpha, C = c * labels, box[rows]
        assert np.all(alpha >= 0.0) and np.all(alpha <= C)
        assert abs(float(c.sum())) <= 1e-9 * max(1.0, float(C.sum()))
        margin = labels * decision[rows, k]
        live = C > 0.0
        violation = np.concatenate([
            (1.0 - margin)[live & (alpha <= 0.0)],
            (margin - 1.0)[live & (alpha >= C)],
            np.abs(margin - 1.0)[live & (alpha > 0.0) & (alpha < C)],
        ])
        worst = max(worst, float(violation.max(initial=0.0)))
    return worst


CASES = {
    # classes
    "K3": ({"K": 3}, {}),
    "K5": ({"K": 5, "n": 1500}, {}),
    "K8": ({"K": 8, "n": 1600}, {}),
    "K10": ({"K": 10, "n": 2000, "d": 16}, {}),
    "K16": ({"K": 16, "n": 2400, "d": 32}, {}),
    "K26": ({"K": 26, "n": 2600, "d": 16}, {}),
    # boxes
    "C0.1": ({"K": 5}, {"C": 0.1}),
    "C10": ({"K": 5}, {"C": 10.0}),
    "C100": ({"K": 4, "kind": "random", "n": 800}, {"C": 100.0}),
    # gamma
    "gamma-auto": ({"K": 4}, {"gamma": "auto"}),
    "gamma-0.05": ({"K": 4, "d": 16}, {"gamma": 0.05}),
    "gamma-3x-scale": ({"K": 4}, {"gamma": "3x"}),
    "gamma-0.3x-scale-C10": ({"K": 4}, {"gamma": "0.3x", "C": 10.0}),
    # data
    "random-labels": ({"K": 4, "kind": "random", "d": 32}, {}),
    "low-rank": ({"K": 4, "kind": "lowrank", "d": 64}, {}),
    "imbalanced": ({"K": 5, "kind": "imbalanced", "n": 1500}, {}),
    # weights
    "balanced": ({"K": 5, "kind": "imbalanced", "n": 1500}, {"class_weight": "balanced"}),
    "class-weight-dict": ({"K": 4}, {"class_weight": {0: 5.0, 1: 0.5, 2: 1.0, 3: 2.0}}),
    "sample-weight": ({"K": 4}, {"sample_weight": "uniform"}),
    "sample-weight-zeros": ({"K": 4}, {"sample_weight": "zeros"}),
    # stopping
    "tol-1e-5": ({"K": 4}, {"tol": 1e-5}),
}


def _resolve(params, X, seed):
    params = dict(params)
    gamma = params.get("gamma")
    if isinstance(gamma, str) and gamma.endswith("x"):
        params["gamma"] = float(gamma[:-1]) / (X.shape[1] * X.var())
    sample_weight = params.pop("sample_weight", None)
    rng = np.random.default_rng(seed + 7)
    if sample_weight == "uniform":
        sample_weight = rng.uniform(0.5, 2.0, len(X))
    elif sample_weight == "zeros":
        sample_weight = rng.uniform(0.5, 2.0, len(X))
        sample_weight[rng.random(len(X)) < 0.2] = 0.0
    return params, sample_weight


@pytest.mark.parametrize("case", list(CASES))
@needs_kernel
def test_parity_with_scikit_learn(case):
    data, params = CASES[case]
    X, y = _data(**data)
    params, sample_weight = _resolve(params, X, seed=len(case))
    mine, theirs = _fits(X, y, sample_weight, **params)
    assert_ovo_parity(mine, theirs, X, y, sample_weight, tol=params.get("tol", TOL))


@pytest.mark.parametrize("shape", ["ovr", "ovo"])
@needs_kernel
def test_decision_function_shape_and_break_ties(shape):
    X, y = _data(K=5, n=1500)
    mine, theirs = _fits(X, y, decision_function_shape=shape)
    assert mine.decision_function(X).shape == theirs.decision_function(X).shape
    assert_ovo_parity(mine, theirs, X, y)
    if shape == "ovr":
        mine.break_ties = theirs.break_ties = True
        assert np.mean(mine.predict(X) == theirs.predict(X)) >= LABELS
    else:
        mine.break_ties = True
        with pytest.raises(ValueError, match="break_ties"):
            mine.predict(X)


@needs_kernel
def test_a_pair_with_no_free_dual_variable_takes_libsvms_bias(monkeypatch):
    """Random labels at a small box: every dual variable of every pair ends at a bound,
    so the problem leaves the bias anywhere in an interval, and LIBSVM takes it from a
    sample its rounding left a hair inside a bound. The kernel's double-float state
    rounds elsewhere; read off it, the biases were up to 0.86 from scikit-learn's.
    Each such pair is solved again by the host loop, which rounds as LIBSVM does."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(900, 32))
    y = rng.choice(4, size=900, p=[0.3, 0.27, 0.23, 0.2])
    params = {"C": 0.2, "class_weight": "balanced"}
    mine, theirs = _fits(X, y, **params)
    assert mine.fit_event_.extra["ovo_resolved_no_free"] == 6
    np.testing.assert_allclose(mine._pair_rho_, -theirs.intercept_, atol=1e-5)
    assert_ovo_parity(mine, theirs, X, y)
    with tuning_context(force_backend="mlx", metal_kernels=False):
        host = SVC(**params).fit(X, y)
    np.testing.assert_array_equal(mine._pair_rho_, host._pair_rho_)
    np.testing.assert_array_equal(mine.dual_coef_, host.dual_coef_)

    from mlxlearn.svm import _classes

    monkeypatch.setattr(_classes, "_has_free", lambda alpha, C: True)
    with tuning_context(force_backend="mlx"):
        unruled = SVC(**params).fit(X, y)
    assert np.abs(unruled._pair_rho_ + theirs.intercept_).max() > 0.1, "lost its premise"


@needs_kernel
def test_probability_true_never_reaches_the_kernel(monkeypatch):
    """Layer 1 refuses it before any device work; patched mode hands the fit over."""
    from mlxlearn._kernels import _ovo

    launched = []
    monkeypatch.setattr(_ovo.PairSolver, "solve", lambda *a, **k: launched.append(a))
    X, y = _data(K=3, n=600)
    with tuning_context(force_backend="mlx"), pytest.raises(UnsupportedParameterError):
        SVC(probability=True).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PatchedSVC(probability=True, random_state=0).fit(X, y)
    assert model.execution_backend_ == "sklearn" and launched == []
    assert model.predict_proba(X[:5]).shape == (5, 3)


def _soak_case(rng):
    kind = str(rng.choice(["blobs", "random", "imbalanced", "lowrank"]))
    K = int(rng.integers(3, 13))
    n = int(rng.integers(max(30 * K, 300), 2500))
    d = int(rng.choice([2, 4, 8, 16, 32, 64]))
    X, y = _data(kind, n=n, d=max(d, 5) if kind == "lowrank" else d, K=K,
                 seed=int(rng.integers(1 << 30)))
    params = {"C": float(np.exp(rng.uniform(np.log(0.1), np.log(100.0))))}
    gamma = rng.integers(4)
    if gamma == 1:
        params["gamma"] = "auto"
    elif gamma >= 2:
        params["gamma"] = float(rng.uniform(0.3, 3.0)) / (X.shape[1] * X.var())
    weights = rng.integers(4)
    if weights == 1:
        params["class_weight"] = "balanced"
    elif weights == 2:
        params["class_weight"] = {int(c): float(rng.uniform(0.2, 5.0)) for c in np.unique(y)}
    sample_weight = None
    if rng.random() < 0.3:
        sample_weight = rng.uniform(0.2, 3.0, len(X))
        if rng.random() < 0.5:
            sample_weight[rng.random(len(X)) < 0.1] = 0.0
    return X, y, params, sample_weight


@pytest.mark.slow
@needs_kernel
def test_a_soak_of_random_fits():
    """200 random fits: the kernel's model against the host path's decision values,
    the KKT conditions, determinism, and scikit-learn's decision values and labels on
    every tenth. Measured on MLX 0.32.2: within 1.9e-2 of the host path's values and
    1.8e-2 of scikit-learn's, KKT margins within 8.9e-4; 90 of 5 794 pairs, in 6 fits
    at small boxes, had no free dual variable and were solved again on the host."""
    rng = np.random.default_rng(20261002)
    for k in range(200):
        X, y, params, sample_weight = _soak_case(rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            with tuning_context(force_backend="mlx"):
                mine = SVC(**params).fit(X, y, sample_weight=sample_weight)
                with tuning_context(metal_kernels=False):
                    host = SVC(**params).fit(X, y, sample_weight=sample_weight)
                if k % 20 == 0:
                    again = SVC(**params).fit(X, y, sample_weight=sample_weight)
                    for name in ("dual_coef_", "intercept_", "support_", "n_iter_"):
                        assert np.asarray(getattr(again, name)).tobytes() == np.asarray(
                            getattr(mine, name)
                        ).tobytes(), (k, name)
        label = f"fit {k}: {X.shape} K={len(np.unique(y))} {params} weights={sample_weight is not None}"
        assert mine.fit_event_.extra.get("ovo_kernel_pairs"), label
        assert mine.fit_status_ == host.fit_status_ == 0, label
        np.testing.assert_allclose(
            _values(mine, X, "ovo"), _values(host, X, "ovo"),
            atol=DECISION_ATOL, rtol=DECISION_RTOL, err_msg=label,
        )
        if k % 10 == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                theirs = SkSVC(**params).fit(X, y, sample_weight=sample_weight)
            assert_ovo_parity(mine, theirs, X, y, sample_weight, curated=False)
        else:
            violation = _kkt_violation(mine, X, y, sample_weight, _values(mine, X, "ovo"))
            assert violation <= 2 * TOL, f"{label}: KKT violation {violation:.2e}"
