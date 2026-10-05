"""The one-vs-one SVC kernel (``_kernels/_ovo.py``), its gate and its place in ``SVC.fit``.

The kernel changes where the pairs are solved, not what a fit promises: every pair is
held to the host solver's stop in float64 (``smo.finish_c_svc``), a pair that fails it
is finished on the host, and a gate miss runs the per-pair host loop exactly as before
— bit for bit, events included. These tests prove which path ran, by spying on the
kernel's launches and on the host's verification, and compare the two. Parity with
scikit-learn across classes, boxes, weights and shapes is ``tests/parity/test_svc_ovo.py``.
"""

from __future__ import annotations

import dataclasses
import gc
import pickle
import threading
import warnings

import mlx.core as mx
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

import mlxlearn
from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._kernels import _metal, _ovo
from mlxlearn._kernels.kernels import KernelSpec, kernel_matrix
from mlxlearn._solvers import smo
from mlxlearn.exceptions import BackendExecutionError
from mlxlearn.svm import SVC, _classes
from tests._helpers import join_or_fail

MiB = 1024 * 1024

#: Whether the kernel can run here: a GPU with Metal, and MLX 0.32 or later.
KERNEL = gpu_available() and _metal.mlx_version()[:2] >= _metal.MIN_MLX


def needs_kernel(test):
    skip = pytest.mark.skipif(not KERNEL, reason="the one-vs-one kernel needs the GPU and MLX 0.32")
    return pytest.mark.gpu(skip(test))


def _blobs(n=900, d=8, K=4, seed=0, spread=1.0, weights=None):
    rng = np.random.default_rng(seed)
    p = None if weights is None else np.asarray(weights, float) / np.sum(weights)
    y = rng.choice(K, size=n, p=p)
    centers = rng.normal(scale=2.0, size=(K, d)) / np.sqrt(d) * 3
    return centers[y] + spread * rng.normal(size=(n, d)), y


def _fit(X, y, *, kernels=True, sample_weight=None, backend="mlx", **params):
    with tuning_context(force_backend=backend, metal_kernels=kernels), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return SVC(**params).fit(X, y, sample_weight=sample_weight)


#: Every fitted attribute a fit sets, compared bit for bit.
FITTED = (
    "support_", "support_vectors_", "_n_support", "dual_coef_", "intercept_", "_dual_coef_",
    "_intercept_", "n_iter_", "fit_status_", "_pair_weights_", "_pair_rho_", "shape_fit_",
    "classes_", "class_weight_",
)


def _assert_same_model(a, b):
    for name in FITTED:
        left, right = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        assert left.dtype == right.dtype and left.shape == right.shape, name
        assert left.tobytes() == right.tobytes(), name


def _ovo_values(model, X):
    """One-vs-one decision values: the ovr ones count votes, which a value near zero flips."""
    model.decision_function_shape = "ovo"
    try:
        return model.decision_function(X)
    finally:
        model.decision_function_shape = "ovr"


def _ovo_extra(model):
    return {k: v for k, v in model.fit_event_.extra.items() if k.startswith("ovo_")}


@pytest.fixture
def spies(monkeypatch):
    """Counts of waves and launches the kernel ran, and of pairs the host verified.

    The kernel's first use, and its check's own launches, happen before the spies.
    """
    if KERNEL:
        _ovo.pair_solver("gpu")
    counts = {"waves": 0, "launches": 0, "verified": 0}
    real_run = _ovo.PairSolver._run
    real_finish = _classes.finish_c_svc

    def run(self, *args, **kwargs):
        result = real_run(self, *args, **kwargs)
        counts["waves"] += 1
        counts["launches"] += result.launches
        return result

    def finish(*args, **kwargs):
        counts["verified"] += 1
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(_ovo.PairSolver, "_run", run)
    monkeypatch.setattr(_classes, "finish_c_svc", finish)
    return counts


@pytest.fixture
def host_only(monkeypatch):
    """A fit with the GPU one-vs-one solver taken out: the per-pair host loop, as before."""

    def fit(X, y, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(_classes.MLXSVCMixin, "_solve_pairs_on_gpu", lambda self, *a: None)
            return _fit(X, y, **kwargs)

    return fit


@pytest.fixture
def fresh_first_use():
    _ovo._checked.cache_clear()
    yield
    _ovo._checked.cache_clear()
    _ovo._kernel.cache_clear()


def _unbuildable(patch):
    """Give the kernel a source Metal rejects, under a name of its own, as a compiler
    that cannot build it would: MLX raises only when it evaluates the first launch."""
    patch.setattr(_ovo, "_SOURCE", _ovo._SOURCE.replace("steps += 1;", "steps += 1 + undeclared;"))
    patch.setattr(_ovo, "_SOURCE_VERSION", _ovo._SOURCE_VERSION + 1000)
    _ovo._kernel.cache_clear()


# ----------------------------------------------------------------------------------
# the kernel's source and its first use
# ----------------------------------------------------------------------------------


def test_the_kernel_is_built_under_the_metal_rules(monkeypatch):
    """contract(off) first, ASCII, safe math, precise division, no atomics, stable names."""
    built = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kwargs: built.append(kwargs) or kwargs)
    _ovo._kernel.__wrapped__()
    _ovo._kernel.__wrapped__()
    assert built[0]["name"] == built[1]["name"]
    kwargs = built[0]
    assert kwargs["name"].startswith(_metal.NAME_PREFIX)
    assert kwargs["header"].startswith("#pragma METAL fp contract(off)\n")
    assert kwargs["compile_options"] == {"math_mode": "safe"}
    assert kwargs["ensure_row_contiguous"] is True
    source = kwargs["header"] + kwargs["source"]
    source.encode("ascii")
    for fn in ("exp(", "sqrt(", "rsqrt(", "log(", "fast::", "atomic"):
        assert fn not in source
    # Every floating-point division is the precise one; fma is used, explicitly, for
    # exact products.
    assert source.count("divide(") == source.count("metal::precise::divide(") > 0


def test_threadgroup_width_covers_the_largest_pair():
    assert _ovo.threadgroup_width(1) == 32
    assert _ovo.threadgroup_width(256) == 32
    assert _ovo.threadgroup_width(257) == 64
    assert _ovo.threadgroup_width(_ovo.MAX_PAIR_ROWS) == _ovo.MAX_THREADS == 512
    with pytest.raises(ValueError):
        _ovo.threadgroup_width(_ovo.MAX_PAIR_ROWS + 1)


def test_waves_fit_the_budget_in_libsvm_order():
    sizes = [10, 20, 30, 40]
    assert _ovo.waves(sizes, 10**9) == [(0, 4)]
    assert _ovo.waves(sizes, 4 * (10**2 + 20**2)) == [(0, 2), (2, 3), (3, 4)]
    # A pair larger than the budget still gets a wave of its own; the gate keeps such
    # fits off the kernel.
    assert _ovo.waves(sizes, 1) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert _ovo.waves([], 100) == []


def test_a_wave_never_holds_more_gram_entries_than_int32_offsets_reach():
    """Whatever the budget, as the cache overrides allow: 127 Grams of 4 096 rows are
    2³¹ − 2²⁴ entries, and a 128th would wrap the launch table's offsets."""
    assert _ovo.waves([_ovo.MAX_PAIR_ROWS] * 200, 2**45) == [(0, 127), (127, 200)]
    with pytest.raises(OverflowError, match="Gram entries"):
        _ovo.PairSolver(None)._run(
            mx, None, np.array([0, 2**31 + 5]), [1], [np.ones(2)], [np.ones(2)], [10], 1e-3, 32
        )


@needs_kernel
def test_the_known_problem_passes_and_matches_the_host_solver():
    """The first-use problem: the kernel's answer, held to the host's float64 stop."""
    with use_device("gpu"):
        assert _ovo._known_answer(_ovo.PairSolver(_ovo._kernel()))
    grams, firsts, boxes = _ovo._known_problem()
    for gram, first, C in zip(grams, firsts, boxes, strict=True):
        m = gram.shape[0]
        y = np.where(np.arange(m) < first, 1.0, -1.0)
        state = smo.solve_c_svc(y, C, np.ones(m), gram.__getitem__, smo.SMOConfig(cache_rows=m))
        assert state.converged


@pytest.mark.usefixtures("fresh_first_use")
@needs_kernel
def test_the_kernel_compiles_on_the_first_fit_that_takes_it():
    X, y = _blobs(K=2)
    _fit(X, y)
    _fit(*_blobs(K=3), kernel="poly")
    assert _ovo._checked.cache_info().currsize == 0, "compiled for a fit the kernel cannot serve"
    model = _fit(*_blobs(K=3))
    assert _ovo._checked.cache_info().currsize == 1
    assert _ovo_extra(model)["ovo_kernel_pairs"] == 3


@pytest.mark.parametrize("fault", ["unbuildable", "too wide"])
@pytest.mark.usefixtures("fresh_first_use")
@needs_kernel
def test_a_kernel_that_fails_on_its_first_launch_keeps_the_host_path(
    fault, monkeypatch, host_only, spies
):
    """A library Metal cannot build (MLX's RuntimeError), and a threadgroup wider than
    the pipeline takes (its ValueError): both raised when the check's launch is
    evaluated, where reading it through NumPy first aborted the process. The check
    says no, and every fit keeps the host loop."""
    with monkeypatch.context() as patch:
        if fault == "unbuildable":
            _unbuildable(patch)
        else:
            patch.setattr(_ovo, "MAX_THREADS", 2048)  # past every Apple GPU's 1 024
        _ovo._checked.cache_clear()
        assert _ovo.pair_solver("gpu") is None
        checked = spies["waves"]  # the check's own runs, up to the one that failed
        X, y = _blobs()
        model = _fit(X, y)
        assert _ovo._checked.cache_info().currsize == 1, "the failure was not remembered"
    assert spies["waves"] == checked and _ovo_extra(model) == {}
    _assert_same_model(model, host_only(X, y))


@needs_kernel
def test_an_mlx_error_in_a_fits_launch_raises_and_never_aborts(monkeypatch, spies):
    """An error MLX raises evaluating a fit's own launch — here a library that does not
    build, served after the check passed — is the fit's BackendExecutionError."""
    assert _ovo._checked()
    X, y = _blobs()
    try:
        with monkeypatch.context() as patch:
            _unbuildable(patch)
            with pytest.raises(BackendExecutionError):
                _fit(X, y)
    finally:
        _ovo._kernel.cache_clear()
    assert _ovo_extra(_fit(X, y))["ovo_kernel_pairs"] == 6


@pytest.mark.usefixtures("fresh_first_use")
@needs_kernel
def test_a_kernel_that_answers_wrongly_is_not_used(monkeypatch):
    real = _ovo.PairSolver._run

    def wrong(self, *args, **kwargs):
        result = real(self, *args, **kwargs)
        return dataclasses.replace(result, alpha=result.alpha * 0.5)

    monkeypatch.setattr(_ovo.PairSolver, "_run", wrong)
    assert _ovo.pair_solver("gpu") is None


# ----------------------------------------------------------------------------------
# the gate: every miss is the host loop, unchanged
# ----------------------------------------------------------------------------------


GATE_MISSES = {
    "two classes": ({"K": 2}, {}, {}),
    "poly kernel": ({}, {"kernel": "poly"}, {}),
    "max_iter=0": ({}, {"max_iter": 0}, {}),
    "cpu device": ({}, {}, {"backend": "cpu"}),
    "switched off": ({}, {}, {"kernels": False}),
    "a pair the cache cannot hold": ({}, {"cache_size": 0.5}, {}),
    "a box too large": ({}, {"C": 2.0**40}, {}),
    "a box too small": ({}, {"C": 2.0**-110}, {}),
    "a tol below the kernel's resolution": ({}, {"tol": 1e-14}, {}),
}


@pytest.mark.parametrize("case", sorted(GATE_MISSES))
@needs_kernel
def test_a_gate_miss_runs_the_host_loop_bit_for_bit(case, spies, host_only):
    data, params, how = GATE_MISSES[case]
    X, y = _blobs(**{"n": 700, **data})
    with tuning_context(smo_max_iter_factor=1, smo_min_iter_budget=2000):
        model = _fit(X, y, **params, **how)
        reference = host_only(X, y, **params, **how)
    assert spies["waves"] == 0, f"the kernel ran for {case}"
    assert _ovo_extra(model) == {}
    _assert_same_model(model, reference)
    assert model.fit_event_.extra == reference.fit_event_.extra


@needs_kernel
def test_a_pair_past_the_kernels_rows_runs_the_host_loop(monkeypatch, spies, host_only):
    monkeypatch.setattr(_ovo, "MAX_PAIR_ROWS", 300)
    X, y = _blobs(n=600, K=3)  # pairs of about 400 rows
    model = _fit(X, y)
    assert spies["waves"] == 0
    _assert_same_model(model, host_only(X, y))


@needs_kernel
def test_the_gate_reads_tol_against_the_boxes_and_the_largest_pair():
    """``tol`` at least 2⁻⁴⁰ of ``1 + C·m``: the double-float state's reach."""
    classes = [np.arange(300), np.arange(100), np.arange(50)]
    floor = _classes._OVO_TOL_FLOOR * (1.0 + 10.0 * 400)
    assert SVC(C=10.0, tol=floor)._ovo_solver(classes, np.full(450, 10.0), "mlx") is not None
    assert SVC(C=10.0, tol=floor * 0.99)._ovo_solver(classes, np.full(450, 10.0), "mlx") is None


@needs_kernel
def test_a_tol_the_kernel_cannot_resolve_converges_on_the_host_loop(spies, host_only):
    """tol = 1e-14 at C = 10: the double-float state never saw its violation under it,
    and one pair stepped to max_iter (10⁷ steps, 58 s, at the default) where the host
    loop converged in 8 083. Such a fit is the host loop's."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 3, 600)
    X = rng.normal(size=(600, 4)) + 0.8 * rng.normal(size=(3, 4))[y]
    params = {"C": 10.0, "tol": 1e-14, "max_iter": 200_000}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        with tuning_context(force_backend="mlx"):
            model = SVC(**params).fit(X, y)
    assert model.fit_status_ == 0 and spies["waves"] == 0
    _assert_same_model(model, host_only(X, y, **params))


def test_the_gate_reads_the_largest_pair_and_the_cache():
    model = SVC(cache_size=200)
    box = np.ones(4097 + 10)
    classes = [np.arange(2048), np.arange(2049), np.arange(10)]
    assert model._ovo_solver(classes, box, "mlx") is None  # 4 097 rows
    model = SVC(cache_size=1)  # 1 MiB holds a pair of 512 rows, not 513
    classes = [np.arange(257), np.arange(256), np.arange(10)]
    with tuning_context(metal_kernels=False):  # the rules before the kernel's own gate
        assert model._ovo_solver(classes, np.ones(523), "mlx") is None


def test_the_kernel_is_never_built_below_mlx_032(monkeypatch, host_only, spies):
    """0.31's unlocked library map: not one kernel object, whatever MLX this runs."""
    built = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kwargs: built.append(kwargs))
    monkeypatch.setattr(_metal, "mlx_version", lambda: (0, 31, 2))
    _metal._runtime_ready.cache_clear()
    try:
        assert _ovo.pair_solver("gpu") is None
        X, y = _blobs()
        model = _fit(X, y)
    finally:
        _metal._runtime_ready.cache_clear()
    assert built == [] and spies["waves"] == 0
    monkeypatch.undo()
    _assert_same_model(model, host_only(X, y))


@pytest.mark.gpu
@pytest.mark.skipif(
    _metal.mlx_version()[:2] >= _metal.MIN_MLX, reason="the floor's gate, on MLX 0.31"
)
def test_mlx_031_never_builds_the_kernel(monkeypatch):
    built = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kwargs: built.append(kwargs))
    X, y = _blobs()
    model = _fit(X, y)
    assert built == [] and _ovo_extra(model) == {}
    assert _ovo.pair_solver("gpu") is None


# ----------------------------------------------------------------------------------
# what the kernel path computes
# ----------------------------------------------------------------------------------


@needs_kernel
def test_every_pair_is_solved_by_the_kernel_and_verified_on_the_host(spies, host_only):
    X, y = _blobs(n=1200, K=5)
    model = _fit(X, y)
    extra = _ovo_extra(model)
    assert extra == {
        "ovo_kernel_pairs": 10, "ovo_host_finished": 0, "ovo_resolved_non_finite": 0,
        "ovo_resolved_no_free": 0,
        "ovo_waves": 1, "ovo_launches": spies["launches"],
    }
    assert spies["waves"] == 1 and spies["verified"] == 10
    reference = host_only(X, y)
    np.testing.assert_allclose(_ovo_values(model, X), _ovo_values(reference, X), atol=2e-2, rtol=5e-3)
    assert np.mean(model.predict(X) == reference.predict(X)) >= 0.995
    # Same Gram, same rule: the same number of support vectors in every pair but at
    # most a tie or two.
    assert abs(len(model.support_) - len(reference.support_)) <= 2


def _pair_rows(model, X, y):
    """Each pair's (alpha, labels, rows) from the public dual_coef_ layout."""
    encoded = np.searchsorted(model.classes_, y)
    offsets = np.concatenate([[0], np.cumsum(model.n_support_)])
    K = len(model.classes_)
    out = []
    for i in range(K):
        for j in range(i + 1, K):
            rows = np.concatenate([np.flatnonzero(encoded == i), np.flatnonzero(encoded == j)])
            coef = np.zeros(len(X))
            si, sj = slice(offsets[i], offsets[i + 1]), slice(offsets[j], offsets[j + 1])
            coef[model.support_[si]] = model.dual_coef_[j - 1, si]
            coef[model.support_[sj]] = model.dual_coef_[i, sj]
            labels = np.where(encoded[rows] == i, 1.0, -1.0)
            out.append((coef[rows] * labels, labels, rows))
    return out


def _kkt_violations(model, X, y, C):
    """The full-set KKT violation of every pair, with float64 kernel values."""
    from sklearn.metrics.pairwise import rbf_kernel

    violations = []
    for alpha, labels, rows in _pair_rows(model, X, y):
        K = rbf_kernel(X[rows], gamma=model._gamma)
        yg = K @ (alpha * labels) - labels
        upper = alpha >= C
        lower = ~upper & (alpha <= 0.0)
        up = np.where(labels > 0, ~upper, ~lower)
        low = np.where(labels > 0, ~lower, ~upper)
        violations.append(float((-yg[up]).max() + yg[low].max()))
    return np.array(violations)


@needs_kernel
def test_a_pair_the_host_rejects_is_finished_on_the_host(monkeypatch, host_only):
    """The kernel stopped at fifty times ``tol``: every pair fails the host's float64
    test and is finished there, warm-started, to the host solver's own stop."""
    real = _ovo.PairSolver.solve

    def early(self, *args, tol, **kwargs):
        return real(self, *args, tol=tol * 50, **kwargs)

    monkeypatch.setattr(_ovo.PairSolver, "solve", early)
    X, y = _blobs(n=1200, K=4, spread=1.5)
    model = _fit(X, y)
    extra = _ovo_extra(model)
    assert extra["ovo_host_finished"] == 6 and extra["ovo_resolved_non_finite"] == 0
    assert _kkt_violations(model, X, y, 1.0).max() <= 2e-3
    reference = host_only(X, y)
    np.testing.assert_allclose(_ovo_values(model, X), _ovo_values(reference, X), atol=2e-2, rtol=5e-3)


@needs_kernel
def test_a_non_finite_pair_is_solved_again_by_the_host_loop(monkeypatch, host_only):
    """A pair the kernel reports non-finite, and one whose alpha reads back as NaN, are
    each solved from scratch by the per-pair host path, which gives the host loop's own
    coefficients for them, bit for bit."""
    real = _ovo.PairSolver._run

    def broken(self, *args, **kwargs):
        result = real(self, *args, **kwargs)
        stop = result.stop.copy()
        stop[0] = _ovo.STOP_NON_FINITE
        alpha = result.alpha.copy()
        alpha[result.offsets[2]] = np.nan
        return dataclasses.replace(result, stop=stop, alpha=alpha)

    monkeypatch.setattr(_ovo.PairSolver, "_run", broken)
    X, y = _blobs(n=900, K=3)
    model = _fit(X, y)
    assert _ovo_extra(model)["ovo_resolved_non_finite"] == 2
    reference = host_only(X, y)
    for k in (0, 2):
        np.testing.assert_array_equal(model._pair_rho_[k], reference._pair_rho_[k])
        np.testing.assert_array_equal(model.n_iter_[k], reference.n_iter_[k])


@needs_kernel
def test_the_iteration_budget_is_the_host_loops(host_only):
    X, y = _blobs(n=900, K=4, spread=2.0)
    for max_iter in (1, 2, 30):
        with pytest.warns(ConvergenceWarning, match=f"max_iter={max_iter}"):
            with tuning_context(force_backend="mlx"):
                model = SVC(C=10.0, max_iter=max_iter).fit(X, y)
        reference = host_only(X, y, C=10.0, max_iter=max_iter)
        assert model.fit_status_ == reference.fit_status_ == 1
        np.testing.assert_array_equal(model.n_iter_, reference.n_iter_)
        assert set(model.n_iter_.tolist()) == {max_iter}


@needs_kernel
def test_shrinking_changes_nothing_on_the_kernel_path():
    """Pairs the kernel solved, and the host finished, are the same either way."""
    X, y = _blobs(n=900, K=4)
    _assert_same_model(_fit(X, y, shrinking=True), _fit(X, y, shrinking=False))


@needs_kernel
def test_a_pair_solved_again_honors_shrinking_as_the_host_path_does(host_only):
    """Every pair here ends with no free dual variable and is solved again by the host
    loop, which honors ``shrinking``: each setting gives the host path's model, bit for
    bit, and the two settings differ in the biases' last bits."""
    rng = np.random.default_rng(238)
    K, n = int(rng.integers(3, 6)), int(rng.integers(90, 400))
    X = rng.normal(size=(n, int(rng.integers(2, 6))))
    y = rng.integers(0, K, n)
    params = {"C": float(10 ** rng.uniform(-1.5, 1.0)), "gamma": float(10 ** rng.uniform(-3, 1.5))}
    params["class_weight"] = "balanced"
    models = {}
    for shrinking in (True, False):
        models[shrinking] = _fit(X, y, shrinking=shrinking, **params)
        assert _ovo_extra(models[shrinking])["ovo_resolved_no_free"] == 6, "lost its premise"
        _assert_same_model(models[shrinking], host_only(X, y, shrinking=shrinking, **params))
    assert models[True]._pair_rho_.tobytes() != models[False]._pair_rho_.tobytes()


# ----------------------------------------------------------------------------------
# the Gram, bit for bit
# ----------------------------------------------------------------------------------


def _packed_against_kernel_matrix(X, y, spec, budget=None):
    classes = [np.flatnonzero(y == c) for c in np.unique(y)]
    rows = _ovo.class_rows(X, classes)
    pairs = [(i, j) for i in range(len(classes)) for j in range(i + 1, len(classes))]
    overrides = {} if budget is None else {"distance_block_bytes": budget}
    with tuning_context(**overrides), use_device("gpu"):
        gram, offsets = _ovo._pair_grams(mx, spec, rows, pairs, "gpu", 0)
        host = np.array(gram)
        for k, (i, j) in enumerate(pairs):
            index = np.concatenate([classes[i], classes[j]])
            want = kernel_matrix(spec, np.ascontiguousarray(X[index]), backend="mlx", device="gpu")
            got = host[offsets[k] : offsets[k + 1]].reshape(want.shape)
            np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize(
    "n, d, K", [(600, 16, 3), (1024, 512, 8), (1500, 256, 4), (900, 2048, 3)]
)
@needs_kernel
def test_the_packed_grams_are_kernel_matrix_bit_for_bit(n, d, K):
    X, y = _blobs(n=n, d=d, K=K, seed=n + d)
    X = (X - X.mean(0)).astype(np.float32)
    spec = KernelSpec(kernel="rbf", gamma=1.0 / (d * float(X.astype(np.float64).var())))
    _packed_against_kernel_matrix(X, y, spec)


@needs_kernel
def test_a_pair_in_several_blocks_and_a_refined_pair_are_kernel_matrix_bit_for_bit():
    """Row blocks as kernel_matrix cuts them under a small budget; and tight clusters
    far apart, whose rbf entries the host recomputes in float64."""
    X, y = _blobs(n=900, d=8, K=3)
    X = (X - X.mean(0)).astype(np.float32)
    spec = KernelSpec(kernel="rbf", gamma=0.1)
    _packed_against_kernel_matrix(X, y, spec, budget=64 * 1024)
    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, 600)
    X = (np.where(rng.random(600) < 0.5, 0.0, 2000.0)[:, None]
         + rng.normal(scale=0.3, size=(3, 4))[y] + 0.2 * rng.normal(size=(600, 4)))
    X = (X - X.mean(0)).astype(np.float32)
    spec = KernelSpec(kernel="rbf", gamma=1.0)
    sq = float(np.max(np.einsum("ij,ij->i", X, X)))
    from mlxlearn._kernels.kernels import _RbfRefiner

    assert _RbfRefiner.needed(spec, 4, sq, sq), "the test lost its premise"
    _packed_against_kernel_matrix(X, y, spec)


# ----------------------------------------------------------------------------------
# determinism: waves, launches and widths change nothing
# ----------------------------------------------------------------------------------


@needs_kernel
def test_repeated_fits_are_bit_for_bit_identical():
    X, y = _blobs(n=1500, d=16, K=6, spread=1.5)
    first = _fit(X, y, C=3.0)
    for _ in range(3):
        _assert_same_model(_fit(X, y, C=3.0), first)


@needs_kernel
def test_waves_under_a_small_cache_change_nothing(spies):
    """A cache that holds one pair's Gram: a wave per pair, the same model bit for bit."""
    X, y = _blobs(n=1000, d=8, K=5)
    whole = _fit(X, y)
    assert _ovo_extra(whole)["ovo_waves"] == 1
    largest = max(
        np.sum(y == i) + np.sum(y == j) for i in range(5) for j in range(i + 1, 5)
    )
    cache_mib = 4 * largest * largest / MiB * 1.01
    waved = _fit(X, y, cache_size=cache_mib)
    assert _ovo_extra(waved)["ovo_waves"] == 10
    _assert_same_model(waved, whole)


@needs_kernel
def test_relaunches_and_threadgroup_widths_change_nothing(monkeypatch):
    X, y = _blobs(n=1200, d=8, K=4, spread=2.0)
    reference = _fit(X, y, C=10.0)
    monkeypatch.setattr(_ovo, "FIRST_STEPS", 3)
    monkeypatch.setattr(_ovo, "MIN_STEPS", 3)
    monkeypatch.setattr(_ovo, "LAUNCH_SECONDS", 0.0)
    relaunched = _fit(X, y, C=10.0)
    assert _ovo_extra(relaunched)["ovo_launches"] > 20
    _assert_same_model(relaunched, reference)
    monkeypatch.undo()
    monkeypatch.setattr(_ovo, "threadgroup_width", lambda rows: _ovo.MAX_THREADS)
    _assert_same_model(_fit(X, y, C=10.0), reference)


@pytest.mark.usefixtures("fresh_first_use")
@needs_kernel
def test_concurrent_fits_and_the_first_compile(monkeypatch):
    """Eight threads fit at once on a process's first kernel fit: one compiles and checks
    the kernel, the others wait for it, and every model is the sequential one."""
    checks = []
    real_check = _ovo._known_answer

    def counting_check(solver):
        checks.append(threading.get_ident())
        return real_check(solver)

    monkeypatch.setattr(_ovo, "_known_answer", counting_check)
    data = [_blobs(n=600 + 50 * k, d=8, K=3 + k % 3, seed=k) for k in range(8)]
    n_threads = len(data)
    barrier = threading.Barrier(n_threads, timeout=60)
    results: list = [None] * n_threads
    errors: list[BaseException] = []

    def worker(slot):
        try:
            barrier.wait()
            results[slot] = [_fit(*data[slot]) for _ in range(2)]
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    join_or_fail(threads)
    assert not errors, errors
    assert len(checks) == 1, f"{len(checks)} first-use checks"
    for slot in range(n_threads):
        want = _fit(*data[slot])
        assert _ovo_extra(want)["ovo_kernel_pairs"] > 0
        for got in results[slot]:
            _assert_same_model(got, want)


# ----------------------------------------------------------------------------------
# the record, the pickle, the memory
# ----------------------------------------------------------------------------------


@pytest.mark.usefixtures("fresh_first_use")
@needs_kernel
def test_a_kernel_fit_and_its_check_raise_no_floating_point_warnings():
    """NumPy 2.2's BLAS on macOS left floating-point flags set on finite products, and the
    host's verification of every pair, and the first-use check, warned on every fit."""
    X, y = _blobs(n=900, K=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = _fit(X, y)
    assert _ovo_extra(model)["ovo_kernel_pairs"] == 6


@needs_kernel
def test_one_fit_records_one_event_with_the_counts():
    X, y = _blobs(n=900, K=4)
    mlxlearn.clear_backend_diagnostics()
    model = _fit(X, y)
    events = mlxlearn.get_backend_diagnostics()
    assert [e.operation for e in events] == ["fit"]
    assert events[0] == model.fit_event_
    assert _ovo_extra(model)["ovo_kernel_pairs"] == 6


@needs_kernel
def test_a_pickled_model_is_the_host_paths_and_loads_without_the_kernel(host_only):
    X, y = _blobs(n=900, K=4)
    model = _fit(X, y)
    reference = host_only(X, y)
    assert sorted(vars(model)) == sorted(vars(reference))
    blob = pickle.dumps(model)
    with tuning_context(metal_kernels=False):
        loaded = pickle.loads(blob)
        np.testing.assert_array_equal(loaded.decision_function(X), model.decision_function(X))
    _assert_same_model(loaded, model)


@pytest.mark.parametrize(
    "n, d, K",
    [
        (4000, 16, 6),  # 15 pairs of about 1 333 rows: 7 MiB of Gram each
        (4000, 2048, 20),  # 31 MiB of rows, twice the tiles' budget; pairs of 400
    ],
)
@needs_kernel
def test_the_fit_holds_at_most_its_cache_of_grams_and_releases_them(n, d, K):
    """A wave's packed Grams fit ``cache_size``; the rows on the device and the build's
    transients fit the kernel tiles' own budget, the rows' squares included; the fit
    keeps none of it, and the cache-release rule returns it."""
    X, y = _blobs(n=n, d=d, K=K)
    budget, tiles = 24 * MiB, 16 * MiB
    with tuning_context(distance_block_bytes=tiles):
        _fit(X, y, cache_size=budget / MiB)
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        model = _fit(X, y, cache_size=budget / MiB)
        mx.synchronize()
        peak = mx.get_peak_memory() - base
    assert _ovo_extra(model)["ovo_waves"] > 3
    assert peak <= budget + tiles + 8 * MiB, f"peak {peak / MiB:.1f} MiB"
    del model
    gc.collect()
    assert mx.get_active_memory() - base <= 1 * MiB
    with tuning_context(mlx_cache_release_fraction=1e-9):
        _fit(X, y, cache_size=budget / MiB)
    assert mx.get_cache_memory() == 0
