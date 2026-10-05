"""Direct tests of the SVM kernel module, against ``sklearn.metrics.pairwise``.

The estimator suite exercises these functions only through an SVC fit, where a
wrong kernel entry shows up — if at all — as a slightly different model. Here
each entry point is compared with scikit-learn's own kernel functions on both
MLX devices, including the cases the module docstring singles out: a negative
polynomial base raised to an odd power, ``degree=0``, rbf on data far from the
origin and on tight clusters far from each other, and self-entries that must
equal the closed-form diagonal.

Tolerances are float32 ones (``docs/fp32_policy.md``): every entry is computed in
float32 and compared with a float64 reference, relative to the scale of the
kernel's values.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics.pairwise import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)

from mlxlearn._common.device import gpu_available
from mlxlearn._kernels.kernels import (
    KernelBasis,
    KernelSpec,
    _fused_rbf,
    _query_norms,
    _rbf_refiner,
    _transform,
    apply_frame,
    build_kernel_rows,
    check_frame_resolution,
    check_query_range,
    frame_spec,
    kernel_diagonal,
    kernel_frame,
    kernel_matrix,
)
from mlxlearn.exceptions import UnsupportedInputError

BACKENDS = [
    pytest.param(
        "mlx",
        marks=pytest.mark.skipif(not gpu_available(), reason="needs an MLX GPU device"),
    ),
    "cpu",
]

#: (spec kwargs, why it is here). gamma is deliberately non-round: a fused kernel
#: that captured it as a constant would round it differently from the eager path.
SPECS = [
    ({"kernel": "linear"}, "plain inner product"),
    ({"kernel": "rbf", "gamma": 1.0 / 3.0}, "the default kernel"),
    ({"kernel": "poly", "gamma": 0.37, "degree": 3, "coef0": -0.5}, "negative base, odd degree"),
    ({"kernel": "poly", "gamma": 0.37, "degree": 2, "coef0": 1.0}, "even degree"),
    ({"kernel": "sigmoid", "gamma": 0.11, "coef0": -0.3}, "indefinite kernel"),
]


def _reference(spec: KernelSpec, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``K(A, B)`` in float64 from scikit-learn, on the float32-rounded inputs."""
    A = A.astype(np.float32).astype(np.float64)
    B = B.astype(np.float32).astype(np.float64)
    if spec.kernel == "linear":
        return linear_kernel(A, B)
    if spec.kernel == "rbf":
        return rbf_kernel(A, B, gamma=spec.gamma)
    if spec.kernel == "sigmoid":
        return sigmoid_kernel(A, B, gamma=spec.gamma, coef0=spec.coef0)
    if spec.degree == 0:
        # scikit-learn's polynomial_kernel rejects degree=0; the value is 1.
        return np.ones((A.shape[0], B.shape[0]))
    return polynomial_kernel(A, B, degree=spec.degree, gamma=spec.gamma, coef0=spec.coef0)


def _assert_kernel_close(actual, expected):
    scale = max(float(np.abs(expected).max()), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5 * scale)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(300, 12))
    B = rng.normal(size=(220, 12))
    return A, B


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs, why", SPECS, ids=[why for _, why in SPECS])
def test_kernel_matrix_matches_sklearn(backend, kwargs, why, data):
    A, B = data
    spec = KernelSpec(**kwargs)
    _assert_kernel_close(kernel_matrix(spec, A, B, backend=backend), _reference(spec, A, B))
    _assert_kernel_close(kernel_matrix(spec, A, backend=backend), _reference(spec, A, A))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs, why", SPECS, ids=[why for _, why in SPECS])
def test_kernel_rows_match_sklearn(backend, kwargs, why, data):
    """``KernelRows`` is the path every SVC subproblem above the Gram cutoff takes."""
    A, _ = data
    spec = KernelSpec(**kwargs)
    rows = build_kernel_rows(A, spec, backend=backend)
    expected = _reference(spec, A, A)
    picked = np.array([0, 7, 299, 150, 7])

    batch = rows.rows(picked)
    assert batch.shape == (picked.shape[0], A.shape[0])
    assert batch.dtype == np.float32
    _assert_kernel_close(batch, expected[picked])
    for index in (0, 150, 299):
        _assert_kernel_close(rows.row(index), expected[index])


def _combine(spec, A, B, weights, backend, block_rows=None):
    """``K(A, B) @ weights`` through a fresh basis: the path ``predict`` takes.

    A basis keeps the device copy of the first rows it is given, so each call
    gets its own.
    """
    return KernelBasis(spec).combine(A, B, weights, backend=backend, block_rows=block_rows)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs, why", SPECS, ids=[why for _, why in SPECS])
def test_basis_combine_matches_sklearn(backend, kwargs, why, data):
    A, B = data
    spec = KernelSpec(**kwargs)
    weights = np.random.default_rng(1).normal(size=(B.shape[0], 3))
    expected = _reference(spec, A, B) @ weights
    # Several tiles, so the pipelined loop and its last, short tile both run.
    actual = _combine(spec, A, B, weights, backend, block_rows=64)
    assert actual.dtype == np.float64
    scale = float(np.abs(_reference(spec, A, B)).max()) * np.abs(weights).sum(axis=0).max()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5 * max(scale, 1.0))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs, why", SPECS, ids=[why for _, why in SPECS])
def test_the_host_path_is_the_device_arithmetic(backend, kwargs, why, data):
    """``combine(..., host=True)`` — SVC's small batches — computes each entry as the
    device tiles do, so it matches scikit-learn to the same tolerance and the device
    path to float32 rounding; and it reads the same operand, built once."""
    A, B = data
    spec = KernelSpec(**kwargs)
    weights = np.random.default_rng(1).normal(size=(B.shape[0], 3))
    basis = KernelBasis(spec)
    host = basis.combine(A[:9], B, weights, backend=backend, host=True)
    operand = basis._operand
    device = basis.combine(A[:9], B, weights, backend=backend)
    assert basis._operand is operand
    expected = _reference(spec, A[:9], B) @ weights
    scale = float(np.abs(_reference(spec, A, B)).max()) * np.abs(weights).sum(axis=0).max()
    np.testing.assert_allclose(host, expected, rtol=0, atol=2e-5 * max(scale, 1.0))
    np.testing.assert_allclose(host, device, rtol=0, atol=2e-5 * max(scale, 1.0))


def test_the_host_path_refines_rbf_entries_as_the_device_does():
    """Tight clusters far from each other: entries the Gram identity gets wrong in
    float32 are recomputed on either path."""
    rng = np.random.default_rng(3)
    centers = np.array([[0.0] * 6, [40.0] * 6])
    A = centers[rng.integers(0, 2, size=40)] + 1e-3 * rng.normal(size=(40, 6))
    spec = KernelSpec(kernel="rbf", gamma=2e5)
    weights = np.ones((A.shape[0], 1))
    host = KernelBasis(spec).combine(A[:5], A, weights, backend="cpu", host=True)
    np.testing.assert_allclose(host, _reference(spec, A[:5], A) @ weights, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_degree_zero_is_the_constant_kernel(backend, data):
    A, B = data
    spec = KernelSpec(kernel="poly", gamma=0.5, degree=0, coef0=-2.0)
    np.testing.assert_array_equal(kernel_matrix(spec, A, B, backend=backend), 1.0)
    np.testing.assert_array_equal(build_kernel_rows(A, spec, backend=backend).rows([3, 4]), 1.0)
    np.testing.assert_array_equal(kernel_diagonal(spec, A), 1.0)


@pytest.mark.parametrize("kwargs, why", SPECS, ids=[why for _, why in SPECS])
def test_kernel_diagonal_matches_sklearn(kwargs, why, data):
    A, _ = data
    spec = KernelSpec(**kwargs)
    expected = np.diag(_reference(spec, A, A))
    np.testing.assert_allclose(kernel_diagonal(spec, A), expected, rtol=1e-6, atol=1e-6)


def _framed(spec: KernelSpec, X: np.ndarray):
    """``X`` in ``spec``'s float32 compute frame, and the compute spec: what a fit uses."""
    center, scale = kernel_frame(spec, X)
    return frame_spec(spec, scale), apply_frame(X, center, scale)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_rbf_frame_rescues_data_far_from_the_origin(backend):
    """``‖a‖² + ‖b‖² − 2⟨a, b⟩`` at coordinate 1e3 cancels away float32's digits.

    The frame translates by the float64 mean before the cast, which is exact and
    brings the identity back to the variance scale — ``docs/fp32_policy.md`` §2.1.
    The reference is the float64 kernel of the raw data: nothing is lost to the
    offset.
    """
    rng = np.random.default_rng(3)
    X = 1e3 + rng.normal(size=(200, 6))
    spec = KernelSpec(kernel="rbf", gamma=0.2)
    compute, X32 = _framed(spec, X)
    expected = rbf_kernel(X, X, gamma=0.2)
    _assert_kernel_close(kernel_matrix(compute, X32, backend=backend), expected)
    rows = build_kernel_rows(X32, compute, backend=backend)
    _assert_kernel_close(rows.rows(np.arange(20)), expected[:20])

    # Centred, the identity needs no help. Uncentred, its error — more than 1e-2
    # of the kernel — would call for recomputing entries on the host.
    assert _rbf_refiner(compute, X32, X32) is None
    raw32 = X.astype(np.float32)
    assert _rbf_refiner(spec, raw32, raw32) is not None


def _clusters(separation: float, seed: int = 0) -> np.ndarray:
    """Two unit-spread clusters ``separation`` apart: far from the mean, close within."""
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(size=(150, 3)), separation + rng.normal(size=(150, 3))])


@pytest.mark.parametrize("backend", BACKENDS)
def test_rbf_entries_the_identity_cancels_are_recomputed(backend):
    """Tight clusters 2 000 apart with ``gamma`` matched to the clusters.

    Centering cannot help: every row is ~1 000 from the mean, so the identity's
    float32 error, ``~u·(‖a‖² + ‖b‖²)`` in the frame, is ``gamma·‖a − b‖² ≈ 0.6``
    inside a cluster. The kernel came out wrong by 0.58, the device's own diagonal
    at 0.61 against the solver's exact 1, and an SVC fit's decision values by 6
    against scikit-learn's, silently. Refined, every entry is the float32-frame
    kernel to rounding — on the Gram, on the row source and at inference.
    """
    X = _clusters(2000.0)
    spec = KernelSpec(kernel="rbf", gamma=1.0)
    compute, X32 = _framed(spec, X)
    expected = _reference(compute, X32, X32)

    _assert_kernel_close(kernel_matrix(compute, X32, backend=backend), expected)
    rows = build_kernel_rows(X32, compute, backend=backend)
    picked = np.array([0, 5, 149, 150, 299, 7])
    _assert_kernel_close(rows.rows(picked), expected[picked])
    _assert_kernel_close(rows.row(160), expected[160])

    weights = np.random.default_rng(1).normal(size=(X.shape[0], 2))
    center, scale = kernel_frame(spec, X)
    basis = KernelBasis(compute, center=center, scale=scale)
    actual = basis.combine(X[::7], X, weights, backend=backend, block_rows=16)
    np.testing.assert_allclose(actual, expected[::7] @ weights, rtol=0, atol=1e-4)


def test_ordinary_data_never_pays_for_refinement():
    """Standardized data at ``gamma="scale"`` must stay on the plain identity.

    The refinement is host work per entry; on the data SVC is built for it would
    cost more than the kernel. Its gate stays shut up to 4 096 features, and for
    moderately heavy tails.
    """
    rng = np.random.default_rng(8)
    for d in (8, 256, 4096):
        X = rng.normal(size=(300, d))
        spec = KernelSpec(kernel="rbf", gamma=1.0 / (d * X.var()))
        compute, X32 = _framed(spec, X)
        assert _rbf_refiner(compute, X32, X32) is None, d
        assert build_kernel_rows(X32, compute, backend="cpu")._refiner is None
    X = rng.standard_t(3, size=(2000, 20))
    compute, X32 = _framed(KernelSpec(kernel="rbf", gamma=1.0 / (20 * X.var())), X)
    assert _rbf_refiner(compute, X32, X32) is None


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs, why", SPECS, ids=[why for _, why in SPECS])
def test_self_entries_are_the_closed_form_diagonal(backend, kwargs, why):
    """Every row a solver reads carries ``kernel_diagonal``'s value in its self-entry.

    The solver's curvature uses the closed form and its gradient updates use the
    rows. Left to float32, a poly self-entry near 6e7 differed from the closed
    form by 31, and a step computed for one quadratic but applied to another
    circled the solver's whole iteration budget. Offset data makes that rounding
    large enough to see on every kernel.
    """
    rng = np.random.default_rng(9)
    X32 = (20.0 + 0.01 * rng.normal(size=(120, 10))).astype(np.float32)
    spec = KernelSpec(**kwargs)
    diagonal = kernel_diagonal(spec, X32).astype(np.float32)

    np.testing.assert_array_equal(np.diag(kernel_matrix(spec, X32, backend=backend)), diagonal)
    rows = build_kernel_rows(X32, spec, backend=backend)
    picked = np.array([3, 0, 119, 42])
    np.testing.assert_array_equal(rows.rows(picked)[np.arange(4), picked], diagonal[picked])
    for index in (0, 77):
        assert rows.row(index)[index] == diagonal[index]


def test_resolution_check_refuses_an_offset_float32_cannot_resolve():
    """A large offset with a small spread: the inner product's rounding beats its variation.

    ``make_classification · 0.01 + 20`` with a cubic kernel is the reported case:
    it circled 10⁷ iterations (376 s) or converged to a worse model, silently.
    """
    rng = np.random.default_rng(10)
    base = rng.normal(size=(400, 10))
    for kernel in ("poly", "linear"):
        spec = KernelSpec(kernel=kernel, gamma=0.1, degree=3, coef0=1.0)
        compute, X32 = _framed(spec, base * 0.01 + 20.0)
        with pytest.raises(UnsupportedInputError) as excinfo:
            check_frame_resolution(compute, X32)
        assert excinfo.value.reason == "float32-cancellation"
        assert "StandardScaler" in str(excinfo.value)
        # Standardized, and offset up to ten times the spread, it is served.
        for offset in (0.0, 10.0):
            check_frame_resolution(*_framed(spec, base + offset))
        # Rows float32 holds as identical leave only rounding to learn from — a
        # spread of 1e-10 at an offset of 3 000 fitted a dual 5 000 times off
        # LIBSVM's, and differed from it by chance before — while all-zero rows
        # have nothing to round.
        rows = (base[:2, :5] * 1e-10 - 3000.0).astype(np.float32)
        with pytest.raises(UnsupportedInputError):
            check_frame_resolution(spec, rows)
        check_frame_resolution(spec, np.zeros((4, 5), dtype=np.float32))

    # rbf is translated, and sigmoid measured unaffected: neither is checked.
    for kernel in ("rbf", "sigmoid"):
        check_frame_resolution(*_framed(KernelSpec(kernel=kernel, gamma=0.1), base * 0.01 + 20))


def test_far_queries_are_evaluated_rather_than_refused():
    """Only a query's inner product with the fitted rows has to stay in float32.

    The Gram bound, ``4·‖q‖²``, refused a 1e20 or netCDF 9.97e36 fill value in one
    query row, though its rbf entries are exactly 0 and its sigmoid ones ±1.
    """
    rng = np.random.default_rng(11)
    B = rng.uniform(-1.0, 1.0, size=(30, 4)).astype(np.float32)
    reach = float(np.einsum("ij,ij->i", B, B, dtype=np.float64).max())

    def norms(*rows):
        return _query_norms(np.array(rows, dtype=np.float32))

    for value in (1e20, 9.96921e36):
        q_sq = norms([0, 0, 0, 0], [value, 0, 0, 0])
        assert np.isfinite(q_sq).all()  # computed wide where float32 overflows
        for kernel in ("rbf", "sigmoid", "poly", "linear"):
            check_query_range(KernelSpec(kernel=kernel, gamma=0.5), q_sq, reach)

    # What float32 cannot evaluate is still refused: an inner product past its range,
    with pytest.raises(UnsupportedInputError, match="inner product"):
        check_query_range(KernelSpec(kernel="rbf"), norms([3e38] * 4), reach)
    # an entry the frame could not hold,
    with pytest.raises(UnsupportedInputError):
        check_query_range(KernelSpec(kernel="rbf"), norms([np.inf, 0, 0, 0]), reach)
    # and an overflowing distance whose kernel value, at a vanishing gamma, is not 0.
    with pytest.raises(UnsupportedInputError, match="gamma"):
        check_query_range(KernelSpec(kernel="rbf", gamma=1e-40), norms([1e20, 0, 0, 0]), reach)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_a_row_does_not_depend_on_its_batch(backend, kernel):
    """Row values must not change with the rows fetched alongside them.

    The SMO solver prefetches rows in batches of varying size on the GPU; if a
    row's last bits depended on the batch, which rows happened to be prefetched
    would steer the solver's path. A lone row is therefore evaluated as a
    two-column product rather than a matrix-vector one. On the CPU device, where
    nothing is prefetched, the vectorized transcendental functions' scalar tail
    can still move an entry by an ulp, so there it is a float32-rounding claim.
    Exact on the GPU at this width; at a thousand features and more it is not
    (see the next test).
    """
    rng = np.random.default_rng(11)
    X = rng.normal(size=(1500, 24))
    spec = KernelSpec(kernel=kernel, gamma=0.043, coef0=0.25, degree=3)
    rows = build_kernel_rows(X, spec, backend=backend)
    picked = np.array([5, 17, 1499, 250, 0, 777, 3, 1200, 42, 999, 64, 128, 256])
    together = rows.rows(picked)
    for size in (1, 2, 3, 8):
        for start in range(0, picked.shape[0], size):
            chunk = picked[start : start + size]
            if backend == "mlx":
                np.testing.assert_array_equal(rows.rows(chunk), together[start : start + size])
            else:
                np.testing.assert_allclose(
                    rows.rows(chunk), together[start : start + size], rtol=5e-7, atol=5e-7
                )


@pytest.mark.parametrize("backend", BACKENDS)
def test_wide_rows_agree_with_their_batch_to_rounding(backend):
    """At a thousand features and more the GPU is not batch-independent either.

    MLX picks another matrix-multiply kernel for some shapes there — the Gram's,
    or the prefetch's 33-row batch — so a row, a batch and the Gram can differ in
    the last bits on both devices (``cache_size`` in ``docs/fp32_policy.md``
    §2.4). Only rounding-level agreement is claimed, and asserted.
    """
    rng = np.random.default_rng(13)
    X = (rng.normal(size=(600, 2048)) / np.sqrt(2048)).astype(np.float32)
    spec = KernelSpec(kernel="rbf", gamma=0.5)
    rows = build_kernel_rows(X, spec, backend=backend)
    picked = np.arange(100, 166)
    together = rows.rows(picked)
    np.testing.assert_allclose(rows.rows(picked[:8]), together[:8], rtol=5e-6, atol=5e-6)
    np.testing.assert_allclose(rows.row(int(picked[0])), together[0], rtol=5e-6, atol=5e-6)
    gram = kernel_matrix(spec, X, backend=backend)
    np.testing.assert_allclose(gram[picked], together, rtol=5e-6, atol=5e-6)


@pytest.mark.skipif(not gpu_available(), reason="needs an MLX GPU device")
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_gpu_row_is_the_batched_row(kernel):
    """On the GPU ``row`` goes through ``rows``, so the two never disagree."""
    rng = np.random.default_rng(12)
    X = rng.normal(size=(900, 16))
    rows = build_kernel_rows(X, KernelSpec(kernel=kernel, gamma=0.07), backend="mlx")
    for index in (0, 450, 899):
        np.testing.assert_array_equal(rows.row(index), rows.rows([index, 1, 2])[0])


def test_fused_rbf_is_bit_identical_to_the_eager_transform():
    """gamma travels as an array argument, never as a traced constant.

    Captured as a Python float, MLX rounds it into the fused kernel differently
    from the eager path's float32 scalar for most values — 1/3 and the default
    ``"scale"`` among them — and every kernel entry moves by an ulp or more.
    """
    import mlx.core as mx

    if not gpu_available():
        pytest.skip("the fused kernel is only used on the GPU")
    rng = np.random.default_rng(5)
    A = rng.normal(size=(257, 19)).astype(np.float32)
    B = rng.normal(size=(301, 19)).astype(np.float32)
    for gamma in (1.0 / 3.0, 1.0 / (19 * float(A.var())), 0.05):
        spec = KernelSpec(kernel="rbf", gamma=gamma)
        with mx.stream(mx.gpu):
            left, right = mx.array(A), mx.array(B)
            dot = left @ right.T
            a_sq = mx.sum(left * left, axis=1)
            b_sq = mx.sum(right * right, axis=1)
            eager = _transform(mx, spec, dot, a_sq, b_sq)
            fused = _fused_rbf()(dot, a_sq, b_sq, mx.array(-gamma, dtype=mx.float32))
            mx.eval(eager, fused)
        np.testing.assert_array_equal(np.asarray(fused), np.asarray(eager))


@pytest.mark.parametrize("backend", BACKENDS)
def test_basis_combine_blocking_changes_only_rounding(backend, data):
    A, B = data
    spec = KernelSpec(kernel="rbf", gamma=0.08)
    weights = np.random.default_rng(2).normal(size=(B.shape[0], 6))
    whole = _combine(spec, A, B, weights, backend)
    for block_rows in (1, 7, 64, 299):
        blocked = _combine(spec, A, B, weights, backend, block_rows=block_rows)
        np.testing.assert_allclose(blocked, whole, rtol=1e-6, atol=1e-6)
    # And a given blocking is deterministic.
    np.testing.assert_array_equal(
        _combine(spec, A, B, weights, backend, block_rows=64),
        _combine(spec, A, B, weights, backend, block_rows=64),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_owns_its_memory_and_validates_indices(backend):
    X = np.random.default_rng(4).normal(size=(50, 5)).astype(np.float32)
    rows = build_kernel_rows(X, KernelSpec(kernel="rbf", gamma=0.3), backend=backend)

    batch = rows.rows([1, 2, 3])
    assert batch.flags.owndata and batch.flags.c_contiguous
    assert rows.rows(np.array([], dtype=np.int64)).shape == (0, 50)

    # An MLX gather does not bounds-check, so these must be caught before it.
    with pytest.raises(IndexError):
        rows.rows([0, 50])
    with pytest.raises(IndexError):
        rows.rows([-1])
    with pytest.raises(IndexError):
        rows.row(50)
    with pytest.raises(TypeError):
        rows.rows(np.array([0.5]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_kernel_rows_does_not_alias_the_callers_array(backend):
    X = np.random.default_rng(6).normal(size=(40, 4)).astype(np.float32)
    rows = build_kernel_rows(X, KernelSpec(kernel="linear"), backend=backend)
    before = rows.rows([0, 1])
    X[:] = 0.0
    np.testing.assert_array_equal(rows.rows([0, 1]), before)
    assert not np.shares_memory(rows.X, X)


def test_prefetch_is_offered_only_where_a_call_is_expensive():
    X = np.zeros((10, 2), dtype=np.float32)
    assert build_kernel_rows(X, KernelSpec(kernel="linear"), backend="cpu").prefetch_rows == 0
    if gpu_available():
        gpu = build_kernel_rows(X, KernelSpec(kernel="linear"), backend="mlx")
        assert gpu.prefetch_rows > 0


# --------------------------------------------------------------- a host with no GPU


def test_the_kernels_run_where_they_are_told_and_read_no_configuration(monkeypatch, data):
    """backend="mlx" with a device the caller resolved: MLX's CPU, on a GPU-less host.

    Where a GPU-fitted model runs in a process without a GPU is the estimator's
    decision (inference_device), from the configured device. The kernels used to
    read that configuration themselves, below the layer allowed to; they now take
    the device as an argument. The answer is the CPU backend's, bit for bit: the
    same float32 code, the same (eager) tile transform, the same blocking.
    """
    A, B = data
    spec = KernelSpec(kernel="rbf", gamma=0.08)
    weights = np.random.default_rng(3).normal(size=(B.shape[0], 4))
    expected = {
        "matrix": kernel_matrix(spec, A, B, backend="cpu"),
        "rows": build_kernel_rows(B, spec, backend="cpu").rows([0, 5, 9]),
        "basis": _combine(spec, A, B, weights, "cpu"),
    }

    def no_configuration():
        raise AssertionError("a kernel read the user's configuration")

    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    monkeypatch.setattr("mlxlearn._common.device.get_config", no_configuration)
    rows = build_kernel_rows(B, spec, backend="mlx", device="cpu")
    assert rows.prefetch_rows == 0  # the CPU's policy, not the GPU's
    actual = {
        "matrix": kernel_matrix(spec, A, B, backend="mlx", device="cpu"),
        "rows": rows.rows([0, 5, 9]),
        "basis": KernelBasis(spec).combine(A, B, weights, backend="mlx", device="cpu"),
    }
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value, err_msg=name)
