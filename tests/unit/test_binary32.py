"""Binary32 matrix products where MLX may substitute TF32 operands.

On M5 (GPU generation 17) MLX runs float32 GEMM with TF32 operands by default:
errors reach ~2e4 binary32 ulps of ``|a| @ |b|``. These tests force the
replacement kernel on every GPU, so its accuracy and layouts are covered on
M1-M4 too, and check that the dispatching helpers leave MLX's own expression
untouched wherever MLX keeps binary32 products.
"""

import itertools

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import _binary32 as binary32

requires_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.mark.parametrize("architecture,may_round", [
    ("applegpu_g13s", False), ("applegpu_g13g", False), ("applegpu_g14d", False),
    ("applegpu_g15s", False), ("applegpu_g16s", False),
    ("applegpu_g17d", True), ("applegpu_g17s", True), ("applegpu_g18p", True),
    ("applegpu_g21x", True), ("", True), ("applegpu", True), ("unknown_gpu", True),
])
def test_only_pre_accelerator_architectures_trust_float32_matmul(architecture, may_round):
    assert binary32.architecture_may_round_float32(architecture) is may_round


def _operands(rows, cols, depth, transpose_a, transpose_b, seed):
    rng = np.random.default_rng(seed)
    a = (rng.normal(size=(depth, rows) if transpose_a else (rows, depth)) * 100).astype(np.float32)
    b = rng.normal(size=(cols, depth) if transpose_b else (depth, cols)).astype(np.float32)
    return a, b


def _exact(a, b, transpose_a, transpose_b):
    left = a.astype(np.float64).T if transpose_a else a.astype(np.float64)
    right = b.astype(np.float64).T if transpose_b else b.astype(np.float64)
    return left @ right, np.abs(left) @ np.abs(right)


def _envelope_violation(products, exact, scale, depth):
    """Largest error beyond ``depth`` binary32 epsilons of ``|a| @ |b|`` (<= 0 passes)."""
    error = np.abs(np.asarray(products, dtype=np.float64) - exact)
    return float(np.max(error - depth * 2.0**-23 * scale - 2.0**-126))


SHAPES = [
    (1, 1, 1), (1, 2, 3), (3, 7, 5), (5, 33, 32), (63, 65, 17), (64, 64, 16),
    (65, 100, 129), (70, 70, 36), (8, 9, 4096), (257, 300, 1024),
    # Few output tiles and a long K: split into partial products.
    (2, 300, 20000), (3, 50, 70001), (128, 128, 50000),
]


@pytest.mark.gpu
@requires_metal
@pytest.mark.parametrize("transpose_b", [False, True], ids=["nn", "nt"])
@pytest.mark.parametrize("transpose_a", [False, True], ids=["a", "at"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_float32_matmul_stays_inside_binary32_envelope(shape, transpose_a, transpose_b):
    rows, cols, depth = shape
    a, b = _operands(rows, cols, depth, transpose_a, transpose_b, sum(shape))
    products = binary32.float32_matmul(mx.array(a), mx.array(b),
                                       transpose_a=transpose_a, transpose_b=transpose_b)
    assert products.shape == (rows, cols) and products.dtype == mx.float32
    exact, scale = _exact(a, b, transpose_a, transpose_b)
    assert _envelope_violation(np.array(products), exact, scale, depth) <= 0


@pytest.mark.gpu
@requires_metal
@pytest.mark.parametrize("c_shape", ["row", "matrix", "row2d", "column"])
@pytest.mark.parametrize("transpose_a,transpose_b", list(itertools.product([False, True], repeat=2)))
@pytest.mark.parametrize("shape", [(70, 65, 33), (64, 64, 16), (3, 50, 70001)],
                         ids=lambda s: "x".join(map(str, s)))
def test_float32_addmm_matches_alpha_product_plus_beta_c(shape, transpose_a, transpose_b, c_shape):
    rows, cols, depth = shape
    a, b = _operands(rows, cols, depth, transpose_a, transpose_b, rows + cols)
    shapes = {"row": (cols,), "matrix": (rows, cols), "row2d": (1, cols), "column": (rows, 1)}
    c = np.random.default_rng(depth).normal(size=shapes[c_shape]).astype(np.float32)
    got = binary32.float32_addmm(mx.array(c), mx.array(a), mx.array(b), alpha=-0.5, beta=2.0,
                                 transpose_a=transpose_a, transpose_b=transpose_b)
    exact, scale = _exact(a, b, transpose_a, transpose_b)
    want = -0.5 * exact + 2.0 * c.astype(np.float64)
    # The product's envelope, plus the rounding of the scaling and the sum.
    error = np.abs(np.array(got).astype(np.float64) - want)
    allowed = depth * 2.0**-23 * 0.5 * scale + 4 * 2.0**-23 * np.abs(want) + 2.0**-120
    assert got.shape == (rows, cols) and float(np.max(error - allowed)) <= 0


@pytest.mark.gpu
@requires_metal
@pytest.mark.parametrize("transpose_b", [False, True])
def test_wide_products_launch_in_column_chunks(monkeypatch, transpose_b):
    a, b = _operands(9, 1000, 12, False, transpose_b, 4)
    whole = np.array(binary32.float32_matmul(mx.array(a), mx.array(b), transpose_b=transpose_b))
    monkeypatch.setattr(binary32, "_MAX_LAUNCH_COLUMNS", 192)
    chunked = np.array(binary32.float32_matmul(mx.array(a), mx.array(b), transpose_b=transpose_b))
    np.testing.assert_array_equal(chunked, whole)


@pytest.mark.gpu
@requires_metal
def test_split_and_whole_k_products_share_the_envelope(monkeypatch):
    a, b = _operands(40, 70, 9000, True, False, 11)
    exact, scale = _exact(a, b, True, False)
    assert binary32._split_span(40, 70, 9000) < 9000
    split = np.array(binary32.float32_matmul(mx.array(a), mx.array(b), transpose_a=True))
    monkeypatch.setattr(binary32, "_SPLIT_MIN_SPAN", 10**9)
    assert binary32._split_span(40, 70, 9000) == 9000
    whole = np.array(binary32.float32_matmul(mx.array(a), mx.array(b), transpose_a=True))
    for products in (split, whole):
        assert _envelope_violation(products, exact, scale, 9000) <= 0


def test_split_spans_are_16_aligned_and_cover_k():
    for rows, cols, depth in [(5, 100, 200000), (64, 64, 1024), (1000, 10, 100001), (3, 3, 1535)]:
        span = binary32._split_span(rows, cols, depth)
        assert span == depth or (span % 16 == 0 and 2 <= -(-depth // span) <= 256)
    assert binary32._split_span(4096, 4096, 100000) == 100000


@pytest.mark.parametrize("bad", ["dtype", "ndim", "inner"])
def test_float32_matmul_rejects_invalid_operands(bad):
    a, b = mx.zeros((3, 4), mx.float32), mx.zeros((4, 5), mx.float32)
    if bad == "dtype":
        a = a.astype(mx.float16)
    elif bad == "ndim":
        b = mx.zeros((4,), mx.float32)
    else:
        b = mx.zeros((3, 5), mx.float32)
    with pytest.raises(TypeError if bad != "inner" else ValueError):
        binary32.float32_matmul(a, b)


@pytest.mark.gpu
@requires_metal
def test_empty_products_are_zero():
    out = binary32.float32_matmul(mx.zeros((3, 0), mx.float32), mx.zeros((0, 4), mx.float32))
    np.testing.assert_array_equal(np.array(out), np.zeros((3, 4), np.float32))
    c = mx.array(np.arange(4, dtype=np.float32))
    out = binary32.float32_addmm(c, mx.zeros((3, 0), mx.float32), mx.zeros((0, 4), mx.float32),
                                 beta=2.0)
    np.testing.assert_array_equal(np.array(out), np.tile(2 * np.arange(4, dtype=np.float32), (3, 1)))


def _count_kernel_calls(monkeypatch, may_round):
    calls = []
    original, original_addmm = binary32.float32_matmul, binary32.float32_addmm

    def counted(*args, **kwargs):
        calls.append(None)  # count only: retaining arrays would inflate peak memory
        return original(*args, **kwargs)

    def counted_addmm(*args, **kwargs):
        calls.append(None)
        return original_addmm(*args, **kwargs)

    monkeypatch.setattr(binary32, "matmul_may_round_float32", lambda: may_round)
    monkeypatch.setattr(binary32, "float32_matmul", counted)
    monkeypatch.setattr(binary32, "float32_addmm", counted_addmm)
    return calls


@pytest.mark.gpu
@requires_metal
@pytest.mark.parametrize("transpose_a,transpose_b", list(itertools.product([False, True], repeat=2)))
def test_helpers_evaluate_mlx_expression_where_matmul_keeps_binary32(
    monkeypatch, transpose_a, transpose_b
):
    calls = _count_kernel_calls(monkeypatch, False)
    a, b = (mx.array(x) for x in _operands(70, 65, 33, transpose_a, transpose_b, 3))
    c = mx.array(np.linspace(-1, 1, 65, dtype=np.float32))
    left = a.T if transpose_a else a
    right = b.T if transpose_b else b
    np.testing.assert_array_equal(
        np.array(binary32.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)),
        np.array(left @ right))
    np.testing.assert_array_equal(
        np.array(binary32.addmm(c, a, b, alpha=0.5, beta=-2.0,
                                transpose_a=transpose_a, transpose_b=transpose_b)),
        np.array(mx.addmm(c, left, right, alpha=0.5, beta=-2.0)))
    assert not calls


@pytest.mark.gpu
@requires_metal
def test_helpers_take_the_kernel_only_for_float32_gpu_gemm(monkeypatch):
    calls = _count_kernel_calls(monkeypatch, True)
    a = mx.array(np.random.default_rng(0).normal(size=(40, 30)).astype(np.float32))
    b = mx.array(np.random.default_rng(1).normal(size=(30, 20)).astype(np.float32))
    exact, scale = _exact(np.array(a), np.array(b), False, False)
    out = binary32.matmul(a, b)
    assert len(calls) == 1 and _envelope_violation(np.array(out), exact, scale, 30) <= 0
    binary32.addmm(mx.zeros((20,), mx.float32), a, b)
    assert len(calls) == 2
    # Matrix-vector shapes, 1-D operands, other dtypes and the CPU keep MLX.
    binary32.matmul(a[:1], b)
    binary32.matmul(a, b[:, :1])
    binary32.matmul(a, b[:, 0])
    binary32.matmul(a.astype(mx.float16), b.astype(mx.float16))
    binary32.addmm(mx.zeros((20,), mx.float16), a.astype(mx.float16), b.astype(mx.float16))
    with mx.stream(mx.cpu):
        binary32.matmul(a, b)
    assert len(calls) == 2


@pytest.mark.gpu
@requires_metal
def test_kernel_route_works_inside_compiled_functions(monkeypatch):
    _count_kernel_calls(monkeypatch, True)

    @mx.compile
    def gradient(residual, X):
        return binary32.matmul(residual, X, transpose_a=True) * 0.5

    rng = np.random.default_rng(5)
    residual = rng.normal(size=(3000, 4)).astype(np.float32)
    X = rng.normal(size=(3000, 70)).astype(np.float32)
    exact, scale = _exact(residual, X, True, False)
    out = np.array(gradient(mx.array(residual), mx.array(X))).astype(np.float64)
    assert _envelope_violation(2 * out, exact, scale, 3000) <= 0
