"""SVC Metal fusion against independent double-precision kernel formulas."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._kernels import svc_fused
from mlxlearn._kernels.kernels import KernelSpec, _transform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
@pytest.mark.parametrize("shape", [(1, 1), (3, 31), (17, 257)])
def test_transform_matches_portable_on_odd_tiles(kernel, shape):
    rng = np.random.default_rng(771)
    a = mx.array(rng.normal(size=(shape[0], 33)).astype(np.float32))
    b = mx.array(rng.normal(size=(shape[1], 33)).astype(np.float32))
    spec = KernelSpec(kernel, gamma=0.03, coef0=-0.2, degree=3)
    dot = a @ b.T
    a_sq, b_sq = mx.sum(a * a, axis=1), mx.sum(b * b, axis=1)
    expected = _transform(mx, spec, dot, a_sq, b_sq)
    actual = svc_fused.transform(spec, dot, a_sq, b_sq)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=3e-7, atol=1e-7)


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 8, 17, 2**32 + 1])
def test_transform_polynomial_preserves_integer_power_and_negative_bases(degree):
    base = np.array([[-1.0, -0.5, 0, 0.5, 1]], dtype=np.float32)
    spec = KernelSpec("poly", gamma=1.0, coef0=0, degree=degree)
    actual = np.asarray(svc_fused.transform(spec, mx.array(base)))
    np.testing.assert_allclose(actual, np.power(base.astype(np.float64), degree), atol=0, rtol=0)


@pytest.mark.parametrize("features", [1, 7, 32, 33, 128, 129, 769, 4097])
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_row_matches_direct_double_kernel(features, kernel):
    rng = np.random.default_rng(723 + features)
    x = rng.normal(size=(67, features)).astype(np.float32)
    # Duplicates test the exact RBF diagonal and no distance cancellation.
    x[-1] = x[3]
    spec = KernelSpec(kernel, gamma=0.5 / features, coef0=0.1, degree=3)
    x64 = x.astype(np.float64)
    if kernel == "rbf":
        expected = np.exp(-spec.gamma * np.sum((x64 - x64[3]) ** 2, axis=1))
    else:
        dot = x64 @ x64[3]
        expected = {
            "linear": lambda: dot,
            "poly": lambda: (spec.gamma * dot + spec.coef0) ** spec.degree,
            "sigmoid": lambda: np.tanh(spec.gamma * dot + spec.coef0),
        }[kernel]()
    actual = np.asarray(svc_fused.row(spec, mx.array(x), 3))
    # Linear absolute error scales with row width; transformed kernels have a
    # bounded output scale on this fixture. This compares the formula itself,
    # without using the portable MLX row evaluator as an oracle.
    atol = 8e-7 * np.sqrt(features) if kernel == "linear" else 2e-7
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=atol)
    if kernel == "rbf":
        assert actual[3] == actual[-1] == 1.0


def test_direct_rbf_row_preserves_small_distance_at_large_offsets():
    x = np.full((33, 65), 2**20, dtype=np.float32)
    x[:, 0] += np.arange(33, dtype=np.float32) * 0.125
    x[:, -1] -= np.arange(33, dtype=np.float32) * 0.25
    delta = x.astype(np.float64) - x[11].astype(np.float64)
    expected = np.exp(-0.2 * np.sum(delta * delta, axis=1))
    actual = np.asarray(svc_fused.row(KernelSpec("rbf", gamma=0.2), mx.array(x), 11))
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-8)
    assert actual[11] == 1


def test_rbf_transform_clamps_small_negative_squared_distance():
    dot = mx.array([[1.0, 1.0 + 2**-23]], dtype=mx.float32)
    actual = svc_fused.transform(KernelSpec("rbf", gamma=1e6), dot, mx.array([1.0]), mx.ones(2))
    np.testing.assert_array_equal(np.asarray(actual), [[1.0, 1.0]])


def test_rbf_overflow_underflows_to_zero_without_nan():
    x = mx.array([[0.0, 0.0], [1e20, -1e20]], dtype=mx.float32)
    actual = svc_fused.row(KernelSpec("rbf"), x, 0)
    np.testing.assert_array_equal(np.asarray(actual), [1.0, 0.0])


def test_views_are_made_contiguous_before_fused_reads():
    x = mx.arange(77, dtype=mx.float32).reshape(7, 11).T
    spec = KernelSpec("linear")
    np.testing.assert_allclose(np.asarray(svc_fused.row(spec, x, 4)), np.asarray(x @ x[4]))
    dot = mx.arange(15, dtype=mx.float32).reshape(3, 5).T
    actual = svc_fused.transform(KernelSpec("poly", gamma=1, degree=2), dot)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(dot) ** 2)


def test_shape_and_type_checks_fail_before_gpu_indexing():
    with pytest.raises(ValueError, match="float32 matrix"):
        svc_fused.row(KernelSpec("linear"), mx.ones((2, 3), dtype=mx.float16), 0)
    with pytest.raises(TypeError):
        svc_fused.row(KernelSpec("linear"), mx.ones((2, 3)), 0.5)
    with pytest.raises(IndexError, match="bounds"):
        svc_fused.row(KernelSpec("linear"), mx.ones((2, 3)), -1)
    with pytest.raises(IndexError, match="bounds"):
        svc_fused.row(KernelSpec("linear"), mx.ones((2, 3)), 2)
    with pytest.raises(ValueError, match="norms"):
        svc_fused.transform(KernelSpec("rbf"), mx.ones((2, 3)), mx.ones(3), mx.ones(2))


def test_empty_transform_does_not_launch_gpu_work():
    dot = mx.empty((0, 3), dtype=mx.float32)
    assert svc_fused.transform(KernelSpec("rbf"), dot) is dot
