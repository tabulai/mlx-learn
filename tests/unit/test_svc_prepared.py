"""Prepared SVC inference must preserve mutations, precision, and ownership."""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pytest

from mlxlearn._kernels import svc_cpu
from mlxlearn._kernels.kernels import KernelSpec, kernel_matrix
from mlxlearn._kernels.svc_predict import PreparedKernel


@pytest.fixture
def fixture():
    rng = np.random.default_rng(924)
    supports = rng.normal(size=(37, 7))
    queries = rng.normal(size=(13, 7))
    weights = rng.normal(size=(37, 3))
    bias = rng.normal(size=3)
    return supports, queries, weights, bias


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_prepared_predictions_match_independent_kernel_contraction(fixture, backend, kernel):
    supports, queries, weights, bias = fixture
    spec = KernelSpec(kernel, gamma=0.17, coef0=-0.1, degree=3)
    center = supports.mean(axis=0).astype(np.float32) if kernel == "rbf" else None
    prepared = PreparedKernel(supports, spec, backend=backend, center=center)
    assert prepared.matches(supports, spec, center, backend)
    tile = kernel_matrix(spec, queries, supports, center=center, backend=backend, block_rows=5)
    expected = tile.astype(np.float64) @ weights - bias
    actual = prepared.combine(queries, weights, bias=bias, block_rows=5)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-12)
    right, norms = prepared._right, prepared._norms
    # Neither the query nor weights/bias are cached between calls.
    weights[0] *= -3
    bias += 2
    queries[0] += 1
    expected = kernel_matrix(spec, queries, supports, center=center, backend=backend,
                             block_rows=5).astype(np.float64) @ weights - bias
    np.testing.assert_allclose(prepared.combine(queries, weights, bias=bias, block_rows=5),
                               expected, rtol=1e-12, atol=2e-12)
    assert prepared._right is right
    assert prepared._norms is norms


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prepared_rejects_mutated_supports_center_spec_and_backend(fixture, dtype):
    supports, _, _, _ = fixture
    supports = supports.astype(dtype)
    center = np.zeros(supports.shape[1], dtype)
    spec = KernelSpec("rbf", gamma=0.2)
    prepared = PreparedKernel(supports, spec, backend="cpu", center=center)
    assert not np.shares_memory(supports, prepared._snapshot)
    assert not np.shares_memory(center, prepared._center)
    assert not prepared._snapshot.flags.writeable
    supports[-1, -1] += 0.125
    assert not prepared.matches(supports, spec, center, "cpu")
    supports[-1, -1] -= 0.125
    assert prepared.matches(supports, spec, center, "cpu")
    center[0] = 0.5
    assert not prepared.matches(supports, spec, center, "cpu")
    center[0] = 0
    assert not prepared.matches(supports, replace(spec, gamma=0.3), center, "cpu")
    assert not prepared.matches(supports, spec, center, "mlx")
    assert not prepared.matches(supports[:-1], spec, center, "cpu")
    assert not prepared.matches(supports, spec, center[:-1], "cpu")
    assert not prepared.matches(supports, spec, None, "cpu")


def test_matching_is_exact_at_float32_boundary_not_float64_identity():
    supports = np.ones((5, 3), np.float64)
    spec = KernelSpec("linear")
    prepared = PreparedKernel(supports, spec)
    supports[0, 0] += 2**-30
    assert prepared.matches(supports, spec, None, "cpu")
    supports[0, 0] += 2**-22
    assert not prepared.matches(supports, spec, None, "cpu")
    # A non-RBF center is irrelevant because translating it changes the kernel.
    assert prepared.matches(np.ones((5, 3)), spec, np.ones(3), "cpu")


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_pickle_drops_device_arrays_and_lazily_restores_owned_state(fixture, backend):
    supports, queries, weights, bias = fixture
    spec = KernelSpec("rbf", gamma=0.13)
    center = supports.mean(axis=0).astype(np.float32)
    prepared = PreparedKernel(supports, spec, backend=backend, center=center)
    expected = prepared.combine(queries, weights, bias=bias)
    restored = pickle.loads(pickle.dumps(prepared))
    assert restored._right is restored._right_t is restored._norms is None
    assert restored._center_device is None
    assert not restored._snapshot.flags.writeable
    assert not restored._center.flags.writeable
    assert restored.matches(supports, spec, center, backend)
    np.testing.assert_array_equal(restored.combine(queries, weights, bias=bias), expected)
    assert restored._right is not None


def test_empty_support_and_empty_queries(fixture):
    _, queries, _, bias = fixture
    prepared = PreparedKernel(np.empty((0, 7)), KernelSpec("rbf"))
    weights = np.empty((0, 3))
    np.testing.assert_array_equal(prepared.combine(queries, weights, bias=bias),
                                  np.broadcast_to(-bias, (len(queries), 3)))
    assert prepared.combine(np.empty((0, 7)), weights, bias=bias).shape == (0, 3)


@pytest.mark.parametrize("shape", [(), (1,), (1, 7)])
def test_prepared_preserves_broadcastable_centers(fixture, shape):
    supports, queries, weights, bias = fixture
    center = np.full(shape, 0.25, np.float32)
    spec = KernelSpec("rbf", gamma=0.2)
    prepared = PreparedKernel(supports, spec, center=center)
    assert prepared.matches(supports, spec, center, "cpu")
    expected = kernel_matrix(spec, queries, supports, center=center).astype(np.float64) @ weights - bias
    np.testing.assert_allclose(prepared.combine(queries, weights, bias=bias), expected,
                               atol=2e-14, rtol=2e-14)


def test_missing_extension_keeps_matching_and_predictions_correct(fixture, monkeypatch):
    supports, queries, weights, bias = fixture
    spec = KernelSpec("rbf", gamma=0.2)
    prepared = PreparedKernel(supports, spec)
    expected = prepared.combine(queries, weights, bias=bias)
    monkeypatch.setattr(svc_cpu, "_native", None)
    assert prepared.matches(supports, spec, None, "cpu")
    supports[0, 0] += 1
    assert not prepared.matches(supports, spec, None, "cpu")
    # The owned preparation remains stable until the caller replaces it.
    np.testing.assert_allclose(prepared.combine(queries, weights, bias=bias), expected,
                               atol=2e-14, rtol=2e-14)


def test_missing_native_matcher_preserves_scalar_center_shape(fixture, monkeypatch):
    supports, _, _, _ = fixture
    spec = KernelSpec("rbf", gamma=0.2)
    center = np.array(0.25, dtype=np.float64)
    prepared = PreparedKernel(supports, spec, center=center)
    monkeypatch.setattr(svc_cpu, "matches_float32", lambda *args: None)
    assert prepared.matches(supports, spec, center, "cpu")
    center[...] += 2**-30  # same represented float32 center, despite a float64 edit
    assert prepared.matches(supports, spec, center, "cpu")
    center[...] = 0.5
    assert not prepared.matches(supports, spec, center, "cpu")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_float32_comparison_covers_simd_tails_and_signed_zero(dtype):
    if svc_cpu._native is None or not hasattr(svc_cpu._native, "matches_float32"):
        pytest.skip("native snapshot matcher is not built")
    source = np.linspace(-1, 1, 15, dtype=dtype).reshape(3, 5)
    snapshot = source.astype(np.float32)
    assert svc_cpu.matches_float32(source, snapshot)
    source[-1, -1] += 1
    assert not svc_cpu.matches_float32(source, snapshot)
    source[-1, -1] -= 1
    source[1, 2] = -0.0
    assert not svc_cpu.matches_float32(source, snapshot)
    assert not svc_cpu.matches_float32(source[:-1], snapshot)


def test_native_float32_comparison_accepts_noncontiguous_and_unaligned_sources():
    if svc_cpu._native is None or not hasattr(svc_cpu._native, "matches_float32"):
        pytest.skip("native snapshot matcher is not built")
    unaligned = np.ndarray((3, 5), np.float64, buffer=bytearray(121), offset=1)
    unaligned[:] = np.arange(15).reshape(3, 5)
    snapshot = unaligned.astype(np.float32)
    assert svc_cpu.matches_float32(unaligned, snapshot)
    assert svc_cpu.matches_float32(unaligned[:, ::-1], snapshot[:, ::-1])
    with pytest.raises(ValueError):
        svc_cpu._native.matches_float32(snapshot.astype(np.int32), snapshot)


def test_prepared_validates_internal_shapes(fixture):
    supports, queries, weights, _ = fixture
    spec = KernelSpec("rbf")
    with pytest.raises(ValueError, match="backend"):
        PreparedKernel(supports, spec, backend="other")
    with pytest.raises(ValueError, match="center"):
        PreparedKernel(supports, spec, center=np.zeros(3))
    prepared = PreparedKernel(supports, spec)
    with pytest.raises(ValueError, match="feature"):
        prepared.combine(queries[:, :-1], weights)
    with pytest.raises(ValueError, match="weights"):
        prepared.combine(queries, weights[:-1])
