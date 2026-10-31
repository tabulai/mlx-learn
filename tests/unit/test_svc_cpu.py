"""Independent references for SVC's optional native arithmetic."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from mlxlearn._kernels import svc_cpu
from mlxlearn._kernels.kernels import KernelSpec


@pytest.fixture
def native():
    if svc_cpu._native is None:
        pytest.skip("optional native SVC arithmetic extension is not built")
    return svc_cpu._native


@pytest.mark.parametrize("shape", [(7, 19, 1), (13, 21, 6), (0, 19, 3), (5, 0, 3)])
@pytest.mark.parametrize("biased", [False, True])
def test_contraction_matches_float64_reference(native, shape, biased):
    rows, support, pairs = shape
    rng = np.random.default_rng(94)
    tile = rng.normal(size=(rows, support)).astype(np.float32)
    weights = rng.normal(size=(support, pairs))
    bias = rng.normal(size=pairs) if biased else None
    actual = svc_cpu.contract(tile, weights, bias)
    expected = tile.astype(np.float64) @ weights
    if bias is not None:
        expected -= bias
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)


def test_contraction_retains_small_cancelling_weight_terms(native):
    tile = np.ones((2, 300), np.float32)
    weights = np.tile([1e16, 1.0, -1e16], 100).reshape(-1, 1)
    expected = math.fsum(weights[:, 0]) - 0.25
    np.testing.assert_array_equal(svc_cpu.contract(tile, weights, np.array([0.25])),
                                  np.full((2, 1), expected))


@pytest.mark.parametrize("shape", [(257, 2203, 1), (257, 2203, 6), (11, 17003, 3)])
def test_contraction_bounded_tiles_cover_row_and_support_tails(native, shape):
    rows, support, pairs = shape
    rng = np.random.default_rng(93)
    tile = rng.normal(size=(rows, support)).astype(np.float32)
    weights = rng.normal(size=(support, pairs))
    bias = rng.normal(size=pairs)
    expected = tile.astype(np.float64) @ weights - bias
    np.testing.assert_allclose(svc_cpu.contract(tile, weights, bias), expected,
                               atol=3e-12, rtol=5e-13)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_variance_and_mean_match_independent_references(native, dtype):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(231, 13)).astype(dtype)
    assert svc_cpu.variance(X) == pytest.approx(np.var(X, dtype=np.float64), rel=3e-15)
    np.testing.assert_allclose(svc_cpu.column_mean(X), X.mean(axis=0, dtype=np.float64),
                               atol=5e-16, rtol=5e-15)


@pytest.mark.parametrize("offset,scale", [(1e12, 2**-10), (0.0, 1e-10), (1e30, 1e22)])
def test_statistics_preserve_small_variation_around_large_offsets(native, offset, scale):
    X = offset + np.arange(35, dtype=np.float64).reshape(7, 5) * scale
    # Shift the independent oracle before its mean, including on platforms
    # where np.longdouble is the same precision as float64.
    shifted = X - X[0, 0]
    mean = math.fsum(shifted.ravel()) / X.size
    expected = math.fsum((float(x) - mean) ** 2 for x in shifted.ravel()) / X.size
    assert svc_cpu.variance(X) == pytest.approx(expected, rel=3e-15, abs=0)
    expected_mean = [X[0, c] + math.fsum(X[:, c] - X[0, c]) / len(X)
                     for c in range(X.shape[1])]
    np.testing.assert_array_equal(svc_cpu.column_mean(X), expected_mean)
    assert svc_cpu.variance(np.full((4, 3), offset)) == 0.0


@pytest.mark.parametrize("kernel,degree", [
    ("linear", 3), ("rbf", 3), ("sigmoid", 3),
    ("poly", 0), ("poly", 1), ("poly", 2), ("poly", 3), ("poly", 9),
])
def test_kernel_diagonal_uses_float64_arithmetic(native, kernel, degree):
    X = np.array([[0.0, 0.25, -0.5], [3, 4, 0], [-1e8, 1.25, -1e8]], np.float32)
    spec = KernelSpec(kernel, gamma=0.125, coef0=-3.0, degree=degree)
    squares = np.sum(X.astype(np.float64) ** 2, axis=1)
    if kernel == "linear":
        expected = squares
    elif kernel == "rbf":
        expected = np.ones(len(X))
    elif kernel == "sigmoid":
        expected = np.tanh(spec.gamma * squares + spec.coef0)
    else:
        expected = (spec.gamma * squares + spec.coef0) ** degree
    np.testing.assert_allclose(svc_cpu.diagonal(X, spec), expected, atol=2e-16, rtol=5e-15)


def _aggregate_reference(decision, classes):
    rows = decision.shape[0]
    ovo = np.zeros((rows, classes), np.int64)
    ovr = np.zeros((rows, classes), np.int64)
    confidence = np.zeros((rows, classes), np.float64)
    pair = 0
    for i in range(classes):
        for j in range(i + 1, classes):
            value = decision[:, pair]
            ovo[:, i] += value > 0
            ovo[:, j] += ~(value > 0)
            ovr[:, i] += ~(value < 0)
            ovr[:, j] += value < 0
            confidence[:, i] += value
            confidence[:, j] -= value
            pair += 1
    return ovo.argmax(axis=1), ovr + confidence / (3 * (np.abs(confidence) + 1))


@pytest.mark.parametrize("classes", [2, 3, 5, 10])
def test_multiclass_matches_vote_and_confidence_references(native, classes):
    rng = np.random.default_rng(81)
    decisions = rng.normal(size=(27, classes * (classes - 1) // 2))
    decisions[0] = 0  # exact pair boundary rules differ between OVO and OVR
    decisions[1] = -0.0
    decisions[2, ::2] = 0
    ovo, ovr = _aggregate_reference(decisions, classes)
    np.testing.assert_array_equal(svc_cpu.ovo_vote(decisions, classes), ovo)
    np.testing.assert_array_equal(svc_cpu.ovr_decision(decisions, classes), ovr)
    np.testing.assert_array_equal(svc_cpu.ovo_vote(decisions, classes, break_ties=True),
                                  ovr.argmax(axis=1))
    assert ovo[0] == classes - 1
    assert ovr[0].argmax() == 0


def test_multiclass_empty_rows_and_nonfinite_legacy_argmax(native):
    assert svc_cpu.ovo_vote(np.empty((0, 6)), 4).shape == (0,)
    assert svc_cpu.ovr_decision(np.empty((0, 6)), 4).shape == (0, 4)
    decision = np.array([[1.0, np.nan, 1.0], [np.inf, 0, -np.inf]])
    with np.errstate(invalid="ignore"):
        ovo, ovr = _aggregate_reference(decision, 3)
    np.testing.assert_array_equal(svc_cpu.ovo_vote(decision, 3), ovo)
    np.testing.assert_array_equal(svc_cpu.ovo_vote(decision, 3, break_ties=True),
                                  ovr.argmax(axis=1))


def test_boundary_wrappers_only_allocate_when_native_is_available(monkeypatch):
    monkeypatch.setattr(svc_cpu, "_native", None)
    assert svc_cpu.contract(None, None) is None
    assert svc_cpu.variance(None) is None
    assert svc_cpu.column_mean(None) is None
    assert svc_cpu.diagonal(None, None) is None
    assert svc_cpu.ovo_vote(None, 3) is None
    assert svc_cpu.ovr_decision(None, 3) is None


def test_native_rejects_bad_shapes_types_aliases_and_outputs(native):
    tile = np.ones((2, 3), np.float32)
    weights = np.ones((3, 2))
    output = np.empty((2, 2))
    with pytest.raises(ValueError):
        native.contract(tile.astype(np.float64), weights, None, output)
    with pytest.raises(ValueError):
        native.contract(tile, weights[:2], None, output)
    with pytest.raises(ValueError):
        native.contract(tile, weights, np.ones(3), output)
    with pytest.raises(ValueError):
        native.contract(tile, weights, None, weights[:2])
    with pytest.raises((ValueError, BufferError)):
        native.contract(tile[:, ::-1], weights, None, output)
    unaligned = np.ndarray((2, 2), dtype=np.float64, buffer=bytearray(33), offset=1)
    with pytest.raises(ValueError):
        native.contract(tile, weights, None, unaligned)
    output.flags.writeable = False
    with pytest.raises((ValueError, BufferError)):
        native.contract(tile, weights, None, output)
    with pytest.raises(ValueError):
        native.variance(np.ones((2, 3), dtype=np.int32))
    with pytest.raises(ValueError):
        native.variance(np.empty((0, 3)))
    with pytest.raises(ValueError):
        native.column_mean(np.ones((2, 3)), np.empty(2))
    with pytest.raises(ValueError):
        native.diagonal(tile, 2, 1.0, 0.0, -1, np.empty(2))
    with pytest.raises(ValueError):
        native.aggregate(np.ones((2, 3)), 4, 0, np.empty(2, np.int64))
    with pytest.raises(ValueError):
        native.aggregate(np.ones((2, 3)), 3, 1, np.empty((2, 2)))


def test_wrappers_copy_unaligned_inputs_before_native_call(native):
    tile = np.ndarray((2, 3), dtype=np.float32, buffer=bytearray(25), offset=1)
    weights = np.ndarray((3, 2), dtype=np.float64, buffer=bytearray(49), offset=1)
    bias = np.ndarray((2,), dtype=np.float64, buffer=bytearray(17), offset=1)
    tile[:] = [[1, 2, 3], [4, 5, 6]]
    weights[:] = [[2, 3], [4, 5], [6, 7]]
    bias[:] = [0.5, -0.5]
    np.testing.assert_array_equal(svc_cpu.contract(tile, weights, bias),
                                  tile.astype(np.float64) @ weights - bias)
    np.testing.assert_array_equal(svc_cpu.column_mean(tile), [2.5, 3.5, 4.5])


def test_diagonal_defers_precision_and_oversized_integer_degrees(native):
    X64 = np.array([[1 + 2**-30]], np.float64)
    assert svc_cpu.diagonal(X64, KernelSpec("linear")) is None
    assert svc_cpu.diagonal(X64.astype(np.float32),
                            KernelSpec("poly", degree=sys.maxsize + 1)) is None
