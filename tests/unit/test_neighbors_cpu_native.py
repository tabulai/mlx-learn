"""Native CPU search: independent exact ordering and checked buffer boundaries."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from mlxlearn._kernels import _neighbors_cpu_native as native
from tests.unit.test_neighbors_cpu_exact import exact_oracle

_BUFFER_ERRORS = (TypeError, ValueError, BufferError, OverflowError)


def _run(train, queries, k, *, exclude_self=False, offset=0, return_distance=True):
    distances = np.full((len(queries), k), -773.25, np.float64) if return_distance else None
    indices = np.full((len(queries), k), -317, np.int64)
    result = native.query(train, queries, k, exclude_self, offset, distances, indices)
    assert result is None
    return distances, indices


def _check(train, queries, k, *, exclude_self=False, offset=0, return_distance=True):
    distances, indices = _run(
        train, queries, k, exclude_self=exclude_self, offset=offset,
        return_distance=return_distance,
    )
    if len(queries):
        expected = exact_oracle(train, queries, len(train))
        if exclude_self:
            expected = np.stack([row[row != offset + i][:k] for i, row in enumerate(expected)])
        else:
            expected = expected[:, :k]
    else:
        expected = np.empty((0, k), np.int64)
    np.testing.assert_array_equal(indices, expected)
    if return_distance:
        assert distances.dtype == np.float64 and np.isfinite(distances).all()
        delta = train[expected].astype(np.float64) - queries.astype(np.float64)[:, None]
        reference = np.sqrt(np.sum(delta * delta, axis=2))
        np.testing.assert_allclose(distances, reference, rtol=1e-13, atol=0)
    else:
        assert distances is None
    return distances, indices


@pytest.mark.parametrize("features", [1, 8, 32, 128])
@pytest.mark.parametrize("return_distance", [False, True])
def test_native_arbitrary_finite_float32_patterns(features, return_distance):
    rng = np.random.default_rng(4021 + features)
    bits = rng.integers(0, 2**32, size=(38, features), dtype=np.uint32)
    bits = (bits & np.uint32(0x807fffff)) | (
        rng.integers(0, 255, size=bits.shape, dtype=np.uint32) << np.uint32(23)
    )
    values = bits.view(np.float32)
    train, queries = values[:31].copy(), values[31:].copy()
    _check(train, queries, 7, return_distance=return_distance)


@pytest.mark.parametrize("k", [1, 5, 12])
def test_native_subnormals_extremes_and_float64_rounded_ties(k):
    train = np.zeros((12, 4), np.float32)
    train[:4, 0] = 1
    train[:4, 1] = [2**-90, 2**-100, 0, 2**-100]
    train[4:6, 0] = [2**-149, 2**-148]
    train[6:8, 0] = 1e30
    train[6:8, 1] = [2, 1]
    train[8:10, 0] = [-1, 1]
    train[10:12, 0] = [np.finfo(np.float32).max, -np.finfo(np.float32).max]
    queries = np.array([[0, 0, 0, 0], [1e30, 1, 0, 0]], np.float32)
    _check(train, queries, k)


@pytest.mark.parametrize("features", [4, 32, 64, 65])
def test_native_moderate_inputs_keep_true_order_when_float32_distances_round_equal(features):
    train = np.zeros((17, features), np.float32)
    train[:, 0] = 1
    train[:, 1] = np.arange(17, 0, -1, dtype=np.float32) * np.float32(2**-20)
    train[-2:, 1] = 0
    train[3, 0] = -1
    queries = np.zeros((3, features), np.float32)
    queries[1, 0] = np.nextafter(np.float32(1), np.float32(0))
    queries[2, 1] = np.float32(2**-20)
    _check(train, queries, 5)


@pytest.mark.parametrize("features", [64, 65])
def test_native_float32_screen_coordinate_boundaries(features):
    low, high = np.float32(2**-32), np.float32(2**40)
    bounds = np.array([
        0, -0.0, low, np.nextafter(low, np.float32(0)),
        np.nextafter(low, np.float32(np.inf)), high,
        np.nextafter(high, np.float32(0)), np.nextafter(high, np.float32(np.inf)),
    ], np.float32)
    train = np.zeros((16, features), np.float32)
    train[:, 0] = high
    train[:8, 1], train[8:, 1] = bounds, -bounds
    queries = np.zeros((3, features), np.float32)
    queries[1] = train[7]
    queries[2, 0] = np.nextafter(high, np.float32(np.inf))
    _check(train, queries, 9)


@pytest.mark.parametrize("k", [1, 5, 23])
def test_native_equal_distance_cutoff_uses_original_training_index(k):
    train = np.zeros((23, 4), np.float32)
    train[::4, 0], train[1::4, 0] = 1, -1
    train[2::4, 1], train[3::4, 1] = 1, -1
    queries = np.zeros((2, 4), np.float32)
    _, indices = _check(train, queries, k)
    np.testing.assert_array_equal(indices, np.tile(np.arange(k), (2, 1)))


@pytest.mark.parametrize("return_distance", [False, True])
def test_native_partial_first_eight_features_cannot_discard_late_true_winners(return_distance):
    train = np.zeros((8, 16), np.float32)
    train[:, 0] = 1
    train[:, 8] = [2**-20, 2**-21, 2**-22, 2**-28, 2**-30, 2**-31, 0, 0]
    queries = np.zeros((2, 16), np.float32)
    queries[1, 0] = 1
    # All partial sums tie; for the first query the complete float32 sums
    # also round equal. The winning rows occur after the initial k candidates.
    _, indices = _check(train, queries, 3, return_distance=return_distance)
    np.testing.assert_array_equal(indices, [[6, 7, 5], [6, 7, 5]])


def test_native_true_ties_have_identical_distances_despite_different_float64_reductions():
    train = np.zeros((2, 16), np.float32)
    train[:, 0] = 2**30
    train[0, [4, 8, 12, 1, 2, 3, 5]] = 8
    train[1, [1, 3, 5, 7, 9, 11, 13]] = 8
    queries = np.zeros((1, 16), np.float32)
    distances, indices = _check(train, queries, 2)
    # The exact squared sums are the same integer, but separate SIMD-lane
    # reductions can round them differently before taking a square root.
    expected = math.sqrt(float(2**60 + 7 * 64))
    assert expected > 2**30
    np.testing.assert_array_equal(indices, [[0, 1]])
    np.testing.assert_array_equal(distances, [[expected, expected]])


def test_native_exact_key_distance_conversion_matches_independent_rational_rounding():
    points = np.zeros((8, 16), np.float32)
    points[0, 0], points[1, 0] = 2**-149, 2**-148
    points[2:6, 0] = 1
    points[2, 1] = 2**-27
    points[3, 1:3] = 2**-27
    points[4, 1], points[4, 2:4] = 2**-26, 2**-27
    points[5, 1:3], points[5, 3] = 2**-27, 2**-40
    points[6, :2] = [np.finfo(np.float32).max, -np.finfo(np.float32).max]
    points[7, :3] = [1e30, -1e30, 1]
    # Every nonzero duplicate pair forces exact-key comparison, covering small
    # keys, half-way/guard/sticky rounding, and the largest float32 exponents.
    train = np.repeat(points, 2, axis=0)
    queries = np.zeros((1, 16), np.float32)
    distances, indices = _check(train, queries, len(train))
    expected = [math.sqrt(float(sum(Fraction(float(value)) ** 2 for value in train[index])))
                for index in indices[0]]
    np.testing.assert_array_equal(distances, [expected])


@pytest.mark.parametrize("offset", [0, 4, 17, 21])
@pytest.mark.parametrize("all_neighbors", [False, True])
def test_native_self_queries_exclude_only_the_correct_offset_row(offset, all_neighbors):
    train = np.zeros((23, 8), np.float32)
    train[4:8, :2] = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    train[8:12, 0] = 1
    train[8:12, 1] = [2**-80, 2**-100, 2**-90, 2**-100]
    train[12:, 0] = np.arange(11, dtype=np.float32)
    queries = train[offset:offset + 4].copy()
    k = len(train) - 1 if all_neighbors else 3
    _, indices = _check(train, queries, k, exclude_self=True, offset=offset)
    assert not np.any(indices == np.arange(offset, offset + len(queries))[:, None])


def test_native_readonly_inputs_are_accepted_and_never_modified():
    train = np.arange(96, dtype=np.float32).reshape(24, 4)
    queries = train[[1, 8, 23]].copy()
    train_bits, query_bits = train.view(np.uint32).copy(), queries.view(np.uint32).copy()
    train.setflags(write=False)
    queries.setflags(write=False)
    _check(train, queries, 5)
    np.testing.assert_array_equal(train.view(np.uint32), train_bits)
    np.testing.assert_array_equal(queries.view(np.uint32), query_bits)
    assert not train.flags.writeable and not queries.flags.writeable


@pytest.mark.parametrize("return_distance", [False, True])
def test_native_zero_query_rows_are_valid(return_distance):
    _check(np.zeros((8, 4), np.float32), np.empty((0, 4), np.float32), 3,
           return_distance=return_distance)


def _arguments():
    train = np.arange(136, dtype=np.float32).reshape(17, 8)
    queries = train[[1, 8, 16]].copy()
    distances = np.full((3, 5), -773.25, np.float64)
    indices = np.full((3, 5), -317, np.int64)
    return [train, queries, 5, False, 0, distances, indices]


def _unaligned_like(values):
    result = np.ndarray(values.shape, dtype=values.dtype,
                        buffer=bytearray(values.nbytes + 1), offset=1)
    result[:] = values
    assert result.flags.c_contiguous and not result.flags.aligned
    return result


@pytest.mark.parametrize("argument", [0, 1, 5, 6], ids=["train", "queries", "distances", "indices"])
@pytest.mark.parametrize("kind", [
    "wrong_dtype", "non_native", "fortran", "strided", "negative_stride", "unaligned", "wrong_ndim",
])
def test_native_rejects_invalid_buffer_layouts_and_dtypes(argument, kind):
    args = _arguments()
    value = args[argument]
    if kind == "wrong_dtype":
        dtype = np.float64 if argument in (0, 1) else np.float32 if argument == 5 else np.int32
        invalid = value.astype(dtype)
    elif kind == "non_native":
        invalid = value.astype(value.dtype.newbyteorder("S"))
    elif kind == "fortran":
        invalid = np.asfortranarray(value)
    elif kind == "strided":
        invalid = np.empty((2 * len(value), value.shape[1]), dtype=value.dtype)[::2]
        invalid[:] = value
    elif kind == "negative_stride":
        invalid = value[::-1]
    elif kind == "unaligned":
        invalid = _unaligned_like(value)
    else:
        invalid = value.reshape(-1)
    args[argument] = invalid
    with pytest.raises(_BUFFER_ERRORS):
        native.query(*args)


@pytest.mark.parametrize("argument", [5, 6], ids=["distances", "indices"])
def test_native_rejects_readonly_outputs(argument):
    args = _arguments()
    args[argument].setflags(write=False)
    with pytest.raises(_BUFFER_ERRORS):
        native.query(*args)


@pytest.mark.parametrize("argument", [0, 1], ids=["train", "queries"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_native_rejects_nonfinite_coordinates(argument, bad):
    args = _arguments()
    args[argument][-1, -1] = bad
    with pytest.raises(ValueError, match="finite"):
        native.query(*args)


@pytest.mark.parametrize("case", [
    "no_train", "no_features", "feature_mismatch", "short_distances", "wide_distances",
    "short_indices", "wide_indices", "zero_k", "negative_k", "large_k", "self_large_k",
    "negative_offset", "offset_past_train", "self_rows_past_train", "float_k", "huge_k",
    "float_offset", "huge_offset",
])
def test_native_rejects_invalid_dimensions_counts_and_offsets(case):
    args = _arguments()
    if case == "no_train":
        args[0] = np.empty((0, 8), np.float32)
    elif case == "no_features":
        args[0], args[1] = np.empty((17, 0), np.float32), np.empty((3, 0), np.float32)
    elif case == "feature_mismatch":
        args[1] = np.zeros((3, 9), np.float32)
    elif case in {"short_distances", "wide_distances", "short_indices", "wide_indices"}:
        index = 5 if "distances" in case else 6
        shape = (2, 5) if case.startswith("short") else (3, 6)
        args[index] = np.empty(shape, args[index].dtype)
    elif case in {"zero_k", "negative_k", "large_k", "float_k", "huge_k"}:
        args[2] = {"zero_k": 0, "negative_k": -1, "large_k": 18,
                   "float_k": 5.0, "huge_k": 2**100}[case]
    elif case == "self_large_k":
        args[2], args[3] = 17, True
        args[5], args[6] = np.empty((3, 17), np.float64), np.empty((3, 17), np.int64)
    else:
        args[3] = True
        args[4] = {"negative_offset": -1, "offset_past_train": 18,
                   "self_rows_past_train": 15, "float_offset": 1.0, "huge_offset": 2**100}[case]
    with pytest.raises(_BUFFER_ERRORS):
        native.query(*args)


@pytest.mark.parametrize("argument", [0, 1, 6], ids=["train", "queries", "indices"])
def test_native_required_buffers_cannot_be_none(argument):
    args = _arguments()
    args[argument] = None
    with pytest.raises(_BUFFER_ERRORS):
        native.query(*args)


@pytest.mark.parametrize("case", ["outputs", "distance_train", "index_train", "distance_queries", "index_queries"])
def test_native_rejects_overlapping_output_storage(case):
    args = _arguments()
    if case == "outputs":
        args[6] = args[5].view(np.int64)
    else:
        output = 5 if case.startswith("distance") else 6
        source = 0 if case.endswith("train") else 1
        # Leave enough storage for both valid shapes, with one aligned backing
        # buffer. Each individual view is C-contiguous and correctly typed.
        storage = np.zeros(max(args[source].nbytes, args[output].nbytes), np.uint8)
        input_shape, output_shape = args[source].shape, args[output].shape
        args[source] = np.ndarray(input_shape, np.float32, buffer=storage)
        args[output] = np.ndarray(output_shape, args[output].dtype, buffer=storage)
    with pytest.raises(ValueError, match="overlap"):
        native.query(*args)
