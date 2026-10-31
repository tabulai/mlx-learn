"""Direct selection correctness at SIMD, partition, and precision boundaries."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import _neighbors_fused as core

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not mx.metal.is_available(), reason="requires Metal",
)]


def _oracle(train, queries, k, *, offset=None):
    delta = queries[:, None, :] - train[None, :, :]
    # Float32 pair arithmetic with a high-precision accumulation reference.
    squared = np.sum((delta * delta).astype(np.float64), axis=2).astype(np.float32)
    if offset is not None:
        squared[np.arange(queries.shape[0]), np.arange(queries.shape[0]) + offset] = np.inf
    indices = np.broadcast_to(np.arange(train.shape[0]), squared.shape)
    order = np.lexsort((indices, squared), axis=1)[:, :k]
    return np.take_along_axis(squared, order, axis=1), order


@pytest.mark.parametrize("n_query,n_train,features,k", [
    (1, 1, 1, 1), (3, 31, 3, 5), (5, 33, 7, 32), (7, 65, 8, 24),
    (8, 513, 32, 10), (1, 8193, 5, 32), (9, 2051, 1, 24),
    (3, 67, 769, 7), (257, 515, 2, 5),
])
def test_direct_selection_matches_float32_oracle(n_query, n_train, features, k):
    rng = np.random.default_rng(n_train + features)
    train = rng.normal(size=(n_train, features)).astype(np.float32)
    queries = rng.normal(size=(n_query, features)).astype(np.float32)
    expected_d, expected_i = _oracle(train, queries, k)
    distances, indices = core.fused_neighbors(mx.array(train), mx.array(queries), k)
    np.testing.assert_array_equal(np.asarray(indices), expected_i)
    np.testing.assert_allclose(np.asarray(distances), expected_d, rtol=2e-7, atol=0)


@pytest.mark.parametrize("k", [1, 5, 24, 32])
def test_all_winners_in_one_lane_and_partition(k, monkeypatch):
    train = np.full((4097, 2), 1000, dtype=np.float32)
    winners = np.arange(32) * 32
    train[winners, 0] = np.arange(1, 33, dtype=np.float32)
    train[winners, 1] = 0
    query = np.zeros((1, 2), dtype=np.float32)
    for parts in [1, 4, 32]:
        monkeypatch.setattr(core, "_scan_partitions", lambda n, m, parts=parts: parts)
        distances, indices = core.fused_neighbors(mx.array(train), mx.array(query), k)
        np.testing.assert_array_equal(np.asarray(indices)[0], winners[:k])
        np.testing.assert_array_equal(np.asarray(distances)[0], np.arange(1, k + 1) ** 2)


@pytest.mark.parametrize("k", [1, 5, 24, 32])
def test_ties_at_selection_boundary_choose_lowest_indices(k, monkeypatch):
    train = mx.zeros((2049, 3), mx.float32)
    queries = mx.zeros((3, 3), mx.float32)
    for parts in [1, 7, 32]:
        monkeypatch.setattr(core, "_scan_partitions", lambda n, m, parts=parts: parts)
        distances, indices = core.fused_neighbors(train, queries, k)
        np.testing.assert_array_equal(np.asarray(indices), np.tile(np.arange(k), (3, 1)))
        np.testing.assert_array_equal(np.asarray(distances), 0)


def test_self_exclusion_uses_global_query_offset():
    train = np.zeros((1025, 3), dtype=np.float32)
    train[:, 0] = np.arange(1025, dtype=np.float32)
    offset = 513
    queries = train[offset:offset + 19]
    distances, indices = core.fused_neighbors(
        mx.array(train), mx.array(queries), 32, exclude_self=True, query_offset=offset,
    )
    expected_d, expected_i = _oracle(train, queries, 32, offset=offset)
    np.testing.assert_array_equal(np.asarray(indices), expected_i)
    np.testing.assert_array_equal(np.asarray(distances), expected_d)


@pytest.mark.parametrize("exclude_self", [False, True])
def test_scan_partitioning_is_bitwise_invariant(exclude_self, monkeypatch):
    rng = np.random.default_rng(918)
    train = rng.normal(size=(50003, 32)).astype(np.float32)
    # Mixed scales exercise compensated accumulation; duplicate rows exercise
    # distance/index ties across otherwise independent reference partitions.
    train[:, ::4] *= 1000
    train[::113] = train[0]
    queries = train[701:710] if exclude_self else train[::113][:9]
    X, Q = mx.array(train), mx.array(queries)
    expected = None
    for parts in (1, 3, 16, 32):
        monkeypatch.setattr(core, "_scan_partitions", lambda n, m, parts=parts: parts)
        result = core.fused_neighbors(X, Q, 16, exclude_self=exclude_self,
                                      query_offset=701 if exclude_self else 0)
        actual = tuple(np.asarray(a) for a in result)
        if expected is None:
            expected = actual
        else:
            for value, reference in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(value, reference)


def test_self_exclusion_with_more_duplicates_than_k():
    distances, indices = core.fused_neighbors(
        mx.zeros((1025, 2)), mx.zeros((3, 2)), 5, exclude_self=True, query_offset=2,
    )
    np.testing.assert_array_equal(np.asarray(indices), [[0, 1, 3, 4, 5],
                                                         [0, 1, 2, 4, 5],
                                                         [0, 1, 2, 3, 5]])
    np.testing.assert_array_equal(np.asarray(distances), 0)


def test_large_offset_uses_original_coordinates_without_centering_loss():
    train = np.array([[2**24, 0], [2**24, 1], [-2**24, 0], [2**24, 2]], np.float32)
    queries = np.array([[2**24, 0.25]], np.float32)
    distances, indices = core.fused_neighbors(mx.array(train), mx.array(queries), 3)
    expected_d, expected_i = _oracle(train, queries, 3)
    np.testing.assert_array_equal(np.asarray(indices), expected_i)
    np.testing.assert_array_equal(np.asarray(distances), expected_d)


def test_mixed_feature_scales_preserve_small_distance_contributions():
    train = mx.array([[10000.] + [1.] * 31, [10000., 4.] + [0.] * 30])
    distances, indices = core.fused_neighbors(train, mx.zeros((1, 32)), 2)
    np.testing.assert_array_equal(np.asarray(indices), [[1, 0]])
    np.testing.assert_array_equal(np.asarray(distances), [[100000016., 100000032.]])


def test_strided_inputs_and_empty_query():
    train = mx.arange(200, dtype=mx.float32).reshape(10, 20)[:, ::3]
    queries = train[::2]
    distances, indices = core.fused_neighbors(train, queries, 3)
    expected_d, expected_i = _oracle(np.asarray(train), np.asarray(queries), 3)
    np.testing.assert_array_equal(np.asarray(indices), expected_i)
    np.testing.assert_array_equal(np.asarray(distances), expected_d)
    distances, indices = core.fused_neighbors(train, train[:0], 3)
    assert distances.shape == indices.shape == (0, 3)
    assert distances.dtype == mx.float32 and indices.dtype == mx.int32


def test_finite_inputs_with_overflowed_squared_distance_keep_real_indices():
    train = mx.array([[1e30], [-1e30], [2e30]], mx.float32)
    distances, indices = core.fused_neighbors(train, mx.zeros((1, 1)), 3)
    np.testing.assert_array_equal(np.asarray(indices), [[0, 1, 2]])
    assert np.isinf(np.asarray(distances)).all()


def test_active_cpu_does_not_launch_metal():
    with mx.stream(mx.cpu), pytest.raises(RuntimeError, match="active MLX GPU"):
        core.fused_neighbors(mx.zeros((10, 2)), mx.zeros((1, 2)), 3)


@pytest.mark.parametrize("k,error", [(0, ValueError), (33, ValueError), (True, TypeError),
                                     (1.5, TypeError)])
def test_invalid_k(k, error):
    with pytest.raises(error):
        core.fused_neighbors(mx.zeros((64, 2)), mx.zeros((1, 2)), k)


def test_invalid_shapes_dtype_and_offset():
    with pytest.raises(TypeError, match="float32"):
        core.fused_neighbors(mx.zeros((10, 2), mx.int32), mx.zeros((1, 2)), 3)
    with pytest.raises(ValueError, match="feature count"):
        core.fused_neighbors(mx.zeros((10, 2)), mx.zeros((1, 3)), 3)
    with pytest.raises(ValueError, match="query_offset"):
        core.fused_neighbors(mx.zeros((10, 2)), mx.zeros((2, 2)), 3,
                             exclude_self=True, query_offset=9)


@pytest.mark.parametrize("rows,width,k", [
    (1, 1, 1), (7, 33, 32), (3, 257, 64), (1, 4097, 128),
    (256, 8193, 9), (9, 2051, 68),
])
def test_tile_selection_matches_stable_sort(rows, width, k):
    rng = np.random.default_rng(rows + width)
    values = rng.integers(0, 150, size=(rows, width)).astype(np.float32)
    values[:, ::17] = np.inf
    actual_values, actual_indices = core.select_topk(mx.array(values), k, offset=87)
    expected_indices = np.argsort(values, axis=1, kind="stable")[:, :k]
    np.testing.assert_array_equal(np.asarray(actual_indices), expected_indices + 87)
    np.testing.assert_array_equal(np.asarray(actual_values),
                                  np.take_along_axis(values, expected_indices, axis=1))


@pytest.mark.parametrize("k", [33, 64, 128])
def test_tile_all_winners_in_one_lane(k, monkeypatch):
    values = np.full((1, 8193), 10000, dtype=np.float32)
    winners = np.arange(k) * 32
    values[0, winners] = np.arange(k, dtype=np.float32)
    for parts in (1, 2, 32):
        monkeypatch.setattr(core, "_partitions", lambda n, m, parts=parts: parts)
        actual_values, actual_indices = core.select_topk(mx.array(values), k)
        np.testing.assert_array_equal(np.asarray(actual_indices), winners[None, :])
        np.testing.assert_array_equal(np.asarray(actual_values), np.arange(k)[None, :])


def test_tile_nonfinite_values_retain_real_tied_indices():
    values = mx.array([[float("nan"), float("inf"), 5., 2., float("nan")]])
    actual_values, actual_indices = core.select_topk(values, 5)
    np.testing.assert_array_equal(np.asarray(actual_indices), [[3, 2, 0, 1, 4]])
    np.testing.assert_array_equal(np.asarray(actual_values), [[2., 5., np.inf, np.inf, np.inf]])


def test_tile_empty_strided_and_cpu_guard():
    values = mx.arange(200, dtype=mx.float32).reshape(10, 20)[:, ::3]
    actual_values, actual_indices = core.select_topk(values, 3, offset=11)
    np.testing.assert_array_equal(np.asarray(actual_indices), np.tile(np.arange(11, 14), (10, 1)))
    np.testing.assert_array_equal(np.asarray(actual_values), np.asarray(values[:, :3]))
    assert core.select_topk(values[:0], 3)[0].shape == (0, 3)
    with mx.stream(mx.cpu), pytest.raises(RuntimeError, match="active MLX GPU"):
        core.select_topk(values, 3)


@pytest.mark.parametrize("features,k", [(1, 1), (8, 5), (32, 32), (127, 24), (1024, 5)])
def test_refinement_matches_fused_pair_arithmetic_bit_for_bit(features, k):
    mx.random.seed(features + k)
    train = mx.random.normal((257, features))
    queries = mx.random.normal((37, features))
    distances, indices = core.fused_neighbors(train, queries, k)
    refined = core.refine_neighbors(train, queries, indices)
    np.testing.assert_array_equal(np.asarray(refined), np.asarray(distances))


def test_refinement_mixed_scale_invalid_index_and_empty():
    train = mx.array([[10000.] + [1.] * 31, [10000., 4.] + [0.] * 30])
    queries = mx.zeros((1, 32))
    indices = mx.array([[0, 1, -1, 2147483647]], mx.int32)
    refined = core.refine_neighbors(train, queries, indices)
    np.testing.assert_array_equal(np.asarray(refined), [[100000032., 100000016., np.inf, np.inf]])
    empty = core.refine_neighbors(train, queries[:0], indices[:0])
    assert empty.shape == (0, 4)
    with mx.stream(mx.cpu), pytest.raises(RuntimeError, match="active MLX GPU"):
        core.refine_neighbors(train, queries, indices)
