"""Query grouping preserves each original direct-distance accumulation."""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device._neighbors_fused import fused_neighbors
from mlxlearn_device._neighbors_paired import paired_neighbors

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]


def unpaired_neighbors(train, queries, k, *, exclude_self=False, query_offset=0):
    # Single-query calls always use the original scan, so this remains an
    # independent oracle even when the public dispatcher enables query grouping.
    rows = [
        fused_neighbors(
            train,
            queries[row : row + 1],
            k,
            exclude_self=exclude_self,
            query_offset=query_offset + row,
        )
        for row in range(queries.shape[0])
    ]
    return tuple(mx.concatenate([row[part] for row in rows]) for part in (0, 1))


@pytest.mark.parametrize(
    "n_train,n_query,features,k",
    [
        (33, 7, 3, 32),
        (1025, 7, 32, 5),
        (515, 7, 16, 24),
        (8193, 7, 32, 5),
        (8193, 33, 32, 5),
        (8192, 2, 16, 16),
        (8192, 1, 8, 1),
        (8193, 6, 1, 5),
    ],
)
def test_query_grouping_matches_original_distances(n_train, n_query, features, k):
    rng = np.random.default_rng(51)
    train = mx.array(rng.normal(size=(n_train, features)).astype(np.float32))
    queries = mx.array(rng.normal(size=(n_query, features)).astype(np.float32))
    expected = unpaired_neighbors(train, queries, k)
    for partitions in (1, 7, 16, 32):
        actual = paired_neighbors(train, queries, k, partitions=partitions)
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, e)
    for a, e in zip(fused_neighbors(train, queries, k), expected, strict=True):
        np.testing.assert_array_equal(a, e)


@pytest.mark.parametrize("k", [1, 5, 16, 32])
def test_all_winners_in_one_lane_and_partition(k):
    train = np.full((4097, 2), 1000, dtype=np.float32)
    winners = np.arange(32) * 32
    train[winners, 0] = np.arange(1, 33, dtype=np.float32)
    train[winners, 1] = 0
    for partitions in (1, 4, 32):
        actual = paired_neighbors(mx.array(train), mx.zeros((3, 2)), k, partitions=partitions)
        np.testing.assert_array_equal(actual[1], np.tile(winners[:k], (3, 1)))
        np.testing.assert_array_equal(actual[0], np.tile(np.arange(1, k + 1) ** 2, (3, 1)))


def test_ties_self_exclusion_and_global_offset():
    train = mx.zeros((8193, 3))
    expected = unpaired_neighbors(train, train[2:5], 5, exclude_self=True, query_offset=2)
    for partitions in (1, 7, 16, 32):
        actual = paired_neighbors(
            train, train[2:5], 5, partitions=partitions, exclude_self=True, query_offset=2
        )
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, e)
    public = fused_neighbors(train, train[2:5], 5, exclude_self=True, query_offset=2)
    np.testing.assert_array_equal(public[1], [[0, 1, 3, 4, 5], [0, 1, 2, 4, 5], [0, 1, 2, 3, 5]])


def test_mixed_scales_and_overflow():
    train = mx.array([[10000.0] + [1.0] * 31, [10000.0, 4.0] + [0.0] * 30])
    query = mx.zeros((3, 32))
    actual = paired_neighbors(train, query, 2, partitions=32)
    np.testing.assert_array_equal(actual[1], [[1, 0]] * 3)
    np.testing.assert_array_equal(actual[0], [[100000016.0, 100000032.0]] * 3)
    train = mx.array([[1e30], [-1e30], [2e30]])
    actual = paired_neighbors(train, mx.zeros((2, 1)), 3, partitions=32)
    np.testing.assert_array_equal(actual[1], [[0, 1, 2], [0, 1, 2]])
    assert np.isinf(np.asarray(actual[0])).all()


@pytest.mark.parametrize("features", [3, 32])
def test_overflow_before_last_feature_matches_guarded_scan(features):
    train = np.ones((8193, features), np.float32)
    train[:, 0] = np.resize(np.array([1e30, -1e30, 2e30], np.float32), len(train))
    train[37] = 0
    X, Q = mx.array(train), mx.zeros((3, features))
    expected = unpaired_neighbors(X, Q, 5)
    for partitions in (1, 7, 32):
        actual = paired_neighbors(X, Q, 5, partitions=partitions)
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, e)


def test_noncontiguous_input_views():
    train = mx.arange(120, dtype=mx.float32).reshape(3, 40).T
    queries = mx.arange(18, dtype=mx.float32).reshape(3, 6).T
    for a, e in zip(
        paired_neighbors(train, queries, 5, partitions=3),
        unpaired_neighbors(train, queries, 5),
        strict=True,
    ):
        np.testing.assert_array_equal(a, e)


def test_packed_loads_materialize_noncontiguous_features():
    train = mx.arange(64 * 32, dtype=mx.float32).reshape(32, 64).T
    queries = train[::5]
    actual = paired_neighbors(train, queries, 5, partitions=7)
    expected = unpaired_neighbors(train, queries, 5)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("features,k", [(8, 5), (16, 5), (32, 5), (32, 8), (32, 16)])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_packed_loads_allow_float_aligned_contiguous_views(features, k, exclude_self):
    rng = np.random.default_rng(features + k)
    # The row-contiguous buffers start one float past their base allocation.
    # packed_float4 requires only float alignment, unlike a float4 pointer.
    train = mx.array(rng.normal(size=8193 * features + 1).astype(np.float32))[1:]
    train = train.reshape(8193, features)
    query = mx.array(rng.normal(size=33 * features + 1).astype(np.float32))[1:]
    query = train[19:52] if exclude_self else query.reshape(33, features)
    offset = 19 if exclude_self else 0
    expected = unpaired_neighbors(train, query, k, exclude_self=exclude_self,
                                   query_offset=offset)
    for parts in (1, 7, 32):
        actual = paired_neighbors(train, query, k, partitions=parts,
                                  exclude_self=exclude_self, query_offset=offset)
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, e)
    actual = fused_neighbors(train, query, k, exclude_self=exclude_self, query_offset=offset)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, e)


def test_five_register_candidates_keep_compensation_and_overflow():
    train = mx.array([[10000.] + [1.] * 31, [10000., 4.] + [0.] * 30,
                      [-1e30] + [1.] * 31, [1e30] + [1.] * 31, [0.] * 32])
    query = mx.zeros((3, 32))
    expected = unpaired_neighbors(train, query, 5)
    for parts in (1, 7, 32):
        actual = paired_neighbors(train, query, 5, partitions=parts)
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, e)


@pytest.mark.parametrize("features,k", [(3, 5), (32, 5), (32, 8)])
def test_empty_paired_query_preserves_shapes_and_dtypes(features, k):
    distances, indices = paired_neighbors(mx.zeros((33, features)), mx.zeros((0, features)),
                                          k, partitions=7)
    assert distances.shape == indices.shape == (0, k)
    assert distances.dtype == mx.float32 and indices.dtype == mx.int32
