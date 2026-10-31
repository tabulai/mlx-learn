"""Small and single-query searches retain exact original-coordinate ordering."""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import fit_neighbors

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]


def _reference(train, queries, k, *, exclude_self=False):
    """Direct float64 reference; bounded dyadic fixtures have exact sums."""
    ids = np.arange(len(train))
    values, indices = [], []
    for row, query in enumerate(queries.astype(np.float64)):
        delta = train.astype(np.float64) - query
        squared = np.sum(delta * delta, axis=1)
        if exclude_self:
            squared[row] = np.inf
        order = np.lexsort((ids, squared))[:k]
        values.append(np.sqrt(squared[order]))
        indices.append(order)
    return np.asarray(values), np.asarray(indices)


def _exact_indices(train, queries, k):
    """Compare binary32 coordinates as Python integers in units of 2**-149."""
    def row_integers(row):
        result = []
        for raw in row.view(np.uint32):
            raw = int(raw)
            exponent = (raw >> 23) & 255
            mantissa = (raw & 0x7fffff) | (0x800000 if exponent else 0)
            value = mantissa << max(0, exponent - 1)
            result.append(-value if raw >> 31 else value)
        return result

    points = [row_integers(row) for row in train]
    result = []
    for query in queries:
        q = row_integers(query)
        keys = [(sum((a - b) ** 2 for a, b in zip(q, point, strict=True)), index)
                for index, point in enumerate(points)]
        result.append([index for _, index in sorted(keys)[:k]])
    return np.asarray(result)


@pytest.mark.parametrize("n,features,rows", [
    (120, 4, 30), (142, 13, 36), (455, 30, 114), (1541, 64, 17),
    (8193, 8, 1), (8193, 32, 1),
])
def test_small_shapes_match_exhaustive_original_coordinate_reference(n, features, rows):
    rng = np.random.default_rng(19200 + features)
    train = rng.integers(-256, 257, (n, features)).astype(np.float32) / 16
    queries = rng.integers(-256, 257, (rows, features)).astype(np.float32) / 16
    queries[0] = train[-1]
    expected_d, expected_i = _reference(train, queries, 5)
    index = fit_neighbors(mx.array(train))
    distances, indices = index.query(mx.array(queries), 5)
    np.testing.assert_array_equal(indices, expected_i)
    np.testing.assert_allclose(distances, expected_d, rtol=2e-7, atol=1e-6)
    omitted, only_indices = index.query(mx.array(queries), 5, return_distance=False)
    assert omitted is None
    np.testing.assert_array_equal(only_indices, expected_i)


@pytest.mark.parametrize("n,features", [(129, 4), (8193, 8), (8193, 32)])
def test_small_and_single_queries_distinguish_rounded_distances_and_true_ties(n, features):
    train = np.zeros((n, features), np.float32)
    train[:, 0] = np.arange(n, dtype=np.float32) + 32
    # The larger-index points are mathematically closer, even though all three
    # squared distances round to1. Spread winners across lanes and partitions.
    train[0, :2] = [1, 2**-14]
    train[32, 0] = 1
    train[-1, 0] = -1
    query = np.zeros((1, features), np.float32)
    expected = _exact_indices(train, query, 3)
    np.testing.assert_array_equal(expected, [[32, n - 1, 0]])
    index = fit_neighbors(mx.array(train))
    for k in (1, 2, 3):
        distances, indices = index.query(mx.array(query), k)
        np.testing.assert_array_equal(indices, expected[:, :k])
        np.testing.assert_array_equal(distances, np.ones((1, k), np.float32))


@pytest.mark.parametrize("kind", ["overflow", "subnormal"])
def test_small_exact_order_survives_extreme_coordinate_ranges(kind):
    train = np.zeros((65, 8), np.float32)
    query = np.zeros((1, 8), np.float32)
    train[:, 1] = np.arange(65, 0, -1, dtype=np.float32)
    if kind == "overflow":
        train[:, 0] = np.float32(1e30)
    else:
        train[:, 1] = 0
        train[:, 0] = np.arange(65, 0, -1, dtype=np.uint32).view(np.float32)
    expected = _exact_indices(train, query, 5)
    index = fit_neighbors(mx.array(train))
    distances, indices = index.query(mx.array(query), 5)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(expected, [[64, 63, 62, 61, 60]])
    if kind == "overflow":
        assert np.isinf(np.asarray(distances)).all()
    else:
        np.testing.assert_array_equal(distances, 0)


@pytest.mark.parametrize("features", [8, 32])
def test_single_queries_use_owned_training_when_deferred_outputs_are_reversed(features):
    rng = np.random.default_rng(19300 + features)
    train = rng.integers(-64, 65, (8193, features)).astype(np.float32) / 8
    first, second = train[[17]].copy(), train[[8192]].copy()
    first[0, 0] += np.float32(0.0625)
    second[0, 0] -= np.float32(0.0625)
    source = mx.array(train.copy())
    index = fit_neighbors(source)
    source[:] = 10_000
    result_a = index.query(mx.array(first), 5)
    result_b = index.query(mx.array(second), 5)
    mx.eval(*result_b, *result_a)
    for result, query in ((result_a, first), (result_b, second)):
        expected_d, expected_i = _reference(train, query, 5)
        np.testing.assert_array_equal(result[1], expected_i)
        np.testing.assert_allclose(result[0], expected_d, rtol=2e-7, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(index.X).view(np.uint32), train.view(np.uint32))


def test_small_self_query_keeps_duplicate_order_with_explicit_tiles_and_tail():
    train = np.zeros((67, 13), np.float32)
    train[:, 0] = np.arange(67, dtype=np.float32) / 4
    train[[1, 32, 66]] = train[0]
    expected_d, expected_i = _reference(train, train, 5, exclude_self=True)
    index = fit_neighbors(mx.array(train))
    for options in ({}, {"query_block_size": 11, "train_block_size": 33}):
        result = index.query(mx.array(train), 5, exclude_self=True, **options)
        np.testing.assert_array_equal(result[1], expected_i)
        np.testing.assert_array_equal(result[0], expected_d)


@pytest.mark.parametrize("n,features", [(120, 4), (8193, 32)])
def test_equal_shaped_small_fits_keep_independent_compiled_inputs(n, features):
    rng = np.random.default_rng(19400 + features)
    train_a = rng.integers(-64, 65, (n, features)).astype(np.float32) / 8
    train_b = rng.integers(-64, 65, (n, features)).astype(np.float32) / 8 + 128
    query_a, query_b = train_a[[17]].copy(), train_b[[-1]].copy()
    query_a[0, 0] += np.float32(0.0625)
    query_b[0, 0] -= np.float32(0.0625)
    source_a, source_b = mx.array(train_a.copy()), mx.array(train_b.copy())
    index_a, index_b = fit_neighbors(source_a), fit_neighbors(source_b)
    source_a[:] = -10_000
    source_b[:] = 10_000
    result_a = index_a.query(mx.array(query_a), 5)
    result_b = index_b.query(mx.array(query_b), 5)
    # Both graphs have the same shapes and static options. A compiled graph
    # must still read each index's dynamic training argument, not cached data.
    mx.eval(*result_b, *result_a)
    for index, result, train, query in (
        (index_a, result_a, train_a, query_a),
        (index_b, result_b, train_b, query_b),
    ):
        expected_d, expected_i = _reference(train, query, 5)
        np.testing.assert_array_equal(expected_i, _exact_indices(train, query, 5))
        np.testing.assert_array_equal(result[1], expected_i)
        np.testing.assert_allclose(result[0], expected_d, rtol=2e-7, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(index.X).view(np.uint32), train.view(np.uint32))
