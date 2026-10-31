"""Integration checks for owned compact preparation across public query calls."""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.config import tuning_context
from mlxlearn.neighbors import NearestNeighbors
from mlxlearn_device import fit_neighbors
from mlxlearn_device._neighbors_fused import fused_neighbors


@pytest.mark.parametrize("features,k", [(16, 1), (16, 8), (32, 5)])
def test_compact_preparation_is_owned_and_reused_without_caching_queries(features, k):
    rng = np.random.default_rng(233)
    original = rng.normal(size=(16384, features)).astype(np.float32)
    source = mx.array(original)
    index = fit_neighbors(source)
    source[:] = 777
    for shift in (0.0, 0.5):
        queries = mx.array(rng.normal(size=(129, features)).astype(np.float32) + shift)
        expected = fused_neighbors(mx.array(original), queries, k)
        actual = index.query(queries, k, query_block_size=64, _squared=True)
        mx.eval(*actual, *expected)
        for result, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(result, reference)
        if shift == 0:
            prepared = index._compact
            assert prepared is not None
        else:
            assert index._compact is prepared
        omitted, indices = index.query(queries, k, return_distance=False, query_block_size=64)
        assert omitted is None
        np.testing.assert_array_equal(indices, expected[1])


def test_compact_automatic_query_blocks_and_small_tail_preserve_order():
    rng = np.random.default_rng(234)
    train = mx.array(rng.normal(size=(16384, 16)).astype(np.float32))
    queries = mx.array(rng.normal(size=(1057, 16)).astype(np.float32))
    index = fit_neighbors(train)
    expected = fused_neighbors(train, queries, 5)
    actual = index.query(queries, 5, _squared=True)
    mx.eval(*actual, *expected)
    assert index._compact is not None
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)


def test_compact_training_preparation_is_replaced_by_refit():
    rng = np.random.default_rng(235)
    train = rng.normal(size=(16384, 16)).astype(np.float32)
    queries = rng.normal(size=(64, 16)).astype(np.float32)
    with tuning_context(force_backend="mlx"):
        model = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(train)
        first = model.kneighbors(queries)
        prepared = model._index_.device_index._compact
        train[:] = 0
        unchanged = model.kneighbors(queries)
        for result, reference in zip(unchanged, first, strict=True):
            np.testing.assert_array_equal(result, reference)
        model.fit(train)
        # Lazy preparation and eagerly fused preparation both belong to this fit.
        assert model._index_.device_index._compact is not prepared
        _, indices = model.kneighbors(queries)
        np.testing.assert_array_equal(indices, np.tile(np.arange(5), (64, 1)))


def test_explicit_small_query_blocks_do_not_prepare_compact_storage():
    rng = np.random.default_rng(236)
    train = mx.array(rng.normal(size=(16384, 16)).astype(np.float32))
    queries = mx.array(rng.normal(size=(64, 16)).astype(np.float32))
    index = fit_neighbors(train)
    expected = fused_neighbors(train, queries, 5)
    actual = index.query(queries, 5, query_block_size=32, _squared=True)
    mx.eval(*actual, *expected)
    assert index._compact is None
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
