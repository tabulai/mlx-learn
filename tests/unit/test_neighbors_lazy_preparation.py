"""Deferred Gram preparation preserves owned inputs and exact query results."""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import fit_neighbors
from mlxlearn_device._neighbors import DeviceNeighborIndex

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]


def _reference(train, queries, k, *, exclude_self=False):
    """Exhaustive original-coordinate reference with canonical index ties."""
    train64 = train.astype(np.float64)
    ids = np.arange(len(train))
    distances, indices = [], []
    for row, query in enumerate(queries.astype(np.float64)):
        delta = train64 - query
        squared = np.sum(delta * delta, axis=1)
        if exclude_self:
            squared[row] = np.inf
        order = np.lexsort((ids, squared))[:k]
        distances.append(np.sqrt(squared[order]))
        indices.append(order)
    return np.asarray(distances), np.asarray(indices)


def _check(actual, train, queries, k=5, **options):
    expected_distances, expected_indices = _reference(train, queries, k, **options)
    np.testing.assert_array_equal(actual[1], expected_indices)
    np.testing.assert_allclose(actual[0], expected_distances, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("features", [32, 128, 1024])
def test_fresh_general_queries_share_owned_preparation_when_evaluated_in_reverse(features):
    rng = np.random.default_rng(930 + features)
    train = rng.normal(size=(193, features)).astype(np.float32)
    first = rng.normal(size=(9, features)).astype(np.float32)
    first[:2] = train[:2]
    second = first + np.linspace(-0.5, 0.5, features, dtype=np.float32)
    source = mx.array(train.copy())
    index = fit_neighbors(source)
    # The deferred mean and centered training storage must use the owned fit,
    # even when the original caller buffer changes before the first query.
    source[:] = 100_000.0
    q1, q2 = mx.array(first), mx.array(second)
    # Explicit single-tile general search also covers the otherwise fused32D
    # shape. Construct both graphs before evaluating either returned result.
    options = {"train_block_size": len(train), "_validated": True}
    a = index.query(q1, 5, **options)
    b = index.query(q2, 5, **options)
    mx.eval(*b, *a)
    _check(a, train, first)
    _check(b, train, second)
    np.testing.assert_array_equal(np.asarray(index.X).view(np.uint32), train.view(np.uint32))
    np.testing.assert_array_equal(np.asarray(q1).view(np.uint32), first.view(np.uint32))
    np.testing.assert_array_equal(np.asarray(q2).view(np.uint32), second.view(np.uint32))


@pytest.mark.parametrize("features", [32, 128, 1024])
def test_deferred_general_preparation_retains_tiled_self_exclusion_and_duplicate_order(features):
    rng = np.random.default_rng(941 + features)
    train = rng.normal(size=(257, features)).astype(np.float32)
    train[1] = train[0]
    train[-1] = train[0]
    index = fit_neighbors(mx.array(train))
    actual = index.query(
        mx.array(train), 5, exclude_self=True,
        query_block_size=65, train_block_size=128,
    )
    mx.eval(*actual)
    _check(actual, train, train, exclude_self=True)
    np.testing.assert_array_equal(np.asarray(actual[1])[[0, 1, 256], :2], [[1, 256], [0, 256], [0, 1]])


@pytest.mark.parametrize("features", [32, 128, 1024])
def test_fresh_general_offset_preparation_preserves_rounded_unequal_neighbors(features):
    center = np.full(features, 10_000, np.float32)
    train = np.tile(center + np.float32(30), (65, 1))
    train[:3] = center
    train[0, 0] += 4
    train[0, 1] += np.float32(2**-10)
    train[1, 0] += 4
    train[2, 0] -= 4
    query = center[None, :].copy()
    exact = np.sum((train[:3].astype(np.float64) - center.astype(np.float64)) ** 2, axis=1)
    assert exact[1] == exact[2] < exact[0]
    assert exact.astype(np.float32)[0] == exact.astype(np.float32)[1]
    index = fit_neighbors(mx.array(train))
    actual = index.query(mx.array(query), 3, train_block_size=len(train))
    mx.eval(*actual)
    _check(actual, train, query, 3)
    np.testing.assert_array_equal(actual[1], [[1, 2, 0]])


@pytest.mark.parametrize("kind", ["large_constant", "subnormal"])
def test_deferred_general_preparation_keeps_original_values_outside_fast_ranges(kind):
    train = np.zeros((65, 128), np.float32)
    query = np.zeros((1, 128), np.float32)
    if kind == "large_constant":
        train[:, 0] = query[:, 0] = np.float32(1e30)
        train[:, 1] = np.arange(65, 0, -1, dtype=np.float32)
    else:
        train[:, 0] = np.arange(65, 0, -1, dtype=np.uint32).view(np.float32)
    index = fit_neighbors(mx.array(train))
    actual = index.query(mx.array(query), 5, train_block_size=len(train))
    mx.eval(*actual)
    _check(actual, train, query)
    np.testing.assert_array_equal(actual[1], [[64, 63, 62, 61, 60]])


def test_cpu_general_queries_do_not_use_gpu_gram_preparation(monkeypatch):
    rng = np.random.default_rng(959)
    train = rng.normal(size=(129, 32)).astype(np.float32)
    query = rng.normal(size=(7, 32)).astype(np.float32)

    def forbidden(self):
        raise AssertionError("CPU search attempted GPU Gram preparation")

    monkeypatch.setattr(DeviceNeighborIndex, "_prepare_gram", forbidden)
    index = fit_neighbors(mx.array(train), device="cpu")
    actual = index.query(mx.array(query), 5, train_block_size=64)
    mx.eval(*actual)
    _check(actual, train, query)


def test_equal_shaped_fits_keep_independent_deferred_training_preparation():
    rng = np.random.default_rng(961)
    train_a = rng.normal(size=(193, 128)).astype(np.float32)
    train_b = (rng.normal(size=(193, 128)) + 10_000).astype(np.float32)
    query_a = rng.normal(size=(7, 128)).astype(np.float32)
    query_b = (rng.normal(size=(7, 128)) + 10_000).astype(np.float32)
    index_a, index_b = fit_neighbors(mx.array(train_a)), fit_neighbors(mx.array(train_b))
    a = index_a.query(mx.array(query_a), 5, _validated=True)
    b = index_b.query(mx.array(query_b), 5, _validated=True)
    mx.eval(*b, *a)
    _check(a, train_a, query_a)
    _check(b, train_b, query_b)
