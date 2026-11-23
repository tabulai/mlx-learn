"""Exact-match, unlimited-radius and deferred preparation at the public entry points."""

import math

import numpy as np
import pytest

from mlxlearn._kernels.distance import build_index
from mlxlearn._kernels.radius_result import concatenate
from mlxlearn.neighbors import NearestNeighbors


@pytest.mark.parametrize("radius", [0.0, 2.0**-150])
@pytest.mark.parametrize("self_query", [True, False])
def test_row_lookup_signed_zero_subnormals_and_block_offsets(radius, self_query):
    tiny = np.float32(2.0**-149)
    X = np.zeros((1024, 2), dtype=np.float32)
    X[1::3, 0] = -0.0
    X[:100, 1] = tiny
    X[600:750, 1] = -tiny
    Q = X if self_query else X[::-1].copy()
    if not self_query:
        Q[::4, 0] = 2
    index = build_index(X, backend="cpu", _auto=True)
    parts = index.query_radius_csr(Q, radius, exclude_self=self_query, sort_results=True,
                                   query_block_size=512)
    assert all(p.route == "row-lookup" for p in parts)
    result = concatenate(parts)
    np.testing.assert_array_equal(result.distances, np.zeros(result.nnz))
    for row in range(len(Q)):
        matches = np.all(X == Q[row], axis=1)
        if self_query:
            matches[row] = False
        lo, hi = result.indptr[row:row + 2]
        np.testing.assert_array_equal(result.indices[lo:hi], np.flatnonzero(matches))

    # The minimum positive separation itself is inclusive and uses the usual engine.
    part, = index.query_radius_csr(Q, float(tiny), need_distance=False)
    assert part.route != "row-lookup"
    assert part.nnz > result.nnz


def test_explicit_brute_zero_radius_avoids_lookup_index():
    X = np.arange(512, dtype=np.float32).reshape(-1, 1)
    model = NearestNeighbors(algorithm="brute", radius=0.0).fit(X)
    rows = model.radius_neighbors(X, return_distance=False)
    assert model._index_._radius_search.last_plan.route == "brute"
    assert model._index_._radius_search._lookup is None
    np.testing.assert_array_equal(np.concatenate(rows), np.arange(len(X)))


def test_zero_radius_prediction_and_refit():
    from mlxlearn.neighbors import RadiusNeighborsRegressor

    X = np.repeat(np.arange(512, dtype=np.float32), 2).reshape(-1, 1)
    y = np.arange(len(X), dtype=np.float32)
    model = RadiusNeighborsRegressor(radius=0.0).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), np.repeat(y[::2] + 0.5, 2))
    model.fit(X[::-1], y)
    np.testing.assert_array_equal(model.predict(X), np.repeat(y[::2][::-1] + 0.5, 2))


@pytest.mark.parametrize("radius", [math.inf, 1e200])
@pytest.mark.parametrize("self_query", [False, True])
def test_unlimited_indices_for_query_blocks(radius, self_query):
    X = np.arange(75, dtype=np.float32).reshape(25, 3)
    index = build_index(X, backend="cpu", _auto=True)
    Q = X if self_query else X[::3]
    result = concatenate(index.query_radius_csr(
        Q, radius, exclude_self=self_query, need_distance=False, query_block_size=7))
    assert result.distances is None
    rows = np.arange(len(X))
    for i in range(len(Q)):
        lo, hi = result.indptr[i:i + 2]
        np.testing.assert_array_equal(result.indices[lo:hi], rows[rows != i] if self_query else rows)


def test_low_dimensional_radius_fit_defers_gpu_and_train_statistics(monkeypatch):
    from mlxlearn._common import device

    def unexpected():
        pytest.fail("The low-dimensional radius tree should not probe the GPU")

    monkeypatch.setattr(device, "gpu_available", unexpected)
    X = np.random.default_rng(61).normal(size=(32768, 4)).astype(np.float32)
    model = NearestNeighbors(radius=0.0).fit(X)
    indices = model.radius_neighbors(X[:256], return_distance=False)
    for i, row in enumerate(indices):
        np.testing.assert_array_equal(row, [i])
    assert model._index_._radius_search.last_plan.route == "tree"
    assert model._index_._device_index is None
    assert not ({"center", "sample", "integral", "max_abs"} & vars(model._index_._radius_search.stats).keys())


def test_wide_common_translation_can_use_metal():
    from mlxlearn._common.device import gpu_available

    if not gpu_available():
        pytest.skip("requires Metal")
    X = np.full((800, 16), np.float32(2.0**100))
    X[:, 1] = np.arange(len(X), dtype=np.float32)
    Q = X[200:220].copy()
    index = build_index(X, backend="mlx")
    part, = index.query_radius_csr(Q, 1.0, sort_results=True)
    assert part.route == "metal"
    for i in range(len(Q)):
        lo, hi = part.indptr[i:i + 2]
        np.testing.assert_array_equal(part.indices[lo:hi], [200 + i, 199 + i, 201 + i])
        np.testing.assert_array_equal(part.distances[lo:hi], [0.0, 1.0, 1.0])
