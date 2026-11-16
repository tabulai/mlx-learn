"""General GPU path: partial-selection routing, sorted-tile width limit and block heights."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._kernels import distance
from mlxlearn_device import _neighbors_fused as fused
from mlxlearn_device import _neighbors_general as general
from mlxlearn_device import fit_neighbors

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not mx.metal.is_available(), reason="requires Metal",
)]


def _exact(train, queries, k):
    """Float64 direct distances ordered by (distance, index), one query at a time."""
    train64 = train.astype(np.float64)
    order_all, dist_all = [], []
    for query in queries.astype(np.float64):
        squared = ((train64 - query) ** 2).sum(axis=1)
        # Every point tied with the kth distance competes on index order.
        candidates = np.flatnonzero(squared <= np.partition(squared, k - 1)[k - 1])
        order = candidates[np.lexsort((candidates, squared[candidates]))][:k]
        order_all.append(order)
        dist_all.append(squared[order])
    return np.sqrt(np.array(dist_all)), np.array(order_all)


@pytest.mark.parametrize("rows,width,take,expected", [
    (1, 10, 16, True), (1, 2**21, 16, True),            # small candidate sets: unchanged
    (7, 2**20, 17, False), (8, 2**18, 41, True),         # wide tiles need 8+ rows
    (64, 2**20, 42, False),                              # adversarial order loses past 41
    (8, 2**14, 32, True), (8, 2**14, 33, False),         # medium tiles up to 32 candidates
    (8, 2**14 - 1, 17, False),
    (256, 2**14, 41, True), (1024, 4096, 41, True),      # tall tiles
    (1024, 1024, 17, False), (255, 2**16, 41, False),
])
def test_partial_select_regions(rows, width, take, expected):
    assert general._use_partial_select(rows, width, take) is expected


@pytest.mark.parametrize("rows,n_query,expected", [
    (67, 2048, 64), (89, 2048, 64), (90, 2048, 64), (91, 2048, 91), (64, 2048, 64),
    (50, 2048, 50), (128, 2048, 128), (67, 67, 67), (89, 40, 89),
])
def test_gram_block_rows(rows, n_query, expected):
    assert general._gram_block_rows(rows, n_query) == expected


@pytest.mark.parametrize("k", [17, 20, 24, 31, 32])
@pytest.mark.parametrize("order", ["random", "descending"])
def test_padded_selection_matches_stable_sort(k, order):
    rng = np.random.default_rng(k)
    values = rng.integers(0, 150, size=(37, 9001)).astype(np.float32)
    if order == "descending":
        values = -np.sort(-values, axis=1)
    values[:, ::23] = np.inf
    actual_values, actual_indices = fused.select_topk(mx.array(values), k, offset=5)
    expected = np.argsort(values, axis=1, kind="stable")[:, :k]
    np.testing.assert_array_equal(np.asarray(actual_indices), expected + 5)
    np.testing.assert_array_equal(np.asarray(actual_values),
                                  np.take_along_axis(values, expected, axis=1))


@pytest.mark.parametrize("k", [17, 32])
def test_padded_selection_all_winners_in_one_lane(k, monkeypatch):
    values = np.full((1, 8193), 10000, dtype=np.float32)
    winners = np.arange(k) * 32
    values[0, winners] = np.arange(k, dtype=np.float32)
    for parts in (1, 2, 32):
        monkeypatch.setattr(fused, "_partitions", lambda n, m, parts=parts: parts)
        actual_values, actual_indices = fused.select_topk(mx.array(values), k)
        np.testing.assert_array_equal(np.asarray(actual_indices), winners[None, :])
        np.testing.assert_array_equal(np.asarray(actual_values), np.arange(k)[None, :])


@pytest.mark.parametrize("n_train,n_query,k", [
    (20000, 64, 9),     # medium tile, 18 candidates
    (20000, 64, 23),    # medium tile, 32 candidates
    (16384, 256, 32),   # tall tile, 41 candidates
])
def test_general_partial_selection_is_exact(n_train, n_query, k, monkeypatch):
    rng = np.random.default_rng(n_train + k)
    train = rng.normal(size=(n_train, 40)).astype(np.float32)
    queries = rng.normal(size=(n_query, 40)).astype(np.float32)
    calls = []
    original = fused.select_topk

    def counting(values, take, **kwargs):
        calls.append(take)
        return original(values, take, **kwargs)

    monkeypatch.setattr(fused, "select_topk", counting)
    # The public memory plan keeps each query block in one reference tile here.
    distances, indices = distance.build_index(train, backend="mlx").query(queries, k)
    expected_d, expected_i = _exact(train, queries, k)
    np.testing.assert_array_equal(indices, expected_i)
    np.testing.assert_allclose(distances, expected_d, rtol=1e-6)
    assert calls and set(calls) == {k + 9}

    # The direct device API returns the same exact neighbors.
    direct_d, direct_i = fit_neighbors(mx.array(train)).query(mx.array(queries), k)
    np.testing.assert_array_equal(np.asarray(direct_i), expected_i)
    np.testing.assert_allclose(np.asarray(direct_d), expected_d, rtol=1e-6)


@pytest.mark.parametrize("n_query,k,sorted_tiles", [
    (3, 17, True),      # fewer than 8 rows: full sorts on limited tiles
    (16, 40, True),     # 49 candidates: full sorts on limited tiles
    (16, 20, False),    # partial selection over the whole reference range
])
def test_public_query_beyond_mlx_sort_width_is_exact(n_query, k, sorted_tiles, monkeypatch):
    # MLX's Metal argsort is wrong on float32 rows of 2**21+ columns, and the
    # public memory plan uses one reference tile this wide for small queries.
    n_train = 2**21 + 3
    rng = np.random.default_rng(n_query + k)
    train = rng.normal(size=(n_train, 4)).astype(np.float32)
    queries = rng.normal(size=(n_query, 4)).astype(np.float32)
    widths = []
    original = general._gram_lower

    def recording(query, train_tile, *args):
        widths.append(int(train_tile.shape[0]))
        return original(query, train_tile, *args)

    monkeypatch.setattr(general, "_gram_lower", recording)
    plan = distance.resolve_blocks(n_query, n_train, backend="mlx", k_search=k + distance._OVERSELECT)
    assert plan.train_cols == n_train
    index = distance.build_index(train, backend="mlx")
    actual_d, actual_i = index.query(queries, k)
    expected_d, expected_i = _exact(train, queries, k)
    np.testing.assert_array_equal(actual_i, expected_i)
    np.testing.assert_allclose(actual_d, expected_d, rtol=1e-6)
    if sorted_tiles:
        assert max(widths) == general._SORT_WIDTH_LIMIT
    else:
        assert widths == [n_train]
