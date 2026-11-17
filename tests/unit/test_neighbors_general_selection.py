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
    (7, 2**20, 17, False), (8, 2**18, 41, True),         # wide tiles need 8+ rows
    (64, 2**20, 42, False),                              # adversarial order loses past 41
    (8, 2**14, 32, True), (8, 2**14, 33, False),         # medium tiles up to 32 candidates
    (8, 2**14 - 1, 17, False),
    (256, 2**14, 41, True), (1024, 4096, 41, True),      # tall tiles
    (1024, 1024, 17, False), (255, 2**16, 41, False),
])
def test_calibrated_partial_select_regions(rows, width, take, expected):
    assert general._calibrated_partial_select(rows, width, take) is expected


@pytest.mark.parametrize("family,expected", [(16, "partial"), (13, "sort"), (17, "sort"), (None, "sort")])
def test_larger_partial_selection_requires_a_calibrated_gpu(family, expected, monkeypatch):
    monkeypatch.setattr(general, "_gpu_family", lambda: family)
    monkeypatch.setattr(general, "_filtered_select_wins", lambda *shape: False)
    assert general._selection(64, 2**20, 19) == expected
    # Up to 16 candidates use partial selection on every device, as before.
    assert general._selection(1, 2**21, 16) == "partial"


@pytest.mark.parametrize("family,expected", [
    (13, "filtered"), (16, "filtered"), (17, "filtered"), (14, "sort"), (None, "sort"),
])
def test_filtered_selection_requires_a_calibrated_gpu(family, expected, monkeypatch):
    monkeypatch.setattr(general, "_gpu_family", lambda: family)
    assert general._selection(64, 2**20, 41) == expected
    assert general._selection(64, 2**20, 12) == ("partial" if expected == "sort" else "filtered")
    # Narrow tiles keep their routes on every device.
    assert general._selection(8, 4096, 12) == "partial"
    assert general._selection(8, 4096, 41) == "sort"


@pytest.mark.parametrize("architecture,family", [
    ("applegpu_g16s", 16), ("applegpu_g13d", 13), ("", None), ("unknown", None),
])
def test_gpu_family_parses_mlx_architecture(architecture, family, monkeypatch):
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": architecture}, raising=False)
    assert general._gpu_family.__wrapped__() == family


@pytest.mark.parametrize("rows,n_query,expected", [
    (67, 2048, 64), (89, 2048, 64), (90, 2048, 64), (91, 2048, 91), (64, 2048, 64),
    (50, 2048, 50), (128, 2048, 128), (67, 67, 67), (89, 40, 89),
])
@pytest.mark.parametrize("family", [13, 16])
def test_gram_block_rows(rows, n_query, expected, family, monkeypatch):
    monkeypatch.setattr(general, "_gpu_family", lambda: family)
    assert general._gram_block_rows(rows, n_query) == expected


@pytest.mark.parametrize("family", [17, None])
def test_gram_block_rows_keep_planner_heights_elsewhere(family, monkeypatch):
    monkeypatch.setattr(general, "_gpu_family", lambda: family)
    assert general._gram_block_rows(67, 2048) == 67
    assert general._gram_block_rows(89, 2048) == 89


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


def _filter_patterns(rng, rows, width):
    base = rng.random((rows, width), dtype=np.float32)
    yield "random", base
    yield "descending", -np.sort(-base, axis=1)
    yield "ties", rng.integers(0, 150, size=(rows, width)).astype(np.float32)
    yield "all_equal", np.full((rows, width), 3.0, np.float32)
    lane = base + 10
    lane[:, ::32] = -np.sort(-base[:, ::32], axis=1)   # every winner in one lane residue
    yield "one_lane", lane
    front = base + 10
    front[:, :2048] = -np.sort(-base[:, :2048], axis=1)  # every winner in the first chunks
    yield "front", front
    special = base.copy()
    special[:, ::7] = np.inf
    special[:, 1::5] = np.nan
    yield "inf_nan", special
    sparse = np.full((rows, width), np.inf, np.float32)
    sparse[:, ::997] = base[:, ::997]                   # chunks without a finite minimum
    yield "mostly_inf", sparse


@pytest.mark.parametrize("width,rows", [(100, 3), (2048, 5), (4097, 9), (40001, 6)])
def test_filtered_selection_matches_stable_sort(width, rows):
    rng = np.random.default_rng(width)
    for name, values in _filter_patterns(rng, rows, width):
        clean = np.where(np.isnan(values), np.inf, values)
        order = np.argsort(clean, axis=1, kind="stable")
        for k in (1, 2, 16, 17, 32, 33, 41, 128):
            if k > width:
                continue
            actual_values, actual_indices = fused.select_topk(mx.array(values), k, offset=7,
                                                              filtered=True)
            expected = order[:, :k]
            np.testing.assert_array_equal(np.asarray(actual_indices), expected + 7, err_msg=name)
            np.testing.assert_array_equal(np.asarray(actual_values),
                                          np.take_along_axis(clean, expected, axis=1), err_msg=name)


def test_selection_limit_bounds_the_kth_value():
    rng = np.random.default_rng(3)
    values = rng.random((4, 5000), dtype=np.float32)
    values[1] = -np.sort(-values[1])
    values[2, 1::2] = np.nan
    for k in (1, 8, 39):
        limit = np.asarray(fused._selection_limit(mx.array(values), k))
        kth = np.sort(np.where(np.isnan(values), np.inf, values), axis=1)[:, k - 1]
        assert np.all(limit >= kth)
    # 5000 columns form 40 chunks of 128: too few to bound 41 candidates.
    assert fused._selection_limit(mx.array(values), 41) is None
    # Wide rows use larger chunks, keeping at most 2**16 minima per row.
    wide = mx.zeros((1, 2**16 * 128 + 1), mx.float32)
    assert fused._selection_limit(wide, 4) is not None


def test_filtered_partitions():
    assert fused._filtered_partitions(64, 2**20) == 32
    assert fused._filtered_partitions(256, 2**20) == 8
    assert fused._filtered_partitions(1024, 4096) == 2
    assert fused._filtered_partitions(1, 1000) == 1


@pytest.mark.parametrize("n_train,n_query,k", [
    (20000, 64, 9),     # medium tile, 18 candidates
    (20000, 64, 23),    # medium tile, 32 candidates
    (16384, 256, 32),   # tall tile, 41 candidates
])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_general_partial_selection_is_exact(n_train, n_query, k, filtered, descending, monkeypatch):
    rng = np.random.default_rng(n_train + k)
    train = rng.normal(size=(n_train, 40)).astype(np.float32)
    queries = rng.normal(size=(n_query, 40)).astype(np.float32)
    if descending:
        # Training rows ordered by decreasing distance from the queries: each
        # Gram row then arrives in descending order, the unfiltered worst case.
        train[:, 0] += np.linspace(60, 0, n_train, dtype=np.float32)
    calls = []
    original = fused.select_topk

    def counting(values, take, **kwargs):
        calls.append((take, kwargs.get("filtered", False)))
        return original(values, take, **kwargs)

    # Exercise the calibrated routes on any GPU: exactness does not depend on them.
    monkeypatch.setattr(general, "_gpu_family", lambda: 16)
    monkeypatch.setattr(general, "_filtered_select_wins", lambda *shape: filtered)
    monkeypatch.setattr(fused, "select_topk", counting)
    # The public memory plan keeps each query block in one reference tile here.
    distances, indices = distance.build_index(train, backend="mlx").query(queries, k)
    expected_d, expected_i = _exact(train, queries, k)
    np.testing.assert_array_equal(indices, expected_i)
    np.testing.assert_allclose(distances, expected_d, rtol=1e-6)
    assert calls and set(calls) == {(k + 9, filtered)}

    # The direct device API returns the same exact neighbors.
    direct_d, direct_i = fit_neighbors(mx.array(train)).query(mx.array(queries), k)
    np.testing.assert_array_equal(np.asarray(direct_i), expected_i)
    np.testing.assert_allclose(np.asarray(direct_d), expected_d, rtol=1e-6)


def test_filtered_selection_excludes_self_exactly(monkeypatch):
    rng = np.random.default_rng(5)
    train = rng.normal(size=(3000, 36)).astype(np.float32)
    train[:, 0] += np.linspace(30, 0, 3000, dtype=np.float32)
    train[1500:1510] = train[1490]          # duplicates compete on index order
    calls = []
    original = fused.select_topk

    def counting(values, take, **kwargs):
        calls.append(kwargs.get("filtered", False))
        return original(values, take, **kwargs)

    monkeypatch.setattr(general, "_gpu_family", lambda: 13)
    monkeypatch.setattr(general, "_filtered_select_wins", lambda *shape: True)
    monkeypatch.setattr(fused, "select_topk", counting)
    index = fit_neighbors(mx.array(train))
    actual_d, actual_i = index.query(mx.array(train), 12, exclude_self=True)
    assert calls and all(calls)
    train64 = train.astype(np.float64)
    squared = np.concatenate([((train64[start:start + 100, None, :] - train64[None]) ** 2).sum(axis=2)
                              for start in range(0, 3000, 100)])
    np.fill_diagonal(squared, np.inf)
    columns = np.broadcast_to(np.arange(3000), squared.shape)
    order = np.lexsort((columns, squared), axis=1)[:, :12]
    np.testing.assert_array_equal(np.asarray(actual_i), order)
    np.testing.assert_allclose(np.asarray(actual_d), np.sqrt(np.take_along_axis(squared, order, axis=1)),
                               rtol=1e-6)


@pytest.mark.parametrize("n_query,k,family,filtered,sorted_tiles", [
    (3, 17, 16, False, True),      # fewer than 8 rows: full sorts on limited tiles
    (16, 40, 16, False, True),     # 49 candidates: full sorts on limited tiles
    (16, 20, 16, False, False),    # calibrated GPU: partial selection over all references
    (16, 20, 13, False, True),     # other GPUs: full sorts on limited tiles
    (16, 40, 13, True, False),     # filtered selection over all references
])
def test_public_query_beyond_mlx_sort_width_is_exact(n_query, k, family, filtered, sorted_tiles,
                                                     monkeypatch):
    # MLX's Metal argsort is wrong on float32 rows of 2**21+ columns, and the
    # public memory plan uses one reference tile this wide for small queries.
    # Pin the budget so low-memory runners exercise the same query-row count:
    # splitting 16 rows into blocks below eight changes partial-selection routing.
    monkeypatch.setattr(distance, "_distance_budget_bytes", lambda backend: 1 << 30)
    monkeypatch.setattr(general, "_gpu_family", lambda: family)
    monkeypatch.setattr(general, "_filtered_select_wins", lambda *shape: filtered)
    n_train = 2**21 + 3
    rng = np.random.default_rng(n_query + k)
    train = rng.normal(size=(n_train, 4)).astype(np.float32)
    queries = rng.normal(size=(n_query, 4)).astype(np.float32)
    widths = []
    original = general.gram_lower

    def recording(query, train_tile, *args, **kwargs):
        widths.append(int(train_tile.shape[0]))
        return original(query, train_tile, *args, **kwargs)

    monkeypatch.setattr(general, "gram_lower", recording)
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
