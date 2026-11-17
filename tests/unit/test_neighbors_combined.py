"""Integrated binary32 screening, filtered selection and exact fallback."""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._kernels.distance import build_index
from mlxlearn_device import _binary32 as binary32
from mlxlearn_device import _neighbors_fallback as fallback
from mlxlearn_device import _neighbors_fused as fused
from mlxlearn_device import _neighbors_general as general
from mlxlearn_device import fit_neighbors

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]


def _exact(train, queries, k, *, exclude_self=False):
    """Direct float64 oracle; integer fixtures make all squared sums exact."""
    train64 = train.astype(np.float64)
    ids = np.arange(len(train))
    found, distances = [], []
    for row, query in enumerate(queries.astype(np.float64)):
        squared = np.sum((train64 - query) ** 2, axis=1)
        if exclude_self:
            squared[row] = np.inf
        order = np.lexsort((ids, squared))[:k]
        found.append(order)
        distances.append(np.sqrt(squared[order]))
    return np.asarray(distances), np.asarray(found)


def _combined_routes(monkeypatch):
    """Exercise the M5 product and new selectors on any available Apple GPU."""
    events = []
    originals = {
        "matmul": binary32.float32_matmul,
        "filtered": fused._filtered_value_kernel,
        "gram": general.gram_lower,
        "finish": fused.certified_neighbors,
        "fallback": fallback.partitioned_neighbors,
        "eval": mx.eval,
    }

    def matmul(*args, **kwargs):
        events.append("binary32")
        return originals["matmul"](*args, **kwargs)

    def filtered():
        # Reaching the factory proves the threshold was usable, rather than
        # merely requesting filtered=True on a tile with too few chunks.
        events.append("filtered")
        return originals["filtered"]()

    def gram(*args, **kwargs):
        events.append("tile")
        return originals["gram"](*args, **kwargs)

    def finish(train, queries, k, values, indices, certificate, **kwargs):
        events.append("finish")
        # Withdraw certificates after all tile merging/refinement so this
        # covers the real combined pipeline independent of GPU rounding.
        certificate = mx.zeros_like(certificate)
        return originals["finish"](
            train, queries, k, values, indices, certificate, **kwargs,
        )

    def partitioned(*args, **kwargs):
        events.append("partitioned")
        return originals["fallback"](*args, **kwargs)

    def evaluate(*args, **kwargs):
        events.append("eval")
        return originals["eval"](*args, **kwargs)

    monkeypatch.setattr(binary32, "matmul_may_round_float32", lambda: True)
    monkeypatch.setattr(binary32, "float32_matmul", matmul)
    monkeypatch.setattr(general, "_gpu_family", lambda: 17)
    monkeypatch.setattr(general, "_filtered_select_wins", lambda *shape: True)
    monkeypatch.setattr(general, "gram_lower", gram)
    monkeypatch.setattr(fused, "_filtered_value_kernel", filtered)
    monkeypatch.setattr(fused, "_use_partitioned_fallback", lambda *shape: True)
    monkeypatch.setattr(fused, "certified_neighbors", finish)
    monkeypatch.setattr(fallback, "partitioned_neighbors", partitioned)
    monkeypatch.setattr(mx, "eval", evaluate)
    return events


def _assert_routes_and_bounded_tiles(events):
    assert "binary32" in events
    assert "filtered" in events
    assert "partitioned" in events
    # Every query block evaluates its accumulated candidates before starting
    # another reference tile. This prevents the entire tiled search graph
    # from retaining all reference-tile working sets at once.
    pending_tile = False
    for event in events:
        if event == "tile":
            assert not pending_tile, "another reference tile started before evaluation"
            pending_tile = True
        elif event in {"eval", "finish"}:
            pending_tile = False


@pytest.mark.parametrize("api", ["numpy", "device"])
def test_filtered_binary32_multitile_query_recomputes_exact_neighbors(monkeypatch, api):
    rng = np.random.default_rng(20261117)
    train = rng.integers(-3, 4, size=(6147, 40)).astype(np.float32)
    queries = rng.integers(-3, 4, size=(35, 40)).astype(np.float32)
    # Equal nearest points cross a tile boundary and compete by global index.
    train[3065:3085] = train[50]
    queries[:3] = train[50]
    k = 12
    expected_d, expected_i = _exact(train, queries, k)
    if api == "numpy":
        index, query = build_index(train, backend="mlx"), queries
    else:
        index, query = fit_neighbors(mx.array(train)), mx.array(queries)
    events = _combined_routes(monkeypatch)
    # Two full tiles and a single-row final tile, plus a partial query block.
    actual_d, actual_i = index.query(
        query, k, query_block_size=17, train_block_size=3073,
    )
    np.testing.assert_array_equal(np.asarray(actual_i), expected_i)
    np.testing.assert_allclose(np.asarray(actual_d), expected_d, rtol=1e-6)
    _assert_routes_and_bounded_tiles(events)


def test_combined_self_query_keeps_duplicate_ties_and_singleton_tail(monkeypatch):
    rng = np.random.default_rng(20261118)
    train = rng.integers(-2, 3, size=(2049, 36)).astype(np.float32)
    train[1020:1030] = train[17]
    train[-1] = train[17]
    k = 5
    expected_d, expected_i = _exact(train, train, k, exclude_self=True)
    index, query = fit_neighbors(mx.array(train)), mx.array(train)
    events = _combined_routes(monkeypatch)
    # 1024-column tiles have enough disjoint chunks for the k+3 threshold.
    # For the final query row, the last reference tile contains only itself.
    actual_d, actual_i = index.query(
        query, k, query_block_size=257, train_block_size=1024, exclude_self=True,
    )
    found = np.asarray(actual_i)
    np.testing.assert_array_equal(found, expected_i)
    np.testing.assert_allclose(np.asarray(actual_d), expected_d, rtol=1e-6)
    assert not np.any(found == np.arange(len(train))[:, None])
    _assert_routes_and_bounded_tiles(events)
