"""Uncertain rows of the fused small-feature screen: exact fallback searches.

The fused direct scan proves most rows from FP32 intervals: a "clear" row
keeps its fast order and a "membership" row is re-sorted exactly. Any other
row needs a complete exact search of the original coordinates. For k > 8 that
search used to run on one GPU thread per row; integer data, whose exact ties
leave most rows uncertain, took 316 ms at 262144 x 32 and k=9 against 1.3 ms
at k=8 on M4 Max. These tests force uncertain rows by zeroing their omission
barrier, or use integer data, whose ties make rows uncertain on every GPU.
"""

import time

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import _neighbors_fused as fused
from mlxlearn_device import _neighbors_screen as screen
from mlxlearn_device import fit_neighbors

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]


def _oracle(train, queries, k, excluded=None):
    """Indices in exact float64 (distance, index) order."""
    train64 = train.astype(np.float64)
    ids = np.arange(len(train))
    result = []
    for row, query in enumerate(queries.astype(np.float64)):
        delta = train64 - query
        squared = np.sum(delta * delta, axis=1)
        if excluded is not None:
            squared[excluded[row]] = np.inf
        result.append(np.lexsort((ids, squared))[:k])
    return np.asarray(result)


def _boundary_ties(train, queries, k):
    """Fraction of rows whose k-th and (k+1)-th exact distances are equal."""
    tied = 0
    for query in queries.astype(np.float64):
        squared = np.sort(np.sum((train.astype(np.float64) - query) ** 2, axis=1))
        tied += squared[k - 1] == squared[k]
    return tied / len(queries)


def _route(monkeypatch, route, forced=None):
    """Pick the finish route; zero the barrier of ``forced`` query rows.

    A zero omission barrier proves neither order nor membership, so those
    rows must be searched again (unless every reference is a neighbor).
    """
    monkeypatch.setattr(screen, "_use_partitioned_finish",
                        lambda n, features, k: route == "partitioned")
    if forced is None:
        return
    original = screen._finish

    def finish(train, queries, k, values, indices, barriers, **options):
        mask = np.zeros(queries.shape[0], bool)
        mask[forced] = True
        barriers = mx.where(mx.array(mask), mx.array(0.0, mx.float32), barriers)
        return original(train, queries, k, values, indices, barriers, **options)

    monkeypatch.setattr(screen, "_finish", finish)


def _check_distances(train, queries, distances, found):
    # Every row reports the compensated FP32 sums shared by all direct scans.
    direct = fused.refine_neighbors(mx.array(train), mx.array(queries), mx.array(found))
    np.testing.assert_array_equal(distances.view(np.uint32), np.array(direct).view(np.uint32))


# The single-dispatch finish exists for k <= 8 only (see _finish_kernel).
_ROUTES = [(k, route) for k in (1, 8, 9, 16, 32)
           for route in (["single", "partitioned"] if k <= 8 else ["partitioned"])]


@pytest.mark.parametrize(("n", "rows"), [(3000, 40), (9000, 64)])  # the second pairs queries
@pytest.mark.parametrize("features", [3, 32])
@pytest.mark.parametrize(("k", "route"), _ROUTES)
def test_forced_uncertain_rows_match_float64_order(k, route, features, n, rows, monkeypatch):
    rng = np.random.default_rng(97 * k + features + n)
    train = rng.standard_normal((n, features)).astype(np.float32)
    queries = rng.standard_normal((rows, features)).astype(np.float32)
    _route(monkeypatch, route, forced=[0, 1, 7, rows - 1])
    distances, found = fused.fused_neighbors(mx.array(train), mx.array(queries), k)
    distances, found = np.array(distances), np.array(found)
    np.testing.assert_array_equal(found, _oracle(train, queries, k))
    _check_distances(train, queries, distances, found)


def test_forced_rows_are_uncertain_and_others_certified(monkeypatch):
    rng = np.random.default_rng(3)
    train = rng.standard_normal((5000, 7)).astype(np.float32)
    queries = rng.standard_normal((50, 7)).astype(np.float32)
    forced = [2, 3, 40]
    _route(monkeypatch, "partitioned", forced=forced)
    seen = []
    original = screen._certify_kernel

    def spy():
        kernel = original()

        def run(**options):
            outputs = kernel(**options)
            seen.append(np.array(outputs[2]))
            return outputs
        return run

    monkeypatch.setattr(screen, "_certify_kernel", spy)
    distances, found = fused.fused_neighbors(mx.array(train), mx.array(queries), 9)
    np.testing.assert_array_equal(np.array(found), _oracle(train, queries, 9))
    assert not seen[0][forced].any()
    assert seen[0].sum() >= len(queries) - len(forced) - 2  # Gaussian rows certify


@pytest.mark.parametrize(("k", "route"), [(5, "single"), (5, "partitioned"), (12, "partitioned")])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_every_reference_a_neighbor_is_certified_exactly(k, route, exclude_self, monkeypatch):
    # With n - exclude_self == k the omission barrier is irrelevant.
    rng = np.random.default_rng(5)
    n = k + int(exclude_self)
    train = rng.integers(-1, 2, (n, 4)).astype(np.float32)
    queries = train if exclude_self else rng.integers(-1, 2, (6, 4)).astype(np.float32)
    _route(monkeypatch, route, forced=list(range(len(queries))))
    distances, found = fused.fused_neighbors(mx.array(train), mx.array(queries), k,
                                             exclude_self=exclude_self)
    excluded = np.arange(n) if exclude_self else None
    np.testing.assert_array_equal(np.array(found), _oracle(train, queries, k, excluded))


@pytest.mark.parametrize(("n", "features"), [(1000, 8), (40000, 16)])
@pytest.mark.parametrize("k", [9, 12, 16])
def test_integer_ties_through_public_query(n, features, k):
    rng = np.random.default_rng(n + k)
    train = rng.integers(-2, 3, (n, features)).astype(np.float32)
    queries = rng.integers(-2, 3, (96, features)).astype(np.float32)
    assert _boundary_ties(train, queries, k) > 0.25  # many rows need the fallback
    distances, found = fit_neighbors(mx.array(train)).query(mx.array(queries), k)
    expected = _oracle(train, queries, k)
    np.testing.assert_array_equal(np.array(found), expected)
    reference = np.sqrt(np.sum((train[expected].astype(np.float64) - queries[:, None, :]) ** 2, axis=2))
    np.testing.assert_allclose(np.array(distances), reference, rtol=1e-6)


@pytest.mark.parametrize("k", [5, 9, 16])
def test_integer_ties_exclude_each_query_identity(k):
    rng = np.random.default_rng(31 + k)
    train = rng.integers(-2, 3, (2500, 16)).astype(np.float32)
    train[100:140] = train[60:100]  # identities with equal-distance duplicates
    _, found = fit_neighbors(mx.array(train)).query(mx.array(train), k, exclude_self=True)
    found = np.array(found)
    np.testing.assert_array_equal(found, _oracle(train, train, k, excluded=np.arange(len(train))))
    assert not np.any(found == np.arange(len(train))[:, None])


def test_routing_partitions_large_k_and_large_small_k_searches():
    assert screen._use_partitioned_finish(20, 2, 9)
    assert screen._use_partitioned_finish(262144, 32, 16)
    assert screen._use_partitioned_finish(1024, 32, 8)
    assert screen._use_partitioned_finish(16384, 2, 5)
    assert not screen._use_partitioned_finish(1023, 32, 8)
    assert not screen._use_partitioned_finish(4096, 4, 5)


@pytest.mark.benchmark
def test_tied_rows_cost_little_more_than_certified_rows(monkeypatch):
    """Uncertain rows must not dominate a fused query (the old serial cliff).

    Integer data leave most of these 256 rows uncertain at k=9. The old
    one-thread-per-row search took about 45 ms here on M4 Max, around 60
    times the query with every row forced certified. The bound is relative,
    so slower GPUs and background load affect both sides alike.
    """
    rng = np.random.default_rng(11)
    train = rng.integers(-2, 3, (65536, 16)).astype(np.float32)
    queries = rng.integers(-2, 3, (256, 16)).astype(np.float32)
    assert _boundary_ties(train, queries[:32], 9) > 0.25
    index, block = fit_neighbors(mx.array(train)), mx.array(queries)
    original = screen._merge_kernel

    def certify_all():
        # A huge omission barrier proves membership for every row. The answer
        # may be wrong, but no row needs the fallback: the query's floor.
        kernel = original()

        def run(**options):
            values, indices, barriers = kernel(**options)
            return values, indices, mx.maximum(barriers, 1e30)
        return run

    def best():
        mx.eval(index.query(block, 9))
        times = []
        for _ in range(5):
            start = time.perf_counter()
            mx.eval(index.query(block, 9))
            times.append(time.perf_counter() - start)
        return min(times)

    uncertain = best()
    monkeypatch.setattr(screen, "_merge_kernel", certify_all)
    certified = best()
    assert uncertain < 4 * certified + 0.02, (uncertain, certified)
