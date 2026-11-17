"""Uncertified Gram rows: exact, partitioned fallback search on the GPU.

Rows whose Gram screen cannot certify their candidates are searched again from
the original coordinates. That search used to run on one GPU thread per row
for k > 8; one such row at 131072 x 1024 took about two seconds. These tests
force uncertified rows directly, so they do not depend on how a particular
GPU rounds its Gram products.
"""

import time

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import _neighbors_fallback as fallback
from mlxlearn_device import _neighbors_fused as fused
from mlxlearn_device import fit_neighbors

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]


def _oracle(train, queries, k, excluded=None):
    """Sorted float64 differences, ties broken by training index.

    ``excluded`` gives one training row per query to leave out.
    """
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


def _clustered(n, features, rows, clusters, seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, features)).astype(np.float32) * 4
    train = centers[rng.integers(0, clusters, n)] + rng.standard_normal((n, features), np.float32)
    queries = centers[rng.integers(0, clusters, rows)] + rng.standard_normal((rows, features), np.float32)
    return train.astype(np.float32), queries.astype(np.float32)


def _fallback(train, queries, k, certified, *, exclude_self=False, query_offset=0, seed=0):
    """Run the routed fallback with deliberately wrong candidates."""
    rng = np.random.default_rng(seed)
    rows = len(queries)
    values = mx.array(rng.standard_normal((rows, k)).astype(np.float32))
    indices = mx.array(rng.integers(0, len(train), (rows, k)).astype(np.int32))
    distances, found = fused.certified_neighbors(
        mx.array(train), mx.array(queries), k, values, indices, mx.array(certified),
        exclude_self=exclude_self, query_offset=query_offset,
    )
    mx.eval(distances, found)
    return np.array(values), np.array(indices), np.array(distances), np.array(found)


def _check(train, queries, k, certified):
    values, indices, distances, found = _fallback(train, queries, k, certified)
    # Certified rows are trusted verbatim, even these arbitrary candidates.
    np.testing.assert_array_equal(found[certified], indices[certified])
    np.testing.assert_array_equal(distances[certified].view(np.uint32),
                                  values[certified].view(np.uint32))
    redo = ~certified
    if not redo.any():
        return
    np.testing.assert_array_equal(found[redo], _oracle(train, queries[redo], k))
    # Distances are the compensated FP32 sums used by every direct scan.
    direct = fused.refine_neighbors(mx.array(train), mx.array(queries[redo]), mx.array(found[redo]))
    np.testing.assert_array_equal(distances[redo].view(np.uint32), np.array(direct).view(np.uint32))


@pytest.mark.parametrize("k", [1, 5, 8, 9, 10, 31, 33, 100])
@pytest.mark.parametrize("features", [3, 48, 300])
def test_forced_uncertified_rows_match_float64_order(k, features):
    rng = np.random.default_rng(1000 * k + features)
    n, rows = 5000, 70
    train = rng.standard_normal((n, features)).astype(np.float32)
    queries = rng.standard_normal((rows, features)).astype(np.float32)
    certified = rng.random(rows) < 0.7
    certified[:3] = False
    _check(train, queries, k, certified)


@pytest.mark.parametrize("k", [4, 12, 40])
def test_partitioned_search_resolves_ties_and_duplicates_exactly(k):
    # Small integers give many equal and nearly equal distances; duplicated
    # training rows leave only the index to break a tie.
    rng = np.random.default_rng(k)
    base = rng.integers(-2, 3, (1500, 40)).astype(np.float32)
    train = np.concatenate([base, base[::3], base[:7]])
    queries = np.concatenate([train[rng.integers(0, len(train), 20)],
                              rng.integers(-2, 3, (20, 40)).astype(np.float32)])
    certified = np.zeros(len(queries), bool)
    certified[::4] = True
    values, indices, distances, found = _fallback(train, queries, k, certified)
    redo = ~certified
    np.testing.assert_array_equal(found[redo], _oracle(train, queries[redo], k))
    np.testing.assert_array_equal(found[certified], indices[certified])


@pytest.mark.parametrize("k", [6, 16])
def test_partitioned_search_excludes_each_query_identity(k):
    rng = np.random.default_rng(77)
    train = rng.integers(-3, 4, (3000, 36)).astype(np.float32)
    train[100:140] = train[60:100]  # identities with equal-distance duplicates
    offset, rows = 50, 120
    queries = train[offset:offset + rows]
    certified = rng.random(rows) < 0.5
    values, indices, distances, found = _fallback(
        train, queries, k, certified, exclude_self=True, query_offset=offset,
    )
    redo = np.flatnonzero(~certified)
    expected = _oracle(train, queries[redo], k, excluded=offset + redo)
    np.testing.assert_array_equal(found[redo], expected)
    assert not np.any(found[redo] == (offset + redo)[:, None])


def test_partitioned_search_handles_more_rows_than_scan_groups():
    # Every row uncertified and more rows than the usual group budget: each
    # row still gets its own chunk list, and the grid grows with the block.
    rng = np.random.default_rng(5)
    rows = fallback._TARGET_GROUPS + 37
    train = rng.standard_normal((700, 5)).astype(np.float32)
    queries = rng.standard_normal((rows, 5)).astype(np.float32)
    assert fallback.scan_groups(rows, len(train), 9) == rows
    _check(train, queries, 9, np.zeros(rows, bool))


@pytest.mark.parametrize(("n", "k"), [(10, 9), (40, 33), (129, 32), (4099, 200), (70000, 11)])
def test_chunk_boundaries_and_short_lists_stay_exact(n, k):
    # Chunk counts that do not divide n, k close to n, and lists longer than
    # a chunk all keep every reference exactly once.
    rng = np.random.default_rng(n + k)
    train = rng.standard_normal((n, 34)).astype(np.float32)
    queries = rng.standard_normal((9, 34)).astype(np.float32)
    certified = np.zeros(9, bool)
    certified[4] = True
    _check(train, queries, k, certified)


def test_routing_partitions_large_k_and_large_small_k_scans():
    assert fused._use_partitioned_fallback(100, 4, 9)
    assert fused._use_partitioned_fallback(131072, 1024, 10)
    assert fused._use_partitioned_fallback(16384, 64, 8)
    assert not fused._use_partitioned_fallback(4096, 48, 8)


def test_public_query_recomputes_uncertified_rows_exactly(monkeypatch):
    train, queries = _clustered(8192, 1024, 96, 16, seed=3)
    forced = np.zeros(len(queries), bool)
    forced[[0, 17, 18, 95]] = True
    original = fused.certified_neighbors
    seen = []

    def forget(train_, queries_, k, values, indices, certificate, **options):
        mask = mx.array(~forced[options["query_offset"]:][:queries_.shape[0]])
        seen.append(int(mx.sum(~(certificate & mask)).item()))
        return original(train_, queries_, k, values, indices, certificate & mask, **options)

    monkeypatch.setattr(fused, "certified_neighbors", forget)
    distances, indices = fit_neighbors(mx.array(train)).query(mx.array(queries), 10)
    assert sum(seen) >= forced.sum()
    expected = _oracle(train, queries, 10)
    np.testing.assert_array_equal(np.array(indices), expected)
    ref = np.sqrt(np.sum((train[expected].astype(np.float64) - queries[:, None, :]) ** 2, axis=2))
    np.testing.assert_allclose(np.array(distances), ref, rtol=1e-5)


@pytest.mark.benchmark
def test_uncertified_rows_cost_little_more_than_certified_rows(monkeypatch):
    """A few uncertified rows must not dominate a query (the old cliff).

    At this shape one serial row scan took about 250 ms on M4 Max, around 50
    times the whole certified query; the partitioned search adds well under
    one certified query's time. The bound is relative, so slower GPUs and
    background load affect both sides alike.
    """
    train, queries = _clustered(8192, 1024, 256, 16, seed=11)
    index, block = fit_neighbors(mx.array(train)), mx.array(queries)
    original = fused.certified_neighbors

    def forced(keep):
        # Replace the certificate outright: this clustered data leaves some
        # rows uncertified naturally on M1-M4, which would slow both sides.
        def run(train_, queries_, k, values, indices, certificate, **options):
            mask = np.ones(queries_.shape[0], bool)
            if not keep:
                mask[[5, 200]] = False
            return original(train_, queries_, k, values, indices, mx.array(mask), **options)
        return run

    def best(keep):
        monkeypatch.setattr(fused, "certified_neighbors", forced(keep))
        mx.eval(index.query(block, 10))
        times = []
        for _ in range(5):
            start = time.perf_counter()
            mx.eval(index.query(block, 10))
            times.append(time.perf_counter() - start)
        return min(times)

    certified, uncertified = best(True), best(False)
    assert uncertified < 4 * certified + 0.02, (uncertified, certified)
