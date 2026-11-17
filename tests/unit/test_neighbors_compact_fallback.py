"""Uncertain rows of the compact and integer screens: exact fallback searches.

Both screens (16 <= D <= 32, k <= 8) certify most rows from a shortlist of
lower bounds. A row they cannot certify needs a complete exact search. That
search used to run on the row's single SIMD group over every training row.
Integer data, whose exact ties leave many rows uncertain, took 21 ms at
262144 x 16 and k=8 on M4 Max, against 1.1 ms for Gaussian data. These tests
force uncertain rows, or use integer data, whose ties make rows uncertain on
every GPU.
"""

import time

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import _neighbors_compact as compact
from mlxlearn_device import _neighbors_fallback as fallback
from mlxlearn_device import _neighbors_fused as fused
from mlxlearn_device import _neighbors_integer as integer
from mlxlearn_device import fit_neighbors

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]

_SENTINEL = 2147483647


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


def _check_distances(train, queries, distances, found):
    # Every row reports the compensated FP32 sums shared by all direct scans.
    direct = fused.refine_neighbors(mx.array(train), mx.array(queries), mx.array(found))
    np.testing.assert_array_equal(distances.view(np.uint32), np.array(direct).view(np.uint32))


def _route(monkeypatch, route):
    monkeypatch.setattr(compact, "_use_partitioned_finish",
                        lambda n, features: route == "partitioned")


def _mask(rows, forced):
    mask = np.zeros(rows, bool)
    mask[forced] = True
    return mx.array(mask)


def _force_unsafe_lists(monkeypatch, forced):
    """Give ``forced`` rows -inf lower bounds: valid, but they certify nothing.

    Their shortlists become arbitrary references, so the exact search runs
    with a loose bound (or none) on either route.
    """
    original = compact.finish_neighbors

    def finish(partial_distances, *args, **options):
        mask = _mask(partial_distances.shape[0], forced)[:, None, None]
        partial_distances = mx.where(mask, mx.array(-np.inf, mx.float32), partial_distances)
        return original(partial_distances, *args, **options)

    monkeypatch.setattr(compact, "finish_neighbors", finish)
    monkeypatch.setattr(integer, "finish_neighbors", finish)


def _force_uncertified(monkeypatch, forced):
    """Withdraw the certificate of ``forced`` rows after certification.

    Their shortlists stay exactly sorted, so the partitioned search runs with
    the tightest bound: the row's own k-th distance, ties included.
    """
    original = compact._certify_kernel

    def spy():
        kernel = original()

        def run(**options):
            values, indices, certified = kernel(**options)
            keep = mx.logical_not(_mask(certified.shape[0], forced))
            return values, indices, mx.logical_and(certified, keep)
        return run

    monkeypatch.setattr(compact, "_certify_kernel", spy)


def _data(kind, n, rows, features, seed):
    rng = np.random.default_rng(seed)
    if kind == "grid":
        return (rng.integers(-2, 3, (n, features)).astype(np.float32),
                rng.integers(-2, 3, (rows, features)).astype(np.float32))
    return (rng.standard_normal((n, features)).astype(np.float32),
            rng.standard_normal((rows, features)).astype(np.float32))


def _search(path, train, queries, k, **options):
    X, Q = mx.array(train), mx.array(queries)
    if path == "integer":
        if not integer.is_supported():
            pytest.skip("requires integer tensor operations")
        return integer.integer_neighbors(X, Q, k, integer.prepare_integer(X),
                                         return_certificate=True, **options)
    return compact.compact_neighbors(X, Q, k, compact.prepare_compact(X),
                                     return_certificate=True, **options)


_PATHS = [("compact", 16), ("compact", 28), ("integer", 32)]


@pytest.mark.parametrize(("path", "features"), _PATHS)
@pytest.mark.parametrize("kind", ["gauss", "grid"])
@pytest.mark.parametrize("k", [1, 5, 8])
@pytest.mark.parametrize("route", ["single", "partitioned"])
def test_unsafe_lists_match_float64_order(path, features, kind, k, route, monkeypatch):
    rows = 70
    train, queries = _data(kind, 6000, rows, features, 13 * k + features)
    forced = [0, 1, 7, rows - 1]
    _route(monkeypatch, route)
    _force_unsafe_lists(monkeypatch, forced)
    distances, found, certified = (np.array(a) for a in _search(path, train, queries, k))
    np.testing.assert_array_equal(found, _oracle(train, queries, k))
    _check_distances(train, queries, distances, found)
    assert not certified[forced].any()


@pytest.mark.parametrize(("path", "features"), _PATHS)
@pytest.mark.parametrize("k", [1, 2, 5, 8])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_tight_bounds_keep_boundary_ties(path, features, k, exclude_self, monkeypatch):
    # Values in {-1, 0, 1} tie almost every k-th distance with many others.
    # Withdrawing every certificate leaves each row's exact shortlist as its
    # bound, so references tied with the k-th must still be searched.
    rng = np.random.default_rng(71 + k + features)
    train = rng.integers(-1, 2, (20000, features)).astype(np.float32)
    train[500:900] = train[100:500]  # duplicates share distances and order by index
    offset = 300
    queries = (train[offset:offset + 80] if exclude_self
               else rng.integers(-1, 2, (80, features)).astype(np.float32))
    _route(monkeypatch, "partitioned")
    _force_uncertified(monkeypatch, np.arange(len(queries)))
    options = {"exclude_self": True, "query_offset": offset} if exclude_self else {}
    distances, found, certified = (np.array(a) for a in _search(path, train, queries, k, **options))
    excluded = offset + np.arange(len(queries)) if exclude_self else None
    np.testing.assert_array_equal(found, _oracle(train, queries, k, excluded))
    _check_distances(train, queries, distances, found)
    assert not certified.any()


def test_forced_rows_are_uncertain_and_others_certified(monkeypatch):
    train, queries = _data("gauss", 30000, 96, 16, 3)
    forced = [2, 3, 40, 95]
    _force_uncertified(monkeypatch, forced)
    distances, found, certified = (np.array(a) for a in _search("compact", train, queries, 6))
    np.testing.assert_array_equal(found, _oracle(train, queries, 6))
    _check_distances(train, queries, distances, found)
    assert not certified[forced].any()
    assert certified.sum() >= len(queries) - len(forced) - 2  # Gaussian rows certify


@pytest.mark.parametrize("route", ["single", "partitioned"])
@pytest.mark.parametrize("k", [5, 8])
def test_integer_ties_exclude_each_query_identity(route, k, monkeypatch):
    # Query blocks with offsets, as the dispatch layer issues them.
    rng = np.random.default_rng(31 + k)
    train = rng.integers(-2, 3, (2500, 16)).astype(np.float32)
    train[100:140] = train[60:100]  # identities with equal-distance duplicates
    _route(monkeypatch, route)
    X = mx.array(train)
    prepared = compact.prepare_compact(X)
    found, uncertain = [], 0
    for start in range(0, len(train), 640):
        block = mx.array(train[start:start + 640])
        _, indices, certified = compact.compact_neighbors(
            X, block, k, prepared, exclude_self=True, query_offset=start, return_certificate=True,
        )
        found.append(np.array(indices))
        uncertain += int((~np.array(certified)).sum())
    found = np.concatenate(found)
    np.testing.assert_array_equal(found, _oracle(train, train, k, excluded=np.arange(len(train))))
    assert not np.any(found == np.arange(len(train))[:, None])
    assert uncertain > 30  # the ties exercise the exact search (68 at k=5 on M4 Max)


@pytest.mark.parametrize(("features", "k"), [(16, 5), (16, 8), (32, 8)])
def test_integer_ties_through_public_query(features, k):
    rng = np.random.default_rng(features + k)
    train = rng.integers(-2, 3, (40000, features)).astype(np.float32)
    queries = rng.integers(-2, 3, (96, features)).astype(np.float32)
    distances, found = fit_neighbors(mx.array(train)).query(mx.array(queries), k)
    expected = _oracle(train, queries, k)
    np.testing.assert_array_equal(np.array(found), expected)
    reference = np.sqrt(np.sum((train[expected].astype(np.float64) - queries[:, None, :]) ** 2, axis=2))
    np.testing.assert_allclose(np.array(distances), reference, rtol=1e-6)


def test_bound_keeps_lower_indices_tied_with_the_last_candidate():
    """The bound skips only references strictly farther than the k-th."""
    # Distances from the origin: 0 -> 1, 1..6 -> 2, 7 -> 4, rest farther.
    train = np.full((600, 4), 3.0, np.float32)
    train[0] = [1, 0, 0, 0]
    for index in range(1, 7):
        train[index] = [1, 1, 0, 0]
    train[7] = [2, 0, 0, 0]
    train[400] = [1, 1, 0, 0]  # also tied at 2, late in the scan
    queries = np.zeros((3, 4), np.float32)
    k = 3
    # Row 0's shortlist ends with a tied reference of higher index (5), row 1's
    # with a strictly farther one (7), row 2 has no valid bound.
    candidates = np.array([[0, 2, 5], [0, 1, 7], [0, 1, _SENTINEL]], np.int32)
    values = fused.refine_neighbors(mx.array(train), mx.array(queries),
                                    mx.array(np.minimum(candidates, 599)))
    values = mx.where(mx.array(candidates == _SENTINEL), mx.array(np.inf, mx.float32), values)
    distances, found = fallback.partitioned_neighbors(
        mx.array(train), mx.array(queries), k, values, mx.array(candidates),
        mx.zeros((3,), mx.bool_), bounded=True,
    )
    np.testing.assert_array_equal(np.array(found), [[0, 1, 2]] * 3)
    _check_distances(train, queries, np.array(distances), np.array(found))


def test_routing_partitions_large_searches():
    assert compact._use_partitioned_finish(4096, 16)
    assert compact._use_partitioned_finish(16384, 16)  # smallest dispatched compact search
    assert compact._use_partitioned_finish(2**21, 32)
    assert not compact._use_partitioned_finish(4095, 16)
    assert not compact._use_partitioned_finish(1024, 32)


@pytest.mark.benchmark
def test_tied_rows_cost_little_more_than_certified_rows(monkeypatch):
    """Uncertain rows must not dominate a compact query (the old serial cliff).

    Integer data leave about 95 of these 256 rows uncertain at k=8. The old
    one-SIMD-group-per-row search took 21 ms here on M4 Max, 19 times the
    query with every row forced certified (1.1 ms); the partitioned search
    takes about 2.1 ms. The bound is relative, so slower GPUs and background
    load affect both sides alike.
    """
    train, queries = _data("grid", 262144, 256, 16, 11)
    index, block = fit_neighbors(mx.array(train)), mx.array(queries)
    original = compact._scan_kernel

    def certify_all():
        # Uniform huge lower bounds prove every shortlist complete. The answer
        # is wrong, but no row needs the exact search: the query's floor.
        kernel = original()

        def run(**options):
            distances, indices = kernel(**options)
            return mx.full(distances.shape, 1e30, mx.float32), indices
        return run

    def best():
        mx.eval(index.query(block, 8))
        times = []
        for _ in range(5):
            start = time.perf_counter()
            mx.eval(index.query(block, 8))
            times.append(time.perf_counter() - start)
        return min(times)

    uncertain = best()
    monkeypatch.setattr(compact, "_scan_kernel", certify_all)
    certified = best()
    assert uncertain < 4 * certified + 0.004, (uncertain, certified)
