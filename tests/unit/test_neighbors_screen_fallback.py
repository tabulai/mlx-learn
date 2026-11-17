"""Uncertain rows of the fused small-feature screen: exact fallback searches.

The fused direct scan proves most rows from FP32 intervals: a "clear" row
keeps its fast order and a "membership" row is re-sorted exactly. Any other
row needs a complete exact search of the original coordinates. For k > 8 that
search used to run on one GPU thread per row; integer data, whose exact ties
leave most rows uncertain, took 316 ms at 262144 x 32 and k=9 against 1.3 ms
at k=8 on M4 Max. These tests force uncertain rows by zeroing their omission
barrier, or use integer data, whose ties make rows uncertain on every GPU.

The partitioned search is bounded by the largest fast distance of each
uncertain row, so references provably beyond the row's k-th distance skip the
exact comparisons that ties against the running k-th would cost. Withdrawing
certificates after certification gives the tightest bounds, ties included.
"""

import math
import time
from fractions import Fraction

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import _neighbors_fallback as fallback
from mlxlearn_device import _neighbors_fused as fused
from mlxlearn_device import _neighbors_screen as screen
from mlxlearn_device import fit_neighbors

_SENTINEL = 2147483647

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


def _exact_oracle(train, queries, k):
    """Exact rational (distance, index) order for data float64 cannot rank.

    Float64 errors here are below 1e-14 relative, so every possible member
    lies within the float64 window kept around the k-th distance.
    """
    train64 = train.astype(np.float64)
    result = []
    for query in queries:
        squared = np.sum((train64 - query.astype(np.float64)) ** 2, axis=1)
        kth = np.partition(squared, k - 1)[k - 1]
        coordinates = [Fraction(float(value)) for value in query]
        exact = {int(row): sum((Fraction(float(value)) - center) ** 2
                               for value, center in zip(train[row], coordinates, strict=True))
                 for row in np.nonzero(squared <= kth * (1 + 1e-9))[0]}
        result.append(sorted(exact, key=lambda row: (exact[row], row))[:k])
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


def _mask(rows, forced):
    mask = np.zeros(rows, bool)
    mask[forced] = True
    return mx.array(mask)


def _force_uncertified(monkeypatch, forced, seen=None):
    """Withdraw the certificate of ``forced`` rows after certification.

    A certified row's outputs are exactly sorted, so the partitioned search
    runs with the tightest bound: the row's own k-th distance, ties included.
    ``seen`` collects each certify call's fast lists and outputs.
    """
    original = screen._certify_kernel

    def spy():
        kernel = original()

        def run(**options):
            values, indices, certified = kernel(**options)
            if seen is not None:
                fast_values, fast_indices = options["inputs"][2:4]
                seen.append(tuple(np.array(a) for a in (fast_values, fast_indices,
                                                        values, indices, certified)))
            if forced is not None:
                certified = mx.logical_and(certified, mx.logical_not(_mask(certified.shape[0], forced)))
            return values, indices, certified
        return run

    monkeypatch.setattr(screen, "_certify_kernel", spy)


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


@pytest.mark.parametrize("features", [3, 16, 32])
@pytest.mark.parametrize("k", [1, 5, 8, 9, 16, 32])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_tight_bounds_keep_boundary_ties(features, k, exclude_self, monkeypatch):
    # Values in {-1, 0, 1} tie almost every k-th distance with many others
    # (at D=3 only 27 points exist). Withdrawing every certificate leaves each
    # row's exactly sorted output as its bound, so references tied with the
    # k-th must still be searched. 80 queries pair for k <= 16.
    rng = np.random.default_rng(71 + k + features)
    train = rng.integers(-1, 2, (20000, features)).astype(np.float32)
    train[500:900] = train[100:500]  # duplicates share distances and order by index
    offset = 300
    queries = (train[offset:offset + 80] if exclude_self
               else rng.integers(-1, 2, (80, features)).astype(np.float32))
    _route(monkeypatch, "partitioned")
    _force_uncertified(monkeypatch, np.arange(len(queries)))
    options = {"exclude_self": True, "query_offset": offset} if exclude_self else {}
    distances, found = fused.fused_neighbors(mx.array(train), mx.array(queries), k, **options)
    distances, found = np.array(distances), np.array(found)
    excluded = offset + np.arange(len(queries)) if exclude_self else None
    np.testing.assert_array_equal(found, _oracle(train, queries, k, excluded))
    _check_distances(train, queries, distances, found)


@pytest.mark.parametrize("kind", ["gauss", "grid"])
@pytest.mark.parametrize(("features", "k"), [(12, 5), (12, 16), (32, 9)])
def test_uncertain_rows_carry_their_largest_fast_distance_last(kind, features, k, monkeypatch):
    # The bound of an uncertain row is its fast list, largest distance last.
    rng = np.random.default_rng(5 * k + features)
    rows = 64
    train = (rng.integers(-2, 3, (9000, features)) if kind == "grid"
             else rng.standard_normal((9000, features))).astype(np.float32)
    queries = (rng.integers(-2, 3, (rows, features)) if kind == "grid"
               else rng.standard_normal((rows, features))).astype(np.float32)
    forced = [0, 1, 7, rows - 1]
    seen = []
    _route(monkeypatch, "partitioned", forced=forced)
    _force_uncertified(monkeypatch, None, seen)
    distances, found = (np.array(a) for a in fused.fused_neighbors(mx.array(train), mx.array(queries), k))
    np.testing.assert_array_equal(found, _oracle(train, queries, k))
    _check_distances(train, queries, distances, found)
    (fast_values, fast_indices, values, indices, certified), = seen
    assert not certified[forced].any()
    for row in np.nonzero(~certified)[0]:
        np.testing.assert_array_equal(indices[row], fast_indices[row, :k])
        np.testing.assert_array_equal(values[row, :k - 1].view(np.uint32),
                                      fast_values[row, :k - 1].view(np.uint32))
        assert values[row, k - 1] == fast_values[row, :k].max()


def test_bound_is_the_largest_fast_distance(monkeypatch):
    """Exercise the bound with crafted fast lists of every shape.

    Every query sits at the origin; reference j of the first 600 lies at
    distance j * j, except two that tie: 300 with 2 (distance 4) and 450 with
    3 (distance 9). The exact top five are 0, 1, 2, 300, 3.
    """
    train = np.zeros((600, 4), np.float32)
    train[:, 0] = np.arange(600)
    train[300, 0], train[450, 0] = 2, 3
    queries = np.zeros((4, 4), np.float32)
    k = 5
    lists = np.array([
        [500, 0, 1, 2, 300],  # unsorted: slot k-1 is closer than the true k-th
        [0, _SENTINEL, 1, 2, 300],  # four real references: no bound, whatever the distances
        [0, 1, 2, 300, 450],  # tight: 3 ties the last at 9 and must survive
        [597, 598, 599, 596, 595],  # loose
    ], np.int32)
    fast = fused.refine_neighbors(mx.array(train), mx.array(queries), mx.array(np.minimum(lists, 599)))
    fast = mx.where(mx.array(lists == _SENTINEL), mx.array(0.0, mx.float32), fast)
    seen = []
    _route(monkeypatch, "partitioned")
    _force_uncertified(monkeypatch, None, seen)
    distances, found = screen._finish(
        mx.array(train), mx.array(queries), k, fast, mx.array(lists), mx.zeros((4,), mx.float32),
        width=k, exclude_self=False, query_offset=0,
    )
    np.testing.assert_array_equal(np.array(found), [[0, 1, 2, 300, 3]] * 4)
    _check_distances(train, queries, np.array(distances), np.array(found))
    (_, _, values, indices, certified), = seen
    assert not certified.any()
    np.testing.assert_array_equal(values[:, k - 1], [500.0**2, 4.0, 9.0, 599.0**2])
    np.testing.assert_array_equal(indices[:, k - 1], [300, -1, 450, 595])


@pytest.mark.parametrize("bound", ["fast", "withdrawn"])
@pytest.mark.parametrize("k", [9, 16])
def test_near_ties_below_float32_resolution_match_exact_order(bound, k, monkeypatch):
    # Cluster centers with full float32 mantissas plus a quarter grid leave
    # many distances within an FP32 ulp of each other, where neither the fast
    # order nor float64 sums are exact. "fast" bounds every row by its fast
    # list; "withdrawn" by its certified output.
    rng = np.random.default_rng(17 + k)
    centers = (rng.standard_normal((5, 12)) * 10).astype(np.float32)
    def sample(count):
        points = centers[rng.integers(0, 5, count)] + rng.integers(-1, 2, (count, 12)) * 0.25
        return points.astype(np.float32)
    train, queries = sample(4000), sample(48)
    rows = np.arange(len(queries))
    _route(monkeypatch, "partitioned", forced=rows if bound == "fast" else None)
    _force_uncertified(monkeypatch, rows)
    distances, found = (np.array(a) for a in fused.fused_neighbors(mx.array(train), mx.array(queries), k))
    np.testing.assert_array_equal(found, _exact_oracle(train, queries, k))
    _check_distances(train, queries, distances, found)


def _circle(features, seed):
    """Integer rows on a circle whose FP32 distances split one exact tie.

    N = 2*5*13*17*29*41*101 = 265396690 has 32 representations a*a + b*b.
    With signs and swaps they give 256 rows at exactly distance N from the
    origin, but their 26-bit squares round, so the compensated FP32 distances
    take two values an ulp apart. Exact order breaks the tie by index; FP32
    order puts the smaller value first. Three closer rows follow the circle
    and 9000 farther rows surround it.
    """
    n_squared = 2 * 5 * 13 * 17 * 29 * 41 * 101
    roots = [(a, math.isqrt(n_squared - a * a)) for a in range(1, math.isqrt(n_squared // 2) + 1)
             if math.isqrt(n_squared - a * a) ** 2 == n_squared - a * a]
    points = sorted({(sa * x, sb * y) for a, b in roots for x, y in ((a, b), (b, a))
                     for sa in (-1, 1) for sb in (-1, 1)})
    rng = np.random.default_rng(seed)
    circle = np.zeros((len(points), features), np.float32)
    circle[:, :2] = np.array(points)[rng.permutation(len(points))]
    near = np.zeros((3, features), np.float32)
    near[:, 0] = [1, 2, 3]
    far = np.zeros((9000, features), np.float32)
    far[:, :2] = rng.integers(17000, 20000, (9000, 2)) * rng.choice([-1, 1], (9000, 2))
    return np.concatenate([far[:4000], circle, near, far[4000:]])


@pytest.mark.parametrize(("features", "k"), [(2, 5), (2, 9), (2, 32), (4, 16), (4, 32)])
@pytest.mark.parametrize("rows", [8, 40])  # 40 rows pair queries for k <= 16
def test_exact_ties_split_across_fp32_values_keep_index_order(features, k, rows):
    # Some of the exact top k lie strictly above every distance of the fast
    # top k, so only the FP32 interval of the bound keeps them in the search.
    train = _circle(features, features)
    queries = np.zeros((rows, features), np.float32)
    X = mx.array(train)
    every = np.arange(len(train), dtype=np.int32)
    fast = np.array(fused.refine_neighbors(X, mx.array(queries[:1]), mx.array(every[None])))[0]
    expected = _oracle(train, queries, k)
    by_value = np.lexsort((every, fast))[:k]  # the fast top k (FP32 values, then index)
    assert np.sum(fast[expected[0]] > fast[by_value].max()) > 0
    distances, found = (np.array(a) for a in fused.fused_neighbors(X, mx.array(queries), k))
    np.testing.assert_array_equal(found, expected)
    _check_distances(train, queries, distances, found)


@pytest.mark.benchmark
def test_bound_skips_exact_comparisons_on_tied_rows(monkeypatch):
    """The bounded search must stay well below the unbounded one on ties.

    Integer data leave most of these 256 rows uncertain at k=9. Without the
    bound, exact comparisons against each chunk's tied running k-th took
    about half of an 8 ms query at 262144 x 16 on M4 Max; here bounded is
    about 0.3-0.4 of unbounded. The bound is relative, so slower GPUs and
    background load affect both sides alike.
    """
    rng = np.random.default_rng(11)
    train = rng.integers(-2, 3, (65536, 16)).astype(np.float32)
    queries = rng.integers(-2, 3, (256, 16)).astype(np.float32)
    index, block = fit_neighbors(mx.array(train)), mx.array(queries)

    def best():
        mx.eval(index.query(block, 9))
        times = []
        for _ in range(5):
            start = time.perf_counter()
            mx.eval(index.query(block, 9))
            times.append(time.perf_counter() - start)
        return min(times)

    bounded = best()
    original = fallback.partitioned_neighbors
    monkeypatch.setattr(fallback, "partitioned_neighbors",
                        lambda *args, **options: original(*args, **{**options, "bounded": False}))
    unbounded = best()
    assert bounded < 0.75 * unbounded, (bounded, unbounded)


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
