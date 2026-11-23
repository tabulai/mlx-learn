"""Native radius engine: canonical distances, exact resolution, finisher, capsules, threads."""

from __future__ import annotations

import gc
import math
import os
import threading
import time
from fractions import Fraction

import numpy as np
import pytest

native = pytest.importorskip("mlxlearn._kernels._radius_cpu_native")

from mlxlearn._kernels import radius_cpu as rc  # noqa: E402
from mlxlearn._kernels import radius_host as rh  # noqa: E402
from mlxlearn._kernels.radius_contract import Thresholds, exact_le, thresholds  # noqa: E402
from mlxlearn._kernels.radius_reduce import reduce_mean_csr, reduce_votes_csr  # noqa: E402

from ._radius_native_helpers import assert_same_csr, oracle, sequential_d2  # noqa: E402

ALL = os.cpu_count() or 1


def _bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def _random_pairs(rng, n, m, k):
    return rng.integers(0, m, size=k), rng.integers(0, n, size=k)


# ----------------------------------------------------------------------------------
# canonical distances
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 3, 7, 16, 33, 64])
def test_canonical_distances_equal_a_sequential_numpy_sum(d, rng):
    n, m, k = 400, 300, 40_000
    X = rng.normal(size=(n, d)).astype(np.float32)
    Q = rng.normal(size=(m, d)).astype(np.float32)
    # Mix magnitudes, exact zeros and float32 subnormals into both sides.
    X[::7] *= np.float32(2.0**-140)
    Q[::5] *= np.float32(2.0**-135)
    X[1::11] *= np.float32(2.0**60)
    Q[2::13, 0] = 0.0
    rows, cols = _random_pairs(rng, n, m, k)
    expected = sequential_d2(X, Q, rows, cols)
    for threads in (1, ALL):
        actual = rc.canonical_distances(X, Q, rows, cols, threads=threads)
        np.testing.assert_array_equal(_bits(actual), _bits(expected))


def test_canonical_distances_accept_int32_pairs(rng):
    X = rng.normal(size=(50, 5)).astype(np.float32)
    Q = rng.normal(size=(20, 5)).astype(np.float32)
    rows, cols = _random_pairs(rng, 50, 20, 500)
    a = rc.canonical_distances(X, Q, rows.astype(np.int32), cols.astype(np.int32))
    b = rc.canonical_distances(X, Q, rows, cols)
    np.testing.assert_array_equal(_bits(a), _bits(b))


# ----------------------------------------------------------------------------------
# exact resolution
# ----------------------------------------------------------------------------------


def _exact_S(q, x) -> Fraction:
    return sum((Fraction(float(a)) - Fraction(float(b))) ** 2 for a, b in zip(q, x, strict=True))


def _native_le(X, Q, rows, cols, R2):
    out = np.empty(len(rows), dtype=np.uint8)
    native.exact_le_pairs(X, Q, np.asarray(rows, np.int64), np.asarray(cols, np.int64), float(R2),
                          out, 1)
    return out.astype(bool)


def test_exact_resolution_on_exact_ties_and_neighbouring_ulps():
    # Pairs whose exact S is a float64 value: S == R2 exactly, and R2 one ulp
    # either side of S.
    cases = [
        ([0.0, 0.0], [3.0, 4.0]),
        ([1.5], [-1.5]),
        ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
        ([2.0**-149], [0.0]),
        ([2.0**-149, -(2.0**-149)], [0.0, 2.0**-148]),
        ([3.0e38], [-3.0e38]),
        ([1.0, 2.0**-30], [1.0 + 2.0**-23, 0.0]),
        ([0.75, -0.5, 0.125], [0.25, 0.5, -0.375]),
    ]
    for q, x in cases:
        Q = np.asarray([q], dtype=np.float32)
        X = np.asarray([x], dtype=np.float32)
        S = _exact_S(Q[0], X[0])
        base = float(S)
        for R2 in (base, math.nextafter(base, -math.inf), math.nextafter(base, math.inf)):
            if R2 < 0:
                continue
            truth = S <= Fraction(R2)
            assert exact_le(Q[0], X[0], R2) == truth
            assert _native_le(X, Q, [0], [0], R2)[0] == truth, (q, x, R2)


@pytest.mark.parametrize("d", [1, 3, 8, 24])
def test_exact_resolution_matches_fractions_on_random_pairs(d, rng):
    n = 60
    X = rng.normal(size=(n, d)).astype(np.float32)
    Q = rng.normal(size=(n, d)).astype(np.float32)
    X[::4] *= np.float32(2.0**-130)
    Q[::4] *= np.float32(2.0**-130)
    rows = np.arange(n)
    cols = rng.permutation(n)
    for i in range(n):
        S = _exact_S(Q[rows[i]], X[cols[i]])
        nearest = float(S)
        for R2 in (nearest, math.nextafter(nearest, -math.inf), math.nextafter(nearest, math.inf)):
            if R2 < 0:
                continue
            truth = S <= Fraction(R2)
            assert _native_le(X, Q, rows[i:i + 1], cols[i:i + 1], R2)[0] == truth
            assert exact_le(Q[rows[i]], X[cols[i]], R2) == truth


def test_exact_resolution_large_radius_shortcuts():
    X = np.asarray([[-3e38, 3e38]], dtype=np.float32)
    Q = np.asarray([[3e38, -3e38]], dtype=np.float32)
    assert _native_le(X, Q, [0], [0], 2.0**342)[0]
    assert _native_le(X, Q, [0], [0], math.inf)[0]
    assert not _native_le(X, Q, [0], [0], 1e77)[0]


# ----------------------------------------------------------------------------------
# finish_pairs
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [2, 3, 8, 17, 64])
def test_finish_pairs_matches_host_refine(d, rng):
    n, m = 2000, 300
    X = rng.normal(size=(n, d)).astype(np.float32)
    Q = rng.normal(size=(m, d)).astype(np.float32)
    R2 = 0.8 * d
    thr = thresholds(d, R2)
    # A superset screen: every pair within a looser radius.
    loose = oracle(X, Q, 1.3 * R2)
    rows, cols = loose.row_ids(), loose.indices
    keep, squared = rh.refine(X, Q, rows, cols, thr)
    expected_rows, expected_cols = rows[keep], cols[keep]
    canonical = sequential_d2(X, Q, expected_rows, expected_cols)
    for presorted in (True, False):
        order = np.arange(len(rows)) if presorted else rng.permutation(len(rows))
        result = rc.finish_pairs(X, Q, rows[order], cols[order], thr, presorted=presorted).to_csr()
        np.testing.assert_array_equal(result.row_ids(), expected_rows)
        np.testing.assert_array_equal(result.indices, expected_cols)
        np.testing.assert_array_equal(_bits(result.distances),
                                      _bits(np.sqrt(np.minimum(canonical, R2))))
        np.testing.assert_allclose(result.distances, np.sqrt(squared[keep]), rtol=1e-14)


def test_finish_pairs_exclude_self_offset_and_duplicates(rng):
    base = rng.normal(size=(100, 3)).astype(np.float32)
    X = np.vstack([base, base])  # every row duplicated at i and i + 100
    thr = thresholds(3, 0.5)
    q_offset = 40
    Q = X[q_offset:q_offset + 60]
    rows = np.repeat(np.arange(60), len(X))
    cols = np.tile(np.arange(len(X)), 60)
    result = rc.finish_pairs(X, Q, rows, cols, thr, exclude_self=True, q_offset=q_offset).to_csr()
    expected = oracle(X, Q, 0.5, exclude_self=True, q_offset=q_offset)
    assert_same_csr(result, expected)
    own = result.row_ids() + q_offset == result.indices
    assert not own.any()
    # The duplicate of each query (distance 0) stays.
    for row in range(60):
        twin = (row + q_offset + 100) % 200
        assert twin in result.indices[result.indptr[row]:result.indptr[row + 1]]


def test_finish_pairs_with_every_pair_uncertain_is_still_exact(rng):
    d = 3
    X = rng.integers(-3, 4, size=(300, d)).astype(np.float32)
    Q = rng.integers(-3, 4, size=(40, d)).astype(np.float32)
    R2 = 4.0
    wide = Thresholds(R2, d, 0.0, math.inf, math.inf, 0.0, None, 0.0, 0.0, False)
    rows = np.repeat(np.arange(40), 300)
    cols = np.tile(np.arange(300), 40)
    result = rc.finish_pairs(X, Q, rows, cols, wide).to_csr()
    assert_same_csr(result, oracle(X, Q, R2, integer_exact=True))


def test_finish_pairs_sort_mode_orders_by_distance_then_index(rng):
    X = rng.integers(-2, 3, size=(500, 2)).astype(np.float32)  # many exact distance ties
    Q = rng.integers(-2, 3, size=(30, 2)).astype(np.float32)
    thr = thresholds(2, 5.0)
    rows = np.repeat(np.arange(30), 500)
    cols = np.tile(np.arange(500), 30)
    result = rc.finish_pairs(X, Q, rows, cols, thr, sort=True).to_csr()
    plain = rc.finish_pairs(X, Q, rows, cols, thr).to_csr()
    for row in range(30):
        a, b = result.indptr[row], result.indptr[row + 1]
        order = np.lexsort((plain.indices[a:b], plain.distances[a:b]))
        np.testing.assert_array_equal(result.indices[a:b], plain.indices[a:b][order])
        np.testing.assert_array_equal(_bits(result.distances[a:b]), _bits(plain.distances[a:b][order]))


def test_finish_pairs_rejects_invalid_pairs(rng):
    X = rng.normal(size=(10, 2)).astype(np.float32)
    Q = rng.normal(size=(5, 2)).astype(np.float32)
    thr = thresholds(2, 1.0)
    with pytest.raises(ValueError, match="out of range"):
        rc.finish_pairs(X, Q, [0, 1], [3, 10], thr)
    with pytest.raises(ValueError, match="out of range"):
        rc.finish_pairs(X, Q, [5], [0], thr, presorted=True)
    with pytest.raises(ValueError, match="non-decreasing"):
        rc.finish_pairs(X, Q, [2, 1], [0, 0], thr, presorted=True)
    with pytest.raises(ValueError, match="thresholds"):
        native.finish_pairs(X, Q, np.zeros(1, np.int64), np.zeros(1, np.int64),
                            (1.0, 2.0, 0.5, 3.0, 0.1), 0, False, 1, 1, False)


def test_finish_pairs_with_no_pairs_returns_empty_rows(rng):
    X = rng.normal(size=(10, 2)).astype(np.float32)
    Q = rng.normal(size=(4, 2)).astype(np.float32)
    result = rc.finish_pairs(X, Q, np.empty(0, np.int64), np.empty(0, np.int64), thresholds(2, 1.0))
    csr = result.to_csr()
    assert result.sizes() == (4, 0)
    np.testing.assert_array_equal(csr.indptr, np.zeros(5, np.int64))


# ----------------------------------------------------------------------------------
# threads, errors, GIL, lifetime, FPCR
# ----------------------------------------------------------------------------------


def test_outputs_are_bitwise_identical_for_any_thread_count(rng):
    d = 5
    X = rng.normal(size=(20_000, d)).astype(np.float32)
    Q = rng.normal(size=(3000, d)).astype(np.float32)
    thr = thresholds(d, 1.5)
    tree = rc.build_tree(X)
    reference = None
    for threads in (1, 2, ALL):
        tree_csr = rc.tree_query(tree, X, Q, thr, threads=threads).to_csr()
        brute_csr = rc.brute_query(X, Q, thr, threads=threads).to_csr()
        finish_csr = rc.finish_pairs(X, Q, tree_csr.row_ids()[::-1], tree_csr.indices[::-1], thr,
                                     threads=threads).to_csr()
        if reference is None:
            reference = tree_csr
        assert_same_csr(tree_csr, reference, context=f"tree threads={threads}")
        assert_same_csr(brute_csr, reference, context=f"brute threads={threads}")
        assert_same_csr(finish_csr, reference, context=f"finish threads={threads}")


def test_task_exceptions_propagate_as_python_errors(rng):
    X = rng.normal(size=(20_000, 3)).astype(np.float32)
    Q = rng.normal(size=(2000, 3)).astype(np.float32)
    thr = thresholds(3, 0.5)
    tree = rc.build_tree(X)
    args = rc.threshold_tuple(thr)
    fail = rc.NEED_DISTANCE | rc._FAIL_TEST
    for threads in (1, ALL):
        with pytest.raises(RuntimeError, match="injected failure"):
            native.tree_query(tree._capsule, Q, args, 0, False, fail, threads)
        with pytest.raises(RuntimeError, match="injected failure"):
            native.brute_query(X, Q, args, 0, False, fail, threads)
        with pytest.raises(RuntimeError, match="injected failure"):
            native.brute_query(X, Q[:2], args, 0, False, fail, threads)
        rows = np.repeat(np.arange(10), 10)
        with pytest.raises(RuntimeError, match="injected failure"):
            native.finish_pairs(X, Q, rows, rows, args, 0, False, fail, threads, True)
    # The engine still works after a failed call.
    assert rc.tree_query(tree, X, Q, thr).nnz == rc.brute_query(X, Q, thr).nnz


def test_native_calls_release_the_gil(rng):
    X = rng.normal(size=(60_000, 16)).astype(np.float32)
    Q = rng.normal(size=(400, 16)).astype(np.float32)
    thr = thresholds(16, 4.0)
    ticks = 0
    done = threading.Event()

    def spin():
        nonlocal ticks
        while not done.is_set():
            ticks += 1

    worker = threading.Thread(target=spin)
    worker.start()
    try:
        time.sleep(0.02)
        start_ticks = ticks
        start = time.perf_counter()
        rc.brute_query(X, Q, thr, threads=1)  # single-threaded, ~0.1-0.3 s
        elapsed = time.perf_counter() - start
        advanced = ticks - start_ticks
    finally:
        done.set()
        worker.join()
    assert elapsed > 0.03
    # With the GIL held for the whole call the spinner would barely move.
    assert advanced > 20_000, (advanced, elapsed)


def test_results_and_trees_survive_deletion_of_their_inputs(rng):
    X = rng.normal(size=(3000, 3)).astype(np.float32)
    Q = rng.normal(size=(200, 3)).astype(np.float32)
    thr = thresholds(3, 0.4)
    expected = oracle(X, Q, 0.4)
    tree = rc.build_tree(X.copy())
    result = rc.tree_query(tree, None, Q.copy(), thr)
    brute = rc.brute_query(X.copy(), Q.copy(), thr)
    rows, cols = expected.row_ids(), expected.indices
    finished = rc.finish_pairs(X.copy(), Q.copy(), rows.copy(), cols.copy(), thr)
    del tree, rows, cols
    gc.collect()
    for item in (result, brute, finished):
        assert_same_csr(item.to_csr(), expected)
    tree = rc.build_tree(X.copy())
    gc.collect()
    assert_same_csr(rc.tree_query(tree, None, Q, thr).to_csr(), expected)


def test_count_only_result_reports_more_than_2_31_pairs_without_columns():
    n, m = 1 << 16, (1 << 15) + 1
    X = np.arange(n, dtype=np.float32)[:, None]
    Q = np.zeros((m, 1), dtype=np.float32)
    tree = rc.build_tree(X)
    result = rc.tree_query(tree, X, Q, thresholds(1, math.inf), count_only=True)
    assert result.sizes() == (m, n * m)
    assert n * m > 2**31
    np.testing.assert_array_equal(result.counts(), np.full(m, n))
    with pytest.raises(ValueError, match="count-only"):
        result.to_csr()


def test_flush_to_zero_on_the_calling_thread_does_not_change_results(rng):
    d = 3
    scale = np.float32(2.0**-138)
    X = (rng.integers(-60, 61, size=(4000, d)) * scale).astype(np.float32)
    Q = (rng.integers(-60, 61, size=(300, d)) * scale).astype(np.float32)
    R2 = float(400.0 * float(scale) ** 2)
    thr = thresholds(d, R2)
    tree = rc.build_tree(X)
    rows, cols = _random_pairs(rng, 4000, 300, 5000)
    expected = [rc.tree_query(tree, X, Q, thr, threads=t).to_csr() for t in (1, ALL)]
    expected_d2 = rc.canonical_distances(X, Q, rows, cols, threads=1)
    assert expected[0].nnz > 0
    previous = native._fpcr()
    try:
        native._fpcr(previous | (1 << 24))
        assert native._fpcr() & (1 << 24)
        actual = [rc.tree_query(tree, X, Q, thr, threads=t).to_csr() for t in (1, ALL)]
        actual_brute = rc.brute_query(X, Q, thr, threads=1).to_csr()
        actual_d2 = rc.canonical_distances(X, Q, rows, cols, threads=1)
    finally:
        native._fpcr(previous)
    assert native._fpcr() == previous
    for a, e in zip(actual, expected, strict=True):
        assert_same_csr(a, e)
    assert_same_csr(actual_brute, expected[0])
    np.testing.assert_array_equal(_bits(actual_d2), _bits(expected_d2))


def test_invalid_inputs_are_rejected(rng):
    X = rng.normal(size=(10, 2)).astype(np.float32)
    thr = thresholds(2, 1.0)
    with pytest.raises(ValueError, match="float32"):
        native.brute_query(X.astype(np.float64), X, rc.threshold_tuple(thr), 0, False, 1, 1)
    with pytest.raises(ValueError, match="features"):
        native.brute_query(X, X[:, :1].copy(), rc.threshold_tuple(thr), 0, False, 1, 1)
    with pytest.raises(ValueError, match="SORT requires"):
        native.brute_query(X, X, rc.threshold_tuple(thr), 0, False, rc.SORT, 1)
    with pytest.raises(ValueError, match="finite"):
        rc.build_tree(np.asarray([[0.0, np.nan]], dtype=np.float32))
    result = rc.brute_query(X, X, thr, need_distance=False)
    with pytest.raises(ValueError, match="without distances"):
        result.to_csr(need_distance=True)
    with pytest.raises(ValueError):
        native.result_fill_csr(result._capsule, np.empty(3, np.int64), np.empty(0, np.int64), None)


# ----------------------------------------------------------------------------------
# fused reductions
# ----------------------------------------------------------------------------------


def _reduction_case(rng):
    base = rng.normal(size=(800, 3)).astype(np.float32)
    X = np.vstack([base, base[:200]])  # duplicates give zero distances
    Q = np.vstack([base[:50], rng.normal(size=(150, 3)).astype(np.float32),
                   np.full((10, 3), 50.0, np.float32)])  # coincident, generic, empty rows
    return X, Q, thresholds(3, 0.6)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("engine", ["tree", "brute"])
def test_fused_votes_equal_radius_reduce_bitwise(weighted, engine, rng):
    X, Q, thr = _reduction_case(rng)
    codes = rng.integers(0, 4, size=len(X)).astype(np.int32)
    if engine == "tree":
        result = rc.tree_query(rc.build_tree(X), X, Q, thr, need_distance=weighted)
    else:
        result = rc.brute_query(X, Q, thr, need_distance=weighted)
    for threads in (1, ALL):
        result._threads = threads
        votes, counts = result.votes(codes, 4, weighted)
        ref_votes, ref_counts = reduce_votes_csr(result.to_csr(), codes, 4, weighted=weighted)
        np.testing.assert_array_equal(votes.view(np.uint64), ref_votes.view(np.uint64))
        np.testing.assert_array_equal(counts, ref_counts)
    votes64, _ = result.votes(codes.astype(np.int64), 4, weighted)
    np.testing.assert_array_equal(votes64, votes)
    assert (counts[-10:] == 0).all() and (votes[-10:] == 0).all()


@pytest.mark.parametrize("weighted", [False, True])
def test_fused_means_equal_radius_reduce_bitwise(weighted, rng):
    X, Q, thr = _reduction_case(rng)
    y = rng.normal(size=len(X)) * 10.0
    result = rc.tree_query(rc.build_tree(X), X, Q, thr, need_distance=weighted)
    for threads in (1, ALL):
        result._threads = threads
        pred, counts = result.mean(y, weighted)
        ref_pred, ref_counts = reduce_mean_csr(result.to_csr(), y, weighted=weighted)
        np.testing.assert_array_equal(pred.view(np.uint64), ref_pred.view(np.uint64))
        np.testing.assert_array_equal(counts, ref_counts)
    assert np.isnan(pred[-10:]).all()


def test_fused_reductions_validate_inputs(rng):
    X, Q, thr = _reduction_case(rng)
    result = rc.brute_query(X, Q, thr, need_distance=False)
    with pytest.raises(ValueError, match="out of range"):
        result.votes(np.full(len(X), 7, np.int32), 4, False)
    with pytest.raises(ValueError, match="distances"):
        result.votes(np.zeros(len(X), np.int32), 4, True)
    with pytest.raises(ValueError, match="sizes"):
        result.mean(np.zeros(3), False)
