"""Native KD-tree: exact membership, determinism, boxes, two-pass fill and pilot."""

from __future__ import annotations

import math
import os
import pickle
import threading

import numpy as np
import pytest

pytest.importorskip("mlxlearn._kernels._radius_cpu_native")

from mlxlearn._kernels import radius_cpu as rc  # noqa: E402
from mlxlearn._kernels.radius_contract import Thresholds, thresholds  # noqa: E402
from mlxlearn._kernels.radius_result import sort_rows  # noqa: E402

from ._radius_native_helpers import assert_same_csr, generators, oracle  # noqa: E402

ALL = os.cpu_count() or 1


def _query(tree, X, Q, thr, **kwargs):
    return rc.tree_query(tree, X, Q, thr, **kwargs).to_csr()


@pytest.mark.parametrize("d", [1, 2, 3, 4, 7, 8, 9, 12, 16])
def test_tree_membership_and_distances_match_the_exact_oracle(d, rng):
    n, m = 700, 120
    for name, (X, Q, R2, integer_exact) in generators(rng, n, m, d).items():
        thr = thresholds(d, R2, integer_exact=integer_exact)
        expected = oracle(X, Q, R2, integer_exact=integer_exact)
        tree = rc.build_tree(X, leaf_size=8)
        assert_same_csr(_query(tree, X, Q, thr), expected, context=f"{name} d={d}")


@pytest.mark.parametrize("R2", [0.0, math.inf])
def test_tree_handles_zero_and_infinite_radius(R2, rng):
    base = rng.integers(-2, 3, size=(300, 3)).astype(np.float32)
    X = np.vstack([base, base[:50]])
    Q = np.vstack([base[:40], rng.normal(size=(20, 3)).astype(np.float32)])
    thr = thresholds(3, R2)
    tree = rc.build_tree(X, leaf_size=4)
    expected = oracle(X, Q, R2)
    assert_same_csr(_query(tree, X, Q, thr), expected)
    assert_same_csr(_query(tree, X, Q, thr, sort=True), sort_rows(expected))
    if R2 == 0.0:
        assert (expected.distances == 0).all() and expected.nnz >= 40


@pytest.mark.parametrize("d", [2, 3, 6])
def test_tree_self_queries_exclude_only_the_own_index(d, rng):
    base = rng.normal(size=(400, d)).astype(np.float32)
    X = np.vstack([base, base[:100], base[:50]])  # duplicates at several indices
    thr = thresholds(d, 0.3 * d)
    tree = rc.build_tree(X, leaf_size=8)
    full = oracle(X, X, thr.R2, exclude_self=True)
    assert_same_csr(_query(tree, X, X, thr, exclude_self=True), full)
    # Batches of a self-query use q_offset.
    for start, stop in ((0, 137), (137, 401), (401, len(X))):
        batch = _query(tree, X, X[start:stop], thr, exclude_self=True, q_offset=start)
        assert_same_csr(batch, oracle(X, X[start:stop], thr.R2, exclude_self=True, q_offset=start))
    # Each duplicate of a query is still returned.
    result = _query(tree, X, X[:50], thr, exclude_self=True)
    for row in range(50):
        found = result.indices[result.indptr[row]:result.indptr[row + 1]]
        assert row + 400 in found and row + 500 in found and row not in found


def test_tree_output_is_identical_across_threads_and_leaf_sizes(rng):
    d = 4
    X = rng.normal(size=(30_000, d)).astype(np.float32)
    Q = rng.normal(size=(2500, d)).astype(np.float32)
    thr = thresholds(d, 0.8)
    reference = None
    for leaf_size in (1, 5, 16, 33, 200):
        tree = rc.build_tree(X, leaf_size=leaf_size)
        for threads in (1, 2, ALL):
            for sort in (False, True):
                result = _query(tree, X, Q, thr, threads=threads, sort=sort)
                key = "sorted" if sort else "plain"
                if reference is None:
                    reference = {}
                reference.setdefault(key, result)
                assert_same_csr(result, reference[key], context=f"leaf={leaf_size} t={threads}")
    assert_same_csr(reference["sorted"], sort_rows(reference["plain"]))


@pytest.mark.parametrize("kind", ["uniform", "identical", "lattice"])
def test_tree_build_is_independent_of_thread_count(kind, rng):
    n, d = 300_000, 3
    if kind == "uniform":
        X = rng.random((n, d), dtype=np.float32)
    elif kind == "identical":
        X = np.full((n, d), 1.25, dtype=np.float32)
        X[::1000, 1] = -0.0
        X[1::1000, 1] = 0.0
    else:
        X = rng.integers(0, 4, size=(n, d)).astype(np.float32)
    exports = [rc.build_tree(X, threads=t).export() for t in (1, 3, ALL)]
    for other in exports[1:]:
        for key in ("perm", "points", "lo", "hi"):
            np.testing.assert_array_equal(other[key].view(np.uint32 if key != "perm" else np.int32),
                                          exports[0][key].view(np.uint32 if key != "perm" else np.int32),
                                          err_msg=key)


def test_tree_structure_boxes_and_leaf_order(rng):
    X = rng.normal(size=(5000, 3)).astype(np.float32)
    X[100:300] = X[0]  # a block of duplicates
    tree = rc.build_tree(X, leaf_size=16)
    parts = tree.export()
    perm, points, lo, hi = parts["perm"], parts["points"], parts["lo"], parts["hi"]
    np.testing.assert_array_equal(np.sort(perm), np.arange(len(X)))
    np.testing.assert_array_equal(points, X[perm])
    # Recompute node ranges of the implicit heap and check exact boxes.
    begin = np.zeros(tree.n_nodes, np.int64)
    end = np.zeros(tree.n_nodes, np.int64)
    end[0] = len(X)
    first_leaf = (1 << tree.depth) - 1
    for h in range(first_leaf):
        mid = begin[h] + (end[h] - begin[h]) // 2
        begin[2 * h + 1], end[2 * h + 1] = begin[h], mid
        begin[2 * h + 2], end[2 * h + 2] = mid, end[h]
    for h in range(tree.n_nodes):
        block = points[begin[h]:end[h]]
        np.testing.assert_array_equal(lo[h], block.min(axis=0))
        np.testing.assert_array_equal(hi[h], block.max(axis=0))
        if h >= first_leaf:
            assert end[h] - begin[h] <= 16
            assert (np.diff(perm[begin[h]:end[h]]) > 0).all()
    assert math.ceil(len(X) / 2**tree.depth) <= 16 < math.ceil(len(X) / 2 ** (tree.depth - 1))


def test_tree_prune_and_bulk_tests_are_sound_at_the_thresholds(rng):
    # Integer lattices put many box corners exactly on the radius: with
    # integer-exact thresholds a box whose farthest corner is at R2 is bulk
    # included and one whose nearest point is at R2 is still visited.
    d = 3
    grid = np.stack(np.meshgrid(*[np.arange(-4, 5)] * d, indexing="ij"), -1).reshape(-1, d)
    X = np.repeat(grid, 2, axis=0).astype(np.float32)
    Q = grid[::7].astype(np.float32)
    for R2 in (1.0, 2.0, 3.0, 4.0, 9.0, 12.0):
        for integer_exact in (True, False):
            thr = thresholds(d, R2, integer_exact=integer_exact)
            expected = oracle(X, Q, R2, integer_exact=integer_exact)
            for leaf_size in (1, 2, 8):
                tree = rc.build_tree(X, leaf_size=leaf_size)
                assert_same_csr(_query(tree, X, Q, thr), expected, context=f"R2={R2}")
    # Thresholds that force exact resolution everywhere and forbid pruning and
    # bulk inclusion still give the exact answer.
    wide = Thresholds(4.0, d, 0.0, math.inf, math.inf, 0.0, None, 0.0, 0.0, False)
    tree = rc.build_tree(X, leaf_size=4)
    assert_same_csr(_query(tree, X, Q, wide), oracle(X, Q, 4.0, integer_exact=True))


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_two_pass_fill_equals_one_pass(sort, exclude_self, rng):
    d = 3
    X = rng.random((40_000, d), dtype=np.float32)
    Q = X[:3000] if exclude_self else rng.random((3000, d), dtype=np.float32)
    thr = thresholds(d, 0.03**2)
    tree = rc.build_tree(X)
    one = _query(tree, X, Q, thr, exclude_self=exclude_self, sort=sort)
    for threads in (1, ALL):
        two = rc.tree_query_csr(tree, X, Q, thr, exclude_self=exclude_self, sort=sort,
                                threads=threads)
        assert_same_csr(two, one)
    two = rc.tree_query_csr(tree, X, Q, thr, exclude_self=exclude_self, need_distance=False)
    assert two.distances is None
    np.testing.assert_array_equal(two.indices, _query(tree, X, Q, thr, exclude_self=exclude_self).indices)


def test_tree_queries_without_distances_return_the_same_members(rng):
    X = rng.normal(size=(20_000, 3)).astype(np.float32)
    Q = rng.normal(size=(500, 3)).astype(np.float32)
    thr = thresholds(3, 4.0)  # dense enough for bulk inclusion
    tree = rc.build_tree(X)
    with_distances = _query(tree, X, Q, thr)
    without = rc.tree_query(tree, X, Q, thr, need_distance=False).to_csr()
    assert without.distances is None
    np.testing.assert_array_equal(without.indptr, with_distances.indptr)
    np.testing.assert_array_equal(without.indices, with_distances.indices)


@pytest.mark.parametrize("need_distance", [False, True])
def test_dense_tree_rows_keep_indices_and_distances_paired(need_distance, rng):
    # Non-multiple-of-64 size, empty words and members on both sides of a word
    # boundary exercise the rank calculation. Reuse the scratch on a new row.
    X = rng.random((1027, 2), dtype=np.float32)
    X[128:320, 0] += 4
    X[[63, 64, 1026]] = (0.1, 0.2)
    Q = np.array([[0.25, 0.25], [0.75, 0.75], [4.5, 0.5]], dtype=np.float32)
    thr = thresholds(X.shape[1], 0.25)
    tree = rc.build_tree(X, leaf_size=5)
    expected = oracle(X, Q, thr.R2)
    assert np.min(np.diff(expected.indptr)) >= 128
    for result in (rc.tree_query(tree, X, Q, thr, need_distance=need_distance, threads=1).to_csr(),
                   rc.tree_query_csr(tree, X, Q, thr, need_distance=need_distance, threads=1)):
        assert_same_csr(result, expected, distances=need_distance)


def test_tree_pilot_counts_are_deterministic_and_consistent(rng):
    X = rng.normal(size=(20_000, 3)).astype(np.float32)
    Q = rng.normal(size=(64, 3)).astype(np.float32)
    R2 = 0.25
    tree = rc.build_tree(X)
    stats = rc.tree_pilot(tree, Q, R2, visit_cap=len(X))
    np.testing.assert_array_equal(stats, rc.tree_pilot(tree, Q, R2, visit_cap=len(X), threads=ALL))
    expected = np.diff(oracle(X, Q, R2).indptr)
    np.testing.assert_array_equal(stats[:, 2], expected)  # no exact ties in this data
    assert (stats[:, 0] >= 1).all() and (stats[:, 1] >= stats[:, 2]).all()
    capped = rc.tree_pilot(tree, Q, R2, visit_cap=32)
    assert (capped[:, 1] <= stats[:, 1]).all()
    assert (capped[:, 1] < 32 + 2 * tree.leaf_size + len(X)).all()


def test_tree_edge_shapes(rng):
    thr = thresholds(1, 1.0)
    empty = rc.build_tree(np.empty((0, 1), np.float32))
    result = rc.tree_query(empty, None, np.zeros((3, 1), np.float32), thr)
    assert result.sizes() == (3, 0)
    tiny = np.asarray([[0.0], [0.5], [3.0]], np.float32)
    tree = rc.build_tree(tiny)
    assert tree.depth == 0
    assert_same_csr(_query(tree, tiny, tiny, thr), oracle(tiny, tiny, 1.0))
    no_queries = rc.tree_query(tree, tiny, np.empty((0, 1), np.float32), thr)
    assert no_queries.sizes() == (0, 0)
    assert no_queries.to_csr().indptr.tolist() == [0]
    with pytest.raises(ValueError, match="features"):
        rc.tree_query(tree, None, np.zeros((2, 2), np.float32), thresholds(2, 1.0))
    with pytest.raises(ValueError, match="built on"):
        rc.tree_query(tree, np.zeros((4, 1), np.float32), tiny, thr)


def test_tree_is_safe_to_query_from_many_threads(rng):
    X = rng.normal(size=(50_000, 3)).astype(np.float32)
    Q = rng.normal(size=(2000, 3)).astype(np.float32)
    thr = thresholds(3, 0.2)
    tree = rc.build_tree(X)
    expected = _query(tree, X, Q, thr)
    results = [None] * 8

    def work(slot):
        results[slot] = _query(tree, X, Q, thr, threads=4)

    workers = [threading.Thread(target=work, args=(slot,)) for slot in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    for result in results:
        assert_same_csr(result, expected)


def test_tree_is_not_picklable(rng):
    tree = rc.build_tree(rng.normal(size=(10, 2)).astype(np.float32))
    with pytest.raises(TypeError, match="pickled"):
        pickle.dumps(tree)


@pytest.mark.parametrize("d", [2, 3, 5, 8, 13])
def test_float32_screens_and_box_tests_do_not_change_the_output(d, rng):
    # The float32 box tests and leaf pre-screen only skip work; every variant
    # (including the plain float64 path) gives bitwise-identical output.
    X = rng.normal(size=(20_000, d)).astype(np.float32)
    X[::50] = X[1]  # duplicates
    Q = np.vstack([X[:300], rng.normal(size=(700, d)).astype(np.float32)])
    native = rc._native
    tree = rc.build_tree(X)
    for R2 in (0.0, 0.05 * d, 0.5 * d, 2.0**130):
        thr = thresholds(d, R2)
        args = rc.threshold_tuple(thr)
        for base in (rc.NEED_DISTANCE, 0, rc.NEED_DISTANCE | rc.SORT):
            outputs = []
            for extra in (0, rc._NO_SCREEN, 1 << 22, 1 << 23):
                result = rc.NativeResult(native.tree_query(tree._capsule, Q, args, 0, False,
                                                           base | extra, ALL), ALL)
                outputs.append(result.to_csr())
            for other in outputs[1:]:
                assert_same_csr(other, outputs[0], distances=bool(base))
        if R2 == 0.5 * d:
            assert_same_csr(outputs[0], sort_rows(oracle(X, Q, R2)))


def test_dense_tree_results_stay_within_the_memory_budget(rng):
    # One-pass results hold int32 columns and float64 squared distances with
    # bounded slack (<= 16 B per pair); the CSR copy adds 16 B per pair, so a
    # dense result and its CSR stay within 32 B per pair. Two-pass mode writes
    # the CSR directly.
    import tracemalloc

    X = rng.random((60_000, 3), dtype=np.float32)
    thr = thresholds(3, 0.06**2)
    tree = rc.build_tree(X)
    result = rc.tree_query(tree, X, X, thr, exclude_self=True)
    pairs = result.nnz
    assert pairs > 3_000_000
    rows = len(X)
    assert result.memory_bytes() <= 16 * pairs + 24 * rows + (1 << 20)
    tracemalloc.start()
    csr = result.to_csr()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= 16 * pairs + 8 * (rows + 1) + (1 << 20)
    tracemalloc.start()
    two = rc.tree_query_csr(tree, X, X, thr, exclude_self=True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= 16 * pairs + 24 * (rows + 1) + (1 << 20)
    assert_same_csr(two, csr)
