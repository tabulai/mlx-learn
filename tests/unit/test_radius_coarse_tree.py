"""Sampled routing to exact, coarse cells in high-dimensional radius searches."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mlxlearn._kernels import radius_cpu
from mlxlearn._kernels.distance import build_index

pytestmark = pytest.mark.skipif(not radius_cpu.AVAILABLE, reason="requires the native radius tree")


def _clustered():
    rng = np.random.default_rng(837)
    centers = rng.normal(size=(64, 32)).astype(np.float32) * 20
    centers[0] = 0
    train = centers[np.arange(8192) % len(centers)] + rng.normal(
        0, 0.02, (8192, 32)).astype(np.float32)
    # A misplaced row, two ties, and one exterior point whose squared distance
    # rounds to the cutoff in float64. The pilot may miss any of these rows.
    train[:4] = 0
    train[1:4, 0] = [1, -1, 1]
    train[3, 1] = 2.0**-32
    return train, centers[np.arange(256) % len(centers)]


def test_coarse_tree_is_exact_and_independent_of_forced_tree():
    train, queries = _clustered()
    search = build_index(train, backend="cpu", _auto=True)._radius()

    def query(route):
        parts, plan = search.query_with_plan(
            queries, 1.0, need_distance=True, sort_results=True, exclude_self=False,
            fixed="cpu", forced=route, n_jobs=2)
        return parts[0], plan

    # Concurrent first queries can request trees with different leaf sizes.
    with ThreadPoolExecutor(max_workers=2) as pool:
        (coarse, plan), (fine, forced) = pool.map(query, ["auto", "tree"])
    assert plan.route == forced.route == "tree"
    assert plan.leaf_size == 128 and forced.leaf_size == 0
    assert search._trees[128].leaf_size == 128 and search._trees[0].leaf_size == 32
    np.testing.assert_array_equal(coarse.indptr, fine.indptr)
    np.testing.assert_array_equal(coarse.indices, fine.indices)
    np.testing.assert_array_equal(coarse.distances, fine.distances)
    first = coarse.indices[:coarse.indptr[1]]
    assert all(i in first for i in (0, 1, 2)) and 3 not in first

    # Sub-batches of training rows use the global row offset to remove self.
    kwargs = {"need_distance": False, "sort_results": False, "exclude_self": True,
                  "q_offset": 17, "fixed": "cpu", "n_jobs": 2}
    qs = train[17:273]
    (coarse,), plan = search.query_with_plan(qs, 1.0, **kwargs)
    (direct,), _ = search.query_with_plan(qs, 1.0, forced="brute", **kwargs)
    assert plan.leaf_size == 128
    np.testing.assert_array_equal(coarse.indptr, direct.indptr)
    np.testing.assert_array_equal(coarse.indices, direct.indices)


def test_coarse_pilot_rejects_diffuse_data_and_honors_explicit_engines():
    rng = np.random.default_rng(5463)
    train = rng.normal(size=(8192, 32)).astype(np.float32)
    query = rng.normal(size=(256, 32)).astype(np.float32)
    search = build_index(train, backend="cpu", _auto=True)._radius()
    opts = {"algorithm": "auto", "forced": "auto", "exclude_self": False, "n_jobs": 1, "patched": False}
    plan = search.plan(query, 5.0, fixed="cpu", has_gpu=lambda: False, **opts)
    assert plan.route == "brute" and search._pilot_tree is not None and not search._trees

    train, query = _clustered()
    search = build_index(train, backend="cpu", _auto=True)._radius()
    for changes, expected in [({"fixed": "mlx"}, "metal"), ({"algorithm": "brute"}, "brute"),
                              ({"forced": "brute"}, "brute")]:
        options = {"fixed": "cpu", **opts, **changes}
        assert search.plan(query, 1.0, has_gpu=lambda: True, **options).route == expected
    assert search._pilot_tree is None and not search._trees
