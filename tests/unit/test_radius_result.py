"""CSR assembly, row sorting, reductions and router purity for radius search."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlxlearn._kernels.radius_reduce import reduce_mean_csr, reduce_votes_csr
from mlxlearn._kernels.radius_result import (
    RadiusCSR,
    concatenate,
    csr_to_objects,
    from_grouped,
    pairs_to_csr,
    sort_rows,
    to_ragged,
)
from mlxlearn._kernels.radius_router import (
    DensitySample,
    RouteContext,
    density_sample,
    plan_radius,
)


def _random_csr(rng, nq=50, n=200, max_len=30, ties=False):
    lengths = rng.integers(0, max_len, nq)
    cols = [np.sort(rng.choice(n, size=k, replace=False)) for k in lengths]
    if ties:
        dist = [rng.integers(0, 4, size=k).astype(np.float64) for k in lengths]
    else:
        dist = [rng.random(k) for k in lengths]
    cols = np.concatenate(cols) if nq else np.empty(0, dtype=np.int64)
    dist = np.concatenate(dist) if nq else np.empty(0)
    return from_grouped(lengths, cols, dist)


def test_objects_are_views_of_one_buffer_with_sklearn_dtypes(rng):
    result = _random_csr(rng)
    dist, ind = to_ragged([result], return_distance=True)
    assert ind.dtype == object and dist.dtype == object
    assert len(ind) == result.n_query
    for row in range(result.n_query):
        assert ind[row].dtype == np.int64 and dist[row].dtype == np.float64
        assert ind[row].ndim == 1
        if len(ind[row]):
            assert np.shares_memory(ind[row], result.indices)


def test_equal_length_rows_stay_one_dimensional():
    result = from_grouped(np.array([2, 2]), np.array([0, 1, 0, 1]), np.zeros(4))
    rows = csr_to_objects(result.indptr, result.indices)
    assert rows.shape == (2,) and rows[0].shape == (2,)


def test_empty_queries_and_rows():
    empty = to_ragged([], return_distance=True)
    assert empty[0].shape == (0,) and empty[1].shape == (0,)
    result = from_grouped(np.zeros(3, dtype=np.int64), np.empty(0, np.int64), np.empty(0))
    dist, ind = to_ragged([result], return_distance=True)
    assert all(r.shape == (0,) for r in ind)


@pytest.mark.parametrize("ties", [False, True])
def test_sort_rows_matches_lexsort(rng, ties):
    result = _random_csr(rng, ties=ties)
    ordered = sort_rows(result)
    rows = result.row_ids()
    reference = np.lexsort((result.indices, result.distances, rows))
    np.testing.assert_array_equal(ordered.indices, result.indices[reference])
    np.testing.assert_array_equal(ordered.distances, result.distances[reference])


def test_sort_rows_handles_float32_key_collisions():
    # Distinct float64 distances that share one float32 image must still be
    # ordered by their float64 values.
    base = 1.0
    close = np.nextafter(base, 2.0)
    result = from_grouped(np.array([3]), np.array([0, 1, 2]), np.array([close, base, close]))
    ordered = sort_rows(result)
    np.testing.assert_array_equal(ordered.indices, [1, 0, 2])


def test_pairs_to_csr_groups_and_orders(rng):
    rows = rng.integers(0, 10, 200)
    cols = rng.permutation(200) % 37
    keep = np.unique(rows * 1000 + cols, return_index=True)[1]
    rows, cols = rows[keep], cols[keep]
    values = rng.random(len(rows))
    shuffle = rng.permutation(len(rows))
    result = pairs_to_csr(rows[shuffle], cols[shuffle], values[shuffle], 10, 37)
    order = np.lexsort((cols, rows))
    np.testing.assert_array_equal(result.indices, cols[order])
    np.testing.assert_array_equal(result.distances, values[order])
    np.testing.assert_array_equal(np.diff(result.indptr), np.bincount(rows, minlength=10))


def test_concatenate_offsets(rng):
    parts = [_random_csr(rng, nq=7), _random_csr(rng, nq=5), _random_csr(rng, nq=0)]
    joined = concatenate(parts)
    assert joined.n_query == 12
    np.testing.assert_array_equal(joined.indices, np.concatenate([p.indices for p in parts]))
    assert joined.indptr[-1] == sum(p.nnz for p in parts)


def _ragged_votes(result, codes, n_classes, weighted):
    votes = np.zeros((result.n_query, n_classes))
    for row in range(result.n_query):
        start, stop = result.indptr[row], result.indptr[row + 1]
        cols = result.indices[start:stop]
        if weighted:
            d = result.distances[start:stop]
            w = (d == 0).astype(float) if np.any(d == 0) else 1.0 / d
            votes[row] = np.bincount(codes[cols], weights=w, minlength=n_classes)
        else:
            votes[row] = np.bincount(codes[cols], minlength=n_classes)
    return votes


@pytest.mark.parametrize("weighted", [False, True])
def test_vectorized_votes_match_the_per_row_loop(rng, weighted):
    result = _random_csr(rng, n=200)
    distances = result.distances.copy()
    distances[::7] = 0.0  # coincident points take all the weight of their rows
    result = RadiusCSR(result.indptr, result.indices, distances)
    codes = rng.integers(0, 4, 200)
    votes, counts = reduce_votes_csr(result, codes, 4, weighted=weighted)
    np.testing.assert_array_equal(votes, _ragged_votes(result, codes, 4, weighted))
    np.testing.assert_array_equal(counts, result.lengths())


@pytest.mark.parametrize("weighted", [False, True])
def test_vectorized_means_match_numpy(rng, weighted):
    result = _random_csr(rng, n=200)
    y = rng.normal(size=200)
    pred, counts = reduce_mean_csr(result, y, weighted=weighted)
    for row in range(result.n_query):
        start, stop = result.indptr[row], result.indptr[row + 1]
        values = y[result.indices[start:stop]]
        if stop == start:
            assert math.isnan(pred[row])
        elif weighted:
            w = 1.0 / result.distances[start:stop]
            np.testing.assert_allclose(pred[row], np.average(values, weights=w), rtol=1e-13)
        else:
            np.testing.assert_allclose(pred[row], values.mean(), rtol=1e-13)


def _context(**overrides):
    base = {"n": 100_000, "d": 16, "n_query": 1000, "R2": 1.0, "algorithm": "auto",
            "n_jobs": None, "patched": False, "gpu": True, "metal_ok": True, "native": False,
            "sample": DensitySample(10.0, 40.0, 1.0), "forced": "auto"}
    base.update(overrides)
    return RouteContext(**base)


@pytest.mark.parametrize("overrides", [
    {}, {"d": 3}, {"d": 12, "gpu": False}, {"n_query": 2}, {"R2": math.inf},
    {"patched": True, "sample": DensitySample(5000.0, 10000.0, 1.0)},
    {"algorithm": "brute", "d": 3},
])
def test_router_is_a_pure_function(overrides):
    ctx = _context(**overrides)
    plans = {plan_radius(ctx) for _ in range(5)}
    assert len(plans) == 1


def test_router_rules():
    assert plan_radius(_context(d=3)).route == "tree"
    assert plan_radius(_context(d=3, algorithm="brute")).route == "metal"
    assert plan_radius(_context(n_query=2)).route == "brute"
    assert plan_radius(_context(d=12, gpu=False)).route == "tree"
    assert plan_radius(_context(d=32, gpu=False)).route == "brute"
    dense = DensitySample(5000.0, 10000.0, 1.0)
    assert plan_radius(_context(patched=True, sample=dense)).route == "stock"
    assert plan_radius(_context(patched=False, sample=dense)).route == "metal"
    with pytest.raises(ValueError):
        plan_radius(_context(forced="stock"))
    with pytest.raises(ValueError):
        plan_radius(_context(forced="metal", gpu=False))
    assert plan_radius(_context(forced="metal", metal_ok=False)).route == "brute"


def test_density_sample_is_deterministic_and_skips_self_pairs(rng):
    from mlxlearn._kernels.radius_router import TrainSample

    X = rng.normal(size=(5000, 4)).astype(np.float32)
    sample = TrainSample.from_train(X, X.mean(axis=0, dtype=np.float64))
    first = density_sample(sample, X, 0.0, exclude_self=True, n=len(X))
    assert first.m_hat == 0.0
    second = density_sample(sample, X, 0.0, exclude_self=False, n=len(X))
    assert second.m_hat > 0.0
    assert density_sample(sample, X[:100], 1.0, exclude_self=False, n=len(X)) == density_sample(
        sample, X[:100], 1.0, exclude_self=False, n=len(X))


def test_sample_rows_cover_the_whole_training_set(rng):
    from mlxlearn._kernels.radius_router import TrainSample

    X = rng.normal(size=(6000, 3)).astype(np.float32)
    sample = TrainSample.from_train(X, np.zeros(3))
    assert sample.index[0] == 0 and sample.index[-1] == len(X) - 1


def test_non_selective_gram_screen_avoids_metal():
    flooded = DensitySample(10.0, 40.0, 1.0, c_hat=50_000.0)
    assert plan_radius(_context(d=12, sample=flooded)).route == "tree"
    assert plan_radius(_context(d=64, sample=flooded)).route == "brute"
    selective = DensitySample(10.0, 40.0, 1.0, c_hat=12.0)
    assert plan_radius(_context(d=64, sample=selective)).route == "metal"


def test_dense_outputs_on_small_screens_stay_on_stock_when_patched():
    dense = DensitySample(1500.0, 4000.0, 1.0, c_hat=1600.0)
    small = _context(patched=True, native=True, d=64, n=100_000, n_query=1000, sample=dense)
    assert plan_radius(small).route == "stock"
    bitmap = _context(patched=True, native=True, d=32, n=32_000, n_query=256, sample=dense)
    assert plan_radius(bitmap).route == "metal"
    large = _context(patched=True, native=True, d=64, n=1_000_000, n_query=1000, sample=dense)
    assert plan_radius(large).route == "metal"
    assert plan_radius(_context(patched=False, native=True, d=64, n=100_000, n_query=1000,
                                sample=dense)).route == "metal"


def test_patched_tree_plans_without_a_gpu_stay_owned():
    for native in (True, False):
        ctx = _context(patched=True, gpu=False, native=native, d=12, n=60_000, n_query=60_000)
        assert plan_radius(ctx).route == "tree"
