"""Exact float32 integer Gram selection, ordering and routing boundaries."""

import math

import numpy as np
import pytest

from mlxlearn._kernels import radius_cpu
from mlxlearn._kernels.distance import build_index
from mlxlearn._kernels.radius_result import concatenate

pytestmark = pytest.mark.skipif(not radius_cpu.AVAILABLE, reason="requires native radius selection")


@pytest.mark.parametrize("radius", [np.nextafter(3.0, 0.0), 3.0, math.sqrt(2**20)])
@pytest.mark.parametrize("ordered", [False, True])
def test_integer_gram_matches_integer_oracle_across_tiles(radius, ordered, monkeypatch):
    monkeypatch.setattr(radius_cpu, "_INTEGER_GRAM_TILE_ELEMENTS", 1 << 17)
    rng = np.random.default_rng(41)
    # The last feature attains the binary32 proof limit, so nearby rows
    # subtract large dot products. Repetitions produce many tied distances.
    base = rng.integers(-1, 2, (512, 16), dtype=np.int32).astype(np.float32)
    X = np.tile(base, (2, 1))
    X[:, -1] = 512
    Q = X[:257].copy()
    model = build_index(X, backend="cpu", _auto=True)
    parts = model.query_radius_csr(Q, radius, need_distance=ordered, sort_results=ordered)
    assert len(parts) >= 2 and model._radius_search.last_plan.route == "integer-gram"
    result = concatenate(parts)
    for i in (0, 19, 255, 256):
        squared = np.sum((Q[i].astype(np.int64) - X.astype(np.int64))**2, axis=1)
        expected = np.flatnonzero(squared <= radius * radius)
        if ordered:
            expected = expected[np.argsort(squared[expected], kind="stable")]
        row = slice(result.indptr[i], result.indptr[i + 1])
        np.testing.assert_array_equal(result.indices[row], expected)
        if ordered:
            np.testing.assert_array_equal(result.distances[row], np.sqrt(squared[expected]))


def test_self_exclusion_uses_global_offsets_and_preserves_duplicates():
    X = np.tile(np.arange(16, dtype=np.float32), (513, 1))
    index = build_index(X, backend="cpu", _auto=True)
    parts = index.query_radius_csr(X, 1.0, exclude_self=True, query_block_size=171)
    assert index._radius_search.last_plan.route == "integer-gram"
    result = concatenate(parts)
    for i in (0, 170, 171, 342, 512):
        lo, hi = result.indptr[i:i + 2]
        ids = np.arange(len(X))
        np.testing.assert_array_equal(result.indices[lo:hi], ids[ids != i])


@pytest.mark.parametrize("exception", [513.0, -0.5, np.float32(2.0**-149)])
@pytest.mark.parametrize("training", [True, False])
def test_every_coordinate_is_certified(exception, training):
    X = np.zeros((2048, 16), np.float32)
    Q = np.zeros((32, 16), np.float32)
    (X if training else Q)[-1, -1] = exception
    index = build_index(X, backend="cpu", _auto=True)
    search = index._radius()
    plan = search.plan(Q, 1.0, exclude_self=False, algorithm="auto", n_jobs=1,
                       patched=False, fixed="cpu", has_gpu=lambda: False, forced="auto")
    assert plan.route != "integer-gram"


def test_large_integer_batch_respects_available_matrix_backend():
    train = np.ones((32768, 128), np.float32)
    queries = train[:257]
    search = build_index(train, backend="cpu", _auto=True)._radius()
    options = {"exclude_self": False, "algorithm": "brute", "n_jobs": 1, "patched": False, "forced": "auto"}
    assert search.plan(queries, 1.0, fixed=None, has_gpu=lambda: True, **options).route == "metal"
    assert search.plan(queries, 1.0, fixed=None, has_gpu=lambda: False, **options).route == "integer-gram"
    assert search.plan(queries, 1.0, fixed="cpu", has_gpu=lambda: True, **options).route == "integer-gram"


def test_radius_immediately_below_largest_certified_squared_distance():
    X = np.full((4096, 16), 512.0, dtype=np.float32)
    X[::2] *= -1
    Q = X[1:2].repeat(16, axis=0)
    index = build_index(X, backend="cpu", _auto=True)
    for radius, expected in [(4096.0, np.arange(len(X))),
                             (math.nextafter(4096.0, 0.0), np.arange(1, len(X), 2))]:
        parts = index.query_radius_csr(Q, radius, need_distance=False)
        assert index._radius_search.last_plan.route == "integer-gram"
        row = concatenate(parts)
        np.testing.assert_array_equal(row.indices[:row.indptr[1]], expected)
