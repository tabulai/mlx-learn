"""Distribution routing preserves original neighbors, fit ownership and queries."""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.config import tuning_context
from mlxlearn._kernels import distance
from mlxlearn.neighbors import NearestNeighbors
from mlxlearn_device import _neighbors as neighbors
from mlxlearn_device import _neighbors_compact as compact
from mlxlearn_device import _neighbors_fused as fused
from mlxlearn_device import _neighbors_general as general
from mlxlearn_device import _neighbors_integer as integer
from mlxlearn_device import fit_neighbors

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]


def _fixture(kind, features=32, nq=65, seed=872):
    rng = np.random.default_rng(seed)
    n = 32768 if features == 32 else 4096
    train = rng.normal(size=(n, features))
    queries = rng.normal(size=(nq, features))
    if kind == "clustered":
        centers = rng.normal(size=(4, features)) * 8
        train = centers[rng.integers(0, 4, n)] + 0.2 * train
        queries = centers[rng.integers(0, 4, nq)] + 0.2 * queries
    elif kind == "offset":
        offset = np.full(features, 10_000)
        offset[::2] *= -1
        train += offset
        queries += offset
    elif kind == "anisotropic":
        scales = np.logspace(-2, 2, features)
        train *= scales
        queries *= scales
    train, queries = train.astype(np.float32), queries.astype(np.float32)
    queries[:2] = train[:2]
    return train, queries


def _reference(train, queries, k=5):
    train64 = train.astype(np.float64)
    ids = np.arange(len(train))
    distances, indices = [], []
    for query in queries.astype(np.float64):
        delta = train64 - query
        squared = np.sum(delta * delta, axis=1)
        order = np.lexsort((ids, squared))[:k]
        distances.append(np.sqrt(squared[order]))
        indices.append(order)
    return np.asarray(distances), np.asarray(indices)


def _check(actual, train, queries):
    expected_distances, expected_indices = _reference(train, queries)
    np.testing.assert_array_equal(actual[1], expected_indices)
    np.testing.assert_allclose(actual[0], expected_distances, atol=1e-6, rtol=1e-6)


@pytest.fixture
def supported():
    if not integer.is_supported():
        pytest.skip("requires integer tensor operations")


@pytest.fixture
def routes(monkeypatch):
    calls = []
    for name, module, attr in (
        ("direct", fused, "fused_neighbors"),
        ("compact", compact, "compact_neighbors"),
        ("integer", integer, "integer_neighbors"),
    ):
        original = getattr(module, attr)

        def observe(*args, name=name, original=original, **kwargs):
            calls.append((name, kwargs.get("center")))
            return original(*args, **kwargs)

        monkeypatch.setattr(module, attr, observe)
    return calls


@pytest.mark.parametrize("kind,features,expected", [
    ("gaussian", 32, "integer"),
    ("anisotropic", 32, "compact"),
    ("offset", 32, "integer"),
    ("clustered", 32, "direct"),
    ("clustered", 128, "direct"),
])
def test_distribution_routes_match_exhaustive_original_coordinate_neighbors(
    supported, routes, kind, features, expected,
):
    train, queries = _fixture(kind, features)
    index = fit_neighbors(mx.array(train))
    actual = index.query(mx.array(queries), 5)
    mx.eval(*actual)
    _check(actual, train, queries)
    assert routes and all(route == expected for route, _ in routes)
    if kind == "offset":
        assert all(center is not None for _, center in routes)
    elif expected == "integer":
        assert all(center is None for _, center in routes)


def test_geometry_reads_owned_fit_once_without_caching_or_modifying_queries(supported, monkeypatch, routes):
    train, first = _fixture("offset")
    second = first[:64].copy()
    second[:, 0] += np.float32(0.25)
    source = mx.array(train.copy())
    index = fit_neighbors(source)
    source[:] = 0
    observed = []
    original = neighbors._training_metadata

    def observe(values):
        observed.append(values)
        return original(values)

    monkeypatch.setattr(neighbors, "_training_metadata", observe)
    q1, q2 = mx.array(first.copy()), mx.array(second.copy())
    a = index.query(q1, 5)
    b = index.query(q2, 5)
    mx.eval(*b, *a)
    _check(b, train, second)
    _check(a, train, first)
    np.testing.assert_array_equal(np.asarray(q1).view(np.uint32), first.view(np.uint32))
    np.testing.assert_array_equal(np.asarray(q2).view(np.uint32), second.view(np.uint32))
    assert len(observed) == 1 and observed[0] is index.X
    assert index._geometry == "offset"
    assert all(route == "integer" and center is not None for route, center in routes)
    omitted, indices = index.query(q2, 5, return_distance=False)
    assert omitted is None
    np.testing.assert_array_equal(indices, b[1])
    assert len(observed) == 1


def test_public_refit_replaces_geometry_and_centered_preparation(supported):
    with tuning_context(force_backend="mlx"):
        model = NearestNeighbors(n_neighbors=5, algorithm="brute")
        previous = None
        for kind in ("offset", "gaussian", "clustered"):
            train, queries = _fixture(kind)
            model.fit(train)
            index = model._index_.device_index
            assert index is not previous
            assert index._geometry is None and index._integer_center is None
            actual = model.kneighbors(queries)
            _check(actual, train, queries)
            assert index._geometry == ("ordinary" if kind == "gaussian" else kind)
            if kind == "offset":
                assert index._integer_center is not None
            else:
                assert index._integer_center is None
            previous = index


def test_public_128_cluster_query_uses_direct_route_despite_automatic_memory_plan(routes):
    train, queries = _fixture("clustered", features=128)
    with tuning_context(force_backend="mlx"):
        model = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(train)
        actual = model.kneighbors(queries)
    _check(actual, train, queries)
    assert routes and all(route == "direct" for route, _ in routes)
    assert model._index_.device_index._geometry == "clustered"


def test_public_ordinary_128_query_retains_automatic_general_memory_plan(monkeypatch, routes):
    train, queries = _fixture("gaussian", features=128)
    # An explicit query height makes this automatic byte budget require
    # reference tiling. It must reach the general path unchanged.
    monkeypatch.setattr(distance, "_distance_budget_bytes", lambda backend: len(queries) * 512 * 16)
    calls = []
    original = general.gpu_neighbors

    def observe(*args, **kwargs):
        calls.append((kwargs["rows"], kwargs["cols"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(general, "gpu_neighbors", observe)
    plan = distance.resolve_blocks(
        len(queries), len(train), backend="mlx", k_search=5 + distance._OVERSELECT,
        query_block_size=len(queries),
    )
    assert plan.train_cols == 512
    index = distance.build_index(train, backend="mlx")
    actual = index.query(queries, 5, query_block_size=len(queries))
    _check(actual, train, queries)
    assert calls == [(plan.query_rows, plan.train_cols)]
    assert not routes
    assert index.device_index._geometry == "ordinary"


@pytest.mark.parametrize("features,train_columns", [(32, 8192), (128, 1024)])
def test_explicit_public_train_tiling_keeps_general_search_and_bypasses_geometry(
    monkeypatch, routes, features, train_columns,
):
    train, queries = _fixture("clustered", features=features)
    calls = []
    original = general.gpu_neighbors

    def observe(*args, **kwargs):
        calls.append((kwargs["rows"], kwargs["cols"]))
        return original(*args, **kwargs)

    def forbidden(values):
        raise AssertionError("explicit training tiles must bypass distribution sampling")

    monkeypatch.setattr(general, "gpu_neighbors", observe)
    monkeypatch.setattr(neighbors, "_sample_geometry", forbidden)
    monkeypatch.setattr(neighbors, "_training_metadata", forbidden)
    index = distance.build_index(train, backend="mlx")
    actual = index.query(
        queries, 5, query_block_size=len(queries), train_block_size=train_columns,
    )
    _check(actual, train, queries)
    assert calls == [(len(queries), train_columns)]
    assert not routes
    assert index.device_index._geometry is None
