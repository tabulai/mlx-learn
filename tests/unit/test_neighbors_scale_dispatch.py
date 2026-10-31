"""Scale-aware dispatch keeps original-coordinate neighbors and owned fit state."""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.config import tuning_context
from mlxlearn.neighbors import NearestNeighbors
from mlxlearn_device import _neighbors as neighbors
from mlxlearn_device import _neighbors_compact as compact
from mlxlearn_device import _neighbors_fused as fused
from mlxlearn_device import _neighbors_integer as integer
from mlxlearn_device import fit_neighbors

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]


def fixture_arrays(kind, *, n=32768, nq=65, seed=947):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 32)).astype(np.float32)
    Q = rng.normal(size=(nq, 32)).astype(np.float32)
    if kind == "anisotropic":
        scales = np.logspace(-2, 2, 32).astype(np.float32)
        X *= scales
        Q *= scales
    if nq > 1:
        Q[:2] = X[:2]
    return X, Q


def assert_original_neighbors(index, X, Q, k=5, **options):
    """Compare against direct original-coordinate compensated scanning."""
    actual = index.query(Q, k, _squared=True, **options)
    expected = fused.fused_neighbors(X, Q, k)
    mx.eval(*actual, *expected)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)
    return actual


@pytest.fixture
def supported():
    if not integer.is_supported():
        pytest.skip("requires integer tensor operations")


@pytest.fixture
def executed_routes(monkeypatch):
    """Observe executed kernels instead of inferring routes from cached arrays."""
    calls = []
    for name, module, attr in (
        ("integer", integer, "integer_neighbors"),
        ("compact", compact, "compact_neighbors"),
    ):
        original = getattr(module, attr)

        def record(*args, name=name, original=original, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, attr, record)
    return calls


@pytest.mark.parametrize("kind,k", [
    ("gaussian", 5), ("anisotropic", 1), ("anisotropic", 5), ("anisotropic", 8),
])
def test_scale_dispatch_executes_expected_filter_without_changing_metric(supported, executed_routes, kind, k):
    host, queries = fixture_arrays(kind)
    X, Q = mx.array(host), mx.array(queries)
    index = fit_neighbors(X)
    actual = assert_original_neighbors(index, X, Q, k)
    expected_route = "integer" if kind == "gaussian" else "compact"
    assert executed_routes == [expected_route]
    assert index._integer_preferred is (kind == "gaussian")
    omitted, indices = index.query(Q, k, return_distance=False)
    assert omitted is None
    np.testing.assert_array_equal(indices, actual[1])
    assert executed_routes == [expected_route, expected_route]


@pytest.mark.parametrize("kind", ["gaussian", "anisotropic"])
@pytest.mark.parametrize("transform", ["permutation", "tiny", "large", "offset"])
def test_scale_preference_is_stable_under_coordinate_symmetries(kind, transform):
    X, _ = fixture_arrays(kind, n=257, nq=1)
    if transform == "permutation":
        X = X[:, np.random.default_rng(948).permutation(32)]
    elif transform == "tiny":
        X = X * np.float32(2.0 ** -60)
    elif transform == "large":
        X = X * np.float32(2.0 ** 60)
    elif transform == "offset":
        X = X + np.float32(10000)
    assert neighbors._prefer_integer_training(mx.array(X)) is (kind == "gaussian")


@pytest.mark.parametrize("kind", ["zero", "constant", "single_feature", "outlier"])
def test_degenerate_and_outlier_training_keeps_correct_neighbors(supported, executed_routes, kind):
    host, queries = fixture_arrays("gaussian")
    if kind == "zero":
        host[:] = 0
        queries[:] = 0
    elif kind == "constant":
        host[:] = 7
        queries[:] = 7
    elif kind == "single_feature":
        host[:, 1:] = 0
        queries[:, 1:] = 0
    else:
        host[0, 0] = np.float32(1e6)
        queries[0] = host[0]
    X, Q = mx.array(host), mx.array(queries)
    index = fit_neighbors(X)
    assert_original_neighbors(index, X, Q)
    assert index._integer_preferred is False
    assert executed_routes == ["compact"]


@pytest.mark.parametrize("layout", ["row_stride", "feature_stride", "transposed"])
def test_preference_reads_strided_buffers_as_their_logical_values(layout):
    rng = np.random.default_rng(949)
    expected = True
    if layout == "row_stride":
        values = rng.normal(size=(514, 32)).astype(np.float32)
        values[1::2, 0] *= np.float32(1e6)
        train = mx.array(values)[::2]
    elif layout == "feature_stride":
        values = rng.normal(size=(257, 64)).astype(np.float32)
        values[:, 1::2] *= np.float32(1e6)
        train = mx.array(values)[:, ::2]
    else:
        values = rng.normal(size=(32, 257)).astype(np.float32)
        values *= np.logspace(-2, 2, 32).astype(np.float32)[:, None]
        train = mx.array(values).T
        expected = False
    assert neighbors._prefer_integer_training(train) is expected
    scales = mx.array(np.logspace(-2, 2, 32).astype(np.float32))
    assert neighbors._prefer_integer_training(train * scales) is False


def test_owned_preference_is_cached_without_caching_queries_or_losing_tail(supported, monkeypatch, executed_routes):
    host, first = fixture_arrays("anisotropic", nq=65)
    original, source = mx.array(host.copy()), mx.array(host.copy())
    index = fit_neighbors(source)
    # The guard must inspect the fitted copy, even before its first eligible query.
    source[:] = mx.array(fixture_arrays("gaussian")[0])
    decisions = []
    metadata = neighbors._training_metadata

    def observe(train):
        decisions.append(train)
        return metadata(train)

    monkeypatch.setattr(neighbors, "_training_metadata", observe)
    first = mx.array(first)
    second = first + mx.array(np.linspace(-0.4, 0.4, 32).astype(np.float32))
    out_first = index.query(first, 5, query_block_size=64, _squared=True)
    out_second = index.query(second, 5, query_block_size=64, _squared=True)
    # Evaluate outstanding graphs in the opposite order to construction.
    mx.eval(*out_second, *out_first)
    for query, actual in ((first, out_first), (second, out_second)):
        expected = fused.fused_neighbors(original, query, 5)
        mx.eval(*expected)
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(value, reference)
    assert len(decisions) == 1 and decisions[0] is index.X
    assert index._integer_preferred is False
    assert executed_routes == ["compact", "compact"]
    assert not np.array_equal(np.asarray(out_first[0]), np.asarray(out_second[0]))


@pytest.mark.parametrize("first_kind", ["gaussian", "anisotropic"])
def test_public_refit_replaces_the_preference_and_filter(supported, first_kind):
    second_kind = "anisotropic" if first_kind == "gaussian" else "gaussian"
    with tuning_context(force_backend="mlx"):
        model = NearestNeighbors(n_neighbors=5, algorithm="brute")
        previous = None
        for kind in (first_kind, second_kind):
            X, Q = fixture_arrays(kind, nq=64)
            model.fit(X)
            index = model._index_.device_index
            assert index is not previous and index._integer_preferred is None
            actual_distances, actual_indices = model.kneighbors(Q)
            expected_sq, expected_indices = fused.fused_neighbors(mx.array(X), mx.array(Q), 5)
            mx.eval(expected_sq, expected_indices)
            expected_distances = np.sqrt(np.asarray(expected_sq, dtype=np.float64))
            np.testing.assert_array_equal(actual_indices, expected_indices)
            np.testing.assert_array_equal(actual_distances, expected_distances)
            assert index._integer_preferred is (kind == "gaussian")
            previous = index


@pytest.mark.parametrize("mode", ["short_queries", "small_blocks"])
def test_ineligible_query_never_reads_host_dispatch_sample(monkeypatch, mode):
    host, queries = fixture_arrays("gaussian", nq=63 if mode == "short_queries" else 65)
    X, Q = mx.array(host), mx.array(queries)
    index = fit_neighbors(X)

    def forbidden(train):
        raise AssertionError("ineligible query inspected host dispatch sample")

    monkeypatch.setattr(neighbors, "_prefer_integer_training", forbidden)
    monkeypatch.setattr(neighbors, "_training_metadata", forbidden)
    monkeypatch.setattr(neighbors, "_sample_geometry", forbidden)
    options = {"query_block_size": 32} if mode == "small_blocks" else {}
    assert_original_neighbors(index, X, Q, **options)
    assert index._integer_preferred is None


def test_unavailable_buffer_uses_compact_without_losing_correctness(supported, monkeypatch, executed_routes):
    host, queries = fixture_arrays("gaussian")
    X, Q = mx.array(host), mx.array(queries)

    def unavailable_buffer(value):
        raise BufferError("buffer interface unavailable")

    monkeypatch.setattr(neighbors, "memoryview", unavailable_buffer, raising=False)
    index = fit_neighbors(X)
    assert_original_neighbors(index, X, Q)
    assert index._integer_preferred is False
    assert executed_routes == ["compact"]
