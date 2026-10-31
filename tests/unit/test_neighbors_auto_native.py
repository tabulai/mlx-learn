"""Owned native feature permutations preserve the original Euclidean problem."""

from __future__ import annotations

import numpy as np

from mlxlearn._kernels import neighbors_cpu
from mlxlearn.neighbors import KNeighborsClassifier, NearestNeighbors
from tests.unit.test_neighbors_cpu_exact import exact_oracle


def _fixture():
    rng = np.random.default_rng(843)
    train = rng.integers(-4, 5, size=(256, 16)).astype(np.float32) / 16
    train[:, 8:] *= 1024
    train[:12] = 0
    train[:8, 0] = 1
    train[:8, 1] = [2**-90, 2**-100, 0, 2**-100, 2**-80, 0, 2**-110, 2**-100]
    train[8:10, 0] = [-1, 1]
    queries = np.zeros((16, 16), np.float32)
    queries[8:] = train[20:28]
    return train, queries


def _check(result, train, queries, k, *, exclude_self=False):
    distances, indices = result
    expected = exact_oracle(train, queries, k, exclude_self=exclude_self)
    np.testing.assert_array_equal(indices, expected)
    delta = train[expected].astype(np.float64) - queries.astype(np.float64)[:, None]
    np.testing.assert_allclose(distances, np.sqrt(np.sum(delta * delta, axis=2)), rtol=1e-13, atol=0)


def test_native_permutation_preserves_exact_original_coordinate_order_and_owned_source():
    train, queries = _fixture()
    before_train, before_queries = train.view(np.uint32).copy(), queries.view(np.uint32).copy()
    index = neighbors_cpu.CPUIndex(train)
    _check(index.query(queries, 7), train, queries, 7)
    assert index.X is train and index._native_train is not None
    assert np.any(index._native_order[:8] >= 8)
    assert not np.shares_memory(index._native_train, train)
    np.testing.assert_array_equal(index._native_train, train[:, index._native_order])
    np.testing.assert_array_equal(train.view(np.uint32), before_train)
    np.testing.assert_array_equal(queries.view(np.uint32), before_queries)


def test_native_permutation_keeps_self_identity_across_query_tile_tails():
    train, _ = _fixture()
    index = neighbors_cpu.CPUIndex(train)
    tiled = index.query(train, 5, exclude_self=True, query_block_size=17)
    _check(tiled, train, train, 5, exclude_self=True)
    assert index.X is train and index._native_train is not None
    assert not np.any(tiled[1] == np.arange(len(train))[:, None])
    np.testing.assert_array_equal(index.query(train, 5, exclude_self=True)[1], tiled[1])


def test_native_permutation_is_deferred_for_single_queries_then_reused(monkeypatch):
    train, queries = _fixture()
    index = neighbors_cpu.CPUIndex(train)
    original = neighbors_cpu._native_query
    assert original is not None
    calls = []

    def observe(X, Q, *args):
        calls.append((X, Q.copy()))
        return original(X, Q, *args)

    monkeypatch.setattr(neighbors_cpu, "_native_query", observe)
    monkeypatch.setattr(neighbors_cpu, "_native_query_concentrated", observe)
    _check(index.query(queries[:1], 5), train, queries[:1], 5)
    assert index._native_order is None and index._native_train is None
    assert calls[-1][0] is train
    _check(index.query(queries, 5), train, queries, 5)
    prepared, order = index._native_train, index._native_order
    assert prepared is not None and calls[-1][0] is prepared
    np.testing.assert_array_equal(calls[-1][1], queries[:, order])
    _check(index.query(queries[-1:], 5), train, queries[-1:], 5)
    assert index._native_train is prepared and index._native_order is order
    assert index.X is train and calls[-1][0] is prepared
    np.testing.assert_array_equal(calls[-1][1], queries[-1:, order])


def test_native_query_keeps_its_matching_snapshot_when_another_query_publishes_permutation(monkeypatch):
    train, queries = _fixture()
    query = queries[-1:].copy()  # Non-symmetric columns expose a mismatched permutation.
    index = neighbors_cpu.CPUIndex(train)
    original_prepare = index._prepare_native_train
    original_query = neighbors_cpu._native_query
    assert original_query is not None
    calls = []

    def publish_after_snapshot(query_count):
        snapshot = original_prepare(query_count)
        assert snapshot[0] is train and snapshot[1] is None
        prepared, order = original_prepare(16)
        assert prepared is index._native_train and order is index._native_order
        assert prepared is not train and np.any(order[:8] >= 8)
        # Simulate another thread finishing its batch preparation between this
        # query obtaining its original-coordinate snapshot and launching search.
        return snapshot

    def observe(X, Q, *args):
        calls.append(True)
        assert X is train
        np.testing.assert_array_equal(Q, query)
        return original_query(X, Q, *args)

    monkeypatch.setattr(index, "_prepare_native_train", publish_after_snapshot)
    monkeypatch.setattr(neighbors_cpu, "_native_query", observe)
    _check(index.query(query, 5), train, query, 5)
    assert calls == [True] and index._native_train is not None


def test_public_auto_neighbors_accept_unaligned_inputs_and_readonly_queries(monkeypatch):
    from mlxlearn._common import base

    monkeypatch.setattr(base, "resolve_device", lambda requested: "cpu")
    expected_train = np.arange(256, dtype=np.float32).reshape(64, 4) / 16
    train = np.ndarray(expected_train.shape, np.float32,
                       buffer=bytearray(expected_train.nbytes + 1), offset=1)
    train[:] = expected_train
    expected_queries = expected_train[3:6] + np.float32(0.125)
    queries = np.ndarray(expected_queries.shape, np.float32,
                         buffer=bytearray(expected_queries.nbytes + 1), offset=1)
    queries[:] = expected_queries
    readonly = expected_queries.copy()
    readonly.setflags(write=False)
    assert train.flags.c_contiguous and not train.flags.aligned
    assert queries.flags.c_contiguous and not queries.flags.aligned
    model = NearestNeighbors(n_neighbors=3).fit(train)
    for query in (queries, readonly):
        _check(model.kneighbors(query), expected_train, expected_queries, 3)
        assert model.last_query_backend_ == "cpu"
    np.testing.assert_array_equal(train, expected_train)
    np.testing.assert_array_equal(queries, expected_queries)
    assert not readonly.flags.writeable


def test_native_equal_distance_class_votes_do_not_depend_on_float64_accumulation_order(monkeypatch):
    from mlxlearn._common import base

    monkeypatch.setattr(base, "resolve_device", lambda requested: "cpu")
    train = np.zeros((2, 16), np.float32)
    train[:, 0] = 2**30
    train[0, [4, 8, 12, 1, 2, 3, 5]] = 8
    train[1, [1, 3, 5, 7, 9, 11, 13]] = 8
    queries = np.zeros((1, 16), np.float32)
    model = KNeighborsClassifier(n_neighbors=2, weights="distance").fit(train, ["z", "a"])
    np.testing.assert_array_equal(model.predict_proba(queries), [[0.5, 0.5]])
    np.testing.assert_array_equal(model.predict(queries), ["a"])
    assert model.last_query_backend_ == "cpu"
