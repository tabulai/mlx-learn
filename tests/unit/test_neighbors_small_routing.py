"""CPU-only dispatch checks protecting the bounded small-search fast path."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from mlxlearn_device import _neighbors as neighbors
from mlxlearn_device import _neighbors_general as general
from mlxlearn_device import _neighbors_small as small


@pytest.mark.parametrize("n,rows,features,k,expected", [
    (2048, 256, 64, 8, True),
    (2049, 256, 64, 8, False),
    (2048, 257, 64, 8, False),
    (2048, 256, 65, 8, False),
    (2048, 256, 64, 9, False),
    (16384, 4, 32, 8, True),
    (16385, 4, 32, 8, False),
    (16384, 5, 32, 8, False),
    (16384, 4, 33, 8, False),
    (16384, 4, 32, 9, False),
    # Established Gaussian, clustered, offset and unequal-scale batch shapes
    # keep their existing search routes, regardless of values or distribution.
    (50000, 256, 32, 5, False),
    (4096, 128, 128, 5, False),
    (8192, 256, 32, 5, False),
    (2048, 32, 1024, 5, False),
])
def test_small_search_scope_preserves_batch_routes(n, rows, features, k, expected):
    assert neighbors._use_small_search(n, rows, features, k) is expected


@pytest.mark.parametrize("device,train_tile,automatic,expected", [
    ("gpu", None, False, "small"),
    ("gpu", 33, False, "general_gpu"),
    ("gpu", 33, True, "small"),
    ("cpu", None, False, "general_cpu"),
    ("cpu", 33, True, "general_cpu"),
])
def test_explicit_training_tiles_and_cpu_override_small_route(
    monkeypatch, device, train_tile, automatic, expected,
):
    # Fake matrices and a no-op stream exercise Python dispatch without
    # constructing, evaluating or launching any MLX array or GPU kernel.
    train, query = SimpleNamespace(shape=(120, 4)), SimpleNamespace(shape=(3, 4))
    index = neighbors.DeviceNeighborIndex(train, device=device)
    distances, indices = object(), object()
    calls = []

    def record(route):
        def operation(*args, **kwargs):
            calls.append((route, args, kwargs))
            return distances, indices
        return operation

    monkeypatch.setattr(neighbors, "_matrix", lambda *args, **kwargs: None)
    monkeypatch.setattr(neighbors.mx, "stream", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(small, "small_neighbors", record("small"))
    monkeypatch.setattr(general, "gpu_neighbors", record("general_gpu"))
    monkeypatch.setattr(general, "cpu_neighbors", record("general_cpu"))
    result = index.query(
        query, 5, train_block_size=train_tile, _auto_train_tiling=automatic,
        _validated=True, _squared=True,
    )
    assert result == (distances, indices)
    assert len(calls) == 1 and calls[0][0] == expected
    if expected == "small":
        assert calls[0][1] == (train, query, 5)
        assert calls[0][2]["partitions"] == 4
    else:
        assert calls[0][2]["cols"] == (train_tile or train.shape[0])
