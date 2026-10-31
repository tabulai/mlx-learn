"""The faster conservative cutoff retains exact membership and boundary ties."""

import numpy as np
import pytest

from mlxlearn._kernels import _neighbors_cpu_native as native
from mlxlearn._kernels import neighbors_cpu
from tests.unit.test_neighbors_cpu_exact import exact_oracle


def _run(train, queries, k, *, exclude_self=False, offset=0):
    distances = np.empty((len(queries), k), np.float64)
    indices = np.empty((len(queries), k), np.int64)
    native.query_concentrated(train, queries, k, exclude_self, offset, distances, indices)
    expected = exact_oracle(train, queries, len(train))
    if exclude_self:
        expected = np.stack([row[row != offset + i][:k] for i, row in enumerate(expected)])
    else:
        expected = expected[:, :k]
    np.testing.assert_array_equal(indices, expected)
    delta = train[expected].astype(np.float64) - queries.astype(np.float64)[:, None]
    np.testing.assert_allclose(distances, np.sqrt(np.sum(delta * delta, axis=2)), rtol=1e-13, atol=0)
    return distances, indices


@pytest.mark.parametrize("features", [1, 8, 16, 32, 64, 65])
@pytest.mark.parametrize("scale", [2**-32, 1.0, 2**40])
def test_concentrated_screen_boundaries_ties_and_late_winners(features, scale):
    train = np.zeros((37, features), np.float32)
    train[:, 0] = scale
    train[::3, min(1, features - 1)] = -scale
    train[::5, -1] = scale
    train[-1] = 0
    queries = np.zeros((3, features), np.float32)
    queries[1, 0], queries[2, -1] = -scale, scale
    _run(train, queries, 5)


@pytest.mark.parametrize("tail", [2**-15, 2**-149, np.finfo(np.float32).max])
def test_concentrated_screen_keeps_rounded_and_true_ties_with_exact_fallback(tail):
    train = np.zeros((33, 32), np.float32)
    train[:, 0] = 2
    train[:3, 0] = [1, 1, -1]
    train[:3, -1] = [tail, tail / 2, tail / 2]
    train[-1, 0] = np.nextafter(np.float32(1), np.float32(0))
    queries = np.zeros((1, 32), np.float32)
    _run(train, queries, 3)


def test_concentrated_screen_self_tail_and_all_neighbors():
    train = np.zeros((33, 32), np.float32)
    train[:, 0] = np.arange(33) % 5
    train[:, 16] = np.arange(33) % 3
    _run(train, train[29:].copy(), len(train) - 1, exclude_self=True, offset=29)


def test_cpu_concentrated_query_uses_new_optional_entry_and_older_binary_fallback(monkeypatch):
    rng = np.random.default_rng(10631)
    train = rng.integers(-4, 5, size=(256, 16)).astype(np.float32) / 16
    train[:, 8:] *= 1024
    queries = train[:16].copy()
    calls = []
    original = neighbors_cpu._native_query_concentrated
    assert original is not None

    def record(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(neighbors_cpu, "_native_query_concentrated", record)
    result = neighbors_cpu.CPUIndex(train).query(queries, 5)
    assert calls
    monkeypatch.setattr(neighbors_cpu, "_native_query_concentrated", None)
    fallback = neighbors_cpu.CPUIndex(train).query(queries, 5)
    np.testing.assert_array_equal(result[1], fallback[1])
    np.testing.assert_allclose(result[0], fallback[0], rtol=1e-13, atol=0)
