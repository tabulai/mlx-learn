"""Gram screening keeps binary32 products where MLX may substitute TF32.

On M5 (GPU generation 17) MLX runs float32 ``matmul`` with TF32 operands by
default. Gram lower bounds then exceeded exact distances by up to ~8e8 on the
monotone fixture, and the general path certified wrong neighbors. Forcing the
binary32 product on earlier GPUs keeps that route covered everywhere. The
kernel itself is tested in ``test_binary32.py``.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._kernels.distance import build_index
from mlxlearn_device import _binary32 as binary32
from mlxlearn_device import _neighbors as neighbors
from mlxlearn_device import _neighbors_general as general


def _monotone():
    train = np.arange(8192 * 32, dtype=np.float32).reshape(8192, 32)
    return train, train[[0, 3, 4, 4096, 8191]].copy()


def _clustered(features):
    rng = np.random.default_rng(872)
    n = 32768 if features == 32 else 4096
    centers = rng.normal(size=(4, features)) * 8
    train = centers[rng.integers(0, 4, n)] + 0.2 * rng.normal(size=(n, features))
    queries = centers[rng.integers(0, 4, 65)] + 0.2 * rng.normal(size=(65, features))
    train, queries = train.astype(np.float32), queries.astype(np.float32)
    queries[:2] = train[:2]
    return train, queries


FIXTURES = {
    "monotone": (_monotone, 8192),
    "clustered32": (lambda: _clustered(32), 8192),
    "clustered128": (lambda: _clustered(128), 1024),
}


@pytest.mark.gpu
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("exact_products", [None, True], ids=["dispatch", "binary32"])
@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_gram_lower_bounds_never_exceed_exact_distances(fixture, exact_products):
    make, tile = FIXTURES[fixture]
    train, queries = make()
    center, centered, train_sq = neighbors._prepare_gram_arrays(mx.array(train))
    features = train.shape[1]
    error_factor = (2 * features + 16) * 2.0**-23
    q64, t64 = queries.astype(np.float64), train.astype(np.float64)
    for offset in range(0, len(train), tile):
        stop = min(offset + tile, len(train))
        lower = general.gram_lower(
            mx.array(queries), centered[offset:stop], center, train_sq[offset:stop],
            error_factor, features * 1e-30, exact_products=exact_products,
        )
        delta = q64[:, None, :] - t64[None, offset:stop, :]
        exact = np.sum(delta * delta, axis=2)
        excess = np.array(lower).astype(np.float64) - exact
        assert excess.max() <= 0, f"lower bound exceeds exact distance by {excess.max()}"


def _oracle(train, queries, k):
    train64, ids = train.astype(np.float64), np.arange(len(train))
    indices = []
    for query in queries.astype(np.float64):
        delta = train64 - query
        indices.append(np.lexsort((ids, np.sum(delta * delta, axis=1)))[:k])
    return np.asarray(indices)


@pytest.mark.gpu
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_general_path_takes_binary32_products_where_matmul_may_round(monkeypatch, fixture):
    make, tile = FIXTURES[fixture]
    train, queries = make()
    calls = []
    original = binary32.float32_matmul

    def counted(a, b, **kwargs):
        calls.append(None)  # count only: retaining arrays would inflate peak memory
        return original(a, b, **kwargs)

    monkeypatch.setattr(binary32, "matmul_may_round_float32", lambda: True)
    monkeypatch.setattr(binary32, "float32_matmul", counted)
    _, indices = build_index(train, backend="mlx").query(
        queries, 5, query_block_size=len(queries), train_block_size=tile,
    )
    np.testing.assert_array_equal(indices, _oracle(train, queries, 5))
    assert len(calls) == -(-len(train) // tile)
