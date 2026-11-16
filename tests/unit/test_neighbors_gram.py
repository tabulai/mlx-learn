"""Gram screening keeps binary32 products where MLX may substitute TF32.

On M5 (GPU generation 17) MLX runs float32 ``matmul`` with TF32 operands by
default. Gram lower bounds then exceeded exact distances by up to ~8e8 on the
monotone fixture, and the general path certified wrong neighbors. Forcing the
binary32 product on earlier GPUs keeps that route covered everywhere.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._kernels.distance import build_index
from mlxlearn_device import _neighbors as neighbors
from mlxlearn_device import _neighbors_general as general
from mlxlearn_device import _neighbors_gram as gram


@pytest.mark.parametrize("architecture,may_round", [
    ("applegpu_g13s", False), ("applegpu_g13g", False), ("applegpu_g14d", False),
    ("applegpu_g15s", False), ("applegpu_g16s", False),
    ("applegpu_g17d", True), ("applegpu_g17s", True), ("applegpu_g18p", True),
    ("applegpu_g21x", True), ("", True), ("applegpu", True), ("unknown_gpu", True),
])
def test_only_pre_accelerator_architectures_trust_float32_matmul(architecture, may_round):
    assert gram.architecture_may_round_float32(architecture) is may_round


def _envelope_violation(a, b, products):
    """Largest error beyond K binary32 epsilons of ``|a| @ |b|.T`` (<= 0 passes)."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    scale = np.abs(a64) @ np.abs(b64).T
    error = np.abs(np.asarray(products, dtype=np.float64) - a64 @ b64.T)
    return float(np.max(error - a.shape[1] * 2.0**-23 * scale - 2.0**-126))


@pytest.mark.gpu
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("shape", [
    (1, 1, 1), (1, 2, 3), (3, 7, 5), (5, 8192, 32), (63, 65, 17), (64, 64, 16),
    (65, 1000, 129), (70, 70, 36), (8, 9, 4096), (257, 300, 1024),
])
def test_float32_gram_stays_inside_binary32_envelope(shape):
    rows, cols, features = shape
    rng = np.random.default_rng(sum(shape))
    a = (rng.normal(size=(rows, features)) * 100).astype(np.float32)
    b = rng.normal(size=(cols, features)).astype(np.float32)
    products = gram.float32_gram(mx.array(a), mx.array(b))
    assert products.shape == (rows, cols) and products.dtype == mx.float32
    assert _envelope_violation(a, b, np.array(products)) <= 0


@pytest.mark.gpu
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_float32_gram_launches_wide_training_tiles_in_chunks(monkeypatch):
    rng = np.random.default_rng(4)
    a = rng.normal(size=(9, 12)).astype(np.float32)
    b = rng.normal(size=(1000, 12)).astype(np.float32)
    whole = np.array(gram.float32_gram(mx.array(a), mx.array(b)))
    monkeypatch.setattr(gram, "_MAX_LAUNCH_COLUMNS", 192)
    chunked = np.array(gram.float32_gram(mx.array(a), mx.array(b)))
    np.testing.assert_array_equal(chunked, whole)


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
    original = gram.float32_gram

    def counted(a, b):
        calls.append(None)  # count only: retaining arrays would inflate peak memory
        return original(a, b)

    monkeypatch.setattr(gram, "matmul_may_round_float32", lambda: True)
    monkeypatch.setattr(gram, "float32_gram", counted)
    _, indices = build_index(train, backend="mlx").query(
        queries, 5, query_block_size=len(queries), train_block_size=tile,
    )
    np.testing.assert_array_equal(indices, _oracle(train, queries, 5))
    assert len(calls) == -(-len(train) // tile)
