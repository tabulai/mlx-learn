"""Centered screening must retain original-coordinate membership and ordering.

Expected neighbors come from exhaustive float64 differences or decoded Python
integers, never another GPU search implementation. The optional center affects
screening only; it must not replace the coordinates used by precise refinement.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import _neighbors_integer as integer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]


@pytest.fixture
def supported():
    if not integer.is_supported():
        pytest.skip("requires integer tensor operations")


def _reference(train, queries, k, *, exclude_self=False, query_offset=0):
    train64 = train.astype(np.float64)
    ids = np.arange(len(train))
    distances, indices = [], []
    for row, query in enumerate(queries.astype(np.float64)):
        delta = train64 - query
        squared = np.sum(delta * delta, axis=1)
        if exclude_self:
            squared[row + query_offset] = np.inf
        order = np.lexsort((ids, squared))[:k]
        distances.append(squared[order])
        indices.append(order)
    return np.asarray(distances), np.asarray(indices)


def _check(actual, train, queries, k=5, **options):
    expected_squared, expected_indices = _reference(train, queries, k, **options)
    mx.eval(*actual)
    np.testing.assert_array_equal(actual[1], expected_indices)
    np.testing.assert_allclose(actual[0], expected_squared, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("offset", ["positive", "negative", "mixed_sign", "zero"])
def test_centered_integer_uses_a_nonvacuous_original_coordinate_certificate(supported, offset):
    rng = np.random.default_rng(728)
    center = np.full(32, 10_000, np.float32)
    if offset == "negative":
        center *= -1
    elif offset == "mixed_sign":
        center[::2] *= -1
    elif offset == "zero":
        center[:] = 0
    train = (rng.normal(size=(4099, 32)) + center).astype(np.float32)
    queries = (rng.normal(size=(9, 32)) + center).astype(np.float32)
    queries[0] = train[4098]
    X, Q, C = mx.array(train), mx.array(queries), mx.array(center)
    prepared = integer.prepare_integer(X, center=C)
    actual = integer.integer_neighbors(X, Q, 5, prepared, center=C, return_certificate=True)
    _check(actual, train, queries)
    assert np.any(np.asarray(actual[2])), "centered path never certified an ordinary Gaussian row"
    assert np.asarray(actual[0])[0, 0] == 0


@pytest.mark.parametrize("k", [1, 3])
def test_centered_integer_preserves_rounded_unequal_pairs_and_true_ties(supported, k):
    center = np.full(32, 10_000, np.float32)
    train = np.tile(center + np.float32(30), (513, 1))
    train[:3] = center
    train[0, 0] += 4
    train[0, 1] += np.float32(2**-10)
    train[1, 0] += 4
    train[2, 0] -= 4
    queries = np.tile(center, (9, 1))
    exact = np.sum((train[:3].astype(np.float64) - center.astype(np.float64)) ** 2, axis=1)
    assert exact[1] == exact[2] < exact[0]
    assert exact.astype(np.float32)[0] == exact.astype(np.float32)[1]
    X, Q, C = mx.array(train), mx.array(queries), mx.array(center)
    actual = integer.integer_neighbors(X, Q, k, integer.prepare_integer(X, center=C), center=C)
    _check(actual, train, queries, k)
    np.testing.assert_array_equal(np.asarray(actual[1])[0], [1] if k == 1 else [1, 2, 0])


def test_centered_integer_rechecks_later_queries_outside_translation_guard(supported):
    rng = np.random.default_rng(729)
    center = np.full(32, 10_000, np.float32)
    train = (rng.normal(size=(1027, 32)) + center).astype(np.float32)
    first = (rng.normal(size=(9, 32)) + center).astype(np.float32)
    later = first.copy()
    later[:3] = np.asarray([0, -10_000, 30_000], np.float32)[:, None]
    X, C = mx.array(train), mx.array(center)
    prepared = integer.prepare_integer(X, center=C)
    a = integer.integer_neighbors(X, mx.array(first), 5, prepared, center=C, return_certificate=True)
    b = integer.integer_neighbors(X, mx.array(later), 5, prepared, center=C, return_certificate=True)
    # Query graphs may be evaluated in the reverse of their construction order.
    _check(b, train, later)
    _check(a, train, first)
    assert not np.asarray(b[2])[:3].any()


def test_centered_integer_marks_an_unsafe_training_row_without_discarding_it(supported):
    rng = np.random.default_rng(730)
    center = np.full(32, 10_000, np.float32)
    train = (rng.normal(size=(1027, 32)) + center).astype(np.float32)
    train[-1] = -10_000
    queries = np.vstack((train[-1:], train[1:9])).copy()
    X, Q, C = mx.array(train), mx.array(queries), mx.array(center)
    prepared = integer.prepare_integer(X, center=C)
    actual = integer.integer_neighbors(X, Q, 5, prepared, center=C, return_certificate=True)
    _check(actual, train, queries)
    assert np.isneginf(np.asarray(prepared[1])[-1])
    assert np.asarray(actual[1])[0, 0] == len(train) - 1
    assert np.asarray(actual[0])[0, 0] == 0


def test_centered_integer_keeps_global_self_exclusion_at_reference_and_query_tails(supported):
    rng = np.random.default_rng(731)
    center = np.full(32, -10_000, np.float32)
    train = (rng.normal(size=(1027, 32)) + center).astype(np.float32)
    train[-2] = train[-1]
    queries = train[-9:].copy()
    X, Q, C = mx.array(train), mx.array(queries), mx.array(center)
    options = {"exclude_self": True, "query_offset": len(train) - len(queries)}
    actual = integer.integer_neighbors(
        X, Q, 5, integer.prepare_integer(X, center=C), center=C, **options,
    )
    _check(actual, train, queries, **options)
    assert np.asarray(actual[1])[-1, 0] == len(train) - 2


def _integer_row(row):
    result = []
    for raw in row.view(np.uint32):
        raw = int(raw)
        exponent = (raw >> 23) & 255
        mantissa = (raw & 0x7fffff) | (0x800000 if exponent else 0)
        integer_value = mantissa << max(exponent - 1, 0)
        result.append(-integer_value if raw >> 31 else integer_value)
    return result


@pytest.mark.parametrize("kind", ["subnormal", "large", "guard_low", "guard_high"])
def test_centered_integer_precision_edges_keep_exact_original_order(supported, kind):
    train = np.zeros((65, 32), np.float32)
    query = np.zeros((1, 32), np.float32)
    center = np.zeros(32, np.float32)
    if kind == "subnormal":
        train[:, 0] = np.arange(65, 0, -1, dtype=np.uint32).view(np.float32)
        center[0] = np.array([2], np.uint32).view(np.float32)[0]
    elif kind == "large":
        train[:, 0] = np.float32(1e30)
        train[:, 1] = np.arange(65, 0, -1, dtype=np.float32)
        center[0] = np.float32(1e30)
    else:
        edge = np.float32(2**-30 if kind == "guard_low" else 2**30)
        outside = np.nextafter(edge, np.float32(0 if kind == "guard_low" else np.inf))
        train[:] = edge
        train[:, 0] = outside
        train[:, 1] = edge * np.linspace(0.75, 1.25, 65, dtype=np.float32)
        query[:] = edge
        center[:] = edge
    q = _integer_row(query[0])
    expected = sorted(
        range(len(train)),
        key=lambda i: (sum((a - b)**2 for a, b in zip(q, _integer_row(train[i]), strict=True)), i),
    )[:5]
    X, Q, C = mx.array(train), mx.array(query), mx.array(center)
    squared, indices, certificate = integer.integer_neighbors(
        X, Q, 5, integer.prepare_integer(X, center=C), center=C, return_certificate=True,
    )
    mx.eval(squared, indices, certificate)
    np.testing.assert_array_equal(indices, [expected])
    if kind == "subnormal":
        np.testing.assert_array_equal(np.asarray(squared), 0)
    elif kind == "large":
        assert np.isinf(np.asarray(squared)).all()
