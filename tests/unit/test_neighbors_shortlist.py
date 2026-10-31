"""Independent precision and omission-bound checks for fused shortlists."""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._kernels.distance import build_index
from mlxlearn_device._neighbors_shortlist import refine_certified_candidates

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]

SENTINEL = np.iinfo(np.int32).max


def _direct_candidates(train, queries, candidates, k):
    """Independent float64 differences; invalid slots follow every real point."""
    values = np.full((len(queries), k), np.inf, np.float64)
    indices = np.full((len(queries), k), SENTINEL, np.int32)
    for row, query in enumerate(queries.astype(np.float64)):
        ids = candidates[row]
        ids = ids[(ids >= 0) & (ids < len(train))]
        delta = train[ids].astype(np.float64) - query
        squared = np.sum(delta * delta, axis=1, dtype=np.float64)
        order = np.lexsort((ids, squared))[:k]
        values[row, :len(order)] = squared[order]
        indices[row, :len(order)] = ids[order]
    return values, indices


def _run(train, queries, candidates, k, barriers=None, certificate=None, *, check=True):
    rows = len(queries)
    if barriers is None:
        barriers = np.full(rows, np.inf, np.float32)
    if certificate is None:
        certificate = np.ones(rows, bool)
    result = refine_certified_candidates(
        mx.array(train), mx.array(queries), mx.array(candidates), k,
        mx.array(barriers), mx.array(certificate), check_barrier=check,
    )
    return tuple(np.asarray(array) for array in result)


@pytest.mark.parametrize("features,width,k", [
    (33, 1, 1), (65, 7, 1), (65, 13, 5), (128, 7, 5),
    (128, 13, 8), (256, 7, 5), (256, 13, 8), (256, 16, 8),
])
def test_shortlist_matches_direct_oracle_across_launch_shapes(features, width, k):
    rng = np.random.default_rng(7200 + features + width)
    train = rng.normal(size=(53, features)).astype(np.float32)
    queries = rng.normal(size=(9, features)).astype(np.float32)
    candidates = np.stack([rng.choice(len(train), width, replace=False) for _ in queries]).astype(np.int32)
    queries[0] = train[candidates[0, 0]]
    expected_values, expected_indices = _direct_candidates(train, queries, candidates, k)
    values, indices, certificate = _run(train, queries, candidates, k)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(values, expected_values, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(values[0, 0], 0)
    np.testing.assert_array_equal(certificate, True)


@pytest.fixture(scope="module")
def _offset_pair():
    rng = np.random.default_rng(251)
    train = (rng.normal(size=(50_000, 32)) + 10_000).astype(np.float32)
    queries = (rng.normal(size=(256, 32)) + 10_000).astype(np.float32)
    return train[[1902, 43153]], queries[79:80]


@pytest.mark.parametrize("features,k", [(65, 1), (128, 2)])
def test_shortlist_preserves_the_recorded_offset_order(features, k, _offset_pair):
    pair, query32 = _offset_pair
    train = np.zeros((7, features), np.float32)
    query = np.zeros((1, features), np.float32)
    train[:2, :32], query[:, :32] = pair, query32
    train[2:, :32] = query32 + np.arange(2, 7, dtype=np.float32)[:, None] * 20
    candidates = np.array([[6, 4, 2, 0, 1, 5, 3]], np.int32)
    expected_values, expected_indices = _direct_candidates(train, query, candidates, k)
    assert expected_indices[0, 0] == 1
    squared_pair = np.sum((pair.astype(np.float64) - query32.astype(np.float64))**2, axis=1)
    assert squared_pair[1] < squared_pair[0]
    assert squared_pair.astype(np.float32)[0] == squared_pair.astype(np.float32)[1]
    values, indices, certificate = _run(train, query, candidates, k)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(values, expected_values, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(certificate, True)


@pytest.mark.parametrize("k", [1, 8])
def test_shortlist_true_ties_use_global_index_order(k):
    train = np.zeros((13, 65), np.float32)
    train[np.arange(13), np.arange(13)] = np.where(np.arange(13) % 2, -1, 1)
    queries = np.zeros((3, 65), np.float32)
    candidates = np.tile(np.arange(13, dtype=np.int32)[::-1], (3, 1))
    values, indices, certificate = _run(train, queries, candidates, k)
    np.testing.assert_array_equal(indices, np.tile(np.arange(k), (3, 1)))
    np.testing.assert_array_equal(values, 1)
    np.testing.assert_array_equal(certificate, True)


@pytest.mark.parametrize("check", [False, True])
def test_shortlist_incomplete_rows_keep_invalid_slots_and_prior_certificate(check):
    train = np.zeros((6, 128), np.float32)
    train[:, 0] = np.arange(6)
    queries = np.zeros((2, 128), np.float32)
    candidates = np.array([[5, -1, 3, SENTINEL, 0, -7, 6], [-1, -2, 6, 7, 8, 9, SENTINEL]], np.int32)
    expected_values, expected_indices = _direct_candidates(train, queries, candidates, 5)
    values, indices, certificate = _run(
        train, queries, candidates, 5, certificate=np.array([True, False]), check=check,
    )
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(certificate, [not check, False])


def _integer_coordinates(row):
    integers = []
    for raw in row.view(np.uint32):
        raw = int(raw)
        exponent = (raw >> 23) & 255
        mantissa = (raw & 0x7fffff) | (0x800000 if exponent else 0)
        value = mantissa << max(exponent - 1, 0)
        integers.append(-value if raw >> 31 else value)
    return integers


@pytest.mark.parametrize("regime", ["subnormal", "overflow", "outside_double_single_range"])
def test_shortlist_order_matches_exact_integers_outside_fast_precision_range(regime):
    train = np.zeros((7, 128), np.float32)
    permutation = np.array([7, 1, 2, 5, 3, 6, 4], np.uint32)
    if regime == "subnormal":
        train[:, 0] = permutation.view(np.float32)
    else:
        train[:, 0] = np.float32(2.0**(100 if regime == "overflow" else 40))
        train[:, 1] = permutation.astype(np.float32)
    queries = np.zeros((1, 128), np.float32)
    candidates = np.arange(7, dtype=np.int32)[None, :]
    exact_keys = [(sum(value * value for value in _integer_coordinates(point)), index)
                  for index, point in enumerate(train)]
    expected = np.array([[index for _, index in sorted(exact_keys)[:5]]])
    values, indices, certificate = _run(train, queries, candidates, 5)
    np.testing.assert_array_equal(indices, expected)
    if regime == "subnormal":
        np.testing.assert_array_equal(values, 0)
    elif regime == "overflow":
        assert np.all(np.isinf(values))
    else:
        np.testing.assert_array_equal(values, np.float32(2.0**80))
    np.testing.assert_array_equal(certificate, regime != "overflow")


@pytest.mark.parametrize("check", [False, True])
def test_shortlist_certificate_is_strict_at_the_documented_upper_endpoint(check):
    train = np.zeros((7, 128), np.float32)
    train[:, 0] = np.arange(1, 8)
    queries = np.zeros((5, 128), np.float32)
    candidates = np.tile(np.arange(7, dtype=np.int32), (5, 1))
    # The true cutoff is exactly one. The documented outer allowance is 12u;
    # its tiny absolute floor rounds away here. Equality cannot certify safety.
    upper = np.float32(1 + 6 * 2.0**-23)
    barriers = np.array([
        np.nextafter(upper, np.float32(-np.inf)), upper,
        np.nextafter(upper, np.float32(np.inf)), np.nan, np.inf,
    ], np.float32)
    prior = np.array([True, True, True, True, False])
    values, indices, certificate = _run(
        train, queries, candidates, 1, barriers, prior, check=check,
    )
    np.testing.assert_array_equal(values, 1)
    np.testing.assert_array_equal(indices, 0)
    np.testing.assert_array_equal(certificate, [False, False, True, False, False] if check else prior)


def test_general_128_feature_path_keeps_self_exclusion_and_final_tile_tail():
    train = np.zeros((67, 128), np.float32)
    train[:7, 0] = [1, -1, 2, -2, 3, -3, 4]
    candidates = np.tile(np.arange(67, dtype=np.int32), (67, 1))
    candidates[np.arange(67), np.arange(67)] = -1
    expected_values, expected_indices = _direct_candidates(train, train, candidates, 5)
    values, indices = build_index(train, backend="mlx").query(
        train, 5, exclude_self=True, query_block_size=11, train_block_size=33,
    )
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(values, np.sqrt(expected_values), rtol=1e-6, atol=1e-6)


def test_empty_shortlist_query_preserves_shapes_and_dtypes():
    values, indices, certificate = refine_certified_candidates(
        mx.zeros((10, 128), mx.float32), mx.zeros((0, 128), mx.float32),
        mx.zeros((0, 7), mx.int32), 5, mx.zeros((0,), mx.float32),
        mx.zeros((0,), mx.bool_), check_barrier=True,
    )
    assert values.shape == indices.shape == (0, 5)
    assert certificate.shape == (0,)
    assert (values.dtype, indices.dtype, certificate.dtype) == (mx.float32, mx.int32, mx.bool_)
