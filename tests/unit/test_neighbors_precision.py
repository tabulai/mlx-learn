"""Independent original-coordinate ordering at rounded ties and cancellation.

The oracle uses direct float64 differences on the actual float32 inputs. It
never treats nearby distances as equal or derives expected indices from an
implementation kernel. The offset fixture's squared distances are exact dyadic
values at float64 precision, so its disputed order is unambiguous.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.config import tuning_context
from mlxlearn._kernels.distance import build_index
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

BACKENDS = [
    "cpu",
    pytest.param("mlx", marks=[pytest.mark.gpu, pytest.mark.skipif(
        not mx.metal.is_available(), reason="requires Metal",
    )]),
]


def direct_oracle(train, queries, k, *, exclude_self=False):
    """Sort every candidate by (direct float64 squared distance, index)."""
    train64 = np.asarray(train, dtype=np.float64)
    query64 = np.asarray(queries, dtype=np.float64)
    ids = np.arange(len(train), dtype=np.int64)
    distances, indices = [], []
    for row, query in enumerate(query64):
        delta = train64 - query
        squared = np.sum(delta * delta, axis=1, dtype=np.float64)
        if exclude_self:
            squared[row] = np.inf
        order = np.lexsort((ids, squared))[:k]
        distances.append(np.sqrt(squared[order]))
        indices.append(order)
    return np.asarray(distances), np.asarray(indices)


@pytest.fixture(scope="module")
def offset_seed251():
    rng = np.random.default_rng(251)
    train = (rng.normal(size=(50_000, 32)) + 10_000).astype(np.float32)
    queries = (rng.normal(size=(256, 32)) + 10_000).astype(np.float32)
    # Row 79 is the independently recorded ordering failure. Copying the first
    # two fixture queries from training does not affect this row.
    return train, queries[79:80]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("k", [1, 2])
def test_distinct_original_distances_do_not_become_index_ties(backend, k, offset_seed251):
    full_train, query = offset_seed251
    # Index zero is farther but wins the old float32-only tie break.
    train = full_train[[1902, 43153]]
    delta = train.astype(np.float64) - query.astype(np.float64)
    squared = np.sum(delta * delta, axis=1)
    assert squared[1] < squared[0]
    assert squared.astype(np.float32)[0] == squared.astype(np.float32)[1]
    expected_distances, expected_indices = direct_oracle(train, query, k)
    np.testing.assert_array_equal(expected_indices, [[1]] if k == 1 else [[1, 0]])
    index = build_index(train, backend=backend)
    distances, indices = index.query(query, k)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(distances, expected_distances, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(index.query(query, k, return_distance=False)[1], expected_indices)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("k", [4, 5])
def test_offset251_order_and_k_boundary_match_exhaustive_oracle(backend, k, offset_seed251):
    train, query = offset_seed251
    expected_distances, expected_indices = direct_oracle(train, query, k)
    assert expected_indices[0, 3] == 43153
    if k == 4:
        assert 1902 not in expected_indices[0]
    else:
        assert expected_indices[0, 4] == 1902
    # Sixty-five queries exercise the batched certified route and its query
    # tail, while computing the exhaustive oracle only once.
    queries = np.repeat(query, 65, axis=0)
    index = build_index(train, backend=backend)
    distances, indices = index.query(queries, k)
    np.testing.assert_array_equal(indices, np.repeat(expected_indices, 65, axis=0))
    np.testing.assert_allclose(
        distances, np.repeat(expected_distances, 65, axis=0), atol=1e-6, rtol=1e-6,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("k", [1, 3])
def test_genuinely_equal_distances_still_choose_smallest_indices(backend, k):
    train = np.zeros((8, 32), dtype=np.float32)
    train[:4, :2] = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    train[4:, 0] = 2
    query = np.zeros((1, 32), dtype=np.float32)
    expected_distances, expected_indices = direct_oracle(train, query, k)
    np.testing.assert_array_equal(expected_indices, np.arange(k)[None, :])
    distances, indices = build_index(train, backend=backend).query(query, k)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(distances, expected_distances)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("block", [None, 8192])
def test_monotone_training_copied_rows_survive_gram_cancellation(backend, block):
    train = np.arange(8192 * 32, dtype=np.float32).reshape(8192, 32)
    queries = train[[0, 3, 4, 4096, 8191]].copy()
    expected_distances, expected_indices = direct_oracle(train, queries, 5)
    distances, indices = build_index(train, backend=backend).query(queries, 5, train_block_size=block)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(indices[:, 0], [0, 3, 4, 4096, 8191])
    np.testing.assert_array_equal(distances[:, 0], 0)
    np.testing.assert_allclose(distances, expected_distances, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_general_path_keeps_ties_self_exclusion_and_single_row_tail(backend):
    train = np.zeros((67, 40), dtype=np.float32)
    train[3:7, 0] = np.array([1, -1, 2, -2], np.float32)
    expected_distances, expected_indices = direct_oracle(train, train, 5, exclude_self=True)
    distances, indices = build_index(train, backend=backend).query(
        train, 5, exclude_self=True, query_block_size=11, train_block_size=33,
    )
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(distances, expected_distances, atol=0, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rounded_boundary_tie_does_not_change_prediction(backend, offset_seed251):
    full_train, query = offset_seed251
    train = full_train[[1902, 43153]]
    with tuning_context(force_backend=backend):
        classifier = KNeighborsClassifier(n_neighbors=1, algorithm="brute").fit(train, [0, 1])
        regressor = KNeighborsRegressor(n_neighbors=1, algorithm="brute").fit(train, [0., 10.])
        np.testing.assert_array_equal(classifier.predict(query), [1])
        np.testing.assert_array_equal(classifier.predict_proba(query), [[0., 1.]])
        np.testing.assert_array_equal(regressor.predict(query), [10.])


@pytest.mark.gpu
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_ambiguous_order_matches_exact_integer_oracle_across_float32_range():
    from mlxlearn_device._neighbors_fused import precise_select_candidates

    rng = np.random.default_rng(9021)
    bits = rng.integers(0, 2**32, size=(300, 32), dtype=np.uint32)
    # Replace nonfinite exponents, preserving arbitrary signs and mantissas.
    bits = (bits & np.uint32(0x807fffff)) | (
        rng.integers(0, 255, size=bits.shape, dtype=np.uint32) << np.uint32(23)
    )
    arrays = bits.view(np.float32)
    queries, train = arrays[:100].copy(), arrays[100:].copy()
    queries[:4] = 0
    train[:8] = 0
    train[0, 0], train[1, 0] = np.float32(2**-140), np.float32(2**-141)
    train[2:4, 0] = np.float32(1e30)
    train[2, 1], train[3, 1] = 2, 1
    train[4:6, 0] = 1
    train[4, 1], train[5, 1] = np.float32(0.0002), np.float32(0.0001)
    train[6, 0], train[7, 0] = -1, 1  # a genuine equal-distance tie

    def integers(row):
        result = []
        for raw in row.view(np.uint32):
            raw = int(raw)
            exponent = (raw >> 23) & 255
            mantissa = (raw & 0x7fffff) | (0x800000 if exponent else 0)
            value = mantissa << max(0, exponent - 1)
            result.append(-value if raw >> 31 else value)
        return result

    expected = []
    candidates = np.arange(200, dtype=np.int32).reshape(100, 2)
    for row, query in enumerate(queries):
        q = integers(query)
        keys = []
        for index in candidates[row]:
            keys.append((sum((a - b)**2 for a, b in zip(q, integers(train[index]), strict=True)), int(index)))
        expected.append([index for _, index in sorted(keys)])
    _, indices = precise_select_candidates(mx.array(train), mx.array(queries), mx.array(candidates), 2)
    np.testing.assert_array_equal(indices, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("features", [8, 32, 128, 1024])
def test_close_moderate_scale_pairs_match_exact_integer_order(features):
    """Exercise the precision filters and parallel wide-feature refinement."""
    from mlxlearn_device._neighbors_fused import precise_select_candidates

    rng = np.random.default_rng(5200 + features)
    count = 48
    scales = np.exp2(rng.integers(-20, 21, size=(count, 1))).astype(np.float32)
    queries = (rng.normal(size=(count, features)) * scales).astype(np.float32)
    delta = (rng.normal(size=queries.shape) * scales).astype(np.float32)
    train = np.empty((2 * count, features), np.float32)
    train[::2] = queries + delta
    train[1::2] = queries - delta
    # Mix exact duplicates, adjacent representable perturbations, and changes
    # large enough for the double-single filter to resolve decisively.
    train[1::8] = train[::8]
    train[3::8, 0] = np.nextafter(train[3::8, 0], np.float32(np.inf))
    train[5::8, 0] += scales[2::4, 0] * np.float32(1e-4)

    def integer_row(row):
        result = []
        for raw in row.view(np.uint32):
            raw = int(raw)
            exponent = (raw >> 23) & 255
            mantissa = (raw & 0x7fffff) | (0x800000 if exponent else 0)
            value = mantissa << max(0, exponent - 1)
            result.append(-value if raw >> 31 else value)
        return result

    expected = []
    candidates = np.arange(2 * count, dtype=np.int32).reshape(count, 2)
    for row, query in enumerate(queries):
        q = integer_row(query)
        pairs = [(sum((a - b)**2 for a, b in zip(q, integer_row(train[i]), strict=True)), int(i))
                 for i in candidates[row]]
        expected.append([i for _, i in sorted(pairs)])
    _, actual = precise_select_candidates(mx.array(train), mx.array(queries), mx.array(candidates), 2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("features", [1, 32, 128])
def test_native_fit_preserves_subnormal_coordinate_bits(backend, features):
    from mlxlearn_device import fit_neighbors

    bits = np.zeros((3, features), np.uint32)
    bits[:, 0] = [2, 1, 3]
    train = bits.view(np.float32)
    device = "gpu" if backend == "mlx" else "cpu"
    index = fit_neighbors(mx.array(train), device=device)
    np.testing.assert_array_equal(np.asarray(index.X).view(np.uint32), bits)
    _, actual = index.query(mx.zeros((1, features), mx.float32), 3)
    np.testing.assert_array_equal(actual, [[1, 0, 2]])


@pytest.mark.gpu
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("features", [32, 128])
def test_large_candidate_list_repairs_rounded_order_and_invalid_tail(features, offset_seed251):
    from mlxlearn_device._neighbors_fused import precise_select_candidates

    full_train, query32 = offset_seed251
    train = np.zeros((40, features), np.float32)
    query = np.zeros((1, features), np.float32)
    query[:, :32] = query32
    train[:2, :32] = full_train[[1902, 43153]]
    train[2:, :32] = query32 + np.arange(2, 40, dtype=np.float32)[:, None] * 20
    _, expected = direct_oracle(train, query, 40)
    assert expected[0, :2].tolist() == [1, 0]
    candidates = np.array([list(range(40)) + [-1, 2**31 - 1]], np.int32)
    distances, actual = precise_select_candidates(mx.array(train), mx.array(query), mx.array(candidates), 42)
    np.testing.assert_array_equal(np.asarray(actual)[:, :40], expected)
    np.testing.assert_array_equal(np.asarray(actual)[:, 40:], 2**31 - 1)
    assert np.all(np.isinf(np.asarray(distances)[:, 40:]))
