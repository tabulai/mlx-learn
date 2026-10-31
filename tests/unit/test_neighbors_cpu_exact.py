"""CPU search checked against exact rational values of the original inputs."""

import numpy as np
import pytest

from mlxlearn._kernels.neighbors_cpu import CPUIndex


@pytest.fixture(autouse=True)
def portable_cpu_engine(monkeypatch):
    # These tests exercise the portable certificates even when a compiled
    # extension is installed. Native arithmetic has its own independent suite.
    from mlxlearn._kernels import neighbors_cpu

    monkeypatch.setattr(neighbors_cpu, "_native_query", None)
    monkeypatch.setattr(neighbors_cpu, "_native_query_concentrated", None)
    monkeypatch.setattr(neighbors_cpu, "_native_integer_select", None)
    monkeypatch.setattr(neighbors_cpu, "_native_feature_ranges", None)


def exact_oracle(train, queries, k, *, exclude_self=False):
    """Integer scaling via Python ratios, independent of the engine bit parser."""
    def scaled(row):
        result = []
        for value in row:
            numerator, denominator = float(value).as_integer_ratio()
            result.append(numerator * ((1 << 149) // denominator))
        return result

    points = [scaled(row) for row in train]
    indices = []
    for row, query in enumerate(queries):
        q = scaled(query)
        keys = [(sum((a - b) ** 2 for a, b in zip(q, point, strict=True)), index)
                for index, point in enumerate(points) if not exclude_self or index != row]
        indices.append([index for _, index in sorted(keys)[:k]])
    return np.asarray(indices, np.int64)


@pytest.mark.parametrize("method", ["direct", "gram", "auto"])
@pytest.mark.parametrize("k", [1, 5, 12])
def test_cpu_rounded_distances_and_subnormals_keep_exact_order(method, k):
    train = np.zeros((12, 4), np.float32)
    train[:4, 0] = 1
    train[:4, 1] = [2**-90, 2**-100, 0, 2**-100]
    train[4:6, 0] = [2**-149, 2**-148]
    train[6:8, 0] = 1e30
    train[6:8, 1] = [2, 1]
    train[8:10, 0] = [-1, 1]
    train[10:12, 0] = [np.finfo(np.float32).max, -np.finfo(np.float32).max]
    queries = np.array([[0, 0, 0, 0], [1e30, 1, 0, 0]], np.float32)
    expected = exact_oracle(train, queries, k)
    index = CPUIndex(train)
    distances, actual = index.query(queries, k, method=method)
    np.testing.assert_array_equal(actual, expected)
    assert distances.dtype == np.float64
    assert np.isfinite(distances).all()
    np.testing.assert_array_equal(
        index.query(queries, k, return_distance=False, method=method)[1], expected,
    )


@pytest.mark.parametrize("features", [8, 32, 128])
@pytest.mark.parametrize("method", ["direct", "gram"])
def test_cpu_arbitrary_finite_float32_bit_patterns(features, method):
    rng = np.random.default_rng(8091 + features)
    bits = rng.integers(0, 2**32, size=(38, features), dtype=np.uint32)
    bits = (bits & np.uint32(0x807fffff)) | (
        rng.integers(0, 255, size=bits.shape, dtype=np.uint32) << np.uint32(23)
    )
    data = bits.view(np.float32)
    train, queries = data[:31].copy(), data[31:].copy()
    np.testing.assert_array_equal(
        CPUIndex(train).query(queries, 7, method=method)[1], exact_oracle(train, queries, 7),
    )


@pytest.mark.parametrize("block", [None, 1, 3, 11])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_cpu_duplicate_ties_and_self_exclusion_across_both_tiles(block, exclude_self):
    train = np.zeros((23, 8), np.float32)
    train[4:8, :2] = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    train[8:12, 0] = 1
    train[8:12, 1] = [2**-80, 2**-100, 2**-90, 2**-100]
    train[12:, 0] = np.arange(11, dtype=np.float32)
    queries = train.copy()
    k = len(train) - int(exclude_self)
    distances, actual = CPUIndex(train).query(
        queries, k, exclude_self=exclude_self, query_block_size=4, train_block_size=block,
    )
    np.testing.assert_array_equal(actual, exact_oracle(train, queries, k, exclude_self=exclude_self))
    assert np.isfinite(distances).all()
    if exclude_self:
        assert np.all(actual != np.arange(len(train))[:, None])


def test_cpu_gram_omitted_point_certificate_falls_back_on_cancellation(monkeypatch):
    rng = np.random.default_rng(9271)
    train = (1e8 + 32 * rng.normal(size=(300, 32))).astype(np.float32)
    queries = train[[299, 101, 200]].copy()
    queries[1, 0] = np.nextafter(queries[1, 0], np.float32(np.inf))
    index = CPUIndex(train)
    calls = []
    direct = index._direct

    def record(*args, **kwargs):
        calls.append(True)
        return direct(*args, **kwargs)

    monkeypatch.setattr(index, "_direct", record)
    actual = index.query(queries, 5, method="gram")[1]
    np.testing.assert_array_equal(actual, exact_oracle(train, queries, 5))
    assert calls, "the full lower-bound certificate must reject cancellation"


def test_cpu_integer_gram_at_exact_bound_and_fractional_query_recheck(monkeypatch):
    rng = np.random.default_rng(513)
    train = rng.integers(-256, 257, size=(512, 64)).astype(np.float32)
    train[0], train[1] = 256, -256
    queries = train[:16].copy()
    index = CPUIndex(train)
    calls = []
    gram = index._integer_gram

    def record(*args, **kwargs):
        calls.append(True)
        return gram(*args, **kwargs)

    monkeypatch.setattr(index, "_integer_gram", record)
    np.testing.assert_array_equal(index.query(queries, 5)[1], exact_oracle(train, queries, 5))
    assert len(calls) == 1
    queries[0, 0] += np.float32(0.5)
    np.testing.assert_array_equal(index.query(queries, 5)[1], exact_oracle(train, queries, 5))
    assert len(calls) == 1


def test_cpu_integer_certificate_checks_training_after_rejection_sample():
    train = np.ones((512, 32), np.float32)
    train[-1, 0] = 1.5
    index = CPUIndex(train)
    assert not index._integer_query(np.ones((16, 32), np.float32))
    assert index._integral is False


def test_cpu_empty_queries_and_owned_storage_reference():
    train = np.zeros((5, 3), np.float32)
    index = CPUIndex(train)
    assert index.X is train
    distances, actual = index.query(np.empty((0, 3), np.float32), 2)
    assert distances.shape == actual.shape == (0, 2)


def test_cpu_native_dyadic_repairs_are_exact_and_recheck_each_query():
    train = np.array([[1, 0.1], [1, 0], [-1, 0], [1, -0.1]], np.float32)
    queries = np.zeros((2, 2), np.float32)
    candidates = np.broadcast_to(np.arange(4), (2, 4))
    index = CPUIndex(train)
    np.testing.assert_array_equal(
        index._native_exact_order(queries, candidates, 4), exact_oracle(train, queries, 4),
    )
    assert index._dyadic_train is not None
    # A new query can require finer dyadic units than the fitted data.
    queries[0, 1] = 2**-100
    assert index._native_exact_order(queries, candidates, 4) is None
    np.testing.assert_array_equal(index.query(queries, 4)[1], exact_oracle(train, queries, 4))


def test_cpu_native_dyadic_square_bound_rejects_overflow():
    train = np.array([[2**20, 0.1], [2**20, 0], [-2**20, 0]], np.float32)
    queries = np.zeros((1, 2), np.float32)
    candidates = np.arange(3)[None, :]
    index = CPUIndex(train)
    assert index._native_exact_order(queries, candidates, 3) is None
    np.testing.assert_array_equal(index.query(queries, 3)[1], exact_oracle(train, queries, 3))


def test_cpu_tree_proposals_never_override_full_omitted_point_certificate(monkeypatch):
    rng = np.random.default_rng(491)
    train = rng.normal(size=(300, 16)).astype(np.float32)
    train[:, :8] *= 1000
    queries = train[[299, 257, 201]].copy()
    index = CPUIndex(train)
    index.query(queries, 5, method="gram")
    assert index._proposal_tree is not None

    class WrongTree:
        def query(self, query, *, k, workers):
            # Deliberately omit every exact training match. The tree has no
            # authority over membership; the exhaustive lower bound rejects it.
            return None, np.broadcast_to(np.arange(k), (len(query), k)).copy()

    index._proposal_tree = WrongTree()
    calls = []
    direct = index._direct

    def record(*args, **kwargs):
        calls.append(True)
        return direct(*args, **kwargs)

    monkeypatch.setattr(index, "_direct", record)
    np.testing.assert_array_equal(
        index.query(queries, 5, method="gram")[1], exact_oracle(train, queries, 5),
    )
    assert len(calls) == len(queries)


def test_cpu_small_partition_restores_canonical_zero_boundary():
    train = np.full((120, 4), np.float32(0.1))
    train[2::3, 0] = 0.2
    queries = train[:3].copy()
    index = CPUIndex(train)
    np.testing.assert_array_equal(index.query(queries, 5)[1], exact_oracle(train, queries, 5))

