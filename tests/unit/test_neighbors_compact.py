"""Certified compact scans retain the original exact FP32 neighbor result."""

import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import _neighbors_compact as compact
from mlxlearn_device._neighbors_compact import compact_neighbors, prepare_compact
from mlxlearn_device._neighbors_fused import _precise_merge_kernel, _scan_kernel, refine_neighbors

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]


def original_neighbors(train, queries, k, *, exclude_self=False, query_offset=0):
    """Invoke the original guarded Kahan scan directly, bypassing dispatch."""
    n_train, features = train.shape
    n_query = queries.shape[0]
    parts = 7
    distances, indices = _scan_kernel()(
        inputs=[train, queries, mx.array([n_train, n_query, features, parts, query_offset], mx.int32)],
        template=[("K", k), ("D", features), ("EXCLUDE_SELF", exclude_self)],
        grid=(((n_query * parts * 32 + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n_query, parts, k)] * 2,
        output_dtypes=[mx.float32, mx.int32],
        stream=mx.gpu,
    )
    return tuple(_precise_merge_kernel()(
        inputs=[distances, indices, train, queries, mx.array([parts, n_query, n_train], mx.int32)],
        template=[("K", k), ("D", features)],
        grid=(((n_query * 32 + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n_query, k)] * 2,
        output_dtypes=[mx.float32, mx.int32],
        stream=mx.gpu,
    ))


def check_exact(train, queries, k, **kwargs):
    expected = original_neighbors(train, queries, k, **kwargs)
    actual = compact_neighbors(train, queries, k, prepare_compact(train), **kwargs)
    assert actual[0].dtype == mx.float32 and actual[1].dtype == mx.int32
    assert actual[0].shape == actual[1].shape == (queries.shape[0], k)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)
    return actual


@pytest.mark.parametrize("features", [4, 8, 16, 32])
@pytest.mark.parametrize("k", [1, 5, 8])
def test_compact_random_batches_match_original_scan(features, k):
    rng = np.random.default_rng(740 + features + k)
    train = mx.array(rng.normal(size=(1027, features)).astype(np.float32))
    query = mx.array(rng.normal(size=(7, features)).astype(np.float32))
    check_exact(train, query, k)


@pytest.mark.parametrize("n_train,n_query,features,k", [(50000, 65, 32, 5), (16385, 64, 16, 8)])
def test_compact_large_odd_batches_match_original_scan(n_train, n_query, features, k):
    rng = np.random.default_rng(n_train)
    train = mx.array(rng.normal(size=(n_train, features)).astype(np.float32))
    query = mx.array(rng.normal(size=(n_query, features)).astype(np.float32))
    check_exact(train, query, k)


@pytest.mark.parametrize("features", [8, 32])
@pytest.mark.parametrize("case", [
    "duplicate", "half_rounding_ties", "half_subnormal", "float_subnormal",
    "overflow", "half_overflow_edge", "negative_offset", "tiny_squared", "mixed_scale",
])
def test_compact_adversarial_arithmetic_matches_original_scan(features, case):
    rng = np.random.default_rng(120 + features)
    train = rng.normal(size=(67, features)).astype(np.float32)
    query = rng.normal(size=(5, features)).astype(np.float32)
    if case == "duplicate":
        train[:20] = train[21]
        query[:3] = train[21]
    elif case == "half_rounding_ties":
        train[:] = np.float32(1.00048828125)
        train[:, 0] += np.arange(len(train), dtype=np.float32) * np.float32(2**-23)
        query[:] = train[34]
    elif case == "half_subnormal":
        train *= np.float32(2**-20)
        query *= np.float32(2**-20)
        train[:5, 0] = np.array([2**-24, -(2**-24), 2**-25, 2**-14, -(2**-14)], np.float32)
    elif case == "float_subnormal":
        train *= np.float32(2**-140)
        query *= np.float32(2**-140)
    elif case == "overflow":
        train[::2, 0] = np.float32(1e30)
        query[:, 0] = np.float32(-1e30)
    elif case == "half_overflow_edge":
        train[:] = np.float32(65504)
        train[:, 0] += np.arange(len(train), dtype=np.float32) - 32
        query[:] = np.float32(65504)
    elif case == "negative_offset":
        train -= np.float32(1e6)
        query -= np.float32(1e6)
    elif case == "tiny_squared":
        train *= np.float32(1e-19)
        query *= np.float32(1e-19)
    else:
        train[:, 0] = np.float32(1e20)
        query[:, 0] = np.float32(1e20)
        train[5, 1] = np.float32(-1e30)
    X, Q = mx.array(train), mx.array(query)
    check_exact(X, Q, 5)
    check_exact(X, X[11:16], 5, exclude_self=True, query_offset=11)


@pytest.mark.parametrize("k", [1, 5, 8])
def test_compact_ties_keep_index_order_and_exclude_self(k):
    train = mx.zeros((67, 16), mx.float32)
    actual = check_exact(train, train[2:7], k, exclude_self=True, query_offset=2)
    expected = [[index for index in range(k + 1) if index != row][:k] for row in range(2, 7)]
    np.testing.assert_array_equal(actual[1], expected)


def test_compact_all_infinite_distances_keep_valid_indices():
    train = mx.full((67, 32), 1e30, mx.float32)
    query = mx.full((3, 32), -1e30, mx.float32)
    actual = check_exact(train, query, 8)
    assert np.isinf(np.asarray(actual[0])).all()
    np.testing.assert_array_equal(actual[1], np.broadcast_to(np.arange(8), (3, 8)))


def test_compact_preserves_small_terms_next_to_large_squared_difference():
    train = np.full((67, 32), 30000, np.float32)
    train[0] = [10000.0] + [1.0] * 31
    train[1] = [10000.0, 4.0] + [0.0] * 30
    actual = check_exact(mx.array(train), mx.zeros((3, 32)), 1)
    np.testing.assert_array_equal(actual[1], [[1]] * 3)
    np.testing.assert_array_equal(actual[0], [[100000016.0]] * 3)


@pytest.mark.parametrize("features", [4, 8, 16, 32])
def test_compact_packed_lower_bounds_never_exceed_original_distance(features):
    rng = np.random.default_rng(180 + features)
    train = rng.normal(size=(83, features)).astype(np.float32)
    train[10:20] *= np.float32(2**-20)
    train[20:30] *= np.float32(2**-140)
    train[30:40] += np.float32(-1e6)
    train[40:50] *= np.float32(1e30)
    train[50:60] *= np.float32(1e-19)
    train[60:70] = np.float32(1.00048828125)
    train[60:70, 0] += np.arange(10, dtype=np.float32) * np.float32(2**-23)
    train[70:, 0] = np.resize(np.array([
        2**-24, -(2**-24), 2**-25, 2**-14, -(2**-14), 65504, 65520,
    ], np.float32), 13)
    queries = np.vstack([train[::9], np.zeros((1, features), np.float32)])
    X, Q = mx.array(train), mx.array(queries)
    prepared = prepare_compact(X)
    n, nq, carry = len(train), len(queries), 9
    # One reference per partition exposes every actual packed lower bound,
    # rather than only checking whichever candidates survive selection.
    tasks = ((nq + 1) // 2) * n
    lower, indices = compact._scan_kernel()(
        inputs=[prepared[0], Q, prepared[1], mx.array([n, nq, n, 0], mx.int32)],
        template=[("C", carry), ("D", features), ("EXCLUDE_SELF", False)],
        grid=(((tasks * 32 + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(nq, n, carry)] * 2,
        output_dtypes=[mx.float32, mx.int32],
        stream=mx.gpu,
    )
    all_indices = mx.broadcast_to(mx.arange(n, dtype=mx.int32)[None, :], (nq, n))
    exact = refine_neighbors(X, Q, all_indices)
    np.testing.assert_array_equal(np.asarray(indices)[:, :, 0], np.asarray(all_indices))
    bounds = np.asarray(lower)[:, :, 0]
    assert np.isfinite(bounds).any()
    assert np.all(bounds <= np.asarray(exact))


def test_compact_partition_geometry_bounds_packed_reference_positions():
    for n_train in (1, 255, 256, 8191, 8192, 8193, 50000, 262145, 2**21 - 1, 2**21):
        for n_query in (1, 7, 64, 256, 4096):
            parts = compact._partitions(n_train, n_query)
            assert 1 <= parts <= 256
            largest_partition = (n_train + parts - 1) // parts
            assert (largest_partition + 31) // 32 <= 256


def test_compact_packed_maximum_local_position_does_not_wrap():
    train = np.ones((8192, 16), np.float32)
    train[8191] = 0
    train[8160] = np.float32(0.1)
    train[8159] = np.float32(0.2)
    X, Q = mx.array(train), mx.zeros((3, 16))
    prepared = prepare_compact(X)
    lower, indices = compact._scan_kernel()(
        inputs=[prepared[0], Q, prepared[1], mx.array([8192, 3, 1, 0], mx.int32)],
        template=[("C", 9), ("D", 16), ("EXCLUDE_SELF", False)],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(3, 1, 9)] * 2,
        output_dtypes=[mx.float32, mx.int32],
        stream=mx.gpu,
    )
    mx.eval(lower, indices)
    np.testing.assert_array_equal(np.asarray(indices)[:, 0, :3], [[8191, 8160, 8159]] * 3)


@pytest.mark.parametrize("k", [5, 8])
def test_compact_many_winners_in_one_lane_are_not_lost(k):
    train = np.full((8192, 32), 1000, np.float32)
    winners = np.arange(16, dtype=np.int32) * 32
    train[winners] = 0
    train[winners, 0] = np.arange(1, 17, dtype=np.float32)
    actual = check_exact(mx.array(train), mx.zeros((3, 32)), k)
    np.testing.assert_array_equal(actual[1], np.broadcast_to(winners[:k], (3, k)))


@pytest.mark.parametrize("layout", ["unaligned", "strided_features", "strided_rows"])
def test_compact_input_layouts_preserve_exact_results(layout):
    rng = np.random.default_rng(761)
    features = 32
    if layout == "unaligned":
        train = mx.array(rng.normal(size=1027 * features + 1).astype(np.float32))[1:]
        query = mx.array(rng.normal(size=7 * features + 1).astype(np.float32))[1:]
        train, query = train.reshape(1027, features), query.reshape(7, features)
    elif layout == "strided_features":
        train = mx.array(rng.normal(size=(1027, features * 2)).astype(np.float32))[:, ::2]
        query = mx.array(rng.normal(size=(7, features * 2)).astype(np.float32))[:, ::2]
    else:
        train = mx.array(rng.normal(size=(2054, features)).astype(np.float32))[::2]
        query = mx.array(rng.normal(size=(14, features)).astype(np.float32))[::2]
    check_exact(train, query, 5)


@pytest.mark.parametrize("features,k", [(4, 1), (16, 5), (32, 8)])
def test_compact_empty_query_preserves_shapes_and_dtypes(features, k):
    train = mx.ones((33, features), mx.float32)
    actual = compact_neighbors(train, mx.zeros((0, features)), k, prepare_compact(train))
    assert actual[0].shape == actual[1].shape == (0, k)
    assert actual[0].dtype == mx.float32 and actual[1].dtype == mx.int32


def test_compact_path_imports_without_numpy_or_sklearn():
    source = '''
import importlib.abc
import sys
class BlockHostLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'sklearn'):
            raise AssertionError('Host library imported: ' + fullname)
sys.meta_path.insert(0, BlockHostLibraries())
import mlx.core as mx
from mlxlearn_device._neighbors_compact import compact_neighbors, prepare_compact
mx.random.seed(710)
train = mx.random.normal((16384, 16))
distances, indices = compact_neighbors(train, train[:64], 5, prepare_compact(train))
mx.eval(distances, indices)
assert indices[:, 0].tolist() == list(range(64))
assert 'numpy' not in sys.modules and 'sklearn' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", source], env=os.environ.copy(), check=True, timeout=30)
