"""Resident search contracts independent of sklearn and host result conversion."""

import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import fit_neighbors


@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("k", [1, 5, 32, 40])
def test_resident_search_matches_direct_reference(device, k):
    rng = np.random.default_rng(83)
    X = rng.normal(size=(90, 7)).astype(np.float32)
    Q = rng.normal(size=(11, 7)).astype(np.float32)
    distances, indices = fit_neighbors(mx.array(X), device=device).query(mx.array(Q), k)
    squared = ((Q.astype(np.float64)[:, None, :] - X.astype(np.float64)) ** 2).sum(axis=2)
    expected = np.argsort(squared, axis=1)[:, :k]
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(distances, np.sqrt(np.take_along_axis(squared, expected, axis=1)),
                               rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_ties_spanning_tiles_and_self_exclusion(device):
    X = mx.zeros((80, 3), mx.float32)
    index = fit_neighbors(X, device=device)
    for block in (None, 3, 11, 80):
        _, actual = index.query(X, 5, exclude_self=True, train_block_size=block,
                                query_block_size=9)
        expected = np.array([[i for i in range(80) if i != row][:5] for row in range(80)])
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_owned_storage_and_original_coordinate_refinement(device):
    # Centered float32 coordinates lose 1-unit distinctions due to an outlier.
    X = np.array([[1.], [2.], [3.], [1e10]], dtype=np.float32)
    index = fit_neighbors(mx.array(X), device=device)
    X[:] = -7
    distances, indices = index.query(mx.array([[2.]], mx.float32), 3, train_block_size=4)
    np.testing.assert_array_equal(indices, [[1, 0, 2]])
    np.testing.assert_array_equal(distances, [[0, 1, 1]])


@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_empty_query_indices_only_and_validation(device):
    index = fit_neighbors(mx.zeros((3, 2), mx.float32), device=device)
    dist, idx = index.query(mx.zeros((0, 2), mx.float32), 2, return_distance=False)
    assert dist is None and idx.shape == (0, 2)
    for k in (0, -1, True, 1.5, 4):
        with pytest.raises(ValueError):
            index.query(index.X, k)
    with pytest.raises(ValueError, match="finite"):
        index.query(mx.array([[float("nan"), 0.]]), 1)
    with pytest.raises(ValueError, match="finite"):
        fit_neighbors(mx.array([[float("inf"), 0.]]), device=device)
    with pytest.raises(TypeError, match="float32"):
        fit_neighbors(mx.array([[1, 2]]), device=device)
    with pytest.raises(ValueError, match="feature"):
        index.query(mx.zeros((1, 3)), 1)
    with pytest.raises(ValueError, match="block"):
        index.query(index.X, 1, train_block_size=0)


def test_device_neighbors_without_numpy_or_sklearn():
    source = '''
import importlib.abc
import sys
class BlockHostLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'sklearn'):
            raise AssertionError('Host library imported: ' + fullname)
sys.meta_path.insert(0, BlockHostLibraries())
import mlx.core as mx
from mlxlearn_device import fit_neighbors
assert 'mlxlearn_device._logistic' not in sys.modules
from mlxlearn_device._neighbors import class_probabilities, regression_predictions
for device in ('gpu', 'cpu'):
    X = mx.array([[0., 0.], [1., 0.], [2., 0.]])
    model = fit_neighbors(X, device=device)
    d, i = model.query(mx.array([[.25, 0.]]), 2)
    p = class_probabilities(i, mx.array([0, 1, 1]), 2, distances=d)
    r = regression_predictions(i, mx.array([1., 3., 5.]), distances=d)
    mx.eval(d, i, p, r)
    assert i.tolist() == [[0, 1]]
    assert abs(p.tolist()[0][0] - .75) < 1e-6
    assert abs(r.item() - 1.5) < 1e-6
    if device == 'gpu':
        large = mx.arange(8192 * 8, dtype=mx.float32).reshape(8192, 8)
        d, i = fit_neighbors(large).query(large[:33], 5)
        mx.eval(d, i)
        assert i[:, 0].tolist() == list(range(33))
assert 'numpy' not in sys.modules and 'sklearn' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", source], env=os.environ.copy(), check=True, timeout=30)


@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_self_is_excluded_even_when_other_squared_distances_overflow(device):
    index = fit_neighbors(mx.array([[1e30], [-1e30]]), device=device)
    for train_block in (None, 1, 2):
        _, indices = index.query(index.X, 1, exclude_self=True, train_block_size=train_block)
        np.testing.assert_array_equal(indices, [[1], [0]])
