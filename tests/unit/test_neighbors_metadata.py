"""Shape-plan caching never caches training-dependent dispatch decisions.

The helpers inspect the Python buffer protocol; NumPy inputs exercise their
sampling without dispatching GPU work or comparing one GPU search to another.
"""

import numpy as np
import pytest

from mlxlearn_device import _neighbors as neighbors


def _training(kind, *, features=128):
    rng = np.random.default_rng(872)
    n = 4096
    train = rng.normal(size=(n, features))
    if kind == "clustered":
        centers = rng.normal(size=(4, features)) * 8
        train = centers[rng.integers(0, 4, n)] + 0.01 * train
    elif kind == "offset":
        train += 10_000
    elif kind == "anisotropic":
        train *= np.logspace(-2, 2, features)
    return train.astype(np.float32)


def test_cached_shape_plan_rechecks_different_training_data_and_mutations():
    ordinary = _training("ordinary", features=32)
    offset = _training("offset", features=32)
    unequal = _training("anisotropic", features=32)
    clustered = _training("clustered", features=32)
    for train, expected in (
        (ordinary, (True, "ordinary")),
        (offset, (True, "offset")),
        (unequal, (False, "ordinary")),
        (clustered, (False, "clustered")),
        (ordinary, (True, "ordinary")),
    ):
        assert neighbors._training_metadata(train) == expected
    # Shape and storage identity stay fixed while the values change.
    ordinary[:] = offset
    assert neighbors._training_metadata(ordinary) == (True, "offset")
    ordinary[:] = unequal
    assert neighbors._training_metadata(ordinary) == (False, "ordinary")


@pytest.mark.parametrize("kind", ["ordinary", "offset", "clustered", "anisotropic"])
@pytest.mark.parametrize("layout", ["row_stride", "feature_stride", "transposed", "negative_stride"])
def test_wide_geometry_reads_strided_views_as_their_logical_coordinates(kind, layout):
    values = _training(kind)
    if layout == "row_stride":
        storage = np.zeros((2 * len(values), values.shape[1]), np.float32)
        storage[::2] = values
        view = storage[::2]
    elif layout == "feature_stride":
        storage = np.zeros((len(values), 2 * values.shape[1]), np.float32)
        storage[:, ::2] = values
        view = storage[:, ::2]
    elif layout == "transposed":
        view = np.ascontiguousarray(values.T).T
    else:
        view = np.ascontiguousarray(values[::-1, ::-1])[::-1, ::-1]
    assert not view.flags.c_contiguous
    assert neighbors._sample_geometry(view) == neighbors._sample_geometry(values)
    np.testing.assert_array_equal(view, values)


@pytest.mark.parametrize("shape", [(0, 8), (1, 8), (8, 0), (3, 1), (17, 1), (3, 7)])
def test_geometry_handles_empty_single_coordinate_and_odd_row_samples(shape):
    values = np.zeros(shape, np.float32)
    assert neighbors._sample_geometry(values) == "ordinary"
    assert neighbors._training_metadata(values) == (False, "ordinary")


def test_unavailable_sample_buffer_retains_conservative_decisions(monkeypatch):
    values = _training("offset")

    def unavailable(_):
        raise BufferError("buffer export unavailable")

    monkeypatch.setattr(neighbors, "memoryview", unavailable, raising=False)
    assert neighbors._sample_geometry(values) == "ordinary"
    assert neighbors._training_metadata(values) == (False, "ordinary")
