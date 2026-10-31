"""Working-set kernel batches preserve row ordering and precision."""

import numpy as np
import pytest

from mlxlearn._kernels.kernels import KernelSpec, build_kernel_rows


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
@pytest.mark.parametrize("features", [7, 32, 33, 129])
def test_batch_rows_match_separate_rows(backend, kernel, features):
    rng = np.random.default_rng(809)
    data = rng.normal(size=(37, features)).astype(np.float32)
    spec = KernelSpec(kernel, gamma=0.04, coef0=-0.1, degree=3)
    source = build_kernel_rows(data, spec, backend=backend, center=np.full(features, 0.25, np.float32))
    ids = [36, 0, 15, 15, 7]
    batch = source.rows(ids)
    reference = np.stack([source.row(index) for index in ids])
    assert batch.shape == (len(ids), len(data))
    assert batch.dtype == np.float32
    if backend == "mlx":
        np.testing.assert_array_equal(batch, reference)
    else:
        np.testing.assert_allclose(batch, reference, rtol=8e-6, atol=4e-5)


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_batch_rows_validate_indices_and_handle_empty_batch(backend):
    data = np.arange(28, dtype=np.float32).reshape(7, 4)
    source = build_kernel_rows(data, KernelSpec("rbf"), backend=backend)
    assert source.rows([]).shape == (0, 7)
    for indices in ([-1], [7], [2, 7]):
        with pytest.raises(IndexError):
            source.rows(indices)
    with pytest.raises(TypeError):
        source.rows([1.5])


def test_batch_direct_rbf_keeps_shifted_duplicates_exact():
    rng = np.random.default_rng(23)
    data = (rng.normal(size=(43, 35)) + 1e5).astype(np.float32)
    data[7] = data[2]
    source = build_kernel_rows(data, KernelSpec("rbf", gamma=0.2), backend="mlx")
    batch = source.rows([2, 7])
    np.testing.assert_array_equal(batch[:, [2, 7]], np.ones((2, 2), np.float32))
    reference = np.exp(-0.2 * np.sum((data[2].astype(float) - data.astype(float))**2, axis=1))
    np.testing.assert_allclose(batch[0], reference, rtol=4e-6, atol=1e-7)
