"""Numerical regressions for PCA centering and host solvers."""

import numpy as np
import pytest
from sklearn.decomposition import PCA as SkPCA

from mlxlearn._common.config import tuning_context
from mlxlearn.decomposition import PCA


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_covariance_checks_centering_per_feature(backend):
    # The second feature has a large offset and unit spread. The first feature's
    # high variance masks that offset if centering is tested on the total spread.
    X = np.tile(
        np.array(
            [[1e4, 1e4 + 1], [-1e4, 1e4 + 1], [1e4, 1e4 - 1], [-1e4, 1e4 - 1]],
            dtype=np.float32,
        ),
        (4096, 1),
    )
    reference = SkPCA(n_components=2, svd_solver="full").fit(X.astype(np.float64))
    with tuning_context(force_backend=backend):
        model = PCA(n_components=2, svd_solver="covariance_eigh").fit(X)

    assert model.execution_backend_ == backend
    np.testing.assert_allclose(model.explained_variance_, reference.explained_variance_, rtol=1e-5)


@pytest.mark.parametrize("solver", ["covariance_eigh", "full"])
@pytest.mark.parametrize(("dtype", "offset"), [(np.float32, 1e7), (np.float64, 1e13)])
def test_host_whitened_projection_centers_large_offsets(solver, dtype, offset):
    X = np.random.default_rng(0).normal(offset, 1, (2048, 64)).astype(dtype)
    with tuning_context(force_backend="cpu"):
        model = PCA(n_components=8, svd_solver=solver, whiten=True)
        fitted_scores = model.fit_transform(X)
        scores = model.transform(X)

    X_centered = X.astype(np.float64) - model.mean_
    expected = (X_centered @ model.components_.T) / np.sqrt(model.explained_variance_)
    np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fitted_scores, scores, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scores.std(axis=0, ddof=1), np.ones(8), rtol=2e-3)


def test_gpu_whitened_projection_centers_large_float32_offset():
    residual = np.random.default_rng(1).standard_normal((1024, 64), dtype=np.float32)
    X = np.concatenate([1e7 + residual, 1e7 - residual])
    with tuning_context(force_backend="mlx"):
        model = PCA(n_components=8, svd_solver="covariance_eigh", whiten=True)
        scores = model.fit_transform(X)

    assert model.execution_backend_ == "mlx"
    np.testing.assert_allclose(model.transform(X), scores, atol=2e-5)
    np.testing.assert_allclose(scores.std(axis=0, ddof=1), np.ones(8), atol=2e-5)


@pytest.mark.parametrize("scale", [1e17, 1e18])
def test_host_randomized_total_variance_survives_float32_overflow(scale):
    X = np.random.default_rng(0).standard_normal((1024, 128), dtype=np.float32)
    X *= np.float32(scale)
    with tuning_context(force_backend="cpu"):
        model = PCA(n_components=8, svd_solver="randomized", random_state=0).fit(X)

    total_variance = X.var(axis=0, dtype=np.float64, ddof=1).sum()
    explained = model.explained_variance_.astype(np.float64)
    expected_noise = (total_variance - explained.sum()) / (X.shape[1] - model.n_components_)
    np.testing.assert_allclose(model.explained_variance_ratio_, explained / total_variance, rtol=1e-6)
    np.testing.assert_allclose(model.noise_variance_, expected_noise, rtol=1e-6)
