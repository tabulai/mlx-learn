"""Numerical regressions for PCA centering and host solvers."""

import numpy as np
import pytest
from sklearn.decomposition import PCA as SkPCA

from mlxlearn._common.config import tuning_context
from mlxlearn.decomposition import PCA
from mlxlearn.exceptions import UnsupportedInputError


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


@pytest.mark.parametrize(
    ("fit_dtype", "whiten", "strided", "offset"),
    [
        (np.float64, False, False, 1000),
        (np.float64, True, True, 1000),
        (np.float32, False, False, 0.25),
    ],
)
def test_gpu_fitted_float64_transform_keeps_precision_without_cast(
    fit_dtype, whiten, strided, offset, monkeypatch
):
    import mlxlearn.decomposition._pca as pca_module
    from mlxlearn._solvers import pca as pca_solver

    rng = np.random.default_rng(7)
    X_train = rng.normal(offset, 1, (2048, 64)).astype(fit_dtype)
    with tuning_context(force_backend="mlx"):
        model = PCA(n_components=8, whiten=whiten).fit(X_train)

    assert model.execution_backend_ == "mlx"
    original_cast = pca_module.cast_float32_checked
    cast_calls = []

    def track_cast(X):
        cast_calls.append(X.shape)
        return original_cast(X)

    monkeypatch.setattr(pca_module, "cast_float32_checked", track_cast)
    # Large offsets erase the variation after narrowing. A moderate offset
    # exposes a float32 projected mean in the mixed-dtype host projection.
    queries = model.mean_.astype(np.float64) + rng.normal(0, 2e-6, (2048, 64))
    X = queries[::2] if strided else queries[:1024]
    if fit_dtype == np.float64:
        # Exercise the precision guard independently of the small-batch cutoff.
        monkeypatch.setattr(pca_solver, "_PCA_FLOAT64_GPU_TRANSFORM_MIN_WORK", X.size)
    original = X.copy()
    expected = (original - model.mean_) @ model.components_.T
    if whiten:
        expected /= np.sqrt(model.explained_variance_)

    scores = model.transform(X)
    assert scores.dtype == np.float64
    assert cast_calls == []
    np.testing.assert_allclose(scores, expected, rtol=2e-6, atol=1e-11)
    np.testing.assert_array_equal(X, original)


def test_gpu_float64_projection_checks_original_range_when_centering():
    X_train = np.random.default_rng(9).standard_normal((2048, 64))
    X_train[:, 0] = 2e38
    with tuning_context(force_backend="mlx"):
        model = PCA(n_components=8).fit(X_train)

    assert model.execution_backend_ == "mlx"
    X = X_train[:1024].copy()
    # The original values fit float32 even though subtracting the mean overflows.
    X[:, 0] = -2e38
    expected = (X - model.mean_) @ model.components_.T
    np.testing.assert_allclose(model.transform(X), expected, rtol=2e-6, atol=1e-6)

    # The centered value fits float32, but the original value exceeds its range.
    X[0, 0] = 4e38
    with pytest.raises(UnsupportedInputError, match="float32"):
        model.transform(X)

    X[0, 0] = 2e38
    for value, message in ((np.nan, "NaN"), (np.inf, "infinity")):
        X[0, 1] = value
        with pytest.raises(ValueError, match=message) as error:
            model.transform(X)
        assert not isinstance(error.value, UnsupportedInputError)


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
