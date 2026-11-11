"""Numerical regressions in randomized PCA."""

import numpy as np
import pytest
from scipy import linalg

from mlxlearn._common.config import tuning_context
from mlxlearn.decomposition import PCA


def test_randomized_pca_gram_basis_stays_orthonormal():
    rng = np.random.default_rng(3)
    U, _ = np.linalg.qr(rng.normal(size=(256, 40)))
    V, _ = np.linalg.qr(rng.normal(size=(32_768, 40)))
    U -= U.mean(axis=0)
    scales = 10.0 ** (-np.arange(40) / 8)
    X = ((U * scales) @ V.T).astype(np.float32)
    n_components = 16
    model = PCA(
        n_components=n_components,
        svd_solver="randomized",
        power_iteration_normalizer="LU",
        whiten=True,
        random_state=0,
    )

    transformed = model.fit_transform(X)

    assert model.execution_backend_ == "mlx"
    components = model.components_.astype(np.float64)
    np.testing.assert_allclose(components @ components.T, np.eye(n_components), atol=2e-5)
    # V has orthonormal columns, so the singular values can be computed from
    # the small left factor in float64.
    singular_values = np.linalg.svd(U * scales, compute_uv=False)[:n_components]
    np.testing.assert_allclose(model.singular_values_, singular_values, rtol=1e-4)
    np.testing.assert_allclose(model.transform(X), transformed, atol=2e-4, rtol=2e-4)


def test_randomized_pca_avoids_gram_overflow_on_wide_input():
    rng = np.random.default_rng(42)
    scale = np.float32(1.2e17)
    X = rng.standard_normal((256, 32_768), dtype=np.float32)
    n_components = 32

    model = PCA(n_components=n_components, svd_solver="randomized", random_state=0).fit(X * scale)

    assert model.execution_backend_ == "mlx"
    components = model.components_.astype(np.float64)
    np.testing.assert_allclose(components @ components.T, np.eye(n_components), atol=2e-5)
    centered = X.astype(np.float64) - X.mean(axis=0, dtype=np.float64)
    singular_values = np.sqrt(np.linalg.eigvalsh(centered @ centered.T)[-n_components:][::-1])
    np.testing.assert_allclose(model.singular_values_ / scale, singular_values, rtol=0.06)
    assert np.isfinite(model.explained_variance_).all()
    assert np.isfinite(model.noise_variance_)


@pytest.mark.parametrize(
    ("noise", "randomized_options", "backend", "full_svd"),
    [
        (1e-3, {}, "cpu", False),
        (0.0, {}, "cpu", True),
        (None, {}, "mlx", False),
        (1e-3, {"iterated_power": 1, "power_iteration_normalizer": "QR"}, "mlx", False),
        (1e-3, {"n_oversamples": 12}, "mlx", False),
    ],
)
def test_nearly_full_randomized_sketch_uses_host_only_when_gram_fails(
    noise, randomized_options, backend, full_svd, monkeypatch
):
    rng = np.random.default_rng(34)
    n_samples, n_features, n_components = 256, 600, 192
    X = rng.standard_normal((n_samples, n_features), dtype=np.float32)
    if noise is not None:
        signal = rng.standard_normal((n_samples, 32), dtype=np.float32)
        X = 10 + signal @ rng.standard_normal((32, n_features), dtype=np.float32) + noise * X

    original_svd = linalg.svd
    full_svd_dtypes = []

    def track_svd(matrix, *args, **kwargs):
        if matrix.shape == X.shape:
            full_svd_dtypes.append(matrix.dtype)
        return original_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(linalg, "svd", track_svd)

    with tuning_context(force_backend="mlx"):
        model = PCA(
            n_components=n_components,
            svd_solver="randomized",
            whiten=True,
            random_state=0,
            **randomized_options,
        )
        transformed = model.fit_transform(X)

    assert model.execution_backend_ == backend
    assert model._fit_svd_solver == "randomized"
    assert full_svd_dtypes == ([np.dtype(np.float64)] if full_svd else [])
    components = model.components_.astype(np.float64)
    np.testing.assert_allclose(components @ components.T, np.eye(n_components), atol=2e-5)
    if backend == "cpu":
        centered = X.astype(np.float64) - X.mean(axis=0, dtype=np.float64)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        np.testing.assert_allclose(
            model.singular_values_, singular_values[:n_components], rtol=2e-5, atol=1e-10
        )
        expected_noise = np.mean(singular_values[n_components:] ** 2) / (n_samples - 1)
        np.testing.assert_allclose(model.noise_variance_, expected_noise, rtol=2e-5)
        np.testing.assert_allclose(model.transform(X), transformed, atol=1e-5)
        if noise:
            np.testing.assert_allclose(transformed.std(axis=0, ddof=1), 1, atol=2e-3)


def test_randomized_pca_host_rescales_unnormalized_power_iterations():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 80), dtype=np.float32)
    scale = np.float32(1e10)
    params = {"n_components": 8, "svd_solver": "randomized", "iterated_power": 2, "random_state": 0}
    with tuning_context(force_backend="cpu"):
        model = PCA(**params).fit(X * scale)
        unscaled = PCA(**params).fit(X)

    components = model.components_.astype(np.float64)
    np.testing.assert_allclose(components @ components.T, np.eye(8), atol=2e-5)
    np.testing.assert_allclose(model.singular_values_ / scale, unscaled.singular_values_, rtol=2e-5)
    np.testing.assert_allclose(
        model.explained_variance_ratio_, unscaled.explained_variance_ratio_, rtol=2e-5
    )


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_randomized_pca_recovers_small_residual_variance(backend):
    rng = np.random.default_rng(3)
    n_samples, n_features, rank = 1024, 128, 8
    scores = rng.normal(size=(n_samples, rank))
    directions, _ = np.linalg.qr(rng.normal(size=(n_features, rank)))
    X = (scores @ directions.T + 1e-4 * rng.normal(size=(n_samples, n_features))).astype(np.float32)

    with tuning_context(force_backend=backend):
        model = PCA(n_components=rank, svd_solver="randomized", random_state=0).fit(X)

    centered = X.astype(np.float64) - X.mean(axis=0, dtype=np.float64)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    expected_noise = np.mean(singular_values[rank:] ** 2) / (n_samples - 1)
    np.testing.assert_allclose(model.noise_variance_, expected_noise, rtol=2e-3)
