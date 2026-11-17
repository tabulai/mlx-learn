"""Regression tests for batch-independent PCA log-likelihoods."""

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from mlxlearn._common.config import tuning_context
from mlxlearn.decomposition import PCA


def test_low_noise_scores_match_fitted_covariance_at_every_batch_size():
    rng = np.random.default_rng(3)
    n_samples, n_features = 8000, 1024
    latent = rng.normal(size=(n_samples, 8))
    basis, _ = np.linalg.qr(rng.normal(size=(n_features, 8)))
    signal = latent @ basis.T
    X = (signal + 1e-4 * rng.normal(size=signal.shape)).astype(np.float32)
    # Exercise normal randomized routing, including scoring batch sizes on
    # both sides of the dense-precision and GPU cutoffs.
    model = PCA(n_components=8, svd_solver="randomized", random_state=0).fit(X)
    assert model.execution_backend_ == "mlx"
    assert 0 < model.noise_variance_ < 1e-7

    nv = float(model.noise_variance_)
    ev = model.explained_variance_.astype(np.float64)
    V = model.components_.astype(np.float64)
    cov = (V.T * (ev - nv)) @ V + nv * np.eye(n_features)
    factor = cholesky(cov, lower=True)
    coords = solve_triangular(factor, (X[:8].astype(np.float64) - model.mean_).T, lower=True)
    expected = -0.5 * (
        np.sum(coords**2, axis=0)
        + 2 * np.sum(np.log(factor.diagonal()))
        + n_features * np.log(2 * np.pi)
    )

    for batch_size in (8, 32, 128):
        np.testing.assert_allclose(
            model.score_samples(X[:batch_size])[:8], expected, atol=1e-3, rtol=0
        )


def test_well_conditioned_scores_agree_across_cpu_and_gpu_batch_sizes():
    X = np.random.default_rng(5).standard_normal((2048, 64), dtype=np.float32)
    with tuning_context(force_backend="mlx"):
        model = PCA(n_components=8, svd_solver="covariance_eigh").fit(X)
    assert model.execution_backend_ == "mlx"
    np.testing.assert_allclose(
        model.score_samples(X[:4]), model.score_samples(X[:1024])[:4], atol=5e-5, rtol=0
    )
