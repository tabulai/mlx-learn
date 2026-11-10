"""Parity and numerical accuracy tests for ``mlxlearn.decomposition.PCA`` against scikit-learn."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA as SkPCA

from mlxlearn._common.config import tuning_context
from mlxlearn.decomposition import PCA
from mlxlearn.exceptions import SparseInputError, UnsupportedInputError
from mlxlearn.patching._estimators import PCA as PatchedPCA


def _make_well_conditioned(
    n_samples: int = 1024,
    n_features: int = 64,
    *,
    seed: int = 0,
    dtype=np.float64,
    shift: float = 0.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n_samples, min(n_samples, n_features))))
    V, _ = np.linalg.qr(rng.standard_normal((n_features, min(n_samples, n_features))))
    m = min(n_samples, n_features)
    s = np.linspace(50.0, 2.0, m)
    X = (U * s) @ V.T + shift
    return X.astype(dtype)


def _assert_pca_parity(
    mlx_pca: PCA,
    sk_pca: SkPCA,
    X: np.ndarray,
    *,
    rtol: float = 2e-4,
    atol: float = 2e-4,
) -> None:
    assert mlx_pca.n_components_ == sk_pca.n_components_
    assert mlx_pca.n_samples_ == sk_pca.n_samples_
    assert mlx_pca.n_features_in_ == sk_pca.n_features_in_
    assert mlx_pca._fit_svd_solver == sk_pca._fit_svd_solver

    assert mlx_pca.components_.dtype == sk_pca.components_.dtype
    assert mlx_pca.explained_variance_.dtype == sk_pca.explained_variance_.dtype
    assert mlx_pca.explained_variance_ratio_.dtype == sk_pca.explained_variance_ratio_.dtype
    assert mlx_pca.singular_values_.dtype == sk_pca.singular_values_.dtype
    assert mlx_pca.mean_.dtype == sk_pca.mean_.dtype

    np.testing.assert_allclose(mlx_pca.mean_, sk_pca.mean_, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        mlx_pca.singular_values_, sk_pca.singular_values_, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        mlx_pca.explained_variance_, sk_pca.explained_variance_, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        mlx_pca.explained_variance_ratio_,
        sk_pca.explained_variance_ratio_,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        float(mlx_pca.noise_variance_), float(sk_pca.noise_variance_), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(mlx_pca.components_, sk_pca.components_, rtol=rtol, atol=atol)

    Z_mlx = mlx_pca.transform(X[:128])
    Z_sk = sk_pca.transform(X[:128])
    assert Z_mlx.dtype == Z_sk.dtype
    np.testing.assert_allclose(Z_mlx, Z_sk, rtol=rtol, atol=atol)

    if mlx_pca.n_components_ > 0:
        X_rec_mlx = mlx_pca.inverse_transform(Z_mlx)
        X_rec_sk = sk_pca.inverse_transform(Z_sk)
        assert X_rec_mlx.dtype == X_rec_sk.dtype
        np.testing.assert_allclose(X_rec_mlx, X_rec_sk, rtol=rtol, atol=atol)


@pytest.mark.parametrize("svd_solver", ["auto", "full", "covariance_eigh", "randomized", "arpack"])
@pytest.mark.parametrize("whiten", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_pca_solvers_parity(svd_solver, whiten, dtype):
    X = _make_well_conditioned(1200, 48, seed=42, dtype=dtype)
    k = 16
    with tuning_context(force_backend="mlx"):
        mlx_pca = PCA(n_components=k, svd_solver=svd_solver, whiten=whiten, random_state=0).fit(X)
    sk_pca = SkPCA(n_components=k, svd_solver=svd_solver, whiten=whiten, random_state=0).fit(X)
    assert mlx_pca.execution_backend_ == "mlx"
    _assert_pca_parity(mlx_pca, sk_pca, X)


@pytest.mark.parametrize("n_components", [None, 0, 1, 10, 0.75, "mle"])
def test_pca_n_components_variants(n_components):
    X = _make_well_conditioned(1024, 32, seed=7)
    with tuning_context(force_backend="mlx"):
        mlx_pca = PCA(n_components=n_components, svd_solver="full").fit(X)
    sk_pca = SkPCA(n_components=n_components, svd_solver="full").fit(X)
    _assert_pca_parity(mlx_pca, sk_pca, X)


@pytest.mark.parametrize("svd_solver", ["full", "covariance_eigh", "randomized", "arpack"])
@pytest.mark.parametrize("whiten", [False, True])
def test_pca_fit_transform_matches_fit_then_transform_and_sklearn(svd_solver, whiten):
    X = _make_well_conditioned(1024, 48, seed=11)
    with tuning_context(force_backend="mlx"):
        mlx_ft = PCA(n_components=12, svd_solver=svd_solver, whiten=whiten, random_state=3)
        Z_ft = mlx_ft.fit_transform(X)
        Z_t = PCA(n_components=12, svd_solver=svd_solver, whiten=whiten, random_state=3).fit(X).transform(X)
    sk_est = SkPCA(n_components=12, svd_solver=svd_solver, whiten=whiten, random_state=3)
    Z_sk_ft = sk_est.fit_transform(X)
    Z_sk_t = sk_est.transform(X)

    if svd_solver != "randomized":
        np.testing.assert_allclose(Z_ft, Z_t, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(Z_ft, Z_sk_ft, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(Z_t, Z_sk_t, rtol=2e-4, atol=2e-4)


def test_pca_wide_matrix_parity():
    """Wide matrix (n_samples < n_features) across full and randomized solvers."""
    X = _make_well_conditioned(128, 512, seed=19)
    for solver, k in [("full", None), ("full", 32), ("randomized", 24)]:
        with tuning_context(force_backend="mlx"):
            mlx_pca = PCA(n_components=k, svd_solver=solver, random_state=0).fit(X)
        sk_pca = SkPCA(n_components=k, svd_solver=solver, random_state=0).fit(X)
        assert mlx_pca.components_.shape == sk_pca.components_.shape
        # Check row orthonormality of components_ even when k == n_samples (where last SV is 0)
        gram = mlx_pca.components_ @ mlx_pca.components_.T
        np.testing.assert_allclose(gram, np.eye(mlx_pca.n_components_), atol=1e-5)
        np.testing.assert_allclose(
            mlx_pca.singular_values_, sk_pca.singular_values_, rtol=2e-4, atol=2e-4
        )
        np.testing.assert_allclose(
            mlx_pca.transform(X[:32]), sk_pca.transform(X[:32]), rtol=2e-4, atol=2e-4
        )


def test_pca_shifted_data_numerical_stability():
    """Large constant offset (1e4) must not degrade float32 mean or variance accuracy."""
    X = _make_well_conditioned(2048, 32, seed=23, shift=1e4)
    with tuning_context(force_backend="mlx"):
        mlx_pca = PCA(n_components=10, svd_solver="covariance_eigh").fit(X)
    sk_pca = SkPCA(n_components=10, svd_solver="covariance_eigh").fit(X)
    _assert_pca_parity(mlx_pca, sk_pca, X, rtol=1e-3, atol=1e-3)


def test_pca_ill_conditioned_spectrum_deflated_tail_refinement():
    """Ill-conditioned singular values spanning 1e4 are recovered accurately under svd_solver='full'."""
    rng = np.random.default_rng(101)
    n, d = 1024, 32
    U, _ = np.linalg.qr(rng.standard_normal((n, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.logspace(2.0, -2.0, d)  # condition number 1e4
    X = (U * s) @ V.T
    with tuning_context(force_backend="mlx"):
        mlx_pca = PCA(svd_solver="full").fit(X)
    sk_pca = SkPCA(svd_solver="full").fit(X)
    np.testing.assert_allclose(mlx_pca.singular_values_, sk_pca.singular_values_, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(
        np.abs(mlx_pca.components_), np.abs(sk_pca.components_), rtol=2e-3, atol=2e-3
    )


@pytest.mark.parametrize("normalizer", ["auto", "QR", "LU", "none"])
@pytest.mark.parametrize("iterated_power", ["auto", 0, 3])
def test_pca_randomized_normalizers_and_power_iterations(normalizer, iterated_power):
    X = _make_well_conditioned(1024, 64, seed=31)
    kwargs = {
        "n_components": 12,
        "svd_solver": "randomized",
        "power_iteration_normalizer": normalizer,
        "iterated_power": iterated_power,
        "n_oversamples": 12,
        "random_state": 7,
    }
    with tuning_context(force_backend="mlx"):
        mlx_pca = PCA(**kwargs).fit(X)
    sk_pca = SkPCA(**kwargs).fit(X)
    np.testing.assert_allclose(
        mlx_pca.singular_values_, sk_pca.singular_values_, rtol=5e-3, atol=5e-3
    )


def test_pca_score_samples_and_covariance_precision_parity():
    X = _make_well_conditioned(1024, 48, seed=55)
    for whiten in (False, True):
        with tuning_context(force_backend="mlx"):
            mlx_pca = PCA(n_components=16, whiten=whiten).fit(X)
        sk_pca = SkPCA(n_components=16, whiten=whiten).fit(X)

        np.testing.assert_allclose(
            mlx_pca.get_covariance(), sk_pca.get_covariance(), rtol=5e-4, atol=5e-4
        )
        np.testing.assert_allclose(
            mlx_pca.get_precision(), sk_pca.get_precision(), rtol=5e-4, atol=5e-4
        )
        np.testing.assert_allclose(
            mlx_pca.score_samples(X), sk_pca.score_samples(X), rtol=5e-4, atol=5e-4
        )
        assert mlx_pca.score(X) == pytest.approx(sk_pca.score(X), rel=5e-4, abs=5e-4)


def test_pca_copy_false_contract():
    X = _make_well_conditioned(600, 32, seed=61)

    X_full = X.copy()
    PCA(n_components=8, svd_solver="full", copy=False).fit(X_full)
    np.testing.assert_allclose(X_full.mean(axis=0), np.zeros(32), atol=1e-5)

    X_eigh = X.copy()
    PCA(n_components=8, svd_solver="covariance_eigh", copy=False).fit(X_eigh)
    np.testing.assert_array_equal(X_eigh, X)


def test_pca_sparse_and_float32_overflow_layer1_vs_layer2():
    sparse = pytest.importorskip("scipy.sparse")
    X = sparse.csr_matrix(_make_well_conditioned(300, 20, seed=71))

    with pytest.raises(SparseInputError):
        PCA(n_components=5).fit(X)

    patched = PatchedPCA(n_components=5).fit(X)
    assert patched.execution_backend_ == "sklearn"
    sk = SkPCA(n_components=5).fit(X)
    np.testing.assert_allclose(patched.transform(X), sk.transform(X), rtol=1e-5, atol=1e-5)

    X_huge = _make_well_conditioned(500, 20, seed=73) * 1e50
    with pytest.raises(UnsupportedInputError):
        PCA(n_components=5).fit(X_huge)

    patched_huge = PatchedPCA(n_components=5).fit(X_huge)
    assert patched_huge.execution_backend_ == "sklearn"
