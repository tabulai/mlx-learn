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


@pytest.mark.parametrize("estimator", [PCA, PatchedPCA])
def test_pca_float64_spread_below_float32_resolution(estimator):
    X = np.random.default_rng(0).normal(1e8, 0.5, size=(5200, 520))
    reference = SkPCA(n_components=8, svd_solver="full").fit(X)

    model = estimator(n_components=8, whiten=True)
    scores = model.fit_transform(X)

    assert model._fit_svd_solver == "covariance_eigh"
    assert model.execution_backend_ == "cpu"
    np.testing.assert_allclose(model.singular_values_, reference.singular_values_, rtol=1e-9)
    np.testing.assert_allclose(scores.std(axis=0, ddof=1), np.ones(8), atol=1e-6)
    np.testing.assert_allclose(model.transform(X), scores, atol=1e-6)


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


def _make_fast_decay(
    n_samples: int,
    n_features: int,
    *,
    components_per_decade: float,
    rank: int = 60,
    seed: int = 131,
) -> np.ndarray:
    """Float32 data whose singular values fall by a decade every few components."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n_samples, rank)))
    V, _ = np.linalg.qr(rng.standard_normal((n_features, rank)))
    s = 10.0 ** (-np.arange(rank) / components_per_decade)
    return ((U * s) @ V.T).astype(np.float32)


@pytest.mark.parametrize(
    ("shape", "n_components", "components_per_decade", "iterated_power"),
    [
        ((600, 4000), 16, 3.0, "auto"),
        ((600, 4000), 16, 3.0, 3),
        ((600, 4000), 24, 4.0, "auto"),
        ((600, 4000), 24, 4.0, 3),
        ((1500, 600), 24, 4.0, "auto"),
        ((4000, 1500), 16, 3.0, 3),
        ((4000, 1500), 24, 3.0, 3),
    ],
)
def test_pca_randomized_fast_decay_keeps_float32_accuracy(
    shape, n_components, components_per_decade, iterated_power
):
    """A retained spectrum spanning many decades must not cost orthonormality or accuracy.

    Every case takes the direct GPU subspace iteration. Forming ``M.T @ Q`` from an
    unnormalized ``M.T @ Y`` and a triangular factor cancels catastrophically in
    float32 once the subspace is ill-conditioned: each of these cases returned
    non-orthonormal ``components_`` or trailing singular values wrong by 2% to 500%.
    """
    X = _make_fast_decay(*shape, components_per_decade=components_per_decade)
    centered = X.astype(np.float64)
    centered -= centered.mean(axis=0)
    exact = np.linalg.svd(centered, compute_uv=False)[:n_components]
    # Singular values below 1e-5 of the largest are at the float32 noise floor for
    # any solver, scikit-learn's included; hold the rest to float32 accuracy.
    resolved = exact >= 1e-5 * exact[0]
    assert resolved.sum() >= 15

    with tuning_context(force_backend="mlx"):
        mlx_pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            iterated_power=iterated_power,
            random_state=0,
        ).fit(X)
    assert mlx_pca.execution_backend_ == "mlx"

    components = mlx_pca.components_.astype(np.float64)
    np.testing.assert_allclose(components @ components.T, np.eye(n_components), atol=2e-5)
    np.testing.assert_allclose(
        mlx_pca.singular_values_[resolved], exact[resolved], rtol=5e-3
    )


def test_pca_float64_gpu_path_narrows_once_without_scanning(monkeypatch):
    """A finite float64 input reaches the GPU through one cast and no range scan."""
    import mlxlearn.decomposition._pca as pca_module

    scans: list[tuple[int, ...]] = []
    original = pca_module.check_float32_range

    def counting(X, **kwargs):
        scans.append(np.shape(X))
        return original(X, **kwargs)

    monkeypatch.setattr(pca_module, "check_float32_range", counting)
    X = _make_well_conditioned(4096, 64, seed=83)
    with tuning_context(force_backend="mlx"):
        model = PCA(n_components=8).fit(X)
        Z = model.transform(X)
        model.score_samples(X)
        model.inverse_transform(Z)
        PCA(n_components=8).fit_transform(X)
    assert scans == []

    sk = SkPCA(n_components=8).fit(X)
    np.testing.assert_allclose(Z, sk.transform(X), rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_pca_nonfinite_input_matches_sklearn_error(bad, dtype):
    X = _make_well_conditioned(4096, 64, seed=89, dtype=dtype)
    X_bad = X.copy()
    X_bad[1234, 5] = bad
    with tuning_context(force_backend="mlx"):
        with pytest.raises(ValueError, match="NaN|infinity") as mlx_error:
            PCA(n_components=8).fit(X_bad)
        assert not isinstance(mlx_error.value, UnsupportedInputError)
        model = PCA(n_components=8).fit(X)
        for method in (model.transform, model.score_samples):
            with pytest.raises(ValueError, match="NaN|infinity") as mlx_error:
                method(X_bad)
            assert not isinstance(mlx_error.value, UnsupportedInputError)
    with pytest.raises(ValueError) as sk_error:
        SkPCA(n_components=8).fit(X_bad)
    assert str(mlx_error.value).split("\n")[0] == str(sk_error.value).split("\n")[0]


def test_pca_float32_overflow_on_gpu_path_layer1_vs_layer2():
    """Overflow found by the narrowing cast is still a capability error, at any size."""
    X = _make_well_conditioned(4096, 64, seed=97)
    X_huge = X.copy()
    X_huge[100, 3] = 1e60

    with tuning_context(force_backend="mlx"):
        with pytest.raises(UnsupportedInputError):
            PCA(n_components=8).fit(X_huge)
        model = PCA(n_components=8).fit(X)
        for method in (model.transform, model.score_samples):
            with pytest.raises(UnsupportedInputError):
                method(X_huge)
        with pytest.raises(UnsupportedInputError):
            model.inverse_transform(np.full((4096, 8), 1e60))

    # Even with NaN beside it, a finite overflow is reported as the capability error.
    X_both = X_huge.copy()
    X_both[7, 7] = np.nan
    with tuning_context(force_backend="mlx"), pytest.raises(UnsupportedInputError):
        PCA(n_components=8).fit(X_both)

    # Patched estimators hand the same input to scikit-learn's float64 path. Lower
    # the size floor so the fit reaches the GPU route, where the cast finds it.
    sk = SkPCA(n_components=8, svd_solver="randomized", random_state=0).fit(X_huge)
    with tuning_context(pca_randomized_min_work=0):
        with pytest.warns(Warning, match="float32"):
            patched = PatchedPCA(n_components=8, svd_solver="randomized", random_state=0)
            patched.fit(X_huge)
        assert patched.execution_backend_ == "sklearn"
        np.testing.assert_allclose(patched.singular_values_, sk.singular_values_, rtol=1e-12)
        patched = PatchedPCA(n_components=8, svd_solver="randomized", random_state=0)
        out = patched.fit_transform(X_huge)
        assert patched.execution_backend_ == "sklearn"
    np.testing.assert_allclose(out, sk.transform(X_huge), rtol=1e-6, atol=1e-6 * np.abs(out).max())


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("shift", [0.0, 3.0])
@pytest.mark.parametrize("whiten", [False, True])
def test_pca_host_covariance_matches_float64_reference(dtype, shift, whiten):
    """The host covariance route solves in the input dtype without losing the variance.

    The reference is scikit-learn on a float64 copy: its own float32 covariance
    cancels visibly once the mean is a few standard deviations from zero.
    """
    X = _make_well_conditioned(3000, 24, seed=103, dtype=dtype, shift=shift)
    with tuning_context(force_backend="cpu"):
        mlx_pca = PCA(n_components=6, svd_solver="covariance_eigh", whiten=whiten)
        Z = mlx_pca.fit_transform(X)
        Z_again = mlx_pca.transform(X)
    assert mlx_pca.execution_backend_ == "cpu"
    X64 = X.astype(np.float64)
    reference = SkPCA(n_components=6, svd_solver="covariance_eigh", whiten=whiten)
    Z_reference = reference.fit_transform(X64)

    tol = 1e-9 if dtype == np.float64 else 2e-4
    for name in ("mean_", "components_", "explained_variance_", "singular_values_"):
        assert getattr(mlx_pca, name).dtype == dtype, name
    assert Z.dtype == dtype
    np.testing.assert_allclose(mlx_pca.mean_, reference.mean_, rtol=tol, atol=tol)
    np.testing.assert_allclose(mlx_pca.components_, reference.components_, rtol=tol, atol=tol)
    np.testing.assert_allclose(
        mlx_pca.explained_variance_, reference.explained_variance_, rtol=tol, atol=tol
    )
    np.testing.assert_allclose(
        mlx_pca.explained_variance_ratio_, reference.explained_variance_ratio_, rtol=tol, atol=tol
    )
    np.testing.assert_allclose(
        float(mlx_pca.noise_variance_), float(reference.noise_variance_), rtol=tol, atol=tol
    )
    np.testing.assert_allclose(Z, Z_reference, rtol=tol, atol=tol)
    np.testing.assert_allclose(Z_again, Z_reference, rtol=tol, atol=tol)


def test_pca_host_covariance_dominant_mean_stays_accurate_in_float32():
    """A mean far larger than the spread must not cancel the variance away."""
    X64 = _make_well_conditioned(3000, 24, seed=107, shift=1e4)
    X32 = X64.astype(np.float32)
    with tuning_context(force_backend="cpu"):
        mlx_pca = PCA(n_components=6, svd_solver="covariance_eigh").fit(X32)
    assert mlx_pca.execution_backend_ == "cpu"
    reference = SkPCA(n_components=6, svd_solver="full").fit(X32.astype(np.float64))
    np.testing.assert_allclose(mlx_pca.singular_values_, reference.singular_values_, rtol=1e-3)
    np.testing.assert_allclose(mlx_pca.mean_, reference.mean_, rtol=1e-6)


def test_pca_host_mean_and_gram_guards():
    from mlxlearn._solvers.pca import _host_mean_and_gram

    rng = np.random.default_rng(109)
    X = (rng.standard_normal((1_000_000, 3)) + 2.0).astype(np.float32)
    mean, gram = _host_mean_and_gram(X)
    exact = X.astype(np.float64).mean(axis=0)
    # One float32 sweep over a million rows would be off by about 1e-5 here.
    np.testing.assert_allclose(mean, exact, rtol=0, atol=5e-7)
    centered = X.astype(np.float64) - exact
    np.testing.assert_allclose(gram, centered.T @ centered, rtol=2e-5, atol=2e-5 * 1e6)

    for bad in (np.nan, np.inf, -np.inf):
        X_bad = X[:1000].copy()
        X_bad[17, 1] = bad
        assert _host_mean_and_gram(X_bad) is None
    # Alternating signs keep the centered values large enough to overflow the Gram.
    X_overflow = np.full((50, 2), 1e30, dtype=np.float32)
    X_overflow[::2] *= -1
    assert _host_mean_and_gram(X_overflow) is None

    # A mean that dominates the spread is centered explicitly, not cancelled away.
    shifted = (X[:20_000].astype(np.float64) + 1e4).astype(np.float32)
    mean, gram = _host_mean_and_gram(shifted)
    exact = shifted.astype(np.float64).mean(axis=0)
    centered = shifted.astype(np.float64) - exact
    np.testing.assert_allclose(mean, exact, rtol=0, atol=1e-6)
    np.testing.assert_allclose(gram, centered.T @ centered, rtol=1e-4, atol=1e-4 * 20_000)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("svd_solver", ["covariance_eigh", "full", "randomized", "arpack"])
def test_pca_host_path_rejects_nonfinite_input(bad, svd_solver):
    X = _make_well_conditioned(600, 24, seed=113)
    X[5, 5] = bad
    with tuning_context(force_backend="cpu"), pytest.raises(ValueError, match="NaN|infinity"):
        PCA(n_components=6, svd_solver=svd_solver).fit(X)


@pytest.mark.parametrize("iterated_power", [0, 1, 2, "auto"])
@pytest.mark.parametrize("shape", [(1500, 600), (600, 4000)], ids=["tall", "wide"])
def test_pca_randomized_survives_large_scale_float32_data(shape, iterated_power):
    """Singular values in the hundreds of thousands must not overflow a float32 Gram matrix.

    With ``iterated_power <= 2`` the power iteration is not normalized, so the block
    grows by ``sigma_max ** 2`` per step and its Gram matrix squares that again;
    this raised "array must not contain infs or NaNs" on ordinary unscaled data.
    """
    # Rank 64 from thin factors. A square QR of this size is avoided on purpose:
    # Accelerate's QR overruns a heap buffer at 600 x 600 on macOS 26 (it faults
    # under guard malloc with NumPy alone), which crashed this process at random.
    rng = np.random.default_rng(127)
    U, _ = np.linalg.qr(rng.standard_normal((shape[0], 64)))
    V, _ = np.linalg.qr(rng.standard_normal((shape[1], 64)))
    X = ((U * np.linspace(50.0, 2.0, 64)) @ V.T * 1e4).astype(np.float32)
    with tuning_context(force_backend="mlx"):
        mlx_pca = PCA(
            n_components=8, svd_solver="randomized", iterated_power=iterated_power, random_state=0
        ).fit(X)
    assert mlx_pca.execution_backend_ == "mlx"
    sk_pca = SkPCA(
        n_components=8, svd_solver="randomized", iterated_power=iterated_power, random_state=0
    ).fit(X)
    np.testing.assert_allclose(mlx_pca.singular_values_, sk_pca.singular_values_, rtol=1e-3)
    components = mlx_pca.components_.astype(np.float64)
    np.testing.assert_allclose(components @ components.T, np.eye(8), atol=2e-5)


@pytest.mark.parametrize(
    ("shape", "params", "solver", "direct_backend", "patched_backend"),
    [
        # covariance_eigh: narrow and under 8e6 elements stays on the host ...
        ((20_000, 64), {"n_components": 8}, "covariance_eigh", "cpu", "sklearn"),
        # ... while 512 features or 8e6 elements is enough for the GPU on its own.
        ((5_200, 520), {"n_components": 8}, "covariance_eigh", "mlx", "mlx"),
        ((130_000, 64), {"n_components": 8}, "covariance_eigh", "mlx", "mlx"),
        # randomized: 8e6 elements.
        ((3_000, 1_200), {"n_components": 8}, "randomized", "cpu", "sklearn"),
        ((8_000, 1_100), {"n_components": 8}, "randomized", "mlx", "mlx"),
        # full and arpack are served on the host at every size.
        ((2_000, 300), {}, "full", "cpu", "sklearn"),
        ((2_000, 300), {"n_components": 8, "svd_solver": "arpack"}, "arpack", "cpu", "sklearn"),
    ],
)
def test_pca_routing_follows_the_measured_crossovers(
    shape, params, solver, direct_backend, patched_backend
):
    rng = np.random.default_rng(137)
    X = rng.standard_normal(shape, dtype=np.float32) * np.linspace(3.0, 1.0, shape[1], dtype=np.float32)
    sk = SkPCA(random_state=0, **params).fit(X)
    assert sk._fit_svd_solver == solver

    for cls, expected in ((PCA, direct_backend), (PatchedPCA, patched_backend)):
        model = cls(random_state=0, **params).fit(X)
        assert model._fit_svd_solver == solver
        assert model.execution_backend_ == expected, cls.__name__
        np.testing.assert_allclose(
            model.explained_variance_, sk.explained_variance_, rtol=2e-3, atol=2e-3
        )
        Z = model.transform(X[:256])
        assert Z.dtype == np.float32
        np.testing.assert_allclose(np.abs(Z), np.abs(sk.transform(X[:256])), rtol=5e-2, atol=5e-2)


def test_pca_patched_small_fit_pays_no_mlx_input_work(monkeypatch):
    """A fit handed to scikit-learn for its size is not scanned or cast first."""
    import mlxlearn.decomposition._pca as pca_module

    def unexpected(*args, **kwargs):
        raise AssertionError("small patched fits must not prepare MLX input")

    monkeypatch.setattr(pca_module, "check_float32_range", unexpected)
    monkeypatch.setattr(pca_module, "cast_float32_checked", unexpected)
    X = _make_well_conditioned(4096, 64, seed=139)
    for params in ({"n_components": 8}, {"n_components": 8, "svd_solver": "randomized"}, {}):
        model = PatchedPCA(random_state=0, **params)
        Z = model.fit_transform(X)
        assert model.execution_backend_ == "sklearn"
        sk = SkPCA(random_state=0, **params)
        np.testing.assert_allclose(Z, sk.fit_transform(X), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_pca_host_inference_rejects_nonfinite_input(dtype, bad):
    """The host route settles finiteness with one sum of squares; it must still reject."""
    X = _make_well_conditioned(600, 24, seed=149, dtype=dtype)
    with tuning_context(force_backend="cpu"):
        model = PCA(n_components=6).fit(X)
        assert model.execution_backend_ == "cpu"
        X_bad = X.copy()
        X_bad[10, 3] = bad  # an even row, so the strided view below keeps it
        for method in (model.transform, model.score_samples, model.score):
            with pytest.raises(ValueError, match="NaN|infinity") as error:
                method(X_bad)
            assert not isinstance(error.value, UnsupportedInputError)
        with pytest.raises(ValueError, match="NaN|infinity"):
            model.inverse_transform(np.full((4, 6), bad, dtype=dtype))
        # Fortran-ordered and strided inputs take the exact checks instead.
        np.testing.assert_allclose(
            model.transform(np.asfortranarray(X)), model.transform(X), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            model.transform(X[::2]), model.transform(np.ascontiguousarray(X[::2])), rtol=0, atol=0
        )
        with pytest.raises(ValueError, match="NaN|infinity"):
            model.transform(X_bad[::2])


def test_pca_host_inference_float32_capability_and_large_finite_values():
    X = _make_well_conditioned(600, 24, seed=151)
    with tuning_context(force_backend="cpu"):
        model = PCA(n_components=6).fit(X)
        X_huge = X.copy()
        X_huge[5, 5] = 1e60  # finite in float64, infinite in float32
        with pytest.raises(UnsupportedInputError):
            model.transform(X_huge)
        with pytest.raises(UnsupportedInputError):
            PCA(n_components=6).fit(X_huge)
        # Squares overflow float32 but the values themselves are legal float32.
        X_big = (X * 1e25).astype(np.float32)
        Z = model.transform(X_big)
        assert Z.dtype == np.float64 and bool(np.isfinite(Z).all())
