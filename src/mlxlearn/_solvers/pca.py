"""Metal GPU and Unified-Memory Accelerate solvers for Principal Component Analysis.

This module is part of the pure solver layer (``mlxlearn._solvers``): it does not
import ``scikit-learn`` or read global configuration. All solver parameters and
tuning thresholds are received explicitly via :class:`PCAConfig`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from .._common.arrays import to_numpy
from .._common.preparation import cast_float32_checked
from .._kernels.pca_input import pca_stats, prepare_pca_centered


def ensure_f32_2d(array: np.ndarray) -> np.ndarray:
    """Return ``array`` as C-contiguous float32, narrowing large float64 input in parallel."""
    if array.dtype == np.float32:
        return np.asarray(array, dtype=np.float32, order="C")
    return cast_float32_checked(array)[0]


def to_mlx_f32(array: np.ndarray) -> mx.array:
    return mx.array(np.asarray(array, dtype=np.float32, order="C"))


def borrow_mlx_f32(c_f32_array: np.ndarray) -> mx.array:
    return mx.asarray(c_f32_array) if hasattr(mx, "asarray") else mx.array(c_f32_array)

__all__ = [
    "PCAConfig",
    "PCASolution",
    "choose_svd_solver",
    "pca_inverse_transform",
    "pca_score_samples",
    "pca_transform",
    "solve_pca",
]

SvdSolver = Literal["auto", "full", "covariance_eigh", "arpack", "randomized"]
PowerNormalizer = Literal["auto", "QR", "LU", "none"]


@dataclass(frozen=True)
class PCAConfig:
    """Explicit solver configuration for :func:`solve_pca`."""

    n_components: int | float | Literal["mle"] | None = None
    whiten: bool = False
    svd_solver: SvdSolver = "auto"
    tol: float = 0.0
    iterated_power: int | Literal["auto"] = "auto"
    n_oversamples: int = 10
    power_iteration_normalizer: PowerNormalizer = "auto"
    seed: int = 0
    prefer_gpu: bool = True


@dataclass(frozen=True)
class PCASolution:
    """Fitted state returned by :func:`solve_pca`."""

    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray
    mean: np.ndarray
    n_components: int
    n_samples: int
    n_features: int
    noise_variance: float | np.floating
    fit_svd_solver: str
    used_gpu: bool
    transformed: np.ndarray | None = None
    device_mean: mx.array | None = None
    device_components: mx.array | None = None


def _svd_flip(
    u: np.ndarray | None,
    v: np.ndarray,
    *,
    u_based_decision: bool = False,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Deterministic sign correction for SVD / eigendecomposition outputs."""
    if v.size == 0:
        return u, v
    if u_based_decision and u is not None:
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        shift = np.arange(u.shape[1])
        indices = max_abs_cols * u.shape[1] + shift
        signs = np.sign(np.take(np.reshape(u, (-1,)), indices, axis=0))
    else:
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        shift = np.arange(v.shape[0])
        indices = max_abs_rows + shift * v.shape[1]
        signs = np.sign(np.take(np.reshape(v, (-1,)), indices, axis=0))
    signs[signs == 0] = 1
    v = v * signs[:, np.newaxis]
    if u is not None:
        u = u * signs
    return u, v


def _assess_dimension(spectrum: np.ndarray, rank: int, n_samples: int) -> float:
    """Compute Minka's log-likelihood for a given rank on a non-increasing spectrum."""
    n_features = spectrum.shape[0]
    if not 1 <= rank < n_features:
        raise ValueError(f"the tested rank should be in [1, n_features - 1], got {rank}")

    eps = 1e-15
    if spectrum[rank - 1] < eps:
        return -np.inf

    pu = -rank * math.log(2.0)
    for i in range(1, rank + 1):
        pu += (
            math.lgamma((n_features - i + 1) / 2.0)
            - math.log(math.pi) * (n_features - i + 1) / 2.0
        )

    pl = float(np.sum(np.log(spectrum[:rank])))
    pl = -pl * n_samples / 2.0

    v = max(eps, float(np.sum(spectrum[rank:]) / (n_features - rank)))
    pv = -math.log(v) * n_samples * (n_features - rank) / 2.0

    m = n_features * rank - rank * (rank + 1.0) / 2.0
    pp = math.log(2.0 * math.pi) * (m + rank) / 2.0

    pa = 0.0
    spectrum_ = spectrum.astype(np.float64, copy=True)
    spectrum_[rank:n_features] = v
    for i in range(rank):
        si = float(spectrum[i])
        inv_si = 1.0 / float(spectrum_[i])
        for j in range(i + 1, n_features):
            diff_term = (si - float(spectrum[j])) * (1.0 / float(spectrum_[j]) - inv_si)
            if diff_term <= 0.0:
                diff_term = eps
            pa += math.log(diff_term) + math.log(n_samples)

    return float(pu + pl + pv + pp - pa / 2.0 - rank * math.log(n_samples) / 2.0)


def _infer_dimension(spectrum: np.ndarray, n_samples: int) -> int:
    """Infer intrinsic dimension from a spectrum via Minka's MLE."""
    ll = np.empty(spectrum.shape[0], dtype=np.float64)
    ll[0] = -np.inf
    for rank in range(1, spectrum.shape[0]):
        ll[rank] = _assess_dimension(spectrum, rank, n_samples)
    return int(ll.argmax())


def choose_svd_solver(
    n_samples: int,
    n_features: int,
    n_components: int | float | Literal["mle"] | None,
    svd_solver: SvdSolver,
) -> Literal["full", "covariance_eigh", "arpack", "randomized"]:
    """Resolve ``svd_solver='auto'`` using scikit-learn's exact dispatch rules."""
    if svd_solver != "auto":
        return svd_solver
    if n_features <= 1_000 and n_samples >= 10 * n_features:
        return "covariance_eigh"
    if max(n_samples, n_features) <= 500 or n_components == "mle":
        return "full"
    n_comp_eff = min(n_samples, n_features) if n_components is None else n_components
    if isinstance(n_comp_eff, (int, np.integer)) and 1 <= n_comp_eff < 0.8 * min(
        n_samples, n_features
    ):
        return "randomized"
    return "full"


def _resolve_full_n_components(
    n_components: int | float | Literal["mle"] | None,
    explained_variance: np.ndarray,
    explained_variance_ratio: np.ndarray,
    n_samples: int,
    n_features: int,
) -> int:
    """Resolve ``n_components`` for ``full`` and ``covariance_eigh`` solvers."""
    if n_components is None:
        return min(n_samples, n_features)
    if n_components == "mle":
        return _infer_dimension(explained_variance, n_samples)
    if isinstance(n_components, (float, np.floating)) and 0.0 < float(n_components) < 1.0:
        ratio_cumsum = np.cumsum(explained_variance_ratio)
        return min(
            int(np.searchsorted(ratio_cumsum, n_components, side="right") + 1),
            min(n_samples, n_features),
        )
    return int(n_components)


def _complete_orthonormal_rows(Vt_valid: np.ndarray, target_rows: int, n_features: int) -> np.ndarray:
    """Extend an ``(r, n_features)`` row-orthonormal matrix to ``(target_rows, n_features)``."""
    r = Vt_valid.shape[0]
    if r >= target_rows:
        return Vt_valid[:target_rows]
    out = np.zeros((target_rows, n_features), dtype=Vt_valid.dtype)
    if r > 0:
        out[:r] = Vt_valid
    cur = r
    for col in range(n_features):
        if cur >= target_rows:
            break
        vec = np.zeros(n_features, dtype=np.float64)
        vec[col] = 1.0
        if cur > 0:
            basis = out[:cur].astype(np.float64, copy=False)
            vec -= basis.T @ (basis @ vec)
            vec -= basis.T @ (basis @ vec)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-7:
            out[cur] = (vec / norm).astype(Vt_valid.dtype, copy=False)
            cur += 1
    return out


def _transform_from_centered(
    X_c_np: np.ndarray | None,
    X_c_mx: mx.array | None,
    Vt: np.ndarray,
    explained_variance: np.ndarray,
    whiten: bool,
    out_dtype: np.dtype,
    prefer_gpu: bool,
) -> np.ndarray:
    """Compute ``fit_transform`` output consistent with ``transform``."""
    n_samples = X_c_mx.shape[0] if X_c_mx is not None else int(X_c_np.shape[0])  # type: ignore[union-attr]
    k = Vt.shape[0]
    if k == 0:
        return np.zeros((n_samples, 0), dtype=out_dtype)

    if prefer_gpu and X_c_mx is not None and X_c_mx.size >= 16_384:
        V_T = Vt.T.astype(np.float32, copy=True)
        if whiten:
            scale = np.sqrt(explained_variance.astype(np.float64, copy=False))
            min_scale = float(np.finfo(out_dtype).eps)
            scale = np.where(scale < min_scale, min_scale, scale)
            V_T = (V_T / scale.astype(np.float32)).astype(np.float32, copy=False)
        V_mx = to_mlx_f32(V_T)
        Z_mx = X_c_mx @ V_mx
        mx.eval(Z_mx)
        return to_numpy(Z_mx).astype(out_dtype, copy=False)

    if X_c_np is None:
        X_c_np = to_numpy(X_c_mx).astype(out_dtype, copy=False)  # type: ignore[arg-type]
    Z = (X_c_np @ Vt.T).astype(out_dtype, copy=False)
    if whiten:
        scale = np.sqrt(explained_variance)
        min_scale = np.finfo(scale.dtype).eps
        scale = np.where(scale < min_scale, min_scale, scale)
        Z = Z / scale
    return Z.astype(out_dtype, copy=False)


#: Rows whose column sums are taken in the input dtype before the partial sums are
#: accumulated in float64. Short float32 runs keep the column mean at float32
#: accuracy; one float32 sweep over a million rows loses about three digits.
_HOST_SUM_BLOCK_ROWS = 4096
#: Rows sampled to guess, before any Gram matrix exists, whether the mean dominates.
_HOST_SPREAD_SAMPLE_ROWS = 256


def _host_column_sum(X: np.ndarray) -> np.ndarray:
    """Column sums of ``X`` in float64, from short BLAS sweeps in the input dtype."""
    n_samples, n_features = X.shape
    ones = np.ones(min(n_samples, _HOST_SUM_BLOCK_ROWS), dtype=X.dtype)
    column_sum = np.zeros(n_features, dtype=np.float64)
    for start in range(0, n_samples, _HOST_SUM_BLOCK_ROWS):
        block = X[start : start + _HOST_SUM_BLOCK_ROWS]
        column_sum += ones[: block.shape[0]] @ block
    return column_sum


def _host_sum_of_squares(A: np.ndarray) -> float:
    """Sum of squares of a C-contiguous array, accumulated in float64 without a copy."""
    flat = A.reshape(-1)
    total = 0.0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for start in range(0, flat.size, 1 << 16):
            chunk = flat[start : start + (1 << 16)]
            total += float(np.dot(chunk, chunk))
    return total


def _host_mean_and_gram(X: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Column mean (float64) and centered Gram matrix of ``X`` on the host.

    The usual route is ``X.T @ X - n * outer(mean, mean)``: it reads ``X`` twice,
    once for the column sums and once for the symmetric rank-k update, and
    allocates nothing the size of ``X``. It is the formulation scikit-learn's
    ``covariance_eigh`` uses, in the input dtype, so float64 input keeps float64
    accuracy.

    That subtraction cancels the digits that carry the variance once the mean
    dominates the spread, and the error it leaves is first order in the mean, so
    float32 gets a tight bound and float64 a loose one. Past the bound the Gram
    matrix is taken from an explicitly centered copy instead, whose own column
    sums then refine the mean: sums of values near 1e4 carry an absolute error
    that sums of their small residuals do not. A small row sample predicts which
    route applies, so shifted data does not pay for a discarded rank-k update; the
    exact test on the finished Gram matrix still decides.

    Returns ``None`` when the Gram diagonal is not finite: ``X`` holds NaN or
    infinity, or values whose squares overflow the dtype. The caller then
    validates ``X`` explicitly.
    """
    n_samples = X.shape[0]
    limit = 10.0 if X.dtype == np.float32 else 1e6
    # NumPy 2.2 with Accelerate raises spurious floating-point flags from matmul on
    # finite data; finiteness is read off the sums and the Gram diagonal instead.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        column_sum = _host_column_sum(X)
        if not bool(np.isfinite(column_sum).all()):
            return None
        mean = column_sum / n_samples
        mean_ss = n_samples * float(np.sum(mean * mean))

        step = max(1, n_samples // _HOST_SPREAD_SAMPLE_ROWS)
        sample = X[::step].astype(np.float64) - mean
        spread_guess = float(np.sum(sample * sample)) * (n_samples / sample.shape[0])
        if mean_ss <= limit * max(spread_guess, 1e-12):
            gram = X.T @ X
            trace = float(np.sum(np.diagonal(gram), dtype=np.float64))
            if not math.isfinite(trace):
                return None
            if mean_ss <= limit * max(trace - mean_ss, 1e-12):
                gram -= (n_samples * np.outer(mean, mean)).astype(gram.dtype, copy=False)
                return mean, gram

        shift = mean.astype(X.dtype, copy=False)
        centered = X - shift
        residual = _host_column_sum(centered) / n_samples
        gram = centered.T @ centered
        if not math.isfinite(float(np.sum(np.diagonal(gram), dtype=np.float64))):
            return None
        gram -= (n_samples * np.outer(residual, residual)).astype(gram.dtype, copy=False)
        mean = shift.astype(np.float64) + residual
    return mean, gram


def _solve_cpu_full_or_eigh(
    X: np.ndarray,
    config: PCAConfig,
    fit_svd_solver: Literal["full", "covariance_eigh"],
    out_dtype: np.dtype,
    return_transformed: bool,
) -> PCASolution:
    """Host path for ``full`` and ``covariance_eigh``, in the dtype of ``X``.

    Validates finiteness itself: ``covariance_eigh`` reads it off the Gram diagonal
    it needs anyway, and everything else scans ``X`` before centering it.
    """
    n_samples, n_features = X.shape
    X_c: np.ndarray | None = None
    gram = _host_mean_and_gram(X) if fit_svd_solver == "covariance_eigh" else None
    if gram is None:
        if not bool(np.isfinite(X).all()):
            raise ValueError("Input X contains NaN or infinity.")
        mean = np.mean(X, axis=0, dtype=np.float64).astype(out_dtype, copy=False)
        X_c = (X - mean).astype(out_dtype, copy=False)
    else:
        mean = gram[0].astype(out_dtype, copy=False)

    if fit_svd_solver == "full":
        U, S, Vt = scipy.linalg.svd(X_c, full_matrices=False)
        U, Vt = _svd_flip(U, Vt, u_based_decision=False)
        explained_variance = ((S**2) / (n_samples - 1)).astype(out_dtype, copy=False)
    else:
        cov = gram[1] if gram is not None else X_c.T @ X_c  # type: ignore[union-attr]
        w, V = scipy.linalg.eigh(cov, driver="evd")
        w = np.maximum(w, 0.0)
        S = np.sqrt(w)[::-1].astype(out_dtype, copy=False)
        Vt = V[:, ::-1].T.astype(out_dtype, copy=False)
        _, Vt = _svd_flip(None, Vt, u_based_decision=False)
        explained_variance = (w[::-1] / (n_samples - 1)).astype(out_dtype, copy=False)

    total_var = explained_variance.sum()
    explained_variance_ratio = (explained_variance / total_var).astype(out_dtype, copy=False)
    singular_values = S.astype(out_dtype, copy=False)

    k = _resolve_full_n_components(
        config.n_components,
        explained_variance,
        explained_variance_ratio,
        n_samples,
        n_features,
    )
    if k < min(n_features, n_samples):
        noise_variance: float | np.floating = explained_variance[k:].mean()
    else:
        noise_variance = 0.0

    Vt_k = Vt[:k].astype(out_dtype, copy=False)
    ev_k = explained_variance[:k].astype(out_dtype, copy=False)
    evr_k = explained_variance_ratio[:k].astype(out_dtype, copy=False)
    sv_k = singular_values[:k].astype(out_dtype, copy=False)

    transformed = None
    if return_transformed and X_c is not None:
        transformed = _transform_from_centered(
            X_c, None, Vt_k, ev_k, config.whiten, out_dtype, prefer_gpu=False
        )
    elif return_transformed:
        transformed = pca_transform(X, mean, Vt_k, ev_k, whiten=config.whiten, prefer_gpu=False)

    return PCASolution(
        components=Vt_k,
        explained_variance=ev_k,
        explained_variance_ratio=evr_k,
        singular_values=sv_k,
        mean=mean,
        n_components=k,
        n_samples=n_samples,
        n_features=n_features,
        noise_variance=noise_variance,
        fit_svd_solver=fit_svd_solver,
        used_gpu=False,
        transformed=transformed,
    )


def _solve_gpu_full_or_eigh(
    X_mx: mx.array,
    mean_mx: mx.array,
    col_ss_mx: mx.array,
    X_np: np.ndarray,
    config: PCAConfig,
    fit_svd_solver: Literal["full", "covariance_eigh"],
    out_dtype: np.dtype,
    return_transformed: bool,
) -> PCASolution:
    """Metal GPU path for ``covariance_eigh`` and ``full`` SVD."""
    n_samples, n_features = X_mx.shape
    mean_f32 = to_numpy(mean_mx)
    mean = mean_f32.astype(out_dtype, copy=False)
    col_ss_sum = float(np.sum(to_numpy(col_ss_mx), dtype=np.float64))
    mean_ss = float(n_samples) * float(np.sum(mean_f32.astype(np.float64) ** 2))

    # When fit_svd_solver == 'covariance_eigh' and the data shift does not dominate
    # the centered variance, compute G = X.T @ X - n * mean @ mean.T directly without
    # allocating or writing an n x d centered copy X_c.
    if fit_svd_solver == "covariance_eigh" and mean_ss <= 10.0 * max(col_ss_sum, 1e-12):
        X_c_mx: mx.array | None = None
        G_mx = mx.addmm(
            mean_mx[:, None] @ mean_mx[None, :],
            X_mx.T,
            X_mx,
            alpha=1.0,
            beta=-float(n_samples),
        )
    else:
        X_c_mx = prepare_pca_centered(X_mx, mean_mx)

    if fit_svd_solver == "covariance_eigh" or n_samples >= n_features:
        # Primal Gram matrix G = X_c.T @ X_c of shape (n_features, n_features) on GPU.
        if X_c_mx is not None:
            G_mx = X_c_mx.T @ X_c_mx
        w_mx, V_mx = mx.linalg.eigh(G_mx, stream=mx.cpu)
        mx.eval(w_mx, V_mx)

        w_desc = np.maximum(to_numpy(w_mx)[::-1].astype(np.float64), 0.0)
        V_desc = to_numpy(V_mx)[:, ::-1]
        S = np.sqrt(w_desc)
        Vt = V_desc.T.copy()

        max_rank = min(n_samples, n_features) if fit_svd_solver == "full" else n_features
        S = S[:max_rank]
        Vt = Vt[:max_rank]

        explained_variance = w_desc[:max_rank] / (n_samples - 1)
        total_var = float(explained_variance.sum())
        explained_variance_ratio = explained_variance / total_var

        k = _resolve_full_n_components(
            config.n_components,
            explained_variance,
            explained_variance_ratio,
            n_samples,
            n_features,
        )

        # Deflated tail refinement for ill-conditioned spectra under svd_solver='full'.
        # Because Gram matrix squaring loses accuracy below 1e-2 * sigma_max (w_i < 1e-3 * w_0),
        # we orthogonalize the tail subspace against the well-conditioned head on the GPU and
        # run a thin SVD on the small deflated tail matrix.
        if (
            fit_svd_solver == "full"
            and max_rank > 0
            and w_desc[0] > 0.0
            and w_desc[max_rank - 1] < 1e-3 * w_desc[0]
        ):
            idx = int(np.searchsorted(-w_desc[:max_rank], -1e-3 * w_desc[0], side="right"))
            if idx < max_rank:
                V_full_mx = to_mlx_f32(V_desc[:, :max_rank])
                Y_mx = X_c_mx @ V_full_mx
                if idx > 0:
                    inv_s_head = to_mlx_f32((1.0 / S[:idx]).astype(np.float32))
                    U_head_mx = Y_mx[:, :idx] * inv_s_head
                    Y_tail_mx = Y_mx[:, idx:] - U_head_mx @ (U_head_mx.T @ Y_mx[:, idx:])
                else:
                    Y_tail_mx = Y_mx
                mx.eval(Y_tail_mx)
                Y_tail_np = to_numpy(Y_tail_mx)
                _, S_tail, Vt_tail = scipy.linalg.svd(Y_tail_np, full_matrices=False)
                S[idx:max_rank] = S_tail.astype(np.float64)
                Vt[idx:max_rank] = Vt_tail @ V_desc[:, idx:max_rank].T
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    explained_variance = (S**2) / (n_samples - 1)
                    total_var = float(explained_variance.sum())
                    explained_variance_ratio = explained_variance / total_var
                k = _resolve_full_n_components(
                    config.n_components,
                    explained_variance,
                    explained_variance_ratio,
                    n_samples,
                    n_features,
                )

        _, Vt = _svd_flip(None, Vt, u_based_decision=False)
    else:
        # Wide matrix (n_samples < n_features) with fit_svd_solver == "full":
        # Dual Gram matrix G_dual = X_c @ X_c.T of shape (n_samples, n_samples) on GPU.
        G_dual_mx = X_c_mx @ X_c_mx.T
        w_mx, U_mx = mx.linalg.eigh(G_dual_mx, stream=mx.cpu)
        mx.eval(w_mx, U_mx)

        w_desc = np.maximum(to_numpy(w_mx)[::-1].astype(np.float64), 0.0)
        U_desc = to_numpy(U_mx)[:, ::-1]
        S = np.sqrt(w_desc)

        # Since X_c is column-centered, rank(X_c) <= n_samples - 1, so the last
        # eigenvalue w_desc[n_samples - 1] is structurally zero. Check whether any
        # earlier singular values (0 .. n_samples - 2) need deflated tail refinement.
        n_nontrivial = max(n_samples - 1, 0)
        if n_nontrivial > 0 and w_desc[0] > 0.0 and w_desc[n_nontrivial - 1] < 1e-3 * w_desc[0]:
            idx = int(np.searchsorted(-w_desc[:n_nontrivial], -1e-3 * w_desc[0], side="right"))
        else:
            idx = n_nontrivial

        Vt = np.zeros((n_samples, n_features), dtype=np.float32)
        if idx > 0:
            U_head_mx = to_mlx_f32(U_desc[:, :idx])
            inv_s_mx = to_mlx_f32((1.0 / S[:idx]).astype(np.float32))
            Vt_head_mx = (U_head_mx.T @ X_c_mx) * inv_s_mx[:, None]
            Vt_head_mx = Vt_head_mx * mx.rsqrt(
                mx.maximum(mx.sum(Vt_head_mx * Vt_head_mx, axis=1, keepdims=True), 1e-30)
            )
            mx.eval(Vt_head_mx)
            Vt[:idx] = to_numpy(Vt_head_mx)

        if idx < n_nontrivial:
            U_tail_mx = to_mlx_f32(U_desc[:, idx:])
            X_rem_mx = U_tail_mx.T @ X_c_mx
            if idx > 0:
                Vt_head_mx_ref = to_mlx_f32(Vt[:idx])
                X_rem_mx = X_rem_mx - (X_rem_mx @ Vt_head_mx_ref.T) @ Vt_head_mx_ref
            mx.eval(X_rem_mx)
            X_rem_np = to_numpy(X_rem_mx)
            if float(np.linalg.norm(X_rem_np)) > 1e-7 * float(S[0]):
                _, S_tail, Vt_tail = scipy.linalg.svd(X_rem_np, full_matrices=False)
                n_tail = S_tail.shape[0]
                S[idx : idx + n_tail] = S_tail.astype(np.float64)
                Vt[idx : idx + n_tail] = Vt_tail.astype(np.float32, copy=False)
                valid_rows = int(np.sum(S > float(S[0]) * max(n_samples, n_features) * 1e-7))
            else:
                S[idx:] = 0.0
                valid_rows = idx
        else:
            S[idx:] = 0.0
            valid_rows = idx if w_desc[0] > 0.0 else 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            explained_variance = (S**2) / (n_samples - 1)
            total_var = float(explained_variance.sum())
            explained_variance_ratio = explained_variance / total_var

        k = _resolve_full_n_components(
            config.n_components,
            explained_variance,
            explained_variance_ratio,
            n_samples,
            n_features,
        )
        if valid_rows < k:
            Vt_k_raw = _complete_orthonormal_rows(Vt[:valid_rows], k, n_features)
        else:
            Vt_k_raw = Vt[:k]
        _, Vt_k_flipped = _svd_flip(None, Vt_k_raw, u_based_decision=False)
        Vt[:k] = Vt_k_flipped

    if k < min(n_features, n_samples):
        noise_variance: float | np.floating = out_dtype.type(explained_variance[k:].mean())
    else:
        noise_variance = 0.0

    Vt_k = Vt[:k].astype(out_dtype, copy=False)
    ev_k = explained_variance[:k].astype(out_dtype, copy=False)
    evr_k = explained_variance_ratio[:k].astype(out_dtype, copy=False)
    sv_k = S[:k].astype(out_dtype, copy=False)

    dev_components = to_mlx_f32(Vt_k) if k > 0 else None
    transformed = None
    if return_transformed:
        if X_c_mx is not None:
            transformed = _transform_from_centered(
                None, X_c_mx, Vt_k, ev_k, config.whiten, out_dtype, prefer_gpu=True
            )
        else:
            transformed = pca_transform(
                X_np,
                mean,
                Vt_k,
                ev_k,
                whiten=config.whiten,
                prefer_gpu=True,
                device_mean=mean_mx,
                device_components=dev_components,
                X_mx=X_mx,
            )
    return PCASolution(
        components=Vt_k,
        explained_variance=ev_k,
        explained_variance_ratio=evr_k,
        singular_values=sv_k,
        mean=mean,
        n_components=k,
        n_samples=n_samples,
        n_features=n_features,
        noise_variance=noise_variance,
        fit_svd_solver=fit_svd_solver,
        used_gpu=True,
        transformed=transformed,
        device_mean=mean_mx,
        device_components=dev_components,
    )


def _gram_f64(A_mx: mx.array) -> np.ndarray:
    """Return the small Gram matrix ``A.T @ A`` on the host, symmetrized, in float64."""
    G = to_numpy(A_mx.T @ A_mx).astype(np.float64)
    return 0.5 * (G + G.T)


def _unit_scale(A_mx: mx.array) -> mx.array:
    """Divide ``A_mx`` by its largest magnitude, lazily, on the GPU.

    A scalar factor leaves the column span untouched. It keeps an unnormalized
    power iteration from growing by ``sigma_max ** 2`` per step, and keeps the
    Gram matrix of a float32 block, which squares its scale, from overflowing on
    data whose singular values are merely in the thousands.
    """
    return A_mx / mx.maximum(mx.max(mx.abs(A_mx)), 1e-30)


#: Cholesky QR rounds allowed before the block is handed to a host factorization.
_CHOLESKY_QR_MAX_ROUNDS = 8


def _orthonormalize(
    Y_mx: mx.array,
    normalizer: Literal["QR", "LU", "none"],
    *,
    passes: int = 1,
    rescale: bool = True,
) -> mx.array:
    """Normalize the columns of ``Y_mx`` on the GPU by Cholesky QR.

    ``Q = Y @ inv(R)`` is always formed explicitly, so every later product uses the
    basis that was actually computed. One resolved pass leaves ``Q.T @ Q - I`` of
    order ``eps * cond(Y) ** 2``, which keeps a power iteration well scaled. Two
    bring that down to ``eps`` and are what the final range finder needs, because
    a Rayleigh-Ritz step through a non-orthonormal basis leaks the leading
    singular values into the trailing ones.

    The ``size x size`` Gram matrix is computed in float32, so it only resolves
    directions above a relative floor. When the smallest Cholesky pivot falls
    under that floor, or the factorization fails, the round is repeated with the
    floor added to the diagonal: that step is safe for any conditioning, and it
    lifts the unresolved directions by up to ``1 / sqrt(floor)`` so a later round
    can resolve them. Low-rank data with a noise floor, which is most real data,
    converges in a few such rounds, each costing one small GPU Gram matrix; a
    rank-deficient block converges too, with rounding noise completing the basis.
    Pivots can overstate conditioning, so the last of two passes also requires the
    block it starts from to be close to orthonormal.

    Only a block that never converges, or a Gram matrix that is not finite, is
    normalized on the host with a Householder QR (or LU, when that was requested).

    The block is rescaled to unit magnitude first (see :func:`_unit_scale`), which
    is all that ``normalizer="none"`` does. A caller that has bounded the block's
    scale, so that its Gram matrix cannot overflow, passes ``rescale=False``.
    """
    if normalizer == "none":
        return _unit_scale(Y_mx)
    Q_mx = _unit_scale(Y_mx) if rescale else Y_mx
    n_rows = max(int(Y_mx.shape[0]), 1)
    relative_floor = max(1e-5, 8.0 * float(np.finfo(np.float32).eps) * math.sqrt(n_rows))
    clean = 0
    try:
        for _ in range(_CHOLESKY_QR_MAX_ROUNDS):
            G = _gram_f64(Q_mx)
            top = float(np.max(np.diagonal(G)))
            if not math.isfinite(float(np.sum(G))) or not top > 0.0:
                break
            identity = np.eye(G.shape[0], dtype=np.float64)
            if clean and not float(np.max(np.abs(G - identity))) < 0.5:
                clean = 0
            floor = relative_floor * top
            try:
                R = scipy.linalg.cholesky(G, lower=False, check_finite=False)
                resolved = float(np.min(np.diagonal(R))) ** 2 >= floor
            except scipy.linalg.LinAlgError:
                resolved = False
            if not resolved:
                R = scipy.linalg.cholesky(G + floor * identity, lower=False, check_finite=False)
            R_inv = scipy.linalg.solve_triangular(R, identity, lower=False, check_finite=False)
            Q_mx = Q_mx @ to_mlx_f32(R_inv)
            if resolved:
                clean += 1
                if clean >= passes:
                    return Q_mx
    except scipy.linalg.LinAlgError:
        pass
    Y_np = to_numpy(Y_mx)
    if not bool(np.isfinite(Y_np).all()):
        raise ValueError("The randomized range finder overflowed float32.")
    if normalizer == "LU":
        Q_np, _ = scipy.linalg.lu(Y_np, permute_l=True)
    else:
        Q_np, _ = scipy.linalg.qr(Y_np, mode="economic")
    return to_mlx_f32(Q_np)


def _host_normalize(Z: np.ndarray, normalizer: Literal["QR", "LU"]) -> np.ndarray:
    """Normalize the columns of a small float64 host block between power iterations.

    The Gram route only needs a well-conditioned basis here, not an exactly
    orthonormal one, because its next step renormalizes through ``Q.T @ G @ Q``.
    A Cholesky QR in float64 delivers that for a fraction of a Householder QR,
    which was most of the route's run time at a few thousand columns; Householder
    remains the fallback when the block is too ill-conditioned to factor.
    """
    lapack = {"check_finite": False, "overwrite_a": True}
    if normalizer == "LU":
        return scipy.linalg.lu(Z, permute_l=True, **lapack)[0]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        ZtZ = Z.T @ Z
    try:
        R = scipy.linalg.cholesky(0.5 * (ZtZ + ZtZ.T), lower=False)
        return scipy.linalg.solve_triangular(R.T, Z.T, lower=True, check_finite=False).T
    except scipy.linalg.LinAlgError:
        return scipy.linalg.qr(Z, mode="economic", **lapack)[0]


def _solve_randomized(
    X_np: np.ndarray,
    X_mx: mx.array | None,
    mean_mx: mx.array | None,
    col_ss_mx: mx.array | None,
    config: PCAConfig,
    out_dtype: np.dtype,
    use_gpu: bool,
    return_transformed: bool,
) -> PCASolution:
    """Randomized SVD solver on Metal GPU (or CPU below crossover)."""
    n_samples, n_features = X_np.shape
    k = int(config.n_components)  # type: ignore[arg-type]
    n_random = k + int(config.n_oversamples)
    n_iter = (
        7 if config.iterated_power == "auto" and k < 0.1 * min(n_samples, n_features)
        else (4 if config.iterated_power == "auto" else int(config.iterated_power))
    )
    on_host = not use_gpu or X_mx is None or mean_mx is None or col_ss_mx is None
    normalizer: Literal["QR", "LU", "none"]
    if config.power_iteration_normalizer == "auto":
        # The host mirrors scikit-learn, whose automatic normalizer is the cheaper
        # LU. On the GPU both spellings are a Cholesky QR through a small Gram matrix.
        normalizer = "none" if n_iter <= 2 else ("LU" if on_host else "QR")
    else:
        normalizer = config.power_iteration_normalizer

    rng = np.random.RandomState(config.seed)

    if on_host:
        mean = np.mean(X_np, axis=0, dtype=np.float64).astype(out_dtype, copy=False)
        X_c = np.ascontiguousarray((X_np - mean).astype(out_dtype, copy=False))
        total_var = _host_sum_of_squares(X_c) / (n_samples - 1)

        # X was validated finite before this point, and every factorization below
        # owns its freshly computed argument.
        lapack = {"check_finite": False, "overwrite_a": True}
        transpose = n_samples < n_features
        M = X_c.T if transpose else X_c
        Q = rng.normal(size=(M.shape[1], n_random)).astype(out_dtype, copy=False)
        for _ in range(n_iter):
            if normalizer == "none":
                Q = M @ Q
                Q = M.T @ Q
            elif normalizer == "LU":
                Q, _ = scipy.linalg.lu(M @ Q, permute_l=True, **lapack)
                Q, _ = scipy.linalg.lu(M.T @ Q, permute_l=True, **lapack)
            else:
                Q, _ = scipy.linalg.qr(M @ Q, mode="economic", **lapack)
                Q, _ = scipy.linalg.qr(M.T @ Q, mode="economic", **lapack)
        Q, _ = scipy.linalg.qr(M @ Q, mode="economic", **lapack)
        B = Q.T @ M
        Uhat, S, Vt = scipy.linalg.svd(B, full_matrices=False, check_finite=False)
        U = Q @ Uhat
        if transpose:
            U, Vt = Vt.T, U.T
        U, Vt = _svd_flip(U, Vt, u_based_decision=False)
        S_k = S[:k].astype(out_dtype, copy=False)
        Vt_k = Vt[:k].astype(out_dtype, copy=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ev_k = ((S_k**2) / (n_samples - 1)).astype(out_dtype, copy=False)
            evr_k = (ev_k / total_var).astype(out_dtype, copy=False)
        if k < min(n_features, n_samples):
            noise_variance: float | np.floating = out_dtype.type(
                max(total_var - float(ev_k.sum()), 0.0) / (min(n_features, n_samples) - k)
            )
        else:
            noise_variance = 0.0
        if return_transformed:
            U_k = U[:, :k]
            if config.whiten:
                transformed = (U_k * math.sqrt(n_samples - 1)).astype(out_dtype, copy=False)
            else:
                transformed = (U_k * S_k).astype(out_dtype, copy=False)
        else:
            transformed = None
        return PCASolution(
            components=Vt_k,
            explained_variance=ev_k,
            explained_variance_ratio=evr_k,
            singular_values=S_k,
            mean=mean,
            n_components=k,
            n_samples=n_samples,
            n_features=n_features,
            noise_variance=noise_variance,
            fit_svd_solver="randomized",
            used_gpu=False,
            transformed=transformed,
        )

    # Metal GPU path for randomized SVD
    mean = to_numpy(mean_mx).astype(out_dtype, copy=False)
    total_var = float(np.sum(to_numpy(col_ss_mx), dtype=np.float64)) / (n_samples - 1)
    X_c_mx = prepare_pca_centered(X_mx, mean_mx)

    transpose = n_samples < n_features
    M_mx = X_c_mx.T if transpose else X_c_mx
    m_rows, m_cols = M_mx.shape
    size = min(n_random, m_rows, m_cols)

    Q_init = rng.normal(size=(m_cols, n_random)).astype(np.float32, copy=False)
    if size < n_random:
        Q_init = Q_init[:, :size]

    # When m_cols <= 2048 and m_rows >= 2 * m_cols (and m_cols <= 2 * (n_iter + 1) * size)
    # with QR/LU normalizer, precomputing the Gram matrix G = M.T @ M once on the GPU
    # reads the large matrix M only once instead of 2*(n_iter + 1) times.
    use_gram_subspace = (
        normalizer in ("QR", "LU")
        and m_cols <= 2048
        and m_rows >= 2 * m_cols
        and m_cols <= 2 * (n_iter + 1) * size
    )

    if use_gram_subspace:
        G_mx = M_mx.T @ M_mx
        mx.eval(G_mx)
        G_np = to_numpy(G_mx)
        Q_np = Q_init
        # Each iteration on M is Q <- norm(M.T @ norm(M @ Q)).
        # Using Gram G = M.T @ M: (M @ Q)^T (M @ Q) = Q.T @ G @ Q = R_y.T @ R_y,
        # and M.T @ Q_y = (G @ Q) @ R_y^{-1}.
        gram_ok = True
        for _ in range(n_iter):
            GQ = G_np @ Q_np
            YtY = (Q_np.T @ GQ).astype(np.float64)
            YtY = 0.5 * (YtY + YtY.T)
            try:
                Ry = scipy.linalg.cholesky(YtY, lower=False)
                Z = scipy.linalg.solve_triangular(Ry.T, GQ.T, lower=True).T
            except scipy.linalg.LinAlgError:
                gram_ok = False
                break
            # Back to float32, so the next product with G is one single-precision
            # matrix multiply instead of a float64 one on a converted copy of G.
            Q_np = np.ascontiguousarray(_host_normalize(Z, normalizer), dtype=np.float32)

        if gram_ok:
            GQ = G_np @ Q_np
            YtY = (Q_np.T @ GQ).astype(np.float64)
            YtY = 0.5 * (YtY + YtY.T)
            try:
                Ry = scipy.linalg.cholesky(YtY, lower=False)
                B_T = scipy.linalg.solve_triangular(Ry.T, GQ.T, lower=True).T
                B = B_T.T
                Uhat, S, Vt = scipy.linalg.svd(B, full_matrices=False)
                if transpose:
                    # M = X_c.T, so Vt has shape (size, n_samples) and Q_y = (M @ Q) @ Ry^{-1}
                    Q_Rinv = scipy.linalg.solve_triangular(Ry.T, Q_np.T, lower=True).T
                    Q_Uhat_mx = to_mlx_f32((Q_Rinv @ Uhat[:, :k]).astype(np.float32))
                    Vt_k_mx = (M_mx @ Q_Uhat_mx).T
                    mx.eval(Vt_k_mx)
                    Vt_k_raw = to_numpy(Vt_k_mx)
                    U_k_raw = Vt[:k].T
                    U_k_flipped, Vt_k_raw = _svd_flip(U_k_raw, Vt_k_raw, u_based_decision=False)
                else:
                    Uhat_flipped, Vt_flipped = _svd_flip(Uhat, Vt, u_based_decision=False)
                    Vt_k_raw = Vt_flipped[:k]
                    U_k_flipped = None
                S_k = S[:k].astype(out_dtype, copy=False)
                Vt_k = Vt_k_raw.astype(out_dtype, copy=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ev_k = ((S_k.astype(np.float64) ** 2) / (n_samples - 1)).astype(
                        out_dtype, copy=False
                    )
                    evr_k = (ev_k.astype(np.float64) / total_var).astype(out_dtype, copy=False)
                if k < min(n_features, n_samples):
                    noise_variance = out_dtype.type(
                        max(total_var - float(ev_k.astype(np.float64).sum()), 0.0)
                        / (min(n_features, n_samples) - k)
                    )
                else:
                    noise_variance = 0.0
                if return_transformed:
                    scale_vec = (
                        np.full(k, math.sqrt(n_samples - 1), dtype=np.float64)
                        if config.whiten
                        else S[:k].astype(np.float64)
                    )
                    if transpose and U_k_flipped is not None:
                        transformed = (U_k_flipped * scale_vec).astype(out_dtype, copy=False)
                    else:
                        Q_Rinv = scipy.linalg.solve_triangular(Ry.T, Q_np.T, lower=True).T
                        W = (Q_Rinv @ (Uhat_flipped[:, :k] * scale_vec)).astype(np.float32)  # type: ignore[index]
                        Z_mx = M_mx @ to_mlx_f32(W)
                        mx.eval(Z_mx)
                        transformed = to_numpy(Z_mx).astype(out_dtype, copy=False)
                else:
                    transformed = None
                return PCASolution(
                    components=Vt_k,
                    explained_variance=ev_k,
                    explained_variance_ratio=evr_k,
                    singular_values=S_k,
                    mean=mean,
                    n_components=k,
                    n_samples=n_samples,
                    n_features=n_features,
                    noise_variance=noise_variance,
                    fit_svd_solver="randomized",
                    used_gpu=True,
                    transformed=transformed,
                    device_mean=mean_mx,
                    device_components=to_mlx_f32(Vt_k) if k > 0 else None,
                )
            except scipy.linalg.LinAlgError:
                pass

    # Direct GPU subspace iteration. Each half step multiplies by ``M`` or ``M.T`` and
    # then normalizes through the tiny ``size x size`` Gram matrix, forming the
    # normalized basis explicitly before the next product. Building ``M.T @ Q`` as
    # ``(M.T @ Y) @ inv(R)`` instead would cost the same two passes over ``M`` but
    # cancels catastrophically in float32 once ``Y`` is ill-conditioned.
    #
    # A block's Gram matrix is bounded by the squared norm of the block. Normalized
    # iterates have unit scale and the Gaussian start has spectral norm about
    # sqrt(m_cols) + sqrt(size), so with two orders of margin the Gram matrices
    # cannot overflow float32 unless the data's own sum of squares is enormous;
    # only then, or when the iteration is not normalized, are blocks rescaled.
    total_ss = total_var * (n_samples - 1)
    rescale = normalizer == "none" or not (
        400.0 * m_cols * size * total_ss < float(np.finfo(np.float32).max)
    )
    Q_mx = to_mlx_f32(Q_init)
    for _ in range(n_iter):
        if normalizer == "none":
            Q_mx = _unit_scale(M_mx.T @ _unit_scale(M_mx @ Q_mx))
            mx.eval(Q_mx)
            continue
        Q_mx = _orthonormalize(
            M_mx.T @ _orthonormalize(M_mx @ Q_mx, normalizer, rescale=rescale),
            normalizer,
            rescale=rescale,
        )

    Q_y_mx = _orthonormalize(M_mx @ Q_mx, "QR", passes=2, rescale=rescale)
    B_T_mx = M_mx.T @ Q_y_mx
    mx.eval(B_T_mx, Q_y_mx)
    B = to_numpy(B_T_mx).T
    Uhat, S, Vt = scipy.linalg.svd(B, full_matrices=False)
    if transpose:
        Q_y_Uhat_mx = Q_y_mx @ to_mlx_f32(Uhat[:, :k].astype(np.float32))
        mx.eval(Q_y_Uhat_mx)
        Vt_k_raw = to_numpy(Q_y_Uhat_mx).T
        U_k_raw = Vt[:k].T
        U_k_flipped, Vt_k_raw = _svd_flip(U_k_raw, Vt_k_raw, u_based_decision=False)
        Uhat_flipped = None
    else:
        Uhat_flipped, Vt_flipped = _svd_flip(Uhat, Vt, u_based_decision=False)
        Vt_k_raw = Vt_flipped[:k]
        U_k_flipped = None

    S_k = S[:k].astype(out_dtype, copy=False)
    Vt_k = Vt_k_raw.astype(out_dtype, copy=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ev_k = ((S_k.astype(np.float64) ** 2) / (n_samples - 1)).astype(out_dtype, copy=False)
        evr_k = (ev_k.astype(np.float64) / total_var).astype(out_dtype, copy=False)

    if k < min(n_features, n_samples):
        noise_variance = out_dtype.type(
            max(total_var - float(ev_k.astype(np.float64).sum()), 0.0)
            / (min(n_features, n_samples) - k)
        )
    else:
        noise_variance = 0.0

    if return_transformed:
        scale_vec = (
            np.full(k, math.sqrt(n_samples - 1), dtype=np.float64)
            if config.whiten
            else S[:k].astype(np.float64)
        )
        if transpose and U_k_flipped is not None:
            transformed = (U_k_flipped * scale_vec).astype(out_dtype, copy=False)
        else:
            W = (Uhat_flipped[:, :k] * scale_vec).astype(np.float32)  # type: ignore[index]
            Z_mx = Q_y_mx @ to_mlx_f32(W)
            mx.eval(Z_mx)
            transformed = to_numpy(Z_mx).astype(out_dtype, copy=False)
    else:
        transformed = None
    return PCASolution(
        components=Vt_k,
        explained_variance=ev_k,
        explained_variance_ratio=evr_k,
        singular_values=S_k,
        mean=mean,
        n_components=k,
        n_samples=n_samples,
        n_features=n_features,
        noise_variance=noise_variance,
        fit_svd_solver="randomized",
        used_gpu=True,
        transformed=transformed,
        device_mean=mean_mx,
        device_components=to_mlx_f32(Vt_k) if k > 0 else None,
    )


def _solve_arpack(
    X_np: np.ndarray,
    X_mx: mx.array | None,
    mean_mx: mx.array | None,
    col_ss_mx: mx.array | None,
    config: PCAConfig,
    out_dtype: np.dtype,
    use_gpu: bool,
    return_transformed: bool,
) -> PCASolution:
    """ARPACK truncated SVD solver with optional GPU matvec acceleration."""
    n_samples, n_features = X_np.shape
    k = int(config.n_components)  # type: ignore[arg-type]
    rng = np.random.RandomState(config.seed)
    v0 = rng.uniform(-1.0, 1.0, size=min(n_samples, n_features)).astype(out_dtype, copy=False)

    if use_gpu and X_mx is not None and mean_mx is not None and col_ss_mx is not None:
        mean = to_numpy(mean_mx).astype(out_dtype, copy=False)
        total_var = float(np.sum(to_numpy(col_ss_mx), dtype=np.float64)) / (n_samples - 1)
        X_c_mx = prepare_pca_centered(X_mx, mean_mx)
        mx.eval(X_c_mx)
        X_c = (X_np - mean).astype(out_dtype, copy=False)
    else:
        mean = np.mean(X_np, axis=0, dtype=np.float64).astype(out_dtype, copy=False)
        X_c = (X_np - mean).astype(out_dtype, copy=False)
        total_var = float(np.sum(X_c.astype(np.float64) ** 2)) / (n_samples - 1)
        X_c_mx = None

    U, S, Vt = scipy.sparse.linalg.svds(X_c, k=k, tol=config.tol, v0=v0)
    S = S[::-1]
    U, Vt = _svd_flip(U[:, ::-1], Vt[::-1], u_based_decision=False)

    S_k = S.astype(out_dtype, copy=False)
    Vt_k = Vt.astype(out_dtype, copy=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ev_k = ((S_k**2) / (n_samples - 1)).astype(out_dtype, copy=False)
        evr_k = (ev_k / total_var).astype(out_dtype, copy=False)

    if k < min(n_features, n_samples):
        noise_variance: float | np.floating = out_dtype.type(
            max(total_var - float(ev_k.sum()), 0.0) / (min(n_features, n_samples) - k)
        )
    else:
        noise_variance = 0.0

    transformed = (
        _transform_from_centered(
            X_c, X_c_mx, Vt_k, ev_k, config.whiten, out_dtype, prefer_gpu=X_c_mx is not None
        )
        if return_transformed
        else None
    )
    return PCASolution(
        components=Vt_k,
        explained_variance=ev_k,
        explained_variance_ratio=evr_k,
        singular_values=S_k,
        mean=mean,
        n_components=k,
        n_samples=n_samples,
        n_features=n_features,
        noise_variance=noise_variance,
        fit_svd_solver="arpack",
        used_gpu=X_c_mx is not None,
        transformed=transformed,
        device_mean=mean_mx if X_c_mx is not None else None,
        device_components=to_mlx_f32(Vt_k) if (X_c_mx is not None and k > 0) else None,
    )


def solve_pca(
    X: np.ndarray,
    config: PCAConfig,
    *,
    return_transformed: bool = False,
    X_f32: np.ndarray | None = None,
) -> PCASolution:
    """Fit PCA on a dense 2D array ``X`` of dtype ``float64`` or ``float32``.

    ``X_f32`` is an optional C-contiguous float32 copy of ``X`` that the caller has
    already made, so the GPU path does not narrow the input a second time.
    """
    n_samples, n_features = X.shape
    out_dtype = np.dtype(np.float32) if X.dtype == np.float32 else np.dtype(np.float64)
    fit_svd_solver = choose_svd_solver(
        n_samples, n_features, config.n_components, config.svd_solver
    )

    # Degenerate single-sample or zero-component cases are fast on CPU and avoid
    # unnecessary GPU kernel launches while checking finiteness exactly.
    use_gpu = bool(
        config.prefer_gpu
        and n_samples >= 2
        and n_features >= 1
        and n_samples * n_features >= 16_384
    )

    borrowed: np.ndarray | None = None
    try:
        if use_gpu:
            borrowed = ensure_f32_2d(X) if X_f32 is None else X_f32
            X_mx = borrow_mlx_f32(borrowed)
            mean_mx, col_ss_mx, finite_mx = pca_stats(X_mx)
            mx.eval(mean_mx, col_ss_mx, finite_mx)
            if not bool(finite_mx.item()):
                raise ValueError("Input X contains NaN or infinity.")
            # Guard against extreme float32 values (> 1e19) where squared deviations overflow float32.
            col_ss_np = to_numpy(col_ss_mx)
            if not bool(np.all(np.isfinite(col_ss_np))):
                use_gpu = False
        else:
            # The host full/covariance solve validates finiteness itself; the
            # covariance route does it without an extra pass over X.
            if fit_svd_solver in ("randomized", "arpack") and not bool(np.isfinite(X).all()):
                raise ValueError("Input X contains NaN or infinity.")
            X_mx = None
            mean_mx = None
            col_ss_mx = None

        if fit_svd_solver in ("full", "covariance_eigh"):
            if use_gpu and X_mx is not None and mean_mx is not None and col_ss_mx is not None:
                return _solve_gpu_full_or_eigh(
                    X_mx,
                    mean_mx,
                    col_ss_mx,
                    X,
                    config,
                    fit_svd_solver,
                    out_dtype,
                    return_transformed,
                )
            return _solve_cpu_full_or_eigh(
                X, config, fit_svd_solver, out_dtype, return_transformed
            )

        if fit_svd_solver == "randomized":
            return _solve_randomized(
                X,
                X_mx,
                mean_mx,
                col_ss_mx,
                config,
                out_dtype,
                use_gpu,
                return_transformed,
            )

        return _solve_arpack(
            X,
            X_mx,
            mean_mx,
            col_ss_mx,
            config,
            out_dtype,
            use_gpu,
            return_transformed,
        )
    finally:
        if borrowed is not None:
            mx.synchronize()


def pca_transform(
    X: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    explained_variance: np.ndarray,
    *,
    whiten: bool = False,
    prefer_gpu: bool = True,
    device_mean: mx.array | None = None,
    device_components: mx.array | None = None,
    X_mx: mx.array | None = None,
    X_f32: np.ndarray | None = None,
) -> np.ndarray:
    """Project ``X`` onto the fitted principal components.

    ``X_f32`` is an optional float32 copy of ``X`` the caller has already made.
    """
    out_dtype = np.result_type(X.dtype, mean.dtype)
    n_samples = X.shape[0]
    k = components.shape[0]
    if k == 0:
        return np.zeros((n_samples, 0), dtype=out_dtype)

    if prefer_gpu and (X_mx is not None or X.size >= 65_536):
        borrowed: np.ndarray | None = None
        try:
            if X_mx is None:
                borrowed = ensure_f32_2d(X) if X_f32 is None else X_f32
                X_mx = borrow_mlx_f32(borrowed)
            if not whiten and device_components is not None and device_mean is not None:
                V_mx = device_components.T
                bias_mx = -(device_mean @ V_mx)
                Z_mx = mx.addmm(bias_mx, X_mx, V_mx)
                mx.eval(Z_mx)
                return to_numpy(Z_mx).astype(out_dtype, copy=False)
            V_T = components.T.astype(np.float64, copy=True)
            if whiten:
                scale = np.sqrt(explained_variance.astype(np.float64, copy=False))
                min_scale = float(np.finfo(out_dtype).eps)
                scale = np.where(scale < min_scale, min_scale, scale)
                V_T = V_T / scale
            bias = -(mean.astype(np.float64, copy=False) @ V_T)
            V_mx = to_mlx_f32(V_T.astype(np.float32))
            bias_mx = to_mlx_f32(bias.astype(np.float32))
            Z_mx = mx.addmm(bias_mx, X_mx, V_mx)
            mx.eval(Z_mx)
            return to_numpy(Z_mx).astype(out_dtype, copy=False)
        finally:
            if borrowed is not None:
                mx.synchronize()

    # Project first and shift the k-wide result, as scikit-learn does, instead of
    # centering a copy of X.
    X_transformed = np.asarray(X @ components.T, dtype=out_dtype)
    X_transformed -= (mean @ components.T).astype(out_dtype, copy=False)
    if whiten:
        scale = np.sqrt(explained_variance)
        min_scale = np.finfo(scale.dtype).eps
        scale = np.where(scale < min_scale, min_scale, scale)
        X_transformed /= scale.astype(out_dtype, copy=False)
    return X_transformed


def pca_inverse_transform(
    X: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    explained_variance: np.ndarray,
    *,
    whiten: bool = False,
    prefer_gpu: bool = True,
    device_mean: mx.array | None = None,
    device_components: mx.array | None = None,
    X_f32: np.ndarray | None = None,
) -> np.ndarray:
    """Transform data from the latent space back to the original feature space."""
    out_dtype = np.result_type(X.dtype, mean.dtype)
    n_samples = X.shape[0]
    k, n_features = components.shape
    if k != X.shape[1]:
        raise ValueError(
            f"X has {X.shape[1]} features, but PCA is expecting {k} components as input."
        )
    if k == 0:
        return np.broadcast_to(mean.astype(out_dtype, copy=False), (n_samples, n_features)).copy()

    if prefer_gpu and n_samples * n_features >= 65_536:
        X_f32 = ensure_f32_2d(X) if X_f32 is None else X_f32
        try:
            X_mx = borrow_mlx_f32(X_f32)
            if not whiten and device_components is not None and device_mean is not None:
                out_mx = mx.addmm(device_mean, X_mx, device_components)
                mx.eval(out_mx)
                return to_numpy(out_mx).astype(out_dtype, copy=False)
            comp = components.astype(np.float64, copy=True)
            if whiten:
                comp = np.sqrt(explained_variance.astype(np.float64)[:, np.newaxis]) * comp
            comp_mx = to_mlx_f32(comp.astype(np.float32))
            mean_mx = device_mean if device_mean is not None else to_mlx_f32(mean.astype(np.float32))
            out_mx = mx.addmm(mean_mx, X_mx, comp_mx)
            mx.eval(out_mx)
            return to_numpy(out_mx).astype(out_dtype, copy=False)
        finally:
            mx.synchronize()

    if whiten:
        scaled_components = np.sqrt(explained_variance[:, np.newaxis]) * components
        return (X @ scaled_components + mean).astype(out_dtype, copy=False)
    return (X @ components + mean).astype(out_dtype, copy=False)


def pca_score_samples(
    X: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    explained_variance: np.ndarray,
    noise_variance: float | np.floating,
    *,
    prefer_gpu: bool = True,
    X_f32: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the log-likelihood of each sample in ``O(n d k)`` via the Woodbury identity.

    For probabilistic PCA with covariance :math:`C = V^T (\\Lambda - \\sigma^2 I) V + \\sigma^2 I`:

    * :math:`\\log |C| = (d - k) \\log(\\sigma^2) + \\sum_{i=1}^k \\log(\\lambda_i)`
    * :math:`x_c^T C^{-1} x_c = \\frac{1}{\\sigma^2} \\|x_c\\|^2 - \\sum_{i=1}^k \\left(\\frac{1}{\\sigma^2} - \\frac{1}{\\lambda_i}\\right) (x_c \\cdot v_i)^2`
    """
    out_dtype = np.result_type(X.dtype, mean.dtype)
    n_samples, n_features = X.shape
    k = components.shape[0]
    nv = float(noise_variance)
    ev = np.maximum(explained_variance.astype(np.float64, copy=False), nv)

    if prefer_gpu and n_samples * n_features >= 65_536:
        X_f32 = ensure_f32_2d(X) if X_f32 is None else X_f32
        try:
            X_mx = borrow_mlx_f32(X_f32)
            mean_mx = to_mlx_f32(mean.astype(np.float32, copy=False))
            X_c_mx = X_mx - mean_mx
            row_ss_mx = mx.sum(X_c_mx * X_c_mx, axis=1)
            if k > 0:
                V_mx = to_mlx_f32(components.T.astype(np.float32, copy=False))
                Z_mx = X_c_mx @ V_mx
                mx.eval(row_ss_mx, Z_mx)
                row_ss = to_numpy(row_ss_mx).astype(np.float64)
                Z = to_numpy(Z_mx).astype(np.float64)
                weights = (1.0 / nv) - (1.0 / ev)
                quad = (row_ss / nv) - ((Z * Z) @ weights)
            else:
                mx.eval(row_ss_mx)
                row_ss = to_numpy(row_ss_mx).astype(np.float64)
                quad = row_ss / nv
        finally:
            mx.synchronize()
    else:
        X_c = (X - mean).astype(np.float64, copy=False)
        row_ss = np.einsum("ij,ij->i", X_c, X_c)
        if k > 0:
            Z = X_c @ components.T.astype(np.float64, copy=False)
            weights = (1.0 / nv) - (1.0 / ev)
            quad = (row_ss / nv) - ((Z * Z) @ weights)
        else:
            quad = row_ss / nv

    log_det = (n_features - k) * math.log(nv) + float(np.sum(np.log(ev)))
    ll = -0.5 * (quad + log_det + n_features * math.log(2.0 * math.pi))
    return ll.astype(out_dtype, copy=False)
