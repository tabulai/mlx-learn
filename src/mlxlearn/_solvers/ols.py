"""Ordinary least squares, in MLX.

Same contract as every solver here: arrays and a frozen :class:`OLSConfig` in, a
frozen :class:`OLSSolution` out. No estimator, no scikit-learn import, no global
configuration.

The objective is scikit-learn's, exactly
------------------------------------------
For weights ``w`` (the intercept excluded) and per-sample weights ``sw``::

    J(w, b) = (1 / (2 · sw_sum)) · Σᵢ swᵢ · ‖xᵢ w + b − yᵢ‖²

That is residual sum of squares, scaled so that unit sample weights recover mean
squared error. There is no penalty. The intercept is the weighted mean residual
after the slope is fitted, which is what centering the design and the target
before solving produces in exact arithmetic.

How the system is solved
------------------------
scikit-learn's dense path calls ``linalg.lstsq`` (an SVD of the design). MLX's
SVD is CPU-only — ``mx.linalg.svd`` on the GPU raises ``Metal SVD NYI`` — so the
device does the work SVD would have spent on a tall matrix anyway: the products
``XᵀX`` and ``Xᵀy`` (or ``XXᵀ`` when there are more features than rows). The
small Gram matrix is then factored on the host. In exact arithmetic this is the
same min-norm least-squares solution lstsq returns; in float32 the Gram has a
squared condition number, which is why coefficient parity lives at the
tolerances in ``docs/fp32_policy.md`` §3 and the objective is the strong
assertion.

A fit is a pure function of its arguments. Nothing about a previous fit in the
same process can reach this one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._common.arrays import NUMPY_FLOAT
from .._common.device import use_device

__all__ = [
    "OLSConfig",
    "OLSSolution",
    "solve_ols",
    "reference_objective",
]


@dataclass(frozen=True, slots=True)
class OLSConfig:
    """Problem-independent settings for one ordinary-least-squares fit.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Center ``X`` and ``y`` (with ``sample_weight`` when given) and restore
        the intercept afterwards. Without an intercept the raw design is solved,
        because shifting it would be a different problem.
    rcond : float, default=1e-6
        Relative cutoff on the singular values of the (centered, weighted)
        design, matching scikit-learn's ``LinearRegression.tol`` passed to
        ``lstsq``. Singular values smaller than ``rcond · σ_max`` are treated
        as zero.
    device : {"auto", "gpu", "cpu"}, default="auto"
        Passed straight to :func:`~mlxlearn._common.device.use_device`.
    """

    fit_intercept: bool = True
    rcond: float = 1e-6
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.rcond < 0 or not math.isfinite(self.rcond):
            raise ValueError(f"rcond must be finite and non-negative, got {self.rcond}")


@dataclass(frozen=True, slots=True)
class OLSSolution:
    """Fitted coefficients and the factorization diagnostics lstsq reports.

    Attributes
    ----------
    coef : ndarray of shape (n_targets, n_features), float64
        Always two-dimensional; the estimator ravels a single target.
    intercept : ndarray of shape (n_targets,), float64
        Zeros when ``fit_intercept=False``.
    rank : int
        Effective rank of the (centered, weighted) design after ``rcond``.
    singular : ndarray of shape (min(n_samples, n_features),), float64
        Singular values of that design, descending.
    residual_sum_of_squares : float
        Unscaled weighted residual sum of squares of the restored model, in
        float64. The quantity :func:`reference_objective` reports is this
        divided by ``2 · sw_sum``.
    """

    coef: np.ndarray
    intercept: np.ndarray
    rank: int
    singular: np.ndarray
    residual_sum_of_squares: float


def solve_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    config: OLSConfig,
) -> OLSSolution:
    """Fit ordinary least squares.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Dense and finite. Converted to float32 for the device products.
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Numeric target. A one-dimensional ``y`` is treated as a single target
        internally; the returned ``coef`` is still ``(1, n_features)``.
    sample_weight : ndarray of shape (n_samples,), optional
        float64, non-negative. ``None`` means unweighted, and takes a code path
        that never multiplies by ones. All-zero weights raise ``ValueError``
        with a message scikit-learn 1.9's zero-weight check recognizes.
    config : OLSConfig

    Returns
    -------
    OLSSolution
    """
    X_host = np.ascontiguousarray(X, dtype=np.float64)
    y_host = np.asarray(y, dtype=np.float64)
    n_samples, n_features = X_host.shape
    if y_host.shape[0] != n_samples:
        raise ValueError(
            f"y must have {n_samples} samples to match X, got shape {y_host.shape}"
        )
    if y_host.ndim == 1:
        y_host = y_host.reshape(-1, 1)
    elif y_host.ndim != 2:
        raise ValueError(f"y must be 1d or 2d, got shape {y_host.shape}")

    if sample_weight is None:
        sw = None
        sw_sum = float(n_samples)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if sw.shape[0] != n_samples:
            raise ValueError(
                f"sample_weight has {sw.shape[0]} elements but X has {n_samples} samples."
            )
        sw_sum = float(sw.sum())
        if sw_sum <= 0.0:
            # scikit-learn 1.9's check_all_zero_sample_weights_error matches
            # r"weight.*zero|zero.*weight".
            raise ValueError(
                "Sample weights sum to zero: every sample weight is zero, so there "
                "is nothing to fit."
            )

    if config.fit_intercept and n_samples > 0:
        if sw is None:
            x_offset = X_host.mean(axis=0)
            y_offset = y_host.mean(axis=0)
        else:
            x_offset = (sw[:, None] * X_host).sum(axis=0) / sw_sum
            y_offset = (sw[:, None] * y_host).sum(axis=0) / sw_sum
        X_work = X_host - x_offset
        y_work = y_host - y_offset
    else:
        x_offset = np.zeros(n_features, dtype=np.float64)
        y_offset = np.zeros(y_host.shape[1], dtype=np.float64)
        X_work = X_host
        y_work = y_host

    if sw is not None:
        scale = np.sqrt(sw)
        X_work = X_work * scale[:, None]
        y_work = y_work * scale[:, None]

    X32 = np.ascontiguousarray(X_work, dtype=NUMPY_FLOAT)
    y32 = np.ascontiguousarray(y_work, dtype=NUMPY_FLOAT)

    coef_T, rank, singular = _lstsq_products(X32, y32, rcond=float(config.rcond), device=config.device)
    coef = np.ascontiguousarray(coef_T.T, dtype=np.float64)

    if config.fit_intercept:
        intercept = np.ascontiguousarray(y_offset - x_offset @ coef.T, dtype=np.float64)
    else:
        intercept = np.zeros(coef.shape[0], dtype=np.float64)

    rss = _residual_sum_of_squares(coef, intercept, X_host, y_host, sw)
    return OLSSolution(
        coef=coef,
        intercept=intercept,
        rank=rank,
        singular=singular,
        residual_sum_of_squares=rss,
    )


def _lstsq_products(
    X: np.ndarray,
    y: np.ndarray,
    *,
    rcond: float,
    device: str,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Min-norm least squares from device GEMMs of the Gram and the right-hand side.

    ``X`` is ``(n, p)``, ``y`` is ``(n, k)``. Returns ``coef.T`` as ``(p, k)``,
    the rank, and the singular values of ``X`` implied by the Gram.
    """
    import mlx.core as mx

    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        coef = np.zeros((n_features, y.shape[1]), dtype=np.float64)
        return coef, 0, np.zeros(0, dtype=np.float64)

    with use_device(device):
        Xm = mx.array(X, dtype=mx.float32)
        Ym = mx.array(y, dtype=mx.float32)
        if n_samples >= n_features:
            gram = Xm.T @ Xm
            rhs = Xm.T @ Ym
            mx.eval(gram, rhs)
            gram_np = np.asarray(gram, dtype=np.float64)
            rhs_np = np.asarray(rhs, dtype=np.float64)
            coef, rank, singular = _solve_gram(gram_np, rhs_np, rcond=rcond)
            return coef, rank, singular

        gram = Xm @ Xm.T
        mx.eval(gram)
        gram_np = np.asarray(gram, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        if y64.ndim == 1:
            y64 = y64.reshape(-1, 1)
        dual, rank, singular = _solve_gram(gram_np, y64, rcond=rcond)
        coef = np.asarray(X, dtype=np.float64).T @ dual
        return coef, rank, singular


def _solve_gram(
    gram: np.ndarray, rhs: np.ndarray, *, rcond: float
) -> tuple[np.ndarray, int, np.ndarray]:
    """Solve ``gram @ x = rhs`` as a truncated SVD of a Gram matrix.

    ``gram`` is ``AᵀA`` or ``AAᵀ``. Its eigenvalues are the squares of ``A``'s
    singular values, so ``rcond`` is applied to ``√λ`` the way ``lstsq`` applies
    it to ``σ``.
    """
    gram = 0.5 * (gram + gram.T)
    u, s_gram, vt = np.linalg.svd(gram, full_matrices=False)
    s_gram = np.maximum(s_gram, 0.0)
    singular = np.sqrt(s_gram)
    cutoff = rcond * (float(singular[0]) if singular.size else 0.0)
    keep = singular > cutoff
    inv = np.zeros_like(s_gram)
    positive = keep & (s_gram > 0)
    inv[positive] = 1.0 / s_gram[positive]
    rhs_2d = np.asarray(rhs, dtype=np.float64)
    if rhs_2d.ndim == 1:
        rhs_2d = rhs_2d.reshape(-1, 1)
    # gram = U Σ Vᵀ, so gram⁺ = V Σ⁺ Uᵀ.
    tmp = u.T @ rhs_2d
    solution = vt.T @ (inv[:, None] * tmp)
    return solution, int(keep.sum()), singular.astype(np.float64, copy=False)


def _residual_sum_of_squares(
    coef: np.ndarray,
    intercept: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> float:
    pred = X @ coef.T + intercept
    sq = np.sum((pred - y) ** 2, axis=1)
    if sample_weight is None:
        return float(sq.sum())
    return float(np.dot(sample_weight, sq))


def reference_objective(
    coef: np.ndarray,
    intercept: np.ndarray | float,
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """The OLS objective in float64 NumPy, for parity assertions.

    ``docs/fp32_policy.md`` §3 makes the objective the strong assertion and the
    coefficients the weak one. Both coefficient sets are scored here.

    Parameters
    ----------
    coef : ndarray of shape (n_features,) or (n_targets, n_features)
    intercept : float or ndarray of shape (n_targets,)
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
    sample_weight : ndarray of shape (n_samples,), optional

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn._solvers.ols import reference_objective
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0.0, 1.0, 2.0, 3.0])
    >>> float(round(reference_objective(np.array([1.0]), 0.0, X, y), 12))
    0.0
    """
    Xd = np.asarray(X, dtype=np.float64)
    yd = np.asarray(y, dtype=np.float64)
    b = np.asarray(intercept, dtype=np.float64)
    if yd.ndim == 1:
        pred = Xd @ np.asarray(coef, dtype=np.float64).ravel() + b
        sq = (pred - yd) ** 2
    else:
        W = np.atleast_2d(np.asarray(coef, dtype=np.float64))
        pred = Xd @ W.T + b
        sq = np.sum((pred - yd) ** 2, axis=1)

    n_samples = Xd.shape[0]
    if sample_weight is None:
        sw_sum = float(n_samples)
        total = float(sq.sum())
    else:
        sw = np.asarray(sample_weight, dtype=np.float64)
        sw_sum = float(sw.sum())
        total = float(np.dot(sw, sq))
    if sw_sum <= 0.0:
        return 0.0
    return 0.5 * total / sw_sum
