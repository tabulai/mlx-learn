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
The solver forms a Gram matrix on the smaller side of the design and uses a
symmetric eigensolver on the host. The CPU uses a float64 symmetric rank-k
product for small systems and wide fits. Larger products use binary32 with
float64 iterative refinement. For a centered wide design, an orthogonal
reflection removes the known null direction of the weighted design.

The spectrum must resolve the requested rank and the solution must satisfy
the normal-equation residual check. If either check fails, a direct float64
least-squares factorization of the original working design supplies the
coefficients and rank diagnostics.

A fit is a pure function of its arguments. Nothing about a previous fit in the
same process can reach this one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .._common.arrays import NUMPY_FLOAT
from .._common.device import resolve_device, use_device

__all__ = [
    "OLSConfig",
    "OLSSolution",
    "solve_ols",
    "reference_objective",
]


#: Enough for any Gram worth keeping as a preconditioner. Refinement converges
#: geometrically and stops early once the float64 residual stops falling.
_MAX_REFINEMENT_STEPS = 8

#: ``‖Xᵀr‖ / (σ_max ‖y‖)`` a refined solution must reach. Larger means the normal
#: equations do not hold and the float32 Gram was not a usable preconditioner.
_STATIONARITY_TOL = 1e-9


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
        Dense and finite. Centering, weighting and residual checks use float64.
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

    center_direction = None
    if config.fit_intercept and 1 < n_samples <= n_features:
        center_direction = (np.full(n_samples, 1.0 / math.sqrt(n_samples))
                            if sw is None else scale / math.sqrt(sw_sum))
    coef_T, rank, singular = _lstsq_products(
        X_work, y_work, rcond=float(config.rcond), device=config.device,
        center_direction=center_direction,
    )
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
    design: np.ndarray,
    target: np.ndarray,
    *,
    rcond: float,
    device: str,
    center_direction: np.ndarray | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Solve and certify the smaller Gram system on the requested device."""
    n_samples, n_features = design.shape
    if n_samples == 0 or n_features == 0:
        coef = np.zeros((n_features, target.shape[1]), dtype=np.float64)
        return coef, 0, np.zeros(0, dtype=np.float64)

    X, y = design, target
    omitted_norm = 0.0
    if center_direction is not None:
        # A Householder reflection sends the weighted constant direction to the
        # last row. Bound that discarded row before solving in its complement.
        projection = center_direction @ design
        omitted_norm = float(np.linalg.norm(projection))
        projected_cutoff = rcond * (float(np.linalg.norm(design)) / math.sqrt(n_samples))
        if rcond <= np.finfo(np.float64).eps * max(design.shape) or omitted_norm > projected_cutoff / 32:
            return _lstsq_direct(design, target, rcond=rcond)
        leading = center_direction[:-1, None] / (1 + center_direction[-1])
        X = design[:-1] - leading * (projection + design[-1])
        y = target[:-1] - leading * (center_direction @ target + target[-1])

    tall = X.shape[0] >= X.shape[1]
    device = resolve_device(device)
    # A float64 SYRK wins for the measured skinny and small wide CPU systems.
    double = device == "cpu" and (X.shape[1] <= 256 or (not tall and X.shape[0] <= 1024))
    if double:
        gram = linalg.blas.dsyrk(1.0, X.T, trans=int(not tall), lower=1)
    else:
        gram = _device_gram(X, tall=tall, device=device)
    if not np.isfinite(gram).all():
        return _lstsq_direct(design, target, rcond=rcond)

    apply_inverse, rank, singular = _factor_gram(gram, rcond=rcond, lower_only=double)
    largest = float(singular[0])
    # A small residual can hide a lost singular direction. Require the whole
    # reduced spectrum to clear the cutoff and floating-point error envelope.
    if double:
        eps = float(np.finfo(np.float64).eps)
        gram_error = 8 * eps * max(X.shape) * float(np.trace(gram))
        # A direction can resolve rank while still amplifying rounding in
        # its coefficient. Keep that regime on the direct least-squares path.
        coefficient_floor = largest * math.sqrt(eps / _STATIONARITY_TOL)
        uncertainty = max(math.sqrt(max(gram_error, 0.0)), coefficient_floor)
    else:
        uncertainty = largest * math.sqrt(float(np.finfo(NUMPY_FLOAT).eps) * max(X.shape))
    cutoff = math.hypot(rcond * largest, uncertainty) + (1 + rcond) * omitted_norm
    if singular[-1] <= cutoff:
        return _lstsq_direct(design, target, rcond=rcond)

    def step(residual):
        # The dual solution belongs to the row space, which gives minimum norm.
        return apply_inverse(X.T @ residual) if tall else X.T @ apply_inverse(residual)

    if double:
        coef = step(y)
        residual = target - design @ coef
    else:
        coef, residual = _refine(X, y, step)
        if center_direction is not None:
            residual = target - design @ coef
    limit = _STATIONARITY_TOL * largest * float(np.linalg.norm(target))
    if not np.linalg.norm(design.T @ residual) <= limit:
        return _lstsq_direct(design, target, rcond=rcond)
    if center_direction is not None:
        singular = np.append(singular, 0.0)
    return coef, rank, singular


def _device_gram(X: np.ndarray, *, tall: bool, device: str) -> np.ndarray:
    import mlx.core as mx

    from mlxlearn_device._binary32 import matmul

    with use_device(device):
        Xm = mx.array(np.ascontiguousarray(X, dtype=NUMPY_FLOAT), dtype=mx.float32)
        gram = matmul(Xm, Xm, transpose_a=True) if tall else matmul(Xm, Xm, transpose_b=True)
        mx.eval(gram)
        return np.asarray(gram, dtype=np.float64)


def _lstsq_direct(X: np.ndarray, y: np.ndarray, *, rcond: float) -> tuple[np.ndarray, int, np.ndarray]:
    coef, _, rank, singular = np.linalg.lstsq(X, y, rcond=rcond)
    return coef, int(rank), singular


def _refine(X: np.ndarray, y: np.ndarray, step) -> tuple[np.ndarray, np.ndarray]:
    """Iterate ``coef += step(residual)`` until the float64 residual stops falling."""
    coef = np.zeros((X.shape[1], y.shape[1]), dtype=np.float64)
    residual = np.array(y, dtype=np.float64, copy=True)
    best = float(np.sum(residual * residual))
    for _ in range(_MAX_REFINEMENT_STEPS):
        candidate = coef + step(residual)
        residual_next = y - X @ candidate
        norm = float(np.sum(residual_next * residual_next))
        if not norm < best:
            break
        coef, residual, best = candidate, residual_next, norm
    return coef, residual


def _factor_gram(gram: np.ndarray, *, rcond: float, lower_only: bool = False):
    """Factor a symmetric Gram matrix and apply its truncated pseudoinverse."""
    if not lower_only:
        gram = 0.5 * (gram + gram.T)
    eigenvalues, vectors = linalg.eigh(gram, driver="evd", lower=True, check_finite=False)
    roots = np.sqrt(np.maximum(eigenvalues, 0.0))
    keep = roots > rcond * (float(roots[-1]) if roots.size else 0.0)

    def apply_inverse(rhs: np.ndarray) -> np.ndarray:
        projected = vectors.T @ rhs
        projected[~keep] = 0
        projected[keep] /= eigenvalues[keep, None]
        return vectors @ projected

    return apply_inverse, int(keep.sum()), roots[::-1]


def _solve_gram(
    gram: np.ndarray, rhs: np.ndarray, *, rcond: float
) -> tuple[np.ndarray, int, np.ndarray]:
    """Solve ``gram @ x = rhs`` as a truncated SVD of a Gram matrix."""
    apply_inverse, rank, singular = _factor_gram(gram, rcond=rcond)
    rhs_2d = np.asarray(rhs, dtype=np.float64)
    if rhs_2d.ndim == 1:
        rhs_2d = rhs_2d.reshape(-1, 1)
    return apply_inverse(rhs_2d), rank, singular


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
