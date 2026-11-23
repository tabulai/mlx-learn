"""Ordinary least squares on the host, in float64, with certified rank decisions.

Same contract as every solver here: arrays and a frozen :class:`OLSConfig` in, a
frozen :class:`OLSSolution` out. No estimator, no scikit-learn import, no global
configuration.

The problem is scikit-learn's, exactly
--------------------------------------
For per-sample weights ``sw`` and ``fit_intercept=True``, scikit-learn centres
``X`` and ``y`` by their weighted means, scales the rows by ``sqrt(sw)`` and calls
LAPACK ``gelsd`` on the result with a relative singular-value cutoff ``rcond``.
The returned coefficients are the minimum-norm least-squares solution of the
truncated problem, ``rank_`` counts singular values above ``rcond * sigma_max`` and
``singular_`` lists all of them. This module returns the same three things.

How the system is solved
------------------------
1. **One threaded pass** builds the centred, weighted, augmented design
   ``A = [Xc | yc]`` in row chunks and accumulates ``X^T X`` and ``X^T Y`` chunk
   by chunk while each chunk is still in cache. Chunking also makes the rounding
   bound of the Gram tight: each entry carries at most
   ``gamma(rows_per_chunk + chunks)`` relative error instead of ``gamma(n)``.
2. **Eigenvalues** of the feature Gram give every singular value whose square
   stands well above that bound. A rigorous error budget ``E`` (Gram rounding plus
   the eigensolver's backward error) turns each eigenvalue into an interval.
3. **Small directions** (eigenvalues below ``max(K_ACC * E, RHO * lambda_max)``)
   are measured on the data itself: ``W = A V0`` is formed directly and its
   singular values come from a tiled QR, so nothing is squared. Cauchy
   interlacing bounds the smallest singular values of the design from above by
   those of ``W``; the Li-Li quadratic bound limits how far the coupling to the
   rest of the spectrum can pull them down. Together they certify ``rank_`` and
   give ``singular_`` to backward-stable accuracy.
4. **The solve** uses a Cholesky factor of the Gram (full rank) or the spectral
   pseudo-inverse with Ritz values for the small directions (truncated), and
   residual refinement against ``A`` itself. Refinement is accepted when it
   converges, or when it stalls at the accuracy a backward-stable solver reaches,
   ``u * (kappa + kappa^2 * ||r|| / (sigma_max ||b||))``.
5. **Tall designs not certified** by a Gram are reduced by a tiled QR: row tiles
   factored by ``dgeqrf`` on parallel threads (it releases the GIL), merged by
   ``dtpqrt``, then ``gelsd`` on the small triangular factor. Its singular values,
   rank and coefficients carry ``gelsd``'s own semantics; an answer is kept unless
   a singular value lies inside scikit-learn's rounding band of the cutoff.
6. **Everything else** runs scikit-learn's own arithmetic, call for call
   (``_stock_lstsq``): tiny and near-square designs, a cutoff outside (0, 1),
   float16, and ranks only stock's rounding can decide. Its result is stock's to
   the bit; it skips stock's extra copies and writes the Fortran-ordered design
   on parallel threads. ``gelsd`` output is checked for plausibility (Accelerate
   on macOS 27.2 can return garbage silently) and recomputed with ``gelss``.

Wide designs use the dual Gram ``Xc Xc^T`` with the same certification. With
an intercept the constant direction ``q`` of the centred design is deflated by
a reflection; the design then equals the deflated matrix plus a rank-one part
of norm at most ``||Xc^T q||``, which every drop decision clears and which the
reported singular values account for. They fall back to step 6.

Accelerate's blocked QR (``dgeqrf`` with a real workspace, which ``numpy.linalg.qr``
and ``scipy.linalg.qr`` use) overruns a heap buffer for some squarish shapes on
macOS. This module never calls ``numpy``/``scipy`` QR, and calls blocked
``dgeqrf`` only on leaf shapes a guard-malloc sweep found clean (see
``_LEAF_MIN_EXTRA_ROWS``).

A fit is a pure function of its arguments. Nothing about a previous fit in the
same process can reach this one.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.linalg import blas, lapack

__all__ = [
    "OLSConfig",
    "OLSSolution",
    "solve_ols",
    "reference_objective",
]

#: Unit roundoff of float64.
_U = 2.0**-53

#: Target size of one row chunk of the augmented design, in bytes.
_CHUNK_BYTES = 4 << 20

#: Group-1 eigenvalues must exceed ``_K_ACC * E``; their square roots are then
#: within ``1 / (2 * _K_ACC)`` of the singular values even under the rigorous
#: bound. Realistic rounding is many orders of magnitude smaller.
_K_ACC = 1e6

#: Eigenvalues below ``_RHO * lambda_max`` are measured on the data as well, so
#: small singular values are reported to backward-stable accuracy.
_RHO = 1e-6

#: The Ritz path pays for an eigendecomposition with vectors, the product
#: ``A V0`` and a QR of ``n x m0``, so it measures at most this fraction of the
#: directions. On long designs (``n >= _RITZ_LONG * d``) those costs are small
#: next to stock's QR and the limit rises: designs mixing column scales put up
#: to half their directions below the split (one-hot columns next to numeric
#: columns of scale 100) and gain 2-2.8x there, while on shorter designs a
#: measurement that a knife-edge rank then voids costs up to a third of the fit.
#: Graded spectra put 70% or more below the split and are never measured.
_RITZ_FRACTION = 0.3
_RITZ_FRACTION_LONG = 0.6
_RITZ_LONG = 16

#: The pivoted-Cholesky screen skips the eigensolver when more than the Ritz
#: limit plus this margin of directions look small. Its count tracks the
#: eigenvalue count closely (257 of 257, 377 of 377 at d = 512), so refused fits
#: stop before the eigensolver.
_SCREEN_MARGIN = 0.05

#: Largest side for which an eigendecomposition with vectors is cheap enough to
#: risk before the exact path. Up to twice that the vectors still cost only about
#: 1.5x the eigenvalues already computed, so a handful of small directions
#: (``_RITZ_FEW``) is measured there too.
_RITZ_MAX_SIDE = 1024
_RITZ_FEW = 32

#: Above this width the tiled QR runs only with this many leaves of parallel
#: work; with fewer it barely beats gelsd, and a knife-edge rank then pays twice.
_QR_WIDE = 512
_QR_WIDE_MIN_PARTS = 8

#: Below this ``n * (d + k)^2`` the exact path (one gelsd call) is the fastest.
_SMALL_WORK = 1 << 19

#: Designs closer to square than this ratio go straight to the exact path.
_SQUARE_RATIO = 1.7

#: Refinement steps before giving up on a Gram-based solve.
_MAX_STEPS = 12

#: The tiled QR only pays once a design is this tall.
_QR_MIN_ROWS = 2048

#: Below this many elements a fit runs single-threaded.
_PARALLEL_MIN_ELEMENTS = 1 << 18

_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class OLSConfig:
    """Problem-independent settings for one ordinary-least-squares fit.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Centre ``X`` and ``y`` (with ``sample_weight`` when given) and restore
        the intercept afterwards.
    rcond : float, default=1e-6
        Relative cutoff on the singular values of the (centred, weighted)
        design: singular values ``<= rcond * sigma_max`` are dropped, as LAPACK
        ``gelsd`` drops them. The estimator passes scikit-learn's cutoff for the
        installed version.
    threads : int, default=0
        Host threads for the row passes and the tiled QR. ``0`` chooses
        automatically.
    """

    fit_intercept: bool = True
    rcond: float = 1e-6
    threads: int = 0

    def __post_init__(self) -> None:
        if not (self.rcond >= 0 and math.isfinite(self.rcond)):
            raise ValueError(f"rcond must be finite and non-negative, got {self.rcond}")
        if self.threads < 0:
            raise ValueError(f"threads must be non-negative, got {self.threads}")


@dataclass(frozen=True, slots=True)
class OLSSolution:
    """Fitted coefficients and the factorization diagnostics ``gelsd`` reports.

    Attributes
    ----------
    coef : ndarray of shape (n_targets, n_features), float64
        Always two-dimensional; the estimator ravels a single target.
    intercept : ndarray of shape (n_targets,), float64
        Zeros when ``fit_intercept=False``.
    rank : int
        Effective rank of the (centred, weighted) design after ``rcond``.
    singular : ndarray of shape (min(n_samples, n_features),), float64
        Singular values of that design, descending.
    method : str
        Which path produced the solution: ``"gram"`` (certified full rank),
        ``"ritz"`` (certified, with small directions measured on the data),
        ``"dual"`` (wide design with an intercept, through the deflated dual
        Gram), ``"qr"`` (tiled QR, then ``gelsd`` on the triangular factor),
        ``"lstsq"`` (scikit-learn's own arithmetic, bit for bit) or ``"gelss"``
        (that arithmetic with ``gelss``, after ``gelsd`` returned an implausible
        result).
    """

    coef: np.ndarray
    intercept: np.ndarray
    rank: int
    singular: np.ndarray
    method: str


# ------------------------------------------------------------------------------ entry


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
        Dense and finite. float32 and float16 keep their dtype, as in
        scikit-learn: the design is centred and weighted in it. float32 is then
        widened exactly and solved in float64; float16 runs scikit-learn's own
        arithmetic. Any other dtype is converted to float64. Never modified.
        The memory layout is kept, because the exact fallback reproduces
        scikit-learn's arithmetic, whose column means depend on it.
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Numeric target, cast to ``X``'s working dtype as scikit-learn casts it.
    sample_weight : ndarray of shape (n_samples,), optional
        Non-negative, in the dtype scikit-learn's ``_check_sample_weight`` gives
        for ``X``. With ``fit_intercept=True`` the weights must not sum to zero
        (the weighted means are undefined).
    config : OLSConfig

    Returns
    -------
    OLSSolution
    """
    X = np.asarray(X)
    if X.dtype not in (np.float64, np.float32, np.float16):
        X = np.asarray(X, dtype=np.float64)
    working = X.dtype
    y_work = np.asarray(y, dtype=working)
    n_samples, n_features = X.shape
    if y_work.shape[0] != n_samples:
        raise ValueError(f"y must have {n_samples} samples to match X, got shape {y_work.shape}")
    if y_work.ndim not in (1, 2):
        raise ValueError(f"y must be 1d or 2d, got shape {y_work.shape}")
    n_targets = 1 if y_work.ndim == 1 else y_work.shape[1]
    sw_work = sw = None
    if sample_weight is not None:
        # Keep the dtype scikit-learn's _check_sample_weight chose (the estimator
        # validates with it): float32 for float32 X, but float64 for float16 X.
        sw_work = np.asarray(sample_weight).reshape(-1)
        if sw_work.dtype.kind != "f":
            sw_work = sw_work.astype(np.float64)
        if sw_work.shape[0] != n_samples:
            raise ValueError(f"sample_weight has {sw_work.shape[0]} elements but X has {n_samples} samples.")
        sw = sw_work.astype(np.float64, copy=False)
        if not np.isfinite(sw).all():
            # Only reachable with NaN or infinite weights (a NaN scalar weight, or
            # assume_finite): stock's arithmetic then raises its own error.
            return _stock_lstsq(X, y_work, sw_work, config.fit_intercept, float(config.rcond))
        if config.fit_intercept and float(np.sum(sw)) == 0.0:
            # The weighted means are undefined; np.average raises the same.
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")

    columns = n_features + n_targets
    threads = _thread_count(n_samples * columns, config.threads)
    rcond = float(config.rcond)
    eps = float(np.finfo(working).eps)
    if not 0.0 < rcond < 1.0 or working == np.float16:
        # LAPACK's dlalsd replaces a cutoff <= 0 or >= 1 with machine epsilon:
        # a rounding-level cutoff that only stock's own arithmetic reproduces.
        # float16 input keeps scikit-learn's float16 arithmetic, as stock does.
        return _stock_lstsq(X, y_work, sw_work, config.fit_intercept, rcond, threads)
    if n_samples * columns ** 2 <= _SMALL_WORK:
        # Tiny problems: one gelsd call beats a dozen small BLAS/LAPACK calls,
        # and stock's arithmetic without its extra copies is bit-identical.
        return _stock_lstsq(X, y_work, sw_work, config.fit_intercept, rcond, threads)
    if max(n_samples, n_features) < _SQUARE_RATIO * min(n_samples, n_features):
        # Near-square designs are bound by the cubic small stage on every path;
        # stock's own computation, without its extra copies, is as fast as any.
        return _stock_lstsq(X, y_work, sw_work, config.fit_intercept, rcond, threads)
    with np.errstate(all="ignore"):
        prep = _stock_offsets(X, y_work, sw_work, config.fit_intercept)
        if n_samples >= n_features:
            out = _solve_tall(X, prep, sw, rcond, eps, threads)
        else:
            out = _solve_wide(X, prep, sw, rcond, eps, threads)
        if out is None:
            return _stock_lstsq(X, y_work, sw_work, config.fit_intercept, rcond, threads, prep)
        coef, rank, singular, method = out
        coef_T = np.ascontiguousarray(coef.T)
        if config.fit_intercept:
            intercept = prep.y_off64 - coef_T @ prep.x_off64
        else:
            intercept = np.zeros(n_targets)
    return OLSSolution(
        coef=coef_T,
        intercept=np.asarray(intercept, dtype=np.float64),
        rank=int(rank),
        singular=np.ascontiguousarray(singular, dtype=np.float64),
        method=method,
    )


def _stock_lstsq(X, y, sw, fit_intercept, rcond, threads=1, prep=None) -> OLSSolution:
    """scikit-learn's dense computation, operation for operation.

    Same arrays (``X``'s working dtype and memory layout), same NumPy calls in
    the same order (``numpy.average``, in-place centring, in-place ``sqrt(sw)``
    scaling) and the same LAPACK ``gelsd``, so the result is stock's to the bit.
    It decides the cases whose rank sits inside scikit-learn's own rounding.
    Floating-point warnings are silenced (mlxlearn's internals must not warn);
    errors, such as a non-finite centred design, still raise as in stock.
    """
    with np.errstate(all="ignore"):
        return _stock_lstsq_quiet(X, y, sw, fit_intercept, rcond, threads, prep)


def _stock_lstsq_quiet(X, y, sw, fit_intercept, rcond, threads, prep) -> OLSSolution:
    y = np.asarray(y)
    k = 1 if y.ndim == 1 else y.shape[1]
    x_off = y_off = None
    if fit_intercept:
        if prep is None:
            # On y itself: a reduction rounds as on stock's copy of it (same layout).
            x_off = np.average(X, axis=0, weights=sw).astype(X.dtype, copy=False)
            y_off = np.average(y, axis=0, weights=sw)
        else:
            # The same numpy.average calls on the same arrays (_stock_offsets).
            x_off = prep.x_off
            y_off = prep.y_off if y.ndim > 1 else prep.y_off[0]
    root = None if sw is None else (np.sqrt(sw) if prep is None else prep.root)
    # A tall design with 16 or more columns goes to gelsd as a Fortran-ordered
    # copy it may overwrite: scipy then skips its own (single-threaded,
    # transposing) copy, and the result is stock's to the bit. Narrower designs
    # gain nothing from it. A wide one keeps stock's layout and lets scipy copy
    # it, as stock does: scipy lays out its copy of a wide matrix differently,
    # and an in-place Fortran copy differed from stock in the last bit for about
    # one wide shape in ten. The targets always take stock's route.
    tall = X.shape[0] >= X.shape[1] >= _INPLACE_MIN_FEATURES
    order = "F" if tall else "K"
    ys = _stock_targets(y, y_off, root)
    Xs = _stock_design(X, x_off, root, threads, order)
    if Xs.dtype == np.float16:
        # scipy runs float16 through single-precision LAPACK (an exact
        # widening); doing it here is bit-identical, and avoids scipy's
        # float16 deprecation (removal is planned for SciPy 1.20).
        Xs = Xs.astype(np.float32, order=order)
        ys = ys.astype(np.float32)
    # check_finite as stock: centring can overflow, and stock then raises.
    coef, _, rank, singular = linalg.lstsq(Xs, ys, cond=rcond, overwrite_a=tall)
    method = "lstsq"
    del Xs
    cols = _plausible_columns(k)
    checked = ys.reshape(ys.shape[0], -1)[:, cols].astype(np.float64)
    if not _plausible_lstsq(X, x_off, root, checked, cols, coef, rank, singular):
        # Accelerate's dgelsd on macOS 27.2 can return a garbage rank, unsorted
        # singular values and a solution worse than zero (its SVD does not
        # converge and nothing reports it). dgelss applies the same truncation
        # rule through a QR-iteration SVD (given gelsd's effective cutoff, see
        # _gelss_cond); use it rather than stock's garbage.
        Xs = _stock_design(X, x_off, root, threads, order)
        if Xs.dtype == np.float16:
            Xs = Xs.astype(np.float32, order=order)
        coef, _, rank, singular = linalg.lstsq(Xs, ys, cond=_gelss_cond(rcond, Xs.dtype), overwrite_a=tall,
                                               lapack_driver="gelss")
        method = "gelss"
    # _set_intercept casts coef_ to the working dtype before using it, and only
    # with an intercept (a no-op unless scipy computed in a wider type, as it
    # does for float16: without an intercept stock keeps scipy's float32).
    coef = coef.T.astype(X.dtype, copy=False) if fit_intercept else coef.T
    if fit_intercept:
        intercept = y_off - (x_off @ coef if coef.ndim == 1 else x_off @ coef.T)
    else:
        intercept = np.zeros(k)
    return OLSSolution(
        coef=np.ascontiguousarray(np.reshape(coef, (k, -1)), dtype=np.float64),
        intercept=np.reshape(np.asarray(intercept, dtype=np.float64), (k,)),
        rank=int(rank),
        singular=np.ascontiguousarray(singular, dtype=np.float64),
        method=method,
    )


def _gelss_cond(rcond: float, dtype) -> float:
    """The cutoff gelsd actually applies, for a gelss call that must match it.

    dlalsd replaces a cutoff <= 0 or >= 1 with LAPACK's machine epsilon (2^-53
    in double, 2^-24 in single precision, which float16 also runs in); dgelss
    takes it literally (everything dropped at 1, nothing at 0).
    """
    if 0.0 < rcond < 1.0:
        return rcond
    return 2.0 ** -53 if np.dtype(dtype) == np.float64 else 2.0 ** -24


def _stock_targets(y, y_off, root):
    """Stock's ``y -= y_offset; y *= sqrt_sw`` on a copy in y's dtype and layout.

    The copy keeps y's memory order (``np.array(..., copy=True)``), as stock's
    ``check_array(y, copy=True)`` does, and gelsd never overwrites it: letting
    it overwrite a Fortran-ordered copy changed the last bit in some fits.
    """
    ys = np.array(y, copy=True)
    if y_off is not None:
        ys -= y_off
    if root is not None:
        ys *= root if ys.ndim == 1 else root[:, None]
    return ys


def _stock_design(X, x_off, root, threads=1, order="F"):
    """Stock's centred, weighted design, in its dtype, as a working copy.

    The offsets are reductions, so they are taken on X in the layout stock sees;
    centring and scaling are elementwise, so a copy in any layout rounds
    identically. ``order="F"`` (tall designs) saves ``lstsq`` its own copy;
    ``order="K"`` keeps X's layout, as stock's ``check_array(X, copy=True)``.
    Row blocks of about 1 MB are written on parallel threads. A Fortran block is
    centred in a C-ordered temporary and then copied across: writing the
    transpose element by element is up to 8x slower when the column length is
    a power of two (the columns alias in the cache).
    """
    n, d = X.shape
    transpose = order == "F"
    Xs = np.empty_like(X, order=order)

    def fill(blocks):
        # Every write is final, so a block computed twice cannot be scaled twice.
        for a, b in blocks:
            if root is None and not transpose:
                if x_off is not None:
                    np.subtract(X[a:b], x_off, out=Xs[a:b])
                else:
                    Xs[a:b] = X[a:b]
                continue
            if x_off is None and root is None:
                Xs[a:b] = X[a:b]
                continue
            block = X[a:b] - x_off if x_off is not None else np.array(X[a:b])
            if root is not None:
                np.multiply(block, root[a:b, None], out=block, casting="unsafe")
            Xs[a:b] = block

    rows = max(64, (1 << 17) // max(d, 1))
    blocks = [(a, min(n, a + rows)) for a in range(0, n, rows)]
    if threads <= 1 or len(blocks) < 2:
        fill(blocks)
    else:
        _map(fill, [blocks[i::threads] for i in range(threads)], threads)
    return Xs


_PLAUSIBLE_TARGETS = 8

#: The residual part of the plausibility test runs from this many n * d * k.
_PLAUSIBLE_RESIDUAL_WORK = 1 << 18

#: Multiple of sqrt(max(n, d)) eps sigma_max ||b|| by which a correct gelsd
#: residual may exceed the optimum (its backward error) in the plausibility test.
_GELSD_BACKWARD = 32.0

#: Narrower working designs are stored zero-padded to this many columns
#: (Accelerate's GEMM and GEMV are several times slower below it).
_MIN_STORE_COLUMNS = 8

#: The exact path hands gelsd an in-place Fortran copy from this many features.
_INPLACE_MIN_FEATURES = 16


def _plausible_columns(k: int) -> np.ndarray:
    """The targets the residual test reads: at most _PLAUSIBLE_TARGETS, spread out.

    One SVD serves every target, so a failed gelsd shows on any of them.
    """
    return np.unique(np.linspace(0, k - 1, min(k, _PLAUSIBLE_TARGETS)).astype(int))


def _plausible_lstsq(X, x_off, root, target, cols, coef, rank, singular) -> bool:
    """Cheap consistency checks a correct gelsd result always passes.

    The rank lies in range; singular values are finite, non-negative and
    descending; and, on problems with at least _PLAUSIBLE_RESIDUAL_WORK
    entries, no residual exceeds the zero solution's by more than rounding and
    gelsd's backward error. The failure seen on macOS 27.2 (an SVD that does not
    converge) corrupts the rank and the spectrum, which the first checks catch;
    on tiny problems the residual pass would cost as much as the solve. Nothing
    here squares the data, so the checks hold at any magnitude, and they are
    loose (1e-3 relative) so float32 runs pass too.
    """
    s = np.asarray(singular, dtype=np.float64)
    if not (0 <= int(rank) <= min(X.shape)) or not np.all(np.isfinite(s)) or np.any(s < 0):
        return False
    if s.size > 1 and np.any(np.diff(s) > 1e-3 * s[0]):
        return False
    coef64 = np.asarray(coef, dtype=np.float64).reshape(X.shape[1], -1)
    if X.size * coef64.shape[1] < _PLAUSIBLE_RESIDUAL_WORK:
        return True
    if not np.all(np.isfinite(coef64)):
        # Stock returns them too (a solution beyond float64 range); with a sane
        # rank and spectrum there is nothing more to check.
        return True
    coef64 = coef64[:, cols]
    # On stock's own centred matrix (no cancellation between X b and x_off b).
    pred = _centred_product64(X, x_off, coef64)
    if root is not None:
        pred *= root[:, None]
    # The slack covers rounding in the centred product (relative 1e-6) and the
    # backward error of gelsd itself: a backward-stable solution may
    # exceed the optimal residual by about eps sigma_max ||b||, which is large
    # when the cutoff keeps directions at the working precision's noise level.
    eps = float(np.finfo(np.float32 if X.dtype == np.float16 else X.dtype).eps)
    backward = _GELSD_BACKWARD * math.sqrt(max(X.shape)) * eps * (float(s[0]) if s.size else 0.0)
    if target.shape[1] == 1:  # the common case, without the vectorized overhead
        t, p = target[:, 0], pred[:, 0]
        slack1 = 1e-6 * _fast_norm(p) + backward * _fast_norm(coef64[:, 0])
        return bool(_fast_norm(t - p) <= (1 + 1e-3) * _fast_norm(t) + slack1)
    slack = 1e-6 * _column_norms(pred) + backward * _column_norms(coef64)
    residual = _column_norms(target - pred)
    return bool(np.all(residual <= (1 + 1e-3) * _column_norms(target) + slack))


def _centred_product64(X: np.ndarray, x_off, B: np.ndarray) -> np.ndarray:
    """``(X - x_off) @ B`` in float64, centred in X's dtype as stock centres.

    Row chunks bound the extra memory. Centring first avoids the cancellation
    between ``X b`` and ``x_off b`` on offset data with large coefficients, and
    the float64 product avoids float16's rounding, both of which would make a
    correct gelsd result look implausible.
    """
    if x_off is None and X.dtype == np.float64:
        return X @ B
    out = np.empty((X.shape[0], B.shape[1]))
    rows = max(1024, (1 << 20) // max(X.shape[1], 1))
    for a in range(0, X.shape[0], rows):
        block = X[a:a + rows] - x_off if x_off is not None else X[a:a + rows]
        out[a:a + rows] = block.astype(np.float64, copy=False) @ B
    return out


def _column_norms(M: np.ndarray) -> np.ndarray:
    """Euclidean norms of the columns of ``M`` (n x k), scaled so nothing overflows.

    Vectorized over the columns: a strided ``dnrm2`` per column costs one pass
    over ``M`` for each target. Non-finite columns give non-finite norms.
    """
    sq = np.einsum("ij,ij->j", M, M)
    if np.all((sq > 1e-280) & (sq < 1e280)):
        return np.sqrt(sq)  # no column near the range edges: squaring was safe
    top = np.max(np.abs(M), axis=0) if M.shape[0] else np.zeros(M.shape[1])
    safe = np.where((top > 0) & np.isfinite(top), top, 1.0)
    scaled = M / safe
    norms = safe * np.sqrt(np.einsum("ij,ij->j", scaled, scaled))
    return np.where(np.isfinite(top), np.where(top > 0, norms, 0.0), top)


def _rounding_band(eps, n, d, smax, frob):
    """How far scikit-learn's own computed singular values can sit from exact.

    ``gelsd`` is backward stable on the centred design it is given: about
    ``sqrt(max(n, d)) * eps * sigma_max``, plus ``eps * ||Xc||_F`` for the
    reductions inside it. mlxlearn forms that same centred design (stock's
    offsets and arithmetic), so the band does not need to cover centring. A rank
    decision closer to the cutoff than this depends on the order of stock's
    floating-point operations, so it is left to :func:`_stock_lstsq`.
    """
    return 4.0 * eps * (math.sqrt(max(n, d)) * smax + frob)


# ---------------------------------------------------------------------------- threads


def _sysctl(name: str) -> str | None:
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        size = ctypes.c_size_t(0)
        if libc.sysctlbyname(name.encode(), None, ctypes.byref(size), None, 0) != 0 or not size.value:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if libc.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, 0) != 0:
            return None
        raw = buf.raw[: size.value]
        if name.endswith(".name"):
            return raw.rstrip(b"\0").decode(errors="ignore")
        return str(int.from_bytes(raw, "little"))
    except (OSError, AttributeError, ValueError, TypeError):  # not macOS, or no such key
        return None


def _performance_cores() -> int:
    """Cores that are not efficiency cores (Apple's perflevels), else ``os.cpu_count()``."""
    total = 0
    for level in range(4):
        count = _sysctl(f"hw.perflevel{level}.physicalcpu")
        if count is None:
            break
        name = (_sysctl(f"hw.perflevel{level}.name") or "").lower()
        if level > 0 and "efficiency" in name:
            continue
        total += int(count)
    return total or (os.cpu_count() or 1)


_HOST_THREADS = max(1, min(32, _performance_cores()))


def _thread_budget() -> int:
    """Performance cores, capped by ``OMP_NUM_THREADS``/``VECLIB_MAXIMUM_THREADS`` when set.

    joblib's loky workers set ``OMP_NUM_THREADS`` to their share of the
    machine, so nested parallelism (e.g. cross-validation with ``n_jobs``) does
    not oversubscribe the cores.
    """
    budget = _HOST_THREADS
    for key in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        raw = os.environ.get(key, "")
        try:
            cap = int(raw.split(",")[0]) if raw else 0
        except ValueError:
            cap = 0
        if cap > 0:
            budget = min(budget, cap)
    return max(1, budget)


def _thread_count(elements: int, requested: int) -> int:
    if requested:
        return int(requested)
    if elements < _PARALLEL_MIN_ELEMENTS:
        return 1
    return int(max(1, min(_thread_budget(), elements // (_PARALLEL_MIN_ELEMENTS // 2))))


def _pool() -> ThreadPoolExecutor:
    """One process-wide pool, created lazily and never resized or replaced.

    Replacing it while another fit holds a reference would make that fit's
    next submission fail; a fit's own budget bounds its concurrency instead.
    """
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=max(4, _HOST_THREADS), thread_name_prefix="mlxlearn-ols")
        return _POOL


def _forget_pool_after_fork() -> None:
    # A forked child inherits the executor object but none of its threads;
    # submitting to it would wait forever. Start a fresh pool on demand.
    global _POOL, _POOL_LOCK
    _POOL, _POOL_LOCK = None, threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_pool_after_fork)


def _quiet(fn):
    """Run ``fn`` under ``np.errstate(all="ignore")`` (errstate is thread-local)."""

    def run(item):
        with np.errstate(all="ignore"):
            return fn(item)

    return run


def _map(fn, items: list, threads: int) -> list:
    """``[fn(item) for item in items]`` on at most ``threads`` pool workers, in order."""
    if threads <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    run = _quiet(fn)
    # With more work items than the budget each worker takes a fixed stripe, so
    # concurrency honours the budget; results keep their order either way.
    width = min(threads, len(items))
    units = [list(range(i, len(items), width)) for i in range(width)]

    def work(unit):
        return [run(items[i]) for i in unit]

    pool = _pool()
    futures = []
    for j, unit in enumerate(units):
        try:
            futures.append(pool.submit(work, unit))
        except RuntimeError:
            # The interpreter is shutting down (no new futures): run the units
            # not yet submitted here, and still collect the submitted ones, so
            # no item runs twice.
            futures.extend(_Ready(work(rest)) for rest in units[j:])
            break
    out = [None] * len(items)
    for unit, future in zip(units, futures, strict=True):
        for i, value in zip(unit, future.result(), strict=True):
            out[i] = value
    return out


class _Ready:
    """A finished stand-in for a future."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


def _chunk_rows(columns: int) -> int:
    # Narrow designs: a chunk that stays in cache between centring and its
    # Gram update. Wide ones: tall enough that the chunked SYRK runs at the
    # speed of a single call (2048 rows cost 12-14% more at 1025 columns).
    if columns >= 512:
        return 8192
    if columns >= 128:
        return 4096
    return int(min(16384, max(256, _CHUNK_BYTES // (8 * columns))))


def _groups(n_rows: int, rows: int, threads: int) -> tuple[list[list[tuple[int, int]]], int]:
    """Fixed row chunks, split into contiguous groups.

    The grouping depends on the machine (its performance cores), never on the
    thread budget, and group partials are summed in group order, so a fit is
    bit-identical for every budget on one machine.
    """
    bounds = [(a, min(n_rows, a + rows)) for a in range(0, n_rows, rows)]
    parts = max(1, min(_HOST_THREADS, len(bounds)))
    edges = np.linspace(0, len(bounds), parts + 1).astype(int)
    return [bounds[edges[i]:edges[i + 1]] for i in range(parts) if edges[i + 1] > edges[i]], len(bounds)


def _gamma(m: float) -> float:
    t = m * _U
    return t / (1.0 - t) if t < 0.5 else math.inf


def _nrm2(a: np.ndarray) -> float:
    """Scaled 2-norm: no underflow or overflow from squaring the entries."""
    return float(blas.dnrm2(np.ravel(a)))


def _fast_norm(v: np.ndarray) -> float:
    """2-norm of a contiguous float64 vector: one dot product, ``dnrm2`` near the range edges."""
    sq = float(np.dot(v, v))
    if 1e-280 < sq < 1e280:
        return math.sqrt(sq)
    return 0.0 if sq == 0.0 and not np.any(v) else _nrm2(v)


def _hypot_all(values: list[float]) -> float:
    """sqrt(sum(v^2)) without overflow or underflow."""
    top = max(values, default=0.0)
    if not (top > 0.0 and math.isfinite(top)):
        return top
    return top * math.sqrt(sum((v / top) ** 2 for v in values))


# ------------------------------------------------------------------------ eigensolver


def _padded(G: np.ndarray) -> np.ndarray | None:
    """Accelerate's dsyevd is markedly slower at multiples of 512 above ~1500.

    A decoupled diagonal entry moves the size off the pathology; its eigenvalue
    (or eigenvector, e_{d+1}) is removed afterwards. The pad equals the largest
    diagonal entry, which is at most lambda_max, so the error budget is unchanged.
    """
    d = G.shape[0]
    if d < 1536 or d % 512:
        return None
    P = np.zeros((d + 1, d + 1))
    P[:d, :d] = G
    P[d, d] = float(np.max(np.diag(G))) or 1.0
    return P


def _eigvalsh(G: np.ndarray) -> np.ndarray:
    P = _padded(G)
    if P is None:
        return linalg.eigh(G, eigvals_only=True, driver="evd", check_finite=False)
    lam = linalg.eigh(P, eigvals_only=True, driver="evd", check_finite=False)
    return np.delete(lam, int(np.argmin(np.abs(lam - P[-1, -1]))))


def _eigh(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = _padded(G)
    if P is None:
        return linalg.eigh(G, driver="evd", check_finite=False)
    lam, V = linalg.eigh(P, driver="evd", check_finite=False)
    keep = np.arange(P.shape[0]) != int(np.argmax(np.abs(V[-1])))
    return lam[keep], np.ascontiguousarray(V[:-1][:, keep])


# ------------------------------------------------------------------------ preparation


class _Prep:
    """scikit-learn's preprocessing values, computed with its own calls.

    ``x_off``/``y_off`` come from ``numpy.average`` on X and y in their working
    dtype and memory layout (stock's ``_average``), and ``root`` is
    ``sqrt(sample_weight)`` in the weights' dtype, so every centred, weighted
    entry built from them rounds exactly as stock's ``X -= X_offset;
    X *= sqrt_sw[:, None]``. Decisions then concern stock's own matrix.
    """

    __slots__ = ("y", "x_off", "y_off", "root", "x_off64", "y_off64", "k")

    def __init__(self, y, x_off, y_off, root):
        self.y = y.reshape(y.shape[0], -1)
        self.k = self.y.shape[1]
        self.x_off = x_off
        self.y_off = None if y_off is None else np.reshape(y_off, (-1,))
        self.root = root
        self.x_off64 = None if x_off is None else np.asarray(x_off, dtype=np.float64)
        self.y_off64 = None if y_off is None else np.asarray(self.y_off, dtype=np.float64)


def _stock_offsets(X, y_work, sw_work, fit_intercept) -> _Prep:
    x_off = y_off = None
    if fit_intercept:
        x_off = np.average(X, axis=0, weights=sw_work).astype(X.dtype, copy=False)
        y_off = np.average(y_work, axis=0, weights=sw_work)
    root = None if sw_work is None else np.sqrt(sw_work)
    return _Prep(y_work, x_off, y_off, root)


def _stock_block(X, prep, a, b, out):
    """Write stock's centred, weighted rows a:b of [X | y] into ``out`` (float64)."""
    d = X.shape[1]
    # Every write is final, so a chunk computed twice cannot be scaled twice:
    # weighted rows are centred and scaled in a temporary in X's dtype (as stock
    # rounds them), then written (widening float32 exactly).
    if prep.root is None and (X.dtype == np.float64 or prep.x_off is None):
        if prep.x_off is not None:
            np.subtract(X[a:b], prep.x_off, out=out[:, :d])
        else:
            out[:, :d] = X[a:b]
    else:
        block = X[a:b] - prep.x_off if prep.x_off is not None else np.array(X[a:b])
        if prep.root is not None:
            np.multiply(block, prep.root[a:b, None], out=block, casting="unsafe")
        out[:, :d] = block
    yb = prep.y[a:b] - prep.y_off if prep.y_off is not None else np.array(prep.y[a:b])
    if prep.root is not None:
        # In place in y's dtype, as stock's ``y *= sqrt_sw`` rounds.
        np.multiply(yb, prep.root[a:b, None], out=yb, casting="unsafe")
    out[:, d:] = yb


def _prepare_tall(X, prep, threads):
    """Stock's centred, weighted augmented design A = [Xc | yc] and its chunked Gram.

    Returns ``(A, M, depth, top)``: ``M`` holds ``X^T X`` and ``X^T Y``, and
    ``top`` the largest magnitude of each target column (for
    :func:`_targets_out_of_range`, gathered while the chunk is in cache). With
    few targets one SYRK on the augmented block is cheapest; with many, the
    k x k target block would dominate, so ``X^T X`` and ``X^T Y`` are separate
    products.
    """
    n, d = X.shape
    k = prep.k
    c = d + k
    rows = _chunk_rows(c)
    groups, n_chunks = _groups(n, rows, threads)
    # Accelerate's GEMM/GEMV take a slow path below 8 columns (a 16384 x 5 Gram
    # costs 110 us, padded to 8 columns 18 us): narrow designs live in a
    # zero-padded store, and A is its leading columns.
    width = max(c, _MIN_STORE_COLUMNS)
    store = np.zeros((n, width)) if width > c else np.empty((n, c))
    A = store[:, :c] if width > c else store
    augmented = k <= max(8, d // 4)

    def fill(chunks):
        M = np.zeros((width, width) if augmented else (d, c))
        top = np.zeros(k)
        for a, b in chunks:
            block = A[a:b]
            _stock_block(X, prep, a, b, block)
            if augmented:
                full = store[a:b]
                M += full.T @ full
            else:
                xb = block[:, :d]
                M[:d, :d] += xb.T @ xb
                M[:d, d:c] += xb.T @ block[:, d:]
            np.maximum(top, np.max(np.abs(block[:, d:]), axis=0), out=top)
        return M, top

    M = None
    top = None
    for part, part_top in _map(fill, groups, threads):
        M = part if M is None else M + part
        top = part_top if top is None else np.maximum(top, part_top)
    # Rows per chunk inside one product, then the chunk and thread sums.
    depth = min(rows, n) + n_chunks + len(groups)
    return A, M[:d, :c], depth, top


def _prepare_wide(X, prep, threads):
    Xc = _stock_design(X, prep.x_off, prep.root, threads, order="C")
    yc = _stock_targets(prep.y, prep.y_off, prep.root)
    return np.ascontiguousarray(Xc, dtype=np.float64), np.ascontiguousarray(yc, dtype=np.float64)


# ---------------------------------------------------------------------------- tiled QR

#: Blocked ``dgeqrf`` (a real workspace) runs only on leaves at least this much
#: taller than wide, and between ``_LEAF_MIN_COLUMNS`` and ``_LEAF_MAX_COLUMNS``
#: wide. Accelerate's blocked
#: QR overruns a heap buffer on some squarish shapes (600 x 600, 602 x 256-301);
#: a guard-malloc sweep of leaves with 4 to 1280 columns and 2048 to 8192 rows
#: more than that was clean. Other blocks use the unblocked kernel (minimal
#: workspace), which was clean on every shape tried, and the tiled QR path is
#: not taken for designs wider than the swept range.
_LEAF_MIN_EXTRA_ROWS = 2048
_LEAF_MIN_COLUMNS = 4
_LEAF_MAX_COLUMNS = 1280

_LWORK: dict[tuple[int, int], int] = {}


def _leaf_r(B: np.ndarray) -> np.ndarray:
    """Upper-triangular factor of a Fortran-ordered block (overwritten)."""
    m, c = B.shape
    if m >= c + _LEAF_MIN_EXTRA_ROWS and _LEAF_MIN_COLUMNS <= c <= _LEAF_MAX_COLUMNS:
        lwork = _LWORK.get((m, c))
        if lwork is None:
            if len(_LWORK) > 4096:
                _LWORK.clear()
            # The queried workspace, as in the guard-malloc sweep that cleared
            # these shapes.
            lwork = _LWORK[(m, c)] = max(1, int(lapack.dgeqrf_lwork(m, c)[0]))
        qr, _, _, info = lapack.dgeqrf(B, lwork=lwork, overwrite_a=1)
    else:
        qr, _, _, info = lapack.dgeqrf(B, overwrite_a=1)
    if info:
        raise np.linalg.LinAlgError(f"dgeqrf returned info={info}")
    return np.triu(qr[:c])


def _tile_rows(columns: int) -> int:
    return int(max(columns + _LEAF_MIN_EXTRA_ROWS, 4096))


def _qr_range(A: np.ndarray, a: int, b: int, tile: int) -> np.ndarray:
    """Triangular factor of rows ``a:b`` of ``A``: the running factor is stacked on each tile.

    ``dgeqrf`` releases the GIL, so ranges factor in parallel threads.
    """
    c = A.shape[1]
    edges = list(range(a, b, tile))
    if len(edges) > 1 and b - edges[-1] < tile // 2:
        edges.pop()  # the last tile absorbs a short remainder
    R = None
    for i, start in enumerate(edges):
        stop = edges[i + 1] if i + 1 < len(edges) else b
        top = 0 if R is None else c
        B = np.empty((top + stop - start, c), order="F")
        if R is not None:
            B[:c] = R
        B[top:] = A[start:stop]
        R = _leaf_r(B)
    if R.shape[0] < c:  # fewer rows than columns in a tiny range
        R = np.vstack([R, np.zeros((c - R.shape[0], c))])
    return R


def _qr_merge(pair: tuple[np.ndarray, np.ndarray], nb: int) -> np.ndarray:
    top, bottom = pair
    R, _, _, info = lapack.dtpqrt(0, nb, np.asfortranarray(top), np.asfortranarray(bottom),
                                  overwrite_a=1, overwrite_b=1)
    if info:
        raise np.linalg.LinAlgError(f"dtpqrt returned info={info}")
    return np.triu(R)


def _tiled_r(A: np.ndarray, threads: int) -> np.ndarray:
    """Upper-triangular S with S^T S = A^T A, by an orthogonal reduction of A's rows."""
    n, c = A.shape
    tile = _tile_rows(c)
    parts = int(max(1, min(_HOST_THREADS, n // tile)))
    if parts <= 1:
        return _qr_range(A, 0, n, tile)
    edges = np.linspace(0, n, parts + 1).astype(int)
    ranges = [(int(edges[i]), int(edges[i + 1])) for i in range(parts)]
    factors = _map(lambda ab: _qr_range(A, ab[0], ab[1], tile), ranges, threads)
    nb = max(1, min(32, c))
    # Merging two c x c triangles is small; dtpqrt holds the GIL but avoids the
    # squarish blocked-QR shapes (2c x c) that overrun in Accelerate.
    while len(factors) > 1:
        merged = [_qr_merge((factors[i], factors[i + 1]), nb) for i in range(0, len(factors) - 1, 2)]
        factors = merged + ([factors[-1]] if len(factors) % 2 else [])
    return factors[0]


def _tall_svd(W: np.ndarray, threads: int) -> tuple[np.ndarray, np.ndarray]:
    """Singular values (descending) and right singular vectors of a tall ``W``."""
    n, m = W.shape
    if n > m:
        # Reduce to a square triangle first, so gesdd never takes its own QR
        # branch on a tall shape (Accelerate's blocked QR is not trusted there).
        W = _tiled_r(np.ascontiguousarray(W), threads)
    _, s, Qt = linalg.svd(W, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    return s, Qt


# --------------------------------------------------------------------------- exact path


def _qr_tall(A: np.ndarray, d: int, rcond: float, band_of, threads: int):
    """Tiled QR of ``A``, then ``gelsd`` on its triangular factor.

    Returns ``None`` when a singular value lies within twice scikit-learn's
    rounding band of the cutoff: then stock's own rounding decides the rank, and
    only stock's arithmetic reproduces it.
    """
    S = _tiled_r(A, threads)
    if not np.isfinite(S).all():
        return None  # never hand a non-finite matrix to LAPACK
    R11, R12 = S[:d, :d], S[:d, d:]
    if d >= 1000:
        # Zero rows change neither the solution nor the singular values, and
        # keep Accelerate's gelsd off its slow square sizes.
        R11 = np.vstack([R11, np.zeros((8, d))])
        R12 = np.vstack([R12, np.zeros((8, R12.shape[1]))])
    coef, _, rank, singular = linalg.lstsq(R11, R12, cond=rcond, check_finite=False)
    cols = _plausible_columns(R12.shape[1])
    if not _plausible_lstsq(R11, None, None, R12[:, cols], cols, coef, rank, singular):
        # macOS 27.2's dgelsd can fail silently (see _stock_lstsq); dgelss
        # applies the same truncation rule (with gelsd's cutoff mapping).
        coef, _, rank, singular = linalg.lstsq(R11, R12, cond=_gelss_cond(rcond, R11.dtype), check_finite=False,
                                               lapack_driver="gelss")
    if not (singular.size and np.isfinite(singular).all()):
        return None
    band = band_of(float(singular[0]))
    tau = rcond * float(singular[0])
    if np.any(np.abs(singular - tau) <= 2 * band + rcond * band):
        return None
    if not np.isfinite(coef).all():
        return None
    return coef, int(rank), singular, "qr"


# ------------------------------------------------------------------------ certification


#: Verdict of :func:`_certify` when only scikit-learn's arithmetic can decide.
_STOCK = "stock"


def _certify(lam, E, tr, rcond, measure, side, band_of, fraction=_RITZ_FRACTION, extra_hi=0.0):
    """Decide the rank from a Gram's eigenvalues (ascending) and measured small directions.

    ``measure(mask)`` returns ``(sW, R0, epsW)`` for the directions in ``mask``:
    singular values of the design restricted to them (descending), the matching
    Ritz vectors as columns of ``R0`` in the Gram's space, and a bound on the
    product rounding. ``band_of(smax)`` is scikit-learn's rounding band; every
    decision must clear it. Returns ``None`` when the rank is not certified,
    else a dict describing the kept directions and the Ritz data.
    """
    lmax = float(lam[-1])
    if not (math.isfinite(tr) and tr > 1e-250 and lmax > 4 * E):
        return None
    band = band_of(math.sqrt(lmax))
    tau_lo = rcond * (math.sqrt(lmax - E) - band)
    tau_hi = rcond * (math.sqrt(lmax + E) + band)
    if band >= tau_lo and not np.sqrt(max(float(lam[0]) - E, 0.0)) - band > tau_hi:
        # The cutoff lies inside stock's own rounding noise even for a null
        # direction, and some direction is not certainly kept: no measurement
        # can decide this rank, only stock's arithmetic can.
        return _STOCK
    split = max(_K_ACC * E, _RHO * lmax)
    g0 = lam < split
    m0 = int(np.count_nonzero(g0))
    if m0 == 0:
        if not np.all(np.sqrt(np.maximum(lam - E, 0.0)) - band > tau_hi):
            return None  # truncation among well-resolved eigenvalues: rare (huge rcond)
        return {"m0": 0, "rank": side}
    if m0 > fraction * side:
        return None
    if side > _RITZ_MAX_SIDE and (side > 2 * _RITZ_MAX_SIDE or m0 > _RITZ_FEW):
        return None
    # A cost decision, not a correctness one: skip an expensive measurement that
    # would likely fail. An eigenvalue within 4E of zero may be an exact null
    # direction (one-hot columns beside the intercept), which cannot be dropped
    # once the product's own rounding (at least gamma(side + 2) sqrt(tr m0), see
    # ``measure``) reaches the cutoff: the tiny cutoff of scikit-learn 1.7/1.8.
    # It may also be a small but kept direction (columns in very different
    # units), which the measurement certifies; a few of those are always
    # measured, since W = A V0 is then cheap.
    eps_floor = _gamma(side + 2) * math.sqrt(tr) * math.sqrt(m0)
    if m0 > _RITZ_FEW and float(lam[0]) <= 4 * E and eps_floor + band >= tau_lo:
        return None
    g1 = ~g0
    lam1 = lam[g1]
    sW, R0, epsW = measure(g0)
    if not np.all(np.isfinite(sW)):
        return None
    vfac = 1.0 + 8 * side * _U
    b0max = ((float(sW[0]) + epsW) * vfac) ** 2
    l1min = float(lam1.min()) - E
    gap = l1min - b0max
    if not gap > 2 * E:
        return None
    coupling = E * E / gap
    # Li-Li: coupling can only pull the measured values down by `coupling`;
    # Cauchy interlacing bounds them from above with no coupling term. When the
    # measured matrix is the design with a rank-one part removed (the deflated
    # wide path: X^T X = Z^T Z + z z^T, ||z|| <= extra_hi), the design's values
    # can sit up to sqrt(s^2 + ||z||^2), which every drop must clear.
    s0_lo = np.sqrt(np.maximum((np.maximum(sW - epsW, 0.0) / vfac) ** 2 - coupling, 0.0))
    s0_hi = np.sqrt(((sW + epsW) * vfac) ** 2 + extra_hi * extra_hi)
    keep0 = s0_lo - band > tau_hi
    drop0 = s0_hi + band < tau_lo
    keep1 = np.sqrt(np.maximum(lam1 - E - coupling, 0.0)) - band > tau_hi
    if not ((keep0 | drop0).all() and keep1.all()):
        # A measured value so close to the cutoff that the tiled QR's own test
        # (``_qr_tall``: within 2 band + rcond band) would defer as well: skip
        # the QR and let stock's arithmetic decide at once.
        undecided = ~(keep0 | drop0)
        tau = rcond * math.sqrt(lmax)
        if keep1.all() and np.any(np.abs(sW[undecided] - tau) + 2 * epsW <= 2 * band + rcond * band):
            return _STOCK
        return None
    rank = int(np.count_nonzero(keep0)) + int(lam1.size)
    if keep0.any():
        s0k = sW[keep0] ** 2
        rho = E / (math.sqrt(float(s0k.min())) * math.sqrt(l1min)) + coupling / float(s0k.min())
        rho += 2 * epsW / float(sW[keep0].min())
        if not rho < 0.25:
            return None
    return {"m0": m0, "rank": rank, "g1": g1, "lam1": lam1, "sW": sW, "R0": R0, "keep0": keep0}


def _refine(residual, gradient, apply, b, smax, kappa, dim, steps_cap=_MAX_STEPS):
    """Residual refinement, accepted at the backward-stable floor.

    Returns ``(b, ok)``. Stops when the correction falls to rounding level, or
    when it stalls: then the stall level must sit below
    ``64 sqrt(dim) u (kappa + kappa^2 ||r|| / (smax ||b||))``, the accuracy a
    backward-stable solver attains on this problem.
    """
    prev = math.inf
    for step in range(1, steps_cap + 1):
        r = residual(b)
        delta = apply(gradient(r))
        b = b + delta
        nd, nb = _nrm2(delta), _nrm2(b)
        if not (math.isfinite(nd) and math.isfinite(nb)):
            return b, False
        if nb == 0.0 or nd <= 8 * _U * nb:
            return b, True
        if step >= 2 and nd > 0.5 * prev:
            eta = (r.norm if isinstance(r, _Pass) else _nrm2(r)) / (smax * nb)
            floor = 64 * math.sqrt(dim) * _U * (kappa + kappa * kappa * eta)
            return b, nd <= floor * nb
        prev = nd
    return b, False


# --------------------------------------------------------------------------- tall solve


def _solve_tall(X, prep, sw, rcond, eps, threads):
    n, d = X.shape
    A, M, depth, top = _prepare_tall(X, prep, threads)
    # The centred design is stock's own (same offsets, same rounding), so the
    # band only has to cover gelsd's backward error.
    frob = math.sqrt(max(float(np.trace(M[:d, :d])), 0.0))

    def band_of(smax):
        return _rounding_band(eps, n, d, smax, frob)

    out = _gram_tall(A, M, d, rcond, threads, depth, band_of, top)
    c = A.shape[1]
    if out is None and c - d <= d and c <= _LEAF_MAX_COLUMNS and n >= 2 * _tile_rows(c) and (
        c <= _QR_WIDE or n // _tile_rows(c) >= _QR_WIDE_MIN_PARTS
    ):
        # (With more targets than features the QR would mostly reduce targets;
        # gelsd applies its reflectors to many right-hand sides far cheaper.
        # Two full tiles guarantee blocked leaves and some parallel work.)
        out = _qr_tall(A, d, rcond, band_of, threads)
    if out is None or out is _STOCK:
        return None
    return out


def _ritz_fraction(long_side: int, side: int) -> float:
    """The largest fraction of directions the Ritz path measures (see _RITZ_FRACTION)."""
    return _RITZ_FRACTION_LONG if long_side >= _RITZ_LONG * side else _RITZ_FRACTION


def _too_many_small(G: np.ndarray, tr: float, depth: int, fraction: float) -> bool:
    """Pivoted-Cholesky screen: would the Ritz step refuse this many directions?

    ``dpstrf`` costs a tenth of the eigensolver and counts the pivots below the
    group-0 split closely (184 vs 179 at d = 256 on a log-spaced spectrum). It
    only skips work that would be discarded: the split uses a lower bound on
    lambda_max, and the limits sit just above the Ritz path's own (the fraction,
    and the few directions it measures past ``_RITZ_MAX_SIDE``).
    """
    d = G.shape[0]
    if d < 32:
        return False
    lmax_low = float(np.max(np.diag(G)))
    split = max(_K_ACC * (_gamma(depth) * tr + 4 * d * _U * lmax_low), _RHO * lmax_low)
    _, _, rank, info = lapack.dpstrf(G, tol=split, lower=1)
    if info < 0:
        return False
    small = d - int(rank)
    if d > 2 * _RITZ_MAX_SIDE:
        return small > 0
    if d > _RITZ_MAX_SIDE:
        return small > 2 * _RITZ_FEW
    return small > (fraction + _SCREEN_MARGIN) * d


def _targets_out_of_range(Y: np.ndarray, tr: float, top: np.ndarray | None = None) -> bool:
    """Targets whose cross products with X could underflow or overflow.

    Guards the Gram paths (X^T y, the refinement gradient) against a silent
    coef_ = 0 when |x| |y| leaves the normal range although X, y and the
    solution are ordinary floats. Such fits take the exact path.
    """
    # Bounds from each column's largest entry (one pass over Y, which can be
    # large with many targets): max|y| <= ||y|| <= sqrt(n) max|y|.
    sx = math.sqrt(tr)
    root_n = math.sqrt(Y.shape[0])
    if Y.shape[0] == 0:
        return False
    if top is None:
        top = np.maximum(np.max(Y, axis=0), -np.min(Y, axis=0))
    for v in top.tolist():
        if not math.isfinite(v) or v * root_n > 1e150 or (0.0 < v and (v < 1e-150 or sx * v < 1e-290)):
            return True
    return False


def _hopeless(lmax_low: float, lmax_high: float, rcond: float, band_of) -> bool:
    """The cutoff lies inside stock's rounding band even for a null direction."""
    band = band_of(math.sqrt(max(lmax_high, 0.0)))
    return band >= rcond * (math.sqrt(max(lmax_low, 0.0)) - band)


def _gram_tall(A, M, d, rcond, threads, depth, band_of, top=None):
    G = np.ascontiguousarray(M[:d, :d])
    C = np.ascontiguousarray(M[:d, d:])
    tr = float(np.trace(G))
    if not (math.isfinite(tr) and np.isfinite(C).all()):
        return None
    if not tr > 1e-250:
        return _STOCK  # a constant (or vanishing) design: stock's arithmetic decides
    if _targets_out_of_range(A[:, d:], tr, top):
        return _STOCK
    fraction = _ritz_fraction(A.shape[0], d)
    if _too_many_small(G, tr, depth, fraction):
        # Many small directions: the Ritz step would refuse. When the cutoff is
        # inside stock's own rounding band, no QR answer could be accepted either.
        lmax_low = float(np.max(np.diag(G)))
        return _STOCK if _hopeless(lmax_low, tr, rcond, band_of) else None
    lam = _eigvalsh(G)
    lmax = float(lam[-1])
    E = _gamma(depth) * tr * (1 + 4 * _U) + 4 * d * _U * max(lmax, 0.0)
    V = None

    def measure(mask):
        nonlocal V
        _, V = _eigh(G)
        V0 = V[:, mask]
        W = A[:, :d] @ V0  # row-strided view: BLAS reads it with lda = c, no copy
        sW, Qt = _tall_svd(W, threads)
        epsW = _gamma(d + 2) * math.sqrt(tr) * math.sqrt(V0.shape[1]) + 32 * V0.shape[1] * _U * float(sW[0])
        return sW, V0 @ Qt.T, epsW

    cert = _certify(lam, E, tr, rcond, measure, d, band_of, fraction)
    if cert is None or cert is _STOCK:
        return cert
    residual, gradient = _fused_refinement_pass(A, d, threads)

    smax = math.sqrt(lmax)
    if cert["m0"] == 0:
        try:
            factor = linalg.cho_factor(G, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return None

        def apply(g):
            return linalg.cho_solve(factor, g, check_finite=False)

        kappa = math.sqrt(lmax / float(lam[0]))
        b, ok = _refine(residual, gradient, apply, apply(C), smax, kappa, d, steps_cap=6)
        if not ok or not np.isfinite(b).all():
            return None
        return b, d, np.sqrt(lam)[::-1].copy(), "gram"

    b, ok, singular = _ritz_solve(cert, V, C, residual, gradient, smax)
    if not ok:
        return None
    return b, cert["rank"], singular, "ritz"


def _fused_refinement_pass(A: np.ndarray, d: int, threads: int):
    """``residual``/``gradient`` callables for :func:`_refine` that read ``A`` once per step.

    Each row chunk computes ``r = Y_chunk - X_chunk b`` and ``X_chunk^T r`` while
    the chunk is in cache (column views of the contiguous chunk, so BLAS reads
    them in place); chunk sums are added in a fixed order. ``residual`` returns a
    stand-in carrying the gradient and ``||r||``, so the step costs one pass of
    ``O(n d k)`` work for ``k`` targets.
    """
    n, c = A.shape
    k = c - d
    groups, _ = _groups(n, _chunk_rows(c), threads)
    # Below 8 columns Accelerate's GEMV is several times slower: the gradient
    # then reads the whole (zero-padded) row block and keeps its first d entries.
    base = A.base if A.base is not None and A.base.ndim == 2 and A.base.shape[0] == n else A
    wide_grad = d < _MIN_STORE_COLUMNS and base.shape[1] >= _MIN_STORE_COLUMNS

    def residual(b):
        def part(chunks):
            g = np.zeros((d, k))
            norms = []
            for a0, b0 in chunks:
                block = A[a0:b0]
                xb = block[:, :d]
                r = block[:, d:] - xb @ b
                g += (base[a0:b0].T @ r)[:d] if wide_grad else xb.T @ r
                norms.append(_nrm2(r))
            return g, norms

        g = np.zeros((d, k))
        norms: list[float] = []
        for gp, part_norms in _map(part, groups, threads):
            g += gp
            norms.extend(part_norms)
        return _Pass(g, _hypot_all(norms))

    return residual, lambda r: r.gradient


class _Pass:
    """One fused refinement pass: the gradient and the residual norm."""

    __slots__ = ("gradient", "norm")

    def __init__(self, gradient: np.ndarray, norm: float) -> None:
        self.gradient = gradient
        self.norm = norm


def _ritz_solve(cert, V, rhs, residual, gradient, smax):
    """Solve with the spectral pseudo-inverse, Ritz values standing in for small directions.

    The kept subspace comes from the Gram's eigenvectors and the Ritz vectors,
    accurate to about ``E / gap`` in angle, so the solution's component along
    dropped directions (zero in the exact minimum-norm solution) is of that
    relative size: about ``kappa^2 u`` when the gap sits at the cutoff. ``gelsd``
    leaves a component of order ``kappa u`` there; both are far below the
    coefficients' own sensitivity to rounding in the data.
    """
    V1, lam1 = V[:, cert["g1"]], cert["lam1"]
    keep0, sW = cert["keep0"], cert["sW"]
    K0 = cert["R0"][:, keep0]
    s0k = sW[keep0] ** 2

    def apply(g):
        x = V1 @ ((V1.T @ g) / lam1[:, None])
        if K0.shape[1]:
            x += K0 @ ((K0.T @ g) / s0k[:, None])
        return x

    smallest = float(lam1.min())
    if s0k.size:
        smallest = min(smallest, float(s0k.min()))
    kappa = smax / math.sqrt(smallest)
    b, ok = _refine(residual, gradient, apply, apply(rhs), smax, kappa, V.shape[0])
    singular = np.sort(np.concatenate([np.sqrt(np.maximum(lam1, 0.0)), sW]))[::-1]
    return b, ok and bool(np.isfinite(b).all()), singular


# --------------------------------------------------------------------------- wide solve


def _solve_wide(X, prep, sw, rcond, eps, threads):
    n, d = X.shape
    if prep.x_off is not None and rcond <= 4.0 * eps * math.sqrt(d):
        # Even an exact left null direction cannot clear stock's rounding band.
        return None
    Xc, yc = _prepare_wide(X, prep, threads)
    zero_rows = 0
    if sw is not None and not np.all(sw != 0):
        # Zero-weight rows are exact zero rows of the weighted design: they
        # change nothing but add exact zero singular values.
        rows = sw != 0
        zero_rows = n - int(np.count_nonzero(rows))
        Xc, yc, sw = Xc[rows], yc[rows], sw[rows]
        if Xc.shape[0] == 0 or max(Xc.shape) < _SQUARE_RATIO * min(Xc.shape):
            return None
    frob = math.sqrt(max(float(np.einsum("ij,ij->", Xc, Xc)), 0.0))

    def band_of(smax):
        return _rounding_band(eps, n, d, smax, frob)

    q = None
    if prep.x_off is not None:
        # The weighted centring makes q = sqrt(sw)/||sqrt(sw)|| a left null vector.
        q = np.full(Xc.shape[0], 1.0) if sw is None else np.sqrt(sw)
        q /= _nrm2(q)
    out = _gram_wide(Xc, yc, rcond, threads, band_of, q, zero_rows)
    if out is None:
        return None
    coef, rank, singular, method = out
    if zero_rows:
        singular = np.concatenate([singular, np.zeros(zero_rows)])
    return coef, rank, singular, method


_NULL_BLOCK = 32

#: The deflated wide path reports measured values of the deflated matrix only
#: while they are this close (relative to sigma_max) to the design's own.
_DEFLATED_REPORT_TOL = 1e-11


def _null_direction_bound(Xc: np.ndarray, q: np.ndarray, tr: float) -> tuple[float, np.ndarray]:
    """A rigorous upper bound on ``||Xc^T q||`` for a unit vector ``q``, and ``Xc^T q``.

    BLAS sums ``_NULL_BLOCK`` rows at a time, then the block sums are added
    pairwise, so the rounding is at most ``gamma(32 + log2(blocks) + 2)`` times
    ``|| |Xc|^T |q| || <= ||Xc||_F`` instead of ``gamma(n)``. The tighter bound
    matters for the cutoff of scikit-learn 1.7/1.8, about ``max(n, d) eps``.
    """
    n, d = Xc.shape
    full = n // _NULL_BLOCK
    rows = full * _NULL_BLOCK
    parts = []
    if full:
        blocks = Xc[:rows].reshape(full, _NULL_BLOCK, d)
        parts.append(np.matmul(q[:rows].reshape(full, 1, _NULL_BLOCK), blocks)[:, 0, :])
    if rows < n:
        parts.append((q[rows:] @ Xc[rows:])[None, :])
    partial = np.concatenate(parts) if len(parts) > 1 else parts[0]
    levels = 0
    while partial.shape[0] > 1:
        half = partial.shape[0] // 2
        top = partial[:half] + partial[half:2 * half]
        partial = np.concatenate([top, partial[2 * half:]]) if partial.shape[0] % 2 else top
        levels += 1
    frob = math.sqrt(tr / (1.0 - _gamma(n + d)))  # >= the exact ||Xc||_F
    rounding = _gamma(_NULL_BLOCK + levels + 2) * frob * (1 + 16 * _U)
    return _nrm2(partial[0]) * (1 + 16 * _U) + rounding, partial[0]


def _householder(q: np.ndarray):
    """``apply(M)`` = H M for the reflector H with H q = -sign(q[-1]) e_last."""
    v = q.copy()
    v[-1] += math.copysign(1.0, q[-1]) if q[-1] != 0 else 1.0
    scale = 2.0 / float(v @ v)

    def apply(M):
        return M - np.outer(v, scale * (v @ M)) if M.ndim == 2 else M - v * (scale * float(v @ M))

    return apply


def _gram_wide(Xc, yc, rcond, threads, band_of, q, zero_rows=0):
    """Minimum-norm solution through the dual Gram ``K = Xc Xc^T`` (n x n).

    With an intercept the known null direction ``q`` is deflated by a
    Householder reflection: ``K' = (H K H)[:-1, :-1] = Z Z^T`` is usually full
    rank. The design satisfies ``Xc^T Xc = Z^T Z + z z^T`` with ``z = Xc^T q``:
    its values are at least Z's (keep decisions stand) and at most
    ``sqrt(s^2 + ||z||^2)`` (drops must clear that, through ``extra_hi``); the
    deflated direction itself is certified below the cutoff by
    ``||z|| <= U_q`` (Courant-Fischer) and reported as ``||P z||``, P the
    projector off Z's row space, which is the design's smallest singular value
    to first order.
    """
    n, d = Xc.shape
    K = Xc @ Xc.T
    tr = float(np.trace(K))
    if not (math.isfinite(tr) and tr > 1e-250) or _targets_out_of_range(yc, tr):
        return None
    E = _gamma(d + 2) * tr * (1 + 4 * _U)
    H = None
    U_q = 0.0
    if q is not None:
        # Courant-Fischer: some singular value is at most ||Xc^T q|| (+ rounding).
        U_q, z = _null_direction_bound(Xc, q, tr)
        # Fail fast when even the most favourable bounds on sigma_max (the largest
        # row norm below, ||Xc||_F above) leave the constant direction undecided;
        # the exact check follows the eigenvalues.
        band_low = band_of(math.sqrt(float(np.max(np.diag(K)))))
        if not U_q + band_low < rcond * (math.sqrt(tr) - band_low):
            return None
        H = _householder(q)
        Kp = H(H(K).T).T[:-1, :-1]
        Kp = 0.5 * (Kp + Kp.T)
        E += 8 * n * _U * tr  # the two reflections
    else:
        Kp = K
    m = Kp.shape[0]
    fraction = _ritz_fraction(d, m)
    if _too_many_small(Kp, float(np.trace(Kp)), d + 2, fraction):
        return None
    lam = _eigvalsh(Kp)
    lmax = float(lam[-1])
    E += 4 * m * _U * max(lmax, 0.0)
    if zero_rows:
        # Dropped zero-weight rows are exact zero singular values here, but
        # rounding noise up to stock's band in stock's gelsd: they are dropped
        # in stock only if that band sits certainly below the cutoff.
        band = band_of(math.sqrt(max(lmax, 0.0) + E))
        if not band < rcond * (math.sqrt(max(lmax - E, 0.0)) - band):
            return None
    if H is not None:
        # H K H = [[K', b], [b^T, beta]] and K' drops b, so every eigenvalue of K
        # is within ||b|| <= ||K q|| <= sigma_max ||Xc^T q|| of K' (Weyl), and
        # sigma_max <= U_q + sqrt(lmax + E) bounds the exact largest value.
        E += (U_q + math.sqrt(max(lmax, 0.0) + E)) * U_q * (1 + 4 * _U)
        # The constant direction (singular value <= U_q, Courant-Fischer) must be
        # certainly below the cutoff, with sigma_max now known to within E.
        band = band_of(math.sqrt(max(lmax, U_q * U_q) + E))
        if not U_q + band < rcond * (math.sqrt(max(lmax - E, 0.0)) - band):
            return None

    def lift(alpha):
        """Rows of the reduced problem back to rows of Xc: H [alpha; 0]."""
        if H is None:
            return alpha
        full = np.zeros((n,) + alpha.shape[1:])
        full[:m] = alpha
        return H(full)

    def reduce_rows(v):
        return v if H is None else H(v)[:m]

    U = None

    def measure(mask):
        nonlocal U
        _, U = _eigh(Kp)
        U0 = U[:, mask]
        W = Xc.T @ lift(U0)
        sW, Qt = _tall_svd(W, threads)
        epsW = _gamma(n + 2) * math.sqrt(tr) * math.sqrt(U0.shape[1]) + 32 * U0.shape[1] * _U * float(sW[0])
        return sW, U0 @ Qt.T, epsW

    cert = _certify(lam, E, tr, rcond, measure, m, band_of, fraction, extra_hi=U_q)
    if cert is None or cert is _STOCK:
        return None
    if H is not None and cert["m0"] and U_q > _DEFLATED_REPORT_TOL * math.sqrt(max(lmax - E, 0.0)):
        # The measured values are those of the deflated matrix; the design's lie
        # within U_q of them, and singular_ must match stock to 1e-10 sigma_max.
        return None

    def residual(b):
        return yc - Xc @ b

    smax = math.sqrt(lmax)
    if cert["m0"] == 0:
        try:
            factor = linalg.cho_factor(Kp, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return None

        def apply_dual(v):
            return Xc.T @ lift(linalg.cho_solve(factor, reduce_rows(v), check_finite=False))

        kappa = math.sqrt(lmax / float(lam[0]))
        rank = m
        reduced_singular = np.sqrt(lam)
    else:
        U1, lam1 = U[:, cert["g1"]], cert["lam1"]
        keep0, sW = cert["keep0"], cert["sW"]
        K0 = cert["R0"][:, keep0]
        s0k = sW[keep0] ** 2

        def apply_dual(v):
            v = reduce_rows(v)
            alpha = U1 @ ((U1.T @ v) / lam1[:, None])
            if K0.shape[1]:
                alpha += K0 @ ((K0.T @ v) / s0k[:, None])
            return Xc.T @ lift(alpha)

        smallest = float(lam1.min())
        if s0k.size:
            smallest = min(smallest, float(s0k.min()))
        kappa = smax / math.sqrt(smallest)
        rank = cert["rank"]
        reduced_singular = np.concatenate([np.sqrt(np.maximum(lam1, 0.0)), sW])

    b, ok = _refine(residual, lambda r: r, apply_dual, apply_dual(yc), smax, kappa, n,
                    steps_cap=6 if cert["m0"] == 0 else _MAX_STEPS)
    if not ok or not np.isfinite(b).all():
        return None
    parts = [reduced_singular]
    if H is not None:
        if cert["m0"] == 0:
            # The design's smallest singular value is ||P z|| for z = Xc^T q and P
            # the projector off the deflated matrix's row space, to first order in
            # ||z|| / sigma_min (what stock's gelsd reports, not the bound U_q).
            t = linalg.cho_solve(factor, reduce_rows(Xc @ z), check_finite=False)
            parts.append(np.array([_nrm2(z - Xc.T @ lift(t))]))
        else:
            parts.append(np.array([U_q]))  # within _DEFLATED_REPORT_TOL sigma_max (checked above)
    singular = np.sort(np.concatenate(parts))[::-1]
    return b, rank, singular, ("dual" if H is not None else ("gram" if cert["m0"] == 0 else "ritz"))


# ------------------------------------------------------------------------- objective


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
