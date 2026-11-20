"""Conservative center-separation bounds for Euclidean binary32 Lloyd assignment."""

import mlx.core as mx

from mlxlearn_device._binary32 import matmul

from .e_means import selected_distances

_EPS = 2.0**-23
_MIN_NORMAL = 2.0**-126
_CENTER_TILE_BYTES = 64 << 20


def _allowance(d, squared_scale):
    # Each sum of d nonnegative squares or signed products has absolute
    # error <= gamma_(2d) times the sum of absolute products, allowing
    # separate rounding of both multiply and add. Cauchy's inequality gives
    # 2 sum |x_i c_i| <= ||x||² + ||c||². For d <= 1024 this larger factor
    # also covers the rounded input norms, Gram additions and guard arithmetic.
    # The additive term allows flush-to-zero in tiny products and partial sums.
    return (16 * (d + 8) * _EPS) * squared_scale + (32 * (d + 8) * _MIN_NORMAL)


def separation_lower(centers, ids, norms, max_norm):
    """Lower bound each requested center's distance to every other center ID.

    Norms are binary32 squared norms of the current centered coordinates.
    A duplicate at a different ID must participate. Subtract a bound for the
    worst subtraction/dot-product error across the row before taking the
    square root, then round the result down by a few ulps.
    """
    k, d = centers.shape
    if not ids.size:
        return mx.zeros((0,), mx.float32)
    if k == 1:
        return mx.full(ids.shape, float("inf"), mx.float32)
    rows = max(1, _CENTER_TILE_BYTES // (4 * k))
    all_ids = mx.arange(k, dtype=ids.dtype)
    result = []
    for start in range(0, ids.size, rows):
        current = ids[start:start + rows]
        dots = matmul(centers[current], centers, transpose_b=True)
        other = mx.where(all_ids[None, :] == current[:, None], float("inf"),
                         norms[None, :] - 2.0 * dots)
        current_norm = norms[current]
        lower_sq = current_norm + mx.min(other, axis=1) - _allowance(d, current_norm + max_norm)
        lower = mx.sqrt(mx.maximum(lower_sq, 0.0)) * (1.0 - 4 * _EPS)
        lower = mx.where(mx.isfinite(lower_sq), lower, 0.0)
        result.append(lower)
        if start + rows < ids.size:
            mx.eval(lower)
    return mx.concatenate(result)


def reusable_labels(x, centers, labels, separation, max_norm):
    """Certify incumbents against the binary32 scores of exhaustive assignment.

    For point-to-incumbent distance r and center separation s, every other
    squared distance exceeds the incumbent by at least s*(s-2r) if s>2r.
    A positive exact-distance gap alone is insufficient: the production
    argmin compares rounded ||c||²-2*x.c. Require a gap greater than the sum
    of both scores' rounding allowances, so even the production tie breaker
    cannot change the winning ID.
    """
    residual = selected_distances(x, centers, labels)
    if centers.shape[0] == 1:
        return mx.ones(labels.shape, mx.bool_), residual
    d = x.shape[1]
    row_norm = mx.sum(x * x, axis=1)
    error = _allowance(d, row_norm + max_norm)
    # Residuals use direct differences and sums of squares; their error is
    # relative to the residual itself, apart from possible underflow.
    radius = mx.sqrt(residual + _allowance(d, residual)) * (1.0 + 4 * _EPS)
    gap = separation * (separation - 2.0 * radius)
    certain = mx.isfinite(gap) & (gap > 2.0 * error)
    return certain, residual
