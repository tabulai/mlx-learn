"""Bounded direct CPU search and certified GPU Gram screening."""

import re
from functools import lru_cache

import mlx.core as mx


def _cpu_order(values, indices, k):
    # argsort is stable: establish the secondary index key before distance.
    order = mx.argsort(indices, axis=1)
    values = mx.take_along_axis(values, order, axis=1)
    indices = mx.take_along_axis(indices, order, axis=1)
    order = mx.argsort(values, axis=1)[:, :k]
    return mx.take_along_axis(values, order, axis=1), mx.take_along_axis(indices, order, axis=1)


def cpu_neighbors(train, queries, k, *, rows, cols, exclude_self=False):
    """Exhaustive float64 differences on CPU; no Gram candidate truncation."""
    n, features = train.shape
    # Two float64 feature tensors coexist during direct subtraction/squaring.
    rows = min(rows, max(1, (64 << 20) // (max(features, 1) * 16)))
    cols = min(cols, max(1, (64 << 20) // (rows * features * 16)))
    x64 = train.astype(mx.float64)
    q64 = queries.astype(mx.float64)
    out_values, out_indices = [], []
    for start in range(0, queries.shape[0], rows):
        query = q64[start:start + rows]
        best_values = best_indices = None
        for offset in range(0, n, cols):
            stop = min(n, offset + cols)
            delta = query[:, None, :] - x64[None, offset:stop, :]
            values = mx.sum(delta * delta, axis=2)
            indices = mx.broadcast_to(mx.arange(offset, stop, dtype=mx.int32), values.shape)
            if exclude_self:
                own = mx.arange(start, start + query.shape[0])[:, None]
                values = mx.where(indices == own, float("inf"), values)
            values, indices = _cpu_order(values, indices, min(k, stop - offset))
            if best_values is not None:
                values = mx.concatenate([best_values, values], axis=1)
                indices = mx.concatenate([best_indices, indices], axis=1)
                values, indices = _cpu_order(values, indices, min(k, values.shape[1]))
            best_values, best_indices = values, indices
            mx.eval(best_values, best_indices)
        out_values.append(best_values.astype(mx.float32))
        out_indices.append(best_indices)
    return mx.concatenate(out_values), mx.concatenate(out_indices)


#: MLX's Metal argsort returns wrong orders for float32 rows of 2**21 or more
#: columns (MLX 0.29 through at least 0.32.4). Tiles that need a full sort are
#: never wider than this; the per-tile certificates and exact merge below
#: already handle any number of reference tiles.
_SORT_WIDTH_LIMIT = 2**20


#: Apple GPU generations whose larger partial-selection regions are calibrated.
#: 16 is the M4 family (measured on M4 Max). On an M1 Ultra (generation 13),
#: partial selection lost to argsort for 24+ candidates in descending column
#: order at every tested tile width, so other generations keep the full sort.
_CALIBRATED_GPU_FAMILIES = frozenset({16})


@lru_cache(maxsize=1)
def _gpu_family():
    """Return the Apple GPU generation, e.g. 16 for ``applegpu_g16s``, or None."""
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    except Exception:  # pragma: no cover - no Metal device information
        return None
    match = re.match(r"applegpu_g(\d+)", str(info.get("architecture", "")))
    return int(match.group(1)) if match else None


def _use_partial_select(rows, width, take):
    """Choose ``select_topk`` over a full row sort for one Gram tile.

    Up to 16 candidates always use partial selection. Larger candidate sets
    use it only on calibrated GPU generations, inside the regions of
    ``_calibrated_partial_select``.
    """
    if take <= 16:
        return True
    return _gpu_family() in _CALIBRATED_GPU_FAMILIES and _calibrated_partial_select(rows, width, take)


def _calibrated_partial_select(rows, width, take):
    """Tile regions where ``select_topk`` beat ``argsort`` on M4 Max, for 17..41 candidates.

    They were faster for both random and descending column order (each lane's
    worst case): at least 1.07x, and 3-11x for 64-row tiles of a million
    columns. A full sort stays faster for fewer than 8 query rows, for narrow
    tiles, and beyond 41 candidates in adversarial order.
    """
    if take > 41 or rows < 8:
        return False
    if width >= 2**18:
        return True
    if take <= 32 and width >= 2**14:
        return True
    return rows >= 256 and rows * width >= 2**22


#: Apple GPU generations where 64-row Gram blocks measured faster than the
#: planner's 65-90 row heights: M1 (13) and M4 (16). On an M5 Ultra (17) they
#: were slower (0.81-0.95x at k=5 over 0.75-1M rows), so it keeps its heights.
_ROW_ALIGNED_GPU_FAMILIES = frozenset({13, 16})


def _gram_block_rows(rows, n_query):
    """Avoid block heights just above 64 rows when a query spans several blocks.

    On the generations above, MLX's Gram matrix product pads such blocks to a
    second 64-row tile: 67-row blocks measured 1.3-1.8x slower than 64-row
    blocks of the same query on M4 Max, at 96-1024 features; on M1 Ultra,
    64-row blocks made k=5 queries over a million rows 1.1-1.6x faster.
    Heights from 91 rows upward keep their size. Shorter blocks only reduce
    the memory each needs.
    """
    if rows < n_query and 64 < rows <= 90 and _gpu_family() in _ROW_ALIGNED_GPU_FAMILIES:
        return 64
    return rows


@mx.compile
def _gram_lower(query, train, center, train_sq, error_factor, floor):
    centered = query - center
    norms = mx.sum(centered * centered, axis=1)[:, None] + train_sq[None, :]
    gram = norms - 2.0 * (centered @ train.T)
    safe = (mx.isfinite(norms) & (norms <= 2.0**108) & mx.isfinite(gram)
            & (error_factor < 0.125))
    return mx.where(safe, mx.maximum(gram - error_factor * norms - floor, 0.0), 0.0)


def gpu_neighbors(index, queries, k, *, rows, cols, exclude_self=False):
    """Use Gram values only when an omitted-candidate bound proves safety."""
    from ._neighbors_fused import certified_neighbors, precise_select_candidates, select_topk

    index._prepare_gram()
    train = index.X
    n, features = train.shape
    small_shortlist = 33 <= features <= 256 and k <= 8
    # This margin changes screening cost, not correctness: the same omitted
    # lower bound must still certify membership, or the full scan runs.
    carry = min(n, k + (2 if small_shortlist else 8))
    # Standard dot/norm rounding, centering and final subtraction allowances.
    # Unsafe ranges get a zero lower bound and therefore cannot certify.
    error_factor = (2 * features + 16) * 2.0**-23
    floor = features * 1e-30
    rows = _gram_block_rows(rows, queries.shape[0])
    out_values, out_indices = [], []
    for start in range(0, queries.shape[0], rows):
        query = queries[start:start + rows]
        best_values = best_indices = None
        certified = mx.ones((query.shape[0],), mx.bool_)
        own = mx.arange(start, start + query.shape[0])[:, None] if exclude_self else None
        block_cols = min(cols, n)
        if not _use_partial_select(query.shape[0], block_cols, min(block_cols, carry + 1)):
            block_cols = min(block_cols, _SORT_WIDTH_LIMIT)
        for offset in range(0, n, block_cols):
            stop = min(offset + block_cols, n)
            lower = _gram_lower(query, index._train[offset:stop], index._center,
                                index._train_sq[offset:stop], error_factor, floor)
            ids = mx.broadcast_to(mx.arange(offset, stop, dtype=mx.int32), lower.shape)
            if exclude_self:
                lower = mx.where(ids == own, float("inf"), lower)
            width = stop - offset
            take = min(width, carry + 1)
            if _use_partial_select(query.shape[0], width, take):
                bounds, candidates = select_topk(lower, take, offset=offset)
            else:
                # A block that chose partial selection sorts only tiles narrower
                # than 2**18 columns; any other block is limited above.
                assert width <= _SORT_WIDTH_LIMIT
                order = mx.argsort(lower, axis=1)[:, :take]
                bounds = mx.take_along_axis(lower, order, axis=1)
                candidates = mx.take_along_axis(ids, order, axis=1)
            barrier = bounds[:, -1] if width > carry else mx.full((query.shape[0],), float("inf"))
            candidates = candidates[:, :min(carry, width)]
            if exclude_self:
                candidates = mx.where(candidates == own, -1, candidates)
            # A final tile can contain only the excluded identity. An invalid
            # sentinel is retained through the merge, after every real point.
            retained = min(k, width)
            if small_shortlist:
                from ._neighbors_shortlist import refine_certified_candidates

                values, candidates, certified = refine_certified_candidates(
                    train, query, candidates, retained, barrier, certified,
                    check_barrier=width > carry,
                )
            else:
                values, candidates = precise_select_candidates(train, query, candidates, retained)
                if width > carry:
                    upper = values[:, -1] * (1.0 + 6 * 2.0**-23) + floor
                    certified = certified & mx.isfinite(upper) & (barrier > upper)
            if best_values is not None:
                values = mx.concatenate([best_values, values], axis=1)
                candidates = mx.concatenate([best_indices, candidates], axis=1)
                values, candidates = precise_select_candidates(
                    train, query, candidates, min(k, values.shape[1]), values,
                )
            best_values, best_indices = values, candidates
            if stop < n:
                mx.eval(best_values, best_indices, certified)
        values, indices = certified_neighbors(
            train, query, k, best_values, best_indices, certified,
            exclude_self=exclude_self, query_offset=start,
        )
        out_values.append(values)
        out_indices.append(indices)
    return mx.concatenate(out_values), mx.concatenate(out_indices)
