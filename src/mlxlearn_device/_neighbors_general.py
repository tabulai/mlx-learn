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


#: Apple GPU generations where filtered selection was calibrated against the
#: routes below: M1 Ultra (13), M4 Max (16) and M5 Ultra (17).
_FILTERED_GPU_FAMILIES = frozenset({13, 16, 17})


def _selection(rows, width, take):
    """Choose how one Gram tile selects its candidates.

    Returns "filtered" or "partial" (``select_topk`` with or without its
    chunk-minimum filter) or "sort" (a full row sort). Calibrated GPU
    generations use filtered selection inside ``_filtered_select_wins``.
    Otherwise up to 16 candidates use partial selection, as do larger sets
    inside ``_calibrated_partial_select`` on its calibrated generations.
    """
    family = _gpu_family()
    if family in _FILTERED_GPU_FAMILIES and _filtered_select_wins(rows, width, take):
        return "filtered"
    if take <= 16 or (family in _CALIBRATED_GPU_FAMILIES
                      and _calibrated_partial_select(rows, width, take)):
        return "partial"
    return "sort"


def _filtered_select_wins(rows, width, take):
    """Tile regions where filtered selection beat the route it replaces on M1, M4 and M5.

    Measured on 1..1024-row tiles of 4096..2**20 columns, for 2..128 candidates
    in random, descending and single-lane column orders; every region beat
    partial selection (up to 16 candidates) or the full sort by at least 3% on
    all three GPUs in all three orders. Filtered selection does about the same
    work in any order, while the unfiltered scan inserts every value of a
    descending row. Its two extra launches lose on small tiles, and long
    candidate lists lose to a full sort unless the tile is wide. Single rows
    were not measured beyond 2**20 columns and keep their routes.
    """
    area = rows * width
    if rows < 2:
        return False
    if take <= 16:
        # Partial selection is cheap for short lists; tall tiles keep it too.
        if take <= 7:
            return 16 <= rows <= 256 and area >= 2**22
        if take <= 11:
            return 16 <= rows <= 256 and area >= 2**21
        if take <= 15:
            return rows <= 256 and area >= 2**21
        return rows <= 1024 and area >= 2**20
    if take <= 32:
        return rows <= 1024 and width >= 2**14 and area >= 2**20
    if take <= 48:
        return 4 <= rows <= 1024 and width >= 2**16 and area >= 2**20
    if take <= 64:
        return rows <= 1024 and width >= 2**16 and area >= 2**21
    if take <= 96:
        return rows <= 256 and width >= 2**17 and area >= 2**22
    if take <= 128:
        return rows <= 256 and width >= 2**18 and area >= 2**23
    return False


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


@mx.compile
def _gram_lower_from_products(centered, products, train_sq, error_factor, floor):
    norms = mx.sum(centered * centered, axis=1)[:, None] + train_sq[None, :]
    gram = norms - 2.0 * products
    safe = (mx.isfinite(norms) & (norms <= 2.0**108) & mx.isfinite(gram)
            & (error_factor < 0.125))
    return mx.where(safe, mx.maximum(gram - error_factor * norms - floor, 0.0), 0.0)


def gram_lower(query, train, center, train_sq, error_factor, floor, *, exact_products=None):
    """Certified lower bounds on squared distances from a binary32 Gram tile.

    ``error_factor`` bounds binary32 GEMM rounding. MLX may compute float32
    ``matmul`` with TF32 operands on GPUs with neural accelerators, which
    violates that bound, so those GPUs take a dedicated binary32 product.
    """
    from ._binary32 import float32_matmul, matmul_may_round_float32

    if exact_products is None:
        exact_products = matmul_may_round_float32()
    if not exact_products:
        return _gram_lower(query, train, center, train_sq, error_factor, floor)
    centered = query - center
    return _gram_lower_from_products(
        centered, float32_matmul(centered, train, transpose_b=True), train_sq, error_factor, floor,
    )


def gpu_neighbors(index, queries, k, *, rows, cols, exclude_self=False, evaluate_blocks=False):
    """Use Gram values only when an omitted-candidate bound proves safety."""
    from ._binary32 import matmul_may_round_float32
    from ._neighbors_fused import certified_neighbors, precise_select_candidates, select_topk

    index._prepare_gram()
    exact_products = matmul_may_round_float32()
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
        if _selection(query.shape[0], block_cols, min(block_cols, carry + 1)) == "sort":
            block_cols = min(block_cols, _SORT_WIDTH_LIMIT)
        for offset in range(0, n, block_cols):
            stop = min(offset + block_cols, n)
            lower = gram_lower(query, index._train[offset:stop], index._center,
                               index._train_sq[offset:stop], error_factor, floor,
                               exact_products=exact_products)
            ids = mx.broadcast_to(mx.arange(offset, stop, dtype=mx.int32), lower.shape)
            if exclude_self:
                lower = mx.where(ids == own, float("inf"), lower)
            width = stop - offset
            take = min(width, carry + 1)
            method = _selection(query.shape[0], width, take)
            if method == "sort" and width > _SORT_WIDTH_LIMIT:
                # Only a final tile narrower than its block re-chooses; a full
                # sort of it would be wrong, so select it instead.
                method = "partial"
            if method != "sort":
                bounds, candidates = select_topk(lower, take, offset=offset,
                                                 filtered=method == "filtered")
            else:
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
        if evaluate_blocks and start + rows < queries.shape[0]:
            # Finish this block before queuing the next Gram matrix. Only the
            # k results per query need to remain live for concatenation.
            mx.eval(values, indices)
        out_values.append(values)
        out_indices.append(indices)
    return mx.concatenate(out_values), mx.concatenate(out_indices)
