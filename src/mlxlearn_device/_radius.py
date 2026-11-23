"""Certified Gram screening and bounded candidate compaction for radius search.

The screen reads one binary32 product tile ``p = (q - c) . (x - c)`` of centred
queries and centred training rows, the queries' centred squared norms ``qn``
and the index's centred training norms ``xn``. It evaluates, in float32 and in
the same order, the certified lower bound of the k-nearest-neighbour Gram
screen (``_neighbors_general._gram_lower_from_products``)::

    norms = qn + xn
    gram  = norms - 2 p
    safe  = isfinite(norms) && norms <= 2**108 && isfinite(gram)
    lower = safe ? max(gram - factor * norms - floor, 0) : 0

and keeps a pair as a candidate iff ``lower <= cut``. The host chooses ``cut``
at or above the squared radius (float32, rounded up), so every pair within the
radius is a candidate; an unsafe entry has ``lower = 0`` and is always one.

Each row's candidate count is exact even past ``capacity``; only the first
``capacity`` arrivals are stored. Arrival order follows the per-row atomic
counter and is unspecified, so callers sort each row on the host.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

__all__ = ["compact_gram"]

#: Threads per threadgroup, columns per thread and rows per thread.
_THREADS = 128
_COLS = 4
_ROWS = 8
#: ``mx.fast.metal_kernel`` binds an input of fewer than eight elements in the
#: constant address space, which changes the generated source (and before MLX
#: 0.32 rebuilds the library through an unlocked map). Every input is padded.
_MIN_INPUT = 8


@cache
def _compact_kernel():
    # Each thread holds the training norms of COLS columns spaced THREADS
    # apart, so every product load of a SIMD group reads 32 consecutive floats,
    # and reuses them for ROWS consecutive rows. A SIMD group that found any
    # candidate in a row reserves slots with one atomic add on that row's count.
    return mx.fast.metal_kernel(
        name="mlxlearn_radius_gram_compact",
        input_names=["products", "query_norms", "train_norms", "shape", "settings"],
        output_names=["counts", "packed"],
        atomic_outputs=True,
        source=f"""
        constexpr uint THREADS = {_THREADS}u, COLS = {_COLS}u, ROWS = {_ROWS}u;
        const uint width = shape[0], capacity = shape[1], rows = shape[2];
        const uint row_offset = shape[3], col_offset = shape[4];
        const float cut = settings[0], factor = settings[1], floor_value = settings[2];
        const uint lane = thread_index_in_simdgroup;
        const uint first_col = threadgroup_position_in_grid.x * (THREADS * COLS)
                               + thread_position_in_threadgroup.x;
        const uint row_begin = threadgroup_position_in_grid.y * ROWS;
        const uint row_end = metal::min(rows, row_begin + ROWS);
        float train_sq[COLS];
        bool inside[COLS];
        #pragma clang loop unroll(full)
        for (uint e = 0u; e < COLS; ++e) {{
            const uint col = first_col + e * THREADS;
            inside[e] = col < width;
            train_sq[e] = inside[e] ? train_norms[col_offset + col] : 0.0f;
        }}
        for (uint row = row_begin; row < row_end; ++row) {{
            const float query_sq = query_norms[row_offset + row];
            const size_t base = size_t(row) * size_t(width);
            uint hits = 0u;
            #pragma clang loop unroll(full)
            for (uint e = 0u; e < COLS; ++e) {{
                if (inside[e]) {{
                    const float norms = query_sq + train_sq[e];
                    const float gram = norms - 2.0f * products[base + first_col + e * THREADS];
                    const bool safe = metal::isfinite(norms) && norms <= 0x1p108f
                                      && metal::isfinite(gram);
                    const float lower = safe
                        ? metal::max(gram - factor * norms - floor_value, 0.0f) : 0.0f;
                    hits |= uint(lower <= cut) << e;
                }}
            }}
            const uint count = metal::popcount(hits);
            if (metal::simd_any(count != 0u)) {{
                const uint prefix = metal::simd_prefix_exclusive_sum(count);
                const uint total = metal::simd_sum(count);
                uint start = 0u;
                if (lane == 0u)
                    start = atomic_fetch_add_explicit(&counts[row], total, memory_order_relaxed);
                uint slot = metal::simd_broadcast(start, ushort(0)) + prefix;
                #pragma clang loop unroll(full)
                for (uint e = 0u; e < COLS; ++e) {{
                    if ((hits >> e) & 1u) {{
                        if (slot < capacity)
                            atomic_store_explicit(
                                &packed[size_t(row) * size_t(capacity) + slot],
                                col_offset + first_col + e * THREADS, memory_order_relaxed);
                        ++slot;
                    }}
                }}
            }}
        }}
        """,
    )


def _padded(array, stream):
    """A flat view of ``array`` with at least eight elements (zero padded)."""
    if array.size >= _MIN_INPUT:
        return array
    flat = mx.reshape(array, (-1,), stream=stream)
    pad = mx.zeros((_MIN_INPUT - array.size,), dtype=array.dtype, stream=stream)
    return mx.concatenate([flat, pad], axis=0, stream=stream)


def compact_gram(products, query_norms, train_norms, *, capacity, cut, factor, floor,
                 row_offset=0, col_offset=0, stream=None):
    """Screen one product tile and compact its candidates.

    Parameters
    ----------
    products : mx.array, float32 (rows, width)
        Binary32 products of centred query rows and centred training rows
        ``[col_offset, col_offset + width)``.
    query_norms : mx.array, float32
        Centred squared query norms; row ``r`` of the tile reads
        ``query_norms[row_offset + r]``.
    train_norms : mx.array, float32
        Centred squared training norms; column ``j`` reads
        ``train_norms[col_offset + j]``.
    capacity : int
        Stored candidates per row (at least 1).
    cut, factor, floor : float
        Float32 values of the squared-radius cut (rounded up), the relative
        Gram allowance and the absolute floor.

    Returns
    -------
    counts : mx.array, uint32 (rows,)
        Exact candidate count of every row, including those beyond ``capacity``.
    packed : mx.array, uint32 (rows, capacity)
        Global training indices (``col_offset + j``) of the first
        ``min(count, capacity)`` candidates of each row, in unspecified order.
        Slots beyond a row's count are zero.
    """
    stream = mx.gpu if stream is None else stream
    rows, width = products.shape
    capacity = int(capacity)
    if capacity < 1:
        raise ValueError("capacity must be at least 1")
    if rows == 0 or width == 0:
        return (mx.zeros((rows,), mx.uint32, stream=stream),
                mx.zeros((rows, capacity), mx.uint32, stream=stream))
    if col_offset + width >= 2**32 - 1:
        # 0xFFFFFFFF stays free as the host's sort sentinel.
        raise ValueError("training indices must fit below 2**32 - 1")
    shape = mx.array([width, capacity, rows, int(row_offset), int(col_offset), 0, 0, 0],
                     dtype=mx.uint32)
    settings = mx.array([float(cut), float(factor), float(floor), 0.0, 0.0, 0.0, 0.0, 0.0],
                        dtype=mx.float32)
    chunks = -(-width // (_THREADS * _COLS))
    groups = -(-rows // _ROWS)
    counts, packed = _compact_kernel()(
        inputs=[_padded(products, stream), _padded(query_norms, stream),
                _padded(train_norms, stream), shape, settings],
        grid=(chunks * _THREADS, groups, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows,), (rows, capacity)],
        output_dtypes=[mx.uint32, mx.uint32],
        init_value=0,
        stream=stream,
    )
    return counts, packed
