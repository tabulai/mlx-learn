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

import operator
from functools import cache

import mlx.core as mx

__all__ = ["compact_gram", "mask_gram"]

#: Threads per threadgroup, columns per thread and rows per thread.
_THREADS = 128
_COLS = 4
_ROWS = 8
#: ``mx.fast.metal_kernel`` binds an input of fewer than eight elements in the
#: constant address space, which changes the generated source (and before MLX
#: 0.32 rebuilds the library through an unlocked map). Every input is padded.
_MIN_INPUT = 8


_GRAM_HEADER = """
inline bool is_radius_candidate(float query_sq, float train_sq, float product,
                                float cut, float factor, float floor_value) {
    const float norms = query_sq + train_sq;
    const float gram = norms - 2.0f * product;
    const bool safe = metal::isfinite(norms) && norms <= 0x1p108f && metal::isfinite(gram);
    const float lower = safe ? metal::max(gram - factor * norms - floor_value, 0.0f) : 0.0f;
    return lower <= cut;
}
"""


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
        header=_GRAM_HEADER,
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
                    hits |= uint(is_radius_candidate(query_sq, train_sq[e],
                        products[base + first_col + e * THREADS], cut, factor, floor_value)) << e;
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


@cache
def _mask_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_radius_gram_mask",
        input_names=["products", "query_norms", "train_norms", "shape", "settings"],
        output_names=["masks"],
        header=_GRAM_HEADER,
        source="""
        const uint col = thread_position_in_grid.x, row = thread_position_in_grid.y;
        const uint width = shape[0], words = shape[1];
        const uint lane = thread_index_in_simdgroup;
        bool hit = false;
        if (col < width) {
            hit = is_radius_candidate(query_norms[shape[2] + row], train_norms[col],
                products[size_t(row) * width + col], settings[0], settings[1], settings[2]);
        }
        const uint mask = metal::simd_or(uint(hit) << lane);
        if (lane == 0 && col < width)
            masks[size_t(row) * words + (col >> 5)] = mask;
        """,
    )


def mask_gram(products, query_norms, train_norms, *, cut, factor, floor, row_offset, stream):
    """Store the certified candidates in column order, one bit per training row."""
    if products.ndim != 2 or query_norms.ndim != 1 or train_norms.ndim != 1:
        raise ValueError("expected a product matrix and one-dimensional norm arrays")
    if any(array.dtype != mx.float32 for array in (products, query_norms, train_norms)):
        raise ValueError("products and norms must be float32")
    rows, width = products.shape
    row_offset = operator.index(row_offset)
    # Grid extents are signed ints in MLX. The kernel computes norm indices
    # in uint32, and linear product/mask offsets in size_t.
    grid_width = ((width + 127) // 128) * 128
    if grid_width >= 2**31 or rows >= 2**31:
        raise ValueError("tile dimensions exceed the Metal kernel grid limit")
    if not 0 <= row_offset < 2**32 or row_offset + rows > 2**32:
        raise ValueError("query norm indices must fit in uint32")
    if row_offset + rows > query_norms.size or width > train_norms.size:
        raise ValueError("query and training norms must cover the product tile")
    words = (width + 31) // 32
    if rows == 0 or width == 0:
        return mx.zeros((rows, words), dtype=mx.uint32, stream=stream)
    shape = mx.array([width, words, row_offset, 0, 0, 0, 0, 0], dtype=mx.uint32)
    settings = mx.array([cut, factor, floor, 0, 0, 0, 0, 0], dtype=mx.float32)
    return _mask_kernel()(
        inputs=[_padded(products, stream), _padded(query_norms, stream),
                _padded(train_norms, stream), shape, settings],
        grid=(grid_width, rows, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, words)],
        output_dtypes=[mx.uint32],
        stream=stream,
    )[0]
