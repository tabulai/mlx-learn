"""Bounded candidate compaction for Euclidean radius search on Metal."""

from functools import cache

import mlx.core as mx


@cache
def _compact_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_radius_compact",
        input_names=["products", "train_norms", "query_norms", "shape", "settings"],
        output_names=["counts", "indices"],
        atomic_outputs=True,
        source=r"""
        const uint col = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        const uint lane = thread_index_in_simdgroup;
        const uint width = shape[0], capacity = shape[1];
        uint hit = 0u;
        if (col < width) {
            const float norms = train_norms[col] + query_norms[row];
            const float lower = norms * settings[1] - 2.0f * products[size_t(row) * width + col];
            hit = uint(lower <= settings[0] + settings[2]);
        }
        const uint prefix = metal::simd_prefix_exclusive_sum(hit);
        const uint total = metal::simd_sum(hit);
        uint base = 0u;
        if (lane == 0u && total != 0u)
            base = atomic_fetch_add_explicit(&counts[row], total, memory_order_relaxed);
        const uint pos = metal::simd_shuffle(base, ushort(0)) + prefix;
        if (hit && pos < capacity)
            atomic_store_explicit(&indices[size_t(row) * capacity + pos], col,
                                  memory_order_relaxed);
        """,
    )


def compact_radius(products, train_norms, query_norms, settings, capacity):
    """Count all candidates and store at most capacity indices per row."""
    rows, cols = products.shape
    return _compact_kernel()(
        inputs=[products, train_norms, query_norms,
                mx.array([cols, capacity], mx.uint32), settings],
        grid=((cols + 127) // 128 * 128, rows, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows,), (rows, capacity)],
        output_dtypes=[mx.uint32, mx.uint32],
        init_value=0,
        stream=mx.gpu,
    )
