"""Device-only direct Euclidean selection for small neighbor counts.

The streaming local-top-k / SIMD-list-merge design is inspired by the MIT
licensed ``tabulai/mlx-geospatial`` haversine selectors (revision e7af5bac).
This implementation computes Euclidean distances on the original float32
coordinates, partitions long reference scans, and uses SIMD minima to merge
the sorted lists. It does not materialize a query-by-training distance matrix.

Each pair uses a compensated sum of squared differences. Selection and merging
resolve overlapping rounding intervals from the original coordinate bits; only
mathematically equal distances use the training index as a tie breaker. No host array conversion or synchronization
is performed. Input finiteness is the caller's validation responsibility.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import mlx.core as mx

from ._neighbors_precision import PRECISION_HEADER

__all__ = ["fused_neighbors", "select_topk", "refine_neighbors"]

_HEADER = """
inline bool neighbor_precedes(float distance, int index,
                              float other_distance, int other_index) {
    return distance < other_distance ||
           (distance == other_distance && index < other_index);
}
"""

_SCAN = """
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    const uint lane = thread_index_in_simdgroup;
    const uint task = thread_position_in_grid.x / 32;
    const int n_train = shape[0];
    const int n_query = shape[1];
    const int parts = shape[3];
    const int query = int(task / uint(parts));
    if (query >= n_query) return; // uniform across this entire SIMDgroup
    const int part = int(task % uint(parts));
    const int begin = int((size_t(part) * size_t(n_train)) / size_t(parts));
    const int end = int((size_t(part + 1) * size_t(n_train)) / size_t(parts));
    const int sentinel = 2147483647;
    float distances[K];
    int indices[K];
    #pragma clang loop unroll(disable)
    for (int slot = 0; slot < K; ++slot) {
        distances[slot] = INFINITY;
        indices[slot] = sentinel;
    }
    for (int reference = begin + int(lane); reference < end; reference += 32) {
        if (EXCLUDE_SELF && reference == query + shape[4]) continue;
        float distance = 0.0f;
        float correction = 0.0f;
        for (int feature = 0; feature < D; ++feature) {
            const float delta = queries[size_t(query) * D + feature]
                              - train[size_t(reference) * D + feature];
            const float adjusted = delta * delta - correction;
            const float total = distance + adjusted;
            // Mixed scales must not discard every small squared difference.
            // Keep overflow as +inf instead of contaminating the compensation
            // with inf-inf and losing valid reference indices to NaN.
            correction = metal::isfinite(total) ? (total - distance) - adjusted : 0.0f;
            distance = total;
        }
        if (precise_neighbor_precedes<D>(distance, reference, distances[K - 1], indices[K - 1],
                                           train, queries, query, n_train)) {
            int slot = K - 1;
            while (slot > 0 && precise_neighbor_precedes<D>(distance, reference,
                                                 distances[slot - 1], indices[slot - 1],
                                                 train, queries, query, n_train)) {
                distances[slot] = distances[slot - 1];
                indices[slot] = indices[slot - 1];
                --slot;
            }
            distances[slot] = distance;
            indices[slot] = reference;
        }
    }
    // A lane may own all K winners. Advance only the list that supplies the
    // selected head, then reduce again; one winner per lane would be incorrect.
    int cursor = 0;
    #pragma clang loop unroll(disable)
    for (int slot = 0; slot < K; ++slot) {
        const float head = cursor < K ? distances[cursor] : INFINITY;
        const int index = cursor < K ? indices[cursor] : sentinel;
        const auto winner = precise_simd_min<D>(head, index, train, queries, query, n_train);
        const int winner_index = winner.index;
        const float winner_distance = winner.distance;
        if (lane == uint(slot) % 32) {
            out_distances[size_t(task) * K + slot] = winner_distance;
            out_indices[size_t(task) * K + slot] = winner_index;
        }
        if (index == winner_index && cursor < K) ++cursor;
    }
"""

_MERGE = """
    const uint lane = thread_index_in_simdgroup;
    const uint query = thread_position_in_grid.x / 32;
    const int parts = shape[0];
    if (query >= uint(shape[1])) return;
    const int sentinel = 2147483647;
    int cursor = 0;
    // Every partition supplies a sorted list; one lane traverses each list.
    #pragma clang loop unroll(disable)
    for (int slot = 0; slot < K; ++slot) {
        const bool present = lane < uint(parts) && cursor < K;
        const size_t address = (size_t(query) * parts + lane) * K + cursor;
        const float head = present ? partial_distances[address] : INFINITY;
        const int index = present ? partial_indices[address] : sentinel;
        const float winner_distance = metal::simd_min(head);
        const int winner_index = metal::simd_min(head == winner_distance ? index : sentinel);
        if (lane == uint(slot) % 32) {
            out_distances[size_t(query) * K + slot] = winner_distance;
            out_indices[size_t(query) * K + slot] = winner_index;
        }
        if (present && index == winner_index) ++cursor;
    }
"""

_PRECISE_MERGE = _MERGE.replace(
    "const float winner_distance = metal::simd_min(head);\n"
    "        const int winner_index = metal::simd_min(head == winner_distance ? index : sentinel);",
    "const auto winner = precise_simd_min<D>(head, index, train, queries, int(query), shape[2]);\n"
    "        const int winner_index = winner.index;\n"
    "        const float winner_distance = winner.distance;",
)


_VALUE_SCAN = """
    const uint lane = thread_index_in_simdgroup;
    const uint task = thread_position_in_grid.x / 32;
    const int width = shape[0];
    const int rows = shape[1];
    const int parts = shape[2];
    const int row = int(task / uint(parts));
    if (row >= rows) return;
    const int part = int(task % uint(parts));
    const int begin = int((size_t(part) * size_t(width)) / size_t(parts));
    const int end = int((size_t(part + 1) * size_t(width)) / size_t(parts));
    const int sentinel = 2147483647;
    float distances[K];
    int indices[K];
    for (int slot = 0; slot < K; ++slot) {
        distances[slot] = INFINITY;
        indices[slot] = sentinel;
    }
    for (int col = begin + int(lane); col < end; col += 32) {
        float distance = values[size_t(row) * width + col];
        // Overflow in a Gram tile may produce NaN. Keep a real candidate
        // index so the caller can recompute that candidate's direct distance.
        if (metal::isnan(distance)) distance = INFINITY;
        const int index = col + shape[3];
        if (neighbor_precedes(distance, index, distances[K - 1], indices[K - 1])) {
            int slot = K - 1;
            while (slot > 0 && neighbor_precedes(distance, index,
                                                 distances[slot - 1], indices[slot - 1])) {
                distances[slot] = distances[slot - 1];
                indices[slot] = indices[slot - 1];
                --slot;
            }
            distances[slot] = distance;
            indices[slot] = index;
        }
    }
    int cursor = 0;
    for (int slot = 0; slot < K; ++slot) {
        const float head = cursor < K ? distances[cursor] : INFINITY;
        const int index = cursor < K ? indices[cursor] : sentinel;
        const float winner_distance = metal::simd_min(head);
        const int winner_index = metal::simd_min(head == winner_distance ? index : sentinel);
        if (lane == uint(slot) % 32) {
            out_distances[size_t(task) * K + slot] = winner_distance;
            out_indices[size_t(task) * K + slot] = winner_index;
        }
        if (index == winner_index && cursor < K) ++cursor;
    }
"""

_REFINE = """
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    const size_t item = thread_position_in_grid.x;
    const int n_train = shape[0];
    const int n_query = shape[1];
    const int width = shape[2];
    if (item >= size_t(n_query) * width) return;
    const int query = int(item / width);
    const int reference = indices[item];
    if (reference < 0 || reference >= n_train) {
        out_distances[item] = INFINITY;
        return;
    }
    float distance = 0.0f;
    float correction = 0.0f;
    for (int feature = 0; feature < D; ++feature) {
        const float delta = queries[size_t(query) * D + feature]
                          - train[size_t(reference) * D + feature];
        const float adjusted = delta * delta - correction;
        const float total = distance + adjusted;
        correction = metal::isfinite(total) ? (total - distance) - adjusted : 0.0f;
        distance = total;
    }
    out_distances[item] = distance;
"""


@lru_cache(maxsize=1)
def _scan_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_direct_neighbors_scan",
        input_names=["train", "queries", "shape"],
        output_names=["out_distances", "out_indices"],
        source=_SCAN,
        header=_HEADER + PRECISION_HEADER,
    )


@lru_cache(maxsize=1)
def _merge_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_direct_neighbors_merge",
        input_names=["partial_distances", "partial_indices", "shape"],
        output_names=["out_distances", "out_indices"],
        source=_MERGE,
    )


@lru_cache(maxsize=1)
def _precise_merge_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_precise_neighbors_merge",
        input_names=["partial_distances", "partial_indices", "train", "queries", "shape"],
        output_names=["out_distances", "out_indices"],
        source=_PRECISE_MERGE,
        header=_HEADER + PRECISION_HEADER,
    )


@lru_cache(maxsize=1)
def _value_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_neighbor_value_select",
        input_names=["values", "shape"],
        output_names=["out_distances", "out_indices"],
        source=_VALUE_SCAN,
        header=_HEADER,
    )


@lru_cache(maxsize=1)
def _refine_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_neighbor_direct_refine",
        input_names=["train", "queries", "indices", "shape"],
        output_names=["out_distances"],
        source=_REFINE,
    )


def _partitions(n_query: int, n_train: int) -> int:
    """Partition value-only selection without large merge buffers."""
    return min(32, max(1, n_train // 512), max(1, (256 + n_query - 1) // n_query))


def _scan_partitions(n_query: int, n_train: int) -> int:
    """Expose enough independent distance scans to occupy the GPU.

    Direct distances do substantially more work than value-only selection.
    Targeting just 256 SIMDgroups left ordinary batches with long serial scans.
    Up to 4096 groups shortens those scans while retaining at least 512 training
    rows per partition and bounding the GPU merge to 32 sorted lists.
    """
    return min(32, max(1, n_train // 512), max(1, (4096 + n_query - 1) // n_query))


def fused_neighbors(
    train: mx.array,
    queries: mx.array,
    k: int,
    *,
    exclude_self: bool = False,
    query_offset: int = 0,
) -> tuple[mx.array, mx.array]:
    """Return sorted squared float32 distances and int32 training indices.

    Both arguments must be float32 MLX matrices with the same positive feature
    count, and the active device must be Metal. ``1 <= k <= 32``. With
    ``exclude_self``, query row ``i`` excludes training row ``query_offset+i``;
    the caller guarantees the queries are the corresponding training slice.
    The outputs remain lazy MLX arrays, ready for further device computation.
    """
    for name, array in (("train", train), ("queries", queries)):
        if not isinstance(array, mx.array):
            raise TypeError(f"{name} must be an MLX array")
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        if array.dtype != mx.float32:
            raise TypeError(f"{name} must have float32 dtype")
    if train.shape[1] == 0 or train.shape[1] != queries.shape[1]:
        raise ValueError("train and queries must have the same positive feature count")
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError("k must be an integer")
    if not 1 <= k <= min(32, train.shape[0] - int(exclude_self)):
        raise ValueError("k must be between 1 and 32 and no larger than the available references")
    if isinstance(query_offset, bool) or not isinstance(query_offset, Integral):
        raise TypeError("query_offset must be an integer")
    if query_offset < 0 or (exclude_self and query_offset + queries.shape[0] > train.shape[0]):
        raise ValueError("query_offset must locate the query slice within the training matrix")
    if max(train.shape + queries.shape + (query_offset,)) >= 2**31:
        raise ValueError("fused neighbor dimensions must fit int32 indexing")
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("fused_neighbors requires the active MLX GPU device")
    n_query, features = queries.shape
    if n_query == 0:
        return mx.zeros((0, k), mx.float32), mx.zeros((0, k), mx.int32)
    parts = _scan_partitions(n_query, train.shape[0])
    from ._neighbors_screen import screen_neighbors

    paired = train.shape[0] >= 8192 and n_query >= 32 and features <= 32 and k <= 16
    return screen_neighbors(
        train, queries, int(k), partitions=parts, paired=paired,
        exclude_self=exclude_self, query_offset=query_offset,
    )


def select_topk(values: mx.array, k: int, *, offset: int = 0) -> tuple[mx.array, mx.array]:
    """Select sorted row minima without a full-shaped sort-index allocation.

    ``values`` is a float32 matrix; ``1 <= k <= 128``. Returned global column
    indices include ``offset`` and break equal-value ties. NaN entries are
    treated as positive infinity. Both result arrays remain on the GPU.
    """
    if not isinstance(values, mx.array) or values.dtype != mx.float32:
        raise TypeError("values must be a float32 MLX array")
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError("k must be an integer")
    if not 1 <= k <= min(128, values.shape[1]):
        raise ValueError("k must be between 1 and min(128, values.shape[1])")
    if isinstance(offset, bool) or not isinstance(offset, Integral):
        raise TypeError("offset must be an integer")
    if offset < 0 or values.shape[1] + offset >= 2**31 or values.shape[0] >= 2**31:
        raise ValueError("global column indices and row count must fit int32")
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("select_topk requires the active MLX GPU device")
    rows, width = values.shape
    if rows == 0:
        return mx.zeros((0, k), mx.float32), mx.zeros((0, k), mx.int32)
    parts = _partitions(rows, width)
    task_count = rows * parts
    partial_values, partial_indices = _value_kernel()(
        inputs=[values, mx.array([width, rows, parts, offset], mx.int32)],
        template=[("K", int(k))],
        grid=(((task_count * 32 + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, parts, k), (rows, parts, k)],
        output_dtypes=[mx.float32, mx.int32],
        stream=mx.gpu,
    )
    if parts == 1:
        return partial_values.reshape(rows, k), partial_indices.reshape(rows, k)
    return tuple(_merge_kernel()(
        inputs=[partial_values, partial_indices, mx.array([parts, rows], mx.int32)],
        template=[("K", int(k))],
        grid=(((rows * 32 + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, k), (rows, k)],
        output_dtypes=[mx.float32, mx.int32],
        stream=mx.gpu,
    ))


def refine_neighbors(train: mx.array, queries: mx.array, indices: mx.array) -> mx.array:
    """Recompute candidate distances with the fused scan's compensated order.

    One thread accumulates each candidate pair directly from original inputs,
    so temporary storage is proportional to the output rather than B*k*D.
    Invalid candidate indices produce positive infinity without an invalid
    read. All valid pair distances match ``fused_neighbors`` bit for bit.
    """
    for name, array in (("train", train), ("queries", queries)):
        if not isinstance(array, mx.array) or array.dtype != mx.float32:
            raise TypeError(f"{name} must be a float32 MLX array")
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
    if train.shape[1] == 0 or train.shape[1] != queries.shape[1]:
        raise ValueError("train and queries must have the same positive feature count")
    if not isinstance(indices, mx.array) or indices.dtype != mx.int32:
        raise TypeError("indices must be an int32 MLX array")
    if indices.ndim != 2 or indices.shape[0] != queries.shape[0]:
        raise ValueError("indices must be a matrix with one row per query")
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("refine_neighbors requires the active MLX GPU device")
    if indices.size == 0:
        return mx.zeros(indices.shape, mx.float32)
    (distances,) = _refine_kernel()(
        inputs=[train, queries, indices,
                mx.array([train.shape[0], queries.shape[0], indices.shape[1]], mx.int32)],
        template=[("D", train.shape[1])],
        grid=(((indices.size + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[indices.shape],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
    )
    return distances


_WIDE_REFINE = r"""
const uint lane = thread_index_in_simdgroup;
const size_t item = thread_position_in_grid.x / 32;
if (item >= size_t(shape[1]) * shape[2]) return;
const int query = int(item / shape[2]), reference = indices[item];
if (reference < 0 || reference >= shape[0]) {
    if (lane == 0) { out_distances[item] = INFINITY; out_tails[item] = NAN; }
    return;
}
float2 sum = float2(0.0f);
bool safe = D <= 4096;
#pragma clang loop unroll(disable)
for (int feature = int(lane); feature < D; feature += 32) {
    const float q = queries[size_t(query) * D + feature];
    const float x = train[size_t(reference) * D + feature];
    const uint qb = as_type<uint>(q) & 0x7fffffffu, xb = as_type<uint>(x) & 0x7fffffffu;
    safe = safe && (!qb || (qb >= 0x30800000u && qb <= 0x4e800000u))
                && (!xb || (xb >= 0x30800000u && xb <= 0x4e800000u));
    const float2 delta = precise_two_sum(q, -x);
    const float product = delta.x * delta.x;
    const float tail = metal::fma(delta.x, delta.x, -product)
                     + (2.0f * delta.x * delta.y + delta.y * delta.y);
    const float2 added = precise_two_sum(sum.x, product);
    sum = precise_two_sum(added.x, (sum.y + tail) + added.y);
}
for (uint step = 16; step; step >>= 1) {
    const float hi = metal::simd_shuffle_down(sum.x, step);
    const float lo = metal::simd_shuffle_down(sum.y, step);
    const float2 added = precise_two_sum(sum.x, hi);
    sum = precise_two_sum(added.x, (sum.y + lo) + added.y);
}
safe = metal::simd_all(safe);
if (lane == 0) {
    if (safe) { out_distances[item] = sum.x; out_tails[item] = sum.y; }
    else {
        float value = 0.0f, correction = 0.0f;
        #pragma clang loop unroll(disable)
        for (int feature = 0; feature < D; ++feature) {
            const float delta = queries[size_t(query) * D + feature]
                              - train[size_t(reference) * D + feature];
            const float adjusted = delta * delta - correction;
            const float total = value + adjusted;
            correction = metal::isfinite(total) ? (total - value) - adjusted : 0.0f;
            value = total;
        }
        out_distances[item] = value; out_tails[item] = NAN;
    }
}
"""

_CACHED_PRECISION = r"""
template <int D, typename TrainPointer, typename QueryPointer>
inline bool precise_cached_precedes(float distance, int index,
    float other_distance, int other_index, TrainPointer train, QueryPointer queries,
    int query, int n, float tail, float other_tail) {
    if (D <= 4096 && index != other_index && index >= 0 && index < n &&
        other_index >= 0 && other_index < n && metal::isfinite(tail) && metal::isfinite(other_tail)) {
        const float difference = (distance - other_distance) + (tail - other_tail);
        const float bound = (128.0f * float(D) * 0x1p-46f) * (distance + other_distance)
                          + float(D) * 0x1p-90f;
        if (difference < -bound) return true;
        if (difference > bound) return false;
    }
    return precise_neighbor_precedes<D>(distance, index, other_distance, other_index,
                                       train, queries, query, n);
}
"""


@lru_cache(maxsize=1)
def _wide_refine_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_wide_precise_refine",
        input_names=["train", "queries", "indices", "shape"],
        output_names=["out_distances", "out_tails"],
        source=_WIDE_REFINE, header=PRECISION_HEADER,
    )


_CANDIDATE_ORDER = r"""
const size_t item = thread_position_in_grid.x;
const int width = shape[2], k = shape[3];
if (item >= size_t(shape[1]) * width) return;
const int query = int(item / width), column = int(item % width);
const int reference = indices[item];
if (column < k) {
    int available = 0;
    for (int other = 0; other < width; ++other) {
        const int id = indices[size_t(query) * width + other];
        available += int(id >= 0 && id < shape[0]);
    }
    if (column >= available) {
        out_distances[size_t(query) * k + column] = INFINITY;
        out_indices[size_t(query) * k + column] = 2147483647;
    }
}
if (reference < 0 || reference >= shape[0]) return;
const float distance = values[item];
int rank = 0;
for (int other = 0; other < width; ++other) {
    const size_t at = size_t(query) * width + other;
    if (other != column && precise_cached_precedes<D>(
        values[at], indices[at], distance, reference, train, queries, query, shape[0], tails[at], tails[item])) ++rank;
}
if (rank < k) {
    out_distances[size_t(query) * k + rank] = distance;
    out_indices[size_t(query) * k + rank] = reference;
}
"""


@lru_cache(maxsize=1)
def _candidate_order_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_precise_candidate_order",
        input_names=["train", "queries", "indices", "values", "tails", "shape"],
        output_names=["out_distances", "out_indices"],
        source=_CANDIDATE_ORDER, header=_HEADER + PRECISION_HEADER + _CACHED_PRECISION,
    )


@mx.compile
def _sort_cached_candidates(values, indices, tails, n):
    keys = (values.view(mx.uint32).astype(mx.uint64) << 32) | indices.astype(mx.uint64)
    keys = mx.where((indices >= 0) & (indices < n), keys, ~mx.array(0, mx.uint64))
    order = mx.argsort(keys, axis=1)
    return tuple(mx.take_along_axis(a, order, axis=1) for a in (values, indices, tails))


@lru_cache(maxsize=1)
def _sorted_candidate_kernel():
    # Ordinary lists are already ordered by their high components. Verify each
    # adjacent original-coordinate comparison, then rank only disordered rows.
    source = r"""
const uint lane = thread_index_in_simdgroup;
const int query = int(thread_position_in_grid.x / 32);
if (query >= shape[1]) return;
const int n = shape[0], width = shape[2], k = shape[3];
const size_t base = size_t(query) * width;
bool ordered = true;
for (int column = int(lane); column + 1 < width; column += 32) {
    const size_t at = base + column;
    ordered = ordered && !precise_cached_precedes<D>(values[at + 1], indices[at + 1],
        values[at], indices[at], train, queries, query, n, tails[at + 1], tails[at]);
}
if (metal::simd_all(ordered)) {
    for (int column = int(lane); column < k; column += 32) {
        const int id = indices[base + column];
        const bool valid = id >= 0 && id < n;
        out_distances[size_t(query) * k + column] = valid ? values[base + column] : INFINITY;
        out_indices[size_t(query) * k + column] = valid ? id : 2147483647;
    }
    return;
}
for (int column = int(lane); column < width; column += 32) {
    const size_t item = base + column;
""" + _CANDIDATE_ORDER[_CANDIDATE_ORDER.index("const int reference = indices[item];"):].replace(
        "if (reference < 0 || reference >= shape[0]) return;",
        "if (reference < 0 || reference >= shape[0]) continue;",
    ) + "\n}"
    return mx.fast.metal_kernel(
        name="mlxlearn_verify_sorted_candidates",
        input_names=["train", "queries", "indices", "values", "tails", "shape"],
        output_names=["out_distances", "out_indices"], source=source,
        header=_HEADER + PRECISION_HEADER + _CACHED_PRECISION,
    )


@lru_cache(maxsize=1)
def _candidate_simd_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_precise_candidate_simd",
        input_names=["train", "queries", "indices", "values", "shape"],
        output_names=["out_distances", "out_indices"],
        source=r"""
const uint lane = thread_index_in_simdgroup;
const int query = int(thread_position_in_grid.x / 32);
if (query >= shape[1]) return;
const int width = shape[2], k = shape[3], n = shape[0];
int index = lane < uint(width) ? indices[size_t(query) * width + lane] : 2147483647;
if (index < 0 || index >= n) index = 2147483647;
float distance = index < n ? values[size_t(query) * width + lane] : INFINITY;
for (int slot = 0; slot < k; ++slot) {
    const auto winner = precise_simd_min<D>(distance, index, train, queries, query, n);
    if (lane == uint(slot)) {
        out_distances[size_t(query) * k + slot] = winner.distance;
        out_indices[size_t(query) * k + slot] = winner.index;
    }
    if (index == winner.index) { index = 2147483647; distance = INFINITY; }
}
""", header=_HEADER + PRECISION_HEADER,
    )


def precise_select_candidates(train, queries, indices, k, values=None):
    """Order distinct candidate IDs by original-coordinate distance, then ID.

    Non-overlapping float32 error intervals settle ordinary comparisons.
    Guarded double-single arithmetic resolves close pairs; unresolved pairs
    use exact integers on the original coordinate bits. Wide feature vectors
    share their refined distance across comparisons. Invalid IDs sort last.
    Valid IDs in each candidate row must be distinct.
    """
    wide = train.shape[1] > 32 or indices.shape[1] > 32
    if wide:
        values, tails = _wide_refine_kernel()(
            inputs=[train, queries, indices, mx.array([train.shape[0], queries.shape[0], indices.shape[1]], mx.int32)],
            template=[("D", train.shape[1])],
            grid=(((indices.size * 32 + 127) // 128) * 128, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[indices.shape] * 2, output_dtypes=[mx.float32] * 2, stream=mx.gpu,
        )
    else:
        if values is None:
            values = refine_neighbors(train, queries, indices)
        tails = values  # unused for D<=32
    small = indices.shape[1] <= 32 and not wide
    sorted_list = indices.shape[1] > 32
    if sorted_list:
        values, indices, tails = _sort_cached_candidates(values, indices, tails, train.shape[0])
    kernel = (_candidate_simd_kernel() if small else
              _sorted_candidate_kernel() if sorted_list else _candidate_order_kernel())
    threads = queries.shape[0] * 32 if small or sorted_list else indices.size
    return tuple(kernel(
        inputs=[train, queries, indices, values] + ([] if small else [tails]) +
               [mx.array([train.shape[0], queries.shape[0], indices.shape[1], k], mx.int32)],
        template=[("D", train.shape[1])],
        grid=(((threads + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(queries.shape[0], k)] * 2,
        output_dtypes=[mx.float32, mx.int32], stream=mx.gpu,
    ))


_EXHAUSTIVE_LARGE_K = r"""
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
const int query = int(thread_position_in_grid.x);
if (query >= shape[1]) return;
const int n = shape[0], k = shape[2];
const size_t base = size_t(query) * k;
const int sentinel = 2147483647;
for (int slot = 0; slot < k; ++slot) {
    out_distances[base + slot] = INFINITY;
    out_indices[base + slot] = sentinel;
}
for (int reference = 0; reference < n; ++reference) {
    if (EXCLUDE_SELF && reference == query + shape[3]) continue;
    float distance = 0.0f, correction = 0.0f;
    for (int feature = 0; feature < D; ++feature) {
        const float delta = queries[size_t(query) * D + feature]
                          - train[size_t(reference) * D + feature];
        const float adjusted = delta * delta - correction;
        const float total = distance + adjusted;
        correction = metal::isfinite(total) ? (total - distance) - adjusted : 0.0f;
        distance = total;
    }
    if (!precise_neighbor_precedes<D>(distance, reference,
            out_distances[base + k - 1], out_indices[base + k - 1], train, queries, query, n)) continue;
    int slot = k - 1;
    while (slot > 0 && precise_neighbor_precedes<D>(distance, reference,
            out_distances[base + slot - 1], out_indices[base + slot - 1], train, queries, query, n)) {
        out_distances[base + slot] = out_distances[base + slot - 1];
        out_indices[base + slot] = out_indices[base + slot - 1];
        --slot;
    }
    out_distances[base + slot] = distance;
    out_indices[base + slot] = reference;
}
"""


@lru_cache(maxsize=2)
def _certified_kernel(large_k):
    if large_k:
        source = _EXHAUSTIVE_LARGE_K.replace(
            "const int n = shape[0], k = shape[2];",
            """const int n = shape[0], k = shape[2];
if (certified[query]) {
    for (int slot = 0; slot < k; ++slot) {
        out_distances[size_t(query) * k + slot] = candidate_distances[size_t(query) * k + slot];
        out_indices[size_t(query) * k + slot] = candidate_indices[size_t(query) * k + slot];
    }
    return;
}""",
        )
    else:
        source = _SCAN.replace(
            "const int part = int(task % uint(parts));",
            """if (certified[query]) {
        if (lane < uint(K)) {
            out_distances[size_t(query) * K + lane] = candidate_distances[size_t(query) * K + lane];
            out_indices[size_t(query) * K + lane] = candidate_indices[size_t(query) * K + lane];
        }
        return;
    }
    const int part = int(task % uint(parts));""",
        )
    return mx.fast.metal_kernel(
        name=f"mlxlearn_certified_general_finish_{int(large_k)}",
        input_names=["train", "queries", "candidate_distances", "candidate_indices", "certified", "shape"],
        output_names=["out_distances", "out_indices"], source=source,
        header=_HEADER + PRECISION_HEADER,
    )


def certified_neighbors(train, queries, k, values, indices, certificate, *,
                        exclude_self=False, query_offset=0):
    """Keep certified rows and scan every original reference for the rest."""
    rows, features = queries.shape
    large = k > 8
    shape = ([train.shape[0], rows, k, query_offset] if large
             else [train.shape[0], rows, features, 1, query_offset])
    threads = rows if large else rows * 32
    return tuple(_certified_kernel(large)(
        inputs=[train, queries, values, indices, certificate, mx.array(shape, mx.int32)],
        template=([("K", int(k))] if not large else [])
                 + [("D", features), ("EXCLUDE_SELF", bool(exclude_self))],
        grid=(((threads + 127) // 128) * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows, k)] * 2, output_dtypes=[mx.float32, mx.int32], stream=mx.gpu,
    ))
