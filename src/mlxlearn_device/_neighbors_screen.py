"""Fast direct scans with a certified omission boundary and exact fallback.

The scan uses the established FP32 ranking. Its next-omitted distance certifies
membership, and adjacent output intervals certify order. Only uncertain rows
execute the exact-comparison fallback. No host synchronization is required.
"""

from functools import lru_cache

import mlx.core as mx

from ._neighbors_fused import _EXHAUSTIVE_LARGE_K, _HEADER, _MERGE, _SCAN
from ._neighbors_precision import PRECISION_HEADER
from ._neighbors_registers import paired_scan_source


def _scan_source():
    source = _SCAN.replace("precise_neighbor_precedes<D>", "neighbor_precedes")
    source = source.replace(",\n                                           train, queries, query, n_train", "")
    source = source.replace(",\n                                                 train, queries, query, n_train", "")
    source = source.replace(
        "const auto winner = precise_simd_min<D>(head, index, train, queries, query, n_train);\n"
        "        const int winner_index = winner.index;\n        const float winner_distance = winner.distance;",
        "const float winner_distance = metal::simd_min(head);\n"
        "        const int winner_index = metal::simd_min(head == winner_distance ? index : sentinel);",
    )
    source = source.replace("float distances[K];", "float omitted = INFINITY;\n    float distances[K];")
    source = source.replace("        if (neighbor_precedes(distance, reference, distances[K - 1]",
        "        omitted = metal::min(omitted, metal::max(distance, distances[K - 1]));\n"
        "        if (neighbor_precedes(distance, reference, distances[K - 1]")
    source += """
    const float next = metal::simd_min(metal::min(omitted, cursor < K ? distances[cursor] : INFINITY));
    if (lane == 0) out_barriers[task] = next;
"""
    assert "precise_" not in source
    return source


@lru_cache(maxsize=5)
def _scan_kernel(paired, registers, packed):
    source = paired_scan_source(registers=registers, packed=packed, screen=True) if paired else _scan_source()
    assert "precise_" not in source
    return mx.fast.metal_kernel(
        name=f"mlxlearn_screen_scan_{int(paired)}_{int(registers)}_{int(packed)}",
        input_names=["train", "queries", "shape"],
        output_names=["out_distances", "out_indices", "out_barriers"],
        source=source, header=_HEADER,
    )


@lru_cache(maxsize=1)
def _merge_kernel():
    source = _MERGE.replace("slot < K", "slot < O").replace("size_t(query) * K + slot", "size_t(query) * O + slot") + """
    const bool present = lane < uint(parts) && cursor < K;
    const size_t address = (size_t(query) * parts + lane) * K + cursor;
    const float head = present ? partial_distances[address] : INFINITY;
    const float omitted = lane < uint(parts) ? partial_barriers[size_t(query) * parts + lane] : INFINITY;
    const float next = metal::simd_min(metal::min(head, omitted));
    if (lane == 0) out_barriers[query] = next;
"""
    return mx.fast.metal_kernel(
        name="mlxlearn_screen_merge",
        input_names=["partial_distances", "partial_indices", "partial_barriers", "shape"],
        output_names=["out_distances", "out_indices", "out_barriers"], source=source,
    )


@lru_cache(maxsize=2)
def _finish_kernel(large):
    if large:
        source = _EXHAUSTIVE_LARGE_K.replace("const size_t base = size_t(query) * k;", """
const size_t base = size_t(query) * k;
bool clear = true;
for (int slot = 0; slot < k; ++slot) {
    if (slot + 1 == k && n - int(EXCLUDE_SELF) == k) continue;
    const float left = fast_distances[base + slot];
    const float right = slot + 1 < k ? fast_distances[base + slot + 1] : barriers[query];
    clear = clear && precise_distance_lower<D>(right) > precise_distance_upper<D>(left);
}
if (clear) {
    for (int slot = 0; slot < k; ++slot) {
        out_distances[base + slot] = fast_distances[base + slot];
        out_indices[base + slot] = fast_indices[base + slot];
    }
    return;
}
const bool membership = n - int(EXCLUDE_SELF) == k ||
    precise_distance_lower<D>(barriers[query]) > precise_distance_upper<D>(fast_distances[base + k - 1]);
if (membership) {
    for (int slot = 0; slot < k; ++slot) {
        float distance = fast_distances[base + slot];
        int index = fast_indices[base + slot], at = slot;
        while (at > 0 && precise_neighbor_precedes<D>(distance, index,
            out_distances[base + at - 1], out_indices[base + at - 1], train, queries, query, n)) {
            out_distances[base + at] = out_distances[base + at - 1];
            out_indices[base + at] = out_indices[base + at - 1];
            --at;
        }
        out_distances[base + at] = distance; out_indices[base + at] = index;
    }
    return;
}
""", 1)
    else:
        source = _SCAN.replace("const int part = int(task % uint(parts));", """
    bool clear = true;
    if (lane < uint(K) && (lane + 1 < uint(K) || n_train - int(EXCLUDE_SELF) > K)) {
        const float left = fast_distances[size_t(query) * W + lane];
        const float next = lane + 1 < uint(W) ? fast_distances[size_t(query) * W + lane + 1] : INFINITY;
        const float right = lane + 1 < uint(K) ? next : metal::min(next, barriers[query]);
        clear = precise_distance_lower<D>(right) > precise_distance_upper<D>(left);
    }
    if (metal::simd_all(clear)) {
        if (lane < uint(K)) {
            out_distances[size_t(query) * K + lane] = fast_distances[size_t(query) * W + lane];
            out_indices[size_t(query) * K + lane] = fast_indices[size_t(query) * W + lane];
        }
        return;
    }
    const bool membership = n_train - int(EXCLUDE_SELF) == K ||
        precise_distance_lower<D>(barriers[query]) > precise_distance_upper<D>(fast_distances[size_t(query) * W + K - 1]);
    if (membership) {
        float value = lane < uint(W) ? fast_distances[size_t(query) * W + lane] : INFINITY;
        int index = lane < uint(W) ? fast_indices[size_t(query) * W + lane] : 2147483647;
        for (int slot = 0; slot < K; ++slot) {
            const auto winner = precise_simd_min<D>(value, index, train, queries, query, n_train);
            if (lane == uint(slot)) {
                out_distances[size_t(query) * K + slot] = winner.distance;
                out_indices[size_t(query) * K + slot] = winner.index;
            }
            if (index == winner.index) { index = 2147483647; value = INFINITY; }
        }
        return;
    }
    const int part = int(task % uint(parts));
""", 1)
    return mx.fast.metal_kernel(
        name=f"mlxlearn_screen_finish_{int(large)}",
        input_names=["train", "queries", "fast_distances", "fast_indices", "barriers", "shape"],
        output_names=["out_distances", "out_indices"], source=source,
        header=_HEADER + PRECISION_HEADER,
    )


def screen_neighbors(train, queries, k, *, partitions, paired, exclude_self=False, query_offset=0):
    n, features = train.shape
    rows = queries.shape[0]
    tasks = ((rows + 1) // 2 if paired else rows) * partitions
    shape = ([n, rows, partitions, query_offset] if paired
             else [n, rows, features, partitions, query_offset])
    template = [("K", k), ("D", features), ("EXCLUDE_SELF", exclude_self)]
    if paired:
        template.append(("TAIL", bool(rows % 2)))
    values, indices, barriers = _scan_kernel(paired, paired and k == 5, paired and features % 4 == 0)(
        inputs=[train, queries, mx.array(shape, mx.int32)], template=template,
        grid=(((tasks * 32 + 127) // 128) * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows, partitions, k)] * 2 + [(rows, partitions)],
        output_dtypes=[mx.float32, mx.int32, mx.float32], stream=mx.gpu,
    )
    width = k + 1 if partitions > 1 and k <= 8 else k
    if partitions > 1:
        values, indices, barriers = _merge_kernel()(
            inputs=[values, indices, barriers, mx.array([partitions, rows], mx.int32)],
            template=[("K", k), ("O", width)], grid=(((rows * 32 + 127) // 128) * 128, 1, 1),
            threadgroup=(128, 1, 1), output_shapes=[(rows, width)] * 2 + [(rows,)],
            output_dtypes=[mx.float32, mx.int32, mx.float32], stream=mx.gpu,
        )
    else:
        values, indices, barriers = values.reshape(rows, k), indices.reshape(rows, k), barriers.reshape(rows)
    large = k > 8
    shape = ([n, rows, k, query_offset] if large else [n, rows, features, 1, query_offset])
    threads = rows if large else rows * 32
    return tuple(_finish_kernel(large)(
        inputs=[train, queries, values, indices, barriers, mx.array(shape, mx.int32)],
        template=([("K", k), ("W", width)] if not large else []) + [("D", features), ("EXCLUDE_SELF", exclude_self)],
        grid=(((threads + 127) // 128) * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows, k)] * 2, output_dtypes=[mx.float32, mx.int32], stream=mx.gpu,
    ))
