"""Fast direct scans with a certified omission boundary and exact fallback.

The scan uses the established FP32 ranking. Its next-omitted distance certifies
membership, and adjacent output intervals certify order. Only uncertain rows
execute the exact-comparison fallback: small k <= 8 searches scan each such row
with its own SIMD group in the finishing dispatch; all others hand them to the
partitioned search of ``_neighbors_fallback``, bounded by the row's fast
candidates so that references provably beyond its k-th distance skip the
exact comparisons that exact ties would otherwise require. No host
synchronization is required.
"""

from functools import lru_cache

import mlx.core as mx

from ._neighbors_fused import _HEADER, _MERGE, _SCAN
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


# One SIMD group per row. "Clear" rows have separated adjacent intervals and
# keep the fast order; "membership" rows have a certified boundary and are
# re-sorted with exact comparisons. A certified row returns early, recording
# ``{certify}`` first; every other row needs a complete exact search. Lane l
# checks slot l, so K <= 32, and W >= K is the width of the fast lists.
_CERTIFY = """
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
        }{certify}
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
        }{certify}
        return;
    }
"""


@lru_cache(maxsize=1)
def _finish_kernel():
    """Certify rows, then scan each other row with its SIMD group, in one dispatch.

    Only for K <= 8: the scan keeps a K-entry list per lane in a private array,
    and Metal's backend compile time explodes beyond that (M4 Max: under a
    second at K=12, over 90 s at K=16, over 30 minutes at K=32).
    """
    source = _SCAN.replace(
        "const int part = int(task % uint(parts));",
        _CERTIFY.replace("{certify}", "") + "    const int part = int(task % uint(parts));", 1)
    return mx.fast.metal_kernel(
        name="mlxlearn_screen_finish",
        input_names=["train", "queries", "fast_distances", "fast_indices", "barriers", "shape"],
        output_names=["out_distances", "out_indices"], source=source,
        header=_HEADER + PRECISION_HEADER,
    )


# An uncertain row's first K fast entries are K distinct, non-excluded
# training rows, as long as every index is real, each with the compensated
# FP32 distance the exact scans compute. (The paired scan's unguarded
# compensation differs from the guarded one only after a sum overflows, and
# both then report +inf.) The largest of those distances therefore has an
# upper endpoint at least the exact distance of each of the K rows, and so at
# least the row's true K-th distance. The row's outputs are written with that
# largest distance last (index -1 unless all K indices are real), as the
# bounded partitioned search requires; the search recomputes every output.
# The fast list is sorted by FP32 value, so the largest is normally its last
# entry; taking the maximum keeps the bound sound without relying on that.
_UNCERTAIN = """
    const bool listed = lane < uint(K);
    const float fast = listed ? fast_distances[size_t(query) * W + lane] : 0.0f;
    const int fast_index = listed ? fast_indices[size_t(query) * W + lane] : 0;
    const float largest = metal::simd_max(fast);
    const bool real = metal::simd_all(!listed || (fast_index >= 0 && fast_index < n_train));
    if (listed) {
        const bool last = lane + 1 == uint(K);
        out_distances[size_t(query) * K + lane] = last ? largest : fast;
        out_indices[size_t(query) * K + lane] = last && !real ? -1 : fast_index;
    }
    if (lane == 0) certified[query] = false;
"""


@lru_cache(maxsize=1)
def _certify_kernel():
    """Finish certified rows and flag the rest for the bounded partitioned search."""
    source = """
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    const uint lane = thread_index_in_simdgroup;
    const int query = int(thread_position_in_grid.x / 32);
    const int n_train = shape[0];
    if (query >= shape[1]) return; // uniform across this entire SIMDgroup
""" + _CERTIFY.replace("{certify}", "\n        if (lane == 0) certified[query] = true;") + _UNCERTAIN
    return mx.fast.metal_kernel(
        name="mlxlearn_screen_certify",
        input_names=["train", "queries", "fast_distances", "fast_indices", "barriers", "shape"],
        output_names=["out_distances", "out_indices", "certified"], source=source,
        header=_HEADER + PRECISION_HEADER,
    )


#: Training work (rows x features) from which uncertain k <= 8 rows go to the
#: partitioned search. On M4 Max one uncertain row cost a SIMD-group scan about
#: 0.2 ms at 2**15 (0.5 ms at 2**18, 2 ms at 2**21), and exact ties cost far
#: more; the partitioned search was 2-5x faster on tied integer data from
#: n_train ~1500. Its three extra dispatches cost 25-40 us per block when every
#: row certifies: up to ~25% of a 0.15 ms query, ~1% of a millisecond one.
#: Smaller k <= 8 searches keep the single dispatch.
_PARTITIONED_WORK = 2**15


def _use_partitioned_finish(n_train, features, k):
    """Search uncertain rows with the partitioned fallback.

    k > 8 always does: the single-dispatch kernel is limited to K <= 8, and
    the one-thread-per-row scan it used to fall back to cost 0.5 ms or more
    per uncertain row even at n_train = 2049 (hundreds of ms at 262144).
    """
    return k > 8 or n_train * features >= _PARTITIONED_WORK


def _finish(train, queries, k, values, indices, barriers, *, width, exclude_self, query_offset):
    n, features = train.shape
    rows = queries.shape[0]
    shape = mx.array([n, rows, features, 1, query_offset], mx.int32)
    template = [("K", k), ("W", width), ("D", features), ("EXCLUDE_SELF", exclude_self)]
    launch = {"grid": (((rows * 32 + 127) // 128) * 128, 1, 1), "threadgroup": (128, 1, 1),
              "stream": mx.gpu}
    inputs = [train, queries, values, indices, barriers, shape]
    if k <= 8 and not _use_partitioned_finish(n, features, k):
        return tuple(_finish_kernel()(
            inputs=inputs, template=template, output_shapes=[(rows, k)] * 2,
            output_dtypes=[mx.float32, mx.int32], **launch,
        ))
    from ._neighbors_fallback import partitioned_neighbors

    values, indices, certified = _certify_kernel()(
        inputs=inputs, template=template, output_shapes=[(rows, k), (rows, k), (rows,)],
        output_dtypes=[mx.float32, mx.int32, mx.bool_], **launch,
    )
    return partitioned_neighbors(
        train, queries, k, values, indices, certified,
        exclude_self=exclude_self, query_offset=query_offset, bounded=True,
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
    return _finish(train, queries, k, values, indices, barriers, width=width,
                   exclude_self=exclude_self, query_offset=query_offset)
