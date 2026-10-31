"""One-launch original-coordinate neighbor selection for small workloads.

A threadgroup owns each query. Its SIMDgroups scan disjoint training partitions
with the established compensated distances and omitted-candidate bounds. The
first SIMDgroup then merges and certifies the answer in the same launch. It
resolves uncertain order with the original exact comparator and rescans every
reference when membership is uncertain.

The rescan can reject a point only when its conservative distance lower bound
exceeds the upper bound from K distinct fast candidates. Those K candidates
bound the true kth distance even when fast membership was wrong. Strict
inequality preserves boundary ties; unsafe ranges disable the pruning.

All preparation remains lazy and uses dynamic training/query array inputs.
Only the kernel factory and compiled graph are cached, never dataset values or
results. Explicit GPU dispatch and validation belong to DeviceNeighborIndex.
"""

from functools import lru_cache

import mlx.core as mx

from ._neighbors_fused import _HEADER, _SCAN
from ._neighbors_precision import PRECISION_HEADER
from ._neighbors_screen import _scan_source

_MERGE_AND_CERTIFY = r"""
// Every producer reaches this barrier: grid dimensions contain exactly one
// complete threadgroup per query, including groups whose partition is empty.
threadgroup_barrier(mem_flags::mem_threadgroup);
if (thread_position_in_threadgroup.x >= 32) return;
float fallback_cutoff = INFINITY;
{
    const uint lane = thread_index_in_simdgroup;
    const int query = int(threadgroup_position_in_grid.x);
    const int n_train = shape[0];
    const int sentinel = 2147483647;
    int cursor = 0;
    // Keep the merged list distributed over the first SIMDgroup's registers;
    // shuffle operations make the adjacent-order check independent of memory.
    float selected_distance = INFINITY;
    int selected_index = sentinel;
    #pragma clang loop unroll(disable)
    for (int slot = 0; slot < K; ++slot) {
        const bool present = lane < uint(PARTS) && cursor < K;
        const size_t address = size_t(lane) * K + cursor;
        const float head = present ? partial_distances[address] : INFINITY;
        const int index = present ? partial_indices[address] : sentinel;
        const float winner_distance = metal::simd_min(head);
        const int winner_index = metal::simd_min(head == winner_distance ? index : sentinel);
        if (lane == uint(slot)) {
            selected_distance = winner_distance;
            selected_index = winner_index;
        }
        if (present && index == winner_index) ++cursor;
    }
    const bool present = lane < uint(PARTS) && cursor < K;
    const size_t address = size_t(lane) * K + cursor;
    const float head = present ? partial_distances[address] : INFINITY;
    const float omitted = lane < uint(PARTS) ? partial_barriers[lane] : INFINITY;
    const float barrier = metal::simd_min(metal::min(head, omitted));
    const float next = metal::simd_shuffle_down(selected_distance, 1);
    const float last = metal::simd_shuffle(selected_distance, ushort(K - 1));
    // These K distinct valid references bound the true kth distance even if
    // fast membership is wrong. upper() is monotone on finite screen values;
    // unsafe values produce infinity and disable the pruning below.
    fallback_cutoff = precise_distance_upper<D>(last);
    bool clear = true;
    if (lane < uint(K) && (lane + 1 < uint(K) || n_train - int(EXCLUDE_SELF) > K)) {
        const float right = lane + 1 < uint(K) ? next : barrier;
        clear = precise_distance_lower<D>(right) > precise_distance_upper<D>(selected_distance);
    }
    if (metal::simd_all(clear)) {
        if (lane < uint(K)) {
            out_distances[size_t(query) * K + lane] = SQRT ? metal::sqrt(selected_distance) : selected_distance;
            out_indices[size_t(query) * K + lane] = selected_index;
        }
        return;
    }
    const bool membership = n_train - int(EXCLUDE_SELF) == K ||
        precise_distance_lower<D>(barrier) > precise_distance_upper<D>(last);
    if (membership) {
        // The omission certificate proves that these K references contain
        // every answer. Resolve only their order with the unchanged exact
        // comparison, including mathematically equal distances and index ties.
        #pragma clang loop unroll(disable)
        for (int slot = 0; slot < K; ++slot) {
            const auto winner = precise_simd_min<D>(selected_distance, selected_index,
                                                   train, queries, query, n_train);
            if (lane == uint(slot)) {
                out_distances[size_t(query) * K + slot] = SQRT ? metal::sqrt(winner.distance) : winner.distance;
                out_indices[size_t(query) * K + slot] = winner.index;
            }
            if (selected_index == winner.index) {
                selected_index = sentinel;
                selected_distance = INFINITY;
            }
        }
        return;
    }
}
// Membership was not certified. The first SIMDgroup executes an exhaustive
// original-coordinate exact selection; all other SIMDgroups already returned.
"""


def _source():
    # Reuse the established compensated scan and omitted-candidate calculation
    # verbatim except for their scratch destinations. PARTS equals the number
    # of SIMDgroups in the threadgroup, so one group always owns one query.
    scan = _scan_source().replace("const int parts = shape[3];", "const int parts = PARTS;")
    scan = scan.replace("out_distances[size_t(task) * K + slot]", "partial_distances[size_t(part) * K + slot]")
    scan = scan.replace("out_indices[size_t(task) * K + slot]", "partial_indices[size_t(part) * K + slot]")
    scan = scan.replace("out_barriers[task]", "partial_barriers[part]")
    assert "out_distances[" not in scan and "out_indices[" not in scan
    assert "out_barriers[" not in scan
    # The fallback sees a single partition and a query-global task index. Its
    # shape[4] remains the caller's original self-exclusion offset.
    fallback = _SCAN.replace(
        "const uint task = thread_position_in_grid.x / 32;",
        "const uint task = threadgroup_position_in_grid.x;",
    ).replace("const int parts = shape[3];", "const int parts = 1;")
    fallback = fallback.replace(
        "out_distances[size_t(task) * K + slot] = winner_distance;",
        "out_distances[size_t(task) * K + slot] = SQRT ? metal::sqrt(winner_distance) : winner_distance;",
    )
    insertion = "        if (precise_neighbor_precedes<D>(distance, reference, distances[K - 1]"
    assert fallback.count(insertion) == 1
    fallback = fallback.replace(
        insertion,
        "        if (precise_distance_lower<D>(distance) > fallback_cutoff) continue;\n"
        + insertion,
    )
    return ("threadgroup float partial_distances[PARTS * K];\n"
            "threadgroup int partial_indices[PARTS * K];\n"
            "threadgroup float partial_barriers[PARTS];\n{\n" + scan
            + "\n}\n" + _MERGE_AND_CERTIFY + "\n{\n" + fallback + "\n}\n")


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_small_neighbors",
        input_names=["train", "queries", "shape"],
        output_names=["out_distances", "out_indices"],
        source=_source(), header=_HEADER + PRECISION_HEADER,
    )


@mx.compile
def _launch(train, queries, k, partitions, exclude_self, query_offset, sqrt_distances):
    n, features = train.shape
    rows = queries.shape[0]
    threads = partitions * 32
    return tuple(_kernel()(
        inputs=[train, queries, mx.array([n, rows, features, partitions, query_offset], mx.int32)],
        template=[("K", k), ("D", features), ("PARTS", partitions),
                  ("EXCLUDE_SELF", exclude_self), ("SQRT", sqrt_distances)],
        grid=(rows * threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(rows, k)] * 2, output_dtypes=[mx.float32, mx.int32], stream=mx.gpu,
    ))


def small_neighbors(
    train, queries, k, *, partitions=4, exclude_self=False, query_offset=0,
    sqrt_distances=False,
):
    """Search validated float32 matrices using one threadgroup per query.

    The caller supplies 1<=k<=32 valid available references, 1..32 SIMDgroups
    per threadgroup, positive matching feature counts and an active GPU stream.
    With self exclusion, query row i corresponds to training query_offset+i.
    Every output remains lazy. Squared distances are the default; the optional
    final square root shares the selection kernel instead of adding a launch.
    """
    if queries.shape[0] == 0:
        return mx.zeros((0, k), mx.float32), mx.zeros((0, k), mx.int32)
    return _launch(train, queries, int(k), int(partitions), exclude_self,
                   int(query_offset), sqrt_distances)
