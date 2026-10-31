"""Experimental one-launch direct search for small query workloads.

One threadgroup owns each query. Its SIMDgroups produce the existing screened
local lists and omission bounds into threadgroup storage. After one group
barrier, the first SIMDgroup merges and certifies the result. Ambiguous order
uses the existing exact comparator; uncertain membership executes the existing
full exact scan in that same SIMDgroup. This prototype is not production dispatch.

``small_group_neighbors`` accepts the same already-validated float32 matrices
as the internal search functions and returns lazy squared distances by default.
No host evaluations or transfers are introduced.
"""

from functools import lru_cache

import mlx.core as mx

from mlxlearn_device._neighbors_fused import _HEADER, _SCAN
from mlxlearn_device._neighbors_precision import PRECISION_HEADER
from mlxlearn_device._neighbors_screen import _scan_source

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
    // unsafe values produce infinity and disable the optional pruning below.
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
        "        if (FILTER && precise_distance_lower<D>(distance) > fallback_cutoff) continue;\n"
        + insertion,
    )
    return ("threadgroup float partial_distances[PARTS * K];\n"
            "threadgroup int partial_indices[PARTS * K];\n"
            "threadgroup float partial_barriers[PARTS];\n{\n" + scan
            + "\n}\n" + _MERGE_AND_CERTIFY + "\n{\n" + fallback + "\n}\n")


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_experiment_small_group_neighbors",
        input_names=["train", "queries", "shape"],
        output_names=["out_distances", "out_indices"],
        source=_source(), header=_HEADER + PRECISION_HEADER,
    )


def small_group_neighbors(
    train, queries, k, *, partitions=4, exclude_self=False, query_offset=0,
    sqrt_distances=False, filter_fallback=True,
):
    """One complete threadgroup per query; internal validated-input prototype.

    ``partitions`` controls SIMDgroups per threadgroup (1..32). Larger values
    expose more parallelism within a single query but consume more resources.
    The optional final square root is only for separate diagnostic comparison;
    the default squared outputs match other internal neighbor functions.
    The optional fallback filter discards only references whose conservative
    lower bound exceeds the upper bound supplied by K distinct fast candidates.
    """
    n, features = train.shape
    rows = queries.shape[0]
    if not 1 <= k <= min(32, n - int(exclude_self)):
        raise ValueError("k must be between1 and32 and fit the available references")
    if not 1 <= partitions <= 32:
        raise ValueError("partitions must be between1 and32")
    if queries.shape[1] != features or features < 1:
        raise ValueError("train and queries must have the same positive feature count")
    if train.dtype != mx.float32 or queries.dtype != mx.float32:
        raise TypeError("train and queries must have float32 dtype")
    if rows == 0:
        return mx.zeros((0, k), mx.float32), mx.zeros((0, k), mx.int32)
    threads = partitions * 32
    return tuple(_kernel()(
        inputs=[train, queries, mx.array([n, rows, features, partitions, query_offset], mx.int32)],
        template=[("K", int(k)), ("D", features), ("PARTS", int(partitions)),
                  ("EXCLUDE_SELF", exclude_self), ("SQRT", sqrt_distances),
                  ("FILTER", filter_fallback)],
        grid=(rows * threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(rows, k)] * 2, output_dtypes=[mx.float32, mx.int32], stream=mx.gpu,
    ))
