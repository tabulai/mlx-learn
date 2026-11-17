"""Partitioned exact search for query rows whose Gram screen did not certify.

The general GPU path proves most rows from Gram lower bounds. A row that fails
needs a complete scan of the original training coordinates. Before this module
that scan ran on a single GPU thread per row for k > 8: one such row at
131072 x 1024 took two to four seconds on M4 Max, while a block of certified
rows finished in milliseconds.

Here every uncertified row is split into many contiguous chunks of training
rows, and one SIMD group scans each chunk. Rows are compacted on the GPU from
an inclusive prefix count of certified flags, so the grid and scratch are
fixed in advance and no host synchronization is needed: a few uncertified
rows receive hundreds of chunks each, and a block of entirely uncertified rows
still gets at least one chunk per row.

Distances are the same compensated FP32 sums as the original serial scan.
Each chunk keeps its own sorted top-k list, and the lists are merged with the
exact original-coordinate comparisons, so the result is the same (distance,
index) ordering bit for bit. A chunk's list holds the chunk's exact top k
entries, so the union of lists contains the row's top k entries.

Callers whose uncertified rows still carry k exactly sorted real candidates
(the compact and integer screens) may pass ``bounded=True``. The last
candidate then bounds the row's k-th distance, and references provably
farther are skipped before any exact comparison. On data with exact ties
that skips most comparisons; the result is unchanged.
"""

from functools import lru_cache

import mlx.core as mx

from ._neighbors_fused import _HEADER
from ._neighbors_precision import PRECISION_HEADER

#: SIMD groups launched for the scan (more when a block has more rows), with
#: at most ``_SCRATCH_ENTRIES`` list entries of scratch. Up to eight
#: uncertified rows still get the maximum 256 chunks each. Every launched
#: group costs time even when all rows are certified: on M4 Max 4096 groups
#: took 14 us per block and 2048 took 8 us.
_TARGET_GROUPS = 2048
_SCRATCH_ENTRIES = 8 << 20
#: Each merge lane owns up to this many chunk lists, bounding chunks per row
#: at 32 times this value.
_LISTS_PER_LANE = 8

_CHUNKS = r"""
// Chunks per uncertified row. Both kernels evaluate this with the same
// inputs. count * chunks <= groups because groups >= rows >= count.
inline int fallback_chunks(int count, int groups, int n, int k) {
    if (count <= 0) return 0;
    // Short chunks spend their time filling a list rather than scanning.
    const int minimum = k < (1 << 20) ? metal::max(128, 4 * k) : (1 << 22);
    int chunks = metal::min(groups / count, 32 * LISTS);
    chunks = metal::min(chunks, metal::max(1, n / minimum));
    return metal::max(chunks, 1);
}
"""

_SCAN = r"""
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
const uint lane = thread_index_in_simdgroup;
const int group = int(thread_position_in_grid.x / 32);
const int n = shape[0], rows = shape[1], k = shape[2], groups = shape[4];
// positions[r] counts certified rows 0..r, so r + 1 - positions[r] counts
// the uncertified ones.
const int count = rows - positions[rows - 1];
const int chunks = fallback_chunks(count, groups, n, k);
if (group >= count * chunks) return;  // uniform across the SIMD group
// Chunk-major numbering: groups that run together read the same training
// rows for different queries.
const int chunk = group / count, compact = group % count;
int low = 0, high = rows - 1;
while (low < high) {  // first row whose inclusive count exceeds compact
    const int middle = (low + high) / 2;
    if (middle + 1 - positions[middle] > compact) high = middle; else low = middle + 1;
}
const int query = low;
const int begin = int((size_t(chunk) * size_t(n)) / size_t(chunks));
const int end = int((size_t(chunk + 1) * size_t(n)) / size_t(chunks));
const size_t base = size_t(group) * size_t(k);
const int sentinel = 2147483647;
const ushort last = ushort((k - 1) % 32);
// Lane l owns list slots l, l + 32, ...: every slot is read and written by
// its owner only, and values move between lanes through SIMD shuffles.
for (int slot = int(lane); slot < k; slot += 32) {
    out_distances[base + slot] = INFINITY;
    out_indices[base + slot] = sentinel;
}
float worst = INFINITY;
int worst_index = sentinel;
// BOUNDED: a valid last candidate is the k-th of k distinct, non-excluded
// references in exact order, with its compensated distance. The row's true
// k-th distance is then at most that distance's upper endpoint, and a
// reference whose lower endpoint exceeds it is strictly farther than k others.
// Skipping it avoids exact comparisons against tied list entries.
float limit = INFINITY;
if (BOUNDED) {
    const size_t last = size_t(query) * size_t(k) + size_t(k - 1);
    const int bound_index = candidate_indices[last];
    if (bound_index >= 0 && bound_index < n)
        limit = precise_distance_upper<D>(candidate_distances[last]);
}
for (int start = begin; start < end; start += 32) {
    const int reference = start + int(lane);
    bool pending = reference < end && !(EXCLUDE_SELF && reference == query + shape[3]);
    float distance = INFINITY;
    if (pending) {
        float correction = 0.0f;
        distance = 0.0f;
        for (int feature = 0; feature < D; ++feature) {
            const float delta = queries[size_t(query) * D + feature]
                              - train[size_t(reference) * D + feature];
            const float adjusted = delta * delta - correction;
            const float total = distance + adjusted;
            correction = metal::isfinite(total) ? (total - distance) - adjusted : 0.0f;
            distance = total;
        }
        pending = !(BOUNDED && precise_distance_lower<D>(distance) > limit) &&
                  precise_neighbor_precedes<D>(distance, reference, worst, worst_index,
                                               train, queries, query, n);
    }
    while (metal::simd_any(pending)) {
        const ushort leader = ushort(metal::simd_min(pending ? lane : 32u));
        const float value = metal::simd_shuffle(distance, leader);
        const int index = metal::simd_shuffle(reference, leader);
        if (lane == leader) pending = false;
        // An earlier insertion from this batch may have displaced the cutoff.
        if (!precise_neighbor_precedes<D>(value, index, worst, worst_index,
                                          train, queries, query, n)) continue;
        int ahead = 0;
        for (int slot = int(lane); slot < k; slot += 32) {
            ahead += int(precise_neighbor_precedes<D>(out_distances[base + slot],
                out_indices[base + slot], value, index, train, queries, query, n));
        }
        const int rank = metal::simd_sum(ahead);
        float carried = INFINITY;
        int carried_index = sentinel;
        for (int first = (rank / 32) * 32; first < k; first += 32) {
            const int slot = first + int(lane);
            const bool own = slot < k;
            const float held = own ? out_distances[base + slot] : INFINITY;
            const int held_index = own ? out_indices[base + slot] : sentinel;
            float moved = metal::simd_shuffle_up(held, ushort(1));
            int moved_index = metal::simd_shuffle_up(held_index, ushort(1));
            if (lane == 0) {
                moved = carried;
                moved_index = carried_index;
            }
            carried = metal::simd_broadcast(held, ushort(31));
            carried_index = metal::simd_broadcast(held_index, ushort(31));
            if (own && slot >= rank) {
                out_distances[base + slot] = slot == rank ? value : moved;
                out_indices[base + slot] = slot == rank ? index : moved_index;
            }
        }
        const bool owner = lane == uint(last);
        worst = metal::simd_broadcast(owner ? out_distances[base + k - 1] : 0.0f, last);
        worst_index = metal::simd_broadcast(owner ? out_indices[base + k - 1] : 0, last);
    }
}
"""

_MERGE = r"""
const uint lane = thread_index_in_simdgroup;
const int query = int(thread_position_in_grid.x / 32);
const int n = shape[0], rows = shape[1], k = shape[2], groups = shape[4];
if (query >= rows) return;
const size_t out = size_t(query) * size_t(k);
if (certified[query]) {
    for (int slot = int(lane); slot < k; slot += 32) {
        out_distances[out + slot] = candidate_distances[out + slot];
        out_indices[out + slot] = candidate_indices[out + slot];
    }
    return;
}
const int count = rows - positions[rows - 1];
const int compact = query - positions[query];
const int chunks = fallback_chunks(count, groups, n, k);
const int sentinel = 2147483647;
int cursors[LISTS];
#pragma clang loop unroll(full)
for (int list = 0; list < LISTS; ++list) cursors[list] = 0;
for (int slot = 0; slot < k; ++slot) {
    float head = INFINITY;
    int head_index = sentinel, owner = -1;
    #pragma clang loop unroll(full)
    for (int list = 0; list < LISTS; ++list) {
        const int chunk = int(lane) + 32 * list;
        if (chunk < chunks && cursors[list] < k) {
            const size_t at = (size_t(chunk) * size_t(count) + size_t(compact)) * size_t(k)
                            + size_t(cursors[list]);
            const float value = partial_distances[at];
            const int index = partial_indices[at];
            if (precise_neighbor_precedes<D>(value, index, head, head_index,
                                             train, queries, query, n)) {
                head = value;
                head_index = index;
                owner = list;
            }
        }
    }
    const auto winner = precise_simd_min<D>(head, head_index, train, queries, query, n);
    if (lane == uint(slot) % 32) {
        out_distances[out + slot] = winner.distance;
        out_indices[out + slot] = winner.index;
    }
    // Training indices are distinct across chunks, so only one list advances.
    #pragma clang loop unroll(full)
    for (int list = 0; list < LISTS; ++list) {
        if (owner == list && head_index == winner.index) ++cursors[list];
    }
}
"""


def _header():
    return (_HEADER + PRECISION_HEADER
            + f"#define LISTS {_LISTS_PER_LANE}\n" + _CHUNKS)


@lru_cache(maxsize=1)
def _scan_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_uncertified_chunk_scan",
        input_names=["train", "queries", "candidate_distances", "candidate_indices",
                     "positions", "shape"],
        output_names=["out_distances", "out_indices"],
        source=_SCAN, header=_header(),
    )


@lru_cache(maxsize=1)
def _merge_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_uncertified_chunk_merge",
        input_names=["partial_distances", "partial_indices", "candidate_distances",
                     "candidate_indices", "certified", "positions", "train", "queries", "shape"],
        output_names=["out_distances", "out_indices"],
        source=_MERGE, header=_header(),
    )


def scan_groups(rows, n, k):
    """SIMD groups (and scratch lists) for one block; never fewer than rows.

    No more groups are launched than the block's rows could use, which keeps
    small blocks cheap when every row is certified. Mirrors fallback_chunks.
    """
    minimum = max(128, 4 * k) if k < 2**20 else 2**22
    per_row = min(32 * _LISTS_PER_LANE, max(1, n // minimum))
    return max(rows, min(_TARGET_GROUPS, rows * per_row, max(1, _SCRATCH_ENTRIES // k)))


def partitioned_neighbors(train, queries, k, values, indices, certificate, *,
                          exclude_self=False, query_offset=0, bounded=False):
    """Keep certified rows; search every training row for the others.

    ``values``/``indices`` hold each row's sorted candidates and
    ``certificate`` marks rows whose candidates are proven exact. Other rows
    are recomputed from the original coordinates. With ``exclude_self``,
    query row ``i`` skips training row ``query_offset + i``.

    ``bounded`` promises that, for every uncertified row whose last candidate
    index is a valid reference, its k candidates are distinct non-excluded
    references in exact order with their compensated FP32 distances. The
    search then skips references provably farther than that last candidate.
    The result is unchanged; tied data avoid most exact comparisons.
    """
    rows, features = queries.shape
    n = train.shape[0]
    groups = scan_groups(rows, n, k)
    # One prefix count on the GPU compacts the uncertified rows.
    positions = mx.cumsum(certificate)
    if positions.dtype != mx.int32:  # MLX 0.32 already sums bool as int32
        positions = positions.astype(mx.int32)
    shape = mx.array([n, rows, k, query_offset, groups], mx.int32)
    template = [("D", features), ("EXCLUDE_SELF", bool(exclude_self)), ("BOUNDED", bool(bounded))]
    partial_distances, partial_indices = _scan_kernel()(
        inputs=[train, queries, values, indices, positions, shape], template=template,
        grid=(((groups * 32 + 127) // 128) * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(groups, k)] * 2, output_dtypes=[mx.float32, mx.int32],
        stream=mx.gpu,
    )
    return tuple(_merge_kernel()(
        inputs=[partial_distances, partial_indices, values, indices, certificate,
                positions, train, queries, shape],
        template=[("D", features)],
        grid=(((rows * 32 + 127) // 128) * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows, k)] * 2, output_dtypes=[mx.float32, mx.int32],
        stream=mx.gpu,
    ))
