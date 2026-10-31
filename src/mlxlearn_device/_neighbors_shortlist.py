"""Fuse bounded candidate refinement, precise ordering and omission checking.

Each query owns one threadgroup. A SIMD group refines each candidate using the
same double-single arithmetic as the general wide-feature path; the shared
results then feed the unchanged exact-capable ordering comparison. This kernel
only certifies the shortlist. The caller retains the complete-search fallback.
"""

from functools import lru_cache

import mlx.core as mx

from ._neighbors_fused import _CACHED_PRECISION
from ._neighbors_precision import PRECISION_HEADER

_REFINE_AND_ORDER = r"""
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint column = tid / 32;
const int query = int(threadgroup_position_in_grid.x);
const int n = shape[0];
if (query >= shape[1]) return;
threadgroup float refined[C];
threadgroup float tails[C];
threadgroup int references[C];
threadgroup float cutoff;
if (tid == 0) cutoff = INFINITY;

if (column < uint(C)) {
    const int reference = indices[size_t(query) * C + column];
    const bool valid = reference >= 0 && reference < n;
    float2 sum = float2(0.0f);
    bool safe = D <= 4096;
    if (valid) {
        #pragma clang loop unroll(disable)
        for (int feature = int(lane); feature < D; feature += 32) {
            const float q = queries[size_t(query) * D + feature];
            const float x = train[size_t(reference) * D + feature];
            const uint qb = as_type<uint>(q) & 0x7fffffffu;
            const uint xb = as_type<uint>(x) & 0x7fffffffu;
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
    }
    if (lane == 0) {
        references[column] = reference;
        if (!valid) {
            refined[column] = INFINITY;
            tails[column] = NAN;
        } else if (safe) {
            refined[column] = sum.x;
            tails[column] = sum.y;
        } else {
            // Preserve the ordinary compensated result outside the guarded
            // double-single range; NAN disables its cached precision filter.
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
            refined[column] = value;
            tails[column] = NAN;
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Only the first C threads rank candidates. Invalid output slots cannot race
// with valid ranks: the latter occupy exactly [0, available).
if (tid < uint(C)) {
    const int reference = references[tid];
    if (tid < uint(K)) {
        int available = 0;
        for (int other = 0; other < C; ++other)
            available += int(references[other] >= 0 && references[other] < n);
        if (int(tid) >= available) {
            out_distances[size_t(query) * K + tid] = INFINITY;
            out_indices[size_t(query) * K + tid] = 2147483647;
        }
    }
    if (reference >= 0 && reference < n) {
        const float distance = refined[tid];
        int rank = 0;
        for (int other = 0; other < C; ++other) {
            if (other != int(tid) && precise_cached_precedes<D>(
                refined[other], references[other], distance, reference,
                train, queries, query, n, tails[other], tails[tid])) ++rank;
        }
        if (rank < K) {
            out_distances[size_t(query) * K + rank] = distance;
            out_indices[size_t(query) * K + rank] = reference;
            if (rank + 1 == K) cutoff = distance;
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    bool certified = certificate[query];
    if (CHECK_BARRIER) {
        const float upper = cutoff * (1.0f + 6.0f * 0x1p-23f) + distance_floor[0];
        certified = certified && metal::isfinite(upper) && barriers[query] > upper;
    }
    out_certified[query] = certified;
}
"""


@lru_cache(maxsize=1)
def _shortlist_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_refine_certified_shortlist",
        input_names=[
            "train", "queries", "indices", "barriers", "certificate", "shape", "distance_floor",
        ],
        output_names=["out_distances", "out_indices", "out_certified"],
        source=_REFINE_AND_ORDER,
        header=PRECISION_HEADER + _CACHED_PRECISION,
    )


def refine_certified_candidates(
    train, queries, indices, k, barriers, certificate, *, check_barrier,
):
    """Refine C<=16 distinct valid IDs and preserve the omission certificate.

    This internal specialization accepts 33..256 features and k<=8. Invalid
    candidate IDs (including masked self identities) sort after all real points.
    ``barriers`` bounds omitted references in the current training tile, while
    ``certificate`` carries the conjunction from preceding tiles. A tile that
    contains every reference uses ``check_barrier=False``. An uncertain result
    must still reach the caller's complete original-coordinate fallback.
    """
    rows, features = queries.shape
    width = indices.shape[1]
    if not (33 <= features <= 256 and 1 <= k <= min(width, 8) and width <= 16):
        raise ValueError("shortlist refinement requires D=33..256, C=1..16, and k<=min(C,8)")
    if indices.shape[0] != rows or train.shape[1] != features:
        raise ValueError("shortlist inputs must have matching query and feature counts")
    if rows == 0:
        return mx.zeros((0, k), mx.float32), mx.zeros((0, k), mx.int32), mx.zeros((0,), mx.bool_)
    group_size = (1 << (width - 1).bit_length()) * 32
    return tuple(_shortlist_kernel()(
        inputs=[
            train, queries, indices, barriers, certificate,
            mx.array([train.shape[0], rows], mx.int32),
            mx.array([features * 1e-30], mx.float32),
        ],
        template=[("D", features), ("C", width), ("K", int(k)), ("CHECK_BARRIER", bool(check_barrier))],
        grid=(rows * group_size, 1, 1), threadgroup=(group_size, 1, 1),
        output_shapes=[(rows, k), (rows, k), (rows,)],
        output_dtypes=[mx.float32, mx.int32, mx.bool_], stream=mx.gpu,
    ))
