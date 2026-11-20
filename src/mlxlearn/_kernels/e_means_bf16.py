"""Streaming Metal nearest-centroid assignment with FP32 accumulation.

The fused tile kernel keeps scores in registers and writes only one winning
score/index per row and centroid tile. A second SIMD reduction merges tiles.
The tile supports BF16 operands and a binary32 variant for dense FP32 search.

The fragment coordinates and vector load strategy are derived from MLX 0.32.4
``mlx/backend/metal/kernels/steel/gemm/{mma,loader}.h`` (MIT).
Copyright © 2024 Apple Inc.
Modified for mlxlearn on 2026-11-14: BF16 native fragments, fused nearest-center
reduction, guarded ragged loads, bounded row batching, and Python dispatch.
See THIRD_PARTY_NOTICES.md section 1 for the upstream MIT license.
"""

from __future__ import annotations

import operator
from functools import lru_cache

import mlx.core as mx

_UINT32_MAX = 2**32 - 1
_SCRATCH_BYTES = 256 * 1024**2

# Copyright © 2024 Apple Inc.
# Modified for mlxlearn: native BF16 loads and fused centroid argmin, 2026-11-14.
_TILE_SOURCE = r"""

    constexpr uint TM = BM / (8 * WM);
    constexpr uint TN = BN / (8 * WN);
    constexpr uint LD = BK + PAD;
    constexpr uint THREADS = 32 * WM * WN;
    threadgroup T xs[BM * LD];
    // Store centroids in their natural [centroid,feature] layout so that
    // adjacent loading lanes read adjacent global addresses.
    threadgroup T cs[BN * LD];
    uint tid = thread_index_in_threadgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint r0 = threadgroup_position_in_grid.x * BM;
    uint c0 = threadgroup_position_in_grid.y * BN;
    if (r0 >= N || c0 >= K) return;
    uint wm = sg / WN, wn = sg % WN;
    const uint qid_load = lane / 4;
    const uint fm_load = (qid_load & 4) + ((lane / 2) % 4);
    const uint fn_load = (qid_load & 2) * 2 + (lane % 2) * 2;
    metal::simdgroup_float8x8 acc[TM][TN];
    #pragma clang loop unroll(full)
    for (uint m = 0; m < TM; ++m)
        #pragma clang loop unroll(full)
        for (uint n = 0; n < TN; ++n)
            acc[m][n] = metal::simdgroup_float8x8(0);
    #pragma clang loop unroll(disable)
    for (uint step = 0; step < D; step += BK) {
        // MLX Steel's FP32 GEMM accumulates its partial depth tile first.
        // Match its reduction order even when scores differ by one ulp.
        uint f0 = sizeof(T) == sizeof(float) && D % BK
                  ? (step == 0 ? D - D % BK : step - BK) : step;
        if constexpr (VECTOR_LOAD) {
            constexpr uint AV = BM * BK / THREADS;
            constexpr uint BV = BN * BK / THREADS;
            constexpr uint AC = BK / AV, BC = BK / BV;
            constexpr uint AR = THREADS / AC, BR = THREADS / BC;
            struct alignas(T) AVector { T values[AV]; };
            struct alignas(T) BVector { T values[BV]; };
            #pragma clang loop unroll(full)
            for (uint i = 0; i < BM; i += AR) {
                uint r = i + tid / AC, f = (tid % AC) * AV;
                if (NATIVE && r0 + r < N && f0 + f + AV <= D) {
                    *reinterpret_cast<threadgroup AVector*>(xs + r * LD + f) =
                        emeans_bf16_load_vector<AVector>(x + (r0 + r) * D + f0 + f);
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < AV; ++j)
                        xs[r * LD + f + j] = r0 + r < N && f0 + f + j < D ? T(x[(r0 + r) * D + f0 + f + j]) : T(0);
                }
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < BN; i += BR) {
                uint c = i + tid / BC, f = (tid % BC) * BV;
                if (NATIVE && c0 + c < K && f0 + f + BV <= D) {
                    *reinterpret_cast<threadgroup BVector*>(cs + c * LD + f) =
                        emeans_bf16_load_vector<BVector>(centers + (c0 + c) * D + f0 + f);
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < BV; ++j)
                        cs[c * LD + f + j] = c0 + c < K && f0 + f + j < D ? T(centers[(c0 + c) * D + f0 + f + j]) : T(0);
                }
            }
        } else {
            for (uint i = tid; i < BM * BK; i += THREADS) {
                uint r = i / BK, f = i % BK;
                xs[r * LD + f] = r0 + r < N && f0 + f < D ? T(x[(r0 + r) * D + f0 + f]) : T(0);
            }
            for (uint i = tid; i < BN * BK; i += THREADS) {
                uint c = i / BK, f = i % BK;
                cs[c * LD + f] = c0 + c < K && f0 + f < D ? T(centers[(c0 + c) * D + f0 + f]) : T(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint f = 0; f < BK; f += 8) {
            metal::simdgroup_matrix<T,8,8> aa[TM], bb[TN];
            #pragma clang loop unroll(full)
            for (uint m = 0; m < TM; ++m) {
                if constexpr (MANUAL) {
                    uint offset = ((m * WM + wm) * 8 + fm_load) * LD + f + fn_load;
                    reinterpret_cast<thread metal::vec<T,2>&>(aa[m].thread_elements()) = metal::vec<T,2>(xs[offset],xs[offset+1]);
                } else {
                    metal::simdgroup_load(aa[m], xs + (m * WM + wm) * 8 * LD + f, LD);
                }
            }
            #pragma clang loop unroll(full)
            for (uint n = 0; n < TN; ++n) {
                if constexpr (MANUAL) {
                    uint offset = ((n * WN + wn) * 8 + fn_load) * LD + f + fm_load;
                    reinterpret_cast<thread metal::vec<T,2>&>(bb[n].thread_elements()) = metal::vec<T,2>(cs[offset],cs[offset+LD]);
                } else {
                    metal::simdgroup_load(bb[n], cs + (n * WN + wn) * 8 * LD + f, LD, ulong2(0,0), true);
                }
            }
            #pragma clang loop unroll(full)
            for (uint m = 0; m < TM; ++m)
                #pragma clang loop unroll(full)
                for (uint n = 0; n < TN; ++n)
                    metal::simdgroup_multiply_accumulate(acc[m][n], aa[m], bb[n], acc[m][n]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float row_scores[WN * BM];
    threadgroup uint row_indices[WN * BM];
    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);
    const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    #pragma clang loop unroll(full)
    for (uint m = 0; m < TM; ++m) {
        float best = INFINITY;
        uint index = c0;
        #pragma clang loop unroll(full)
        for (uint n = 0; n < TN; ++n) {
            const uint col = c0 + (n * WN + wn) * 8 + fn;
            const auto values = reinterpret_cast<thread float2&>(acc[m][n].thread_elements());
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 2; ++q) {
                float value = col + q < K ? (SPHERICAL ? -values[q] : norms[col + q] - 2.0f * values[q]) : INFINITY;
                if (value < best || (value == best && col + q < index)) {
                    best = value; index = col + q;
                }
            }
        }
        // The four lanes with the same row differ only in lane bits 0 and 3.
        float other = metal::simd_shuffle_xor(best, 1);
        uint other_index = metal::simd_shuffle_xor(index, 1);
        if (other < best || (other == best && other_index < index)) {
            best = other; index = other_index;
        }
        other = metal::simd_shuffle_xor(best, 8);
        other_index = metal::simd_shuffle_xor(index, 8);
        if (other < best || (other == best && other_index < index)) {
            best = other; index = other_index;
        }
        if (fn == 0) {
            uint row = (m * WM + wm) * 8 + fm;
            row_scores[wn * BM + row] = best;
            row_indices[wn * BM + row] = index;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < BM; row += THREADS) {
        if (r0 + row >= N) continue;
        float best = INFINITY;
        uint index = c0;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < WN; ++j) {
            float value = row_scores[j * BM + row];
            uint candidate = row_indices[j * BM + row];
            if (value < best || (value == best && candidate < index)) {
                best = value; index = candidate;
            }
        }
        uint output_index = COALESCED ? (c0 / BN) * N + r0 + row
                                     : (r0 + row) * PARTS + c0 / BN;
        partial_scores[output_index] = best;
        partial_indices[output_index] = index;
    }
"""

@lru_cache(None)
def _tile_kernel():
    return mx.fast.metal_kernel(
        name="emeans_bf16_tiled", input_names=["x", "centers", "norms"],
        output_names=["partial_scores", "partial_indices"], source=_TILE_SOURCE,
        header=r"""
        #include <metal_simdgroup_matrix>
        template <typename V, typename U>
        V emeans_bf16_load_vector(const device U* pointer) {
            return *reinterpret_cast<const device V*>(pointer);
        }
        // MLX binds one-element arrays in the constant address space.
        template <typename V, typename U>
        V emeans_bf16_load_vector(const constant U* pointer) {
            return *reinterpret_cast<const constant V*>(pointer);
        }
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(None)
def _norm_kernel():
    return mx.fast.metal_kernel(
        name="emeans_bf16_norms", input_names=["centers"], output_names=["norms"],
        source=r"""
        uint row = thread_position_in_grid.x / 32;
        uint lane = thread_index_in_simdgroup;
        if (row >= K) return;
        float sum = 0;
        for (uint f = lane; f < D; f += 32) {
            float v = float(centers[row * D + f]);
            sum += v * v;
        }
        sum = metal::simd_sum(sum);
        if (lane == 0) norms[row] = sum;
        """,
    )


@lru_cache(None)
def _merge_kernel():
    return mx.fast.metal_kernel(
        name="emeans_bf16_merge", input_names=["scores", "indices"],
        output_names=["labels", "best_scores"], source=r"""
        uint row = thread_position_in_grid.x / 32;
        uint lane = thread_index_in_simdgroup;
        if (row >= N) return;
        float best = INFINITY;
        uint index = 0;
        for (uint part = lane; part < PARTS; part += 32) {
            float score = scores[row * PARTS + part];
            uint candidate = indices[row * PARTS + part];
            if (score < best || (score == best && candidate < index)) {
                best = score; index = candidate;
            }
        }
        float value = metal::simd_min(best);
        uint winner = metal::simd_min(best == value ? index : 0xffffffff);
        if (lane == 0) {labels[row] = winner; best_scores[row] = value;}
        """,
    )


@lru_cache(None)
def _coalesced_merge_kernel():
    """Merge part-major tiles with contiguous row reads in every SIMD group."""
    return mx.fast.metal_kernel(
        name="emeans_bf16_merge_coalesced", input_names=["scores", "indices"],
        output_names=["labels", "best_scores"], source=r"""
        uint lane = thread_index_in_simdgroup;
        uint warp = simdgroup_index_in_threadgroup;
        uint tid = thread_index_in_threadgroup;
        uint row = threadgroup_position_in_grid.x * 32 + lane;
        float best = INFINITY;
        uint index = 0;
        for (uint part = warp; part < PARTS; part += WARPS) {
            float score = row < N ? scores[part * N + row] : INFINITY;
            uint candidate = row < N ? indices[part * N + row] : 0;
            if (score < best || (score == best && candidate < index)) {
                best = score; index = candidate;
            }
        }
        threadgroup float row_scores[WARPS * 32];
        threadgroup uint row_indices[WARPS * 32];
        row_scores[tid] = best;
        row_indices[tid] = index;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (warp == 0 && row < N) {
            for (uint w = 1; w < WARPS; ++w) {
                float value = row_scores[w * 32 + lane];
                uint candidate = row_indices[w * 32 + lane];
                if (value < best || (value == best && candidate < index)) {
                    best = value; index = candidate;
                }
            }
            labels[row] = index;
            best_scores[row] = best;
        }
        """, compile_options={"math_mode": "safe"},
    )


def _check_centers(centers):
    if centers.ndim != 2 or min(centers.shape) < 1:
        raise ValueError("centers must be a nonempty matrix")
    k, d = centers.shape
    if k * d > _UINT32_MAX or k * 32 > _UINT32_MAX:
        raise ValueError("centers exceed the Metal uint32 indexing limit")
    return k, d


def prepare_centers(centers):
    """Return BF16 centers and FP32 squared norms of the same BF16 values.

    Inputs must be finite and small enough for FP32 norm/dot accumulation.
    The public estimator performs this magnitude check before kernel dispatch.
    """
    k, d = _check_centers(centers)
    represented = centers.astype(mx.bfloat16)
    norms = _norm_kernel()(
        inputs=[represented], template=[("K", k), ("D", d)],
        grid=(k * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(k,)], output_dtypes=[mx.float32], stream=mx.gpu,
    )[0]
    return represented, norms


def streaming_tile(x, centers, norms, *, spherical, bm, bk, pad, coalesced):
    """Compute labels and winning scores for one workspace-bounded FP32 or BF16 tile."""
    rows, d = x.shape
    k = centers.shape[0]
    bn = 64
    wm = wn = 2
    threads = 32 * wm * wn
    parts = (k + bn - 1) // bn
    partial_scores, partial_indices = _tile_kernel()(
        inputs=[x, centers, norms],
        template=[("T", x.dtype), ("N", rows), ("K", k), ("D", d),
                  ("BM", bm), ("BN", bn), ("BK", bk), ("WM", wm),
                  ("WN", wn), ("PAD", pad), ("PARTS", parts),
                  ("MANUAL", True), ("VECTOR_LOAD", True), ("NATIVE", True),
                  ("SPHERICAL", bool(spherical)), ("COALESCED", coalesced)],
        grid=(((rows + bm - 1) // bm) * threads, parts, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(parts, rows), (parts, rows)] if coalesced else [(rows, parts), (rows, parts)],
        output_dtypes=[mx.float32, mx.uint32], stream=mx.gpu,
    )
    merge = _coalesced_merge_kernel() if coalesced else _merge_kernel()
    return tuple(merge(
        inputs=[partial_scores, partial_indices],
        template=[("N", rows), ("PARTS", parts)] + ([("WARPS", 8)] if coalesced else []),
        grid=(((rows + 31) // 32) * 256 if coalesced else rows * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows,), (rows,)],
        output_dtypes=[mx.uint32, mx.float32], stream=mx.gpu,
    ))


def assign_bf16(x, centers, *, norms=None, batch_size=None, spherical=False):
    """Return uint32 labels and FP32 winning assignment scores.

    Euclidean scores are ``||BF16(center)||² - 2*dot(BF16(x), BF16(center))``;
    they omit the row-constant input norm and are not squared distances.
    Spherical scores are ``-dot(BF16(x), BF16(center))``. Ties select the
    smallest centroid index. This low-level routine assumes finite operands
    whose magnitudes are safe for FP32 arithmetic.

    Supplying BF16 ``centers`` and their FP32 ``norms`` avoids preparation.
    ``batch_size`` is an optional row limit; all calls also respect a 256 MiB
    partial-score/index workspace limit and uint32 Metal indexing limits.
    """
    k, d = _check_centers(centers)
    if x.ndim != 2 or x.shape[1] != d or x.shape[0] < 1:
        raise ValueError("x and centers must be nonempty compatible matrices")
    n = x.shape[0]
    if norms is None:
        centers, norms = prepare_centers(centers)
    elif centers.dtype != mx.bfloat16 or norms.shape != (k,) or norms.dtype != mx.float32:
        raise ValueError("supplied norms require BF16 centers and a matching FP32 vector")
    represented_x = x.astype(mx.bfloat16)
    # Tuned on M4 Max; raw dispatch sweeps are retained with benchmark results.
    # Paired M4 Max measurements support the wider tile and coalesced partial
    # layout for these common aligned dimensions. Other shapes retain their
    # previous dispatch; larger dimensions showed no consistent layout gain.
    coalesced = d in (96, 128) and k >= 1024 and n >= 32
    bm, bk = (32, 16) if d <= 128 and not coalesced else (64, 32)
    parts = (k + 63) // 64
    block = min(n, _UINT32_MAX // d, _UINT32_MAX // parts,
                _UINT32_MAX // 32, max(1, _SCRATCH_BYTES // (8 * parts)))
    if batch_size is not None:
        try:
            requested = operator.index(batch_size)
        except TypeError as exc:
            raise ValueError("batch_size must be a positive integer") from exc
        if requested < 1 or isinstance(batch_size, bool):
            raise ValueError("batch_size must be a positive integer")
        block = min(block, requested)
    if block < 1:
        raise ValueError("matrix exceeds the Metal uint32 indexing limit")
    labels, scores = [], []
    for start in range(0, n, block):
        xb = represented_x[start:start + block]
        rows = xb.shape[0]
        batch_labels, batch_scores = streaming_tile(
            xb, centers, norms, spherical=spherical, bm=bm, bk=bk, pad=8, coalesced=coalesced,
        )
        # Materialize each chunk before dispatching another to bound workspace.
        # A single training batch stays lazy for fusion into the fit graph.
        if rows < n:
            mx.eval(batch_labels, batch_scores)
        labels.append(batch_labels)
        scores.append(batch_scores)
    if len(labels) == 1:
        return labels[0], scores[0]
    return mx.concatenate(labels), mx.concatenate(scores)
