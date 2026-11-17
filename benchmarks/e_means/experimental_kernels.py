"""Experimental BF16 Metal assignment and statistics kernels, not runtime code.

BF16 operands are represented explicitly; centroid norms and MMA accumulators
are FP32. No half-precision conversion or implicit centering occurs. Statistics
consume the supplied values (pass BF16 rows for upstream's resident-BF16 path).

The register-coordinate mapping follows Apple's installed MLX Steel
BaseMMAFrag<float,8,8>::get_coord (MIT): an 8x8 SIMD fragment owns two
adjacent columns per lane. Other kernel code is specific to this experiment.
The derived mapping and vector-load strategy originate in MLX 0.32.4's
mlx/backend/metal/kernels/steel/gemm/{mma,loader}.h, Copyright © 2024 Apple Inc.
The Apple MIT notice is reproduced in THIRD_PARTY_NOTICES.md, section 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx


@dataclass(frozen=True)
class BF16Config:
    rows: int = 32
    cols: int = 128
    depth: int = 32
    warps_m: int = 2
    warps_n: int = 2
    native_bf16: bool = True
    register_argmin: bool = True
    padding: int = 8
    manual_fragments: bool = True
    vector_load: bool = True
    swizzle_rows: int = 4
    math_mode: str = "safe"

    def validate(self):
        if min(self.rows, self.cols, self.depth, self.warps_m, self.warps_n) < 1:
            raise ValueError("positive tile sizes and warp counts are required")
        if self.rows % (8 * self.warps_m) or self.cols % (8 * self.warps_n):
            raise ValueError("matrix tiles must divide evenly over 8x8 SIMD fragments")
        if self.depth % 8 or min(self.rows, self.cols, self.depth, self.warps_m, self.warps_n) < 1:
            raise ValueError("positive tiles and a depth divisible by eight are required")
        threads = 32 * self.warps_m * self.warps_n
        if threads > 1024:
            raise ValueError("too many threads")
        element_bytes = 2 if self.native_bf16 else 4
        shared = (self.rows + self.cols) * (self.depth + self.padding) * element_bytes
        shared += self.rows * self.warps_n * 8 if self.register_argmin else self.rows * self.cols * 4
        if shared > 32768:
            raise ValueError(f"tile needs {shared} bytes of threadgroup memory")
        return threads


_PREFIX = r"""
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
    uint r0, c0;
    if constexpr (SWIZZLE > 0) {
        r0 = (threadgroup_position_in_grid.y * SWIZZLE + threadgroup_position_in_grid.x % SWIZZLE) * BM;
        c0 = (threadgroup_position_in_grid.x / SWIZZLE) * BN;
    } else {
        r0 = threadgroup_position_in_grid.x * BM;
        c0 = threadgroup_position_in_grid.y * BN;
    }
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
    for (uint f0 = 0; f0 < D; f0 += BK) {
        if constexpr (VECTOR_LOAD) {
            constexpr uint AV = BM * BK / THREADS;
            constexpr uint BV = BN * BK / THREADS;
            constexpr uint AC = BK / AV, BC = BK / BV;
            constexpr uint AR = THREADS / AC, BR = THREADS / BC;
            struct alignas(2) AVector { uint8_t values[2 * AV]; };
            struct alignas(2) BVector { uint8_t values[2 * BV]; };
            #pragma clang loop unroll(full)
            for (uint i = 0; i < BM; i += AR) {
                uint r = i + tid / AC, f = (tid % AC) * AV;
                if (NATIVE && r0 + r < N && f0 + f + AV <= D) {
                    *reinterpret_cast<threadgroup AVector*>(xs + r * LD + f) =
                        *reinterpret_cast<const device AVector*>(x + (r0 + r) * D + f0 + f);
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
                        *reinterpret_cast<const device BVector*>(centers + (c0 + c) * D + f0 + f);
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
"""

_REGISTER_EPILOGUE = r"""
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
                float value = col + q < K ? norms[col + q] - 2.0f * values[q] : INFINITY;
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
        partial_scores[(r0 + row) * PARTS + c0 / BN] = best;
        partial_indices[(r0 + row) * PARTS + c0 / BN] = index;
    }
"""

_SPILL_EPILOGUE = r"""
    threadgroup float dots[BM * BN];
    #pragma clang loop unroll(full)
    for (uint m = 0; m < TM; ++m)
        #pragma clang loop unroll(full)
        for (uint n = 0; n < TN; ++n)
            metal::simdgroup_store(acc[m][n], dots + (m * WM + wm) * 8 * BN + (n * WN + wn) * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = sg; row < BM; row += WM * WN) {
        if (r0 + row >= N) continue;
        float best = INFINITY;
        uint index = c0;
        for (uint col = lane; col < BN; col += 32) {
            float value = c0 + col < K ? norms[c0 + col] - 2.0f * dots[row * BN + col] : INFINITY;
            if (value < best || (value == best && c0 + col < index)) {
                best = value; index = c0 + col;
            }
        }
        float value = metal::simd_min(best);
        uint winner = metal::simd_min(best == value ? index : 0xffffffff);
        if (lane == 0) {
            partial_scores[(r0 + row) * PARTS + c0 / BN] = value;
            partial_indices[(r0 + row) * PARTS + c0 / BN] = winner;
        }
    }
"""


@lru_cache(None)
def _assignment_kernel(config):
    config.validate()
    return mx.fast.metal_kernel(
        name="emeans_exp_bf16_tiled", input_names=["x", "centers", "norms"],
        output_names=["partial_scores", "partial_indices"],
        source=_PREFIX + (_REGISTER_EPILOGUE if config.register_argmin else _SPILL_EPILOGUE),
        header="#include <metal_simdgroup_matrix>\n",
        compile_options={"math_mode": config.math_mode},
    )


@lru_cache(None)
def _aux_kernel(name):
    if name == "norms":
        return mx.fast.metal_kernel(
            name="emeans_exp_bf16_norms", input_names=["centers"], output_names=["norms"],
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
    if name == "merge":
        return mx.fast.metal_kernel(
            name="emeans_exp_bf16_merge", input_names=["scores", "indices"],
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
    if name == "atomic_stats":
        return mx.fast.metal_kernel(
            name="emeans_exp_atomic_stats", input_names=["x", "labels"],
            output_names=["sums", "counts"], atomic_outputs=True, source=r"""
            uint task = thread_position_in_grid.x;
            constexpr uint FEATURES = (D + VEC - 1) / VEC;
            uint row = task / FEATURES, first = (task % FEATURES) * VEC;
            if (row >= N) return;
            uint cluster = labels[row];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < VEC; ++j) {
                if (first + j < D)
                    atomic_fetch_add_explicit(&sums[cluster * D + first + j], float(x[row * D + first + j]), memory_order_relaxed);
            }
            if (first == 0) atomic_fetch_add_explicit(&counts[cluster], 1.0f, memory_order_relaxed);
            """,
        )
    if name == "linked_bins":
        return mx.fast.metal_kernel(
            name="emeans_exp_linked_bins", input_names=["labels"],
            output_names=["heads", "next"], atomic_outputs=True, source=r"""
            uint row = thread_position_in_grid.x;
            if (row >= N) return;
            uint previous = atomic_exchange_explicit(&heads[labels[row]], row + 1, memory_order_relaxed);
            atomic_store_explicit(&next[row], previous, memory_order_relaxed);
            """,
        )
    if name == "linked_reduce":
        return mx.fast.metal_kernel(
            name="emeans_exp_linked_reduce", input_names=["x", "heads", "next"],
            output_names=["sums", "counts"], source=r"""
            uint i = thread_position_in_grid.x;
            if (i >= K * D) return;
            uint cluster = i / D, feature = i % D;
            uint cursor = heads[cluster], count = 0;
            float sum = 0;
            while (cursor != 0) {
                uint row = cursor - 1;
                sum += float(x[row * D + feature]);
                cursor = next[row];
                ++count;
            }
            sums[i] = sum;
            if (feature == 0) counts[cluster] = float(count);
            """,
        )
    raise ValueError(name)


def prepare_centers(centers):
    """Return BF16 mirror and FP32 norms of that same represented mirror."""
    represented = centers.astype(mx.bfloat16)
    k, d = centers.shape
    norms = _aux_kernel("norms")(
        inputs=[represented], template=[("K", k), ("D", d)],
        grid=(k * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(k,)], output_dtypes=[mx.float32], stream=mx.gpu,
    )[0]
    return represented, norms


def assign_bf16(x, centers, *, config=None, norms=None):
    """Return labels and FP32 ``norm(center_BF16)^2 - 2*dot_BF16`` scores.

    Scores deliberately omit the row-constant input norm, matching the
    reference assignment objective. Inputs are BF16 rounded without centering.
    Passing a prepared BF16 center table and norms amortizes their preparation.
    """
    config = config or BF16Config()
    threads = config.validate()
    n, d = x.shape
    k = centers.shape[0]
    if k < 1 or centers.shape[1] != d or n < 1 or d < 1:
        raise ValueError("nonempty compatible matrices are required")
    if norms is None:
        centers, norms = prepare_centers(centers)
    elif centers.dtype != mx.bfloat16:
        raise ValueError("supplied norms require explicitly BF16 represented centers")
    represented_x = x.astype(mx.bfloat16)
    parts = (k + config.cols - 1) // config.cols
    row_tiles = (n + config.rows - 1) // config.rows
    grid = ((parts * config.swizzle_rows * threads,
             (row_tiles + config.swizzle_rows - 1) // config.swizzle_rows, 1)
            if config.swizzle_rows else (row_tiles * threads, parts, 1))
    scores, indices = _assignment_kernel(config)(
        inputs=[represented_x, centers, norms],
        template=[("T", mx.bfloat16 if config.native_bf16 else mx.float32),
                  ("N", n), ("K", k), ("D", d), ("BM", config.rows),
                  ("BN", config.cols), ("BK", config.depth), ("WM", config.warps_m),
                  ("WN", config.warps_n), ("PAD", config.padding), ("PARTS", parts),
                  ("MANUAL", config.manual_fragments), ("VECTOR_LOAD", config.vector_load),
                  ("NATIVE", config.native_bf16), ("SWIZZLE", config.swizzle_rows)],
        grid=grid,
        threadgroup=(threads, 1, 1), output_shapes=[(n, parts), (n, parts)],
        output_dtypes=[mx.float32, mx.uint32], stream=mx.gpu,
    )
    return tuple(_aux_kernel("merge")(
        inputs=[scores, indices], template=[("N", n), ("PARTS", parts)],
        grid=(n * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(n,), (n,)], output_dtypes=[mx.uint32, mx.float32], stream=mx.gpu,
    ))


def stats_atomic(x, labels, k, *, vector_width=4):
    """FP32 atomic statistics with several contiguous features per thread."""
    n, d = x.shape
    features = (d + vector_width - 1) // vector_width
    return tuple(_aux_kernel("atomic_stats")(
        inputs=[x, labels], template=[("N", n), ("D", d), ("VEC", vector_width)],
        grid=(n * features, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(k, d), (k,)], output_dtypes=[mx.float32, mx.float32],
        init_value=0, stream=mx.gpu,
    ))


def stats_linked(x, labels, k):
    """Build atomic integer lists, then reduce each center without float atomics."""
    n, d = x.shape
    heads, next_rows = _aux_kernel("linked_bins")(
        inputs=[labels], template=[("N", n)], grid=(n, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(k,), (n,)], output_dtypes=[mx.uint32, mx.uint32],
        init_value=0, stream=mx.gpu,
    )
    return tuple(_aux_kernel("linked_reduce")(
        inputs=[x, heads, next_rows], template=[("K", k), ("D", d)],
        grid=(k * d, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(k, d), (k,)], output_dtypes=[mx.float32, mx.float32], stream=mx.gpu,
    ))
