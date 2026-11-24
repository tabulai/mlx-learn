"""Metal primitives for e-means on Apple Silicon.

Assignment uses MLX's Metal GEMM followed by SIMD argmin, or streams score
tiles through SIMD-group registers for large FP32 assignments on M4 Max.
All accumulation and centroid state are float32. Float32 GEMMs keep binary32
products on M5-generation GPUs, where MLX would use TF32 operands (see
``mlxlearn_device._binary32``).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from mlxlearn_device._binary32 import matmul

from .e_means_bf16 import (
    _merge_second_kernel,
    _tile_kernel,
    assign_bf16,
    prepare_centers,
    streaming_tile,
)

_UINT32_MAX = 2**32 - 1


@lru_cache(None)
def _kernel(name):
    if name == "gather_bf16":
        return mx.fast.metal_kernel(
            name="emeans_gather_cast_bf16", input_names=["data", "order"],
            output_names=["batch"], source=r"""
            uint i = thread_position_in_grid.x * 4;
            if (i >= B * D) return;
            if constexpr (D % 4 == 0) {
                uint row = i / D, feature = i % D;
                metal::float4 values = emeans_gather_load4(data + ulong(order[row]) * D + feature);
                *reinterpret_cast<device metal::vec<bfloat16_t,4>*>(batch + i) =
                    metal::vec<bfloat16_t,4>(values);
            } else {
                for (uint j = 0; j < 4 && i + j < B * D; ++j) {
                    uint row = (i + j) / D, feature = (i + j) % D;
                    batch[i + j] = bfloat16_t(data[ulong(order[row]) * D + feature]);
                }
            }
            """, header=r"""
            metal::float4 emeans_gather_load4(const device float* pointer) {
                return *reinterpret_cast<const device metal::float4*>(pointer);
            }
            metal::float4 emeans_gather_load4(const constant float* pointer) {
                return *reinterpret_cast<const constant metal::float4*>(pointer);
            }
            """)
    if name == "argmin":
        return mx.fast.metal_kernel(
            name="emeans_argmin", input_names=["scores", "norms"],
            output_names=["labels"], source=r"""
            const uint row = thread_position_in_grid.x / 32;
            const uint lane = thread_index_in_simdgroup;
            if (row >= N) return;
            float best = INFINITY;
            // Keep a valid index even if unsupported inputs overflow every score.
            // The public estimator rejects unsafe magnitudes before dispatch.
            uint index = 0;
            for (uint j = lane; j < K; j += 32) {
                float v = SPHERICAL ? -float(scores[row * K + j])
                                    : norms[j] - 2.0f * float(scores[row * K + j]);
                if (v < best || (v == best && j < index)) {best = v; index = j;}
            }
            float v = metal::simd_min(best);
            uint i = metal::simd_min(best == v ? index : 0xffffffff);
            if (lane == 0) labels[row] = i;
            """)
    if name == "argmin_second":
        # ``argmin`` that also returns the winning score and the smallest
        # score among the other centroids (equal to the winner's on a tie).
        return mx.fast.metal_kernel(
            name="emeans_argmin_second", input_names=["scores", "norms"],
            output_names=["labels", "best_scores", "second_scores"], source=r"""
            const uint row = thread_position_in_grid.x / 32;
            const uint lane = thread_index_in_simdgroup;
            if (row >= N) return;
            float best = INFINITY, second = INFINITY;
            uint index = 0;
            for (uint j = lane; j < K; j += 32) {
                float v = SPHERICAL ? -float(scores[row * K + j])
                                    : norms[j] - 2.0f * float(scores[row * K + j]);
                bool wins = v < best || (v == best && j < index);
                second = metal::min(second, wins ? best : v);
                if (wins) {best = v; index = j;}
            }
            float v = metal::simd_min(best);
            uint i = metal::simd_min(best == v ? index : 0xffffffff);
            // The winner's lane offers its second score, every other lane its best.
            float other = metal::simd_min(best == v && index == i ? second : best);
            if (lane == 0) {labels[row] = i; best_scores[row] = v; second_scores[row] = other;}
            """)
    if name == "distances":
        return mx.fast.metal_kernel(
            name="emeans_distances", input_names=["x", "centers", "labels"],
            output_names=["distances"], source=r"""
            const uint row = thread_position_in_grid.x / 32;
            const uint lane = thread_index_in_simdgroup;
            if (row >= N) return;
            float sum = 0;
            for (uint j = lane; j < D; j += 32) {
                float a = x[row * D + j], b = centers[labels[row] * D + j];
                float delta = a - b;
                sum += SPHERICAL ? a * b : delta * delta;
            }
            sum = metal::simd_sum(sum);
            if (lane == 0) distances[row] = SPHERICAL ? 2.0f - 2.0f * sum : sum;
            """)
    if name == "stats":
        return mx.fast.metal_kernel(
            name="emeans_stats", input_names=["x", "labels"],
            output_names=["sums", "counts"], atomic_outputs=True, source=r"""
            uint i = thread_position_in_grid.x;
            if (i >= N * D) return;
            uint row = i / D;
            uint feature = i % D;
            uint cluster = labels[row];
            atomic_fetch_add_explicit(&sums[cluster * D + feature], float(x[i]), memory_order_relaxed);
            if (feature == 0)
                atomic_fetch_add_explicit(&counts[cluster], 1.0f, memory_order_relaxed);
            """)
    if name == "ema":
        return mx.fast.metal_kernel(
            name="emeans_ema", input_names=["centers", "counts", "sums", "hits", "alpha"],
            output_names=["new_centers", "new_counts"], source=r"""
            uint i = thread_position_in_grid.x;
            if (i >= K * D) return;
            uint c = i / D;
            float old_weight = (INDEPENDENT_DECAY ? alpha[1] : (1.0f - alpha[0])) * counts[c];
            float count = old_weight + alpha[0] * hits[c];
            // alpha=1 with no hits must preserve the position, avoiding 0/0.
            if (INDEPENDENT_DECAY && count > 0 && hits[c] > 0) {
                float mean = sums[i] / hits[c];
                float weight = (alpha[0] * hits[c]) / count;
                new_centers[i] = weight == 1.0f ? mean :
                    centers[i] + weight * (mean - centers[i]);
            } else {
                new_centers[i] = count > 0 && hits[c] > 0 ?
                    (old_weight * centers[i] + alpha[0] * sums[i]) / count : centers[i];
            }
            if (i % D == 0) new_counts[c] = count;
            """)
    if name == "removal_stats":
        # One SIMD group per row. With y = x - c for the row's centroid c and
        # p = <y, axis>, add p * y to the centroid's axis sums and the row's
        # margin to its removal cost. A margin that is not finite (a single
        # centroid, or overflowing operands) adds nothing; the bit test keeps
        # that rule under fast-math compilation.
        return mx.fast.metal_kernel(
            name="emeans_removal_stats",
            input_names=["x", "labels", "margins", "centers", "axes"],
            output_names=["axis_sums", "costs"], atomic_outputs=True, source=r"""
            const uint row = thread_position_in_grid.x / 32;
            const uint lane = thread_index_in_simdgroup;
            if (row >= N) return;
            const uint base = uint(labels[row]) * D;
            float projection = 0;
            for (uint f = lane; f < D; f += 32) {
                float y = float(x[row * D + f]) - centers[base + f];
                projection += y * axes[base + f];
            }
            projection = metal::simd_sum(projection);
            for (uint f = lane; f < D; f += 32) {
                float y = float(x[row * D + f]) - centers[base + f];
                atomic_fetch_add_explicit(&axis_sums[base + f], projection * y, memory_order_relaxed);
            }
            if (lane == 0) {
                const float margin = margins[row];
                if ((as_type<uint>(margin) & 0x7f800000u) != 0x7f800000u)
                    atomic_fetch_add_explicit(&costs[base / D], margin, memory_order_relaxed);
            }
            """)
    if name == "removal_update":
        # One SIMD group per centroid: the EMA of its axis sums and removal
        # cost, the length of the new axis sum and its direction. The length
        # is formed from components scaled by the largest one, so that it
        # cannot overflow where the length itself is representable. A sum
        # whose largest component is below the normal float32 range counts as
        # zero and keeps the previous direction: dividing by it could overflow
        # once fast-math compilation turns the division into a reciprocal.
        return mx.fast.metal_kernel(
            name="emeans_removal_update",
            input_names=["long_axis", "removal", "axis_sums", "costs", "axes", "rate"],
            output_names=["new_long_axis", "new_removal", "new_axes", "lengths"], source=r"""
            const uint c = thread_position_in_grid.x / 32;
            const uint lane = thread_index_in_simdgroup;
            if (c >= K) return;
            float peak = 0;
            for (uint f = lane; f < D; f += 32) {
                float u = rate[1] * long_axis[c * D + f] + rate[0] * axis_sums[c * D + f];
                new_long_axis[c * D + f] = u;
                peak = metal::max(peak, metal::abs(u));
            }
            peak = metal::simd_max(peak);
            const bool usable = peak >= 1.17549435e-38f;
            float squares = 0;
            if (usable) {
                for (uint f = lane; f < D; f += 32) {
                    float u = new_long_axis[c * D + f] / peak;
                    squares += u * u;
                }
            }
            // At least one scaled component is one, so the scale is at least one.
            const float scale = usable ? metal::sqrt(metal::simd_sum(squares)) : 1.0f;
            const float length = usable ? peak * scale : 0.0f;
            for (uint f = lane; f < D; f += 32)
                new_axes[c * D + f] = usable ? (new_long_axis[c * D + f] / peak) / scale
                                             : axes[c * D + f];
            if (lane == 0) {
                lengths[c] = length;
                new_removal[c] = rate[1] * removal[c] + rate[0] * costs[c];
            }
            """)
    raise ValueError(name)


_FUSED = r"""
    // One threadgroup evaluates a 32 by 64 pair tile, with eight SIMD groups.
    threadgroup T xt[32 * 32];
    threadgroup T ct[32 * 64];
    threadgroup float dots[32 * 64];
    uint tid = thread_index_in_threadgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint r0 = threadgroup_position_in_grid.x * 32;
    uint c0 = threadgroup_position_in_grid.y * 64;
    uint rr = (sg / 4) * 16;
    uint cc = (sg % 4) * 16;
    metal::simdgroup_float8x8 a00(0), a01(0), a10(0), a11(0);
    for (uint f0 = 0; f0 < D; f0 += 32) {
        for (uint i = tid; i < 32 * 32; i += 256) {
            uint row = r0 + i / 32, f = f0 + i % 32;
            xt[i] = row < N && f < D ? T(x[row * D + f]) : T(0);
        }
        for (uint i = tid; i < 32 * 64; i += 256) {
            uint col = c0 + i % 64, f = f0 + i / 64;
            ct[i] = col < K && f < D ? T(centers[col * D + f]) : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint f = 0; f < 32; f += 8) {
            metal::simdgroup_matrix<T, 8, 8> x0, x1, c0m, c1m;
            metal::simdgroup_load(x0, xt + rr * 32 + f, 32);
            metal::simdgroup_load(x1, xt + (rr + 8) * 32 + f, 32);
            metal::simdgroup_load(c0m, ct + f * 64 + cc, 64);
            metal::simdgroup_load(c1m, ct + f * 64 + cc + 8, 64);
            metal::simdgroup_multiply_accumulate(a00, x0, c0m, a00);
            metal::simdgroup_multiply_accumulate(a01, x0, c1m, a01);
            metal::simdgroup_multiply_accumulate(a10, x1, c0m, a10);
            metal::simdgroup_multiply_accumulate(a11, x1, c1m, a11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    metal::simdgroup_store(a00, dots + rr * 64 + cc, 64);
    metal::simdgroup_store(a01, dots + rr * 64 + cc + 8, 64);
    metal::simdgroup_store(a10, dots + (rr + 8) * 64 + cc, 64);
    metal::simdgroup_store(a11, dots + (rr + 8) * 64 + cc + 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Eight SIMD groups cooperatively reduce four rows each.
    uint lane = thread_index_in_simdgroup;
    for (uint r = sg; r < 32; r += 8) {
        uint row = r0 + r;
        if (row >= N) continue;
        float best = INFINITY;
        uint index = c0;
        for (uint c = lane; c < 64; c += 32) {
            uint col = c0 + c;
            if (col < K) {
                float v = SPHERICAL ? -dots[r * 64 + c]
                                    : norms[col] - 2.0f * dots[r * 64 + c];
                if (v < best || (v == best && col < index)) {best = v; index = col;}
            }
        }
        float v = metal::simd_min(best);
        uint idx = metal::simd_min(best == v ? index : 0xffffffff);
        if (lane == 0) {
            uint address = row * PARTS + threadgroup_position_in_grid.y;
            partial_scores[address] = v;
            partial_labels[address] = idx;
        }
    }
"""


# The reduction of ``_FUSED`` starts at this line; the variant below replaces
# it and also keeps each row's second-smallest score of the tile.
_FUSED_REDUCTION = "    // Eight SIMD groups cooperatively reduce four rows each.\n"
_FUSED_SECOND = _FUSED[:_FUSED.index(_FUSED_REDUCTION)] + r"""    // Eight SIMD groups cooperatively reduce four rows each.
    uint lane = thread_index_in_simdgroup;
    for (uint r = sg; r < 32; r += 8) {
        uint row = r0 + r;
        if (row >= N) continue;
        float best = INFINITY, second = INFINITY;
        uint index = c0;
        for (uint c = lane; c < 64; c += 32) {
            uint col = c0 + c;
            if (col < K) {
                float v = SPHERICAL ? -dots[r * 64 + c]
                                    : norms[col] - 2.0f * dots[r * 64 + c];
                bool wins = v < best || (v == best && col < index);
                second = metal::min(second, wins ? best : v);
                if (wins) {best = v; index = col;}
            }
        }
        float v = metal::simd_min(best);
        uint idx = metal::simd_min(best == v ? index : 0xffffffff);
        // The winner's lane offers its second score, every other lane its best.
        float other = metal::simd_min(best == v && index == idx ? second : best);
        if (lane == 0) {
            uint address = row * PARTS + threadgroup_position_in_grid.y;
            partial_scores[address] = v;
            partial_seconds[address] = other;
            partial_labels[address] = idx;
        }
    }
"""


@lru_cache(None)
def _fused_second_kernel():
    return mx.fast.metal_kernel(
        name="emeans_tiled_assign_second", input_names=["x", "centers", "norms"],
        output_names=["partial_scores", "partial_seconds", "partial_labels"],
        source=_FUSED_SECOND, header="#include <metal_simdgroup_matrix>\n")


@lru_cache(None)
def _fused_kernels():
    tiled = mx.fast.metal_kernel(
        name="emeans_tiled_assign", input_names=["x", "centers", "norms"],
        output_names=["partial_scores", "partial_labels"], source=_FUSED,
        header="#include <metal_simdgroup_matrix>\n")
    merge = mx.fast.metal_kernel(
        name="emeans_merge", input_names=["scores", "indices"],
        output_names=["labels"], source=r"""
        uint row = thread_position_in_grid.x / 32;
        uint lane = thread_index_in_simdgroup;
        if (row >= N) return;
        float best = INFINITY;
        uint index = 0;
        for (uint p = lane; p < PARTS; p += 32) {
            float v = scores[row * PARTS + p];
            uint i = indices[row * PARTS + p];
            if (v < best || (v == best && i < index)) {best = v; index = i;}
        }
        float v = metal::simd_min(best);
        uint i = metal::simd_min(best == v ? index : 0xffffffff);
        if (lane == 0) labels[row] = i;
        """)
    return tiled, merge


def selected_distances(x, centers, labels, *, spherical=False):
    """Compute squared residuals directly, avoiding Gram cancellation in MSE."""
    n, d = x.shape
    return _kernel("distances")(
        inputs=[x, centers, labels], template=[("N", n), ("D", d), ("SPHERICAL", spherical)],
        grid=(n * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(n,)], output_dtypes=[mx.float32], stream=mx.gpu)[0]


@lru_cache(None)
def _scratch_budget():
    """Use larger GEMM tiles on unified-memory Macs with sufficient memory."""
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    recommended = info["max_recommended_working_set_size"]
    return min(1024 << 20, max(32 << 20, recommended // 64))


@lru_cache(maxsize=1)
def _streaming_fp32_device():
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    if info.get("device_name") != "Apple M4 Max":
        return False
    try:
        _tile_kernel()
    except TypeError:
        # Earlier supported MLX releases lack the safe-math compile option.
        return False
    return True


def assign(x, centers, *, batch_size=None, spherical=False, method="auto", precision="float32",
           second=False):
    """Assign rows to centers with bounded scratch memory on the GPU.

    ``float16`` only reduces assignment operands; state, accumulation, scores,
    and residual scoring remain float32. This is an Apple-specific precision
    option, not the paper's BF16/FP8 representations. It is approximate and is
    never selected implicitly. Reduced-precision assignment always uses the
    fused kernel: a half-precision GEMM output would round/overflow the scores.
    ``bfloat16`` rounds both assignment operands, accumulates scores in FP32,
    and reports residuals between the represented operands. Its ``auto`` and
    ``fused`` paths use native BF16 matrix tiles with register argmin; ``gemm``
    widens the BF16 values before FP32 GEMM to avoid BF16 score rounding.
    Float32 ``auto`` streams large M4 Max score tiles directly into argmin and
    uses tuned GEMM for the other shapes. ``fused`` exposes the older matrix tiles.

    ``second`` adds a third result: each row's margin, the distance to its
    second-nearest centroid minus the distance to its nearest one, in the
    units of the returned distances. It comes from the assignment scores of
    the same search, so it is zero on a tie and infinite when there is only
    one centroid. Labels and distances are those of the plain call.
    """
    if method not in {"auto", "gemm", "fused"}:
        raise ValueError("method must be auto, gemm, or fused")
    if precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("precision must be float32, float16, or bfloat16")
    n, d = x.shape
    k = centers.shape[0]
    if k < 1 or centers.shape[1] != d:
        raise ValueError("centers must be nonempty and have the same feature dimension")
    if d < 1 or k * d > _UINT32_MAX:
        raise ValueError("centroid matrix exceeds Metal kernel indexing capacity")
    if n == 0:
        empty = (mx.zeros((0,), mx.uint32), mx.zeros((0,), mx.float32))
        return empty + (mx.zeros((0,), mx.float32),) if second else empty
    dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[precision]
    if precision == "float16":
        method = "fused"
    # Bound one GEMM tile to at most 1 GiB (less on smaller Macs). This bounds lazy graphs via
    # evaluation inside the loop, important for full-dataset predict/score.
    native_bf16 = precision == "bfloat16" and method in {"auto", "fused"}
    scratch_per_row = (12 if second else 8) * ((k + 63) // 64) if native_bf16 else 4 * k
    budget = min(256 << 20, _scratch_budget()) if native_bf16 else _scratch_budget()
    block = min(n, max(1, budget // scratch_per_row)) if batch_size is None else int(batch_size)
    if block < 1:
        raise ValueError("batch_size must be positive")
    block = min(block, _UINT32_MAX // k, _UINT32_MAX // d, _UINT32_MAX // 32)
    if precision == "bfloat16":
        center_mirror, norms = prepare_centers(centers)
        represented_centers = center_mirror.astype(mx.float32)
    else:
        center_mirror = centers.astype(dtype)
        represented_centers = center_mirror.astype(mx.float32)
        norms = mx.sum(represented_centers * represented_centers, axis=1)
    # BF16 GEMM outputs would round scores to BF16. Widen the represented
    # operands for the baseline so both implementations accumulate/output FP32.
    gemm_centers = represented_centers if precision == "bfloat16" else centers.astype(dtype)
    ct = gemm_centers.T
    # Spherical scores are -x.c and distances 2 - 2 x.c; Euclidean scores
    # differ from squared distances by the row's own norm only.
    margin_scale = 2.0 if spherical else 1.0
    chunks = []
    for start in range(0, n, block):
        xb = x[start:start + block]
        rows = xb.shape[0]
        best = other = None
        if native_bf16:
            if second:
                labels, best, other = assign_bf16(xb, center_mirror, norms=norms,
                                                  spherical=spherical, second=True)
            else:
                labels, _ = assign_bf16(xb, center_mirror, norms=norms, spherical=spherical)
        elif (method == "auto" and precision == "float32" and rows >= 128
              and k >= 4096 and 64 <= d <= 512 and _streaming_fp32_device()):
            # 64 centers per tile: keep one score and index per row in place
            # of 64 GEMM outputs. Keep the original batching, including GEMV
            # for singleton tails, whose dot reduction is different.
            if second:
                labels, best, other = streaming_tile(
                    xb.astype(mx.float32), gemm_centers, norms, spherical=spherical,
                    bm=64, bk=16, pad=4, coalesced=True, second=True,
                )
            else:
                labels, _ = streaming_tile(
                    xb.astype(mx.float32), gemm_centers, norms, spherical=spherical,
                    bm=64, bk=16, pad=4, coalesced=True,
                )
        elif method == "fused":
            parts = (k + 63) // 64
            template = [("T", dtype), ("N", rows), ("K", k), ("D", d),
                        ("PARTS", parts), ("SPHERICAL", spherical)]
            grid = (((rows + 31) // 32) * 256, parts, 1)
            if second:
                partials = _fused_second_kernel()(
                    inputs=[xb, centers, norms], template=template, grid=grid,
                    threadgroup=(256, 1, 1), output_shapes=[(rows, parts)] * 3,
                    output_dtypes=[mx.float32, mx.float32, mx.uint32], stream=mx.gpu)
                labels, best, other = _merge_second_kernel()(
                    inputs=list(partials), template=[("N", rows), ("PARTS", parts)],
                    grid=(rows * 32, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(rows,)] * 3,
                    output_dtypes=[mx.uint32, mx.float32, mx.float32], stream=mx.gpu)
            else:
                tile, merge = _fused_kernels()
                scores, indices = tile(
                    inputs=[xb, centers, norms], template=template, grid=grid,
                    threadgroup=(256, 1, 1),
                    output_shapes=[(rows, parts), (rows, parts)],
                    output_dtypes=[mx.float32, mx.uint32], stream=mx.gpu)
                labels = merge(
                    inputs=[scores, indices], template=[("N", rows), ("PARTS", parts)],
                    grid=(rows * 32, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(rows,)], output_dtypes=[mx.uint32], stream=mx.gpu)[0]
        else:
            operands = xb.astype(dtype)
            if precision == "bfloat16":
                # BF16 values are exact in TF32, so MLX's accelerator GEMM on
                # M5-class GPUs still forms these products exactly.
                scores = operands.astype(mx.float32) @ ct
            else:
                # Float32 scores need binary32 products on every GPU.
                scores = matmul(operands, gemm_centers, transpose_b=True)
            template = [("N", rows), ("K", k), ("SPHERICAL", spherical)]
            if second:
                labels, best, other = _kernel("argmin_second")(
                    inputs=[scores, norms], template=template,
                    grid=(rows * 32, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(rows,)] * 3,
                    output_dtypes=[mx.uint32, mx.float32, mx.float32], stream=mx.gpu)
            else:
                labels = _kernel("argmin")(
                    inputs=[scores, norms], template=template,
                    grid=(rows * 32, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(rows,)], output_dtypes=[mx.uint32], stream=mx.gpu)[0]
        distance_x = xb.astype(dtype) if precision == "bfloat16" else xb
        distance_centers = center_mirror if precision == "bfloat16" else centers
        distances = selected_distances(distance_x, distance_centers, labels, spherical=spherical)
        chunk = (labels, distances)
        if second:
            chunk += ((other - best) * margin_scale if margin_scale != 1.0 else other - best,)
        if n > block:
            mx.eval(*chunk)
        chunks.append(chunk)
    if len(chunks) == 1:
        return chunks[0]
    return tuple(mx.concatenate(pieces) for pieces in zip(*chunks, strict=True))


def gather_bf16(data, order):
    """Gather validated permutation rows directly into their BF16 batch.

    This preserves ``data[order].astype(bfloat16)`` without its FP32 batch
    intermediate. The caller supplies in-range nonnegative permutation indices;
    no host read or index scan is performed in the training loop. Source offsets
    use 64 bits, so a full resident input may exceed 2**32 elements.
    """
    if data.ndim != 2 or data.dtype != mx.float32 or data.shape[1] < 1:
        raise ValueError("gather data must be a float32 matrix with nonzero dimension")
    if order.ndim != 1 or order.dtype not in (mx.int32, mx.uint32):
        raise ValueError("gather order must be a 32-bit integer vector")
    b, d = order.size, data.shape[1]
    if b * d > _UINT32_MAX:
        raise ValueError("gather batch exceeds Metal uint32 indexing capacity")
    if b == 0:
        return mx.zeros((0, d), mx.bfloat16)
    return _kernel("gather_bf16")(
        inputs=[data, order], template=[("B", b), ("D", d)],
        grid=((b * d + 3) // 4, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(b, d)], output_dtypes=[mx.bfloat16], stream=mx.gpu)[0]


def batch_stats(x, labels, k):
    """Fuse float32 scatter-add of coordinates and cluster counts in Metal."""
    n, d = x.shape
    if n > 2**24:
        raise ValueError("a statistics batch must not exceed 2**24 rows for exact float32 counts")
    if n * d > _UINT32_MAX or k * d > _UINT32_MAX:
        raise ValueError("statistics matrix exceeds Metal kernel indexing capacity")
    return tuple(_kernel("stats")(
        inputs=[x, labels], template=[("N", n), ("D", d)],
        grid=(max(1, n * d), 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(k, d), (k,)], output_dtypes=[mx.float32, mx.float32],
        init_value=0, stream=mx.gpu))


def removal_stats(x, labels, margins, centers, axes):
    """Batch statistics of cost-based revival, fused in Metal.

    For each row, with ``y = x - centers[label]`` and ``p = <y, axes[label]>``,
    adds ``p * y`` to the centroid's axis sums and the row's margin to its
    removal cost; a margin that is not finite adds nothing. ``centers`` are
    the centroids the rows were assigned to, in the coordinates of ``x``, and
    ``axes`` their float32 unit directions. Returns float32 ``(k, d)`` axis
    sums and ``(k,)`` costs. Sums use float32 atomics, as ``batch_stats``.
    """
    n, d = x.shape
    k = centers.shape[0]
    if centers.shape != (k, d) or axes.shape != (k, d):
        raise ValueError("centers and axes must be matrices with one row per centroid")
    if labels.shape != (n,) or margins.shape != (n,):
        raise ValueError("labels and margins must have one entry per row")
    if n * max(d, 32) > _UINT32_MAX or k * d > _UINT32_MAX:
        raise ValueError("statistics matrix exceeds Metal kernel indexing capacity")
    return tuple(_kernel("removal_stats")(
        inputs=[x, labels, margins, centers, axes], template=[("N", n), ("D", d)],
        grid=(max(1, n) * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(k, d), (k,)], output_dtypes=[mx.float32, mx.float32],
        init_value=0, stream=mx.gpu))


def removal_update(long_axis, removal, axis_sums, costs, axes, alpha):
    """EMA of the cost-based revival statistics and the unit long axes.

    ``long_axis`` and ``removal`` decay by ``1 - alpha`` and gain ``alpha``
    times the batch's ``axis_sums`` and ``costs``. Returns the new long-axis
    sums, the new removal costs, their unit directions (the previous
    direction where the sum is zero) and the lengths of the sums.
    """
    k, d = long_axis.shape
    if k * max(d, 32) > _UINT32_MAX:
        raise ValueError("centroid matrix exceeds Metal kernel indexing capacity")
    return tuple(_kernel("removal_update")(
        inputs=[long_axis, removal, axis_sums, costs, axes,
                mx.array([alpha, 1.0 - alpha], mx.float32)],
        template=[("K", k), ("D", d)],
        grid=(k * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(k, d), (k,), (k, d), (k,)],
        output_dtypes=[mx.float32] * 4, stream=mx.gpu))


def ema_update(centers, counts, batch_sums, batch_counts, alpha, *, decay=None):
    """Fused EMA with one thread per centroid coordinate, no host transfer."""
    k, d = centers.shape
    if k * d > _UINT32_MAX:
        raise ValueError("centroid matrix exceeds Metal kernel indexing capacity")
    return tuple(_kernel("ema")(
        inputs=[centers, counts, batch_sums, batch_counts,
                mx.array([alpha, 1.0 - alpha if decay is None else decay], mx.float32)],
        template=[("K", k), ("D", d), ("INDEPENDENT_DECAY", decay is not None)],
        grid=(k * d, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(k, d), (k,)], output_dtypes=[mx.float32, mx.float32], stream=mx.gpu))
