"""Metal primitives for e-means on Apple Silicon.

Assignment uses MLX's tuned Metal GEMM, followed by a fused SIMD argmin, or
an experimental streaming SIMD-group matrix kernel that never materializes
the full distance matrix. All accumulation and centroid state are float32.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .e_means_bf16 import assign_bf16, prepare_centers

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


def assign(x, centers, *, batch_size=None, spherical=False, method="auto", precision="float32"):
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
    Float32 ``auto`` uses tuned GEMM; ``fused`` exposes the older matrix tiles.
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
        return mx.zeros((0,), mx.uint32), mx.zeros((0,), mx.float32)
    dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[precision]
    if precision == "float16":
        method = "fused"
    # Bound one GEMM tile to at most 1 GiB (less on smaller Macs). This bounds lazy graphs via
    # evaluation inside the loop, important for full-dataset predict/score.
    native_bf16 = precision == "bfloat16" and method in {"auto", "fused"}
    scratch_per_row = 8 * ((k + 63) // 64) if native_bf16 else 4 * k
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
    ct = represented_centers.T if precision == "bfloat16" else centers.astype(dtype).T
    labels_out, distances_out = [], []
    for start in range(0, n, block):
        xb = x[start:start + block]
        rows = xb.shape[0]
        if native_bf16:
            labels, _ = assign_bf16(xb, center_mirror, norms=norms, spherical=spherical)
        elif method == "fused":
            tile, merge = _fused_kernels()
            parts = (k + 63) // 64
            scores, indices = tile(
                inputs=[xb, centers, norms],
                template=[("T", dtype), ("N", rows), ("K", k), ("D", d),
                          ("PARTS", parts), ("SPHERICAL", spherical)],
                grid=(((rows + 31) // 32) * 256, parts, 1), threadgroup=(256, 1, 1),
                output_shapes=[(rows, parts), (rows, parts)],
                output_dtypes=[mx.float32, mx.uint32], stream=mx.gpu)
            labels = merge(
                inputs=[scores, indices], template=[("N", rows), ("PARTS", parts)],
                grid=(rows * 32, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(rows,)], output_dtypes=[mx.uint32], stream=mx.gpu)[0]
        else:
            operands = xb.astype(dtype)
            if precision == "bfloat16":
                operands = operands.astype(mx.float32)
            scores = operands @ ct
            labels = _kernel("argmin")(
                inputs=[scores, norms], template=[("N", rows), ("K", k), ("SPHERICAL", spherical)],
                grid=(rows * 32, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(rows,)], output_dtypes=[mx.uint32], stream=mx.gpu)[0]
        distance_x = xb.astype(dtype) if precision == "bfloat16" else xb
        distance_centers = center_mirror if precision == "bfloat16" else centers
        distances = selected_distances(distance_x, distance_centers, labels, spherical=spherical)
        if n > block:
            mx.eval(labels, distances)
        labels_out.append(labels)
        distances_out.append(distances)
    return (labels_out[0], distances_out[0]) if len(labels_out) == 1 else (
        mx.concatenate(labels_out), mx.concatenate(distances_out))


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
