"""Binary32 Gram products for certified neighbor screening.

The general GPU path proves that omitted candidates cannot enter the result
from a float32 error bound on ``centered @ train.T``. On macOS 26.2+ with GPU
generation 17 or later (M5), MLX (verified with 0.32.4) runs float32 GEMM on
the GPU neural accelerators with TF32 operands unless the process started with
``MLX_ENABLE_TF32=0``. The flag is read once per process, with no per-call
override, and covers ``matmul``, ``addmm``, ``einsum`` and ``block_masked_mm``
alike. Operands then keep about eleven significant bits: measured errors are
up to ~2e4 binary32 ulps of ``|a| @ |b|.T``, so Gram cancellation produced
lower bounds above exact distances and certified wrong neighbors.

Where MLX may take that route, this module supplies the product from its own
kernel. Binary32 operands go through ``simdgroup_matrix<float>`` multiply-
accumulate: the same fp32 arithmetic as MLX's GEMM on earlier GPUs, which the
screening bound already assumes, at the speed of MLX's own fp32 GEMM on M5.
Earlier GPUs keep ``mx.matmul`` and therefore unchanged results and speed.
"""

from __future__ import annotations

import re
from functools import lru_cache

import mlx.core as mx

__all__ = ["architecture_may_round_float32", "matmul_may_round_float32", "float32_gram"]

# MLX enables its neural-accelerator GEMM from this generation onward
# (generation 18 for "p" variants). Anything unrecognised is treated as
# capable of it: a slower exact product never changes results.
_FIRST_ACCELERATED_GENERATION = 17
_ARCHITECTURE = re.compile(r"applegpu_g(\d+)[a-z]*")


def architecture_may_round_float32(architecture: str) -> bool:
    """Whether MLX float32 GEMM may round operands on this GPU architecture."""
    match = _ARCHITECTURE.fullmatch(architecture)
    return match is None or int(match.group(1)) >= _FIRST_ACCELERATED_GENERATION


@lru_cache(maxsize=1)
def matmul_may_round_float32() -> bool:
    """Whether ``mx.matmul`` on the default GPU may not be binary32 GEMM."""
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        architecture = str(info.get("architecture", ""))
    except Exception:  # pragma: no cover - unknown runtimes take the exact kernel
        return True
    return architecture_may_round_float32(architecture)


# Each 128-thread group computes a 64x64 output tile as a 2x2 grid of SIMD
# groups, each holding 4x4 accumulators of 8x8. Out-of-range operands load as
# zero and contribute exact zero products. Edge fragments store through
# threadgroup scratch so no lane writes outside the output. Every loop over
# fragments is fully unrolled: dynamically indexed simdgroup matrices spill to
# thread memory, which made an earlier draft 14x slower on M5. Inputs with
# fewer than eight elements arrive in the constant address space.
_GRAM_HEADER = r"""
#include <metal_simdgroup_matrix>
inline float4 gram_load4(const device float* p) { return *(const device float4*)p; }
inline float4 gram_load4(const constant float* p) { return *(const constant float4*)p; }
"""

_GRAM = r"""
constexpr int BM = 64, BN = 64, BK = 16, LD = BK + 4;
threadgroup float4 a_store[BM * LD / 4];
threadgroup float4 b_store[BN * LD / 4];
threadgroup float edge[4 * 64];
threadgroup float* a_tile = (threadgroup float*)a_store;
threadgroup float* b_tile = (threadgroup float*)b_store;
const int M = shape[0], N = shape[1], K = shape[2];
const int row0 = int(threadgroup_position_in_grid.y) * BM;
const int col0 = int(threadgroup_position_in_grid.x) * BN;
const int tid = int(thread_index_in_threadgroup);
const uint group = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int sm = int(group / 2) * 32, sn = int(group % 2) * 32;
simdgroup_float8x8 acc[4][4];
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i)
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
for (int k0 = 0; k0 < K; k0 += BK) {
    if (ALIGNED) {  // K % 4 == 0: a float4 never crosses the end of a row
        #pragma clang loop unroll(full)
        for (int p = 0; p < BM * BK / 512; ++p) {
            const int e = tid + p * 128;
            const int r = e / (BK / 4), c = (e % (BK / 4)) * 4, kk = k0 + c;
            const int ar = row0 + r, br = col0 + r;
            a_store[(r * LD + c) / 4] = (ar < M && kk < K)
                ? gram_load4(a + size_t(ar) * K + kk) : float4(0.0f);
            b_store[(r * LD + c) / 4] = (br < N && kk < K)
                ? gram_load4(b + size_t(br) * K + kk) : float4(0.0f);
        }
    } else {
        #pragma clang loop unroll(full)
        for (int p = 0; p < BM * BK / 128; ++p) {
            const int e = tid + p * 128;
            const int r = e / BK, c = e % BK, kk = k0 + c;
            const int ar = row0 + r, br = col0 + r;
            a_tile[r * LD + c] = (ar < M && kk < K) ? a[size_t(ar) * K + kk] : 0.0f;
            b_tile[r * LD + c] = (br < N && kk < K) ? b[size_t(br) * K + kk] : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int kk = 0; kk < BK; kk += 8) {
        simdgroup_float8x8 af[4], bf[4];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i)
            simdgroup_load(af[i], a_tile + (sm + i * 8) * LD + kk, LD);
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j)
            simdgroup_load(bf[j], b_tile + (sn + j * 8) * LD + kk, LD, ulong2(0, 0), true);
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i)
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
        const int r = row0 + sm + i * 8, c = col0 + sn + j * 8;
        if (r + 8 <= M && c + 8 <= N) {
            simdgroup_store(acc[i][j], out + size_t(r) * N + c, N);
        } else if (r < M && c < N) {  // uniform across this SIMD group
            simdgroup_store(acc[i][j], edge + group * 64, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int e = int(lane); e < 64; e += 32) {
                const int rr = r + e / 8, cc = c + e % 8;
                if (rr < M && cc < N) out[size_t(rr) * N + cc] = edge[group * 64 + e];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"""


@lru_cache(maxsize=1)
def _gram_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_float32_gram",
        input_names=["a", "b", "shape"],
        output_names=["out"],
        source=_GRAM,
        header=_GRAM_HEADER,
    )


# Grid sizes count threads in 32 bits; wider training tiles launch in chunks.
_MAX_LAUNCH_COLUMNS = 1 << 24


def float32_gram(a: mx.array, b: mx.array) -> mx.array:
    """Return ``a @ b.T`` for float32 matrices with binary32 arithmetic."""
    for name, array in (("a", a), ("b", b)):
        if not isinstance(array, mx.array) or array.dtype != mx.float32 or array.ndim != 2:
            raise TypeError(f"{name} must be a two-dimensional float32 MLX array")
    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must have the same number of columns")
    rows, cols = a.shape[0], b.shape[0]
    if rows == 0 or cols == 0 or a.shape[1] == 0:
        return mx.zeros((rows, cols), mx.float32)
    if max(rows, cols, a.shape[1]) >= 2**31:
        raise ValueError("Gram dimensions must fit int32 indexing")
    if cols > _MAX_LAUNCH_COLUMNS:
        return mx.concatenate([
            float32_gram(a, b[start:start + _MAX_LAUNCH_COLUMNS])
            for start in range(0, cols, _MAX_LAUNCH_COLUMNS)
        ], axis=1)
    (out,) = _gram_kernel()(
        inputs=[a, b, mx.array([rows, cols, a.shape[1]], mx.int32)],
        template=[("ALIGNED", a.shape[1] % 4 == 0)],
        grid=(((cols + 63) // 64) * 128, (rows + 63) // 64, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
    )
    return out
