"""Binary32 matrix products on GPUs where MLX may round float32 GEMM to TF32.

On macOS 26.2+ with GPU generation 17 or later (M5), MLX (verified with
0.32.4) runs float32 GEMM on the GPU neural accelerators with TF32 operands
unless the process started with ``MLX_ENABLE_TF32=0``. The flag is read once
per process, with no per-call override, and covers ``matmul``, ``addmm``,
``einsum``, ``block_masked_mm`` and ``gather_mm`` alike. Operands then keep
about eleven significant bits: measured errors reach ~2e4 binary32 ulps of
``|a| @ |b|``, against about five on M4. Only matrix-vector products (one
output row or column) keep binary32 operands.

Where MLX may take that route, :func:`matmul` and :func:`addmm` compute the
product with this module's kernel instead. Binary32 operands go through
``simdgroup_matrix<float>`` multiply-accumulate: the same fp32 arithmetic as
MLX's GEMM on earlier GPUs, at about the speed of MLX's own fp32 GEMM on M5.
Earlier GPUs, the CPU, other dtypes and matrix-vector shapes evaluate the
caller's original MLX expression, so their results and speed are unchanged.
"""

from __future__ import annotations

import re
from functools import cache, lru_cache

import mlx.core as mx

__all__ = [
    "addmm",
    "architecture_may_round_float32",
    "float32_matmul",
    "matmul",
    "matmul_may_round_float32",
]

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
# groups, each holding 4x4 accumulators of 8x8. Tiles are staged in the
# operand's storage order, so device loads stay contiguous (float4 when the
# contiguous extent is a multiple of four) and transposed operands are read
# through transposed simdgroup loads. Out-of-range operands load as zero and
# contribute exact zero products. Every loop over fragments is fully unrolled:
# dynamically indexed simdgroup matrices spill to thread memory, which made an
# earlier draft 14x slower on M5. Each lane stores the two accumulator elements
# it owns (Apple's 8x8 fragment layout, as in MLX's steel GEMM), so edges need
# no scratch. Split launches (grid z) cover consecutive 16-aligned K ranges and
# write partial products for a separate sum. Inputs with fewer than eight
# elements arrive in the constant address space.
_HEADER = r"""
#include <metal_simdgroup_matrix>
inline float4 b32_load4(const device float* p) { return *(const device float4*)p; }
inline float4 b32_load4(const constant float* p) { return *(const constant float4*)p; }
inline float4 b32_loadu4(const device float* p) { return float4(*(const device packed_float4*)p); }
inline float4 b32_loadu4(const constant float* p) { return float4(*(const constant packed_float4*)p); }

// Stages logical rows [base, base + 64) by K [k0, k0 + 16) of one operand into
// threadgroup memory, zero-filling out-of-range elements. KMAJOR operands are
// stored as (K, extent) and staged as 16 rows of 64; the others are stored as
// (extent, K) and staged as 64 rows of 16. Every thread moves two runs of four
// contiguous elements and advances its source pointer by one K block per
// step. ALIGNED runs are float4-aligned and never straddle an end; unchecked
// loads are only issued for blocks wholly inside the operand.
template <bool KMAJOR, bool ALIGNED>
struct b32_loader {
    static constexpr constant int LD = KMAJOR ? 68 : 20;
    static constexpr constant int DR = KMAJOR ? 8 : 32;
    size_t offset, second, step;
    threadgroup float* dst;
    int r, c, base, extent;

    b32_loader(threadgroup float* tile, int tid, int base_, int extent_, int k_begin, int depth) {
        r = KMAJOR ? tid / 16 : tid / 4;
        c = KMAJOR ? (tid % 16) * 4 : (tid % 4) * 4;
        base = base_;
        extent = extent_;
        offset = KMAJOR ? size_t(k_begin + r) * size_t(extent) + size_t(base + c)
                        : size_t(base + r) * size_t(depth) + size_t(k_begin + c);
        second = KMAJOR ? size_t(DR) * size_t(extent) : size_t(DR) * size_t(depth);
        step = KMAJOR ? size_t(16) * size_t(extent) : size_t(16);
        dst = tile + r * LD + c;
    }

    template <typename P>
    float4 get(P p) const { return ALIGNED ? b32_load4(p) : b32_loadu4(p); }

    template <typename P>
    void load(P data) const {
        *(threadgroup float4*)dst = get(data + offset);
        *(threadgroup float4*)(dst + DR * LD) = get(data + offset + second);
    }

    template <typename P>
    void load_checked(P data, int k0, int k_end) const {
        #pragma clang loop unroll(full)
        for (int p = 0; p < 2; ++p) {
            const int row = r + p * DR;
            const int k = KMAJOR ? k0 + row : k0 + c, m = KMAJOR ? base + c : base + row;
            const size_t at = offset + (p ? second : 0);
            float4 v;
            if (ALIGNED) {
                v = (k < k_end && m < extent) ? b32_load4(data + at) : float4(0.0f);
            } else {
                #pragma clang loop unroll(full)
                for (int i = 0; i < 4; ++i) {
                    const bool inside = KMAJOR ? (k < k_end && m + i < extent)
                                               : (m < extent && k + i < k_end);
                    v[i] = inside ? data[at + i] : 0.0f;
                }
            }
            *(threadgroup float4*)(dst + p * DR * LD) = v;
        }
    }

    void next() { offset += step; }
};
"""

_SOURCE = r"""
constexpr int BM = 64, BN = 64, BK = 16;
constexpr int LDA = TA ? BM + 4 : BK + 4;
constexpr int LDB = TB ? BK + 4 : BN + 4;
threadgroup float4 a_store[(TA ? BK * LDA : BM * LDA) / 4];
threadgroup float4 b_store[(TB ? BN * LDB : BK * LDB) / 4];
threadgroup float* a_tile = (threadgroup float*)a_store;
threadgroup float* b_tile = (threadgroup float*)b_store;
const int M = shape[0], N = shape[1], K = shape[2], span = shape[3];
const int row0 = int(threadgroup_position_in_grid.y) * BM;
const int col0 = int(threadgroup_position_in_grid.x) * BN;
const int k_begin = int(threadgroup_position_in_grid.z) * span;
const int k_end = min(K, k_begin + span);
const bool rows_inside = row0 + BM <= M, cols_inside = col0 + BN <= N;
const int tid = int(thread_index_in_threadgroup);
const int group = int(simdgroup_index_in_threadgroup);
const int lane = int(thread_index_in_simdgroup);
const int sm = (group / 2) * 32, sn = (group % 2) * 32;
simdgroup_float8x8 acc[4][4];
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i)
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
b32_loader<TA, VA> a_loader(a_tile, tid, row0, M, k_begin, K);
b32_loader<!TB, VB> b_loader(b_tile, tid, col0, N, k_begin, K);
for (int k0 = k_begin; k0 < k_end; k0 += BK) {
    const bool k_inside = k0 + BK <= k_end;
    // Uniform per threadgroup: interior blocks skip every bounds test.
    if (rows_inside && k_inside) a_loader.load(a); else a_loader.load_checked(a, k0, k_end);
    if (cols_inside && k_inside) b_loader.load(b); else b_loader.load_checked(b, k0, k_end);
    a_loader.next();
    b_loader.next();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int kk = 0; kk < BK; kk += 8) {
        simdgroup_float8x8 af[4], bf[4];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
            if (TA) simdgroup_load(af[i], a_tile + kk * LDA + sm + i * 8, LDA, ulong2(0, 0), true);
            else simdgroup_load(af[i], a_tile + (sm + i * 8) * LDA + kk, LDA);
        }
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
            if (TB) simdgroup_load(bf[j], b_tile + (sn + j * 8) * LDB + kk, LDB, ulong2(0, 0), true);
            else simdgroup_load(bf[j], b_tile + kk * LDB + sn + j * 8, LDB);
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i)
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
const int quad = lane / 4;
const int fm = (quad & 4) + (lane / 2) % 4;
const int fn = (quad & 2) * 2 + (lane % 2) * 2;
device float* target = out + size_t(threadgroup_position_in_grid.z) * size_t(M) * size_t(N);
// Interior tiles with an even row length store each lane's adjacent pair as
// one aligned float2 (fn is even); other tiles check every element.
const bool paired = rows_inside && cols_inside && N % 2 == 0;
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
    const int r = row0 + sm + i * 8 + fm;
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
        const int c = col0 + sn + j * 8 + fn;
        const auto values = acc[i][j].thread_elements();
        if (paired) {
            float2 pair;
            #pragma clang loop unroll(full)
            for (int t = 0; t < 2; ++t) pair[t] = STORE(values[t]);
            *(device float2*)(target + size_t(r) * N + c) = pair;
        } else if (r < M) {
            #pragma clang loop unroll(full)
            for (int t = 0; t < 2; ++t) {
                if (c + t < N) {
                    target[size_t(r) * N + c + t] = STORE(values[t]);
                }
            }
        }
    }
}
"""


# ``alpha * product + beta * c``, with ``c`` a row vector (1) or full matrix (2).
_AXPBY = r"""
#define STORE(value) (coefficients[0] * (value) + coefficients[1] \
    * (EPILOGUE == 1 ? c_in[c + t] : c_in[size_t(r) * N + c + t]))
"""


@cache
def _kernel(epilogue: bool):
    inputs = ["a", "b", "shape"] + (["c_in", "coefficients"] if epilogue else [])
    return mx.fast.metal_kernel(
        name="mlxlearn_binary32_gemm_axpby" if epilogue else "mlxlearn_binary32_gemm",
        input_names=inputs,
        output_names=["out"],
        source=(_AXPBY if epilogue else "#define STORE(value) (value)\n") + _SOURCE,
        header=_HEADER,
    )


# Grid sizes count threads in 32 bits; wider products launch in column chunks.
_MAX_LAUNCH_COLUMNS = 1 << 24
# Split K when the output has too few tiles to occupy the GPU and each
# threadgroup would otherwise run a long serial K loop.
_SPLIT_TARGET_GROUPS = 1024
_SPLIT_MIN_SPAN = 512
_SPLIT_MAX_PARTS = 256


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _split_span(rows: int, cols: int, depth: int) -> int:
    """Return the K range each launch slice covers (``depth`` means no split)."""
    tiles = _ceil_div(rows, 64) * _ceil_div(cols, 64)
    parts = min(_SPLIT_MAX_PARTS, _SPLIT_TARGET_GROUPS // tiles, depth // _SPLIT_MIN_SPAN)
    if parts < 2:
        return depth
    return _ceil_div(_ceil_div(depth, parts), 16) * 16


def _launch(a, b, rows, cols, depth, transpose_a, transpose_b, epilogue=None):
    span = _split_span(rows, cols, depth) if epilogue is None else depth
    parts = _ceil_div(depth, span)
    inputs = [a, b, mx.array([rows, cols, depth, span], mx.int32)]
    template = [
        ("TA", transpose_a),
        ("TB", transpose_b),
        ("VA", (rows if transpose_a else depth) % 4 == 0),
        ("VB", (depth if transpose_b else cols) % 4 == 0),
    ]
    if epilogue is not None:
        c_in, kind, alpha, beta = epilogue
        inputs += [c_in, mx.array([alpha, beta], mx.float32)]
        template.append(("EPILOGUE", kind))
    (out,) = _kernel(epilogue is not None)(
        inputs=inputs,
        template=template,
        grid=(_ceil_div(cols, 64) * 128, _ceil_div(rows, 64), parts),
        threadgroup=(128, 1, 1),
        output_shapes=[(parts, rows, cols) if parts > 1 else (rows, cols)],
        output_dtypes=[mx.float32],
    )
    return mx.sum(out, axis=0) if parts > 1 else out


def _check_operands(a, b, transpose_a, transpose_b):
    for name, array in (("a", a), ("b", b)):
        if not isinstance(array, mx.array) or array.dtype != mx.float32 or array.ndim != 2:
            raise TypeError(f"{name} must be a two-dimensional float32 MLX array")
    rows, depth = (a.shape[1], a.shape[0]) if transpose_a else a.shape
    other, cols = (b.shape[1], b.shape[0]) if transpose_b else b.shape
    if depth != other:
        raise ValueError(f"inner dimensions differ: {depth} and {other}")
    if max(rows, cols, depth) > 2**30:
        raise ValueError("matrix dimensions must leave headroom for int32 tile indexing")
    return rows, cols, depth


def _column_chunks(b, cols, transpose_b):
    for start in range(0, cols, _MAX_LAUNCH_COLUMNS):
        stop = min(start + _MAX_LAUNCH_COLUMNS, cols)
        yield b[start:stop] if transpose_b else b[:, start:stop]


def float32_matmul(a, b, *, transpose_a=False, transpose_b=False):
    """Return ``op(a) @ op(b)`` for float32 matrices with binary32 arithmetic.

    ``op`` transposes its operand when the matching flag is set; pass the
    stored (untransposed) arrays so no transposed copy is made. Products run on
    the GPU with fp32 multiply-accumulate in every case, whatever MLX's own
    float32 GEMM would do on this device.
    """
    rows, cols, depth = _check_operands(a, b, transpose_a, transpose_b)
    if rows == 0 or cols == 0 or depth == 0:
        return mx.zeros((rows, cols), mx.float32)
    if cols > _MAX_LAUNCH_COLUMNS:
        return mx.concatenate([
            float32_matmul(a, part, transpose_a=transpose_a, transpose_b=transpose_b)
            for part in _column_chunks(b, cols, transpose_b)
        ], axis=1)
    return _launch(a, b, rows, cols, depth, transpose_a, transpose_b)


def float32_addmm(c, a, b, *, alpha=1.0, beta=1.0, transpose_a=False, transpose_b=False):
    """Return ``alpha * (op(a) @ op(b)) + beta * c`` with a binary32 product.

    ``c`` broadcasts against the ``(rows, cols)`` product, as in ``mx.addmm``.
    A row vector or full matrix is applied in the kernel's store; split
    products and other broadcasts add it after the partial sums.
    """
    rows, cols, depth = _check_operands(a, b, transpose_a, transpose_b)
    if not isinstance(c, mx.array) or c.dtype != mx.float32:
        raise TypeError("c must be a float32 MLX array")
    alpha, beta = float(alpha), float(beta)
    shape = (rows, cols)
    if mx.broadcast_shapes(c.shape, shape) != shape:
        raise ValueError(f"c with shape {c.shape} does not broadcast to {shape}")
    if rows == 0 or cols == 0:
        return mx.zeros(shape, mx.float32)
    if depth == 0:
        return beta * mx.broadcast_to(c, shape)
    unfused = (cols > _MAX_LAUNCH_COLUMNS or _split_span(rows, cols, depth) < depth
               or c.shape not in {(cols,), (1, cols), shape})
    if unfused:
        product = float32_matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
        return alpha * product + beta * c
    kind = 2 if c.shape == shape and rows > 1 else 1
    return _launch(a, b, rows, cols, depth, transpose_a, transpose_b,
                   epilogue=(c.reshape(-1), kind, alpha, beta))


def _needs_kernel(left, right) -> bool:
    """Whether ``left @ right`` would be a float32 GEMM that MLX may round."""
    return (
        left.dtype == mx.float32
        and right.dtype == mx.float32
        and left.ndim == 2
        and right.ndim == 2
        # Matrix-vector shapes take MLX's gemv kernels, which stay binary32.
        and left.shape[0] > 1
        and right.shape[1] > 1
        and mx.default_device() == mx.gpu
        and matmul_may_round_float32()
    )


def matmul(a, b, *, transpose_a=False, transpose_b=False):
    """``op(a) @ op(b)``, with binary32 products wherever MLX may use TF32.

    Elsewhere this evaluates exactly the MLX expression ``op(a) @ op(b)``.
    """
    left = a.T if transpose_a else a
    right = b.T if transpose_b else b
    if not _needs_kernel(left, right):
        return left @ right
    return float32_matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)


def addmm(c, a, b, *, alpha=1.0, beta=1.0, transpose_a=False, transpose_b=False):
    """``mx.addmm(c, op(a), op(b))``, with binary32 products wherever MLX may use TF32.

    Elsewhere this evaluates exactly ``mx.addmm(c, op(a), op(b), alpha, beta)``.
    """
    left = a.T if transpose_a else a
    right = b.T if transpose_b else b
    if not (_needs_kernel(left, right) and c.dtype == mx.float32):
        return mx.addmm(c, left, right, alpha=alpha, beta=beta)
    return float32_addmm(c, a, b, alpha=alpha, beta=beta,
                         transpose_a=transpose_a, transpose_b=transpose_b)
