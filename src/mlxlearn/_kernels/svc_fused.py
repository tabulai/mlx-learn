"""Fused Metal epilogues and row queries for exact C-SVC kernels.

Large Gram blocks retain MLX's tuned matrix multiplication. One elementwise
kernel then combines the RBF norms, clamp and exponential, or evaluates the
polynomial/sigmoid transform. A cache miss instead computes the requested row
and its transform in a single launch. Its RBF distance uses direct differences,
so near-duplicate points do not subtract two large squared norms.

Only kernel entries are float32 here. The SMO state and the weighted decision
reduction retain their float64 contract outside this module. In particular,
casting dual weights to float32 to fuse prediction would lose information that
compensated summation cannot restore.
"""

from __future__ import annotations

import operator

import mlx.core as mx

__all__ = ["transform", "row", "rows"]

_KINDS = {"linear": 0, "rbf": 1, "poly": 2, "sigmoid": 3}

_HEADER = r"""
inline float svc_power(float base, ulong degree) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    // Exactly the repeated-squaring order used by the portable evaluator.
    float result = 1.0f;
    bool initialized = false;
    while (degree) {
        if (degree & 1) {
            result = initialized ? result * base : base;
            initialized = true;
        }
        degree >>= 1;
        if (degree) base = base * base;
    }
    return result;
}
inline float svc_transform(float value, float gamma, float coef0, uint kind,
                           ulong degree) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    if (kind == 0) return value;
    if (kind == 1) return metal::exp(-gamma * metal::max(value, 0.0f));
    float z = gamma * value + coef0;
    if (kind == 3) return metal::tanh(z);
    return svc_power(z, degree);
}
"""

_transform = mx.fast.metal_kernel(
    name="mlxlearn_svc_transform",
    input_names=["dot", "a_sq", "b_sq", "params", "degree"],
    output_names=["out"],
    header=_HEADER,
    source=r"""
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint index = thread_position_in_grid.x;
    if (index >= dot_shape[0] * dot_shape[1]) return;
    float value = dot[index];
    if (KIND == 1) {
        uint columns = dot_shape[1];
        value = (a_sq[index / columns] + b_sq[index % columns]) - 2.0f * value;
    }
    out[index] = svc_transform(value, params[0], params[1], KIND, degree[0]);
""",
)

_row = mx.fast.metal_kernel(
    name="mlxlearn_svc_row",
    input_names=["data", "index", "params", "degree"],
    output_names=["out"],
    header=_HEADER,
    source=r"""
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint tid = thread_position_in_grid.x;
    uint target = tid / WIDTH;
    uint query = thread_position_in_grid.y;
    uint lane = thread_index_in_simdgroup % WIDTH;
    // WIDTH divides 32, so each row owns an independent SIMD subgroup.
    if (target >= data_shape[0]) return;
    ulong target_offset = ulong(target) * D;
    ulong query_offset = ulong(index[query]) * D;
    float sum = 0.0f, correction = 0.0f;
    for (uint feature = lane; feature < D; feature += WIDTH) {
        float a = data[target_offset + feature];
        float b = data[query_offset + feature];
        float term;
        if (KIND == 1) {
            float difference = a - b;
            term = difference * difference;
        } else {
            term = a * b;
        }
        // Long rows and mixed scales must not discard all small terms.
        float adjusted = term - correction;
        float next = sum + adjusted;
        correction = metal::isfinite(next) ? (next - sum) - adjusted : 0.0f;
        sum = next;
    }
    for (uint shift = WIDTH / 2; shift > 0; shift /= 2) {
        sum += metal::simd_shuffle_down(sum, shift);
    }
    if (lane == 0) out[ulong(query) * data_shape[0] + target] =
        svc_transform(sum, params[0], params[1], KIND, degree[0]);
""",
)


def _parameters(spec):
    # Runtime values avoid compiling a new Metal library for every resolved
    # gamma. A uint64 exponent preserves the integer API for large degrees.
    if not 0 <= int(spec.degree) < 2**64:
        raise ValueError("fused polynomial degree must fit in uint64")
    return (
        mx.array([spec.gamma, spec.coef0], dtype=mx.float32),
        mx.array([int(spec.degree)], dtype=mx.uint64),
    )


def transform(spec, dot, a_sq=None, b_sq=None):
    """Transform a float32 MLX inner-product matrix on the GPU in one pass.

    RBF requires float32 squared-norm vectors matching the two matrix axes.
    Linear kernels return ``dot`` unchanged. This GPU-only helper does not
    change the default device or evaluate/synchronize its result.
    """
    if dot.ndim != 2 or dot.dtype != mx.float32:
        raise ValueError("dot must be a float32 matrix")
    if spec.kernel == "linear" or dot.size == 0:
        return dot
    if spec.kernel == "rbf":
        if (
            a_sq is None or b_sq is None
            or a_sq.shape != (dot.shape[0],) or b_sq.shape != (dot.shape[1],)
            or a_sq.dtype != mx.float32 or b_sq.dtype != mx.float32
        ):
            raise ValueError("RBF norms must be float32 vectors matching dot's axes")
    else:
        # Unused operands keep the kernel signature uniform without making
        # feature-count-sized dummy allocations.
        a_sq = b_sq = mx.array([0.0], dtype=mx.float32)
    params, degree = _parameters(spec)
    return _transform(
        inputs=[dot, a_sq, b_sq, params, degree],
        template=[("KIND", _KINDS[spec.kernel])],
        grid=(dot.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[dot.shape],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
    )[0]


def row(spec, data, index):
    """Return ``K(data[index], data)`` in one GPU launch.

    ``data`` is a float32 MLX matrix. RBF accepts either original or commonly
    centered coordinates. Eight lanes share a row for at most 32 features;
    otherwise the whole SIMD group cooperates. No dense Gram is allocated.
    """
    if data.ndim != 2 or data.dtype != mx.float32 or data.shape[1] == 0:
        raise ValueError("data must be a float32 matrix with at least one feature")
    index = operator.index(index)
    if not 0 <= index < data.shape[0]:
        raise IndexError("kernel row index out of bounds")
    params, degree = _parameters(spec)
    width = 8 if data.shape[1] <= 32 else 32
    return _row(
        inputs=[data, mx.array([index], dtype=mx.uint32), params, degree],
        template=[("KIND", _KINDS[spec.kernel]), ("D", data.shape[1]), ("WIDTH", width)],
        grid=(data.shape[0] * width, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(data.shape[0],)],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
    )[0]


def rows(spec, data, indices):
    """Evaluate a requested working set in one GPU launch, without a full Gram.

    The integer row IDs are a small host sequence. Arithmetic and reductions
    use exactly the same kernel as single-row queries, including direct RBF
    distances. Reordering or repeating IDs preserves the corresponding rows.
    """
    if data.ndim != 2 or data.dtype != mx.float32 or data.shape[1] == 0:
        raise ValueError("data must be a float32 matrix with at least one feature")
    ids = [operator.index(index) for index in indices]
    if any(index < 0 or index >= data.shape[0] for index in ids):
        raise IndexError("kernel row index out of bounds")
    if not ids:
        return mx.zeros((0, data.shape[0]), dtype=mx.float32)
    params, degree = _parameters(spec)
    width = 8 if data.shape[1] <= 32 else 32
    return _row(
        inputs=[data, mx.array(ids, dtype=mx.uint32), params, degree],
        template=[("KIND", _KINDS[spec.kernel]), ("D", data.shape[1]), ("WIDTH", width)],
        grid=(data.shape[0] * width, len(ids), 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(len(ids), data.shape[0])],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
    )[0]
