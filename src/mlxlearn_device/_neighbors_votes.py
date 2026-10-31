"""Deterministic neighbor voting without atomics or host result transfers.

Metal votes use compensated sums and reciprocal corrections in a fixed neighbor
order. Predictions compare those sums before normalization rounds probabilities
to float32. CPU execution uses MLX float64 arithmetic on the current stream.
"""

from __future__ import annotations

import mlx.core as mx

_HEADER = r"""
#pragma clang fp contract(off) reassociate(off)
inline float2 vote_add(float2 a, float2 b) {
    float s = a.x + b.x;
    float v = s - a.x;
    float e = (a.x - (s - v)) + (b.x - v);
    float t = a.y + b.y + e;
    float h = s + t;
    return float2(h, t - (h - s));
}
inline bool vote_greater(float2 a, float2 b) {
    float2 diff = vote_add(a, -b);
    return diff.x > 0.0f || (diff.x == 0.0f && diff.y > 0.0f);
}
inline float vote_ratio(float2 score, float2 total) {
    float p = score.x / total.x;
    return p + ((fma(-p, total.x, score.x) + score.y) - p * total.y) / total.x;
}
inline float2 vote_weight(float distance, float minimum, bool zero, bool squared) {
    if (zero) return float2(distance == 0.0f ? 1.0f : 0.0f, 0.0f);
    if (isinf(distance)) return float2(0.0f);
    if (squared) {
        // Retain the residual of each square root: rounding the public
        // distances first can turn a distinct pair of squared distances into
        // a tied vote. Separate roots also avoid underflow in minimum/distance.
        float numerator = sqrt(minimum), denominator = sqrt(distance);
        float numerator_tail = fma(-numerator, numerator, minimum) / (2.0f * numerator);
        float denominator_tail = fma(-denominator, denominator, distance) / (2.0f * denominator);
        float hi = numerator / denominator;
        float lo = ((fma(-hi, denominator, numerator) + numerator_tail) - hi * denominator_tail) / denominator;
        return float2(hi, lo);
    }
    float hi = minimum / distance;
    return float2(hi, fma(-hi, distance, minimum) / distance);
}
"""

_fused = mx.fast.metal_kernel(
    name="mlxlearn_neighbor_votes",
    input_names=["indices", "targets", "distances"],
    output_names=["probabilities", "predictions"],
    header=_HEADER,
    source=r"""
uint row = thread_position_in_grid.x;
if (row >= N) return;
float minimum = INFINITY;
bool zero = false;
if (WEIGHTED) {
    for (uint j = 0; j < K; ++j) {
        float d = distances[ulong(row) * K + j];
        zero = zero || d == 0.0f;
        if (d > 0.0f) minimum = min(minimum, d);
    }
}
float2 scores[C];
for (uint c = 0; c < C; ++c) scores[c] = float2(0.0f);
float2 total = float2(0.0f);
for (uint j = 0; j < K; ++j) {
    ulong offset = ulong(row) * K + j;
    uint code = uint(targets[indices[offset]]);
    float2 w = WEIGHTED ? vote_weight(distances[offset], minimum, zero, SQUARED) : float2(1.0f, 0.0f);
    scores[code] = vote_add(scores[code], w);
    total = vote_add(total, w);
}
uint best = 0;
if (!(total.x > 0.0f) || !isfinite(total.x)) {
    for (uint c = 0; c < C; ++c) probabilities[ulong(row) * C + c] = NAN;
    predictions[row] = -1;
    return;
}
for (uint c = 0; c < C; ++c) {
    probabilities[ulong(row) * C + c] = vote_ratio(scores[c], total);
    if (vote_greater(scores[c], scores[best])) best = c;
}
predictions[row] = int(best);
""",
)

_class_scores = mx.fast.metal_kernel(
    name="mlxlearn_neighbor_class_scores",
    input_names=["indices", "targets", "distances"],
    output_names=["high", "low"],
    header=_HEADER,
    source=r"""
ulong i = thread_position_in_grid.x;
if (i >= ulong(N) * C) return;
uint row = uint(i / C), code = uint(i % C);
float minimum = INFINITY;
bool zero = false;
if (WEIGHTED) {
    for (uint j = 0; j < K; ++j) {
        float d = distances[ulong(row) * K + j];
        zero = zero || d == 0.0f;
        if (d > 0.0f) minimum = min(minimum, d);
    }
}
float2 score = float2(0.0f);
for (uint j = 0; j < K; ++j) {
    ulong offset = ulong(row) * K + j;
    if (uint(targets[indices[offset]]) == code) {
        float2 w = WEIGHTED ? vote_weight(distances[offset], minimum, zero, SQUARED) : float2(1.0f, 0.0f);
        score = vote_add(score, w);
    }
}
high[i] = score.x;
low[i] = score.y;
""",
)

_parallel_scores = mx.fast.metal_kernel(
    name="mlxlearn_neighbor_parallel_scores",
    input_names=["indices", "targets", "distances"],
    output_names=["high", "low"],
    header=_HEADER,
    source=r"""
uint lane = thread_position_in_threadgroup.x % 32;
ulong i = ulong(thread_position_in_grid.x) / 32;
if (i >= ulong(N) * C) return;
uint row = uint(i / C), code = uint(i % C);
float minimum = INFINITY;
bool zero = false;
if (WEIGHTED) {
    for (uint j = lane; j < K; j += 32) {
        float d = distances[ulong(row) * K + j];
        zero = zero || d == 0.0f;
        if (d > 0.0f) minimum = min(minimum, d);
    }
    minimum = simd_min(minimum);
    zero = simd_any(zero);
}
float2 score = float2(0.0f);
for (uint j = lane; j < K; j += 32) {
    ulong offset = ulong(row) * K + j;
    if (uint(targets[indices[offset]]) == code) {
        float2 w = WEIGHTED ? vote_weight(distances[offset], minimum, zero, SQUARED) : float2(1.0f, 0.0f);
        score = vote_add(score, w);
    }
}
for (uint offset = 16; offset > 0; offset /= 2) {
    float2 next = float2(simd_shuffle_down(score.x, offset), simd_shuffle_down(score.y, offset));
    if (lane < offset) score = vote_add(score, next);
}
if (lane == 0) { high[i] = score.x; low[i] = score.y; }
""",
)

_normalize = mx.fast.metal_kernel(
    name="mlxlearn_neighbor_normalize_votes",
    input_names=["high", "low"],
    output_names=["probabilities", "predictions"],
    header=_HEADER,
    source=r"""
uint row = thread_position_in_grid.x;
if (row >= N) return;
float2 total = float2(0.0f), best_score = float2(-1.0f, 0.0f);
uint best = 0;
for (uint c = 0; c < C; ++c) {
    ulong i = ulong(row) * C + c;
    float2 score = float2(high[i], low[i]);
    total = vote_add(total, score);
    if (vote_greater(score, best_score)) { best = c; best_score = score; }
}
if (!(total.x > 0.0f) || !isfinite(total.x)) {
    for (uint c = 0; c < C; ++c) probabilities[ulong(row) * C + c] = NAN;
    predictions[row] = -1;
    return;
}
for (uint c = 0; c < C; ++c) {
    ulong i = ulong(row) * C + c;
    probabilities[i] = vote_ratio(float2(high[i], low[i]), total);
}
predictions[row] = int(best);
""",
)


def _cpu_votes(indices, targets, n_classes, distances, squared_distances):
    """CPU scatter has a fixed order; float64 preserves close weighted votes."""
    codes = targets[indices]
    if distances is None:
        weights = mx.ones(indices.shape, mx.float64)
    else:
        values = distances.astype(mx.float64)
        if squared_distances:
            values = mx.sqrt(values)
        zero = values == 0
        has_zero = mx.any(zero, axis=1, keepdims=True)
        minimum = mx.min(mx.where(zero, float("inf"), values), axis=1, keepdims=True)
        relative = mx.where(mx.isinf(values), 0, minimum / mx.where(zero, 1, values))
        weights = mx.where(has_zero, zero.astype(mx.float64), relative)
    scores = mx.zeros((indices.shape[0], n_classes), mx.float64)
    rows = mx.broadcast_to(mx.arange(indices.shape[0])[:, None], indices.shape)
    scores = scores.at[rows, codes].add(weights)
    total = mx.sum(scores, axis=1, keepdims=True)
    probabilities = (scores / total).astype(mx.float32)
    predictions = mx.where((total[:, 0] > 0) & mx.isfinite(total[:, 0]),
                           mx.argmax(scores, axis=1).astype(mx.int32), -1)
    return probabilities, predictions


def _votes(indices, targets, n_classes, distances, squared_distances):
    n, k = indices.shape
    if n == 0:
        return mx.zeros((0, n_classes), mx.float32), mx.zeros((0,), mx.int32)
    if mx.default_device() == mx.cpu:
        return _cpu_votes(indices, targets, n_classes, distances, squared_distances)
    inputs = [indices, targets, distances if distances is not None else mx.zeros((1,), mx.float32)]
    template = [("N", n), ("K", k), ("C", n_classes), ("WEIGHTED", distances is not None),
                ("SQUARED", squared_distances)]
    outputs = {"output_shapes": [(n, n_classes), (n,)], "output_dtypes": [mx.float32, mx.int32]}
    if n_classes <= 32 and k <= 64:
        return _fused(
            inputs=inputs, template=template, grid=(n, 1, 1), threadgroup=(128, 1, 1), **outputs,
        )
    kernel, lanes = (_parallel_scores, 32) if k > 64 else (_class_scores, 1)
    high, low = kernel(
        inputs=inputs, template=template, grid=(n * n_classes * lanes, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(n, n_classes)] * 2, output_dtypes=[mx.float32] * 2,
    )
    return _normalize(
        inputs=[high, low], template=[("N", n), ("C", n_classes)], grid=(n, 1, 1),
        threadgroup=(128, 1, 1), **outputs,
    )


def class_probabilities(indices, targets, n_classes, distances=None, *, squared_distances=False):
    """Return deterministic float32 class probabilities on the current stream."""
    return _votes(indices, targets, n_classes, distances, squared_distances)[0]


def class_predictions(indices, targets, n_classes, distances=None, *, squared_distances=False):
    """Choose the largest compensated vote, retaining differences below one ULP."""
    return _votes(indices, targets, n_classes, distances, squared_distances)[1]
