"""Certified neighbor scans using compact training storage and short local lists.

Half precision is only an internal search filter. Measured quantization errors
produce conservative distance lower bounds; returned distances use the original
float32 compensated arithmetic. Any uncertain query runs the complete exact scan.
Overlapping rounding intervals use exact original-coordinate comparison; only
true distance ties use index order. No query result or user buffer is cached.
"""

from functools import lru_cache

import mlx.core as mx

from ._neighbors_fused import _HEADER
from ._neighbors_precision import PRECISION_HEADER

_PREPARE = r"""
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
const int row = int(thread_position_in_grid.x);
if (row >= shape[0]) return;
float4 sums = float4(0.0f);
for (int feature = 0; feature < D; feature += 4) {
    const float4 original = float4(*reinterpret_cast<device const packed_float4*>(
        train + size_t(row) * D + feature));
    const half4 packed = half4(original);
    *reinterpret_cast<device packed_half4*>(
        packed_train + size_t(row) * D + feature) = packed_half4(packed);
    const float4 error = original - float4(packed);
    sums = metal::fma(error, error, sums);
}
errors[row] = (sums.x + sums.y) + (sums.z + sums.w);
"""


def _scan_source():
    """Emit the single bounded specialization: two queries, three keys per lane."""
    lines = [r"""
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
const uint lane = thread_index_in_simdgroup;
const uint task = thread_position_in_grid.x / 32;
const int n = shape[0], nq = shape[1], parts = shape[2];
const int query_base = int(task / uint(parts)) * 2;
if (query_base >= nq) return;
const int part = int(task % uint(parts));
const int begin = int((size_t(part) * n) / parts);
const int end = int((size_t(part + 1) * n) / parts);
const int sentinel = 2147483647;
"""]
    for query in range(2):
        for slot in range(3):
            lines.append(f"uint key{query}_{slot} = 0xffffffffu;")
    lines.append(r"""
for (int reference = begin + int(lane); reference < end; reference += 32) {
    float sum0 = 0.0f, sum1 = 0.0f;
    for (int feature = 0; feature < D; feature += 4) {
        const float4 training = float4(*reinterpret_cast<device const packed_half4*>(
            packed_train + size_t(reference) * D + feature));
        const float4 query0 = float4(*reinterpret_cast<device const packed_float4*>(
            queries + size_t(query_base) * D + feature));
        const float4 query1 = query_base + 1 < nq
            ? float4(*reinterpret_cast<device const packed_float4*>(
                queries + size_t(query_base + 1) * D + feature))
            : float4(0.0f);
""")
    for component in "xyzw":
        for query in range(2):
            lines.append(f"""{{
            const float delta = query{query}.{component} - training.{component};
            sum{query} = metal::fma(delta, delta, sum{query});
        }}""")
    lines.append("    }")
    for query in range(2):
        lines.append(f"""
    if (query_base + {query} < nq &&
        (!EXCLUDE_SELF || reference != query_base + {query} + shape[3])) {{
        const float error = errors[reference];
        const float coarse = sum{query};
        // Young's inequality, with a=2**-11, relates the half-storage
        // squared distance to the original one. The wider coefficients
        // also cover scalar FP32 accumulation, Kahan rounding and FTZ.
        const bool safe = metal::isfinite(coarse) && coarse <= 0x1p110f &&
                          metal::isfinite(error) && error <= 0x1p110f;
        const float lower = safe ? 0.9993f * coarse - 2050.0f * error - 1e-30f
                                 : -INFINITY;
        const uint bits = as_type<uint>(lower);
        const uint ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
        // Dropping the low ordered bits rounds the lower bound downward.
        // Launch geometry guarantees at most 256 references per lane.
        const uint prefix = metal::isfinite(lower) ? (ordered & 0xffffff00u) : 0u;
        const uint position = uint((reference - begin - int(lane)) / 32);
        const uint key = prefix | position;
        if (key < key{query}_2) {{
            key{query}_2 = key;
            if (key < key{query}_1) {{
                key{query}_2 = key{query}_1; key{query}_1 = key;
                if (key < key{query}_0) {{
                    key{query}_1 = key{query}_0; key{query}_0 = key;
                }}
            }}
        }}
    }}""")
    lines.append("}")
    for query in range(2):
        lines.append(f"""
if (query_base + {query} < nq) {{
    // Keep two real candidates per lane; the third key bounds every
    // omitted candidate. A sentinel index carries this bound through
    // the ordinary sorted-list merge without another output buffer.
    const uint barrier_key = key{query}_2;
    const uint barrier_ordered = barrier_key & 0xffffff00u;
    const uint barrier_bits = (barrier_ordered & 0x80000000u)
        ? (barrier_ordered ^ 0x80000000u) : ~barrier_ordered;
    const float local_barrier = barrier_key == 0xffffffffu ? INFINITY
        : (barrier_ordered < 0x00800000u ? -INFINITY : as_type<float>(barrier_bits));
    float barrier = metal::simd_min(local_barrier);
    key{query}_2 = 0xffffffffu;
    for (int slot = 0; slot < C; ++slot) {{
        const uint key = key{query}_0;
        const uint ordered = key & 0xffffff00u;
        const uint bits = (ordered & 0x80000000u)
            ? (ordered ^ 0x80000000u) : ~ordered;
        const float head = key == 0xffffffffu ? INFINITY
            : (ordered < 0x00800000u ? -INFINITY : as_type<float>(bits));
        const int index = key == 0xffffffffu ? sentinel
            : begin + int(lane) + 32 * int(key & 255u);
        const float best = metal::simd_min(head);
        const int best_index = metal::simd_min(head == best ? index : sentinel);
        const bool inject_barrier = barrier < best;
        if (lane == uint(slot)) {{
            const size_t address = (size_t(query_base + {query}) * parts + part) * C + slot;
            out_distances[address] = inject_barrier ? barrier : best;
            out_indices[address] = inject_barrier ? sentinel : best_index;
        }}
        if (inject_barrier) barrier = INFINITY;
        else if (index == best_index) {{
            key{query}_0 = key{query}_1;
            key{query}_1 = key{query}_2;
            key{query}_2 = 0xffffffffu;
        }}
    }}
}}""")
    return "\n".join(lines)


_FINISH = r"""
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
const uint lane = thread_index_in_simdgroup;
const int query = int(thread_position_in_grid.x / 32);
const int n = shape[0], nq = shape[1], parts = shape[2];
if (query >= nq) return;
const int sentinel = 2147483647;
int cursors[LISTS];
for (int list = 0; list < LISTS; ++list) cursors[list] = 0;
int candidate = sentinel;
float omitted_lower = INFINITY;
float last_lower = -INFINITY;
for (int slot = 0; slot < C; ++slot) {
    float head = INFINITY;
    int index = sentinel, chosen = -1;
    #pragma clang loop unroll(full)
    for (int list = 0; list < LISTS; ++list) {
        const int part = int(lane) + list * 32;
        const int cursor = cursors[list];
        if (part < parts && cursor < C) {
            const size_t address = (size_t(query) * parts + part) * C + cursor;
            const float value = partial_distances[address];
            const int reference = partial_indices[address];
            if (neighbor_precedes(value, reference, head, index)) {
                head = value; index = reference; chosen = list;
            }
        }
    }
    const float best = metal::simd_min(head);
    const int best_index = metal::simd_min(head == best ? index : sentinel);
    if (lane == uint(slot)) candidate = best_index;
    if (best_index == sentinel) omitted_lower = metal::min(omitted_lower, best);
    last_lower = best;
    // Sentinels represent distinct barriers. Advance only a winning head;
    // comparing the index alone could also consume a larger barrier.
    if (chosen >= 0 && head == best && index == best_index) ++cursors[chosen];
}
omitted_lower = metal::min(omitted_lower, last_lower);
const bool valid = lane < uint(C) && candidate >= 0 && candidate < n;
const uint real_count = metal::simd_sum(uint(valid));
float exact = INFINITY;
if (valid) {
    exact = 0.0f;
    float correction = 0.0f;
    for (int feature = 0; feature < D; ++feature) {
        const float delta = queries[size_t(query) * D + feature]
                          - train[size_t(candidate) * D + feature];
        const float adjusted = delta * delta - correction;
        const float total = exact + adjusted;
        correction = metal::isfinite(total) ? (total - exact) - adjusted : 0.0f;
        exact = total;
    }
}
int exact_index = valid ? candidate : sentinel;
float cutoff = INFINITY;
for (int slot = 0; slot < K; ++slot) {
    const auto winner = precise_simd_min<D>(exact, exact_index, train, queries, query, n);
    const int best_index = winner.index;
    const float best = winner.distance;
    if (lane == uint(slot)) {
        out_distances[size_t(query) * K + slot] = best;
        out_indices[size_t(query) * K + slot] = best_index;
    }
    cutoff = best;
    if (exact_index == best_index) { exact = INFINITY; exact_index = sentinel; }
}
// The filter bounds the compensated FP32 value. Convert that barrier to a
// real-distance lower endpoint before comparing it with the selected point's
// upper endpoint; a rounded tie must never hide an omitted closer candidate.
const float omitted_exact_lower = omitted_lower == INFINITY ? INFINITY
    : precise_distance_lower<D>(omitted_lower);
const bool accepted = real_count >= uint(K) && metal::isfinite(cutoff)
                   && omitted_exact_lower > precise_distance_upper<D>(cutoff);
if (lane == 0) certified[query] = accepted;
if (accepted) return;
// Uncertain rows redo the complete search with the original arithmetic.
float distances[K];
int indices[K];
for (int slot = 0; slot < K; ++slot) {
    distances[slot] = INFINITY; indices[slot] = sentinel;
}
for (int reference = int(lane); reference < n; reference += 32) {
    if (EXCLUDE_SELF && reference == query + shape[3]) continue;
    float distance = 0.0f, correction = 0.0f;
    for (int feature = 0; feature < D; ++feature) {
        const float delta = queries[size_t(query) * D + feature]
                          - train[size_t(reference) * D + feature];
        const float adjusted = delta * delta - correction;
        const float total = distance + adjusted;
        correction = metal::isfinite(total) ? (total - distance) - adjusted : 0.0f;
        distance = total;
    }
    if (precise_neighbor_precedes<D>(distance, reference, distances[K - 1], indices[K - 1],
                                    train, queries, query, n)) {
        int slot = K - 1;
        while (slot > 0 && precise_neighbor_precedes<D>(
            distance, reference, distances[slot - 1], indices[slot - 1],
            train, queries, query, n)) {
            distances[slot] = distances[slot - 1];
            indices[slot] = indices[slot - 1];
            --slot;
        }
        distances[slot] = distance; indices[slot] = reference;
    }
}
int cursor = 0;
for (int slot = 0; slot < K; ++slot) {
    const float head = cursor < K ? distances[cursor] : INFINITY;
    const int index = cursor < K ? indices[cursor] : sentinel;
    const auto winner = precise_simd_min<D>(head, index, train, queries, query, n);
    const int best_index = winner.index;
    const float best = winner.distance;
    if (lane == uint(slot)) {
        out_distances[size_t(query) * K + slot] = best;
        out_indices[size_t(query) * K + slot] = best_index;
    }
    if (index == best_index && cursor < K) ++cursor;
}
"""


@lru_cache(maxsize=1)
def _prepare_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_compact_neighbors_prepare",
        input_names=["train", "shape"],
        output_names=["packed_train", "errors"],
        source=_PREPARE,
    )


@lru_cache(maxsize=1)
def _scan_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_compact_neighbors_scan",
        input_names=["packed_train", "queries", "errors", "shape"],
        output_names=["out_distances", "out_indices"],
        source=_scan_source(),
        header=_HEADER,
    )


@lru_cache(maxsize=1)
def _finish_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_compact_neighbors_finish",
        input_names=["partial_distances", "partial_indices", "train", "queries", "shape"],
        output_names=["out_distances", "out_indices", "certified"],
        source=_FINISH,
        header=_HEADER + PRECISION_HEADER,
    )


def prepare_compact(train):
    """Build lazy, owned half storage and per-row squared quantization errors."""
    n, features = train.shape
    return tuple(_prepare_kernel()(
        inputs=[train, mx.array([n], mx.int32)],
        template=[("D", features)],
        grid=(((n + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n, features), (n,)],
        output_dtypes=[mx.float16, mx.float32],
        stream=mx.gpu,
    ))


def _partitions(n_train, n_query):
    """Bound local references to eight bits while exposing enough SIMD groups."""
    minimum = (n_train + 8191) // 8192
    target = max(minimum, (8192 + n_query - 1) // n_query)
    power = 1 << (max(1, target) - 1).bit_length()
    return min(256, max(1, n_train // 256), power)


def compact_neighbors(
    train, queries, k, prepared, *, exclude_self=False, query_offset=0,
    return_certificate=False,
):
    """Return exact squared distances and indices from already validated inputs.

    Internal caller contract: float32 arrays, 1<=k<=8, positive D<=32 divisible
    by four, N<=2**21, and a valid self-query offset. The fitted index owns
    ``train`` and caches only the lazy ``prepared`` training representation.
    """
    n, features = train.shape
    nq = queries.shape[0]
    if nq == 0:
        result = (mx.zeros((0, k), mx.float32), mx.zeros((0, k), mx.int32))
        return (*result, mx.zeros((0,), mx.bool_)) if return_certificate else result
    parts = _partitions(n, nq)
    carry = k + 4
    task_count = ((nq + 1) // 2) * parts
    shape = mx.array([n, nq, parts, query_offset], mx.int32)
    partial_distances, partial_indices = _scan_kernel()(
        inputs=[prepared[0], queries, prepared[1], shape],
        template=[("C", int(carry)), ("D", features), ("EXCLUDE_SELF", bool(exclude_self))],
        grid=(((task_count * 32 + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(nq, parts, carry)] * 2,
        output_dtypes=[mx.float32, mx.int32],
        stream=mx.gpu,
    )
    result = tuple(_finish_kernel()(
        inputs=[partial_distances, partial_indices, train, queries, shape],
        template=[("K", int(k)), ("C", int(carry)), ("D", features),
                  ("EXCLUDE_SELF", bool(exclude_self)), ("LISTS", (parts + 31) // 32)],
        grid=(((nq * 32 + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(nq, k), (nq, k), (nq,)],
        output_dtypes=[mx.float32, mx.int32, mx.bool_],
        stream=mx.gpu,
    ))
    return result if return_certificate else result[:2]
