"""Metal screens for k-means++ seeding.

Each k-means++ step compares every row with a few candidate centers, but only
rows nearer to a candidate than to their current center change anything. The
screens prove most (candidate, row) pairs unchanged in float32 and return the
rest for the host to evaluate with its reference float64 arithmetic.

:func:`candidate_screen` reads every row. :func:`projected_screen` first
bounds each distance from below in a low-dimensional image of the rows and
reads full rows only for the pairs that bound cannot settle.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

# A pair is proved unchanged when
#
#     s - slack >= high,    slack = s*e1 + (|x|^2 + |c|^2)*e2 + tiny,
#
# where ``s`` is the float32 sum of squared differences and ``high`` an upper
# bound of the row's closest distance. ``e1`` bounds the relative error of
# ``s``: the terms are nonnegative, each a rounded square of a rounded
# difference followed by at most D rounded additions, so the error is below
# (D + 3) * 2^-24; the bound doubles that with a margin for the slack
# arithmetic itself. ``e2`` bounds the error of the host's float64 expansion
# |x|^2 - 2 x.c + |c|^2, at most 2 (D + 2) 2^-53 (|x|^2 + |c|^2), doubled; the
# norm inputs are upper bounds. ``tiny`` covers products lost to underflow.
# The same slack brackets the reference's float32 distance of an unproved
# pair: it lies between max(s - slack, 0) and s + slack.
_HEADER = r"""
inline float kpp_expansion(uint D) { return float(4 * (D + 2)) * 0x1p-53f; }
inline float kpp_tiny(uint D) { return float(4 * (D + 16)) * FLT_MIN; }
inline float kpp_slack(float s, float norms, uint D) {
    return s * (float(2 * (D + 16)) * 0x1p-24f) + norms * kpp_expansion(D) + kpp_tiny(D);
}
inline float4 kpp_load(const device float* p) { return float4(*(const device packed_float4*)p); }
"""

# One thread per row accumulates the direct differences to every candidate, so
# the input is read once per step. Unproved pairs store their bracket, capped
# by the row's bounds, and set the candidate's bit in the row's mask;
# ``touched`` flags rows with any such pair, as bytes NumPy scans quickly.
_SCREEN = r"""
    const uint N = shape[0], D = shape[1];
    const uint i = thread_position_in_grid.x;
    if (i >= N) return;
    const device float* row = x + size_t(i) * D;
    float4 wide[T];
    float narrow[T];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < T; ++t) { wide[t] = float4(0.0f); narrow[t] = 0.0f; }
    uint f = 0;
    for (; f + 4 <= D; f += 4) {
        const float4 value = kpp_load(row + f);
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            const float4 delta = value - kpp_load(x + size_t(ids[t]) * D + f);
            wide[t] += delta * delta;
        }
    }
    for (; f < D; ++f) {
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            const float delta = row[f] - x[size_t(ids[t]) * D + f];
            narrow[t] += delta * delta;
        }
    }
    const float below = low[i], above = high[i];
    uint open = 0;
    #pragma clang loop unroll(full)
    for (uint t = 0; t < T; ++t) {
        const float s = (wide[t].x + wide[t].y) + (wide[t].z + wide[t].w) + narrow[t];
        const float slack = kpp_slack(s, norms[i] + norms[ids[t]], D);
        const bool unchanged = s - slack >= above;
        const size_t at = size_t(t) * N + i;
        table[at] = unchanged ? below : min(below, max(s - slack, 0.0f));
        table[size_t(T) * N + at] = unchanged ? above : min(above, s + slack);
        open |= unchanged ? 0u : (1u << t);
    }
    mask[i] = open;
    touched[i] = open != 0u;
"""

# The projected rows z are float32 images of the rows under a linear map of
# spectral norm at most sigma, each within eps of its exact image. Then
#
#     |x - c| >= (|z_x - z_c| - eps_x - eps_c) / sigma,
#
# and the reference leaves the row unchanged when |x - c|^2 >= limit, with
# limit = high + (|x|^2 + |c|^2)*e2 + tiny as above. The first test compares
# the float32 sum q of squared image differences, reduced by its own relative
# error, with (sqrt(limit)*(1 + 2^-19) + eps_x + eps_c)^2. The factor on the
# root covers its rounding and sigma, which the host keeps below 1 + 2^-22.
#
# Pairs that fail it need the full rows. Each thread group of G rows packs its
# failed pairs into one list, shares them out evenly among its threads, and
# settles them as the dense screen does, storing their table entries. A
# group's work thus follows its number of failed pairs: with none left, only
# the images are read, and with all of them the kernel does the dense screen's
# work.
_PROJECTED = r"""
    const uint N = shape[0], D = shape[1], M = shape[2];
    const uint tid = thread_index_in_threadgroup;
    const uint first = threadgroup_position_in_grid.x * G;
    const uint i = first + tid;
    const bool valid = i < N;
    threadgroup ushort counts[G];
    threadgroup ushort list[G * T];
    threadgroup uchar verdict[G * T];
    uint failed = 0;
    if (valid) {
        const device float* row = z + size_t(i) * M;
        float4 sums[T];
        float largest = 0.0f;
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            sums[t] = float4(0.0f);
            largest = max(largest, norms[ids[t]]);
        }
        for (uint f = 0; f < M; f += 4) {
            const float4 value = kpp_load(row + f);
            #pragma clang loop unroll(full)
            for (uint t = 0; t < T; ++t) {
                const float4 delta = value - kpp_load(z + size_t(ids[t]) * M + f);
                sums[t] += delta * delta;
            }
        }
        const float limit = high[i] + (norms[i] + largest) * kpp_expansion(D) + kpp_tiny(D);
        const float reach = sqrt(limit) * (1.0f + 0x1p-19f) + eps[i];
        const float shrink = 1.0f - float(2 * (M + 16)) * 0x1p-24f;
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            const float q = ((sums[t].x + sums[t].y) + (sums[t].z + sums[t].w)) * shrink;
            const float bound = (reach + eps[ids[t]]) * (1.0f + 0x1p-21f);
            failed |= q >= bound * bound ? 0u : (1u << t);
        }
    }
    counts[tid] = ushort(popcount(failed));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint offset = 0, total = 0;
    for (uint u = 0; u < G; ++u) {
        const uint c = counts[u];
        offset += u < tid ? c : 0u;
        total += c;
    }
    if (tid == 0) settled[threadgroup_position_in_grid.x] = total;
    uint at = offset;
    for (uint t = 0; t < T; ++t)
        if ((failed >> t) & 1u) list[at++] = ushort((tid << 5) | t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < total; p += G) {
        const uint pair = list[p];
        const uint r = first + (pair >> 5), t = pair & 31u;
        const device float* row = x + size_t(r) * D;
        const device float* other = x + size_t(ids[t]) * D;
        float4 wide = float4(0.0f), second = float4(0.0f);
        uint f = 0;
        for (; f + 8 <= D; f += 8) {
            const float4 a = kpp_load(row + f) - kpp_load(other + f);
            const float4 b = kpp_load(row + f + 4) - kpp_load(other + f + 4);
            wide += a * a;
            second += b * b;
        }
        float narrow = 0.0f;
        for (; f < D; ++f) {
            const float delta = row[f] - other[f];
            narrow += delta * delta;
        }
        wide += second;
        const float s = (wide.x + wide.y) + (wide.z + wide.w) + narrow;
        const float slack = kpp_slack(s, norms[r] + norms[ids[t]], D);
        const float below = low[r], above = high[r];
        const bool unchanged = s - slack >= above;
        const size_t entry = size_t(t) * N + r;
        table[entry] = unchanged ? below : min(below, max(s - slack, 0.0f));
        table[size_t(T) * N + entry] = unchanged ? above : min(above, s + slack);
        verdict[p] = unchanged ? uchar(0) : uchar(1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid) return;
    const float below = low[i], above = high[i];
    uint open = 0;
    at = offset;
    #pragma clang loop unroll(full)
    for (uint t = 0; t < T; ++t) {
        if ((failed >> t) & 1u) {
            // Settled above, by the thread that also stored its table entries.
            open |= verdict[at++] == uchar(0) ? 0u : (1u << t);
        } else {
            const size_t entry = size_t(t) * N + i;
            table[entry] = below;
            table[size_t(T) * N + entry] = above;
        }
    }
    mask[i] = open;
    touched[i] = open != 0u;
"""

_GROUP = 256

# The error bounds assume IEEE rounding of every float32 operation.
_SAFE = {"math_mode": "safe"}


@lru_cache(None)
def _kernel(name):
    if name == "screen":
        return mx.fast.metal_kernel(
            name="kmeans_plusplus_screen", input_names=["x", "low", "high", "norms", "ids", "shape"],
            output_names=["table", "mask", "touched"], source=_SCREEN, header=_HEADER,
            compile_options=_SAFE)
    if name == "projected":
        return mx.fast.metal_kernel(
            name="kmeans_plusplus_projected",
            input_names=["x", "z", "eps", "low", "high", "norms", "ids", "shape"],
            output_names=["table", "mask", "touched", "settled"], source=_PROJECTED,
            header=_HEADER, compile_options=_SAFE)
    raise ValueError(name)


@lru_cache(maxsize=1)
def _safe_math():
    try:
        _kernel("screen")
    except TypeError:
        # Earlier supported MLX releases cannot select Metal's safe math mode.
        return False
    return True


def screen_supported(n, d, trials):
    """Whether the screens cover this problem on this MLX release."""
    # Inputs shorter than eight elements are bound in another address space.
    return n >= 8 and trials <= 32 and n <= 2**32 - 1 and _safe_math()


def candidate_ids(ids):
    """Candidate row indices as the screens read them: uint32, at least eight."""
    padded = [int(i) for i in ids] + [0] * max(0, 8 - len(ids))
    return mx.array(padded, mx.uint32)


def _shape(*values):
    return mx.array([*values, *([0] * (8 - len(values)))], mx.uint32)


def candidate_screen(x, low, high, norms, ids, trials):
    """Screen every (candidate, row) pair against the row's closest distance.

    ``x`` is the resident (n, d) float32 input and ``ids`` the ``trials``
    candidates' row indices from :func:`candidate_ids`. ``low`` and ``high``
    bound each row's squared distance to its nearest chosen center, and
    ``norms`` are upper bounds of the squared row norms.

    Returns ``table`` (2, trials, n), ``mask`` (n,) and ``touched`` (n,).
    ``table[0, t, i]`` and ``table[1, t, i]`` bound row ``i``'s closest
    distance once candidate ``t`` is added. Where the candidate provably
    leaves the row unchanged they are ``low[i]`` and ``high[i]``; otherwise
    bit ``t`` of ``mask[i]`` is set and they bracket the smaller of the row's
    bounds and the reference's float32 distance to the candidate.
    ``touched[i]`` is 1 where ``mask[i]`` is nonzero.
    """
    n, d = x.shape
    return _kernel("screen")(
        inputs=[x, low, high, norms, ids, _shape(n, d)], template=[("T", trials)],
        grid=(n, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(2, trials, n), (n,), (n,)],
        output_dtypes=[mx.float32, mx.uint32, mx.uint8], stream=mx.gpu)


def projected_screen(x, z, eps, low, high, norms, ids, trials):
    """Screen as :func:`candidate_screen`, reading full rows only where needed.

    ``z`` (n, m) holds float32 images of the rows under a linear map whose
    spectral norm is at most 1 + 2^-22, and ``eps`` (n,) bounds each image's
    distance from its exact value; ``m`` is a multiple of four. The first
    three outputs have the meaning of :func:`candidate_screen`'s: a pair is
    left open only if the full rows cannot prove it unchanged either. The
    fourth, ``settled``, counts per thread group the pairs that needed the
    full rows.
    """
    n, d = x.shape
    groups = -(-n // _GROUP)
    return _kernel("projected")(
        inputs=[x, z, eps, low, high, norms, ids, _shape(n, d, z.shape[1])],
        template=[("T", trials), ("G", _GROUP)],
        grid=(groups * _GROUP, 1, 1), threadgroup=(_GROUP, 1, 1),
        output_shapes=[(2, trials, n), (n,), (n,), (groups,)],
        output_dtypes=[mx.float32, mx.uint32, mx.uint8, mx.uint32], stream=mx.gpu)
