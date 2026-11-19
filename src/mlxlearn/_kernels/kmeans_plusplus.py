"""Metal screen for k-means++ seeding.

Each k-means++ step compares every row with a few candidate centers, but only
rows nearer to a candidate than to their current center change anything. The
screen evaluates all (candidate, row) squared distances in float32 and proves
most pairs unchanged; the remaining pairs are returned for the host to
evaluate with its reference float64 arithmetic.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

# One thread per row accumulates the direct differences to every candidate, so
# the input is read once per step. A pair is proved unchanged when
#
#     s - slack >= high,    slack = s*e1 + (|x|^2 + |c|^2)*e2 + tiny,
#
# where ``s`` is the float32 sum and ``high`` an upper bound of the row's
# closest distance. ``e1`` bounds the relative error of ``s``: the terms are
# nonnegative, each a rounded square of a rounded difference followed by at
# most D rounded additions, so the error is below (D + 3) * 2^-24; the bound
# doubles that with a margin for the slack arithmetic itself. ``e2`` bounds
# the error of the host's float64 expansion |x|^2 - 2 x.c + |c|^2, at most
# 2 (D + 2) 2^-53 (|x|^2 + |c|^2), doubled; the norm inputs are upper bounds.
# ``tiny`` covers products lost to underflow. Unproved pairs store -1 (stored
# distances are never negative) and set the candidate's bit in the row's mask;
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
        const float4 value = float4(*(const device packed_float4*)(row + f));
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            const float4 delta =
                value - float4(*(const device packed_float4*)(candidates + size_t(t) * D + f));
            wide[t] += delta * delta;
        }
    }
    for (; f < D; ++f) {
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            const float delta = row[f] - candidates[size_t(t) * D + f];
            narrow[t] += delta * delta;
        }
    }
    const float below = low[i], above = high[i];
    const float own = norms[i];
    const float e1 = float(2 * (D + 16)) * 0x1p-24f;
    const float e2 = float(4 * (D + 2)) * 0x1p-53f;
    const float tiny = float(4 * (D + 16)) * FLT_MIN;
    uint open = 0;
    #pragma clang loop unroll(full)
    for (uint t = 0; t < T; ++t) {
        const float s = (wide[t].x + wide[t].y) + (wide[t].z + wide[t].w) + narrow[t];
        const float slack = s * e1 + (own + candidate_norms[t]) * e2 + tiny;
        const bool unchanged = s - slack >= above;
        const size_t at = size_t(t) * N + i;
        table[at] = unchanged ? below : -1.0f;
        table[size_t(T) * N + at] = unchanged ? above : -1.0f;
        open |= unchanged ? 0u : (1u << t);
    }
    mask[i] = open;
    touched[i] = open != 0u;
"""


@lru_cache(None)
def _screen_kernel():
    return mx.fast.metal_kernel(
        name="kmeans_plusplus_screen",
        input_names=["x", "candidates", "low", "high", "norms", "candidate_norms", "shape"],
        output_names=["table", "mask", "touched"], source=_SCREEN)


def screen_supported(n, d, trials):
    """Whether the screen's inputs and per-row candidate mask cover this problem."""
    # Inputs shorter than eight elements are bound in another address space.
    return n >= 8 and trials * d >= 8 and trials <= 32 and n <= 2**32 - 1


def candidate_screen(x, candidates, low, high, norms, candidate_norms):
    """Screen every (candidate, row) pair against the row's closest distance.

    ``x`` is the resident (n, d) float32 input and ``candidates`` the (trials,
    d) candidate rows. ``low`` and ``high`` bound each row's squared distance
    to its nearest chosen center. ``norms`` and ``candidate_norms`` are upper
    bounds of the squared row norms, the latter padded to at least eight
    elements.

    Returns ``table`` (2, trials, n), ``mask`` (n,) and ``touched`` (n,).
    Where candidate ``t`` provably leaves row ``i`` unchanged, ``table[0, t,
    i]`` is ``low[i]`` and ``table[1, t, i]`` is ``high[i]``; otherwise both
    are -1 and bit ``t`` of ``mask[i]`` is set. ``touched[i]`` is 1 where
    ``mask[i]`` is nonzero.
    """
    n, d = x.shape
    trials = candidates.shape[0]
    shape = mx.array([n, d, 0, 0, 0, 0, 0, 0], mx.uint32)
    return _screen_kernel()(
        inputs=[x, candidates, low, high, norms, candidate_norms, shape], template=[("T", trials)],
        grid=(n, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(2, trials, n), (n,), (n,)],
        output_dtypes=[mx.float32, mx.uint32, mx.uint8], stream=mx.gpu)
