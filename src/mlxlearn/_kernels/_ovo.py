"""SVC's one-vs-one subproblems, solved concurrently on the GPU in one Metal kernel.

A multiclass ``SVC`` fit solves ``K(K−1)/2`` independent binary duals, and the host
SMO loop (``_solvers/smo.py``) solves them one after another, a few tens of
microseconds of NumPy per iteration whatever the subproblem's size. This kernel
gives each pair a threadgroup of its own and runs every pair's SMO at once, one
launch for all of them, so a fit costs about as long as its slowest pair takes on
the GPU rather than the sum of every pair on the host.

**The method** is LIBSVM's, step for step: sequential minimal optimization (Platt,
1998) with the two-threshold optimality test of Keerthi, Shevade, Bhattacharyya and
Murthy (2001), second-order working-set selection (Fan, Chen and Lin, JMLR 6, 2005)
and the stopping rule, box clipping and one-vs-one decomposition of Chang and Lin's
LIBSVM paper (2011; Hsu and Lin, 2002, for one-vs-one). Running SMO's selection as a
parallel reduction over a subproblem's samples on a GPU follows Catanzaro, Sundaram
and Keutzer ("Fast support vector machine training and classification on graphics
processors", ICML 2008); giving each independent subproblem its own group of threads
in one launch is the step this module adds. Written for mlxlearn from those
descriptions. Per pair and per step: ``i`` is the last maximum of ``−y∇f`` over
``I_up``; the stop test runs on the whole set, with no shrinking; ``j`` is the last
maximum of the second-order gain over ``I_low``; the analytic two-variable step is
clipped to the box branch for branch as ``smo._pair_update`` clips it; and the
gradient is updated from the two kernel rows. Each thread holds :data:`ROWS_PER_THREAD`
samples in registers; the maxima are SIMD-group reductions, merged across SIMD
groups through threadgroup memory.

**Precision.** The dual variables and the gradient are double-float — a float32
value plus a float32 residual, about 48 bits — with error-free sums and products
(Dekker, "A floating-point technique for extending the available precision",
1971; Knuth, TAOCP vol. 2, §4.2.2) and ``fma`` for a product's exact error. Only
the working-set key, the gain that ranks the ``j`` candidates, is float32, so two
candidates whose gains agree to float32's precision can be ranked differently from
the host's float64 ranking: a different path to the same optimum. The kernel is not
trusted with the answer: every pair's result is held to the host solver's stop by
``smo.finish_c_svc`` — gradient recomputed in float64, full-set KKT test, the host
loop finishing any pair that fails it (``docs/fp32_policy.md`` §2.5).

**Deterministic.** No result is a sum across threads: every reduction is a maximum
with ties broken by the larger index, which no order of evaluation changes, and the
update of each sample is its own thread's. So a pair's result depends on its data
alone — not on the threadgroup width, on which other pairs share its launch, or on
how its steps are split across launches. There are no atomics.

**Launches stay short.** A launch runs at most a step budget per pair —
:data:`FIRST_STEPS` in the first, then as many as about :data:`LAUNCH_SECONDS` of
the last launch's time per step allows, within :data:`MIN_STEPS`–:data:`MAX_STEPS`
— so no command buffer approaches the GPU watchdog, and the host relaunches while
any pair is unfinished, the state staying on the device. A pair that is done copies
its state through. Every launch advances every pair it runs by at least one step or
stops it, and a pair spends at most its ``max_iter``, so the relaunches are bounded;
a pair a launch neither advanced nor stopped would be a fault of the kernel's, and
is handed to the host solver's finish instead of being relaunched.

**The Gram.** A pair's kernel matrix is built on the device with exactly the MLX
operations :func:`~._kernels.kernels.kernel_matrix` evaluates for that pair — the
same products, norms and fused rbf transform, block for block — so it is the Gram the
host path would have solved against. Pairs whose rbf entries need the host's float64
recomputation (``docs/fp32_policy.md`` §2.4) take :func:`kernel_matrix` itself.
Every pair's Gram is packed into one buffer for the launch; pairs run in **waves**
whose packed Grams fit the caller's budget (:func:`waves`). Building them, the
classes' rows on the device and the build's transients keep within
``Tuning.distance_block_bytes`` (:func:`_pair_grams`), where a pair too large for it
alone takes what :func:`kernel_matrix` takes to build it on the host path.

**When it compiles.** On the first fit in a process that takes this path, and once:
:func:`pair_solver` builds the library, runs it on a fixed problem whose answer the
host checks, and serves it only if the two agree. A library never compiled on the
machine cost that first fit 70-78 ms more on an M4 Max with MLX 0.32.2 (first fits of
108-150 ms against 33-81 ms), once per machine, MLX release and kernel version; the
operating system's shader cache serves it to later processes. The check itself costs
the first fit of every process about 8 ms. MLX holds a lock that stalls the process's
other GPU work while a library builds, so nothing builds ahead of need, as for the
neighbor search's kernel.
"""

from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._common.config import get_tuning
from .._common.device import use_device
from . import _metal
from .kernels import (
    KernelSpec,
    _block_rows,
    _RbfRefiner,
    _self_entries,
    _tile_transform,
    kernel_matrix,
)

__all__ = [
    "MAX_PAIR_ROWS",
    "ClassRows",
    "PairSolver",
    "WaveResult",
    "class_rows",
    "pair_solver",
    "waves",
]

#: Samples each thread holds in registers.
ROWS_PER_THREAD = 8

#: The largest threadgroup the kernel is launched with: 16 SIMD groups.
MAX_THREADS = 512

#: The largest one-vs-one subproblem the kernel serves: a threadgroup of
#: :data:`MAX_THREADS` threads holding :data:`ROWS_PER_THREAD` samples each.
MAX_PAIR_ROWS = MAX_THREADS * ROWS_PER_THREAD

#: Steps per pair in the first launch of a wave; later launches size theirs from its
#: time per step. See the module docstring.
FIRST_STEPS = 2048
MIN_STEPS = 256
MAX_STEPS = 8192
LAUNCH_SECONDS = 0.25

#: LIBSVM's floor for a non-positive quadratic coefficient, as ``smo._TAU``.
_TAU = 1e-12

#: The most Gram entries one wave packs. The launch table carries each pair's Gram
#: offset as an int32, and MLX sizes an array in int32 too; a wave past it would wrap
#: and read other pairs' Grams. 8 GiB of float32: :func:`waves` splits before it.
MAX_WAVE_ENTRIES = 2**31 - 1

#: Stop codes a launch reports per pair.
STOP_BUDGET = 0  # the launch's step budget, or max_iter, ran out
STOP_OPTIMAL = 1  # no violating pair, or the violation below tol
STOP_NON_FINITE = 2  # a gradient or the violation was not finite
STOP_SKIPPED = 3  # not run in this launch: state copied through
STOP_STALLED = 4  # host-side: a launch neither advanced nor stopped the pair

#: Part of the kernel's name: bumped with the source, so no cached library is stale.
_SOURCE_VERSION = 1

_HEADER = r"""
#include <metal_simdgroup>

// A double-float: hi + lo, |lo| <= ulp(hi) / 2.
struct ovo_ff {
  float hi;
  float lo;
};

// Knuth's TwoSum: s + e == a + b exactly, for any a, b.
inline ovo_ff ovo_two_sum(const float a, const float b) {
  const float s = a + b;
  const float bv = s - a;
  const float av = s - bv;
  return ovo_ff{s, (a - av) + (b - bv)};
}

// Dekker's Fast2Sum: exact when |a| >= |b| or a == 0.
inline ovo_ff ovo_fast_two_sum(const float a, const float b) {
  const float s = a + b;
  return ovo_ff{s, b - (s - a)};
}

inline ovo_ff ovo_add(const ovo_ff a, const ovo_ff b) {
  const ovo_ff s = ovo_two_sum(a.hi, b.hi);
  const ovo_ff t = ovo_two_sum(a.lo, b.lo);
  const ovo_ff u = ovo_fast_two_sum(s.hi, s.lo + t.hi);
  return ovo_fast_two_sum(u.hi, u.lo + t.lo);
}

inline ovo_ff ovo_neg(const ovo_ff a) { return ovo_ff{-a.hi, -a.lo}; }

inline ovo_ff ovo_sub(const ovo_ff a, const ovo_ff b) { return ovo_add(a, ovo_neg(b)); }

// a * b for a float b; the product's rounding error is recovered exactly by fma.
inline ovo_ff ovo_mul(const ovo_ff a, const float b) {
  const float p = a.hi * b;
  const float e = metal::fma(a.lo, b, metal::fma(a.hi, b, -p));
  return ovo_fast_two_sum(p, e);
}

inline ovo_ff ovo_div(const ovo_ff a, const ovo_ff b) {
  const float q = metal::precise::divide(a.hi, b.hi);
  const ovo_ff r = ovo_sub(a, ovo_mul(b, q));
  return ovo_fast_two_sum(q, metal::precise::divide(r.hi, b.hi));
}

// g + a * x + b * y for floats x, y: the gradient update with two kernel rows.
inline ovo_ff ovo_update(const ovo_ff g, const ovo_ff a, const float x, const ovo_ff b,
                         const float y) {
  const float px = a.hi * x;
  const float ex = metal::fma(a.lo, x, metal::fma(a.hi, x, -px));
  const float py = b.hi * y;
  const float ey = metal::fma(b.lo, y, metal::fma(b.hi, y, -py));
  const ovo_ff s = ovo_two_sum(g.hi, px);
  const ovo_ff u = ovo_two_sum(s.hi, py);
  return ovo_two_sum(u.hi, (((g.lo + s.lo) + u.lo) + ex) + ey);
}

inline bool ovo_gt(const ovo_ff a, const ovo_ff b) {
  return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}

inline bool ovo_ge(const ovo_ff a, const ovo_ff b) {
  return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
}

// LIBSVM's one alpha status as set bits: 1 for I_up, 2 for I_low. At the upper bound
// is alpha >= C; at the lower bound is alpha <= 0 and not at the upper one, so a
// zero box counts as upper (smo._bound_status).
inline uchar ovo_status(const ovo_ff a, const ovo_ff c, const bool positive) {
  const bool upper = ovo_ge(a, c);
  const bool lower = !upper && a.hi <= 0.0f;
  const bool up = positive ? !upper : !lower;
  const bool low = positive ? !lower : !upper;
  return uchar(up) | uchar(low ? 2 : 0);
}

// The SIMD group's lexicographic maximum of (key_hi, key_lo, index), and the
// winning lane's payload, broadcast. Equal keys go to the larger index: LIBSVM's
// scans keep the last extremum.
template <int N>
inline void ovo_argmax(thread float& key_hi, thread float& key_lo, thread int& index,
                       thread float (&payload)[N], const uint lane) {
  const float top_hi = metal::simd_max(key_hi);
  const float top_lo = metal::simd_max(key_hi == top_hi ? key_lo : -INFINITY);
  const bool tied = key_hi == top_hi && key_lo == top_lo;
  const int top_index = metal::simd_max(tied ? index : -1);
  const int from = metal::simd_max((tied && index == top_index) ? int(lane) : -1);
  const ushort source = ushort(metal::max(from, 0));
  for (int q = 0; q < N; ++q) {
    payload[q] = metal::simd_shuffle(payload[q], source);
  }
  key_hi = top_hi;
  key_lo = top_lo;
  index = top_index;
}

inline void ovo_max(thread float& hi, thread float& lo) {
  const float top = metal::simd_max(hi);
  lo = metal::simd_max(hi == top ? lo : -INFINITY);
  hi = top;
}
"""

# One threadgroup per pair p; thread `tid` holds the pair's samples tid + e * width.
# table[8p:8p+8] = [m, first-class samples, sample offset, Gram offset, step budget].
# params = [width]; fparams = [tol hi, tol lo, tau hi, tau lo]. Per-sample arrays are
# flat over the launch's pairs; a pair's Gram is m x m, row-major, at its offset.
# info[2p:2p+2] = [steps taken, stop code].
_SOURCE = r"""
  const uint tid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint p = threadgroup_position_in_grid.x;
  const int width = params[0];
  const uint groups = uint(width) / 32u;
  const int m = table[8u * p];
  const int first = table[8u * p + 1u];
  const uint off = uint(table[8u * p + 2u]);
  const uint goff = uint(table[8u * p + 3u]);
  const int budget = table[8u * p + 4u];
  const ovo_ff tol = ovo_ff{fparams[0], fparams[1]};
  const ovo_ff tau = ovo_ff{fparams[2], fparams[3]};
  // Partial reductions, one slot per SIMD group, in two banks used in turn: a SIMD
  // group still reading one reduction's bank cannot be overwritten by the next one.
  threadgroup float shared_f[2][10][32];
  threadgroup int shared_i[2][4][32];

  if (budget <= 0) {
    for (int e = 0; e < OVO_E; ++e) {
      const int t = int(tid) + e * width;
      if (t < m) {
        a_hi[off + t] = a_hi_in[off + t];
        a_lo[off + t] = a_lo_in[off + t];
        g_hi[off + t] = g_hi_in[off + t];
        g_lo[off + t] = g_lo_in[off + t];
      }
    }
    if (tid == 0) {
      info[2u * p] = 0;
      info[2u * p + 1u] = 3;
    }
    return;
  }

  float gh[OVO_E], gl[OVO_E], ah[OVO_E], al[OVO_E], ki[OVO_E];
  uchar st[OVO_E];
  for (int e = 0; e < OVO_E; ++e) {
    const int t = int(tid) + e * width;
    gh[e] = 0.0f;
    gl[e] = 0.0f;
    ah[e] = 0.0f;
    al[e] = 0.0f;
    ki[e] = 0.0f;
    st[e] = 0;
    if (t < m) {
      gh[e] = g_hi_in[off + t];
      gl[e] = g_lo_in[off + t];
      ah[e] = a_hi_in[off + t];
      al[e] = a_lo_in[off + t];
      st[e] = ovo_status(ovo_ff{ah[e], al[e]}, ovo_ff{c_hi[off + t], c_lo[off + t]}, t < first);
    }
  }

  int steps = 0;
  int stop = 0;
  uint bank = 0;
  while (true) {
    if (steps >= budget) {
      stop = 0;
      break;
    }

    // ---- i, the last maximum of -yg over I_up; the largest yg over I_low; flags ----
    float i_hi = -INFINITY, i_lo = -INFINITY, low_hi = -INFINITY, low_lo = -INFINITY;
    int i_ix = -1;
    int flags = 0;  // 1: I_up not empty, 2: I_low not empty, 4: a gradient not finite
    float i_pay[3] = {0.0f, 0.0f, 0.0f};
    for (int e = 0; e < OVO_E; ++e) {
      const int t = int(tid) + e * width;
      if (t >= m) {
        continue;
      }
      const bool up = (st[e] & 1) != 0;
      const bool low = (st[e] & 2) != 0;
      const bool nan_g = metal::isnan(gh[e]);
      flags |= (up ? 1 : 0) | (low ? 2 : 0);
      if ((up && (nan_g || gh[e] == -INFINITY)) || (low && (nan_g || gh[e] == INFINITY))) {
        flags |= 4;
      }
      if (up && (-gh[e] > i_hi || (-gh[e] == i_hi && -gl[e] >= i_lo))) {
        i_hi = -gh[e];
        i_lo = -gl[e];
        i_ix = t;
        i_pay[0] = ah[e];
        i_pay[1] = al[e];
        i_pay[2] = t < first ? 1.0f : -1.0f;
      }
      if (low && (gh[e] > low_hi || (gh[e] == low_hi && gl[e] > low_lo))) {
        low_hi = gh[e];
        low_lo = gl[e];
      }
    }
    ovo_argmax<3>(i_hi, i_lo, i_ix, i_pay, lane);
    ovo_max(low_hi, low_lo);
    flags = metal::simd_or(flags);
    if (groups > 1u) {
      if (lane == 0) {
        shared_f[bank][0][sg] = i_hi;
        shared_f[bank][1][sg] = i_lo;
        shared_f[bank][2][sg] = low_hi;
        shared_f[bank][3][sg] = low_lo;
        for (int q = 0; q < 3; ++q) {
          shared_f[bank][4 + q][sg] = i_pay[q];
        }
        shared_i[bank][0][sg] = i_ix;
        shared_i[bank][1][sg] = flags;
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      const bool mine = lane < groups;
      i_hi = mine ? shared_f[bank][0][lane] : -INFINITY;
      i_lo = mine ? shared_f[bank][1][lane] : -INFINITY;
      low_hi = mine ? shared_f[bank][2][lane] : -INFINITY;
      low_lo = mine ? shared_f[bank][3][lane] : -INFINITY;
      for (int q = 0; q < 3; ++q) {
        i_pay[q] = mine ? shared_f[bank][4 + q][lane] : 0.0f;
      }
      i_ix = mine ? shared_i[bank][0][lane] : -1;
      flags = mine ? shared_i[bank][1][lane] : 0;
      ovo_argmax<3>(i_hi, i_lo, i_ix, i_pay, lane);
      ovo_max(low_hi, low_lo);
      flags = metal::simd_or(flags);
      bank ^= 1u;
    }

    // ---- the stop tests, in the host solver's order ----
    if ((flags & 3) != 3) {
      stop = 1;  // no violating pair exists: LIBSVM's convergence
      break;
    }
    if ((flags & 4) != 0 || i_ix < 0) {
      stop = 2;
      break;
    }
    const ovo_ff gmax = ovo_ff{i_hi, i_lo};
    const ovo_ff violation = ovo_add(gmax, ovo_ff{low_hi, low_lo});
    if (!metal::isfinite(violation.hi)) {
      stop = 2;
      break;
    }
    if (ovo_gt(tol, violation)) {
      stop = 1;
      break;
    }

    // ---- j, the last maximum of the second-order gain over I_low, from row i ----
    const int i = i_ix;
    const ovo_ff alpha_i = ovo_ff{i_pay[0], i_pay[1]};
    const float y_i = i_pay[2];
    const ovo_ff kd_i = ovo_ff{kd_hi[off + i], kd_lo[off + i]};
    const ovo_ff c_i = ovo_ff{c_hi[off + i], c_lo[off + i]};
    const uint row_i = goff + uint(i) * uint(m);
    float j_key = -INFINITY, j_zero = 0.0f;
    int j_ix = -1;
    float j_pay[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int e = 0; e < OVO_E; ++e) {
      const int t = int(tid) + e * width;
      ki[e] = t < m ? gram[row_i + uint(t)] : 0.0f;
    }
    for (int e = 0; e < OVO_E; ++e) {
      const int t = int(tid) + e * width;
      if (t >= m || (st[e] & 2) == 0) {
        continue;
      }
      // Eligible: gmax + yg_t > 0, compared exactly on the double-floats.
      if (!(gh[e] > -gmax.hi || (gh[e] == -gmax.hi && gl[e] > -gmax.lo))) {
        continue;
      }
      const float k = ki[e];
      const float kd_hi_t = kd_hi[off + t];
      const float kd_lo_t = kd_lo[off + t];
      // The working-set key, in float32: (gmax + yg_t)^2 / (K_ii + K_tt - 2 K_it).
      const float diff = (gmax.hi + gh[e]) + (gmax.lo + gl[e]);
      float quad = ((kd_i.hi - k) + (kd_hi_t - k)) + (kd_i.lo + kd_lo_t);
      quad = quad > 0.0f ? quad : tau.hi;
      const float key = metal::precise::divide(diff * diff, quad);
      if (key >= j_key) {
        j_key = key;
        j_ix = t;
        j_pay[0] = gh[e];
        j_pay[1] = gl[e];
        j_pay[2] = ah[e];
        j_pay[3] = al[e];
        j_pay[4] = k;
        j_pay[5] = kd_hi_t;
        j_pay[6] = kd_lo_t;
        j_pay[7] = t < first ? 1.0f : -1.0f;
      }
    }
    ovo_argmax<8>(j_key, j_zero, j_ix, j_pay, lane);
    if (groups > 1u) {
      if (lane == 0) {
        shared_f[bank][0][sg] = j_key;
        for (int q = 0; q < 8; ++q) {
          shared_f[bank][1 + q][sg] = j_pay[q];
        }
        shared_i[bank][0][sg] = j_ix;
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      const bool mine = lane < groups;
      j_key = mine ? shared_f[bank][0][lane] : -INFINITY;
      j_zero = 0.0f;
      for (int q = 0; q < 8; ++q) {
        j_pay[q] = mine ? shared_f[bank][1 + q][lane] : 0.0f;
      }
      j_ix = mine ? shared_i[bank][0][lane] : -1;
      ovo_argmax<8>(j_key, j_zero, j_ix, j_pay, lane);
      bank ^= 1u;
    }
    if (j_ix < 0) {
      stop = 1;
      break;
    }
    steps += 1;
    const int j = j_ix;

    // ---- the analytic two-variable step, clipped to the box (smo._pair_update) ----
    const ovo_ff yg_j = ovo_ff{j_pay[0], j_pay[1]};
    const ovo_ff alpha_j = ovo_ff{j_pay[2], j_pay[3]};
    const float k_ij = j_pay[4];
    const ovo_ff kd_j = ovo_ff{j_pay[5], j_pay[6]};
    const float y_j = j_pay[7];
    const ovo_ff c_j = ovo_ff{c_hi[off + j], c_lo[off + j]};
    const ovo_ff zero = ovo_ff{0.0f, 0.0f};
    ovo_ff quad = ovo_add(ovo_add(kd_i, kd_j), ovo_ff{-2.0f * k_ij, 0.0f});
    if (!(quad.hi > 0.0f)) {
      quad = tau;
    }
    const ovo_ff grad_i = y_i > 0.0f ? ovo_neg(gmax) : gmax;
    const ovo_ff grad_j = y_j > 0.0f ? yg_j : ovo_neg(yg_j);
    ovo_ff new_i, new_j;
    if (y_i != y_j) {
      const ovo_ff delta = ovo_div(ovo_neg(ovo_add(grad_i, grad_j)), quad);
      const ovo_ff diff = ovo_sub(alpha_i, alpha_j);
      new_i = ovo_add(alpha_i, delta);
      new_j = ovo_add(alpha_j, delta);
      if (ovo_gt(diff, zero)) {
        if (new_j.hi < 0.0f) {
          new_j = zero;
          new_i = diff;
        }
      } else if (new_i.hi < 0.0f) {
        new_i = zero;
        new_j = ovo_neg(diff);
      }
      if (ovo_gt(diff, ovo_sub(c_i, c_j))) {
        if (ovo_gt(new_i, c_i)) {
          new_i = c_i;
          new_j = ovo_sub(c_i, diff);
        }
      } else if (ovo_gt(new_j, c_j)) {
        new_j = c_j;
        new_i = ovo_add(c_j, diff);
      }
    } else {
      const ovo_ff delta = ovo_div(ovo_sub(grad_i, grad_j), quad);
      const ovo_ff total = ovo_add(alpha_i, alpha_j);
      new_i = ovo_sub(alpha_i, delta);
      new_j = ovo_add(alpha_j, delta);
      if (ovo_gt(total, c_i)) {
        if (ovo_gt(new_i, c_i)) {
          new_i = c_i;
          new_j = ovo_sub(total, c_i);
        }
      } else if (new_j.hi < 0.0f) {
        new_j = zero;
        new_i = total;
      }
      if (ovo_gt(total, c_j)) {
        if (ovo_gt(new_j, c_j)) {
          new_j = c_j;
          new_i = ovo_sub(total, c_j);
        }
      } else if (new_i.hi < 0.0f) {
        new_i = zero;
        new_j = total;
      }
    }
    // yg += y_i (new_i - old_i) K_i + y_j (new_j - old_j) K_j.
    ovo_ff step_i = ovo_sub(new_i, alpha_i);
    ovo_ff step_j = ovo_sub(new_j, alpha_j);
    if (y_i < 0.0f) {
      step_i = ovo_neg(step_i);
    }
    if (y_j < 0.0f) {
      step_j = ovo_neg(step_j);
    }
    const uchar st_i = ovo_status(new_i, c_i, y_i > 0.0f);
    const uchar st_j = ovo_status(new_j, c_j, y_j > 0.0f);
    const uint row_j = goff + uint(j) * uint(m);
    for (int e = 0; e < OVO_E; ++e) {
      const int t = int(tid) + e * width;
      if (t >= m) {
        continue;
      }
      const ovo_ff g = ovo_update(ovo_ff{gh[e], gl[e]}, step_i, ki[e], step_j,
                                  gram[row_j + uint(t)]);
      gh[e] = g.hi;
      gl[e] = g.lo;
      if (t == i) {
        ah[e] = new_i.hi;
        al[e] = new_i.lo;
        st[e] = st_i;
      }
      if (t == j) {
        ah[e] = new_j.hi;
        al[e] = new_j.lo;
        st[e] = st_j;
      }
    }
  }

  for (int e = 0; e < OVO_E; ++e) {
    const int t = int(tid) + e * width;
    if (t < m) {
      a_hi[off + t] = ah[e];
      a_lo[off + t] = al[e];
      g_hi[off + t] = gh[e];
      g_lo[off + t] = gl[e];
    }
  }
  if (tid == 0) {
    info[2u * p] = steps;
    info[2u * p + 1u] = stop;
  }
"""

_INPUTS = (
    "gram", "kd_hi", "kd_lo", "c_hi", "c_lo",
    "a_hi_in", "a_lo_in", "g_hi_in", "g_lo_in", "table", "params", "fparams",
)
_OUTPUTS = ("a_hi", "a_lo", "g_hi", "g_lo", "info")


@functools.lru_cache(maxsize=1)
def _kernel():
    """The kernel object: built, not yet compiled. One variant serves every fit."""
    header = f"#define OVO_E {ROWS_PER_THREAD}\n" + _HEADER
    return _metal.metal_kernel(
        f"svc_ovo_smo_e{ROWS_PER_THREAD}_v{_SOURCE_VERSION}",
        inputs=_INPUTS,
        outputs=_OUTPUTS,
        source=_SOURCE,
        header=header,
    )


def threadgroup_width(rows: int) -> int:
    """Threads per pair for pairs of at most ``rows`` samples: whole SIMD groups."""
    groups = max(1, -(-int(rows) // (32 * ROWS_PER_THREAD)))
    if groups * 32 > MAX_THREADS:
        raise ValueError(f"a pair of {rows} rows exceeds the kernel's {MAX_PAIR_ROWS}")
    return groups * 32


def _split(values) -> tuple[np.ndarray, np.ndarray]:
    """float64 values as double-floats: ``hi`` rounded to float32, ``lo`` the rest."""
    values = np.asarray(values, dtype=np.float64)
    hi = values.astype(np.float32)
    lo = (values - hi.astype(np.float64)).astype(np.float32)
    return hi, lo


# ------------------------------------------------------------------ the problem's rows


@dataclass(frozen=True, slots=True)
class ClassRows:
    """A fit's training rows in the compute frame, class by class, for its pairs' Grams.

    Class ``c``'s rows are ``X32[members[c]]``, ascending, and pair ``(i, j)`` is class
    ``i``'s rows then class ``j``'s, as LIBSVM orders each one-vs-one subproblem.
    ``X32`` is the fit's own float32 copy, read and never copied whole. ``top_sq`` is
    each class's largest squared row norm, from the float32 norms
    :func:`kernel_matrix`'s refinement test reads. ``device`` holds classes' rows on
    the GPU between the Grams that read them, in the order they were last read, within
    the budget :func:`_pair_grams` builds to; it goes with the fit, or :meth:`release`.
    """

    X32: np.ndarray
    members: tuple
    top_sq: np.ndarray
    device: dict

    def count(self, c: int) -> int:
        return int(self.members[c].shape[0])

    def host(self, i: int, j: int) -> np.ndarray:
        """Pair ``(i, j)``'s rows, on the host."""
        return np.concatenate([self.X32[self.members[i]], self.X32[self.members[j]]])

    def on_device(self, mx, c: int):
        """Class ``c``'s rows on the device, uploaded unless resident; now the most recent."""
        rows = self.device.pop(c, None)
        if rows is None:
            rows = mx.array(self.X32[self.members[c]], dtype=mx.float32)
        self.device[c] = rows
        return rows

    def device_bytes(self) -> int:
        return sum(4 * int(rows.size) for rows in self.device.values())

    def evict(self, keep, room: int) -> None:
        """Drop resident classes but ``keep``, least recently used first, to ``room`` bytes."""
        for c in list(self.device):
            if self.device_bytes() <= room:
                return
            if c not in keep:
                del self.device[c]

    def release(self) -> None:
        self.device.clear()


def class_rows(X32: np.ndarray, members: list[np.ndarray]) -> ClassRows:
    """:class:`ClassRows` of ``X32``, whose class ``c`` is ``X32[members[c]]``; none on the device."""
    sq = np.einsum("ij,ij->i", X32, X32)
    top = np.array([float(np.max(sq[rows])) if rows.shape[0] else 0.0 for rows in members])
    return ClassRows(X32=X32, members=tuple(members), top_sq=top, device={})


def waves(sizes, budget_bytes: int) -> list[tuple[int, int]]:
    """Consecutive runs of pairs, in LIBSVM's order, whose packed Grams fit the budget.

    ``sizes`` are the pairs' sample counts; a pair's Gram takes ``4·m²`` bytes, and a
    wave at most :data:`MAX_WAVE_ENTRIES` entries whatever the budget. Every pair must
    fit on its own, which the caller's gate checks. Returns ``[start, end)`` bounds.
    """
    budget = min(int(budget_bytes), 4 * MAX_WAVE_ENTRIES)
    out: list[tuple[int, int]] = []
    start = 0
    used = 0
    for k, m in enumerate(sizes):
        need = 4 * int(m) * int(m)
        if k > start and used + need > budget:
            out.append((start, k))
            start, used = k, 0
        used += need
    if len(sizes):
        out.append((start, len(sizes)))
    return out


def _pair_grams(mx, spec: KernelSpec, rows: ClassRows, pairs, device: str, capacity: int):
    """The packed Gram of ``pairs``: each pair's ``m × m`` kernel matrix, row-major.

    Block for block the MLX operations :func:`kernel_matrix` evaluates for the pair,
    every self-entry the closed-form diagonal; a pair whose rbf entries the host must
    recompute is :func:`kernel_matrix`'s own matrix. The buffer holds at least
    ``capacity`` entries: every wave of a fit allocates the same size, so MLX's
    buffer cache hands each wave the one the wave before it freed, where buffers of
    different sizes would each stay cached. Returns ``(gram, offsets)``.

    Everything else it puts on the device — the classes' rows ``rows`` keeps there
    and each pair's transients — stays within ``Tuning.distance_block_bytes``; a
    pair whose own transients exceed it is built alone, beside its two classes, as
    :func:`kernel_matrix` would build it on the host solver's path.
    """
    sizes = np.asarray([rows.count(i) + rows.count(j) for i, j in pairs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes * sizes)]).astype(np.int64)
    total = int(offsets[-1])
    rbf = spec.kernel == "rbf"
    n_features = int(rows.X32.shape[1])
    transform = _tile_transform(mx, spec, device)
    gram = mx.zeros((max(total, int(capacity), _metal.MIN_INPUT),), dtype=mx.float32)
    # Pairs are built in groups, each evaluated before the next is built: a lazy graph
    # over every pair would hold all their transients at once, and evaluating after
    # every pair cost a synchronization each (51 against 25 ms at 2 048 × 512, K = 16).
    # A group's transients and the resident classes fit the budget kernel_matrix's
    # blocks keep to. When the next pair would overflow it, the group is evaluated and
    # classes the pair does not read are dropped, least recently used first, until the
    # rest take at most half the budget, or less where the pair needs more: a fit whose
    # rows fit in half keeps them all, and one whose rows do not leaves its groups
    # half the budget, not a pair at a time.
    limit = get_tuning().distance_block_bytes
    pending = 0
    for k, (i, j) in enumerate(pairs):
        m = int(sizes[k])
        top = float(max(rows.top_sq[i], rows.top_sq[j]))
        refined = _RbfRefiner.needed(spec, n_features, top, top)
        block = _block_rows(m, m, device, None)
        if refined:
            # kernel_matrix's own: the rows and a block uploaded, their squares, the
            # product and the tile; then its result, uploaded.
            need = 4 * (4 * m * n_features + 3 * m * m)
        else:
            # The pair's rows side by side and their squares, then kernel_matrix's
            # blocks: each one's product and transformed tile, and its rows' squares
            # where it is not the whole pair.
            need = 4 * ((2 * m + (m if block < m else 0)) * n_features + 2 * m * m)
        fresh = () if refined else tuple(c for c in (i, j) if c not in rows.device)
        upload = 4 * n_features * sum(rows.count(c) for c in fresh)
        if pending + rows.device_bytes() + upload + need > limit:
            if pending:
                mx.eval(gram)
                pending = 0
            rows.evict((i, j), min(limit // 2, limit - need) - upload)
        pending += need
        at = int(offsets[k])
        if refined:
            tile = mx.array(kernel_matrix(spec, rows.host(i, j), backend="mlx", device=device))
            gram[at : at + m * m] = tile.reshape(-1)
        else:
            _build_pair(
                mx, gram, at, rows.on_device(mx, i), rows.on_device(mx, j), block, transform,
                rbf,
            )
    # Every self-entry is the closed-form diagonal, rounded to float32, as
    # kernel_matrix writes it.
    selfs = _self_entries(spec, rows.X32)
    diag = []
    values = []
    for k, (i, j) in enumerate(pairs):
        m = int(sizes[k])
        diag.append(int(offsets[k]) + np.arange(m, dtype=np.int64) * (m + 1))
        values += [selfs[rows.members[i]], selfs[rows.members[j]]]
    gram[mx.array(np.concatenate(diag))] = mx.array(np.concatenate(values))
    mx.eval(gram)
    return gram, offsets


def _build_pair(mx, gram, at: int, first, second, block: int, transform, rbf: bool) -> None:
    """Write the kernel matrix of ``first``'s rows then ``second``'s into ``gram`` at ``at``.

    Lazily, in blocks of ``block`` rows, the operations :func:`kernel_matrix`
    evaluates. A block of every row takes the norms of the whole pair, which
    kernel_matrix computes again from an identical array, to the same bits. The
    transients are this function's locals, so nothing but the graph holds them.
    """
    right = mx.concatenate([first, second], axis=0)
    m = int(right.shape[0])
    right_t = right.T
    b_sq = mx.sum(right * right, axis=1) if rbf else None
    for start in range(0, m, block):
        end = min(start + block, m)
        left = right[start:end]
        a_sq = b_sq if end - start == m else (mx.sum(left * left, axis=1) if rbf else None)
        tile = transform(left @ right_t, a_sq, b_sq)
        gram[at + start * m : at + end * m] = tile.reshape(-1)


# -------------------------------------------------------------------------- solving


@dataclass(frozen=True, slots=True)
class WaveResult:
    """What one wave's launches reached, per pair, read back to the host.

    ``alpha`` is float64, flat over the wave's pairs at ``offsets``, every value the
    kernel left at a bound set to that bound exactly. ``grams`` are the pairs' kernel
    matrices as host views of the device buffer — the caller verifies against them,
    and drops them to free it. ``stop`` holds the ``STOP_*`` codes.
    """

    alpha: np.ndarray
    offsets: np.ndarray
    n_iter: np.ndarray
    stop: np.ndarray
    grams: list
    launches: int

    def pair_alpha(self, k: int) -> np.ndarray:
        return self.alpha[self.offsets[k] : self.offsets[k + 1]]


@dataclass(frozen=True, slots=True)
class PairSolver:
    """The kernel, ready to launch: built by :func:`pair_solver` once it passed its check."""

    kernel: Any

    def solve(
        self,
        spec: KernelSpec,
        rows: ClassRows,
        pairs,
        boxes,
        k_diag,
        max_iter,
        *,
        tol: float,
        width: int,
        device: str,
        capacity: int = 0,
    ) -> WaveResult:
        """Solve one wave of one-vs-one pairs, every one in each launch.

        ``pairs`` are ``(i, j)`` class pairs; ``boxes`` and ``k_diag`` each pair's
        float64 per-sample ``C_t`` and ``K(x_t, x_t)`` in its own order (class ``i``'s
        rows, then class ``j``'s); ``max_iter`` each pair's resolved budget, at least
        one; ``width`` the threads per pair (:func:`threadgroup_width`) and
        ``capacity`` the Gram buffer's entries, at least — both the same for every
        wave of a fit.
        """
        import mlx.core as mx

        firsts = [rows.count(i) for i, _ in pairs]
        with use_device(device):
            gram, gram_offsets = _pair_grams(mx, spec, rows, pairs, device, capacity)
            return self._run(mx, gram, gram_offsets, firsts, boxes, k_diag, max_iter, tol, width)

    def _run(
        self, mx, gram, gram_offsets, firsts, boxes, k_diag, max_iter, tol, width,
        first_steps: int | None = None,
    ):
        """The launches of one wave over its packed Gram; see :meth:`solve`.

        ``firsts`` is each pair's count of first-class samples, labelled ``+1``.
        ``first_steps`` is the first launch's step budget per pair, and every later
        launch's floor: :data:`FIRST_STEPS` and :data:`MIN_STEPS` unless given.
        """
        n_pairs = len(boxes)
        sizes = np.array([b.shape[0] for b in boxes], dtype=np.int64)
        firsts = np.asarray(firsts, dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        total = int(offsets[-1])
        flat = max(total, _metal.MIN_INPUT)

        def padded(values, fill):
            out = np.full(flat, fill, dtype=np.float64)
            out[:total] = values
            return out

        box = padded(np.concatenate(boxes), 1.0)
        c_hi, c_lo = _split(box)
        kd_hi, kd_lo = _split(padded(np.concatenate(k_diag), 0.0))
        labels = np.concatenate(
            [np.where(np.arange(m) < f, 1.0, -1.0) for m, f in zip(sizes, firsts, strict=True)]
        )
        if int(gram_offsets[-1]) > MAX_WAVE_ENTRIES:  # waves() splits before this
            raise OverflowError(
                f"a wave of {int(gram_offsets[-1])} Gram entries exceeds {MAX_WAVE_ENTRIES}"
            )
        table = np.zeros((n_pairs, 8), dtype=np.int64)
        table[:, 0] = sizes
        table[:, 1] = firsts
        table[:, 2] = offsets[:-1]
        table[:, 3] = gram_offsets[:-1]
        tol_hi, tol_lo = _split([tol])
        tau_hi, tau_lo = _split([_TAU])
        fixed = [
            gram,
            mx.array(kd_hi), mx.array(kd_lo), mx.array(c_hi), mx.array(c_lo),
        ]
        params = _metal.int_params(mx, width)
        fparams = mx.array(
            np.array([tol_hi[0], tol_lo[0], tau_hi[0], tau_lo[0], 0, 0, 0, 0], np.float32)
        )
        # alpha = 0, so yg = -y.
        state = [
            mx.zeros((flat,), dtype=mx.float32),
            mx.zeros((flat,), dtype=mx.float32),
            mx.array(padded(-labels, 0.0).astype(np.float32)),
            mx.zeros((flat,), dtype=mx.float32),
        ]
        max_iter = np.asarray(max_iter, dtype=np.int64)
        n_iter = np.zeros(n_pairs, dtype=np.int64)
        stop = np.full(n_pairs, STOP_BUDGET, dtype=np.int64)
        active = max_iter > n_iter
        steps = FIRST_STEPS if first_steps is None else int(first_steps)
        floor = MIN_STEPS if first_steps is None else min(int(first_steps), MIN_STEPS)
        launches = 0
        # Every launch advances or stops each pair it runs, so no pair outlives its
        # budget at the smallest step count. A hard bound, for a kernel at fault.
        limit = 2 + int(max_iter.max(initial=0)) // floor
        while active.any():
            if launches >= limit:  # pragma: no cover - a fault in the kernel's progress
                stop[active] = STOP_STALLED
                break
            budget = np.where(active, np.minimum(steps, max_iter - n_iter), 0)
            table[:, 4] = budget
            started = time.perf_counter()
            outputs = _metal.launch(
                self.kernel,
                inputs=[
                    *fixed, *state, mx.array(table.reshape(-1).astype(np.int32)), params, fparams,
                ],
                output_shapes=[(flat,)] * 4 + [(2 * n_pairs,)],
                output_dtypes=[mx.float32] * 4 + [mx.int32],
                grid=(n_pairs * width, 1, 1),
                threadgroup=(width, 1, 1),
            )
            # Evaluated before NumPy reads any of it. An error MLX raises while it
            # evaluates — a library Metal cannot build, a threadgroup wider than the
            # pipeline takes, a failed allocation — is a Python exception from
            # mx.eval, and aborts the process from inside NumPy's buffer protocol
            # (MLX 0.32.2).
            mx.eval(*outputs)
            info = np.array(outputs[4]).reshape(n_pairs, 2).astype(np.int64)
            elapsed = time.perf_counter() - started
            launches += 1
            state = list(outputs[:4])
            ran = budget > 0
            n_iter[ran] += info[ran, 0]
            stop[ran] = info[ran, 1]
            stalled = ran & (info[:, 1] == STOP_BUDGET) & (info[:, 0] < budget)
            stop[stalled] = STOP_STALLED
            active = ran & (stop == STOP_BUDGET) & (n_iter < max_iter)
            most = int(info[ran, 0].max(initial=0))
            if most > 0:
                per_step = elapsed / most
                steps = int(min(MAX_STEPS, max(floor, LAUNCH_SECONDS / max(per_step, 1e-9))))

        alpha = np.asarray(state[0], dtype=np.float64)[:total] + np.asarray(
            state[1], dtype=np.float64
        )[:total]
        # A value the kernel set to its bound is that bound, exactly, in float64.
        bound = (c_hi.astype(np.float64) + c_lo.astype(np.float64))[:total]
        box = box[:total]
        alpha = np.where(alpha >= bound, box, np.clip(alpha, 0.0, box))
        host = np.asarray(gram)
        # A view of the device buffer: writing through it would change the Gram under
        # the lazy graphs built from it.
        host.flags.writeable = False
        grams = [
            host[int(gram_offsets[k]) : int(gram_offsets[k + 1])].reshape(int(m), int(m))
            for k, m in enumerate(sizes)
        ]
        return WaveResult(
            alpha=alpha, offsets=offsets, n_iter=n_iter, stop=stop, grams=grams,
            launches=launches,
        )


# ---------------------------------------------------------------------- first use

#: Serializes first uses: one thread compiles and checks the kernel while any other
#: that needs it waits for the answer. Never held while a fit runs the kernel.
_FIRST_USE = threading.Lock()


def _known_problem():
    """Two pairs of a fixed problem, no random numbers: 40 samples and 300 (two SIMD
    groups), per-sample boxes, an rbf Gram computed on the host in float64."""
    rows = []
    for m in (40, 300):
        t = np.arange(m, dtype=np.float64)
        X = np.stack([np.sin(1.3 * t), np.cos(0.7 * t + 0.4), np.sin(0.37 * t + 1.0)], axis=1)
        sq = np.einsum("ij,ij->i", X, X)
        # Finite data: a flag NumPy's BLAS raises here is spurious (smo._gradient).
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            dist = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
        gram = np.exp(-0.5 * dist).astype(np.float32)
        np.fill_diagonal(gram, 1.0)
        rows.append(gram)
    firsts = (17, 131)
    boxes = (np.full(40, 1.0), 0.5 + (np.arange(300) % 3).astype(np.float64))
    return rows, firsts, boxes


def _known_answer(solver: PairSolver) -> bool:
    """Run ``solver`` on :func:`_known_problem`; True if the host accepts what it returns.

    Each pair must stop optimal, inside its box, on its equality constraint, and
    with a full-set KKT violation below ``tol`` when the host recomputes the
    gradient in float64; and the answer must be the same, bit for bit, whether the
    pairs run in launches of seven steps at the narrowest width or in one launch at
    the widest.
    """
    import mlx.core as mx

    grams, firsts, boxes = _known_problem()
    tol = 1e-3
    sizes = [g.shape[0] for g in grams]
    gram_offsets = np.concatenate([[0], np.cumsum([m * m for m in sizes])]).astype(np.int64)
    packed = mx.array(np.concatenate([g.reshape(-1) for g in grams]))
    k_diag = [np.ones(m) for m in sizes]
    budget = [10_000, 10_000]
    short, wide = (
        solver._run(
            mx, packed, gram_offsets, firsts, list(boxes), k_diag, budget, tol, width,
            first_steps=first,
        )
        for width, first in ((threadgroup_width(max(sizes)), 7), (MAX_THREADS, None))
    )
    if short.launches < 2 or not np.array_equal(short.alpha, wide.alpha):
        return False
    if not np.array_equal(short.n_iter, wide.n_iter):
        return False
    for k, gram in enumerate(grams):
        alpha = short.pair_alpha(k)
        y = np.where(np.arange(sizes[k]) < firsts[k], 1.0, -1.0)
        C = boxes[k]
        if short.stop[k] != STOP_OPTIMAL or short.n_iter[k] < 1:
            return False
        if np.any(alpha < 0.0) or np.any(alpha > C) or abs(float(alpha @ y)) > 1e-9 * C.sum():
            return False
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            yg = gram.astype(np.float64) @ (alpha * y) - y
        upper = alpha >= C
        lower = ~upper & (alpha <= 0.0)
        up = np.where(y > 0, ~upper, ~lower)
        low = np.where(y > 0, ~lower, ~upper)
        violation = float((-yg[up]).max(initial=-np.inf) + yg[low].max(initial=-np.inf))
        if not violation < tol * (1.0 + 1e-6):
            return False
    return True


@functools.lru_cache(maxsize=1)
def _checked() -> bool:
    """Whether the kernel builds and passes its check. Compiles it, on its first launch.

    A kernel that fails to build or answers wrongly keeps every fit of the process
    on the host solver. Run inside a GPU device block.
    """
    try:
        return _known_answer(PairSolver(_kernel()))
    except (RuntimeError, ValueError):
        # What MLX raises, from mx.eval, when Metal cannot build the library or load
        # the kernel (RuntimeError), or when the widest threadgroup exceeds what the
        # compiled pipeline takes (ValueError). The kernel ran only on the fixed problem
        # above, never on a fit's data.
        return False


def pair_solver(device: str) -> PairSolver | None:
    """The concurrent one-vs-one solver for work on ``device``, or ``None``.

    ``None`` — the host solver, pair after pair — unless custom kernels may run for
    ``device`` (:func:`._metal.kernels_enabled`: the GPU, MLX 0.32 or later,
    ``Tuning.metal_kernels``) and the kernel has passed its first-use check, which
    compiles it. Unlike the neighbor search's kernel this one needs no list of
    validated MLX releases: nothing it returns is trusted to match MLX bit for bit,
    since every pair is verified on the host (``smo.finish_c_svc``).
    """
    if not _metal.kernels_enabled(device):
        return None
    with use_device(device), _FIRST_USE:
        ok = _checked()
    return PairSolver(_kernel()) if ok else None
