"""The first pass's distance tile and top-K selection, fused into one Metal kernel.

The neighbor search's first pass forms, for every (query row, training row) pair of a
tile, ``max((‖q‖² + ‖t‖²) − 2G, 0)`` from the Gram tile ``G`` the matmul produced,
and keeps each query row's first ``k_carry`` pairs under ``(value, column)``. MLX's
built-in operations do that in three passes over the tile: the compiled tile
expression, which writes a second tile, and two rounds of ``argpartition``, which
MLX's GPU runs as sorts. On tall, narrow data that selection was three quarters of
the first pass (1 000 000 × 64: 4.8 of 6.5 ms a block), and the second tile and the
sort's temporaries are why a block could hold only 38 query rows there. This kernel
reads ``G`` once and writes only the survivors.

**The method** is the one Johnson, Douze and Jégou describe for k-selection on GPUs
("Billion-scale similarity search with GPUs", IEEE Transactions on Big Data, 2019;
arXiv:1702.08734): each SIMD lane keeps a short sorted list of its best pairs in
registers and rejects almost every element with one comparison against the list's
last value, and a SIMD group's lanes merge their lists with SIMD-wide minimum
reductions. Written for mlxlearn from that description. One SIMD group scans one
chunk of one row's columns, a lane reading four columns in every 128; a second mode
of the same kernel merges the chunks' lists, and the list kept from the train tiles
before, into one per row.

**Order, bit for bit.** A lane meets its columns in increasing order, so a pair whose
value equals one already in its list always has the larger column: inserted after
every equal value, and rejected when it only equals the list's last, the list is
ordered by ``(value, column)``. A merge round emits the smallest head value and,
among the lanes holding it, the smallest column (a second ``simd_min``, on 32-bit
columns: Apple GPUs reduce a 64-bit ``simd_min`` through float32, which is not
exact). The merge mode reads the lists in increasing column ranges, which keeps the
same property. So a query block's result is its first ``k_carry`` columns under
``(value, column)``, ascending — the set MLX's built-in selection keeps, whose GPU
``argpartition`` behaves as a stable sort, and the same largest kept value, which the
certificate is built on. Each value is computed with the tile expression's operations
in its order — ``s = ‖q‖² + ‖t‖²``, then ``s − 2G`` (``2G`` is exact), then MLX's
``maximum(x, 0)`` — under ``fp contract(off)``, so it is the built-in tile's value bit
for bit. That identity rests on how MLX 0.32.2 implements ``argpartition`` and the
matmul, which is why :data:`VALIDATED_MLX` names the releases it was checked on.

**List length and the overflow flag.** A lane's own last value rejects little — it is
the K-th best of a 32nd of the group's elements — so the group also keeps a common
bound, as the paper's warp-wide queue does: the lexicographically largest of the
lanes' m-th pairs, ``32·m ≥ k_carry``, refreshed after every 128 columns. No pair
above it can be among the group's first ``k_carry``. With it, lists stay at most 32
long (lists of 64 spilled registers and ran five times slower) and ``k_carry`` up to
64 is still served: the one way a 32-long list can drop a winner — all 32 of its pairs
at or below the bound when another arrives — is flagged per row, never guessed, and
the caller recomputes a flagged row with the built-in selection. On Gaussian, blob
and low-rank data at 56 000–1 000 000 rows it flagged at most one row in 1 024.

**When it compiles.** Lazily, on the first query in a process whose first pass takes
this kernel, and once per list length (16 or 32): :func:`selection` builds the
library, runs it once on a fixed input whose answer the host computes, and serves it
only if the two agree. A library never compiled on the machine cost that first query
84–88 ms on an M4 Max with MLX 0.32.2, once per machine, MLX release and kernel
version; the operating system's shader cache then serves it to later processes in a
few milliseconds. MLX holds a lock that stalls the process's other GPU work while a
library builds, so it is not built on a background thread: one started at the index
build overlapped only that build's host work, about 8 ms at 30 000 × 256.
"""

from __future__ import annotations

import functools
import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _metal

__all__ = ["BUCKETS", "VALIDATED_MLX", "Selection", "bucket", "selection"]

#: ``k_carry`` is served in one of these buckets; a larger one keeps the built-in path.
BUCKETS = (16, 32, 64)

#: MLX releases, ``(major, minor, patch)``, on which the kernel's answers were checked
#: bit for bit against MLX's built-in selection — ``tests/unit/test_fused_selection.py``,
#: and a grid of 791 cases of data, k and blocking against the search before the
#: kernel existed. The identity rests on two behaviors of these releases, not on
#: promises — the GPU ``argpartition`` keeping the lowest columns among equal values,
#: as a stable sort does, and the matmul giving a row the same bits whatever the
#: number of rows (two or more) and columns in a product wider than it is deep
#: (``distance._tiles_wider``). Which GEMM kernel MLX picks for a shape can change in
#: any release, a patch release included, so each is listed on its own, and one
#: outside the list keeps the built-in selection until it has been checked. A
#: one-time self-test at run time could not stand in for that: both behaviors depend
#: on the shape, and a test small enough to run on a first query would exercise only
#: the shapes it is given.
VALIDATED_MLX = ((0, 32, 2),)

#: Register list length, at most. See the module docstring.
_LIST_MAX = 32

#: SIMD groups per threadgroup: 128 threads.
_SIMD_GROUPS = 4

#: A scan launches about this many SIMD groups, one per row and column chunk, when the
#: tile is large enough; fewer leave the GPU partly idle. Above 16 candidates a chunk's
#: emit and merge cost more, and half as many groups measured 1.15-1.2x faster at
#: 128 x 1 000 000 and 1 024 x 128 000, while at 16 twice as many were 1.1x faster.
#: Tuned on one M4 Max shared with other work, so provisional; other Apple GPUs are
#: untested.
_TARGET_GROUPS = 2048
_TARGET_GROUPS_WIDE = 1024

#: Columns per chunk, in multiples of 128: one load of four columns by each of 32 lanes.
_CHUNK_MIN = 1024
_CHUNK_MAX = 131072

#: Part of every kernel name: bumped with the source, so no cached library is stale.
_SOURCE_VERSION = 1

_HEADER = r"""
#include <metal_simdgroup>
#define MLXL_UNROLL _Pragma("clang loop unroll(full)")

// Insert (d, c) into the ascending list (v, ix) of MLXL_K pairs, dropping the last.
// Only called with d < v[MLXL_K - 1]. A pair already held with value d has a smaller
// column -- columns reach a list in increasing order -- and stays ahead of (d, c).
// Every index is a compile-time constant, so the list stays in registers.
inline void mlxl_insert(thread float* v, thread uint* ix, const float d, const uint c) {
  bool later[MLXL_K];
  MLXL_UNROLL for (int t = 0; t < MLXL_K; ++t) { later[t] = v[t] > d; }
  MLXL_UNROLL for (int t = MLXL_K - 1; t > 0; --t) {
    const bool shift = later[t - 1];
    v[t] = shift ? v[t - 1] : (later[t] ? d : v[t]);
    ix[t] = shift ? ix[t - 1] : (later[t] ? c : ix[t]);
  }
  v[0] = later[0] ? d : v[0];
  ix[0] = later[0] ? c : ix[0];
}

// Offer (d, c) to this lane's list. (bound_v, bound_c) is the SIMD group's bound: no
// pair above it, under (value, column), can be among the group's first n_out. A pair
// at or below it that meets a full list whose last pair is at or below it too may be
// one of those n_out with no room left -- possible only when n_out exceeds MLXL_K
// (watch) -- and sets the overflow flag.
inline void mlxl_offer(thread float* v, thread uint* ix, thread float& last,
                       thread bool& overflow, const bool watch, const float bound_v,
                       const uint bound_c, const float d, const uint c) {
  if (d < bound_v || (d == bound_v && c <= bound_c)) {
    overflow = overflow || (watch && last < INFINITY &&
                            (last < bound_v || (last == bound_v && ix[MLXL_K - 1] <= bound_c)));
    if (d < last) {
      mlxl_insert(v, ix, d, c);
      last = v[MLXL_K - 1];
    }
  }
}

// Write the SIMD group's first n pairs under (value, column), ascending, at out + at.
// Each round takes the smallest head value, then the smallest column among the lanes
// whose head holds it; that lane pops its head.
inline void mlxl_emit(thread float* v, thread uint* ix, const uint lane, const int n,
                      device float* out_v, device int* out_c, const size_t at) {
  for (int r = 0; r < n; ++r) {
    const float head = v[0];
    const uint head_c = ix[0];
    const float low = metal::simd_min(head);
    const uint low_c = metal::simd_min(head == low ? head_c : 0xFFFFFFFFu);
    if (head == low && head_c == low_c) {
      MLXL_UNROLL for (int t = 0; t < MLXL_K - 1; ++t) { v[t] = v[t + 1]; ix[t] = ix[t + 1]; }
      v[MLXL_K - 1] = INFINITY;
      ix[MLXL_K - 1] = 0xFFFFFFFFu;
    }
    if (lane == 0) {
      out_v[at + r] = low;
      out_c[at + r] = as_type<int>(low_c);
    }
  }
}
"""

# Mode 0 scans a Gram tile, one SIMD group per (row, column chunk): params are
# [0, rows, width, chunk width, chunks per row, n_out, first column, four-wide loads].
# Mode 1 merges each row's lists, one SIMD group per row: params are
# [1, rows, entries per row, n_out]; a lane takes four consecutive entries in every 128.
_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint group = threadgroup_position_in_grid.x * MLXL_GROUPS + simdgroup_index_in_threadgroup;
  float v[MLXL_K];
  uint ix[MLXL_K];
  MLXL_UNROLL for (int t = 0; t < MLXL_K; ++t) { v[t] = INFINITY; ix[t] = 0xFFFFFFFFu; }
  float last = INFINITY;
  float bound_v = INFINITY;
  uint bound_c = 0xFFFFFFFFu;
  bool overflow = false;
  int n_out;
  if (params[0] == 0) {
    const uint rows = uint(params[1]);
    const uint width = uint(params[2]);
    const uint chunk = uint(params[3]);
    const uint chunks = uint(params[4]);
    n_out = params[5];
    const uint col0 = uint(params[6]);
    const bool four = params[7] != 0;
    if (group >= rows * chunks) {
      return;
    }
    // The bound is the m-th pair of the lanes' lists, 32 m >= n_out.
    const bool second = n_out > 32;
    const bool watch = n_out > MLXL_K;
    const uint row = group / chunks;
    const uint c0 = (group - row * chunks) * chunk;
    const uint c1 = metal::min(c0 + chunk, width);
    const float q_sq = row_sq[row];
    const device float* g_row = vals + size_t(row) * size_t(width);
    // Every lane takes every step, so each simd_max below sees all 32 lanes.
    for (uint step = c0; step < c1; step += 128u) {
      const uint j = step + 4u * lane;
      if (j < c1) {
        float4 x4;
        uint n = 4u;
        if (four && j + 4u <= c1) {
          const float4 g4 = *((const device float4*)(g_row + j));
          const float4 t4 = *((const device float4*)(col_sq + j));
          const float4 s4 = float4(q_sq) + t4;
          x4 = s4 - 2.0f * g4;
        } else {
          n = metal::min(4u, c1 - j);
          for (uint e = 0; e < n; ++e) {
            const float s = q_sq + col_sq[j + e];
            x4[e] = s - 2.0f * g_row[j + e];
          }
        }
        #pragma clang loop unroll(disable)
        for (uint e = 0; e < n; ++e) {
          const float x = x4[e];
          // MLX's maximum(x, 0): NaN passes through, otherwise the larger.
          const float d = metal::isnan(x) ? x : (x > 0.0f ? x : 0.0f);
          mlxl_offer(v, ix, last, overflow, watch, bound_v, bound_c, d, col0 + j + e);
        }
      }
      const float m_v = second ? v[1] : v[0];
      const uint m_c = second ? ix[1] : ix[0];
      bound_v = metal::simd_max(m_v);
      bound_c = metal::simd_max(m_v == bound_v ? m_c : 0u);
    }
  } else {
    const uint rows = uint(params[1]);
    const uint total = uint(params[2]);
    n_out = params[3];
    if (group >= rows) {
      return;
    }
    const bool second = n_out > 32;
    const bool watch = n_out > MLXL_K;
    const size_t at = size_t(group) * size_t(total);
    // The row's lists, each ascending and each over columns above the one before, are
    // read in order: a pair reaching a lane after an equal value still has the larger
    // column, as in the scan.
    for (uint step = 0; step < total; step += 128u) {
      const uint j = step + 4u * lane;
      const uint end = metal::min(j + 4u, total);
      #pragma clang loop unroll(disable)
      for (uint e = j; e < end; ++e) {
        const float x = vals[at + e];
        // (inf, -1) pads a list shorter than n_out, and is never a candidate.
        if (x < INFINITY) {
          mlxl_offer(v, ix, last, overflow, watch, bound_v, bound_c, x,
                     as_type<uint>(cols_in[at + e]));
        }
      }
      const float m_v = second ? v[1] : v[0];
      const uint m_c = second ? ix[1] : ix[0];
      bound_v = metal::simd_max(m_v);
      bound_c = metal::simd_max(m_v == bound_v ? m_c : 0u);
    }
  }
  mlxl_emit(v, ix, lane, n_out, out_vals, out_cols, size_t(group) * size_t(n_out));
  const bool any_overflow = metal::simd_any(overflow);
  if (lane == 0) {
    out_flags[group] = any_overflow ? 1 : 0;
  }
"""


def bucket(k_carry: int) -> int | None:
    """The bucket ``k_carry`` candidates are served in, or ``None`` above the largest."""
    for size in BUCKETS:
        if k_carry <= size:
            return size
    return None


def _list_length(k_carry: int) -> int:
    return min(bucket(k_carry), _LIST_MAX)


@functools.lru_cache(maxsize=len(BUCKETS))
def _kernel(list_length: int):
    """The kernel object for register lists of ``list_length``: built, not yet compiled."""
    header = f"#define MLXL_K {list_length}\n#define MLXL_GROUPS {_SIMD_GROUPS}\n" + _HEADER
    return _metal.metal_kernel(
        f"knn_topk_l{list_length}_g{_SIMD_GROUPS}_v{_SOURCE_VERSION}",
        inputs=["params", "vals", "row_sq", "col_sq", "cols_in"],
        outputs=["out_vals", "out_cols", "out_flags"],
        source=_SOURCE,
        header=header,
    )


def _chunk_width(rows: int, width: int, k_carry: int) -> int:
    """Columns per SIMD group: enough groups to fill the GPU, within the chunk bounds."""
    target = _TARGET_GROUPS if k_carry <= 16 else _TARGET_GROUPS_WIDE
    per = max(_CHUNK_MIN, min(_CHUNK_MAX, (rows * width) // target))
    return -(-per // 128) * 128


@dataclass(frozen=True, slots=True)
class Selection:
    """The fused first-pass selection of ``k_carry`` candidates, ready to launch.

    Built by :func:`selection` once the gate is open and the kernel has passed its
    first-use check; holds the kernel object and nothing that depends on data.
    """

    kernel: Any
    k_carry: int

    @property
    def may_overflow(self) -> bool:
        """Whether a list can overflow, so that its rows' flags must be read."""
        return self.k_carry > _list_length(self.k_carry)

    def _launch(self, mx, params, vals, row_sq, col_sq, cols_in, groups: int):
        size = max(groups * self.k_carry, 1)
        n_tg = -(-groups // _SIMD_GROUPS)
        return _metal.launch(
            self.kernel,
            inputs=[params, vals, row_sq, col_sq, cols_in],
            output_shapes=[(size,), (size,), (max(groups, 1),)],
            output_dtypes=[mx.float32, mx.int32, mx.int32],
            grid=(max(n_tg, 1) * _SIMD_GROUPS * 32, 1, 1),
            threadgroup=(_SIMD_GROUPS * 32, 1, 1),
        )

    def scan(self, mx, gram, q_sq, t_sq, *, col0: int, t_offset: int):
        """Per (row, column chunk), the first ``k_carry`` pairs of a Gram tile, ascending.

        ``gram`` is ``(rows, width)``, ``q_sq`` the rows' squared norms and ``t_sq`` the
        tile's columns'. Returns ``(vals, cols, n_chunks, flags)``: ``vals`` and
        ``cols`` are ``(rows, n_chunks * k_carry)``, a chunk narrower than ``k_carry``
        padding its list with ``(inf, -1)``, and ``cols`` holds global columns, ``col0``
        plus the tile's; ``flags`` is ``(rows, n_chunks)``, nonzero where a list
        overflowed. ``t_offset`` is ``t_sq``'s element offset in its buffer: four-wide
        loads need it, and the tile's width, to be multiples of four.
        """
        rows, width = (int(s) for s in gram.shape)
        chunk = _chunk_width(rows, width, self.k_carry)
        n_chunks = -(-width // chunk)
        groups = rows * n_chunks
        four = int(width % 4 == 0 and t_offset % 4 == 0)
        params = _metal.int_params(mx, 0, rows, width, chunk, n_chunks, self.k_carry, col0, four)
        vals, cols, flags = self._launch(
            mx,
            params,
            _metal.pad_min8(mx, gram.reshape(-1), 0.0),
            _metal.pad_min8(mx, q_sq.reshape(-1), 0.0),
            _metal.pad_min8(mx, t_sq.reshape(-1), math.inf),
            mx.zeros((_metal.MIN_INPUT,), dtype=mx.int32),
            groups,
        )
        shape = (rows, n_chunks * self.k_carry)
        return vals.reshape(shape), cols.reshape(shape), n_chunks, flags.reshape(rows, n_chunks)

    def merge(self, mx, vals, cols, *, n_lists: int):
        """The first ``k_carry`` pairs of each row's ``n_lists`` ascending lists.

        The lists, ``k_carry`` long each, must cover increasing column ranges from left
        to right; the result is then the first ``k_carry`` under ``(value, column)``.
        Returns ``(vals, cols, flags)``, ``flags`` of shape ``(rows,)``.
        """
        rows = int(vals.shape[0])
        params = _metal.int_params(mx, 1, rows, n_lists * self.k_carry, self.k_carry)
        dummy = mx.zeros((_metal.MIN_INPUT,), dtype=mx.float32)
        out_vals, out_cols, flags = self._launch(
            mx,
            params,
            _metal.pad_min8(mx, vals.reshape(-1), math.inf),
            dummy,
            dummy,
            _metal.pad_min8(mx, cols.reshape(-1), -1),
            rows,
        )
        shape = (rows, self.k_carry)
        return out_vals.reshape(shape), out_cols.reshape(shape), flags[:rows]

    def smallest(self, mx, gram, q_sq, t_sq, *, col0: int, t_offset: int, prev=None):
        """A query block's running first ``k_carry``, after one more train tile.

        ``prev`` is ``(vals, cols)`` from the tiles before, every column of which is
        below ``col0``. Returns ``(vals, cols, flags)``: ``(rows, k_carry)`` each,
        ascending by ``(value, column)``, and per row a flag, nonzero where a list
        overflowed — ``None`` when none can (:attr:`may_overflow`). A flagged row's list
        is not to be used.
        """
        vals, cols, n_chunks, scan_flags = self.scan(
            mx, gram, q_sq, t_sq, col0=col0, t_offset=t_offset
        )
        flagged = self.may_overflow
        if prev is None and n_chunks == 1:
            # The first tile is never narrower than k_carry (resolve_blocks), so its one
            # chunk's list is full.
            return vals, cols, (scan_flags.reshape(-1) if flagged else None)
        n_lists = n_chunks
        if prev is not None:
            vals = mx.concatenate([prev[0], vals], axis=1)
            cols = mx.concatenate([prev[1], cols], axis=1)
            n_lists += 1
        vals, cols, merge_flags = self.merge(mx, vals, cols, n_lists=n_lists)
        if not flagged:
            return vals, cols, None
        return vals, cols, mx.maximum(mx.max(scan_flags, axis=1), merge_flags)


# ------------------------------------------------------------------ first use

#: Serializes first uses, so that one thread compiles and checks a kernel while any
#: other that needs it waits for the answer instead of compiling it again. It guards
#: the cached answer only; it is never held while a query runs a kernel.
_FIRST_USE = threading.Lock()


def _known_answer(select: Selection) -> bool:
    """Run ``select`` on a fixed tile whose answer the host computes; True if it matches.

    Two rows of small integers, so every tile value is exact and most are ties; two
    train tiles, so a previous list is merged, of 2 052 columns (four-wide loads, a
    last chunk four wide, narrower than the list) and 1 031 (single loads).
    """
    import mlx.core as mx

    widths = (2052, 1031)
    n = sum(widths)
    col = np.arange(n)
    gram = ((col[None, :] * 7 + np.arange(2)[:, None] * 3) % 9 - 4).astype(np.float32)
    t_sq = (col % 5).astype(np.float32)
    q_sq = np.array([[40.0], [41.0]], dtype=np.float32)
    tile = np.maximum((q_sq + t_sq) - np.float32(2.0) * gram, np.float32(0.0))
    order = np.lexsort((np.broadcast_to(col, tile.shape), tile), axis=1)[:, : select.k_carry]

    prev = None
    flags = []
    start = 0
    for width in widths:
        vals, cols, flag = select.smallest(
            mx,
            mx.array(gram[:, start : start + width]),
            mx.array(q_sq),
            mx.array(t_sq[start : start + width]),
            col0=start,
            t_offset=0,
            prev=prev,
        )
        prev = (vals, cols)
        if flag is not None:
            flags.append(flag)
        start += width
    mx.eval(*prev, *flags)
    return (
        np.array_equal(np.asarray(prev[1]), order)
        and np.array_equal(
            np.asarray(prev[0]).view(np.uint32),
            np.take_along_axis(tile, order, axis=1).view(np.uint32),
        )
        and not any(np.asarray(f).any() for f in flags)
    )


@functools.lru_cache(maxsize=len(BUCKETS))
def _checked(list_length: int) -> bool:
    """Whether the kernel with lists of ``list_length`` builds and passes its check.

    Compiles it, on its first launch. Every bucket the list length serves is checked:
    for lists of 32, ``k_carry`` of 32, and of 64, which reads the second bound and
    watches for overflow. A kernel that fails to build or answers wrongly keeps every
    query of the process on MLX's built-in selection. Run inside a GPU device block.
    """
    served = [size for size in BUCKETS if min(size, _LIST_MAX) == list_length]
    try:
        return all(_known_answer(Selection(_kernel(list_length), size)) for size in served)
    except RuntimeError:
        # What MLX raises when Metal cannot build the library ("Unable to build metal
        # library") or load the kernel ("Unable to load kernel"). The kernel ran only
        # on the fixed input above, never on a query's data.
        return False


def selection(device: str, k_carry: int) -> Selection | None:
    """The fused selection of ``k_carry`` candidates for work on ``device``, or ``None``.

    ``None`` — MLX's built-in selection — unless custom kernels may run for
    ``device`` (:func:`._metal.kernels_enabled`), MLX is a release this kernel was
    validated on (:data:`VALIDATED_MLX`), ``k_carry`` fits a bucket, and the kernel
    serving it has passed its first-use check, which compiles it. Call it inside the
    query's GPU device block.
    """
    if (
        bucket(k_carry) is None
        or not _metal.kernels_enabled(device)
        or _metal.mlx_version() not in VALIDATED_MLX
    ):
        return None
    length = _list_length(k_carry)
    with _FIRST_USE:
        ok = _checked(length)
    return Selection(_kernel(length), k_carry) if ok else None
