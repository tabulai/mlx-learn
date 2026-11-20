"""Metal primitives for o-means: two-level nearest-centroid assignment.

o-means groups the centroids into coarse cells and keeps a mirror of the
centroid table sorted by cell, so each cell is one contiguous block. A row is
scored against the cell centres, keeps its nearest cells and is compared only
with the centroids in them. The search is approximate: a row whose nearest
centroid lies in a cell it did not search receives the best centroid of the
cells it did search.

The algorithm follows ``contrib/o_means`` of owlgebra-ai/faiss (MIT),
accessed 2026-11-18. The Metal kernels are a separate implementation.

The tiled scan's fragment coordinates and vector load strategy are derived
from MLX 0.32.4 ``mlx/backend/metal/kernels/steel/gemm/{mma,loader}.h`` (MIT),
through ``e_means_bf16.py``.
Copyright © 2024 Apple Inc.
Modified for mlxlearn on 2026-11-18: rows gathered through per-cell buckets,
a loop over the centroid block of one cell, and a running row minimum.
See THIRD_PARTY_NOTICES.md section 1 for the upstream MIT license.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx

from mlxlearn_device._binary32 import matmul
from mlxlearn_device._neighbors_fused import select_topk

from .e_means import assign, batch_stats, gather_bf16
from .e_means_bf16 import prepare_centers

_UINT32_MAX = 2**32 - 1
# An empty cell must lose to every cell that holds a centroid, yet stay finite
# so that the selection order among empty cells is defined.
EMPTY_NORM = 1e37
MAX_PROBES = 128


@dataclass(frozen=True)
class Cells:
    """Coarse cells over the centroids and the cell-sorted table layout."""
    centres: mx.array   # (G, d) float32, in assignment coordinates
    norms: mx.array     # (G,) float32 squared norms; EMPTY_NORM for empty cells
    order: mx.array     # (k,) int32: centroid id at each table position
    rank: mx.array      # (k,) int32: table position of each centroid id
    offsets: mx.array   # (G + 1,) uint32: first table position of each cell
    sizes: mx.array     # (G,) uint32: centroids in each cell

    def arrays(self):
        return (self.centres, self.norms, self.order, self.rank, self.offsets, self.sizes)


def _shape(*values):
    """Sizes passed as data, so that a new batch size compiles nothing."""
    padded = list(values) + [0] * (8 - len(values))
    return mx.array(padded, mx.int32)


@lru_cache(None)
def _kernel(name):
    if name == "histogram":
        return mx.fast.metal_kernel(
            name="omeans_histogram", input_names=["values", "shape"],
            output_names=["counts"], atomic_outputs=True, source=r"""
            uint i = thread_position_in_grid.x;
            if (i >= uint(shape[0])) return;
            atomic_fetch_add_explicit(&counts[uint(values[i])], 1u, memory_order_relaxed);
            """)
    if name == "place":
        # Counting sort of the (row, probe) pairs by cell. Every output of an
        # atomic kernel is atomic, so the pair list uses atomic stores.
        return mx.fast.metal_kernel(
            name="omeans_place_pairs", input_names=["probes", "pair_offsets", "shape"],
            output_names=["cursor", "pairs"], atomic_outputs=True, source=r"""
            uint i = thread_position_in_grid.x;
            if (i >= uint(shape[0])) return;
            uint cell = uint(probes[i]);
            uint slot = atomic_fetch_add_explicit(&cursor[cell], 1u, memory_order_relaxed);
            atomic_store_explicit(&pairs[pair_offsets[cell] + slot], i, memory_order_relaxed);
            """)
    if name == "scalar_scan":
        # One SIMD group per (row, probe) pair. Lanes stride over the centroids
        # of the probed cell. This is the reference fine scan: it needs no
        # bucketing and is used to verify the tiled scan.
        return mx.fast.metal_kernel(
            name="omeans_scalar_scan",
            input_names=["x", "centers", "norms", "probes", "offsets", "shape"],
            output_names=["scores", "positions"], source=r"""
            const uint pair = thread_position_in_grid.x / 32;
            const uint lane = thread_index_in_simdgroup;
            const uint R = uint(shape[0]), K = uint(shape[1]);
            if (pair >= R * P) return;
            const uint row = pair / P;
            const uint cell = uint(probes[pair]);
            const uint lo = uint(offsets[cell]), hi = uint(offsets[cell + 1]);
            float best = INFINITY;
            // An empty cell reports an infinite score at a valid position.
            uint index = metal::min(lo, K - 1u);
            for (uint c = lo + lane; c < hi; c += 32) {
                float dot = 0;
                for (uint f = 0; f < D; ++f)
                    dot += float(x[row * D + f]) * float(centers[c * D + f]);
                float value = norms[c] - 2.0f * dot;
                if (value < best) {best = value; index = c;}
            }
            float value = metal::simd_min(best);
            uint winner = metal::simd_min(best == value ? index : 0xffffffffu);
            if (lane == 0) {scores[pair] = value; positions[pair] = winner;}
            """)
    if name == "reduce":
        # Merge each row's probes, then follow the winner's split links: a
        # centroid split since the last cell build is still filed under its
        # donor's cell, so the donor's child and grandchild are tested too.
        return mx.fast.metal_kernel(
            name="omeans_reduce_links",
            input_names=["x", "centers", "norms", "scores", "positions",
                         "order", "rank", "child", "shape"],
            output_names=["labels", "best_scores"], source=r"""
            const uint row = thread_position_in_grid.x / 32;
            const uint lane = thread_index_in_simdgroup;
            if (row >= uint(shape[0])) return;
            float best = INFINITY;
            uint slot = 0xffffffffu;
            for (uint p = lane; p < P; p += 32) {
                float value = scores[row * P + p];
                if (value < best) {best = value; slot = p;}
            }
            float value = metal::simd_min(best);
            // Equal scores keep the nearer cell. With no finite score every
            // lane holds the sentinel; the first probe then supplies a valid
            // position.
            uint first = metal::simd_min(best == value ? slot : 0xffffffffu);
            if (first == 0xffffffffu) first = 0;
            uint winner = uint(order[positions[row * P + first]]);
            const int c1 = child[winner];
            const int c2 = c1 >= 0 ? child[uint(c1)] : -1;
            if (c1 >= 0) {
                const uint p1 = uint(rank[uint(c1)]);
                const uint p2 = c2 >= 0 ? uint(rank[uint(c2)]) : p1;
                float d1 = 0, d2 = 0;
                for (uint f = lane; f < D; f += 32) {
                    float a = float(x[row * D + f]);
                    d1 += a * float(centers[p1 * D + f]);
                    d2 += a * float(centers[p2 * D + f]);
                }
                float s1 = norms[p1] - 2.0f * metal::simd_sum(d1);
                float s2 = norms[p2] - 2.0f * metal::simd_sum(d2);
                if (s1 < value) {value = s1; winner = uint(c1);}
                if (c2 >= 0 && s2 < value) {value = s2; winner = uint(c2);}
            }
            if (lane == 0) {labels[row] = winner; best_scores[row] = value;}
            """)
    raise ValueError(name)

# Copyright © 2024 Apple Inc.
# Modified for mlxlearn: bucketed row gather, per-cell centroid loop, 2026-11-18.
_TILED_SCAN = r"""
    constexpr uint TM = BM / (8 * WM);
    constexpr uint TN = BN / (8 * WN);
    constexpr uint LD = BK + PAD;
    constexpr uint THREADS = 32 * WM * WN;
    constexpr uint AV = BM * BK / THREADS;
    constexpr uint BV = BN * BK / THREADS;
    constexpr uint AC = BK / AV, BC = BK / BV;
    constexpr uint AR = THREADS / AC, BR = THREADS / BC;
    struct alignas(T) AVector { T values[AV]; };
    struct alignas(T) BVector { T values[BV]; };
    threadgroup T xs[BM * LD];
    // Centroids keep their natural [centroid,feature] layout so that adjacent
    // loading lanes read adjacent global addresses.
    threadgroup T cs[BN * LD];
    threadgroup uint slot_pair[BM];
    threadgroup uint slot_row[BM];
    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint K = uint(shape[1]), G = uint(shape[2]);
    // plan = pair offsets, tile offsets and centroid offsets, G + 1 entries each.
    const uint tile = threadgroup_position_in_grid.x;
    if (tile >= plan[2 * G + 1]) return;
    // The last cell whose first tile is not after this one owns the tile;
    // cells without pairs own no tile and are skipped.
    uint cell = 0, last = G - 1;
    while (cell < last) {
        uint middle = (cell + last + 1) / 2;
        if (plan[G + 1 + middle] <= tile) cell = middle; else last = middle - 1;
    }
    const uint q0 = plan[cell] + (tile - plan[G + 1 + cell]) * BM;
    const uint q1 = plan[cell + 1];
    const uint k_lo = plan[2 * G + 2 + cell], k_hi = plan[2 * G + 3 + cell];
    for (uint i = tid; i < BM; i += THREADS) {
        uint pair = q0 + i < q1 ? uint(pairs[q0 + i]) : 0xffffffffu;
        slot_pair[i] = pair;
        slot_row[i] = pair == 0xffffffffu ? pair : pair / P;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint wm = sg / WN, wn = sg % WN;
    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);
    const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    // Each SIMD group keeps the running minimum of its rows over its share
    // of the columns in registers; the groups are merged once at the end.
    float run_best[TM];
    uint run_index[TM];
    #pragma clang loop unroll(full)
    for (uint m = 0; m < TM; ++m) {
        run_best[m] = INFINITY;
        // An empty cell reports an infinite score at a valid position.
        run_index[m] = metal::min(k_lo, K - 1u);
    }
    #pragma clang loop unroll(disable)
    for (uint c0 = k_lo; c0 < k_hi; c0 += BN) {
        metal::simdgroup_float8x8 acc[TM][TN];
        #pragma clang loop unroll(full)
        for (uint m = 0; m < TM; ++m)
            #pragma clang loop unroll(full)
            for (uint n = 0; n < TN; ++n)
                acc[m][n] = metal::simdgroup_float8x8(0);
        #pragma clang loop unroll(disable)
        for (uint step = 0; step < D; step += BK) {
            // Same depth order as the dense streaming kernel and MLX Steel,
            // which accumulate a partial FP32 depth tile first.
            uint f0 = sizeof(T) == sizeof(float) && D % BK
                      ? (step == 0 ? D - D % BK : step - BK) : step;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < BM; i += AR) {
                uint r = i + tid / AC, f = (tid % AC) * AV;
                uint source = slot_row[r];
                if (source != 0xffffffffu && f0 + f + AV <= D) {
                    *reinterpret_cast<threadgroup AVector*>(xs + r * LD + f) =
                        omeans_load_vector<AVector>(x + source * D + f0 + f);
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < AV; ++j)
                        xs[r * LD + f + j] = source != 0xffffffffu && f0 + f + j < D
                            ? T(x[source * D + f0 + f + j]) : T(0);
                }
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < BN; i += BR) {
                uint c = i + tid / BC, f = (tid % BC) * BV;
                if (c0 + c < k_hi && f0 + f + BV <= D) {
                    *reinterpret_cast<threadgroup BVector*>(cs + c * LD + f) =
                        omeans_load_vector<BVector>(centers + (c0 + c) * D + f0 + f);
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < BV; ++j)
                        cs[c * LD + f + j] = c0 + c < k_hi && f0 + f + j < D
                            ? T(centers[(c0 + c) * D + f0 + f + j]) : T(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint f = 0; f < BK; f += 8) {
                metal::simdgroup_matrix<T,8,8> aa[TM], bb[TN];
                #pragma clang loop unroll(full)
                for (uint m = 0; m < TM; ++m) {
                    uint offset = ((m * WM + wm) * 8 + fm) * LD + f + fn;
                    reinterpret_cast<thread metal::vec<T,2>&>(aa[m].thread_elements()) =
                        metal::vec<T,2>(xs[offset], xs[offset + 1]);
                }
                #pragma clang loop unroll(full)
                for (uint n = 0; n < TN; ++n) {
                    uint offset = ((n * WN + wn) * 8 + fn) * LD + f + fm;
                    reinterpret_cast<thread metal::vec<T,2>&>(bb[n].thread_elements()) =
                        metal::vec<T,2>(cs[offset], cs[offset + LD]);
                }
                #pragma clang loop unroll(full)
                for (uint m = 0; m < TM; ++m)
                    #pragma clang loop unroll(full)
                    for (uint n = 0; n < TN; ++n)
                        metal::simdgroup_multiply_accumulate(acc[m][n], aa[m], bb[n], acc[m][n]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma clang loop unroll(full)
        for (uint m = 0; m < TM; ++m) {
            float best = INFINITY;
            uint index = c0;
            #pragma clang loop unroll(full)
            for (uint n = 0; n < TN; ++n) {
                const uint col = c0 + (n * WN + wn) * 8 + fn;
                const auto values = reinterpret_cast<thread float2&>(acc[m][n].thread_elements());
                #pragma clang loop unroll(full)
                for (uint q = 0; q < 2; ++q) {
                    float value = col + q < k_hi ? norms[col + q] - 2.0f * values[q] : INFINITY;
                    if (value < best || (value == best && col + q < index)) {
                        best = value; index = col + q;
                    }
                }
            }
            // The four lanes with the same row differ only in lane bits 0 and 3.
            float other = metal::simd_shuffle_xor(best, 1);
            uint other_index = metal::simd_shuffle_xor(index, 1);
            if (other < best || (other == best && other_index < index)) {
                best = other; index = other_index;
            }
            other = metal::simd_shuffle_xor(best, 8);
            other_index = metal::simd_shuffle_xor(index, 8);
            if (other < best || (other == best && other_index < index)) {
                best = other; index = other_index;
            }
            if (best < run_best[m] || (best == run_best[m] && index < run_index[m])) {
                run_best[m] = best; run_index[m] = index;
            }
        }
    }
    threadgroup float row_scores[WN * BM];
    threadgroup uint row_indices[WN * BM];
    #pragma clang loop unroll(full)
    for (uint m = 0; m < TM; ++m) {
        if (fn == 0) {
            uint row = (m * WM + wm) * 8 + fm;
            row_scores[wn * BM + row] = run_best[m];
            row_indices[wn * BM + row] = run_index[m];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < BM; row += THREADS) {
        const uint pair = slot_pair[row];
        if (pair == 0xffffffffu) continue;
        float best = row_scores[row];
        uint index = row_indices[row];
        #pragma clang loop unroll(full)
        for (uint j = 1; j < WN; ++j) {
            float value = row_scores[j * BM + row];
            uint candidate = row_indices[j * BM + row];
            if (value < best || (value == best && candidate < index)) {
                best = value; index = candidate;
            }
        }
        scores[pair] = best;
        positions[pair] = index;
    }
"""


@lru_cache(None)
def _tiled_scan_kernel():
    return mx.fast.metal_kernel(
        name="omeans_tiled_scan",
        input_names=["x", "centers", "norms", "pairs", "plan", "shape"],
        output_names=["scores", "positions"], source=_TILED_SCAN,
        header=r"""
        #include <metal_simdgroup_matrix>
        template <typename V, typename U>
        V omeans_load_vector(const device U* pointer) {
            return *reinterpret_cast<const device V*>(pointer);
        }
        // MLX binds arrays of fewer than eight elements in the constant address space.
        template <typename V, typename U>
        V omeans_load_vector(const constant U* pointer) {
            return *reinterpret_cast<const constant V*>(pointer);
        }
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def tiled_scan_available():
    """Whether this MLX release can build the tiled scan."""
    try:
        _tiled_scan_kernel()
    except TypeError:
        # Earlier supported MLX releases lack the safe-math compile option.
        return False
    return True


def histogram(values, bins):
    """Count 32-bit integer values in ``[0, bins)`` with integer atomics."""
    count = values.size
    if count == 0:
        return mx.zeros((bins,), mx.uint32)
    return _kernel("histogram")(
        inputs=[values, _shape(count)], grid=(count, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(bins,)], output_dtypes=[mx.uint32], init_value=0, stream=mx.gpu)[0]


def build_cells(centers, n_cells, iterations, *, previous=None, rng=None):
    """Group centroids into cells by k-means over the centroids themselves.

    ``centers`` are float32 centroids in assignment coordinates. The first
    build draws ``n_cells`` of them as starting cell centres with ``rng``;
    later builds start from ``previous``. Zero iterations re-file every
    centroid under the existing cell centres. An empty cell keeps its centre.
    """
    k, d = centers.shape
    if not 1 <= n_cells <= k:
        raise ValueError("n_cells must be between 1 and the number of centroids")
    if n_cells * k > 2**31 - 1:
        raise ValueError("n_cells * n_clusters exceeds the 32-bit cell sort key")
    if previous is None:
        picks = rng.permutation(k)[:n_cells]
        centres = centers[mx.array(picks, dtype=mx.int32)]
    else:
        centres = previous.centres
    for _ in range(iterations):
        labels, _ = assign(centers, centres)
        sums, counts = batch_stats(centers, labels, n_cells)
        centres = mx.where((counts > 0)[:, None], sums / mx.maximum(counts, 1)[:, None], centres)
    labels, _ = assign(centers, centres)
    sizes = histogram(labels, n_cells)
    offsets = mx.concatenate([mx.zeros((1,), mx.uint32), mx.cumsum(sizes)]).astype(mx.uint32)
    positions = mx.arange(k, dtype=mx.int32)
    # Ties in the sort key are impossible, so the layout is deterministic.
    order = mx.argsort(labels.astype(mx.int32) * k + positions).astype(mx.int32)
    rank = mx.zeros((k,), mx.int32)
    rank[order] = positions
    norms = mx.where(sizes > 0, mx.sum(centres * centres, axis=1), EMPTY_NORM)
    return Cells(centres, norms, order, rank, offsets, sizes)


def sorted_mirror(centers, cells, *, offset=None, precision="float32"):
    """Return the cell-sorted assignment mirror and its float32 squared norms.

    Float32 mirrors are centred on ``offset``. BF16 mirrors round the original
    coordinates; their norms are those of the represented values.
    """
    if precision == "bfloat16":
        return prepare_centers(gather_bf16(centers, cells.order))
    mirror = centers[cells.order]
    if offset is not None:
        mirror = mirror - offset
    return mirror, mx.sum(mirror * mirror, axis=1)


def probe_cells(batch, cells, n_probe):
    """Each row's ``n_probe`` nearest cells, nearest first, as int32 cell ids."""
    scores = cells.norms - 2.0 * matmul(batch.astype(mx.float32), cells.centres, transpose_b=True)
    return select_topk(scores, n_probe)[1]


def scalar_scan(batch, mirror, norms, probes, cells):
    """Best centroid of every probed cell: one score and table position per pair."""
    rows, d = batch.shape
    k = mirror.shape[0]
    n_probe = probes.shape[1]
    return tuple(_kernel("scalar_scan")(
        inputs=[batch, mirror, norms, probes, cells.offsets, _shape(rows, k)],
        template=[("D", d), ("P", n_probe)],
        grid=(rows * n_probe * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, n_probe), (rows, n_probe)],
        output_dtypes=[mx.float32, mx.uint32], stream=mx.gpu))


def reduce_links(batch, mirror, norms, scores, positions, cells, child):
    """Merge probe results into centroid ids, following split links."""
    rows, d = batch.shape
    n_probe = scores.shape[1]
    return tuple(_kernel("reduce")(
        inputs=[batch, mirror, norms, scores, positions, cells.order, cells.rank, child,
                _shape(rows)],
        template=[("D", d), ("P", n_probe)],
        grid=(rows * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows,), (rows,)], output_dtypes=[mx.uint32, mx.float32],
        stream=mx.gpu))


def bucket_pairs(probes, cells, tile_rows):
    """Group the (row, probe) pairs by cell without a sort or a host read.

    Returns the pair ids ``row * n_probe + rank`` ordered by cell, and the
    plan the tiled scan reads: per-cell pair offsets, tile offsets (tiles of
    ``tile_rows`` pairs) and centroid offsets, ``n_cells + 1`` entries each.
    """
    total = probes.size
    cell_count = cells.sizes.shape[0]
    counts = histogram(probes, cell_count)
    zero = mx.zeros((1,), mx.uint32)
    pair_offsets = mx.concatenate([zero, mx.cumsum(counts)]).astype(mx.uint32)
    tiles = (counts + (tile_rows - 1)) // tile_rows
    tile_offsets = mx.concatenate([zero, mx.cumsum(tiles)]).astype(mx.uint32)
    _, pairs = _kernel("place")(
        inputs=[probes, pair_offsets, _shape(total)], grid=(total, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(cell_count,), (total,)], output_dtypes=[mx.uint32, mx.uint32],
        init_value=0, stream=mx.gpu)
    return pairs, mx.concatenate([pair_offsets, tile_offsets, cells.offsets])


def tile_shape(dtype, d):
    """Rows, depth, padding and centroids of one scan tile: ``(BM, BK, PAD, BN)``.

    Measured on M4 Max for float32 and BF16 at 100 and 768 features: this
    shape was the fastest or within noise of it in every case. Narrower
    centroid tiles, and skipping half a tile on short cell tails, were slower.
    """
    return (64, 16, 4, 64)


def tiled_scan(batch, mirror, norms, probes, cells, *, tile=None):
    """Best centroid of every probed cell, by matrix tiles over bucketed rows.

    One threadgroup scores up to ``BM`` rows that probe the same cell against
    that cell's contiguous centroid block, ``BN`` centroids at a time, and
    keeps one score and table position per (row, probe) pair.
    """
    rows, d = batch.shape
    k = mirror.shape[0]
    n_probe = probes.shape[1]
    cell_count = cells.sizes.shape[0]
    bm, bk, pad, bn = tile_shape(batch.dtype, d) if tile is None else tile
    threads = 128
    pairs, plan = bucket_pairs(probes, cells, bm)
    # Every cell rounds its pairs up to whole tiles: at most one extra each.
    tiles = (rows * n_probe + bm - 1) // bm + cell_count
    return tuple(_tiled_scan_kernel()(
        inputs=[batch, mirror, norms, pairs, plan, _shape(rows, k, cell_count)],
        template=[("T", batch.dtype), ("D", d), ("P", n_probe), ("BM", bm), ("BN", bn),
                  ("BK", bk), ("WM", 2), ("WN", 2), ("PAD", pad)],
        grid=(tiles * threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(rows, n_probe), (rows, n_probe)],
        output_dtypes=[mx.float32, mx.uint32], stream=mx.gpu))


def two_level_assign(batch, mirror, norms, cells, n_probe, child, *, scan="auto", tile=None):
    """Assign rows to centroid ids by searching their nearest cells only.

    ``batch`` and ``mirror`` share a dtype (float32 or bfloat16) and
    coordinates; ``mirror`` is cell-sorted with ``norms`` the float32 squared
    norms of its rows. ``child`` holds each centroid's split link or -1.
    ``scan`` selects the matrix-tile scan or the scalar reference scan;
    ``auto`` uses the tiled scan wherever this MLX release can build it.
    Returns ``labels`` (uint32 centroid ids), the winning scores
    ``||c||² - 2 x·c`` and the number of distances evaluated (float32).
    """
    rows, d = batch.shape
    k = mirror.shape[0]
    cell_count = cells.centres.shape[0]
    if mirror.shape[1] != d or cells.centres.shape[1] != d:
        raise ValueError("batch, mirror and cell centres must share a feature dimension")
    if not 1 <= n_probe <= min(cell_count, MAX_PROBES):
        raise ValueError(f"n_probe must be between 1 and min(n_cells, {MAX_PROBES})")
    if rows * max(d, n_probe * 32) > _UINT32_MAX or k * d > _UINT32_MAX:
        raise ValueError("two-level assignment exceeds Metal uint32 indexing capacity")
    if scan not in {"auto", "tiled", "scalar"}:
        raise ValueError("scan must be 'auto', 'tiled' or 'scalar'")
    if scan == "auto":
        scan = "tiled" if tiled_scan_available() else "scalar"
    probes = probe_cells(batch, cells, n_probe)
    if scan == "tiled":
        scores, positions = tiled_scan(batch, mirror, norms, probes, cells, tile=tile)
    else:
        scores, positions = scalar_scan(batch, mirror, norms, probes, cells)
    labels, best = reduce_links(batch, mirror, norms, scores, positions, cells, child)
    evaluated = rows * cell_count + mx.sum(cells.sizes[probes].astype(mx.float32))
    return labels, best, evaluated


def record_splits(child, plan):
    """Update split links after a revival check.

    Each donor now points at its new child, and the child inherits the donor's
    previous link, so two hops from a donor reach its last two children.
    """
    is_child, is_donor, source, target, _ = plan
    return mx.where(is_child, child[source], mx.where(is_donor, target.astype(mx.int32), child))
