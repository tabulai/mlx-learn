"""Blocked exact Euclidean distance and k-nearest-neighbor selection in MLX.

The whole computation stays on the device. The ancestor materialized each tile's
top-k back to the host and merged it in a Python loop, which meant the GPU did the
matmul and the interpreter did everything else; for a hundred thousand queries and
a hundred tiles that is on the order of 10⁸ interpreted iterations, and it is why
its measured KNN speedups ranged down to 0.4×. Here, the running top-k is an MLX
array, tiles are merged with a concatenate-and-repartition on the device, and the
only host transfer is the final ``(n_query, k)`` result — one copy per query block.

Three correctness properties this module is responsible for:

**Exactness under float32.** Candidates are *selected* with the expanded identity
``‖q‖² + ‖t‖² − 2⟨q,t⟩`` — one matmul, and numerically hostile — on training rows
translated by their mean and scaled by a power of two, and the survivors' distances
are then *refined* directly as ``Σ(t − q)²`` from the untranslated coordinates. A
fixed over-selection margin alone is not enough: on clustered or near-duplicate data
the Gram noise exceeds the gaps between neighbors and true neighbors are silently
dropped. So every row is **certified**. A rigorous float32 error bound on the
identity proves that no training row left out of the candidate set can beat the
k-th refined distance; a row it cannot certify is searched again with a much wider
candidate set, and if that fails too — or if a count on a sample of the training
rows estimates that it would — exactly, by direct differences against every
training row. See ``docs/fp32_policy.md`` §2.1.

**A total order.** Results are the first ``k`` training rows under ``(refined
distance, training index)``, so exact ties break on the smaller index — including
ties at the k-th boundary, which the certificate treats as uncertain — and the
answer for ``k`` is a prefix of the answer for any larger ``k``. Across devices the
answers agree wherever the refined distances are exact (integer and binary data, for
example); the GPU and the CPU may round a long sum differently.

**Blocking invariance.** Any combination of query- and train-block sizes yields
identical output. The tests sweep this.

On the GPU, a first pass over enough (query row, training row) pairs forms its tile and
selects its candidates in one Metal kernel (:mod:`._topk`) instead of MLX's tile
expression and ``argpartition``. It keeps the same candidates, and the same largest kept
value, bit for bit, so every certificate, every second pass and every count is what the
built-in selection gives; see :func:`_first_pass_blocks` for what that takes.
"""

from __future__ import annotations

import functools
import math
import threading
from dataclasses import InitVar, dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from .._common.arrays import INDEX_DTYPE, NUMPY_FLOAT
from .._common.config import get_tuning
from .._common.device import physical_memory_bytes, use_device
from ..exceptions import UnsupportedInputError
from . import _topk

__all__ = ["NeighborIndex", "build_index", "resolve_blocks"]

#: Extra candidates carried per tile beyond ``k`` on the first pass — at least this
#: many, and more for wide data and large ``k`` (see :func:`_first_pass_carry`). This
#: is not what makes the answer exact — the certificate is — but it is what lets the
#: certificate succeed on the first pass for ordinary data.
_OVERSELECT = 4

#: Candidates carried on the second pass over rows the first pass could not certify:
#: ``max(4 * k_carry, k_carry + _WIDEN_EXTRA)``. When a cluster is tight relative to
#: its distance from the mean, all of its members sit inside the Gram noise band
#: together, so the second pass has to be able to carry a whole cluster —
#: ``make_blobs`` at ``cluster_std=0.01`` puts ~250 points in each.
_WIDEN_EXTRA = 256

#: A row the first pass could not certify skips the second pass, and is searched
#: exactly at once, when an estimate of the training rows inside its Gram noise band
#: reaches this multiple of the second pass's carry: with that many rows in the band,
#: the second pass has nothing to certify with (see ``_Search.cannot_widen``). On one
#: unscaled column at 1e6 x 16, where most rows need the exact search anyway, the skip
#: made a query 9-14% faster. Where the second pass does certify the rows — queries
#: beside training rows jittered or repeated many times over — the count is a device
#: round trip spent for nothing: after an idle gap, a query of 256 rows took 1.03-1.05x
#: the time, and one of a single row about 1.25x.
_SKIP_MARGIN = 2

#: Sampled rows expected in the band at the decision point. The estimate counts one
#: training row in each ``_SKIP_MARGIN * k_wide // _SAMPLE_HITS`` — one in 16 at k=10 —
#: so a true count of ``k_wide``, where the second pass starts to fail, shows about 17
#: hits on average and is called at 34 or more; the count's arithmetic is about a
#: sixteenth of a first pass.
_SAMPLE_HITS = 32

#: Seed of the hash that places the count's sampled rows (:func:`_band_offset`): fixed,
#: so a query takes the same route, and the same time, every time it is asked.
_SAMPLE_SEED = 0x5EED

#: Segment widths for two-stage selection on the GPU: at least 512, at least four
#: times the number kept, at most 8192. MLX's GPU ``argpartition`` is effectively a
#: full sort of the row, which made selection cost several times the Gram matmul on
#: a wide tile. Partitioning segments first and then the survivors measured 2.5-4×
#: faster keeping 14 of 100 000 and 1.2-2.5× keeping 64-1030, with identical sets.
_SELECT_SEGMENT = 512
_SELECT_SEGMENT_MAX = 8192

#: Peak device memory per byte of distance tile. The budget below is for the whole
#: working set, not the tile alone: the matmul output and the fused tile expression
#: each cost a tile, and a full-row GPU ``argpartition`` — still used when ``k`` is
#: too large for segmented selection — allocates five more. Measured with
#: ``mx.get_peak_memory``: the unfactored budget peaked at 7.0× (7.2 GB against
#: 1 GiB); with this factor, 0.2-0.6 GB. Query time was flat across factors 1-7. The
#: CPU evaluates the tile expression unfused, a tile or two more, which the GPU's
#: argpartition share covers there.
_TILE_WORKSET_FACTOR = 7

#: The same, for a first pass that selects with the fused kernel (:mod:`._topk`): the
#: matmul output, and room for the kernel's per-chunk lists and MLX's temporaries. No
#: second tile and no sort exists, so a block holds 3.5 times the rows — 114 query rows
#: against 38 at 1 000 000 × 64 — and the kernel's chunks fill the GPU from fewer
#: queries. Peak device memory for 4 096 queries, against a 1 GiB budget: 326 → 436 MiB
#: at 1 000 000 × 64, 298 → 596 MiB at 128 000 × 768. A block is a whole number of the
#: built-in path's blocks, at least one, and its tiles are as wide as this budget allows
#: for the rows it actually holds (:func:`_first_pass_blocks`).
_TILE_WORKSET_FUSED = 2

#: A query's first pass selects with the fused kernel only from this many (query row,
#: training row) pairs; below, MLX's tile expression and ``argpartition`` keep it.
#: Engaged on small work the kernel measured slower — 4 096 × 2 048 at 8-16 queries,
#: some 65 000 pairs — and engaged from this many pairs on, no query of its prototype's
#: benchmark grid (1 224 cells over two scikit-learn versions) was significantly slower.
#: Provisional: no grid has calibrated the crossover itself. It is not a dispatch
#: threshold — both sides are MLX, and their answers agree bit for bit — and no
#: dispatch threshold moved with it.
_FUSED_MIN_PAIRS = 1 << 22

#: And above 16 candidates, from this many training rows. On a narrow training set one
#: SIMD group scans a row's few hundred columns and then spends ``k_carry`` rounds
#: emitting its list, where MLX sorts a short row: Gaussian data at 4 and 32 features,
#: 2²² pairs, after an idle gap, 13 interleaved pairs a cell, measured the kernel at
#: 0.81x-1.09x of MLX's selection at 64-1 024 rows with 17-64 candidates, 20 of 51
#: cells losing (0.84x-0.94x; 0.89x at 40 rows, 110 000 queries and 40 candidates);
#: 0.88x-1.26x at 2 048 rows and 0.94x-1.76x from 4 096 to 32 768, none significantly
#: slower. At 16 candidates and fewer, 0.95x-2.15x at every width, none significantly
#: slower. Measured on one M4 Max, so provisional, like the pair floor.
_FUSED_MIN_TRAIN_ROWS = 4096

#: A query gives the fused kernel up for the rest of its blocks once more than one row
#: in this many of a block overflowed its register lists: each such row is selected
#: again by MLX's built-in path, so a block where most do costs both passes. Rows
#: repeated with a period that is a multiple of 128 — a set stacked 33 times or more
#: — overflow every row of a 64-candidate list: a set of 1 024 rows stacked 100 times,
#: k = 30, cost a query of one block 1.05x-1.21x the built-in path's time on two sets of
#: queries, and with this rule one of four blocks 1.01x. Gaussian, blob and low-rank
#: data flagged at most one row in 1 024.
_FUSED_MAX_REDO = 8

#: Peak device memory per byte of the ``(rows, k_carry, n_features)`` refinement
#: gather: the gathered rows, the fused difference, and — for float64 input — the
#: gathered residuals. Unbudgeted, ``k=1000`` at 768 features reached 7.4 GB.
_REFINE_WORKSET_FACTOR = 4

#: The direct-difference exact search takes at least this many training rows per
#: tile, and more when fewer query rows leave the budget room: as many as it allows,
#: in power-of-two multiples of this, so that calls whose row counts differ a little
#: reuse MLX's cached buffers. A fixed tile of this width, evaluated one at a time,
#: cost 245 synchronizations over a million training rows however few rows were
#: searched. With budget-sized tiles, queries on one unscaled column at 1e6 x 16 ran
#: 12-21x faster for one row and about 3x for 256; the skip and the 32-bit keys
#: without them, 1.02-1.05x.
_EXACT_MIN_COLS = 4096

#: Device memory per (query row, training row) pair that the exact search's key and
#: selection add to its ``3 * d * 4``-byte difference: the uint64 key and the
#: partition's temporaries, counted generously as ``_TILE_WORKSET_FACTOR`` of them.
_KEY_WORKSET_BYTES = _TILE_WORKSET_FACTOR * 8

#: Queries on small-integer training data stay on the exact integer path when they
#: are multiples of ``2**-_DYADIC_SHIFTS`` — half-integers, quarters — because
#: scaling both sides by a power of two keeps every product an integer.
_DYADIC_SHIFTS = 4

#: float32 unit roundoff.
_U = 2.0**-24

#: float32's largest finite value.
_FLOAT32_MAX = float(np.finfo(np.float32).max)

#: Largest distance from the training mean mlxlearn will subtract in float32. Twice
#: this must stay finite, because refinement subtracts two such coordinates before
#: scaling them.
_SPAN_LIMIT = _FLOAT32_MAX / 8.0

#: Room, in bits, the scaling leaves between the training span and float32's
#: overflow for queries lying farther out: one about 2**20 times the span from the
#: training mean is still served. Every bit of it is a bit less for resolving small
#: gaps under a large span; see _span_exponent.
_QUERY_HEADROOM_BITS = 20

#: The smallest scaled squared distance refinement is trusted with. Below it, the
#: squares of per-coordinate differences may have left float32's normal range — the
#: GPU flushes those to zero and the CPU keeps them as subnormals, so the devices
#: disagree and neither is right. At or above it, all that underflow can remove,
#: d·2**-126, is 2**-30·d of the value: far inside refinement's own rounding.
_RESOLVED_SQ = 2.0**-96

#: At fit, a training set whose own rows' nearest neighbors come within this factor
#: of _RESOLVED_SQ is refused, so that such data falls back at fit under
#: patch_sklearn() rather than failing its queries at inference.
_RESOLVED_MARGIN = 2.0**20

#: Training rows searched at fit for that check, spread evenly over the rows.
_PROBE_ROWS = 64

#: The exact search packs a training index into 32 bits of its sort key.
_MAX_TRAIN_ROWS = 2**32

#: Training matrices up to this many elements have their column extremes found on
#: the host at build: faster than the device round trip that reduction otherwise
#: costs (0.1 ms on the GPU), up to about 8000 x 32, where the two measured even.
_HOST_REDUCE_ELEMENTS = 1 << 18


@dataclass(frozen=True, slots=True)
class BlockPlan:
    """How a query is split into tiles."""

    query_rows: int
    train_cols: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"query_rows={self.query_rows}, train_cols={self.train_cols}"


def _distance_budget_bytes(backend: str) -> int:
    """Bytes the query working set may occupy on the device.

    Derived from installed memory and clamped, because a fixed constant is wrong by
    an order of magnitude on both an 8 GB Air and a 128 GB Studio. It bounds the
    *working set* — tile, selection temporaries and the refinement gather together —
    not merely the distance tile: bounding only the tile let the real peak reach
    seven times the budget, and MLX's buffer cache then kept that peak resident after
    the call returned. Wider tiles bought nothing once selection was segmented.
    """
    tuning = get_tuning()
    if backend != "mlx":
        return tuning.cpu_distance_block_bytes
    total = physical_memory_bytes()
    if total is None:
        return tuning.distance_block_bytes
    return int(
        min(max(total // 32, tuning.distance_block_bytes), tuning.distance_block_bytes_max)
    )


def _align_down(value: int, align: int) -> int:
    return (value // align) * align if align > 1 and value >= align else value


def resolve_blocks(
    n_query: int,
    n_train: int,
    *,
    backend: str,
    k_search: int,
    query_block_size: int | None = None,
    train_block_size: int | None = None,
    workset_factor: int = _TILE_WORKSET_FACTOR,
) -> BlockPlan:
    """Choose tile sizes so the distance tile's working set fits the budget.

    Unlike the ancestor, the device kind is passed in rather than inferred from
    whether the byte budget happens to exceed a GPU-ish constant — conflating a
    memory budget with a device kind meant an explicit CPU budget above 256 MiB
    silently switched on GPU alignment, and the CPU path could never shrink its
    block no matter how wide the training set was.

    The refinement gather, which grows with ``k * n_features`` rather than with the
    training set, is budgeted separately, by row chunks inside a block, so that a
    large ``k`` does not also shrink the matmul.

    ``workset_factor`` is the working set in tiles: ``_TILE_WORKSET_FACTOR`` for MLX's
    built-in selection, ``_TILE_WORKSET_FUSED`` for the fused kernel's.
    """
    if n_query <= 0 or n_train <= 0:
        return BlockPlan(query_rows=0, train_cols=0)

    tuning = get_tuning()
    budget = _distance_budget_bytes(backend)
    tile_budget = max(budget // workset_factor, 1)
    itemsize = 4

    if query_block_size is not None:
        rows = max(1, min(int(query_block_size), n_query))
    else:
        rows = tuning.query_block_rows if backend == "mlx" else tuning.cpu_query_block_rows
        # Keep a full-width tile inside the budget when the training set is small
        # enough to be scanned in one pass; otherwise the train tiling handles it.
        full_width_rows = tile_budget // max(n_train * itemsize, 1)
        if 0 < full_width_rows < rows:
            align = tuning.query_block_align if backend == "mlx" else 1
            rows = _align_down(full_width_rows, align)
        rows = max(1, min(n_query, rows))

    min_cols = min(k_search, n_train)
    if train_block_size is not None:
        cols = max(min_cols, min(int(train_block_size), n_train))
    else:
        cols = tile_budget // max(rows * itemsize, 1)
        cols = max(min_cols, min(n_train, int(cols)))

    return BlockPlan(query_rows=int(rows), train_cols=int(cols))


def _compute_device(backend: str, device: str | None = None) -> str:
    """The MLX device an index built for ``backend`` runs on: ``device``, or else its own.

    ``"cpu"`` is MLX's CPU and ``"mlx"`` the GPU. A process with none — a model
    fitted on a Mac, unpickled somewhere without Metal — runs the identical float32
    code on MLX's CPU instead, unless the configuration names the GPU. That is the
    estimator's decision (:func:`~mlxlearn._common.device.inference_device`), made
    where the configuration may be read and passed in as ``device``: the kernels
    read no configuration, and the device a query runs on is the one diagnostics
    report.
    """
    if device is not None:
        return device
    return "gpu" if backend == "mlx" else "cpu"


@functools.lru_cache(maxsize=2)
def _kernels(device: str) -> SimpleNamespace:
    """The elementwise pieces for ``device``: fused on the GPU, eager on the CPU.

    On the GPU ``mx.compile`` fuses each expression into one kernel, so the tile
    costs one temporary where the unfused expression cost four, and the refinement
    difference one where it cost three. On MLX's CPU device the compiled graphs ran
    no faster, and each one shells out to a C compiler once per MLX version and
    ``$TMPDIR``: a 200-row fit and predict took 3-4 s on a fresh machine, against
    0.1 s eager. So the CPU evaluates the same expressions eagerly, as
    ``kernels._tile_transform`` does for the rbf tile; the answers are identical.
    Compiled functions are cached by MLX on input shape and stream only — never on
    data — and are built lazily so that importing mlxlearn does not import MLX.
    """
    import mlx.core as mx

    def center_scale(hi, center, scale):
        return (hi - center) * scale

    def center_scale_lo(hi, lo, center, center_lo, scale):
        return ((hi - center) + (lo - center_lo)) * scale

    def tile(q_sq, t_sq, gram):
        # The clamp only ever moves a value toward the truth: distances are >= 0.
        return mx.maximum((q_sq + t_sq) - 2.0 * gram, 0.0)

    def tile_key(q_sq, t_sq, gram, index, shift):
        # The integer path's tile packed as ``distance << shift | index``. Every value
        # there is an integer below 2**24, so each step, the cast included, is exact
        # fused or not.
        return mx.left_shift(tile(q_sq, t_sq, gram).astype(mx.uint32), shift) | index

    def pair_sq(t, q, scale):
        diff = (t - q) * scale
        return diff * diff

    def pair_sq_lo(t, t_lo, q, q_lo, scale):
        diff = ((t - q) + (t_lo - q_lo)) * scale
        return diff * diff

    def band_rows(window, stride):
        # The row ``_Search.band_count`` counts in each window of ``stride`` rows.
        return window * stride + _band_offset(window, stride)

    fns = {
        "center_scale": center_scale,
        "center_scale_lo": center_scale_lo,
        "tile": tile,
        "tile_key": tile_key,
        "pair_sq": pair_sq,
        "pair_sq_lo": pair_sq_lo,
        "band_rows": band_rows,
    }
    if device == "gpu":
        fns = {name: mx.compile(fn) for name, fn in fns.items()}
    return SimpleNamespace(**fns)


def _translate(fns, hi, lo, center, center_lo, scale, zero):
    """``(hi + lo − center − center_lo) · scale`` in float32, into the index's frame.

    ``lo`` is a float64 row's residual and ``center_lo`` a float64 center's, either
    of them ``None``. Every row, training or query, residual or not, is translated
    here, so all of them share one frame; with neither residual it is the plain
    ``(hi − center) · scale`` it always was.
    """
    if lo is None and center_lo is None:
        return fns.center_scale(hi, center, scale)
    lo = zero if lo is None else lo
    center_lo = zero if center_lo is None else center_lo
    return fns.center_scale_lo(hi, lo, center, center_lo, scale)


def _gamma(n: int) -> float:
    """Higham's γₙ = n·u / (1 − n·u): the relative error bound of an n-term float32 sum."""
    nu = n * _U
    return nu / (1.0 - nu) if nu < 0.5 else math.inf


def _gram_err(n_features: int) -> float:
    """``C``: the Gram tile's float32 error is at most ``C·(‖q‖² + ‖t‖²)``."""
    return 2.0 * (2.0 * _gamma(n_features + 2) + 8.0 * _U)


def _scaled_limit(n_features: int) -> float:
    """Largest scaled coordinate whose Gram tile — ``4·d`` squares — stays finite."""
    return math.sqrt(_FLOAT32_MAX / (4.0 * max(n_features, 1)))


def _span_exponent(spread: float, n_features: int) -> int:
    """The index scales by ``2**-exponent``: exact, and only as far as float32 needs.

    A span below 1 is scaled up into [0.5, 1): data around 1e-20 squares into the
    subnormal range, which the GPU flushes to zero. A span whose squares, with
    ``2**_QUERY_HEADROOM_BITS`` of room for queries beyond it, would overflow is
    scaled down just below that, and no further. Anything between is left alone.

    Scaling down shrinks the gaps between neighbors along with the span. Every span
    used to be normalized into [0.5, 1), and one 1e20 fill value among unit-scale
    rows then put their squared gaps near 2**-134: the GPU returned every distance
    as zero and the neighbors in index order, with nothing raised. What even the
    smallest scale-down cannot resolve is refused; see :func:`_require_resolvable`.
    """
    if spread == 0.0:
        return 0
    exponent = math.frexp(spread)[1]
    if exponent <= 0:
        return max(exponent, -126)
    top = math.floor(math.log2(_scaled_limit(n_features))) - _QUERY_HEADROOM_BITS
    return max(0, exponent - top)


def _first_pass_carry(k_search: int, n_features: int, n_train: int) -> int:
    """Candidates to carry on the first pass.

    The certificate's noise band is proportional to ``γ_d``, and the number of
    neighbors crowding into it near the k-th distance grows with ``d`` and ``k``: on
    Gaussian data the rows that failed a first pass with a fixed margin of 4 went
    from none at d ≤ 64 to 29 of 200 at d=768, k=100 and all 200 at k=1000, and the
    margin that brought each back to zero tracked ``d²·u·k``. A second pass costs a
    whole extra search, so erring wide here is the cheaper mistake; the carry never
    more than doubles ``k``.
    """
    band = math.ceil(4.0 * n_features * n_features * _U * k_search)
    margin = max(_OVERSELECT, min(k_search, band))
    return min(n_train, k_search + margin)


@dataclass(frozen=True, slots=True)
class _FirstPass:
    """How a query's first pass runs: its blocks of query rows, its tiles, its selection."""

    bounds: list  # [(start, end)] of query rows, in order
    train_cols: int
    fused: _topk.Selection | None  # the fused kernel's selection, or None: MLX's own
    group_rows: int  # rows a block of the built-in path holds
    group_cols: int  # columns a train tile of the built-in path holds

    def blocks(self, n_query: int, search: _Search):
        """``(start, end, train_cols)`` of each block the first pass searches, in order.

        The fused kernel's blocks while ``search`` keeps the kernel, and the built-in
        path's once it gives the kernel up (:meth:`_Search._fused_candidates`): a fused
        block is a whole number of built-in ones, so the built-in path takes over at
        the first row of one of its own blocks.
        """
        ends = dict(self.bounds)
        start = 0
        while start < n_query:
            if search.fused is None:
                end, cols = min(start + self.group_rows, n_query), self.group_cols
            else:
                end, cols = ends[start], self.train_cols
            yield start, end, cols
            start = end


def _tiles_wider(n_train: int, train_cols: int, n_features: int) -> bool:
    """Whether every train tile of ``train_cols`` columns, the last too, is wider than deep.

    MLX 0.32.2 multiplies a product of at most ``n_features`` columns, from 128
    features, with a kernel whose bits depend on the product's row count, and a
    one-row product of a few columns too; wider than ``n_features``, a row's bits were
    those of any wider product at every row count from 2 to 1 024 and every width
    measured, 64-4 096 features. The fused path needs every product of both paths there.
    """
    cols = max(1, min(train_cols, n_train))
    return min(cols, n_train % cols or cols) > n_features


def _first_pass_blocks(
    n_query: int,
    n_train: int,
    n_features: int,
    k_carry: int,
    plan: BlockPlan,
    *,
    device: str,
    plan_backend: str,
    query_block_size: int | None = None,
    train_block_size: int | None = None,
) -> _FirstPass:
    """The blocks a first pass searches, and whether it selects with the fused kernel.

    ``plan`` is the built-in path's (``resolve_blocks`` at ``_TILE_WORKSET_FACTOR``).
    The fused kernel takes the first pass when the gate allows it for ``device``
    (:func:`._topk.selection`, which compiles the kernel on its first use, so this is
    called inside the query's device block) and the query has ``_FUSED_MIN_PAIRS``
    pairs — and, above 16 candidates, ``_FUSED_MIN_TRAIN_ROWS`` training rows. Its
    blocks are then budgeted at ``_TILE_WORKSET_FUSED``, and a block's rows still get
    the built-in path's bits. A row's Gram values do not depend on how many rows its
    product has, or how many columns, as long as every product is wider than it is
    deep (:func:`_tiles_wider`, which both paths' tiles must pass, or the built-in
    path runs) and has two rows or more: a one-row product is a matrix-vector kernel
    that rounds differently. So a fused block is a union of whole built-in blocks, the
    built-in path's one-row last block, when it has one, stays a block of its own, and
    a query whose built-in blocks are single rows keeps the built-in path. The rows a
    block cannot certify are resolved in the groups the built-in blocks put them in
    (``group_rows``), since the second pass's products depend on the grouping the
    same way, and the rows the kernel flags are selected again on the built-in path's
    tiles (``group_cols``).

    A fused block's tiles are as wide as ``_TILE_WORKSET_FUSED`` allows for the rows
    the block holds. On a training set too long for the built-in path to search a
    block's rows in one tile, the fused budget can hold fewer rows than a built-in
    block — two or three, against 1 024 at 10 million rows and a 256 MiB budget — and
    the block is then one built-in block, tiled for its own rows.
    """
    base_rows, base_cols = plan.query_rows, plan.train_cols
    base = [(s, min(s + base_rows, n_query)) for s in range(0, n_query, base_rows)]
    builtin = _FirstPass(
        bounds=base, train_cols=base_cols, fused=None, group_rows=base_rows, group_cols=base_cols
    )
    if (
        n_query * n_train < _FUSED_MIN_PAIRS
        or (k_carry > _topk.BUCKETS[0] and n_train < _FUSED_MIN_TRAIN_ROWS)
        or (base_rows < 2 and n_query > 1)
        or n_train >= 2**31  # the kernel's columns are 32-bit
        or not _tiles_wider(n_train, base_cols, n_features)
    ):
        return builtin
    wide = resolve_blocks(
        n_query,
        n_train,
        backend=plan_backend,
        k_search=k_carry,
        query_block_size=query_block_size,
        workset_factor=_TILE_WORKSET_FUSED,
    )
    group = max(1, wide.query_rows // base_rows) * base_rows
    train_cols = resolve_blocks(
        n_query,
        n_train,
        backend=plan_backend,
        k_search=k_carry,
        query_block_size=group,
        train_block_size=train_block_size,
        workset_factor=_TILE_WORKSET_FUSED,
    ).train_cols
    if (
        not _tiles_wider(n_train, train_cols, n_features)
        # The kernel reads a tile as one array, whose length MLX keeps in 32 bits.
        or min(group, n_query) * train_cols >= 2**31
    ):
        return builtin
    fused = _topk.selection(device, k_carry)
    if fused is None:
        return builtin
    return _FirstPass(
        bounds=_fused_bounds(n_query, base_rows, group),
        train_cols=train_cols,
        fused=fused,
        group_rows=base_rows,
        group_cols=base_cols,
    )


def _fused_bounds(n_query: int, base_rows: int, group: int) -> list[tuple[int, int]]:
    """Fused blocks of ``group`` rows, a whole number of built-in blocks of ``base_rows``.

    A built-in last block of one row stays a block of its own; see
    :func:`_first_pass_blocks`.
    """
    single = base_rows >= 2 and n_query > 1 and n_query % base_rows == 1
    limit = n_query - 1 if single else n_query
    bounds = [(s, min(s + group, limit)) for s in range(0, limit, group)]
    if single:
        bounds.append((n_query - 1, n_query))
    return bounds


@dataclass(frozen=True, slots=True)
class _DeviceState:
    """Everything the device holds for one index. Built whole, published once.

    Immutable, and assigned to the index only after every array in it has been
    evaluated, so a thread that sees it can use it — see ``_ensure_device``.
    """

    raw: Any  # (n, d) float32, uncentered; the index's ``X`` is a view of it
    raw_lo: Any  # (n, d) float32 residual of float64 input, or None
    sel: Any  # (n, d) (X - center) * 2**-exponent: the Gram operand
    sel_t: Any  # sel.T
    sel_sq: Any  # (n,) squared row norms of sel
    center: Any  # (d,) float32
    center_lo: Any  # (d,) float32 residual of a float64 center, or None
    scale: Any  # 0-d, 2**-exponent
    zero: Any  # 0-d, 0.0
    exponent: int
    spread: float  # largest |X - center|, unscaled
    norms_sorted: np.ndarray  # ascending float64 row norms of sel, for the certificate
    raw_norm_bound: float  # upper bound on any training row's ||x||, unscaled
    int_limit: int  # > 0 when the rows are small integers and the Gram is exact
    int_max: float  # largest |x| in the training rows, integer path only
    raw_t: Any  # raw.T, integer path only
    raw_sq: Any  # (n,) exact squared norms of raw, integer path only
    col_min: np.ndarray | None  # (d,) float64 column minima, integer path only
    col_max: np.ndarray | None  # (d,) float64 column maxima, integer path only
    index_dtype: Any


@dataclass(slots=True)
class _QueryBlock:
    """One block of query rows, in every representation the search needs."""

    hi: Any  # (B, d) float32 query rows
    lo: Any  # (B, d) float32 residual of float64 queries, or None
    sel: Any  # (B, d) translated and scaled like the training rows
    sq: Any  # (B, 1) squared norms of sel
    raw_norm: np.ndarray | None  # (B,) ||q||, only when a residual is in play
    int_shift: int | None  # 2**shift makes both sides small integers; None if not
    # With int_shift: no squared distance from these rows to a training row, both
    # lifted by 2**int_shift, exceeds this. None when the index has no column extremes.
    int_max_sq: float | None = None

    @property
    def n(self) -> int:
        return int(self.hi.shape[0])

    def take(self, mx, rows: np.ndarray) -> _QueryBlock:
        index = mx.array(rows.astype(np.int32))

        def pick(a):
            return None if a is None else mx.take(a, index, axis=0)

        return _QueryBlock(
            hi=pick(self.hi),
            lo=pick(self.lo),
            sel=pick(self.sel),
            sq=pick(self.sq),
            raw_norm=None if self.raw_norm is None else self.raw_norm[rows],
            int_shift=self.int_shift,
            int_max_sq=self.int_max_sq,  # still a bound for a subset of the rows
        )


@dataclass(slots=True, eq=False)
class NeighborIndex:
    """Device-resident training data for repeated neighbor queries.

    Owned by the fitted estimator and released with it. Deliberately *not* a
    module-level cache keyed on the identity of a user's array: the ancestor did
    that and it produced stale results whenever a caller mutated an array in
    place, leaked a full float32 copy plus its transpose for the lifetime of the
    process, and was not thread-safe. An agent refitting a probe on new data must
    never get the previous fit's coefficients back.

    Attributes
    ----------
    X : ndarray of shape (n_samples, n_features)
        The training rows in float32, C-ordered, and never the caller's buffer: an
        estimator that aliases the caller's array changes its own predictions when
        the caller mutates it. Once the device copy exists this is a read-only,
        zero-copy view of it, so on unified memory the host and the device share
        one buffer instead of holding two.
    center : ndarray of shape (n_features,)
        The translation applied before the Gram identity: the training mean.
        Distances are translation-invariant, so any shift is exact in real
        arithmetic; the mean is the one that keeps float32 cancellation manageable.
        float64 for wider-than-float32 input, and applied as a float32 value and
        residual, like ``X``: rounded to float32, a mean near 1.7e9 is up to 64 away
        from the true one, and that offset alone swamped the certificate for
        timestamps a few seconds apart. float32 — the mean, rounded — for float32
        input.
    backend : {"mlx", "cpu"}
        Which MLX device the index was built for.
    device : {"gpu", "cpu"}, optional
        Init only: the MLX device to build on, as the caller resolved it; by
        default the one ``backend`` names. Not kept, and not pickled: each query
        says where it runs.
    X_lo : ndarray of shape (n_samples, n_features) or None
        For wider-than-float32 input, the float32 residual ``X64 − float32(X64)``.
        Refinement and centering use ``X + X_lo``, so a float64 caller's precision
        is not thrown away by the cast — a column of Unix timestamps keeps its
        seconds. ``None`` when the input was float32 or exactly representable.
    """

    X: np.ndarray
    center: np.ndarray
    backend: str
    X_lo: np.ndarray | None = None
    _state: _DeviceState | None = field(default=None, repr=False)
    _host64: np.ndarray | None = field(default=None, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)
    device: InitVar[str | None] = None

    # ------------------------------------------------------------------ build

    def __post_init__(self, device: str | None) -> None:
        self._ensure_device(device)

    def _ensure_device(self, device: str | None = None) -> _DeviceState:
        """Build and publish the device state. Idempotent and thread-safe.

        It doubles as the rebuild path after unpickling, and there the first query
        may well be a burst from a thread pool. This method used to publish the
        device arrays one at a time, still unevaluated, so a second thread could pass
        the "already built" check and read ``None`` — or evaluate a lazy graph
        belonging to another thread's stream, which MLX answers with "There is no
        Stream(gpu, N) in current thread" or a segfault. Here the state is built
        into locals, evaluated, and published with one assignment, under
        double-checked locking so that concurrent first queries also do not each
        upload their own copy of the training set.
        """
        state = self._state
        if state is not None:
            return state
        with self._lock:
            if self._state is None:
                state, x_view, lo_view = _build_state(
                    self.X, self.X_lo, self.center, _compute_device(self.backend, device)
                )
                self.X = x_view
                self.X_lo = lo_view
                self._state = state  # published last: the unlocked check reads only this
            return self._state

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    # ------------------------------------------------------------------ pickle

    def __getstate__(self) -> dict:
        """Pickle the host data only; the device tensors are derived.

        Keeping them would multiply the pickle — the translated matrix, its row
        norms and their sorted copy are all recomputable from ``X``, ``X_lo`` and
        ``center`` — and would embed device buffers in a file that may be loaded on
        a machine with a different MLX build or no GPU at all.
        """
        state = {"X": self.X, "center": self.center, "backend": self.backend}
        if self.X_lo is not None:
            state["X_lo"] = self.X_lo
        return state

    def __setstate__(self, state: dict) -> None:
        self.X = state["X"]
        self.center = state["center"]
        self.backend = state["backend"]
        self.X_lo = state.get("X_lo")
        # A lock is process-local and never pickled; each unpickled index gets its own.
        self._lock = threading.Lock()
        self._state = None
        self._host64 = None
        # Rebuilt lazily on the first query rather than here, so that unpickling
        # never touches a device, and on the device that query runs on; see
        # _compute_device.

    def host_rows(self) -> np.ndarray:
        """The training rows in float64, translated by ``center``: built once and kept.

        ``X + X_lo − center``, the input to about 2⁻⁴⁸ of each coordinate (float32
        input exactly) and translated by the same mean the search uses, for a caller
        that answers a query on the host instead of searching the index: translated
        by ``center`` too, its queries get the distances a search around the origin
        would lose to cancellation on data far from it. Read-only. Built on the first
        call, under the index's lock so concurrent first calls build one copy; like
        the device state, it is derived, and not pickled.
        """
        rows = self._host64
        if rows is not None:
            return rows
        with self._lock:
            if self._host64 is None:
                rows = self.X.astype(np.float64)
                if self.X_lo is not None:
                    rows += self.X_lo
                rows -= self.center.astype(np.float64)
                rows.flags.writeable = False
                self._host64 = rows
            return self._host64

    # ------------------------------------------------------------------ probe

    def probe(self, k: int, *, rows: int = _PROBE_ROWS, device: str | None = None) -> dict:
        """How the first pass fares on the index's own rows: what a query will cost.

        Searches ``rows`` training rows (all of them, if the index has fewer),
        spread evenly over the index, for their ``k`` nearest (each row finds
        itself, or a duplicate, first) with the first float32 pass alone, and counts
        the rows its certificate could not certify. A query row like those is
        searched again with a much wider candidate set, and exactly if that fails
        too (:meth:`query`), which costs several first passes: on duplicated rows,
        tight clusters or one column on a far larger scale than the rest, nearly
        every row. Nothing is resolved here, so the probe costs one first pass over
        the index for ``rows`` queries. The caller chooses ``rows``: the estimator
        passes ``Tuning.knn_probe_rows``, since the index reads no configuration.

        Integer data is searched on the exact integer path, which has no
        certificate; the probe says so instead of searching.

        Returns
        -------
        dict
            ``{"rows": probed, "uncertified": count, "integer": bool}``.
        """
        import mlx.core as mx

        device = _compute_device(self.backend, device)
        state = self._ensure_device(device)
        n_train = self.n_samples
        if state.int_limit > 0:
            return {"rows": 0, "uncertified": 0, "integer": True}
        rows = np.unique(np.linspace(0, n_train - 1, min(n_train, rows)).astype(np.intp))
        k_search = max(1, min(int(k), n_train))
        k_carry = _first_pass_carry(k_search, self.n_features, n_train)
        if rows.size == 0 or k_carry >= n_train:
            # Every row is a candidate, so the first pass is the exact answer.
            return {"rows": int(rows.size), "uncertified": 0, "integer": False}
        probe = self.X[rows]
        if self.X_lo is not None:
            probe = probe.astype(np.float64) + self.X_lo[rows]
        plan_backend = "mlx" if device == "gpu" else "cpu"
        plan = resolve_blocks(rows.size, n_train, backend=plan_backend, k_search=k_carry)
        uncertified = 0
        with use_device(device):
            # The probe is a query like any other: from _FUSED_MIN_PAIRS pairs it selects
            # with the fused kernel, compiling it if this is the process's first.
            first = _first_pass_blocks(
                rows.size,
                n_train,
                self.n_features,
                k_carry,
                plan,
                device=device,
                plan_backend=plan_backend,
            )
            search = _Search(
                index=self,
                state=state,
                mx=mx,
                k_search=k_search,
                k_carry=k_carry,
                segmented=device == "gpu",
                fns=_kernels(device),
                plan_backend=plan_backend,
                train_block_size=None,
                counts={"rows": int(rows.size), "widened_rows": 0, "widen_skipped_rows": 0,
                        "exact_rows": 0},
                fused=first.fused,
                group_rows=first.group_rows,
                group_cols=first.group_cols,
            )
            for start, end, train_cols in first.blocks(rows.size, search):
                for _, _, block in search.query_blocks(probe, start, end, self_query=False):
                    cand, gmax = search.first_pass(block, train_cols)
                    refined = search.refine(block, cand)
                    mx.eval(cand, refined, gmax, block.sq)
                    _, sq = _sort_candidates(np.asarray(cand), np.asarray(refined))
                    ok = search.certified(block, np.asarray(gmax), sq[:, k_search - 1])
                    uncertified += int(np.count_nonzero(~ok))
        return {"rows": int(rows.size), "uncertified": uncertified, "integer": False}

    # ------------------------------------------------------------------ query

    def query(
        self,
        Q: np.ndarray,
        k: int,
        *,
        return_distance: bool = True,
        exclude_self: bool = False,
        query_block_size: int | None = None,
        train_block_size: int | None = None,
        stats: dict | None = None,
        device: str | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Return the ``k`` nearest training rows for each row of ``Q``.

        Parameters
        ----------
        Q : ndarray of shape (n_queries, n_features)
            Query rows, float32 or float64. float64 rows are split into a float32
            value and residual exactly as the training rows are, so their precision
            is used rather than discarded by a cast.
        k : int
            Neighbors to return, before ``exclude_self`` removes one.
        return_distance : bool, default=True
            Whether to compute and return Euclidean distances.
        exclude_self : bool, default=False
            Treat ``Q`` as the training matrix itself and drop each row's own
            index. Requires ``len(Q) == n_samples``. Passing this index's own ``X``
            queries with the device-resident training rows directly.
        stats : dict, optional
            If given, filled with how the rows were answered: ``"rows"``,
            ``"widened_rows"`` (the first pass could not certify them),
            ``"widen_skipped_rows"`` (of those, the rows a sampled count estimated a
            wider pass could not certify either, so it was skipped) and
            ``"exact_rows"`` (searched exactly: those, and the rows the wider pass
            could not certify). The kernel knows nothing of estimators, so it is the
            caller's to record.
        device : {"gpu", "cpu"}, optional
            The MLX device to search on, as the caller resolved it; by default the
            one the index was built for (``backend``).

        Returns
        -------
        distances : ndarray of shape (n_queries, k) or None
            Euclidean (not squared) distances, float64, ascending.
        indices : ndarray of shape (n_queries, k)
            Training-row indices, int64: the first ``k`` under ``(distance, index)``.

        Notes
        -----
        ``exclude_self`` compares against each row's **global** index. The
        ancestor compared against the block-local row number, so every query block
        after the first tested the wrong identity and a row could be returned as
        its own nearest neighbor — verified on 1500 rows with default blocking.
        """
        import mlx.core as mx

        # Decided before the device build, which swaps ``X`` for a view of the
        # device copy: the caller handed us the old object.
        self_query = exclude_self and Q is self.X
        device = _compute_device(self.backend, device)
        state = self._ensure_device(device)
        n_query = int(Q.shape[0])
        n_train = self.n_samples
        k = int(k)

        if exclude_self and n_query != n_train:
            raise ValueError(
                "exclude_self requires the query matrix to be the training matrix: "
                f"got {n_query} query rows and {n_train} training rows."
            )
        k_search = k + int(exclude_self)
        if k_search > n_train:
            limit = n_train - int(exclude_self)
            raise ValueError(f"n_neighbors must be <= {limit}, got {k}.")

        if n_query == 0:
            empty_idx = np.empty((0, k), dtype=INDEX_DTYPE)
            empty_dist = np.empty((0, k), dtype=np.float64) if return_distance else None
            return empty_dist, empty_idx

        if not self_query:
            Q = np.asarray(Q)
            if Q.ndim != 2 or Q.shape[1] != self.n_features:
                raise ValueError(
                    f"Q must have shape (n_queries, {self.n_features}); got {Q.shape}."
                )

        # Carry extra candidates so the certificate usually succeeds first time.
        k_carry = _first_pass_carry(k_search, self.n_features, n_train)
        # Budgeted for the device the query runs on, which for a GPU-fitted index
        # loaded on a GPU-less host is MLX's CPU, not the device it was built for.
        plan_backend = "mlx" if device == "gpu" else "cpu"
        plan = resolve_blocks(
            n_query,
            n_train,
            backend=plan_backend,
            k_search=k_carry,
            query_block_size=query_block_size,
            train_block_size=train_block_size,
        )

        out_idx = np.empty((n_query, k), dtype=INDEX_DTYPE)
        out_dist = np.empty((n_query, k), dtype=np.float64) if return_distance else None

        with use_device(device):
            first = _first_pass_blocks(
                n_query,
                n_train,
                self.n_features,
                k_carry,
                plan,
                device=device,
                plan_backend=plan_backend,
                query_block_size=query_block_size,
                train_block_size=train_block_size,
            )
            search = _Search(
                index=self,
                state=state,
                mx=mx,
                k_search=k_search,
                k_carry=k_carry,
                segmented=device == "gpu",
                fns=_kernels(device),
                plan_backend=plan_backend,
                train_block_size=train_block_size,
                counts={"rows": n_query, "widened_rows": 0, "widen_skipped_rows": 0,
                        "exact_rows": 0},
                fused=first.fused,
                group_rows=first.group_rows,
                group_cols=first.group_cols,
            )
            for start, end, train_cols in first.blocks(n_query, search):
                for b_start, b_end, block in search.query_blocks(
                    Q, start, end, self_query=self_query
                ):
                    idx_np, sq_np = search.block(block, train_cols)

                    if exclude_self:
                        idx_np, sq_np = _drop_self(idx_np, sq_np, offset=b_start)

                    out_idx[b_start:b_end] = idx_np[:, :k]
                    if return_distance:
                        out_dist[b_start:b_end] = np.sqrt(sq_np[:, :k])

        if stats is not None:
            stats.update(search.counts)
        return out_dist, out_idx


@dataclass(slots=True)
class _Search:
    """One ``query`` call's worth of search machinery over a built index."""

    index: NeighborIndex
    state: _DeviceState
    mx: Any
    k_search: int
    k_carry: int
    segmented: bool
    fns: SimpleNamespace  # _kernels() for the device the search runs on
    plan_backend: str  # "mlx" or "cpu": whose memory budget the blocks follow
    train_block_size: int | None
    counts: dict
    # The fused kernel's selection of k_carry candidates, while it takes the first pass
    # (_first_pass_blocks); None runs MLX's tile expression and argpartition. A search
    # sets it to None when it gives the kernel up (_fused_candidates).
    fused: _topk.Selection | None = None
    # Rows the built-in path's query blocks hold. A fused block is a union of them: it
    # makes their choices (query_blocks), and the rows it cannot certify are resolved
    # in those groups; 0 resolves a block's rows together.
    group_rows: int = 0
    # Columns the built-in path's train tiles hold: the tiles a row the fused kernel
    # flags is selected again on. 0 when no kernel runs.
    group_cols: int = 0

    # ------------------------------------------------------------- query rows

    def query_blocks(self, Q, start: int, end: int, *, self_query: bool) -> list:
        """Rows ``start:end`` as the ``(start, end, block)`` the search answers in turn.

        One block, unless rows ``start:end`` are a fused block whose built-in blocks
        (``group_rows`` each) would not all choose alike. :meth:`query_block` makes
        two choices for a block as a whole: whether its rows carry a float64 residual —
        and with it whether the certificate allows for the residual's rounding — and,
        on integer training data, whether they take the exact integer path. A union of
        blocks that choose alike chooses as each of them does; otherwise each run of
        such blocks is a block of its own, and a block on the integer path always is,
        so every row is searched as the built-in path searches it.
        """
        group = self.group_rows
        if self_query or not group or end - start <= group:
            return [(start, end, self.query_block(Q, start, end, self_query=self_query))]
        state = self.state
        hi_np, rows64, lo_np = _split_query(Q[start:end])
        # Per row, one pass each, as query_block makes them per block.
        offsets = np.max(np.abs(rows64 - self.index.center.astype(np.float64)), axis=1)
        residual_rows = None if lo_np is None else lo_np.any(axis=1)
        runs: list[list] = []  # [first row, end, key], relative to start
        for a in range(0, end - start, group):
            b = min(a + group, end - start)
            # Raised for the first block that would raise, as the built-in path does.
            self.check_offset(float(np.max(offsets[a:b])))
            residual = residual_rows is not None and bool(residual_rows[a:b].any())
            integer = (
                state.int_limit > 0
                and not residual
                and _dyadic_shift(hi_np[a:b], state.int_max, state.int_limit) is not None
            )
            key = ("integer", a) if integer else ("float", residual)
            if runs and runs[-1][2] == key:
                runs[-1][1] = b
            else:
                runs.append([a, b, key])
        return [
            (start + a, start + b, self.host_block(
                hi_np[a:b], rows64[a:b], None if lo_np is None else lo_np[a:b],
                residual=key[1] if key[0] == "float" else False,
            ))
            for a, b, key in runs
        ]

    def query_block(self, Q, start: int, end: int, *, self_query: bool) -> _QueryBlock:
        """Prepare rows ``start:end`` of the query in every representation needed."""
        state = self.state

        if self_query:
            # The training rows themselves: already on the device, already
            # translated, and identical bit for bit to what they are compared with.
            lo = None if state.raw_lo is None else state.raw_lo[start:end]
            raw_norm = None
            if lo is not None:
                host = np.asarray(self.index.X[start:end], dtype=np.float64)
                raw_norm = np.sqrt(np.einsum("ij,ij->i", host, host))
            integer = state.int_limit > 0
            return _QueryBlock(
                hi=state.raw[start:end],
                lo=lo,
                sel=state.sel[start:end],
                sq=state.sel_sq[start:end, None],
                raw_norm=raw_norm,
                int_shift=0 if integer else None,
                int_max_sq=(
                    _integer_sq_bound(state, state.col_min, state.col_max, 0) if integer else None
                ),
            )

        hi_np, rows64, lo_np = _split_query(Q[start:end])
        self.check_offset(float(np.max(np.abs(rows64 - self.index.center.astype(np.float64)))))
        return self.host_block(hi_np, rows64, lo_np)

    def check_offset(self, offset: float) -> None:
        """Raise if query rows lying up to ``offset`` from the training mean overflow float32.

        Far enough out, a query overflows float32 even after scaling. Inference cannot
        fall back — the backend is sticky — so this is an error rather than a row of
        infinities.
        """
        state = self.state
        if not math.isfinite(offset) or offset > _SPAN_LIMIT or (
            math.ldexp(offset, -state.exponent) > _scaled_limit(self.index.n_features)
        ):
            raise UnsupportedInputError(
                f"A query row lies {offset:.3g} from the training data's mean, which "
                "overflows float32 distance arithmetic at the training data's scale. "
                "Rescale the queries, or compute these neighbors with scikit-learn.",
                reason="float32-overflow",
            )

    def host_block(
        self, hi_np: np.ndarray, rows64: np.ndarray, lo_np, *, residual: bool | None = None
    ) -> _QueryBlock:
        """A :class:`_QueryBlock` of query rows split by :func:`_split_query`.

        The block's choices are made here, for its rows together: the residual, when
        any row has one — ``residual``, when the caller has already looked — and the
        integer path, when every row can take it.
        """
        mx, state = self.mx, self.state
        fns = self.fns
        if residual is None:
            residual = lo_np is not None and bool(lo_np.any())
        if not residual:
            lo_np = None

        hi = mx.array(hi_np)
        lo = None if lo_np is None else mx.array(lo_np)
        sel = _translate(fns, hi, lo, state.center, state.center_lo, state.scale, state.zero)
        sq = mx.sum(sel * sel, axis=1, keepdims=True)

        raw_norm = None
        if lo is not None or state.raw_lo is not None:
            raw_norm = np.sqrt(np.einsum("ij,ij->i", rows64, rows64))

        int_shift = int_max_sq = None
        if state.int_limit > 0 and lo_np is None:
            int_shift = _dyadic_shift(hi_np, state.int_max, state.int_limit)
            if int_shift is not None:
                int_max_sq = _integer_sq_bound(
                    state, hi_np.min(axis=0), hi_np.max(axis=0), int_shift
                )
        return _QueryBlock(
            hi=hi, lo=lo, sel=sel, sq=sq, raw_norm=raw_norm, int_shift=int_shift,
            int_max_sq=int_max_sq,
        )

    # ------------------------------------------------------------------ search

    def block(self, qb: _QueryBlock, train_cols: int) -> tuple[np.ndarray, np.ndarray]:
        """The first ``k_search`` neighbors of each row of ``qb``.

        Returns host arrays: indices, and float64 squared distances in the data's
        own units, ordered by ``(distance, index)``.
        """
        mx, state = self.mx, self.state
        n_train = self.index.n_samples

        if qb.int_shift is not None:
            # Small integers: the Gram identity on the raw rows is exact, so an
            # exact search is one matmul pass with no certificate to fail.
            idx, sq = self.exact(qb, integer=True)
            return idx, np.ldexp(sq.astype(np.float64), -2 * qb.int_shift)

        cand, gmax = self.first_pass(qb, train_cols)
        refined = self.refine(qb, cand)
        mx.eval(cand, refined, gmax, qb.sq)
        idx, sq = _sort_candidates(np.asarray(cand), np.asarray(refined))
        idx, sq = idx[:, : self.k_search], sq[:, : self.k_search]

        if self.k_carry < n_train:
            gmax = np.asarray(gmax)
            ok = self.certified(qb, gmax, sq[:, -1])
            if not ok.all():
                bad = np.flatnonzero(~ok)
                groups = [bad] if not self.group_rows else np.split(
                    bad, np.flatnonzero(np.diff(bad // self.group_rows)) + 1
                )
                for rows in groups:
                    idx[rows], sq[rows] = self.resolve(qb.take(mx, rows), gmax[rows])

        if state.exponent > 0:
            self.require_resolved(qb, idx, sq)
        return idx, np.ldexp(sq.astype(np.float64), 2 * state.exponent)

    def require_resolved(self, qb: _QueryBlock, idx: np.ndarray, sq: np.ndarray) -> None:
        """Raise if a neighbor returned for ``qb`` is nearer than float32 resolves.

        Only a span scaled down can put a distance below ``_RESOLVED_SQ``, where
        underflow has taken terms from it — all of them on the GPU. An exact
        duplicate of the query is the one neighbor that legitimately sits there.
        Any other would sort first and be returned, so checking the returned
        neighbors is enough; and the certificate, which compares against the k-th
        refined distance, is sound only once that distance is resolved.
        """
        rows, cols = np.nonzero(sq < _RESOLVED_SQ)
        if rows.size == 0:
            return
        train = idx[rows, cols]
        differ = np.any(self.index.X[train] != np.asarray(qb.hi)[rows], axis=1)
        if self.index.X_lo is not None or qb.lo is not None:
            t_lo = 0.0 if self.index.X_lo is None else self.index.X_lo[train]
            q_lo = 0.0 if qb.lo is None else np.asarray(qb.lo)[rows]
            differ |= np.any(t_lo != q_lo, axis=1)
        if differ.any():
            raise UnsupportedInputError(
                "A query row lies nearer to a training row than float32 can resolve at "
                "the training data's scale: the data spans so far from its mean that "
                "the squares of gaps this small underflow. Rescale the data, or compute "
                "these neighbors with scikit-learn.",
                reason="float32-overflow",
            )

    def first_pass(self, qb: _QueryBlock, train_cols: int):
        """The first pass's ``k_carry`` candidates and largest kept Gram value per row.

        By the fused kernel when it takes the pass, else by :meth:`candidates`; the two
        agree bit for bit (:mod:`._topk`).
        """
        if self.fused is None:
            return self.candidates(qb, self.k_carry, train_cols)
        return self._fused_candidates(qb, train_cols)

    def _fused_candidates(self, qb: _QueryBlock, train_cols: int):
        """:meth:`candidates` for ``k_carry``, by the fused kernel: the same set and value.

        One kernel per train tile forms the tile from the matmul output, selects, and
        merges with the tiles before. The list comes back ascending by ``(value,
        index)``, so its last value is the largest kept. A row whose register lists may
        have dropped a candidate (possible only above 32 candidates, and flagged, never
        guessed) is searched again by :meth:`candidates`, on the built-in path's tiles.
        When more than one row in ``_FUSED_MAX_REDO`` was, the search gives the kernel
        up: its later blocks are the built-in path's (:meth:`_FirstPass.blocks`).
        """
        mx, state = self.mx, self.state
        n_train = self.index.n_samples
        k_carry = self.k_carry
        budget = _distance_budget_bytes(self.plan_backend)
        col_bytes = qb.n * 4 * _TILE_WORKSET_FUSED  # a tile column, as resolve_blocks budgets it
        in_flight = 0
        best = flags = None
        for t_start in range(0, n_train, train_cols):
            t_end = min(t_start + train_cols, n_train)
            gram = qb.sel @ state.sel_t[:, t_start:t_end]
            # sel_sq is an array of its own, so the slice starts t_start elements into
            # its buffer: what the kernel's four-wide loads need to know.
            vals, idx, flagged = self.fused.smallest(
                mx, gram, qb.sq, state.sel_sq[t_start:t_end], col0=t_start, t_offset=t_start,
                prev=best,
            )
            best = (vals, idx)
            if flagged is not None:
                flags = flagged if flags is None else mx.maximum(flags, flagged)
            # Bounded as in :meth:`candidates`.
            in_flight += (t_end - t_start) * col_bytes
            following = min(train_cols, n_train - t_end)
            if following and in_flight + following * col_bytes > budget:
                mx.eval(*best, *([] if flags is None else [flags]))
                mx.synchronize()
                in_flight = 0
        vals, idx = best
        gmax = vals[:, k_carry - 1]
        if flags is None:
            return idx, gmax
        bad = np.flatnonzero(np.asarray(flags))
        if bad.size == 0:
            return idx, gmax
        # On the built-in path's tiles, each wider than deep (_tiles_wider), so that a
        # product of fewer rows than a built-in block gives its rows the same bits — and
        # a built-in block's rows at most at once, so that the tile stays in its budget.
        group = self.group_rows or qb.n
        parts = []
        for rows in np.split(bad, np.flatnonzero(np.diff(bad // group)) + 1):
            take = rows
            if rows.size == 1 and qb.n > 1:
                # Searched alone, the row would be a one-row product, a matrix-vector
                # kernel that rounds differently: a second row, discarded, keeps its bits.
                take = np.array([rows[0], 1 if rows[0] == 0 else 0])
            part_idx, part_gmax = self.candidates(qb.take(mx, take), k_carry, self.group_cols)
            parts.append((part_idx[: rows.size], part_gmax[: rows.size]))
        sub_idx = mx.concatenate([p[0] for p in parts], axis=0)
        sub_gmax = mx.concatenate([p[1] for p in parts], axis=0)
        redo = np.zeros(qb.n, dtype=bool)
        redo[bad] = True
        at = np.zeros(qb.n, dtype=np.int32)
        at[bad] = np.arange(bad.size, dtype=np.int32)
        redo_dev, at_dev = mx.array(redo), mx.array(at)
        idx = mx.where(redo_dev[:, None], mx.take(sub_idx, at_dev, axis=0), idx)
        gmax = mx.where(redo_dev, mx.take(sub_gmax, at_dev, axis=0), gmax)
        if bad.size * _FUSED_MAX_REDO > qb.n:
            self.fused = None
        return idx, gmax

    def candidates(self, qb: _QueryBlock, k_carry: int, train_cols: int):
        """Running top-``k_carry`` by Gram value over train tiles, entirely on device.

        Returns the candidate indices and, per row, the largest Gram value kept.
        Every training row left out has a Gram value at least that large — each
        partition and merge keeps the smallest — which is what the certificate
        builds on.
        """
        mx, state = self.mx, self.state
        fns = self.fns
        n_train = self.index.n_samples
        budget = _distance_budget_bytes(self.plan_backend)
        col_bytes = qb.n * 4 * _TILE_WORKSET_FACTOR  # a tile column, as resolve_blocks budgets it
        in_flight = 0
        best_idx = None
        best_val = None

        for t_start in range(0, n_train, train_cols):
            t_end = min(t_start + train_cols, n_train)
            gram = qb.sel @ state.sel_t[:, t_start:t_end]
            tile = fns.tile(qb.sq, state.sel_sq[None, t_start:t_end], gram)

            take = min(k_carry, t_end - t_start)
            local, vals = _select_smallest(mx, tile, take, segmented=self.segmented)
            idx = local.astype(state.index_dtype) + t_start

            if best_idx is None:
                best_idx, best_val = idx, vals
            else:
                merged_idx = mx.concatenate([best_idx, idx], axis=1)
                merged_val = mx.concatenate([best_val, vals], axis=1)
                keep = min(k_carry, merged_val.shape[1])
                sel = mx.argpartition(merged_val, kth=keep - 1, axis=1)[:, :keep]
                best_idx = mx.take_along_axis(merged_idx, sel, axis=1)
                best_val = mx.take_along_axis(merged_val, sel, axis=1)

            # Evaluated once one more tile would take the working set in flight past
            # the budget, as in :meth:`exact`, but never after the last tile: the
            # caller evaluates the result with its refinement, and a block searched in
            # one tile — the usual case — gains no synchronization. Left lazy, every
            # tile was alive at once: 40 000 000 x 2, 64 queries, peaked 7.0 GiB above
            # the index against 1 GiB, in this pass and in the second. ``eval``
            # returns once the values are ready, but MLX takes the tiles' buffers back
            # only when their command buffer completes, on a thread of its own; so the
            # stream is synchronized too, or the next tiles are sometimes allocated
            # beside the last ones: 8-10 MiB against a 4 MiB budget in 7 of 120 runs
            # of the tests that hold this pass and the fused one to it.
            in_flight += (t_end - t_start) * col_bytes
            following = min(train_cols, n_train - t_end)
            if following and in_flight + following * col_bytes > budget:
                mx.eval(best_idx, best_val)
                mx.synchronize()
                in_flight = 0

        return best_idx, mx.max(best_val, axis=1)

    def refine(self, qb: _QueryBlock, cand_idx):
        """Recompute the candidates' squared distances directly.

        ``Σ(t − q)²`` from the *untranslated* coordinates, not the expanded
        identity and not the translated copy: translation rounds every coordinate
        to a precision set by its distance from the mean, which for data far from
        its own mean — two clusters, a heavy-tailed column — is far coarser than the
        input itself. Subtracting the raw values rounds once, relative to the
        difference. float64 input also carries its residual, so the difference is
        taken to nearly double precision before it is squared.
        """
        mx, state = self.mx, self.state
        fns = self.fns
        n_rows, k_carry = (int(s) for s in cand_idx.shape)

        # The gather is (rows, k_carry, n_features): with a large k on wide data it
        # dwarfs the distance tile (7.4 GB at k=1000, d=768 before this was
        # budgeted), so it runs in row chunks sized to the budget. Each chunk is
        # evaluated before the next is built: MLX encodes lazily and runs ahead of
        # the GPU, so unevaluated chunks all hold their temporaries at once — lazy
        # chunking alone measured 3.2 GB against a 1 GiB budget.
        gather = _REFINE_WORKSET_FACTOR * k_carry * self.index.n_features * 4
        chunk = max(1, _distance_budget_bytes(self.plan_backend) // max(gather, 1))
        if chunk < n_rows:
            # Finish selection first, so its temporaries are gone before the
            # first chunk's gather is allocated.
            mx.eval(cand_idx)

        parts = []
        for start in range(0, n_rows, chunk):
            end = min(start + chunk, n_rows)
            idx = cand_idx[start:end]
            t = mx.take(state.raw, idx, axis=0)
            q = qb.hi[start:end, None, :]
            if state.raw_lo is None and qb.lo is None:
                sq = fns.pair_sq(t, q, state.scale)
            else:
                t_lo = state.zero if state.raw_lo is None else mx.take(state.raw_lo, idx, axis=0)
                q_lo = state.zero if qb.lo is None else qb.lo[start:end, None, :]
                sq = fns.pair_sq_lo(t, t_lo, q, q_lo, state.scale)
            parts.append(mx.sum(sq, axis=2))
            if chunk < n_rows:
                mx.eval(parts[-1])
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)

    # ------------------------------------------------------------ certificate

    def certified(self, qb: _QueryBlock, gmax: np.ndarray, r_k: np.ndarray) -> np.ndarray:
        """Rows whose candidate set provably contains their first ``k_search``.

        A training row left out of the candidates has Gram value ``G ≥ gmax``. The
        float32 error of the identity is at most ``C·(‖q‖² + ‖t‖²)``, and the
        reverse triangle inequality gives ``(‖t‖ − ‖q‖)²`` — which is what keeps a
        single far-out training row from making every row uncertifiable. The lower
        bound on its true distance, discounted by refinement's own rounding, must
        exceed the k-th refined distance *strictly*: a row whose excluded
        neighbors might merely tie is not certified, because which tied row makes
        the cut is decided by index, and only an exact search knows the indices.
        """
        state = self.state
        d = self.index.n_features
        q_sq = np.asarray(qb.sq, dtype=np.float64).reshape(-1)
        bound = _excluded_lower_bound(
            q_sq,
            gmax.astype(np.float64),
            state.norms_sorted,
            gram_err=_gram_err(d),
            norm_err=2.0 * _gamma(d + 2),
        )
        lower = bound * (1.0 - 2.0 * _gamma(d + 4))
        if qb.raw_norm is not None:
            # The float64 residuals — the rows' and the center's — are themselves
            # rounded, at u² of the raw magnitudes. The center is the training mean,
            # inside the rows' bounding box, so raw_norm_bound covers it as well.
            absolute = math.ldexp(12.0 * _U * _U, -state.exponent) * (
                qb.raw_norm + state.raw_norm_bound
            )
            lower = lower - 2.0 * np.sqrt(np.maximum(bound, 0.0)) * absolute - absolute**2
        return lower > r_k.astype(np.float64)

    # --------------------------------------------------------------- fallback

    def resolve(self, qb: _QueryBlock, gmax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Search rows the first pass could not certify: wider, then exactly.

        ``gmax`` is each row's largest Gram value kept by the first pass. The rows
        :meth:`cannot_widen` picks out skip the wider pass and are searched exactly at
        once.
        """
        mx = self.mx
        n_train = self.index.n_samples
        k_search = self.k_search
        self.counts["widened_rows"] += qb.n
        out_idx = np.empty((qb.n, k_search), dtype=INDEX_DTYPE)
        out_sq = np.empty((qb.n, k_search), dtype=NUMPY_FLOAT)
        pending = np.arange(qb.n)

        k_wide = min(n_train, max(4 * self.k_carry, self.k_carry + _WIDEN_EXTRA))
        if k_wide < n_train:
            skip = self.cannot_widen(qb, gmax, k_wide)
            self.counts["widen_skipped_rows"] += int(np.count_nonzero(skip))
            wide = np.flatnonzero(~skip)
            still = [np.flatnonzero(skip)]
            plan = resolve_blocks(
                wide.size,
                n_train,
                backend=self.plan_backend,
                k_search=k_wide,
                train_block_size=self.train_block_size,
            )
            # An empty plan, of zero rows, when every row skips.
            for start in range(0, wide.size, plan.query_rows or 1):
                rows = wide[start : start + plan.query_rows]
                sub = qb.take(mx, rows)
                cand, gmax_wide = self.candidates(sub, k_wide, plan.train_cols)
                refined = self.refine(sub, cand)
                mx.eval(cand, refined, gmax_wide, sub.sq)
                idx, sq = _sort_candidates(np.asarray(cand), np.asarray(refined))
                idx, sq = idx[:, :k_search], sq[:, :k_search]
                ok = self.certified(sub, np.asarray(gmax_wide), sq[:, -1])
                out_idx[rows[ok]] = idx[ok]
                out_sq[rows[ok]] = sq[ok]
                still.append(rows[~ok])
            pending = np.sort(np.concatenate(still))

        if pending.size:
            self.counts["exact_rows"] += int(pending.size)
            idx, sq = self.exact(qb.take(mx, pending), integer=False)
            out_idx[pending] = idx
            out_sq[pending] = sq
        return out_idx, out_sq

    def cannot_widen(self, qb: _QueryBlock, gmax: np.ndarray, k_wide: int) -> np.ndarray:
        """Rows of ``qb`` a wider pass carrying ``k_wide`` would, by an estimate, fail to certify.

        Which search answers a row decides only the time: the exact search computes
        each pair's distance as refinement does and ranks every training row by the
        same ``(distance, index)``, so it returns what a certified wider pass would.
        A row goes there directly when ``k_wide`` training rows lie inside its Gram
        noise band ``C·‖q‖²``: the wider pass's bound is then at most its largest
        kept Gram value less ``C·‖q‖²`` (the ``f`` branch of
        :func:`_excluded_lower_bound`), which is at most zero, and it has to exceed
        the k-th distance. The band is counted, on a sample (:meth:`band_count`),
        only for rows whose first-pass ``gmax`` already lies inside it — as it must
        when ``k_wide`` rows do, the first pass carrying fewer — and a row whose
        estimate reaches ``_SKIP_MARGIN * k_wide`` skips the wider pass. Not a proof —
        a sample, and a bound that takes its ``g`` branch when every training norm
        lies beyond the split — hence the margin; a wrong call costs time, never the
        answer.
        """
        skip = np.zeros(qb.n, dtype=bool)
        need = _SKIP_MARGIN * k_wide
        if need > self.index.n_samples:
            return skip
        q_sq = np.asarray(qb.sq, dtype=np.float64).reshape(-1)
        band = _gram_err(self.index.n_features) * q_sq
        suspects = np.flatnonzero(gmax.astype(np.float64) <= band)
        if suspects.size:
            sub = qb if suspects.size == qb.n else qb.take(self.mx, suspects)
            skip[suspects] = self.band_count(sub, band[suspects], need) >= need
        return skip

    def band_count(self, qb: _QueryBlock, band: np.ndarray, need: int) -> np.ndarray:
        """Per row of ``qb``, about how many training rows have a Gram value ``<= band``.

        Counted on one training row in each window of ``stride`` consecutive rows,
        with ``stride`` set so that about ``_SAMPLE_HITS`` sampled rows sit at the
        decision point ``need``, and scaled back up; the same tile expression as the
        search. Each window's row sits at an offset hashed from the window's index
        (:func:`_band_offset`), so every training row is counted with the same
        chance, ``1 / stride``, however the rows are ordered. Every ``stride``-th row
        was not: rows that repeat with a period that is a multiple of the stride — a
        set tiled or stacked on itself — put every copy of one row in the sample and
        no copy of the next, so a query near such a row counted its copies ``stride``
        times over, and skipped a second pass that would have certified it. Were the
        offsets independent of the data, a true count of ``k_wide`` would reach the
        call at twice that with at most a Poisson tail's chance, of order 1e-4; they
        are fixed, so that is an estimate, not a bound.
        """
        mx, state = self.mx, self.state
        stride = max(1, need // _SAMPLE_HITS)
        n_sample = _band_windows(self.index.n_samples, stride)
        # Built on the device for this call alone, and gathered from tile by tile.
        sample = self.fns.band_rows(
            mx.arange(n_sample, dtype=mx.uint32), mx.array(stride, dtype=mx.uint32)
        )
        plan = resolve_blocks(qb.n, n_sample, backend=self.plan_backend, k_search=1)
        limit = mx.array(band.astype(NUMPY_FLOAT)[:, None])
        counts = []
        for r_start in range(0, qb.n, plan.query_rows):
            r_end = min(r_start + plan.query_rows, qb.n)
            total = None
            for t_start in range(0, n_sample, plan.train_cols):
                rows = sample[t_start : t_start + plan.train_cols]
                gram = qb.sel[r_start:r_end] @ mx.take(state.sel, rows, axis=0).T
                t_sq = mx.take(state.sel_sq, rows)[None]
                tile = self.fns.tile(qb.sq[r_start:r_end], t_sq, gram)
                hits = mx.sum((tile <= limit[r_start:r_end]).astype(mx.int32), axis=1)
                total = hits if total is None else total + hits
                mx.eval(total)
            counts.append(np.array(total))
        return np.concatenate(counts).astype(np.int64) * stride

    def exact(self, qb: _QueryBlock, *, integer: bool) -> tuple[np.ndarray, np.ndarray]:
        """Exact first ``k_search`` against every training row, by composite key.

        Each value is packed with its index into one ``uint64`` —
        ``bits(distance) << 32 | index`` — which orders exactly as ``(distance,
        index)`` because non-negative float32 bit patterns order as their values.
        Partitioning on the key therefore breaks every tie on the index, at every
        tile and every merge, with nothing left to certify.

        With ``integer=True`` the values come from the Gram identity on the raw
        rows, both sides scaled by ``2**qb.int_shift``, which is exact for small
        integers; otherwise from direct differences, exactly as :meth:`refine`
        computes them, so an exact answer and a certified one never disagree about
        a distance. Returns host indices and float32 squared distances — scaled like
        :meth:`refine`'s, or by ``4**int_shift`` when ``integer``.

        The integer path's distances are integers below ``2**24``. When the largest
        this block can reach and the largest index fit in 32 bits together
        (:func:`_int_key_bits`), its key is ``distance << bits | index`` in ``uint32``
        instead, which orders the same and moves half the bytes through selection:
        binary 1e6 x 16 with 256 queries ran 1.6x faster, and no faster without it.
        """
        mx, state = self.mx, self.state
        fns = self.fns
        n_train = self.index.n_samples
        d = self.index.n_features
        k_search = self.k_search
        budget = _distance_budget_bytes(self.plan_backend)
        index_bits = None

        if integer:
            plan = resolve_blocks(
                qb.n, n_train, backend=self.plan_backend, k_search=k_search,
                train_block_size=self.train_block_size,
            )
            rows, cols = plan.query_rows, plan.train_cols
            pair_bytes = 4 * _TILE_WORKSET_FACTOR  # the tile as resolve_blocks budgets it
            lift = float(2**qb.int_shift)
            q_int = qb.hi * lift if qb.int_shift else qb.hi
            q_sq = mx.sum(q_int * q_int, axis=1, keepdims=True)
            index_bits = _int_key_bits(qb.int_max_sq, n_train)
            if index_bits is not None:
                shift = mx.array(index_bits, dtype=mx.uint32)
        else:
            # A (rows, cols, d) difference per tile: cols as wide as the budget
            # allows for this many rows, rows as many as it then allows.
            diff_bytes = 3 * d * 4
            pair_bytes = diff_bytes + _KEY_WORKSET_BYTES
            cols = self.train_block_size or _exact_cols(budget, qb.n, pair_bytes)
            cols = max(min(k_search, n_train), min(int(cols), n_train))
            rows = max(1, min(qb.n, budget // max(diff_bytes * cols, 1)))

        keys = []
        for r_start in range(0, qb.n, rows):
            r_end = min(r_start + rows, qb.n)
            best = None
            in_flight = 0
            for t_start in range(0, n_train, cols):
                t_end = min(t_start + cols, n_train)
                if integer:
                    gram = q_int[r_start:r_end] @ state.raw_t[:, t_start:t_end]
                    t_sq = state.raw_sq[None, t_start:t_end]
                    if qb.int_shift:
                        # Both sides lifted by 2**shift: products and sums stay
                        # integers below 2**24, so this is still exact.
                        gram = gram * lift
                        t_sq = t_sq * (lift * lift)
                    if index_bits is None:
                        vals = fns.tile(q_sq[r_start:r_end], t_sq, gram)
                        key = _composite_keys(mx, vals, t_start)
                    else:
                        index = mx.arange(t_end - t_start, dtype=mx.uint32) + mx.array(
                            t_start, dtype=mx.uint32
                        )
                        key = fns.tile_key(q_sq[r_start:r_end], t_sq, gram, index[None], shift)
                else:
                    t = state.raw[None, t_start:t_end]
                    q = qb.hi[r_start:r_end, None, :]
                    if state.raw_lo is None and qb.lo is None:
                        sq = fns.pair_sq(t, q, state.scale)
                    else:
                        t_lo = (
                            state.zero if state.raw_lo is None
                            else state.raw_lo[None, t_start:t_end]
                        )
                        q_lo = state.zero if qb.lo is None else qb.lo[r_start:r_end, None, :]
                        sq = fns.pair_sq_lo(t, t_lo, q, q_lo, state.scale)
                    key = _composite_keys(mx, mx.sum(sq, axis=2), t_start)

                take = min(k_search, t_end - t_start)
                _, part = _select_smallest(mx, key, take, segmented=self.segmented)
                if best is None:
                    best = part
                else:
                    merged = mx.concatenate([best, part], axis=1)
                    keep = min(k_search, merged.shape[1])
                    sel = mx.argpartition(merged, kth=keep - 1, axis=1)[:, :keep]
                    best = mx.take_along_axis(merged, sel, axis=1)
                # Evaluated once one more tile would take the working set in flight
                # past the budget: a budget-sized tile on its own, narrow ones several
                # at a time. Left lazy, every tile would be alive at once; see
                # ``refine``.
                tile_bytes = (r_end - r_start) * (t_end - t_start) * pair_bytes
                in_flight += tile_bytes
                if in_flight + tile_bytes > budget:
                    mx.eval(best)
                    in_flight = 0
            keys.append(np.array(best))

        return _unpack_keys(np.sort(np.concatenate(keys, axis=0), axis=1), index_bits)


def _exact_cols(budget: int, rows: int, pair_bytes: int) -> int:
    """Training rows per exact-search tile for ``rows`` query rows.

    As many as ``budget`` holds at ``pair_bytes`` a pair, rounded down to a
    power-of-two multiple of ``_EXACT_MIN_COLS``, and never fewer than that.
    """
    wide = max(1, budget // max(pair_bytes * rows * _EXACT_MIN_COLS, 1))
    return _EXACT_MIN_COLS << (wide.bit_length() - 1)


def _band_offset(window, stride):
    """Where in its window of ``stride`` rows each window in ``window`` is sampled.

    Wellons's ``lowbias32`` hash of the window's index, seeded with
    ``_SAMPLE_SEED``, reduced modulo ``stride``. Plain ``uint32`` arithmetic, which
    wraps alike in NumPy and in MLX on either device, so the host can say where any
    window's row falls without asking the device.
    """
    h = window ^ _SAMPLE_SEED
    h = (h ^ (h >> 16)) * 0x7FEB352D
    h = (h ^ (h >> 15)) * 0x846CA68B
    return (h ^ (h >> 16)) % stride


def _band_windows(n_train: int, stride: int) -> int:
    """How many windows of ``stride`` rows :meth:`_Search.band_count` takes a row from.

    Every window that starts inside the index, less the last when it is short and
    its row falls past the end. So every row is counted with the same chance.
    """
    windows = -(-n_train // stride)
    last = windows - 1
    offset = int(_band_offset(np.array([last], dtype=np.uint32), np.uint32(stride))[0])
    return windows - int(last * stride + offset >= n_train)


def _integer_sq_bound(state: _DeviceState, q_min, q_max, shift: int) -> float | None:
    """The largest squared distance from a row inside ``[q_min, q_max]`` to a training row.

    Column by column, the farthest apart the two can be given the training columns'
    extremes, lifted by ``2**shift`` like both sides of the integer path; every
    coordinate there is a small integer, so the sum is exact. ``None`` for a state
    without the extremes, whose exact search then keys in 64 bits.
    """
    if state.col_min is None or state.col_max is None:
        return None
    q_min = np.asarray(q_min, dtype=np.float64)
    q_max = np.asarray(q_max, dtype=np.float64)
    span = np.maximum(q_max - state.col_min, state.col_max - q_min)
    return math.ldexp(float(np.dot(span, span)), 2 * shift)


def _int_key_bits(max_sq: float | None, n_train: int) -> int | None:
    """Index bits for a ``uint32`` ``distance << bits | index`` key, or ``None``.

    ``None`` when the key needs 64 bits — the largest index, ``n_train - 1``, and the
    largest distance, ``max_sq``, take more than 32 between them — or when
    ``max_sq`` is unknown.
    """
    if max_sq is None:
        return None
    index_bits = max(n_train - 1, 0).bit_length()
    if index_bits >= 32 or index_bits + int(max_sq).bit_length() > 32:
        return None
    return index_bits


def _unpack_keys(packed: np.ndarray, index_bits: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Indices and float32 values from keys: ``uint64`` ones when ``index_bits`` is None."""
    if index_bits is None:
        idx = (packed & np.uint64(0xFFFFFFFF)).astype(INDEX_DTYPE)
        return idx, (packed >> np.uint64(32)).astype(np.uint32).view(NUMPY_FLOAT)
    idx = (packed & np.uint32((1 << index_bits) - 1)).astype(INDEX_DTYPE)
    return idx, (packed >> np.uint32(index_bits)).astype(NUMPY_FLOAT)


def _composite_keys(mx, vals, t_start: int):
    """``bits(vals) << 32 | index`` for a tile whose first column is row ``t_start``.

    32 bits of index, which is why :func:`build_index` refuses ``2**32`` rows. The
    indices are built from 0 and offset, because MLX 0.31's ``arange`` rejects a
    start or stop of ``2**31`` or more.
    """
    index = mx.arange(int(vals.shape[1]), dtype=mx.uint64) + t_start
    return mx.left_shift(mx.view(vals, mx.uint32).astype(mx.uint64), 32) | index[None, :]


def _select_smallest(mx, vals, take: int, *, segmented: bool):
    """Indices and values of the ``take`` smallest entries of each row.

    On the GPU a wide row is partitioned in two stages — each segment first, then
    the survivors — because MLX's GPU ``argpartition`` costs a full sort of the row.
    Exact: every row's global ``take`` smallest are among its segments' ``take``
    smallest. The ragged tail is appended to the survivors whole rather than padded,
    so no padding value can ever tie with, and displace, a real one.
    """
    n_rows, width = vals.shape
    seg = max(_SELECT_SEGMENT, 1 << (4 * take - 1).bit_length())
    if not segmented or seg > _SELECT_SEGMENT_MAX or width < 4 * seg:
        local = mx.argpartition(vals, kth=take - 1, axis=1)[:, :take]
        return local, mx.take_along_axis(vals, local, axis=1)

    n_seg = width // seg
    head = vals[:, : n_seg * seg].reshape(n_rows, n_seg, seg)
    part = mx.argpartition(head, kth=take - 1, axis=2)[:, :, :take]
    cand_val = mx.take_along_axis(head, part, axis=2).reshape(n_rows, n_seg * take)
    offsets = (mx.arange(n_seg, dtype=mx.uint32) * seg)[None, :, None]
    cand_idx = (part + offsets).reshape(n_rows, n_seg * take)

    tail = width - n_seg * seg
    if tail:
        tail_idx = mx.arange(n_seg * seg, width, dtype=mx.uint32)[None, :]
        cand_val = mx.concatenate([cand_val, vals[:, n_seg * seg :]], axis=1)
        cand_idx = mx.concatenate(
            [cand_idx, mx.broadcast_to(tail_idx, (n_rows, tail))], axis=1
        )

    sel = mx.argpartition(cand_val, kth=take - 1, axis=1)[:, :take]
    return mx.take_along_axis(cand_idx, sel, axis=1), mx.take_along_axis(cand_val, sel, axis=1)


def _sort_candidates(idx: np.ndarray, sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order each row's candidates by ``(distance, index)``, on the host.

    A lexicographic sort over a ``(block, k_carry)`` array — small enough that host
    ``np.lexsort`` is both simpler and faster than any device trick, and it is the
    one place a total order is imposed on the first pass.
    """
    idx = idx.astype(INDEX_DTYPE, copy=False)
    order = np.lexsort((idx, sq), axis=-1)
    rows = np.arange(idx.shape[0])[:, None]
    return idx[rows, order], sq[rows, order]


def _excluded_lower_bound(
    q_sq: np.ndarray,
    gmax: np.ndarray,
    norms: np.ndarray,
    *,
    gram_err: float,
    norm_err: float,
) -> np.ndarray:
    """Per row, a lower bound on the true distance of every excluded training row.

    For an excluded row ``t`` with norm ``τ = ‖t‖`` two bounds hold:
    ``f(τ) = gmax − C·(‖q‖² + τ²)`` from the Gram error, which falls with ``τ``, and
    ``g(τ) = (τ(1 − η) − ‖q‖(1 + η))² − 16u·(‖q‖² + τ²)`` from the reverse triangle
    inequality, which rises with ``τ`` beyond ``t₁ ≈ ‖q‖``. Splitting the sorted
    training norms at any ``t* ≥ t₁``, rows at or below it are bounded by ``f`` at
    the largest such norm and rows above it by ``g`` at the smallest, so the
    minimum of those two values bounds them all. ``t*`` is placed where ``f`` and
    ``g`` cross, which is the best split; correctness needs only ``t* ≥ t₁``.

    Using the one largest training norm for every row instead would let a single
    outlier make every row uncertifiable.
    """
    C, eta = gram_err, norm_err
    denom = (1.0 - eta) ** 2 - 16.0 * _U
    if not math.isfinite(C) or denom <= 0.0:
        return np.full(q_sq.shape, -np.inf)

    qn = np.sqrt(q_sq)
    a = gmax - C * q_sq
    disc = (1.0 + C) * a - C * q_sq
    cross = (qn + np.sqrt(np.maximum(disc, 0.0))) / (1.0 + C)
    t1 = qn * (1.0 - eta * eta) / denom
    t_star = np.maximum(cross, t1)

    n = norms.shape[0]
    pos = np.searchsorted(norms, t_star, side="right")
    left = norms[np.clip(pos - 1, 0, n - 1)]
    right = norms[np.clip(pos, 0, n - 1)]
    f_left = np.where(pos > 0, a - C * left * left, np.inf)
    g_right = np.where(
        pos < n,
        (right * (1.0 - eta) - qn * (1.0 + eta)) ** 2 - 16.0 * _U * (q_sq + right * right),
        np.inf,
    )
    return np.minimum(f_left, g_right)


def _drop_self(idx: np.ndarray, dist: np.ndarray, *, offset: int):
    """Remove each query's own training index from its neighbor list.

    ``offset`` converts the block-local row number into the global row index. That
    single term is the difference between a correct self-query and the ancestor's,
    which returned a row as its own nearest neighbor whenever the training set
    exceeded one query block.

    A row with no self match — more than ``k`` exact duplicates precede it by index
    — drops its *last* candidate. Under ``(distance, index)`` the row itself sorts
    after every candidate kept, so that leaves the first ``k`` excluding it, and a
    prefix of the answer for any larger ``k``. scikit-learn drops column 0 there,
    but its column 0 is an arbitrary one of the tied duplicates; mlxlearn's is the
    lowest-indexed, the one its own order keeps.
    """
    n_rows, width = idx.shape
    global_rows = (np.arange(n_rows, dtype=idx.dtype) + offset)[:, None]
    mask = idx != global_rows

    missing = np.all(mask, axis=1)
    if np.any(missing):
        mask[missing, -1] = False

    keep = width - 1
    out_idx = idx[mask].reshape(n_rows, keep)
    out_dist = dist[mask].reshape(n_rows, keep)
    return out_idx, out_dist


def _row_sq_norms(mx, a, block: int = 1 << 16):
    """``Σ a²`` per row, in row blocks so no full-size squared temporary exists.

    Each block is evaluated before the next is built; left lazy, MLX encodes them
    all ahead of the GPU and every block's square is alive at once.
    """
    n = int(a.shape[0])
    if n <= block:
        return mx.sum(a * a, axis=1)
    parts = []
    for i in range(0, n, block):
        part = a[i : i + block]
        parts.append(mx.sum(part * part, axis=1))
        mx.eval(parts[-1])
    return mx.concatenate(parts)


def _split_query(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Query rows as ``(float32 value, float64 rows, float32 residual)``.

    The residual is ``None`` for float32 rows, and otherwise every row's, zero or not:
    whether a block keeps it is the block's choice (``_Search.host_block``).
    """
    hi_np = np.ascontiguousarray(rows, dtype=NUMPY_FLOAT)
    rows64 = np.asarray(rows, dtype=np.float64)
    lo_np = None
    if rows.dtype != NUMPY_FLOAT:
        lo_np = (rows64 - hi_np).astype(NUMPY_FLOAT)
    return hi_np, rows64, lo_np


def _dyadic_shift(rows: np.ndarray, int_max: float, int_limit: int) -> int | None:
    """The smallest ``s`` making ``rows * 2**s`` integers the exact path can take.

    ``None`` when there is none: a non-dyadic value, or magnitudes that would push
    a product past ``2**24`` once both sides are lifted.
    """
    largest = max(float(np.max(np.abs(rows))) if rows.size else 0.0, int_max)
    for shift in range(_DYADIC_SHIFTS + 1):
        if math.ldexp(largest, shift) > int_limit:
            return None
        lifted = np.ldexp(rows, shift)
        if np.array_equal(lifted, np.rint(lifted)):
            return shift
    return None


def _all_integral(mx, a, block: int = 1 << 16) -> bool:
    """Whether every entry of ``a`` is an integer, a row block at a time.

    Blocked so the rounded copy never exists at full size, and it stops at the first
    block that fails, which for real-valued data is the first.
    """
    for i in range(0, int(a.shape[0]), block):
        part = a[i : i + block]
        if not bool(mx.all(part == mx.round(part)).item()):
            return False
    return True


def _build_state(X: np.ndarray, X_lo: np.ndarray | None, center: np.ndarray, device: str):
    """Upload and derive everything ``device`` needs. Returns ``(state, X, X_lo)``.

    The returned host arrays are read-only zero-copy views of the device arrays;
    the caller swaps them in for the ones it had, which frees those.
    """
    import mlx.core as mx

    fns = _kernels(device)
    n, d = (int(s) for s in X.shape)
    with use_device(device):
        # mx.array copies. This is the index's one private copy of the training
        # rows; nothing below allocates another.
        raw = mx.array(np.ascontiguousarray(X, dtype=NUMPY_FLOAT))
        raw_lo = None if X_lo is None else mx.array(np.ascontiguousarray(X_lo, dtype=NUMPY_FLOAT))
        # The column extremes set the scale, which everything below needs on the
        # host. A small matrix is reduced there in less than a device round trip.
        if n * d <= _HOST_REDUCE_ELEMENTS:
            host = np.asarray(X, dtype=NUMPY_FLOAT)
            col_max = host.max(axis=0)
            col_min = host.min(axis=0)
        else:
            col_max = mx.max(raw, axis=0)
            col_min = mx.min(raw, axis=0)
            mx.eval(raw, col_max, col_min, *([] if raw_lo is None else [raw_lo]))

        c64 = np.asarray(center, dtype=np.float64)
        hi_col = np.asarray(col_max, dtype=np.float64)
        lo_col = np.asarray(col_min, dtype=np.float64)
        spread = float(max(np.max(np.abs(hi_col - c64)), np.max(np.abs(lo_col - c64))))
        if not math.isfinite(spread):
            raise ValueError("The training data contains NaN or infinity.")
        if spread > _SPAN_LIMIT:
            # The remedy is advice: a patched estimator falls back on this at fit,
            # and reports the limit without telling its caller to use scikit-learn.
            raise UnsupportedInputError(
                f"The training data spans {spread:.3g} from its mean, too wide for "
                "float32 to subtract two of its coordinates.",
                reason="float32-overflow",
                advice="Rescale it, or compute these neighbors with scikit-learn.",
            )
        # Scale by an exact power of two, and only where float32 needs it: data
        # around 1e19, whose squares overflow float32, and data around 1e-20, whose
        # squares underflow it (and which the GPU flushes to zero), are both brought
        # back to where float32 squares them faithfully. Scaling by 2**k is exact,
        # and ordinary data is not scaled at all.
        exponent = _span_exponent(spread, d)
        scale = mx.array(math.ldexp(1.0, -exponent), dtype=mx.float32)
        c_hi = c64.astype(NUMPY_FLOAT)
        c_lo = (c64 - c_hi).astype(NUMPY_FLOAT)
        center_dev = mx.array(c_hi)
        center_lo = mx.array(c_lo) if c_lo.any() else None
        zero = mx.array(0.0, dtype=mx.float32)

        sel = _translate(fns, raw, raw_lo, center_dev, center_lo, scale, zero)
        sel_sq = _row_sq_norms(mx, sel)
        # Everything that goes into the state is evaluated here, the transposes
        # included: an unevaluated node belongs to this thread's stream, and a query
        # from another thread that forces it fails with "There is no Stream(gpu, N)
        # in current thread".
        sel_t = sel.T
        pending = [raw, sel, sel_t, sel_sq, *([] if raw_lo is None else [raw_lo])]
        # The certificate's sorted norms. MLX's CPU sort took a third of a CPU
        # build, four times NumPy's; the float32 square roots are the same either way.
        norms = mx.sort(mx.sqrt(sel_sq)) if device == "gpu" else None
        if norms is not None:
            pending.append(norms)

        # Small integers — binary, one-hot, counts — make the raw Gram identity
        # exact in float32: every product and partial sum is an integer below 2**24.
        # Such data is also where ties are heaviest, and an exact search needs no
        # certificate, so it gets its own path.
        max_abs = float(max(np.max(np.abs(hi_col)), np.max(np.abs(lo_col)))) if n else 0.0
        int_limit = int(math.isqrt((1 << 24) // (4 * d))) if d else 0
        raw_t = raw_sq = None
        if raw_lo is None and 0 < int_limit and max_abs <= int_limit:
            # Real-valued data within the bound — standardized features, say —
            # nearly always shows it in its first rows, which the host can check
            # without a pass over the device copy or a synchronization.
            head = np.asarray(X[:256], dtype=NUMPY_FLOAT)
            if np.array_equal(head, np.rint(head)) and _all_integral(mx, raw):
                raw_t = raw.T
                raw_sq = _row_sq_norms(mx, raw)
                pending += [raw_t, raw_sq]
            else:
                int_limit = 0
        else:
            int_limit = 0
        mx.eval(*pending)

    state = _DeviceState(
        raw=raw,
        raw_lo=raw_lo,
        sel=sel,
        sel_t=sel_t,
        sel_sq=sel_sq,
        center=center_dev,
        center_lo=center_lo,
        scale=scale,
        zero=zero,
        exponent=exponent,
        spread=spread,
        norms_sorted=(
            np.sort(np.sqrt(np.asarray(sel_sq))) if norms is None else np.asarray(norms)
        ).astype(np.float64),
        raw_norm_bound=float(np.sqrt(np.sum(np.maximum(np.abs(hi_col), np.abs(lo_col)) ** 2))),
        int_limit=int_limit,
        int_max=max_abs,
        raw_t=raw_t,
        raw_sq=raw_sq,
        # What the integer path's exact search bounds its distances by, to key them
        # in 32 bits when they fit (_integer_sq_bound).
        col_min=lo_col if int_limit > 0 else None,
        col_max=hi_col if int_limit > 0 else None,
        index_dtype=mx.int32 if n < 2**31 else mx.int64,
    )
    x_view = _readonly_view(raw)
    lo_view = None if raw_lo is None else _readonly_view(raw_lo)
    return state, x_view, lo_view


def _readonly_view(array) -> np.ndarray:
    """A zero-copy NumPy view of an evaluated MLX array, write-protected.

    MLX arrays are immutable, and the device state assumes it; a writeable view
    would let a caller change the index underneath its own derived arrays.
    """
    view = np.asarray(array)
    view.flags.writeable = False
    return view


def _split_float64(X: np.ndarray, block: int = 1 << 16) -> tuple[np.ndarray, np.ndarray | None]:
    """``X`` as a float32 value plus a float32 residual, ``hi + lo ≈ X`` to ~u².

    Row-blocked, so the float64 difference never exists at full size. The residual
    is ``None`` when ``X`` is exactly representable in float32 — integers, binary
    features, data that was float32 all along.
    """
    hi = np.asarray(X, dtype=NUMPY_FLOAT, order="C")
    lo = np.empty_like(hi)
    for start in range(0, X.shape[0], block):
        stop = start + block
        lo[start:stop] = X[start:stop] - hi[start:stop]
    return hi, (lo if lo.any() else None)


def build_index(X: np.ndarray, *, backend: str, device: str | None = None) -> NeighborIndex:
    """Build a :class:`NeighborIndex` from a training matrix, on ``device`` if given.

    float32 input is uploaded as given. The upload is the copy — the index then
    keeps a read-only view of its own device buffer, never the caller's array —
    because an index that aliases the caller's array silently changes its own
    answers when that array is mutated. Wider input (float64, integers) is split
    into a float32 value and a float32 residual, and translated by its float64
    mean, carried as a float32 pair the same way, so the float32 cast does not
    quantize away what the caller supplied: Unix timestamps collapse to 128-second
    steps in float32, but not here.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional; got shape {X.shape}.")
    if X.shape[0] > _MAX_TRAIN_ROWS:
        raise UnsupportedInputError(
            f"The training data has {X.shape[0]} rows; the neighbor index addresses at "
            f"most {_MAX_TRAIN_ROWS}.",
            advice="Compute these neighbors with scikit-learn.",
        )
    if X.dtype == NUMPY_FLOAT:
        hi = np.ascontiguousarray(X)
        lo = None
        center = hi.mean(axis=0, dtype=np.float64).astype(NUMPY_FLOAT)
    else:
        X64 = np.asarray(X, dtype=np.float64)
        hi, lo = _split_float64(X64)
        center = X64.mean(axis=0)
    index = NeighborIndex(X=hi, center=center, backend=backend, X_lo=lo, device=device)
    _require_resolvable(index, device)
    return index


def _require_resolvable(index: NeighborIndex, device: str | None) -> None:
    """Refuse, at fit, training rows whose own neighbors float32 cannot resolve.

    Only reachable when the span had to be scaled down: a few values far larger than
    the rest — an unmasked fill value, 1e20 or the NetCDF default 9.97e36, among
    unit-scale rows — leave the rest's gaps too small to square at that scale. A
    query would then raise at inference, where nothing can fall back, so a sample
    of the training rows is searched here instead, and a fit whose rows sit within
    ``_RESOLVED_MARGIN`` of the limit is a capability mismatch: a directly imported
    estimator raises, and a patched one falls back to scikit-learn.
    """
    state = index._ensure_device(device)
    n = index.n_samples
    if state.exponent <= 0 or n < 2:
        return
    rows = np.unique(np.linspace(0, n - 1, min(n, _PROBE_ROWS)).astype(np.intp))
    probe = index.X[rows]
    if index.X_lo is not None:
        probe = probe.astype(np.float64) + index.X_lo[rows]
    try:
        # Each row finds itself, or a duplicate, first; the second is its nearest.
        dist, idx = index.query(probe, 2, device=device)
    except UnsupportedInputError:
        unresolved = True  # a neighbor already below the limit itself
    else:
        near = idx[:, 1]
        close = np.ldexp(np.square(dist[:, 1]), -2 * state.exponent) < (
            _RESOLVED_SQ * _RESOLVED_MARGIN
        )
        differ = np.any(index.X[near] != index.X[rows], axis=1)
        if index.X_lo is not None:
            differ |= np.any(index.X_lo[near] != index.X_lo[rows], axis=1)
        unresolved = bool(np.any(close & differ))
    if unresolved:
        raise UnsupportedInputError(
            f"The training data spans {state.spread:.3g} from its mean, too wide for float32 to "
            "also resolve the distances between its nearest rows: their squares underflow "
            "at that scale.",
            reason="float32-overflow",
            advice="A fill or sentinel value left in the data does this; mask it, or "
            "rescale the data, or compute these neighbors with scikit-learn.",
        )
