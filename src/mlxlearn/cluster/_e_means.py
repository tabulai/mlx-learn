"""Apple Silicon implementation of e-means (arXiv:2611.09501v1).

The numerical work runs on the GPU: tiled nearest-centroid search, grouped
statistics, EMA updates and donor splitting. NumPy generates reproducible
permutations and random donor keys; no centroid statistics are copied to the
CPU during optimization. This implementation follows the published equations
and exposes the current Faiss recipe separately, because its constants differ.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Integral, Real

import mlx.core as mx
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from mlxlearn_device._binary32 import matmul

from .._kernels.e_means import (
    assign,
    batch_stats,
    ema_update,
    gather_bf16,
    removal_stats,
    removal_update,
    twin_route,
)
from ._recipe import resolve_recipe, scheduled_alpha


@dataclass(frozen=True)
class PreparedEMeansFit:
    """Validated resident inputs; preparation is separate from optimization."""
    data: mx.array
    initial: mx.array
    offset: mx.array
    recipe: dict
    seed: int
    signature: tuple
    # Per-feature (lower, upper) range of the rows and initial centroids;
    # cost-based revival keeps split centroids inside it.
    bounds: tuple | None = None


@dataclass(frozen=True)
class DeviceEMeansFit:
    """Device outputs; copying these arrays to NumPy is an explicit operation."""
    centers: mx.array
    ema_centers: mx.array
    last_epoch_centers: mx.array
    slow_ema_centers: mx.array | None
    assignments: mx.array | None
    history: tuple
    split_count: mx.array
    steps: int
    epochs: int
    seconds: float


def _normalize(x):
    """Normalize safely without overflowing the norm; preserve zero rows."""
    maximum = mx.max(mx.abs(x), axis=1, keepdims=True)
    scaled = x / mx.where(maximum > 0, maximum, 1)
    norm = mx.sqrt(mx.sum(scaled * scaled, axis=1, keepdims=True))
    return scaled / mx.where(norm > 0, norm, 1)


def split_plan(counts, uniform, threshold):
    """Pair dead clusters with mass-weighted donors, entirely on the GPU.

    Returns ``is_child, is_donor, source, target, number``. ``source[t]`` is
    the donor of child slot ``t`` and ``target[s]`` the child slot of donor
    ``s``; each is meaningful only under its mask. The lowest-index dead
    slots become children. Unused outputs stay lazy and are never computed.
    """
    k = counts.shape[0]
    ids = mx.arange(k, dtype=mx.int32)
    dead = counts < (threshold * mx.sum(counts) / k)
    available = (~dead) & (counts > 0)
    number = mx.minimum(mx.sum(dead), mx.sum(available)).astype(mx.int32)
    uniform = mx.clip(uniform, 1e-30, 1 - np.finfo(np.float32).eps)
    keys = mx.where(available, mx.log(mx.maximum(counts, 1e-30)) - mx.log(-mx.log(uniform)), -mx.inf)
    donor_order = mx.argsort(-keys)
    target_order = mx.argsort(mx.where(dead, ids, ids + k))
    donor_rank = mx.argsort(donor_order)
    target_rank = mx.argsort(target_order)
    is_child = target_rank < number
    is_donor = donor_rank < number
    source = donor_order[target_rank]
    target = target_order[donor_rank]
    return is_child, is_donor, source, target, number


def revive(centers, counts, uniform, threshold, split_epsilon, companion=None, *,
           precision="float32", plan=None):
    """Split underused clusters with weighted sampling without replacement.

    ``uniform`` contains independent U(0,1) samples, one per centroid. Gumbel
    top-k implements the reference's mass-proportional donor sampling. The
    whole selection stays on the GPU, including the dynamic number of dead
    clusters. ``companion`` optionally supplies slow-EMA (centers, counts),
    which undergo the same donor/child mapping using their own statistics.
    BF16 repairs a collapsed split by moving the child's first coordinate to
    the next BF16 value toward positive infinity, or inward at the finite
    maximum. Other precisions retain the original multiplicative convention.
    ``plan`` reuses a :func:`split_plan` result computed from the same inputs.
    Returns ``centers, counts, number_of_splits, companion``.
    """
    d = centers.shape[1]
    is_child, is_donor, source, _, number = (
        split_plan(counts, uniform, threshold) if plan is None else plan)
    offset = mx.where(mx.arange(d) % 2 == 0, split_epsilon, -split_epsilon)

    def split(position, mass):
        child = position[source] * (1 + offset)
        donor = position * (1 - offset)
        if precision == "bfloat16":
            donor_bits = donor[source].astype(mx.bfloat16).view(mx.uint16)
            child_bits = child.astype(mx.bfloat16).view(mx.uint16)
            # Compare represented codes, treating signed zeros as equal. Bit
            # operations also preserve subnormals on GPUs that flush FP32
            # subnormal arithmetic, including the first positive BF16 value.
            both_zero = ((donor_bits & 0x7fff) == 0) & ((child_bits & 0x7fff) == 0)
            collapsed = mx.all((donor_bits == child_bits) | both_zero, axis=1)
            bits = child_bits[:, 0].astype(mx.uint32)
            negative, zero = (bits & 0x8000) != 0, (bits & 0x7fff) == 0
            upward = mx.where(zero, 1, mx.where(negative, bits - 1, bits + 1))
            inward = mx.where(zero, 0x8001, mx.where(negative, bits + 1, bits - 1))
            neighbor_bits = mx.where((upward & 0x7fff) < 0x7f80, upward, inward)
            neighbor = (neighbor_bits.astype(mx.uint32) << 16).view(mx.float32)
            child = mx.concatenate((mx.where(collapsed, neighbor, child[:, 0])[:, None], child[:, 1:]), axis=1)
        new_position = mx.where(is_child[:, None], child, mx.where(is_donor[:, None], donor, position))
        new_mass = mx.where(is_child, mass[source] * 0.5, mx.where(is_donor, mass * 0.5, mass))
        return new_position, new_mass

    centers, counts = split(centers, counts)
    if companion is not None:
        companion = split(*companion)
    return centers, counts, number, companion


# Cost-based revival moves a centroid whose removal cost is zero or below this
# fraction of the median cost (facebookresearch/faiss, contrib/e_means/base.py,
# ``_REMOVAL_THRESHOLD``, accessed 2026-11-24).
REMOVAL_THRESHOLD = 0.3
# MLX's Metal argsort is verified below this many float32 keys.
_MAX_PLAN_CLUSTERS = 2**21


def epoch_seed(seed, epoch, *, distinct=False):
    """Seed of one pass's row permutation.

    ``distinct`` keeps a pass from replaying the initialization draw, which
    uses the stream seeded with ``seed``: a pass seed equal to ``seed`` moves
    by the golden-ratio constant, as in the reference since cost-based
    revival. That happens at pass 0 for seeds 0 and 2**31, and at one later
    even pass for two other seeds each; every other pass seed is unchanged.
    """
    value = (seed * 1000003 + epoch) % 2**32
    if distinct and value == seed % 2**32:
        value = (value + 0x9E3779B9) % 2**32
    return value


def removal_plan(removal, mass, lengths, threshold=REMOVAL_THRESHOLD):
    """Pair centroids that are cheap to remove with the donors to split, on the GPU.

    ``removal`` holds each centroid's running removal cost, the sum over its
    rows of the distance to their second-nearest centroid minus the distance
    to their nearest one. A centroid whose cost is zero or below ``threshold``
    times the median cost moves, cheapest first; centroids without cost go in
    index order. Donors are the other centroids with mass, by decreasing
    ``lengths``, the lengths of their long-axis sums; a length that is not a
    number makes no donor. Nothing is drawn at random. Returns ``is_child,
    is_donor, source, target, number`` with the meaning of :func:`split_plan`.
    """
    k = removal.shape[0]
    ids = mx.arange(k, dtype=mx.int32)
    ordered = mx.sort(removal)
    median = 0.5 * (ordered[(k - 1) // 2] + ordered[k // 2])
    candidate = (removal <= 0) | (removal < threshold * median)
    available = (~candidate) & (mass > 0) & (lengths >= 0)
    number = mx.minimum(mx.sum(candidate), mx.sum(available)).astype(mx.int32)
    # Costs are never negative. Indices below every positive cost order the
    # centroids without cost; float32 holds them exactly for k below 2**24.
    free = (ids - k).astype(mx.float32)
    target_order = mx.argsort(mx.where(candidate, mx.where(removal > 0, removal, free), mx.inf))
    donor_order = mx.argsort(mx.where(available, -lengths, mx.inf))
    target_rank = mx.argsort(target_order)
    donor_rank = mx.argsort(donor_order)
    is_child = target_rank < number
    is_donor = donor_rank < number
    source = donor_order[target_rank]
    target = target_order[donor_rank]
    return is_child, is_donor, source, target, number


def _adjacent_bf16(donor, child, lower, upper):
    """Coordinate zero of ``child``, moved to an adjacent BF16 value where the
    represented child equals the represented donor (signed zeros are equal).

    The value moves up if the result stays at or below ``upper``, otherwise
    down if that stays at or above ``lower``; otherwise it is kept and the two
    stay coincident. From zero it moves to the smallest normal BF16 value,
    because MLX's cast returns the subnormal ones to zero.
    """
    donor_bits = donor.astype(mx.bfloat16).view(mx.uint16)
    child_bits = child.astype(mx.bfloat16).view(mx.uint16)
    both_zero = ((donor_bits & 0x7fff) == 0) & ((child_bits & 0x7fff) == 0)
    collapsed = mx.all((donor_bits == child_bits) | both_zero, axis=1)
    bits = child_bits[:, 0].astype(mx.uint32)
    negative, tiny = (bits & 0x8000) != 0, (bits & 0x7f80) == 0
    upward = mx.where(tiny, 0x0080, mx.where(negative, bits - 1, bits + 1))
    inward = mx.where(tiny, 0x8080, mx.where(negative, bits + 1, bits - 1))
    above = (upward.astype(mx.uint32) << 16).view(mx.float32)
    below = (inward.astype(mx.uint32) << 16).view(mx.float32)
    rise = ((upward & 0x7fff) < 0x7f80) & (above <= upper[0])
    fall = ((inward & 0x7fff) < 0x7f80) & (below >= lower[0])
    neighbor = mx.where(rise, above, mx.where(fall, below, child[:, 0]))
    return mx.where(collapsed, neighbor, child[:, 0])


def _adjacent_float32(donor, child, lower, upper):
    """As :func:`_adjacent_bf16` for float32 values. From zero the value moves
    to the adjacent subnormal, as in the reference."""
    collapsed = mx.all(donor == child, axis=1)
    value = child[:, 0]
    bits = value.view(mx.uint32)
    negative, zero = (bits & 0x80000000) != 0, (bits & 0x7fffffff) == 0
    upward = mx.where(zero, 1, mx.where(negative, bits - 1, bits + 1))
    inward = mx.where(zero, 0x80000001, mx.where(negative, bits + 1, bits - 1))
    # A value strictly inside the range has that neighbour inside it too.
    neighbor = mx.where(value < upper[0], upward.view(mx.float32),
                        mx.where(value > lower[0], inward.view(mx.float32), value))
    return mx.where(collapsed, neighbor, value)


def revive_by_cost(centers, mass, long_axis, removal, axes, lengths, plan, bounds, *,
                   companion=None, precision="float32"):
    """Move the plan's centroids onto its donors, splitting each donor along its axis.

    A donor at ``c`` with mass ``N`` and long-axis sum ``U`` becomes two
    centroids at ``c -/+ s u`` with ``u = U / |U|`` and ``s = sqrt(2 |U| /
    (pi N))``, the half-means of a Gaussian along ``u``; ``s`` is zero when
    ``|U|`` or ``N`` is. Both keep half of the donor's mass, long-axis sum and
    removal cost, and the moved centroid takes the donor's axis. Positions are
    clipped to ``bounds``, the per-feature range of the rows and initial
    centroids, so that the magnitudes checked before training keep holding.
    Coincident copies are separated by one adjacent value in coordinate zero
    (BF16 values for ``precision="bfloat16"``) where that stays inside the
    bounds. Such copies still score alike: the one with the higher index
    receives no rows and moves again at a later check. ``companion`` optionally
    supplies slow-EMA (centers, counts), displaced by the same vectors.
    Returns ``centers, mass, long_axis, removal, axes, number, companion``.
    """
    is_child, is_donor, source, _, number = plan
    lower, upper = bounds
    valid = (lengths > 0) & (mass > 0)
    ratio = mx.where(valid, lengths, 0.0) / mx.where(valid, mass, 1.0)
    # The ratio is at most the largest squared row-to-centroid distance; a
    # value that is not finite could only come from unchecked operands.
    ratio = mx.where(mx.isfinite(ratio), ratio, 0.0)
    offset = mx.sqrt((2.0 / math.pi) * ratio)[:, None] * axes
    adjacent = _adjacent_bf16 if precision == "bfloat16" else _adjacent_float32
    child_offset = offset[source]

    def split(position, weight):
        donor = mx.clip(position - offset, lower, upper)
        child = mx.clip(position[source] + child_offset, lower, upper)
        child = mx.concatenate((adjacent(donor[source], child, lower, upper)[:, None], child[:, 1:]),
                               axis=1)
        new_position = mx.where(is_child[:, None], child, mx.where(is_donor[:, None], donor, position))
        new_weight = mx.where(is_child, weight[source] * 0.5, mx.where(is_donor, weight * 0.5, weight))
        return new_position, new_weight

    centers, mass = split(centers, mass)
    long_axis = mx.where(is_child[:, None], long_axis[source] * 0.5,
                         mx.where(is_donor[:, None], long_axis * 0.5, long_axis))
    removal = mx.where(is_child, removal[source] * 0.5, mx.where(is_donor, removal * 0.5, removal))
    axes = mx.where(is_child[:, None], axes[source], axes)
    if companion is not None:
        companion = split(*companion)
    return centers, mass, long_axis, removal, axes, number, companion


# A closing-pass revival check is kept as one int32 per centroid until the
# rows before it have their twins; the checks kept at once stay below this.
_CLOSING_STASH_BYTES = 128 << 20


class ClosingLog:
    """Rows and moves of the closing pass under cost-based revival.

    A centroid's closing mean is the mean of the closing-pass rows that belong
    to it when training ends. Rows are assigned while centroids still move, so
    two kinds of rows are looked at again at the end:

    stale rows
        were assigned to a centroid that was moved onto a donor afterwards;
    twin rows
        were assigned to a centroid that was split afterwards. The first such
        split names the row's twin, the centroid that took the other half,
        provided that centroid did not move again.

    Times count the closing-pass rows assigned so far. The pass is kept as the
    intervals between revival checks, each with its labels. A check is kept as
    the child of every donor until a backward sweep over the kept checks gives
    each earlier row the first later split of its centroid; the sweep runs
    once the kept checks reach ``stash_bytes``, and when the pass ends.
    """

    def __init__(self, k, stash_bytes=None):
        self.k = int(k)
        self.done = 0
        self.moved_at = mx.full((self.k,), -1, mx.int32)
        self.moves = mx.array(0, mx.int32)
        budget = _CLOSING_STASH_BYTES if stash_bytes is None else stash_bytes
        self._segment = max(1, int(budget) // (4 * self.k))
        self._batches = []      # labels assigned since the last check
        self._intervals = []    # [labels, time of the check that follows, twin, time of that split]
        self._stash = []        # (child of each donor or -1, time) of the checks not swept yet

    def rows(self, labels):
        """Record one batch of closing-pass labels."""
        self._batches.append(labels)
        self.done += labels.shape[0]

    def _interval(self):
        labels = (mx.zeros((0,), mx.uint32) if not self._batches else
                  self._batches[0] if len(self._batches) == 1 else mx.concatenate(self._batches))
        self._batches = []
        return labels

    def check(self, plan):
        """Record a revival check; returns the arrays to evaluate with the step."""
        is_child, is_donor, _, target, number = plan
        self.moved_at = mx.where(is_child, self.done, self.moved_at)
        self.moves = self.moves + number
        self._intervals.append([self._interval(), self.done, None, None])
        self._stash.append((mx.where(is_donor, target.astype(mx.int32), -1), self.done))
        pending = [self.moved_at, self.moves, self._intervals[-1][0], self._stash[-1][0]]
        if len(self._stash) >= self._segment:
            pending.extend(self._sweep())
        return pending

    def _sweep(self):
        """Give the rows before each kept check the first later split of their centroid."""
        child = mx.full((self.k,), -1, mx.int32)
        born = mx.full((self.k,), -1, mx.int32)
        first = len(self._intervals) - len(self._stash)
        changed = []
        for position in range(len(self._stash) - 1, -1, -1):
            children, time = self._stash[position]
            split = children >= 0
            child, born = mx.where(split, children, child), mx.where(split, time, born)
            entry = self._intervals[first + position]
            entry[2], entry[3] = child[entry[0]], born[entry[0]]
            changed.extend(entry[2:])
        # Rows of earlier sweeps without a twin so far take the first split kept here.
        for entry in self._intervals[:first]:
            unset = entry[2] < 0
            entry[2], entry[3] = (mx.where(unset, child[entry[0]], entry[2]),
                                  mx.where(unset, born[entry[0]], entry[3]))
            changed.extend(entry[2:])
        self._stash = []
        return changed

    def finish(self):
        """End the pass. Returns ``labels, stale, twins`` in the order of assignment.

        ``stale`` marks the rows whose centroid moved after they were assigned.
        ``twins`` holds each other row's twin, or -1: a row has one when its
        centroid was split after the row was assigned and the centroid that
        took the other half at the first such split has not moved since.
        """
        if self._stash:
            self._sweep()
        tail = self._interval()
        if tail.size or not self._intervals:
            none = mx.full(tail.shape, -1, mx.int32)
            self._intervals.append([tail, self.done, none, none])
        labels, stale, twins = [], [], []
        for index, end, twin, born in self._intervals:
            late = self.moved_at[index] >= end
            kept = (~late) & (twin >= 0) & (self.moved_at[mx.maximum(twin, 0)] == born)
            labels.append(index)
            stale.append(late)
            twins.append(mx.where(kept, twin, -1))
        if len(labels) == 1:
            return labels[0], stale[0], twins[0]
        return mx.concatenate(labels), mx.concatenate(stale), mx.concatenate(twins)


def _search_chunks(count, batch_size):
    """Sizes of the assignment calls that look ``count`` rows up again.

    Assignment kernels are compiled per number of rows, so the calls take
    sizes that recur across fits: whole training batches, then powers of two
    from 1,024 (from 128 for fewer rows than that). Returns ``(rows, size)``
    pairs; where ``size`` exceeds ``rows`` the call is padded. A remainder is
    rounded up to a power of two only when that pads by less than a third,
    and is otherwise split.
    """
    chunks, smallest = [], 1024 if count >= 1024 else 128
    while count > 0:
        if count >= batch_size:
            size = batch_size
        else:
            size = smallest
            while 2 * size <= count:
                size *= 2
            # ``size`` is the largest power of two within ``count``, or the smallest call.
            if count > size + size // 2 and 2 * size <= batch_size:
                size *= 2
            size = min(size, batch_size)
        rows = min(count, size)
        chunks.append((rows, size))
        count -= rows
    return chunks


class EMeans(ClusterMixin, TransformerMixin, BaseEstimator):
    """Mini-batch clustering with scheduled EMA updates on the Metal GPU.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of centroids.
    max_iter : float, default=3
        Epoch budget, including fractional epochs. For budgets >= 1 a full
        final pass is reserved for last-epoch means, matching Faiss.
    batch_size : int or None
        Statistical mini-batch size. None uses the fitted batch-size law.
    alpha0 : float or None
        Initial EMA step in (0,1]. None uses the fitted step-size law.
    recipe : {'paper', 'faiss'}, default='paper'
        Published Table 2 recipe or the current reference-code recipe. The
        resolved coefficients and conventions are recorded in ``recipe_``.
    batch_law : {'auto', 'throughput', 'quality'}, default='auto'
        Quality divides the unrounded throughput batch size by eight. Paper
        auto selects quality above 2M rows; Faiss auto selects it at d <= 16.
    finalization : {'last_epoch', 'ema', 'slow_ema'}, default='last_epoch'
        Last-pass assigned means, the primary EMA, or a second, slower EMA.
    slow_ema_half_life : float or None
        Half-life in logical batches for the optional second EMA. Otherwise
        its constant step is ``slow_ema_multiplier * 0.1 * alpha0``.
    slow_ema_multiplier : float, default=1
        Slow-EMA step relative to the final cosine floor (Appendix I).
    random_state : int, RandomState or None, default=1234
        Initialization, epoch shuffling and donor-sampling random seed.
    init : {'random'} or array of shape (n_clusters, n_features)
        Initial centroids. Random initialization samples without replacement.
    spherical : bool, default=False
        Normalize input rows and assignment centroid directions. EMA state
        retains unprojected centers. Zero input rows remain zero.
    unused_threshold : float or None
        Revival threshold as a fraction of mean mass; 0 disables revival.
        It applies to ``revival='mass'`` only.
    revival : {'mass', 'cost'}, default='mass'
        Rule that reuses poorly placed centroids, every five mini-batches.
        ``mass`` refills centroids whose mass is below ``unused_threshold``
        times the mean by splitting donors drawn in proportion to their mass.
        ``cost`` moves the centroids that are cheapest to remove, by the sum
        over their rows of the distance to the second-nearest centroid minus
        the distance to the nearest, onto the clusters with the longest axes
        and splits those at the Gaussian half-means along the axis. It needs
        the second-nearest distance of every row and two more statistics per
        centroid; see ``docs/e-means.md``.
    assignment : {'auto', 'gemm', 'fused'}, default='auto'
        Metal assignment implementation; useful for performance comparisons.
    precision : {'float32', 'float16', 'bfloat16'}, default='float32'
        Assignment arithmetic. EMA state and accumulated sums remain float32.
        Reduced precision is explicitly approximate. bfloat16 also rounds rows
        before statistics, matching the Faiss H100 cast route; it uses raw
        coordinates, represented-centroid norms, and float32 state.
    assignment_batch_size : int or None
        Maximum rows in an assignment tile, independent of statistical B.
    compute_labels : bool, default=True
        Reassign all input rows after training for sklearn-style ``labels_``
        and ``inertia_``. False omits this extra pass and sets both to None.
    verbose : bool, default=False
        Print one progress line per pass.

    Notes
    -----
    ``last_epoch_centers_`` and ``ema_centers_`` are always available.
    ``slow_ema_centers_`` is available when slow EMA was requested. These
    finalizers use moving assignments, not a fresh Lloyd pass. Final labels
    (when enabled) are reassigned against the chosen final centroids.

    With ``revival='cost'`` centroids keep moving during the last pass, so
    that pass assigns rows to centroids that move later. When it ends, a row
    whose centroid was moved onto a donor afterwards is assigned again to the
    final centroids, and a row whose centroid was split afterwards goes to
    the nearer of that centroid and the one that took the other half.
    ``last_epoch_centers_`` holds the means of the last-pass rows under these
    labels (the EMA position for a centroid left without rows). This keeps
    the labels of the last pass on the GPU, four bytes per row, and reads one
    mask of its rows back to the host. Splits are clipped to the per-feature
    range of the rows and initial centroids, and the selection draws no
    random numbers.

    Remaining batches are kept, unlike Algorithm 1's floor notation. GPU
    float32 reductions need not be bitwise identical to the CPU reference.
    Faiss recipe mode matches its statistical recipe, not bitwise execution:
    donor keys use float32. Float32/float16 multiplicative splits leave zero
    donors zero; bfloat16 repairs collapsed represented splits by moving one
    child coordinate to the adjacent representable value, as in Faiss.
    This estimator requires MLX's Metal GPU; it never silently falls back.
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        max_iter=3,
        batch_size=None,
        alpha0=None,
        recipe="paper",
        batch_law="auto",
        finalization="last_epoch",
        slow_ema_half_life=None,
        slow_ema_multiplier=1.0,
        random_state=1234,
        init="random",
        spherical=False,
        unused_threshold=None,
        revival="mass",
        assignment="auto",
        precision="float32",
        assignment_batch_size=None,
        compute_labels=True,
        verbose=False,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.recipe = recipe
        self.batch_law = batch_law
        self.finalization = finalization
        self.slow_ema_half_life = slow_ema_half_life
        self.slow_ema_multiplier = slow_ema_multiplier
        self.random_state = random_state
        self.init = init
        self.spherical = spherical
        self.unused_threshold = unused_threshold
        self.revival = revival
        self.assignment = assignment
        self.precision = precision
        self.assignment_batch_size = assignment_batch_size
        self.compute_labels = compute_labels
        self.verbose = verbose

    def _validate_parameters(self):
        if self.finalization not in {"last_epoch", "ema", "slow_ema"}:
            raise ValueError("finalization must be 'last_epoch', 'ema', or 'slow_ema'")
        if self.assignment not in {"auto", "gemm", "fused"}:
            raise ValueError("assignment must be 'auto', 'gemm', or 'fused'")
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("precision must be 'float32', 'float16', or 'bfloat16'")
        if self.revival not in {"mass", "cost"}:
            raise ValueError("revival must be 'mass' or 'cost'")
        if self.revival == "cost" and self.unused_threshold is not None:
            raise ValueError("unused_threshold applies to revival='mass'; leave it None with revival='cost'")
        if self.assignment_batch_size is not None and (
            isinstance(self.assignment_batch_size, bool)
            or not isinstance(self.assignment_batch_size, Integral)
            or self.assignment_batch_size < 1
        ):
            raise ValueError("assignment_batch_size must be a positive integer or None")
        for name in ("slow_ema_half_life", "slow_ema_multiplier"):
            value = getattr(self, name)
            if value is None and name == "slow_ema_multiplier":
                raise ValueError("slow_ema_multiplier must be finite and positive")
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")

    def _input(self, X, *, fit=False, return_bound=False):
        if isinstance(X, mx.array):
            if X.ndim != 2 or min(X.shape) < 1:
                raise ValueError("X must be a nonempty two-dimensional matrix")
            if X.shape[0] > np.iinfo(np.int32).max:
                raise ValueError("X has too many rows for 32-bit GPU permutation indices")
            if mx.issubdtype(X.dtype, mx.complexfloating):
                raise ValueError("X must contain real values")
            data = X.astype(mx.float32)
            finite, magnitude = mx.all(mx.isfinite(data)), mx.max(mx.abs(data))
            mx.eval(finite, magnitude)
            maximum = float(magnitude.item())
            if not bool(finite.item()) or not math.isfinite(maximum):
                raise ValueError("X must contain finite values representable in float32")
        else:
            array = np.asarray(X)
            if array.ndim != 2 or min(array.shape) < 1:
                raise ValueError("X must be a nonempty two-dimensional matrix")
            if array.shape[0] > np.iinfo(np.int32).max:
                raise ValueError("X has too many rows for 32-bit GPU permutation indices")
            if np.iscomplexobj(array):
                raise ValueError("X must contain real values")
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.asarray(array, dtype=np.float32, order="C")
            # Reuse validation to obtain a magnitude bound without allocating
            # an array-sized abs/isfinite temporary on large embeddings.
            lower, upper = float(array.min()), float(array.max())
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError("X must contain finite values representable in float32")
            maximum = max(abs(lower), abs(upper))
            data = mx.array(array)
        if not fit and data.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {data.shape[1]} features, expected {self.n_features_in_}")
        if self.spherical:
            data, maximum = _normalize(data), 0.0 if maximum == 0 else 1.0
        return (data, maximum) if return_bound else data

    def _assign(self, data, centers, *, second=False):
        if not self.spherical and self.precision != "bfloat16":
            data, centers = data - self._offset_mx, centers - self._offset_mx
        return assign(
            data,
            centers,
            batch_size=self.assignment_batch_size,
            spherical=self.spherical,
            method=self.assignment,
            precision=self.precision,
            **({"second": True} if second else {}),
        )

    def _resolve_recipe(self, n, d):
        recipe = resolve_recipe(
            n, self.n_clusters, d, self.max_iter, self.batch_size, self.alpha0,
            self.batch_law, self.recipe, self.unused_threshold,
        )
        if self.revival == "cost":
            # Neither the mass threshold nor the multiplicative split applies.
            recipe.update(unused_threshold=None, split_epsilon=None,
                          removal_threshold=REMOVAL_THRESHOLD)
        return recipe

    def _configuration_signature(self):
        primitive = (str, int, float, bool, type(None))
        return tuple((key, value if isinstance(value, primitive) else id(value))
                     for key, value in sorted(self.get_params(deep=False).items()))

    def prepare_device(self, X):
        """Validate/upload data and explicit initialization before a timed fit.

        Returns reusable resident inputs for :meth:`train_device`. Input scans,
        upload, random initial-center selection and the FP32 assignment origin
        are preparation. Epoch permutations, state initialization, BF16 casts,
        training, online labels and finalization remain in ``train_device``.
        """
        self._validate_parameters()
        if not mx.metal.is_available():
            raise RuntimeError("EMeans requires an available Apple Silicon Metal GPU")
        with mx.stream(mx.gpu):
            data, data_maximum = self._input(X, fit=True, return_bound=True)
            n, d = data.shape
            self.recipe_ = self._resolve_recipe(n, d)
            k = int(self.n_clusters)
            if k * d > np.iinfo(np.uint32).max or self.recipe_["batch_size"] * d > np.iinfo(np.uint32).max:
                raise ValueError("centroid or statistical batch size exceeds 32-bit Metal indexing; reduce n_clusters or batch_size")
            cost = getattr(self, "revival", "mass") == "cost"
            if cost and k >= _MAX_PLAN_CLUSTERS:
                raise ValueError(f"revival='cost' supports fewer than {_MAX_PLAN_CLUSTERS} centroids")
            self.recipe_["revival"] = ("cost" if cost else
                                       "mass" if self.recipe_["unused_threshold"] > 0 else "none")
            rng = check_random_state(self.random_state)
            seed = int(self.random_state) if isinstance(self.random_state, Integral) else int(rng.randint(2**32))
            if not 0 <= seed < 2**32:
                raise ValueError("random_state integer must be in [0, 2**32)")
            if isinstance(self.init, str):
                if self.init != "random":
                    raise ValueError("init must be 'random' or an initial centroid matrix")
                indices = np.random.RandomState(seed).permutation(n)[:k]
                centers = data[mx.array(indices, dtype=mx.int32)]
                initial_maximum = data_maximum
            else:
                centers, initial_maximum = self._input(self.init, fit=True, return_bound=True)
                if centers.shape != (k, d):
                    raise ValueError(f"init must have shape {(k, d)}")
            self._check_float32_range(data_maximum, initial_maximum, n=n, d=d, training=True)
            # BF16 must round original coordinates: recentering before rounding
            # would change the represented values used by the H100 algorithm.
            centered = not self.spherical and self.precision != "bfloat16"
            self._offset_mx = mx.mean(data, axis=0) if centered else mx.zeros((d,), mx.float32)
            use_slow = self.finalization == "slow_ema" or self.slow_ema_half_life is not None
            slow_alpha = (-math.expm1(-math.log(2) / self.slow_ema_half_life)
                          if self.slow_ema_half_life is not None else
                          min(1.0, self.slow_ema_multiplier * 0.1 * self.recipe_["alpha0"]))
            smallest_step = min(self.recipe_["alpha0"] * 0.1, slow_alpha) if use_slow else self.recipe_["alpha0"] * 0.1
            if smallest_step < float(np.finfo(np.float32).tiny):
                raise ValueError("EMA step is too small for float32; increase alpha0 or reduce the slow EMA half-life")
            self.recipe_.update(
                seed=seed, backend="metal", precision=self.precision,
                assignment=self.assignment, finalization=self.finalization,
                spherical=bool(self.spherical), slow_alpha=slow_alpha if use_slow else None,
                assignment_centering="global_mean" if centered else "none",
                statistics_representation="bfloat16" if self.precision == "bfloat16" else "float32",
            )
            if self.precision == "float16":
                self._check_half_range(data, centers, training=True)
            bounds = None
            if cost:
                # Rows and their means stay inside the per-feature range of the
                # rows and initial centroids; splits are clipped to it. BF16
                # rows are rounded, which can move the range by one BF16 step.
                lower = mx.minimum(mx.min(data, axis=0), mx.min(centers, axis=0))
                upper = mx.maximum(mx.max(data, axis=0), mx.max(centers, axis=0))
                if self.precision == "bfloat16":
                    lower = mx.minimum(lower, lower.astype(mx.bfloat16).astype(mx.float32))
                    upper = mx.maximum(upper, upper.astype(mx.bfloat16).astype(mx.float32))
                bounds = (lower, upper)
                self.recipe_.update(closing="last_epoch_mean_of_rerouted_rows",
                                    epoch_seed="distinct_from_initialization")
                mx.eval(lower, upper)
            mx.eval(data, centers, self._offset_mx)
            return PreparedEMeansFit(data, centers, self._offset_mx, dict(self.recipe_), seed,
                                     self._configuration_signature(), bounds)

    def _reroute_closing(self, data, order, log, centers, *, keep_labels=False):
        """Closing sums of a cost-based fit whose centroids moved during the last pass.

        Stale rows are assigned again to the final centroids, and twin rows go
        to the nearer of their centroid and its twin (see :class:`ClosingLog`).
        The sums and counts are then rebuilt from the closing-pass rows under
        these labels, batch by batch as in training. One host read finds the
        stale rows. Returns ``sums, counts, labels``; ``labels`` (in the order
        of assignment) only with ``keep_labels``.
        """
        k, batch_size = int(self.n_clusters), self.recipe_["batch_size"]
        rows = log.done
        labels, stale, twins = log.finish()
        readout = _normalize(centers) if self.spherical else centers

        def gather(batch_order):
            return gather_bf16(data, batch_order) if self.precision == "bfloat16" else data[batch_order]

        positions = np.flatnonzero(np.asarray(stale))
        if positions.size:
            found, start = [], 0
            for taken, size in _search_chunks(positions.size, batch_size):
                # Pad by repetition to a size the assignment kernels know.
                padded = np.resize(positions[start:start + taken], size).astype(np.int32)
                found.append(self._assign(gather(order[mx.array(padded)]), readout)[0][:taken])
                mx.eval(found[-1])
                start += taken
            labels[mx.array(positions.astype(np.int32))] = found[0] if len(found) == 1 else mx.concatenate(found)
        sums, counts = mx.zeros((k, data.shape[1]), mx.float32), mx.zeros((k,), mx.float32)
        routed = []
        for start in range(0, rows, batch_size):
            stop = min(start + batch_size, rows)
            batch = gather(order[start:stop])
            final = twin_route(batch, labels[start:stop], twins[start:stop], readout)
            batch_sums, batch_counts = batch_stats(batch, final, k)
            sums, counts = sums + batch_sums, counts + batch_counts
            if keep_labels:
                routed.append(final)
            mx.async_eval(sums, counts, final)
        return sums, counts, ((routed[0] if len(routed) == 1 else mx.concatenate(routed)) if keep_labels else None)

    def train_device(self, prepared, *, capture_assignments=False):
        """Run a complete fit from prepared inputs, returning resident outputs.

        Reusing ``prepared`` restarts from the same initial centers and seed.
        There is no upload, finite-value scan, or NumPy centroid export here.
        All epoch permutations are generated inside this call. Optional labels
        are captured during the final pass, not reassigned after training.
        With ``revival='cost'`` a final pass in which centroids moved ends by
        sorting out its rows (see the class notes): one row mask is read back
        to find the rows of moved centroids, their positions are sent to the
        GPU and they are assigned again. The captured labels are then those
        the closing means were taken over.
        This low-level method does not populate sklearn's host fitted attributes.
        """
        if not isinstance(prepared, PreparedEMeansFit) or prepared.signature != self._configuration_signature():
            raise ValueError("prepared inputs do not match this estimator configuration")
        started = time.perf_counter()
        with mx.stream(mx.gpu):
            data, centers = prepared.data, prepared.initial
            n, d = data.shape
            k = int(self.n_clusters)
            self.recipe_, self._offset_mx = dict(prepared.recipe), prepared.offset
            mass = mx.full((k,), self.recipe_["init_usage"], dtype=mx.float32)
            slow_alpha = self.recipe_["slow_alpha"]
            slow = (centers, mass) if slow_alpha is not None else None
            split_rng = np.random.RandomState(prepared.seed)
            total_splits = mx.array(0, dtype=mx.int32)
            history = []
            step = 0
            pass_sums, pass_counts = mx.zeros((k, d), mx.float32), mx.zeros((k,), mx.float32)
            batch_size = self.recipe_["batch_size"]
            final_labels = []
            revival = self.recipe_["revival"]
            cost = revival == "cost"
            if cost:
                # Unit starting axes from a fixed key, so that a seed's fit repeats.
                axes = _normalize(mx.random.normal((k, d), key=mx.random.key(prepared.seed)))
                long_axis, removal = mx.zeros((k, d), mx.float32), mx.zeros((k,), mx.float32)
                closing_log = ClosingLog(k)
            for epoch, rows in enumerate(self.recipe_["epoch_rows"]):
                epoch_start = time.perf_counter()
                order_seed = epoch_seed(prepared.seed, epoch, distinct=cost)
                order = mx.array(np.random.RandomState(order_seed).permutation(n), dtype=mx.int32)
                final_epoch = epoch == len(self.recipe_["epoch_rows"]) - 1
                objectives = []
                report_counts = mx.zeros((k,), mx.float32)
                for start in range(0, rows, batch_size):
                    batch_order = order[start:min(start + batch_size, rows)]
                    batch = gather_bf16(data, batch_order) if self.precision == "bfloat16" else data[batch_order]
                    readout = _normalize(centers) if self.spherical else centers
                    if cost:
                        labels, distances, margins = self._assign(batch, readout, second=True)
                        axis_sums, costs = removal_stats(batch, labels, margins, readout, axes)
                    else:
                        labels, distances = self._assign(batch, readout)
                    sums, counts = batch_stats(batch, labels, k)
                    objectives.append(mx.sum(distances))
                    if self.precision == "bfloat16":
                        report_counts = report_counts + counts
                    if final_epoch:
                        pass_sums, pass_counts = pass_sums + sums, pass_counts + counts
                        if capture_assignments:
                            final_labels.append(labels)
                        if cost:
                            closing_log.rows(labels)
                    recorded = []
                    alpha = scheduled_alpha(self.recipe_["alpha0"], step, self.recipe_["total_steps"])
                    previous_mass = mass
                    if self.precision == "bfloat16":
                        centers, mass = ema_update(centers, mass, sums, counts, alpha, decay=1.0 - alpha)
                    else:
                        centers, mass = ema_update(centers, mass, sums, counts, alpha)
                    if self.recipe == "faiss" and alpha == 1.0:
                        mass = mx.where(counts > 0, mass, previous_mass)
                    if cost:
                        long_axis, removal, axes, lengths = removal_update(
                            long_axis, removal, axis_sums, costs, axes, alpha)
                    if slow is not None:
                        slow = ema_update(*slow, sums, counts, slow_alpha,
                                          **({"decay": 1.0 - slow_alpha} if self.precision == "bfloat16" else {}))
                    if (step + 1) % 5 == 0 and cost:
                        plan = removal_plan(removal, mass, lengths)
                        centers, mass, long_axis, removal, axes, nsplit, slow = revive_by_cost(
                            centers, mass, long_axis, removal, axes, lengths, plan, prepared.bounds,
                            companion=slow, precision=self.precision)
                        total_splits = total_splits + nsplit
                        if final_epoch:
                            recorded = closing_log.check(plan)
                    elif (step + 1) % 5 == 0 and revival == "mass":
                        uniform = mx.array(split_rng.random_sample(k), dtype=mx.float32)
                        centers, mass, nsplit, slow = revive(
                            centers, mass, uniform, self.recipe_["unused_threshold"],
                            self.recipe_["split_epsilon"], companion=slow, precision=self.precision)
                        total_splits = total_splits + nsplit
                    pending = [centers, mass, objectives[-1], total_splits]
                    if cost:
                        pending.extend((long_axis, removal, axes))
                        pending.extend(recorded)
                    if self.precision == "bfloat16":
                        pending.append(report_counts)
                    if final_epoch:
                        pending.extend((pass_sums, pass_counts))
                        if capture_assignments or cost:
                            pending.append(labels)
                    if slow is not None:
                        pending.extend(slow)
                    mx.async_eval(*pending)
                    step += 1
                # Match the reference's per-epoch reporting boundary. This
                # diagnostic reduction is FP32: Metal has no FP64 arithmetic.
                objective = float(mx.sum(mx.stack(objectives)).item()) if objectives else 0.0
                mx.eval(centers, mass, total_splits)
                stat = {"epoch": epoch + 1, "rows": rows, "steps": step,
                        "moving_objective": objective, "moving_mse": objective / rows if rows else 0.,
                        "seconds": time.perf_counter() - epoch_start,
                        "cumulative_seconds": time.perf_counter() - started,
                        "splits": int(total_splits.item())}
                history.append(stat)
                if self.precision == "bfloat16":
                    stat["pass_count_squares"] = float(mx.sum(report_counts * report_counts).item())
                if self.verbose:
                    print(f"e-means pass {epoch + 1}: {rows} rows, moving MSE={stat['moving_mse']:.7g}, {stat['seconds']:.3f}s", flush=True)
            rerouted = None
            if cost and int(closing_log.moves.item()) > 0:
                # Centroids moved during the closing pass: its rows are sorted
                # out before the means are taken.
                pass_sums, pass_counts, rerouted = self._reroute_closing(
                    data, order, closing_log, centers, keep_labels=capture_assignments)
            closing = mx.where((pass_counts > 0)[:, None], pass_sums / mx.maximum(pass_counts[:, None], 1), centers)
            if self.spherical:
                closing, centers = _normalize(closing), _normalize(centers)
            slow_centers = (_normalize(slow[0]) if self.spherical else slow[0]) if slow is not None else None
            chosen = {"ema": centers, "last_epoch": closing, "slow_ema": slow_centers}[self.finalization]
            captured = None
            if capture_assignments:
                # Metal scatter supports 32-bit integers; widen the completed
                # captured vector to the reference API's int64 representation.
                # After a closing-pass move these are the labels the closing
                # means were taken over, not the ones of the first assignment.
                moving_labels = (mx.concatenate(final_labels) if rerouted is None else rerouted).astype(mx.int32)
                captured = mx.full((n,), -1, dtype=mx.int32)
                captured[order[:len(moving_labels)]] = moving_labels
                captured = captured.astype(mx.int64)
            pending = [chosen, centers, closing, total_splits]
            if slow_centers is not None:
                pending.append(slow_centers)
            if captured is not None:
                pending.append(captured)
            mx.eval(*pending)
            return DeviceEMeansFit(chosen, centers, closing, slow_centers, captured,
                                   tuple(history), total_splits, step, len(history), time.perf_counter() - started)

    def fit(self, X, y=None):
        """Fit centroids and populate sklearn-style NumPy fitted attributes."""
        started = time.perf_counter()
        prepared = self.prepare_device(X)
        result = self.train_device(prepared)
        with mx.stream(mx.gpu):
            self.offset_ = np.array(prepared.offset)
            self.ema_centers_ = np.array(result.ema_centers)
            self.last_epoch_centers_ = np.array(result.last_epoch_centers)
            self.slow_ema_centers_ = None if result.slow_ema_centers is None else np.array(result.slow_ema_centers)
            self._centers_mx = result.centers
            self.cluster_centers_ = np.array(result.centers)
            self._centers_magnitude_ = max(abs(float(self.cluster_centers_.min())), abs(float(self.cluster_centers_.max())))
            self.n_features_in_ = prepared.data.shape[1]
            self.n_steps_, self.n_iter_ = result.steps, result.epochs
            self.n_splits_ = int(result.split_count.item())
            self.iteration_stats_ = list(result.history)
            self.execution_backend_ = "metal"
            self.optimization_time_ = time.perf_counter() - started
            self.labels_, self.inertia_ = None, None
            if self.compute_labels:
                labels, distances = self._assign(prepared.data, result.centers)
                mx.eval(labels, distances)
                self.labels_ = np.array(labels, dtype=np.int64)
                self.inertia_ = float(np.asarray(distances).sum(dtype=np.float64))
            self.fit_time_ = time.perf_counter() - started
        return self

    def predict(self, X):
        """Return nearest final-centroid indices."""
        check_is_fitted(self, "cluster_centers_")
        with mx.stream(mx.gpu):
            data = self._inference_input(X)
            if self.precision == "float16":
                self._check_half_range(data, self._centers_mx)
            labels, _ = self._assign(data, self._centers_mx)
            return np.array(labels, dtype=np.int64)

    def transform(self, X):
        """Return Euclidean (spherical: chord) distances to all centers."""
        check_is_fitted(self, "cluster_centers_")
        with mx.stream(mx.gpu):
            data = self._inference_input(X)
            centers = self._centers_mx
            if not self.spherical:
                data, centers = data - self._offset_mx, centers - self._offset_mx
            chunks = []
            size = self.assignment_batch_size or max(1, (16 << 20) // (4 * self.n_clusters))
            for start in range(0, len(data), size):
                batch = data[start:start + size]
                if self.spherical:
                    distances = 2 - 2 * matmul(batch, centers, transpose_b=True)
                else:
                    distances = mx.sum(batch * batch, axis=1)[:, None] + mx.sum(centers * centers, axis=1)[None, :] - 2 * matmul(batch, centers, transpose_b=True)
                chunks.append(np.array(mx.sqrt(mx.maximum(distances, 0))))
            return np.concatenate(chunks)

    def fit_predict(self, X, y=None):
        """Fit and return final labels, computing them even when disabled."""
        self.fit(X, y)
        return self.labels_ if self.labels_ is not None else self.predict(X)

    def score(self, X, y=None):
        """Negative sum of squared distances (larger is better)."""
        check_is_fitted(self, "cluster_centers_")
        with mx.stream(mx.gpu):
            data = self._inference_input(X)
            if self.precision == "float16":
                self._check_half_range(data, self._centers_mx)
            _, distances = self._assign(data, self._centers_mx)
            return -float(np.array(distances).sum(dtype=np.float64))

    def _inference_input(self, X):
        data, maximum = self._input(X, return_bound=True)
        # The fixed assignment origin can be larger than the final centers.
        origin = max(abs(float(self.offset_.min())), abs(float(self.offset_.max())))
        self._check_float32_range(max(maximum, origin), self._centers_magnitude_, n=1, d=self.n_features_in_)
        return data

    def _check_float32_range(self, data_maximum, initial_maximum, *, n, d, training=False):
        """Bound float32 distances, sums and objectives before GPU execution.

        A coordinate hull bounds every EMA mean. Every possible mass-based
        revival can enlarge that hull by at most (1 + epsilon); cost-based
        revival clips its splits to the hull. Centering can double the
        coordinate magnitude, and subtraction can double it again. The margin
        below bounds squared residuals and their float32 per-pass reduction,
        as well as original-coordinate batch/closing sums. This conservative
        check requires no per-batch synchronization.
        """
        maximum = max(data_maximum, initial_maximum)
        if self.precision == "bfloat16":
            # Cover BF16 rounding and the smallest nonzero repaired donor.
            maximum = max(maximum, 2.0**-133) * (1 + 2.0**-7)
        if training and maximum > 0 and self.recipe_["revival"] == "mass":
            checks = self.recipe_["total_steps"] // 5
            per_check = math.log1p(self.recipe_["split_epsilon"])
            if self.precision == "bfloat16":
                per_check += math.log1p(2.0**-7)
            exponent = checks * per_check
            maximum = math.inf if exponent > 700 else maximum * math.exp(exponent)
        largest = float(np.finfo(np.float32).max)
        safe_maximum = min(largest / (8 * n), math.sqrt(largest / (64 * n * d)))
        if not math.isfinite(maximum) or maximum > safe_maximum:
            raise ValueError("input or initial centroid magnitude is too large for safe float32 distances and accumulation; rescale the data")

    def _check_half_range(self, data, centers, training=False):
        """Reject unsafe half casts without synchronizing every GPU update.

        EMA means remain in the coordinate hull of their operands. Multiplicative
        splits can extend it: conservatively bound that extension over every
        possible revival. This can reject borderline data whose actual run would
        stay in range; float32 remains available for those inputs.
        """
        shift = self._offset_mx
        maximum = mx.maximum(mx.max(mx.abs(data - shift)), mx.max(mx.abs(centers - shift)))
        limit = float(maximum.item())
        if training and not self.spherical and self.recipe_["revival"] == "mass":
            checks = self.recipe_["total_steps"] // 5
            growth_log = checks * math.log1p(self.recipe_["split_epsilon"])
            if growth_log > 700:
                limit = math.inf
            else:
                raw_max = float(mx.maximum(mx.max(mx.abs(data)), mx.max(mx.abs(centers))).item())
                limit += raw_max * math.expm1(growth_log)
        if not math.isfinite(limit) or limit > float(np.finfo(np.float16).max):
            raise ValueError("centered assignment operands may exceed float16 range; use precision='float32'")
