"""k-means++ seeding that selects scikit-learn's centers with a Metal screen.

scikit-learn's k-means++ draws a few candidates for every new center and, for
each of them, evaluates float64 squared distances to all rows. Those distances
dominate its time, yet a candidate only changes the rows it is nearer to than
their current center: about half of them for the second center and one in
thousands later on.

Here a Metal kernel compares every (candidate, row) pair in float32 and proves
most pairs unchanged with a rigorous error bound. The host evaluates the rest
in float64. Sampling, candidate potentials and the random stream then use the
reference's NumPy expressions, so a ``random_state`` selects the centers
``sklearn.cluster.kmeans_plusplus`` selects on the same machine.

That is certified rather than assumed. The reference rounds a float64 result
to float32, and its float64 dot products come from a BLAS GEMM whose rounding
depends on the shape of the call, so a product over selected rows may differ
from the reference's in the last bits. Every evaluated distance is therefore
an interval that contains the reference value:

* Nearly always both ends round to the same float32, which is then the
  reference's value.
* A row that coincides with a candidate has only rounding noise for a
  distance, between zero and a tiny bound. It keeps that interval.
* In the remaining cases the reference's own chunk is recomputed with its
  shapes.

Float32 sums are monotone in every term for a fixed summation order, so sums
over the lower and the upper ends bracket the reference's. A step is accepted
when both ends give the same candidates and the same potentials. If they ever
differ, as when more centers are requested than there are distinct rows and
only noise is left to sample, the reference routine runs instead from the
initial random state.

The same brackets save work. The kernel returns a float32 bracket for every
pair it cannot prove, so the potentials of all candidates are bracketed before
any float64 arithmetic, and only candidates that can still win are evaluated.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache

import mlx.core as mx
import numpy as np
import sklearn
from sklearn.cluster import kmeans_plusplus as _reference_plusplus

from .._kernels.kmeans_plusplus import (
    candidate_ids,
    candidate_screen,
    projected_screen,
    screen_supported,
)

# scikit-learn 1.8 changed the sampling prefix sums from float64 to the input's
# float32, which changes the candidates a seed draws.
_FLOAT64_PREFIX = tuple(int(part) for part in sklearn.__version__.split(".")[:2]) < (1, 8)

# The reference routine costs about trials * n * (d + 48) per center; below
# this value of that product it is as fast as a GPU round trip.
_MIN_SCREEN_WORK = 1 << 22

# The projected screen: image dimensions, the feature count from which the
# images are cheaper to compare than the rows, the remaining centers needed to
# pay for building them, and the first step that uses them (earlier candidates
# change too many rows for the images to settle much). When the images leave
# more than ``_PROJECT_SHARE`` of the pairs to the full rows, the dense screen
# is faster and takes over for a growing number of steps.
_SUBSPACE = 64
_PROJECT_WIDTH = 256
_PROJECT_STEPS = 256
_PROJECT_FROM = 32
_PROJECT_SHARE = 0.5

# With more rows than this to evaluate, two extra potential sums that rule
# out losing candidates cost less than evaluating those candidates' rows.
_CONTEST_ROWS = 384

_STAGES = ("prepare", "project", "draw", "screen", "collect", "evaluate", "select")


def _block_rows(d):
    """Rows per float64 work block: at most 128 MiB."""
    return max(1, min(2048, (1 << 24) // d))


class _Reference:
    """scikit-learn's float32 squared Euclidean distances, for chosen rows.

    For float32 input the reference forms ``-2 c.x + |c|^2 + |x|^2`` in
    float64 from chunks of upcast rows and rounds the result to float32. The
    row norms do not depend on the candidates and are computed once; each is a
    sum over one row whose rounding does not depend on the other rows.
    """

    def __init__(self, host):
        self.host = host
        n, d = host.shape
        self.norms = np.empty(n)
        step = _block_rows(d)
        for start in range(0, n, step):
            block = host[start:start + step].astype(np.float64)
            self.norms[start:start + step] = np.einsum("ij,ij->i", block, block)
        # Two float64 sums of the same exact products, in any order and with
        # or without fused multiply-add, differ by at most this multiple of
        # |c| |x|: twice the summation bound d * 2^-53, with room for rounding
        # the interval ends and the norms themselves.
        self.spread = 2 * (d + 1) * 2.0**-53 * (1 + 2.0**-30 + 4 * d * 2.0**-53)

    def length(self, count):
        """Rows and candidates per chunk in the reference, for ``count`` candidates."""
        n, d = self.host.shape
        budget = max(((count + n) * d + count * n) / 10, 10 * 2**17)
        span = 2 * d
        return max(int((-span + math.sqrt(span**2 + 4 * budget)) / 2), 1)

    def chunk(self, candidates, index, length):
        """The reference's distances from ``candidates`` to its row chunk ``index``.

        The products have the reference's shapes and operands, so they round
        identically.
        """
        rows = slice(index * length, (index + 1) * length)
        block = self.host[rows].astype(np.float64)
        norms = self.norms[rows][None, :]
        out = np.empty((len(candidates), block.shape[0]), np.float32)
        for start in range(0, len(candidates), length):
            group = candidates[start:start + length].astype(np.float64)
            distances = -2 * (group @ block.T)
            distances += np.einsum("ij,ij->i", group, group)[:, None]
            distances += norms
            out[start:start + length] = distances
        np.maximum(out, 0, out=out)
        return out

    def every(self, candidates):
        """The reference's distances from ``candidates`` to every row."""
        n = len(self.host)
        length = self.length(len(candidates))
        out = np.empty((len(candidates), n), np.float32)
        for index in range(-(-n // length)):
            out[:, index * length:(index + 1) * length] = self.chunk(candidates, index, length)
        return out

    def bounds(self, candidates, rows, which, position):
        """Float32 bounds of the reference distances for selected pairs.

        Pair ``j`` is candidate ``which[j]`` and row ``rows[position[j]]``.
        The reference value lies between the returned ``least`` and ``most``.
        """
        used, which = np.unique(which, return_inverse=True)
        centers = candidates[used].astype(np.float64)
        own = np.einsum("ij,ij->i", centers, centers)
        products = np.empty((len(centers), len(rows)))
        step = _block_rows(self.host.shape[1])
        for start in range(0, len(rows), step):
            block = self.host[rows[start:start + step]].astype(np.float64)
            products[:, start:start + step] = centers @ block.T
        dots = products[which, position]
        ours = own[which]
        theirs = self.norms[rows][position]
        spread = self.spread * np.sqrt(ours * theirs)
        # The reference's rounded expression is monotone in the dot product.
        least = ((-2 * (dots + spread) + ours) + theirs).astype(np.float32)
        most = ((-2 * (dots - spread) + ours) + theirs).astype(np.float32)
        return np.maximum(least, 0, out=least), np.maximum(most, 0, out=most)


class _Projection:
    """Low-dimensional float32 images of the rows, for the projected screen.

    The images are ``A x - b`` for a matrix ``A`` whose rows are nearly
    orthonormal and span the leading directions of a row sample, computed in
    float64 and rounded to float32. Any linear map bounds distances from
    below; one aligned with the data's variance makes the bound tight.
    ``eps`` bounds each image's distance from its exact value.
    """

    def __init__(self, host, norms):
        n, d = host.shape
        m = _SUBSPACE
        # A row sample of at most 256 MiB as float64.
        count = min(n, 8192, max(4 * m, (1 << 25) // d))
        rows = np.arange(n) if count == n else np.linspace(0, n - 1, count).astype(np.intp)
        sample = host[rows].astype(np.float64)
        center = sample.mean(axis=0)
        sample -= center
        # A fixed generator: the caller's random stream is not consumed.
        basis = np.random.default_rng(0).standard_normal((d, m))
        for _ in range(3):
            basis, _ = np.linalg.qr(sample.T @ (sample @ basis))
        basis = np.ascontiguousarray(basis.T)
        largest = math.sqrt(np.einsum("ij,ij->i", basis, basis).max())
        # The squared spectral norm of the exact map is at most 1 + excess.
        excess = np.linalg.norm(basis @ basis.T - np.eye(m)) + m * (d + 2) * 2.0**-52 * largest**2
        self.valid = bool(excess < 2.0**-23)
        shift = basis @ center
        images = np.empty((n, m), np.float32)
        eps = np.empty(n, np.float32)
        # Per image: float32 rounding of its length, and the float64 products'
        # error (d + 2) 2^-53 |a| |x| in each of the m coordinates.
        products = math.sqrt(m) * (d + 2) * 2.0**-53 * largest
        step = _block_rows(d)
        for start in range(0, n, step):
            stop = min(start + step, n)
            block = host[start:stop].astype(np.float64) @ basis.T
            block -= shift
            images[start:stop] = block
            block[:] = images[start:stop]
            length = np.sqrt(np.einsum("ij,ij->i", block, block))
            eps[start:stop] = ((2.0**-24 * length + products * np.sqrt(norms[start:stop]))
                               * (1 + 2.0**-20) + 2.0**-120)
        self.images = mx.array(images)
        self.eps = mx.array(eps)


def _screened(n, d, trials):
    return trials * n * (d + 48) >= _MIN_SCREEN_WORK and screen_supported(n, d, trials)


def plusplus_indices(data, host, n_clusters, random_state, *, profile=None):
    """Row indices of the k-means++ centers for a NumPy ``RandomState``.

    ``data`` is the resident (n, d) float32 MLX input and ``host`` a NumPy
    view of the same values. The result equals the indices returned by
    ``sklearn.cluster.kmeans_plusplus(host, n_clusters=n_clusters,
    random_state=random_state)`` and consumes the same random numbers. Small
    problems, and seedings whose decisions cannot be certified, call that
    routine directly.

    ``profile``, if given, is a dict that receives seconds per stage and
    counts of the host's work.
    """
    n, d = host.shape
    trials = 2 + int(np.log(n_clusters))
    if n_clusters >= 2 and _screened(n, d, trials):
        state = random_state.get_state()
        indices = _certified(data, host, n_clusters, trials, random_state, profile)
        if indices is not None:
            return indices
        random_state.set_state(state)
    return _reference_plusplus(host, n_clusters=n_clusters, random_state=random_state)[1]


def _host_prefix(values):
    return np.cumsum(values, dtype=np.float64) if _FLOAT64_PREFIX else np.cumsum(values)


def _scan_prefix(values):
    with mx.stream(mx.cpu):
        scanned = mx.array(values)
        if _FLOAT64_PREFIX:
            scanned = scanned.astype(mx.float64)
        scanned = mx.cumsum(scanned)
        mx.eval(scanned)
    return np.asarray(scanned)


@lru_cache(maxsize=1)
def _prefix_routine():
    """The faster routine for the reference's sequential prefix sums.

    NumPy's accumulation reloads every partial sum from memory. MLX's CPU scan
    performs the same additions in the same order at about twice the speed,
    which this check confirms once per process on sums that round at every
    step; any other order would show.
    """
    values = np.ldexp(np.random.default_rng(0).random(4096), np.arange(4096) % 40 - 20)
    values = values.astype(np.float32)
    same = np.array_equal(_scan_prefix(values), _host_prefix(values))
    return _scan_prefix if same else _host_prefix


def _certified(data, host, n_clusters, trials, random_state, profile):
    n = len(host)
    clock = time.perf_counter
    seconds = dict.fromkeys(_STAGES, 0.0)
    counts = dict.fromkeys(("pairs", "rows", "dense_steps", "reference_chunks", "second_prefix",
                            "projected_steps", "contests"), 0)
    if profile is not None:
        profile.update(trials=trials, certified=False, steps=0)
    mark = clock()
    reference = _Reference(host)
    length = reference.length(trials)
    # Float32 upper bounds of the squared norms, for the screen's error bound.
    upper = (reference.norms * (1 + 2.0**-20)).astype(np.float32)
    bounds = mx.array(upper)
    shifts = np.arange(trials, dtype=np.uint32)
    project = host.shape[1] >= _PROJECT_WIDTH
    projection = None
    resume, pause = _PROJECT_FROM, 8
    weights = np.ones(n, np.float32)
    column = weights.reshape(-1, 1)
    indices = np.full(n_clusters, -1, dtype=int)
    indices[0] = random_state.choice(n, p=weights / weights.sum())
    first = reference.every(host[indices[:1]])
    potential = first @ weights
    # Bounds of every row's squared distance to its nearest chosen center.
    # They are equal except on ``unsure`` rows.
    low = high = first[0]
    prefix_sums = _prefix_routine()
    unsure = np.empty(0, np.intp)
    listed = np.zeros(n, bool)
    seconds["prepare"] = clock() - mark
    for step in range(1, n_clusters):
        mark = clock()
        draws = random_state.uniform(size=trials) * potential
        prefix = prefix_sums(low)
        ids = np.searchsorted(prefix, draws)
        if len(unsure):
            # The prefix sums are sequential: they are the same for both ends
            # if every unsure row is absorbed where it is added.
            before = np.where(unsure > 0, prefix[unsure - 1], 0)
            if not np.array_equal(before + high[unsure], prefix[unsure]):
                counts["second_prefix"] += 1
                if not np.array_equal(ids, np.searchsorted(prefix_sums(high), draws)):
                    return None
        np.clip(ids, None, n - 1, out=ids)
        candidates = host[ids]
        after = clock()
        seconds["draw"] += after - mark
        mark = after
        chosen, floor, ceiling = candidate_ids(ids), mx.array(low), mx.array(high)
        if project and step >= resume and (projection is not None
                                           or n_clusters - step >= _PROJECT_STEPS):
            if projection is None:
                projection = _Projection(host, reference.norms)
                project = projection.valid
                after = clock()
                seconds["project"] += after - mark
                mark = after
        if project and projection is not None and step >= resume:
            kept, mask, touched, settled = projected_screen(
                data, projection.images, projection.eps, floor, ceiling, bounds, chosen, trials)
            mx.eval(kept, mask, touched, settled)
            counts["projected_steps"] += 1
            if int(np.asarray(settled).sum()) > _PROJECT_SHARE * trials * n:
                pause = min(2 * pause, 1024)
                resume = step + 1 + pause
            else:
                pause = 8
        else:
            kept, mask, touched = candidate_screen(data, floor, ceiling, bounds, chosen, trials)
            mx.eval(kept, mask, touched)
        after = clock()
        seconds["screen"] += after - mark
        mark = after
        rows = np.flatnonzero(np.asarray(touched).view(np.bool_))
        after = clock()
        seconds["collect"] += after - mark
        mark = after
        tables = np.asarray(kept)
        if not tables.flags.writeable:
            tables = np.array(tables)
        below, above = tables[0], tables[1]
        opened = np.asarray(mask)[rows]
        contenders = None
        if len(rows) > _CONTEST_ROWS:
            # The tables bracket every candidate's potential. A candidate
            # whose lower end exceeds another's upper end cannot win, and its
            # open pairs need no evaluation.
            contenders = (below @ column)[:, 0] <= (above @ column).min()
            if contenders.all():
                contenders = None
            else:
                opened = opened & np.uint32(np.left_shift(1, np.flatnonzero(contenders)).sum())
                rows, opened = rows[opened != 0], opened[opened != 0]
                counts["contests"] += 1
        which = columns = unsettled = None
        if 3 * len(rows) >= 2 * n:
            # Early centers move most rows: evaluate every pair as the reference does.
            exact = reference.every(candidates)
            below = np.minimum(low, exact)
            above = np.minimum(high, exact, out=exact)
            contenders = None
            counts["dense_steps"] += 1
            counts["rows"] += n
        elif len(rows):
            position, which = np.nonzero((opened[:, None] >> shifts) & 1)
            columns = rows[position]
            least, most = reference.bounds(candidates, rows, which, position)
            unsettled = least != most
            ambiguous = unsettled & (least > 0)
            if ambiguous.any():
                # The float32 result depends on the reference's last bits.
                for index in np.unique(columns[ambiguous] // length):
                    exact = reference.chunk(candidates, index, length)
                    inside = columns // length == index
                    least[inside] = exact[which[inside], columns[inside] - index * length]
                    most[inside] = least[inside]
                    counts["reference_chunks"] += 1
                unsettled = least != most
            where = which * n + columns
            below.reshape(-1)[where] = np.minimum(low[columns], least)
            above.reshape(-1)[where] = np.minimum(high[columns], most)
            counts["pairs"] += len(where)
            counts["rows"] += len(rows)
        after = clock()
        seconds["evaluate"] += after - mark
        mark = after
        potentials = below @ column
        members = slice(None) if contenders is None else np.flatnonzero(contenders)
        if (len(unsure) or (unsettled is not None and unsettled.any())) and not np.array_equal(
                potentials[members], (above @ column)[members]):
            return None
        best = np.argmin(potentials[members])
        if contenders is not None:
            best = members[best]
        potential = potentials[best]
        low, high = below[best], above[best]
        indices[step] = ids[best]
        if unsettled is not None:
            fresh = columns[unsettled & (which == best)]
            fresh = fresh[~listed[fresh]]
            if len(fresh):
                listed[fresh] = True
                unsure = np.concatenate([unsure, fresh])
        seconds["select"] += clock() - mark
        if profile is not None:
            profile["steps"] = step
    if profile is not None:
        profile.update(seconds, **counts, unsure_rows=len(unsure), certified=True)
    return indices
