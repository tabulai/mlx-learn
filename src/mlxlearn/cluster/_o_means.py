"""o-means on Apple Silicon: e-means with a two-level assignment.

o-means keeps e-means' objective, starting centroids, mini-batch EMA update,
batch and step laws, cosine schedule and split revival. During training each
row is compared only with the centroids of a few nearby coarse cells instead
of with every centroid, and fits shorter than two epochs end with a
recency-weighted mean. The algorithm follows ``contrib/o_means`` of
owlgebra-ai/faiss (MIT), accessed 2026-11-18; the Metal implementation is
separate from its CUDA kernels.
"""

from __future__ import annotations

import time
from functools import lru_cache
from numbers import Integral

import mlx.core as mx
import numpy as np

from .._kernels.e_means import assign, batch_stats, ema_update, gather_bf16, selected_distances
from .._kernels.o_means import (
    MAX_PROBES,
    build_cells,
    record_splits,
    sorted_mirror,
    tiled_scan_available,
    two_level_assign,
)
from ._e_means import DeviceEMeansFit, EMeans, PreparedEMeansFit, revive, split_plan
from ._o_schedule import closing_weight, expected_fraction, resolve_schedule
from ._recipe import scheduled_alpha

# Largest centroid count of the two-level search: its cell sort stays inside
# MLX's verified sort widths, and no larger table has been evaluated.
MAX_TWO_LEVEL_CLUSTERS = 2**16
# ``auto`` searches cells only while that at least halves the comparisons:
# the tiled scan evaluates a distance at roughly 70% of the dense rate and
# adds probing and bucketing, so it stops paying somewhat above one half.
_AUTO_MAX_FRACTION = 0.5
# GPUs on which whole two-level fits were measured faster than dense ones
# (MLX 0.32.4, 2026-11-18). Other GPUs keep the dense search under ``auto``.
_VALIDATED_DEVICES = frozenset({"Apple M1 Ultra", "Apple M4 Max", "Apple M5 Ultra"})


@lru_cache(maxsize=1)
def _two_level_device():
    """Whether the fast two-level scan is validated on this GPU."""
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return info.get("device_name") in _VALIDATED_DEVICES and tiled_scan_available()


def split_closing(close_sums, close_counts, plan):
    """Give a donor and its new child half of the donor's closing mass each."""
    is_child, is_donor, source = plan[:3]
    sums = mx.where(is_child[:, None], close_sums[source] * 0.5,
                    mx.where(is_donor[:, None], close_sums * 0.5, close_sums))
    counts = mx.where(is_child, close_counts[source] * 0.5,
                      mx.where(is_donor, close_counts * 0.5, close_counts))
    return sums, counts


class OMeans(EMeans):
    """e-means with a two-level nearest-centroid search on the Metal GPU.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of centroids.
    max_iter : float, default=3
        Epoch budget, including fractional epochs. Below one epoch the batch
        and step sizes of a one-epoch fit are used.
    batch_size : int or None
        Statistical mini-batch size. None uses the fitted batch-size law.
    alpha0 : float or None
        Initial EMA step in (0,1]. None uses the fitted step-size law.
    batch_law : {'auto', 'throughput', 'quality'}, default='auto'
        ``auto`` selects the throughput law up to two million rows and the
        quality law above.
    n_cells : int or None
        Coarse cells over the centroids. None uses ``n_clusters // 128``
        rounded up to a multiple of 16, at least 16.
    n_probe : int or None
        Cells each row searches, at most 128. None uses 16, 12 from 512
        features, and 24 above 16,384 centroids.
    random_state : int, RandomState or None, default=1234
        Initialization, epoch shuffling, donor sampling and cell seed.
    init : {'random'} or array of shape (n_clusters, n_features)
        Initial centroids. Random initialization samples without replacement.
    unused_threshold : float or None
        Revival threshold as a fraction of mean mass; 0 disables revival.
    assignment : {'auto', 'two_level', 'dense'}, default='auto'
        Search used while training. ``two_level`` searches the nearest cells
        only and is approximate. ``dense`` compares every row with every
        centroid, as e-means does. ``auto`` selects ``two_level`` where it is
        validated to be faster and ``dense`` elsewhere. It also returns to
        ``dense`` for the rest of a fit whose cells turn out so uneven that
        rows are compared with more than half of the centroids.
    precision : {'float32', 'bfloat16'}, default='float32'
        Assignment arithmetic. EMA state and accumulated sums remain float32.
    assignment_batch_size : int or None
        Maximum rows in a dense assignment tile.
    compute_labels : bool, default=True
        Reassign all input rows after training with a full search.
    verbose : bool, default=False
        Print one progress line per pass.

    Notes
    -----
    The recipe is the Faiss e-means recipe. Fits shorter than two epochs end
    with a mean over the whole run in which each batch is weighted by
    ``exp(-(rows processed afterwards) / n_samples)``; longer fits end with
    the mean of the last pass, as e-means does. ``recipe_`` records the
    resolved cells, probes, rebuild cadence and closing mean.

    Labels captured during training come from the approximate search against
    moving centroids. ``labels_``, ``predict`` and ``score`` always use a full
    search. Batch sums use float32 atomics, so repeated fits can differ in
    the last bits. This estimator requires MLX's Metal GPU.
    """

    recipe = "faiss"
    finalization = "last_epoch"
    slow_ema_half_life = None
    slow_ema_multiplier = 1.0
    spherical = False

    def __init__(
        self,
        n_clusters=8,
        *,
        max_iter=3,
        batch_size=None,
        alpha0=None,
        batch_law="auto",
        n_cells=None,
        n_probe=None,
        random_state=1234,
        init="random",
        unused_threshold=None,
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
        self.batch_law = batch_law
        self.n_cells = n_cells
        self.n_probe = n_probe
        self.random_state = random_state
        self.init = init
        self.unused_threshold = unused_threshold
        self.assignment = assignment
        self.precision = precision
        self.assignment_batch_size = assignment_batch_size
        self.compute_labels = compute_labels
        self.verbose = verbose

    def _validate_parameters(self):
        if self.assignment not in {"auto", "two_level", "dense"}:
            raise ValueError("assignment must be 'auto', 'two_level', or 'dense'")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be 'float32' or 'bfloat16'")
        if self.assignment_batch_size is not None and (
            isinstance(self.assignment_batch_size, bool)
            or not isinstance(self.assignment_batch_size, Integral)
            or self.assignment_batch_size < 1
        ):
            raise ValueError("assignment_batch_size must be a positive integer or None")

    def _assign(self, data, centers):
        """Full search over every centroid, used outside the training search."""
        if self.precision != "bfloat16":
            data, centers = data - self._offset_mx, centers - self._offset_mx
        return assign(data, centers, batch_size=self.assignment_batch_size,
                      method="auto", precision=self.precision)

    def _resolve_recipe(self, n, d):
        recipe = resolve_schedule(
            n, self.n_clusters, d, self.max_iter, self.batch_size, self.alpha0,
            self.batch_law, self.n_cells, self.n_probe, self.unused_threshold,
        )
        k, cells, probes = int(self.n_clusters), recipe["n_cells"], recipe["n_probe"]
        supported = (k <= MAX_TWO_LEVEL_CLUSTERS and probes <= MAX_PROBES
                     and cells * k <= 2**31 - 1)
        if self.assignment == "two_level":
            if not supported:
                raise ValueError(
                    f"two-level assignment supports at most {MAX_TWO_LEVEL_CLUSTERS} centroids "
                    f"and {MAX_PROBES} probes; use assignment='dense'")
            search, reason = "two_level", "requested"
        elif self.assignment == "dense":
            search, reason = "dense", "requested"
        elif not supported:
            search, reason = "dense", "outside the two-level limits"
        elif expected_fraction(k, cells, probes) > _AUTO_MAX_FRACTION:
            search, reason = "dense", "probes cover most centroids"
        elif not _two_level_device():
            search, reason = "dense", "two-level scan not validated on this GPU"
        else:
            search, reason = "two_level", "validated"
        recipe.update(search=search, search_reason=reason,
                      expected_fraction=expected_fraction(k, cells, probes))
        return recipe

    def train_device(self, prepared, *, capture_assignments=False):
        """Run a complete fit from prepared inputs, returning resident outputs.

        Reusing ``prepared`` restarts from the same initial centers and seed.
        Nothing is read back from the GPU between batches. Optional labels
        are captured during the final pass with the training search; rows it
        did not visit have label -1.
        """
        if not isinstance(prepared, PreparedEMeansFit) or prepared.signature != self._configuration_signature():
            raise ValueError("prepared inputs do not match this estimator configuration")
        started = time.perf_counter()
        with mx.stream(mx.gpu):
            data, centers = prepared.data, prepared.initial
            n, d = data.shape
            k = int(self.n_clusters)
            recipe = self.recipe_ = dict(prepared.recipe)
            self._offset_mx = prepared.offset
            bf16 = self.precision == "bfloat16"
            two_level = recipe["search"] == "two_level"
            # ``auto`` verifies the saving it expected: after the first batch,
            # after the first revival check, and at every pass boundary.
            guarded = two_level and self.assignment == "auto"
            recipe["search_fallback_step"] = None
            recency = recipe["closing"] == "recency_weighted"
            cell_count, n_probe = recipe["n_cells"], recipe["n_probe"]
            batch_size = recipe["batch_size"]
            mass = mx.full((k,), recipe["init_usage"], dtype=mx.float32)
            split_rng = np.random.RandomState(prepared.seed)
            cell_rng = np.random.RandomState((prepared.seed * 7919 + 17) % 2**32)
            total_splits = mx.array(0, dtype=mx.int32)
            close_sums, close_counts = mx.zeros((k, d), mx.float32), mx.zeros((k,), mx.float32)
            child = mx.full((k,), -1, dtype=mx.int32)
            cells, since_build, builds = None, 0, 0
            history, final_labels = [], []
            step = rows_done = 0
            last_epoch = len(recipe["epoch_rows"]) - 1
            for epoch, rows in enumerate(recipe["epoch_rows"]):
                epoch_start = time.perf_counter()
                order_seed = (prepared.seed * 1000003 + epoch) % 2**32
                order = mx.array(np.random.RandomState(order_seed).permutation(n), dtype=mx.int32)
                objectives, evaluations = [], []
                # Comparisons of the searches alone; cell builds are amortized.
                scans, scanned_rows = [], 0
                searched = "two_level" if two_level else "dense"
                for start in range(0, rows, batch_size):
                    batch_order = order[start:min(start + batch_size, rows)]
                    count = batch_order.shape[0]
                    batch = gather_bf16(data, batch_order) if bf16 else data[batch_order]
                    pending = []
                    if two_level:
                        # Most splits happen during the first pass: re-file the
                        # centroids after every revival check there, so that new
                        # children land in their own cells.
                        first_pass = rows_done < n
                        every = recipe["check_unused_every"] if first_pass else recipe["rebuild_every"]
                        if cells is None or since_build >= every:
                            iterations = (recipe["first_build_iterations"] if cells is None else
                                          recipe["first_pass_rebuild_iterations"] if first_pass else
                                          recipe["rebuild_iterations"])
                            search_centers = centers if bf16 else centers - self._offset_mx
                            cells = build_cells(search_centers, cell_count, iterations,
                                                previous=cells, rng=cell_rng)
                            child = mx.full((k,), -1, dtype=mx.int32)
                            evaluations.append(mx.array(float(k * cell_count * (iterations + 1))))
                            since_build, builds = 0, builds + 1
                            pending.extend(cells.arrays())
                        since_build += 1
                        mirror, norms = sorted_mirror(
                            centers, cells, offset=None if bf16 else self._offset_mx,
                            precision=self.precision)
                        search_batch = batch if bf16 else batch - self._offset_mx
                        labels, _, evaluated = two_level_assign(
                            search_batch, mirror, norms, cells, n_probe, child)
                        distances = selected_distances(
                            batch, centers.astype(mx.bfloat16) if bf16 else centers, labels)
                        evaluations.append(evaluated)
                        scans.append(evaluated)
                        scanned_rows += count
                    else:
                        labels, distances = self._assign(batch, centers)
                        evaluations.append(mx.array(float(count * k)))
                    rows_done += count
                    sums, counts = batch_stats(batch, labels, k)
                    objectives.append(mx.sum(distances))
                    omega = closing_weight(recipe, n, epoch, rows_done)
                    if omega is not None:
                        close_sums = close_sums + (sums if omega == 1.0 else omega * sums)
                        close_counts = close_counts + (counts if omega == 1.0 else omega * counts)
                    if capture_assignments and epoch == last_epoch:
                        final_labels.append(labels)
                    alpha = scheduled_alpha(recipe["alpha0"], step, recipe["total_steps"])
                    previous_mass = mass
                    if bf16:
                        centers, mass = ema_update(centers, mass, sums, counts, alpha, decay=1.0 - alpha)
                    else:
                        centers, mass = ema_update(centers, mass, sums, counts, alpha)
                    if alpha == 1.0:
                        mass = mx.where(counts > 0, mass, previous_mass)
                    if (step + 1) % recipe["check_unused_every"] == 0 and recipe["unused_threshold"] > 0:
                        uniform = mx.array(split_rng.random_sample(k), dtype=mx.float32)
                        plan = split_plan(mass, uniform, recipe["unused_threshold"])
                        centers, mass, nsplit, _ = revive(
                            centers, mass, uniform, recipe["unused_threshold"],
                            recipe["split_epsilon"], precision=self.precision, plan=plan)
                        total_splits = total_splits + nsplit
                        if two_level:
                            child = record_splits(child, plan)
                        if recency:
                            close_sums, close_counts = split_closing(close_sums, close_counts, plan)
                    pending.extend((centers, mass, objectives[-1], evaluations[-1], total_splits,
                                    close_sums, close_counts, child))
                    if capture_assignments and epoch == last_epoch:
                        pending.append(labels)
                    mx.async_eval(*pending)
                    step += 1
                    if guarded and two_level and step in (1, recipe["check_unused_every"]):
                        if float(mx.sum(mx.stack(scans)).item()) > _AUTO_MAX_FRACTION * scanned_rows * k:
                            two_level, recipe["search_fallback_step"] = False, step
                objective = float(mx.sum(mx.stack(objectives)).item()) if objectives else 0.0
                evaluated = float(np.asarray(mx.stack(evaluations)).sum(dtype=np.float64)) if evaluations else 0.0
                mx.eval(centers, mass, total_splits)
                stat = {"epoch": epoch + 1, "rows": rows, "steps": step,
                        "moving_objective": objective, "moving_mse": objective / rows if rows else 0.,
                        "seconds": time.perf_counter() - epoch_start,
                        "cumulative_seconds": time.perf_counter() - started,
                        "splits": int(total_splits.item()),
                        "evaluated_distances": evaluated,
                        "evaluated_fraction": evaluated / (rows * k) if rows else 0.,
                        "cell_builds": builds, "search": searched}
                history.append(stat)
                if guarded and two_level and scans and (
                        float(mx.sum(mx.stack(scans)).item()) > _AUTO_MAX_FRACTION * scanned_rows * k):
                    two_level, recipe["search_fallback_step"] = False, step
                if self.verbose:
                    print(f"o-means pass {epoch + 1}: {rows} rows, moving MSE={stat['moving_mse']:.7g}, "
                          f"{stat['evaluated_fraction']:.3f} of dense, {stat['seconds']:.3f}s", flush=True)
            # A centroid with no closing mass, or a negligible weighted mass,
            # keeps its EMA position.
            closed = close_counts > (1e-3 if recency else 0)
            closing = mx.where(closed[:, None],
                               close_sums / mx.where(closed, close_counts, 1)[:, None], centers)
            captured = None
            if capture_assignments:
                moving_labels = mx.concatenate(final_labels).astype(mx.int32)
                captured = mx.full((n,), -1, dtype=mx.int32)
                captured[order[:len(moving_labels)]] = moving_labels
                captured = captured.astype(mx.int64)
            pending = [closing, centers, total_splits]
            if captured is not None:
                pending.append(captured)
            mx.eval(*pending)
            return DeviceEMeansFit(closing, centers, closing, None, captured,
                                   tuple(history), total_splits, step, len(history),
                                   time.perf_counter() - started)
