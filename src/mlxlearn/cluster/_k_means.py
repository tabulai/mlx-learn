"""Dense FP32 Lloyd k-means on Metal, with bounded assignment/statistics tiles.

The resident interface keeps input preparation outside a timed optimization.
All centroid statistics and updates stay on the GPU. Lloyd shares the dense
assignment kernels with e-means and can certify previous labels against the
current centers before issuing a dense search.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from numbers import Integral, Real

import mlx.core as mx
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .._kernels.e_means import (
    _scratch_budget,
    _streaming_fp32_device,
    assign,
    batch_stats,
    selected_distances,
)
from .._kernels.lloyd_bounds import reusable_labels, separation_lower

_UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class PreparedKMeansFit:
    """Owned raw FP32 input and initializations, reusable for independent fits."""

    data: mx.array
    initial: tuple[mx.array, ...]
    offset: mx.array
    tolerance: float
    signature: tuple


@dataclass(frozen=True)
class DeviceKMeansFit:
    """Resident outputs from the best restart; labels are freshly reassigned."""

    centers: mx.array
    labels: mx.array
    inertia: mx.array
    n_iter: int
    history: tuple
    convergence_reason: str
    restart_index: int
    seconds: float
    restart_inertias: tuple[float, ...]

    @property
    def assignments(self):
        return self.labels


def _positive_integer(value, name, *, optional=False):
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _range_check(maximum, n, d):
    # Centering can double the coordinate magnitude; differences double it
    # again. Also bound full-iteration sums, inertia and variance reductions.
    largest = float(np.finfo(np.float32).max)
    bound = min(largest / (8 * n), math.sqrt(largest / (64 * n * d)))
    if not math.isfinite(maximum) or maximum > bound:
        raise ValueError("input or initial centroid magnitude is too large for safe float32 distances and accumulation; rescale the data")


def _matrix(X, *, name="X", own=False):
    if isinstance(X, mx.array):
        if X.ndim != 2 or min(X.shape) < 1:
            raise ValueError(f"{name} must be a nonempty two-dimensional dense matrix")
        if mx.issubdtype(X.dtype, mx.complexfloating):
            raise ValueError(f"{name} must contain real values")
        data = X.astype(mx.float32)
        low, high, finite = mx.min(data), mx.max(data), mx.all(mx.isfinite(data))
        mx.eval(low, high, finite)
        lower, upper = float(low.item()), float(high.item())
        if not bool(finite.item()):
            raise ValueError(f"{name} must contain finite values representable in float32")
        if own:
            # Own storage even when the supplied MLX array already has FP32
            # dtype. The bitwise copy preserves subnormal input coordinates.
            data = (data.view(mx.uint32) | mx.zeros((), mx.uint32)).view(mx.float32)
    else:
        array = np.asarray(X)
        if array.ndim != 2 or min(array.shape) < 1:
            raise ValueError(f"{name} must be a nonempty two-dimensional dense matrix")
        if np.iscomplexobj(array):
            raise ValueError(f"{name} must contain real values")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(array, dtype=np.float32, order="C")
        lower, upper = float(array.min()), float(array.max())
        data = mx.array(array)
    if data.shape[0] > np.iinfo(np.int32).max:
        raise ValueError(f"{name} has too many rows for 32-bit cluster/index arrays")
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"{name} must contain finite values representable in float32")
    return data, max(abs(lower), abs(upper))


def _relocate_empty(data, offset, labels, distances, sums, counts):
    """Move the farthest assigned rows into empty clusters without host copies.

    As in sklearn Lloyd, remove each row from its old cluster's sum/count and
    insert it into an empty cluster. Current labels are not changed. Farthest
    point ties and their mapping to empty IDs can differ from NumPy argpartition.
    """
    empty_count = int(mx.sum(counts == 0).item())
    if empty_count and float(mx.max(distances).item()) > 0:
        k = counts.shape[0]
        ids = mx.arange(k, dtype=mx.int32)
        empty = mx.argsort(mx.where(counts == 0, ids, ids + k))[:empty_count]
        far = mx.argpartition(-distances, kth=empty_count - 1)[:empty_count]
        # Order selected rows by distance; equal-distance choices are not a
        # promise of sklearn's version-dependent argpartition ordering.
        far = far[mx.argsort(-distances[far])]
        donors = labels[far]
        values = data[far] - offset
        removed_sums, removed_counts = batch_stats(values, donors, k)
        sums = (sums - removed_sums).at[empty].add(values)
        counts = (counts - removed_counts).at[empty].add(mx.ones((empty_count,), mx.float32))
    means = sums / mx.maximum(counts[:, None], 1)
    largest = mx.argmax(counts)
    # Relocation can empty a donor. With coincident data, relocation is skipped
    # and every empty center also receives the largest occupied cluster's mean.
    centers = mx.where((counts > 0)[:, None], means, means[largest])
    return centers, empty_count


def _pruned_tile(x, centers, old, radii, max_norm):
    keep, distances = reusable_labels(x, centers, old, radii[old], max_norm)
    uncertain = len(x) - int(mx.sum(keep).item())
    if uncertain == 0:
        return old, distances, len(x)
    # Only compact rows on shapes with the streaming kernel's fixed dot order.
    # Keep at least 128 rows in its call, including for a single ambiguous row.
    if x.shape[1] > 512 or max(uncertain, 128) >= len(x):
        labels, distances = assign(x, centers)
        return labels, distances, 0
    ids = mx.argsort(keep.astype(mx.int32))[:uncertain]
    work = x[ids]
    if uncertain < 128:
        work = mx.concatenate([work, mx.broadcast_to(work[:1], (128 - uncertain, x.shape[1]))])
    labels, updated = assign(work, centers)
    return (mx.put_along_axis(old, ids, labels[:uncertain], axis=0),
            mx.put_along_axis(distances, ids, updated[:uncertain], axis=0), len(x) - uncertain)


class KMeans(ClusterMixin, TransformerMixin, BaseEstimator):
    """Full-batch Lloyd clustering using FP32 Metal assignment and updates.

    Parameters follow sklearn KMeans where supported: ``init`` accepts
    ``'k-means++'``, ``'random'``, or an explicit center matrix; ``n_init='auto'``
    means one run for k-means++/explicit init and ten for random init. Explicit
    initialization always uses one run. Only ``algorithm='lloyd'`` is supported.
    ``copy_x=False`` is accepted but inputs are still preserved. Sample weights,
    sparse data and callable initializers are unsupported.

    ``assignment_batch_size`` bounds rows passed to an assignment tile;
    ``statistics_batch_size`` (default 65536) bounds each statistics tile.
    The complete input remains resident, but global-mean centering is applied
    in tiles, avoiding a second complete centered input. Numerical reductions
    and empty-cluster tie choices need not be bitwise identical to sklearn.

    The convergence threshold is ``tol * mean(feature_variances)``. Labels
    unchanged from the preceding iteration are checked before squared center
    shift. Even ``tol=0`` can stop on unchanged labels or exactly zero shift.
    Every restart always ends with fresh labels and direct residual inertia
    against the returned original-coordinate FP32 centers.
    """

    def __init__(self, n_clusters=8, *, init="k-means++", n_init="auto", max_iter=300,
                 tol=1e-4, verbose=False, random_state=None, copy_x=True, algorithm="lloyd",
                 assignment_batch_size=None, statistics_batch_size=65536):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.assignment_batch_size = assignment_batch_size
        self.statistics_batch_size = statistics_batch_size

    def _validate_parameters(self):
        _positive_integer(self.n_clusters, "n_clusters")
        _positive_integer(self.max_iter, "max_iter")
        _positive_integer(self.assignment_batch_size, "assignment_batch_size", optional=True)
        _positive_integer(self.statistics_batch_size, "statistics_batch_size")
        if self.n_init != "auto":
            _positive_integer(self.n_init, "n_init")
        if isinstance(self.tol, bool) or not isinstance(self.tol, Real) or not math.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be finite and nonnegative")
        if self.algorithm != "lloyd":
            raise ValueError("only algorithm='lloyd' is supported")
        if not isinstance(self.copy_x, (bool, np.bool_)):
            raise ValueError("copy_x must be a boolean")
        if isinstance(self.init, str) and self.init not in {"random", "k-means++"} or callable(self.init):
            raise ValueError("init must be 'random', 'k-means++', or an initial center matrix")

    def _configuration_signature(self):
        primitive = (str, int, float, bool, type(None))
        return tuple((key, value if isinstance(value, primitive) else id(value))
                     for key, value in sorted(self.get_params(deep=False).items()))

    def _tile_rows(self, n, d):
        return min(n, int(self.statistics_batch_size), 2**24, _UINT32_MAX // d)

    def prepare_device(self, X, *, sample_weight=None):
        """Validate and own input/init storage before resident optimization.

        This includes upload, initialization and the global mean and tolerance.
        K-means++ initialization uses sklearn's public CPU helper. Preparation
        can be reused by a model with the same configuration; non-scalar init
        and RandomState parameters must be the same objects. Input mutation
        after this call cannot change the prepared snapshots.
        """
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported by Metal KMeans")
        self._validate_parameters()
        if not mx.metal.is_available():
            raise RuntimeError("KMeans requires an available Apple Silicon Metal GPU")
        with mx.stream(mx.gpu):
            data, maximum = _matrix(X, own=True)
            n, d = data.shape
            k = int(self.n_clusters)
            if k > n:
                raise ValueError(f"n_samples={n} must be >= n_clusters={k}")
            if k * d > _UINT32_MAX:
                raise ValueError("centroid matrix exceeds 32-bit Metal indexing capacity")
            _range_check(maximum, n, d)
            offset = mx.mean(data, axis=0)
            rng = check_random_state(self.random_state)
            explicit = not isinstance(self.init, str)
            restarts = self.n_init
            if self.n_init == "auto":
                restarts = 10 if not explicit and self.init == "random" else 1
            if explicit and restarts != 1:
                warnings.warn("Explicit initial center position passed: performing only one init instead of n_init="
                              f"{restarts}.", RuntimeWarning, stacklevel=2)
                restarts = 1
            initials = []
            if explicit:
                centers, bound = _matrix(self.init, name="init", own=True)
                if centers.shape != (k, d):
                    raise ValueError(f"init must have shape {(k, d)}")
                _range_check(max(maximum, bound), n, d)
                initials.append(centers)
            else:
                # k-means++ is preparation, not part of timed resident Lloyd.
                # Explicit/random initialization never exports the full input.
                host = np.asarray(data) if self.init == "k-means++" else None
                for _ in range(int(restarts)):
                    if self.init == "random":
                        indices = rng.choice(n, size=k, replace=False)
                        centers = data[mx.array(indices, mx.int32)]
                    else:
                        values, _ = kmeans_plusplus(host, n_clusters=k, random_state=rng)
                        centers = mx.array(values, mx.float32)
                    initials.append(centers)
            tolerance = 0.0
            if self.tol:
                total = mx.array(0, mx.float32)
                rows = self._tile_rows(n, d)
                for start in range(0, n, rows):
                    centered = data[start:start + rows] - offset
                    total = total + mx.sum(centered * centered)
                    mx.eval(total)
                tolerance = float(total.item()) / (n * d) * float(self.tol)
            mx.eval(data, offset, *initials)
            return PreparedKMeansFit(data, tuple(initials), offset, tolerance, self._configuration_signature())

    def _exclusion_radii(self, data, centers, offset, previous):
        # The full radii cost k² center pairs. First try a small deterministic
        # sample of point/center pairs; dense embeddings often fail this test.
        n, d = data.shape
        k = len(centers)
        if (previous is None or k < 4096 or n < 8 * k or not 64 <= d <= 1024
                or self.assignment_batch_size is not None or not _streaming_fp32_device()):
            return None
        sample = mx.arange(128) * (n // 128)
        ids = previous[sample]
        norms = mx.sum(centers * centers, axis=1)
        max_norm = mx.max(norms)
        sample_radii = separation_lower(centers, ids, norms, max_norm)
        keep, _ = reusable_labels(data[sample] - offset, centers, ids, sample_radii, max_norm)
        if int(mx.sum(keep).item()) < 96:
            return None
        return separation_lower(centers, mx.arange(k), norms, max_norm), max_norm

    def _assignment(self, data, centered_centers, offset, *, original_centers=None, previous=None):
        labels, distances = [], []
        n, d = data.shape
        rows = self._tile_rows(n, d)
        bounds = self._exclusion_radii(data, centered_centers, offset, previous)
        pruned = 0
        for start in range(0, n, rows):
            block = data[start:start + rows]
            tile = len(block) if bounds is None else max(1, _scratch_budget() // (4 * len(centered_centers)))
            for i in range(0, len(block), tile):
                points = block[i:i + tile]
                centered = points - offset
                if bounds is None:
                    indices, squared = assign(centered, centered_centers, batch_size=self.assignment_batch_size)
                else:
                    old = previous[start + i:start + i + len(points)]
                    indices, squared, saved = _pruned_tile(centered, centered_centers, old, *bounds)
                    pruned += saved
                if original_centers is not None:
                    squared = selected_distances(points, original_centers, indices)
                mx.eval(indices, squared)
                labels.append(indices)
                distances.append(squared)
        return mx.concatenate(labels), mx.concatenate(distances), pruned

    def _statistics(self, data, labels, offset):
        n, d = data.shape
        rows = self._tile_rows(n, d)
        sums = mx.zeros((self.n_clusters, d), mx.float32)
        counts = mx.zeros((self.n_clusters,), mx.float32)
        for start in range(0, n, rows):
            batch_sums, batch_counts = batch_stats(data[start:start + rows] - offset, labels[start:start + rows], self.n_clusters)
            sums, counts = sums + batch_sums, counts + batch_counts
            # Bound both the graph and live scatter buffers independently of n.
            mx.eval(sums, counts)
        return sums, counts

    def train_device(self, prepared):
        """Run all restarts, returning synchronized resident best-fit outputs.

        No upload, full validation, initialization or NumPy output export occurs
        here. This method records ``fit_result_`` but does not populate the
        sklearn host fitted attributes; use ``fit`` for that boundary.
        """
        if not isinstance(prepared, PreparedKMeansFit) or prepared.signature != self._configuration_signature():
            raise ValueError("prepared inputs do not match this estimator configuration")
        started = time.perf_counter()
        best = None
        restart_inertias = []
        with mx.stream(mx.gpu):
            data, offset = prepared.data, prepared.offset
            for restart, initial in enumerate(prepared.initial):
                centers = initial - offset
                previous = None
                history = []
                reason = "max_iter"
                for iteration in range(int(self.max_iter)):
                    iteration_start = time.perf_counter()
                    labels, distances, pruned = self._assignment(data, centers, offset, previous=previous)
                    sums, counts = self._statistics(data, labels, offset)
                    updated, empty_count = _relocate_empty(data, offset, labels, distances, sums, counts)
                    shift = mx.sum((updated - centers) ** 2)
                    unchanged = mx.array(False) if previous is None else mx.all(labels == previous)
                    objective = mx.sum(distances)
                    mx.eval(updated, shift, unchanged, objective)
                    shift_value = float(shift.item())
                    history.append({"iteration": iteration + 1, "rows": data.shape[0], "inertia_before_update": float(objective.item()),
                                    "center_shift_squared": shift_value, "empty_clusters": empty_count,
                                    "pruned_rows": pruned,
                                    "seconds": time.perf_counter() - iteration_start})
                    centers = updated
                    if self.verbose:
                        print(f"KMeans restart {restart + 1}, iteration {iteration + 1}: "
                              f"inertia={history[-1]['inertia_before_update']:.8g}, shift={shift_value:.8g}", flush=True)
                    if bool(unchanged.item()):
                        reason = "labels"
                        break
                    if shift_value <= prepared.tolerance:
                        reason = "tolerance"
                        break
                    previous = labels
                restored = centers + offset
                # Fresh assignment also accounts for FP32 restoration rounding.
                labels, distances, _ = self._assignment(data, restored - offset, offset,
                                                        original_centers=restored, previous=labels)
                labels = labels.astype(mx.int64)
                inertia = mx.sum(distances)
                mx.eval(restored, labels, inertia)
                value = float(inertia.item())
                restart_inertias.append(value)
                if best is None or value < best[0]:
                    best = (value, restored, labels, inertia, iteration + 1, tuple(history), reason, restart)
            mx.synchronize()
        _, centers, labels, inertia, iterations, history, reason, restart = best
        result = DeviceKMeansFit(centers, labels, inertia, iterations, history, reason, restart,
                                time.perf_counter() - started, tuple(restart_inertias))
        self.fit_result_ = result
        return result

    def fit(self, X, y=None, sample_weight=None):
        """Fit and expose NumPy centers, labels and inertia, preserving inputs."""
        started = time.perf_counter()
        prepared = self.prepare_device(X, sample_weight=sample_weight)
        result = self.train_device(prepared)
        self.cluster_centers_ = np.array(result.centers)
        self.labels_ = np.array(result.labels, dtype=np.int64)
        self.inertia_ = float(result.inertia.item())
        self.n_iter_ = result.n_iter
        self.n_features_in_ = prepared.data.shape[1]
        self._n_features_out = self.n_clusters
        self.offset_ = np.array(prepared.offset)
        self._offset_mx = prepared.offset
        self.iteration_stats_ = list(result.history)
        self.convergence_reason_ = result.convergence_reason
        self.n_init_ = len(prepared.initial)
        self.best_restart_ = result.restart_index
        self.execution_backend_ = "metal"
        self.optimization_time_ = result.seconds
        self.fit_time_ = time.perf_counter() - started
        distinct = np.unique(self.labels_).size
        if distinct < self.n_clusters:
            warnings.warn(f"Number of distinct clusters ({distinct}) found smaller than n_clusters ({self.n_clusters}). "
                          "Possibly due to duplicate points in X.", ConvergenceWarning, stacklevel=2)
        return self

    def _inference(self, X):
        check_is_fitted(self, "cluster_centers_")
        data, maximum = _matrix(X)
        if data.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {data.shape[1]} features, expected {self.n_features_in_}")
        centers, center_maximum = _matrix(self.cluster_centers_, name="cluster_centers_")
        origin = max(abs(float(self.offset_.min())), abs(float(self.offset_.max())))
        _range_check(max(maximum, center_maximum, origin), data.shape[0], data.shape[1])
        return data, centers

    def predict(self, X):
        """Return the nearest final center index for every input row."""
        with mx.stream(mx.gpu):
            data, centers = self._inference(X)
            labels, _, _ = self._assignment(data, centers - self._offset_mx, self._offset_mx)
            return np.array(labels, dtype=np.int64)

    def transform(self, X):
        """Return Euclidean distances to every center using bounded direct tiles.

        The returned dense n_samples by n_clusters NumPy matrix is necessarily
        unbounded in total size; temporary feature differences are bounded.
        """
        with mx.stream(mx.gpu):
            data, centers = self._inference(X)
            n, d = data.shape
            k = centers.shape[0]
            result = np.empty((n, k), dtype=np.float32)
            cols = min(k, 256)
            rows = max(1, min(n, (16 << 20) // (4 * d * cols)))
            for start in range(0, n, rows):
                for column in range(0, k, cols):
                    delta = data[start:start + rows, None, :] - centers[None, column:column + cols, :]
                    values = mx.sqrt(mx.sum(delta * delta, axis=2))
                    result[start:start + rows, column:column + cols] = np.array(values)
            return result

    def score(self, X, y=None, sample_weight=None):
        """Return negative fresh-assignment inertia; sample weights unsupported."""
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported by Metal KMeans")
        with mx.stream(mx.gpu):
            data, centers = self._inference(X)
            _, distances, _ = self._assignment(data, centers - self._offset_mx, self._offset_mx,
                                               original_centers=centers)
            return -float(mx.sum(distances).item())

    def fit_predict(self, X, y=None, sample_weight=None):
        """Fit and return final fresh nearest-center labels."""
        return self.fit(X, y, sample_weight=sample_weight).labels_
