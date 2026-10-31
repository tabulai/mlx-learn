"""MLX-resident Euclidean search and neighbor aggregation, without NumPy.

Small-k GPU searches fuse direct distances and selection. The general path
uses exhaustive float64 differences on CPU and certified Gram screening on GPU.
Ambiguous GPU comparisons use exact original-coordinate ordering. A small sample of owned training
storage supplies cached dispatch metadata; all search results stay in MLX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import isfinite
from numbers import Integral
from operator import itemgetter

import mlx.core as mx

__all__ = ["DeviceNeighborIndex", "fit_neighbors"]


def _use_fused_search(n_features, k):
    # Direct scanning wins for modest feature/neighbor counts; register-heavy
    # larger lists use the GEMM path. See the checked-in before/after panel.
    return n_features <= 32 and k <= 16


def _use_small_search(n_train, n_query, n_features, k):
    """Amortize a single GPU launch where a multi-kernel screen is overhead."""
    return k <= 8 and (
        (n_train <= 2048 and n_query <= 256 and n_features <= 64)
        or (n_train <= 16384 and n_query <= 4 and n_features <= 32)
    )


def _use_compact_search(n_train, n_query, n_features, k):
    # Amortize compact training storage over a substantial batch. Small shapes
    # keep the direct scan; explicit reference tiling keeps the general path.
    return (16384 <= n_train <= 2**21 and n_query >= 64
            and 16 <= n_features <= 32 and n_features % 4 == 0 and k <= 8)


def _use_integer_search(n_train, n_query, n_features, k):
    # Integer matrix screening amortizes its preparation on larger batches.
    # Older Metal runtimes retain the scalar compact implementation.
    if not (32768 <= n_train <= 2**21 and n_query >= 64
            and n_features == 32 and k <= 8):
        return False
    from ._neighbors_integer import is_supported

    return is_supported()


def _prefer_integer_training(train):
    """Keep concentrated feature variation on the more precise compact filter.

    Integer quantization uses one scale per row. With energy in only a few
    features, its conservative bounds frequently require an expensive complete
    scan even when compact bounds certify the answer. Estimate the effective
    number of varying features from differences between pairs of 16 evenly
    spaced owned rows. Differences avoid sensitivity to a common offset. This
    is a performance heuristic, never a numerical certificate or a change to
    the metric: either route retains exact refinement and its complete fallback.

    Read MLX's buffer interface without a NumPy dependency or an explicit full
    training copy. Exporting a lazy buffer may evaluate it. Python double
    arithmetic avoids FP32 overflow/underflow in the diagnostic. Only this small
    sample is read on the host, once per eligible fitted index; query results
    are never cached. Unsupported buffer exports conservatively retain compact.
    """
    return _training_metadata(train)[0]


@lru_cache(maxsize=128)
def _training_sample_plan(n, features):
    """Cache immutable shape-derived indices, never training or query values."""
    count, width = min(16, n), min(8, features)
    rows = tuple(row * (n - 1) // max(count - 1, 1) for row in range(count))
    offsets = tuple(row * features for row in rows)
    columns = tuple(i * (features - 1) // max(width - 1, 1) for i in range(width))
    return rows, offsets, columns, itemgetter(*columns) if columns else None


def _training_metadata(train):
    """Share the owned-row sample between integer preference and geometry.

    Production calls this only for eligible 32-feature integer shapes. Reading
    sixteen short contiguous rows through the buffer's native conversion avoids
    both duplicate sampling and a Python buffer lookup for each coordinate.
    Only those sampled rows are copied; there is no full training copy or NumPy
    dependency. Strided buffers retain their logical indexing semantics.
    """
    n, features = train.shape
    count = min(16, n)
    if count < 2 or features == 0:
        return False, "ordinary"
    rows, offsets, columns, _ = _training_sample_plan(n, features)
    try:
        with memoryview(train) as values:
            if values.c_contiguous and values.format == "f" and values.itemsize == 4:
                with values.cast("B").cast("f") as flat:
                    sample = [flat[offset:offset + features].tolist() for offset in offsets]
            else:
                sample = [[values[row, feature] for feature in range(features)] for row in rows]
    except (TypeError, BufferError, NotImplementedError):
        return False, "ordinary"
    energies = [0.0] * features
    pairs = count // 2
    for pair in range(pairs):
        left, right = sample[2 * pair], sample[2 * pair + 1]
        for feature in range(features):
            delta = left[feature] - right[feature]
            energies[feature] += delta * delta
    total = sum(energies)
    # Participation ratio (sum E)^2 / sum(E^2): equal energies give D,
    # whereas a single dominant feature gives 1. Zero columns are harmless.
    preferred = (isfinite(total) and total > 0.0
                 and total * total >= 16.0 * sum(energy * energy for energy in energies))
    return preferred, _geometry_from_sample(sample, energies, pairs, columns)


def _sample_geometry(train):
    """Choose a screening route from a small sample of owned training storage.

    A shared offset makes raw integer codes unnecessarily coarse. Tight groups
    separated by much larger gaps can likewise defeat a small Gram or quantized
    shortlist. Detect these shapes from sixteen rows and at most eight features;
    no query values or results are retained. This is only a performance hint:
    every chosen route keeps its original-coordinate certificate and fallback.
    """
    n, features = train.shape
    count, width = min(16, n), min(8, features)
    if count < 2 or width == 0:
        return "ordinary"
    rows, offsets, columns, select = _training_sample_plan(n, features)
    try:
        with memoryview(train) as values:
            if values.c_contiguous and values.format == "f" and values.itemsize == 4:
                with values.cast("B").cast("f") as flat:
                    if width == 1:
                        sample = [(flat[offset],) for offset in offsets]
                    else:
                        sample = [select(flat[offset:offset + features]) for offset in offsets]
            else:
                sample = [[values[row, feature] for feature in columns] for row in rows]
    except (TypeError, BufferError, NotImplementedError):
        return "ordinary"
    energies = [0.0] * width
    pairs = count // 2
    for pair in range(pairs):
        left, right = sample[2 * pair], sample[2 * pair + 1]
        for feature in range(width):
            delta = left[feature] - right[feature]
            energies[feature] += delta * delta
    return _geometry_from_sample(sample, energies, pairs, range(width))


def _geometry_from_sample(sample, energies, pairs, columns):
    energies = [energies[feature] for feature in columns]
    width = len(columns)
    total = sum(energies)
    variation = total / pairs
    if not isfinite(variation) or variation <= 0.0:
        return "ordinary"
    magnitude = sum(sample[0][feature] * sample[0][feature] for feature in columns)
    if isfinite(magnitude) and magnitude > 64.0 * variation:
        return "offset"
    # A few dominant coordinates also create close sorted neighbors. Retain
    # the existing compact route for that geometry instead of calling it a
    # tight group in an otherwise broadly varying feature space.
    if total * total < (width / 2) * sum(energy * energy for energy in energies):
        return "ordinary"
    # Sorting one varying coordinate brings repeated tight groups together.
    # Most ordinary pairs exceed the limit after just one or two coordinates.
    dominant = columns[max(range(width), key=energies.__getitem__)]
    sample.sort(key=lambda row: row[dominant])
    limit = variation / 64.0
    for left, right in zip(sample, sample[1:], strict=False):
        distance = 0.0
        for feature in columns:
            delta = left[feature] - right[feature]
            distance += delta * delta
            if distance > limit:
                break
        else:
            return "clustered"
    return "ordinary"


def _matrix(X, *, n_features=None, allow_empty=False):
    if not isinstance(X, mx.array):
        raise TypeError("X must be an MLX array")
    if X.dtype != mx.float32:
        raise TypeError("X must have float32 dtype")
    if X.ndim != 2 or X.shape[1] == 0 or (not allow_empty and X.shape[0] == 0):
        raise ValueError("X must be a nonempty two-dimensional matrix")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("X must have the fitted feature count")


def _order_keys(values, indices, excluded=None):
    keys = (values.view(mx.uint32).astype(mx.uint64) << 32) | indices.astype(mx.uint64)
    if excluded is not None:
        # A masked identity must sort after *all* real candidates, including
        # distances that overflow to infinity. Infinity alone is not a mask.
        keys = mx.where(excluded, ~mx.array(0, mx.uint64), keys)
    return keys


@mx.compile
def _ordered_take(values, indices, k, excluded=None):
    """Select using exact float32 distance bits followed by the global index.

    Distances are nonnegative, so their IEEE bit pattern has numeric order.
    A uint64 key avoids adding an epsilon that can change genuinely unequal
    distances. MLX supports this sort on both CPU and Metal.
    """
    keys = _order_keys(values, indices, excluded)
    order = mx.argpartition(keys, kth=k - 1, axis=1)[:, :k]
    return mx.take_along_axis(values, order, axis=1), mx.take_along_axis(indices, order, axis=1)


@mx.compile
def _ordered_sort(values, indices, k, excluded=None):
    keys = _order_keys(values, indices, excluded)
    order = mx.argsort(keys, axis=1)[:, :k]
    return mx.take_along_axis(values, order, axis=1), mx.take_along_axis(indices, order, axis=1)


@mx.compile
def _prepare_gram_arrays(train):
    # Anchoring and quarter scaling avoid overflow in the mean and preserve
    # constant columns. These arrays depend only on the owned training data.
    anchor = train[0] * 0.25
    center = (anchor + mx.mean(train * 0.25 - anchor, axis=0)) * 4.0
    centered = train - center
    return center, centered, mx.sum(centered * centered, axis=1)


@dataclass(slots=True)
class DeviceNeighborIndex:
    """Owned float32 training data with reusable device preparation.

    Create with :func:`fit_neighbors`. ``query`` returns ``(distances, indices)``
    as MLX float32/int32 arrays, or ``(None, indices)`` when distances are omitted.
    Explicit ``device='cpu'`` ranks direct float64 differences with MLX CPU
    operations. The general GPU path certifies candidate selection and scans
    every original reference when its Gram screen is uncertain.
    """

    X: mx.array
    device: str = "gpu"
    _center: object = field(default=None, init=False, repr=False)
    _train: object = field(default=None, init=False, repr=False)
    _train_sq: object = field(default=None, init=False, repr=False)
    _compact: object = field(default=None, init=False, repr=False)
    _integer: object = field(default=None, init=False, repr=False)
    _integer_preferred: bool | None = field(default=None, init=False, repr=False)
    _geometry: str | None = field(default=None, init=False, repr=False)
    _integer_center: object = field(default=None, init=False, repr=False)

    def _prepare_gram(self):
        if self._train is None:
            # The first query evaluates preparation in the same graph as its
            # search. A separate eval here needlessly submits and synchronizes
            # GPU work on every fresh fit. Later queries reuse the owned arrays.
            self._center, self._train, self._train_sq = _prepare_gram_arrays(self.X)

    def query(
        self, Q, k, *, return_distance=True, exclude_self=False,
        query_block_size=None, train_block_size=None, _validated=False, _squared=False,
        _auto_train_tiling=False,
    ):
        """Find neighbors, ordered by distance then training index.

        ``exclude_self`` treats Q as the training rows in their original order.
        It masks each global identity before selection, including duplicate
        rows whose identity would otherwise lie outside the selected prefix.
        Results are evaluated by the caller, or by conversion to a host array.
        """
        _matrix(Q, n_features=self.X.shape[1], allow_empty=True)
        if not isinstance(k, Integral) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        k = int(k)
        n_query, n_train = Q.shape[0], self.X.shape[0]
        if exclude_self and n_query != n_train:
            raise ValueError("exclude_self requires the query matrix to be the training matrix")
        limit = n_train - int(exclude_self)
        if k > limit:
            raise ValueError(f"n_neighbors must be <= {limit}, got {k}.")
        for value in (query_block_size, train_block_size):
            if value is not None and (not isinstance(value, Integral) or value < 1):
                raise ValueError("block sizes must be positive integers")
        with mx.stream(mx.gpu if self.device == "gpu" else mx.cpu):
            if not _validated and not bool(mx.all(mx.isfinite(Q))):
                raise ValueError("X must contain only finite values")
            if n_query == 0:
                return (mx.zeros((0, k), mx.float32) if return_distance else None,
                        mx.zeros((0, k), mx.int32))

            # Explicit train tiling is also a useful way to exercise the general
            # path; fused selection owns its own bounded reference partitions.
            features = self.X.shape[1]
            fused = self.device == "gpu" and _use_fused_search(features, k)
            geometry = "ordinary"
            # The NumPy boundary supplies a memory-budget tile even when the
            # caller requested none. Only an explicit tile should force Gram;
            # ordinary Gram queries still use that unchanged automatic plan.
            automatic_tiling = train_block_size is None or _auto_train_tiling
            if (self.device == "gpu" and automatic_tiling
                    and _use_small_search(n_train, n_query, features, k)):
                from ._neighbors_small import small_neighbors

                rows = min(n_query, query_block_size or 256)
                partitions = 16 if n_train > 2048 else 4
                options = {
                    "partitions": partitions, "exclude_self": exclude_self,
                    "sqrt_distances": return_distance and not _squared,
                }
                if rows == n_query:
                    distances, indices = small_neighbors(self.X, Q, k, **options)
                    return distances if return_distance else None, indices
                out_dist, out_idx = [], []
                for start in range(0, n_query, rows):
                    distances, indices = small_neighbors(
                        self.X, Q[start:start + rows], k, query_offset=start, **options,
                    )
                    out_idx.append(indices)
                    if return_distance:
                        out_dist.append(distances)
                    if start + rows < n_query:
                        mx.eval(indices, *([] if not return_distance else [distances]))
                return (mx.concatenate(out_dist) if return_distance else None,
                        mx.concatenate(out_idx))
            geometry_eligible = (
                self.device == "gpu" and automatic_tiling and k <= 8
                and min(n_query, query_block_size or n_query) >= 64
                and ((features == 32 and _use_compact_search(n_train, n_query, features, k))
                     or (33 <= features <= 256 and n_train >= 1024))
            )
            if geometry_eligible:
                if self._geometry is None:
                    if features == 32 and n_train >= 32768:
                        self._integer_preferred, self._geometry = _training_metadata(self.X)
                    else:
                        self._geometry = _sample_geometry(self.X)
                geometry = self._geometry
                if geometry == "clustered":
                    fused = True
            if fused and automatic_tiling:
                from ._neighbors_fused import fused_neighbors

                compact = _use_compact_search(n_train, n_query, self.X.shape[1], k)
                rows = query_block_size or (min(n_query, 1024) if compact else n_query)
                integer = geometry != "clustered" and _use_integer_search(
                    n_train, min(n_query, rows), self.X.shape[1], k,
                )
                if integer:
                    if self._integer_preferred is None:
                        self._integer_preferred = _prefer_integer_training(self.X)
                    integer = self._integer_preferred
                if geometry == "clustered" or (geometry == "offset" and not integer):
                    compact = False
                if geometry == "offset" and integer and self._integer_center is None:
                    self._integer_center = mx.mean(self.X, axis=0)
                out_dist, out_idx = [], []
                for start in range(0, n_query, rows):
                    block = Q[start:start + rows]
                    if integer and block.shape[0] >= 64:
                        from ._neighbors_integer import integer_neighbors, prepare_integer

                        centered = ({"center": self._integer_center}
                                    if self._integer_center is not None else {})
                        if self._integer is None:
                            self._integer = prepare_integer(self.X, **centered)
                        sq, indices = integer_neighbors(
                            self.X, block, k, self._integer,
                            exclude_self=exclude_self, query_offset=start,
                            **centered,
                        )
                    elif compact and block.shape[0] >= 64:
                        from ._neighbors_compact import compact_neighbors, prepare_compact

                        if self._compact is None:
                            # These arrays are lazy, owned preparation of X.
                            # Their first evaluation shares the query's graph.
                            self._compact = prepare_compact(self.X)
                        sq, indices = compact_neighbors(
                            self.X, block, k, self._compact,
                            exclude_self=exclude_self, query_offset=start,
                        )
                    else:
                        sq, indices = fused_neighbors(
                            self.X, block, k,
                            exclude_self=exclude_self, query_offset=start,
                        )
                    out_idx.append(indices)
                    if return_distance:
                        out_dist.append(sq if _squared else mx.sqrt(sq))
                    if start + rows < n_query:
                        mx.eval(indices, *([] if not return_distance else [out_dist[-1]]))
                return (mx.concatenate(out_dist) if return_distance else None,
                        mx.concatenate(out_idx))

            from ._neighbors_general import cpu_neighbors, gpu_neighbors

            rows = min(n_query, query_block_size or 256)
            cols = min(n_train, train_block_size or max(k + 4, (32 << 20) // (rows * 16)))
            if self.device == "cpu":
                squared, indices = cpu_neighbors(
                    self.X, Q, k, rows=rows, cols=cols, exclude_self=exclude_self,
                )
            else:
                squared, indices = gpu_neighbors(
                    self, Q, k, rows=rows, cols=cols, exclude_self=exclude_self,
                )
            return (squared if _squared else mx.sqrt(squared)) if return_distance else None, indices



def fit_neighbors(X, *, device="gpu", _validated=False):
    """Copy a finite float32 MLX matrix into an owned resident search index.

    NumPy and sklearn are neither imported nor required by this API. Shape and
    finiteness are checked once at fit, and independently for each public query.
    """
    if device not in ("gpu", "cpu"):
        raise ValueError("device must be 'gpu' or 'cpu'")
    _matrix(X)
    if X.shape[0] >= 2**31:
        raise ValueError("training row count must fit int32 indices")
    with mx.stream(mx.gpu if device == "gpu" else mx.cpu):
        # A bitwise copy owns storage without flushing subnormal coordinates.
        # Evaluate before the caller may reuse a borrowed input buffer.
        owned = (X.view(mx.uint32) | mx.zeros((), mx.uint32)).view(mx.float32)
        finite = mx.all(mx.isfinite(X)) if not _validated else None
        mx.eval(owned, *([] if finite is None else [finite]))
        if finite is not None and not bool(finite):
            raise ValueError("X must contain only finite values")
        return DeviceNeighborIndex(owned, device=device)


def _weights(distances):
    zero = distances == 0
    has_zero = mx.any(zero, axis=1, keepdims=True)
    # Scaling reciprocal weights by the minimum positive distance preserves
    # their normalized result without overflowing for subnormal distances.
    minimum = mx.min(mx.where(zero, float("inf"), distances), axis=1, keepdims=True)
    relative = minimum / mx.where(zero, 1.0, distances)
    return mx.where(has_zero, zero.astype(mx.float32), relative)


def class_probabilities(indices, targets, n_classes, distances=None, *, squared_distances=False):
    """Device class votes; targets are integer codes, already resident."""
    from ._neighbors_votes import class_probabilities as probabilities

    return probabilities(indices, targets, n_classes, distances,
                         squared_distances=squared_distances)


def regression_predictions(indices, targets, distances=None):
    """Resident float32 regression; sklearn's float64 target path is separate."""
    values = targets[indices]
    if distances is None:
        return mx.mean(values, axis=1)
    weights = _weights(distances)
    return mx.sum(values * weights, axis=1) / mx.sum(weights, axis=1)
