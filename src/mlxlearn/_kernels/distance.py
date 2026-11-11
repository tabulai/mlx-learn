"""Compatibility boundary and memory planning for exact Euclidean neighbors.

Fixed device search uses ``mlxlearn_device._neighbors``. Automatic NumPy
estimators can instead use the certified host search for latency-sensitive
workloads, while retaining the same owned training coordinates on both devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._common.arrays import INDEX_DTYPE, NUMPY_FLOAT
from .._common.config import get_tuning
from .._common.device import physical_memory_bytes, use_device
from .neighbors_cpu import CPUIndex

__all__ = ["NeighborIndex", "build_index", "resolve_blocks"]

#: Extra candidates carried per tile beyond ``k``. Cheap insurance against a
#: candidate that residual float32 noise in the Gram identity ranked just outside
#: the cut; the refinement pass then orders the enlarged set exactly.
_OVERSELECT = 4


@dataclass(frozen=True, slots=True)
class BlockPlan:
    """How a query is split into tiles."""

    query_rows: int
    train_cols: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"query_rows={self.query_rows}, train_cols={self.train_cols}"


def _distance_budget_bytes(backend: str) -> int:
    """Bytes to spend on one distance tile.

    On unified memory a bigger tile means fewer, wider matmuls, which is what the
    GPU wants. The budget is derived from installed memory and clamped, because a
    fixed constant is wrong by an order of magnitude on both an 8 GB Air and a
    128 GB Studio.
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


def resolve_blocks(
    n_query: int,
    n_train: int,
    *,
    backend: str,
    k_search: int,
    query_block_size: int | None = None,
    train_block_size: int | None = None,
) -> BlockPlan:
    """Choose tile sizes so a distance tile fits the budget.

    Unlike the ancestor, the device kind is passed in rather than inferred from
    whether the byte budget happens to exceed a GPU-ish constant — conflating a
    memory budget with a device kind meant an explicit CPU budget above 256 MiB
    silently switched on GPU alignment, and the CPU path could never shrink its
    block no matter how wide the training set was.
    """
    if n_query <= 0 or n_train <= 0:
        return BlockPlan(query_rows=0, train_cols=0)

    tuning = get_tuning()
    budget = _distance_budget_bytes(backend)
    # Distance values, packed ordering keys and selection indices coexist.
    itemsize = 16

    if query_block_size is not None:
        rows = max(1, min(int(query_block_size), n_query))
    else:
        rows = tuning.query_block_rows if backend == "mlx" else tuning.cpu_query_block_rows
        # Keep a full-width tile inside the budget when the training set is small
        # enough to be scanned in one pass; otherwise the train tiling handles it.
        max_rows = budget // max(n_train * itemsize, 1)
        if 0 < max_rows < rows:
            align = tuning.query_block_align if backend == "mlx" else 1
            aligned = (max_rows // align) * align if align > 1 else max_rows
            rows = aligned if aligned > 0 else max_rows
        rows = max(1, min(n_query, rows))

    min_cols = min(k_search, n_train)
    if train_block_size is not None:
        cols = max(min_cols, min(int(train_block_size), n_train))
    else:
        cols = budget // max(rows * itemsize, 1)
        cols = max(min_cols, min(n_train, int(cols)))

    return BlockPlan(query_rows=int(rows), train_cols=int(cols))


@dataclass(init=False, slots=True)
class NeighborIndex:
    """Owned search state and the sklearn compatibility boundary.

    Host X remains available for sklearn's private fitted state and portable
    pickles. It is independent of both authoritative search buffers. Automatic
    fits prepare CPU or GPU search state lazily when the query policy needs it;
    explicit device fits retain the resident NumPy-free MLX index.
    """

    _host_X: np.ndarray | None
    backend: str
    _device_index: object = None
    auto_policy: bool = False
    _gpu_available: bool | None = True
    _cpu_X: object = None
    _cpu_index: object = None
    last_query_backend: str | None = None

    def __init__(self, X, backend, _device_index=None, *, _auto=False, _gpu_available=True):
        self._host_X = None if _auto else X
        self.backend = backend
        self._device_index = _device_index
        self.auto_policy = _auto
        self._gpu_available = _gpu_available
        self._cpu_X = X if _auto else None
        self._cpu_index = None
        self.last_query_backend = None
        if not _auto or backend == "mlx":
            self._ensure_device()

    @property
    def training_data(self):
        if self._cpu_X is not None:
            return self._cpu_X
        if self._device_index is not None:
            return self._device_index.X
        return self._host_X

    def _ensure_cpu(self):
        if self._cpu_index is None:
            if self._cpu_X is None:
                self._cpu_X = np.asarray(self.training_data, dtype=NUMPY_FLOAT)
            self._cpu_index = CPUIndex(self._cpu_X)
        return self._cpu_index

    def _query_policy(self, n_query, k, Q=None):
        if not self.auto_policy:
            return self.backend, None
        if self._gpu_available is False:
            return "cpu", None
        n, features = self.n_samples, self.n_features
        if k <= 32 and features <= 32:
            if ((n_query <= 4 and n * n_query <= 16384)
                    or (n <= 256 and n_query <= 64)
                    or (n <= 512 and n_query <= 32)):
                return "cpu", None
            if 256 <= n <= 512 and 16 <= n_query <= 128 and 8 < features <= 32:
                cpu = self._ensure_cpu()
                cpu._prepare_native_train(n_query)
                if cpu._native_concentrated:
                    return "cpu", None
        if k <= 8 and n <= 2048 and 32 < features <= 64 and n_query <= 256:
            cpu = self._ensure_cpu()
            if cpu.sparse_integer_batch(None if Q is None else np.asarray(Q, dtype=NUMPY_FLOAT)):
                return "cpu", True if Q is not None else None
        if self._gpu_available is None:
            from .._common.device import gpu_available

            self._gpu_available = gpu_available()
        return ("mlx" if self._gpu_available else "cpu"), None

    def backend_for_query(self, n_query, k, Q=None):
        return self._query_policy(n_query, k, Q)[0]

    @property
    def X(self):
        if self._host_X is None:
            # A separate copy preserves the historical isolation between
            # sklearn's mutable private _fit_X and resident search state.
            self._host_X = np.array(
                self.training_data, dtype=NUMPY_FLOAT, order="C", copy=True,
            )
        return self._host_X

    @X.setter
    def X(self, value):
        self._host_X = value

    def _ensure_device(self):
        if self._device_index is not None:
            return
        import mlx.core as mx

        from mlxlearn_device import DeviceNeighborIndex

        X = self.training_data
        if X is None or X.ndim != 2 or 0 in X.shape:
            raise ValueError("X must be a nonempty two-dimensional matrix")
        if X.shape[0] >= 2**31:
            raise ValueError("training row count must fit int32 indices")
        device = "gpu" if self.auto_policy or self.backend == "mlx" else "cpu"
        with use_device(device):
            # Keep independently owned MLX storage even for automatic host
            # fits, including when GPU preparation is deferred until a query.
            self._device_index = DeviceNeighborIndex(
                mx.array(X, dtype=mx.float32),
                device=device,
            )

    @property
    def device_index(self):
        self._ensure_device()
        return self._device_index

    @property
    def n_samples(self):
        return int(self.training_data.shape[0])

    @property
    def n_features(self):
        return int(self.training_data.shape[1])

    def __getstate__(self):
        return {"X": np.array(self.training_data, copy=True) if self.auto_policy else self.X,
                "backend": self.backend, "auto_policy": self.auto_policy,
                "gpu_available": self._gpu_available}

    def __setstate__(self, state):
        # Legacy pickles may contain a host center, now derived on the device.
        self._host_X, self.backend = state["X"], state["backend"]
        self._device_index = None
        self.auto_policy = state.get("auto_policy", False)
        # An automatic pickle can move to a CPU-only machine. Re-probe only
        # when a GPU-sized query actually arrives; small queries remain host-only.
        self._gpu_available = None if state.get("gpu_available", True) is not False else False
        self._cpu_X = self._host_X if self.auto_policy else None
        if self.auto_policy:
            self._host_X = None
        self._cpu_index = None
        self.last_query_backend = None

    def query_device(
        self, Q, k, *, return_distance=True, exclude_self=False,
        query_block_size=None, train_block_size=None, _squared=False,
    ):
        """Return MLX results, without an intermediate host transfer."""
        import mlx.core as mx

        from mlxlearn_device._neighbors import _use_fused_search

        self._ensure_device()
        backend = "mlx" if self.auto_policy else self.backend
        self.last_query_backend = backend
        with use_device("gpu" if backend == "mlx" else "cpu"):
            query = Q if isinstance(Q, mx.array) else mx.array(Q, dtype=mx.float32)
            # Keep the existing memory policy, allowing the fused kernel to own
            # its reference partitions unless tiling is explicitly requested.
            plan = resolve_blocks(
                query.shape[0], self.n_samples, backend=backend,
                k_search=min(self.n_samples, int(k) + _OVERSELECT),
                query_block_size=query_block_size, train_block_size=train_block_size,
            )
            fused = backend == "mlx" and _use_fused_search(self.n_features, k)
            cols = train_block_size if fused else (plan.train_cols or None)
            return self._device_index.query(
                query, k, return_distance=return_distance, exclude_self=exclude_self,
                query_block_size=plan.query_rows or None, train_block_size=cols,
                _validated=True, _squared=_squared,
                _auto_train_tiling=train_block_size is None,
            )

    def query(
        self, Q, k, *, return_distance=True, exclude_self=False,
        query_block_size=None, train_block_size=None, _policy=None,
    ):
        """Return float64 distances and int64 indices at the sklearn boundary."""
        policy = _policy or self._query_policy(Q.shape[0], k, Q)
        if self.auto_policy and policy[0] == "cpu":
            self._ensure_cpu()
            plan = resolve_blocks(
                Q.shape[0], self.n_samples, backend="cpu", k_search=k,
                query_block_size=query_block_size, train_block_size=train_block_size,
            )
            result = self._cpu_index.query(
                np.asarray(Q, dtype=NUMPY_FLOAT), k, return_distance=return_distance,
                exclude_self=exclude_self, query_block_size=plan.query_rows,
                train_block_size=plan.train_cols, _integer_exact=policy[1],
            )
            self.last_query_backend = "cpu"
            return result
        import mlx.core as mx

        distances, indices = self.query_device(
            Q, k, return_distance=return_distance, exclude_self=exclude_self,
            query_block_size=query_block_size, train_block_size=train_block_size,
            _squared=True,
        )
        mx.eval(indices, *([] if distances is None else [distances]))
        return (None if distances is None else np.sqrt(np.asarray(distances, dtype=np.float64)),
                np.asarray(indices, dtype=INDEX_DTYPE))

    def _radius_backend(self):
        if not self.auto_policy or self.backend != "mlx":
            return self.backend
        if self._gpu_available is None:
            from .._common.device import gpu_available

            self._gpu_available = gpu_available()
        return "mlx" if self._gpu_available else "cpu"

    def query_radius(
        self,
        Q: np.ndarray,
        radius: float,
        *,
        return_distance: bool = True,
        exclude_self: bool = False,
        sort_results: bool = False,
        query_block_size: int | None = None,
        train_block_size: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return every training row within ``radius`` of each row of ``Q``.

        Parameters
        ----------
        Q : ndarray of shape (n_queries, n_features)
            Query rows, float32.
        radius : float
            Inclusive Euclidean radius. Negative values are rejected.
        return_distance : bool, default=True
            Whether to return Euclidean distances alongside indices.
        exclude_self : bool, default=False
            Drop each query's own training index. Requires ``len(Q) == n_samples``.
        sort_results : bool, default=False
            If True, each row is ordered by ``(distance, index)``. If False the
            order is still deterministic — the same total order — because an
            unspecified order would make block size and device visible in the
            output. scikit-learn does not define an order unless this flag is
            set; we always have one.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries,) of 1-d arrays
            Euclidean distances, float64. Only when ``return_distance=True``.
        neigh_ind : ndarray of shape (n_queries,) of 1-d arrays
            Training-row indices, int64.

        Notes
        -----
        Membership is decided on direct coordinate differences, not the Gram
        identity. ``query`` screens with Gram tiles because it certifies its
        selection and rescans, but a radius has no selection to certify and
        Metal's float32 matmul rounds far enough to lose true neighbors.
        """
        import mlx.core as mx

        n_query = int(Q.shape[0])
        n_train = self.n_samples
        radius = float(radius)
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")

        if exclude_self and n_query != n_train:
            raise ValueError(
                "exclude_self requires the query matrix to be the training matrix: "
                f"got {n_query} query rows and {n_train} training rows."
            )

        if n_query == 0:
            empty_ind = np.empty(0, dtype=object)
            if return_distance:
                return np.empty(0, dtype=object), empty_ind
            return empty_ind

        radius_sq = radius * radius
        backend = self._radius_backend()
        plan = resolve_blocks(
            n_query,
            n_train,
            backend=backend,
            k_search=1,
            query_block_size=query_block_size,
            train_block_size=train_block_size,
        )
        # A direct tile holds every coordinate difference, not one value per pair.
        train_cols = plan.train_cols
        if train_block_size is None:
            train_cols = max(1, train_cols * 4 // max(self.n_features, 4))

        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        dist_parts: list[np.ndarray] = []

        with use_device("gpu" if backend == "mlx" else "cpu"):
            X = self.training_data
            train = X if isinstance(X, mx.array) else mx.array(X, dtype=mx.float32)
            query = Q if isinstance(Q, mx.array) else mx.array(
                np.asarray(Q, dtype=NUMPY_FLOAT), dtype=mx.float32
            )

            for q_start in range(0, n_query, plan.query_rows):
                block = query[q_start:q_start + plan.query_rows, None, :]

                for t_start in range(0, n_train, train_cols):
                    diff = block - train[t_start:t_start + train_cols][None, :, :]
                    tile = mx.sum(diff * diff, axis=2)
                    mx.eval(tile)
                    tile_np = np.asarray(tile, dtype=np.float64)
                    local_rows, local_cols = np.nonzero(tile_np <= radius_sq)
                    if local_rows.size == 0:
                        continue

                    squared = tile_np[local_rows, local_cols]
                    rows = local_rows.astype(INDEX_DTYPE, copy=False) + np.int64(q_start)
                    cols = local_cols.astype(INDEX_DTYPE, copy=False) + np.int64(t_start)
                    if exclude_self:
                        keep = cols != rows
                        rows, cols, squared = rows[keep], cols[keep], squared[keep]
                        if rows.size == 0:
                            continue

                    row_parts.append(rows)
                    col_parts.append(cols)
                    dist_parts.append(np.sqrt(squared))

        self.last_query_backend = backend
        # Always apply a total order so tile sizes cannot leak into the output.
        # sort_results=False is still deterministic; scikit-learn does not
        # define an order unless the flag is set.
        sort_results = True
        neigh_ind, neigh_dist = _pack_radius_results(
            n_query, row_parts, col_parts, dist_parts, sort_results=sort_results
        )
        if return_distance:
            return neigh_dist, neigh_ind
        return neigh_ind


def _pack_radius_results(
    n_query: int,
    row_parts: list[np.ndarray],
    col_parts: list[np.ndarray],
    dist_parts: list[np.ndarray],
    *,
    sort_results: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ragged ``(n_query,)`` object arrays from concatenated triples.

    Rows are ordered by ``(distance, index)`` when ``sort_results`` is true —
    a total order, so the result does not depend on tile sizes.
    """
    neigh_ind = np.empty(n_query, dtype=object)
    neigh_dist = np.empty(n_query, dtype=object)
    empty_idx = np.empty(0, dtype=INDEX_DTYPE)
    empty_dist = np.empty(0, dtype=np.float64)

    if not row_parts:
        for i in range(n_query):
            neigh_ind[i] = empty_idx
            neigh_dist[i] = empty_dist
        return neigh_ind, neigh_dist

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts).astype(INDEX_DTYPE, copy=False)
    dists = np.concatenate(dist_parts).astype(np.float64, copy=False)

    if sort_results:
        order = np.lexsort((cols, dists, rows))
    else:
        order = np.argsort(rows, kind="mergesort")
    rows = rows[order]
    cols = cols[order]
    dists = dists[order]

    counts = np.bincount(rows, minlength=n_query)
    starts = np.cumsum(np.concatenate(([0], counts)))
    for i in range(n_query):
        sl = slice(int(starts[i]), int(starts[i + 1]))
        neigh_ind[i] = cols[sl]
        neigh_dist[i] = dists[sl]
    return neigh_ind, neigh_dist


def build_index(X: np.ndarray, *, backend: str, _owned: bool = False,
                _auto: bool = False, _gpu_available: bool | None = True) -> NeighborIndex:
    """Own training storage eagerly, deferring automatic CPU search preparation.

    ``_owned`` transfers a freshly allocated, validated conversion buffer from
    the wrapper. It must never be used for a borrowed user input array.
    """
    if backend == "mlx" and not _owned:
        import mlx.core as mx

        from mlxlearn_device import DeviceNeighborIndex

        Xf = np.asarray(X, dtype=NUMPY_FLOAT, order="C")
        if Xf.ndim != 2 or 0 in Xf.shape:
            raise ValueError("X must be a nonempty two-dimensional matrix")
        if Xf.shape[0] >= 2**31:
            raise ValueError("training row count must fit int32 indices")
        with use_device("gpu"):
            # mx.array eagerly owns a copy, including on MLX 0.29. Ordinary
            # queries need only that buffer; private state and pickling obtain
            # an independent host mirror through X when required.
            device_index = DeviceNeighborIndex(mx.array(Xf, dtype=mx.float32), device="gpu")
        return NeighborIndex(X=None, backend=backend, _device_index=device_index,
                             _auto=_auto, _gpu_available=_gpu_available)
    Xf = (np.asarray(X, dtype=NUMPY_FLOAT, order="C") if _owned
          else np.array(X, dtype=NUMPY_FLOAT, order="C", copy=True))
    return NeighborIndex(X=Xf, backend=backend, _auto=_auto, _gpu_available=_gpu_available)
