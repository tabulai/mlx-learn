"""Compatibility boundary and memory planning for exact Euclidean neighbors.

Fixed device search uses ``mlxlearn_device._neighbors``. Automatic NumPy
estimators can instead use the certified host search for latency-sensitive
workloads, while retaining the same owned training coordinates on both devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlxlearn_common._neighbors_plan import BlockPlan, PlanSettings, distance_budget_bytes
from mlxlearn_common._neighbors_plan import resolve_blocks as _resolve_blocks

from .._common.arrays import INDEX_DTYPE, NUMPY_FLOAT
from .._common.config import get_tuning
from .._common.device import physical_memory_bytes, use_device
from .neighbors_cpu import CPUIndex
from .radius import RadiusSearch, pack_radius

__all__ = ["NeighborIndex", "build_index", "resolve_blocks"]

#: Extra candidates carried per tile beyond ``k``. Cheap insurance against a
#: candidate that residual float32 noise in the Gram identity ranked just outside
#: the cut; the refinement pass then orders the enlarged set exactly.
_OVERSELECT = 4


def _plan_settings() -> PlanSettings:
    tuning = get_tuning()
    return PlanSettings(
        distance_block_bytes=tuning.distance_block_bytes,
        distance_block_bytes_max=tuning.distance_block_bytes_max,
        query_block_rows=tuning.query_block_rows,
        query_block_align=tuning.query_block_align,
        cpu_distance_block_bytes=tuning.cpu_distance_block_bytes,
        cpu_query_block_rows=tuning.cpu_query_block_rows,
    )


def _distance_budget_bytes(backend: str) -> int:
    return distance_budget_bytes(backend, _plan_settings(), physical_memory_bytes())


def resolve_blocks(
    n_query: int,
    n_train: int,
    *,
    backend: str,
    k_search: int,
    query_block_size: int | None = None,
    train_block_size: int | None = None,
) -> BlockPlan:
    """Use the shared planner with the current host interface's scoped tuning."""
    return _resolve_blocks(
        n_query, n_train, backend=backend, k_search=k_search,
        query_block_size=query_block_size, train_block_size=train_block_size,
        settings=_plan_settings(), budget_bytes=_distance_budget_bytes(backend),
    )


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
    _radius_search: RadiusSearch | None = None
    last_query_backend: str | None = None

    def __init__(self, X, backend, _device_index=None, *, _auto=False, _gpu_available=True):
        self._host_X = None if _auto else X
        self.backend = backend
        self._device_index = _device_index
        self.auto_policy = _auto
        self._gpu_available = _gpu_available
        self._cpu_X = X if _auto else None
        self._cpu_index = None
        self._radius_search = None
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
        self._radius_search = None
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
                _gram_tile=plan.tight_gram_tile,
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

    def _radius_gpu_available(self):
        if self._gpu_available is None:
            from .._common.device import gpu_available

            self._gpu_available = gpu_available()
        return self._gpu_available

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
        tree_allowed: bool = True,
        stock_factory=None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return the ragged rows inside an inclusive Euclidean radius.

        Candidate screening is bounded, and ambiguous or unrepresentable scores
        are checked from the original float32 coordinates in float64. A sorted
        row uses increasing distance with ties broken by training index.
        """
        n_query = int(Q.shape[0])
        radius = float(radius)
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        if exclude_self and n_query != self.n_samples:
            raise ValueError(
                "exclude_self requires the query matrix to be the training matrix: "
                f"got {n_query} query rows and {self.n_samples} training rows."
            )

        if n_query == 0:
            empty_ind = np.empty(0, dtype=object)
            if return_distance:
                return np.empty(0, dtype=object), empty_ind
            return empty_ind

        query = np.asarray(Q, dtype=NUMPY_FLOAT, order="C")
        radius_sq = radius * radius
        search = self._radius_search
        if search is None:
            search = self._radius_search = RadiusSearch(self.training_data)
        options = {
            "return_distance": return_distance,
            "exclude_self": exclude_self,
            "sort_results": sort_results,
        }
        route = search.route(
            query, radius_sq, fixed=None if self.auto_policy else self.backend,
            tree_allowed=tree_allowed, stock_factory=stock_factory, has_gpu=self._radius_gpu_available,
        )
        backend = "cpu" if route == "tree" else route
        if route == "tree":
            result = search.tree(query, radius_sq, **options)
        elif route == "sklearn":
            result = search.stock(query, radius_sq, factory=stock_factory, **options)
        else:
            plan = resolve_blocks(
                n_query, self.n_samples, backend=backend, k_search=1,
                query_block_size=query_block_size, train_block_size=train_block_size,
            )
            if route == "mlx" and not search.metal_supported(query, radius_sq):
                result = self._radius_direct(Q, radius_sq, plan, train_block_size, **options)
            else:
                resident = self._device_index.X if self._device_index is not None else None
                result = search.screen(query, radius_sq, plan, backend=backend,
                                       train_device=resident, **options)
        self.last_query_backend = backend
        return result

    def _radius_direct(self, Q, radius_sq, plan, train_block_size, *,
                       exclude_self, sort_results, return_distance):
        """Use the range-safe direct scan for extreme inputs on Metal."""
        import mlx.core as mx

        train_cols = plan.train_cols
        if train_block_size is None:
            train_cols = max(1, train_cols * 4 // max(self.n_features, 4))

        row_parts, col_parts, squared_parts = [], [], []
        with use_device("gpu"):
            X = self.training_data
            train = X if isinstance(X, mx.array) else mx.array(X, dtype=mx.float32)
            query = Q if isinstance(Q, mx.array) else mx.array(Q, dtype=mx.float32)
            for q_start in range(0, len(query), plan.query_rows):
                query_block = query[q_start:q_start + plan.query_rows]
                for t_start in range(0, len(train), train_cols):
                    train_block = train[t_start:t_start + train_cols]
                    diff = query_block[:, None, :] - train_block[None, :, :]
                    tile = mx.sum(diff * diff, axis=2)
                    mx.eval(tile)
                    squared = np.asarray(tile, dtype=np.float64)
                    _refine_radius_tile(squared, query_block, train_block)
                    local_rows, local_cols = np.nonzero(squared <= radius_sq)
                    if local_rows.size:
                        row_parts.append(local_rows + q_start)
                        col_parts.append(local_cols + t_start)
                        squared_parts.append(squared[local_rows, local_cols])

        empty = np.empty(0, dtype=INDEX_DTYPE)
        rows = np.concatenate(row_parts) if row_parts else empty
        cols = np.concatenate(col_parts) if col_parts else empty
        squared = np.concatenate(squared_parts) if squared_parts else np.empty(0)
        if exclude_self:
            keep = rows != cols
            rows, cols, squared = rows[keep], cols[keep], squared[keep]
        return pack_radius(rows, cols, squared, len(Q),
                           sort_results=sort_results, return_distance=return_distance)


def _refine_radius_tile(squared: np.ndarray, query, train) -> None:
    """Recover squared distances outside the reliable float32 range in place."""
    # A tile entry can overflow during subtraction, squaring or summation.
    # At the small end, each summand can round to a subnormal or zero.
    small_sum = float(np.finfo(NUMPY_FLOAT).tiny) * train.shape[1]
    unsafe = ~np.isfinite(squared) | (squared < small_sum)
    rows = np.flatnonzero(unsafe.any(axis=1))
    if rows.size == 0:
        return

    query = np.asarray(query)
    train = np.asarray(train)
    # Work one query row at a time to bound the float64 scratch space.
    for row in rows:
        cols = unsafe[row]
        diff = train[cols].astype(np.float64)
        diff -= query[row]
        squared[row, cols] = np.einsum("ij,ij->i", diff, diff)



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
