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

    Host X remains available for inherited sklearn radius queries and portable
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
            # queries need only that buffer; radius queries and pickling obtain
            # an independent host mirror through X when required.
            device_index = DeviceNeighborIndex(mx.array(Xf, dtype=mx.float32), device="gpu")
        return NeighborIndex(X=None, backend=backend, _device_index=device_index,
                             _auto=_auto, _gpu_available=_gpu_available)
    Xf = (np.asarray(X, dtype=NUMPY_FLOAT, order="C") if _owned
          else np.array(X, dtype=NUMPY_FLOAT, order="C", copy=True))
    return NeighborIndex(X=Xf, backend=backend, _auto=_auto, _gpu_available=_gpu_available)
