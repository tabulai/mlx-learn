"""Owned SVC inference data with exact mutation checks at the host boundary.

Only immutable preparation is reused: original float32 support coordinates,
their resident centered representation, and RBF norms. Query results, weights,
and bias are never cached. Public support-vector/center mutations are checked
against the owned snapshots before a caller reuses this object.
"""

from __future__ import annotations

import numpy as np

from .._common.device import use_device
from . import svc_cpu


def _same_float32(source, snapshot):
    source = np.asarray(source)
    if source.shape != snapshot.shape:
        return False
    # The native checker accepts matrices, including the center as one row.
    matched = svc_cpu.matches_float32(source.reshape(1, -1), snapshot.reshape(1, -1))
    if matched is not None:
        return matched
    # ascontiguousarray promotes a scalar center to shape (1,); restore its
    # original shape so exact matching agrees with the native flat comparison.
    converted = np.ascontiguousarray(source, dtype=np.float32).reshape(snapshot.shape)
    return bool(np.array_equal(converted.view(np.uint32), snapshot.view(np.uint32)))


class PreparedKernel:
    """An owned support-vector preparation for one resolved kernel/backend.

    Construction prepares the device arrays immediately. Pickling keeps only
    owned host snapshots and restores device state on the next ``combine``.
    ``matches`` must be called with current public model data before reuse.
    """

    def __init__(self, B, spec, *, backend="cpu", center=None):
        if backend not in ("cpu", "mlx"):
            raise ValueError("backend must be 'cpu' or 'mlx'")
        snapshot = np.array(B, dtype=np.float32, order="C", copy=True)
        if snapshot.ndim != 2:
            raise ValueError("support vectors must be a two-dimensional matrix")
        self.spec = spec
        self.backend = backend
        self._snapshot = snapshot
        self._center = None
        if spec.uses_center and center is not None:
            self._center = np.array(center, dtype=np.float32, order="C", copy=True)
            try:
                centered_shape = np.broadcast_shapes(self._center.shape, snapshot.shape)
            except ValueError as exc:
                raise ValueError("center must broadcast over support vectors") from exc
            if centered_shape != snapshot.shape:
                raise ValueError("center must broadcast over support vectors")
        self._freeze_snapshots()
        self._drop_device_state()
        self._ensure_prepared()

    def _freeze_snapshots(self):
        self._snapshot.flags.writeable = False
        if self._center is not None:
            self._center.flags.writeable = False

    def _drop_device_state(self):
        self._right = None
        self._right_t = None
        self._norms = None
        self._center_device = None

    def _ensure_prepared(self):
        if self._right is not None:
            return
        import mlx.core as mx

        from .kernels import _device

        with use_device(_device(self.backend)):
            right = mx.array(self._snapshot, dtype=mx.float32)
            center_device = None
            if self._center is not None:
                center_device = mx.array(self._center, dtype=mx.float32)
                right = right - center_device
            norms = mx.sum(right * right, axis=1) if self.spec.kernel == "rbf" else None
            right_t = right.T
            mx.eval(right, right_t, *(() if norms is None else (norms,)))
        # Publish fully evaluated arrays together; immutable prepared data can
        # then be read by simultaneous prediction calls without scratch sharing.
        self._right_t = right_t
        self._norms = norms
        self._center_device = center_device
        self._right = right

    def matches(self, B, spec, center, backend):
        """Whether current model data has exactly the prepared float32 values."""
        if self.backend != backend or self.spec != spec:
            return False
        if not _same_float32(B, self._snapshot):
            return False
        effective_center = center if spec.uses_center else None
        if self._center is None:
            return effective_center is None
        return effective_center is not None and _same_float32(effective_center, self._center)

    def combine(self, A, weights, *, bias=None, block_rows=None):
        """Evaluate ``K(A, prepared_supports) @ weights - bias`` in bounded tiles."""
        import mlx.core as mx

        from .kernels import _block_rows, _device, _prepare, _transform

        query = _prepare(A)
        if query.ndim != 2 or query.shape[1] != self._snapshot.shape[1]:
            raise ValueError("query and support vectors must agree in feature count")
        weights = np.asarray(weights, dtype=np.float64)
        support = self._snapshot.shape[0]
        if weights.ndim != 2 or weights.shape[0] != support:
            raise ValueError("weights must have one row per support vector")
        count = query.shape[0]
        output = np.empty((count, weights.shape[1]), dtype=np.float64)
        if not count:
            return output
        if not support:
            native = svc_cpu.contract(np.empty((count, 0), np.float32), weights, bias)
            if native is not None:
                return native
            output.fill(0)
            return output if bias is None else output - bias

        self._ensure_prepared()
        rows = _block_rows(count, support, self.backend, block_rows)
        with use_device(_device(self.backend)):
            for start in range(0, count, rows):
                stop = min(start + rows, count)
                block = mx.array(query[start:stop], dtype=mx.float32)
                if self._center_device is not None:
                    block = block - self._center_device
                norms = mx.sum(block * block, axis=1) if self._norms is not None else None
                tile = _transform(mx, self.spec, block @ self._right_t, norms,
                                  self._norms, backend=self.backend)
                mx.eval(tile)
                host_tile = np.asarray(tile)
                native = svc_cpu.contract(host_tile, weights, bias)
                if native is not None:
                    output[start:stop] = native
                else:
                    combined = host_tile.astype(np.float64) @ weights
                    output[start:stop] = combined if bias is None else combined - bias
        return output

    def __getstate__(self):
        return {"spec": self.spec, "backend": self.backend,
                "_snapshot": self._snapshot, "_center": self._center}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._freeze_snapshots()
        self._drop_device_state()
