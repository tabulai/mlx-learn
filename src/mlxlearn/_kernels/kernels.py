"""Blocked SVM kernel evaluation in MLX.

Four kernels — ``linear``, ``poly``, ``rbf``, ``sigmoid`` — with scikit-learn's
``gamma`` resolution, evaluated one row block at a time so that the working set on
the device is bounded by a byte budget rather than by ``n_samples²``. Nothing here
knows what an estimator is; everything takes and returns arrays and frozen
plain-data objects.

Three things this module is responsible for.

**Never requiring a full Gram.** :class:`KernelRows` answers "give me row ``i``"
from device-resident training data, and :func:`kernel_combine` folds
``K(A, B) @ W`` block by block. The dense ``n × n`` matrix is materialized only
when a caller explicitly asks for it via :func:`kernel_matrix`, which is what an
SMO fit does when — and only when — its row-cache budget already permits holding
every row.

**Exactness under float32.** The squared distance inside the RBF kernel comes from
the expanded identity ``‖a‖² + ‖b‖² − 2⟨a, b⟩``, which is hostile in float32 for
data far from the origin (``docs/fp32_policy.md`` §2.1). Distances are
translation-invariant, so both matrices are translated by a common ``center``
before the identity is applied — exact mathematics, and it is what keeps the
cancellation at the variance scale instead of the coordinate scale. Centering is
applied *only* to the RBF kernel: ``linear``, ``poly`` and ``sigmoid`` read the raw
inner product, which is not translation-invariant, and shifting it would compute a
different kernel.

**Integer powers by repeated squaring.** ``poly`` is ``(γ⟨a,b⟩ + coef0)ᵈ`` with an
integer degree. Evaluating that as a floating-point power returns NaN whenever the
base is negative, which happens routinely for ``coef0=0`` and a negative inner
product. LIBSVM uses binary exponentiation and so does this module, so a negative
base raised to an odd degree yields the negative value it should.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._common.arrays import NUMPY_FLOAT
from .._common.config import get_tuning
from .._common.device import use_device
from ..exceptions import UnsupportedParameterError
from . import svc_cpu

__all__ = [
    "SUPPORTED_KERNELS",
    "KernelSpec",
    "KernelRows",
    "resolve_gamma",
    "kernel_matrix",
    "kernel_combine",
    "kernel_diagonal",
]

#: The kernels mlxlearn evaluates. ``"precomputed"`` and callables are absent on
#: purpose: they are not slower here, they are simply not this module's job, and a
#: caller must be told so rather than handed a different kernel's answer.
SUPPORTED_KERNELS = ("linear", "poly", "rbf", "sigmoid")


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """A fully resolved kernel: no ``"scale"``, no ``"auto"``, no estimator.

    Attributes
    ----------
    kernel : {"linear", "poly", "rbf", "sigmoid"}
    gamma : float
        Already resolved by :func:`resolve_gamma`. Unused by ``"linear"``.
    degree : int
        Integer polynomial degree. Unused unless ``kernel="poly"``.
    coef0 : float
        Independent term. Unused by ``"linear"`` and ``"rbf"``.
    """

    kernel: str
    gamma: float = 1.0
    degree: int = 3
    coef0: float = 0.0

    def __post_init__(self) -> None:
        if self.kernel not in SUPPORTED_KERNELS:
            raise UnsupportedParameterError(
                f"kernel={self.kernel!r} is not evaluated by mlxlearn. Supported "
                f"kernels are: {', '.join(repr(k) for k in SUPPORTED_KERNELS)}.",
                parameter="kernel",
                value=self.kernel,
            )
        if int(self.degree) < 0:
            raise ValueError(f"degree must be non-negative, got {self.degree!r}")

    @property
    def uses_center(self) -> bool:
        """Whether a common translation may be applied before evaluation.

        True only for ``"rbf"``, whose squared-distance argument is
        translation-invariant. Translating the inputs of any other kernel here
        would silently evaluate a different kernel.
        """
        return self.kernel == "rbf"


def resolve_gamma(gamma, X: np.ndarray) -> float:
    """Resolve ``gamma`` the way scikit-learn resolves it.

    Parameters
    ----------
    gamma : {"scale", "auto"} or float
    X : ndarray of shape (n_samples, n_features)
        The validated training matrix, float64. The variance is taken over *all*
        entries — ``X.var()``, not a per-feature variance — because that is what
        scikit-learn does and ``gamma`` has to mean the same number here.

    Returns
    -------
    float

    Notes
    -----
    ``"scale"`` is ``1 / (n_features * X.var())``, falling back to ``1.0`` when the
    variance is at or below ``1e-12``; a constant ``X`` would otherwise divide by
    zero. ``"auto"`` is ``1 / n_features``.

    The variance is computed on the float64 validated matrix, not on the float32
    copy mlxlearn computes with. A float32 variance differs in the ninth
    significant digit, and that difference would propagate into every kernel entry
    of every fit — a systematic offset from scikit-learn bought for nothing.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn._kernels.kernels import resolve_gamma
    >>> X = np.array([[0.0, 1.0], [2.0, 3.0]])
    >>> resolve_gamma("auto", X)
    0.5
    >>> resolve_gamma(0.25, X)
    0.25
    """
    if isinstance(gamma, str):
        if gamma == "scale":
            variance = svc_cpu.variance(X)
            if variance is None:
                variance = float(X.var())
            return 1.0 / (X.shape[1] * variance) if variance > 1e-12 else 1.0
        if gamma == "auto":
            return 1.0 / X.shape[1]
        raise UnsupportedParameterError(
            f"gamma={gamma!r} is not a value mlxlearn resolves. Use 'scale', 'auto', "
            "or a positive float.",
            parameter="gamma",
            value=gamma,
        )
    value = float(gamma)
    if value <= 0.0:
        # scikit-learn's constraint is Interval(Real, 0.0, None, closed="left"),
        # so gamma=0.0 is legal there and produces a degenerate but well-defined
        # model. mlxlearn's kernels do not implement it — and because
        # scikit-learn *can* serve the request, this must be a CapabilityError so
        # that patched mode falls back instead of raising. A plain ValueError here
        # propagated straight through mlx_guard and broke Layer 2 too.
        raise UnsupportedParameterError(
            f"mlxlearn does not implement gamma={value!r}; the MLX kernels require a "
            "strictly positive gamma. scikit-learn accepts it, so under "
            "patch_sklearn() this falls back rather than failing.",
            parameter="gamma",
            value=value,
        )
    return value


def kernel_diagonal(spec: KernelSpec, X: np.ndarray) -> np.ndarray:
    """``K(x, x)`` for every row, computed in closed form, float64.

    Reading the diagonal off a materialized Gram would force one to exist; the
    closed forms below do not. They are also what LIBSVM uses, and it evaluates
    them in double while holding the cached kernel rows in single — the same split
    mlxlearn uses, which is why the two agree on the working-set quadratic
    coefficient to better than the tolerance the solver stops on.
    """
    native = svc_cpu.diagonal(X, spec)
    if native is not None:
        return native
    if spec.kernel == "rbf":
        # ‖x − x‖² = 0 exactly, so exp(0) = 1 exactly. Computing it would only
        # introduce a rounding error the analytic value does not have.
        return np.ones(X.shape[0], dtype=np.float64)

    squared = np.einsum("ij,ij->i", X, X, dtype=np.float64)
    if spec.kernel == "linear":
        return squared
    z = spec.gamma * squared + spec.coef0
    if spec.kernel == "sigmoid":
        return np.tanh(z)
    return _host_integer_power(z, int(spec.degree))


def _host_integer_power(base: np.ndarray, degree: int) -> np.ndarray:
    """``base ** degree`` by repeated squaring, correct for negative bases."""
    if degree == 0:
        return np.ones_like(base)
    result: np.ndarray | None = None
    factor = base
    exponent = degree
    while exponent:
        if exponent & 1:
            result = factor.copy() if result is None else result * factor
        exponent >>= 1
        if exponent:
            factor = factor * factor
    assert result is not None
    return result


# --------------------------------------------------------------------------------------
# device evaluation
# --------------------------------------------------------------------------------------


def _device(backend: str) -> str:
    return "gpu" if backend == "mlx" else "cpu"


def _block_rows(n_rows: int, n_cols: int, backend: str, block_rows: int | None) -> int:
    """How many rows of the left operand to evaluate at once.

    Derived from the same byte budget the distance kernel uses, halved because a
    block needs both the inner-product tile and the transformed tile alive at the
    same time. A fixed constant would be wrong by an order of magnitude on both an
    8 GB laptop and a 128 GB desktop.
    """
    if block_rows is not None:
        return max(1, min(int(block_rows), max(n_rows, 1)))
    tuning = get_tuning()
    budget = (
        tuning.distance_block_bytes if backend == "mlx" else tuning.cpu_distance_block_bytes
    )
    rows = int(budget // max(8 * n_cols, 1))
    return max(1, min(max(n_rows, 1), rows))


def _integer_power(mx, base, degree: int):
    """``base ** degree`` on device, by repeated squaring.

    ``mx.power`` with a floating exponent returns NaN for a negative base, and a
    negative base is the normal case for ``coef0=0`` with an obtuse angle between
    two samples. Repeated squaring is also exactly what LIBSVM's ``powi`` does, so
    the rounding sequence matches.
    """
    if degree == 0:
        return mx.ones_like(base)
    result = None
    factor = base
    exponent = degree
    while exponent:
        if exponent & 1:
            result = factor if result is None else result * factor
        exponent >>= 1
        if exponent:
            factor = factor * factor
    return result


def _transform(mx, spec: KernelSpec, dot, a_sq, b_sq, *, backend="cpu"):
    """Turn an inner-product tile into a kernel tile, on device."""
    if backend == "mlx" and 0 <= int(spec.degree) < 2**64:
        from .svc_fused import transform

        return transform(spec, dot, a_sq, b_sq)
    if spec.kernel == "linear":
        return dot
    if spec.kernel == "rbf":
        squared = a_sq[:, None] + b_sq[None, :] - 2.0 * dot
        # Clamp before the exponential: the identity can produce a small negative
        # value for two nearly identical rows, and exp(-gamma * negative) would
        # return a kernel entry greater than one.
        return mx.exp(-spec.gamma * mx.maximum(squared, 0.0))
    z = spec.gamma * dot + spec.coef0
    if spec.kernel == "sigmoid":
        return mx.tanh(z)
    return _integer_power(mx, z, int(spec.degree))


def _prepare(A) -> np.ndarray:
    """Convert the input buffer; translation is evaluated on the MLX device."""
    return np.ascontiguousarray(A, dtype=NUMPY_FLOAT)


def _device_array(mx, spec, data, center):
    """Upload once and translate on device, keeping array arithmetic out of NumPy."""
    out = mx.array(data, dtype=mx.float32)
    if center is not None and spec.uses_center:
        out = out - mx.array(center, dtype=mx.float32)
    return out


def kernel_matrix(
    spec: KernelSpec,
    A: np.ndarray,
    B: np.ndarray | None = None,
    *,
    backend: str = "cpu",
    center: np.ndarray | None = None,
    block_rows: int | None = None,
) -> np.ndarray:
    """The dense kernel matrix ``K(A, B)``, float32.

    Parameters
    ----------
    spec : KernelSpec
    A : ndarray of shape (n_a, n_features)
    B : ndarray of shape (n_b, n_features), optional
        Defaults to ``A``.
    backend : {"mlx", "cpu"}, default="cpu"
        Which MLX device evaluates the blocks.
    center : ndarray of shape (n_features,), optional
        Common translation, applied only for ``kernel="rbf"``.
    block_rows : int, optional
        Rows of ``A`` per block. Defaults to a memory-derived size.

    Returns
    -------
    ndarray of shape (n_a, n_b), float32

    Notes
    -----
    This is the one function here that does materialize an ``n_a × n_b`` result, so
    callers who cannot afford that should use :func:`kernel_combine` or
    :class:`KernelRows` instead. The *evaluation* is still blocked: the device
    never holds more than one block at a time regardless of the output size.
    """
    import mlx.core as mx

    A32 = _prepare(A)
    B32 = A32 if B is None else _prepare(B)
    n_a, n_b = A32.shape[0], B32.shape[0]
    out = np.empty((n_a, n_b), dtype=NUMPY_FLOAT)
    if n_a == 0 or n_b == 0:
        return out

    rows = _block_rows(n_a, n_b, backend, block_rows)
    needs_norms = spec.kernel == "rbf"

    with use_device(_device(backend)):
        right = _device_array(mx, spec, B32, center)
        right_t = right.T
        b_sq = mx.sum(right * right, axis=1) if needs_norms else None
        for start in range(0, n_a, rows):
            end = min(start + rows, n_a)
            block = right[start:end] if B is None else _device_array(
                mx, spec, A32[start:end], center
            )
            a_sq = None
            if needs_norms:
                a_sq = b_sq[start:end] if B is None else mx.sum(block * block, axis=1)
            tile = _transform(mx, spec, block @ right_t, a_sq, b_sq, backend=backend)
            mx.eval(tile)
            out[start:end] = np.asarray(tile)
    return out


def kernel_combine(
    spec: KernelSpec,
    A: np.ndarray,
    B: np.ndarray,
    weights: np.ndarray,
    *,
    backend: str = "cpu",
    center: np.ndarray | None = None,
    block_rows: int | None = None,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """``K(A, B) @ weights``, blocked over ``A``, returned as float64.

    Parameters
    ----------
    spec : KernelSpec
    A : ndarray of shape (n_a, n_features)
    B : ndarray of shape (n_b, n_features)
    weights : ndarray of shape (n_b, n_outputs)
    backend : {"mlx", "cpu"}, default="cpu"
    center : ndarray of shape (n_features,), optional
    block_rows : int, optional
    bias : ndarray of shape (n_outputs,), optional
        Subtract this intercept convention in the native reduction.

    Returns
    -------
    ndarray of shape (n_a, n_outputs), float64

    Notes
    -----
    Kernel entries are computed in float32 on device. Native CPU code contracts
    them against float64 weights and subtracts the optional bias in float64.
    This preserves the decision-value precision contract without NumPy numerical
    work. Installations without the optional extension use the portable NumPy
    reduction. GPU evaluation fuses each nonlinear kernel's transform into one
    pass after MLX's tuned matrix multiply.

    Different ``block_rows`` remain numerically equivalent within float32
    kernel precision; matrix multiplication and transcendental routines can
    select different accumulation/vectorization orders for different shapes.
    """
    import mlx.core as mx

    A32 = _prepare(A)
    B32 = _prepare(B)
    n_a, n_b = A32.shape[0], B32.shape[0]
    W = np.asarray(weights, dtype=np.float64)
    out = np.zeros((n_a, W.shape[1]), dtype=np.float64)
    if n_a == 0:
        return out
    if n_b == 0:
        native = svc_cpu.contract(np.empty((n_a, 0), dtype=NUMPY_FLOAT), W, bias)
        if native is not None:
            return native
        return out if bias is None else out - bias

    rows = _block_rows(n_a, n_b, backend, block_rows)
    needs_norms = spec.kernel == "rbf"

    with use_device(_device(backend)):
        right = _device_array(mx, spec, B32, center)
        right_t = right.T
        b_sq = mx.sum(right * right, axis=1) if needs_norms else None
        for start in range(0, n_a, rows):
            end = min(start + rows, n_a)
            block = _device_array(mx, spec, A32[start:end], center)
            a_sq = mx.sum(block * block, axis=1) if needs_norms else None
            tile = _transform(mx, spec, block @ right_t, a_sq, b_sq, backend=backend)
            mx.eval(tile)
            native = svc_cpu.contract(np.asarray(tile), W, bias)
            if native is not None:
                out[start:end] = native
            else:
                block_out = np.asarray(tile).astype(np.float64) @ W
                out[start:end] = block_out if bias is None else block_out - bias
    return out


@dataclass(slots=True)
class KernelRows:
    """Device-resident training data answering single-row kernel queries.

    Owned by whoever built it and released with it. Deliberately not a
    module-level cache keyed on the identity of the caller's array: an SMO fit
    that silently reused another fit's device tensors would produce a wrong model
    that no test distinguishes from a right one.

    Attributes
    ----------
    X : ndarray of shape (n_samples, n_features)
        Owned float32 C-ordered copy of the original coordinates.
    spec : KernelSpec
    backend : {"mlx", "cpu"}

    Notes
    -----
    Each row streams the resident ``(n, d)`` design. The GPU fuses its dot
    products or direct squared distances with the kernel transform; CPU MLX
    uses a matrix-vector product. Computing rows on demand keeps large
    decomposition problems within the cache budget.
    """

    X: np.ndarray
    spec: KernelSpec
    backend: str
    center: np.ndarray | None = None
    _device_X: object = None  # mx.array (n, d)
    _device_sq: object = None  # mx.array (n,), only for rbf

    def __post_init__(self) -> None:
        import mlx.core as mx

        with use_device(_device(self.backend)):
            data = _device_array(mx, self.spec, self.X, self.center)
            self._device_X = data
            if self.spec.kernel == "rbf" and self.backend != "mlx":
                self._device_sq = mx.sum(data * data, axis=1)
                mx.eval(data, self._device_sq)
            else:
                mx.eval(data)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    def row(self, index: int) -> np.ndarray:
        """``K(X[index], X)`` as a float32 array of length ``n_samples``."""
        import mlx.core as mx

        with use_device(_device(self.backend)):
            data = self._device_X
            if self.backend == "mlx" and 0 <= int(self.spec.degree) < 2**64:
                from .svc_fused import row

                out = row(self.spec, data, index)
                mx.eval(out)
                return np.asarray(out, dtype=NUMPY_FLOAT)
            dot = (data @ data[index][:, None])[:, 0]
            if self.spec.kernel == "linear":
                out = dot
            elif self.spec.kernel == "rbf":
                sq = self._device_sq
                if sq is None:
                    sq = mx.sum(data * data, axis=1)
                squared = sq[index] + sq - 2.0 * dot
                out = mx.exp(-self.spec.gamma * mx.maximum(squared, 0.0))
            else:
                z = self.spec.gamma * dot + self.spec.coef0
                if self.spec.kernel == "sigmoid":
                    out = mx.tanh(z)
                else:
                    out = _integer_power(mx, z, int(self.spec.degree))
            mx.eval(out)
            return np.asarray(out, dtype=NUMPY_FLOAT)

    def rows(self, indices) -> np.ndarray:
        """Evaluate a working set together and synchronize once for the batch."""
        import operator

        import mlx.core as mx

        ids = [operator.index(index) for index in indices]
        if any(index < 0 or index >= self.n_samples for index in ids):
            raise IndexError("kernel row index out of bounds")
        if not ids:
            return np.empty((0, self.n_samples), dtype=NUMPY_FLOAT)
        with use_device(_device(self.backend)):
            if self.backend == "mlx" and 0 <= int(self.spec.degree) < 2**64:
                from .svc_fused import rows

                out = rows(self.spec, self._device_X, ids)
            else:
                data = self._device_X
                selected = mx.array(ids, dtype=mx.int32)
                left = data[selected]
                a_sq = b_sq = None
                if self.spec.uses_center:
                    b_sq = self._device_sq
                    if b_sq is None:
                        b_sq = mx.sum(data * data, axis=1)
                    a_sq = b_sq[selected]
                out = _transform(mx, self.spec, left @ data.T, a_sq, b_sq, backend=self.backend)
            mx.eval(out)
            return np.asarray(out, dtype=NUMPY_FLOAT)


def build_kernel_rows(
    X: np.ndarray,
    spec: KernelSpec,
    *,
    backend: str,
    center: np.ndarray | None = None,
) -> KernelRows:
    """Build a :class:`KernelRows` over a float32 copy of ``X``.

    The copy is deliberate, and forced rather than incidental: ``_prepare`` returns
    its argument unchanged when it is already float32 and C-ordered, so without the
    explicit copy a row source could alias the caller's buffer and change its own
    answers when the caller mutated their array, mid-fit.
    """
    prepared = _prepare(X)
    if prepared is X or np.shares_memory(prepared, X):
        prepared = prepared.copy()
    return KernelRows(X=prepared, spec=spec, backend=backend, center=center)
