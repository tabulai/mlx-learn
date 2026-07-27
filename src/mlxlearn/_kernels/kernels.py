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


def _transform(mx, spec: KernelSpec, dot, a_sq, b_sq):
    """Turn an inner-product tile into a kernel tile, on device."""
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


def _prepare(spec: KernelSpec, A, center) -> np.ndarray:
    """Host-side float32 copy of ``A``, translated when the kernel allows it."""
    out = np.ascontiguousarray(A, dtype=NUMPY_FLOAT)
    if center is not None and spec.uses_center:
        out = np.ascontiguousarray(out - np.asarray(center, dtype=NUMPY_FLOAT))
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

    A32 = _prepare(spec, A, center)
    B32 = A32 if B is None else _prepare(spec, B, center)
    n_a, n_b = A32.shape[0], B32.shape[0]
    out = np.empty((n_a, n_b), dtype=NUMPY_FLOAT)
    if n_a == 0 or n_b == 0:
        return out

    rows = _block_rows(n_a, n_b, backend, block_rows)
    needs_norms = spec.kernel == "rbf"

    with use_device(_device(backend)):
        right = mx.array(B32, dtype=mx.float32)
        right_t = right.T
        b_sq = mx.sum(right * right, axis=1) if needs_norms else None
        for start in range(0, n_a, rows):
            end = min(start + rows, n_a)
            block = mx.array(A32[start:end], dtype=mx.float32)
            a_sq = mx.sum(block * block, axis=1) if needs_norms else None
            tile = _transform(mx, spec, block @ right_t, a_sq, b_sq)
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

    Returns
    -------
    ndarray of shape (n_a, n_outputs), float64

    Notes
    -----
    The kernel entries are computed in float32 on device — that is where the work
    is, ``n_a · n_b · n_features`` — and the contraction against ``weights`` is
    done in float64 on the host. The contraction is only ``n_a · n_b · n_outputs``
    with ``n_outputs`` equal to the number of one-vs-one classifiers, so it is
    negligible next to the kernel, while doing it in float32 would add an
    accumulation error over every support vector on top of the kernel's own
    rounding. Buying an order of magnitude of decision-value accuracy for a cost
    that does not show up in a profile is not a trade worth declining.

    The result is independent of ``block_rows``: each row of ``A`` is reduced on
    its own, so any chunking gives bit-identical output.
    """
    import mlx.core as mx

    A32 = _prepare(spec, A, center)
    B32 = _prepare(spec, B, center)
    n_a, n_b = A32.shape[0], B32.shape[0]
    W = np.asarray(weights, dtype=np.float64)
    out = np.zeros((n_a, W.shape[1]), dtype=np.float64)
    if n_a == 0 or n_b == 0:
        return out

    rows = _block_rows(n_a, n_b, backend, block_rows)
    needs_norms = spec.kernel == "rbf"

    with use_device(_device(backend)):
        right = mx.array(B32, dtype=mx.float32)
        right_t = right.T
        b_sq = mx.sum(right * right, axis=1) if needs_norms else None
        for start in range(0, n_a, rows):
            end = min(start + rows, n_a)
            block = mx.array(A32[start:end], dtype=mx.float32)
            a_sq = mx.sum(block * block, axis=1) if needs_norms else None
            tile = _transform(mx, spec, block @ right_t, a_sq, b_sq)
            mx.eval(tile)
            out[start:end] = np.asarray(tile).astype(np.float64) @ W
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
        Float32 C-ordered copy, already translated when ``spec.uses_center``.
    spec : KernelSpec
    backend : {"mlx", "cpu"}

    Notes
    -----
    A row costs one ``(n, d) @ (d,)`` product. That is deliberate: a decomposition
    solver touches a small, scattered subset of rows, so computing the rows it
    actually asks for beats materializing all ``n`` of them — and it is the only
    shape of this problem that fits in bounded memory at large ``n``.
    """

    X: np.ndarray
    spec: KernelSpec
    backend: str
    _device_X: object = None  # mx.array (n, d)
    _device_sq: object = None  # mx.array (n,), only for rbf

    def __post_init__(self) -> None:
        import mlx.core as mx

        with use_device(_device(self.backend)):
            data = mx.array(self.X, dtype=mx.float32)
            self._device_X = data
            if self.spec.kernel == "rbf":
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
            dot = (data @ data[index][:, None])[:, 0]
            if self.spec.kernel == "linear":
                out = dot
            elif self.spec.kernel == "rbf":
                sq = self._device_sq
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
    prepared = _prepare(spec, X, center)
    if prepared is X or np.shares_memory(prepared, X):
        prepared = prepared.copy()
    return KernelRows(X=prepared, spec=spec, backend=backend)
