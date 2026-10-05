"""Blocked SVM kernel evaluation in MLX.

Four kernels — ``linear``, ``poly``, ``rbf``, ``sigmoid`` — with scikit-learn's
``gamma`` resolution, evaluated one row block at a time so that the working set on
the device is bounded by a byte budget rather than by ``n_samples²``. Nothing here
knows what an estimator is; everything takes and returns arrays and frozen
plain-data objects.

Three things this module is responsible for.

**Never requiring a full Gram.** :class:`KernelRows` answers "give me rows
``i, j, …``" from device-resident training data, in one round trip, and
:meth:`KernelBasis.combine` folds ``K(Q, B) @ W`` block by block. The dense
``n × n`` matrix is materialized only when a caller explicitly asks for it via
:func:`kernel_matrix`, which is what an SMO fit does when — and only when — its
row-cache budget already permits holding every row.

**Accuracy under float32.** :func:`kernel_frame` maps the data in float64 *before*
the cast: an exact power-of-two scale, folded back into ``gamma`` by
:func:`frame_spec`, so no magnitude leaves float32's range, and — for the RBF
kernel only — a translation by the training mean. The squared distance inside the
RBF kernel comes from the expanded identity ``‖a‖² + ‖b‖² − 2⟨a, b⟩``, whose float32
error grows with ``‖a‖² + ‖b‖²`` rather than with the distance
(``docs/fp32_policy.md`` §2.1); distances are translation-invariant, so centering
keeps that error at the variance scale. Where it is not enough — two rows close
together but far from the mean, in tight well-separated clusters — the entries
the identity may have got wrong are recomputed on the host, in float64
(:class:`_RbfRefiner`).
``linear``, ``poly`` and ``sigmoid`` read the raw inner product, which is not
translation-invariant, so they are not translated; :func:`check_frame_resolution`
refuses data whose variation float32's inner product cannot resolve. And every
row a solver reads carries its self-entry ``K(x, x)`` exactly as
:func:`kernel_diagonal` states it, so the solver's curvature and its gradient
updates see one quadratic.

**Integer powers by repeated squaring.** ``poly`` is ``(γ⟨a,b⟩ + coef0)ᵈ`` with an
integer degree. Evaluating that as a floating-point power returns NaN whenever the
base is negative, which happens routinely for ``coef0=0`` and a negative inner
product. LIBSVM uses binary exponentiation and so does this module, so a negative
base raised to an odd degree yields the negative value it should.
"""

from __future__ import annotations

import functools
import math
import threading
from dataclasses import dataclass, replace

import numpy as np

from .._common.arrays import _FLOAT64_FIT_ADVICE, FLOAT32_MAX, NUMPY_FLOAT
from .._common.config import get_tuning
from .._common.device import use_device
from ..exceptions import UnsupportedInputError, UnsupportedParameterError

__all__ = [
    "SUPPORTED_KERNELS",
    "KernelSpec",
    "KernelBasis",
    "KernelRows",
    "resolve_gamma",
    "kernel_matrix",
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


def resolve_gamma(gamma, X: np.ndarray, *, kernel: str | None = None) -> float:
    """Resolve ``gamma`` the way scikit-learn resolves it.

    Parameters
    ----------
    gamma : {"scale", "auto"} or float
    X : ndarray of shape (n_samples, n_features)
        The validated training matrix, float64. The variance is taken over *all*
        entries — ``X.var()``, not a per-feature variance — because that is what
        scikit-learn does and ``gamma`` has to mean the same number here.
    kernel : str, optional
        When ``"linear"``, a numeric ``gamma`` is returned as given, including
        ``0.0``: the linear kernel never reads it, and scikit-learn stores it
        unvalidated for exactly that reason.

    Returns
    -------
    float

    Notes
    -----
    ``"scale"`` is ``1 / (n_features * X.var())``, falling back to ``1.0`` only when
    the variance is exactly zero — scikit-learn's ``X_var != 0``, not a threshold.
    This was ``variance > 1e-12``, which quietly resolved ``gamma=1`` for any data
    whose features sit around 1e-6 or below, where the correct value is 1e11 and
    up: a kernel that is 1 everywhere and a model at chance, with no warning.
    ``"auto"`` is ``1 / n_features``.

    The variance is computed on the float64 validated matrix, not on the float32
    copy mlxlearn computes with. A float32 variance differs in the ninth
    significant digit, and that difference would propagate into every kernel entry
    of every fit — a systematic offset from scikit-learn bought for nothing.

    The result may lie outside float32's range — ``"scale"`` on data around 1e-20
    gives about 1e38. That is not an error here: :func:`frame_spec` folds the
    power-of-two scale of the compute frame into it, and only the folded value is
    ever evaluated in float32.

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
            return 1.0 / (X.shape[1] * variance) if variance != 0.0 else 1.0
        if gamma == "auto":
            return 1.0 / X.shape[1]
        raise UnsupportedParameterError(
            f"gamma={gamma!r} is not a value mlxlearn resolves. Use 'scale', 'auto', "
            "or a positive float.",
            parameter="gamma",
            value=gamma,
        )
    value = float(gamma)
    if kernel == "linear":
        return value
    if value <= 0.0:
        # scikit-learn's constraint is Interval(Real, 0.0, None, closed="left"),
        # so gamma=0.0 is legal there and produces a degenerate but well-defined
        # model. mlxlearn's kernels do not implement it — and because
        # scikit-learn *can* serve the request, this must be a CapabilityError so
        # that patched mode falls back instead of raising. A plain ValueError here
        # propagated straight through mlx_guard and broke Layer 2 too.
        raise UnsupportedParameterError(
            f"mlxlearn does not implement gamma={value!r}; the MLX kernels require a "
            "strictly positive gamma.",
            parameter="gamma",
            value=value,
            advice="scikit-learn accepts it, so under patch_sklearn() this falls back "
            "rather than failing.",
        )
    return value


def kernel_diagonal(spec: KernelSpec, X: np.ndarray) -> np.ndarray:
    """``K(x, x)`` for every row, computed in closed form, float64.

    Reading the diagonal off a materialized Gram would force one to exist; the
    closed forms below do not. The solver takes its curvature
    ``K_ii + K_jj − 2K_ij`` from these values and its gradient updates from the
    rows, so the rows must agree with them: :func:`kernel_matrix` and
    :class:`KernelRows` write exactly these values, rounded to float32, into every
    self-entry, as LIBSVM's rows hold its double-precision diagonal rounded to
    float. Left to the float32 evaluation, a self-entry differed from its closed
    form by the rounding of a value as large as ``K`` itself — 31 on a poly kernel
    near 6e7 — and a step computed for one quadratic, applied to another, left the
    solver circling a violation for its whole iteration budget.
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
# the float32 compute frame
# --------------------------------------------------------------------------------------

_FLOAT32_MAX = FLOAT32_MAX

#: float32's unit roundoff.
_U = 2.0**-24


def _inner_product_error(n_features: int) -> float:
    """``κ``: the float32 error of an inner product or squared distance, per ``‖a‖² + ‖b‖²``.

    A model, not a proof. Measured on both MLX devices, the Gram identity's
    ``|computed − exact| / (u·(‖a‖² + ‖b‖²))`` peaked at 4.6 for 2–20 features and
    7.1 at 4 096 (the blocked sums grow like ``log d``, not the worst case's
    ``d``); this is two to four times that. The worst-case bound, ``γ_d``, is a
    hundred times larger at 4 096 features, and would send ordinary wide data
    down the slow paths it guards.
    """
    return _U * (8.0 + 2.0 * math.log2(n_features + 1))

#: Elements per host block when translating rows in float64: small enough that the
#: float64 buffer stays cache-resident, however large the input is.
_FRAME_BLOCK_ELEMENTS = 1 << 16


def kernel_frame(spec: KernelSpec, X: np.ndarray) -> tuple[np.ndarray | None, float]:
    """The exact map ``x ↦ (x − center) · scale`` under which ``spec`` runs in float32.

    Two things float32 cannot absorb are removed by exact arithmetic in float64,
    *before* the cast, rather than hoped away after it.

    **Offset** (``rbf`` only, the one translation-invariant kernel). The float64
    training mean is subtracted first. Casting first — what this module used to do —
    rounds away exactly the digits that distinguish samples: Unix timestamps within
    one hour collapse to 29 distinct float32 values out of 5 000, and no later
    translation can bring them back.

    **Magnitude.** An exact power of two puts the largest mapped entry in
    ``[0.5, 1)``, so squares and inner products can neither overflow (centred
    ``|x| ≳ 1.8e19 / √d``) nor sink into float32's subnormal range, which the GPU
    flushes to zero. :func:`frame_spec` folds the scale back into ``gamma``, so the
    kernel is unchanged. Scaling by a power of two commutes with float32 rounding,
    so for data already in range every intermediate rounds exactly as it did
    unscaled — the result is bit-identical, not merely close.

    ``"linear"`` gets the identity: its value *is* the raw inner product, with no
    ``gamma`` to absorb a scale, so :func:`check_frame_range` guards it instead.

    Parameters
    ----------
    spec : KernelSpec
    X : ndarray of shape (n_samples, n_features)
        The validated training matrix, float64.

    Returns
    -------
    center : ndarray of shape (n_features,), float64, or None
    scale : float
        An exact power of two; ``1.0`` when there is nothing to scale.
    """
    X = np.asarray(X, dtype=np.float64)
    if spec.kernel == "linear" or X.size == 0:
        return None, 1.0
    if spec.uses_center:
        center = X.mean(axis=0)
        # Rounded subtraction is monotone, so the largest |x − c| is attained at a
        # column's max or min: two reductions and no n × d temporary.
        extent = max(
            float(np.max(X.max(axis=0) - center)), float(np.max(center - X.min(axis=0)))
        )
    else:
        center = None
        extent = max(abs(float(X.max())), abs(float(X.min())))
    return center, _power_of_two_scale(extent)


def _power_of_two_scale(extent: float) -> float:
    """``2**-e`` with ``extent = m · 2**e``, ``m ∈ [0.5, 1)``; ``1.0`` if degenerate."""
    if not (0.0 < extent < np.inf):
        return 1.0
    _, exponent = np.frexp(extent)
    scale = float(np.ldexp(1.0, -int(exponent)))
    # Below 2**-1022 the reciprocal is not a finite double. Data that small is zero
    # in float32 either way, and so is every kernel distance LIBSVM would compute.
    return scale if 0.0 < scale < np.inf else 1.0


def frame_spec(spec: KernelSpec, scale: float) -> KernelSpec:
    """``spec`` with the frame's scale folded into ``gamma``: ``γ' = γ / scale²``.

    ``γ'‖s·a − s·b‖² = γ‖a − b‖²`` and ``γ'⟨s·a, s·b⟩ = γ⟨a, b⟩`` exactly, so the kernel
    evaluated on mapped inputs is the kernel of the original ones. This is also what
    lets a ``gamma`` outside float32's range be served: ``"scale"`` on data around
    1e-20 resolves to about 4e38, and folds back to order one.

    Raises
    ------
    UnsupportedParameterError
        When the folded ``gamma``, or ``coef0``, is not finite in float32 — an
        exponent the device cannot represent. scikit-learn evaluates the kernel in
        float64 and can serve it, so this is a capability limit, not bad input.
    """
    if spec.kernel == "linear":
        return spec
    # Two divisions rather than one by scale**2, which can overflow for tiny data.
    gamma = spec.gamma / scale / scale
    if not abs(gamma) <= _FLOAT32_MAX:
        raise UnsupportedParameterError(
            f"gamma={spec.gamma!r} is {gamma:.3g} once folded into mlxlearn's float32 "
            "compute frame, which float32 cannot represent.",
            parameter="gamma",
            value=spec.gamma,
            advice="scikit-learn evaluates the kernel in float64 and can serve it, so "
            "under patch_sklearn() this falls back rather than failing.",
        )
    if not abs(spec.coef0) <= _FLOAT32_MAX:
        raise UnsupportedParameterError(
            f"coef0={spec.coef0!r} is not finite in float32, where mlxlearn evaluates "
            "the kernel.",
            parameter="coef0",
            value=spec.coef0,
            advice="scikit-learn evaluates it in float64, so under patch_sklearn() this "
            "falls back rather than failing.",
        )
    return replace(spec, gamma=gamma)


def apply_frame(X, center: np.ndarray | None, scale: float) -> np.ndarray:
    """``(X − center) · scale``, rounded to float32 once, as a new array.

    The translation is computed in float64 — that is its whole point — through a
    small reused buffer rather than a float64 copy of ``X``. The power-of-two scale
    is exact in either precision, so it is folded into the final store. Every entry
    is mapped independently, so the blocking cannot change a single bit. The output
    never aliases ``X``. Entries that land beyond float32 become ``inf`` rather than
    raising here; :func:`check_frame_range` is what turns that into an error.
    """
    X = np.asarray(X)
    out = np.empty(X.shape, dtype=NUMPY_FLOAT)
    if X.size == 0:
        return out
    with np.errstate(over="ignore"):
        if center is None:
            if scale == 1.0:
                np.copyto(out, X, casting="same_kind")
            else:
                np.multiply(X, scale, out=out)
            return out
        step = max(1, _FRAME_BLOCK_ELEMENTS // max(int(X.shape[1]), 1))
        buffer = np.empty((min(step, X.shape[0]), X.shape[1]), dtype=np.float64)
        for start in range(0, X.shape[0], step):
            stop = min(start + step, X.shape[0])
            block = buffer[: stop - start]
            np.subtract(X[start:stop], center, out=block)
            np.multiply(block, scale, out=out[start:stop])
    return out


def check_frame_range(
    spec: KernelSpec, X32: np.ndarray, *, name: str = "X", fallback_advice: bool = True
) -> None:
    """Raise when float32 kernel arithmetic among mapped rows ``X32`` could overflow.

    ``rbf`` evaluates ``‖a‖² + ‖b‖² − 2⟨a, b⟩``, at most four times the largest
    squared row norm; the other kernels evaluate ``⟨a, b⟩``, bounded by it. Within
    the bound no partial sum can overflow either (Cauchy–Schwarz on each prefix).

    This is the check for the rows of a Gram — a fit's training data. After
    :func:`kernel_frame` those are always in range, so what it catches is a
    linear-kernel fit with ``‖x‖² > 3.4e38``. Queries against fitted rows are
    checked by :func:`check_query_range`. Values past ``3.4e38`` themselves are
    rejected earlier, on the raw input. ``fallback_advice=False`` leaves out the
    ``patch_sklearn()`` advice, which only a fit can act on.

    Raises
    ------
    UnsupportedInputError
        With ``reason="float32-overflow"``, so patched mode can fall back at fit.
    """
    if X32.size == 0:
        return
    factor = 4.0 if spec.kernel == "rbf" else 1.0
    # ‖x‖² ≤ d · max|x|² costs two reductions and no temporary; the exact norms are
    # computed only when that cheap bound cannot rule an overflow out.
    peak = max(float(X32.max()), -float(X32.min()))
    if factor * X32.shape[1] * peak * peak <= _FLOAT32_MAX:
        return
    largest = float(np.einsum("ij,ij->i", X32, X32, dtype=np.float64).max())
    if not factor * largest <= _FLOAT32_MAX:
        raise UnsupportedInputError(
            f"{name} has a row whose squared norm is {largest:.3g} in mlxlearn's float32 "
            f"compute frame, so the {spec.kernel!r} kernel would overflow float32 (the "
            f"limit is {_FLOAT32_MAX:.3g}) and return infinity or NaN. Rescale the "
            "data.",
            reason="float32-overflow",
            advice=_FLOAT64_FIT_ADVICE if fallback_advice else "",
        )


def check_query_range(
    spec: KernelSpec, q_sq: np.ndarray, partner_sq: float, *, name: str = "X"
) -> None:
    """Raise when queries cannot be evaluated against fitted rows in float32.

    ``q_sq`` holds the mapped queries' squared norms — infinite for an entry the
    frame could not hold, and not overflowed by float32 itself (see
    :func:`_query_norms`) — and ``partner_sq`` the largest squared norm among the
    fitted rows. Only ``⟨q, b⟩`` has to stay finite:
    within ``‖q‖·‖b‖ ≤ FLOAT32_MAX / 2`` no partial sum overflows. A query's own
    ``‖q‖²`` may overflow, and that is harmless for the kernels that read it: the
    rbf squared distance becomes ``+inf`` and the entry ``exp(−inf) = 0`` — the
    right value whenever even the nearest fitted row is past float32's ``exp``,
    which is checked. ``tanh`` saturates to ±1, correctly. A poly power that
    overflows is caught by the caller's finiteness check on the decision values.

    The Gram bound of :func:`check_frame_range` used to be applied here, so a CMIP
    ``1e20`` or netCDF ``9.97e36`` fill value in one query row failed the whole
    ``predict`` — values scikit-learn answers, and so did mlxlearn 0.1.0a2.

    Raises
    ------
    UnsupportedInputError
        With ``reason="float32-overflow"``. Inference cannot fall back, so no
        ``patch_sklearn()`` advice is given.
    """
    if q_sq.size == 0:
        return
    partner = math.sqrt(partner_sq)
    reach = math.sqrt(float(q_sq.max()))
    if not reach * partner <= _FLOAT32_MAX / 2:
        raise UnsupportedInputError(
            f"{name} has a row whose inner product with the fitted rows reaches "
            f"{reach * partner:.3g} in mlxlearn's float32 compute frame, past float32's "
            f"limit of {_FLOAT32_MAX:.3g}: the {spec.kernel!r} kernel would return "
            "infinity or NaN. Rescale the queries (or the training data).",
            reason="float32-overflow",
        )
    if spec.kernel != "rbf" or reach + partner <= math.sqrt(_FLOAT32_MAX):
        return
    # Where ‖q‖² + ‖b‖² + 2‖q‖‖b‖ can pass float32's range, the entry comes out as
    # exp(−inf) = 0. That is exact when the true distance, at least ‖q‖ − ‖b‖, puts
    # every entry below float32's smallest exp (e^−104); a vanishing gamma would not.
    norms = np.sqrt(q_sq)
    far = norms + partner > math.sqrt(_FLOAT32_MAX)
    if not spec.gamma * float(np.min(norms[far]) - partner) ** 2 >= 104.0:
        raise UnsupportedInputError(
            f"{name} has a row so far from the fitted rows that its squared distance "
            f"overflows float32, while gamma={spec.gamma:.3g} in the compute frame is too "
            "small for its kernel value to be zero. Rescale the queries (or the "
            "training data).",
            reason="float32-overflow",
        )


def _query_norms(Q32: np.ndarray) -> np.ndarray:
    """Squared row norms of mapped queries: float32 where they fit, else float64.

    A float32 sum is a fifth of the float64 one's cost, and only a query whose
    norm float32 cannot hold — a fill value far outside the training data — needs
    the wider one. An entry the frame could not hold stays ``inf`` in either.
    """
    q_sq = np.einsum("ij,ij->i", Q32, Q32)
    if not math.isfinite(float(q_sq.max(initial=0.0))):
        q_sq = np.einsum("ij,ij->i", Q32, Q32, dtype=np.float64)
    return q_sq


#: How much of the data's variation the inner product's float32 error may reach
#: before :func:`check_frame_resolution` refuses the data. See there.
_RESOLUTION_LIMIT = 1e-3


def check_frame_resolution(spec: KernelSpec, X32: np.ndarray) -> None:
    """Raise when float32 cannot resolve the variation of an untranslated kernel.

    ``linear`` and ``poly`` read ``⟨a, b⟩`` itself, which is not
    translation-invariant, so the frame cannot centre them. Their float32 error is
    about ``κ·‖a‖‖b‖`` (:func:`_inner_product_error`), while what the solver
    learns from is how the kernel *varies* between rows: its curvature
    ``K_ii + K_jj − 2K_ij`` scales with ``‖a − b‖²``, and a polynomial's degree
    multiplies both sides alike. Data with a large offset and a small spread —
    ``make_classification · 0.01 + 20`` — makes the one larger than the other: the
    poly kernel came out wrong by 40 near 6.5e7 against a curvature of 3–100, the
    fit either circled its whole budget (376 s against scikit-learn's 0.001 s) or
    converged, silently, to a model 12 points less accurate. So the largest squared
    norm ``R²`` is compared with the mean squared distance between rows,
    ``S² = 2·Σ var``, and, for ``poly``, ``coef0``'s own rounding with ``gamma·S²``.
    The ratio overstates the measured kernel error per median curvature by a
    factor of 2–3; past :data:`_RESOLUTION_LIMIT` the data is refused. Measured on
    the linear kernel, where scikit-learn still converges: the decision values
    moved by 1.4e-3 of their range at a ratio of 1.7e-4, 1.1e-2 at 1.7e-3, and 6e-2,
    with 98.8% of labels agreeing, at 1.5e-2.

    ``sigmoid`` reads the inner product too, but is not checked: on the same
    offsets, up to a thousand times the spread and with ``gamma`` from saturating
    to linear, it agreed with scikit-learn to 3e-4 or better. And this is a
    global test: variation that lives only at a finer scale than ``S`` — tight
    clusters far from the origin and from each other — is not seen by it.

    Raises
    ------
    UnsupportedInputError
        With ``reason="float32-cancellation"``, so patched mode falls back at fit.
    """
    if spec.kernel not in ("linear", "poly") or X32.shape[0] < 2:
        return
    reach = float(np.einsum("ij,ij->i", X32, X32).max())
    spread = 2.0 * float(np.var(X32, axis=0, dtype=np.float64).sum())
    if reach == 0.0:
        return  # every row is zero, and so is every entry, exactly
    # Rows float32 holds as identical — often rows that were not, rounded at a
    # large offset — leave only rounding for the solver to learn from.
    ratio = math.inf
    if spread > 0.0:
        ratio = _inner_product_error(X32.shape[1]) * reach / spread
        if spec.kernel == "poly":
            ratio += _U * abs(spec.coef0) / (spec.gamma * spread)
    if ratio > _RESOLUTION_LIMIT:
        raise UnsupportedInputError(
            f"The {spec.kernel!r} kernel reads the raw inner product, and float32 "
            "cannot resolve this data's variation in it: the rows are far from the "
            f"origin compared with how far they are from each other (largest squared "
            f"norm {reach:.3g}, mean squared distance {spread:.3g} in mlxlearn's compute "
            f"frame), so its rounding would reach about {ratio:.2g} of the kernel's "
            f"variation (the limit is {_RESOLUTION_LIMIT:g}). Standardize the features, "
            "e.g. with StandardScaler.",
            reason="float32-cancellation",
            advice=_FLOAT64_FIT_ADVICE,
        )


#: An rbf entry is recomputed when the identity's error bound, in the exponent,
#: exceeds this. On standardized data at ``gamma="scale"`` the bound is ~1e-5 even at
#: 4 096 features, so ordinary data never reaches it.
_RBF_REFINE_BOUND = 1e-4

#: ... unless the entry, computed and true, is provably below this.
_RBF_NEGLIGIBLE = 1e-8
_RBF_NEGLIGIBLE_LOG = -math.log(_RBF_NEGLIGIBLE)

#: float32's smallest normal value: an entry computed as zero was at most this.
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)


class _RbfRefiner:
    """Recomputes, directly, the rbf entries the Gram identity may have got wrong.

    The identity's error in ``‖a − b‖²`` is about ``κ·(‖a‖² + ‖b‖²)``
    (:func:`_inner_product_error`), which centering keeps small only where rows
    are near the training mean. Between two rows close to each other but far from
    the mean — tight clusters far apart, with ``gamma`` matched to the clusters —
    kernel values came out wrong by up to 0.58, and SVC decision values by 3.5
    against scikit-learn's, silently.

    An entry needs recomputing when its error, in the exponent, can exceed
    :data:`_RBF_REFINE_BOUND` and its value, computed or true, can exceed
    :data:`_RBF_NEGLIGIBLE`. Distance is at least ``|‖a‖ − ‖b‖|``, so for a row of
    norm ``α`` only the rows in a band of norms around it can be non-negligible,
    and only those far enough out for the error to matter: a binary search over the
    sorted norms finds them, so a lone far-out row costs its own band, not a pass
    over the tile. The band's entries are recomputed from the float32 rows in
    float64, accurate to float32's own rounding for the values the device holds
    (``docs/fp32_policy.md`` §3.4). Where the identity's float64 error — 2⁻²⁹ of
    float32's — stays below that rounding, that is one matrix-vector product
    over the band's rows, kept contiguous in norm order (gathering the rows and
    differencing measured 39 times slower on dense bands); beyond it, direct
    differences ``Σ(b − a)²`` of the entries that can matter. On ordinary data
    :meth:`needed` is false and nothing runs.
    """

    __slots__ = ("gamma", "kappa", "order", "sq", "norms", "rows")

    def __init__(self, gamma: float, rows: np.ndarray) -> None:
        self.gamma = float(gamma)
        self.kappa = _inner_product_error(rows.shape[1])
        sq = np.einsum("ij,ij->i", rows, rows, dtype=np.float64)
        self.order = np.argsort(sq, kind="stable")
        # All in norm order, so that a band is a contiguous slice.
        self.sq = sq[self.order]
        self.norms = np.sqrt(self.sq)
        self.rows = rows[self.order].astype(np.float64)

    @staticmethod
    def needed(spec: KernelSpec, n_features: int, left_sq: float, right_sq: float) -> bool:
        """Whether any entry between rows of these largest squared norms can need it."""
        if spec.kernel != "rbf":
            return False
        bound = spec.gamma * _inner_product_error(n_features) * (left_sq + right_sq)
        return bool(bound > _RBF_REFINE_BOUND)

    def __call__(self, tile: np.ndarray, left: np.ndarray) -> None:
        """Correct ``tile = K(left, rows)``, a writable float32 host array, in place."""
        g, k = self.gamma, self.kappa
        left64 = left.astype(np.float64)
        left_sq = np.einsum("ij,ij->i", left64, left64)
        alpha = np.sqrt(left_sq)
        with np.errstate(divide="ignore", over="ignore"):
            # (α − β)² − κ(α² + β²) < L/γ: where an entry can be non-negligible ...
            slack = np.sqrt(left_sq * k * (2.0 - k) + (1.0 - k) * _RBF_NEGLIGIBLE_LOG / g)
            # ... and γκ(α² + β²) > bound: where its error can matter.
            floor = np.sqrt(np.maximum(_RBF_REFINE_BOUND / (g * k) - left_sq, 0.0))
        first = np.searchsorted(self.norms, np.maximum((alpha - slack) / (1.0 - k), floor))
        last = np.searchsorted(self.norms, (alpha + slack) / (1.0 - k), side="right")
        # The float64 identity's own error, in the exponent, against float32's rounding.
        k64 = k * 2.0**-29
        for r in np.flatnonzero(last > first):
            lo, hi = int(first[r]), int(last[r])
            cols = self.order[lo:hi]
            if g * k64 * (left_sq[r] + self.sq[hi - 1]) <= _U:
                # Float32 rows, widened: their products cannot overflow float64, so
                # a flag NumPy's BLAS raises here is spurious (smo._gradient).
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    dots = self.rows[lo:hi] @ left64[r]
                squared = np.maximum(left_sq[r] + self.sq[lo:hi] - 2.0 * dots, 0.0)
                tile[r, cols] = np.exp(-g * squared)
                continue
            exponent_error = np.minimum(g * k * (left_sq[r] + self.sq[lo:hi]), 700.0)
            computed = tile[r, cols].astype(np.float64)
            upper = np.maximum(computed, _FLOAT32_TINY) * np.exp(exponent_error)
            keep = upper >= _RBF_NEGLIGIBLE
            if keep.any():
                diff = self.rows[lo:hi][keep] - left64[r]
                tile[r, cols[keep]] = np.exp(-g * np.einsum("ij,ij->i", diff, diff))


class KernelBasis:
    """A fixed right-hand kernel operand — an SVC's support vectors — kept on device.

    Every ``predict`` used to re-map, re-upload and re-reduce the support vectors:
    0.2–0.4 ms of fixed cost at 4 000 × 256, against about 0.25 ms for scikit-learn's
    *entire* single-row predict. This does that work once.

    Owned by the fitted estimator and released with it — never a cache keyed on the
    identity of an array. It holds no host copy of its own: the device tensors are
    built on first use from the rows the owner passes in, and dropped on pickle, so a
    pickled model carries its support vectors exactly once and can be loaded where
    the device tensors could not have been. The build is locked, so concurrent first
    predictions build it once. A small batch is evaluated on the host
    (``combine(..., host=True)``) from zero-copy views of those same tensors — on
    unified memory the host reads the device's buffers directly.

    Attributes
    ----------
    spec : KernelSpec
        The *compute* spec, with the frame's scale already folded in by
        :func:`frame_spec`.
    center, scale
        The frame, from :func:`kernel_frame`.
    """

    __slots__ = ("spec", "center", "scale", "_lock", "_operand")

    def __init__(
        self, spec: KernelSpec, *, center: np.ndarray | None = None, scale: float = 1.0
    ) -> None:
        self.spec = spec
        self.center = None if center is None else np.array(center, dtype=np.float64)
        self.scale = float(scale)
        self._lock = threading.Lock()
        # (mx.array (n, d), mx.array (n,) or None, the largest squared row norm,
        #  and zero-copy host views of the two arrays)
        self._operand: tuple | None = None

    def __getstate__(self) -> dict:
        """Pickle the frame only; the device tensors are derived from the owner's rows."""
        return {"spec": self.spec, "center": self.center, "scale": self.scale}

    def __setstate__(self, state: dict) -> None:
        self.spec = state["spec"]
        self.center = state["center"]
        self.scale = state["scale"]
        self._lock = threading.Lock()
        self._operand = None

    def prepare(self, X, *, name: str = "X", fallback_advice: bool = True) -> np.ndarray:
        """``X`` mapped into this basis's float32 frame, range-checked as Gram rows."""
        mapped = apply_frame(X, self.center, self.scale)
        check_frame_range(self.spec, mapped, name=name, fallback_advice=fallback_advice)
        return mapped

    def _device_operand(self, rows: np.ndarray, device: str) -> tuple:
        operand = self._operand
        if operand is not None:
            return operand
        with self._lock:
            if self._operand is None:
                import mlx.core as mx

                prepared = self.prepare(rows, name="support_vectors_", fallback_advice=False)
                sq = np.einsum("ij,ij->i", prepared, prepared, dtype=np.float64)
                reach = float(sq.max(initial=0.0))
                with use_device(device):
                    right = mx.array(prepared, dtype=mx.float32)
                    if self.spec.kernel == "rbf":
                        right_sq = mx.sum(right * right, axis=1)
                        mx.eval(right, right_sq)
                    else:
                        right_sq = None
                        mx.eval(right)
                views = (
                    np.asarray(right),
                    None if right_sq is None else np.asarray(right_sq),
                )
                # Published whole, in one assignment, so an unlocked reader sees
                # either nothing or a complete operand.
                self._operand = (right, right_sq, reach, *views)
            return self._operand

    def combine(
        self,
        Q,
        rows: np.ndarray,
        weights: np.ndarray,
        *,
        backend: str,
        device: str | None = None,
        block_rows: int | None = None,
        host: bool = False,
    ) -> np.ndarray:
        """``K(Q, rows) @ weights`` for raw queries ``Q``, float64.

        Parameters
        ----------
        Q : ndarray of shape (n_queries, n_features)
            Unmapped queries, float64 or float32; mapped here.
        rows : ndarray of shape (n_rows, n_features)
            The owner's unmapped basis rows. Read only when the device operand has
            not been built yet.
        weights : ndarray of shape (n_rows, n_outputs)
        backend : {"mlx", "cpu"}
        device : {"gpu", "cpu"}, optional
            The MLX device to evaluate on, as the caller resolved it; by default
            the one ``backend`` names.
        host : bool, default=False
            Evaluate on the host instead, in NumPy float32, from zero-copy views of
            the device operand: the same arithmetic, entry for entry, as the device
            tiles (:func:`_host_tile`), so the result differs from theirs only in
            the rounding of the BLAS that computes the inner products. For a batch
            of a few rows, where the device's cost is not arithmetic but the call
            itself — 0.6-1 ms once the GPU has idled — the caller asks
            for this (``SVC``'s ``svc_host_inference_work``).

        Notes
        -----
        The kernel entries are computed in float32 on device — that is where the
        arithmetic is, ``n_q · n_rows · n_features`` — and the contraction against
        ``weights`` is done in float64 on the host, because doing it in float32
        would add an accumulation error over every row on top of the kernel's own
        rounding. That precision is not free: with many one-vs-one pairs the host
        contraction measured 39–47% of ``predict``. It is therefore overlapped with
        the device, which computes the next tile while the host contracts this one
        (:func:`_combine_pipelined`), a chunk of rows at a time rather than through
        a float64 copy of the whole tile.

        Each query is reduced on its own, so ``block_rows`` changes the result only
        at the level of rounding — a BLAS may choose a different kernel for a block
        of a handful of rows — and, for a given input and blocking, the output is
        deterministic. Queries are range-checked by :func:`check_query_range`, and
        rbf entries the Gram identity may have got wrong are recomputed
        (:class:`_RbfRefiner`), exactly as at fit.
        """
        import mlx.core as mx

        W = np.asarray(weights, dtype=np.float64)
        Q32 = apply_frame(Q, self.center, self.scale)
        out = np.zeros((Q32.shape[0], W.shape[1]), dtype=np.float64)
        if Q32.shape[0] == 0 or rows.shape[0] == 0:
            return out

        device = _compute_device(backend, device)
        right, right_sq, reach, right_host, right_sq_host = self._device_operand(rows, device)
        q_sq = _query_norms(Q32)
        check_query_range(self.spec, q_sq, reach)
        refine = None
        if _RbfRefiner.needed(self.spec, Q32.shape[1], float(q_sq.max()), reach):
            refine = _RbfRefiner(self.spec.gamma, apply_frame(rows, self.center, self.scale))
        if host:
            tile = _host_tile(self.spec, Q32, q_sq, right_host, right_sq_host)
            if refine is not None:
                refine(tile, Q32)
            _contract_tile(tile, W, out)
            return out
        with use_device(device):
            _combine_tiles(
                mx, self.spec, device, Q32, right, right_sq, W, out, block_rows, refine
            )
        return out


# --------------------------------------------------------------------------------------
# device evaluation
# --------------------------------------------------------------------------------------


def _compute_device(backend: str, device: str | None) -> str:
    """The MLX device to evaluate on: ``device``, or else the one ``backend`` names.

    The kernels read no configuration. Where a GPU-fitted model runs in a process
    with no GPU is decided by the estimator that owns it
    (:func:`~mlxlearn._common.device.inference_device`) and passed in as ``device``.
    """
    if device is not None:
        return device
    return "gpu" if backend == "mlx" else "cpu"


def _block_rows(n_rows: int, n_cols: int, device: str, block_rows: int | None) -> int:
    """How many rows of the left operand to evaluate at once on ``device``.

    Derived from the same byte budget the distance kernel uses, halved because a
    block needs both the inner-product tile and the transformed tile alive at the
    same time. A fixed constant would be wrong by an order of magnitude on both an
    8 GB laptop and a 128 GB desktop.
    """
    if block_rows is not None:
        return max(1, min(int(block_rows), max(n_rows, 1)))
    tuning = get_tuning()
    budget = (
        tuning.distance_block_bytes if device == "gpu" else tuning.cpu_distance_block_bytes
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


@functools.lru_cache(maxsize=1)
def _fused_rbf():
    """The rbf branch of :func:`_transform` compiled into one GPU kernel.

    Evaluated eagerly, that branch is six full passes over the tile — broadcast
    add, scale, subtract, clamp, scale, exponentiate — each writing an ``n_a × n_b``
    intermediate. Fused, it is one pass, which measured 1.5–5.7× faster per tile.

    ``gamma`` is an *argument*, a 0-d float32 array, and never a constant captured
    in the traced function. Captured, MLX embeds its own rounding of the Python
    float into the kernel, which differs from the eager path's float32 scalar by
    an ulp for most values — including the default ``gamma="scale"`` — and every
    kernel entry then moves with it. Passed as an array, the fused kernel is
    bit-identical to :func:`_transform`, and it is one compiled function for every
    ``gamma`` rather than one per fit, so this is a cache of *code*, keyed on
    nothing: the "no cache keyed on data" rule is about the other kind.

    ``shapeless=True`` because tile shapes vary with every query batch, and there
    is nothing shape-dependent in the graph to specialize on.
    """
    import mlx.core as mx

    def rbf(dot, a_sq, b_sq, neg_gamma):
        squared = a_sq[:, None] + b_sq[None, :] - 2.0 * dot
        return mx.exp(neg_gamma * mx.maximum(squared, 0.0))

    return mx.compile(rbf, shapeless=True)


def _tile_transform(mx, spec: KernelSpec, device: str):
    """``f(dot, a_sq, b_sq)`` turning an inner-product tile into a kernel tile.

    The fused rbf kernel is used on the GPU only — the device the work runs on,
    not the backend it was fitted for. On MLX's CPU device the same compiled
    graph measured 2.3–2.6× *slower* than the eager ops, and its first call per
    process shells out to a C compiler for about a second — neither of which the
    CPU path, which Layer 1 takes for every problem below the SVC crossover, nor
    a GPU-fitted model loaded on a GPU-less host, should pay for an identical
    answer.
    """
    if spec.kernel == "rbf" and device == "gpu":
        fused = _fused_rbf()
        neg_gamma = mx.array(-spec.gamma, dtype=mx.float32)
        return lambda dot, a_sq, b_sq: fused(dot, a_sq, b_sq, neg_gamma)
    return lambda dot, a_sq, b_sq: _transform(mx, spec, dot, a_sq, b_sq)


#: Rows of a kernel tile contracted per host matmul in :meth:`KernelBasis.combine`.
#: Converting a whole tile to float64 first materializes a twin twice its size
#: (256 MiB for a full-budget tile) and streams it through memory once more than
#: necessary; a few hundred rows at a time keep the float64 copy in cache, and are
#: still enough rows for BLAS to run its full-width kernel.
_HOST_CONTRACT_ROWS = 256


def _contract_tile(tile, weights: np.ndarray, out: np.ndarray, refine=None) -> None:
    """``out[:] = tile @ weights`` in float64, a chunk of rows at a time.

    Blocks until ``tile`` has been evaluated — the only synchronization point in
    the pipelined loops below. ``refine``, when given, corrects a writable host
    copy of the tile first.
    """
    host = np.asarray(tile)
    if refine is not None:
        host = np.array(host, dtype=NUMPY_FLOAT)
        refine(host)
    for start in range(0, host.shape[0], _HOST_CONTRACT_ROWS):
        stop = start + _HOST_CONTRACT_ROWS
        rows = host[start:stop].astype(np.float64)
        # A flag NumPy's BLAS raises on a finite tile is spurious (smo._gradient); a
        # tile that overflowed — a poly power — leaves these values non-finite, for
        # the caller's finiteness check to report.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            out[start:stop] = rows @ weights


def _combine_pipelined(
    mx, transform, n_a: int, rows: int, left, right, b_sq, weights, out, refine=None
):
    """Fold ``K(left, right) @ weights`` into ``out``, overlapping device and host.

    ``left(start, stop)`` returns one block of the left operand and its squared
    norms (``None`` unless rbf). Tile ``i + 1`` is submitted with ``mx.async_eval``
    *before* the host contracts tile ``i``, so the device computes the next tile
    while the host is busy with this one; evaluated serially, the GPU sat idle for
    the contraction, measured at 39–47% of multiclass predict time. ``refine``,
    when given, is ``refine(tile, start, stop)`` and corrects the host copy of the
    tile for rows ``start:stop``.

    Memory: at most three tiles are alive on the device — the one being contracted
    and the next one's inner product and result — where the serial loop held two
    plus a float64 host twin of one, i.e. the equivalent of four. The chunked
    contraction is what pays for the pipelining, so the block size derived from
    the byte budget does not have to shrink.
    """
    right_t = right.T

    def submit(start):
        block, a_sq = left(start, min(start + rows, n_a))
        tile = transform(block @ right_t, a_sq, b_sq)
        mx.async_eval(tile)
        return tile

    tile = submit(0)
    for start in range(0, n_a, rows):
        stop = min(start + rows, n_a)
        following = submit(stop) if stop < n_a else None
        fix = None if refine is None else functools.partial(refine, start=start, stop=stop)
        _contract_tile(tile, weights, out[start:stop], fix)
        # Rebinding releases this tile before the next submission, which is what
        # keeps the device at three live tiles rather than four.
        tile = following


def _combine_tiles(
    mx, spec, device, left32, right, right_sq, weights, out, block_rows, refiner=None
) -> None:
    """``out[:] = K(left32, right) @ weights``: the tile loop behind :meth:`KernelBasis.combine`.

    ``left32`` is a host float32 matrix already in the kernel's frame; ``right``
    and ``right_sq`` (its squared row norms, rbf only) are on the device. Blocked
    by :func:`_block_rows` for ``device``, transformed by :func:`_tile_transform`,
    corrected by ``refiner`` (an :class:`_RbfRefiner` over ``right``'s host rows)
    when given, and pipelined by :func:`_combine_pipelined`, inside the caller's
    :func:`~mlxlearn._common.device.use_device` block for that device.
    """
    n_left = int(left32.shape[0])
    rows = _block_rows(n_left, int(right.shape[0]), device, block_rows)
    needs_norms = spec.kernel == "rbf"

    def left(start, stop):
        block = mx.array(left32[start:stop], dtype=mx.float32)
        return block, (mx.sum(block * block, axis=1) if needs_norms else None)

    refine = None
    if refiner is not None:

        def refine(tile, *, start, stop):
            refiner(tile, left32[start:stop])

    transform = _tile_transform(mx, spec, device)
    _combine_pipelined(
        mx, transform, n_left, rows, left, right, right_sq, weights, out, refine
    )


def _host_tile(
    spec: KernelSpec, left32: np.ndarray, left_sq, right32: np.ndarray, right_sq
) -> np.ndarray:
    """``K(left32, right32)`` in NumPy float32: :func:`_transform`'s arithmetic on the host.

    Operation for operation what a device tile computes — the inner products, the
    rbf squared distance from the same float32 norms (``left_sq`` may be float64
    where float32 could not hold a far query's; it is rounded, overflowing to
    ``inf`` as the device's own sum would), the clamp, and poly's repeated squaring
    — so only the BLAS's rounding of the inner products differs. Returns a
    writable float32 array.
    """
    # Within check_query_range's bound no partial sum overflows, so a flag NumPy's
    # BLAS raises here is spurious (smo._gradient) — and the device's product, which
    # this one stands in for, raises none.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        dot = left32 @ right32.T
    if spec.kernel == "linear":
        return dot
    if spec.kernel == "rbf":
        with np.errstate(over="ignore", invalid="ignore"):
            a_sq = np.asarray(left_sq).astype(NUMPY_FLOAT)
            squared = a_sq[:, None] + right_sq[None, :] - NUMPY_FLOAT(2.0) * dot
            np.maximum(squared, NUMPY_FLOAT(0.0), out=squared)
            squared *= NUMPY_FLOAT(-spec.gamma)
            return np.exp(squared)
    with np.errstate(over="ignore", invalid="ignore"):
        z = NUMPY_FLOAT(spec.gamma) * dot + NUMPY_FLOAT(spec.coef0)
        if spec.kernel == "sigmoid":
            return np.tanh(z)
        return np.asarray(_host_integer_power(z, int(spec.degree)), dtype=NUMPY_FLOAT)


def _host_float32(A) -> np.ndarray:
    """Host-side C-ordered float32 view or copy of ``A``."""
    return np.ascontiguousarray(A, dtype=NUMPY_FLOAT)


def _self_entries(spec: KernelSpec, X32: np.ndarray) -> np.ndarray:
    """:func:`kernel_diagonal` of float32 rows, rounded to float32: every self-entry's value.

    A value past float32's range becomes ``inf``, as the float32 evaluation would
    have made it; the solver reports that as non-finite.
    """
    with np.errstate(over="ignore"):
        return kernel_diagonal(spec, X32).astype(NUMPY_FLOAT)


def _rbf_refiner(spec: KernelSpec, left32: np.ndarray, right32: np.ndarray, right_sq=None):
    """An :class:`_RbfRefiner` over ``right32`` when entries against ``left32`` may need one.

    ``right_sq`` may pass the right rows' squared norms already on hand (float32 is
    enough for the test); the left rows' are computed here.
    """
    if spec.kernel != "rbf" or left32.size == 0 or right32.size == 0:
        return None
    if right_sq is None:
        right_sq = np.einsum("ij,ij->i", right32, right32)
    left_sq = right_sq if left32 is right32 else np.einsum("ij,ij->i", left32, left32)
    if not _RbfRefiner.needed(
        spec, right32.shape[1], float(np.max(left_sq)), float(np.max(right_sq))
    ):
        return None
    return _RbfRefiner(spec.gamma, right32)


def kernel_matrix(
    spec: KernelSpec,
    A: np.ndarray,
    B: np.ndarray | None = None,
    *,
    backend: str = "cpu",
    device: str | None = None,
    block_rows: int | None = None,
) -> np.ndarray:
    """The dense kernel matrix ``K(A, B)``, float32.

    Parameters
    ----------
    spec : KernelSpec
        The compute spec: ``A`` and ``B`` are already in its frame.
    A : ndarray of shape (n_a, n_features)
    B : ndarray of shape (n_b, n_features), optional
        Defaults to ``A``, in which case the diagonal is :func:`kernel_diagonal`'s
        closed form, rounded to float32 — the value the solver's curvature uses.
    backend : {"mlx", "cpu"}, default="cpu"
        Which MLX device evaluates the blocks: the GPU for ``"mlx"``.
    device : {"gpu", "cpu"}, optional
        The MLX device to evaluate on, as the caller resolved it, in place of the
        one ``backend`` names.
    block_rows : int, optional
        Rows of ``A`` per block. Defaults to a memory-derived size.

    Returns
    -------
    ndarray of shape (n_a, n_b), float32

    Notes
    -----
    This is the one function here that does materialize an ``n_a × n_b`` result, so
    callers who cannot afford that should use :class:`KernelBasis` or
    :class:`KernelRows` instead. The *evaluation* is still blocked: the device
    never holds more than one block at a time regardless of the output size. rbf
    entries the Gram identity may have got wrong are recomputed on the host
    (:class:`_RbfRefiner`).
    """
    import mlx.core as mx

    A32 = _host_float32(A)
    B32 = A32 if B is None else _host_float32(B)
    n_a, n_b = A32.shape[0], B32.shape[0]
    out = np.empty((n_a, n_b), dtype=NUMPY_FLOAT)
    if n_a == 0 or n_b == 0:
        return out

    device = _compute_device(backend, device)
    rows = _block_rows(n_a, n_b, device, block_rows)
    needs_norms = spec.kernel == "rbf"
    refiner = _rbf_refiner(spec, A32, B32)

    with use_device(device):
        transform = _tile_transform(mx, spec, device)
        right = mx.array(B32, dtype=mx.float32)
        right_t = right.T
        b_sq = mx.sum(right * right, axis=1) if needs_norms else None
        for start in range(0, n_a, rows):
            end = min(start + rows, n_a)
            block = mx.array(A32[start:end], dtype=mx.float32)
            a_sq = mx.sum(block * block, axis=1) if needs_norms else None
            tile = transform(block @ right_t, a_sq, b_sq)
            mx.eval(tile)
            out[start:end] = np.asarray(tile)
            if refiner is not None:
                refiner(out[start:end], A32[start:end])
    if B is None:
        np.fill_diagonal(out, _self_entries(spec, A32))
    return out


#: See :attr:`KernelRows.prefetch_rows`. 16–64 measured alike at 8 000 × 128;
#: much more fills the cache with rows fetched before the ranking meant anything.
_GPU_PREFETCH_ROWS = 32


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
        Float32 C-ordered copy, already in ``spec``'s compute frame.
    spec : KernelSpec
    backend : {"mlx", "cpu"}
    device : {"gpu", "cpu"} or None
        The MLX device to evaluate on, as the caller resolved it; ``None`` for
        the one ``backend`` names.

    Notes
    -----
    Rows are computed on demand, never all ``n`` of them: a decomposition solver
    touches a scattered subset, and on-demand rows are the only shape of this
    problem that fits in bounded memory at large ``n``.

    What a row costs on the GPU is almost entirely the round trip — one
    synchronous evaluation, about 150–250 µs whether the row is ``n = 2 000`` or
    ``20 000`` long — so :meth:`rows` exists to pay it once for many rows: ``k``
    rows are one ``(n, d) @ (d, k)`` product, measured at 8–9 µs per row for
    ``k = 32`` against 200–250 µs for one at a time.

    :meth:`rows` evaluates every row, a single one included, through that product
    and the same tile transform :func:`kernel_matrix` uses; a lone row is a
    two-column product with itself repeated. A one-column product would be a
    matrix-*vector* product, which the backends reduce in a different order, so a
    row's value would depend on how many other rows were fetched alongside it — and
    a solver prefetching rows would take a different path from one that did not.
    At narrow widths, on the GPU, a row's value does not depend on its batch and
    equals the corresponding row of :func:`kernel_matrix`'s Gram (measured on MLX
    0.32 up to 512 features; neither is a documented property of MLX, so neither
    is promised). At about a thousand features and more MLX picks a different
    matrix-multiply kernel for some shapes — the Gram's, or a batch of 33 rows,
    the GPU prefetch's own — and the two properties fail in the last bits: the
    solver's path then depends on which rows arrived together, and so on
    ``cache_size`` (9.2e-4 in SVC decision values at 2 048 × 2 048). On the CPU
    device a row can differ in the last bit
    whatever the width: the vectorized ``exp``/``tanh`` finish a tile with a
    scalar tail whose rounding differs, and where an entry falls in the tile
    depends on the batch. What is guaranteed on both is that a given request
    returns the same bits every time, and that every self-entry is exactly
    :func:`kernel_diagonal`'s value, rounded to float32.

    :meth:`row` is the one-at-a-time path, and it differs by device. On the GPU it
    is ``rows([index])[0]``, at no extra cost. On the CPU device the per-call
    overhead is small, a batch buys little (13–22 µs per row, against 40–75 µs for
    a matrix-vector product), and the two-column product costs twice the
    matrix-vector one — so there :meth:`row` *is* the matrix-vector product, and
    its last bits can differ from :meth:`rows`'. A solver should therefore draw
    the rows it iterates on from one of the two, on the CPU device; the SVC fit
    uses :meth:`rows` there only to rebuild a gradient in bulk.
    """

    X: np.ndarray
    spec: KernelSpec
    backend: str
    device: str | None = None
    _device_X: object = None  # mx.array (n, d)
    _device_sq: object = None  # mx.array (n,), only for rbf
    _tile: object = None  # f(dot, a_sq, b_sq), from _tile_transform
    _device_kind: str = "cpu"  # the MLX device the rows are evaluated on
    _diagonal: object = None  # (n,) float32, every self-entry
    _refiner: object = None  # _RbfRefiner, or None where no entry can need one

    def __post_init__(self) -> None:
        import mlx.core as mx

        self._device_kind = _compute_device(self.backend, self.device)
        self._diagonal = _self_entries(self.spec, self.X)
        self._refiner = _rbf_refiner(self.spec, self.X, self.X)
        with use_device(self._device_kind):
            data = mx.array(self.X, dtype=mx.float32)
            self._device_X = data
            self._tile = _tile_transform(mx, self.spec, self._device_kind)
            if self.spec.kernel == "rbf":
                self._device_sq = mx.sum(data * data, axis=1)
                mx.eval(data, self._device_sq)
            else:
                mx.eval(data)

    def _finish(self, tile: np.ndarray, index: np.ndarray) -> np.ndarray:
        """Correct host rows ``tile = K(X[index], X)`` in place: refinement, self-entries."""
        if self._refiner is not None:
            self._refiner(tile, self.X[index])
        tile[np.arange(index.shape[0]), index] = self._diagonal[index]
        return tile

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def prefetch_rows(self) -> int:
        """Further rows worth fetching alongside a cache miss, on this device.

        ``32`` on the GPU, where a call costs a round trip however many rows it
        carries: prefetching the near-violators this way measured 1.5–2.4× on rbf
        SVC fits of 8 000–16 000 rows (1.0× at 32 000, where the solver's own
        per-iteration arithmetic dominates instead). ``0`` on the CPU device, where a batched row
        costs a third of a lone one rather than a twentieth: prefetching 32 rows
        measured 0.82–0.87× there, and 8 rows 1.02–1.03×, which does not buy
        the batch-dependent last bits that mixing :meth:`row` and :meth:`rows`
        brings on that device (see Notes).
        """
        return _GPU_PREFETCH_ROWS if self._device_kind == "gpu" else 0

    def row(self, index: int) -> np.ndarray:
        """``K(X[index], X)`` as a float32 array of length ``n_samples``."""
        if self._device_kind == "gpu":
            return self.rows(np.array([index]))[0]

        import mlx.core as mx

        index = int(index)
        if not 0 <= index < self.n_samples:
            raise IndexError(f"row index must lie in [0, {self.n_samples}); got {index}")
        with use_device(self._device_kind):
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
            tile = np.array(np.asarray(out)[None, :], dtype=NUMPY_FLOAT)
        return self._finish(tile, np.array([index]))[0]

    def rows(self, indices) -> np.ndarray:
        """``K(X[indices], X)`` as a float32 array of shape ``(len(indices), n_samples)``.

        One device round trip however many rows are asked for. The result owns
        its memory, so a caller may keep some rows and drop others without the
        ones it keeps pinning a buffer shared with the ones it dropped.
        """
        import mlx.core as mx

        index = np.asarray(indices).reshape(-1)
        count = int(index.shape[0])
        n = self.n_samples
        if count == 0:
            return np.empty((0, n), dtype=NUMPY_FLOAT)
        if not np.issubdtype(index.dtype, np.integer):
            raise TypeError(f"row indices must be integers, got dtype {index.dtype}")
        # An MLX gather does not bounds-check: an index past the end reads
        # whatever is there and returns it as a kernel row.
        if int(index.min()) < 0 or int(index.max()) >= n:
            raise IndexError(f"row indices must lie in [0, {n}); got {index.min()}..{index.max()}")
        # See the class Notes: one row is a two-column product, never a GEMV.
        chosen = np.repeat(index, 2) if count == 1 else index

        with use_device(self._device_kind):
            data = self._device_X
            selected = mx.array(chosen.astype(np.int32))
            a_sq = b_sq = None
            if self.spec.kernel == "rbf":
                b_sq = self._device_sq
                a_sq = b_sq[selected]
            # (n, d) @ (d, k) rather than (k, d) @ (d, n): the same entries, but
            # on the CPU device the latter's reduction order changed with k for
            # small shapes, and batch-independence is the point of this method.
            dot = (data @ data[selected].T).T
            tile = self._tile(dot, a_sq, b_sq)
            mx.eval(tile)
            host = np.array(np.asarray(tile)[:count], dtype=NUMPY_FLOAT, order="C")
        return self._finish(host, index)


def build_kernel_rows(
    X: np.ndarray, spec: KernelSpec, *, backend: str, device: str | None = None
) -> KernelRows:
    """Build a :class:`KernelRows` over a float32 copy of ``X``.

    The copy is deliberate, and forced rather than incidental: the float32
    conversion returns its argument unchanged when it is already float32 and
    C-ordered, so without the explicit copy a row source could alias the caller's
    buffer and change its own answers when the caller mutated their array, mid-fit.
    """
    prepared = _host_float32(X)
    if prepared is X or np.shares_memory(prepared, X):
        prepared = prepared.copy()
    return KernelRows(X=prepared, spec=spec, backend=backend, device=device)
