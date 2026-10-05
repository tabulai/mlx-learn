"""C-support-vector classification, accelerated with MLX.

The estimator subclasses :class:`sklearn.svm.SVC`. That is what makes the drop-in
promise real: the constructor signature, ``get_params``/``set_params``, the
parameter constraints and the estimator tags are inherited rather than restated,
so ``clone``, ``Pipeline``, ``GridSearchCV`` and ``isinstance`` keep working, and
``super().fit(...)`` *is* stock scikit-learn for the patching layer to fall back
to. Layer 1 never calls that fallback — ``_mlxlearn_allow_fallback`` is ``False``,
so a capability mismatch raises.

What is actually solved here is LIBSVM's C-SVC dual, exactly, by the SMO in
``_solvers/smo.py`` — or, for a multiclass rbf fit on the GPU, by the same SMO run for
every one-vs-one pair at once in ``_kernels/_ovo.py``, each pair then held to the host
solver's stop and finished there if it fails. There is no random-feature
approximation, no subsampled training set and no ridge surrogate anywhere in this
path. Those things make a different classifier; shipping one under this class's name
would mean every accuracy number a user measured was measured on something other
than what the name says.

The three places where matching scikit-learn is fiddlier than it looks:

**The one-vs-one subproblem ordering.** LIBSVM builds the subproblem for classes
``(i, j)`` as *all class-i rows in ascending order, then all class-j rows*, labels
``+1`` then ``-1``. It is not the ascending union. Everything downstream —
``support_``'s grouping, ``dual_coef_``'s block layout, and the working-set
tie-breaks that decide which margin points end up with a nonzero α — is stated in
that ordering, so mlxlearn adopts it rather than reindexing afterwards.

**The binary sign flip.** For two classes scikit-learn negates the public
``dual_coef_`` and ``intercept_`` relative to LIBSVM's, and negates
``decision_function`` to match, so that a positive decision value means
``classes_[1]``. For three or more classes it does not. mlxlearn keeps LIBSVM's
convention internally, in ``_pair_weights_``/``_pair_rho_``, and applies the flip
only at the public boundary — one place to be right instead of several.

**``dual_coef_``'s block layout.** Pair ``(i, j)``'s weights are split across two
different *rows* of ``dual_coef_``: the class-i half lands in row ``j-1`` and the
class-j half in row ``i``. Every other entry of those two column ranges stays
exactly zero. The layout is not verifiable by looking at the array, and it is not
verified by mlxlearn's own ``decision_function`` either, which reads
``_pair_weights_`` and never ``dual_coef_`` — a scrambled layout would leave it
untouched. What checks it is evaluating the *public* state: stock LIBSVM's decision
function run on mlxlearn's fitted attributes, a decision function rebuilt from
``dual_coef_`` and ``support_vectors_`` alone, and the multiclass linear ``coef_``
scikit-learn derives from them. ``tests/parity/test_svc.py`` does all three.
"""

from __future__ import annotations

import inspect
import math
import warnings
from typing import ClassVar, NamedTuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC as _SkSVC
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import column_or_1d

from .._common.base import BackendMixin, _public_call, mlx_guard
from .._common.config import get_tuning
from .._common.device import inference_device, physical_memory_bytes
from .._common.validation import (
    _safe_shape,
    check_consistent_length,
    check_float32_range,
    check_is_fitted,
    check_sample_weight,
    compute_class_weight,
    count_labels,
    is_sparse,
    require_dense,
    require_supported_params,
    validate_data,
    validate_float32_input,
)
from .._kernels import _ovo
from .._kernels.kernels import (
    SUPPORTED_KERNELS,
    KernelBasis,
    KernelSpec,
    build_kernel_rows,
    check_frame_resolution,
    frame_spec,
    kernel_diagonal,
    kernel_frame,
    kernel_matrix,
    resolve_gamma,
)
from .._solvers.smo import SMOConfig, finish_c_svc, solve_c_svc
from ..exceptions import (
    CapabilityError,
    InvalidInputError,
    UnsupportedInputError,
    UnsupportedParameterError,
)

__all__ = ["SVC", "MLXSVCMixin"]

_SUPPORTED_SHAPES = ("ovo", "ovr")

#: Kernels patched dispatch never sends to MLX, whatever the size, with the fits that
#: show why (``docs/benchmarks.md``, the ``svc-linear``, ``svc-sigmoid`` and
#: ``svc-poly`` tables). Patched mode sends only ``"rbf"`` to MLX.
_NEVER_ON_MLX = {
    "linear": "every size tried (0.07x-0.10x of LIBSVM)",
    "sigmoid": (
        "C = 10 and 100 at nearly every size tried from 128 features (down to 0.06x "
        "of LIBSVM), and at C = 1 at 1 000 x 256 (0.40x-0.44x)"
    ),
    "poly": (
        "C = 0.5 and 1 on digits upsampled to 256-784 pixels, on five latent directions "
        "with random labels and with gamma above 'scale' (down to 0.46x of LIBSVM)"
    ),
}

#: Of those, the kernels Layer 1 also keeps off the GPU at every size, on MLX's CPU
#: device (:meth:`MLXSVCMixin._below_crossover`). Layer 1 keeps poly on its size
#: rules; only patched mode hands it to LIBSVM (:meth:`MLXSVCMixin._kernel_handoff`).
_BELOW_CROSSOVER_KERNELS = ("linear", "sigmoid")

#: scikit-learn's default ``cache_size``, in MB, read off its signature: patched mode
#: hands a fit with a smaller cache to LIBSVM (:meth:`MLXSVCMixin._kernel_handoff`).
_DEFAULT_CACHE_SIZE = float(inspect.signature(_SkSVC.__init__).parameters["cache_size"].default)

#: The relative tolerance the mean box is compared with: balanced class weights, or
#: sample weights normalized to mean 1, average to one up to rounding, and must leave
#: the box at C (:meth:`MLXSVCMixin._kernel_handoff`).
_BOX_RTOL = 1e-9

#: The largest ``C_t · m`` — a box times its pair's sample count, which bounds every
#: gradient the pair's SMO can reach — the one-vs-one kernel is launched for. Its
#: double-float state and float32 working-set key square gradient differences and
#: divide them by quadratic coefficients as small as LIBSVM's ``tau``, 1e-12; a bound
#: of 2⁴⁰ keeps that inside float32's range. Larger boxes run on the host solver, and
#: so do boxes below 2⁻¹⁰⁰, whose double-float residual would leave float32's normal
#: range, which the GPU flushes to zero.
_OVO_MAX_BOX_ROWS = 2.0**40
_OVO_MIN_BOX = 2.0**-100

#: The smallest ``tol`` the one-vs-one kernel is launched for, relative to
#: ``1 + C_t · m``, the same bound on every gradient. The kernel's stop test reads its
#: double-float gradient, good to about 2⁻⁴⁸ of that scale before its rounding
#: accumulates over the steps; with ``tol`` below what it resolves, a pair never saw its
#: violation under ``tol`` and stepped to ``max_iter``, which the host's finish then
#: reports as exhausted. Measured at ``tol=1e-14``, C = 10, three classes of 200 rows
#: and four features: 10⁷ steps and 58 s on one pair, where the host solver converged
#: in 8 083. 2⁻⁴⁰ leaves that a margin of 2⁸ and the pair's size; a smaller ``tol``
#: runs on the host solver.
_OVO_TOL_FLOOR = 2.0**-40

#: A dual variable counts as free, for :meth:`MLXSVCMixin._solve_pairs_on_gpu`'s bias
#: rule, only this far inside its box, relative to the box: well above the residue a
#: clipped update leaves in double-float (measured up to about 2⁻⁴², 2e-13).
_OVO_FREE_MARGIN = 2.0**-32

#: scikit-learn's message for a non-finite fitted model, verbatim so that code
#: matching on it keeps working, plus what went wrong on the MLX path.
_NON_FINITE_MODEL = (
    "The dual coefficients or intercepts are not finite. The input data may contain "
    "large values and need to be preprocessed. (mlxlearn evaluates the kernel in "
    "float32, where these kernel values overflowed.)"
)

#: Whether the installed scikit-learn deprecates ``probability`` (1.9+), read off the
#: inherited signature rather than a version string: the sentinel is the fact that
#: matters, and it is what scikit-learn's own fit tests.
_PROBABILITY_DEPRECATED = (
    inspect.signature(_SkSVC.__init__).parameters["probability"].default == "deprecated"
)


def _libsvm_sample_weight(sample_weight, X: np.ndarray) -> np.ndarray | None:
    """Validate ``sample_weight`` with LIBSVM's meaning for a negative weight.

    scikit-learn's LIBSVM keeps only the rows with ``W > 0`` (``remove_zero_weight``
    in ``svm.cpp``), so a negative weight is *exactly* a zero weight: stock
    scikit-learn fitted with one and with the other returns bit-identical models.
    The shared validator rejects negatives because a negative weight makes a loss
    such as the logistic one non-convex; that rationale does not reach an SVC box
    constraint, and rejecting them here broke ``patch_sklearn()`` for code stock
    scikit-learn runs. So they are clipped to zero, and ``_fit_ovo`` drops them with
    every other empty box.
    """
    if sample_weight is not None:
        raw = np.asarray(sample_weight, dtype=np.float64)
        if np.any(raw < 0.0):
            # NaN compares false and survives, for the validator to reject.
            sample_weight = np.where(raw < 0.0, 0.0, raw)
    return check_sample_weight(sample_weight, X, dtype=np.float64)


def _has_free(alpha: np.ndarray, C: np.ndarray) -> bool:
    """Whether any dual variable is free by more than :data:`_OVO_FREE_MARGIN` of its box."""
    margin = _OVO_FREE_MARGIN * C
    return bool(np.any((alpha > margin) & (alpha < C - margin)))


def _is_none_or_balanced(class_weight) -> bool:
    return class_weight is None or (isinstance(class_weight, str) and class_weight == "balanced")


def ovr_decision_function(
    predictions: np.ndarray, confidences: np.ndarray, n_classes: int
) -> np.ndarray:
    """Turn one-vs-one votes into one-vs-rest decision values.

    Reimplements the ~15 lines of scikit-learn's ``_ovr_decision_function``, which
    is private and whose rename in any release would make this module
    unimportable. The result is the vote count plus a tie-breaking term built from
    the summed pairwise confidences and scaled into ``(-1/3, 1/3)`` so that it can
    order samples inside a vote level without ever moving one across a level.

    Parameters
    ----------
    predictions : ndarray of shape (n_samples, n_pairs)
        ``0`` when the pair's first class won, ``1`` when the second did.
    confidences : ndarray of shape (n_samples, n_pairs)
        Signed pair confidences, positive for the pair's second class.
    n_classes : int

    Returns
    -------
    ndarray of shape (n_samples, n_classes), float64
    """
    n_samples = predictions.shape[0]
    votes = np.zeros((n_samples, n_classes), dtype=np.float64)
    summed = np.zeros((n_samples, n_classes), dtype=np.float64)
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            summed[:, i] -= confidences[:, k]
            summed[:, j] += confidences[:, k]
            votes[predictions[:, k] == 0, i] += 1
            votes[predictions[:, k] == 1, j] += 1
            k += 1
    return votes + summed / (3 * (np.abs(summed) + 1))


def ovo_vote(decision: np.ndarray, n_classes: int) -> np.ndarray:
    """Class index per row from one-vs-one decision values, LIBSVM's rule.

    Pair ``(i, j)`` votes for ``i`` when its decision value is strictly positive
    and for ``j`` otherwise — note that an exact zero votes for ``j``, which is
    what puts a sample sitting precisely on a binary decision boundary in
    ``classes_[1]``. Ties in the vote count go to the lowest class index, matching
    LIBSVM's strict ``>`` scan and :func:`numpy.argmax`.
    """
    n_samples = decision.shape[0]
    votes = np.zeros((n_samples, n_classes), dtype=np.int64)
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            positive = decision[:, k] > 0
            votes[positive, i] += 1
            votes[~positive, j] += 1
            k += 1
    return votes.argmax(axis=1)


class _KernelProblem(NamedTuple):
    """The kernel side of a fit, resolved and range-checked before anything is solved."""

    spec: KernelSpec  # as the user asked for it; ``gamma`` is scikit-learn's ``_gamma``
    basis: KernelBasis  # the float32 compute frame, with the frame's scale in its spec
    X32: np.ndarray  # the training matrix, mapped into that frame


class MLXSVCMixin(BackendMixin):
    """Backend selection, the one-vs-one fit, and inference for :class:`SVC`."""

    #: Fitted state only the MLX path creates. A refit that ends on scikit-learn
    #: never overwrites it, so without clearing it the previous fit's pair weights
    #: and device-resident support vectors would stay in memory and in every pickle.
    _mlxlearn_mlx_attrs: ClassVar[tuple[str, ...]] = (
        "_pair_weights_",
        "_pair_rho_",
        "_pair_kkt_violation_",
        "_kernel_spec_",
        "_kernel_center_",
        "_kernel_basis_",
    )

    # --------------------------------------------------------------- capability

    def _check_svc_capability(self, X) -> None:
        """Raise :class:`CapabilityError` for anything the MLX path cannot serve.

        Only what is visible without scanning the data: sparsity and parameters.
        The data-dependent limits — float32 range, a ``gamma`` the compute frame
        cannot represent, variation float32 cannot resolve, a class with no
        positive weight — are checked after validation, by :meth:`fit`, and only
        once the problem is known to be headed for MLX. Scanning an array that
        patched mode is about to hand to scikit-learn anyway is pure overhead.
        """
        name = type(self).__name__
        require_dense(X, estimator=name)

        if callable(self.kernel):
            raise UnsupportedParameterError(
                f"{name} does not implement a callable kernel on the MLX path. "
                f"Supported kernels are: {', '.join(repr(k) for k in SUPPORTED_KERNELS)}.",
                parameter="kernel",
                value=self.kernel,
            )
        require_supported_params(
            {
                "kernel": self.kernel,
                "decision_function_shape": self.decision_function_shape,
            },
            {
                "kernel": SUPPORTED_KERNELS,
                "decision_function_shape": _SUPPORTED_SHAPES,
            },
            estimator=name,
        )

        # Platt scaling needs a cross-validated inner fit and its own Newton
        # solver; 0.1.0 ships neither, and an uncalibrated sigmoid fitted on
        # in-sample margins — the shortcut that makes this look cheap — is
        # systematically overconfident. Better to name the supported alternative.
        # scikit-learn 1.9 deprecated `probability` and defaults it to the string
        # sentinel "deprecated", which is truthy. Reading it directly therefore
        # rejected every default-constructed SVC on 1.9 — a version pair
        # pyproject.toml declares supported.
        probability = getattr(self, "probability", False)
        if probability == "deprecated":
            probability = False
        if probability:
            raise UnsupportedParameterError(
                f"{name} does not implement probability=True on the MLX path: "
                "Platt calibration is not part of mlxlearn 0.1.0. Wrap the "
                "estimator in sklearn.calibration.CalibratedClassifierCV to get "
                "calibrated probabilities, or use sklearn.svm.SVC directly.",
                parameter="probability",
                value=probability,
            )

        # A non-positive numeric gamma is checked here rather than deep inside
        # kernel evaluation. Raising it from inside mlx_guard meant a gamma
        # mlxlearn cannot serve propagated straight out of Layer 2 as well, even
        # though scikit-learn accepts gamma=0.0 and would have produced a model.
        # A capability limit has to be discovered while there is still somewhere
        # to fall back to. The linear kernel never reads gamma, so any legal value
        # — scikit-learn's constraint admits only >= 0 — is served there.
        #
        # Only the numeric case is checked, and deliberately without touching X:
        # "scale" and "auto" always resolve to something positive, and reaching
        # into an unvalidated X here would pre-empt the input errors scikit-learn's
        # own validation is supposed to raise.
        if not isinstance(self.gamma, str) and not callable(self.kernel):
            try:
                gamma_value = float(self.gamma)
            except (TypeError, ValueError):
                gamma_value = None
            if gamma_value is not None and gamma_value <= 0.0 and self.kernel != "linear":
                raise UnsupportedParameterError(
                    f"{name} does not implement gamma={gamma_value!r} with "
                    f"kernel={self.kernel!r}; the MLX kernels require a strictly "
                    "positive gamma.",
                    parameter="gamma",
                    value=gamma_value,
                    advice="scikit-learn accepts it, so under patch_sklearn() this "
                    "falls back rather than failing.",
                )

    def _below_crossover(
        self, n_samples: int, n_features: int, n_classes: int | None = None
    ) -> tuple[bool, str]:
        """Whether stock scikit-learn is expected to win at this size.

        The linear and sigmoid kernels are excluded outright, and rbf and poly must
        clear a **width** floor as well as the sample and work floors, because
        width is what decides on isotropic Gaussian data. Patched mode also hands
        poly over at every size, and reads C and ``cache_size``
        (:meth:`_kernel_handoff`), which on data of lower intrinsic dimension decide
        at any width. The calibration is ``docs/benchmarks.md``'s ``svc``,
        ``svc-hard``, ``svc-structured``, ``svc-params``, ``svc-poly``,
        ``svc-cache``, ``svc-sigmoid``, ``svc-linear``, ``svc-narrow``,
        ``svc-multi`` and ``svc-multi-overlap`` tables, measured against
        scikit-learn 1.9.0 and 1.7.2.

        Where the time goes, measured. Up to the row cache's capacity — about
        7 240 rows per one-vs-one subproblem at the default ``cache_size=200`` — a
        subproblem's whole Gram is built by **one** blocked MLX matmul, and the SMO
        loop then runs over it in NumPy without touching the device again. Iteration
        counts match LIBSVM's, so the comparison is per iteration: at 4 000 rows
        with C=10, mlxlearn spent 27-41 µs an iteration from 16 to 256 features,
        nearly flat, and LIBSVM 11 µs at 16 features rising to 103 µs at 256,
        because each kernel row it misses costs ``O(n·d)``, and a few µs once its
        cache holds the rows. On the benchmark's binary target they cross near 100
        features: at 16-64 features rbf and poly lost at C = 10 and 100, while at
        128 features they won at every C tried, 0.1 to 100, from 3 000 rows. But
        the labels decide how many rows become support vectors, and a fit cannot
        see that before it runs: on labels with little or no signal nearly every
        row is one, and at 128 features 3 000-6 000 rows lost at C = 10 and 100
        (0.83x-0.94x), as did one-vs-one pairs of 3 000 x 128 (0.81x-0.91x). From
        256 features every label law measured on Gaussian data won at every C, from
        1 500 x 256 and 1 024 x 512 (1.21x-5.9x) — but not on data of lower intrinsic
        dimension above C = 1 (:meth:`_kernel_handoff`). So the width floor is 256
        features; the work floor, 1 500 x 256, keeps 1 000 x 256 on LIBSVM, and
        ``svc_min_samples`` lets 1 024 x 512 through. Narrow data past the cache, where rows are
        fetched on demand, loses at C = 1 too. A work floor alone cannot see any of
        that: the first release's ``n·d >= 10⁶`` could not tell 16 000 x 64 from
        4 000 x 256.

        C and the kernel move that crossing, through the iteration count: a larger
        box, or a kernel whose iteration count the data's structure can multiply,
        takes many iterations per row over rows LIBSVM has cached, and loses at any
        width (:meth:`_kernel_handoff`). The linear kernel's many cheap iterations are
        exactly the case where
        per-iteration overhead is the whole cost, and it is excluded at every size;
        so is the sigmoid kernel, which won at C = 0.1 and lost at C = 10 and 100 at
        nearly every width, down to 0.06x.

        **Many classes.** One-vs-one solves ``K(K−1)/2`` subproblems of ``2n/K``
        rows on average, each paying its own Gram and its own loop, so a problem
        whose total clears every floor can still be a pile of tiny losses. When
        ``n_classes`` is given, the *mean* pair's work ``(2n/K)·d`` must also
        clear ``svc_min_pair_work``, the binary work floor: on separated blobs
        pairs of 256 x 512 (K=16) ran 0.73x-0.81x and of 500 x 256 (K=32)
        0.70x-0.77x (``svc-multi``), while pairs of 1 500 x 256 and 1 024 x 512 won
        on overlapping classes and on random labels (``svc-multi-overlap``,
        ``svc-hard``). The mean, not the largest
        pair: with imbalanced classes one big pair can hide thousands of tiny ones,
        and it is their count that costs. For two classes the mean pair is the
        whole problem.

        None of this is a property of the objective, which is solved correctly:
        the primal agrees with LIBSVM to 5.2e-07 relative and the KKT conditions
        hold on an independent check. See ``docs/benchmarks.md``.

        Every figure above was measured on the host solver. A multiclass rbf fit on
        the GPU now solves its pairs concurrently (:meth:`_solve_pairs_on_gpu`) — 1.4x
        to 16x the host solver's speed on the fits these rules send to MLX, and
        0.89x-0.96x where every pair ends with no free dual variable and is solved
        again on the host, still 2.3x-2.7x stock's — and the rules have not been
        recalibrated for it: they send to MLX exactly the fits they did before, until
        a calibration on both scikit-learn versions moves them.
        """
        tuning = get_tuning()

        if self.kernel in _BELOW_CROSSOVER_KERNELS:
            return True, (
                f"the {self.kernel} kernel was measured slower on the MLX path at "
                f"{_NEVER_ON_MLX[self.kernel]}, so it is always handed to scikit-learn"
            )

        work = n_samples * n_features
        if (
            n_features < tuning.svc_min_features
            or n_samples < tuning.svc_min_samples
            or work < tuning.svc_min_work
        ):
            return True, (
                f"{n_samples} samples x {n_features} features is below the measured "
                f"SVC crossover ({tuning.svc_min_features} features, "
                f"{tuning.svc_min_samples} samples and {tuning.svc_min_work} "
                "element-ops, all required)"
            )
        if n_classes is not None and n_classes > 2:
            pair_rows = 2.0 * n_samples / n_classes
            if pair_rows * n_features < tuning.svc_min_pair_work:
                return True, (
                    f"{n_classes} classes split {n_samples} samples into "
                    f"{n_classes * (n_classes - 1) // 2} one-vs-one subproblems of "
                    f"{pair_rows:.0f} rows on average, and {pair_rows:.0f} x "
                    f"{n_features} is below the measured per-pair crossover "
                    f"({tuning.svc_min_pair_work} element-ops)"
                )
        return False, ""

    def _kernel_handoff(self, mean_box: float | None = None) -> tuple[str, str] | None:
        """Patched mode's handoff by kernel, C and ``cache_size``: ``(reason, detail)``.

        ``None`` when the fit may run on MLX. Three rules, each read from the
        parameters, and each simple on purpose.

        **Only the rbf kernel goes to MLX.** The linear and sigmoid kernels are
        handed over at every size already (:meth:`_below_crossover`); patched mode
        hands poly over too. Even at C <= 1 it lost on data the other grids did not
        hold — digits upsampled to 256-784 pixels, a few latent directions with
        random labels — and with ``gamma`` above ``"scale"``, down to 0.46x, where
        rbf won on the same data (``docs/benchmarks.md``, table ``svc-poly``). Layer 1
        keeps poly on its size rules, and on the GPU above them.

        **C, through the mean box.** How many SMO iterations a fit takes per row
        decides it — mlxlearn pays tens of microseconds for each, LIBSVM a few once
        its cache holds the rows they revisit — and C, the box every dual variable
        lives in, is what lets them multiply. How much depends on the data's
        intrinsic dimension, which nothing a fit can cheaply see reflects: on
        isotropic Gaussian data every binary fit above the size floors won at C = 10
        and 100 (tables ``svc``, ``svc-hard``); on data of low intrinsic dimension
        they lost at C = 10 (table ``svc-structured``), and at C <= 1 every rbf fit
        won, at every ``gamma`` measured. So a fit needs a box of at most
        ``svc_max_c``: its mean box ``mean_box``, C times the mean class and sample
        weight over the samples it keeps (:meth:`_mean_box`) — C itself without
        weights, with balanced class weights, and with sample weights normalized to
        mean 1. Those average to one only up to rounding, so the comparison allows a
        relative ``1e-9``. ``None`` means it is not known yet: the weights need the
        validated targets, and the box waits for them.

        **A cache smaller than scikit-learn's default**, 200 MB, goes to LIBSVM: rows
        a cache that small cannot hold are evaluated on demand, a device call at a
        time, where LIBSVM pays only its own row cost (table ``svc-cache``). The size
        rules were measured at the default, and a larger cache only helps MLX build
        the whole Gram at once.

        Patched mode only (``_select_backend``'s ``handoff``), as the neighbor
        estimators' width rules are: what it measures is LIBSVM winning, and a
        directly imported class, which has no LIBSVM to hand the fit to, runs the
        same SMO loop on either MLX device, where the GPU builds the Gram faster.
        Recorded as ``"below-crossover"``.
        """
        tuning = get_tuning()
        if self.kernel != "rbf":
            # Only poly and a kernel that is not a string get here: linear and sigmoid
            # are handed over by _below_crossover, and the capability check refuses
            # the rest before either.
            if self.kernel not in _NEVER_ON_MLX:
                return None
            return "below-crossover", (
                f"the {self.kernel} kernel was measured slower on the MLX path at "
                f"{_NEVER_ON_MLX[self.kernel]}, so patched dispatch always hands it to "
                "scikit-learn; only the rbf kernel runs on MLX"
            )
        C = float(self.C)
        if mean_box is not None and not mean_box <= tuning.svc_max_c * (1.0 + _BOX_RTOL):
            what = f"C={C:g}" if math.isclose(mean_box, C, rel_tol=_BOX_RTOL) else (
                f"C={C:g} times the mean class and sample weight, {mean_box:g},"
            )
            return "below-crossover", (
                f"{what} is above the measured SVC crossover for the box constraint "
                f"(at most {tuning.svc_max_c:g}): a larger box lets SMO revisit the "
                "same rows many times over, which LIBSVM does faster once its cache "
                "holds them"
            )
        if not float(self.cache_size) >= _DEFAULT_CACHE_SIZE:
            return "below-crossover", (
                f"cache_size={self.cache_size:g} is below scikit-learn's default of "
                f"{_DEFAULT_CACHE_SIZE:g} MB, where the SVC crossover was measured: kernel "
                "rows a smaller cache cannot hold are evaluated on demand, a device call "
                "at a time, which LIBSVM does faster"
            )
        return None

    def _mean_box(self, y_encoded: np.ndarray, sample_weight: np.ndarray | None) -> float:
        """The mean box ``C · class_weight · sample_weight`` over the samples it keeps.

        :meth:`_sample_box`'s per-sample bounds, averaged over those above zero —
        the samples LIBSVM and :meth:`_fit_ovo` actually solve for. Weights that
        are all ``w`` make it ``C · w``, exactly the problem with that C; balanced
        class weights, and sample weights normalized to mean 1, average to one and
        leave it ``C`` — up to rounding, which :meth:`_kernel_handoff` allows for.
        """
        box = float(self.C) * self.class_weight_[y_encoded]
        if sample_weight is not None:
            box = box * sample_weight
        box = box[box > 0.0]
        return float(np.mean(box)) if box.size else float(self.C)

    def _crossover_from_raw(self, shape: tuple[int, int], y) -> tuple[bool, str]:
        """:meth:`_below_crossover` from the raw input, before any validation.

        The class count only matters once both floors are cleared, so ``y`` is
        read only then, and only as a size hint
        (:func:`~mlxlearn._common.validation.count_labels`, which counts strings
        and Python objects too): labels it cannot read count as two classes, which
        leaves the many-class rule to the check :meth:`fit` repeats after
        validation. Counting them there instead cost a float64 copy and a
        finiteness scan of ``X`` on every many-class fit patched mode hands to
        scikit-learn by design.
        """
        below, detail = self._below_crossover(*shape)
        if below:
            return below, detail
        n_classes = count_labels(y)
        if n_classes is None:
            return below, detail
        return self._below_crossover(*shape, n_classes=n_classes)

    # ------------------------------------------------------------------ targets

    def _validate_svc_targets(self, y) -> np.ndarray:
        """Set ``classes_`` and ``class_weight_``; return integer class codes.

        Reimplements scikit-learn's ``BaseSVC._validate_targets`` from public
        pieces, including its error message for a single class, so code written
        against that contract keeps working.
        """
        y_flat = column_or_1d(y, warn=True)
        check_classification_targets(y_flat)
        classes, encoded = np.unique(y_flat, return_inverse=True)
        # compute_class_weight is what rejects an unknown class_weight string and
        # a dict naming a class that is not in y. The ancestor accepted both
        # silently as all-ones, which turns a typo into a model that quietly
        # ignores the weighting it was asked for.
        self.class_weight_ = np.asarray(
            compute_class_weight(self.class_weight, classes=classes, y=y_flat),
            dtype=np.float64,
        )
        if len(classes) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got "
                f"{len(classes)} class"
            )
        self.classes_ = classes
        return np.asarray(encoded, dtype=np.intp)

    # --------------------------------------------------------------------- fit

    def _kernel_cache_rows(self, n_samples: int) -> int:
        """Kernel rows the LRU may hold for a subproblem of this size.

        ``cache_size`` is scikit-learn's knob and it is in mebibytes, so it maps
        onto the row cache directly: ``rows = bytes / (4 · n_samples)``, rows
        being float32 exactly as LIBSVM caches them.

        Only the *ceiling* comes from tuning. A user who asks for a small cache
        gets a small cache — the cache is a performance knob (it can move where
        inside ``tol`` the solver stops, KKT-optimal either way; see
        ``cache_size``), so honoring the request costs little but time, whereas
        quietly overriding it downward would mean ``cache_size`` had no observable
        effect at all and upward would mean a request for 32 GiB on an 8 GB
        laptop got granted.
        """
        rows = self._kernel_cache_bytes() // (4 * max(n_samples, 1))
        return int(max(2, min(max(n_samples, 2), rows)))

    def _kernel_cache_bytes(self) -> int:
        """The kernel memory a fit may hold at once: ``cache_size``, within the ceiling.

        ``cache_size`` in bytes, capped by ``Tuning.smo_kernel_cache_max_bytes`` and
        ``smo_kernel_cache_fraction`` of installed memory, and never below
        ``smo_kernel_cache_min_bytes`` on a machine that will not report its memory.
        It sizes the host solver's row cache (:meth:`_kernel_cache_rows`) and the
        GPU one-vs-one solver's waves (:meth:`_solve_pairs_on_gpu`).
        """
        tuning = get_tuning()
        budget = int(float(self.cache_size) * 1024 * 1024)
        ceiling = tuning.smo_kernel_cache_max_bytes
        installed = physical_memory_bytes()
        if installed is not None:
            ceiling = min(ceiling, int(installed * tuning.smo_kernel_cache_fraction))
        # A machine that will not report its memory must still get a usable cache.
        ceiling = max(ceiling, tuning.smo_kernel_cache_min_bytes)
        return min(budget, ceiling)

    def _resolve_max_iter(self, n_samples: int) -> int:
        """Turn ``max_iter=-1`` into the concrete budget the solver needs.

        ``-1`` becomes upstream LIBSVM's ``max(10⁷, 100 · n_samples)``. That is a
        deliberate deviation from scikit-learn, whose bundled LIBSVM treats ``-1``
        as unlimited: mlxlearn's SMO iterates in Python at roughly 20 µs a step,
        and a problem scikit-learn needs 7·10⁸ iterations for would run for hours
        rather than stop with a warning. The ConvergenceWarning names the budget.

        ``max_iter=0`` is a legal scikit-learn value — its constraint is
        ``Interval(Integral, -1, None, closed="left")`` — and LIBSVM honors it,
        returning an unfitted-looking model with no support vectors. Only ``-1``
        means "no limit".

        The comparison here is ``>= 0`` for that reason. It was ``> 0``, which
        sent ``max_iter=0`` down the unlimited branch: the user asked for zero
        iterations and got 462, silently. Under ``patch_sklearn()`` that reached
        code expecting scikit-learn's behavior, which is exactly the class of
        quiet substitution this library exists to avoid.
        """
        if self.max_iter >= 0:
            return int(self.max_iter)
        tuning = get_tuning()
        return max(tuning.smo_min_iter_budget, tuning.smo_max_iter_factor * n_samples)

    def _prepare_kernel_problem(self, X64: np.ndarray) -> _KernelProblem:
        """Resolve the kernel and map the training data into its float32 frame.

        Everything that can make this data unservable in float32 surfaces here, as
        a :class:`CapabilityError`, before a single subproblem is solved — which is
        what lets :meth:`fit` still hand the problem to scikit-learn in patched
        mode: a range the frame cannot hold, and variation an untranslated kernel's
        float32 inner product cannot resolve
        (:func:`~mlxlearn._kernels.kernels.check_frame_resolution`). See
        :func:`~mlxlearn._kernels.kernels.kernel_frame` for the frame.
        """
        gamma = resolve_gamma(self.gamma, X64, kernel=self.kernel)
        spec = KernelSpec(
            kernel=self.kernel,
            gamma=gamma,
            degree=int(self.degree),
            coef0=float(self.coef0),
        )
        center, scale = kernel_frame(spec, X64)
        basis = KernelBasis(frame_spec(spec, scale), center=center, scale=scale)
        X32 = basis.prepare(X64)
        check_frame_resolution(basis.spec, X32)
        return _KernelProblem(spec=spec, basis=basis, X32=X32)

    def _solve_pair(self, X_pair, y_pair, C_pair, spec, backend):
        """Solve one one-vs-one subproblem.

        The full Gram is precomputed only when the row cache was already sized to
        hold every row — at which point the memory has been accounted for and one
        blocked matmul beats ``n`` separate row evaluations. Otherwise rows are
        evaluated on demand and the LRU bounds what is resident, so a subproblem
        larger than the cache never needs an ``n × n`` matrix to exist.

        ``_fit_ovo`` passes rows already mapped into the compute frame, with the
        frame folded into ``spec``. The diagonal is the closed form on exactly the
        float32 values the rows are built from, and both row sources write it into
        every self-entry, so the solver's curvature and its gradient updates agree.
        """
        n_pair = int(X_pair.shape[0])
        cache_rows = self._kernel_cache_rows(n_pair)
        k_diag = kernel_diagonal(spec, X_pair)
        # Resolved here, where the configuration may be read; the kernels read none.
        device = inference_device(backend)

        fetch_rows = None
        prefetch_rows = 0
        if cache_rows >= n_pair:
            gram = kernel_matrix(spec, X_pair, backend=backend, device=device)
            # Rows of a materialized Gram are views, so the LRU adds no memory here.
            fetch = gram.__getitem__
        else:
            rows = build_kernel_rows(X_pair, spec, backend=backend, device=device)
            fetch = rows.row
            # Many rows per device round trip: a miss prefetches the near-violators
            # (on the GPU, where a call costs ~200 us however many rows it carries),
            # and shrinking rebuilds its gradient in bulk.
            fetch_rows = rows.rows
            prefetch_rows = rows.prefetch_rows

        config = SMOConfig(
            tol=float(self.tol),
            max_iter=self._resolve_max_iter(n_pair),
            cache_rows=max(2, min(cache_rows, n_pair)),
            prefetch_rows=prefetch_rows,
            shrinking=bool(self.shrinking),
        )
        return solve_c_svc(y_pair, C_pair, k_diag, fetch, config, fetch_rows=fetch_rows)

    def _sample_box(self, y_encoded, sample_weight) -> np.ndarray:
        """Each sample's upper bound ``C_t``; raise if a class has none that is positive.

        ``C_t = C · class_weight[class(t)] · sample_weight[t]``, exactly how
        scikit-learn's LIBSVM builds its per-sample box. A sample whose box is
        ``[0, 0]`` is dropped by :meth:`_fit_ovo`, as LIBSVM drops it.

        A class whose every sample is dropped leaves one-vs-one pairs with one
        side empty. LIBSVM fits the remaining classes anyway — a model whose
        ``classes_`` still names the dropped class and whose ``decision_function``
        then fails — and scikit-learn raises for two classes or a zero
        ``class_weight``. mlxlearn builds no such model, and this is a
        :class:`CapabilityError` rather than an input error because scikit-learn
        can serve some of these requests: :meth:`fit` checks it before any device
        work, so Layer 1 raises and patched mode hands the fit to scikit-learn,
        whatever scikit-learn then makes of it. It raised ``InvalidInputError``
        from inside the fit, which patched mode could not act on.
        """
        C_all = float(self.C) * self.class_weight_[y_encoded]
        if sample_weight is not None:
            C_all = C_all * sample_weight
        active = C_all > 0.0
        if not active.all():
            for c in range(len(self.classes_)):
                if not active[y_encoded == c].any():
                    raise UnsupportedInputError(
                        f"Every sample of class {self.classes_[c]!r} has an effective "
                        "weight of zero, so mlxlearn cannot fit that class. Remove it "
                        "from y or give it a positive sample_weight or class_weight.",
                        reason="empty-class",
                        advice="Or run under patch_sklearn(), which hands such a fit to "
                        "scikit-learn.",
                    )
        return C_all

    def _ovo_solver(self, class_rows, C_all, backend):
        """The GPU one-vs-one solver for this fit, with its device, or ``None``.

        ``(solver, device)`` when every pair can run in the kernel of
        ``_kernels/_ovo.py``: three classes or more, the rbf kernel, a positive
        ``max_iter``, work resolved to the GPU on an MLX release custom kernels may
        run on and ``Tuning.metal_kernels`` on (:func:`~._kernels._ovo.pair_solver`),
        every pair of at most :data:`~._kernels._ovo.MAX_PAIR_ROWS` samples whose
        Gram fits the kernel cache (:meth:`_kernel_cache_bytes`) — the pairs the host
        solver would have materialized — boxes the kernel's state can hold
        (:data:`_OVO_MAX_BOX_ROWS`), and a ``tol`` it can resolve
        (:data:`_OVO_TOL_FLOOR`). Anything else is ``None``, and the host solver runs
        every pair, as before. The kernel compiles here, last, so a fit the other
        rules refuse never builds it.
        """
        if len(class_rows) < 3 or self.kernel != "rbf" or self.max_iter == 0:
            return None
        device = inference_device(backend)
        if device != "gpu":
            return None
        counts = sorted(rows.shape[0] for rows in class_rows)
        largest = counts[-1] + counts[-2]
        if largest > _ovo.MAX_PAIR_ROWS or 4 * largest * largest > self._kernel_cache_bytes():
            return None
        box = C_all[C_all > 0.0]
        scale = float(box.max()) * largest
        if not (box.min() >= _OVO_MIN_BOX and scale <= _OVO_MAX_BOX_ROWS):
            return None
        if float(self.tol) < _OVO_TOL_FLOOR * (1.0 + scale):
            return None
        solver = _ovo.pair_solver(device)
        return None if solver is None else (solver, device)

    def _solve_pairs_on_gpu(self, class_rows, pairs, C_all, backend, problem):
        """Every one-vs-one subproblem at once, on the GPU; ``None`` where it cannot run.

        The pairs run in the kernel of ``_kernels/_ovo.py``, in waves whose Grams fit
        :meth:`_kernel_cache_bytes` together, and every pair's result is held to the
        host solver's stop by :func:`~._solvers.smo.finish_c_svc`: gradient
        recomputed in float64, full-set KKT test, and the host loop, warm-started,
        finishing a pair that fails it. Two kinds of pair are solved again from
        scratch by :meth:`_solve_pair`, as every pair is without the kernel: one whose
        state went non-finite on the device, and one with no free dual variable —
        every one at a bound, up to rounding. Such a pair's bias is not determined
        by the problem: any value in an interval is optimal, and LIBSVM reads it off
        whichever sample its own rounding left a hair inside a bound. Measured on a
        pair of a 9-class fit at C = 0.19 with balanced class weights: LIBSVM and the
        host loop took the bias, 0.46, from a sample 3.5e-15 below its box, and the
        double-float state, whose rounding left a different sample 4e-14 below its
        box, gave -0.38 — two KKT points, decision values 0.85 apart. The host loop
        rounds as LIBSVM does, so it gives LIBSVM's. Returns one
        :class:`~._solvers.smo.SMOState` per pair, in ``pairs``' order, and records
        on the fit's event how many pairs the kernel solved, the host finished, and
        :meth:`_solve_pair` solved again, for each reason.

        The kernel and its finish do not shrink: they take every sample at every
        step, whatever ``shrinking`` says — a speed heuristic of the host loop, on
        which neither the optimum nor the stopping test depends. A pair solved again
        by :meth:`_solve_pair` is the host loop's, ``shrinking`` included, so that it
        is the host path's pair bit for bit.
        """
        plan = self._ovo_solver(class_rows, C_all, backend)
        if plan is None:
            return None
        solver, device = plan
        spec = problem.basis.spec
        rows = _ovo.class_rows(problem.X32, class_rows)
        k_diag = kernel_diagonal(spec, problem.X32)
        sizes, boxes, diagonals, budgets, members = [], [], [], [], []
        for i, j in pairs:
            index = np.concatenate([class_rows[i], class_rows[j]])
            n_i = class_rows[i].shape[0]
            sizes.append(index.shape[0])
            boxes.append(C_all[index])
            diagonals.append(k_diag[index])
            budgets.append(self._resolve_max_iter(index.shape[0]))
            members.append(n_i)
        width = _ovo.threadgroup_width(max(sizes))
        tol = float(self.tol)
        states: list = [None] * len(pairs)
        finished = non_finite = no_free = launches = 0
        bounds = _ovo.waves(sizes, self._kernel_cache_bytes())
        entries = np.asarray(sizes, dtype=np.int64) ** 2
        capacity = max(int(entries[start:end].sum()) for start, end in bounds)
        for start, end in bounds:
            wave = solver.solve(
                spec, rows, pairs[start:end], boxes[start:end], diagonals[start:end],
                budgets[start:end], tol=tol, width=width, device=device, capacity=capacity,
            )
            launches += wave.launches
            for k in range(start, end):
                at = k - start
                if wave.stop[at] == _ovo.STOP_NON_FINITE:
                    non_finite += 1
                    continue
                config = SMOConfig(
                    tol=tol, max_iter=budgets[k], cache_rows=max(2, sizes[k]), shrinking=False
                )
                y_pair = np.where(np.arange(sizes[k]) < members[k], 1.0, -1.0)
                state = finish_c_svc(
                    y_pair, boxes[k], diagonals[k], wave.grams[at], wave.pair_alpha(at),
                    int(wave.n_iter[at]), config,
                )
                if state.stop_reason == "non-finite":
                    non_finite += 1
                elif not _has_free(state.alpha, boxes[k]):
                    no_free += 1
                else:
                    finished += int(state.n_iter != wave.n_iter[at])
                    states[k] = state
            # The wave's device buffers go with it, before anything else is built; and
            # the rows kept on the device go before the host solver builds its own.
            del wave
            if any(states[k] is None for k in range(start, end)):
                rows.release()
            for k in range(start, end):
                if states[k] is None:
                    i, j = pairs[k]
                    index = np.concatenate([class_rows[i], class_rows[j]])
                    states[k] = self._solve_pair(
                        np.ascontiguousarray(problem.X32[index]),
                        np.where(np.arange(sizes[k]) < members[k], 1.0, -1.0),
                        C_all[index], spec, backend,
                    )
        self._amend_fit_event(
            ovo_kernel_pairs=len(pairs),
            ovo_host_finished=finished,
            ovo_resolved_non_finite=non_finite,
            ovo_resolved_no_free=no_free,
            ovo_waves=len(bounds),
            ovo_launches=launches,
        )
        return states

    def _fit_ovo(self, X64, y_encoded, C_all, backend, problem) -> None:
        """Fit every one-vs-one subproblem and assemble LIBSVM's model layout.

        ``C_all`` is :meth:`_sample_box`'s per-sample box. On the GPU the pairs run
        concurrently where the one-vs-one kernel serves the fit
        (:meth:`_solve_pairs_on_gpu`); otherwise, and always elsewhere, one after
        another on the host solver.
        """
        n_samples, n_features = X64.shape
        n_classes = int(len(self.classes_))
        X32 = problem.X32
        compute_spec = problem.basis.spec

        # A sample whose box is [0, 0] can never leave the lower bound, but it is
        # still a member of I_low, so its gradient keeps feeding the stopping
        # criterion and the solver runs to max_iter without ever being able to fix
        # the violation it is reporting -- measured: 10^7 iterations and a model
        # 16% worse than scikit-learn's. scikit-learn's LIBSVM drops rows whose
        # weight is not positive before solving, so mlxlearn drops them too, on
        # the effective box rather than on the raw weight so that
        # class_weight={c: 0} is handled the same way.
        active = C_all > 0.0
        class_rows = [np.flatnonzero((y_encoded == c) & active) for c in range(n_classes)]
        pairs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
        n_pairs = len(pairs)

        pair_rows: list[np.ndarray] = []
        pair_coef: list[np.ndarray] = []
        rho = np.empty(n_pairs, dtype=np.float64)
        n_iter = np.empty(n_pairs, dtype=np.int32)
        violation = np.empty(n_pairs, dtype=np.float64)
        fit_status = 0
        exhausted_budget = 0
        solved = self._solve_pairs_on_gpu(class_rows, pairs, C_all, backend, problem)

        for k, (i, j) in enumerate(pairs):
            rows_i = class_rows[i]
            rows_j = class_rows[j]
            index = np.concatenate([rows_i, rows_j])
            y_pair = np.concatenate(
                [np.ones(rows_i.shape[0]), -np.ones(rows_j.shape[0])]
            )
            if solved is None:
                state = self._solve_pair(
                    np.ascontiguousarray(X32[index]), y_pair, C_all[index], compute_spec,
                    backend,
                )
            else:
                state = solved[k]
            if state.stop_reason == "non-finite":
                # The solver saw an infinite or NaN kernel value or gradient, so
                # there is no model to return; see the backstop below.
                raise InvalidInputError(_NON_FINITE_MODEL)
            pair_rows.append(index)
            # LIBSVM folds the label into the dual coefficient before storing it.
            pair_coef.append(state.alpha * y_pair)
            rho[k] = state.rho
            n_iter[k] = state.n_iter
            violation[k] = state.max_kkt_violation
            if not state.converged:
                fit_status = 1
                exhausted_budget = max(exhausted_budget, self._resolve_max_iter(index.shape[0]))

        # A row is a support vector if ANY pair gave it a nonzero coefficient, and
        # LIBSVM's test is exactly alpha != 0 -- not a threshold. The box branches
        # clip to 0 exactly, so the comparison is meaningful.
        is_support = np.zeros(n_samples, dtype=bool)
        for index, coef in zip(pair_rows, pair_coef, strict=True):
            is_support[index] |= np.abs(coef) > 0.0

        support_parts = [rows[is_support[rows]] for rows in class_rows]
        n_support = np.array([part.shape[0] for part in support_parts], dtype=np.int32)
        support = (
            np.concatenate(support_parts).astype(np.int32)
            if n_support.sum()
            else np.empty(0, dtype=np.int32)
        )
        offsets = np.concatenate([[0], np.cumsum(n_support)]).astype(np.intp)
        n_sv = int(n_support.sum())

        pair_weights = np.zeros((n_sv, n_pairs), dtype=np.float64)
        for k, (i, j) in enumerate(pairs):
            coef = pair_coef[k]
            count_i = class_rows[i].shape[0]
            pair_weights[offsets[i] : offsets[i + 1], k] = coef[:count_i][
                is_support[class_rows[i]]
            ]
            pair_weights[offsets[j] : offsets[j + 1], k] = coef[count_i:][
                is_support[class_rows[j]]
            ]

        dual_coef = np.zeros((max(n_classes - 1, 1), n_sv), dtype=np.float64)
        for k, (i, j) in enumerate(pairs):
            dual_coef[j - 1, offsets[i] : offsets[i + 1]] = pair_weights[
                offsets[i] : offsets[i + 1], k
            ]
            dual_coef[i, offsets[j] : offsets[j + 1]] = pair_weights[
                offsets[j] : offsets[j + 1], k
            ]

        # scikit-learn's own post-fit check, with its message verbatim so code that
        # matches on it keeps working. A kernel that overflowed float32 produced
        # NaN gradients, the solver found no eligible working pair and reported
        # convergence after one iteration, and the fit returned a NaN intercept
        # with fit_status_=0 -- a model predicting one class for everything,
        # silently. The solver now reports that as stop_reason="non-finite",
        # raised above; this stays as the backstop for anything non-finite that
        # reaches the assembled model. Checked before any attribute is assigned, so
        # a failed fit leaves no half-populated estimator behind. InvalidInputError,
        # a ValueError, because this runs inside mlx_guard, which reports a plain
        # ValueError as an MLX failure.
        if not (np.isfinite(rho).all() and np.isfinite(dual_coef).all()):
            raise InvalidInputError(_NON_FINITE_MODEL)

        self.support_ = support
        self.support_vectors_ = X64[support] if n_sv else np.empty((0, n_features))
        self._n_support = n_support
        self.shape_fit_ = (int(n_samples), int(n_features))
        self.fit_status_ = fit_status
        self.n_iter_ = n_iter
        self._num_iter = n_iter
        self._gamma = problem.spec.gamma
        self._sparse = False
        self._probA = np.empty(0)
        self._probB = np.empty(0)

        # LIBSVM's convention throughout the internals: decision = K @ w - rho.
        self._pair_weights_ = pair_weights
        self._pair_rho_ = rho
        self._pair_kkt_violation_ = violation
        self._kernel_spec_ = problem.spec
        # Read only by a model pickled by 0.1.0a2 (see _kernel_basis); the basis
        # carries the frame, scale included.
        self._kernel_center_ = problem.basis.center
        # Owns the support vectors' device copy for inference. Created here, not on
        # the first predict, so predict never changes the estimator's __dict__; the
        # device tensors inside it are built lazily and never pickled.
        self._kernel_basis_ = problem.basis

        self._dual_coef_ = dual_coef
        self._intercept_ = -rho
        if n_classes == 2:
            self.dual_coef_ = -dual_coef
            self.intercept_ = rho.copy()
        else:
            self.dual_coef_ = dual_coef
            self.intercept_ = -rho

        if fit_status:
            if self.max_iter == -1:
                limit = (
                    f"max_iter=-1, which mlxlearn resolves to a hard budget of "
                    f"{exhausted_budget} iterations per one-vs-one subproblem; unlike "
                    "scikit-learn's bundled LIBSVM it does not treat -1 as unlimited"
                )
                advice = "Pass a larger explicit max_iter to keep iterating, or consider"
            else:
                limit = f"max_iter={self.max_iter}"
                advice = "Consider"
            warnings.warn(
                f"Solver terminated early ({limit}). {advice} pre-processing your "
                "data with StandardScaler or MinMaxScaler.",
                ConvergenceWarning,
                stacklevel=2,
            )

    def _warn_sklearn_deprecations(self) -> None:
        """Emit the FutureWarnings scikit-learn's own ``fit`` emits for these params.

        Called only on the MLX path — scikit-learn's fallback warns for itself —
        so a user gets the same advance notice of a removal whichever path ran.
        Layer 1 otherwise never showed it at all.
        """
        if _PROBABILITY_DEPRECATED and self.probability != "deprecated":
            warnings.warn(
                "The `probability` parameter was deprecated in 1.9 and will be removed "
                "in version 1.11. Use `CalibratedClassifierCV(SVC(), ensemble=False)` "
                "instead of `SVC(probability=True)`",
                FutureWarning,
                stacklevel=3,
            )

    # --------------------------------------------------------------- inference

    def _kernel_basis(self) -> KernelBasis:
        basis = getattr(self, "_kernel_basis_", None)
        if basis is not None:
            return basis
        # A model pickled by mlxlearn 0.1.0a2, which predates the compute frame. Its
        # centre and unscaled spec still describe a valid frame exactly.
        return KernelBasis(self._kernel_spec_, center=self._kernel_center_, scale=1.0)

    def _pairwise_decision(self, X, operation: str = "decision_function") -> np.ndarray:
        """One-vs-one decision values in LIBSVM's convention, ``(n, n_pairs)``.

        Positive means the pair's *first* class. This is the single quantity every
        public inference method is derived from, so ``predict`` and
        ``decision_function`` cannot disagree about which side of a boundary a
        sample is on.

        Queries are range-checked before they reach float32, as ``fit`` checks its
        input. Unchecked, a 1e39 query became ``inf``, the decision values NaN,
        and ``ovo_vote`` — for which ``NaN > 0`` is false — silently flipped the
        label. Inference cannot fall back (the backend is sticky), so a query
        float32 cannot represent, or whose inner product with a support vector it
        cannot (:func:`~mlxlearn._kernels.kernels.check_query_range`), raises
        :class:`~mlxlearn.exceptions.UnsupportedInputError` in both layers. A far
        query that stays within those bounds — a ``1e20`` fill value — is
        answered, as scikit-learn answers it; a non-finite decision value is the
        backstop for what they do not cover, a poly power that overflows.
        """
        if is_sparse(X):
            # scikit-learn's own error for a dense-fitted model, class and message.
            raise ValueError(
                f"cannot use sparse input in {type(self).__name__!r} trained on dense data"
            )
        # float32 queries are kept as float32: the frame is applied in float64 block
        # by block anyway, and float32 -> float64 -> float32 is exact. A float64
        # query is range-checked after validation, one min/max pass.
        Xv = validate_float32_input(
            self,
            X,
            operation=operation,
            dtype=(np.float64, np.float32),
            order="C",
            reset=False,
        )
        # The fitted backend, sticky. A GPU-fitted model in a process without a
        # usable GPU runs the same MLX code on MLX's CPU (recorded as
        # "gpu-unavailable"), unless device="gpu" names the GPU explicitly. That is
        # inference_device()'s rule, applied here and handed to the kernels, which
        # read no configuration.
        # A small batch is computed on the host, from the same float32 operand
        # (KernelBasis.combine's host path): what the device charges for it is not
        # arithmetic but the call, 0.64-0.98 ms once the GPU has idled, against a
        # stock single-row predict of 0.22-0.82 ms.
        work = Xv.shape[0] * self.support_vectors_.shape[0] * Xv.shape[1]
        host = work <= get_tuning().svc_host_inference_work
        backend = self._note_inference(
            operation,
            shape=_safe_shape(Xv),
            **({"host": True, "compute": "numpy-float32"} if host else {}),
        )
        n_pairs = self._pair_weights_.shape[1]
        if Xv.shape[0] == 0:
            return np.empty((0, n_pairs), dtype=np.float64)
        # The host path still builds the device operand once, the first time, and
        # reads it through zero-copy views; built on MLX's CPU, it needs no GPU.
        device = "cpu" if host else inference_device(backend)
        with mlx_guard(type(self).__name__, operation):
            decision = self._kernel_basis().combine(
                Xv, self.support_vectors_, self._pair_weights_, backend=backend,
                device=device, host=host,
            )
        decision -= self._pair_rho_
        if not np.isfinite(decision).all():
            raise UnsupportedInputError(
                "The decision values for these queries are not finite: the "
                f"{self.kernel!r} kernel overflowed float32, where mlxlearn evaluates "
                "it. Rescale the queries (or the training data) so the kernel values "
                "stay below ~3.4e38.",
                reason="float32-overflow",
            )
        return decision


class SVC(MLXSVCMixin, _SkSVC):
    """C-support-vector classification, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.svm.SVC` for dense data and the four
    built-in kernels. The dual is solved exactly, by LIBSVM's sequential minimal
    optimization with second-order working-set selection — the same objective, the
    same stopping criterion and the same per-sample box that scikit-learn solves.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter, the upper bound on every dual variable before
        class and sample weights scale it.
    kernel : {"linear", "poly", "rbf", "sigmoid"}, default="rbf"
        ``"precomputed"`` and callable kernels are not implemented on the MLX
        path and raise :class:`~mlxlearn.exceptions.UnsupportedParameterError`.
        Under ``patch_sklearn()`` only ``"rbf"`` is sent to MLX; the other three
        were measured slower there and always go to scikit-learn.
    degree : int, default=3
        Degree of the polynomial kernel. Evaluated by repeated squaring, so a
        negative base raised to an odd degree gives a negative value rather than
        NaN.
    gamma : {"scale", "auto"} or float, default="scale"
        ``"scale"`` is ``1 / (n_features * X.var())``, ``"auto"`` is
        ``1 / n_features``, both resolved exactly as scikit-learn resolves them —
        including its fallback to ``1.0`` only for an exactly-zero variance. The
        linear kernel does not read it, so ``gamma=0.0`` is accepted there.
    coef0 : float, default=0.0
    shrinking : bool, default=True
        LIBSVM's shrinking heuristic, honored by the host solver: samples pinned
        at a bound that cannot form a violating pair are set aside periodically,
        so iterations run over the active set only, and the full gradient is
        rebuilt before the solver may stop. The stopping test always covers every
        sample, so the optimum is the same either way. As in LIBSVM, the path to
        it can differ within ``tol`` — the support-vector set at the margin and
        the last digits of the decision values can move. The GPU's one-vs-one
        kernel (see Notes) and the host's finish of its pairs do not shrink,
        whatever this says: it is a speed heuristic, and nothing about the optimum
        or the stop depends on it. A pair the host solver solves again from
        scratch there honors it, as on the host path.
    probability : bool, default=False
        ``True`` is not implemented; see Notes. On scikit-learn 1.9, where the
        parameter is deprecated and defaults to the ``"deprecated"`` sentinel,
        passing it explicitly emits scikit-learn's FutureWarning on the MLX path
        too.
    tol : float, default=1e-3
        Stopping tolerance on the maximal KKT violation.
    cache_size : float, default=200
        Kernel cache in MiB. Mapped onto the solver's LRU row cache:
        ``rows = cache_size · 2²⁰ / (4 · n_samples)``, clamped by installed memory.

        **Unlike scikit-learn, this is not purely a performance knob.** When the
        budget is large enough to hold every row, mlxlearn materializes the whole
        Gram in one blocked matmul instead of evaluating rows on demand, and the
        two can round differently. On the MLX GPU they agreed bit for bit at
        narrow widths (rbf, poly, linear and sigmoid at 2 000 × 10, ``cache_size``
        0.5 against 2000) — a property of MLX's matrix multiply, not a promise —
        but not at a thousand features and more, where MLX picks another kernel
        for some shapes: at 2 048 × 2 048 and 3 000 × 4 096, ``cache_size=5``
        moved the decision values by up to 9.2e-04. On MLX's CPU device, which
        Layer 1 uses below the crossover, they accumulate in different orders at
        any width: the 2 000 × 10 comparison moved decision values by up to
        2.2e-03 and changed the iteration count. Every result is KKT-optimal and
        within the tolerance in ``docs/fp32_policy.md``. Pin ``cache_size``
        explicitly, on either device, if you need bit-reproducible fits.

        Under ``patch_sklearn()`` a ``cache_size`` below the default of 200 hands
        the fit to scikit-learn: rows a smaller cache cannot hold are evaluated on
        demand, a device call at a time, and LIBSVM was faster there.

        On the GPU's one-vs-one kernel (see Notes) it bounds the pairs' Grams held
        on the device at once: pairs run in waves whose Grams fit it together, and
        any value that holds the largest pair's Gram gives the same model, bit for
        bit.
    class_weight : dict or "balanced", default=None
    verbose : bool, default=False
        Accepted and ignored.
    max_iter : int, default=-1
        ``-1`` resolves to upstream LIBSVM's budget, ``max(10⁷, 100 · n)`` per
        one-vs-one subproblem of ``n`` rows. **This deviates from scikit-learn**,
        whose bundled LIBSVM treats ``-1`` as unlimited: mlxlearn's SMO iterates in
        Python, and uncapped it would turn a fit scikit-learn finishes in a minute
        into hours. A fit that stops on the budget sets ``fit_status_=1`` and names
        the budget in its ConvergenceWarning; pass a larger explicit ``max_iter``
        to keep going. The GPU's one-vs-one kernel spends the same budget per pair,
        its steps and the host's finish together.
    decision_function_shape : {"ovo", "ovr"}, default="ovr"
        Read at call time, not frozen at fit time, matching scikit-learn. Ignored
        for two classes, which always return a 1-D decision function.
    break_ties : bool, default=False
    random_state : int, RandomState or None, default=None
        Unused: this solver is deterministic and 0.1.0 has no probability path.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    support_ : ndarray of shape (n_SV,), int32
        Training indices of the support vectors, grouped by class and ascending
        within each class.
    support_vectors_ : ndarray of shape (n_SV, n_features), float64
    n_support_ : ndarray of shape (n_classes,), int32
    dual_coef_ : ndarray of shape (n_classes - 1, n_SV), float64
        LIBSVM's one-vs-one block layout, negated for the binary case exactly as
        scikit-learn negates it.
    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,), float64
    class_weight_ : ndarray of shape (n_classes,), float64
    n_iter_ : ndarray of shape (n_pairs,), int32
        SMO updates performed per one-vs-one subproblem: on the GPU's one-vs-one
        kernel, its steps plus any the host took to finish the pair.
    fit_status_ : int
        ``0`` if every one-vs-one subproblem converged, ``1`` if any hit
        ``max_iter`` — which, as in scikit-learn, also raises a
        :class:`~sklearn.exceptions.ConvergenceWarning`.
    shape_fit_ : tuple of int
    n_features_in_ : int
    execution_backend_ : {"mlx", "cpu"}

    Notes
    -----
    **Not implemented in 0.1.0.** ``kernel="precomputed"``, callable kernels,
    sparse ``X``, and ``probability=True``. Each raises a
    :class:`~mlxlearn.exceptions.UnsupportedParameterError` or
    :class:`~mlxlearn.exceptions.SparseInputError` naming what is missing; none of
    them silently computes something else. For probabilities, wrap this estimator
    in :class:`sklearn.calibration.CalibratedClassifierCV`, which cross-validates
    the calibration instead of fitting a sigmoid on in-sample margins.

    **Precision.** Kernel entries are float32 and the dual variables, gradients
    and working-set arithmetic are float64 — the split LIBSVM uses, which caches
    kernel rows as C ``float``. Decision values contract the float32 kernel
    against float64 coefficients in float64, so the float32 error that reaches
    the output is the kernel's. Before anything is cast, the data is mapped
    exactly in float64: translated by the training mean (rbf only) and scaled by
    a power of two that is folded back into ``gamma``. A large offset —
    timestamps, projected coordinates — or a magnitude around 1e±19 therefore
    never reaches float32 at all. rbf entries between rows close to each other
    but far from the mean, whose float32 squared distance the Gram identity
    cannot resolve, are recomputed on the host in float64. The other kernels
    read the raw inner product, and data whose variation it cannot resolve in
    float32 — a large offset with a small spread, for ``linear`` and ``poly`` —
    is refused. See ``docs/fp32_policy.md`` §2.4.

    **Multiclass fits on the GPU.** With three classes or more, the rbf kernel and
    a fit resolved to the GPU, every one-vs-one pair is solved at once, in a Metal
    kernel of mlxlearn's own — one threadgroup per pair, the whole working set at
    every step — rather than pair after pair by the host solver. Its dual variables
    and gradient are double-float, about 48 bits, and it ranks working-set
    candidates in float32, so its path can part from the host solver's where two
    candidates tie that closely; each pair's result is therefore held to the host
    solver's stop — the gradient recomputed in float64, the full-set KKT test — and
    finished by the host solver when it fails. A pair with no free dual variable,
    whose bias LIBSVM takes from its own rounding, is solved again by the host
    solver, as is one that went non-finite. The fit's diagnostics event counts
    them (``ovo_*`` in ``fit_event_.extra``). Pairs of more than 4 096 rows, or
    whose Gram does not fit ``cache_size``, a ``tol`` below what the kernel's state
    resolves (about 2⁻⁴⁰ of ``1 + C·m``), and MLX 0.31, keep the host solver for
    every pair. See ``docs/fp32_policy.md`` §2.5.

    **Where inference runs.** A ``predict`` or ``decision_function`` batch of at most
    2²⁵ query rows × support vectors × features is evaluated on the host, in NumPy
    float32, with the device tiles' own arithmetic; larger batches on the fitted
    device. What a device call costs a batch of a few rows is the call itself —
    0.64-0.98 ms after the GPU has idled, against 0.22-0.82 ms for LIBSVM's
    single-row predict — while the host answers one row in 0.19-0.35 ms. The two paths
    differ only in how the BLAS rounds the inner products, as a different
    ``block_rows`` would. The event records ``compute="numpy-float32"`` for a batch
    answered on the host, which needs no GPU, so an explicit ``device="gpu"`` has
    nothing to refuse there.

    What float32 cannot represent — a value past ``3.4e38``, a linear-kernel
    inner product that large, a ``gamma`` that cannot be folded into range —
    raises :class:`~mlxlearn.exceptions.UnsupportedInputError` or
    :class:`~mlxlearn.exceptions.UnsupportedParameterError` at ``fit``, which
    patched mode hands to scikit-learn, and at ``predict``, which it cannot: the
    fitted backend is sticky, and a NaN decision would silently flip the label.
    A query far from the training data is answered — an rbf entry that far out
    is zero — unless its inner product with a support vector leaves float32.

    **Zero and negative weights, and where ``support_`` then differs.** A sample
    whose effective ``C`` is not positive — from a zero or negative
    ``sample_weight``, or a zero ``class_weight`` — is dropped from the
    optimization, which is what scikit-learn's LIBSVM does and produces the same
    model. scikit-learn then reports ``support_`` as indices into the *filtered*
    problem, so ``support_vectors_ != X[support_]`` and the indices point at
    unrelated rows; mlxlearn reports indices into the training data the caller
    passed, so ``support_vectors_ == X[support_]`` always holds. This is the one
    fitted attribute where mlxlearn deliberately does not reproduce
    scikit-learn's value, and it is deliberate because the alternative is
    returning indices that mean nothing. A class whose every sample is dropped is
    refused with :class:`~mlxlearn.exceptions.UnsupportedInputError`, which
    patched mode hands to scikit-learn: its LIBSVM fits the remaining classes
    into a model whose ``decision_function`` then fails.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn.svm import SVC
    >>> X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = SVC(kernel="linear").fit(X, y)
    >>> clf.predict([[-1.5], [1.5]])
    array([0, 1])
    >>> clf.n_support_
    array([1, 1], dtype=int32)
    >>> clf.decision_function([[1.5]]) > 0
    array([ True])
    """

    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample multipliers on ``C``. Must be finite. A zero or negative
            weight drops the sample, exactly as scikit-learn's LIBSVM does.

        Returns
        -------
        self : SVC
        """
        # Inherited fit is decorated with scikit-learn's validation context;
        # overriding it means the constraint check has to be requested explicitly,
        # or C=-1 would reach the solver instead of raising InvalidParameterError.
        self._validate_params()
        self._clear_backend()

        # The size is decided from the raw shape first — and, for the many-class
        # rule, the raw labels' count — after only the parameter and sparsity
        # checks, so a capability mismatch is still reported as one. Most of what
        # patched mode sees is below the crossover, and everything below would
        # otherwise be validated, range-scanned and target-checked here only for
        # scikit-learn to do it all again: measured 1.4x-1.6x stock at 200 x 5.
        backend = self._select_backend(
            "fit",
            shape=_safe_shape(X),
            crossover=lambda n, d: self._crossover_from_raw((n, d), y),
            # The kernel and cache_size need no data, and nor does the box without
            # weights, or with balanced class weights, which average to one; other
            # weights scale C sample by sample and are read below, once the targets
            # are.
            handoff=lambda n, d: self._kernel_handoff(
                float(self.C)
                if sample_weight is None and _is_none_or_balanced(self.class_weight)
                else None
            ),
            capability_check=lambda: self._check_svc_capability(X),
        )
        if backend == "sklearn":
            return self._fit_with_sklearn(X, y, sample_weight)

        X64 = validate_data(
            self,
            X,
            accept_sparse=False,
            dtype=np.float64,
            order="C",
            ensure_min_samples=1,
            reset=True,
        )
        y_encoded = self._validate_svc_targets(y)
        check_consistent_length(X64, y_encoded)
        weights = _libsvm_sample_weight(sample_weight, X64)

        # Decided again now that the classes are known — the raw labels may not
        # have been countable, and input with no shape before validation had no
        # size at all. Below the crossover, Layer 1 runs on the MLX CPU device and
        # Layer 2 hands the fit to scikit-learn — whatever device is configured:
        # device="cpu" selects MLX's CPU, which is the same float32 path and no
        # faster, not scikit-learn's. Only MLXLEARN_FORCE_BACKEND outranks the crossover.
        n_classes = len(self.classes_)
        backend = self._select_backend(
            "fit",
            shape=X64.shape,
            crossover=lambda n, d: self._below_crossover(n, d, n_classes=n_classes),
            # Patched mode only, and only once the size rules have passed the fit.
            handoff=lambda n, d: self._kernel_handoff(self._mean_box(y_encoded, weights)),
        )
        if backend == "sklearn":
            return self._fit_with_sklearn(X, y, sample_weight)

        # The data-dependent capability limits, checked only now that MLX will run.
        # Raised this late they bypass _select_backend, so they are routed through
        # the same policy by hand: Layer 1 raises, Layer 2 falls back or raises
        # under fallback_policy="raise". The float32 range check runs on the validated
        # float64 array, which also covers object-dtype input the raw scan skipped.
        try:
            box = self._sample_box(y_encoded, weights)
            check_float32_range(X64)
            problem = self._prepare_kernel_problem(X64)
        except CapabilityError as exc:
            self._handle_capability_error(exc, "fit", shape=X64.shape)
            return self._fit_with_sklearn(X, y, sample_weight)

        self._warn_sklearn_deprecations()
        with mlx_guard(type(self).__name__, "fit"):
            self._fit_ovo(X64, y_encoded, box, backend, problem)
        self._set_backend(backend)
        return self

    def _fit_with_sklearn(self, X, y, sample_weight):
        super().fit(X, y, sample_weight)
        self._set_backend("sklearn")
        return self

    def decision_function(self, X):
        """Signed distance of each sample to the separating hyperplanes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        dec : ndarray, float64
            ``(n_samples,)`` for two classes, where positive means
            ``classes_[1]``. Otherwise ``(n_samples, n_classes)`` when
            ``decision_function_shape="ovr"`` and
            ``(n_samples, n_classes * (n_classes - 1) / 2)`` when it is ``"ovo"``.

        Notes
        -----
        ``decision_function_shape`` is read here rather than frozen at fit time,
        so changing it after fitting changes the output shape. That matches
        scikit-learn.

        The two-class case returns 1-D regardless of ``decision_function_shape``.
        """
        check_is_fitted(self)
        if self._fitted_backend("decision_function") == "sklearn":
            self._note_inference("decision_function")
            return super().decision_function(X)

        decision = self._pairwise_decision(X)
        n_classes = len(self.classes_)
        if n_classes == 2:
            # scikit-learn's public binary convention is the negation of
            # LIBSVM's, so that a positive value means classes_[1].
            return -decision[:, 0]
        if self.decision_function_shape == "ovr":
            return ovr_decision_function(decision < 0, -decision, n_classes)
        return decision

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Labels drawn from ``classes_``.

        Notes
        -----
        Multiclass prediction is the one-vs-one vote, with ties going to the
        lowest class index. ``break_ties=True`` replaces the vote with the argmax
        of the one-vs-rest decision function, and — as in scikit-learn — is only
        meaningful with ``decision_function_shape="ovr"``.
        """
        check_is_fitted(self)
        if self._fitted_backend("predict") == "sklearn":
            self._note_inference("predict")
            return super().predict(X)

        if self.break_ties and self.decision_function_shape == "ovo":
            raise ValueError(
                "break_ties must be False when decision_function_shape is 'ovo'"
            )
        n_classes = len(self.classes_)
        if self.break_ties and self.decision_function_shape == "ovr" and n_classes > 2:
            indices = np.argmax(self.decision_function(X), axis=1)
        else:
            indices = ovo_vote(self._pairwise_decision(X, "predict"), n_classes)
        return self.classes_.take(np.asarray(indices, dtype=np.intp))

    # scikit-learn's predict_proba and predict_log_proba are `available_if`
    # descriptors, which BackendMixin leaves unwrapped so that they keep not
    # existing when probability=False. They are only ever reached on a model
    # scikit-learn fitted (probability=True is handed to it), and they recorded
    # no diagnostics event at all. Each is redeclared here with scikit-learn's own
    # availability rule and a public-call scope inside the descriptor, so a call
    # records exactly one event even though predict_log_proba calls predict_proba.

    def _has_sklearn_proba(self) -> bool:
        return hasattr(super(), "predict_proba")

    def predict_proba(self, X):
        """scikit-learn's Platt-scaled probabilities; see :class:`sklearn.svm.SVC`."""
        check_is_fitted(self)
        self._note_inference("predict_proba")
        return super().predict_proba(X)

    def predict_log_proba(self, X):
        """scikit-learn's log-probabilities; see :class:`sklearn.svm.SVC`."""
        check_is_fitted(self)
        self._note_inference("predict_log_proba")
        return super().predict_log_proba(X)

    predict_proba = available_if(_has_sklearn_proba)(_public_call("predict_proba", predict_proba))
    predict_log_proba = available_if(_has_sklearn_proba)(
        _public_call("predict_log_proba", predict_log_proba)
    )
