"""C-support-vector classification, accelerated with MLX.

The estimator subclasses :class:`sklearn.svm.SVC`. That is what makes the drop-in
promise real: the constructor signature, ``get_params``/``set_params``, the
parameter constraints and the estimator tags are inherited rather than restated,
so ``clone``, ``Pipeline``, ``GridSearchCV`` and ``isinstance`` keep working, and
``super().fit(...)`` *is* stock scikit-learn for the patching layer to fall back
to. Layer 1 never calls that fallback — ``_mlxlearn_allow_fallback`` is ``False``,
so a capability mismatch raises.

What is actually solved here is LIBSVM's C-SVC dual, exactly, by the SMO in
``_solvers/smo.py``. There is no random-feature approximation, no subsampled
training set and no ridge surrogate anywhere in this path. Those things make a
different classifier; shipping one under this class's name would mean every
accuracy number a user measured was measured on something other than what the
name says.

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
exactly zero. The layout is not verifiable by looking at the array — it is
verifiable by checking that the decision function it produces matches
scikit-learn's, which is how it was checked.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC as _SkSVC
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import column_or_1d

from .._common.arrays import NUMPY_FLOAT, as_float32
from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_tuning
from .._common.device import physical_memory_bytes
from .._common.validation import (
    check_consistent_length,
    check_finite_float32,
    check_is_fitted,
    check_sample_weight,
    compute_class_weight,
    require_dense,
    require_supported_params,
    validate_data,
)
from .._kernels.kernels import (
    SUPPORTED_KERNELS,
    KernelSpec,
    build_kernel_rows,
    kernel_combine,
    kernel_diagonal,
    kernel_matrix,
    resolve_gamma,
)
from .._solvers.smo import SMOConfig, solve_c_svc
from ..exceptions import UnsupportedParameterError

__all__ = ["SVC", "MLXSVCMixin"]

_SUPPORTED_SHAPES = ("ovo", "ovr")


def _safe_shape(X) -> tuple[int, ...] | None:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))


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


class MLXSVCMixin(BackendMixin):
    """Backend selection, the one-vs-one fit, and inference for :class:`SVC`."""

    # --------------------------------------------------------------- capability

    def _check_svc_capability(self, X) -> None:
        """Raise :class:`CapabilityError` for anything the MLX path cannot serve."""
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
        if self.probability:
            raise UnsupportedParameterError(
                f"{name} does not implement probability=True on the MLX path: "
                "Platt calibration is not part of mlxlearn 0.1.0. Wrap the "
                "estimator in sklearn.calibration.CalibratedClassifierCV to get "
                "calibrated probabilities, or use sklearn.svm.SVC directly.",
                parameter="probability",
                value=self.probability,
            )

    def _below_crossover(self, n_samples: int, n_features: int) -> tuple[bool, str]:
        """Whether stock scikit-learn is expected to win at this size.

        SMO is a sequential algorithm: each iteration touches two kernel rows and
        cannot start before the previous one finished. Below the measured
        crossover the per-iteration device dispatch costs more than LIBSVM's
        cache-resident inner loop, so handing the problem away is the mechanism
        behind "never slower", not an admission of defeat.
        """
        tuning = get_tuning()
        if n_samples < tuning.svc_min_samples:
            return True, (
                f"{n_samples} samples x {n_features} features is below the measured "
                f"SVC crossover ({tuning.svc_min_samples} samples)"
            )
        return False, ""

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
        gets a small cache — the cache is a pure performance knob, so honoring the
        request costs nothing but time, whereas quietly overriding it downward
        would mean ``cache_size`` had no observable effect at all and upward would
        mean a request for 32 GiB on an 8 GB laptop got granted.
        """
        tuning = get_tuning()
        budget = int(float(self.cache_size) * 1024 * 1024)
        ceiling = tuning.smo_kernel_cache_max_bytes
        installed = physical_memory_bytes()
        if installed is not None:
            ceiling = min(ceiling, int(installed * tuning.smo_kernel_cache_fraction))
        # A machine that will not report its memory must still get a usable cache.
        ceiling = max(ceiling, tuning.smo_kernel_cache_min_bytes)
        rows = min(budget, ceiling) // (4 * max(n_samples, 1))
        return int(max(2, min(max(n_samples, 2), rows)))

    def _resolve_max_iter(self, n_samples: int) -> int:
        """Turn ``max_iter=-1`` into the concrete budget the solver needs."""
        if self.max_iter > 0:
            return int(self.max_iter)
        tuning = get_tuning()
        return max(tuning.smo_min_iter_budget, tuning.smo_max_iter_factor * n_samples)

    def _solve_pair(self, X_pair, y_pair, C_pair, spec, center, backend):
        """Solve one one-vs-one subproblem.

        The full Gram is precomputed only when the row cache was already sized to
        hold every row — at which point the memory has been accounted for and one
        blocked matmul beats ``n`` separate row evaluations. Otherwise rows are
        evaluated on demand and the LRU bounds what is resident, so a subproblem
        larger than the cache never needs an ``n × n`` matrix to exist.
        """
        n_pair = int(X_pair.shape[0])
        cache_rows = self._kernel_cache_rows(n_pair)
        k_diag = kernel_diagonal(spec, X_pair)

        if cache_rows >= n_pair:
            gram = kernel_matrix(spec, X_pair, backend=backend, center=center)
            # Rows of a materialized Gram are views, so the LRU adds no memory here.
            fetch = gram.__getitem__
        else:
            rows = build_kernel_rows(X_pair, spec, backend=backend, center=center)
            fetch = rows.row

        config = SMOConfig(
            tol=float(self.tol),
            max_iter=self._resolve_max_iter(n_pair),
            cache_rows=max(2, min(cache_rows, n_pair)),
        )
        return solve_c_svc(y_pair, C_pair, k_diag, fetch, config)

    def _fit_ovo(self, X64, y_encoded, sample_weight, backend) -> None:
        """Fit every one-vs-one subproblem and assemble LIBSVM's model layout."""
        n_samples, n_features = X64.shape
        n_classes = int(len(self.classes_))

        gamma = resolve_gamma(self.gamma, X64)
        spec = KernelSpec(
            kernel=self.kernel,
            gamma=gamma,
            degree=int(self.degree),
            coef0=float(self.coef0),
        )
        X32 = as_float32(X64)
        check_finite_float32(X32)
        # Translating both sides of the RBF squared distance is exact and is what
        # keeps float32 cancellation at the variance scale rather than the
        # coordinate scale; see docs/fp32_policy.md §2.1.
        center = (
            X32.mean(axis=0, dtype=np.float64).astype(NUMPY_FLOAT)
            if spec.uses_center
            else None
        )

        # C_t = C * class_weight[class(t)] * sample_weight[t], which is exactly
        # how scikit-learn's LIBSVM builds its per-sample box.
        C_all = float(self.C) * self.class_weight_[y_encoded]
        if sample_weight is not None:
            C_all = C_all * sample_weight

        # A sample whose box is [0, 0] can never leave the lower bound, but it is
        # still a member of I_low, so its gradient keeps feeding the stopping
        # criterion and the solver runs to max_iter without ever being able to fix
        # the violation it is reporting -- measured: 10^7 iterations and a model
        # 16% worse than scikit-learn's. scikit-learn's LIBSVM drops zero-weight
        # rows from the problem before solving, so mlxlearn drops them too, on the
        # effective box rather than on the raw weight so that class_weight={c: 0}
        # is handled the same way.
        active = C_all > 0.0
        if not active.all():
            empty = [
                self.classes_[c]
                for c in range(n_classes)
                if not active[y_encoded == c].any()
            ]
            if empty:
                raise ValueError(
                    "Every sample of class "
                    f"{empty[0]!r} has an effective weight of zero, so that class "
                    "cannot be fitted. Remove it from y or give it a nonzero "
                    "sample_weight or class_weight."
                )

        class_rows = [np.flatnonzero((y_encoded == c) & active) for c in range(n_classes)]
        pairs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
        n_pairs = len(pairs)

        pair_rows: list[np.ndarray] = []
        pair_coef: list[np.ndarray] = []
        rho = np.empty(n_pairs, dtype=np.float64)
        n_iter = np.empty(n_pairs, dtype=np.int32)
        violation = np.empty(n_pairs, dtype=np.float64)
        fit_status = 0

        for k, (i, j) in enumerate(pairs):
            rows_i = class_rows[i]
            rows_j = class_rows[j]
            index = np.concatenate([rows_i, rows_j])
            y_pair = np.concatenate(
                [np.ones(rows_i.shape[0]), -np.ones(rows_j.shape[0])]
            )
            state = self._solve_pair(
                np.ascontiguousarray(X32[index]),
                y_pair,
                C_all[index],
                spec,
                center,
                backend,
            )
            pair_rows.append(index)
            # LIBSVM folds the label into the dual coefficient before storing it.
            pair_coef.append(state.alpha * y_pair)
            rho[k] = state.rho
            n_iter[k] = state.n_iter
            violation[k] = state.max_kkt_violation
            if not state.converged:
                fit_status = 1

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

        self.support_ = support
        self.support_vectors_ = X64[support] if n_sv else np.empty((0, n_features))
        self._n_support = n_support
        self.shape_fit_ = (int(n_samples), int(n_features))
        self.fit_status_ = fit_status
        self.n_iter_ = n_iter
        self._num_iter = n_iter
        self._gamma = gamma
        self._sparse = False
        self._probA = np.empty(0)
        self._probB = np.empty(0)

        # LIBSVM's convention throughout the internals: decision = K @ w - rho.
        self._pair_weights_ = pair_weights
        self._pair_rho_ = rho
        self._pair_kkt_violation_ = violation
        self._kernel_spec_ = spec
        self._kernel_center_ = center

        self._dual_coef_ = dual_coef
        self._intercept_ = -rho
        if n_classes == 2:
            self.dual_coef_ = -dual_coef
            self.intercept_ = rho.copy()
        else:
            self.dual_coef_ = dual_coef
            self.intercept_ = -rho

        if fit_status:
            warnings.warn(
                f"Solver terminated early (max_iter={self.max_iter}). Consider "
                "pre-processing your data with StandardScaler or MinMaxScaler.",
                ConvergenceWarning,
                stacklevel=2,
            )

    # --------------------------------------------------------------- inference

    def _pairwise_decision(self, X, operation: str = "decision_function") -> np.ndarray:
        """One-vs-one decision values in LIBSVM's convention, ``(n, n_pairs)``.

        Positive means the pair's *first* class. This is the single quantity every
        public inference method is derived from, so ``predict`` and
        ``decision_function`` cannot disagree about which side of a boundary a
        sample is on.
        """
        Xv = validate_data(
            self, X, accept_sparse=False, dtype=np.float64, order="C", reset=False
        )
        backend = self._note_inference(operation, shape=_safe_shape(Xv))
        n_pairs = self._pair_weights_.shape[1]
        if Xv.shape[0] == 0:
            return np.empty((0, n_pairs), dtype=np.float64)
        with mlx_guard(type(self).__name__, operation):
            decision = kernel_combine(
                self._kernel_spec_,
                Xv,
                self.support_vectors_,
                self._pair_weights_,
                backend=backend,
                center=self._kernel_center_,
            )
        return decision - self._pair_rho_


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
    degree : int, default=3
        Degree of the polynomial kernel. Evaluated by repeated squaring, so a
        negative base raised to an odd degree gives a negative value rather than
        NaN.
    gamma : {"scale", "auto"} or float, default="scale"
        ``"scale"`` is ``1 / (n_features * X.var())``, ``"auto"`` is
        ``1 / n_features``, both resolved exactly as scikit-learn resolves them.
    coef0 : float, default=0.0
    shrinking : bool, default=True
        **Accepted and ignored.** Shrinking is a pure speed heuristic that
        provably does not change the optimum; mlxlearn 0.1.0 does not implement
        it, and passing ``shrinking=False`` therefore also changes nothing. It is
        not raised on because doing so would reject the scikit-learn default.
    probability : bool, default=False
        ``True`` is not implemented; see Notes.
    tol : float, default=1e-3
        Stopping tolerance on the maximal KKT violation.
    cache_size : float, default=200
        Kernel cache in MiB. Mapped onto the solver's LRU row cache:
        ``rows = cache_size · 2²⁰ / (4 · n_samples)``, clamped by installed memory.
    class_weight : dict or "balanced", default=None
    verbose : bool, default=False
        Accepted and ignored.
    max_iter : int, default=-1
        ``-1`` resolves to LIBSVM's budget, ``max(10⁷, 100 · n_samples)``.
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
        SMO updates performed per one-vs-one subproblem.
    fit_status_ : int
        ``1`` when any subproblem hit ``max_iter``. Unlike the estimator this
        replaces, that also raises a :class:`~sklearn.exceptions.ConvergenceWarning`
        instead of reporting success.
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
    and working-set arithmetic are float64 — the same split LIBSVM uses, which
    caches kernel rows as C ``float``. Decision values contract the float32 kernel
    against float64 coefficients in float64, so the only float32 error that
    reaches the output is the kernel's own rounding. See ``docs/fp32_policy.md``.

    **Zero-weight samples, and where ``support_`` then differs.** A sample whose
    effective ``C`` is zero — from ``sample_weight=0`` or a zero ``class_weight``
    — is dropped from the optimization, which is what scikit-learn's LIBSVM does
    and produces the same model. scikit-learn then reports ``support_`` as indices
    into the *filtered* problem, so ``support_vectors_ != X[support_]`` and the
    indices point at unrelated rows; mlxlearn reports indices into the training
    data the caller passed, so ``support_vectors_ == X[support_]`` always holds.
    This is the one fitted attribute where mlxlearn deliberately does not
    reproduce scikit-learn's value, and it is deliberate because the alternative
    is returning indices that mean nothing.

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
            Per-sample multipliers on ``C``. Non-negative and finite.

        Returns
        -------
        self : SVC
        """
        # Inherited fit is decorated with scikit-learn's validation context;
        # overriding it means the constraint check has to be requested explicitly,
        # or C=-1 would reach the solver instead of raising InvalidParameterError.
        self._validate_params()
        self._clear_backend()

        backend = self._select_backend(
            "fit",
            capability_check=lambda: self._check_svc_capability(X),
            shape=_safe_shape(X),
        )
        if backend == "sklearn":
            super().fit(X, y, sample_weight)
            self._set_backend("sklearn")
            return self

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
        weights = check_sample_weight(sample_weight, X64, dtype=np.float64)

        below, detail = self._below_crossover(*X64.shape)
        if below and backend != "cpu":
            backend = self._select_backend(
                "fit",
                below_crossover=True,
                crossover_detail=detail,
                shape=X64.shape,
            )
            if backend == "sklearn":
                super().fit(X, y, sample_weight)
                self._set_backend("sklearn")
                return self

        with mlx_guard(type(self).__name__, "fit"):
            self._fit_ovo(X64, y_encoded, weights, backend)
        self._set_backend(backend)
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
