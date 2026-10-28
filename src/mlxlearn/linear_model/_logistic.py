"""Logistic regression, accelerated with MLX.

Subclassing :class:`sklearn.linear_model.LogisticRegression` is what makes the
drop-in promise real: the constructor signature, ``get_params``/``set_params``, the
parameter constraints and the estimator tags are all inherited, so ``clone``,
``Pipeline``, ``GridSearchCV`` and ``isinstance`` keep working, and
``super().fit(...)`` *is* stock scikit-learn for the patching layer to fall back to.

Scope for 0.1.0 is one solver — L-BFGS on an L2-penalized (or unpenalized)
objective — and everything outside it raises a
:class:`~mlxlearn.exceptions.CapabilityError` naming the parameter. That is
narrower than the ancestor, which accepted ``multi_class='ovr'`` and silently fit a
multinomial model instead: a wrong answer delivered quickly is worse than a
fallback.
"""

from __future__ import annotations

import math
import warnings
from contextlib import ExitStack, contextmanager
from dataclasses import replace

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as _SkLogisticRegression

from .._common.arrays import NUMPY_FLOAT
from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_tuning
from .._common.device import use_device
from .._common.preparation import cast_float32_checked
from .._common.validation import (
    check_array,
    check_finite_float32,
    check_float32_range,
    check_is_fitted,
    check_sample_weight,
    compute_class_weight,
    require_dense,
    require_supported_params,
    type_of_target,
    validate_data,
)
from .._solvers.logistic import (
    LogisticConfig,
    _use_native_solver,
    sketch_supported,
    solve_logistic,
)
from ..exceptions import UnsupportedInputError, UnsupportedParameterError

__all__ = ["LogisticRegression", "MLXLogisticMixin"]

_SUPPORTED_SOLVERS = ("lbfgs",)
_SUPPORTED_PENALTIES = ("l2", None)
#: ``"deprecated"`` is scikit-learn 1.7's default sentinel and ``"auto"`` is the
#: value it replaced; both mean "binary problems get one weight vector, everything
#: else gets the multinomial loss", which is what this implementation does.
_SUPPORTED_MULTI_CLASS = ("deprecated", "auto")


class MLXLogisticMixin(BackendMixin):
    """Capability gating and crossover dispatch for the logistic estimators."""

    # ------------------------------------------------------------ version shims

    def _effective_penalty(self):
        """Resolve the penalty the way the installed scikit-learn resolves it.

        scikit-learn deprecated ``penalty`` in 1.8 in favor of ``l1_ratio``, with
        removal in 1.10, so the two spellings coexist across the range
        ``pyproject.toml`` declares:

        * **1.7** — ``penalty`` is real, default ``"l2"``; ``l1_ratio`` defaults
          to ``None``.
        * **1.9** — ``penalty`` defaults to the sentinel ``"deprecated"`` and the
          effective penalty comes from ``l1_ratio`` (0 → l2, 1 → l1, else
          elasticnet), with ``C=inf`` meaning unpenalized. ``l1_ratio`` defaults
          to ``0.0``.

        Reading the raw attributes broke on both sides of that change: ``penalty``
        read as the literal string ``"deprecated"`` and was rejected as
        unsupported, and the old ``l1_ratio is not None`` guard rejected 1.9's
        own default. Capability is decided on the *resolved* value instead, so
        the same request means the same thing on every supported version.
        """
        penalty = getattr(self, "penalty", "deprecated")
        if penalty != "deprecated":
            return penalty

        l1_ratio = getattr(self, "l1_ratio", None)
        if l1_ratio is None or l1_ratio == 0:
            resolved = "l2"
        elif l1_ratio == 1:
            resolved = "l1"
        else:
            resolved = "elasticnet"
        return None if self.C == math.inf else resolved

    # --------------------------------------------------------------- capability

    def _check_logistic_capability(self, X, y):
        """Check capabilities and return a safely prepared dense input."""
        name = type(self).__name__
        require_dense(X, estimator=name)

        params = {
            "solver": self.solver,
            "penalty": self._effective_penalty(),
            "dual": self.dual,
            "warm_start": self.warm_start,
        }
        supported = {
            "solver": _SUPPORTED_SOLVERS,
            "penalty": _SUPPORTED_PENALTIES,
            "dual": (False,),
            "warm_start": (False,),
        }

        # scikit-learn 1.9 removed multi_class outright; 1.7 and 1.8 still carry it
        # as a deprecated sentinel. Reading it unconditionally raised AttributeError
        # on 1.9 — a version pair pyproject.toml declares supported and CI schedules.
        # Checked only where it exists, rather than pinned away.
        if hasattr(self, "multi_class"):
            params["multi_class"] = self.multi_class
            supported["multi_class"] = _SUPPORTED_MULTI_CLASS

        require_supported_params(params, supported, estimator=name)

        # l1_ratio is already accounted for: on scikit-learn 1.8+ it *is* how the
        # penalty is spelled, so _effective_penalty has resolved it and anything
        # other than pure L2 was rejected above. On 1.7 the penalty is explicit
        # and scikit-learn itself ignores l1_ratio unless penalty='elasticnet',
        # so mlxlearn ignores it in exactly the same cases — being stricter here
        # would send a request to scikit-learn that mlxlearn can serve identically.
        #
        # What must not happen is the ancestor's behavior: accepting an
        # elastic-net request and quietly fitting a ridge. That cannot occur,
        # because 'elasticnet' never resolves to a supported penalty.

        # intercept_scaling only affects the liblinear solver, where it scales a
        # synthetic feature. Accepting a non-default value under lbfgs would mean
        # ignoring it.
        if self.fit_intercept and self.intercept_scaling != 1.0:
            raise UnsupportedParameterError(
                f"{name} implements intercept_scaling=1.0 only; got "
                f"{self.intercept_scaling!r}. It has an effect solely for "
                "solver='liblinear', which is not implemented here.",
                parameter="intercept_scaling",
                value=self.intercept_scaling,
            )

        y_array = np.asarray(y)
        if y_array.ndim > 1 and y_array.shape[1] > 1:
            raise UnsupportedInputError(
                f"{name} implements single-output classification only; got y with "
                f"shape {y_array.shape}. Wrap the estimator in "
                "OneVsRestClassifier for multi-label targets.",
                reason="multi-label-target",
            )
        return _prepare_logistic_input(X, return_finite=True)

    def _below_crossover(self, n_samples: int, n_features: int) -> tuple[bool, str]:
        """Whether stock scikit-learn is expected to win at this size.

        Either the original wide-workload floors or the measured native tall-
        workload floors must be met. Small problems retain their previous route.
        See ``docs/native-logistic-benchmarks.md`` for the measured shapes.
        """
        tuning = get_tuning()
        work = n_samples * n_features
        wide = n_samples >= tuning.logreg_min_samples and work >= tuning.logreg_min_work
        tall = (
            n_samples >= tuning.logreg_tall_min_samples
            and work >= tuning.logreg_tall_min_work
        )
        if not (wide or tall):
            return True, (
                f"{n_samples} samples x {n_features} features is below the measured "
                f"logistic-regression crossover ({tuning.logreg_min_samples} samples "
                f"and {tuning.logreg_min_work} elements, or "
                f"{tuning.logreg_tall_min_samples} samples and "
                f"{tuning.logreg_tall_min_work} elements)"
            )
        return False, ""


class LogisticRegression(MLXLogisticMixin, _SkLogisticRegression):
    """L2-penalized logistic regression with native and fused MLX L-BFGS.

    A drop-in replacement for :class:`sklearn.linear_model.LogisticRegression`
    restricted to ``solver="lbfgs"`` with ``penalty="l2"`` or ``penalty=None`` on
    dense, single-output data. The objective minimized is scikit-learn's, term for
    term, so the two agree on the objective value at their respective solutions.
    Supported binary GPU fits use native Metal C++ kernels for optimizer history,
    batched Armijo backtracking, and fused matrix reductions. Several iterations
    run between host checks, and a direct objective/gradient evaluation certifies
    convergence. Multinomial, CPU, and larger workspaces use the generic fused
    optimizer with a strong-Wolfe line search.

    Parameters
    ----------
    penalty : {"l2", None}, default="l2"
        ``"l1"`` and ``"elasticnet"`` are not implemented and raise.
    dual : bool, default=False
        Only ``False`` is implemented.
    tol : float, default=1e-4
        Tolerance on ``max|gradient|``, the same criterion and scale scikit-learn's
        lbfgs path uses.
    C : float, default=1.0
        Inverse regularization strength.
    fit_intercept : bool, default=True
    intercept_scaling : float, default=1
        Only ``1.0`` is accepted; it affects nothing outside ``solver="liblinear"``.
    class_weight : dict or "balanced", default=None
        Folded into the sample weights exactly as scikit-learn folds them, before
        the penalty scale is derived from their sum.
    random_state : int, RandomState or None, default=None
        Accepted and unused: L-BFGS from a zero start is deterministic.
    solver : {"lbfgs"}, default="lbfgs"
    max_iter : int, default=100
    multi_class : str, default="deprecated"
        Only scikit-learn 1.7's default is implemented — multinomial for three or
        more classes, a single weight vector for two. ``"ovr"`` and
        ``"multinomial"`` raise rather than being reinterpreted.
    verbose : int, default=0
        Accepted and ignored.
    warm_start : bool, default=False
        Only ``False`` is implemented.
    n_jobs : int, default=None
        Accepted and ignored; the MLX path is not joblib-parallel.
    l1_ratio : float, default=None
        Only ``None`` is accepted.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Ascending, with the dtype of ``y``.
    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        float64. Binary problems get a single weight vector for ``classes_[1]``,
        matching scikit-learn.
    intercept_ : ndarray of shape (1,) or (n_classes,)
        float64. Zeros when ``fit_intercept=False``.
    n_iter_ : ndarray of shape (1,), dtype int32
        Accepted L-BFGS steps. Always length one, because there is no
        one-versus-rest path that would fit several independent problems.
    n_features_in_ : int
    execution_backend_ : {"mlx", "cpu", "sklearn"}

    Notes
    -----
    **Precision.** Computation is float32 (``docs/fp32_policy.md``); ``coef_`` and
    ``intercept_`` are float64 containers. The objective is evaluated through
    ``logaddexp``/``logsumexp`` and never exponentiates a raw score, so separable
    data with logits in the hundreds produces finite losses rather than the floored
    plateaus that a ``clip``-based formulation gives in float32.

    **Convergence is reported.** A run that exhausts ``max_iter`` or stalls in the
    line search emits :class:`sklearn.exceptions.ConvergenceWarning`, as
    scikit-learn does. ``n_iter_`` counts accepted steps, so a problem already
    optimal at the zero start reports ``0``.

    **No warm start, no caches.** ``fit`` is a pure function of ``(X, y,
    sample_weight, params)``. Nothing about a previous fit in the same process can
    reach this one.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn.linear_model import LogisticRegression
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = LogisticRegression().fit(X, y)
    >>> clf.coef_.shape
    (1, 1)
    >>> clf.predict([[2.5]])
    array([1])
    """

    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), default=None

        Returns
        -------
        self : LogisticRegression
        """
        # scikit-learn applies its parameter constraints through a private fit
        # decorator this class does not use, so they are invoked explicitly. The
        # ancestor skipped them entirely, which turned C=-1 into a silent clamp to
        # 1e-12 instead of the InvalidParameterError scikit-learn raises.
        self._validate_params()

        self._clear_backend()
        prepared_X = X
        finite_checked = False

        def check_capability():
            nonlocal prepared_X, finite_checked
            prepared_X, finite_checked = self._check_logistic_capability(X, y)

        backend = self._select_backend(
            "fit",
            capability_check=check_capability,
            shape=_safe_shape(X),
        )
        if backend == "sklearn":
            super().fit(X, y, sample_weight)
            self._set_backend("sklearn")
            return self

        shape = _safe_shape(prepared_X)
        gpu_validation = (
            backend == "mlx"
            and type(prepared_X) is np.ndarray
            and prepared_X.dtype == np.float32
            and shape is not None
            and (get_tuning().force_backend == "mlx" or not self._below_crossover(*shape)[0])
        )
        X_checked, y_checked = validate_data(
            self,
            prepared_X,
            y,
            accept_sparse=False,
            dtype=NUMPY_FLOAT,
            order="C",
            ensure_min_samples=1,
            ensure_all_finite=not (finite_checked or gpu_validation),
            reset=True,
        )
        check_finite_float32(X_checked)
        _check_classification_target(y_checked)

        below, detail = self._below_crossover(*X_checked.shape)
        if below and backend != "cpu":
            backend = self._select_backend(
                "fit",
                below_crossover=True,
                crossover_detail=detail,
                shape=X_checked.shape,
            )
            if backend == "sklearn":
                super().fit(X, y, sample_weight)
                self._set_backend("sklearn")
                return self

        self.classes_, y_encoded = _encode_labels(y_checked)
        n_classes = self.classes_.shape[0]
        if n_classes < 2:
            # scikit-learn's wording, verbatim: code written against its error
            # contract matches on this string.
            raise ValueError(
                "This solver needs samples of at least 2 classes in the data, but "
                f"the data contains only one class: {self.classes_[0]!r}"
            )

        weights = self._resolve_weights(X_checked, y_checked, y_encoded, sample_weight)

        # Resolved rather than read, so that C=inf on scikit-learn 1.9 means
        # "unpenalized" here exactly as it does there.
        if self._effective_penalty() is None:
            if self.C not in (1.0, math.inf):
                warnings.warn(
                    "Setting penalty=None will ignore the C and l1_ratio parameters",
                    stacklevel=2,
                )
            C = math.inf
        else:
            C = float(self.C)

        tuning = get_tuning()
        config = LogisticConfig(
            C=C,
            fit_intercept=bool(self.fit_intercept),
            tol=float(self.tol),
            max_iter=int(self.max_iter),
            history_size=tuning.lbfgs_history,
            max_line_search=tuning.lbfgs_max_line_search,
            device="gpu" if backend == "mlx" else "cpu",
            sketch_size=tuning.lbfgs_sketch_size,
        )

        device_input = center = initial_parts = None
        with mlx_guard(type(self).__name__, "fit"), ExitStack() as input_lifetime:
            if backend == "mlx" and (finite_checked or gpu_validation):
                # Read caller storage only while this context owns the borrowed
                # wrapper. The owned copy is independent, centered when an
                # intercept can absorb the shift, and the binary solver's
                # initialization shares its required write.
                targets = 1 if n_classes == 2 else n_classes
                device_input, center, finite, initial_parts = input_lifetime.enter_context(
                    _prepare_centered_logistic_input(
                        X_checked, y_encoded, weights,
                        native=_use_native_solver(config, targets, X_checked.shape[1]),
                        center=config.fit_intercept,
                    )
                )
                if not finite:
                    # mlx_guard passes normal validation errors through intact.
                    check_array(X_checked, dtype=NUMPY_FLOAT, input_name="X", estimator=self)

            if config.sketch_size and not sketch_supported(
                config, 1 if n_classes == 2 else n_classes, X_checked.shape[1], initial_parts
            ):
                # The opt-in sketch is a capability of the prepared native
                # loop, not a requirement: fits it cannot seed run unsketched
                # at the requested tolerance.
                config = replace(config, sketch_size=0)
            solution = solve_logistic(
                X_checked,
                y_encoded.astype(np.int32, copy=False),
                n_classes=n_classes,
                sample_weight=weights,
                config=config,
                X_device=device_input,
                X_center=center,
                initial_parts=initial_parts,
                certification_X=X,
            )

        self.coef_ = solution.coef
        self.intercept_ = solution.intercept
        self.n_iter_ = np.array([solution.n_iter], dtype=np.int32)
        self._set_backend(backend)

        if not solution.converged:
            # The ancestor unpacked `converged` and discarded it at every call
            # site, so a model that stopped at max_iter was indistinguishable from
            # one that reached the tolerance.
            target_tol = self.tol if solution.required_tol is None else solution.required_tol
            safety_note = (
                "" if target_tol == self.tol
                else f" (internal safety threshold; requested tol={self.tol:g})"
            )
            warnings.warn(
                f"{type(self).__name__} failed to converge "
                f"({solution.stop_reason}) after {solution.n_iter} iterations; "
                f"max|gradient| is {solution.grad_max_abs:.3g} against tol="
                f"{target_tol:g}{safety_note}. Increase max_iter, scale the data, or relax tol.",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    # ------------------------------------------------------------------ weights

    def _resolve_weights(self, X, y, y_encoded, sample_weight):
        """Combine ``sample_weight`` and ``class_weight`` the way scikit-learn does.

        The order matters and is not arbitrary: class weights multiply the sample
        weights *first*, and only then is ``sw_sum`` taken, because the penalty
        scale is ``1/(C·sw_sum)``. Folding them in the other order changes the
        effective regularization, and scikit-learn has a test asserting that
        passing ``class_weight`` is equivalent to passing the corresponding
        ``sample_weight``.
        """
        weights = check_sample_weight(sample_weight, X, dtype=np.float64)
        if self.class_weight is None:
            return weights

        if weights is None:
            weights = np.ones(X.shape[0], dtype=np.float64)
        else:
            weights = weights.copy()
        # compute_class_weight is given the sample weights as well, because
        # "balanced" means "inversely proportional to the *weighted* class
        # frequencies" in scikit-learn 1.7.
        class_weights = compute_class_weight(
            self.class_weight, classes=self.classes_, y=y, sample_weight=weights
        )
        weights *= np.asarray(class_weights, dtype=np.float64)[y_encoded]
        return weights

    # ---------------------------------------------------------------- inference

    # scikit-learn computes all three of these from coef_, intercept_ and
    # classes_, in float64, and an MLX fit populates exactly those attributes with
    # exactly those shapes. Reimplementing them in MLX would trade float64
    # exactness for nothing: the work is one (n, d) x (d, K) matmul, far below any
    # crossover. So these overrides add only the sticky-backend bookkeeping —
    # _note_inference asserts that a fit completed and records which backend owns
    # this estimator — and then defer. The ancestor did reimplement them, in
    # float32, with `score >= 0` where scikit-learn uses `score > 0`, so its
    # predict and its decision_function could disagree on the decision boundary.

    def decision_function(self, X):
        """Confidence scores, ``X @ coef_.T + intercept_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            One-dimensional for binary problems.
        """
        check_is_fitted(self)
        self._note_inference("decision_function", shape=_safe_shape(X))
        return super().decision_function(X)

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Labels drawn from ``classes_``.
        """
        check_is_fitted(self)
        self._note_inference("predict", shape=_safe_shape(X))
        return super().predict(X)

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            float64, rows sum to 1, columns ordered by ``classes_``.
        """
        check_is_fitted(self)
        self._note_inference("predict_proba", shape=_safe_shape(X))
        return super().predict_proba(X)


def _check_classification_target(y: np.ndarray) -> None:
    """Reject regression targets with scikit-learn's message.

    Uses the public ``type_of_target`` rather than scikit-learn's
    ``check_classification_targets``, which is not part of its documented API.
    """
    # A validated 1-D integer/bool target is discrete by construction. Avoid
    # sorting all labels merely to distinguish binary from multiclass here.
    if y.ndim == 1 and y.dtype.kind in "biu":
        return
    target_type = type_of_target(y, input_name="y")
    if target_type in ("binary", "multiclass"):
        return
    if target_type in ("multilabel-indicator", "multiclass-multioutput", "continuous-multioutput"):
        raise UnsupportedInputError(
            "LogisticRegression implements single-output classification only; got a "
            f"{target_type!r} target.",
            reason="multi-label-target",
        )
    raise ValueError(f"Unknown label type: {target_type!r}")


def _encode_labels(y):
    """Sorted class encoding without sorting every row of a binary target."""
    if y.dtype.kind in "biu" and y.size:
        low, high = y.min(), y.max()
        positive = y == high
        if np.all(positive | (y == low)):
            classes = np.array([low] if low == high else [low, high], dtype=y.dtype)
            codes = positive.astype(np.int32) if low != high else np.zeros(y.size, np.int32)
            return classes, codes
    classes = np.unique(y)
    return classes, np.searchsorted(classes, y).astype(np.int32)


def _prepare_logistic_input(X, *, return_finite=False):
    """Reuse a checked float32 cast instead of scanning float64 twice first.

    Only NumPy arrays take this path, preserving DataFrame feature names and
    ordinary validation for other containers. On conversion overflow, inspect
    the original values so finite overflow remains a capability error, while
    NaN/Inf remain invalid-input errors raised by ``validate_data``.
    """
    if type(X) is np.ndarray and X.dtype.kind == "f" and X.dtype.itemsize > 4:
        prepared, finite = cast_float32_checked(X)
        if not finite:
            check_float32_range(X)
        return (prepared, finite) if return_finite else prepared
    check_float32_range(X)
    return (X, False) if return_finite else X


@contextmanager
def _prepare_centered_logistic_input(X, y, weights, *, native, center=True):
    """Own a validated device copy of ``X`` for the duration of the fit.

    With ``center`` the copy is shifted by the per-column mean, which is
    yielded so the fitted intercept can be restored; without it the copy
    holds the original values and the yielded center is ``None``. Keep
    borrowed storage alive until all work using the owned result ends.
    """
    import mlx.core as mx

    from .._kernels.logistic_input import prepare_centered

    with use_device("gpu"):
        # Older supported MLX versions may not expose asarray. Their explicit
        # copy retains the same ownership and numerical behavior.
        borrowed = mx.asarray(X) if hasattr(mx, "asarray") else mx.array(X)
        labels = mx.array(y, dtype=mx.float32) if native else None
        sample_weights = (
            mx.array(weights, dtype=mx.float32) if native and weights is not None else None
        )
        owned, shift, finite, parts = prepare_centered(
            borrowed, labels, sample_weights, center=center
        )
        try:
            mx.eval(owned, finite, *(value for value in (shift, parts) if value is not None))
            yield owned, shift, bool(finite), parts
        finally:
            # Evaluation can finish before worker-side graph cleanup. Drain it
            # while the NumPy-backed wrapper is still owned on this thread, so
            # releasing the caller's Python buffer cannot block a GPU worker.
            mx.synchronize()


def _safe_shape(X) -> tuple[int, ...] | None:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))
