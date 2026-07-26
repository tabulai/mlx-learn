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

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as _SkLogisticRegression

from .._common.arrays import NUMPY_FLOAT
from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_tuning
from .._common.validation import (
    check_finite_float32,
    check_is_fitted,
    check_sample_weight,
    compute_class_weight,
    require_dense,
    require_supported_params,
    type_of_target,
    validate_data,
)
from .._solvers.logistic import LogisticConfig, solve_logistic
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

    # --------------------------------------------------------------- capability

    def _check_logistic_capability(self, X, y) -> None:
        """Raise :class:`CapabilityError` for anything the MLX path cannot serve."""
        name = type(self).__name__
        require_dense(X, estimator=name)

        require_supported_params(
            {
                "solver": self.solver,
                "penalty": self.penalty,
                "dual": self.dual,
                "warm_start": self.warm_start,
                "multi_class": self.multi_class,
            },
            {
                "solver": _SUPPORTED_SOLVERS,
                "penalty": _SUPPORTED_PENALTIES,
                "dual": (False,),
                "warm_start": (False,),
                "multi_class": _SUPPORTED_MULTI_CLASS,
            },
            estimator=name,
        )

        # l1_ratio is meaningful only for penalty='elasticnet', which is already
        # rejected above; a non-None value here would otherwise be accepted and
        # ignored, which is exactly how the ancestor turned an elastic-net request
        # into an unannounced ridge fit.
        if self.l1_ratio is not None:
            raise UnsupportedParameterError(
                f"{name} implements penalty='l2' and penalty=None only, so l1_ratio "
                f"has no meaning on the MLX path; got l1_ratio={self.l1_ratio!r}.",
                parameter="l1_ratio",
                value=self.l1_ratio,
            )

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

    def _below_crossover(self, n_samples: int, n_features: int) -> tuple[bool, str]:
        """Whether stock scikit-learn is expected to win at this size.

        The threshold is **high**, and both conditions must be met, because the
        measurements say so. See ``docs/benchmarks.md``.

        The iteration counts match scikit-learn's almost exactly — 17 against 17,
        20 against 19 — so the solver is not doing extra work. What it pays is
        per-iteration overhead: at 20 000 x 16 an L-BFGS iteration costs about
        5.9 ms while the matmul inside it costs 163 us. The line search evaluates
        the objective several times per step and each evaluation forces a host
        synchronization, so the fixed cost of crossing the boundary dominates
        everything MLX is good at until the arithmetic per iteration becomes large.

        Measured: 0.02x-0.11x up to 60 000 x 128, reaching 1.26x only at
        20 000 x 1024 and 1.34x at 60 000 x 1024. So MLX is used only for tall
        *and* wide problems; everything else goes to scikit-learn, which is faster
        and says so honestly rather than being dressed up by a threshold that
        pretends otherwise.

        This is a known optimization target, not a permanent property — see
        ``docs/benchmarks.md`` for what would change it.
        """
        tuning = get_tuning()
        work = n_samples * n_features
        if n_samples < tuning.logreg_min_samples or work < tuning.logreg_min_work:
            return True, (
                f"{n_samples} samples x {n_features} features is below the measured "
                f"logistic-regression crossover ({tuning.logreg_min_samples} samples "
                f"and {tuning.logreg_min_work} element-ops, both required)"
            )
        return False, ""


class LogisticRegression(MLXLogisticMixin, _SkLogisticRegression):
    """L2-penalized logistic regression fitted with L-BFGS on MLX.

    A drop-in replacement for :class:`sklearn.linear_model.LogisticRegression`
    restricted to ``solver="lbfgs"`` with ``penalty="l2"`` or ``penalty=None`` on
    dense, single-output data. The objective minimized is scikit-learn's, term for
    term, so the two agree on the objective value at their respective solutions.

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
        backend = self._select_backend(
            "fit",
            capability_check=lambda: self._check_logistic_capability(X, y),
            shape=_safe_shape(X),
        )
        if backend == "sklearn":
            super().fit(X, y, sample_weight)
            self._set_backend("sklearn")
            return self

        X_checked, y_checked = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            dtype=NUMPY_FLOAT,
            order="C",
            ensure_min_samples=1,
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

        self.classes_, y_encoded = np.unique(y_checked, return_inverse=True)
        n_classes = self.classes_.shape[0]
        if n_classes < 2:
            # scikit-learn's wording, verbatim: code written against its error
            # contract matches on this string.
            raise ValueError(
                "This solver needs samples of at least 2 classes in the data, but "
                f"the data contains only one class: {self.classes_[0]!r}"
            )

        weights = self._resolve_weights(X_checked, y_checked, y_encoded, sample_weight)

        if self.penalty is None:
            if self.C != 1.0:
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
        )

        with mlx_guard(type(self).__name__, "fit"):
            solution = solve_logistic(
                X_checked,
                y_encoded.astype(np.int32, copy=False),
                n_classes=n_classes,
                sample_weight=weights,
                config=config,
            )

        self.coef_ = solution.coef
        self.intercept_ = solution.intercept
        self.n_iter_ = np.array([solution.n_iter], dtype=np.int32)
        self._set_backend(backend)

        if not solution.converged:
            # The ancestor unpacked `converged` and discarded it at every call
            # site, so a model that stopped at max_iter was indistinguishable from
            # one that reached the tolerance.
            warnings.warn(
                f"{type(self).__name__} failed to converge "
                f"({solution.stop_reason}) after {solution.n_iter} iterations; "
                f"max|gradient| is {solution.grad_max_abs:.3g} against tol="
                f"{self.tol:g}. Increase max_iter, scale the data, or relax tol.",
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


def _safe_shape(X) -> tuple[int, ...] | None:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))
