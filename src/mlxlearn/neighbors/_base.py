"""Shared neighbor-estimator machinery.

The three neighbor estimators differ only in what they do with the indices they
get back, so everything up to and including ``kneighbors`` lives here.

Each mlxlearn estimator subclasses its scikit-learn counterpart. That is not
laziness — it is what makes the drop-in promise real. Subclassing inherits the
exact constructor signature, ``get_params``/``set_params``, the parameter
constraints, and the estimator tags, so ``clone``, ``Pipeline``, ``GridSearchCV``
and ``isinstance`` all keep working without mlxlearn re-deriving any of it. It
also gives the patching layer an exact scikit-learn implementation to fall back
to: ``super().fit(...)`` *is* stock scikit-learn.

Layer 1 never calls that fallback. ``_mlxlearn_allow_fallback`` is ``False`` here,
so a capability mismatch raises. Layer 2 flips the flag; nothing else differs.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from sklearn.neighbors import NearestNeighbors as _SkNearestNeighbors

from .._common.arrays import INDEX_DTYPE, NUMPY_FLOAT
from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_tuning
from .._common.validation import (
    check_classification_targets,
    check_finite_float32,
    check_float32_range,
    check_is_fitted,
    require_dense,
    require_supported_params,
    validate_data,
)
from .._kernels.distance import build_index
from ..exceptions import UnsupportedParameterError

__all__ = ["MLXNeighborsMixin", "SUPPORTED_METRICS"]

#: Metrics whose squared-distance form is exactly what the blocked Gram kernel
#: computes. Anything else is a capability mismatch, not an approximation
#: opportunity: returning Euclidean results for a Manhattan request would be a
#: wrong answer delivered quickly.
SUPPORTED_METRICS = ("euclidean", "l2", "minkowski")

_SUPPORTED_ALGORITHMS = ("auto", "brute")


class MLXNeighborsMixin(BackendMixin):
    """``fit`` and ``kneighbors`` for the MLX brute-force neighbor path."""

    # --------------------------------------------------------------- capability

    def _check_neighbors_capability(self, X, y=None) -> None:
        """Raise :class:`CapabilityError` for anything the MLX path cannot serve.

        Runs before ``validate_data``, so a float64 value that overflows float32
        is reported as a capability limit — which scikit-learn's float64 path can
        serve, and therefore which patched mode can fall back on — rather than as
        scikit-learn's generic post-conversion "contains infinity" ``ValueError``.
        """
        require_dense(X, estimator=type(self).__name__)
        check_float32_range(X)

        require_supported_params(
            {"algorithm": self.algorithm, "metric": self.metric},
            {"algorithm": _SUPPORTED_ALGORITHMS, "metric": SUPPORTED_METRICS},
            estimator=type(self).__name__,
        )

        # 'minkowski' is only Euclidean at p == 2. The ancestor short-circuited
        # this check whenever the metric was spelled 'euclidean' or 'l2' and then
        # ignored p entirely; here p is checked against the metric that will
        # actually be used.
        if self.metric == "minkowski" and self.p != 2:
            raise UnsupportedParameterError(
                f"{type(self).__name__} implements metric='minkowski' only at p=2 "
                f"(Euclidean) on the MLX path; got p={self.p!r}.",
                parameter="p",
                value=self.p,
            )

        # metric_params was accepted and silently ignored by the ancestor, so a
        # weighted-Minkowski request returned unweighted Euclidean answers with no
        # warning. Wrong answers are worse than a fallback.
        if getattr(self, "metric_params", None):
            raise UnsupportedParameterError(
                f"{type(self).__name__} does not implement metric_params on the MLX "
                f"path; got {self.metric_params!r}. mlxlearn will not ignore it and "
                "return unweighted Euclidean distances.",
                parameter="metric_params",
                value=self.metric_params,
            )

        weights = getattr(self, "weights", "uniform")
        if callable(weights):
            raise UnsupportedParameterError(
                f"{type(self).__name__} does not implement a callable weights "
                "function on the MLX path.",
                parameter="weights",
                value=weights,
            )

        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.ndim > 1 and y_arr.shape[1] > 1:
                raise UnsupportedParameterError(
                    f"{type(self).__name__} supports single-output targets only on the "
                    f"MLX path; got y with shape {y_arr.shape}.",
                    parameter="y",
                    value=y_arr.shape,
                )

    def _below_crossover(self, n_samples: int, n_features: int) -> tuple[bool, str]:
        """Whether stock scikit-learn is expected to win at this size.

        Uses the measured crossover from ``docs/benchmarks.md``. Handing small
        problems away is the mechanism behind "never slower than scikit-learn";
        without it, kernel-launch and transfer overhead dominates and the
        acceleration is negative.
        """
        tuning = get_tuning()
        work = n_samples * n_features
        if n_samples < tuning.knn_min_train_samples and work < tuning.knn_min_work:
            return True, (
                f"{n_samples} training samples x {n_features} features is below the "
                f"measured KNN crossover ({tuning.knn_min_train_samples} samples)"
            )
        return False, ""

    # --------------------------------------------------------------------- fit

    def _fit_neighbors(self, X, y=None, *, target: str | None = None):
        """Validate, choose a backend, and build the index. Eager, always.

        Parameters
        ----------
        target : {"classification", "regression"}, optional
            What ``y`` is. ``None`` means unsupervised and ``y`` is ignored.
            Passed through to scikit-learn's own validation so that a continuous
            target given to a classifier, a target containing NaN, and a missing
            ``y`` all produce scikit-learn's error rather than an ``IndexError``
            from somewhere further in.

        Returns
        -------
        str
            The chosen backend, or ``"sklearn"`` (Layer 2 only), in which case
            the caller must run scikit-learn's own ``fit``.
        """
        self._clear_backend()

        backend = self._select_backend(
            "fit",
            capability_check=lambda: self._check_neighbors_capability(X, y),
            shape=_safe_shape(X),
        )
        if backend == "sklearn":
            return "sklearn"

        if target is None:
            X_checked = validate_data(
                self, X, accept_sparse=False, dtype=NUMPY_FLOAT,
                ensure_min_samples=1, reset=True,
            )
        else:
            X_checked, y_checked = validate_data(
                self,
                X,
                y,
                accept_sparse=False,
                dtype=NUMPY_FLOAT,
                ensure_min_samples=1,
                multi_output=False,
                y_numeric=target == "regression",
                reset=True,
            )
            if target == "classification":
                check_classification_targets(y_checked)
            self._y_checked_ = y_checked
        check_finite_float32(X_checked)

        below, detail = self._below_crossover(*X_checked.shape)
        if below and backend != "cpu":
            backend = self._select_backend(
                "fit", below_crossover=True, crossover_detail=detail,
                shape=X_checked.shape,
            )
            if backend == "sklearn":
                return "sklearn"

        with mlx_guard(type(self).__name__, "fit"):
            self._index_ = build_index(X_checked, backend=backend)

        self.n_samples_fit_ = int(X_checked.shape[0])
        self.effective_metric_ = "euclidean"
        self.effective_metric_params_ = {}
        # scikit-learn's RadiusNeighborsMixin is inherited unchanged, so it needs
        # the state its own brute-force path reads. Populating it means
        # radius_neighbors stays correct on an MLX-fitted estimator; it runs on
        # the CPU in float32, which docs/fp32_policy.md records.
        self._fit_method = "brute"
        self._fit_X = self._index_.X
        self._tree = None
        self._set_backend(backend)
        return backend

    # -------------------------------------------------------------- kneighbors

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Find the k nearest neighbors of each point.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), default=None
            Query points. ``None`` means "the training data", and each point's own
            index is excluded from its neighbors.
        n_neighbors : int, default=None
            Defaults to the estimator's ``n_neighbors``.
        return_distance : bool, default=True
            Whether to return distances alongside indices.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Euclidean distances, float64, ascending. Only when
            ``return_distance=True``.
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Training indices, int64, ordered by ``(distance, index)``.
        """
        check_is_fitted(self)
        if self._fitted_backend("kneighbors") == "sklearn":
            self._note_inference("kneighbors")
            return super().kneighbors(X, n_neighbors, return_distance)

        query_is_train = X is None
        k = self._resolve_n_neighbors(n_neighbors, query_is_train)

        if query_is_train:
            Q = self._index_.X
        else:
            require_dense(X, estimator=type(self).__name__, operation="kneighbors")
            Q = validate_data(self, X, accept_sparse=False, dtype=NUMPY_FLOAT, reset=False)
            check_finite_float32(Q)

        self._note_inference("kneighbors", shape=Q.shape)
        with mlx_guard(type(self).__name__, "kneighbors"):
            dist, idx = self._index_.query(
                Q, k, return_distance=return_distance, exclude_self=query_is_train
            )
        return (dist, idx) if return_distance else idx

    def _resolve_n_neighbors(self, n_neighbors, query_is_train: bool) -> int:
        """Validate ``n_neighbors`` with scikit-learn's exact error messages."""
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")
        elif not isinstance(n_neighbors, Integral):
            raise TypeError(
                f"n_neighbors does not take {type(n_neighbors)} value, "
                "enter integer value"
            )

        n_samples_fit = self.n_samples_fit_
        limit = n_samples_fit - int(query_is_train)
        if n_neighbors > limit:
            inequality = "n_neighbors < n_samples_fit" if query_is_train else (
                "n_neighbors <= n_samples_fit"
            )
            raise ValueError(
                f"Expected {inequality}, but n_neighbors = {n_neighbors}, "
                f"n_samples_fit = {n_samples_fit}"
            )
        return int(n_neighbors)

    # ------------------------------------------------------------------ helpers

    def _query_indices(self, X, *, need_distance: bool):
        """One neighbor-acquisition path for ``predict`` and ``predict_proba``.

        The ancestor had three — a prefix-cached ordered query for small k, an
        unordered query for large k, and a distance-returning query for
        ``weights='distance'`` — with nothing asserting they agreed on the
        candidate set. One path cannot disagree with itself.
        """
        dist, idx = self.kneighbors(X, return_distance=True)
        return (dist if need_distance else None), idx


def _safe_shape(X) -> tuple[int, ...] | None:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))


def compute_weights(dist: np.ndarray | None, weights) -> np.ndarray | None:
    """Turn neighbor distances into per-neighbor weights.

    Reimplements the ~15 lines of ``sklearn.neighbors._base._get_weights`` that
    mlxlearn needs, rather than importing a private symbol whose rename in any
    scikit-learn release would make the module unimportable.

    ``'distance'`` weighting is ``1/d``. A query that coincides exactly with a
    training point would give that neighbor infinite weight, so — matching
    scikit-learn — any row containing a zero distance is replaced by the 0/1
    indicator of its zero entries: coincident points get all the weight and every
    other neighbor gets none.
    """
    if weights in (None, "uniform"):
        return None
    if weights != "distance":
        raise ValueError(f"weights not recognized: should be 'uniform' or 'distance', got {weights!r}")
    if dist is None:
        raise ValueError("weights='distance' requires neighbor distances.")

    with np.errstate(divide="ignore"):
        w = 1.0 / np.asarray(dist, dtype=np.float64)

    coincident = np.isinf(w)
    rows_with_coincident = np.any(coincident, axis=1)
    if np.any(rows_with_coincident):
        w[rows_with_coincident] = coincident[rows_with_coincident].astype(np.float64)

    if np.any(np.all(w == 0, axis=1)):
        raise ValueError(
            "All neighbors of some sample are getting zero weights. Please modify "
            "'weights' to avoid this case if you are using a user-defined function."
        )
    return w


def weighted_mode(codes: np.ndarray, weights: np.ndarray | None, n_classes: int) -> np.ndarray:
    """Per-row weighted majority vote over integer class codes.

    Ties go to the smallest class code, matching scikit-learn. Vectorized: the
    ancestor imported ``sklearn.utils.extmath.weighted_mode`` and
    ``sklearn.utils.fixes._mode``, both private.
    """
    n_queries, k = codes.shape
    scores = np.zeros((n_queries, n_classes), dtype=np.float64)
    rows = np.repeat(np.arange(n_queries), k)
    flat_codes = codes.ravel()
    flat_weights = np.ones(n_queries * k) if weights is None else weights.ravel()
    np.add.at(scores, (rows, flat_codes), flat_weights)
    # argmax returns the first maximum, i.e. the smallest class code on a tie.
    return scores.argmax(axis=1).astype(INDEX_DTYPE)


def class_probabilities(codes: np.ndarray, weights: np.ndarray | None, n_classes: int) -> np.ndarray:
    """Per-row normalized class scores, shape ``(n_queries, n_classes)``, float64."""
    n_queries, k = codes.shape
    proba = np.zeros((n_queries, n_classes), dtype=np.float64)
    rows = np.repeat(np.arange(n_queries), k)
    flat_weights = np.ones(n_queries * k) if weights is None else weights.ravel()
    np.add.at(proba, (rows, codes.ravel()), flat_weights)
    total = proba.sum(axis=1, keepdims=True)
    np.divide(proba, np.where(total > 0, total, 1.0), out=proba)
    return proba


# Re-exported so the estimator modules do not each import scikit-learn's base.
_SK_NEAREST_NEIGHBORS = _SkNearestNeighbors
