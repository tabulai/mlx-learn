"""Shared neighbor-estimator machinery.

The three neighbor estimators differ only in what they do with the indices they
get back, so everything up to and including ``kneighbors`` lives here.

The direct estimators use scikit-learn's public BaseEstimator and task mixins, with
matching constructors, parameters and tags. They support cloning, pipelines,
metadata routing and serialization without eagerly importing sklearn.neighbors.
They are not subclasses of its concrete neighbor classes: genuine isinstance
checks against a captured stock neighbor class intentionally differ. Stock
classes load only for optional radius operations, uncommon parameter validation
or an explicitly permitted capability fallback. Patched twins additionally
inherit the original stock classes because patching already imports them.

Layer 1 never calls that fallback. ``_mlxlearn_allow_fallback`` is ``False`` here,
so a capability mismatch raises. Layer 2 flips the flag to permit an original
sklearn delegate on capability mismatches; owned numerical search is unchanged.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from sklearn import get_config as get_sklearn_config

from .._common.arrays import NUMPY_FLOAT
from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_config, get_tuning
from .._common.diagnostics import record_event
from .._common.preparation import cast_float32_checked
from .._common.validation import (
    check_classification_targets,
    check_float32_range,
    check_is_fitted,
    require_dense,
    require_supported_params,
    validate_data,
)
from .._kernels.distance import build_index
from ..exceptions import CapabilityError, UnsupportedParameterError

__all__ = ["MLXNeighborsMixin", "SUPPORTED_METRICS"]

#: Metrics whose squared-distance form is exactly what the blocked Gram kernel
#: computes. Anything else is a capability mismatch, not an approximation
#: opportunity: returning Euclidean results for a Manhattan request would be a
#: wrong answer delivered quickly.
SUPPORTED_METRICS = ("euclidean", "l2", "minkowski")

_SUPPORTED_ALGORITHMS = ("auto", "brute")

_STOCK_NEIGHBORS = {}
# Private seam for radius callers/tests; None means resolve the original lazily.
_SK_NEAREST_NEIGHBORS = None


def _register_sklearn_neighbor_class(name, original):
    """Capture originals before the patching layer rebinds public sklearn names."""
    _STOCK_NEIGHBORS.setdefault(name, original)


def _stock_neighbor_class(name):
    if name not in _STOCK_NEIGHBORS:
        from importlib import import_module

        original = getattr(import_module("sklearn.neighbors"), name)
        if issubclass(original, MLXNeighborsMixin):
            raise RuntimeError("The original sklearn neighbor class was not captured before patching")
        _STOCK_NEIGHBORS[name] = original
    return _STOCK_NEIGHBORS[name]


class _ParameterConstraints:
    """Retain sklearn's version-specific constraints without eager tree imports."""

    def __get__(self, instance, owner):
        return _stock_neighbor_class(owner._sklearn_class_name)._parameter_constraints


_DEFAULT_PARAMETER_CONSTRAINTS = _ParameterConstraints()


class MLXNeighborsMixin(BackendMixin):
    """``fit`` and ``kneighbors`` for the MLX brute-force neighbor path."""

    _parameter_constraints = _DEFAULT_PARAMETER_CONSTRAINTS

    def _validate_params(self):
        # Extensions can replace the private constraints on an instance or any
        # class in its MRO, including after construction. Only the exact default
        # descriptor justifies the stock-valid whitelist; never cache this check.
        if "_parameter_constraints" in self.__dict__:
            return super()._validate_params()
        for cls in type(self).__mro__:
            if "_parameter_constraints" in cls.__dict__:
                if cls.__dict__["_parameter_constraints"] is not _DEFAULT_PARAMETER_CONSTRAINTS:
                    return super()._validate_params()
                break
        # Accept only forms known to satisfy sklearn's constraints. Other
        # values use its own validator, preserving version-specific errors.
        weights = getattr(self, "weights", "uniform")
        radius = getattr(self, "radius", None)
        if (
            (self.n_neighbors is None or isinstance(self.n_neighbors, Integral) and self.n_neighbors >= 1)
            and isinstance(self.algorithm, str) and self.algorithm in {"auto", "brute", "kd_tree", "ball_tree"}
            and isinstance(self.leaf_size, Integral) and self.leaf_size >= 1
            and (self.p is None or isinstance(self.p, Real) and self.p > 0)
            and isinstance(self.metric, str) and self.metric in SUPPORTED_METRICS
            and (self.metric_params is None or isinstance(self.metric_params, dict))
            and (self.n_jobs is None or isinstance(self.n_jobs, Integral))
            and isinstance(weights, str) and weights in {"uniform", "distance"}
            and (radius is None or isinstance(radius, Real) and radius >= 0)
        ):
            return
        return super()._validate_params()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = self._mlxlearn_allow_fallback
        tags.input_tags.pairwise = self.metric == "precomputed"
        tags.input_tags.positive_only = tags.input_tags.pairwise
        tags.input_tags.allow_nan = self.metric == "nan_euclidean"
        tags.target_tags.multi_output = self._mlxlearn_allow_fallback
        if tags.classifier_tags is not None:
            tags.classifier_tags.multi_label = self._mlxlearn_allow_fallback
        return tags

    def _fit_sklearn(self, X, y=None):
        model = _stock_neighbor_class(self._sklearn_class_name)(**self.get_params(deep=False))
        model.fit(X, y)
        self.__dict__.pop("_index_", None)
        self.__dict__.update(model.__dict__)
        self._sklearn_model_ = model
        self._set_backend("sklearn")
        return self

    def _sklearn_delegate(self):
        model = self.__dict__.get("_sklearn_model_")
        if model is None:
            model = self._knn_query_state_.get("fallback")
        if model is None:
            # Old sklearn-fitted pickles stored the stock fitted attributes on
            # this object. Recover their behavior without refitting.
            model = _stock_neighbor_class(self._sklearn_class_name)(**self.get_params(deep=False))
            model.__dict__.update({key: value for key, value in self.__dict__.items()
                                   if key not in {"_index_", "_knn_query_state_", "_sklearn_model_"}})
            self._knn_query_state_["fallback"] = model
        # Parameter updates affect inference, but the fit backend stays sticky.
        model.set_params(**self.get_params(deep=False))
        return model

    def _radius_delegate(self):
        if self._fitted_backend("radius_neighbors") == "sklearn":
            return self._sklearn_delegate()
        factory = _SK_NEAREST_NEIGHBORS or _stock_neighbor_class("NearestNeighbors")
        model = factory(**self.get_params(deep=False))
        # Reuse exactly the validated brute-force state formerly read by the
        # inherited radius methods, including the mutable compatibility mirror.
        model.__dict__.update({key: value for key, value in self.__dict__.items()
                               if key not in {"_index_", "_knn_query_state_", "_sklearn_model_"}})
        model._fit_X = self._fit_X
        return model

    @property
    def execution_policy_(self):
        if "_execution_backend_" not in self.__dict__:
            raise AttributeError("execution_policy_")
        return self.__dict__.get("_knn_policy_", "fixed")

    @property
    def last_query_backend_(self):
        backend = self.__dict__.get("_knn_query_state_", {}).get("backend")
        if "_execution_backend_" not in self.__dict__ or backend is None:
            raise AttributeError("last_query_backend_")
        return backend

    def _set_query_backend(self, backend):
        # Diagnostics and lazy search caches live inside fitted state; inference
        # does not add or replace estimator attributes (sklearn's contract).
        self._knn_query_state_["backend"] = backend

    def _set_backend(self, backend):
        super()._set_backend(backend)
        self._knn_query_state_ = {"backend": None}

    @property
    def _fit_X(self):
        # Upstream sklearn writes this attribute during fallback fits. On an
        # MLX fit, defer its independent host mirror until radius queries or
        # explicit private-state access need it.
        if "_fit_X" in self.__dict__:
            return self.__dict__["_fit_X"]
        index = self.__dict__.get("_index_")
        if index is None:
            raise AttributeError("_fit_X")
        return index.X

    @_fit_X.setter
    def _fit_X(self, value):
        self.__dict__["_fit_X"] = value

    # --------------------------------------------------------------- capability

    def _check_neighbors_capability(self, X, y=None):
        """Raise :class:`CapabilityError` for anything the MLX path cannot serve.

        Runs before ``validate_data``, so a float64 value that overflows float32
        is reported as a capability limit — which scikit-learn's float64 path can
        serve, and therefore which patched mode can fall back on — rather than as
        scikit-learn's generic post-conversion "contains infinity" ``ValueError``.
        """
        require_dense(X, estimator=type(self).__name__)
        prepared, finite = _prepare_neighbors_input(X)

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
        return prepared, finite

    # --------------------------------------------------------------------- fit

    def _fit_neighbors(self, X, y=None, *, target: str | None = None):
        """Validate and own training data, deferring search preparation as needed.

        Validation and private storage are eager. Automatic host fits can defer
        device selection and GPU preparation until a query requires them.

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
        self.__dict__.pop("_knn_policy_", None)
        self.__dict__.pop("_knn_query_state_", None)
        # Targets cached on a device belong to this fit alone, including when a
        # refit changes backend or fails partway through validation.
        self.__dict__.pop("_y_device_", None)
        self.__dict__.pop("_y_checked_", None)
        self.__dict__.pop("_sklearn_model_", None)
        self._validate_params()
        automatic = get_config().device == "auto" and not get_tuning().force_backend
        prepared_X, finite_checked = X, False

        def check_capability():
            nonlocal prepared_X, finite_checked
            prepared_X, finite_checked = self._check_neighbors_capability(X, y)

        if automatic:
            # An owned host fit does not need a usable GPU yet. Capability
            # policy still runs before validation, but the real device probe
            # can wait until a validated workload actually needs GPU storage.
            try:
                check_capability()
            except CapabilityError as exc:
                return self._handle_capability_error(exc, "fit", shape=_safe_shape(X))
        else:
            backend = self._select_backend(
                "fit",
                capability_check=check_capability,
                shape=_safe_shape(X),
            )
            if backend == "sklearn":
                return "sklearn"

        if target is None:
            if (automatic and type(prepared_X) is np.ndarray and prepared_X.ndim == 2
                    and prepared_X.shape[0] <= 2048 and prepared_X.shape[1] <= 64
                    and _canonical_float32_matrix(prepared_X)):
                # This owned-search boundary needs exactly the same canonical
                # matrix requirements as query validation. sklearn still resets
                # feature metadata; other containers use its ordinary checks.
                X_checked = validate_data(self, prepared_X, skip_check_array=True, reset=True)
            else:
                X_checked = validate_data(
                    self, prepared_X, accept_sparse=False, dtype=NUMPY_FLOAT, order="C",
                    ensure_min_samples=1, ensure_all_finite=not finite_checked, reset=True,
                )
        else:
            X_checked, y_checked = validate_data(
                self,
                prepared_X,
                y,
                accept_sparse=False,
                dtype=NUMPY_FLOAT,
                order="C",
                ensure_min_samples=1,
                ensure_all_finite=not finite_checked,
                multi_output=False,
                y_numeric=target == "regression",
                reset=True,
            )
            if target == "classification":
                check_classification_targets(y_checked)
            self._y_checked_ = y_checked

        if automatic:
            n_samples, n_features = X_checked.shape
            if ((n_samples <= 2048 and n_features <= 64)
                    or (n_samples <= 16384 and n_features <= 32)):
                backend = "cpu"
                gpu_available = None
                record_event(type(self).__name__, "fit", backend,
                             reason="auto-fit-storage", shape=X_checked.shape)
            else:
                backend = self._select_backend("fit", shape=X_checked.shape)
                gpu_available = backend == "mlx"
        else:
            gpu_available = backend == "mlx"

        with mlx_guard(type(self).__name__, "fit"):
            # A checked wide-float cast already allocated private contiguous
            # storage. Transfer it when validation returned that same buffer;
            # ordinary float32/user containers still require an owned copy.
            owns_cast = finite_checked and prepared_X is not X and X_checked is prepared_X
            self._index_ = build_index(
                X_checked, backend=backend, _owned=owns_cast,
                _auto=automatic, _gpu_available=gpu_available,
            )

        self.n_samples_fit_ = int(X_checked.shape[0])
        self.effective_metric_ = "euclidean"
        self.effective_metric_params_ = {}
        # Optional radius methods delegate to sklearn's public brute-force
        # implementation. Preserve the state it reads, including the float32
        # compatibility mirror used by the preceding inherited implementation.
        self._fit_method = "brute"
        self.__dict__.pop("_fit_X", None)
        self._tree = None
        self._set_backend(backend)
        self._knn_policy_ = "auto" if automatic else "fixed"
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
            self._set_query_backend("sklearn")
            return self._sklearn_delegate().kneighbors(X, n_neighbors, return_distance)

        Q, k, query_is_train = self._prepare_neighbor_query(X, n_neighbors)
        with mlx_guard(type(self).__name__, "kneighbors"):
            dist, idx = self._index_.query(
                Q, k, return_distance=return_distance, exclude_self=query_is_train
            )
        self._record_neighbor_query("kneighbors", Q.shape)
        return (dist, idx) if return_distance else idx

    def kneighbors_graph(self, X=None, n_neighbors=None, mode="connectivity"):
        """Return the CSR graph of selected neighbors, excluding self for X=None."""
        check_is_fitted(self)
        if mode not in ("connectivity", "distance"):
            raise ValueError('Unsupported mode, must be one of "connectivity", '
                             f'or "distance" but got "{mode}" instead')
        if mode == "distance":
            distances, indices = self.kneighbors(X, n_neighbors, return_distance=True)
            values = distances.ravel()
        else:
            indices = self.kneighbors(X, n_neighbors, return_distance=False)
            values = np.ones(indices.size)
        from scipy import sparse

        count, k = indices.shape
        indptr = np.arange(0, indices.size + 1, k)
        constructor = (sparse.csr_array if get_sklearn_config().get("sparse_interface") == "sparray"
                       else sparse.csr_matrix)
        return constructor((values, indices.ravel(), indptr), shape=(count, self.n_samples_fit_))

    def _prepare_neighbor_query(self, X, n_neighbors=None):
        """Validate once before either public or device-resident inference."""
        query_is_train = X is None
        k = self._resolve_n_neighbors(n_neighbors, query_is_train)
        if query_is_train:
            Q = self._index_.training_data
        else:
            require_dense(X, estimator=type(self).__name__, operation="kneighbors")
            Q = _validate_neighbor_query(self, X)
        return Q, k, query_is_train

    def _record_neighbor_query(self, operation, shape):
        backend = self._index_.last_query_backend
        self._set_query_backend(backend)
        if self.execution_policy_ == "auto":
            record_event(type(self).__name__, operation, backend,
                         reason="auto-query", shape=shape)
        else:
            self._note_inference(operation, shape=shape)

    def _resolve_n_neighbors(self, n_neighbors, query_is_train: bool) -> int:
        """Validate ``n_neighbors`` with scikit-learn's exact error messages."""
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if not isinstance(n_neighbors, Integral):
            raise TypeError(
                f"n_neighbors does not take {type(n_neighbors)} value, "
                "enter integer value"
            )
        if n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")

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
        if need_distance:
            return self.kneighbors(X, return_distance=True)
        return None, self.kneighbors(X, return_distance=False)

    def _query_device_indices(self, X, *, need_distance: bool, squared: bool = False):
        """Keep neighbors on-device for voting before any host conversion."""
        Q, k, query_is_train = self._prepare_neighbor_query(X)
        with mlx_guard(type(self).__name__, "kneighbors"):
            result = self._index_.query_device(
                Q, k, return_distance=need_distance, exclude_self=query_is_train,
                _squared=squared,
            )
        self._record_neighbor_query("kneighbors", Q.shape)
        return result

    def __getstate__(self):
        # Encoded labels remain in _y; device copies can be rebuilt lazily, just
        # like the search index, without embedding GPU buffers in the pickle.
        state = super().__getstate__()
        state.pop("_y_device_", None)
        # The mirrored stock state is sufficient to recover fallback inference.
        # Do not serialize a second estimator or rely on its public binding
        # remaining unchanged while sklearn is patched.
        state.pop("_sklearn_model_", None)
        if "_knn_query_state_" in state:
            state["_knn_query_state_"] = {
                "backend": state["_knn_query_state_"].get("backend")
            }
        if state.get("_execution_backend_") in {"mlx", "cpu"}:
            # Preserve the prior pickle layout for existing model readers.
            state["_fit_X"] = self._fit_X
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        # Older fitted estimators predate per-query diagnostics.
        if "_execution_backend_" in self.__dict__:
            self.__dict__.setdefault("_knn_query_state_", {"backend": None})


def _safe_shape(X) -> tuple[int, ...] | None:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))


def _prepare_neighbors_input(X):
    """Check dense float inputs once while preserving container metadata.

    Wide-float ndarray casts allocate private contiguous float32 storage.
    Finiteness is checked on that buffer in the same pipeline used by
    logistic regression. Inspect the original only on failure to distinguish a
    finite value outside float32's range from invalid NaN/Inf input.

    For large native float32 arrays, the vectorized elementwise check avoids
    sklearn's slower sum-based scan. Invalid arrays still go through sklearn's
    normal validation and error messages. The input remains caller-owned.
    """
    if type(X) is np.ndarray and X.dtype == np.float32 and X.size >= 1 << 18:
        return X, bool(np.isfinite(X).all())
    if type(X) is np.ndarray and X.dtype.kind == "f" and X.dtype.itemsize > 4:
        prepared, finite = cast_float32_checked(X)
        if not finite:
            check_float32_range(X)
        return prepared, finite
    check_float32_range(X)
    return X, False


def _canonical_float32_matrix(X):
    return (type(X) is np.ndarray and X.dtype == NUMPY_FLOAT and X.dtype.metadata is None
            and X.ndim == 2 and X.shape[0] > 0 and X.shape[1] > 0
            and X.flags.c_contiguous and np.isfinite(X).all())


def _validate_neighbor_query(estimator, X):
    """Check an already canonical array without repeating generic conversion.

    This narrow case satisfies every array requirement of the query boundary:
    native float32, C order, two nonempty dimensions and finite coordinates.
    sklearn still checks feature names and the fitted feature count. All other
    inputs keep its ordinary conversion, validation and error messages.
    """
    if _canonical_float32_matrix(X):
        return validate_data(estimator, X, skip_check_array=True, reset=False)
    prepared, finite = _prepare_neighbors_input(X)
    return validate_data(
        estimator, prepared, accept_sparse=False, dtype=NUMPY_FLOAT, order="C",
        ensure_all_finite=not finite, reset=False,
    )


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
