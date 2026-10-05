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

import warnings
from numbers import Integral
from typing import ClassVar

import numpy as np
from sklearn import config_context as _sklearn_config_context
from sklearn.base import BaseEstimator
from sklearn.neighbors import VALID_METRICS, BallTree, KDTree
from sklearn.neighbors import NearestNeighbors as _SkNearestNeighbors

from .._common.arrays import INDEX_DTYPE
from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_tuning
from .._common.device import inference_device
from .._common.diagnostics import record_inference
from .._common.validation import (
    SparseInputError,
    _safe_shape,
    check_float32_range,
    check_is_fitted,
    require_dense,
    require_supported_params,
    validate_data,
    validate_float32_input,
)
from .._kernels.distance import build_index
from ..exceptions import CapabilityError, UnsupportedInputError, UnsupportedParameterError

__all__ = ["MLXNeighborsMixin", "SUPPORTED_METRICS"]

#: Metrics whose squared-distance form is exactly what the blocked Gram kernel
#: computes — ``"sqeuclidean"`` is that form itself. Anything else is a capability
#: mismatch, not an approximation opportunity: returning Euclidean results for a
#: Manhattan request would be a wrong answer delivered quickly.
SUPPORTED_METRICS = ("euclidean", "l2", "minkowski", "sqeuclidean")

_SUPPORTED_ALGORITHMS = ("auto", "brute")

#: Validation keeps float64 input as float64: the index translates it by its float64
#: mean and keeps a float32 residual, so a cast here would throw away precision the
#: caller supplied (Unix timestamps collapse to 128-second steps in float32).
_INPUT_DTYPES = (np.float64, np.float32)

#: scikit-learn's ``algorithm="auto"`` never builds a tree above this many features.
_SKLEARN_TREE_MAX_FEATURES = 15


class MLXNeighborsMixin(BackendMixin):
    """``fit`` and ``kneighbors`` for the MLX brute-force neighbor path."""

    #: Neighbor search never looks at ``y``; a multi-output target changes only the
    #: host-side vote or average over the neighbors found, which is done per output
    #: exactly as scikit-learn does it. So both layers take them, and the Layer 1
    #: tags say so.
    _mlxlearn_supports_multi_output: ClassVar[bool] = True

    #: Fitted state only the MLX path creates, dropped at the start of every fit so
    #: that a refit which ends up on scikit-learn — or raises — does not keep the
    #: previous training set alive in memory and in pickles. ``_fit_X`` is here
    #: because on the MLX path it is a view of the index's host copy; scikit-learn's
    #: own fit sets it again.
    _mlxlearn_mlx_attrs: ClassVar[tuple[str, ...]] = ("_index_", "_fit_X", "_host_brute_")

    # --------------------------------------------------------------- capability

    def _check_neighbors_capability(self, X) -> None:
        """Raise :class:`CapabilityError` for anything the MLX path cannot serve.

        The input's kind and the parameters only: nothing here scans the data. The
        one check that does — float32 range — is the fit's ``data_check``, which
        patched mode skips when it hands a fit to scikit-learn on size alone:
        scikit-learn validates the data itself, and scanning it first is the kind
        of overhead that made patched fits slower than the stock ones they
        replaced.
        """
        name = type(self).__name__

        # scikit-learn's own fit also accepts a prebuilt KDTree/BallTree or another
        # fitted neighbors estimator in place of X. There is nothing for the MLX
        # path to index in one; saying so is what lets patched mode hand it over
        # instead of failing inside float32 conversion with a TypeError.
        if isinstance(X, (KDTree, BallTree, BaseEstimator)):
            raise UnsupportedInputError(
                f"{name}.fit received a prebuilt {type(X).__name__} as X. mlxlearn "
                "builds its own index from an array and cannot reuse a scikit-learn "
                "tree or fitted estimator. Pass the training array instead.",
                reason="prebuilt-index",
                advice="Under patch_sklearn(), scikit-learn handles a prebuilt index.",
            )
        require_dense(X, estimator=name)

        require_supported_params(
            {"algorithm": self.algorithm, "metric": self.metric},
            {"algorithm": _SUPPORTED_ALGORITHMS, "metric": SUPPORTED_METRICS},
            estimator=name,
        )

        # 'minkowski' is only Euclidean at p == 2. The ancestor short-circuited
        # this check whenever the metric was spelled 'euclidean' or 'l2' and then
        # ignored p entirely; here p is checked against the metric that will
        # actually be used — including a p given in metric_params, which
        # scikit-learn lets override the p parameter.
        if self.metric == "minkowski" and self._minkowski_p() != 2:
            raise UnsupportedParameterError(
                f"{name} implements metric='minkowski' only at p=2 "
                f"(Euclidean) on the MLX path; got p={self._minkowski_p()!r}.",
                parameter="p",
                value=self._minkowski_p(),
            )

        # metric_params was accepted and silently ignored by the ancestor, so a
        # weighted-Minkowski request returned unweighted Euclidean answers with no
        # warning. Wrong answers are worse than a fallback. The one entry with an
        # exact meaning here is Minkowski's p, resolved and checked just above;
        # a weight vector 'w', or parameters of any other metric, is refused.
        params = getattr(self, "metric_params", None) or {}
        unserved = set(params) - ({"p"} if self.metric == "minkowski" else set())
        if unserved:
            raise UnsupportedParameterError(
                f"{name} does not implement metric_params {sorted(unserved, key=str)!r} on "
                f"the MLX path; got {self.metric_params!r}. mlxlearn will not ignore "
                "them and return unweighted Euclidean distances.",
                parameter="metric_params",
                value=self.metric_params,
            )

    def _below_crossover(self, n_samples: int, n_features: int) -> tuple[bool, str]:
        """Whether this training set is too small for the MLX search, in either layer.

        Below it, patched mode hands the fit to scikit-learn and a directly
        imported estimator searches on MLX's CPU device. Both floors must be
        cleared, rows *and* rows x features, because a fit cannot see how its index
        will be queried, and each floor guards a different query: below 4 096 rows
        a 4096-query batch lost to stock brute force at every width measured
        (0.44x-0.79x at 2 048 x 256-1 024), and below 6 million elements batches of
        1-32 queries lost at 256-1 024 features (0.70x-0.95x at 4 096-8 192 rows).
        Patched mode asks more of anything narrower than ``knn_wide_features``
        (:meth:`_patched_handoff`). The calibration is ``docs/benchmarks.md``'s
        ``knn-rows`` and ``knn`` tables.
        """
        tuning = get_tuning()
        work = n_samples * n_features
        if n_samples < tuning.knn_min_train_samples or work < tuning.knn_min_work:
            return True, (
                f"{n_samples} training samples x {n_features} features is below the "
                f"measured KNN crossover ({tuning.knn_min_train_samples} samples and "
                f"{tuning.knn_min_work} elements, both required)"
            )
        return False, ""

    def _patched_handoff(self, n_samples: int, n_features: int) -> tuple[str, str] | None:
        """Patched mode's handoff by width and rows: ``(reason, detail)``, or ``None``.

        Three rules, each about query batches a fit cannot see, and all measured
        against stock scikit-learn with every MLX call made after the GPU has
        idled — as the benchmark's are, and as a call made after other work is.

        **Narrow data** — fewer than ``knn_min_features`` (16) features — goes to
        scikit-learn at any size. Mostly that is its KD-tree: with
        ``algorithm="auto"`` scikit-learn builds one whenever there are at most 15
        features, ``n_neighbors < n_samples // 2`` and the metric is one the tree
        supports, and the tree answered a single row faster than MLX at 2-8
        features at every size measured (0.13x-0.43x for MLX), and at 12 features up
        to 65 536 rows; at 2-5 features it won every batch. Those fits are recorded
        as ``"tree-wins"``. The rest go to scikit-learn's brute force, recorded as
        ``"below-crossover"``: at 2 features it answered some batch faster than MLX
        at every size measured, up to 2 097 152 rows, and at 8 features up to
        131 072 rows (0.62x-0.91x on one version or both). The floor stood at 64
        features until the first pass selected its candidates with the fused kernel
        (:mod:`mlxlearn._kernels._topk`): at 16-48 features and a million rows
        batches of 16 and 1 024 queries had lost (0.62x-0.84x); with it, every
        batch won or tied there on both versions, at k = 1-50 and on every law of
        data the fit-time probe passes (1.26x at worst).

        **Everything narrower than** ``knn_wide_features`` (256) needs at least
        ``knn_narrow_min_samples`` rows (1 000 000). An idled GPU runs a call of a
        few milliseconds' work up to two or three times slower than a busy one, so on
        a mid-size batch MLX loses to a stock search that costs about as much: at
        16-128 features a batch of 1-128 queries lost on one version or both at 27
        of the 50 sizes measured below 1 000 000 rows (down to 0.55x), all of them
        at 180 000 rows or fewer. From 262 144 rows the dense grid's batches won,
        but below 2**22 (query row, training row) pairs a query runs MLX's own
        selection, and the largest such batch lost at 262 144 x 16 (15 queries,
        0.90x) and only tied at 400 000 rows; and below a million rows nothing but
        Gaussian data at k = 10 was measured. From 1 000 000 rows at 16-128
        features every batch of 1-4096 queries won or tied on both versions, at
        every k up to 50 and on every law the fit-time probe passes. At 256 features the stock search costs enough
        that every batch won or tied from 33 000 rows, the first size the work
        floor lets through. These fits are recorded as ``"below-crossover"``.

        **Everything from** ``knn_wide_features`` **features** needs at least
        ``knn_wide_min_samples`` rows (8 192). Right after a stock query, whose
        OpenMP threads are still spinning, an MLX call of some batch sizes — about
        256 KiB of float32 queries, 32 rows at 2 048 features and 16 at 4 096 — ran
        up to twice as slow as its neighbours, and on a short training set a stock
        search costs about as much: at 4 096-7 500 rows and 2 048-4 096 features
        batches of 16-32 queries lost (0.77x-0.90x) or only tied, while from 8 192
        rows every batch of 8-64 queries won or tied (1.20x at worst; table
        ``knn-batches``). The size floors alone send these fits to MLX from about
        1 500 features. Recorded as ``"below-crossover"``.

        What the rules give up is real: batches of 1024 queries and more never
        lost below 256 features at these sizes (2.0x-15x), and on Gaussian data,
        a KD-tree's worst case, the MLX scan answers 256 queries many times faster
        than the tree from 8 features. Imported directly, the estimator searches on
        MLX regardless, which is what a workload known to query in large batches
        should do. See ``docs/benchmarks.md``, tables ``knn``, ``knn-boundary``,
        ``knn-narrow`` and ``knn-tree``.

        A fit wide or tall enough for MLX is then held to the measured range of
        ``n_neighbors`` (:meth:`_neighbors_handoff`).

        Only patched mode acts on these (``_select_backend``'s ``handoff``), because
        only patched mode has scikit-learn to hand the work to: what they measure is
        stock scikit-learn winning, and the Layer 1 CPU path is brute force as well.
        """
        tuning = get_tuning()
        min_features = tuning.knn_min_features
        if n_features >= min_features and n_features >= tuning.knn_wide_features:
            if n_samples >= tuning.knn_wide_min_samples:
                return self._neighbors_handoff(n_samples)
            return "below-crossover", (
                f"{n_samples} training samples x {n_features} features is below the "
                "measured KNN crossover for patched dispatch: it needs "
                f"{tuning.knn_wide_min_samples} samples, because below that an MLX call "
                "made right after a stock scikit-learn one can run some mid-size query "
                "batches up to twice as slow, and scikit-learn's brute force answers "
                "them in about as long"
            )
        if n_features >= min_features:
            if n_samples >= tuning.knn_narrow_min_samples:
                return self._neighbors_handoff(n_samples)
            return "below-crossover", (
                f"{n_samples} training samples x {n_features} features is below the "
                "measured KNN crossover for patched dispatch: with fewer than "
                f"{tuning.knn_wide_features} features it needs "
                f"{tuning.knn_narrow_min_samples} samples, because below that "
                "scikit-learn's brute force answers mid-size query batches faster "
                "than an MLX call made after the GPU has idled"
            )
        if (
            self.algorithm == "auto"
            and n_features <= _SKLEARN_TREE_MAX_FEATURES
            and (self.n_neighbors is None or self.n_neighbors < n_samples // 2)
            and self._effective_metric()[0] in VALID_METRICS["kd_tree"]
        ):
            return "tree-wins", (
                f"{n_samples} training samples x {n_features} features with "
                "algorithm='auto': scikit-learn answers this with a KD-tree, which "
                "answers single rows and small batches on narrow data faster than "
                "an MLX call"
            )
        return "below-crossover", (
            f"{n_features} features is below the measured KNN crossover for patched "
            f"dispatch ({min_features} features): scikit-learn's brute force answers "
            "some query batches on narrow data faster than an MLX call, at any size"
        )

    def _neighbors_handoff(self, n_samples: int) -> tuple[str, str] | None:
        """Patched mode's handoff by ``n_neighbors``: ``(reason, detail)``, or ``None``.

        The size rules were measured at k = 10, and what MLX pays per query grows with
        k where a stock brute-force search pays nearly nothing more: the first pass
        keeps ``k`` plus a margin of candidates per row through every tile's
        selection and refines them against the full rows, a gather of ``k`` rows of
        every feature per query. So a fit whose ``n_neighbors`` exceeds
        ``knn_max_neighbors`` goes to scikit-learn, recorded as
        ``"below-crossover"`` (``docs/benchmarks.md``, table ``knn-k``). At k = 40-50
        with fewer than 150 rows per neighbor MLX only tied; the row floors leave
        more than 160 at every k this allows.

        It reads the constructor's ``n_neighbors``, the one a fit can see. A query
        that asks ``kneighbors(n_neighbors=...)`` for a k outside the same range is
        handed to scikit-learn's brute force at query time instead
        (:meth:`_query_neighbors_handoff`).
        """
        k = self.n_neighbors
        if k is None:
            return None
        detail = _neighbors_limit(int(k))
        return None if detail is None else ("below-crossover", detail)

    def _query_neighbors_handoff(self, k: int) -> str | None:
        """Patched mode's handoff of one *query* by its ``k``: the detail, or ``None``.

        The fit decided by the constructor's ``n_neighbors``; a query may ask for a
        different k (``kneighbors(n_neighbors=...)``, or ``set_params`` after the
        fit), and one above ``knn_max_neighbors`` costs MLX what it would have cost a
        fit with that k (tables ``knn-k`` and ``knn-query-k``: down to 0.73x at
        8 192 x 2 048 with k = 200). Such a query is answered by scikit-learn's brute
        force over the float64 training rows instead (:meth:`_kneighbors_brute`),
        which kept up with a stock model's on float64 data at every shape measured,
        and on float32 data at a million rows lost by a few percent (0.93x-0.95x),
        the copy being float64 where stock searches float32; see
        ``docs/benchmarks.md``.

        Patched mode only, and never on a backend forced by
        ``MLXLEARN_FORCE_BACKEND`` — at the query, or at the fit that built the
        index. Recorded as ``"below-crossover"`` on the call's event.
        """
        if not self._mlxlearn_allow_fallback or get_tuning().force_backend:
            return None
        fit_event = self.__dict__.get("_fit_event_")
        if fit_event is not None and fit_event.reason == "forced":
            return None
        return _neighbors_limit(int(k))

    def _patched_data_handoff(self, index, device) -> tuple[str, str, dict] | None:
        """Patched mode's handoff by the *data*: ``(reason, detail, extra)``, or ``None``.

        The size rules were calibrated on Gaussian data, where the first float32
        pass certifies every query row. Other data does not always let it, and a
        row it cannot certify is searched again — wider, then exactly — while the
        whole batch waits: the cost is paid per *batch*, so a few such rows in a
        hundred, which put one in most batches of 16-64 queries, lose to stock brute
        force, whose float64 search needs no certificate. On rows repeated about
        twice, one column on 100 times the scale of the rest, or 100 clusters of std
        0.4-0.5, some batch lost at 32 of 40 shapes measured on and near the size
        floors, down to 0.36x, with as few as 3 of 512 probed rows uncertified; on
        rows repeated about 20 times, 100 clusters of std 0.1 or one column in raw
        units, at every shape, down to 0.16x (``docs/benchmarks.md``, tables
        ``knn-structured`` and ``knn-data``). A fit cannot see its queries, but it
        can see its own rows, so patched mode searches ``knn_probe_rows`` (512) of
        them with the first pass alone
        (:meth:`NeighborIndex.probe <mlxlearn._kernels.distance.NeighborIndex.probe>`,
        one pass for 512 rows) and hands the fit to scikit-learn when more than
        ``knn_probe_max_uncertified`` of them — at 0, any of them — could not be
        certified. Where one row in a hundred is, 256 rows saw none in 9 of 128 draws
        of the data, and 8 of those lost a mid-size batch; 512 saw some in every one
        (table ``knn-probe``). Gaussian and uniform data left none of 256 uncertified
        at k = 10 up to 3 072 features (table ``knn-k``), and stayed on MLX; of 512, 2
        of 8 draws of Gaussian data at 8 192 x 1 024 left one each, and go to
        scikit-learn. The first pass's noise band grows with the width and with k, and
        at 4 096 features, or 3 072 at k = 5, or 256 at k = 200, even Gaussian data
        left a few (table ``knn-k``), and goes to scikit-learn. The probe searches with
        the fitted ``n_neighbors`` for that reason.

        Integer-valued data — binary, one-hot, counts — is searched on an exact
        integer path instead, with no certificate to fail; it kept up with stock
        brute force from ``knn_wide_features`` features, and at a million rows and
        16 or 32 features lost batches of 16 queries and, of counts, of 256 and
        1 024 (0.53x-0.85x). Narrower than ``knn_wide_features`` it goes to
        scikit-learn too — conservatively at 33-255 features, where it was not
        measured on MLX.

        Recorded as ``"data-probe"``. Only patched mode acts on this, as on the
        width rules, and never on a backend forced by ``MLXLEARN_FORCE_BACKEND``.
        The calibration is ``docs/benchmarks.md``'s ``knn-data``, ``knn-structured``
        and ``knn-probe`` tables.
        """
        if not self._mlxlearn_allow_fallback or get_tuning().force_backend:
            return None
        found = index.probe(self.n_neighbors, rows=get_tuning().knn_probe_rows, device=device)
        handed = self._probe_handoff(index.n_samples, index.n_features, found)
        if handed is None:
            return None
        extra = {
            "probe_rows": found["rows"],
            "probe_uncertified": found["uncertified"],
            "integer_path": found["integer"],
        }
        return (*handed, extra)

    @staticmethod
    def _probe_handoff(n_samples: int, n_features: int, found: dict) -> tuple[str, str] | None:
        """The data rule on a probe's result (:meth:`_patched_data_handoff`)."""
        tuning = get_tuning()
        if found["integer"]:
            if n_features >= tuning.knn_wide_features:
                return None
            return "data-probe", (
                f"{n_samples} training samples x {n_features} integer-valued features: "
                "mlxlearn searches integer data on an exact path, which keeps up with "
                f"scikit-learn's brute force only from {tuning.knn_wide_features} "
                "features"
            )
        if found["uncertified"] <= tuning.knn_probe_max_uncertified * found["rows"]:
            return None
        limit = tuning.knn_probe_max_uncertified
        allowed = "none allowed" if limit == 0 else f"more than {limit:.0%}"
        return "data-probe", (
            f"the first float32 pass could not certify the neighbors of "
            f"{found['uncertified']} of {found['rows']} training rows probed at fit "
            f"({allowed}): on this data — "
            "duplicated rows, clusters, or a column on a far larger scale than the "
            "rest — most query batches would hold a row searched again, and "
            "scikit-learn's float64 brute force answers them faster"
        )

    def _minkowski_p(self):
        """The Minkowski exponent scikit-learn uses: ``metric_params['p']`` wins over ``p``."""
        params = getattr(self, "metric_params", None) or {}
        return params.get("p", self.p)

    def _effective_metric(self) -> tuple[str, dict]:
        """``(effective_metric_, effective_metric_params_)`` as scikit-learn sets them.

        Only meaningful for what the capability check lets through: ``'minkowski'``
        at ``p=2``, however p was given, resolves to ``'euclidean'`` with no
        parameters left over; ``'euclidean'``, ``'l2'`` and ``'sqeuclidean'`` keep
        their spelling, as they do in scikit-learn.
        """
        if self.metric == "minkowski":
            return "euclidean", {}
        return self.metric, {}

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
            from somewhere further in. The validated target is handed to
            :meth:`_fit_targets`.

        Returns
        -------
        str
            The chosen backend, or ``"sklearn"`` (Layer 2 only), in which case
            the caller must run scikit-learn's own ``fit``.
        """
        # scikit-learn validates constructor parameters in the _fit_context
        # decorator on its own fit, which overriding fit bypasses. Without this,
        # n_neighbors=0 "fitted" and predicted classes_[0] for every row, and an
        # invalid algorithm was reported as a capability mismatch instead of the
        # InvalidParameterError scikit-learn raises. LogisticRegression and SVC
        # do the same.
        self._validate_params()
        self._clear_backend()

        X = _as_array(X)
        shape = _safe_shape(X)
        backend = self._select_backend(
            "fit",
            shape=shape,
            crossover=self._below_crossover,
            handoff=self._patched_handoff,
            capability_check=lambda: self._check_neighbors_capability(X),
            # Before validate_data, so a float64 value that overflows float32 is a
            # capability limit — which scikit-learn's float64 path can serve, and
            # patched mode can fall back on — rather than scikit-learn's generic
            # post-conversion "contains infinity" ValueError.
            data_check=lambda: check_float32_range(X),
        )
        if backend == "sklearn":
            return "sklearn"

        if "p" in (getattr(self, "metric_params", None) or {}) and self.p is not None:
            # scikit-learn's own fit says this, so the MLX path does too.
            warnings.warn(
                "Parameter p is found in metric_params. The corresponding parameter "
                "from __init__ is ignored.",
                SyntaxWarning,
                stacklevel=3,
            )
        _reject_string_data(X)
        if target == "regression":
            # Stock scikit-learn stores a string target as given and fails in
            # predict; the float64 copy _fit_targets makes would parse "1.5"
            # instead, so under patch_sklearn() whether string targets worked
            # depended on the size of the data. Refused here, in scikit-learn's
            # words for string data.
            y = _reject_string_data(y)
        if target is None:
            X_checked = validate_data(
                self, X, accept_sparse=False, dtype=_INPUT_DTYPES,
                ensure_min_samples=1, reset=True,
            )
            y_checked = None
        else:
            # multi_output=True, as scikit-learn validates it: a column-vector y
            # stays (n, 1), so a regressor predicts (n, 1) exactly as the stock one
            # does, instead of the shape depending on which backend fitted it.
            X_checked, y_checked = validate_data(
                self,
                X,
                y,
                accept_sparse=False,
                dtype=_INPUT_DTYPES,
                ensure_min_samples=1,
                multi_output=True,
                y_numeric=target == "regression",
                reset=True,
            )
        # No second pass over X here. The data check has already range-checked the
        # raw input (_select_backend runs it before choosing MLX), validation
        # keeps float64 as float64 and float32 as float32 — rejecting non-finite
        # values in either — so nothing is cast to float32 before build_index, which
        # translates and splits float64 itself and refuses a span float32 cannot hold.

        if shape is None:
            # The input had no shape to read before validation (an array-like
            # without a shape attribute, say; lists were converted above), so the
            # size rules — and the device — are decided now.
            backend = self._select_backend(
                "fit",
                shape=X_checked.shape,
                crossover=self._below_crossover,
                handoff=self._patched_handoff,
            )
            if backend == "sklearn":
                return "sklearn"

        # Resolved here, where the configuration may be read; the index reads none.
        device = inference_device(backend)
        try:
            with mlx_guard(type(self).__name__, "fit"):
                index = build_index(X_checked, backend=backend, device=device)
                handed = self._patched_data_handoff(index, device)
        except CapabilityError as exc:
            # The index finds one limit itself: a span from the training mean that
            # float32 cannot subtract (past FLOAT32_MAX / 8, below FLOAT32_MAX, so
            # the range check passed it). _select_backend has already returned, so
            # it is routed through the same policy by hand, as SVC does: Layer 1
            # raises, Layer 2 falls back — or raises under fallback_policy="raise".
            return self._handle_capability_error(exc, "fit", shape=X_checked.shape)
        if handed is not None:
            reason, detail, extra = handed
            self._stage_event(
                "fit", "sklearn", reason=reason, detail=detail, shape=X_checked.shape,
                fallback=True, **extra,
            )
            return "sklearn"

        # After the index, so that a fit handed to scikit-learn above encodes (and
        # warns about) its target once, in scikit-learn's own fit.
        if y_checked is not None:
            self._fit_targets(y_checked)
        self._index_ = index
        self.n_samples_fit_ = int(X_checked.shape[0])
        self.effective_metric_, self.effective_metric_params_ = self._effective_metric()
        # scikit-learn's RadiusNeighborsMixin is inherited, so it needs the state
        # its own brute-force path reads. Populating it means radius_neighbors
        # stays correct on an MLX-fitted estimator, and it is also what lets
        # patched mode answer a sparse query with scikit-learn's brute force.
        self._fit_method = "brute"
        self._fit_X = index.X
        self._tree = None
        self._set_backend(backend)
        return backend

    def _fit_targets(self, y: np.ndarray) -> None:
        """Store the validated target. The supervised estimators implement this."""
        raise NotImplementedError

    # -------------------------------------------------------------- kneighbors

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Find the k nearest neighbors of each point.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), default=None
            Query points. ``None`` means "the training data", and each point's own
            index is excluded from its neighbors.
        n_neighbors : int, default=None
            Defaults to the estimator's ``n_neighbors``. Patched dispatch decides the
            fit by the constructor's ``n_neighbors``, the one a fit can see; under
            ``patch_sklearn()`` a query on an MLX-fitted model that asks for more
            than 50 neighbors is answered by scikit-learn's brute force over a
            float64 copy of the training rows, translated by their mean, built on
            the first such query and kept with the model: that query pays for the
            copy, ``8 · n_samples · n_features`` bytes (25-85 ms at a million rows
            and 64 features).
        return_distance : bool, default=True
            Whether to return distances alongside indices.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Euclidean distances, or squared Euclidean distances for
            ``metric="sqeuclidean"``, float64, ascending. Only when
            ``return_distance=True``.
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Training indices, int64, ordered by ``(distance, index)``.

        Notes
        -----
        On an MLX-fitted estimator a sparse ``X`` raises
        :class:`~mlxlearn.exceptions.SparseInputError`. Under ``patch_sklearn()``
        it is answered by scikit-learn's brute-force search instead, which reads
        the fitted state directly — the one inference call that is not served by
        the fitted backend, and it is recorded as a fallback. That search runs
        against the float32 training copy, so its distances carry the float32
        rounding of the training data (``docs/fp32_policy.md`` §3.4).
        """
        check_is_fitted(self)
        if self._fitted_backend("kneighbors") == "sklearn":
            self._note_inference("kneighbors")
            return super().kneighbors(X, n_neighbors, return_distance)
        return self._kneighbors("kneighbors", X, n_neighbors, return_distance)

    def _kneighbors(self, operation: str, X, n_neighbors=None, return_distance=True):
        """``kneighbors`` on an MLX-fitted estimator, recorded once as ``operation``.

        ``predict`` and ``predict_proba`` call this directly rather than through
        ``kneighbors``, so a public call records exactly one event, named after
        the call the user made.
        """
        n_neighbors = self._resolve_n_neighbors(n_neighbors)
        query_is_train = X is None

        if not query_is_train:
            try:
                require_dense(X, estimator=type(self).__name__, operation=operation)
            except SparseInputError as exc:
                # Layer 1 raises here. Layer 2 answers with scikit-learn's brute
                # force, which reads the state _fit_neighbors populated for it —
                # scikit-learn serves this query, so raising would break code that
                # works unpatched. fallback_policy still applies. The fallback is
                # this call's one event, named after the method the caller invoked.
                self._handle_capability_error(
                    exc, self._claim_call_event(operation) or operation, shape=_safe_shape(X)
                )
                X = _sparse_query_for_sklearn(X, self._fit_X.dtype)
                return super().kneighbors(X, n_neighbors, return_distance)

        handed = self._query_neighbors_handoff(n_neighbors)
        if handed is not None:
            return self._kneighbors_brute(operation, X, n_neighbors, return_distance, handed)

        if query_is_train:
            Q = self._index_.X
        else:
            X = _reject_string_data(X)
            # A float64 query stays float64 and is range-checked once validated (one
            # min/max pass), so a value float32 cannot hold is an
            # UnsupportedInputError, not an infinity inside the kernel. A float32
            # query is finite once validated and is not scanned again.
            Q = validate_float32_input(self, X, operation=operation, dtype=_INPUT_DTYPES)
        k = self._check_n_neighbors_bound(n_neighbors, query_is_train, n_queries=Q.shape[0])

        # Resolved here, where the configuration may be read; the index reads none.
        device = inference_device(self._execution_backend_)
        stats: dict = {}
        with mlx_guard(type(self).__name__, operation):
            dist, idx = self._index_.query(
                Q, k, return_distance=return_distance, exclude_self=query_is_train, stats=stats,
                device=device,
            )
        if self._fit_X is not self._index_.X:
            # The first query after unpickling or deepcopy rebuilt the index's device
            # copy and pointed its X at that; left on the loaded host array, _fit_X
            # kept a second full copy of the training set alive, and pickled it too.
            self._fit_X = self._index_.X
        # Recorded once the search has run, so the call's one event can also say how
        # it was served (_search_diagnostics).
        detail, extra = _search_diagnostics(stats)
        self._note_inference(operation, shape=Q.shape, detail=detail, **extra)
        if not return_distance:
            return idx
        if self.effective_metric_ == "sqeuclidean":
            # The index reports Euclidean distances in float64; squaring them back
            # costs a few float64 ulps, far inside the float32 error they carry.
            dist = np.square(dist)
        return dist, idx

    def _kneighbors_brute(self, operation: str, X, n_neighbors, return_distance: bool, detail):
        """``kneighbors`` by scikit-learn's brute force, for a query handed off by its k.

        :meth:`_query_neighbors_handoff` decides; this answers, with a stock
        ``NearestNeighbors(algorithm="brute")`` fitted on the index's float32 rows plus
        their float64 residuals — the caller's float64 input to about 2⁻⁴⁸, float32
        input exactly — translated by the index's mean, and queries translated by the
        same mean. Distances do not change under a translation; what does is how much
        of them float64 keeps, and scikit-learn's search, which expands each squared
        distance around the origin, loses data far from the origin to cancellation
        (Unix timestamps in float64 come back with distances of 0), where this keeps
        the ``kneighbors`` precision of ``docs/fp32_policy.md`` §3.4. Elsewhere it is
        stock scikit-learn's answer to rounding. The model and its rows,
        ``8 · n_samples · n_features`` bytes, are built on the first such query and
        kept, and neither is pickled: that first query costs 2-29 ms more at 8 192 rows
        and 1 024-2 048 features or 65 536 x 256, and 25-85 ms at 1 000 000 x 64 (table
        ``knn-query-k``).
        """
        center = self._index_.center.astype(np.float64)
        if X is not None:
            if (
                isinstance(X, np.ndarray)
                and X.ndim == 2
                and X.dtype.kind in "fiu"
                and X.shape[1] == center.shape[0]
                and "feature_names_in_" not in self.__dict__
            ):
                # The common case, checked by scikit-learn's own validation below —
                # finiteness included — rather than twice. The difference is float64,
                # the rows' dtype, which keeps scikit-learn on its fast path.
                X = X - center
            else:
                # Anything else is validated against this estimator first: its feature
                # names and count, which the stock model, fitted on a bare array,
                # cannot check.
                X = _reject_string_data(X)
                X = validate_data(
                    self, X, accept_sparse=False, dtype=np.float64, order="C", reset=False
                ) - center
        name = self._claim_call_event(operation)
        if name is not None:
            fit_event = self.__dict__.get("_fit_event_")
            record_inference(
                type(self).__name__,
                name,
                "sklearn",
                "below-crossover",
                detail,
                self._fit_X.shape if X is None else X.shape,
                {} if fit_event is None else {"fit_reason": fit_event.reason},
            )
        return self._host_brute().kneighbors(X, n_neighbors, return_distance)

    def _host_brute(self):
        """The stock brute-force model :meth:`_kneighbors_brute` answers with, built once."""
        brute = self.__dict__.get("_host_brute_")
        if brute is None:
            brute = _SkNearestNeighbors(
                algorithm="brute", metric=self.effective_metric_, n_jobs=self.n_jobs
            )
            # The rows are finite by construction; checking 8 bytes per element again
            # would cost the first query as much as the search.
            with _sklearn_config_context(assume_finite=True):
                brute.fit(self._index_.host_rows())
            self.__dict__["_host_brute_"] = brute
        return brute

    def __getstate__(self):
        # The stock model and its float64 rows are derived from the index, and rebuilt
        # on the first query that needs them; pickling them would double the file.
        state = dict(super().__getstate__())
        state.pop("_host_brute_", None)
        return state

    def _resolve_n_neighbors(self, n_neighbors):
        """Validate an explicit ``n_neighbors`` with scikit-learn's exact messages.

        Only the checks scikit-learn makes before validating the query; the upper
        bound needs the query's row count and comes after.
        """
        if n_neighbors is None:
            return self.n_neighbors
        if n_neighbors <= 0:
            # int(), as scikit-learn's "%d" truncates: -1.5 is reported as -1.
            raise ValueError(f"Expected n_neighbors > 0. Got {int(n_neighbors)}")
        if not isinstance(n_neighbors, Integral):
            raise TypeError(
                f"n_neighbors does not take {type(n_neighbors)} value, "
                "enter integer value"
            )
        return n_neighbors

    def _check_n_neighbors_bound(self, n_neighbors, query_is_train: bool, *, n_queries: int) -> int:
        """The ``n_neighbors <= n_samples_fit`` check, message and all."""
        n_samples_fit = self.n_samples_fit_
        limit = n_samples_fit - int(query_is_train)
        if n_neighbors > limit:
            inequality = "n_neighbors < n_samples_fit" if query_is_train else (
                "n_neighbors <= n_samples_fit"
            )
            raise ValueError(
                f"Expected {inequality}, but n_neighbors = {n_neighbors}, "
                f"n_samples_fit = {n_samples_fit}, n_samples = {n_queries}"
            )
        return int(n_neighbors)

    # ------------------------------------------------------------------ helpers

    def _query_indices(self, X, operation: str, *, need_distance: bool):
        """One neighbor-acquisition path for ``predict`` and ``predict_proba``.

        The ancestor had three — a prefix-cached ordered query for small k, an
        unordered query for large k, and a distance-returning query for
        ``weights='distance'`` — with nothing asserting they agreed on the
        candidate set. One path cannot disagree with itself.
        """
        dist, idx = self._kneighbors(operation, X, return_distance=True)
        return (dist if need_distance else None), idx


def _neighbors_limit(k: int) -> str | None:
    """Why ``k`` neighbors is outside what MLX answers faster, or ``None``.

    At most ``knn_max_neighbors`` neighbors: the rule behind both the fit's handoff
    and a query's (table ``knn-k``).
    """
    tuning = get_tuning()
    if k > tuning.knn_max_neighbors:
        return (
            f"n_neighbors={k} is above the measured KNN crossover for patched "
            f"dispatch (at most {tuning.knn_max_neighbors} neighbors): MLX's "
            "search pays per query for every neighbor it keeps, and scikit-learn's "
            "brute force barely does"
        )
    return None


def _search_diagnostics(stats: dict) -> tuple[str, dict]:
    """``(detail, extra)`` for a query event from ``NeighborIndex.query``'s ``stats``.

    Empty for the ordinary case, where the first float32 pass certified every row.
    Otherwise the event says how many rows the first pass could not certify, and how
    those were answered: some went straight to the exact search, when a count on a
    sample of the training rows estimated that the wider candidate search could not
    certify them either; the rest were searched wider, and some of those needed the
    exact search after all. The parts add up to the whole. The answer is the same
    either way, but the time is not, and a slow ``kneighbors`` on near-duplicate or
    tied data is explained by exactly this.
    """
    widened = int(stats.get("widened_rows", 0))
    skipped = int(stats.get("widen_skipped_rows", 0))
    exact = int(stats.get("exact_rows", 0))
    if not (widened or exact):
        return "", {}
    rows = int(stats.get("rows", 0))
    detail = f"{widened} of {rows} query rows could not be certified by the first float32 pass"
    if skipped:
        # Skipped rows are searched exactly too, so exact - skipped is what the wider
        # search could not certify.
        searched, after = widened - skipped, exact - skipped
        detail += (
            f"; {skipped} of them went straight to the exact search, where a count on a "
            "sample of the training rows estimated that more candidates could not certify "
            "them either"
        )
        if searched:
            detail += f"; {searched} were searched again with more candidates, " + (
                f"and {after} of those needed the exact search too" if after
                else "which certified them"
            )
    else:
        detail += " and were searched again with more candidates" + (
            f"; {exact} of them needed the exact search" if exact else ""
        )
    detail += ". The neighbors returned are exact either way."
    return detail, {
        "search": "certificate-fallback",
        "widened_rows": widened,
        "widen_skipped_rows": skipped,
        "exact_rows": exact,
    }


def _as_array(X):
    """A list or tuple converted to an ndarray, once; anything else as given.

    ``validate_data`` makes the same conversion. Making it first is what lets the
    string check, the range check and the size rules read the array rather than
    each converting the list again. A ragged list is left for ``validate_data``
    to report in scikit-learn's words.
    """
    if isinstance(X, (list, tuple)):
        try:
            return np.asarray(X)
        except ValueError:
            return X
    return X


def _reject_string_data(X):
    """Refuse string data the way scikit-learn's neighbor estimators do.

    They validate with ``dtype="numeric"``, which rejects arrays of strings and
    bytes. mlxlearn validates with an explicit float dtype so the conversion
    happens once, and an explicit dtype makes ``check_array`` parse ``"1.5"`` into
    1.5 instead — so the MLX path accepted input stock scikit-learn rejects, and
    under ``patch_sklearn()`` whether it did depended on the size of the data.
    Object arrays are left alone: scikit-learn converts those as well.

    Returns ``X``, a list or tuple converted to an ndarray (:func:`_as_array`),
    so the validation after this check does not convert it a second time.
    """
    X = _as_array(X)
    dtype = getattr(X, "dtype", None)
    if isinstance(dtype, np.dtype) and dtype.kind in "USV":
        raise ValueError(
            "dtype='numeric' is not compatible with arrays of bytes/strings."
            "Convert your data to numeric values explicitly instead."
        )
    return X


def _sparse_query_for_sklearn(X, dtype):
    """A sparse query in the training copy's dtype, for scikit-learn's brute force.

    scikit-learn's search requires the query and the training copy to share a
    dtype. A float64 query against the float32 copy fell to a chunked
    pairwise-distance path several times slower than stock, which cannot compute
    ``"sqeuclidean"`` on sparse input at all and raised ``TypeError``. The cast is
    one pass over the stored values, range-checked first, so a value float32
    cannot hold is an ``UnsupportedInputError`` as it is for a dense query.

    A pandas frame of sparse columns is handed over as it is: pandas casts it by
    densifying it. So is complex data, which scikit-learn rejects in its words.
    """
    if not hasattr(X, "tocsr"):
        return X
    X = X.tocsr()  # the format scikit-learn converts the query to anyway
    if X.dtype != dtype and X.dtype.kind in "biuf":
        check_float32_range(X.data, fallback_advice=False)
        X = X.astype(dtype)
    return X


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

    A callable gets the distances ``kneighbors`` returns and gives back the
    weights, as in scikit-learn. Weighting is host code on distances the search
    has already returned, so there is nothing about a callable for the MLX path
    to refuse. A row whose weights are all zero is the classifier's to reject;
    scikit-learn's regressor returns NaN for it.
    """
    if weights in (None, "uniform"):
        return None
    if dist is None:
        raise ValueError(f"weights={weights!r} requires neighbor distances.")
    if callable(weights):
        return np.asarray(weights(dist), dtype=np.float64)
    if weights != "distance":
        raise ValueError(
            "weights not recognized: should be 'uniform', 'distance', or a callable "
            f"function; got {weights!r}"
        )

    with np.errstate(divide="ignore"):
        w = 1.0 / np.asarray(dist, dtype=np.float64)

    coincident = np.isinf(w)
    rows_with_coincident = np.any(coincident, axis=1)
    if np.any(rows_with_coincident):
        w[rows_with_coincident] = coincident[rows_with_coincident].astype(np.float64)
    return w


def _vote_totals(codes: np.ndarray, weights: np.ndarray | None, n_classes: int) -> np.ndarray:
    """Per-row total weight of each class code, shape ``(n_queries, n_classes)``, float64.

    One ``np.bincount`` over row-offset codes rather than ``np.add.at``, which is
    3-4x slower. Both accumulate each (row, class) cell in neighbor order, so the
    sums are bitwise identical.
    """
    n_queries = codes.shape[0]
    flat = (np.arange(n_queries, dtype=np.intp)[:, None] * n_classes + codes).ravel()
    totals = np.bincount(
        flat,
        weights=None if weights is None else weights.ravel(),
        minlength=n_queries * n_classes,
    )
    return totals.reshape(n_queries, n_classes).astype(np.float64, copy=False)


def weighted_mode(codes: np.ndarray, weights: np.ndarray | None, n_classes: int) -> np.ndarray:
    """Per-row weighted majority vote over integer class codes.

    Ties go to the smallest class code, matching scikit-learn. Vectorized: the
    ancestor imported ``sklearn.utils.extmath.weighted_mode`` and
    ``sklearn.utils.fixes._mode``, both private.
    """
    # argmax returns the first maximum, i.e. the smallest class code on a tie.
    return _vote_totals(codes, weights, n_classes).argmax(axis=1).astype(INDEX_DTYPE)


def class_probabilities(codes: np.ndarray, weights: np.ndarray | None, n_classes: int) -> np.ndarray:
    """Per-row normalized class scores, shape ``(n_queries, n_classes)``, float64."""
    proba = _vote_totals(codes, weights, n_classes)
    total = proba.sum(axis=1, keepdims=True)
    np.divide(proba, np.where(total > 0, total, 1.0), out=proba)
    return proba
