"""Merge gates 2 and 5 from docs/development.md, for every registered estimator.

Gate 2 is "works inside scikit-learn's machinery": ``joblib``, cross-validation
with process-parallel workers, ``Pipeline`` inside ``GridSearchCV``. Gate 5 is the
sticky backend: inference reuses whatever ``fit`` chose and never decides again.
Both were asserted only for ``KNeighborsClassifier`` while all five estimators were
registered, and the registry's own rule is that nothing is registered before every
gate passes. These run each gate over both layers, so a new estimator gets them by
appearing in ``LAYER1``.

sklearn's check suite (``tests/test_sklearn_compliance.py``) covers clone,
``get_params``/``set_params``, refit and ``n_features_in_``; ``test_invariants.py``
covers pickle and fit purity.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.base import is_classifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlxlearn import get_backend_diagnostics
from mlxlearn._common.config import tuning_context
from mlxlearn._common.diagnostics import clear_backend_diagnostics
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from mlxlearn.patching import _estimators as patched
from mlxlearn.svm import SVC

#: (class, constructor kwargs, a small grid for GridSearchCV)
_CASES = [
    (KNeighborsClassifier, {"n_neighbors": 5}, {"n_neighbors": [3, 5]}),
    (KNeighborsRegressor, {"n_neighbors": 5}, {"n_neighbors": [3, 5]}),
    (LogisticRegression, {}, {"C": [0.1, 1.0]}),
    (SVC, {}, {"C": [0.5, 1.0]}),
    (NearestNeighbors, {"n_neighbors": 5}, None),
]
LAYER1 = [pytest.param(*case, id=case[0].__name__) for case in _CASES]
LAYER2 = [
    pytest.param(getattr(patched, cls.__name__), kwargs, grid, id=f"patched.{cls.__name__}")
    for cls, kwargs, grid in _CASES
]
BOTH = LAYER1 + LAYER2
SUPERVISED = [case for case in BOTH if case.values[2] is not None]

#: Every inference method any of the five exposes; each test calls those present.
INFERENCE = ("predict", "predict_proba", "decision_function", "kneighbors")


@pytest.fixture(scope="module")
def data():
    """Below every default crossover except KNN's, and quick to fit on any backend."""
    return make_classification(
        n_samples=600, n_features=8, n_informative=5, n_redundant=1, random_state=0
    )


def _fit(cls, kwargs, X, y, backend):
    with tuning_context(force_backend=backend):
        return cls(**kwargs).fit(X, y)


def _outputs(model, X):
    """Every inference result the model offers, keyed by method."""
    out = {}
    for method in INFERENCE:
        if hasattr(model, method):
            result = getattr(model, method)(X)
            out[method] = result[1] if method == "kneighbors" else result
    return out


# ----------------------------------------------------------------------------------
# gate 2: joblib, cross-validation, Pipeline + GridSearchCV
# ----------------------------------------------------------------------------------


# joblib's memory-map loader sets ndarray.shape, which NumPy 2.5 deprecates.
@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
@pytest.mark.parametrize("backend", ["mlx", "cpu"])
@pytest.mark.parametrize("cls, kwargs, grid", BOTH)
def test_joblib_round_trip_preserves_backend_and_predictions(
    cls, kwargs, grid, backend, data, tmp_path
):
    """``joblib.dump``/``load``, including the read-only memory map joblib hands
    to process-parallel workers — the state must not depend on writable buffers."""
    X, y = data
    model = _fit(cls, kwargs, X, y, backend)
    expected = _outputs(model, X[:100])

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    for mmap_mode in (None, "r"):
        restored = joblib.load(path, mmap_mode=mmap_mode)
        assert type(restored) is cls
        assert restored.execution_backend_ == model.execution_backend_ == backend
        actual = _outputs(restored, X[:100])
        assert actual.keys() == expected.keys()
        for method, value in expected.items():
            np.testing.assert_array_equal(actual[method], value, err_msg=method)


# loky starts its workers with fork+exec, which Python 3.12+ flags in a threaded process.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.parametrize("cls, kwargs, grid", SUPERVISED)
def test_process_parallel_cross_validation_matches_sequential(cls, kwargs, grid, data):
    """``n_jobs=2`` pickles the estimator into fresh worker processes, which import
    mlxlearn and MLX from scratch. Their scores must equal the in-process ones.

    Configuration is thread-local and does not reach the workers, so this runs at
    the default configuration on both sides: KNN on MLX, LogisticRegression and SVC
    on Layer 1's CPU path or Layer 2's scikit-learn dispatch.
    """
    X, y = data
    parallel = cross_validate(cls(**kwargs), X, y, cv=3, n_jobs=2, return_estimator=True)
    sequential = cross_validate(cls(**kwargs), X, y, cv=3, n_jobs=1, return_estimator=True)

    np.testing.assert_array_equal(parallel["test_score"], sequential["test_score"])
    assert [m.execution_backend_ for m in parallel["estimator"]] == [
        m.execution_backend_ for m in sequential["estimator"]
    ]


@pytest.mark.gpu
@pytest.mark.parametrize("cls, kwargs, grid", SUPERVISED)
def test_grid_search_over_a_pipeline_runs_on_mlx(cls, kwargs, grid, data):
    """Parameters routed through ``Pipeline`` into a cloned estimator, fitted on the
    MLX path, and the refit best estimator is an MLX-fitted mlxlearn object."""
    X, y = data
    step = cls.__name__.lower()
    search = GridSearchCV(
        make_pipeline(StandardScaler(), cls(**kwargs)),
        {f"{step}__{name}": values for name, values in grid.items()},
        cv=2,
    )
    with tuning_context(force_backend="mlx"):
        search.fit(X, y)

    best = search.best_estimator_[-1]
    assert type(best) is cls
    assert best.execution_backend_ == "mlx"
    for name, values in grid.items():
        assert getattr(best, name) == search.best_params_[f"{step}__{name}"]
        assert getattr(best, name) in values
    assert search.predict(X[:10]).shape == (10,)
    if is_classifier(best):
        assert set(search.classes_) == set(np.unique(y))


@pytest.mark.parametrize(
    "cls", [NearestNeighbors, patched.NearestNeighbors], ids=["NearestNeighbors", "patched"]
)
def test_unsupervised_estimator_in_a_pipeline(cls, data):
    X, _ = data
    with tuning_context(force_backend="mlx"):
        pipeline = make_pipeline(StandardScaler(), cls(n_neighbors=5)).fit(X)
    model = pipeline[-1]
    assert model.execution_backend_ == "mlx"
    _, idx = model.kneighbors(pipeline[:-1].transform(X[:10]))
    np.testing.assert_array_equal(idx[:, 0], np.arange(10))


# ----------------------------------------------------------------------------------
# gate 5: the sticky backend
# ----------------------------------------------------------------------------------


def _inference_events(model, X):
    """Call each inference method the model has; return the events each recorded."""
    recorded = {}
    for method in INFERENCE:
        if hasattr(model, method):
            clear_backend_diagnostics()
            getattr(model, method)(X)
            recorded[method] = get_backend_diagnostics()
    return recorded


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["mlx", "cpu"])
@pytest.mark.parametrize("cls, kwargs, grid", BOTH)
def test_inference_reuses_the_fitted_backend(cls, kwargs, grid, backend, data):
    """Fit on a forced backend, then run every inference method *outside* the
    override. A re-decision would pick something else for this size (the CPU path
    or scikit-learn below the crossover, the GPU above it), so every call must
    record events that carry the fitted backend and the reason ``"sticky"``."""
    X, y = data
    model = _fit(cls, kwargs, X, y, backend)

    recorded = _inference_events(model, X[:50])
    assert recorded
    for method, events in recorded.items():
        assert events, f"{method} recorded no backend event"
        assert all(e.reason == "sticky" for e in events), (method, events)
        assert all(e.backend == backend and not e.fallback for e in events), (method, events)


@pytest.mark.parametrize("cls, kwargs, grid", LAYER2)
def test_inference_after_a_fallback_fit_stays_on_sklearn(cls, kwargs, grid, data):
    """The Layer 2 half: a fit dispatched to scikit-learn keeps inference there,
    recorded as a sticky fallback, even with MLX forced for the inference call."""
    X, y = data
    model = cls(**kwargs).fit(X[:120], y[:120])  # below every crossover
    assert model.execution_backend_ == "sklearn"

    with tuning_context(force_backend="mlx"):
        recorded = _inference_events(model, X[:50])
    assert recorded
    for method, events in recorded.items():
        assert events, f"{method} recorded no backend event"
        assert all(e.reason == "sticky" for e in events), (method, events)
        assert all(e.backend == "sklearn" and e.fallback for e in events), (method, events)
