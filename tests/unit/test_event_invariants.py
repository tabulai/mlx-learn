"""One fit, one event; one public call, one event — every estimator, every path.

``tests/unit/test_runtime.py`` checks the mechanism on the common paths. This file
walks all of them, because each double record found so far lived on a path the
common ones do not reach: a patched fit that passed the size gates and then failed
SVC's per-pair gate, a KD-tree handoff, a sparse query answered by scikit-learn
from MLX-fitted state, a GPU-fitted model on a host without one, scikit-learn's own
``score`` and ``kneighbors_graph`` reaching the work through overridden methods.

The two invariants:

* A completed ``fit`` records exactly one ``"fit"`` event, its backend is
  ``execution_backend_``, and it is the model's ``fit_event_``.
* Every public inference call records exactly one event, named after the method
  the caller invoked, with the fitted (sticky) backend.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning

import mlxlearn
from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn.exceptions import MLXLearnError, MLXLearnFallbackWarning, SparseInputError
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from mlxlearn.patching import _estimators as patched
from mlxlearn.svm import SVC

needs_gpu = pytest.mark.skipif(not gpu_available(), reason="needs a working MLX GPU")

ESTIMATORS = {
    "KNeighborsClassifier": (KNeighborsClassifier, patched.KNeighborsClassifier, {"n_neighbors": 3}),
    "KNeighborsRegressor": (KNeighborsRegressor, patched.KNeighborsRegressor, {"n_neighbors": 3}),
    "NearestNeighbors": (NearestNeighbors, patched.NearestNeighbors, {"n_neighbors": 3}),
    "LogisticRegression": (LogisticRegression, patched.LogisticRegression, {}),
    "SVC": (SVC, patched.SVC, {}),
}
NAMES = sorted(ESTIMATORS)
KNN = ["KNeighborsClassifier", "KNeighborsRegressor", "NearestNeighbors"]

#: Every size gate open, so a small problem is selected for MLX the normal way.
NO_CROSSOVER = {
    "knn_min_train_samples": 0,
    "knn_min_work": 0,
    "knn_min_features": 0,
    "knn_narrow_min_samples": 0,
    "knn_probe_max_uncertified": 1.0,
    "knn_wide_features": 0,
    "knn_max_neighbors": 2**31,
    "knn_wide_min_samples": 0,
    "logreg_min_samples": 0,
    "logreg_multinomial_min_features": 0,
    "logreg_min_max_iter": 0,
    "logreg_binary_min_samples": 0,
    "logreg_binary_max_features": 2**31,
    "svc_min_features": 0,
    "svc_min_samples": 0,
    "svc_min_work": 0,
    "svc_max_c": float("inf"),
    "svc_min_pair_work": 0,
}

#: fit path -> (tuning, input transform, layers it exists on, expected backend per layer)
#: A backend of None means "whatever MLX device selection gives" (mlx, or cpu
#: without a GPU).
FIT_PATHS = {
    "selected": (NO_CROSSOVER, None, ("L1", "L2"), {"L1": None, "L2": None}),
    "below-crossover": ({}, None, ("L1", "L2"), {"L1": "cpu", "L2": "sklearn"}),
    "forced-cpu": ({"force_backend": "cpu"}, None, ("L1", "L2"), {"L1": "cpu", "L2": "cpu"}),
    "forced-mlx": ({"force_backend": "mlx"}, None, ("L1", "L2"), {"L1": "mlx", "L2": "mlx"}),
    "forced-sklearn": ({"force_backend": "sklearn"}, None, ("L2",), {"L2": "sklearn"}),
    "capability": (NO_CROSSOVER, sp.csr_matrix, ("L2",), {"L2": "sklearn"}),
}


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=240, n_features=6, n_informative=4, n_redundant=0, n_classes=3,
        n_clusters_per_class=1, random_state=0,
    )
    return X, y


def _make(name, layer, **overrides):
    layer1, layer2, kwargs = ESTIMATORS[name]
    return (layer1 if layer == "L1" else layer2)(**{**kwargs, **overrides})


def _target(name, y):
    return y.astype(np.float64) if name == "KNeighborsRegressor" else y


def _public_calls(name, X, y):
    """``{method: call(model)}`` for every public inference method ``name`` has."""
    Q, yq = X[:17], _target(name, y)[:17]
    calls = {
        "KNeighborsClassifier": {
            "predict": lambda m: m.predict(Q),
            "predict_proba": lambda m: m.predict_proba(Q),
            "kneighbors": lambda m: m.kneighbors(Q),
            "kneighbors_graph": lambda m: m.kneighbors_graph(Q),
            "score": lambda m: m.score(Q, yq),
        },
        "KNeighborsRegressor": {
            "predict": lambda m: m.predict(Q),
            "kneighbors": lambda m: m.kneighbors(Q),
            "kneighbors_graph": lambda m: m.kneighbors_graph(Q),
            "score": lambda m: m.score(Q, yq),
        },
        "NearestNeighbors": {
            "kneighbors": lambda m: m.kneighbors(Q),
            "kneighbors_graph": lambda m: m.kneighbors_graph(Q),
            "radius_neighbors": lambda m: m.radius_neighbors(Q, radius=1.5),
            "radius_neighbors_graph": lambda m: m.radius_neighbors_graph(Q, radius=1.5),
        },
        "LogisticRegression": {
            "predict": lambda m: m.predict(Q),
            "predict_proba": lambda m: m.predict_proba(Q),
            "predict_log_proba": lambda m: m.predict_log_proba(Q),
            "decision_function": lambda m: m.decision_function(Q),
            "score": lambda m: m.score(Q, yq),
        },
        "SVC": {
            "predict": lambda m: m.predict(Q),
            "decision_function": lambda m: m.decision_function(Q),
            "score": lambda m: m.score(Q, yq),
        },
    }
    return calls[name]


def _events(call):
    mlxlearn.clear_backend_diagnostics()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MLXLearnFallbackWarning)
        call()
    return mlxlearn.get_backend_diagnostics()


def _fit(model, X, y, tuning):
    with tuning_context(**tuning), warnings.catch_warnings():
        warnings.simplefilter("ignore", MLXLearnFallbackWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlxlearn.clear_backend_diagnostics()
        model.fit(X, y)
    return mlxlearn.get_backend_diagnostics()


def _assert_one_fit_event(model, events, expected_backend):
    fits = [e for e in events if e.operation == "fit"]
    assert len(fits) == 1, fits
    assert len(events) == 1, f"a fit recorded more than its fit event: {events}"
    (event,) = fits
    assert event.backend == model.execution_backend_
    assert model.fit_event_ == event
    assert event.fallback == (event.backend == "sklearn")
    # "selected" is what _set_backend records for a decision nobody staged.
    assert not (event.backend == "sklearn" and event.reason == "selected"), event
    if expected_backend is None:
        expected_backend = "mlx" if gpu_available() else "cpu"
    assert model.execution_backend_ == expected_backend


def _assert_one_event_per_call(model, name, X, y, *, reason=None):
    backend = model.execution_backend_
    for method, call in _public_calls(name, X, y).items():
        events = _events(lambda call=call: call(model))
        assert [(e.operation, e.backend) for e in events] == [(method, backend)], (
            f"{name}.{method} on a {backend}-fitted model recorded {events}"
        )
        if reason is not None:
            assert events[0].reason == reason, events[0]


# ======================================================================================
# fit paths shared by every estimator
# ======================================================================================

_CASES = [
    pytest.param(
        name, path, layer,
        id=f"{name}-{path}-{layer}",
        marks=[needs_gpu] if path == "forced-mlx" else [],
    )
    for name in NAMES
    for path, (_, _, layers, _) in FIT_PATHS.items()
    for layer in layers
]


@pytest.mark.parametrize("name, path, layer", _CASES)
def test_one_event_per_fit_and_per_public_call(data, name, path, layer):
    X, y = data
    tuning, transform, _, expected = FIT_PATHS[path]
    model = _make(name, layer)
    X_fit = X if transform is None else transform(X)
    events = _fit(model, X_fit, _target(name, y), tuning)
    _assert_one_fit_event(model, events, expected[layer])
    _assert_one_event_per_call(model, name, X, y)


@pytest.mark.parametrize("name", NAMES)
def test_forcing_sklearn_on_layer1_raises_and_records_no_fit(data, name):
    X, y = data
    with pytest.raises(MLXLearnError, match="Layer 1 estimators never dispatch"):
        _fit(_make(name, "L1"), X, _target(name, y), {"force_backend": "sklearn"})
    assert [e for e in mlxlearn.get_backend_diagnostics() if e.operation == "fit"] == []


@pytest.mark.parametrize("name", NAMES)
def test_a_layer1_capability_refusal_records_one_raised_event(data, name):
    X, y = data
    model = _make(name, "L1")
    with pytest.raises(SparseInputError):
        _fit(model, sp.csr_matrix(X), _target(name, y), NO_CROSSOVER)
    events = mlxlearn.get_backend_diagnostics()
    assert len(events) == 1 and events[0].extra.get("raised") is True, events
    assert not hasattr(model, "_execution_backend_")


# ======================================================================================
# paths particular to one estimator family
# ======================================================================================


@pytest.mark.parametrize("name", KNN)
def test_knn_tree_wins_handoff(data, name):
    """Patched low-dimensional algorithm="auto" fits go to scikit-learn's KD-tree."""
    X, y = data
    model = _make(name, "L2")
    tuning = {**NO_CROSSOVER, "knn_min_features": 16}  # 6 features, a KD-tree config
    events = _fit(model, X, _target(name, y), tuning)
    _assert_one_fit_event(model, events, "sklearn")
    assert model.fit_event_.reason == "tree-wins"
    _assert_one_event_per_call(model, name, X, y)


@pytest.mark.parametrize("layer", ["L1", "L2"])
def test_svc_per_pair_gate(data, layer):
    """The size gates pass on the whole problem, and the per-pair gate then fails.

    SVC used to record ('mlx', 'selected') and then ('sklearn', 'below-crossover')
    here: the total-size fast path could not see the class count.
    """
    X, y = data
    model = _make("SVC", layer)
    tuning = {
        "svc_min_features": 0, "svc_min_samples": 0, "svc_min_work": 0, "svc_min_pair_work": 10**9
    }
    events = _fit(model, X, y, tuning)
    _assert_one_fit_event(model, events, "sklearn" if layer == "L2" else "cpu")
    assert model.fit_event_.reason == "below-crossover"
    assert "one-vs-one" in model.fit_event_.detail
    _assert_one_event_per_call(model, "SVC", X, y)


@pytest.mark.parametrize("break_ties", [False, True])
def test_svc_predict_through_decision_function_records_once(data, break_ties):
    X, y = data
    with tuning_context(force_backend="cpu"):
        model = SVC(break_ties=break_ties).fit(X, y)
    events = _events(lambda: model.predict(X[:9]))
    assert [(e.operation, e.backend) for e in events] == [("predict", "cpu")]


@pytest.mark.parametrize("name", KNN)
def test_a_sparse_query_on_an_mlx_fitted_patched_knn_is_one_fallback_event(data, name):
    """The one inference carve-out: scikit-learn's brute force serves it, recorded once."""
    X, y = data
    with tuning_context(force_backend="cpu"):
        model = _make(name, "L2").fit(X, _target(name, y))
    for method, call in _public_calls(name, sp.csr_matrix(X), y).items():
        if method.startswith("radius"):
            continue  # scikit-learn's code on every backend; not a fallback
        events = _events(lambda call=call: call(model))
        assert [(e.operation, e.backend, e.reason) for e in events] == [
            (method, "sklearn", "sparse-input")
        ], events
        assert "patch_sklearn()" not in events[0].detail


def test_a_call_made_inside_another_estimators_call_records_its_own_event(data):
    """A weights callable that queries a second neighbor estimator.

    The neighbor estimators silenced nested calls with a flag of their own, set
    per thread rather than per estimator: while a scikit-learn-fitted model's
    predict ran, another estimator's kneighbors recorded nothing.
    """
    X, y = data
    with tuning_context(force_backend="cpu"):
        other = NearestNeighbors(n_neighbors=2).fit(X)

    def weights(dist):
        other.kneighbors(X[:3])
        return np.ones_like(dist)

    model = patched.KNeighborsClassifier(n_neighbors=3, weights=weights).fit(X, y)
    assert model.execution_backend_ == "sklearn"
    events = _events(lambda: model.predict(X[:5]))
    assert [(e.estimator, e.operation) for e in events] == [
        ("KNeighborsClassifier", "predict"),
        ("NearestNeighbors", "kneighbors"),
    ]


@pytest.mark.parametrize("name", KNN)
def test_a_sparse_query_on_layer1_knn_raises_with_one_event(data, name):
    X, y = data
    with tuning_context(force_backend="cpu"):
        model = _make(name, "L1").fit(X, _target(name, y))
    mlxlearn.clear_backend_diagnostics()
    with pytest.raises(SparseInputError):
        model.kneighbors(sp.csr_matrix(X[:5]))
    events = mlxlearn.get_backend_diagnostics()
    assert [(e.operation, e.extra.get("raised")) for e in events] == [("kneighbors", True)]


# ======================================================================================
# a GPU-fitted model on a host without a GPU
# ======================================================================================


@needs_gpu
@pytest.mark.parametrize("name", NAMES)
def test_gpu_fitted_inference_without_a_gpu_is_one_event(monkeypatch, data, name):
    """One 'gpu-unavailable' event per call, however many methods the call went through.

    SVC recorded it outside the public-call scope, so predict with break_ties — which
    reaches decision_function — recorded it twice, as backend 'cpu'.
    """
    X, y = data
    with tuning_context(force_backend="mlx"):
        model = _make(name, "L1", **({"break_ties": True} if name == "SVC" else {}))
        model.fit(X, _target(name, y))
    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    # LogisticRegression predicts in host NumPy float64 on every backend: no GPU is
    # wanted, so none is missed. NearestNeighbors.radius_neighbors* are
    # scikit-learn's code on the fitted state, likewise host-side. So is SVC's
    # inference on batches this small (svc_host_inference_work); with that threshold
    # at 0 it takes the device path, and the GPU is missed there.
    for method, call in _public_calls(name, X, y).items():
        for host_work in (None, 0) if name == "SVC" else (None,):
            overrides = {} if host_work is None else {"svc_host_inference_work": host_work}
            with tuning_context(**overrides):
                events = _events(lambda call=call: call(model))
                assert [(e.operation, e.backend) for e in events] == [(method, "mlx")], events
                host_side = (
                    name == "LogisticRegression"
                    or method.startswith("radius")
                    or (name == "SVC" and host_work is None)
                )
                expected = "sticky" if host_side else "gpu-unavailable"
                assert events[0].reason == expected, events[0]

                # An explicit device="gpu" is refused where MLX would have to run, and
                # has nothing to refuse where the call is host code on the fitted state.
                with mlxlearn.config_context(device="gpu"):
                    if host_side:
                        call(model)
                    else:
                        with pytest.raises(mlxlearn.DeviceError):
                            call(model)


# ======================================================================================
# how a neighbor query was served
# ======================================================================================


def _tight_clusters(seed=0):
    """Clusters whose spread is 1e-3 of their distance from the mean: the first
    float32 pass cannot certify their members' order."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(100, 16))
    X = np.repeat(centers, 30, axis=0) + rng.normal(scale=1e-3, size=(3000, 16))
    Q = centers[rng.integers(0, 100, 40)] + rng.normal(scale=1e-3, size=(40, 16))
    return X.astype(np.float32), Q.astype(np.float32)


def test_a_certificate_fallback_is_on_the_calls_one_event():
    X, Q = _tight_clusters()
    with tuning_context(force_backend="cpu"):
        model = NearestNeighbors(n_neighbors=5).fit(X)
        classifier = KNeighborsClassifier(n_neighbors=5).fit(X, np.arange(3000) % 3)

    events = _events(lambda: model.kneighbors(Q))
    assert len(events) == 1, events
    (event,) = events
    assert (event.operation, event.reason) == ("kneighbors", "sticky")
    assert event.extra["search"] == "certificate-fallback"
    assert 0 < event.extra["widened_rows"] <= 40
    assert 0 <= event.extra["exact_rows"] <= event.extra["widened_rows"]
    assert f"{event.extra['widened_rows']} of 40 query rows" in event.detail

    # Through predict, still one event, named predict.
    events = _events(lambda: classifier.predict(Q))
    assert [(e.operation, e.extra.get("search")) for e in events] == [
        ("predict", "certificate-fallback")
    ]


def test_an_ordinary_query_carries_no_search_detail(data):
    X, y = data
    with tuning_context(force_backend="cpu"):
        model = NearestNeighbors(n_neighbors=3).fit(X)
    (event,) = _events(lambda: model.kneighbors(X[:20]))
    assert "search" not in event.extra and event.detail == ""
