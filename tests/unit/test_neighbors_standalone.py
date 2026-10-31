"""Public estimator contracts without importing or inheriting stock neighbors."""

from __future__ import annotations

import inspect
import io
import os
import pickle
import subprocess
import sys
import textwrap
import zipfile
from numbers import Integral
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags

import mlxlearn
from mlxlearn._common.config import config_context
from mlxlearn.exceptions import MLXLearnError
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors

ESTIMATORS = [NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor]
ROOT = Path(__file__).resolve().parents[2]


class NamedSearch(NearestNeighbors):
    _mlxlearn_allow_fallback = True


class NamedClassifier(KNeighborsClassifier):
    _mlxlearn_allow_fallback = True


class NamedRegressor(KNeighborsRegressor):
    _mlxlearn_allow_fallback = True


def _arrays():
    X = np.random.default_rng(20732).normal(size=(32, 3)).astype(np.float32)
    queries = X[[1, 7, 19]].copy() + np.float32(0.03125)
    classes = np.where(X[:, 0] > 0, "positive", "negative")
    targets = X[:, 0].astype(np.float64) - 0.125 * X[:, 1].astype(np.float64)
    return X, queries, classes, targets


def _target(estimator, classes, targets):
    if estimator.__name__ == "KNeighborsClassifier":
        return classes
    if estimator.__name__ == "KNeighborsRegressor":
        return targets
    return None


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_owned_import_fit_query_and_pickle_do_not_import_sklearn_neighbors(estimator):
    program = textwrap.dedent("""
        import pickle
        import sys
        import numpy as np
        assert not any(name == 'sklearn.neighbors' or name.startswith('sklearn.neighbors.')
                       for name in sys.modules)
        from mlxlearn import neighbors
        cls = getattr(neighbors, sys.argv[1])
        assert not any(name == 'sklearn.neighbors' or name.startswith('sklearn.neighbors.')
                       for name in sys.modules)
        X = np.arange(48, dtype=np.float32).reshape(16, 3) / 8
        y = np.arange(len(X)) % 3
        model = cls(n_neighbors=2, algorithm='brute')
        model.fit(X) if cls.__name__ == 'NearestNeighbors' else model.fit(X, y)
        model.kneighbors(X[[2, 9]])
        if cls.__name__ != 'NearestNeighbors':
            model.predict(X[[2, 9]])
        if cls.__name__ == 'KNeighborsClassifier':
            model.predict_proba(X[[2, 9]])
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.kneighbors(X[[2, 9]])[1],
                                      model.kneighbors(X[[2, 9]])[1])
        assert model.execution_backend_ == 'cpu' and model.last_query_backend_ == 'cpu'
        assert not any(name == 'sklearn.neighbors' or name.startswith('sklearn.neighbors.')
                       for name in sys.modules)
    """)
    result = subprocess.run([sys.executable, "-c", program, estimator.__name__],
                            cwd=ROOT, env=os.environ.copy(), capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_standalone_constructor_signature_params_clone_and_unfitted_pickle(estimator):
    import sklearn.neighbors

    stock = getattr(sklearn.neighbors, estimator.__name__)
    assert inspect.signature(estimator) == inspect.signature(stock)
    model = estimator(n_neighbors=3, algorithm="brute", leaf_size=17, n_jobs=1)
    expected = stock(n_neighbors=3, algorithm="brute", leaf_size=17, n_jobs=1)
    assert not isinstance(model, stock)  # The explicit startup/identity tradeoff.
    assert model.get_params(deep=True) == expected.get_params(deep=True)
    for restored in (clone(model), pickle.loads(pickle.dumps(model))):
        assert type(restored) is estimator
        assert restored.get_params() == expected.get_params()
        assert not hasattr(restored, "execution_backend_")
        with pytest.raises(NotFittedError):
            restored.kneighbors(np.ones((1, 3), np.float32))
    assert model.set_params(n_neighbors=2) is model
    assert model.get_params()["n_neighbors"] == 2


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_standalone_estimator_tags_and_kind(estimator):
    model = estimator()
    kind = ("classifier" if estimator is KNeighborsClassifier else
            "regressor" if estimator is KNeighborsRegressor else None)
    tags = get_tags(model)
    assert tags.estimator_type == kind
    assert tags.target_tags.required == (kind is not None)
    assert tags.input_tags.two_d_array
    assert is_classifier(model) == (kind == "classifier")
    assert is_regressor(model) == (kind == "regressor")


@pytest.mark.parametrize("estimator", [KNeighborsClassifier, KNeighborsRegressor])
def test_standalone_weighted_score_and_metadata_routing(estimator):
    from sklearn import config_context as sklearn_config
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.pipeline import Pipeline

    X, queries, classes, targets = _arrays()
    y = _target(estimator, classes, targets)
    model = estimator(n_neighbors=3, weights="distance").fit(X, y)
    actual = model.predict(queries)
    expected_y = y[[1, 7, 19]]
    sample_weight = np.array([1.0, 2.0, 4.0])
    score = accuracy_score if estimator is KNeighborsClassifier else r2_score
    expected = score(expected_y, actual, sample_weight=sample_weight)
    assert model.score(queries, expected_y, sample_weight=sample_weight) == pytest.approx(expected)
    with sklearn_config(enable_metadata_routing=True):
        routed = estimator(n_neighbors=3, weights="distance").set_score_request(sample_weight=True)
        assert routed.get_metadata_routing().consumes(method="score", params=["sample_weight"]) == {"sample_weight"}
        pipeline = Pipeline([("neighbors", routed)]).fit(X, y)
        assert pipeline.score(queries, expected_y, sample_weight=sample_weight) == pytest.approx(expected)
        copied = clone(routed)
        assert copied.get_metadata_routing().consumes(method="score", params=["sample_weight"]) == {"sample_weight"}


@pytest.mark.parametrize("estimator", [KNeighborsClassifier, KNeighborsRegressor])
def test_standalone_pipeline_grid_search_and_parameter_nesting(estimator):
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, queries, classes, targets = _arrays()
    y = _target(estimator, classes, targets)
    pipeline = Pipeline([("scale", StandardScaler()), ("neighbors", estimator())])
    search = GridSearchCV(pipeline, {"neighbors__n_neighbors": [1, 3]}, cv=3,
                          n_jobs=1, error_score="raise").fit(X, y)
    assert type(search.best_estimator_["neighbors"]) is estimator
    assert search.best_params_["neighbors__n_neighbors"] in (1, 3)
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    assert search.predict(queries).shape == (len(queries),)
    assert search.best_estimator_["neighbors"].last_query_backend_ == "cpu"


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("mode", ["connectivity", "distance"])
@pytest.mark.parametrize("self_query", [False, True])
def test_owned_neighbor_graph_retains_canonical_edges_and_self_exclusion(estimator, mode, self_query):
    X, queries, classes, targets = _arrays()
    X[1] = X[0]  # Off-diagonal zero-distance edges must remain explicitly present.
    model = estimator(n_neighbors=3).fit(X, _target(estimator, classes, targets))
    query = None if self_query else queries
    distances, indices = model.kneighbors(query)
    graph = model.kneighbors_graph(query, mode=mode)
    assert graph.format == "csr" and graph.shape == (len(indices), len(X))
    np.testing.assert_array_equal(np.diff(graph.indptr), 3)
    np.testing.assert_array_equal(graph.indices.reshape(-1, 3), indices)
    np.testing.assert_allclose(graph.data.reshape(-1, 3), distances if mode == "distance" else 1)
    if self_query:
        assert not np.any(indices == np.arange(len(X))[:, None])
    with pytest.raises(ValueError):
        model.kneighbors_graph(query, mode="invalid")


@pytest.mark.parametrize("self_query", [False, True])
def test_lazy_radius_delegate_preserves_owned_training_and_radius_graph(self_query):
    from sklearn.neighbors import NearestNeighbors as StockNeighbors

    X = np.array([[0.0], [0.0], [1.0], [3.0], [7.0], [15.0]], np.float32)
    snapshot = X.copy()
    queries = np.array([[0.2], [2.4], [8.5]], np.float32)
    model = NearestNeighbors(n_neighbors=2, radius=2.5, algorithm="brute").fit(X)
    stock = StockNeighbors(n_neighbors=2, radius=2.5, algorithm="brute").fit(snapshot)
    X[:] = 1000
    query = None if self_query else queries
    actual = model.radius_neighbors(query, sort_results=True)
    expected = stock.radius_neighbors(query, sort_results=True)
    for actual_rows, expected_rows in zip(actual, expected, strict=True):
        for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
            np.testing.assert_allclose(actual_row, expected_row, rtol=1e-6, atol=1e-6)
    if self_query:
        assert all(i not in row for i, row in enumerate(actual[1]))
    for mode in ("connectivity", "distance"):
        graph = model.radius_neighbors_graph(query, mode=mode, sort_results=mode == "distance")
        reference = stock.radius_neighbors_graph(query, mode=mode, sort_results=mode == "distance")
        np.testing.assert_allclose(graph.toarray(), reference.toarray(), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.sort(model.kneighbors(queries)[1], axis=1),
                                  np.sort(stock.kneighbors(queries)[1], axis=1))


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_fallback_delegate_survives_patch_unpatch_pickle_and_refit(estimator):
    import joblib
    import sklearn.neighbors

    X, queries, classes, targets = _arrays()
    y = _target(estimator, classes, targets)
    if estimator is KNeighborsClassifier:
        # Stock sklearn 1.7/1.8's float32 Manhattan classifier rejects string
        # labels in its optimized prediction kernel. That parity is tested
        # separately; this case exercises successful serialization and refits.
        y = (classes == "positive").astype(np.int64)
    stock = getattr(sklearn.neighbors, estimator.__name__)
    expected = stock(n_neighbors=3, metric="manhattan", algorithm="brute").fit(X, y)
    mlxlearn.patch_sklearn(estimator.__name__)
    patched = getattr(sklearn.neighbors, estimator.__name__)
    with config_context(fallback_policy="silent"):
        model = patched(n_neighbors=3, metric="manhattan", algorithm="brute").fit(X, y)
    assert model.execution_backend_ == "sklearn"
    restored = pickle.loads(pickle.dumps(model))
    storage = io.BytesIO()
    joblib.dump(model, storage)
    storage.seek(0)
    joblib_restored = joblib.load(storage)
    # Both transports must work while the public sklearn name is still patched.
    for copied in (restored, joblib_restored):
        np.testing.assert_array_equal(copied.kneighbors(queries)[1], expected.kneighbors(queries)[1])
    mlxlearn.unpatch_sklearn(estimator.__name__)
    assert getattr(sklearn.neighbors, estimator.__name__) is stock
    for actual in (model, restored, joblib_restored):
        np.testing.assert_array_equal(actual.kneighbors(queries)[1], expected.kneighbors(queries)[1])
        if estimator is not NearestNeighbors:
            np.testing.assert_array_equal(actual.predict(queries), expected.predict(queries))
        if estimator is KNeighborsClassifier:
            np.testing.assert_allclose(actual.predict_proba(queries), expected.predict_proba(queries))
        actual.set_params(metric="minkowski", p=2).fit(X[::-1].copy(), None if y is None else y[::-1].copy())
        assert actual.execution_policy_ == "auto" and actual.execution_backend_ == "cpu"
        assert actual.kneighbors(queries)[1].shape == (len(queries), 3)
        assert actual.last_query_backend_ == "cpu"
        with pytest.raises(ValueError):
            actual.fit(np.full_like(X, np.nan), y)
        assert not hasattr(actual, "execution_backend_")
        with pytest.raises((NotFittedError, MLXLearnError)) as failed_fit:
            actual.kneighbors(queries)
        if isinstance(failed_fit.value, MLXLearnError):
            assert "no recorded execution backend" in str(failed_fit.value)
            assert "fit did not complete" in str(failed_fit.value)


@pytest.mark.parametrize("operation", ["predict", "predict_proba"])
def test_string_label_manhattan_fallback_matches_stock_values_or_specific_error(operation):
    import sklearn.neighbors

    X, queries, classes, _ = _arrays()
    stock = sklearn.neighbors.KNeighborsClassifier
    reference = stock(n_neighbors=3, metric="manhattan", algorithm="brute").fit(X, classes)
    expected_error = None
    try:
        expected = getattr(reference, operation)(queries)
    except ValueError as error:
        # Evaluate stock first: its older optimized float32 kernel may reject
        # string class labels. Fallback must preserve that error, not hide it.
        expected_error = type(error), str(error)

    mlxlearn.patch_sklearn("KNeighborsClassifier")
    with config_context(fallback_policy="silent"):
        model = sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=3, metric="manhattan", algorithm="brute",
        ).fit(X, classes)
    assert model.execution_backend_ == "sklearn"
    if expected_error is None:
        np.testing.assert_array_equal(getattr(model, operation)(queries), expected)
    else:
        with pytest.raises(expected_error[0]) as actual:
            getattr(model, operation)(queries)
        assert str(actual.value) == expected_error[1]


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("include_self", [False, True])
def test_patched_neighbors_keep_captured_stock_identity_and_public_graph_helper(estimator, include_self):
    import sklearn.neighbors

    X, _, classes, targets = _arrays()
    stock = getattr(sklearn.neighbors, estimator.__name__)
    graph_helper = sklearn.neighbors.kneighbors_graph
    mlxlearn.patch_sklearn(estimator.__name__)
    patched = getattr(sklearn.neighbors, estimator.__name__)
    model = patched(n_neighbors=3).fit(X, _target(estimator, classes, targets))
    assert isinstance(model, stock)
    assert isinstance(model, patched)
    actual = graph_helper(model, 3, mode="distance", include_self=include_self)
    expected = model.kneighbors_graph(X if include_self else None, mode="distance")
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_allclose(actual.data, expected.data)


@pytest.mark.parametrize("include_self", [False, True])
def test_patched_radius_search_keeps_public_radius_graph_helper(include_self):
    import sklearn.neighbors

    X, _, _, _ = _arrays()
    helper = sklearn.neighbors.radius_neighbors_graph
    mlxlearn.patch_sklearn("NearestNeighbors")
    model = sklearn.neighbors.NearestNeighbors(radius=1.25).fit(X)
    actual = helper(model, 1.25, mode="distance", include_self=include_self)
    expected = model.radius_neighbors_graph(X if include_self else None, mode="distance")
    np.testing.assert_allclose(actual.toarray(), expected.toarray())


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("parameter,value", [("n_neighbors", 0), ("n_neighbors", 1.5),
                                             ("algorithm", "invalid"), ("leaf_size", 0),
                                             ("metric", 37)])
def test_standalone_invalid_constructor_parameters_keep_stock_error_type(estimator, parameter, value):
    import sklearn.neighbors

    X, _, classes, targets = _arrays()
    y = _target(estimator, classes, targets)
    stock = getattr(sklearn.neighbors, estimator.__name__)
    with pytest.raises((TypeError, ValueError)) as expected:
        stock(**{parameter: value}).fit(X, y)
    with pytest.raises(type(expected.value)) as actual:
        estimator(**{parameter: value}).fit(X, y)
    assert parameter in str(actual.value)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("parameter,value", [("n_neighbors", None), ("n_neighbors", True),
                                             ("n_neighbors", np.int64(3)), ("leaf_size", None),
                                             ("leaf_size", True), ("p", None), ("p", 0),
                                             ("p", np.float32(2)), ("p", np.inf),
                                             ("metric_params", []), ("n_jobs", "all")])
def test_standalone_parameter_validation_boundaries_match_stock(estimator, parameter, value):
    import sklearn.neighbors

    stock = getattr(sklearn.neighbors, estimator.__name__)
    try:
        stock(**{parameter: value})._validate_params()
    except (TypeError, ValueError) as error:
        with pytest.raises(type(error)):
            estimator(**{parameter: value})._validate_params()
    else:
        estimator(**{parameter: value})._validate_params()


@pytest.mark.parametrize("value", [None, -1, 0, np.float32(0.25), np.inf, "all"])
def test_standalone_radius_parameter_validation_matches_stock(value):
    from sklearn.neighbors import NearestNeighbors as StockNeighbors

    try:
        StockNeighbors(radius=value)._validate_params()
    except (TypeError, ValueError) as error:
        with pytest.raises(type(error)):
            NearestNeighbors(radius=value)._validate_params()
    else:
        NearestNeighbors(radius=value)._validate_params()


def _neighbor_constraints(estimator, upper=2):
    from sklearn.utils._param_validation import Interval

    constraints = dict(estimator._parameter_constraints)
    constraints["n_neighbors"] = [Interval(Integral, 1, upper, closed="both")]
    return constraints


def _constraint_base(estimator, patched):
    if not patched:
        return estimator
    from mlxlearn.patching import _estimators

    return getattr(_estimators, estimator.__name__)


def _assert_custom_constraint(model, estimator, *, valid):
    X, _, classes, targets = _arrays()
    y = _target(estimator, classes, targets)
    if valid:
        BaseEstimator._validate_params(model)
        model._validate_params()
        assert model.fit(X, y) is model
    else:
        with pytest.raises(ValueError) as expected:
            BaseEstimator._validate_params(model)
        with pytest.raises(type(expected.value)) as actual:
            model._validate_params()
        assert str(actual.value) == str(expected.value)
        with pytest.raises(type(expected.value)) as fitted:
            model.fit(X, y)
        assert str(fitted.value) == str(expected.value)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("patched", [False, True])
@pytest.mark.parametrize("inherited", [False, True])
def test_subclass_and_child_custom_parameter_constraints_are_honored(estimator, patched, inherited):
    base = _constraint_base(estimator, patched)

    class TightNeighbors(base):
        _parameter_constraints = _neighbor_constraints(base)

    class InheritedTightNeighbors(TightNeighbors):
        pass

    cls = InheritedTightNeighbors if inherited else TightNeighbors
    model = cls(n_neighbors=3)
    _assert_custom_constraint(model, estimator, valid=False)
    model.set_params(n_neighbors=2)
    _assert_custom_constraint(model, estimator, valid=True)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("patched", [False, True])
def test_class_constraint_replacement_and_inplace_change_apply_after_construction(estimator, patched):
    base = _constraint_base(estimator, patched)

    class MutableNeighbors(base):
        pass

    model = MutableNeighbors(n_neighbors=3)
    _assert_custom_constraint(model, estimator, valid=True)
    MutableNeighbors._parameter_constraints = _neighbor_constraints(base)
    _assert_custom_constraint(model, estimator, valid=False)
    # A previous validation decision cannot be cached on the instance/class.
    MutableNeighbors._parameter_constraints["n_neighbors"] = _neighbor_constraints(base, 4)["n_neighbors"]
    _assert_custom_constraint(model, estimator, valid=True)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("patched", [False, True])
def test_instance_parameter_constraint_override_and_removal_are_honored(estimator, patched):
    base = _constraint_base(estimator, patched)
    model = base(n_neighbors=3)
    _assert_custom_constraint(model, estimator, valid=True)
    model._parameter_constraints = _neighbor_constraints(base)
    _assert_custom_constraint(model, estimator, valid=False)
    model.set_params(n_neighbors=2)
    _assert_custom_constraint(model, estimator, valid=True)
    del model._parameter_constraints
    model.set_params(n_neighbors=3)
    _assert_custom_constraint(model, estimator, valid=True)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_replacing_default_mixin_constraint_descriptor_is_observed(estimator):
    from mlxlearn.neighbors._base import MLXNeighborsMixin

    model = estimator(n_neighbors=3)
    _assert_custom_constraint(model, estimator, valid=True)
    constraints = _neighbor_constraints(estimator)
    original = vars(MLXNeighborsMixin)["_parameter_constraints"]
    try:
        MLXNeighborsMixin._parameter_constraints = constraints
        _assert_custom_constraint(model, estimator, valid=False)
        model.set_params(n_neighbors=2)
        _assert_custom_constraint(model, estimator, valid=True)
    finally:
        MLXNeighborsMixin._parameter_constraints = original
    model.set_params(n_neighbors=3)
    _assert_custom_constraint(model, estimator, valid=True)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("mode", [[], {}])
def test_unhashable_neighbor_graph_modes_keep_stock_validation_error(estimator, mode):
    import sklearn.neighbors

    X, queries, classes, targets = _arrays()
    y = _target(estimator, classes, targets)
    stock = getattr(sklearn.neighbors, estimator.__name__)(n_neighbors=3).fit(X, y)
    model = estimator(n_neighbors=3).fit(X, y)
    with pytest.raises(ValueError) as expected:
        stock.kneighbors_graph(queries, mode=mode)
    with pytest.raises(type(expected.value)) as actual:
        model.kneighbors_graph(queries, mode=mode)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("metric", ["precomputed", "nan_euclidean"])
def test_standalone_metric_dependent_tags_match_stock(estimator, metric):
    import sklearn.neighbors

    own_tags = get_tags(estimator(metric=metric))
    stock_tags = get_tags(getattr(sklearn.neighbors, estimator.__name__)(metric=metric))
    assert own_tags.input_tags.pairwise == stock_tags.input_tags.pairwise
    assert own_tags.input_tags.allow_nan == stock_tags.input_tags.allow_nan


@pytest.mark.parametrize("estimator,base", [(NamedSearch, NearestNeighbors),
                                          (NamedClassifier, KNeighborsClassifier),
                                          (NamedRegressor, KNeighborsRegressor)])
@pytest.mark.parametrize("fallback", [False, True])
def test_named_subclasses_keep_validation_tags_and_stock_delegate_identity(estimator, base, fallback):
    import sklearn.neighbors

    X, queries, classes, targets = _arrays()
    y = _target(base, classes, targets)
    metric = "manhattan" if fallback else "minkowski"
    with config_context(fallback_policy="silent"):
        model = estimator(n_neighbors=3, algorithm="brute", metric=metric).fit(X, y)
    reference = getattr(sklearn.neighbors, base.__name__)(
        n_neighbors=3, algorithm="brute", metric=metric,
    ).fit(X, y)
    np.testing.assert_array_equal(model.kneighbors(queries)[1], reference.kneighbors(queries)[1])
    assert get_tags(model).estimator_type == get_tags(base()).estimator_type
    assert type(clone(model)) is estimator
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.kneighbors(queries)[1], reference.kneighbors(queries)[1])
    with pytest.raises((TypeError, ValueError)):
        estimator(n_neighbors=0).fit(X, y)


@pytest.fixture(scope="module")
def legacy_neighbor_pickles(tmp_path_factory):
    """Generate actual pre-delegate state using the immutable previous source."""
    folder = tmp_path_factory.mktemp("legacy-neighbor-pickles")
    source = folder / "source"
    with zipfile.ZipFile(ROOT / "benchmarks/reference/knn-latency-final-2026-10-30.zip") as archive:
        archive.extractall(source)
    destination = folder / "models.pkl"
    program = textwrap.dedent("""
        import pickle
        import sys
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        import numpy as np
        import mlxlearn
        from mlxlearn import neighbors
        from mlxlearn._common.config import config_context
        from mlxlearn.patching import _estimators as patched
        assert Path(mlxlearn.__file__).is_relative_to(Path(sys.argv[1]))
        X = np.arange(48, dtype=np.float32).reshape(16, 3) / 8
        queries = X[[2, 9]] + np.float32(0.0625)
        result = {}
        for name in ('NearestNeighbors', 'KNeighborsClassifier', 'KNeighborsRegressor'):
            y = None if name == 'NearestNeighbors' else np.arange(len(X)) % 3
            for fallback in (False, True):
                cls = getattr(patched if fallback else neighbors, name)
                model = cls(n_neighbors=2, algorithm='brute',
                            metric='manhattan' if fallback else 'minkowski')
                with config_context(fallback_policy='silent'):
                    model.fit(X, y)
                data = {'model': model, 'X': X, 'queries': queries, 'y': y,
                        'neighbors': model.kneighbors(queries)}
                if name != 'NearestNeighbors':
                    data['prediction'] = model.predict(queries)
                if name == 'KNeighborsClassifier':
                    data['probabilities'] = model.predict_proba(queries)
                result[name, fallback] = data
        with open(sys.argv[2], 'wb') as stream:
            pickle.dump(result, stream)
    """)
    process = subprocess.run([sys.executable, "-c", program, str(source), str(destination)],
                             cwd=folder, capture_output=True, text=True, timeout=45)
    assert process.returncode == 0, process.stdout + process.stderr
    return destination.read_bytes()


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("fallback", [False, True])
def test_actual_legacy_owned_and_fallback_pickle_migration(legacy_neighbor_pickles, estimator, fallback):
    data = pickle.loads(legacy_neighbor_pickles)[estimator.__name__, fallback]
    model, queries = data["model"], data["queries"]
    actual = model.kneighbors(queries)
    np.testing.assert_array_equal(actual[1], data["neighbors"][1])
    np.testing.assert_allclose(actual[0], data["neighbors"][0], rtol=1e-13, atol=0)
    if estimator is not NearestNeighbors:
        np.testing.assert_array_equal(model.predict(queries), data["prediction"])
    if estimator is KNeighborsClassifier:
        np.testing.assert_allclose(model.predict_proba(queries), data["probabilities"])
    assert model.execution_backend_ == ("sklearn" if fallback else "cpu")
    model.set_params(metric="minkowski", p=2).fit(data["X"], data["y"])
    assert model.execution_policy_ == "auto" and model.execution_backend_ == "cpu"
    np.testing.assert_array_equal(model.kneighbors(data["X"][[3]])[1][0, 0], 3)
