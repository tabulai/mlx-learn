"""End-to-end ownership and invalidation of SVC's prepared inference model."""

from __future__ import annotations

import pickle
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.config import tuning_context
from mlxlearn._kernels.kernels import kernel_combine
from mlxlearn.patching._estimators import SVC as PatchedSVC
from mlxlearn.svm import SVC


@pytest.fixture
def data():
    rng = np.random.default_rng(847)
    y = np.tile(np.arange(3), 24)
    centers = np.array([[-2, 0, 1, 0, 0], [1, 2, -1, 0, 0], [1, -2, 0, 1, 0]])
    X = (centers[y] + rng.normal(scale=0.7, size=(len(y), 5))).astype(np.float32)
    query = np.vstack([X[:11], rng.normal(size=(8, 5))]).astype(np.float32)
    return X, y, query


def _fit(data, backend, *, estimator=SVC, **kwargs):
    if backend == "mlx" and not mx.metal.is_available():
        pytest.skip("requires Metal")
    X, y, _ = data
    with tuning_context(force_backend=backend):
        return estimator(gamma=0.2, decision_function_shape="ovo", **kwargs).fit(X, y)


def _direct(model, query):
    return kernel_combine(
        model._kernel_spec_, query, model.support_vectors_, model._pair_weights_,
        center=model._kernel_center_, backend=model.execution_backend_, bias=model._pair_rho_,
    )


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_fitted_preparation_is_reused_and_predictions_match_direct_expansion(data, backend):
    model = _fit(data, backend)
    query = data[2]
    prepared = model._prepared_kernel_
    assert prepared._right is not None  # eager preparation includes the first query
    expected = _direct(model, query)
    for _ in range(2):
        np.testing.assert_allclose(model.decision_function(query), expected,
                                   rtol=2e-12, atol=2e-12)
        prediction = model.predict(query)
        np.testing.assert_array_equal(prediction, model.predict(query.copy()))
        assert model._prepared_kernel_ is prepared

    # Subsequent configuration does not move a fitted model to another device.
    with tuning_context(force_backend="cpu" if backend == "mlx" else "mlx"):
        np.testing.assert_array_equal(model.predict(query), prediction)
    assert model.execution_backend_ == backend
    assert model._prepared_kernel_ is prepared


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize("mutation", ["supports", "center", "spec"])
def test_public_model_mutations_rebuild_before_prediction(data, backend, mutation):
    model = _fit(data, backend)
    query = data[2]
    old_prepared = model._prepared_kernel_
    if mutation == "supports":
        model.support_vectors_[0, :] += 1.5
    elif mutation == "center":
        model._kernel_center_[:] += 0.5
    else:
        model._kernel_spec_ = replace(model._kernel_spec_, gamma=0.3)
    expected = _direct(model, query)
    np.testing.assert_allclose(model.decision_function(query), expected, rtol=2e-12, atol=2e-12)
    assert model._prepared_kernel_ is not old_prepared
    assert model._prepared_kernel_.matches(
        model.support_vectors_, model._kernel_spec_, model._kernel_center_, backend,
    )


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_live_weights_and_bias_do_not_require_support_preparation(data, backend):
    model = _fit(data, backend)
    query = data[2]
    before = model.decision_function(query)
    prepared = model._prepared_kernel_
    model._pair_weights_ *= 0.5
    model._pair_rho_ += 0.25
    expected = _direct(model, query)
    assert not np.allclose(before, expected)
    np.testing.assert_allclose(model.decision_function(query), expected, rtol=2e-12, atol=2e-12)
    assert model._prepared_kernel_ is prepared


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_estimator_pickle_restores_inference_without_serializing_device_state(data, backend):
    model = _fit(data, backend)
    query = data[2]
    expected = model.decision_function(query)
    expected_prediction = model.predict(query)
    restored = pickle.loads(pickle.dumps(model))
    assert restored._prepared_kernel_._right is None
    assert restored._prepared_kernel_._center_device is None
    np.testing.assert_array_equal(restored.decision_function(query), expected)
    np.testing.assert_array_equal(restored.predict(query), expected_prediction)
    assert restored._prepared_kernel_._right is not None
    # Models serialized before prepared inference existed must also work.
    del restored._prepared_kernel_
    np.testing.assert_array_equal(restored.decision_function(query), expected)


@pytest.mark.parametrize("new_backend", ["cpu", "mlx"])
def test_refit_replaces_preparation_and_uses_new_backend(data, new_backend):
    if new_backend == "mlx" and not mx.metal.is_available():
        pytest.skip("requires Metal")
    model = _fit(data, "cpu")
    previous = model._prepared_kernel_
    X, y, query = data
    changed = X * 1.5 + 0.25
    with tuning_context(force_backend=new_backend):
        model.fit(changed, y)
        fresh = SVC(gamma=0.2, decision_function_shape="ovo").fit(changed, y)
    assert model._prepared_kernel_ is not previous
    assert model._prepared_kernel_.backend == new_backend
    np.testing.assert_array_equal(model.predict(query), fresh.predict(query))
    np.testing.assert_array_equal(model.decision_function(query), fresh.decision_function(query))


def test_refit_to_sklearn_drops_prepared_device_model(data):
    model = _fit(data, "cpu", estimator=PatchedSVC)
    assert hasattr(model, "_prepared_kernel_")
    with tuning_context(force_backend="sklearn"):
        model.fit(data[0], data[1])
    assert model.execution_backend_ == "sklearn"
    assert not hasattr(model, "_prepared_kernel_")
    assert model.predict(data[2]).shape == (len(data[2]),)


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_query_validation_still_precedes_preparation_or_reuse(data, backend):
    model = _fit(data, backend)
    prepared = model._prepared_kernel_
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, data[0].shape[1] + 1)))
    with pytest.raises(ValueError, match="NaN"):
        model.decision_function(np.full((2, data[0].shape[1]), np.nan))
    assert model._prepared_kernel_ is prepared


def test_failed_refit_does_not_leave_prepared_inference_state(data):
    model = _fit(data, "cpu")
    with tuning_context(force_backend="cpu"), pytest.raises(ValueError, match="class"):
        model.fit(data[0], np.zeros_like(data[1]))
    assert not hasattr(model, "_prepared_kernel_")
    assert not hasattr(model, "_execution_backend_")
