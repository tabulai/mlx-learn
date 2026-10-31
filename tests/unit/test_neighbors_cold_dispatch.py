"""CPU-eligible fits defer GPU work without moving it to CPU inference."""

from __future__ import annotations

import builtins
import functools
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import mlxlearn
from mlxlearn._common.config import config_context
from mlxlearn.exceptions import DeviceError, UnsupportedParameterError
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors


def _no_device_work(monkeypatch):
    from mlxlearn._common import base, device
    from mlxlearn._kernels import distance

    def reject(*args, **kwargs):
        raise AssertionError("CPU-only fit/query attempted device initialization")

    monkeypatch.setattr(base, "resolve_device", reject)
    monkeypatch.setattr(device, "gpu_available", reject)
    monkeypatch.setattr(distance, "use_device", reject)
    original_import = builtins.__import__

    def no_device_import(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx.") or name.startswith("mlxlearn_device"):
            raise AssertionError(f"CPU-only fit/query imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_device_import)


@pytest.mark.parametrize("shape", [(150, 4), (569, 30), (2048, 64), (8192, 8), (16384, 32)])
def test_cpu_eligible_fit_has_no_gpu_probe_or_resident_device_state(monkeypatch, shape):
    _no_device_work(monkeypatch)
    X = np.zeros(shape, np.float32)
    model = NearestNeighbors(n_neighbors=5).fit(X)
    assert model.execution_policy_ == "auto" and model.execution_backend_ == "cpu"
    assert model._index_._gpu_available is None
    assert model._index_._device_index is None and model._index_._cpu_index is None
    assert not hasattr(model, "last_query_backend_")


@pytest.mark.parametrize("estimator", [NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor])
@pytest.mark.parametrize("portable", [False, True])
def test_single_query_stays_host_only_in_native_and_portable_contexts(monkeypatch, estimator, portable):
    from mlxlearn._kernels import neighbors_cpu

    if portable:
        monkeypatch.setattr(neighbors_cpu, "_native_query", None)
    _no_device_work(monkeypatch)
    X = np.random.default_rng(718).integers(-32, 33, size=(8192, 8)).astype(np.float32) / 8
    Q = X[[7]].copy()
    model = estimator(n_neighbors=1)
    if estimator is NearestNeighbors:
        model.fit(X)
        np.testing.assert_array_equal(model.kneighbors(Q, return_distance=False), [[7]])
    else:
        y = np.arange(len(X)) % 3
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(Q), [y[7]])
        if estimator is KNeighborsClassifier:
            np.testing.assert_array_equal(model.predict_proba(Q), [[0.0, 1.0, 0.0]])
            assert model._y_device_ is None
    assert model.last_query_backend_ == "cpu" and model._index_._gpu_available is None
    assert model._index_._device_index is None


@pytest.mark.parametrize("container", ["list", "dataframe"])
def test_small_other_containers_also_defer_device_selection(monkeypatch, container):
    X = np.arange(32, dtype=np.float32).reshape(8, 4)
    if container == "dataframe":
        pd = pytest.importorskip("pandas")
        values = pd.DataFrame(X, columns=list("abcd"))
    else:
        values = X.tolist()
    _no_device_work(monkeypatch)
    model = NearestNeighbors(n_neighbors=1).fit(values)
    assert model.execution_backend_ == "cpu" and model._index_._gpu_available is None
    if container == "dataframe":
        np.testing.assert_array_equal(model.feature_names_in_, list("abcd"))


def test_invalid_auto_fit_does_not_probe_or_publish_fitted_backend(monkeypatch):
    _no_device_work(monkeypatch)
    model = NearestNeighbors(n_neighbors=1)
    X = np.ones((8, 4), np.float32)
    model.fit(X)
    X[-1, -1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X)
    assert not hasattr(model, "execution_backend_")
    assert not hasattr(model, "execution_policy_")
    assert not hasattr(model, "last_query_backend_")
    assert "_knn_query_state_" not in vars(model)


def test_lazy_auto_selection_preserves_capability_fallback_policy(monkeypatch):
    import sklearn.neighbors

    # Patching all estimator families may import their device solvers; only
    # the neighbor fit and query below promise to avoid device initialization.
    mlxlearn.patch_sklearn()
    _no_device_work(monkeypatch)
    X = np.arange(32, dtype=np.float32).reshape(8, 4)
    with pytest.raises(UnsupportedParameterError):
        NearestNeighbors(algorithm="kd_tree").fit(X)
    with config_context(fallback_policy="silent"):
        model = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(X)
    assert model.execution_backend_ == "sklearn" and model.execution_policy_ == "fixed"
    np.testing.assert_array_equal(model.kneighbors(X[[2]], return_distance=False), [[2]])


def test_explicit_gpu_still_resolves_and_raises_if_unavailable(monkeypatch):
    from mlxlearn._common import base

    requests = []

    def unavailable(requested):
        requests.append(requested)
        raise DeviceError("requested GPU unavailable")

    monkeypatch.setattr(base, "resolve_device", unavailable)
    with config_context(device="gpu"), pytest.raises(DeviceError, match="unavailable"):
        NearestNeighbors().fit(np.ones((8, 4), np.float32))
    assert requests == ["gpu"]


def test_large_gaussian_fit_keeps_gpu_selection_and_only_one_private_conversion(monkeypatch):
    from mlxlearn._common import base
    from mlxlearn.neighbors import _base

    X = np.zeros((50000, 32), np.float64)
    conversions, selections, builds = [], [], []

    def convert(values):
        conversions.append(values)
        return values.astype(np.float32), True

    def select(requested):
        selections.append(requested)
        return "gpu"

    def build(values, **kwargs):
        builds.append((values, kwargs))
        return SimpleNamespace(X=values)

    monkeypatch.setattr(_base, "cast_float32_checked", convert)
    monkeypatch.setattr(base, "resolve_device", select)
    monkeypatch.setattr(_base, "build_index", build)
    model = NearestNeighbors().fit(X)
    assert len(conversions) == 1 and conversions[0] is X
    assert selections == ["auto"]
    assert model.execution_backend_ == "mlx" and model.execution_policy_ == "auto"
    assert len(builds) == 1 and builds[0][1] == {
        "backend": "mlx", "_owned": True, "_auto": True, "_gpu_available": True,
    }


def test_known_unavailable_gpu_remains_cpu_after_pickle(monkeypatch):
    from mlxlearn._common import device

    X = np.random.default_rng(543).integers(-32, 33, size=(512, 16)).astype(np.float32) / 8
    Q = X[:128].copy()
    monkeypatch.setattr(device, "gpu_available", lambda: False)
    model = NearestNeighbors(n_neighbors=1).fit(X)
    expected = model.kneighbors(Q)
    assert model._index_._gpu_available is False and model.last_query_backend_ == "cpu"
    restored = pickle.loads(pickle.dumps(model))

    def reject():
        raise AssertionError("known unavailable GPU was probed again")

    monkeypatch.setattr(device, "gpu_available", reject)
    actual = restored.kneighbors(Q)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert restored._index_._gpu_available is False and restored.last_query_backend_ == "cpu"


@pytest.mark.gpu
def test_unknown_gpu_survives_pickle_and_is_probed_only_when_a_gpu_query_arrives(monkeypatch):
    from mlxlearn._common import device

    original_probe = device.gpu_available
    probes = []

    @functools.lru_cache(maxsize=1)
    def probe_once():
        probes.append(True)
        return original_probe()

    monkeypatch.setattr(device, "gpu_available", probe_once)
    X = np.random.default_rng(543).integers(-32, 33, size=(512, 16)).astype(np.float32) / 8
    model = NearestNeighbors(n_neighbors=1).fit(X)
    assert not probes and model._index_._gpu_available is None
    restored = pickle.loads(pickle.dumps(model))
    assert not probes and restored._index_._gpu_available is None
    restored.kneighbors(X[:1])
    assert not probes and restored.last_query_backend_ == "cpu"
    for _ in range(2):
        np.testing.assert_array_equal(restored.kneighbors(X[:128], return_distance=False),
                                      np.arange(128)[:, None])
        assert restored.last_query_backend_ == "mlx"
    assert probes == [True] and restored._index_._gpu_available is True
