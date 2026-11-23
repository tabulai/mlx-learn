"""Regression tests for defects found while verifying the radius rewrite."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
from sklearn import neighbors as sk_neighbors

from mlxlearn import config_context
from mlxlearn._common.config import tuning_context
from mlxlearn._kernels import radius as radius_module
from mlxlearn.exceptions import CapabilityError
from mlxlearn.neighbors import (
    NearestNeighbors,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from mlxlearn.patching import _estimators as patched


def _data(rng, n=2000, d=6, nq=50):
    X = rng.normal(size=(n, d)).astype(np.float32)
    Q = rng.normal(size=(nq, d)).astype(np.float32)
    return X, Q


@pytest.mark.parametrize("cls", [RadiusNeighborsClassifier, RadiusNeighborsRegressor])
def test_weights_none_means_uniform(rng, cls):
    X, Q = _data(rng)
    y = (X[:, 0] > 0).astype(int)
    kwargs = {"outlier_label": "most_frequent"} if cls is RadiusNeighborsClassifier else {}
    none = cls(radius=1.5, weights=None, **kwargs).fit(X, y)
    uniform = cls(radius=1.5, weights="uniform", **kwargs).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_array_equal(none.predict(Q), uniform.predict(Q))


@pytest.mark.parametrize("route", ["tree", "brute", "metal"])
@pytest.mark.parametrize("native", [True, False])
def test_distance_weighted_all_outlier_batches(rng, route, native, monkeypatch):
    if not native:
        monkeypatch.setattr(radius_module, "native_engine", lambda: None)
    X, _ = _data(rng, n=3000, d=12)
    y = (X[:, 0] > 0).astype(int)
    far = np.full((40, 12), 50.0, dtype=np.float32)
    clf = RadiusNeighborsClassifier(radius=1.0, weights="distance", outlier_label=1).fit(X, y)
    sk = sk_neighbors.RadiusNeighborsClassifier(
        radius=1.0, weights="distance", outlier_label=1).fit(X.astype(np.float64), y)
    try:
        with tuning_context(radius_route=route):
            proba = clf.predict_proba(far)
            labels = clf.predict(far)
    except ValueError as exc:  # no GPU for the forced Metal route
        pytest.skip(str(exc))
    np.testing.assert_array_equal(proba, sk.predict_proba(far.astype(np.float64)))
    np.testing.assert_array_equal(labels, sk.predict(far.astype(np.float64)))
    assert proba.dtype == np.float64


def test_stock_twin_keeps_feature_names(rng):
    pd = pytest.importorskip("pandas")
    X, Q = _data(rng)
    columns = [f"f{i}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=columns)
    model = patched.NearestNeighbors(radius=1.0).fit(frame)
    query = pd.DataFrame(Q, columns=columns)
    radii = np.full(len(Q), 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.radius_neighbors(query, radius=radii)  # delegated: per-query radii
    with pytest.raises(ValueError, match="feature names"):
        model.radius_neighbors(query[columns[::-1]], radius=radii)


def test_capability_delegation_honors_fallback_policy(rng):
    X, Q = _data(rng)
    model = patched.NearestNeighbors(radius=1.0).fit(X)
    radii = np.full(len(Q), 1.0)
    with config_context(fallback_policy="raise"):
        with pytest.raises(CapabilityError):
            model.radius_neighbors(Q, radius=radii)
    with config_context(fallback_policy="warn"):
        with pytest.warns(Warning):
            model.radius_neighbors(Q, radius=radii)


def test_stock_twin_uses_fit_time_parameters(rng):
    X, Q = _data(rng, d=4)
    model = patched.NearestNeighbors(radius=1.0, algorithm="auto").fit(X)
    model.set_params(algorithm="brute", radius=0.5)
    radii = np.full(len(Q), 0.5)
    model.radius_neighbors(Q, radius=radii)
    twin = model._knn_query_state_["stock_radius"]
    assert twin.algorithm == "auto" and twin.radius == 0.5


@pytest.mark.parametrize("label", [[1], (1,), np.array([1])])
def test_single_output_outlier_label_as_a_sequence(rng, label):
    X, Q = _data(rng)
    y = (X[:, 0] > 0).astype(int)
    try:
        stock = sk_neighbors.RadiusNeighborsClassifier(radius=0.1, outlier_label=label).fit(
            X.astype(np.float64), y)
    except (TypeError, ValueError) as exc:  # scikit-learn rejects it: so must we
        with pytest.raises(type(exc)):
            RadiusNeighborsClassifier(radius=0.1, outlier_label=label).fit(X, y)
        return
    ours = RadiusNeighborsClassifier(radius=0.1, outlier_label=label).fit(X, y)
    np.testing.assert_array_equal(ours.predict(Q), stock.predict(Q.astype(np.float64)))


def test_regressor_keeps_column_targets_and_float32(rng):
    X, Q = _data(rng)
    y = rng.normal(size=(len(X), 1)).astype(np.float32)
    ours = patched.RadiusNeighborsRegressor(radius=2.0).fit(X, y)
    stock = sk_neighbors.RadiusNeighborsRegressor(radius=2.0).fit(X.astype(np.float64), y)
    pred, ref = ours.predict(Q), stock.predict(Q.astype(np.float64))
    assert pred.shape == ref.shape == (len(Q), 1)
    assert pred.dtype == ref.dtype == np.float32
    # scikit-learn averages in float32; mlx-learn in float64, then casts.
    np.testing.assert_allclose(pred, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("route", ["auto", "tree", "brute", "metal"])
def test_misaligned_queries(rng, route):
    X, Q = _data(rng, n=4000, d=8)
    buffer = bytearray(Q.nbytes + 2)
    buffer[2:] = Q.tobytes()
    misaligned = np.frombuffer(buffer, dtype=np.float32, offset=2).reshape(Q.shape)
    assert not misaligned.flags.aligned
    model = NearestNeighbors(radius=1.5).fit(X)
    try:
        with tuning_context(radius_route=route):
            dist, ind = model.radius_neighbors(misaligned)
            ref_dist, ref_ind = model.radius_neighbors(Q)
    except ValueError as exc:  # no GPU for the forced Metal route
        pytest.skip(str(exc))
    for a, b in zip(ind, ref_ind, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(dist, ref_dist, strict=True):
        np.testing.assert_array_equal(a, b)


def test_thread_policy_follows_joblib():
    import os

    cores = os.cpu_count() or 1
    assert radius_module.resolve_threads(None) == cores
    assert radius_module.resolve_threads(-1) == cores
    assert radius_module.resolve_threads(-2) == max(1, cores - 1)
    assert radius_module.resolve_threads(3) == 3
    with tuning_context(radius_threads=2):
        assert radius_module.resolve_threads(-1) == 2


def test_forked_children_run_single_threaded(monkeypatch):
    monkeypatch.setattr(radius_module, "_FORKED", True)
    assert radius_module.resolve_threads(None) == 1


def test_missing_extension_reports_unavailable():
    code = textwrap.dedent("""
        import sys
        sys.modules["mlxlearn._kernels._radius_cpu_native"] = None
        from mlxlearn._kernels import radius_cpu, radius
        assert radius_cpu.AVAILABLE is False
        assert radius.native_engine() is None
        import numpy as np
        from mlxlearn.neighbors import NearestNeighbors
        X = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
        d, i = NearestNeighbors(radius=0.5).fit(X).radius_neighbors(X[:5])
        print("ok", len(i))
    """)
    import os

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("ok")


def test_native_vote_buffers_are_checked_without_overflow():
    native = radius_module.native_engine()
    if native is None:
        pytest.skip("native extension not built")
    from mlxlearn._kernels.radius_contract import thresholds

    X = np.zeros((4, 2), dtype=np.float32)
    result = native.brute_query(X, X, thresholds(2, 1.0), need_distance=False, threads=1)
    raw = native._native
    with pytest.raises(ValueError):
        raw.result_fill_votes(result._capsule if hasattr(result, "_capsule") else result._result,
                              np.zeros(4, np.int64), 2**62, False, np.zeros(0),
                              np.zeros(4, np.int64))
