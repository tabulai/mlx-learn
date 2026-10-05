"""The shared runtime's failure modes, asserted directly.

Configuration scope across threads and processes, the device gate, fork safety,
the MLX guard, buffer-cache release, one event per fit and per public call, and
what "not fitted" means after a fit that did not complete. Each of these is a
property of ``_common``, not of any one estimator, so the tests run it across all
five estimators and both layers where that is cheap.
"""

from __future__ import annotations

import asyncio
import pickle
import re
import subprocess
import sys
import textwrap
import threading
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors

import mlxlearn
from mlxlearn._common import base as base_module
from mlxlearn._common.base import mlx_guard, release_mlx_cache
from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available, inference_device, use_device
from mlxlearn._common.diagnostics import BackendEvent, diagnostics_summary, record_event
from mlxlearn._common.notfitted import IncompleteFitError
from mlxlearn.exceptions import (
    BackendExecutionError,
    CapabilityError,
    DeviceError,
    InvalidInputError,
    MLXLearnError,
    MLXLearnFallbackWarning,
    SparseInputError,
    UnsupportedInputError,
)
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from mlxlearn.patching import _estimators as patched
from mlxlearn.svm import SVC
from tests._helpers import join_or_fail

ESTIMATORS = [
    (KNeighborsClassifier, patched.KNeighborsClassifier, {"n_neighbors": 3}),
    (KNeighborsRegressor, patched.KNeighborsRegressor, {"n_neighbors": 3}),
    (NearestNeighbors, patched.NearestNeighbors, {"n_neighbors": 3}),
    (LogisticRegression, patched.LogisticRegression, {}),
    (SVC, patched.SVC, {}),
]
IDS = [layer1.__name__ for layer1, _, _ in ESTIMATORS]

#: Every crossover at zero, so a small problem takes the MLX path by the normal
#: route rather than through force_backend.
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
}

needs_gpu = pytest.mark.skipif(not gpu_available(), reason="needs a working MLX GPU")


@pytest.fixture(scope="module")
def data():
    return make_classification(
        n_samples=240, n_features=6, n_informative=4, n_redundant=0, n_classes=3,
        n_clusters_per_class=1, random_state=0,
    )


def _estimator(pair, layer):
    layer1, layer2, kwargs = pair
    return (layer1 if layer == "L1" else layer2)(**kwargs)


def _infer(model, X):
    if hasattr(model, "predict"):
        return model.predict(X)
    return model.kneighbors(X)[1]


def _fit_events():
    return [e for e in mlxlearn.get_backend_diagnostics() if e.operation == "fit"]


def _without_device_error(call):
    """``call()``, failing on a DeviceError: the GPU policy would report it as a skip.

    For tests that simulate a host without a GPU, where a DeviceError is the bug
    under test rather than a missing device.
    """
    try:
        return call()
    except DeviceError as exc:
        pytest.fail(f"DeviceError where no GPU was needed: {exc}")


def _run(code: str, **env) -> subprocess.CompletedProcess:
    import os

    full_env = {**os.environ, **env}
    full_env["PYTHONPATH"] = os.pathsep.join(
        [str(__import__("pathlib").Path(mlxlearn.__file__).parents[1])]
        + ([full_env["PYTHONPATH"]] if full_env.get("PYTHONPATH") else [])
    )
    return subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", textwrap.dedent(code)],
        capture_output=True, text=True, timeout=120, env=full_env,
    )


# ======================================================================================
# configuration scope
# ======================================================================================


def test_set_config_reaches_other_threads():
    """set_config is the process-wide default; a per-thread one vanished in n_jobs workers."""
    mlxlearn.set_config(fallback_policy="raise", device="cpu")
    seen = {}

    def worker():
        config = mlxlearn.get_config()
        seen["value"] = (config.fallback_policy, config.device)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen["value"] == ("raise", "cpu")


def test_fallback_policy_raise_holds_in_threaded_cross_validation(data):
    """The reproduced bug: n_jobs=2 on the threading backend completed through a fallback."""
    from joblib import parallel_backend
    from sklearn.model_selection import cross_val_score

    X, y = data
    mlxlearn.set_config(fallback_policy="raise")
    estimator = patched.KNeighborsClassifier(n_neighbors=3, metric="manhattan")
    with parallel_backend("threading", n_jobs=2):
        with pytest.raises(CapabilityError, match="fallback_policy='raise'"):
            cross_val_score(estimator, X, y, cv=2, error_score="raise")


def test_config_context_reaches_asyncio_to_thread():
    """Context variables travel with asyncio; a thread-local did not."""

    async def main():
        with mlxlearn.config_context(device="cpu"):
            return await asyncio.to_thread(lambda: mlxlearn.get_config().device)

    assert asyncio.run(main()) == "cpu"


def test_set_config_inside_a_context_changes_the_default_not_the_override():
    with mlxlearn.config_context(fallback_policy="raise"):
        mlxlearn.set_config(fallback_policy="silent", device="cpu")
        # The context's own field wins; the rest shows the new default at once.
        assert mlxlearn.get_config().fallback_policy == "raise"
        assert mlxlearn.get_config().device == "cpu"
    assert mlxlearn.get_config().fallback_policy == "silent"
    assert mlxlearn.get_config().device == "cpu"


def test_nested_contexts_restore_in_order():
    with mlxlearn.config_context(device="cpu"):
        with mlxlearn.config_context(fallback_policy="raise"):
            config = mlxlearn.get_config()
            assert (config.device, config.fallback_policy) == ("cpu", "raise")
        assert mlxlearn.get_config().fallback_policy == "warn"
        assert mlxlearn.get_config().device == "cpu"
    assert mlxlearn.get_config().device == "auto"


def test_invalid_context_changes_nothing():
    with pytest.raises(ValueError):
        with mlxlearn.config_context(device="cpu", fallback_policy="ignore"):
            pass  # pragma: no cover
    assert mlxlearn.get_config().device == "auto"


def test_reset_config_restores_the_process_default():
    mlxlearn.set_config(fallback_policy="silent")
    mlxlearn.reset_config()
    assert mlxlearn.get_config().fallback_policy == "warn"


def test_environment_sets_the_process_default(monkeypatch):
    monkeypatch.setenv("MLXLEARN_FALLBACK_POLICY", "raise")
    monkeypatch.setenv("MLXLEARN_DEVICE", "cpu")
    monkeypatch.setenv("MLXLEARN_DIAGNOSTICS", "false")
    monkeypatch.setenv("MLXLEARN_RANDOM_STATE", "7")
    mlxlearn.reset_config()
    config = mlxlearn.get_config()
    assert (config.fallback_policy, config.device, config.diagnostics, config.random_state) == (
        "raise", "cpu", False, 7,
    )


@pytest.mark.parametrize(
    "name, value",
    [
        ("MLXLEARN_FALLBACK_POLICY", "loud"),
        ("MLXLEARN_DIAGNOSTICS", "maybe"),
        ("MLXLEARN_RANDOM_STATE", "seven"),
        ("MLXLEARN_DEVICE", "mps"),
    ],
)
def test_malformed_configuration_environment_is_an_error(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        mlxlearn.reset_config()


def test_a_fresh_process_starts_from_the_environment():
    """What a loky or multiprocessing worker sees: it imports mlxlearn afresh."""
    result = _run(
        """
        import mlxlearn
        config = mlxlearn.get_config()
        print(config.fallback_policy, config.device)
        """,
        MLXLEARN_FALLBACK_POLICY="raise",
        MLXLEARN_DEVICE="cpu",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["raise", "cpu"]


# ======================================================================================
# the device gate
# ======================================================================================


def test_use_device_yields_a_stream_on_the_resolved_device():
    import mlx.core as mx

    with use_device("cpu") as stream:
        assert stream.device == mx.cpu
        assert mx.default_device() == mx.cpu


def test_use_device_leaves_a_matching_default_alone(monkeypatch):
    """The common case — the default already is the device — changes no global."""
    import mlx.core as mx

    calls = []
    original = mx.set_default_device
    monkeypatch.setattr(mx, "set_default_device", lambda d: calls.append(d) or original(d))
    before = mx.default_device()
    kind = "gpu" if before == mx.gpu else "cpu"
    with use_device(kind):
        pass
    assert calls == []


def test_nested_use_device_switches_and_restores():
    import mlx.core as mx

    before = mx.default_device()
    with use_device("cpu"):
        with use_device("cpu"):
            assert mx.default_device() == mx.cpu
        if gpu_available():
            with use_device("gpu"):
                assert mx.default_device() == mx.gpu
            assert mx.default_device() == mx.cpu
    assert mx.default_device() == before


@needs_gpu
def test_concurrent_blocks_on_different_devices_never_see_the_wrong_one():
    """mx.stream is process-global in MLX 0.32: one thread's block changed another's device."""
    import mlx.core as mx

    before = mx.default_device()
    wrong: list[tuple[str, object]] = []
    errors: list[BaseException] = []
    start = threading.Barrier(6, timeout=60)

    def worker(kind):
        target = mx.gpu if kind == "gpu" else mx.cpu
        try:
            start.wait()
            for _ in range(40):
                with use_device(kind):
                    for _ in range(3):
                        x = mx.ones((64, 64)) @ mx.ones((64, 64))
                        mx.eval(x)
                        if mx.default_device() != target:
                            wrong.append((kind, mx.default_device()))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(k,), daemon=True) for k in ("gpu", "cpu") * 3
    ]
    for thread in threads:
        thread.start()
    join_or_fail(threads, timeout=60)
    assert not errors, errors
    assert not wrong, wrong[:5]
    assert mx.default_device() == before


# ======================================================================================
# fork safety
# ======================================================================================


def test_a_forked_child_gets_a_device_error_not_a_crash():
    """Metal is not fork-safe: the child used to die with SIGSEGV on any MLX call."""
    result = _run(
        """
        import os, sys, warnings
        import numpy as np
        from mlxlearn.neighbors import KNeighborsClassifier
        from mlxlearn.patching._estimators import KNeighborsClassifier as Patched
        from mlxlearn._common.config import tuning_context
        from mlxlearn.exceptions import DeviceError

        rng = np.random.default_rng(0)
        X = rng.normal(size=(600, 8)); y = (X[:, 0] > 0).astype(int)
        KNeighborsClassifier().fit(X, y).predict(X[:5])   # the parent really uses MLX

        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                KNeighborsClassifier().fit(X, y)
                code = 10                     # Layer 1 must refuse
            except DeviceError:
                pass
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with tuning_context(
                    knn_min_train_samples=0, knn_min_work=0, knn_min_features=0,
                    knn_narrow_min_samples=0,
                ):
                    model = Patched().fit(X, y)   # Layer 2 falls back
            if model.execution_backend_ != "sklearn":
                code = 11
            os._exit(code)
        _, status = os.waitpid(pid, 0)
        print(os.waitstatus_to_exitcode(status))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0", result.stderr


# ======================================================================================
# inference on a host without a GPU
# ======================================================================================


def test_inference_device_degrades_only_when_the_device_was_not_named(monkeypatch):
    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    assert inference_device("mlx") == "cpu"
    assert inference_device("cpu") == "cpu"
    with mlxlearn.config_context(device="cpu"):
        assert inference_device("mlx") == "cpu"
    with mlxlearn.config_context(device="gpu"):
        with pytest.raises(DeviceError, match="fitted on the MLX GPU backend"):
            inference_device("mlx")


def test_inference_device_keeps_the_fitted_device_when_it_exists(monkeypatch):
    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: True)
    with mlxlearn.config_context(device="cpu"):
        assert inference_device("mlx") == "gpu"


@needs_gpu
@pytest.mark.parametrize("kind", ["NearestNeighbors", "SVC"])
def test_a_gpu_fitted_model_answers_on_the_mlx_cpu_without_a_gpu(monkeypatch, data, kind):
    """Unpickled where no GPU works: MLX's CPU under device="auto" or "cpu".

    The estimator resolves the device from the configuration and hands it to the
    kernels; an explicit device="gpu" is still refused rather than substituted.
    """
    X, y = data
    with tuning_context(force_backend="mlx"):
        model = SVC().fit(X, y) if kind == "SVC" else NearestNeighbors(n_neighbors=3).fit(X)

    def answer(m):
        if kind == "NearestNeighbors":
            return m.kneighbors(X[:20])[1]
        # Twenty rows are a batch SVC would answer on the host; this is about the
        # device path (test_small_svc_batches_are_answered_on_the_host is the other).
        with tuning_context(svc_host_inference_work=0):
            return m.decision_function(X[:20])

    expected = answer(model)
    payload = pickle.dumps(model)
    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    for device in ("auto", "cpu"):
        with mlxlearn.config_context(device=device):
            actual = _without_device_error(lambda: answer(pickle.loads(payload)))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    with mlxlearn.config_context(device="gpu"), pytest.raises(DeviceError, match="fitted on the MLX GPU"):
        answer(pickle.loads(payload))


def test_gpu_fitted_inference_on_a_gpu_less_host_is_recorded(monkeypatch, data):
    X, y = data
    with tuning_context(force_backend="mlx"):
        model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    model.predict(X[:10])
    assert mlxlearn.get_last_backend_event().reason == "gpu-unavailable"


@needs_gpu
def test_small_svc_batches_are_answered_on_the_host(monkeypatch, data):
    """At or below svc_host_inference_work SVC evaluates the batch in NumPy float32.

    The same arithmetic as the device tiles, so the answer agrees with the device
    path's to float32 rounding; no device is used, so a GPU-less host, even under
    device="gpu", answers it.
    """
    X, y = data
    with tuning_context(force_backend="mlx"):
        model = SVC().fit(X, y)
    with tuning_context(svc_host_inference_work=0):
        device = model.decision_function(X[:7])
    host = model.decision_function(X[:7])
    event = mlxlearn.get_last_backend_event()
    assert (event.backend, event.reason) == ("mlx", "sticky")
    assert event.extra["compute"] == "numpy-float32"
    np.testing.assert_allclose(host, device, rtol=1e-5, atol=1e-5)

    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    restored = pickle.loads(pickle.dumps(model))
    with mlxlearn.config_context(device="gpu"):
        np.testing.assert_allclose(restored.decision_function(X[:7]), host, rtol=1e-6, atol=1e-6)
    assert mlxlearn.get_last_backend_event().extra["compute"] == "numpy-float32"


def test_host_side_inference_is_not_gpu_unavailable(monkeypatch, data):
    """LogisticRegression predicts in NumPy float64 on every backend: no GPU is missed."""
    X, y = data
    with tuning_context(force_backend="mlx"):
        model = LogisticRegression().fit(X, y)
    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    model.predict(X[:10])
    event = mlxlearn.get_last_backend_event()
    assert (event.backend, event.reason) == ("mlx", "sticky")
    assert event.extra["compute"] == "numpy-float64"
    assert event.extra["fit_reason"] == "forced"


# ======================================================================================
# mlx_guard
# ======================================================================================


@pytest.mark.parametrize(
    "error",
    [
        MLXLearnError("deliberate"),
        InvalidInputError("bad input"),
        UnsupportedInputError("capability"),
        MemoryError(),
        RuntimeWarning("escalated by the user's filter"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_mlx_guard_passes_deliberate_errors_through(error):
    with pytest.raises(type(error)) as caught:
        with mlx_guard("Est", "fit"):
            raise error
    assert caught.value is error


@pytest.mark.parametrize(
    "error",
    [
        ValueError("[matmul] Last dimension of first input must match"),
        TypeError("astype(): incompatible function arguments."),
        RuntimeError("something broke."),
        OSError("disk"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_mlx_guard_wraps_everything_else(error):
    """MLX raises its own shape errors as ValueError; they are bugs, not bad input."""
    with pytest.raises(BackendExecutionError) as caught:
        with mlx_guard("Est", "fit"):
            raise error
    assert caught.value.__cause__ is error
    assert caught.value.original is error
    assert "Est.fit failed inside the MLX backend" in str(caught.value)
    assert ".." not in str(caught.value)


def test_mlx_guard_describes_an_allocation_failure_as_one():
    error = RuntimeError(
        "[metal::malloc] Attempting to allocate 102400000000 bytes which is greater "
        "than the maximum allowed buffer size of 86586540032 bytes."
    )
    with pytest.raises(BackendExecutionError) as caught:
        with mlx_guard("Est", "kneighbors"):
            raise error
    message = str(caught.value)
    assert "could not allocate" in message
    assert "bytes.." not in message


#: Where each fit's MLX work is; a crash at any of them must raise. SVC's three-class
#: fit runs its pairs in the one-vs-one kernel on the GPU, and the host loop elsewhere.
_CRASH_SITES = {
    "KNeighborsClassifier": ("mlxlearn.neighbors._base.build_index",),
    "KNeighborsRegressor": ("mlxlearn.neighbors._base.build_index",),
    "NearestNeighbors": ("mlxlearn.neighbors._base.build_index",),
    "LogisticRegression": ("mlxlearn.linear_model._logistic.solve_logistic",),
    "SVC": (
        "mlxlearn.svm._classes.solve_c_svc",
        "mlxlearn._kernels._ovo.PairSolver.solve",
    ),
}


@pytest.mark.parametrize("policy", ["warn", "silent", "raise"])
@pytest.mark.parametrize("layer", ["L1", "L2"])
@pytest.mark.parametrize("pair", ESTIMATORS, ids=IDS)
def test_an_mlx_crash_raises_and_never_falls_back(monkeypatch, data, pair, layer, policy):
    """The invariant the whole fallback design rests on, under every policy."""
    X, y = data

    def crash(*args, **kwargs):
        raise RuntimeError("simulated MLX failure")

    for site in _CRASH_SITES[pair[0].__name__]:
        monkeypatch.setattr(site, crash)
    model = _estimator(pair, layer)
    with mlxlearn.config_context(fallback_policy=policy), tuning_context(force_backend="mlx"):
        with pytest.raises(BackendExecutionError) as caught:
            model.fit(X, y)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert not any(e.fallback for e in mlxlearn.get_backend_diagnostics())
    assert not hasattr(model, "_execution_backend_")


@pytest.mark.parametrize("layer", ["L1", "L2"])
def test_an_mlx_crash_at_inference_raises_too(monkeypatch, data, layer):
    from mlxlearn._kernels.distance import NeighborIndex

    X, y = data
    with tuning_context(force_backend="mlx"):
        model = _estimator(ESTIMATORS[0], layer).fit(X, y)

    def crash(self, *args, **kwargs):
        raise RuntimeError("simulated MLX failure")

    monkeypatch.setattr(NeighborIndex, "query", crash)
    mlxlearn.clear_backend_diagnostics()
    with pytest.raises(BackendExecutionError):
        model.predict(X[:5])
    assert not any(e.fallback for e in mlxlearn.get_backend_diagnostics())


# ======================================================================================
# MLX buffer-cache release
# ======================================================================================


def _grow_cache(mx, n_bytes: int) -> None:
    # A computed buffer, freed, lands in MLX's cache. The trailing tiny eval makes
    # MLX drop its own last reference to the big one. The synchronize waits for the
    # command buffer's completion handler, which frees that eval's own temporaries
    # on another thread: without it they could land in the cache just *after* the
    # release under test, and the "== 0" checks failed about one run in three.
    buffer = mx.ones((n_bytes // 4,), dtype=mx.float32) * 2
    mx.eval(buffer)
    del buffer
    mx.eval(mx.zeros((1,)))
    mx.synchronize()


def test_a_block_that_grew_the_cache_releases_it():
    import mlx.core as mx

    mx.clear_cache()
    with tuning_context(mlx_cache_release_fraction=1e-9):
        with release_mlx_cache():
            _grow_cache(mx, 64 * 1024 * 1024)
    assert mx.get_cache_memory() == 0


def test_release_can_be_disabled_and_nested_blocks_defer_to_the_outermost():
    import mlx.core as mx

    mx.clear_cache()
    with tuning_context(mlx_cache_release_fraction=0.0):
        with release_mlx_cache():
            _grow_cache(mx, 64 * 1024 * 1024)
    assert mx.get_cache_memory() > 0

    mx.clear_cache()
    with tuning_context(mlx_cache_release_fraction=1e-9):
        with release_mlx_cache():
            with release_mlx_cache():
                _grow_cache(mx, 64 * 1024 * 1024)
            assert mx.get_cache_memory() > 0  # the inner block left it alone
    assert mx.get_cache_memory() == 0


def test_growth_below_the_threshold_is_kept():
    """Clearing after every small call measured ~20% slower on repeated queries."""
    import mlx.core as mx

    mx.clear_cache()
    with tuning_context(mlx_cache_release_fraction=0.5):
        with release_mlx_cache():
            _grow_cache(mx, 16 * 1024 * 1024)
    assert mx.get_cache_memory() > 0
    mx.clear_cache()


def test_kneighbors_does_not_leave_its_tiles_cached(data):
    """The reported case: one kneighbors call left gigabytes of idle MLX buffers."""
    import mlx.core as mx

    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 32))
    mx.clear_cache()
    with tuning_context(force_backend="mlx", mlx_cache_release_fraction=1e-9):
        model = NearestNeighbors(n_neighbors=5).fit(X)
        model.kneighbors(X[:2000])
    assert mx.get_cache_memory() == 0


# ======================================================================================
# one fit, one event
# ======================================================================================


@pytest.mark.parametrize("size", ["below", "above", "forced"])
@pytest.mark.parametrize("layer", ["L1", "L2"])
@pytest.mark.parametrize("pair", ESTIMATORS, ids=IDS)
def test_exactly_one_fit_event_per_fit(data, pair, layer, size):
    """It used to be two: a provisional ('mlx', 'selected') before the crossover was known."""
    X, y = data
    tuning = {"below": {}, "above": NO_CROSSOVER, "forced": {"force_backend": "mlx"}}[size]
    with tuning_context(**tuning):
        model = _estimator(pair, layer).fit(X, y)

    fits = _fit_events()
    assert len(fits) == 1, fits
    assert fits[0].backend == model.execution_backend_
    assert model.fit_event_ == fits[0]
    if size == "below":
        assert not any(key.endswith("-> mlx") for key in diagnostics_summary())


def test_a_capability_fallback_records_one_event_and_one_warning(data):
    X, y = data
    with pytest.warns(MLXLearnFallbackWarning) as caught:
        model = patched.KNeighborsClassifier(n_neighbors=3).fit(sp.csr_matrix(X), y)
    assert len([w for w in caught if w.category is MLXLearnFallbackWarning]) == 1
    fits = _fit_events()
    assert [(e.backend, e.reason, e.fallback) for e in fits] == [("sklearn", "sparse-input", True)]
    assert model.fit_event_.reason == "sparse-input"


def test_a_fit_that_fails_after_deciding_records_its_decision(data):
    """It recorded nothing, so get_last_backend_event() described an older call.

    The decision is recorded when it is made: the fit that failed is on the record,
    once, and the estimator is still unfitted.
    """
    X, y = data
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    model.predict(X[:2])
    mlxlearn.clear_backend_diagnostics()
    with pytest.raises(ValueError):
        model.fit(X_bad, y)
    (event,) = mlxlearn.get_backend_diagnostics()
    assert (event.operation, event.backend, event.reason) == ("fit", "cpu", "below-crossover")
    assert mlxlearn.get_last_backend_event() is event
    with pytest.raises(NotFittedError):
        model.predict(X[:2])


def test_a_fallback_that_scikit_learn_then_rejects_is_warned_and_recorded(data):
    """The warning and the record came with a completed fit: here, never."""
    X, y = data
    y_bad = y.astype(np.float64)
    y_bad[0] = np.nan
    model = patched.KNeighborsClassifier(n_neighbors=3, metric="manhattan")
    with pytest.warns(MLXLearnFallbackWarning, match="manhattan"):
        with pytest.raises(ValueError, match="NaN"):
            model.fit(X, y_bad)
    (event,) = _fit_events()
    assert (event.backend, event.reason) == ("sklearn", "unsupported-parameter:metric")
    assert mlxlearn.get_last_backend_event() is event
    with pytest.raises(NotFittedError):
        model.predict(X[:2])


@pytest.mark.parametrize("layer", ["L1", "L2"])
@pytest.mark.parametrize("pair", ESTIMATORS, ids=IDS)
def test_a_list_decided_again_after_validation_is_one_event(data, pair, layer):
    """Input with no shape is decided before validation and again after it."""
    X, y = data
    for tuning in ({}, NO_CROSSOVER):
        mlxlearn.clear_backend_diagnostics()
        with tuning_context(**tuning):
            model = _estimator(pair, layer).fit(X.tolist(), y.tolist())
        assert mlxlearn.get_backend_diagnostics() == [model.fit_event_]


def test_the_fit_event_is_kept_on_the_model(data):
    """Evicted from the ring buffer and absent in loky workers, it used to be simply lost."""
    X, y = data
    with pytest.raises(AttributeError):
        LogisticRegression().fit_event_
    with mlxlearn.config_context(diagnostics=False):
        model = LogisticRegression().fit(X, y)
    assert mlxlearn.get_backend_diagnostics() == []
    assert model.fit_event_.reason == "below-crossover"

    restored = pickle.loads(pickle.dumps(model))
    assert restored.fit_event_ == model.fit_event_
    assert restored.fit_event_.detail == model.fit_event_.detail


def test_backend_events_hash_and_pickle():
    event = record_event("A", "fit", "mlx", reason="selected", shape=(3, 2), custom=[1, 2])
    assert hash(event) == hash(pickle.loads(pickle.dumps(event)))
    assert pickle.loads(pickle.dumps(event)).extra == {"custom": [1, 2]}
    assert len({event, BackendEvent("A", "fit", "mlx", shape=(3, 2))}) == 1


# ======================================================================================
# one public call, one event
# ======================================================================================


def _events_of(call):
    mlxlearn.clear_backend_diagnostics()
    call()
    return [(e.operation, e.backend) for e in mlxlearn.get_backend_diagnostics()]


def test_logistic_predict_records_once(data):
    """scikit-learn's predict calls the overridden decision_function, which recorded again."""
    X, y = data
    model = LogisticRegression().fit(X, y)
    assert _events_of(lambda: model.predict(X)) == [("predict", "cpu")]
    assert _events_of(lambda: model.predict_proba(X)) == [("predict_proba", "cpu")]
    # Named after the call the user made: LogisticRegression overrides
    # predict_log_proba and score itself, so they are not recorded as the
    # predict_proba and predict they use.
    assert _events_of(lambda: model.predict_log_proba(X)) == [("predict_log_proba", "cpu")]
    assert _events_of(lambda: model.score(X, y)) == [("score", "cpu")]


def test_mlx_neighbors_predict_is_recorded_as_predict(data):
    X, y = data
    with tuning_context(force_backend="mlx"):
        model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    assert _events_of(lambda: model.predict(X)) == [("predict", "mlx")]
    assert _events_of(lambda: model.predict_proba(X)) == [("predict_proba", "mlx")]
    assert _events_of(lambda: model.kneighbors(X)) == [("kneighbors", "mlx")]


def test_svc_break_ties_predict_records_once(data):
    X, y = data
    with tuning_context(force_backend="mlx"):
        model = SVC(break_ties=True).fit(X, y)
    assert _events_of(lambda: model.predict(X)) == [("predict", "mlx")]


def test_sklearn_fitted_predict_records_once(data):
    X, y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MLXLearnFallbackWarning)
        model = patched.KNeighborsClassifier(n_neighbors=3, metric="manhattan").fit(X, y)
    assert _events_of(lambda: model.predict(X)) == [("predict", "sklearn")]


def test_sticky_events_carry_the_fit_reason(data):
    X, y = data
    model = LogisticRegression().fit(X, y)
    model.predict(X)
    assert mlxlearn.get_last_backend_event().extra["fit_reason"] == "below-crossover"


def test_separate_estimators_each_record(data):
    X, y = data
    first = LogisticRegression().fit(X, y)
    second = LogisticRegression().fit(X, y)
    assert _events_of(lambda: (first.predict(X), second.predict(X))) == [
        ("predict", "cpu"),
        ("predict", "cpu"),
    ]


# ======================================================================================
# fallback warnings: attribution and deduplication
# ======================================================================================


def _sparse_fit(data):
    X, y = data
    return patched.KNeighborsClassifier(n_neighbors=3).fit(sp.csr_matrix(X), y)


@pytest.mark.parametrize("pair", ESTIMATORS, ids=IDS)
def test_the_fallback_warning_points_at_the_callers_line(data, pair):
    """A fixed stacklevel landed on _common/base.py, which no user filter can match.

    On scikit-learn 1.9, LogisticRegression's warning landed on scikit-learn's
    callback decorator, which wraps its fit.
    """
    X, y = data
    model = _estimator(pair, "L2")
    with pytest.warns(MLXLearnFallbackWarning) as caught:
        model.fit(sp.csr_matrix(X), y)
    assert [w.filename for w in caught if w.category is MLXLearnFallbackWarning] == [__file__]


def test_a_module_filter_written_against_user_code_matches(data):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings(
            "ignore", category=MLXLearnFallbackWarning, module=re.escape(__name__)
        )
        _sparse_fit(data)
    assert not [w for w in caught if w.category is MLXLearnFallbackWarning]


def test_a_filtered_warning_does_not_use_up_deduplication(data):
    """A first fallback inside an 'ignore' block used to silence the warning for good."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _sparse_fit(data)
    with pytest.warns(MLXLearnFallbackWarning):
        _sparse_fit(data)


def test_an_error_filter_raises_on_every_fallback(data):
    """The documented way to assert full acceleration across n_jobs workers."""
    X, y = data
    model = patched.KNeighborsClassifier(n_neighbors=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MLXLearnFallbackWarning)
        for _ in range(2):
            with pytest.raises(MLXLearnFallbackWarning):
                model.fit(sp.csr_matrix(X), y)
            # Like fallback_policy="raise": the fit that raised did not complete.
            with pytest.raises(AttributeError):
                model.execution_backend_


def test_an_error_filter_raises_before_scikit_learn_does_the_work(monkeypatch, data):
    """The warning came only once scikit-learn had run the whole fallback fit.

    Measured at 2.4 s for a patched SVC(probability=True) at 6000 x 64, against
    nothing when the warning comes with the decision — and the refused fit was
    left holding scikit-learn's fitted attributes.
    """
    from sklearn.neighbors import KNeighborsClassifier as SkKNeighborsClassifier

    def stock_fit(self, X, y):
        raise AssertionError("scikit-learn's fit ran before the fallback warning")

    monkeypatch.setattr(SkKNeighborsClassifier, "fit", stock_fit)
    X, y = data
    model = patched.KNeighborsClassifier(n_neighbors=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MLXLearnFallbackWarning)
        with pytest.raises(MLXLearnFallbackWarning):
            model.fit(sp.csr_matrix(X), y)
    assert [e.reason for e in _fit_events()] == ["sparse-input"]
    with pytest.raises(NotFittedError):
        model.predict(X[:2])


def _overflowing(X):
    X = X.copy()
    X[0, 0] = 1e39  # finite in float64, infinite in float32
    return X


#: (Layer 1 estimator, patched estimator, input it cannot serve) per mismatch kind.
_MISMATCHES = {
    "sparse": lambda X: (
        KNeighborsClassifier(n_neighbors=3),
        patched.KNeighborsClassifier(n_neighbors=3),
        sp.csr_matrix(X),
    ),
    "float32-overflow": lambda X: (LogisticRegression(), patched.LogisticRegression(), _overflowing(X)),
    "gamma": lambda X: (SVC(gamma=0.0), patched.SVC(gamma=0.0), X),
    "prebuilt-index": lambda X: (
        NearestNeighbors(n_neighbors=3),
        patched.NearestNeighbors(n_neighbors=3),
        __import__("sklearn.neighbors", fromlist=["KDTree"]).KDTree(X),
    ),
}


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("kind", sorted(_MISMATCHES))
def test_only_layer1_is_told_to_run_under_patch_sklearn(data, kind):
    """A patched estimator's warning advised running under patch_sklearn().

    The advice is Layer 1's alone. A patched estimator reports the same mismatch
    without it and says what it did instead: handed the fit over, or — under
    fallback_policy="raise" — refused to.
    """
    X, y = data
    layer1, layer2, X_bad = _MISMATCHES[kind](X)
    with tuning_context(**NO_CROSSOVER):
        with pytest.raises(CapabilityError) as raised:
            layer1.fit(X_bad, y)
        with pytest.warns(MLXLearnFallbackWarning) as caught:
            layer2.fit(X_bad, y)
        with mlxlearn.config_context(fallback_policy="raise"):
            with pytest.raises(CapabilityError) as refused:
                clone_of_layer2 = type(layer2)(**layer2.get_params())
                clone_of_layer2.fit(X_bad, y)

    assert "patch_sklearn()" in str(raised.value)
    assert raised.value.advice and raised.value.advice in str(raised.value)

    warned = [str(w.message) for w in caught if w.category is MLXLearnFallbackWarning]
    assert len(warned) == 1
    assert "patch_sklearn()" not in warned[0]
    assert warned[0].startswith(raised.value.summary)  # the same mismatch, same words
    assert "handed this fit to scikit-learn" in warned[0]
    assert layer2.execution_backend_ == "sklearn"
    assert "patch_sklearn()" not in layer2.fit_event_.detail

    assert "patch_sklearn()" not in str(refused.value)
    assert "fallback_policy='raise'" in str(refused.value)


@pytest.mark.parametrize(
    "layer1, layer2",
    [
        (NearestNeighbors, patched.NearestNeighbors),
        (KNeighborsClassifier, patched.KNeighborsClassifier),
        (KNeighborsRegressor, patched.KNeighborsRegressor),
    ],
    ids=["NearestNeighbors", "KNeighborsClassifier", "KNeighborsRegressor"],
)
def test_a_training_span_float32_cannot_subtract_falls_back_when_patched(data, layer1, layer2):
    """1e38 is finite in float32, so the range check passes it — but the index
    cannot subtract two coordinates that far apart, and says so only once it is
    built, after backend selection. Patched, that raised instead of falling back.
    """
    X, y = data
    X = X.copy()
    X[0, 0] = 1e38
    kwargs = {"n_neighbors": 3}
    with tuning_context(**NO_CROSSOVER):
        with pytest.raises(UnsupportedInputError) as raised:
            layer1(**kwargs).fit(X, y)
        with pytest.warns(MLXLearnFallbackWarning) as caught:
            model = layer2(**kwargs).fit(X, y)
        with mlxlearn.config_context(fallback_policy="raise"):
            with pytest.raises(CapabilityError, match="fallback_policy='raise'"):
                layer2(**kwargs).fit(X, y)

    assert raised.value.reason == "float32-overflow"
    assert "scikit-learn" in raised.value.advice
    warned = [str(w.message) for w in caught if w.category is MLXLearnFallbackWarning]
    assert len(warned) == 1 and warned[0].startswith(raised.value.summary)
    assert "handed this fit to scikit-learn" in warned[0]
    assert model.execution_backend_ == "sklearn"
    assert model.fit_event_.reason == "float32-overflow" and model.fit_event_.fallback
    expected = SkNearestNeighbors(n_neighbors=3).fit(X).kneighbors(X[:5])[1]
    np.testing.assert_array_equal(model.kneighbors(X[:5])[1], expected)


@pytest.mark.parametrize("layer", ["L1", "L2"])
@pytest.mark.parametrize("pair", [ESTIMATORS[0], ESTIMATORS[4]], ids=["KNN", "SVC"])
def test_an_inference_error_does_not_advise_a_fallback_inference_cannot_take(data, pair, layer):
    """The fitted backend is sticky in both layers, so patch_sklearn() cannot help here."""
    X, y = data
    with tuning_context(force_backend="cpu"):
        model = _estimator(pair, layer).fit(X, y)
    with pytest.raises(UnsupportedInputError) as raised:
        _infer(model, _overflowing(X))
    assert raised.value.reason == "float32-overflow"
    assert "patch_sklearn()" not in str(raised.value)


def test_reset_fallback_warnings_rearms_without_dropping_events(data):
    with pytest.warns(MLXLearnFallbackWarning):
        _sparse_fit(data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _sparse_fit(data)
    assert not [w for w in caught if w.category is MLXLearnFallbackWarning]

    count = len(mlxlearn.get_backend_diagnostics())
    from mlxlearn._common.diagnostics import reset_fallback_warnings

    reset_fallback_warnings()
    assert len(mlxlearn.get_backend_diagnostics()) == count
    with pytest.warns(MLXLearnFallbackWarning):
        _sparse_fit(data)


def test_refusals_and_patch_bookkeeping_are_not_counted_as_work(data):
    X, y = data
    with pytest.raises(SparseInputError):
        KNeighborsClassifier().fit(sp.csr_matrix(X), y)
    record_event("SVC", "patch", "sklearn", reason="patched")
    assert diagnostics_summary() == {}
    events = mlxlearn.get_backend_diagnostics()
    assert events[0].extra["raised"] is True


# ======================================================================================
# not fitted, and what a fit owns
# ======================================================================================


@pytest.mark.parametrize("layer", ["L1", "L2"])
@pytest.mark.parametrize("pair", ESTIMATORS, ids=IDS)
def test_inference_after_a_failed_refit_is_not_fitted_error(data, pair, layer):
    """It raised a bare MLXLearnError, which scikit-learn-style handlers do not catch."""
    X, y = data
    model = _estimator(pair, layer).fit(X, y)
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.fit(X_bad, y)
    with pytest.raises(NotFittedError):
        _infer(model, X[:5])
    with pytest.raises(AttributeError):
        model.execution_backend_


def test_the_incomplete_fit_error_is_both_families():
    error = IncompleteFitError("x")
    assert isinstance(error, MLXLearnError)
    assert isinstance(error, NotFittedError)
    assert isinstance(pickle.loads(pickle.dumps(error)), IncompleteFitError)


def test_an_unfitted_estimator_is_not_fitted(data):
    from sklearn.utils.validation import check_is_fitted

    with pytest.raises(NotFittedError):
        check_is_fitted(LogisticRegression())


def test_state_assigned_from_outside_counts_as_fitted_for_scikit_learn():
    """No mlxlearn fit involved, so scikit-learn's own attribute rule applies."""
    from sklearn.utils.validation import check_is_fitted

    model = LogisticRegression()
    model.coef_ = np.zeros((1, 2))
    check_is_fitted(model)


class _ProbeNeighbors(patched.KNeighborsClassifier):
    _mlxlearn_mlx_attrs = ("_index_",)


def test_a_fallback_refit_drops_the_previous_mlx_state(data):
    """The old index — a full float32 copy of the training data — survived into pickles."""
    X, y = data
    with tuning_context(force_backend="mlx"):
        model = _ProbeNeighbors(n_neighbors=3).fit(X, y)
    assert hasattr(model, "_index_")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MLXLearnFallbackWarning)
        model.fit(sp.csr_matrix(X), y)
    assert model.execution_backend_ == "sklearn"
    assert not hasattr(model, "_index_")


# ======================================================================================
# tags and array-API dispatch
# ======================================================================================


@pytest.mark.parametrize("pair", ESTIMATORS, ids=IDS)
def test_layer1_does_not_advertise_array_api_support(pair):
    layer1, layer2, kwargs = pair
    assert layer1(**kwargs).__sklearn_tags__().array_api_support is False
    parent = layer1.__mro__[[c.__module__.startswith("sklearn") for c in layer1.__mro__].index(True)]
    assert (
        layer2(**kwargs).__sklearn_tags__().array_api_support
        == parent(**kwargs).__sklearn_tags__().array_api_support
    )


def test_under_array_api_dispatch_patched_logistic_hands_off(monkeypatch, data):
    """What scikit-learn serves in the input's namespace goes back to it.

    Stock LogisticRegression advertises array-API support from scikit-learn 1.9
    on; before that there is nothing to hand back, and the patched class keeps
    the work, exactly as it does for SVC and the neighbors on every version.
    """
    from sklearn.linear_model import LogisticRegression as SkLogisticRegression

    X, y = data
    stock_supports = SkLogisticRegression().__sklearn_tags__().array_api_support
    monkeypatch.setattr(base_module, "_array_api_dispatch_enabled", lambda: True)
    with tuning_context(**NO_CROSSOVER), warnings.catch_warnings():
        warnings.simplefilter("ignore", MLXLearnFallbackWarning)
        patched_model = patched.LogisticRegression().fit(X, y)
        direct = LogisticRegression().fit(X, y)
    if stock_supports:
        assert patched_model.execution_backend_ == "sklearn"
        assert patched_model.fit_event_.reason == "array-api-dispatch"
    else:
        assert patched_model.execution_backend_ != "sklearn"
        assert patched_model.fit_event_.reason != "array-api-dispatch"
    assert direct.execution_backend_ != "sklearn"


# ======================================================================================
# device="cpu" and the crossover
# ======================================================================================


@pytest.mark.parametrize("pair", ESTIMATORS, ids=IDS)
def test_patched_small_problems_go_to_sklearn_on_the_cpu_device_too(data, pair):
    """device='cpu' skipped the crossover, so patched small fits ran float32 MLX-CPU."""
    X, y = data
    with mlxlearn.config_context(device="cpu"):
        model = _estimator(pair, "L2").fit(X, y)
    assert model.execution_backend_ == "sklearn"


@pytest.mark.parametrize("pair", ESTIMATORS, ids=IDS)
def test_patched_small_problems_go_to_sklearn_without_a_gpu(monkeypatch, data, pair):
    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    X, y = data
    model = _estimator(pair, "L2").fit(X, y)
    assert model.execution_backend_ == "sklearn"


@pytest.mark.parametrize("as_list", [False, True], ids=["ndarray", "list"])
@pytest.mark.parametrize("pair", ESTIMATORS, ids=IDS)
def test_an_explicit_gpu_device_does_not_override_the_crossover(monkeypatch, data, pair, as_list):
    """Documented: device picks the MLX device once dispatch has chosen MLX.

    On a host with no usable GPU, a directly imported SVC resolved the device
    before the size decision and raised DeviceError on a problem it runs on MLX's
    CPU; so did every estimator for input with no shape before validation.
    """
    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    X, y = data
    if as_list:
        X, y = X.tolist(), y.tolist()
    with mlxlearn.config_context(device="gpu"):
        model = _without_device_error(lambda: _estimator(pair, "L1").fit(X, y))
        assert (model.execution_backend_, model.fit_event_.reason) == ("cpu", "below-crossover")
        patched_model = _without_device_error(lambda: _estimator(pair, "L2").fit(X, y))
        assert patched_model.execution_backend_ == "sklearn"
        # Above the crossover the device is named and missing: refused, not substituted.
        with tuning_context(**NO_CROSSOVER, svc_min_pair_work=0):
            with pytest.raises(DeviceError):
                _estimator(pair, "L1").fit(X, y)


def test_the_linear_kernel_never_needs_the_gpu(monkeypatch, data):
    """Always below the crossover, so it raised DeviceError at every size."""
    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    X, y = data
    with mlxlearn.config_context(device="gpu"), tuning_context(**NO_CROSSOVER):
        model = _without_device_error(lambda: SVC(kernel="linear").fit(X, y))
    assert model.execution_backend_ == "cpu"


@pytest.mark.parametrize("pair", ESTIMATORS, ids=IDS)
def test_forcing_scikit_learn_outranks_the_data_checks(data, pair):
    """MLXLEARN_FORCE_BACKEND comes before the data checks, as it does the size rules.

    The neighbor estimators and LogisticRegression ran their float32 range scan
    as a capability check, ahead of it: under fallback_policy="raise", a fit forced
    onto scikit-learn — which takes the data — raised for a limit of the path it
    was not taking. SVC already skipped the scan.
    """
    X, y = data
    with tuning_context(force_backend="sklearn"), mlxlearn.config_context(fallback_policy="raise"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # scikit-learn's own overflow
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = _estimator(pair, "L2").fit(_overflowing(X), y)
    assert (model.execution_backend_, model.fit_event_.reason) == ("sklearn", "forced")


def test_the_raw_shape_crossover_decides_before_validation(monkeypatch, data):
    """A patched fit headed for scikit-learn pays for none of mlxlearn's preparation."""
    X, y = data
    model = patched.LogisticRegression()

    def data_check():
        raise AssertionError("data checks must not run for work going to scikit-learn")

    backend = model._select_backend(
        "fit",
        shape=X.shape,
        crossover=model._below_crossover,
        data_check=data_check,
    )
    assert backend == "sklearn"
    model._set_backend(backend)
    assert model.fit_event_.reason == "below-crossover"


def test_data_checks_still_refuse_on_layer1():
    model = LogisticRegression()

    def data_check():
        raise UnsupportedInputError("overflows float32", reason="float32-overflow")

    with pytest.raises(UnsupportedInputError):
        model._select_backend(
            "fit", shape=(10, 2), crossover=model._below_crossover, data_check=data_check
        )


# ======================================================================================
# rng: staged for the estimators that will draw random numbers
# ======================================================================================


def test_resolve_seed_precedence_and_masking():
    from mlxlearn._common.rng import resolve_seed

    assert resolve_seed(5) == 5
    assert resolve_seed(np.int64(2**40 + 3)) == 3
    assert resolve_seed(None) == 0  # deterministic=True: a repeated run repeats
    with mlxlearn.config_context(random_state=11):
        assert resolve_seed(None) == 11
        assert resolve_seed(4) == 4  # the estimator's own random_state wins
    with mlxlearn.config_context(deterministic=False):
        assert len({resolve_seed(None) for _ in range(8)}) > 1
    assert 0 <= resolve_seed(np.random.RandomState(0)) < 2**32
    assert 0 <= resolve_seed(np.random.default_rng(0)) < 2**32
    with pytest.raises(TypeError, match="random_state must be"):
        resolve_seed("7")


def test_derived_seeds_are_independent_substreams():
    from mlxlearn._common.rng import derive_seed, numpy_generator

    assert derive_seed(0, "init") == derive_seed(0, "init")
    assert derive_seed(0, "init") != derive_seed(0, "permute")
    assert derive_seed(0, "init") != derive_seed(1, "init")
    np.testing.assert_array_equal(
        numpy_generator(3, "x").random(4), numpy_generator(3, "x").random(4)
    )
    assert not np.array_equal(numpy_generator(3, "x").random(4), numpy_generator(3, "y").random(4))


def test_mlx_draws_come_from_explicit_keys():
    from mlxlearn._common.rng import mlx_uniform

    first = np.asarray(mlx_uniform((6,), 5, "u"))
    np.testing.assert_array_equal(first, np.asarray(mlx_uniform((6,), 5, "u")))
    assert not np.array_equal(first, np.asarray(mlx_uniform((6,), 5, "v")))
    assert ((first >= 0) & (first < 1)).all()
