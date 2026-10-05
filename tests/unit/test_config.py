"""Configuration: the public surface, its validation, and its refusals.

Scope across threads and processes is covered in ``test_runtime.py``.
"""

from __future__ import annotations

import threading

import pytest

import mlxlearn
from mlxlearn._common.config import (
    PUBLIC_CONFIG_FIELDS,
    Tuning,
    _tuning_from_env,
    get_tuning,
    tuning_context,
)


def test_public_surface_is_exactly_six_options():
    """The plan fixes this list. It is asserted so it cannot grow by accident.

    The ancestor exposed roughly fifty environment variables as tuning surface.
    Promoting tuning knobs to public options freezes today's implementation into
    the API, so crossover thresholds and block sizes stay private.
    """
    assert set(PUBLIC_CONFIG_FIELDS) == {
        "device",
        "fallback_policy",
        "output_type",
        "deterministic",
        "random_state",
        "diagnostics",
    }


def test_defaults():
    config = mlxlearn.get_config()
    assert config.device == "auto"
    assert config.fallback_policy == "warn"
    assert config.output_type == "numpy"
    assert config.deterministic is True
    assert config.diagnostics is True


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"device": "mps"}, "device must be one of"),
        ({"device": "cuda"}, "device must be one of"),
        ({"fallback_policy": "ignore"}, "fallback_policy must be one of"),
        ({"deterministic": "yes"}, "deterministic must be a bool"),
        ({"random_state": 1.5}, "random_state must be an int"),
    ],
)
def test_invalid_values_rejected(kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        mlxlearn.set_config(**kwargs)


def test_mps_is_not_a_device_name():
    """MLX has several backends; 'mps' names a PyTorch one and would age badly."""
    with pytest.raises(ValueError):
        mlxlearn.set_config(device="mps")


def test_unknown_option_is_an_error_not_a_silent_store():
    with pytest.raises(TypeError, match="unknown configuration option"):
        mlxlearn.set_config(allow_approximate=True)


def test_output_type_reserved_for_next_release():
    """0.2.x will implement these. Accepting them now would be a silent lie."""
    for value in ("mlx", "input"):
        with pytest.raises(ValueError, match="reserved for the 0.2.x"):
            mlxlearn.set_config(output_type=value)


def test_config_context_restores_on_exception():
    mlxlearn.set_config(fallback_policy="silent")
    with pytest.raises(RuntimeError):
        with mlxlearn.config_context(fallback_policy="raise"):
            assert mlxlearn.get_config().fallback_policy == "raise"
            raise RuntimeError("boom")
    assert mlxlearn.get_config().fallback_policy == "silent"


def test_config_context_does_not_leak_into_other_threads():
    """A scoped policy must not leak into a thread the caller did not choose.

    set_config, by contrast, is the process-wide default and does reach them;
    see tests/unit/test_runtime.py.
    """
    seen = {}

    def worker():
        seen["value"] = mlxlearn.get_config().fallback_policy

    with mlxlearn.config_context(fallback_policy="raise"):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert seen["value"] == "warn"


def test_config_is_immutable():
    config = mlxlearn.get_config()
    with pytest.raises((AttributeError, TypeError)):
        config.device = "cpu"  # type: ignore[misc]


def test_tuning_is_not_public():
    assert not hasattr(mlxlearn, "set_tuning")
    assert not hasattr(mlxlearn, "Tuning")


def test_tuning_env_override(monkeypatch):
    monkeypatch.setenv("MLXLEARN_KNN_MIN_TRAIN_SAMPLES", "17")
    assert _tuning_from_env(Tuning()).knn_min_train_samples == 17


def test_malformed_env_override_is_an_error(monkeypatch):
    """An env var that silently does nothing is worse than no env var at all."""
    monkeypatch.setenv("MLXLEARN_KNN_MIN_TRAIN_SAMPLES", "lots")
    with pytest.raises(ValueError, match="is not a valid value"):
        _tuning_from_env(Tuning())


def test_tuning_context_scopes():
    original = get_tuning().knn_min_train_samples
    with tuning_context(knn_min_train_samples=1):
        assert get_tuning().knn_min_train_samples == 1
    assert get_tuning().knn_min_train_samples == original


def test_negative_tuning_rejected():
    with pytest.raises(ValueError, match="must be non-negative"):
        Tuning(knn_min_train_samples=-1)


def test_only_mlxlearn_prefixed_env_is_read(monkeypatch):
    """The ancestor's environment surface does not come along under another prefix."""
    monkeypatch.setenv("LEGACY_KNN_MIN_TRAIN_SAMPLES", "3")
    monkeypatch.setenv("KNN_MIN_TRAIN_SAMPLES", "5")
    assert _tuning_from_env(Tuning()).knn_min_train_samples == Tuning().knn_min_train_samples


# ------------------------------------------------------------------ value normalization


def test_numpy_scalars_are_accepted_and_normalized():
    """np.bool_ was rejected with the baffling message 'must be a bool, got bool'."""
    import numpy as np

    config = mlxlearn.set_config(
        deterministic=np.bool_(True), diagnostics=np.bool_(False), random_state=np.int64(3)
    )
    assert config.deterministic is True
    assert config.diagnostics is False
    assert type(config.random_state) is int and config.random_state == 3


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"random_state": -1}, ValueError, r"random_state must be in \[0, 2\*\*32\)"),
        ({"random_state": 2**32}, ValueError, "random_state must be in"),
        ({"random_state": True}, TypeError, "random_state must be an int or None, got True"),
        ({"deterministic": 1}, TypeError, r"deterministic must be a bool, got 1 \(int\)"),
        ({"diagnostics": "yes"}, TypeError, "diagnostics must be a bool, got 'yes'"),
    ],
)
def test_bad_values_are_named_in_the_error(kwargs, error, message):
    with pytest.raises(error, match=message):
        mlxlearn.set_config(**kwargs)


def test_inert_options_say_so():
    """deterministic and random_state have no effect in 0.1.x; the docs must not promise one."""
    doc = " ".join(mlxlearn.Config.__doc__.split())
    assert doc.count("no observable effect") == 2
    assert "it does not seed scikit-learn" in doc


# ---------------------------------------------------------------------- tuning validity


@pytest.mark.parametrize("value", ["gpu", "MLX", "CPU", "auto"])
def test_force_backend_is_validated(value):
    """'gpu' used to be accepted, reported as execution_backend_, and run on the CPU."""
    with pytest.raises(ValueError, match="force_backend must be one of"):
        Tuning(force_backend=value)


def test_force_backend_from_the_environment_is_validated(monkeypatch):
    monkeypatch.setenv("MLXLEARN_FORCE_BACKEND", "gpu")
    with pytest.raises(ValueError, match="MLXLEARN_FORCE_BACKEND"):
        _tuning_from_env(Tuning())


def test_force_backend_is_validated_in_tuning_context_too():
    with pytest.raises(ValueError, match="force_backend"):
        with tuning_context(force_backend="gpu"):
            pass  # pragma: no cover


@pytest.mark.parametrize(
    "name",
    ["lbfgs_history", "lbfgs_max_line_search", "query_block_rows", "query_block_align",
     "distance_block_bytes", "cpu_query_block_rows"],
)
def test_sizes_that_cannot_be_zero_are_rejected_at_configuration_time(name):
    with pytest.raises(ValueError, match=f"{name} must be at least 1"):
        Tuning(**{name: 0})


def test_zero_is_still_meaningful_where_it_means_something():
    """A zero crossover means 'never hand away'; a zero SMO budget means max_iter=0."""
    Tuning(knn_min_train_samples=0, smo_min_iter_budget=0, smo_max_iter_factor=0)


@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan")])
def test_cache_release_fraction_is_a_fraction(value):
    with pytest.raises(ValueError, match="mlx_cache_release_fraction"):
        Tuning(mlx_cache_release_fraction=value)


def test_cache_release_fraction_reads_from_the_environment(monkeypatch):
    monkeypatch.setenv("MLXLEARN_MLX_CACHE_RELEASE_FRACTION", "0")
    assert _tuning_from_env(Tuning()).mlx_cache_release_fraction == 0.0


def test_every_thread_sees_the_same_tuning_default(monkeypatch):
    """The environment was re-read per thread, so threads could disagree mid-process."""
    import threading

    main = get_tuning().knn_min_train_samples
    monkeypatch.setenv("MLXLEARN_KNN_MIN_TRAIN_SAMPLES", "17")
    seen = {}
    thread = threading.Thread(target=lambda: seen.setdefault("v", get_tuning().knn_min_train_samples))
    thread.start()
    thread.join()
    assert seen["v"] == main
    mlxlearn.reset_config()
    assert get_tuning().knn_min_train_samples == 17
