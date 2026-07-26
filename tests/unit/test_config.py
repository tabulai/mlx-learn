"""Configuration: the public surface, its scoping, and its refusals."""

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


def test_config_is_thread_local():
    """A scoped policy must not leak into a thread the caller did not choose."""
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
