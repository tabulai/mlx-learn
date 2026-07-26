"""Device resolution."""

from __future__ import annotations

import pytest

import mlxlearn
from mlxlearn._common.device import device_info, gpu_available, resolve_device, use_device
from mlxlearn.exceptions import DeviceError


def test_auto_prefers_gpu_when_available():
    assert resolve_device("auto") == ("gpu" if gpu_available() else "cpu")


def test_cpu_is_honored_exactly():
    assert resolve_device("cpu") == "cpu"


def test_explicit_gpu_raises_when_unavailable(monkeypatch):
    """A user who names a device wants that device.

    Quietly computing somewhere else and reporting success makes a benchmark
    meaningless and a reproducibility claim false.
    """
    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    with pytest.raises(DeviceError, match="no usable MLX GPU"):
        resolve_device("gpu")


def test_auto_degrades_quietly(monkeypatch):
    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_mps_is_not_a_device_name():
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("mps")


def test_resolve_reads_configuration():
    with mlxlearn.config_context(device="cpu"):
        assert resolve_device() == "cpu"


def test_use_device_restores_the_previous_stream():
    import mlx.core as mx

    before = mx.default_stream(mx.default_device())
    with use_device("cpu"):
        pass
    assert mx.default_stream(mx.default_device()) == before


def test_use_device_restores_on_exception():
    import mlx.core as mx

    before = mx.default_stream(mx.default_device())
    with pytest.raises(RuntimeError):
        with use_device("cpu"):
            raise RuntimeError("boom")
    assert mx.default_stream(mx.default_device()) == before


def test_device_info_describes_the_environment():
    info = device_info()
    assert info.kind in ("gpu", "cpu")
    assert isinstance(info.gpu_available, bool)
    assert info.mlx_version != ""


def test_gpu_probe_actually_executes_work():
    """Importing mlx successfully is not evidence that Metal works."""
    assert isinstance(gpu_available(), bool)
