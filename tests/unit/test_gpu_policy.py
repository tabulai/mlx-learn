"""The suite's GPU policy (tests/_gpu.py), exercised on an inner pytest run."""

from __future__ import annotations

import textwrap

import pytest

from tests import _gpu

_INNER = """
import pytest

from mlxlearn.exceptions import DeviceError


@pytest.mark.gpu
def test_marked():
    pass


def test_reaches_the_gpu_implicitly():
    raise DeviceError("device='gpu' was requested but no usable MLX GPU device is available")


def test_unrelated_failure():
    assert False, "a real failure"
"""


class _Outcomes:
    def __init__(self):
        self.by_test: dict[str, str] = {}

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.outcome != "passed":
            self.by_test[report.nodeid.rsplit("::", 1)[-1]] = report.outcome


def _run_inner(tmp_path, monkeypatch, *, gpu: bool, required: bool):
    monkeypatch.setattr(_gpu, "gpu_present", lambda: gpu)
    if required:
        monkeypatch.setenv(_gpu.REQUIRE_GPU_ENV, "1")
    else:
        monkeypatch.delenv(_gpu.REQUIRE_GPU_ENV, raising=False)
    (tmp_path / "test_inner_gpu_policy.py").write_text(textwrap.dedent(_INNER))
    outcomes = _Outcomes()
    args = [str(tmp_path), "-q", "-p", "no:cacheprovider", "--import-mode=importlib"]
    args += ["-W", "ignore::pytest.PytestUnknownMarkWarning"]
    status = pytest.main(args, plugins=[_gpu, outcomes])
    return status, outcomes.by_test


def test_without_a_gpu_gpu_tests_skip_and_real_failures_still_fail(tmp_path, monkeypatch):
    status, outcomes = _run_inner(tmp_path, monkeypatch, gpu=False, required=False)
    assert outcomes == {
        "test_marked": "skipped",
        "test_reaches_the_gpu_implicitly": "skipped",
        "test_unrelated_failure": "failed",
    }
    assert status == pytest.ExitCode.TESTS_FAILED


def test_requiring_a_gpu_that_is_missing_stops_the_session(tmp_path, monkeypatch):
    status, outcomes = _run_inner(tmp_path, monkeypatch, gpu=False, required=True)
    assert status == pytest.ExitCode.USAGE_ERROR
    assert outcomes == {}


def test_with_a_gpu_nothing_is_skipped_or_excused(tmp_path, monkeypatch):
    status, outcomes = _run_inner(tmp_path, monkeypatch, gpu=True, required=True)
    assert outcomes == {
        "test_marked": "passed",
        "test_reaches_the_gpu_implicitly": "failed",
        "test_unrelated_failure": "failed",
    }
    assert status == pytest.ExitCode.TESTS_FAILED


@pytest.mark.parametrize(("value", "required"), [("", False), ("0", False), ("1", True), ("yes", True)])
def test_the_requirement_variable(monkeypatch, value, required):
    monkeypatch.setenv(_gpu.REQUIRE_GPU_ENV, value)
    assert _gpu.gpu_required() is required


def test_the_probe_reports_a_broken_metal_device_as_no_gpu(monkeypatch):
    # The policy is only as honest as gpu_available(). Asserting it returns a bool
    # says nothing; this makes GPU evaluation fail and checks the probe notices.
    # Evaluation, not mx.stream: the probe runs on explicit streams so that it never
    # touches MLX's process-global default device, and any probe has to evaluate.
    import mlx.core as mx

    from mlxlearn._common.device import gpu_available, reset_device_cache

    def broken_eval(*args, **kwargs):
        raise RuntimeError("Metal device unavailable")

    reset_device_cache()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(mx, "eval", broken_eval)
            assert gpu_available() is False
    finally:
        reset_device_cache()  # the next caller re-probes the real device
