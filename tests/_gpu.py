"""The suite's GPU policy, as pytest hooks. ``tests/conftest.py`` re-exports them.

On a host without a working Metal device — a VM, a CPU-only MLX build — tests that
need the GPU are *skipped*, with a reason, instead of failing by the dozen with one
cause buried in every traceback. Two kinds of test count as needing it:

* tests marked ``@pytest.mark.gpu``, which say so up front;
* tests that fail with :class:`~mlxlearn.exceptions.DeviceError`, which is exactly
  mlxlearn's statement that the GPU they asked for is not there. It catches a test
  whose forced MLX fit names the GPU device, without a marker on each — but only
  when the error reaches pytest unwrapped, not inside GridSearchCV's "all fits
  failed". Any *other* failure on such a host still fails.

A test that asserts the MLX path ran needs the marker: without a GPU nothing raises,
because auto-selection picks the CPU and a forced MLX fit of every estimator runs on
MLX's CPU device. ``tests/tooling/test_gpu_policy.py`` runs the suite on a simulated
GPU-less host to hold every test to this.

A skip is the wrong answer where the GPU is the point, as on CI. There,
``MLXLEARN_REQUIRE_GPU=1`` turns a missing GPU into an immediate, single, loud
session failure instead, so a runner image that loses Metal cannot quietly skip
its way to green. The variable is read only by the test suite and the CI tooling,
never by the library.
"""

from __future__ import annotations

import os

import pytest

from mlxlearn._common import device
from mlxlearn.exceptions import DeviceError

REQUIRE_GPU_ENV = "MLXLEARN_REQUIRE_GPU"


def gpu_required() -> bool:
    return os.environ.get(REQUIRE_GPU_ENV, "").strip() not in ("", "0")


def gpu_present() -> bool:
    return device.gpu_available()


def _skip_reason(detail: str = "") -> str:
    reason = f"needs a working MLX GPU device; set {REQUIRE_GPU_ENV}=1 to fail instead"
    return f"{reason} ({detail})" if detail else reason


def pytest_sessionstart(session):
    if gpu_required() and not gpu_present():
        pytest.exit(
            f"{REQUIRE_GPU_ENV}=1 is set, but no working MLX GPU device was found "
            f"({device.device_info()}). Refusing to run a suite whose GPU tests would "
            "all be skipped.",
            returncode=pytest.ExitCode.USAGE_ERROR,
        )


def pytest_collection_modifyitems(config, items):
    marked = [item for item in items if item.get_closest_marker("gpu") is not None]
    if not marked or gpu_present():
        return
    skip = pytest.mark.skip(reason=_skip_reason())
    for item in marked:
        item.add_marker(skip)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (
        report.failed
        and call.excinfo is not None
        and call.excinfo.errisinstance(DeviceError)
        and not gpu_required()
        and not gpu_present()
    ):
        report.outcome = "skipped"
        path, lineno, _ = item.location
        report.longrepr = (path, (lineno or 0) + 1, f"Skipped: {_skip_reason(str(call.excinfo.value))}")
