"""The GPU policy (tests/_gpu.py) on a host without a working GPU.

Only a marker or a :class:`~mlxlearn.exceptions.DeviceError` makes a test skip there.
A test that asserts the MLX path ran gets neither: auto-selection quietly picks the
CPU, and a forced MLX fit of every estimator runs on MLX's CPU device. Such a test
has to carry the ``gpu`` marker, or it fails on exactly the hosts the skip exists for.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from ._load import REPO

# The library and the policy both read the answer from this one probe, so replacing
# it is how a host without Metal looks to both.
_NO_GPU_PLUGIN = """
import functools

from mlxlearn._common import device


@functools.lru_cache(maxsize=1)
def _no_gpu():
    return False


device.gpu_available = _no_gpu
"""


@pytest.mark.slow
def test_the_default_suite_skips_rather_than_fails_without_a_gpu(tmp_path):
    (tmp_path / "_no_gpu_host.py").write_text(_NO_GPU_PLUGIN, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "MLXLEARN_REQUIRE_GPU"}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    # Process-parallel workers start without the plugin and see the real GPU, so
    # there the simulation, not the policy, is what would fail.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "_no_gpu_host", "-p", "no:cacheprovider",
         "-k", "not process_parallel"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = proc.stdout.strip().splitlines()
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "skipped" in lines[-1], lines[-1]
