"""Process-controller failures must never publish misleading benchmark rows."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from benchmarks.e_means import compare


@pytest.fixture
def config(tmp_path):
    return {
        "dataset": "example", "method": "metal", "threads": 8, "veclib_threads": None,
        "trial": 0, "memory_interval_seconds": 0.01,
        "output": str(tmp_path / "result.json"), "worker_output": str(tmp_path / "worker.tmp"),
        "expected_source_sha256": {"compare.py": "source"}, "expected_versions": {"mlx": "version"},
        "expected_host": {"cpu": "test-mac"},
    }


@pytest.fixture
def fake_worker(monkeypatch):
    import memory

    processes = []
    state = {"complete": True, "started": 0, "stopped": 0, "start_error": False, "stop_error": False}
    original = subprocess.Popen

    class Sampler:
        def __init__(self, pid, interval_seconds):
            self.pid = pid

        def start(self):
            state["started"] += 1
            if state["start_error"]:
                raise OSError("initial rusage denied")
            return self

        def stop(self):
            state["stopped"] += 1
            if state["stop_error"]:
                raise RuntimeError("sampler stop failed")
            return {"complete": state["complete"], "peak_physical_footprint_bytes": 1024}

    monkeypatch.setattr(memory, "MemorySampler", Sampler)

    def install(program):
        def spawn(command, **kwargs):
            process = original([sys.executable, "-c", program, command[-1]], **kwargs)
            processes.append(process)
            return process

        monkeypatch.setattr(compare.subprocess, "Popen", spawn)
        return state, processes

    return install


_READY = '''
import json, sys, time
from pathlib import Path
config = json.loads(sys.argv[1])
print(json.dumps({'phase': 'FIT_READY', 'n': 20, 'd': 3, 'k': 2}), flush=True)
assert sys.stdin.readline().strip() == 'GO'
'''
_DONE = '''
print(json.dumps({'phase': 'FIT_DONE', 'train_seconds': 0.1}), flush=True)
assert sys.stdin.readline().strip() == 'STOP_ACK'
result = {
    'status': 'completed', 'kind': 'faiss_direct_comparison', 'config': config,
    'source_sha256': config['expected_source_sha256'],
    'versions': config['expected_versions'], 'hardware': config['expected_host'],
    'train_seconds': 0.1, 'test_mse': 1.0,
}
'''
_WRITE = "Path(config['worker_output']).write_text(json.dumps(result))\n"


def test_controller_success_publishes_only_completed_measurement(config, fake_worker):
    state, processes = fake_worker(_READY + _DONE + _WRITE)
    compare.measured_run(config, timeout=2)
    result = json.loads(Path(config["output"]).read_text())
    assert result["memory"]["complete"]
    assert result["source_sha256"] == config["expected_source_sha256"]
    assert not Path(config["worker_output"]).exists()
    assert state["started"] == state["stopped"] == 1
    assert processes[0].poll() == 0


def test_worker_crash_does_not_publish_stale_temporary_result(config, fake_worker):
    Path(config["worker_output"]).write_text('{"status":"completed"}')
    _, processes = fake_worker("import sys; sys.exit(7)")
    with pytest.raises(RuntimeError, match="Worker exited 7"):
        compare.measured_run(config, timeout=2)
    assert not Path(config["output"]).exists()
    assert not Path(config["worker_output"]).exists()
    assert processes[0].poll() == 7


def test_timeout_terminates_child_and_sampler_without_publishing(config, fake_worker):
    state, processes = fake_worker(_READY + "time.sleep(30)\n")
    with pytest.raises(TimeoutError):
        compare.measured_run(config, timeout=0.15)
    assert processes[0].poll() is not None
    assert state["started"] == state["stopped"] == 1
    assert not Path(config["output"]).exists()


def test_incomplete_memory_sampling_is_not_reported_as_a_peak(config, fake_worker):
    state, processes = fake_worker(_READY + _DONE + _WRITE)
    state["complete"] = False
    with pytest.raises(RuntimeError, match="Incomplete fit-window memory"):
        compare.measured_run(config, timeout=2)
    assert processes[0].poll() is not None
    assert not Path(config["output"]).exists()


def test_sampler_start_failure_preserves_error_and_terminates_waiting_child(config, fake_worker):
    state, processes = fake_worker(_READY + _DONE + _WRITE)
    state["start_error"] = True
    with pytest.raises(OSError, match="initial rusage denied"):
        compare.measured_run(config, timeout=2)
    assert state["started"] == 1 and state["stopped"] == 0
    assert processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed
    assert not Path(config["output"]).exists()


def test_sampler_stop_failure_cannot_bypass_child_and_pipe_cleanup(config, fake_worker):
    state, processes = fake_worker(_READY + _DONE + _WRITE)
    state["stop_error"] = True
    with pytest.raises(RuntimeError, match="sampler stop failed"):
        compare.measured_run(config, timeout=2)
    assert state["started"] == 1 and state["stopped"] >= 1
    assert processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed
    assert not Path(config["output"]).exists()


@pytest.mark.parametrize("field", ["source_sha256", "versions", "hardware", "config"])
def test_controller_rejects_mismatched_worker_provenance(config, fake_worker, field):
    fake_worker(_READY + _DONE + f"result[{field!r}] = {{'wrong': 'value'}}\n" + _WRITE)
    with pytest.raises(RuntimeError, match="provenance"):
        compare.measured_run(config, timeout=2)
    assert not Path(config["output"]).exists()
