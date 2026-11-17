"""CPU-only protocol, process cleanup, provenance and matched-precision checks."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks/e_means"
sys.path.insert(0, str(BENCH))
import large_compare as large  # noqa: E402


def evidence(mode="metal-fp32"):
    metadata = {
        "train_sample_sha256": "train",
        "test_sha256": "test",
        "train_rows_used": 100,
        "test_rows_used": 10,
        "dimension": 9,
    }
    recipe = {
        "batch_law": "quality",
        "batch_size": 32,
        "alpha0": 0.2,
        "nepochs": 2.0,
        "epoch_rows": [100, 100],
        "epoch_steps": [4, 4],
        "total_steps": 8,
        "alpha_min_factor": 0.1,
        "init_usage": 2**-16,
        "split_epsilon": 1e-4,
        "unused_threshold": 0.1,
        "check_unused_every": 5,
        "tail_policy": "keep",
        "seed": 10,
        "precision": "bfloat16" if mode == "metal-bf16" else "float32",
        "finalization": "last_epoch",
    }
    entry = {
        "dataset": "example",
        "n": 100,
        "d": 9,
        "k": 3,
        "epochs": 2.0,
        "seed_ids": [10],
        "dataset_identity": metadata,
        "recipe_expected": large.algorithm_comparison(recipe),
    }
    config = {
        "dataset": "example",
        "mode": mode,
        "seed": 10,
        "epochs": 2.0,
        "epochs_override": None,
        "rows": None,
        "k_override": None,
        "diagnostic_subset": False,
        "precision": recipe["precision"],
        "recipe": "faiss",
        "batch_law": "quality",
        "finalization": "last_epoch",
        "assignment": "auto",
        "capture_assignments": True,
        "warmup": True,
        "threads": 8,
        "train_score_rows": 10,
        "memory": True,
        "memory_interval_seconds": 0.01,
        "protocol_config": entry,
        "expected_source_sha256": {"driver": "hash"},
        "expected_reference_source_sha256": {"upstream": "hash"},
        "expected_versions": {"python": "test"},
        "expected_host": {"cpu": "test"},
        "output": "/tmp/example.json",
        "worker_output": "/tmp/example.tmp",
    }
    return {
        "status": "completed",
        "kind": large.KIND,
        "config": config,
        "dataset": dict(metadata),
        "n": 100,
        "d": 9,
        "k": 3,
        "train_seconds": 3.0,
        "train_seconds_samples": [3.0],
        "quality_fit_index": 0,
        "test_mse": 1.0,
        "train_score_sample_sha256": "sample",
        "initial_centroids_sha256": "initial",
        "final_centroids_sha256": "final",
        "recipe_resolved": recipe,
        "algorithm_comparison": large.algorithm_comparison(recipe),
        "iteration_history": [{"rows": 100, "steps": 4}, {"rows": 100, "steps": 8}],
        "n_steps": 8,
        "captured_assignments": {"shape": [100], "dtype": "int64"},
        "memory_samples": [{"complete": True}],
        "source_sha256": config["expected_source_sha256"],
        "reference_source_sha256": config["expected_reference_source_sha256"],
        "faiss_reference_revision": large.base.FAISS_REVISION,
        "versions": config["expected_versions"],
        "hardware": config["expected_host"],
        "effective_omp_threads": 8,
        "thread_environment": {"OMP_NUM_THREADS": "8", "VECLIB_MAXIMUM_THREADS": None},
        "full_host_input_released": mode != "faiss-fp32",
        "memory_policy": "same policy",
        "timing_scope": "same timing",
        "scoring": "same scoring",
        "memory_scope": "same window",
    }


@pytest.mark.parametrize("mode", large.MODES)
def test_complete_modes_validate(mode):
    row = evidence(mode)
    large.validate_result(row, row["config"], row["memory_samples"])


def test_only_matched_fp32_modes_pair_and_recipe_float_roundoff_is_allowed():
    metal, cpu, supplement = evidence(), evidence("faiss-fp32"), evidence("metal-bf16")
    cpu["train_seconds"] = 12.0
    cpu["train_seconds_samples"] = [12.0]
    cpu["recipe_resolved"]["alpha0"] += 1e-15
    cpu["algorithm_comparison"]["alpha0"] += 1e-15
    pairs, unmatched = large.pair_results([metal, cpu, supplement])
    assert len(pairs) == 1 and pairs[0]["speedup"] == 4.0
    assert unmatched == [supplement]
    with pytest.raises(RuntimeError, match="Duplicate"):
        large.pair_results([metal, copy.deepcopy(metal)])


@pytest.mark.parametrize(
    "change",
    [
        "kind",
        "source",
        "reference",
        "dataset",
        "identity",
        "missing_identity",
        "epoch",
        "seed",
        "batch",
        "alpha",
        "history",
        "steps",
        "labels",
        "release",
        "threads",
        "precision",
        "warmup",
    ],
)
def test_mismatched_evidence_rejected(change):
    row = evidence()
    c = row["config"]
    if change == "kind":
        row["kind"] = "historical"
    elif change == "source":
        row["source_sha256"] = {"changed": "source"}
    elif change == "reference":
        row["reference_source_sha256"] = {"changed": "reference"}
    elif change == "dataset":
        c["dataset"] = "different"
    elif change == "identity":
        row["dataset"]["test_sha256"] = "wrong"
    elif change == "missing_identity":
        del c["protocol_config"]["dataset_identity"]
    elif change == "epoch":
        c["epochs"] = 3
    elif change == "seed":
        c["seed"] = 99
    elif change in {"batch", "alpha"}:
        key = "batch_size" if change == "batch" else "alpha0"
        row["recipe_resolved"][key] *= 2
        row["algorithm_comparison"][key] *= 2
    elif change == "history":
        row["iteration_history"][0]["rows"] = 99
    elif change == "steps":
        row["n_steps"] = 7
    elif change == "labels":
        row["captured_assignments"]["dtype"] = "int32"
    elif change == "release":
        row["full_host_input_released"] = False
    elif change == "threads":
        row["thread_environment"]["VECLIB_MAXIMUM_THREADS"] = "1"
    elif change == "precision":
        c["precision"] = "bfloat16"
    elif change == "warmup":
        c["warmup"] = False
    with pytest.raises(RuntimeError):
        large.validate_result(row, c, row["memory_samples"])


def test_different_initialization_or_scoring_sample_cannot_pair():
    for field in ("initial_centroids_sha256", "train_score_sample_sha256", "timing_scope"):
        metal, cpu = evidence(), evidence("faiss-fp32")
        cpu[field] = "different"
        paired, unpaired = large.pair_results([metal, cpu])
        assert not paired and len(unpaired) == 2


@pytest.fixture
def fake_controller(monkeypatch, tmp_path):
    import memory

    row = evidence()
    row["config"].update(
        output=str(tmp_path / "result.json"), worker_output=str(tmp_path / "worker.tmp")
    )
    state = {"start_error": False, "stop_error": False, "complete": True, "starts": 0, "stops": 0}
    processes = []
    original = subprocess.Popen

    class Sampler:
        def __init__(self, pid, interval_seconds):
            pass

        def start(self):
            state["starts"] += 1
            if state["start_error"]:
                raise OSError("start failed")
            return self

        def stop(self):
            state["stops"] += 1
            if state["stop_error"]:
                raise RuntimeError("stop failed")
            return {"complete": state["complete"]}

    monkeypatch.setattr(memory, "MemorySampler", Sampler)

    def install(script):
        def launch(command, **kwargs):
            assert Path(command[1]) == Path(large.__file__)
            assert kwargs["env"]["OMP_NUM_THREADS"] == "8"
            assert "VECLIB_MAXIMUM_THREADS" not in kwargs["env"]
            process = original(
                [sys.executable, "-c", script, command[-1], json.dumps(row)], **kwargs
            )
            processes.append(process)
            return process

        monkeypatch.setattr(large.subprocess, "Popen", launch)

    return row, state, processes, install


def worker_program(seconds=3.0, count=1, mutation=""):
    return f"""
import json,sys
from pathlib import Path
c=json.loads(sys.argv[1]);r=json.loads(sys.argv[2]);r['config']=c
for i in range({count}):
 print(json.dumps({{'phase':'FIT_READY','call':i}}),flush=True)
 assert sys.stdin.readline().strip()=='GO'
 print(json.dumps({{'phase':'FIT_DONE','call':i,'train_seconds':{seconds}}}),flush=True)
 assert sys.stdin.readline().strip()=='STOP_ACK'
r['train_seconds']={seconds};r['train_seconds_samples']=[{seconds}]*{count}
{mutation}
Path(c['worker_output']).write_text(json.dumps(r))
"""


@pytest.mark.parametrize("seconds,count", [(3.0, 1), (0.1, 3), (2.0, 1)])
def test_controller_adaptive_calls_and_memory_windows(fake_controller, seconds, count):
    row, state, processes, install = fake_controller
    install(worker_program(seconds, count))
    large.measured_run(row["config"], 2)
    output = json.loads(Path(row["config"]["output"]).read_text())
    assert len(output["memory_samples"]) == count
    assert state["starts"] == state["stops"] == count and processes[0].poll() == 0


@pytest.mark.parametrize(
    "failure", ["crash", "timeout", "start", "stop", "incomplete", "source", "adaptive", "order"]
)
def test_controller_failures_cleanup_child_and_never_publish(fake_controller, failure):
    row, state, processes, install = fake_controller
    script = worker_program()
    if failure == "crash":
        script = "raise RuntimeError('crash')"
    elif failure == "timeout":
        script = "import time;time.sleep(30)"
    elif failure in {"start", "stop"}:
        state[failure + "_error"] = True
    elif failure == "incomplete":
        state["complete"] = False
    elif failure == "source":
        script = worker_program(mutation="r['reference_source_sha256']={}")
    elif failure == "adaptive":
        script = worker_program(0.1, 1)
    elif failure == "order":
        script = worker_program().replace("'call':i", "'call':i+1")
    install(script)
    Path(row["config"]["worker_output"]).write_text('{"status":"completed"}')
    with pytest.raises((RuntimeError, OSError, TimeoutError)):
        large.measured_run(row["config"], 0.15 if failure == "timeout" else 2)
    assert processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed
    assert not Path(row["config"]["output"]).exists()


@pytest.mark.parametrize("failure", ["construct", "register", "close"])
def test_selector_failures_always_cleanup_child(fake_controller, monkeypatch, failure):
    row, _, processes, install = fake_controller
    install(worker_program())
    original = large.selectors.DefaultSelector

    class Selector:
        def __init__(self):
            if failure == "construct":
                raise RuntimeError("construct failed")
            self.inner = original()

        def register(self, *args):
            if failure == "register":
                raise RuntimeError("register failed")
            return self.inner.register(*args)

        def select(self, *args):
            return self.inner.select(*args)

        def close(self):
            self.inner.close()
            if failure == "close":
                raise RuntimeError("close failed")

    monkeypatch.setattr(large.selectors, "DefaultSelector", Selector)
    with pytest.raises(RuntimeError, match="failed"):
        large.measured_run(row["config"], 2)
    assert processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed


def test_cli_defaults_five_datasets_three_modes_and_marks_diagnostic(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(large, "sources", lambda _: {"driver": "hash"})
    monkeypatch.setattr(large, "reference_sources", lambda _: {"reference": "hash"})
    monkeypatch.setattr(large.base, "package_versions", lambda: {"python": "test"})
    monkeypatch.setattr(large.base, "host_identity", lambda: {"cpu": "test"})
    monkeypatch.setattr(large, "measured_run", lambda config, timeout: observed.append(config))
    monkeypatch.setattr(sys, "argv", [large.__file__, "--out", str(tmp_path)])
    large.main()
    assert len(observed) == 15 and all(not c["diagnostic_subset"] for c in observed)
    assert all(
        c["batch_law"] == "quality" and c["capture_assignments"] and c["warmup"] for c in observed
    )
    assert {c["precision"] for c in observed if c["mode"] in large.PRIMARY_MODES} == {"float32"}
    observed.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            large.__file__,
            "--datasets",
            "deep1b-10M",
            "--rows",
            "100",
            "--k",
            "3",
            "--epochs",
            "1",
            "--out",
            str(tmp_path),
        ],
    )
    large.main()
    assert len(observed) == 3 and all(c["diagnostic_subset"] for c in observed)
