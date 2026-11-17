"""CPU-only checks for the integer-pass, fresh-final-label benchmark contract."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks/e_means"
sys.path.insert(0, str(BENCH))
import kmeans_compare as bench  # noqa: E402


def evidence(mode="metal-kmeans"):
    metadata = {
        "train_sample_sha256": "train",
        "test_sha256": "test",
        "train_rows_used": 100,
        "test_rows_used": 10,
        "dimension": 3,
    }
    entry = {
        "dataset": "example",
        "n": 100,
        "d": 3,
        "k": 4,
        "epochs": 2,
        "seed_ids": [10],
        "dataset_identity": metadata,
    }
    config = {
        "dataset": "example",
        "mode": mode,
        "seed": 10,
        "epochs": 2,
        "rows": None,
        "k_override": None,
        "epochs_override": None,
        "diagnostic_subset": False,
        "precision": "float32",
        "n_init": 1,
        "tol": 0.0,
        "final_labels": True,
        "warmup": True,
        "threads": 8,
        "train_score_rows": 10000,
        "memory": True,
        "memory_interval_seconds": 0.01,
        "protocol_config": entry,
        "expected_source_sha256": {"driver": "source"},
        "expected_reference_source_sha256": {"reference": "source"},
        "expected_sklearn_sha256": {"sklearn": "source"},
        "expected_versions": {"python": "test"},
        "expected_host": {"cpu": "test"},
        "output": "/tmp/result.json",
        "worker_output": "/tmp/result.tmp",
    }
    recipe = {
        "nepochs": 2,
        "batch_law": "quality",
        "epoch_rows": [100, 100],
        "seed": 10,
        "precision": "float32",
        "finalization": "last_epoch",
        "assignment": "auto",
    }
    return {
        "status": "completed",
        "kind": bench.KIND,
        "config": config,
        "source_sha256": config["expected_source_sha256"],
        "reference_source_sha256": config["expected_reference_source_sha256"],
        "sklearn_source_sha256": config["expected_sklearn_sha256"],
        "versions": config["expected_versions"],
        "hardware": config["expected_host"],
        "dataset": dict(metadata),
        "n": 100,
        "d": 3,
        "k": 4,
        "train_seconds": 3.0,
        "train_seconds_samples": [3.0],
        "memory_samples": [{"complete": True}],
        "actual_iterations": 2,
        "final_labels": {
            "shape": [100],
            "dtype": "int32",
            "semantics": "nearest final centroids; no update after final assignment",
        },
        "quality_fit_index": 0,
        "test_mse": 1.0,
        "train_sample_mse": 0.8,
        "final_inertia": 80.0,
        "label_sample_mse": 0.8,
        "label_sample_agreement": 1.0,
        "train_score_rows": 100,
        "initial_centroids_sha256": "a" * 64,
        "final_centroids_sha256": "b" * 64,
        "final_labels_sha256": "c" * 64,
        "train_score_sample_sha256": "d" * 64,
        "full_host_input_released": mode.startswith("metal-"),
        "effective_omp_threads": 8,
        "sklearn_effective_threads": 8 if mode == "sklearn-kmeans" else None,
        "thread_environment": {"OMP_NUM_THREADS": "8", "VECLIB_MAXIMUM_THREADS": None},
        "recipe_resolved": recipe if mode.endswith("emeans") else None,
        "iteration_history": [{"epoch": 1}, {"epoch": 2}] if mode.endswith("emeans") else [],
    }


@pytest.mark.parametrize("mode", bench.MODES)
def test_four_native_fp32_modes_validate(mode):
    row = evidence(mode)
    bench.validate_result(row, row["config"], row["memory_samples"])


@pytest.mark.parametrize("mode", ["metal-kmeans", "sklearn-kmeans"])
def test_lloyd_can_stop_on_strict_convergence_before_cap(mode):
    row = evidence(mode)
    row["actual_iterations"] = 1
    bench.validate_result(row, row["config"], row["memory_samples"])


@pytest.mark.parametrize(
    "change",
    [
        "n_init",
        "tol",
        "capture",
        "warmup",
        "override",
        "identity",
        "precision",
        "iterations",
        "online_labels",
        "label_hash",
        "label_quality",
        "quality_fit",
        "sample_count",
        "release",
        "threads",
        "sklearn_source",
        "label_agreement",
    ],
)
def test_changed_protocol_or_final_output_rejected(change):
    row = evidence()
    c = row["config"]
    if change == "n_init":
        c["n_init"] = 2
    elif change == "tol":
        c["tol"] = 1e-4
    elif change == "capture":
        c["final_labels"] = False
    elif change == "warmup":
        c["warmup"] = False
    elif change == "override":
        c["epochs_override"] = 2
    elif change == "identity":
        row["dataset"]["test_sha256"] = "different"
    elif change == "precision":
        c["precision"] = "bfloat16"
    elif change == "iterations":
        row["actual_iterations"] = 3
    elif change == "online_labels":
        row["final_labels"]["semantics"] = "online moving labels"
    elif change == "label_hash":
        row["final_labels_sha256"] = ""
    elif change == "label_quality":
        row["label_sample_mse"] = 1.1
    elif change == "quality_fit":
        row["quality_fit_index"] = 1
    elif change == "sample_count":
        row["train_score_rows"] = 99
    elif change == "release":
        row["full_host_input_released"] = False
    elif change == "threads":
        row["thread_environment"]["VECLIB_MAXIMUM_THREADS"] = "1"
    elif change == "sklearn_source":
        row["sklearn_source_sha256"] = {}
    elif change == "label_agreement":
        row["label_sample_agreement"] = float("nan")
    with pytest.raises(RuntimeError):
        bench.validate_result(row, c, row["memory_samples"])


@pytest.mark.parametrize("mode", ["metal-emeans", "faiss-emeans"])
def test_emeans_cannot_report_early_stop_or_fractional_pass_history(mode):
    for mutate in (
        lambda r: r.update(actual_iterations=1),
        lambda r: r["recipe_resolved"].update(epoch_rows=[100, 60]),
        lambda r: r.update(iteration_history=[{}]),
    ):
        row = evidence(mode)
        mutate(row)
        with pytest.raises(RuntimeError, match="pass budget"):
            bench.validate_result(row, row["config"], row["memory_samples"])


def test_label_ties_may_disagree_if_selected_residual_quality_still_matches():
    row = evidence()
    row["label_sample_agreement"] = 0.5
    bench.validate_result(row, row["config"], row["memory_samples"])


@pytest.fixture
def child(monkeypatch, tmp_path):
    import memory

    row = evidence()
    row["config"].update(
        output=str(tmp_path / "result.json"), worker_output=str(tmp_path / "worker.tmp")
    )
    processes = []
    original = subprocess.Popen

    class Sampler:
        def __init__(self, pid, interval_seconds):
            pass

        def start(self):
            return self

        def stop(self):
            return {"complete": True}

    monkeypatch.setattr(memory, "MemorySampler", Sampler)

    def install(program):
        def launch(command, **kwargs):
            process = original(
                [sys.executable, "-c", program, command[-1], json.dumps(row)], **kwargs
            )
            processes.append(process)
            return process

        monkeypatch.setattr(bench.subprocess, "Popen", launch)

    return row, processes, install


def program(seconds, count, mutation=""):
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
def test_adaptive_timed_call_rule_and_final_output_validation(child, seconds, count):
    row, processes, install = child
    install(program(seconds, count))
    bench.measured_run(row["config"], 2)
    saved = json.loads(Path(row["config"]["output"]).read_text())
    assert len(saved["memory_samples"]) == count and processes[0].poll() == 0


@pytest.mark.parametrize("failure", ["timeout", "online_labels", "short_repetition", "provenance"])
def test_controller_failure_cleans_child_and_does_not_publish(child, failure):
    row, processes, install = child
    scripts = {
        "timeout": "import time;time.sleep(30)",
        "online_labels": program(3.0, 1, "r['final_labels']['semantics']='online'"),
        "short_repetition": program(0.1, 1),
        "provenance": program(3.0, 1, "r['sklearn_source_sha256']={}"),
    }
    install(scripts[failure])
    Path(row["config"]["worker_output"]).write_text("{}")
    with pytest.raises((RuntimeError, TimeoutError)):
        bench.measured_run(row["config"], 0.15 if failure == "timeout" else 2)
    assert processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed
    assert not Path(row["config"]["output"]).exists()


@pytest.mark.parametrize("timeout", ["nan", "inf", "-1", "0"])
def test_cli_rejects_nonfinite_or_nonpositive_timeout(monkeypatch, timeout):
    monkeypatch.setattr(sys, "argv", [bench.__file__, "--timeout", timeout])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


def test_default_cli_four_modes_five_datasets_integer_budgets(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(bench, "sources", lambda _: {"driver": "hash"})
    monkeypatch.setattr(bench.previous, "reference_sources", lambda _: {"reference": "hash"})
    monkeypatch.setattr(bench, "sklearn_sources", lambda: {"sklearn": "hash"})
    monkeypatch.setattr(bench.base, "package_versions", lambda: {"python": "test"})
    monkeypatch.setattr(bench.base, "host_identity", lambda: {"cpu": "test"})
    monkeypatch.setattr(
        bench, "measured_run", lambda config, timeout: seen.append(copy.deepcopy(config))
    )
    monkeypatch.setattr(sys, "argv", [bench.__file__, "--out", str(tmp_path)])
    bench.main()
    assert len(seen) == 20 and {c["mode"] for c in seen} == set(bench.MODES)
    assert [seen[i]["epochs"] for i in range(0, 20, 4)] == [2, 2, 2, 2, 3]
    assert all(
        c["precision"] == "float32"
        and c["warmup"]
        and c["final_labels"]
        and not c["diagnostic_subset"]
        for c in seen
    )
