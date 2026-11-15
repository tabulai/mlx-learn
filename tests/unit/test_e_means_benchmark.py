"""Benchmark evidence must not silently mix code, hosts, or data splits."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.e_means import run


def _result(method):
    return {
        "status": "completed", "n": 20, "k": 3, "train_seconds": 2 if method == "metal" else 10,
        "test_mse": 1, "initial_centroids_sha256": "same-start",
        "config": {"dataset": "example", "method": method, "precision": "float32", "seed": 10,
                   "assignment": "auto", "epochs": 3, "lloyd_iters": 25},
        "dataset": {"test_sha256": "same-test", "train_sample_sha256": "same-train"},
        "source_sha256": {"run.py": "same-runner", "kernels.py": "same-kernels"},
        "versions": {"mlx": "same-version"}, "hardware": {"cpu": "same-mac"},
    }


@pytest.mark.parametrize("mismatch", [None, "source_sha256", "versions", "hardware", "dataset", "initial_centroids_sha256"])
def test_summary_only_pairs_matching_evidence(tmp_path, capsys, mismatch):
    current, baseline = _result("metal"), _result("metal-lloyd")
    if mismatch is not None:
        baseline[mismatch] = {"different": "value"} if isinstance(baseline[mismatch], dict) else "different"
    (tmp_path / "current.json").write_text(json.dumps(current))
    (tmp_path / "baseline.json").write_text(json.dumps(baseline))
    run.summarize(tmp_path)
    text = capsys.readouterr().out
    assert ("5.00x faster" in text) == (mismatch is None)


def test_campaign_resume_invalidates_changed_source_environment_and_host(tmp_path, monkeypatch):
    state = {"source": {"kernel": "v1"}, "versions": {"mlx": "v1"}, "host": {"cpu": "mac1"}}
    monkeypatch.setattr(run, "source_hashes", lambda: copy.deepcopy(state["source"]))
    monkeypatch.setattr(run, "package_versions", lambda: copy.deepcopy(state["versions"]))
    monkeypatch.setattr(run, "host_identity", lambda: copy.deepcopy(state["host"]))
    calls = []

    def measured_worker(command, **kwargs):
        config = json.loads(command[-1])
        calls.append(config)
        Path(config["output"]).write_text(json.dumps({
            "status": "completed", "source_sha256": config["expected_source_sha256"],
            "versions": config["expected_versions"], "hardware": config["expected_host"],
        }))

    monkeypatch.setattr(run.subprocess, "run", measured_worker)
    args = SimpleNamespace(
        datasets=["example"], suite="core1m", seeds=[10], methods=["metal"], epochs=[3],
        lloyd_iters=25, k=3, rows=20, threads=1, precision="float32", recipe="paper",
        finalization="last_epoch", assignment="auto", batch_size=None, batch_law="auto",
        repeats=1, no_warmup=False, train_score_rows=0, cache=tmp_path / "cache",
        faiss_source=tmp_path / "faiss", out=tmp_path, force=False, timeout=1,
    )
    run.run_campaign(args)
    run.run_campaign(args)
    assert len(calls) == 1
    state["source"]["kernel"] = "v2"
    run.run_campaign(args)
    state["versions"]["mlx"] = "v2"
    run.run_campaign(args)
    state["host"]["cpu"] = "mac2"
    run.run_campaign(args)
    assert len(calls) == 4
    assert len({call["output"] for call in calls}) == 4
