"""Guard duration-based warmups against unbounded runs and hidden audit failures."""

from __future__ import annotations

import argparse

import pytest
from benchmarks import knn_pipeline, knn_warmed


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_warmup_cli_requires_finite_positive_seconds(value):
    with pytest.raises(argparse.ArgumentTypeError):
        knn_warmed.positive_seconds(value)


def test_warmup_cli_accepts_duration():
    assert knn_warmed.positive_seconds("1.25") == 1.25


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_worker_rejects_unbounded_warmup_before_execution(duration):
    def unexpected(*args):
        pytest.fail("Invalid warmup must not execute a model")

    with pytest.raises(ValueError):
        knn_pipeline.run_block(
            unexpected,
            {
                "repeats": 2,
                "warmup": 0,
                "warmup_seconds": duration,
                "operation": "kneighbors",
                "case": {},
            },
        )


def test_duration_warmup_retains_count_and_failed_audit(monkeypatch):
    clock = [0.0]
    calls = []
    monkeypatch.setattr(knn_pipeline.time, "perf_counter", lambda: clock[0])

    def execute(operation, case):
        calls.append((operation, case))
        clock[0] += 0.25
        return {"ms": 1.0, "audit": {"pass": len(calls) != 1}, "backend": "mlx"}

    result = knn_pipeline.run_block(
        execute,
        {
            "repeats": 3,
            "warmup": 0,
            "warmup_seconds": 1.0,
            "operation": "fit_query",
            "case": {"k": 5},
        },
    )
    assert len(calls) == 7
    assert result["warmup_count"] == 4
    assert result["warmup_elapsed_seconds"] == 1.0
    assert result["warmup_min_seconds"] == 1.0
    assert not result["warmup_quality_pass"]
    assert result["warmup_audits"][0] == {"pass": False}
    assert result["quality_pass"]
    assert len(result["samples"]) == 3
    assert all(sample["backend"] == "mlx" for sample in result["samples"])


def test_existing_count_only_blocks_keep_same_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(knn_pipeline.time, "perf_counter", lambda: 0.0)

    def execute(operation, case):
        calls.append(operation)
        return {"ms": 1.0, "audit": {"pass": True}}

    result = knn_pipeline.run_block(
        execute,
        {
            "repeats": 9,
            "warmup": 3,
            "operation": "kneighbors",
            "case": {},
        },
    )
    assert len(calls) == 12
    assert result["warmup_count"] == 3
    assert result["warmup_min_seconds"] == 0
    assert len(result["samples"]) == 9
