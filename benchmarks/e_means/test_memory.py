"""CPU-only checks that phase memory peaks cannot become lifetime peaks."""

from __future__ import annotations

import ctypes
import errno

import pytest
from benchmarks.e_means import memory


def observation(t, rss, footprint, *, lifetime=900, incarnation=123):
    return {
        "supported": True,
        "source": "proc_pid_rusage",
        "pid": 1234,
        "rusage_flavor": 4,
        "monotonic_ns": t,
        "collection_duration_ns": 1000,
        "rss_bytes": rss,
        "physical_footprint_bytes": footprint,
        "lifetime_peak_physical_footprint_bytes": lifetime,
        "process_start_abstime": incarnation,
        "process_exit_abstime": 0,
    }


def test_fit_peak_uses_current_samples_after_larger_warmup(monkeypatch):
    readings = iter([
        observation(0, 100, 80),
        observation(10_000_000, 300, 250),
        observation(20_000_000, 120, 90),
    ])
    monkeypatch.setattr(memory, "snapshot", lambda pid: next(readings))
    # Poll manually while the background thread waits. stop() wakes it
    # immediately, so this test neither sleeps nor depends on scheduling.
    sampler = memory.MemorySampler(1234, interval_seconds=60, keep_samples=True).start()
    try:
        assert sampler._observe()
    finally:
        result = sampler.stop()
    assert result["complete"]
    assert result["sample_count"] == 3
    assert result["peak_rss_bytes"] == 300
    assert result["peak_physical_footprint_bytes"] == 250
    assert result["rss_peak_increase_bytes"] == 200
    assert result["physical_footprint_peak_increase_bytes"] == 170
    assert result["lifetime_peak_physical_footprint_bytes"] == 900
    assert result["end"]["physical_footprint_bytes"] == 90
    assert result["duration_seconds"] == pytest.approx(0.02)
    assert result["maximum_sample_gap_seconds"] == pytest.approx(0.01)
    assert len(result["samples"]) == 3
    assert sampler.stop() is result


def test_reused_pid_cannot_contaminate_phase_peak(monkeypatch):
    calls = 0

    def read(pid):
        nonlocal calls
        calls += 1
        return observation(calls * 10_000_000, 100 if calls == 1 else 10_000,
                           80 if calls == 1 else 9000,
                           incarnation=123 if calls == 1 else 456)

    monkeypatch.setattr(memory, "snapshot", read)
    sampler = memory.MemorySampler(1234, interval_seconds=60).start()
    try:
        assert not sampler._observe()
    finally:
        result = sampler.stop()
    assert not result["complete"]
    assert result["sample_count"] == 1
    assert result["peak_physical_footprint_bytes"] == 80
    assert result["peak_rss_bytes"] == 100
    assert all(error["errno"] == errno.ESRCH for error in result["sampling_errors"])
    assert "samples" not in result  # default keeps only constant-memory aggregates


def test_exited_process_is_an_error_instead_of_zero_memory(monkeypatch):
    calls = 0

    def read(pid):
        nonlocal calls
        calls += 1
        if calls == 1:
            return observation(0, 100, 80)
        raise ProcessLookupError(errno.ESRCH, "child exited")

    monkeypatch.setattr(memory, "snapshot", read)
    sampler = memory.MemorySampler(1234, interval_seconds=60).start()
    result = sampler.stop()
    assert not result["complete"]
    assert result["end"]["rss_bytes"] == 100
    assert result["end"]["physical_footprint_bytes"] == 80
    assert result["sampling_errors"][0]["errno"] == errno.ESRCH


def test_snapshot_v0_fallback_keeps_missing_lifetime_peak_explicit(monkeypatch):
    flavors = []

    def rusage(pid, flavor, target):
        flavors.append(flavor)
        if flavor == 4:
            ctypes.set_errno(errno.EINVAL)
            return -1
        info = ctypes.cast(target, ctypes.POINTER(memory._RusageInfoV0)).contents
        info.ri_resident_size = 234
        info.ri_phys_footprint = 123
        info.ri_proc_start_abstime = 99
        return 0

    monkeypatch.setattr(memory, "_rusage_function", lambda: rusage)
    result = memory.snapshot(1234)
    assert flavors == [4, 0]
    assert result["rusage_flavor"] == 0
    assert result["rss_bytes"] == 234
    assert result["physical_footprint_bytes"] == 123
    assert result["lifetime_peak_physical_footprint_bytes"] is None
    assert result["process_start_abstime"] == 99


def test_snapshot_permission_error_is_not_unsupported_flavor(monkeypatch):
    flavors = []

    def rusage(pid, flavor, target):
        flavors.append(flavor)
        ctypes.set_errno(errno.EPERM)
        return -1

    monkeypatch.setattr(memory, "_rusage_function", lambda: rusage)
    with pytest.raises(PermissionError) as caught:
        memory.snapshot(1234)
    assert caught.value.errno == errno.EPERM
    assert flavors == [4]
