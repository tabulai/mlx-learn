"""Integration evidence must reject source mixing and preserve failed rows."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks import knn_integration as bench


def _entry(ms=2.0):
    sample = {"ms": ms, "sha256": ["distances", "indices"], "audit": {"pass": True}}
    return {"preparation": {"fixture_sha256": ["train", "query"]},
            "warmup": [copy.deepcopy(sample)], "samples": [copy.deepcopy(sample)]}


def test_comparison_retains_failed_warmup_and_suppresses_speedup():
    variants = {"baseline": _entry(4), "current": _entry()}
    variants["current"]["warmup"][0]["audit"]["pass"] = False
    assert not bench.summarize_case(variants, "baseline")
    assert variants["current"]["median_ms"] == 2
    assert variants["current"]["speedup_vs_baseline"] is None
    assert variants["baseline"]["quality_pass"]


@pytest.mark.parametrize("failure", ["fixture", "bits", "oracle", "error", "missing"])
def test_comparison_cannot_turn_bad_results_into_speed_claims(failure):
    variants = {"baseline": _entry(4), "current": _entry()}
    entry = variants["current"]
    if failure == "fixture":
        entry["preparation"]["fixture_sha256"][0] = "other train"
    elif failure == "bits":
        entry["samples"][0]["sha256"][0] = "different distances"
    elif failure == "oracle":
        entry["samples"][0]["audit"]["pass"] = False
    elif failure == "error":
        entry["error"] = "timeout after first sample"
    else:
        entry["samples"] = []
    assert not bench.summarize_case(variants, "baseline")
    assert entry["speedup_vs_baseline"] is None


def test_failed_baseline_suppresses_every_speed_claim():
    variants = {"baseline": _entry(4), "current": _entry()}
    variants["baseline"]["error"] = "failed"
    assert not bench.summarize_case(variants, "baseline")
    assert all(entry["speedup_vs_baseline"] is None for entry in variants.values())


def test_valid_comparison_uses_medians():
    variants = {"baseline": _entry(4), "current": _entry()}
    assert bench.summarize_case(variants, "baseline")
    assert variants["current"]["speedup_vs_baseline"] == 2


def test_setup_failure_leaves_a_failed_report(tmp_path, monkeypatch):
    def fake_snapshot(root, revision, destination):
        destination.mkdir()
        (destination / "source.py").write_text("pass\n")
        return {"files": bench.manifest(destination)}

    class FailedWorker:
        def __init__(self, source, timeout):
            pass

        def request(self, value):
            raise RuntimeError("GPU unavailable during initialization")

        def close(self):
            pass

    monkeypatch.setattr(bench, "snapshot", fake_snapshot)
    monkeypatch.setattr(bench, "Worker", FailedWorker)
    output = tmp_path / "failed.json"
    args = SimpleNamespace(baseline="HEAD", reference=[], cases=["small_certified"],
                           profile="quick", apis=["device"], output=output, seed=0,
                           repeats=3, warmup=2, oracle_rows=1, timeout=1)
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        bench.run(args)
    result = json.loads(output.read_text())
    assert result["quality_pass"] is False
    assert "GPU unavailable" in result["fatal_error"]
    assert (tmp_path / "failed-sources/knn_integration.py").is_file()


def test_exact_oracle_keeps_low_indices_at_a_tied_cutoff():
    train = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [2, 0]], np.float32)
    distances, indices = bench.exact_oracle(train, np.zeros((1, 2), np.float32), 3)
    np.testing.assert_array_equal(indices, [[0, 1, 2]])
    np.testing.assert_array_equal(distances, [[1, 1, 1]])


def test_snapshot_freezes_working_edits_and_omits_extensions(tmp_path):
    root = tmp_path / "repo"
    for name in ("mlxlearn", "mlxlearn_device", "mlxlearn_common"):
        package = root / "src" / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VERSION = 1\n")
        (package / "native.so").write_bytes(b"not portable")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "*.py"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "fixture"], check=True)
    (root / "src/mlxlearn/__init__.py").write_text("VERSION = 2\n")
    current = bench.snapshot(root, None, tmp_path / "current")
    previous = bench.snapshot(root, "HEAD", tmp_path / "previous")
    assert current["sha256"] != previous["sha256"]
    assert not any(path.endswith(".so") for path in current["files"])
    assert (tmp_path / "previous/src/mlxlearn/__init__.py").read_text() == "VERSION = 1\n"
    assert (tmp_path / "current/src/mlxlearn/__init__.py").read_text() == "VERSION = 2\n"
    assert "src/mlxlearn_common/__init__.py" in current["files"]
    assert "src/mlxlearn_common/__init__.py" in previous["files"]


def _source_tar(entries):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        for name, kind, payload in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            else:
                member.linkname = payload
                archive.addfile(member)
    data.seek(0)
    return tarfile.open(fileobj=data)


def test_source_archive_copy_does_not_depend_on_tarfile_extraction_filters(tmp_path, monkeypatch):
    def unsupported(*args, **kwargs):
        raise AssertionError("tarfile extraction APIs must not be used")

    monkeypatch.setattr(tarfile.TarFile, "extractall", unsupported)
    monkeypatch.setattr(tarfile.TarFile, "extract", unsupported)
    with _source_tar([
        ("src/", tarfile.DIRTYPE, ""),
        ("src/mlxlearn_common/__init__.py", tarfile.REGTYPE, b"VALUE = 1\n"),
    ]) as archive:
        bench.extract_source_archive(archive, tmp_path)
    assert (tmp_path / "src/mlxlearn_common/__init__.py").read_bytes() == b"VALUE = 1\n"


@pytest.mark.parametrize("name,kind,payload", [
    ("../escape.py", tarfile.REGTYPE, b"bad"),
    ("/absolute.py", tarfile.REGTYPE, b"bad"),
    ("src/../../escape.py", tarfile.REGTYPE, b"bad"),
    ("src\\..\\escape.py", tarfile.REGTYPE, b"bad"),
    ("src/link", tarfile.SYMTYPE, "../../escape.py"),
    ("src/link", tarfile.LNKTYPE, "src/safe.py"),
    ("src/fifo", tarfile.FIFOTYPE, ""),
    ("src/device", tarfile.CHRTYPE, ""),
])
def test_source_archive_rejects_unsafe_members_before_writing(tmp_path, name, kind, payload):
    with _source_tar([
        ("src/safe.py", tarfile.REGTYPE, b"safe"),
        (name, kind, payload),
    ]) as archive:
        with pytest.raises(ValueError, match="Unsafe source archive"):
            bench.extract_source_archive(archive, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_source_archive_rejects_existing_symlink_components(tmp_path):
    destination, outside = tmp_path / "snapshot", tmp_path / "outside"
    destination.mkdir()
    outside.mkdir()
    (destination / "src").symlink_to(outside, target_is_directory=True)
    with _source_tar([("src/escape.py", tarfile.REGTYPE, b"bad")]) as archive:
        with pytest.raises(ValueError, match="Unsafe source archive path"):
            bench.extract_source_archive(archive, destination)
    assert list(outside.iterdir()) == []


def test_source_archive_cannot_overwrite_an_existing_hardlink(tmp_path):
    destination, outside = tmp_path / "snapshot", tmp_path / "outside.py"
    destination.mkdir()
    outside.write_bytes(b"original")
    os.link(outside, destination / "source.py")
    with _source_tar([("source.py", tarfile.REGTYPE, b"replacement")]) as archive:
        with pytest.raises(FileExistsError):
            bench.extract_source_archive(archive, destination)
    assert outside.read_bytes() == b"original"


@pytest.mark.parametrize("package", ["mlxlearn_device", "mlxlearn_common"])
def test_loaded_source_guard_rejects_external_package(tmp_path, monkeypatch, package):
    fake = SimpleNamespace(__file__=str(tmp_path / "other/__init__.py"))
    monkeypatch.setattr(sys, "modules", {package: fake})
    with pytest.raises(RuntimeError, match="Mixed source"):
        bench.loaded_sources(tmp_path / "snapshot")


@pytest.mark.parametrize("value", ["current=HEAD", "../escape=HEAD", "label", "label=", "=HEAD"])
def test_reference_labels_cannot_escape_archive_directory(value):
    with pytest.raises(argparse.ArgumentTypeError):
        bench.parse_reference(value)
