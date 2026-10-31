"""CPU-only native identity checks; fake binaries are never imported."""

import copy
import hashlib
import importlib.util
from types import SimpleNamespace

import pytest

from benchmarks.knn_auto_verify import check_native_binary_metadata
from benchmarks.knn_broad_worker import native_cpu_metadata


def native_files(tmp_path, monkeypatch):
    root = tmp_path / "src"
    source = root / "mlxlearn/_kernels/_neighbors_cpu_native.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("// test source\n")
    binary = source.with_suffix(".so")
    binary.write_bytes(b"fake binary: must never be imported")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(binary)))
    return root, source, binary


def reports(root, metadata):
    worker = {"source_root": str(root), "native_cpu_extension": metadata}
    broad = {"workers": {"current": copy.deepcopy(worker)},
             "source_hashes_before": {metadata["source"]["relative_path"]: metadata["source"]["sha256"]}}
    control = {"workers": {v: copy.deepcopy(worker) for v in ("current", "current_b", "current_gpu")}}
    # Previous frozen source roots are intentionally absent from this mapping:
    # verification must not require their deleted temporary directories.
    return broad, control


def test_frozen_gpu_source_does_not_search_live_extension(tmp_path, monkeypatch):
    def unexpected_search(name):
        raise AssertionError("Old source must not discover the live extension")

    monkeypatch.setattr(importlib.util, "find_spec", unexpected_search)
    metadata = native_cpu_metadata(tmp_path)
    assert not metadata["available"]
    assert metadata["source"] is metadata["binary"] is None


def test_records_bytes_without_importing_extension(tmp_path, monkeypatch):
    root, source, binary = native_files(tmp_path, monkeypatch)
    metadata = native_cpu_metadata(root)
    assert metadata["available"]
    assert metadata["binary"]["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert metadata["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    result = check_native_binary_metadata(*reports(root, metadata))
    assert result["all_four_current_source_workers_match"]
    assert len(result["current_files_verified_after_run"]) == 2


@pytest.mark.parametrize("change", ["binary_on_disk", "cpp_on_disk", "worker_binary", "worker_cpp"])
def test_rejects_changed_native_identity(tmp_path, monkeypatch, change):
    root, source, binary = native_files(tmp_path, monkeypatch)
    metadata = native_cpu_metadata(root)
    broad, control = reports(root, metadata)
    if change == "binary_on_disk":
        binary.write_bytes(b"changed")
    elif change == "cpp_on_disk":
        source.write_text("// changed\n")
    else:
        kind = "binary" if change == "worker_binary" else "source"
        control["workers"]["current_gpu"]["native_cpu_extension"][kind]["sha256"] = "changed"
    with pytest.raises(ValueError):
        check_native_binary_metadata(broad, control)
