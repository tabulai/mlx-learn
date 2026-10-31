"""Restore a frozen native binary only into an isolated benchmark reference."""

from __future__ import annotations

import platform
import sys
import sysconfig
import zipfile
from pathlib import Path

from .knn_freeze_runtime import CPP, verify_runtime_archive
from .knn_pipeline import sha256, source_hashes


def current_abi():
    return {
        "implementation": sys.implementation.name, "cache_tag": sys.implementation.cache_tag,
        "SOABI": sysconfig.get_config_var("SOABI"), "EXT_SUFFIX": sysconfig.get_config_var("EXT_SUFFIX"),
        "machine": platform.machine(), "system": platform.system(),
    }


def install_reference_runtime(archive, destination, source_manifest):
    """Keep archived source untouched and reject incompatible/stale binaries."""
    archive, destination = Path(archive), Path(destination).resolve()
    runtime = verify_runtime_archive(archive)
    if runtime["source_files"] != source_manifest["files"] or (
        runtime["source_archive"]["sha256"] != source_manifest["archive"]["sha256"]
    ):
        raise ValueError("Reference runtime and source archive do not identify the same implementation")
    for key, value in current_abi().items():
        if runtime["abi"].get(key) != value:
            raise ValueError(f"Reference native ABI mismatch: {key}")
    before = source_hashes(destination)
    if before != source_manifest["files"]:
        raise ValueError("Extracted reference source differs from its manifest")
    relative = "mlxlearn/_kernels/_neighbors_cpu_native" + current_abi()["EXT_SUFFIX"]
    binary = runtime["native_extension"]["binary"]
    if binary["archive_member"] != "src/" + relative:
        raise ValueError("Unexpected reference binary member path")
    target = destination / relative
    if target.exists():
        raise FileExistsError("Reference binary already exists; never overwrite a worker implementation")
    with zipfile.ZipFile(archive) as saved:
        data = saved.read(binary["archive_member"])
    if sha256(data) != binary["sha256"]:
        raise ValueError("Reference binary bytes differ from their manifest")
    with target.open("xb") as output:
        output.write(data)
    target.chmod(0o755)
    if source_hashes(destination) != before or sha256(target.read_bytes()) != binary["sha256"]:
        raise RuntimeError("Reference implementation changed during native restoration")
    return runtime


def validate_reference_worker(worker, runtime):
    metadata = worker.get("native_cpu_extension") or {}
    if not metadata.get("available"):
        raise ValueError("Archived automatic worker did not discover its native binary")
    for kind in ("binary", "source"):
        if metadata[kind]["sha256"] != runtime["native_extension"][kind]["sha256"]:
            raise ValueError(f"Archived automatic worker loaded the wrong native {kind}")
    if worker["source_hashes"] != runtime["source_files"]:
        raise ValueError("Archived automatic worker loaded the wrong source inventory")
    root = Path(worker["source_root"]).resolve()
    relative = "mlxlearn/_kernels/_neighbors_cpu_native" + runtime["abi"]["EXT_SUFFIX"]
    if Path(metadata["binary"]["path"]).resolve() != root / relative:
        raise ValueError("Archived automatic worker loaded a binary outside its isolated source")
    if Path(metadata["source"]["path"]).resolve() != root / CPP:
        raise ValueError("Archived automatic worker loaded C++ metadata outside its isolated source")


def reference_runtime_unchanged(root, runtime):
    relative = runtime["native_extension"]["binary"]["archive_member"].removeprefix("src/")
    return (sha256((Path(root) / relative).read_bytes()) == runtime["native_extension"]["binary"]["sha256"]
            and source_hashes(Path(root)) == runtime["source_files"])
