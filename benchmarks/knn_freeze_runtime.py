"""Freeze an installed native kNN runtime alongside its source-only snapshot.

This does not compile, import or restore the extension. Run with the interpreter
used to build/benchmark it, after source and native builds have stopped changing.
The source-only archive still excludes binaries; this separate artifact records
the exact measured binary, source, build files/log and compiler/ABI provenance.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import struct
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

from .knn_freeze import verify_source
from .knn_pipeline import ROOT, sha256, source_hashes

CPP = "mlxlearn/_kernels/_neighbors_cpu_native.cpp"
BUILD_FILES = ("setup.py", "MANIFEST.in", "pyproject.toml")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect_environment(binary):
    def output(command):
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()

    return {
        "python_executable": sys.executable, "python_version": sys.version,
        "implementation": sys.implementation.name, "cache_tag": sys.implementation.cache_tag,
        "version_info": list(sys.version_info[:3]), "byteorder": sys.byteorder,
        "pointer_bits": struct.calcsize("P") * 8, "machine": platform.machine(),
        "system": platform.system(), "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0], "sysconfig_platform": sysconfig.get_platform(),
        "sysconfig": {name: sysconfig.get_config_var(name) for name in (
            "SOABI", "EXT_SUFFIX", "CC", "CXX", "CFLAGS", "LDFLAGS", "LDSHARED",
            "MACOSX_DEPLOYMENT_TARGET", "HOST_GNU_TYPE",
        )},
        "capture_process_environment": {name: os.environ.get(name) for name in (
            "ARCHFLAGS", "MACOSX_DEPLOYMENT_TARGET", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS",
        )},
        "compiler": {
            "version_command": ["xcrun", "clang++", "--version"],
            "full_version": output(["xcrun", "clang++", "--version"]),
            "path": output(["xcrun", "--find", "clang++"]),
            "sdk_version": output(["xcrun", "--show-sdk-version"]),
        },
        "binary_format": output(["file", "-b", str(binary)]),
        "binary_architectures": output(["xcrun", "lipo", "-archs", str(binary)]),
        "configuration_note": "Compile/link invocations in build.log record actual build flags. "
        "Sysconfig and capture_process_environment describe this provenance-capture process; "
        "setup.py may set target/architecture flags inside the separate build process.",
    }


def verify_runtime_archive(archive):
    archive = Path(archive)
    manifest = json.loads(archive.with_suffix(".manifest.json").read_text())
    require(manifest.get("kind") == "knn_native_runtime", "Unexpected runtime artifact kind")
    require(sha256(archive.read_bytes()) == manifest["archive"]["sha256"], "Runtime archive hash changed")
    with zipfile.ZipFile(archive) as saved:
        names = saved.namelist()
        require(len(names) == len(set(names)) and set(names) == set(manifest["files"]),
                "Runtime archive inventory changed")
        for name in names:
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe runtime member path")
            require(sha256(saved.read(name)) == manifest["files"][name], f"Runtime member changed: {name}")
    for kind in ("binary", "source"):
        member = manifest["native_extension"][kind]
        require(manifest["files"][member["archive_member"]] == member["sha256"],
                f"Runtime {kind} identity disagrees with its member")
    require(manifest["native_extension"]["source"]["sha256"] == manifest["source_files"][CPP],
            "Runtime C++ differs from source snapshot identity")
    return manifest


def freeze_runtime(project_root, source_root, source_archive, binary, build_log, archive, *, build_command):
    project_root, source_root = Path(project_root).resolve(), Path(source_root).resolve()
    source_archive, binary = Path(source_archive).resolve(), Path(binary).resolve()
    build_log, archive = Path(build_log).resolve(), Path(archive).resolve()
    manifest_path = archive.with_suffix(".manifest.json")
    if archive.suffix != ".zip":
        raise ValueError("Runtime archive must have a .zip suffix")
    if archive.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite a runtime archive or manifest")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix or binary.name != "_neighbors_cpu_native" + suffix:
        raise ValueError("Native binary ABI suffix does not match the capture interpreter")
    if not binary.is_relative_to(source_root):
        raise ValueError("Native binary must belong to the selected source root")
    if not build_command.strip():
        raise ValueError("Record the actual build command")
    frozen = verify_source(source_archive, source_root)
    before = source_hashes(source_root)
    if before != frozen["files"]:
        raise RuntimeError("Source changed after the source snapshot was verified")
    source_manifest_path = source_archive.with_suffix(".manifest.json")
    snapshot_hashes = {str(p): sha256(p.read_bytes()) for p in (source_archive, source_manifest_path)}
    binary_member = "src/" + str(binary.relative_to(source_root))
    source_member = "src/" + CPP
    inputs = {
        binary_member: binary, source_member: source_root / CPP,
        **{name: project_root / name for name in BUILD_FILES},
        "provenance/build.log": build_log,
    }
    captured = {name: path.read_bytes() for name, path in inputs.items()}
    if sha256(captured[source_member]) != before[CPP]:
        raise RuntimeError("C++ source changed during capture")
    environment = collect_environment(binary)
    captured["provenance/environment.json"] = (json.dumps(environment, indent=2, sort_keys=True) + "\n").encode()
    captured["provenance/build-command.txt"] = (build_command.rstrip() + "\n").encode()
    if source_hashes(source_root) != before:
        raise RuntimeError("Source inventory changed during runtime capture")
    if any(sha256(path.read_bytes()) != sha256(captured[name]) for name, path in inputs.items()):
        raise RuntimeError("Runtime binary, source, build files or log changed during capture")
    if any(sha256(Path(path).read_bytes()) != digest for path, digest in snapshot_hashes.items()):
        raise RuntimeError("Source snapshot or manifest changed during runtime capture")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as saved:
        for name, data in sorted(captured.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (0o100755 if name == binary_member else 0o100644) << 16
            entry.compress_type = zipfile.ZIP_STORED
            saved.writestr(entry, data)
    archive_bytes = buffer.getvalue()
    def stored_path(path):
        return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    manifest = {
        "schema_version": 1, "kind": "knn_native_runtime",
        "archive": {"path": stored_path(archive), "sha256": sha256(archive_bytes)},
        "files": {name: sha256(data) for name, data in sorted(captured.items())},
        "source_archive": {"path": stored_path(source_archive),
                           "sha256": snapshot_hashes[str(source_archive)],
                           "manifest_sha256": snapshot_hashes[str(source_manifest_path)]},
        "source_files": before, "source_revision": frozen.get("revision"),
        "native_extension": {
            "module": "mlxlearn._kernels._neighbors_cpu_native",
            "binary": {"archive_member": binary_member, "sha256": sha256(captured[binary_member])},
            "source": {"archive_member": source_member, "sha256": sha256(captured[source_member])},
        },
        "build": {"command": build_command, "log_sha256": sha256(captured["provenance/build.log"])},
        "abi": {"implementation": environment["implementation"], "cache_tag": environment["cache_tag"],
                "SOABI": environment["sysconfig"]["SOABI"], "EXT_SUFFIX": suffix,
                "machine": environment["machine"], "system": environment["system"],
                "sysconfig_platform": environment["sysconfig_platform"]},
        "restore_policy": "Manual restoration requires matching CPython ABI, extension suffix, arm64 "
        "architecture and compatible macOS runtime. Otherwise rebuild from the source archive and included "
        "build files. Source-only portable fallback is supported but is not the measured native configuration.",
    }
    archive.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with archive.open("xb") as output:
            created.append(archive)
            output.write(archive_bytes)
        with manifest_path.open("x") as output:
            created.append(manifest_path)
            output.write(json.dumps(manifest, indent=2) + "\n")
        verify_runtime_archive(archive)
        if source_hashes(source_root) != before or any(
            sha256(path.read_bytes()) != sha256(captured[name]) for name, path in inputs.items()
        ):
            raise RuntimeError("Source or runtime files changed before the freeze completed")
        if any(sha256(Path(path).read_bytes()) != digest for path, digest in snapshot_hashes.items()):
            raise RuntimeError("Source snapshot changed before the freeze completed")
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=ROOT / "src")
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--build-command", required=True, help="Exact command that produced this binary/log; recorded, not executed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_runtime(args.project_root, args.source_root, args.source_archive, args.binary,
                              args.build_log, args.output, build_command=args.build_command)
    print(json.dumps({"archive": manifest["archive"], "native_extension": manifest["native_extension"],
                      "abi": manifest["abi"]}, sort_keys=True))


if __name__ == "__main__":
    main()
