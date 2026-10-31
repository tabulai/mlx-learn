"""Runtime artifact tests using fake extension bytes; nothing is built/imported."""

import importlib
import sysconfig

import pytest

from benchmarks.knn_auto_verify import check_runtime_artifact
from benchmarks.knn_freeze import freeze_source
from benchmarks.knn_freeze_runtime import freeze_runtime, verify_runtime_archive


@pytest.fixture
def runtime_inputs(tmp_path, monkeypatch):
    root, src = tmp_path, tmp_path / "src"
    cpp = src / "mlxlearn/_kernels/_neighbors_cpu_native.cpp"
    cpp.parent.mkdir(parents=True)
    cpp.write_text("// fixture C++\n")
    binary = cpp.with_name("_neighbors_cpu_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    binary.write_bytes(b"fake extension bytes; never import")
    for name in ("setup.py", "MANIFEST.in", "pyproject.toml"):
        (root / name).write_text("test build configuration\n")
    log = root / "build.log"
    log.write_text("clang++ -O3 fixture.cpp\n")
    source_archive = root / "source.zip"
    source_manifest = freeze_source(src, source_archive, source_state="test", revision="test")
    environment = {"implementation": "cpython", "cache_tag": "fixture-cpython",
                   "sysconfig": {"SOABI": sysconfig.get_config_var("SOABI")},
                   "machine": "arm64", "system": "Darwin", "sysconfig_platform": "test-platform"}
    module = importlib.import_module("benchmarks.knn_freeze_runtime")
    monkeypatch.setattr(module, "collect_environment", lambda path: environment)
    return root, src, source_archive, binary, log, cpp, source_manifest, module


def freeze(inputs, name="runtime.zip"):
    root, src, source_archive, binary, log, *_ = inputs
    return freeze_runtime(root, src, source_archive, binary, log, root / name,
                          build_command="python3 setup.py build_ext --inplace --force")


def test_runtime_bytes_are_reproducible_and_reject_overwrite(runtime_inputs):
    root = runtime_inputs[0]
    first, second = freeze(runtime_inputs), freeze(runtime_inputs, "runtime-second.zip")
    assert (root / "runtime.zip").read_bytes() == (root / "runtime-second.zip").read_bytes()
    assert first["native_extension"] == second["native_extension"]
    assert len(first["files"]) == 8
    assert verify_runtime_archive(root / "runtime.zip") == first
    with pytest.raises(FileExistsError):
        freeze(runtime_inputs)


@pytest.mark.parametrize("kind", ["source", "binary", "build_file", "log"])
def test_change_during_capture_leaves_no_artifact(runtime_inputs, monkeypatch, kind):
    root, _, _, binary, log, cpp, _, module = runtime_inputs
    original = module.collect_environment
    path = {"source": cpp, "binary": binary, "build_file": root / "setup.py", "log": log}[kind]

    def changing_environment(binary):
        path.write_bytes(b"changed during capture")
        return original(binary)

    monkeypatch.setattr(module, "collect_environment", changing_environment)
    with pytest.raises(RuntimeError, match="changed"):
        freeze(runtime_inputs)
    assert not (root / "runtime.zip").exists()
    assert not (root / "runtime.manifest.json").exists()


def test_runtime_verifier_links_exact_measured_binary_and_source(runtime_inputs):
    root, _, _, _, _, _, source, _ = runtime_inputs
    manifest = freeze(runtime_inputs)
    broad = {"source_hashes_before": source["files"], "source_snapshot": source["archive"]}
    native = {"available": True, "binary_sha256": manifest["native_extension"]["binary"]["sha256"],
              "cpp_sha256": manifest["native_extension"]["source"]["sha256"]}
    result = check_runtime_artifact(root / "runtime.zip", broad, native)
    assert result["matches_measured_source_and_binary"]
    native["binary_sha256"] = "different measured binary"
    with pytest.raises(ValueError, match="measured worker binary"):
        check_runtime_artifact(root / "runtime.zip", broad, native)


def test_rejects_tampered_runtime_archive(runtime_inputs):
    root = runtime_inputs[0]
    freeze(runtime_inputs)
    with (root / "runtime.zip").open("ab") as stream:
        stream.write(b"unexpected bytes")
    with pytest.raises(ValueError, match="archive hash"):
        verify_runtime_archive(root / "runtime.zip")
