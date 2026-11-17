"""CPU-only archive, isolation, controller and paired-provenance checks."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import py_compile
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gap_bench as gap  # noqa: E402


@contextlib.contextmanager
def fresh_runtime():
    saved = {name: module for name, module in sys.modules.items()
             if name == "mlxlearn" or name.startswith("mlxlearn.")}
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "mlxlearn" or name.startswith("mlxlearn."):
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture
def snapshot(tmp_path):
    files = {
        gap.CORE_MODULES["mlxlearn.cluster"]: b"from ._e_means import EMeans\n",
        gap.CORE_MODULES["mlxlearn.cluster._e_means"]: b"from .._kernels.e_means import TOKEN\nfrom ._recipe import RECIPE\nclass EMeans: marker = TOKEN + RECIPE\n",
        gap.CORE_MODULES["mlxlearn.cluster._recipe"]: b"RECIPE = ' recipe'\n",
        gap.CORE_MODULES["mlxlearn._kernels.e_means"]: b"from .e_means_bf16 import TOKEN\n",
        gap.CORE_MODULES["mlxlearn._kernels.e_means_bf16"]: b"TOKEN = 'frozen'\n",
    }
    archive = tmp_path / "snapshot.zip"
    with zipfile.ZipFile(archive, "w") as target:
        for name, content in files.items():
            target.writestr(name, content)
    metadata = {"archive_sha256": gap.base.file_sha256(archive),
                "source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(json.dumps(metadata))
    return archive, manifest, files, metadata


def test_frozen_archive_and_all_members_are_verified(snapshot):
    archive, manifest, files, metadata = snapshot
    actual, contents = gap.archive_contents(archive, manifest)
    assert actual == metadata and contents == files
    assert gap.runtime_sources("previous", archive, manifest) == metadata["source_sha256"]


@pytest.mark.parametrize("change", ["archive", "member", "extra", "missing"])
def test_corrupt_or_incomplete_archive_is_rejected_before_import(snapshot, change):
    archive, manifest, files, metadata = snapshot
    if change == "archive":
        archive.write_bytes(archive.read_bytes() + b"changed")
    else:
        files = dict(files)
        if change == "member":
            files[gap.CORE_MODULES["mlxlearn.cluster._recipe"]] = b"RECIPE = 'changed'\n"
        elif change == "extra":
            files["unexpected.py"] = b"pass\n"
        else:
            del files[gap.CORE_MODULES["mlxlearn.cluster._recipe"]]
        with zipfile.ZipFile(archive, "w") as target:
            for name, content in files.items():
                target.writestr(name, content)
        metadata["archive_sha256"] = gap.base.file_sha256(archive)
        manifest.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="mismatch|missing"):
        gap.archive_contents(archive, manifest)


def test_only_selected_source_tree_is_imported_and_actual_paths_hashes_proven(tmp_path, snapshot):
    _, _, files, metadata = snapshot
    root = tmp_path / "extracted"
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with fresh_runtime():
        gap.install_runtime(root)
        assert sys.modules["mlxlearn.cluster"].EMeans.marker == "frozen recipe"
        proof = gap.runtime_proof(root, metadata["source_sha256"])
        assert set(proof["modules"]) == set(gap.CORE_MODULES)
        assert all(Path(item["path"]).is_relative_to(root) for item in proof["modules"].values())
        with pytest.raises(RuntimeError, match="fresh process"):
            gap.install_runtime(root)
        (root / gap.CORE_MODULES["mlxlearn.cluster._recipe"]).write_text("RECIPE = 'changed'\n")
        with pytest.raises(RuntimeError, match="hash"):
            gap.runtime_proof(root, metadata["source_sha256"])


def test_loaded_module_from_current_tree_cannot_leak_into_previous_runtime(tmp_path, snapshot):
    _, _, files, metadata = snapshot
    root = tmp_path / "extracted"
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with fresh_runtime():
        gap.install_runtime(root)
        extra = types.ModuleType("mlxlearn.current_leak")
        extra.__file__ = str(tmp_path / "outside.py")
        sys.modules[extra.__name__] = extra
        with pytest.raises(RuntimeError, match="escaped"):
            gap.runtime_proof(root, metadata["source_sha256"])


def test_selected_runtime_compiles_verified_source_not_stale_same_timestamp_bytecode(tmp_path, snapshot):
    _, _, files, metadata = snapshot
    root = tmp_path / "extracted"
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    kernel = root / gap.CORE_MODULES["mlxlearn._kernels.e_means_bf16"]
    kernel.write_bytes(b"TOKEN = 'leaked'\n")
    timestamp = kernel.stat().st_mtime_ns
    py_compile.compile(str(kernel), doraise=True)
    kernel.write_bytes(files[gap.CORE_MODULES["mlxlearn._kernels.e_means_bf16"]])
    os.utime(kernel, ns=(timestamp, timestamp))
    with fresh_runtime():
        gap.install_runtime(root, metadata["source_sha256"])
        assert sys.modules["mlxlearn.cluster"].EMeans.marker == "frozen recipe"
        gap.runtime_proof(root, metadata["source_sha256"])


def test_import_rejects_source_changed_from_campaign_snapshot(tmp_path, snapshot):
    _, _, files, metadata = snapshot
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (tmp_path / gap.CORE_MODULES["mlxlearn.cluster._recipe"]).write_text("RECIPE='changed'\n")
    with fresh_runtime(), pytest.raises(RuntimeError, match="changed before import"):
        gap.install_runtime(tmp_path, metadata["source_sha256"])


def record(mode="previous", seed=10):
    runtime = {variant: {path: f"{variant}-{name}" for name, path in gap.CORE_MODULES.items()}
               for variant in gap.MODES}
    shared = {"gap_bench.py": "harness", "h100.py": "unchanged"}
    config = {"dataset": "example", "mode": mode, "seed": seed, "assignment": "auto",
              "epochs": 2, "precision": "bfloat16", "recipe": "faiss", "finalization": "last_epoch",
              "memory": False, "threads": 8, "memory_interval_seconds": .01,
              "runtime_comparison": runtime, "shared_source_sha256": shared,
              "expected_source_sha256": {**shared, **runtime[mode]}, "archive_sha256": "archive",
              "expected_versions": {"python": "test"}, "expected_host": {"cpu": "test"},
              "output": f"/tmp/{mode}.json", "worker_output": f"/tmp/{mode}.tmp"}
    root = Path("/tmp/previous-runtime" if mode == "previous" else "/tmp/current-runtime")
    return {"status": "completed", "kind": gap.KIND, "config": config,
            "n": 100, "d": 9, "k": 3, "dataset": {"train": "same", "test": "same"},
            "initial_centroids_sha256": "same-init", "train_seconds": 3., "train_seconds_samples": [3.],
            "test_mse": 1., "memory_samples": [None], "versions": config["expected_versions"],
            "hardware": config["expected_host"], "metal_device": {"device": "same"},
            "source_sha256": config["expected_source_sha256"], "runtime_source_sha256": runtime[mode],
            "shared_source_sha256": shared, "recipe_resolved": {"alpha0": .2, "assignment": "auto", "seed": seed},
            "timing_scope": "same", "scoring": "same", "memory_scope": "same",
            "loaded_runtime": {"verified": True, "root": str(root),
                               "origin": "archive" if mode == "previous" else "current_source_tree",
                               "modules": {name: {"relative_path": path, "path": str(root / path), "sha256": runtime[mode][path]}
                                           for name, path in gap.CORE_MODULES.items()}}}


def test_pairing_allows_declared_runtime_difference_with_identical_workload():
    previous, optimized = record(), record("optimized")
    optimized["train_seconds"], optimized["train_seconds_samples"] = 1.5, [1.5] * 3
    optimized["memory_samples"] = [None] * 3
    pairs, unpaired = gap.pair_results([previous, optimized])
    assert not unpaired and len(pairs) == 1
    assert pairs[0]["speedup"] == 2 and pairs[0]["quality_delta_pct"] == 0


@pytest.mark.parametrize("field", ["dataset", "initial_centroids_sha256", "recipe_resolved", "timing_scope"])
def test_pairing_rejects_different_workload_even_with_allowed_sources(field):
    previous, optimized = record(), record("optimized")
    optimized[field] = {"different": True} if isinstance(optimized[field], dict) else "different"
    pairs, unpaired = gap.pair_results([previous, optimized])
    assert not pairs and len(unpaired) == 2


def test_historical_h100_records_and_unverified_loaded_sources_cannot_pair():
    for mutation in (lambda r: r.update(kind="h100_protocol_metal"),
                     lambda r: r["loaded_runtime"].update(origin="current_source_tree"),
                     lambda r: r["loaded_runtime"]["modules"]["mlxlearn.cluster"].update(sha256="changed")):
        row = record()
        mutation(row)
        with pytest.raises(RuntimeError):
            gap.pair_results([row])


def test_new_runtime_comparison_cannot_reuse_previous_campaign_as_matched_pair():
    previous, optimized = record(), record("optimized")
    optimized["config"]["runtime_comparison"]["previous"] = {"different": "snapshot"}
    pairs, unpaired = gap.pair_results([previous, optimized])
    assert not pairs and len(unpaired) == 2
    with pytest.raises(RuntimeError, match="Duplicate"):
        gap.pair_results([previous, copy.deepcopy(previous)])


def test_adapter_targets_new_worker_and_restores_unchanged_h100_globals():
    original = (gap.h100.__file__, gap.h100.KIND, gap.h100.ASSIGNMENT, gap.h100.validate_result)
    with gap.adapter(record()["config"]):
        assert Path(gap.h100.__file__) == Path(gap.__file__)
        assert gap.h100.KIND == gap.KIND
        assert gap.h100.ASSIGNMENT == {"previous": "auto", "optimized": "auto"}
    assert (gap.h100.__file__, gap.h100.KIND, gap.h100.ASSIGNMENT, gap.h100.validate_result) == original


@pytest.mark.parametrize("corrupt", [False, True])
def test_existing_controller_launches_adapter_and_rejects_missing_runtime_proof(tmp_path, monkeypatch, corrupt):
    row = record()
    config = row["config"]
    config.update(output=str(tmp_path / "result.json"), worker_output=str(tmp_path / "result.tmp"))
    if corrupt:
        row["loaded_runtime"]["verified"] = False
    program = """
import json,sys
from pathlib import Path
c=json.loads(sys.argv[1]);r=json.loads(sys.argv[2]);r['config']=c
print(json.dumps({'phase':'FIT_READY','call':0}),flush=True)
assert sys.stdin.readline().strip()=='GO'
print(json.dumps({'phase':'FIT_DONE','call':0,'train_seconds':3.}),flush=True)
assert sys.stdin.readline().strip()=='STOP_ACK'
Path(c['worker_output']).write_text(json.dumps(r))
"""
    original = subprocess.Popen
    processes = []

    def launch(command, **kwargs):
        assert Path(command[1]) == Path(gap.__file__).resolve()
        child = original([sys.executable, "-c", program, command[-1], json.dumps(row)], **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr(gap.h100.subprocess, "Popen", launch)
    with gap.adapter(config):
        if corrupt:
            with pytest.raises(RuntimeError, match="loaded-runtime proof"):
                gap.h100.measured_run(config, 2)
        else:
            gap.h100.measured_run(config, 2)
    assert processes[0].poll() == 0
    assert Path(config["output"]).exists() is not corrupt


def test_default_cli_selects_all_datasets_three_seeds_and_balanced_order(monkeypatch, tmp_path, snapshot):
    archive, manifest, _, _ = snapshot
    entries = [{"dataset": f"dataset-{i}", "suite": "core1m" if i < 6 else "vibe", "epochs": 2}
               for i in range(17)]
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"configurations": entries}))
    observed = []
    monkeypatch.setattr(sys, "argv", [gap.__file__, "--protocol", str(protocol), "--archive", str(archive),
                                    "--archive-manifest", str(manifest), "--out", str(tmp_path / "results")])
    monkeypatch.setattr(gap.base, "package_versions", lambda: {"python": "test"})
    monkeypatch.setattr(gap.base, "host_identity", lambda: {"cpu": "test"})
    monkeypatch.setattr(gap.h100, "measured_run", lambda config, timeout: observed.append(config))
    gap.main()
    assert len(observed) == 17 * 3 * 2
    assert {c["seed"] for c in observed} == {10, 11, 12}
    assert [c["mode"] for c in observed[:4]] == ["previous", "optimized", "optimized", "previous"]
    assert all(c["assignment"] == "auto" and c["capture_assignments"] and c["warmup"] for c in observed)
