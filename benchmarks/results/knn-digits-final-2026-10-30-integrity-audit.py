"""Post-timing byte/provenance audit. Never launches estimators or benchmarks.

Prepared while timings run; execute only after the six-panel runner and the
independent/report audits complete. Writes new files and refuses overwrites.
"""

import argparse
import datetime
import email.parser
import hashlib
import json
import mmap
import re
import struct
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results"
REFERENCES = ROOT / "benchmarks/reference"
PREFIX = "knn-digits-final-2026-10-30"
PRIOR = "knn-startup-final2-2026-10-30"
ORDER = ["cold", "broad", "control", "legacy-control", "preservation-repeat", "first-query"]
EXPECTED = {
    "source": "5a4b3c5c130c043640b8755b3e26aaf53911729cd5e56acc37699e681da268db",
    "runtime": "b92e7cb2dc879eaffbcb375c5f6ea9e9dc8427f0ca68f325afb9ca9ad37a5bae",
    "harnesses": "8718e3cdfdadaca62ead81ac22faae1d673a6db4bef1c72237c04c185a8b96a3",
    "native": "5cb92f7ce3d70e8e1803f9d0b81c9753ec8e611f5c02d375c382fccb094d833a",
    "cpp": "d3b116cd2c95c8d2ad05aa7de4e3b5a82761693b7577aa6f7afbfe4d0d5b5bce",
}
TOP_KEY = re.compile(rb'(?m)^  "([^"\\]+)":')
_HASHES = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    path = Path(path).resolve()
    before = path.stat()
    key = (str(path), before.st_size, before.st_mtime_ns)
    if key not in _HASHES:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
        _HASHES[key] = hasher.hexdigest()
    return _HASHES[key]


def relative(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_absolute() and path.is_relative_to(ROOT) else str(path)


def file_record(path):
    path = Path(path)
    return {"path": relative(path), "sha256": digest(path), "bytes": path.stat().st_size}


def save_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def metadata_only(path):
    """Read pretty-printed root fields without materializing huge cases arrays.

    All harness JSON uses indent=2. Nested fields begin at four or more spaces;
    JSON strings escape literal newlines. Exact two-space root-key markers thus
    delimit complete values. Raw files are independently hashed in full.
    """
    result = {}
    with Path(path).open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        assert data[:1] == b"{" and data[-2:].strip() == b"}", path
        keys = list(TOP_KEY.finditer(data))
        assert keys and sum(match[1] == b"cases" for match in keys) == 1, path
        for position, match in enumerate(keys):
            key = match[1].decode("ascii")
            assert key not in result
            if key == "cases":
                continue
            end = keys[position + 1].start() if position + 1 < len(keys) else len(data)
            value = data[match.end():end].strip()
            if position + 1 < len(keys):
                assert value[-1:] == b",", (path, key)
            else:
                assert value[-1:] == b"}", (path, key)
            result[key] = json.loads(value[:-1])
    return result


def archive_record(prefix, suffix=""):
    manifest_path = REFERENCES / f"{prefix}{suffix}.manifest.json"
    manifest = read(manifest_path)
    path = ROOT / manifest["archive"]["path"]
    assert digest(path) == manifest["archive"]["sha256"], path
    with zipfile.ZipFile(path) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(manifest["files"])
        assert set(archive.namelist()) == set(manifest["files"])
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, (path, name)
    record = {**file_record(path), "manifest": relative(manifest_path),
              "manifest_sha256": digest(manifest_path), "member_count": len(manifest["files"]),
              "member_inventory_and_hashes_verified": True}
    return manifest, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-execution-record", type=Path,
                        help="Optional actual commands/status record from the independent audit owner")
    parser.add_argument("--package-only", action="store_true",
                        help="Verify archives/live files/packages and save package validation before other audits finish")
    args = parser.parse_args()
    outputs = {name: RESULTS / f"{PREFIX}-{name}.json"
               for name in ["execution", "post-run-integrity", "package-validation"]}
    assert not any(path.exists() for name, path in outputs.items() if name != "package-validation"), "Refusing to overwrite audit outputs"
    if args.package_only:
        assert not outputs["package-validation"].exists(), "Refusing to overwrite package validation"
    runner_path = RESULTS / f"{PREFIX}-runner-status.json"
    runner = read(runner_path)
    assert runner["status"] == "complete" and [x["name"] for x in runner["panels"]] == ORDER
    assert all(x["status"] == "complete" and x["returncode"] == 0 for x in runner["panels"])
    assert runner["frozen_hashes_before"] == runner["frozen_hashes_after"]
    for name, expected in runner["frozen_hashes_before"].items():
        assert digest(ROOT / name) == expected, name
    commands_path = RESULTS / f"{PREFIX}-commands.json"
    commands = read(commands_path)
    assert digest(commands_path) == runner["commands_sha256"]
    assert digest(RESULTS / f"{PREFIX}-run-panels.py") == runner["runner_sha256"]
    assert digest(ROOT / commands["predeclaration"]) == runner["predeclaration_sha256"]
    for planned, actual in zip(commands["panels"], runner["panels"], strict=True):
        assert planned["argv"] == actual["declared_argv"]
        assert actual["executed_argv"][1:] == planned["argv"][1:]
        assert actual["executed_argv"][0] == runner["python"]
        assert planned["output"] == actual["output"] and planned["log"] == actual["log"]

    manifests, archives = {}, {}
    for name, suffix in [("source", ""), ("runtime", "-runtime"), ("harnesses", "-harnesses")]:
        manifests[name], archives[name] = archive_record(PREFIX, suffix)
        assert archives[name]["sha256"] == EXPECTED[name]
    source, runtime, harness = (manifests[key] for key in ["source", "runtime", "harnesses"])
    assert len(source["files"]) == 63 and len(harness["files"]) == 148
    for name, expected in source["files"].items():
        assert digest(ROOT / "src" / name) == expected, name
    suffixes = {".py", ".typed", ".metal", ".h", ".hpp", ".cpp", ".mm"}
    inventory = {str(path.relative_to(ROOT / "src"))
                 for package in ["mlxlearn", "mlxlearn_device"]
                 for path in (ROOT / "src" / package).rglob("*")
                 if path.is_file() and path.suffix in suffixes}
    assert inventory == set(source["files"])
    for name, expected in harness["files"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in runtime["files"].items():
        if not name.startswith("provenance/"):
            assert digest(ROOT / name) == expected, name
    assert runtime["source_files"] == source["files"]
    assert runtime["source_archive"]["sha256"] == archives["source"]["sha256"]
    assert runtime["source_archive"]["manifest_sha256"] == archives["source"]["manifest_sha256"]
    native = runtime["native_extension"]
    assert native["binary"]["sha256"] == EXPECTED["native"]
    assert native["source"]["sha256"] == EXPECTED["cpp"]
    native_path = ROOT / native["binary"]["archive_member"]
    magic, cpu, _, filetype = struct.unpack("<IIII", native_path.read_bytes()[:16])
    assert (magic, cpu, filetype) == (0xfeedfacf, 0x0100000c, 8)
    with zipfile.ZipFile(ROOT / runtime["archive"]["path"]) as archive:
        assert archive.read("provenance/build-command.txt").decode().strip() == runtime["build"]["command"]
        build_log = archive.read("provenance/build.log")
        assert hashlib.sha256(build_log).hexdigest() == runtime["build"]["log_sha256"]
        assert all(flag in build_log.decode() for flag in
                   ["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off", "-arch arm64"])
        environment = json.loads(archive.read("provenance/environment.json"))
    assert digest(RESULTS / "knn-digits-gap-fused-build-2026-10-30.log") == runtime["build"]["log_sha256"]
    prior, prior_record = archive_record(PRIOR)
    prior_runtime, prior_runtime_record = archive_record(PRIOR, "-runtime")
    old_gpu, old_gpu_record = archive_record("knn-small-final-2026-10-30")
    changed = sorted(name for name, value in source["files"].items() if prior["files"].get(name) != value)
    assert changed == ["mlxlearn/_kernels/_neighbors_cpu_native.cpp"]
    assert set(source["files"]) == set(prior["files"])
    gpu = {name: value for name, value in source["files"].items() if name.startswith("mlxlearn_device/")}
    assert len(gpu) == 21 and all(old_gpu["files"][name] == prior["files"][name] == value
                                for name, value in gpu.items())

    build_directory = Path("/tmp/mlxlearn-knn-digits-final-build")
    wheels, sdists = list(build_directory.glob("*.whl")), list(build_directory.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1
    wheel, sdist = wheels[0], sdists[0]
    with zipfile.ZipFile(wheel) as archive:
        for name, expected in source["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
        member = native["binary"]["archive_member"].removeprefix("src/")
        assert hashlib.sha256(archive.read(member)).hexdigest() == EXPECTED["native"]
        metadata = email.parser.Parser().parsestr(archive.read(next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))).decode())
        assert any(value.lower().startswith("scipy") for value in metadata.get_all("Requires-Dist", []))
        tags = archive.read(next(name for name in archive.namelist() if name.endswith(".dist-info/WHEEL"))).decode()
        assert "Tag: cp313-cp313-macosx_11_0_arm64" in tags
    with tarfile.open(sdist, "r:gz") as archive:
        top = archive.getnames()[0].split("/")[0]
        for name, expected in source["files"].items():
            assert hashlib.sha256(archive.extractfile(f"{top}/src/{name}").read()).hexdigest() == expected, name
        for name in ["setup.py", "pyproject.toml", "MANIFEST.in"]:
            assert hashlib.sha256(archive.extractfile(f"{top}/{name}").read()).hexdigest() == harness["files"][name]
    package = {"validation_pass": True, "source_archive_sha256": archives["source"]["sha256"],
               "source_files_checked_in_wheel_and_sdist": 63, "wheel": file_record(wheel),
               "sdist": file_record(sdist), "native_binary_sha256": EXPECTED["native"],
               "wheel_tag": "cp313-cp313-macosx_11_0_arm64", "scipy_direct_dependency_verified": True,
               "build_files_match": {name: True for name in ["setup.py", "pyproject.toml", "MANIFEST.in"]}}
    if args.package_only:
        save_new(outputs["package-validation"], package)
        print(json.dumps({"package_validation": file_record(outputs["package-validation"]),
                          "archives_and_live_files_verified": True,
                          "changed_production_files": changed}, indent=2))
        return

    panels, fixtures, metadata = {}, {}, {}
    previous_finish = None
    for run in runner["panels"]:
        kind = run["name"]
        current = metadata_only(ROOT / run["output"])
        metadata[kind] = current
        assert current["status"] == "complete" and current["source_snapshot"]["sha256"] == archives["source"]["sha256"]
        assert current["source_hashes_before"] == current["source_hashes_after"] == source["files"]
        before = {relative(key): value for key, value in current["harness_hashes_before"].items()}
        after = {relative(key): value for key, value in current["harness_hashes_after"].items()}
        assert before == after and all(harness["files"][name] == value for name, value in before.items())
        assert current["harness_integrity_pass"] and current["fixture_integrity_pass"]
        assert current["fixture_hashes_before"] == current["fixture_hashes_after"]
        assert current.get("acceptance_pass", current.get("accuracy_and_integrity_pass")) is True
        assert current.get("current_source_integrity_pass", current.get("source_integrity_pass")) is True
        assert len(current["fixtures"]) == current["planned_cases"]
        assert run["started_at"] <= current["started_at"] <= current["finished_at"] <= run["finished_at"]
        if previous_finish is not None:
            assert run["started_at"] >= previous_finish
        previous_finish = run["finished_at"]
        panel = {"path": run["output"], "case_count": len(current["fixtures"]),
                 "variants": current["variants"], "started_at": current["started_at"],
                 "finished_at": current["finished_at"], "acceptance_and_integrity_pass": True,
                 "quality_by_variant": current["quality_by_variant"], "executed_argv": run["executed_argv"]}
        for key in ["repeats", "warmup", "pairs", "warmup_seconds_per_variant_operation", "order_policy",
                    "planned_case_operation_count", "target_speedup", "target_operations",
                    "expected_launches", "completed_launches", "timing_count_pass"]:
            if key in current:
                panel[key] = current[key]
        worker_native = {}
        for name, worker in current.get("workers", {}).items():
            if name.startswith("current"):
                assert worker["source_hashes"] == source["files"]
                extension = worker["native_cpu_extension"]
                assert extension["binary"]["sha256"] == EXPECTED["native"]
                assert extension["source"]["sha256"] == EXPECTED["cpp"]
                worker_native[name] = {"binary_sha256": EXPECTED["native"], "cpp_sha256": EXPECTED["cpp"]}
            elif name == "previous_auto":
                assert worker["source_hashes"] == prior["files"]
                assert worker["native_cpu_extension"]["binary"]["sha256"] == prior_runtime["native_extension"]["binary"]["sha256"]
        panel["verified_current_worker_native"] = worker_native
        if "native_hashes_before" in current:
            assert current["native_hashes_before"] == current["native_hashes_after"]
            assert current["native_hashes_before"]["current"] == EXPECTED["native"]
            assert current["native_integrity_pass"] and current["reference_runtime_integrity_pass"]
            panel["native_before_after"] = current["native_hashes_before"]
        panels[kind] = panel
        for fixture in current["fixtures"]:
            name = fixture["case"]["name"]
            entry = {key: fixture[key] for key in ["case", "seed", "fixture_sha256", "array_sha256", "array_shapes", "array_dtypes"]}
            assert current["fixture_hashes_before"][name + ".npz"] == entry["fixture_sha256"]
            assert name not in fixtures or fixtures[name] == entry
            fixtures[name] = entry
    assert panels["cold"]["completed_launches"] == 450
    assert panels["broad"]["case_count"] == 72
    assert len(panels["broad"]["target_operations"]) == 9
    assert panels["control"]["case_count"] == 48 and panels["control"]["planned_case_operation_count"] == 222
    assert panels["control"]["order_policy"] == "balanced"
    assert panels["legacy-control"]["planned_case_operation_count"] == 60
    assert panels["preservation-repeat"]["planned_case_operation_count"] == 18
    assert panels["first-query"]["case_count"] * len(panels["first-query"]["variants"]) * panels["first-query"]["repeats"] == 2700
    panels["first-query"].update(fresh_models=2700, completed_queries=5400)

    test_commands_path = RESULTS / "knn-digits-test-matrix-commands-2026-10-30.json"
    test_status_path = RESULTS / "knn-digits-test-matrix-status-2026-10-30.json"
    test_commands, test_status = read(test_commands_path), read(test_status_path)
    assert test_status["status"] == "complete" and len(test_status["runs"]) == 3
    test_results = []
    for planned, actual in zip(test_commands, test_status["runs"], strict=True):
        assert planned["command"] == actual["command"] and planned["log"] == actual["log"]
        assert actual["exit_code"] == 0 and actual["finished_at"] < runner["started_at"]
        text = (ROOT / actual["log"]).read_text()
        match = re.search(r"(\d+) passed, (\d+) deselected, (\d+) warnings in ([\d.]+)s", text)
        assert match and int(match[1]) == 1474 and int(match[2]) == 5
        test_results.append({**actual, "passed": 1474, "deselected": 5,
                             "warnings": int(match[3]), "seconds": float(match[4])})
    harness_log = RESULTS / "knn-digits-harness-tests-2026-10-30.log"
    assert "65 passed" in harness_log.read_text()

    independent_path = RESULTS / f"{PREFIX}-independent-audit.json"
    independent = read(independent_path)
    assert independent["evidence_validation_pass"] and independent["accuracy_and_integrity_pass"]
    assert independent["native_cpu_extension"]["binary_sha256"] == EXPECTED["native"]
    for entry in independent["evidence"].values():
        assert digest(entry["path"]) == entry["sha256"]
    reporting_path = RESULTS / f"{PREFIX}-additional-audit.json"
    reporting = read(reporting_path)
    assert reporting["validation_pass"] and reporting["current_source_harness_runtime_identity_pass"]
    driver = RESULTS / f"{PREFIX}-report-audit.py"
    assert reporting["driver"]["sha256"] == digest(driver)
    for kind in ["legacy-control", "preservation-repeat"]:
        audit_path = RESULTS / f"{PREFIX}-{kind}-audit.json"
        audit = read(audit_path)
        assert audit["validation_pass"] and audit["acceptance_pass"]
        assert audit["source_fixture_harness_archive_and_disk_identity_pass"]
        assert audit["driver"]["sha256"] == digest(driver)
        assert audit["raw_evidence"]["sha256"] == digest(audit["raw_evidence"]["path"])
        assert audit["full_control_sha256"] == digest(RESULTS / f"{PREFIX}-control.json")
        assert reporting["repeat_panels"][kind]["audit_sha256"] == digest(audit_path)

    provenance_path = RESULTS / "knn-digits-gap-probe-provenance-2026-10-30.json"
    provenance = read(provenance_path)
    for entry in provenance["drivers"]:
        assert digest(ROOT / entry["path"]) == entry["sha256"]
        raw = read(ROOT / entry["output"])
        assert raw["driver"] == (ROOT / entry["path"]).read_text()
    artifacts = set()
    for run in runner["panels"]:
        artifacts.update(ROOT / run[key] for key in ["output", "log"])
    for pattern in [f"{PREFIX}-*audit*.json", f"{PREFIX}-*audit*.log", f"{PREFIX}-*audit*.py",
                    f"{PREFIX}-*commands.json", f"{PREFIX}-runner-status.json", f"{PREFIX}-run-panels.py",
                    "knn-digits-final-predeclaration-2026-10-30.json", "knn-digits-gap-*.json",
                    "knn-digits-gap-*.log", "knn-digits-test-matrix-*.json", "knn-digits-sk*-tests-2026-10-30.log",
                    "knn-digits-harness-tests-2026-10-30.log", "knn-digits-package-build-2026-10-30.log",
                    "knn-digits-candidate-2026-10-30-control.json", "knn-digits-candidate-2026-10-30-control.log"]:
        artifacts.update(RESULTS.glob(pattern))
    artifacts.update(ROOT / entry["path"] for entry in provenance["drivers"])
    if args.audit_execution_record:
        assert args.audit_execution_record.is_file()
        artifacts.add(args.audit_execution_record.resolve())
        audit_execution = read(args.audit_execution_record)
        assert audit_execution["status"] == "complete" and audit_execution["mode"] == "actual_execution"
        assert audit_execution["started_at"] >= runner["finished_at"]
        assert all(item["status"] == "complete" and item["returncode"] == 0
                   for item in audit_execution["commands"])
        assert audit_execution["report_driver_sha256"] == digest(driver)
        assert audit_execution["command_metadata_sha256"] == digest(audit_execution["command_metadata"])
        audit_plan = read(audit_execution["command_metadata"])
        for planned, actual in zip(audit_plan["commands"], audit_execution["commands"], strict=True):
            assert planned["argv"] == actual["declared_argv"]
            assert planned["argv"][1:] == actual["executed_argv"][1:]
    else:
        audit_execution = None
    artifacts.update(RESULTS.glob(f"{PREFIX}-*-report.log"))
    artifact_records = {relative(path): file_record(path) for path in sorted(artifacts)}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    post = {"schema_version": 1, "written_at": now, "integrity_pass": True, "archives": archives,
            "live_source_files": 63, "live_harness_files": 148, "live_runtime_and_build_files_match": True,
            "changed_production_files_vs_immediate_prior": changed, "gpu_device_files_unchanged": 21,
            "all_python_production_sources_unchanged": True, "package_source_files_checked": 63,
            "native_binary_sha256": EXPECTED["native"], "all_six_panels_completed_serially": True,
            "all_panel_fixture_source_harness_and_native_metadata_match": True,
            "unique_recorded_fixtures": len(fixtures), "tests_per_sklearn_version": 1474,
            "separate_harness_tests": 65, "distinct_current_tests": 1539,
            "new_benchmarks_or_production_mutations": False,
            "document_hashes_deferred_to_final_report_qa": True,
            "limits": ["This is a provenance and byte-identity audit, not an independent speedup claim.",
                       "Temporary NPZ fixtures were not regenerated; retained per-file/per-array hashes were cross-checked.",
                       "Native performance evidence concerns the recorded Apple M4 Max CPython3.13 arm64 build."]}
    if outputs["package-validation"].exists():
        assert read(outputs["package-validation"]) == package, "Previously recorded package contents changed"
    else:
        save_new(outputs["package-validation"], package)
    save_new(outputs["post-run-integrity"], post)
    for path in [outputs["package-validation"], outputs["post-run-integrity"]]:
        artifact_records[relative(path)] = file_record(path)
    current_worker = metadata["broad"]["workers"]["current"]
    execution = {"schema_version": 2, "written_at": now, "workspace": str(ROOT),
                 "git_head": source.get("revision"), "archives": archives,
                 "reference_archives": {"immediate_prior_source": prior_record,
                                        "immediate_prior_runtime": prior_runtime_record,
                                        "historical_gpu_source": old_gpu_record},
                 "hardware": current_worker["hardware"], "platform": metadata["broad"]["platform"],
                 "python": current_worker["python"],
                 "versions": {key: current_worker[key] for key in ["numpy", "scipy", "sklearn", "mlx"]},
                 "runtime_abi": runtime["abi"], "production_files_changed": changed,
                 "gpu_and_python_production_sources_unchanged": True,
                 "native_build": {**runtime["build"], "binary_sha256": EXPECTED["native"],
                                  "cpp_sha256": EXPECTED["cpp"], "force_flag_used": False,
                                  "task_environment_overrides": False,
                                  "capture_environment": environment["capture_process_environment"]},
                 "package_build_command": "python3 -m build --wheel --sdist --no-isolation --outdir /tmp/mlxlearn-knn-digits-final-build > benchmarks/results/knn-digits-package-build-2026-10-30.log 2>&1",
                 "harness_test_command": "python3 -m pytest -q benchmarks/test_knn_*.py > benchmarks/results/knn-digits-harness-tests-2026-10-30.log 2>&1",
                 "test_matrix_driver_command": "python3 benchmarks/results/knn-digits-run-test-matrix-2026-10-30.py",
                 "timing_runner_command": "python3 benchmarks/results/knn-digits-final-2026-10-30-run-panels.py",
                 "shell_command_provenance": "Build, harness, matrix-driver and timing-runner shell invocations confirmed by executing root agent; actual child argv are retained in completed status records.",
                 "package_validation": package, "serial_timing_order": ORDER,
                 "timing_command_provenance": {"planned": relative(commands_path), "actual": relative(runner_path),
                                               "executed_argv_and_zero_returncodes_verified": True},
                 "timing_runner": runner, "panels": panels,
                 "test_matrix": {"commands": relative(test_commands_path), "status": relative(test_status_path),
                                 "runs": test_results, "separate_harness_passed": 65,
                                 "distinct_current_tests": 1539,
                                 "counting_policy": "1474 current-version tests plus65 separate harness tests; cross-version repeats are not new distinct tests."},
                 "raw_input_fixture_hashes": fixtures,
                 "fixture_note": "Original temporary NPZ files are absent by design; file and array hashes retained in raw panels were cross-checked before/after and across shared cases.",
                 "independent_audit": {"path": relative(independent_path), "validation_pass": True},
                 "reporting_audit": {"path": relative(reporting_path), "driver": file_record(driver), "validation_pass": True},
                 "audit_execution_record": relative(args.audit_execution_record.resolve()) if args.audit_execution_record else None,
                 "audit_execution": audit_execution,
                 "probe_provenance": provenance,
                 "artifacts": dict(sorted(artifact_records.items())),
                 "documentation_hashes": "Deferred to root final report QA; no mutable report text or figure is frozen by this record.",
                 "limits": ["Integrity/accuracy acceptance does not mean all benchmarks are faster or universally 2x faster.",
                            "Component probes are development evidence and are not pooled with completed final panels.",
                            "Fresh interpreters do not imply cold OS, filesystem, bytecode or Metal caches.",
                            "The fallback Python engine is tested with native hooks disabled; a full non-NEON native build was not executed.",
                            "Direct sklearn concrete-class identity policy is unchanged from the immediate prior authorized startup revision."]}
    save_new(outputs["execution"], execution)
    print(json.dumps({"integrity_pass": True, "source_files": 63, "harness_files": 148,
                      "changed_production_files": changed, "fixtures": len(fixtures),
                      "outputs": {name: file_record(path) for name, path in outputs.items()}}, indent=2))


if __name__ == "__main__":
    main()
