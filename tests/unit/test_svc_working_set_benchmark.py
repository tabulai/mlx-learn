"""Source isolation and evidence gates; no performance assertions or timed fits."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks import svc_working_set as bench


def test_verified_reference_extraction_preserves_every_source(tmp_path):
    manifest = bench.extract_reference(tmp_path)
    assert len(manifest["files"]) >= 70
    for relative, digest in manifest["files"].items():
        assert bench.sha256(tmp_path / relative) == digest
    # Reusing an untouched extraction is safe; modified sources are not replaced.
    bench.extract_reference(tmp_path)
    altered = tmp_path / "src/mlxlearn/_solvers/native_smo.py"
    altered.write_text(altered.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="overwrite changed"):
        bench.extract_reference(tmp_path)
    assert altered.read_text().endswith("# changed\n")


def test_reference_archive_tamper_is_rejected_before_extraction(tmp_path):
    archive = tmp_path / "reference.zip"
    archive.write_bytes(bench.REFERENCE.read_bytes() + b"changed")
    archive.with_suffix(".manifest.json").write_text(bench.REFERENCE.with_suffix(".manifest.json").read_text())
    destination = tmp_path / "extract"
    with pytest.raises(ValueError, match="archive hash"):
        bench.extract_reference(destination, archive)
    assert not destination.exists()


def test_unrecorded_python_file_cannot_enter_cached_snapshot(tmp_path):
    manifest = bench.extract_reference(tmp_path)
    (tmp_path / "src/mlxlearn/_unrecorded.py").write_text("pass\n")
    with pytest.raises(ValueError, match="source inventory"):
        bench.verify_source_tree(tmp_path, manifest)


def test_baseline_cache_reuses_only_verified_build(tmp_path, monkeypatch):
    builds = []

    def fake_build(command, **kwargs):
        builds.append(command)
        assert command[0] == bench.sys.executable
        assert "--inplace" in command and "--force" in command
        assert kwargs["env"]["VECLIB_MAXIMUM_THREADS"] == "1"
        base = Path(kwargs["cwd"]) / "src/mlxlearn"
        for directory, name in (("_solvers", "_smo_native"), ("_kernels", "_svc_cpu_native")):
            (base / directory / f"{name}.test.so").write_bytes(name.encode())
        return SimpleNamespace(returncode=0, stdout="built", stderr="")

    monkeypatch.setattr(bench.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(bench.subprocess, "run", fake_build)
    _, first = bench.prepare_baseline(tmp_path)
    _, second = bench.prepare_baseline(tmp_path)
    assert not first["reused"]
    assert second["reused"]
    assert len(builds) == 1
    extension = next((tmp_path / "src").rglob("*.so"))
    extension.write_bytes(b"different binary")
    with pytest.raises(ValueError, match="extension hashes changed"):
        bench.prepare_baseline(tmp_path)


def test_optional_build_failure_does_not_silently_benchmark_portable_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="optional build skipped", stderr=""))
    with pytest.raises(RuntimeError, match="was not built"):
        bench.prepare_baseline(tmp_path)


def test_worker_environment_discards_inherited_import_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/wrong/current/package")
    monkeypatch.setenv("VECLIB_MAXIMUM_THREADS", "8")
    environment = bench.controlled_environment(tmp_path)
    paths = environment["PYTHONPATH"].split(os.pathsep)
    assert paths == [str(tmp_path.resolve() / "src"), str(bench.ROOT)]
    assert all(environment[name] == "1" for name in bench.THREAD_VARIABLES)
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_origin_check_catches_extension_leak_even_with_correct_top_package(tmp_path):
    source = tmp_path / "selected"
    modules = {
        "mlxlearn": SimpleNamespace(__file__=str(source / "src/mlxlearn/__init__.py")),
        "mlxlearn._solvers._smo_native": SimpleNamespace(__file__=str(tmp_path / "other/_smo_native.so")),
    }
    with pytest.raises(RuntimeError, match="wrong source"):
        bench.check_import_roots(source, modules)
    modules.pop("mlxlearn._solvers._smo_native")
    assert "mlxlearn" in bench.check_import_roots(source, modules)


def test_worker_provenance_rejects_different_source_or_binary(tmp_path):
    relative = "src/mlxlearn/_solvers/_smo_native.test.so"
    result = {"environment": {
        "source_sha256": {"a.py": "source"},
        "native_extensions": {"native": {"path": str(tmp_path / relative), "sha256": "binary"}},
    }}
    bench.verify_worker_provenance(result, tmp_path, {"a.py": "source"}, {relative: "binary"})
    with pytest.raises(ValueError, match="source hashes"):
        bench.verify_worker_provenance(result, tmp_path, {"a.py": "changed"}, {relative: "binary"})
    with pytest.raises(ValueError, match="extension hash"):
        bench.verify_worker_provenance(result, tmp_path, {"a.py": "source"}, {relative: "changed"})


def comparison_fixture(kernel="rbf", classes=2):
    target = np.arange(100) % classes
    if classes == 2:
        margins = (target * 2 - 1).astype(float)
    else:
        margins = np.arange(100 * classes * (classes - 1) // 2, dtype=float).reshape(100, -1) / 100
    case = ("test", kernel, 100, 8, classes, "cpu", 2.0)
    results = {}
    for variant in bench.VARIANTS:
        trials = []
        for index in range(3):
            trials.append({
                "fit_s": 100.0 if index == 0 else index / 100,
                "first_predict_s": 90.0 if index == 0 else index / 1000,
                "repeat_predict_s": 80.0 if index == 0 else index / 10000,
                "prediction": target.tolist(), "first_prediction": target.tolist(),
                "margin": margins.tolist(), "model": {
                    "fit_status": 0, "n_iter": [2], "n_support": [20] * classes,
                    "support_count": 20 * classes, "backend": "cpu",
                    "pair_solver": ["native"], "pair_kkt_violation": [0.0001],
                },
            })
        results[variant] = {
            "variant": variant, "case": "test", "seed": 41, "repeats": 2,
            "fixture_sha256": "fixture", "trials": trials, "environment": {},
        }
    return case, results, target


def test_reports_first_and_repeat_predictions_separately_without_startup_in_medians():
    case, results, target = comparison_fixture()
    report = bench.compare_case(case, results, target)
    assert report["quality_passed"]
    assert report["first_call"]["current"]["fit_s"] == 100
    assert report["fit"]["current"]["samples_s"] == [0.01, 0.02]
    assert report["first_predict"]["current"]["samples_s"] == [0.001, 0.002]
    assert report["repeat_predict"]["current"]["samples_s"] == [0.0001, 0.0002]
    assert report["fit"]["current"]["median_ms"] == pytest.approx(15)
    assert "trials" not in report["workers"]["current"]
    assert "prediction" not in report["quality_samples_including_first"]["current"][0]
    json.dumps(report)


@pytest.mark.parametrize("variant", bench.VARIANTS)
@pytest.mark.parametrize("trial", [0, 1, 2])
def test_any_nonconverged_trial_fails_gate(variant, trial):
    case, results, target = comparison_fixture()
    results[variant]["trials"][trial]["model"]["fit_status"] = 1
    assert not bench.compare_case(case, results, target)["quality_passed"]


def test_matching_predictions_cannot_hide_wrong_continuous_margins():
    case, results, target = comparison_fixture(classes=4)
    results["current"]["trials"][1]["margin"][0][0] += 0.1
    assert not bench.compare_case(case, results, target)["quality_passed"]


def test_multiclass_gate_rejects_ovr_shaped_data():
    case, results, target = comparison_fixture(classes=4)
    for result in results.values():
        for trial in result["trials"]:
            trial["margin"] = np.zeros((100, 4)).tolist()
    with pytest.raises(ValueError, match="OVO margin shapes"):
        bench.compare_case(case, results, target)


def test_sigmoid_relaxed_label_gate_does_not_relax_margin_or_first_prediction_checks():
    case, results, target = comparison_fixture("sigmoid")
    for trial in results["current"]["trials"]:
        trial["prediction"][:4] = (1 - target[:4]).tolist()
        trial["first_prediction"][:4] = trial["prediction"][:4]
    assert bench.compare_case(case, results, target)["quality_passed"]
    results["current"]["trials"][0]["margin"][0] = 100
    assert not bench.compare_case(case, results, target)["quality_passed"]


def test_quality_compares_sklearn_even_if_current_and_previous_agree():
    case, results, target = comparison_fixture()
    for variant in ("current", "previous"):
        for trial in results[variant]["trials"]:
            trial["prediction"][:3] = (1 - target[:3]).tolist()
            trial["first_prediction"][:3] = trial["prediction"][:3]
    assert not bench.compare_case(case, results, target)["quality_passed"]


@pytest.mark.parametrize("change", ["fixture", "seed", "count", "identity", "timing"])
def test_mismatched_worker_results_rejected(change):
    case, results, target = comparison_fixture()
    altered = results["previous"]
    if change == "fixture":
        altered["fixture_sha256"] = "different"
    elif change == "seed":
        altered["seed"] = 42
    elif change == "count":
        altered["repeats"] = 5
    elif change == "identity":
        altered["variant"] = "current"
    else:
        altered["trials"][0]["fit_s"] = float("nan")
    with pytest.raises(ValueError):
        bench.compare_case(case, results, target)


def test_nonfinite_margin_and_changed_repeated_prediction_fail():
    case, results, target = comparison_fixture()
    changed = copy.deepcopy(results)
    changed["current"]["trials"][1]["margin"][0] = float("nan")
    assert not bench.compare_case(case, changed, target)["quality_passed"]
    results["current"]["trials"][1]["first_prediction"][0] = 99
    assert not bench.compare_case(case, results, target)["quality_passed"]


def test_default_smoke_and_optional_medium_case_selection():
    assert len(bench.select_cases(None, False)) == 7
    assert len(bench.select_cases(None, True)) == 2
    assert bench.select_cases(["linear_medium,rbf_narrow"], False)[0][2:4] == (2500, 32)
    with pytest.raises(ValueError):
        bench.select_cases(["linear"], True)
    with pytest.raises(ValueError):
        bench.select_cases(["linear", "linear"], False)


def test_fixture_is_float32_and_reproducible(tmp_path):
    # Data generation is independent of estimator variant and outside timing.
    case = ("fixture", "rbf", 12, 8, 2, "cpu", 1.0)
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    bench.make_fixture(case, 42, first)
    bench.make_fixture(case, 42, second)
    with np.load(first) as left, np.load(second) as right:
        assert left["train"].dtype == np.float32
        assert left["query"].shape == (257, 8)
        for name in left.files:
            np.testing.assert_array_equal(left[name], right[name])


def test_driver_import_does_not_load_either_estimator(tmp_path):
    result = bench.subprocess.run(
        [bench.sys.executable, "-c", "import sys; import benchmarks.svc_working_set; assert 'mlxlearn' not in sys.modules; assert 'numpy' not in sys.modules; assert 'mlx.core' not in sys.modules"],
        cwd=tmp_path, env=bench.controlled_environment(tmp_path),
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_launch_uses_selected_package_in_an_isolated_process(tmp_path, monkeypatch):
    source = tmp_path / "baseline"
    output = tmp_path / "worker.json"

    def fake_run(command, **kwargs):
        assert command[:3] == [bench.sys.executable, "-m", "benchmarks.svc_working_set"]
        assert command[command.index("--source-root") + 1] == str(source)
        assert command[command.index("--variant") + 1] == "previous"
        assert kwargs["cwd"] == tmp_path
        assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(source / "src")
        output.write_text(json.dumps({"variant": "previous"}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    result = bench.launch_worker(bench.CASES[0], "previous", source, tmp_path / "input.npz", output, 5, 41)
    assert result["variant"] == "previous"
    assert "--worker" in result["worker_command"]
