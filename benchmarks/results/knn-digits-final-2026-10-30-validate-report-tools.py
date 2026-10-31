"""Synthetic CPU-only checks for reporting tools; never launches benchmark workers."""

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PREFIX = "knn-digits-final-2026-10-30"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def check_plan(plan, declaration):
    assert [p["name"] for p in plan["panels"]] == [p["name"] for p in declaration["panels"]]
    assert len(plan["panels"]) == 6
    for panel in plan["panels"]:
        argv = panel["argv"]
        assert argv[argv.index("--output") + 1] == panel["output"]
        assert argv[argv.index("--source-manifest") + 1] == plan["source_manifest"]
        if panel["name"] != "broad":
            assert argv[argv.index("--previous-auto-archive") + 1] == declaration["baseline_roles"]["previous_auto_source"]
            assert argv[argv.index("--previous-auto-runtime") + 1] == declaration["baseline_roles"]["previous_auto_runtime"]
    for repeated in declaration["runs"]:
        panel = next(p for p in plan["panels"] if p["name"] == repeated["name"])
        selected = [panel["argv"][i + 1] for i, value in enumerate(panel["argv"]) if value == "--case"]
        assert selected == repeated["case_names"]
        assert len(selected) == repeated["case_count"] == 9


def check_runner(runner, plan, *, fail_at=None, existing=False, change_archive=False):
    with tempfile.TemporaryDirectory(prefix="knn-report-tools-test-") as directory:
        root = Path(directory)
        (root / "benchmarks/results").mkdir(parents=True)
        commands, status = root / "commands.json", root / "status.json"
        commands.write_text(json.dumps(plan))
        (root / plan["predeclaration"]).write_text("{}")
        calls, checks = [], []

        def frozen(_):
            checks.append(True)
            return {"synthetic_snapshot": "changed" if change_archive and len(checks) >= 3 else "unchanged"}

        def invoke(argv, *, cwd, stdout, stderr, check):
            calls.append(argv)
            assert cwd == root and check is False
            output = root / argv[argv.index("--output") + 1]
            output.write_text('{"synthetic_test_only": true}')
            stdout.write(b"Synthetic test; no child process launched.\n")
            return SimpleNamespace(returncode=7 if len(calls) == fail_at else 0)

        if existing:
            (root / plan["panels"][0]["output"]).write_text("preserve me")
        with patch.object(runner, "ROOT", root), patch.object(runner, "COMMANDS", commands), \
             patch.object(runner, "STATUS", status), patch.object(runner, "verify_frozen", frozen), \
             patch.object(runner.subprocess, "run", invoke), contextlib.redirect_stdout(io.StringIO()):
            if existing:
                try:
                    runner.main()
                except FileExistsError:
                    pass
                else:
                    raise AssertionError("Runner overwrote existing evidence")
                assert not calls and not status.exists()
                assert (root / plan["panels"][0]["output"]).read_text() == "preserve me"
                return
            result = runner.main()
            saved = json.loads(status.read_text())
            expected_calls = 1 if change_archive else fail_at or 6
            assert len(calls) == expected_calls == len(saved["panels"])
            expected_status = "error" if change_archive else "failed" if fail_at else "complete"
            assert saved["status"] == expected_status
            assert result == (1 if change_archive else 7 if fail_at else 0)
            assert all(p["started_at"] and p["finished_at"] for p in saved["panels"])
            for panel in plan["panels"][expected_calls:]:
                assert not (root / panel["output"]).exists()
                assert not (root / panel["log"]).exists()
            try:
                runner.main()
            except FileExistsError:
                pass
            else:
                raise AssertionError("Runner resumed or overwrote a completed/failed run")
            assert len(calls) == expected_calls


def check_acceptance(driver, declaration):
    policy = declaration["digits_acceptance"]
    cells = []
    for seed in policy["seeds"]:
        for operation in policy["operations"]:
            cells.append({"case": f"real_digits_raw_k5_seed{seed}", "family": "real_digits_raw_k5", "seed": seed,
                          "operation": operation,
                          "median_ms": {"current": 1, "current_b": 1, "current_gpu": 1.2, "previous_gpu": 1.2},
                          "comparisons": {b: {r: {m: 1.2 for m in policy["statistics"]}
                                               for r in policy["replicas"]} for b in policy["comparators"]}})
    screens = [driver.digits_screen(p, cells, declaration) for p in policy["protocols"]]
    assert sum(s["comparison_count"] for s in screens) == 432
    assert sum(s["margin_comparison_count"] for s in screens) == 48
    assert all(s["all_comparisons_pass"] and s["development_margin_target_pass"] for s in screens)
    changed = copy.deepcopy(cells)
    target = next(c for c in changed if c["seed"] == 252 and c["operation"] == "regress_distance")
    for metric in policy["statistics"]:
        target["comparisons"]["previous_gpu"]["current"][metric] = .99
    screen = driver.digits_screen("legacy_control", changed, declaration)
    assert not screen["all_comparisons_pass"] and len(screen["failures"]) == 2
    assert screen["passed_comparisons"] == 214
    for replica in policy["replicas"]:
        for metric in policy["statistics"]:
            target["comparisons"]["previous_gpu"][replica][metric] = 1.05
    screen = driver.digits_screen("legacy_control", changed, declaration)
    assert screen["all_comparisons_pass"] and not screen["development_margin_target_pass"]
    assert len(screen["margin_failures"]) == 4
    try:
        driver.digits_screen("legacy_control", cells[:-1], declaration)
    except AssertionError:
        pass
    else:
        raise AssertionError("Acceptance allowed a missing operation")


def main():
    plan = json.loads((HERE / (PREFIX + "-commands.json")).read_text())
    declaration = json.loads((HERE / "knn-digits-final-predeclaration-2026-10-30.json").read_text())
    check_plan(plan, declaration)
    runner = module("digits_runner_test", HERE / (PREFIX + "-run-panels.py"))
    check_runner(runner, plan)
    check_runner(runner, plan, fail_at=3)
    check_runner(runner, plan, existing=True)
    check_runner(runner, plan, change_archive=True)
    driver = module("digits_audit_test", HERE / (PREFIX + "-report-audit.py"))
    check_acceptance(driver, declaration)
    print(json.dumps({"validation_pass": True, "real_subprocesses_launched": 0,
                      "checks": ["command_inventory", "serial_success", "fail_stop", "no_overwrite", "frozen_change_stop",
                                 "all432_acceptance_checks", "all48_margin_checks", "miss_retention", "margin_separation", "missing_cell_rejection"]}))


if __name__ == "__main__":
    main()
