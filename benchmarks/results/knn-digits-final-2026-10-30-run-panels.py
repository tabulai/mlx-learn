"""Run the declared six panels serially, preserving logs and stopping on failure."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "knn-digits-final-2026-10-30"
COMMANDS = Path(__file__).with_name(PREFIX + "-commands.json")
STATUS = Path(__file__).with_name(PREFIX + "-runner-status.json")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status_write(state):
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, STATUS)


def verify_frozen(plan):
    hashes = {}
    for relative in plan["required_frozen_archives"]:
        path = ROOT / relative
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        actual = digest(path)
        if actual != manifest["archive"]["sha256"]:
            raise ValueError(f"Frozen archive hash differs: {relative}")
        hashes[relative] = actual
        hashes[str(manifest_path.relative_to(ROOT))] = digest(manifest_path)
    runtime = json.loads((ROOT / plan["current_runtime_manifest"]).read_text())
    source = json.loads((ROOT / plan["source_manifest"]).read_text())
    if runtime["source_files"] != source["files"] or runtime["source_archive"]["sha256"] != source["archive"]["sha256"]:
        raise ValueError("Current native runtime and source snapshots differ")
    current_abi = {"implementation": sys.implementation.name, "cache_tag": sys.implementation.cache_tag,
                   "SOABI": sysconfig.get_config_var("SOABI"), "EXT_SUFFIX": sysconfig.get_config_var("EXT_SUFFIX"),
                   "machine": platform.machine(), "system": platform.system()}
    if any(runtime["abi"][key] != value for key, value in current_abi.items()):
        raise ValueError("Runner interpreter ABI differs from the frozen native runtime")
    return hashes


def main():
    if STATUS.exists():
        raise FileExistsError("Runner status already exists; this runner never resumes or overwrites a run")
    plan = json.loads(COMMANDS.read_text())
    if plan["status"] != "prepared_not_executed" or plan["prefix"] != PREFIX:
        raise ValueError("Unexpected prepared command metadata")
    expected_order = ["cold", "broad", "control", "legacy-control", "preservation-repeat", "first-query"]
    if [p["name"] for p in plan["panels"]] != expected_order:
        raise ValueError("Declared serial panel order changed")
    for panel in plan["panels"]:
        for field in ("output", "log"):
            if (ROOT / panel[field]).exists():
                raise FileExistsError(f"Refusing to overwrite existing {field}: {panel[field]}")
        if panel["argv"][panel["argv"].index("--output") + 1] != panel["output"]:
            raise ValueError("Command output differs from declared result path")
    frozen = verify_frozen(plan)
    state = {"schema_version": 1, "status": "running", "started_at": stamp(), "cwd": str(ROOT),
             "python": sys.executable, "runner_sha256": digest(__file__), "commands_sha256": digest(COMMANDS),
             "predeclaration_sha256": digest(ROOT / plan["predeclaration"]), "frozen_hashes_before": frozen,
             "panels": []}
    status_write(state)
    result = 1
    try:
        for panel in plan["panels"]:
            if digest(COMMANDS) != state["commands_sha256"] or digest(__file__) != state["runner_sha256"]:
                raise ValueError("Runner/command metadata changed during execution")
            if digest(ROOT / plan["predeclaration"]) != state["predeclaration_sha256"] or verify_frozen(plan) != frozen:
                raise ValueError("Predeclaration or frozen archives changed during execution")
            if (ROOT / panel["output"]).exists():
                raise FileExistsError("Result appeared before its panel; refusing to overwrite")
            argv = [sys.executable, *panel["argv"][1:]]
            item = {"name": panel["name"], "status": "running", "started_at": stamp(),
                    "declared_argv": panel["argv"], "executed_argv": argv,
                    "output": panel["output"], "log": panel["log"]}
            state["panels"].append(item)
            status_write(state)
            print(json.dumps({"panel": panel["name"], "status": "running", "started_at": item["started_at"]}), flush=True)
            with (ROOT / panel["log"]).open("xb") as log:
                completed = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            item.update(returncode=completed.returncode, finished_at=stamp(),
                        status="complete" if completed.returncode == 0 else "failed")
            status_write(state)
            print(json.dumps({"panel": panel["name"], "status": item["status"], "returncode": completed.returncode}), flush=True)
            if completed.returncode != 0:
                state["status"] = "failed"
                result = completed.returncode
                break
        else:
            state["frozen_hashes_after"] = verify_frozen(plan)
            if state["frozen_hashes_after"] != frozen:
                raise ValueError("Frozen archive hashes changed")
            state["status"] = "complete"
            result = 0
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        if state["panels"] and state["panels"][-1]["status"] == "running":
            state["panels"][-1].update(status="interrupted", finished_at=stamp())
        result = 130
    except Exception as error:
        state.update(status="error", error=f"{type(error).__name__}: {error}")
        if state["panels"] and state["panels"][-1]["status"] == "running":
            state["panels"][-1].update(status="error", finished_at=stamp())
        result = 1
    finally:
        state["finished_at"] = stamp()
        status_write(state)
        print(json.dumps({"runner_status": state["status"], "status_file": str(STATUS)}), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
