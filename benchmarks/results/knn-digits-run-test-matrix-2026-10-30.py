"""Run the declared compatibility matrix serially, retaining every test log."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = ROOT / "benchmarks/results/knn-digits-test-matrix-commands-2026-10-30.json"
STATUS = ROOT / "benchmarks/results/knn-digits-test-matrix-status-2026-10-30.json"
rows = json.loads(COMMANDS.read_text())
if STATUS.exists() or any((ROOT / row["log"]).exists() for row in rows):
    raise FileExistsError("Refusing to replace retained test evidence")
state = {"status": "running", "runs": []}
for row in rows:
    run = {**row, "started_at": datetime.now(timezone.utc).isoformat()}
    state["runs"].append(run)
    STATUS.write_text(json.dumps(state, indent=2) + "\n")
    print("Starting", row["version"], flush=True)
    with (ROOT / row["log"]).open("x") as stream:
        result = subprocess.run(row["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    run.update(exit_code=result.returncode, finished_at=datetime.now(timezone.utc).isoformat())
    STATUS.write_text(json.dumps(state, indent=2) + "\n")
    print("Finished", row["version"], "exit", result.returncode, flush=True)
    if result.returncode:
        state["status"] = "failed"
        STATUS.write_text(json.dumps(state, indent=2) + "\n")
        raise SystemExit(result.returncode)
state["status"] = "complete"
STATUS.write_text(json.dumps(state, indent=2) + "\n")
