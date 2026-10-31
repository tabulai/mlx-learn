"""Close the post-run record after its audit stdout log has finished writing."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "knn-digits-final-2026-10-30"
RESULTS = ROOT / "benchmarks/results"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


path = RESULTS / f"{PREFIX}-execution.json"
record = json.loads(path.read_text())
assert "integrity_record_finalization" not in record
log_relative = f"benchmarks/results/{PREFIX}-integrity-audit.log"
for name, entry in record["artifacts"].items():
    artifact = ROOT / name
    actual = digest(artifact)
    if name == log_relative:
        assert entry["bytes"] == 0, "Only the still-open audit log may change after capture"
        assert json.loads(artifact.read_text())["integrity_pass"] is True
        entry.update(sha256=actual, bytes=artifact.stat().st_size)
    else:
        assert actual == entry["sha256"] and artifact.stat().st_size == entry["bytes"], name
script_relative = str(Path(__file__).resolve().relative_to(ROOT))
record["artifacts"][script_relative] = {"path": script_relative,
    "sha256": digest(Path(__file__)), "bytes": Path(__file__).stat().st_size}
record["integrity_record_finalization"] = {
    "command": f"python3 {script_relative}",
    "scope": "Rehashed every recorded artifact after audit completion; only the initially empty audit stdout log required updating. No timing/source/package/probe evidence changed.",
    "audit_log_note": "The audit command's stdout reports the execution digest before this log-closure metadata step; the final execution digest is emitted by this finalizer and retained in final report QA.",
    "package_audit_command": f"python3 benchmarks/results/{PREFIX}-integrity-audit.py --package-only > benchmarks/results/{PREFIX}-package-integrity-audit.log 2>&1",
    "full_audit_command": f"python3 benchmarks/results/{PREFIX}-integrity-audit.py --audit-execution-record benchmarks/results/{PREFIX}-audit-execution.json > benchmarks/results/{PREFIX}-integrity-audit.log 2>&1",
}
record["artifacts"] = dict(sorted(record["artifacts"].items()))
path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
print(json.dumps({"final_execution_sha256": digest(path), "all_recorded_artifacts_rehashed": True,
                  "artifact_count": len(record["artifacts"]), "path": str(path)}, indent=2))
