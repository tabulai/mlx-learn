#!/usr/bin/env python3
"""Generate a reproducible Phase 0 source and provenance inventory."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import subprocess
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "phase0" / "audit_policy.json"
DEFAULT_OUTPUT = ROOT / "phase0" / "generated"
TEXT_SUFFIXES = {
    "",
    ".c",
    ".cfg",
    ".cpp",
    ".css",
    ".csv",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".pxd",
    ".pyx",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
UPSTREAM_MARKERS = (
    "Intel Corporation",
    "Fujitsu Limited",
    "oneAPI Data Analytics Library",
    "daal4py",
)


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def tracked_files(repo: Path) -> list[str]:
    output = run_git(repo, "ls-files", "-z")
    return sorted(path for path in output.split("\0") if path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 2_000_000:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def normalized_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    # Provenance review is interested in retained source lines, not character
    # edit distance. Line-level matching is also bounded for generated C files.
    return SequenceMatcher(
        None,
        left.splitlines(),
        right.splitlines(),
        autojunk=True,
    ).ratio()


def matching_upstream_path(relative: str, upstream: Path) -> Path | None:
    exact = upstream / relative
    if exact.is_file():
        return exact
    return None


def category(relative: str) -> str:
    path = Path(relative)
    if path.suffix in {".so", ".dylib", ".pyd", ".dll"}:
        return "compiled-binary"
    if relative.startswith("legacy/"):
        return "legacy"
    if relative.startswith("mpsbackend/"):
        return "backend"
    if relative.startswith("sklearnex/"):
        return "compatibility"
    if relative.startswith("tests/"):
        return "tests"
    if relative.startswith("benchmarks/") or relative.startswith("benchmark_results/"):
        return "benchmarks"
    if relative.startswith(("doc/", "examples/")):
        return "docs-examples"
    if path.name in {"pyproject.toml", "setup.py", "setup.cfg", "Makefile", "justfile"}:
        return "build"
    return "repository"


def migration_decision(relative: str, policy: dict[str, Any]) -> tuple[str, str]:
    for rule in policy["migration_rules"]:
        if fnmatch.fnmatchcase(relative, rule["glob"]):
            return rule["disposition"], rule["reason"]
    raise RuntimeError(f"No migration rule matched {relative}")


def provisional_provenance(
    *,
    relative: str,
    text: str | None,
    upstream_relative: str,
    similarity: float | None,
    threshold: float,
) -> tuple[str, str]:
    path = Path(relative)
    if path.suffix in {".so", ".dylib", ".pyd", ".dll"}:
        return "generated-binary", "Compiled artifact."
    if path.suffix == ".c" and relative.startswith("mpsbackend/native/"):
        return "generated-source", "Generated C beside a Cython source."
    markers = [marker for marker in UPSTREAM_MARKERS if text and marker in text]
    if markers:
        return "derived", f"Contains upstream marker(s): {', '.join(markers)}."
    if similarity is not None and similarity >= threshold:
        return (
            "derived",
            f"Text similarity {similarity:.3f} to {upstream_relative}.",
        )
    if relative.startswith("mpsbackend/"):
        return (
            "original-candidate",
            "Backend path with no automated upstream derivation signal.",
        )
    if relative.startswith(("sklearnex/", "legacy/")):
        return (
            "review-required",
            "Compatibility or legacy path lacks conclusive automated evidence.",
        )
    return (
        "project-authored-or-administrative",
        "No automated upstream derivation signal.",
    )


def must_review(disposition: str, provenance: str) -> bool:
    if disposition in {"drop", "generated-do-not-copy", "behavioral-reference"}:
        return False
    return disposition in {"port-candidate", "rewrite", "review"} or provenance in {
        "derived",
        "original-candidate",
        "review-required",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-revision-mismatch", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    upstream = args.upstream.resolve()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_revision = run_git(source, "rev-parse", "HEAD")
    expected_revision = policy["source_revision"]
    if source_revision != expected_revision and not args.allow_revision_mismatch:
        raise SystemExit(
            f"Source revision {source_revision} does not match frozen "
            f"revision {expected_revision}."
        )

    source_files = tracked_files(source)
    upstream_files = set(tracked_files(upstream))
    rows: list[dict[str, Any]] = []
    for relative in source_files:
        source_path = source / relative
        source_text = read_text(source_path)
        upstream_path = matching_upstream_path(relative, upstream)
        upstream_relative = (
            upstream_path.relative_to(upstream).as_posix() if upstream_path else ""
        )
        similarity: float | None = None
        if upstream_path and source_text is not None:
            upstream_text = read_text(upstream_path)
            if upstream_text is not None:
                similarity = normalized_similarity(source_text, upstream_text)
        disposition, disposition_reason = migration_decision(relative, policy)
        provenance, evidence = provisional_provenance(
            relative=relative,
            text=source_text,
            upstream_relative=upstream_relative,
            similarity=similarity,
            threshold=float(policy["similarity_threshold"]),
        )
        review = policy.get("reviewed_files", {}).get(relative)
        review_required = must_review(disposition, provenance)
        review_status = (
            review["status"]
            if review
            else ("pending" if review_required else "not-required")
        )
        rows.append(
            {
                "path": relative,
                "sha256": sha256(source_path),
                "bytes": source_path.stat().st_size,
                "lines": source_text.count("\n") + 1 if source_text is not None else "",
                "category": category(relative),
                "upstream_path": upstream_relative,
                "upstream_similarity": (
                    f"{similarity:.6f}" if similarity is not None else ""
                ),
                "provisional_provenance": provenance,
                "evidence": evidence,
                "migration_disposition": disposition,
                "disposition_reason": disposition_reason,
                "human_review_required": str(review_required).lower(),
                "human_review_status": review_status,
                "human_review_note": review["note"] if review else "",
            }
        )

    fields = [
        "path",
        "sha256",
        "bytes",
        "lines",
        "category",
        "upstream_path",
        "upstream_similarity",
        "provisional_provenance",
        "evidence",
        "migration_disposition",
        "disposition_reason",
        "human_review_required",
        "human_review_status",
        "human_review_note",
    ]
    write_csv(output_dir / "provenance.csv", rows, fields)

    estimator_rows = policy["estimators"]
    estimator_fields = [
        "estimator",
        "source",
        "release",
        "disposition",
        "implementation",
        "required_gates",
    ]
    write_csv(output_dir / "estimator_support.csv", estimator_rows, estimator_fields)

    status = run_git(source, "status", "--porcelain")
    snapshot = {
        "schema_version": 1,
        "source": {
            "path": str(source),
            "revision": source_revision,
            "remote": run_git(source, "remote", "get-url", "origin"),
            "working_tree_clean": not bool(status),
            "tracked_files": len(source_files),
        },
        "upstream": {
            "path": str(upstream),
            "revision": run_git(upstream, "rev-parse", "HEAD"),
            "remote": run_git(upstream, "remote", "get-url", "origin"),
            "tracked_files": len(upstream_files),
        },
        "policy_sha256": sha256(args.policy.resolve()),
        "provenance_counts": dict(
            sorted(Counter(row["provisional_provenance"] for row in rows).items())
        ),
        "disposition_counts": dict(
            sorted(Counter(row["migration_disposition"] for row in rows).items())
        ),
        "human_review_pending": sum(
            row["human_review_status"] == "pending" for row in rows
        ),
    }
    (output_dir / "source_snapshot.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )

    pending_ports = [
        row
        for row in rows
        if row["migration_disposition"] == "port-candidate"
        and row["human_review_status"] == "pending"
    ]
    knn_pending = [
        row for row in pending_ports if row["path"].startswith("mpsbackend/neighbors/")
    ]
    baseline_manifest = ROOT / "phase0" / "baseline" / "manifest.json"
    baseline_captured = baseline_manifest.is_file()
    summary = [
        "# Phase 0 audit summary",
        "",
        f"- Frozen source revision: `{source_revision}`",
        f"- Source working tree clean: `{not bool(status)}`",
        f"- Tracked source files inventoried: **{len(rows)}**",
        f"- Public estimators classified: **{len(estimator_rows)}**",
        f"- Files with pending human review: **{snapshot['human_review_pending']}**",
        f"- Pending KNN port-candidate reviews: **{len(knn_pending)}**",
        "",
        "## Provisional provenance",
        "",
    ]
    for key, value in snapshot["provenance_counts"].items():
        summary.append(f"- `{key}`: {value}")
    summary.extend(["", "## Migration dispositions", ""])
    for key, value in snapshot["disposition_counts"].items():
        summary.append(f"- `{key}`: {value}")
    summary.extend(
        [
            "",
            "## Gate status",
            "",
            "- [x] Frozen revision verified",
            "- [x] Every tracked file inventoried",
            "- [x] Estimator support matrix generated",
            (
                "- [x] Historical correctness and performance baseline captured"
                if baseline_captured
                else "- [ ] Historical correctness and performance baseline captured"
            ),
            (
                "- [x] Technical provenance review completed for first KNN port candidates"
                if not knn_pending
                else "- [ ] Technical provenance review completed for first KNN port candidates"
            ),
            "",
            "Automated classifications are evidence, not legal conclusions. "
            "See `phase0/README.md`.",
            "",
        ]
    )
    (output_dir / "audit_summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    print(f"Inventoried {len(rows)} files at {source_revision}.")
    print(f"Wrote Phase 0 audit artifacts to {output_dir}.")
    print(f"Pending human review: {snapshot['human_review_pending']} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
