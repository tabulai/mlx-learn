#!/usr/bin/env python3
"""Mechanical compliance gates for mlxlearn.

These checks are necessary conditions for a release, never sufficient ones: the
human items live in ``phase0/legal_review_checklist.md``.

Checks
------
branding
    No Intel / oneDAL / sklearnex branding anywhere in the tree. The notices and
    provenance files are exempt, because removing a vendor's branding is not the
    same as removing a vendor's attribution.
private-imports
    No ``sklearn._*`` private imports outside the quarantined ``patching/``
    package, and inside it only the explicitly allowlisted ones.
headers
    No file carries an upstream copyright header that ``THIRD_PARTY_NOTICES.md``
    does not account for.
licenses
    Every runtime dependency's license is on the allowlist.
layout
    The package layout matches the one the plan fixes, and no compiled artifact
    or generated C source is tracked.

Run all of them::

    python tools/compliance.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------------------

#: Terms that must not appear in mlxlearn's own sources or docs.
BANNED_TERMS = (
    "sklearnex",
    "scikit-learn-intelex",
    "scikit_learn_intelex",
    "daal4py",
    "onedal",
    "oneDAL",
    "Intel(R)",
    "mpsbackend",
    "SKMPS_",
)

#: Files allowed to mention the banned terms, because their job is to record lineage.
BRANDING_EXEMPT = {
    "ACKNOWLEDGMENTS.md",
    "THIRD_PARTY_NOTICES.md",
    "CHANGELOG.md",
    "tools/compliance.py",
    "phase0/README.md",
    "phase0/attestation.md",
    "phase0/audit_policy.json",
    "phase0/cython_decision.md",
    "phase0/legal_review_checklist.md",
    "phase0/generated/audit_summary.md",
    "phase0/generated/estimator_support.csv",
    "phase0/generated/provenance.csv",
    "phase0/generated/source_snapshot.json",
    "phase0/baseline/benchmark.log",
    "phase0/baseline/benchmark_results.json",
    "phase0/baseline/environment.json",
    "phase0/baseline/manifest.json",
    "phase0/baseline/summary.md",
    "phase0/baseline/test.log",
    "tools/phase0_audit.py",
    "tools/run_phase0_baseline.py",
    # Measures the ancestor read-only and must name its environment variable to
    # toggle the setting under test. Exempt for the same reason the notices files
    # are: recording lineage is not the same as carrying branding.
    "tools/measure_native_core.py",
    "docs/provenance.md",
}

#: The only private scikit-learn imports mlxlearn may use, and only from ``patching/``.
#: Each entry needs a justification, because every one of them is a maintenance liability.
ALLOWED_PRIVATE_SKLEARN_IMPORTS: dict[str, str] = {}

#: Licenses acceptable in the runtime dependency closure.
ALLOWED_LICENSES = (
    "apache",
    "bsd",
    "mit",
    "psf",
    "python software foundation",
    "isc",
    "mpl-2.0",
    "unlicense",
    "public domain",
)

RUNTIME_DEPENDENCIES = ("numpy", "scikit-learn", "mlx")

#: Upstream copyright signatures that would require an entry in THIRD_PARTY_NOTICES.md.
UPSTREAM_HEADER_PATTERNS = (
    re.compile(r"Copyright\s*\(c\)\s*\d{4}.*Intel", re.IGNORECASE),
    re.compile(r"Copyright\s+\d{4}\s+Intel", re.IGNORECASE),
    re.compile(r"Copyright.*Fujitsu", re.IGNORECASE),
    re.compile(r"Copyright.*scikit-learn developers", re.IGNORECASE),
    re.compile(r"Copyright.*NumPy Developers", re.IGNORECASE),
    re.compile(r"Copyright.*Apple Inc", re.IGNORECASE),
)

HEADER_EXEMPT = {
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "ACKNOWLEDGMENTS.md",
    "tools/compliance.py",
}

REQUIRED_PACKAGES = (
    "src/mlxlearn/__init__.py",
    "src/mlxlearn/_common/__init__.py",
    "src/mlxlearn/_kernels/__init__.py",
    "src/mlxlearn/_solvers/__init__.py",
    "src/mlxlearn/linear_model/__init__.py",
    "src/mlxlearn/neighbors/__init__.py",
    "src/mlxlearn/svm/__init__.py",
    "src/mlxlearn/patching/__init__.py",
)

FORBIDDEN_TRACKED_SUFFIXES = (".so", ".dylib", ".pyd", ".o", ".a")

TEXT_SUFFIXES = {
    ".py", ".md", ".toml", ".cfg", ".txt", ".yml", ".yaml", ".json", ".csv", ".log", ".ini",
}


# --------------------------------------------------------------------------------------
# machinery
# --------------------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    failures: list[str] = field(default_factory=list)
    checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures

    def report(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        head = f"[{mark}] {self.name} ({self.checked} checked)"
        if self.ok:
            return head
        body = "\n".join(f"       - {line}" for line in self.failures)
        return f"{head}\n{body}"


def tracked_files() -> list[Path]:
    """Every file git tracks, plus untracked-but-not-ignored files.

    Compliance must see a file the moment it is written, not only once it is
    committed, or the gate is trivially bypassed by not committing yet.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO / line for line in out.stdout.splitlines() if line]


def read_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------


def check_branding(paths: list[Path]) -> Result:
    result = Result("branding")
    for path in paths:
        rel = path.relative_to(REPO).as_posix()
        if rel in BRANDING_EXEMPT or rel.startswith("phase0/"):
            continue
        text = read_text(path)
        if text is None:
            continue
        result.checked += 1
        for term in BANNED_TERMS:
            if term in text:
                line_no = next(
                    (i for i, line in enumerate(text.splitlines(), 1) if term in line), 0
                )
                result.failures.append(f"{rel}:{line_no}: banned term {term!r}")
    return result


def _private_sklearn_imports(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("sklearn.") and "._" in f".{alias.name}":
                    parts = alias.name.split(".")
                    if any(p.startswith("_") for p in parts[1:]):
                        found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "sklearn" or module.startswith("sklearn."):
                parts = module.split(".")
                if any(p.startswith("_") for p in parts[1:]):
                    found.append((node.lineno, module))
                else:
                    for alias in node.names:
                        if alias.name.startswith("_"):
                            found.append((node.lineno, f"{module}.{alias.name}"))
    return found


def check_private_imports(paths: list[Path]) -> Result:
    result = Result("private-imports")
    for path in paths:
        if path.suffix != ".py":
            continue
        rel = path.relative_to(REPO).as_posix()
        if not rel.startswith("src/mlxlearn/"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, OSError) as exc:
            result.failures.append(f"{rel}: unparseable ({exc})")
            continue
        result.checked += 1
        quarantined = rel.startswith("src/mlxlearn/patching/")
        for lineno, name in _private_sklearn_imports(tree):
            if not quarantined:
                result.failures.append(
                    f"{rel}:{lineno}: private import {name!r} outside patching/"
                )
            elif name not in ALLOWED_PRIVATE_SKLEARN_IMPORTS:
                result.failures.append(
                    f"{rel}:{lineno}: private import {name!r} is not on the allowlist"
                )
    return result


def check_headers(paths: list[Path]) -> Result:
    result = Result("headers")
    for path in paths:
        rel = path.relative_to(REPO).as_posix()
        if rel in HEADER_EXEMPT or rel.startswith("phase0/"):
            continue
        text = read_text(path)
        if text is None:
            continue
        result.checked += 1
        head = "\n".join(text.splitlines()[:60])
        for pattern in UPSTREAM_HEADER_PATTERNS:
            if pattern.search(head):
                result.failures.append(
                    f"{rel}: upstream copyright header not recorded in THIRD_PARTY_NOTICES.md "
                    f"({pattern.pattern})"
                )
    return result


def check_licenses(_paths: list[Path]) -> Result:
    from importlib import metadata

    result = Result("licenses")
    for name in RUNTIME_DEPENDENCIES:
        try:
            meta = metadata.metadata(name)
        except metadata.PackageNotFoundError:
            result.failures.append(f"{name}: not installed, cannot verify license")
            continue
        result.checked += 1
        declared = meta.get("License-Expression") or meta.get("License") or ""
        classifiers = [c for c in meta.get_all("Classifier") or [] if c.startswith("License")]
        blob = f"{declared} {' '.join(classifiers)}".lower()
        if not any(token in blob for token in ALLOWED_LICENSES):
            result.failures.append(
                f"{name}: license {declared or classifiers or 'UNKNOWN'!r} not on the allowlist"
            )
    return result


def check_layout(paths: list[Path]) -> Result:
    result = Result("layout")
    relatives = {p.relative_to(REPO).as_posix() for p in paths}
    for required in REQUIRED_PACKAGES:
        result.checked += 1
        if required not in relatives:
            result.failures.append(f"missing required module {required}")
    for rel in sorted(relatives):
        if Path(rel).suffix in FORBIDDEN_TRACKED_SUFFIXES:
            result.failures.append(f"{rel}: compiled artifact must never be tracked")
        if rel.startswith("src/") and rel.endswith(".c"):
            result.failures.append(f"{rel}: generated C must never be tracked")
    return result


CHECKS = {
    "branding": check_branding,
    "private-imports": check_private_imports,
    "headers": check_headers,
    "licenses": check_licenses,
    "layout": check_layout,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checks", nargs="*", choices=[*CHECKS, []], help="default: all")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    selected = args.checks or list(CHECKS)
    paths = tracked_files()
    results = [CHECKS[name](paths) for name in selected]

    if args.json:
        print(
            json.dumps(
                {r.name: {"ok": r.ok, "checked": r.checked, "failures": r.failures} for r in results},
                indent=2,
            )
        )
    else:
        for r in results:
            print(r.report())

    failed = [r.name for r in results if not r.ok]
    if failed:
        print(f"\ncompliance FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\ncompliance passed (mechanical checks only; see phase0/legal_review_checklist.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
