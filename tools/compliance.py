#!/usr/bin/env python3
"""Mechanical compliance gates for mlxlearn.

These checks cover the repository's mechanically enforceable compliance policy.

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
    Every package in the runtime dependency *closure* — the dependencies declared
    in ``pyproject.toml`` and everything they pull in, as installed — carries a
    license on the allowlist. SPDX expressions are parsed, not substring-matched.
layout
    The package layout matches the one the plan fixes, and no compiled artifact
    or generated C source is tracked.
layering
    ``_solvers`` and ``_kernels`` never import scikit-learn, never touch the user
    configuration (``get_config`` and friends), and never import the estimator
    layer; nothing in the package seeds a global random generator.

Run all of them, or name the ones to run::

    python tools/compliance.py
    python tools/compliance.py licenses layering --json

With ``--json`` the only thing on stdout is the JSON document; pass or fail is the
exit code.
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
    "tools/phase0_audit.py",
    # Measures the ancestor read-only and must name its environment variable to
    # toggle the setting under test. Exempt for the same reason the notices files
    # are: recording lineage is not the same as carrying branding.
    "tools/measure_native_core.py",
}

#: Directories whose every file records lineage. Listed once here rather than file by
#: file, so a new audit artifact cannot be forgotten and a stale path cannot linger.
BRANDING_EXEMPT_DIRS = ("phase0/",)

#: The only private scikit-learn imports mlxlearn may use, and only from ``patching/``.
#: Each entry needs a justification, because every one of them is a maintenance liability.
ALLOWED_PRIVATE_SKLEARN_IMPORTS: dict[str, str] = {}

#: SPDX license identifiers acceptable anywhere in the runtime dependency closure,
#: compared exactly (case-insensitively), never as substrings: a substring match
#: accepted "Proprietary; unlicensed" because it contains "unlicense". Zlib and
#: CC0-1.0 are here because NumPy vendors components under them.
ALLOWED_SPDX = frozenset(
    s.lower()
    for s in (
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "0BSD",
        "MIT",
        "MIT-0",
        "ISC",
        "PSF-2.0",
        "Python-2.0",
        "MPL-2.0",
        "Unlicense",
        "CC0-1.0",
        "Zlib",
    )
)

#: Trove classifiers accepted when a package declares no SPDX expression. Several
#: classifiers on one package conventionally mean "any of these", so one match is
#: enough.
ALLOWED_CLASSIFIERS = frozenset(
    {
        "License :: OSI Approved :: Apache Software License",
        "License :: OSI Approved :: BSD License",
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: MIT No Attribution License (MIT-0)",
        "License :: OSI Approved :: ISC License (ISCL)",
        "License :: OSI Approved :: Python Software Foundation License",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "License :: OSI Approved :: The Unlicense (Unlicense)",
        "License :: OSI Approved :: zlib/libpng License",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
    }
)

#: Free-text values of the legacy ``License`` field that name an allowed license
#: unambiguously, mapped to their SPDX identifier. Only consulted for a short,
#: single-line field; a field holding a whole license text is not parsed at all.
LICENSE_ALIASES = {
    "apache 2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "bsd 3-clause": "BSD-3-Clause",
    "bsd 3-clause license": "BSD-3-Clause",
    "new bsd": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause",
    "mit license": "MIT",
}

#: Reviewed licenses for packages whose metadata cannot be read mechanically, keyed
#: by canonical project name: ``{name: (SPDX expression, justification)}``. Like the
#: private-import allowlist, every entry is a maintenance liability and needs a reason.
LICENSE_OVERRIDES: dict[str, tuple[str, str]] = {}

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

#: The numerical core: typed arrays and frozen dataclasses in and out, testable
#: without an estimator. ``_kernels`` may read ``get_tuning()`` for block sizes —
#: private, typed knobs — but neither package reads the *user's* configuration.
LAYERED_PACKAGES = ("src/mlxlearn/_solvers/", "src/mlxlearn/_kernels/")

#: Names whose appearance in the numerical core means it is reading, or scoping, the
#: user's configuration instead of taking what it needs as an argument.
USER_CONFIG_NAMES = frozenset({"get_config", "set_config", "config_context"})

#: Subpackages the numerical core must not import: the estimator layer and Layer 2.
ESTIMATOR_PACKAGES = frozenset({"neighbors", "linear_model", "svm", "patching"})

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
        if rel in BRANDING_EXEMPT or rel.startswith(BRANDING_EXEMPT_DIRS):
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


class LicenseExpressionError(ValueError):
    """An SPDX license expression that does not parse."""


_SPDX_TOKEN = re.compile(r"\s*(\(|\)|[A-Za-z0-9.+:-]+)")


def _tokenize_spdx(expression: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    text = expression.strip()
    while position < len(text):
        match = _SPDX_TOKEN.match(text, position)
        if match is None:
            raise LicenseExpressionError(f"unexpected {text[position:]!r} in {expression!r}")
        tokens.append(match.group(1))
        position = match.end()
    if not tokens:
        raise LicenseExpressionError("empty license expression")
    return tokens


def spdx_allows(expression: str, allowed: frozenset[str] = ALLOWED_SPDX) -> bool:
    """Whether an SPDX license expression is acceptable under ``allowed``.

    ``A OR B`` needs one acceptable branch, since the licensee may choose; ``A AND B``
    needs every operand, since all of them apply. ``A WITH exception`` is judged on
    ``A``: an exception only ever grants additional permissions. ``AND`` binds
    tighter than ``OR``, as the SPDX specification defines.

    Raises
    ------
    LicenseExpressionError
        If the expression does not parse. A malformed expression is not evidence
        of an acceptable license.
    """
    tokens = _tokenize_spdx(expression)
    position = 0

    def peek() -> str | None:
        return tokens[position] if position < len(tokens) else None

    def take() -> str:
        nonlocal position
        if position >= len(tokens):
            raise LicenseExpressionError(f"unexpected end of {expression!r}")
        token = tokens[position]
        position += 1
        return token

    def parse_or() -> bool:
        ok = parse_and()
        while (peek() or "").upper() == "OR":
            take()
            ok = parse_and() or ok
        return ok

    def parse_and() -> bool:
        ok = parse_with()
        while (peek() or "").upper() == "AND":
            take()
            ok = parse_with() and ok
        return ok

    def parse_with() -> bool:
        ok = parse_atom()
        if (peek() or "").upper() == "WITH":
            take()
            exception = take()
            if exception in ("(", ")") or exception.upper() in ("AND", "OR", "WITH"):
                raise LicenseExpressionError(f"missing exception name in {expression!r}")
        return ok

    def parse_atom() -> bool:
        token = take()
        if token == "(":
            ok = parse_or()
            if take() != ")":
                raise LicenseExpressionError(f"unbalanced parentheses in {expression!r}")
            return ok
        if token == ")" or token.upper() in ("AND", "OR", "WITH"):
            raise LicenseExpressionError(f"unexpected {token!r} in {expression!r}")
        return token.lower() in allowed

    verdict = parse_or()
    if position != len(tokens):
        raise LicenseExpressionError(f"trailing {tokens[position]!r} in {expression!r}")
    return verdict


def license_verdict(name: str, meta) -> tuple[bool, str]:
    """Decide one distribution's license from its metadata: ``(acceptable, what was read)``.

    The sources are consulted from most to least precise, and the first one present
    decides: a reviewed override, the SPDX ``License-Expression``, the ``License ::``
    classifiers, and only then a short, single-line legacy ``License`` field. A
    field holding a full license text (SciPy's is 47 KB and names the GPL runtime
    libraries it bundles) is never scanned for words, because scanning prose is how
    both false passes and false failures happen.
    """
    override = LICENSE_OVERRIDES.get(name)
    if override is not None:
        expression, _justification = override
        return spdx_allows(expression), f"reviewed override {expression!r}"

    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        try:
            return spdx_allows(expression), f"License-Expression {expression!r}"
        except LicenseExpressionError as exc:
            return False, f"unparseable License-Expression ({exc})"

    classifiers = [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        return any(c in ALLOWED_CLASSIFIERS for c in classifiers), f"classifiers {classifiers}"

    legacy = (meta.get("License") or "").strip()
    if legacy and "\n" not in legacy and len(legacy) <= 80:
        alias = LICENSE_ALIASES.get(" ".join(legacy.lower().split()))
        if alias is not None:
            return True, f"License {legacy!r} (= {alias})"
        try:
            return spdx_allows(legacy), f"License {legacy!r}"
        except LicenseExpressionError:
            return False, f"License {legacy!r} is neither an SPDX expression nor a known name"
    return False, "no machine-readable license; add a reviewed entry to LICENSE_OVERRIDES"


def declared_runtime_dependencies(pyproject: Path | None = None) -> list[str]:
    """The ``[project] dependencies`` of ``pyproject.toml``, as requirement strings.

    Read from the source tree rather than from installed metadata, which in an
    editable install is frozen at install time and silently goes stale. Parsed with a
    regex because ``tomllib`` does not exist on Python 3.10.
    """
    text = (pyproject or REPO / "pyproject.toml").read_text(encoding="utf-8")
    project = re.search(r"^\[project\]\s*$(.*?)(?=^\[)", text, re.M | re.S)
    if project is None:
        raise ValueError("pyproject.toml has no [project] table")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", project.group(1), re.M | re.S)
    if block is None:
        raise ValueError("pyproject.toml [project] declares no dependencies list")
    lines = (line.split("#", 1)[0] for line in block.group(1).splitlines())
    return re.findall(r"""["']([^"']+)["']""", "\n".join(lines))


def runtime_closure(roots: list[str]) -> tuple[dict[str, object], list[str]]:
    """Walk installed metadata from ``roots``: ``({name: metadata}, missing)``.

    Environment markers are evaluated for the running interpreter and platform, so
    the closure is the one this environment would actually install — ``mlx-metal``
    is in it on macOS and not elsewhere. Extras are followed only where a
    requirement asks for them.
    """
    from importlib import metadata

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    found: dict[str, object] = {}
    missing: list[str] = []
    queue: list[tuple[str, frozenset[str]]] = []
    for spec in roots:
        req = Requirement(spec)
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            queue.append((req.name, frozenset(req.extras)))
    seen: set[tuple[str, frozenset[str]]] = set()
    while queue:
        raw_name, extras = queue.pop()
        name = canonicalize_name(raw_name)
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            if name not in missing:
                missing.append(name)
            continue
        found[name] = dist.metadata
        for spec in dist.requires or []:
            req = Requirement(spec)
            active = req.marker is None or any(
                req.marker.evaluate({"extra": extra}) for extra in (extras or {""})
            )
            if active:
                queue.append((req.name, frozenset(req.extras)))
    return found, missing


def check_licenses(_paths: list[Path]) -> Result:
    result = Result("licenses")
    try:
        closure, missing = runtime_closure(declared_runtime_dependencies())
    except ImportError as exc:
        result.failures.append(
            f"cannot evaluate dependency markers without 'packaging' ({exc}); "
            "pip install packaging"
        )
        return result
    for name in missing:
        result.failures.append(f"{name}: not installed, cannot verify license")
    for name, meta in sorted(closure.items()):
        result.checked += 1
        acceptable, source = license_verdict(name, meta)
        if not acceptable:
            result.failures.append(f"{name}: {source} is not on the allowlist")
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


def _attribute_chain(node: ast.AST) -> list[str]:
    """``mx.random.seed`` -> ``["mx", "random", "seed"]``; ``[]`` if not a plain chain."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return []


def layering_violations(tree: ast.AST, *, numerical_core: bool) -> list[tuple[int, str]]:
    """Static layering violations in one module's AST.

    Static because it has to be: importing ``mlxlearn._solvers`` runs the package
    ``__init__``, which loads scikit-learn through the estimators, so a runtime
    ``sys.modules`` check could never pass. What it cannot see is an indirect read,
    such as a solver calling ``use_device(None)``, which resolves the configured
    device; solvers pass their config's device explicitly instead.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if numerical_core and top == "sklearn":
                    found.append((node.lineno, f"imports scikit-learn ({alias.name})"))
                if numerical_core and alias.name.startswith("mlxlearn."):
                    if alias.name.split(".")[1] in ESTIMATOR_PACKAGES:
                        found.append((node.lineno, f"imports the estimator layer ({alias.name})"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            parts = module.split(".") if module else []
            if numerical_core and node.level == 0 and parts[:1] == ["sklearn"]:
                found.append((node.lineno, f"imports scikit-learn ({module})"))
            if numerical_core:
                # Relative imports from the numerical core reach the estimators as
                # ``..neighbors``; absolute ones as ``mlxlearn.neighbors``.
                package = parts[1] if node.level == 0 and parts[:1] == ["mlxlearn"] else None
                if node.level >= 2 and parts:
                    package = parts[0]
                if package in ESTIMATOR_PACKAGES:
                    found.append((node.lineno, f"imports the estimator layer ({module})"))
                for alias in node.names:
                    if alias.name in USER_CONFIG_NAMES:
                        found.append((node.lineno, f"imports {alias.name} from {module or '.'}"))
                    if node.level >= 2 and not parts and alias.name in ESTIMATOR_PACKAGES:
                        found.append((node.lineno, f"imports the estimator layer ({alias.name})"))
            if module in ("mlx.core.random", "numpy.random", "random") and any(
                alias.name == "seed" for alias in node.names
            ):
                found.append((node.lineno, f"imports the global-RNG seed from {module}"))
        elif isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if chain[-2:] == ["random", "seed"]:
                found.append((node.lineno, f"seeds a global random generator ({'.'.join(chain)})"))
            elif numerical_core and node.attr in USER_CONFIG_NAMES:
                found.append((node.lineno, f"reads the user configuration ({node.attr})"))
        elif isinstance(node, ast.Name) and numerical_core and node.id in USER_CONFIG_NAMES:
            found.append((node.lineno, f"reads the user configuration ({node.id})"))
    return sorted(set(found))


def check_layering(paths: list[Path]) -> Result:
    result = Result("layering")
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
        core = rel.startswith(LAYERED_PACKAGES)
        for lineno, what in layering_violations(tree, numerical_core=core):
            result.failures.append(f"{rel}:{lineno}: {what}")
    return result


CHECKS = {
    "branding": check_branding,
    "private-imports": check_private_imports,
    "headers": check_headers,
    "licenses": check_licenses,
    "layout": check_layout,
    "layering": check_layering,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # No ``choices=``: with ``nargs="*"``, argparse on Python <= 3.11 validates the
    # empty default against the choices and rejects a bare run. Validated below.
    parser.add_argument(
        "checks", nargs="*", metavar="CHECK", help=f"default: all of {', '.join(CHECKS)}"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print only a JSON document on stdout; pass or fail is the exit code",
    )
    args = parser.parse_args(argv)
    unknown = sorted(set(args.checks) - CHECKS.keys())
    if unknown:
        parser.error(f"unknown check(s): {', '.join(unknown)}; choose from {', '.join(CHECKS)}")

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

    # The verdict line goes to stderr in --json mode, so stdout stays one parseable
    # document; the exit code carries the verdict either way.
    failed = [r.name for r in results if not r.ok]
    verdict_stream = sys.stderr if args.json or failed else sys.stdout
    if failed:
        print(f"\ncompliance FAILED: {', '.join(failed)}", file=verdict_stream)
        return 1
    print("\ncompliance passed (mechanical checks only)", file=verdict_stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
