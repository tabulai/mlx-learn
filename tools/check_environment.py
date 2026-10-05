#!/usr/bin/env python3
"""Print, and check, the environment a CI job is about to test.

A test run is only as meaningful as the versions and the device it ran on, and a
resolver or a runner image can change either without anyone noticing. This prints
both, then fails loudly when they are not what the job claims to test:

* every runtime requirement of the installed ``mlx-learn`` is satisfied — a
  ``--no-deps`` install or a quietly downgrading resolver would otherwise go unseen
  (``--ignore-requirements`` reports them instead, for the job that leaves the
  declared ranges on purpose);
* the installed metadata and the imported package agree on the version, which a
  stale install does not;
* with ``--constraints FILE``, every package pinned there with ``==`` is installed
  at exactly that version, which is how the minimum-versions job proves it tested
  the declared floors rather than whatever resolved;
* with ``--require-gpu``, or ``MLXLEARN_REQUIRE_GPU=1`` in the environment, a
  working MLX GPU device is present.

Usage::

    python tools/check_environment.py --require-gpu
    python tools/check_environment.py --constraints .github/constraints/minimum-versions.txt
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from importlib import metadata
from pathlib import Path

DISTRIBUTION = "mlx-learn"
REQUIRE_GPU_ENV = "MLXLEARN_REQUIRE_GPU"

_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)")


def read_pins(path: Path) -> dict[str, str]:
    """``{name: version}`` for every ``name==version`` line of a constraints file."""
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _PIN.match(line)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def unmet_requirements(distribution: str = DISTRIBUTION) -> list[str]:
    """Runtime requirements of ``distribution`` that the environment does not satisfy."""
    from packaging.requirements import Requirement

    problems: list[str] = []
    for spec in metadata.requires(distribution) or []:
        req = Requirement(spec)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            problems.append(f"{req.name}: required ({req.specifier}) but not installed")
            continue
        if not req.specifier.contains(installed, prereleases=True):
            problems.append(f"{req.name}: {installed} installed, {req.specifier} required")
    return problems


def pin_mismatches(pins: dict[str, str]) -> list[str]:
    from packaging.version import Version

    problems: list[str] = []
    for name, wanted in sorted(pins.items()):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name}: pinned to {wanted} but not installed")
            continue
        if Version(installed) != Version(wanted):
            problems.append(f"{name}: {installed} installed, constraints pin {wanted}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--constraints", type=Path, help="require exactly these == pins")
    parser.add_argument(
        "--ignore-requirements",
        action="store_true",
        help="report unmet requirements without failing, for a job that installs "
        "outside the declared ranges on purpose",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        default=os.environ.get(REQUIRE_GPU_ENV, "").strip() not in ("", "0"),
        help=f"fail without a working MLX GPU device (default: {REQUIRE_GPU_ENV})",
    )
    args = parser.parse_args(argv)

    import mlxlearn

    print(f"python {sys.version.split()[0]} on {sys.platform}")
    for name in (DISTRIBUTION, "mlx", "mlx-metal", "numpy", "scipy", "scikit-learn"):
        try:
            print(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"{name} not installed")
    print(f"mlxlearn.__version__ {mlxlearn.__version__}")
    print(f"device: {mlxlearn.device_info()}")

    problems: list[str] = []
    try:
        installed_version = metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        installed_version = None
        problems.append(f"{DISTRIBUTION} is not installed; there are no requirements to check")
    if installed_version is not None:
        for unmet in unmet_requirements():
            if args.ignore_requirements:
                print(f"note: {unmet}")
            else:
                problems.append(unmet)
        if installed_version != mlxlearn.__version__:
            problems.append(
                f"installed metadata says {DISTRIBUTION} {installed_version}, but the imported "
                f"package says {mlxlearn.__version__}: a stale install"
            )
    if args.constraints is not None:
        problems += pin_mismatches(read_pins(args.constraints))
    if args.require_gpu and not mlxlearn.gpu_available():
        problems.append("a working MLX GPU device is required, and none was found")

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
