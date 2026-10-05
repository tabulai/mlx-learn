"""Cross-file invariants of the repository: packaging, README, workflows."""

from __future__ import annotations

import ast
import re

import pytest

from ._load import REPO, load_tool

check_environment = load_tool("check_environment")
GITHUB = "https://github.com/tabulai/mlx-learn/"
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))


def _pyproject():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ version


def test_the_version_has_one_source():
    project = _pyproject()
    assert "version" not in project["project"]
    assert project["project"]["dynamic"] == ["version"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "mlxlearn.__version__"}


def test_the_version_is_a_literal_setuptools_can_read_without_importing():
    # If __version__ became computed, setuptools would fall back to importing
    # mlxlearn — and with it MLX and scikit-learn — just to build the metadata.
    tree = ast.parse((REPO / "src" / "mlxlearn" / "__init__.py").read_text(encoding="utf-8"))
    literals = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and [getattr(t, "id", None) for t in node.targets] == ["__version__"]
        and isinstance(node.value, ast.Constant)
    ]
    assert len(literals) == 1 and isinstance(literals[0], str)


# ------------------------------------------------------------------------- README

_LINK = re.compile(r"\]\(([^)\s]+)\)")


def test_readme_links_are_absolute_because_it_is_also_the_pypi_page():
    targets = _LINK.findall((REPO / "README.md").read_text(encoding="utf-8"))
    relative = [t for t in targets if not t.startswith(("https://", "http://", "#", "mailto:"))]
    assert not relative, f"relative links 404 on pypi.org: {relative}"


def test_readme_links_into_the_repository_point_at_files_that_exist():
    targets = _LINK.findall((REPO / "README.md").read_text(encoding="utf-8"))
    ours = [t for t in targets if t.startswith(GITHUB + "blob/main/") or t.startswith(GITHUB + "tree/main/")]
    assert ours
    for target in ours:
        path = target.split("/main/", 1)[1].split("#", 1)[0]
        kind = "blob" if "/blob/main/" in target else "tree"
        assert (REPO / path).is_file() if kind == "blob" else (REPO / path).is_dir(), target


def test_the_readme_documents_the_public_backend_property():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "estimator.execution_backend_" in text
    assert re.search(r"[.`]_execution_backend_", text) is None, "README names the private attribute"


# ------------------------------------------------------------------ minimum versions


def _floor(spec: str) -> tuple[str, str]:
    name, floor = re.match(r"^\s*([A-Za-z0-9._-]+).*?>=\s*([0-9][^,;\s]*)", spec).groups()
    return name, floor


def test_the_minimum_versions_job_installs_exactly_the_declared_floors():
    from packaging.version import Version

    project = _pyproject()["project"]
    declared = dict(_floor(s) for s in project["dependencies"])
    declared.update(_floor(s) for s in project["optional-dependencies"]["test"])
    pins = check_environment.read_pins(REPO / ".github" / "constraints" / "minimum-versions.txt")
    assert set(pins) == set(declared)
    for name, floor in declared.items():
        assert Version(pins[name]) == Version(floor), (name, pins[name], floor)
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert ".github/constraints/minimum-versions.txt" in ci


def _ci_test_job() -> str:
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    return ci.split("\n  test:\n", 1)[1].split("\n  slow:\n", 1)[0]


def _ci_test_cells() -> list[dict[str, str]]:
    """The test job's matrix entries as ``{key: value}``, read without a YAML parser."""
    include = _ci_test_job().split("        include:\n", 1)[1].split("    steps:\n", 1)[0]
    cells: list[dict[str, str]] = []
    for line in include.splitlines():
        match = re.match(r'^\s*(- )?(\w+):\s*"?([^"#]*?)"?\s*$', line)
        if match is None:
            continue
        if match.group(1):
            cells.append({})
        cells[-1][match.group(2)] = match.group(3)
    return cells


def test_a_ci_cell_tests_the_newest_scikit_learn_the_range_admits():
    # pip keeps a pinned scikit-learn through a change to the declared range, so with
    # every cell pinned, a Dependabot PR raising the cap passed CI without one test
    # running on the release it let in.
    cells = _ci_test_cells()
    assert {"3.10", "3.14"} <= {cell["python"] for cell in cells}, cells
    assert any("sklearn" not in cell and "constraints" not in cell for cell in cells), cells
    assert 'elif [ -n "${{ matrix.sklearn }}" ]; then' in _ci_test_job()


def test_every_ci_test_run_installs_pandas():
    # The pandas cases in tests/unit/test_validation.py (sparse frames, per-column
    # range reads) skip without it, and nothing else pulls it in: before it joined the
    # test extra they skipped on every CI leg.
    from packaging.requirements import Requirement

    extras = _pyproject()["project"]["optional-dependencies"]
    for extra in ("test", "dev"):
        assert "pandas" in {Requirement(spec).name for spec in extras[extra]}, extra


# ---------------------------------------------------------------------- workflows


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_commit_with_its_version_named(workflow):
    for line in workflow.read_text(encoding="utf-8").splitlines():
        match = re.search(r"uses:\s*(\S+)(.*)$", line)
        if match is None or match.group(1).startswith("./"):
            continue
        ref = match.group(1).split("@", 1)[1]
        assert re.fullmatch(r"[0-9a-f]{40}", ref), f"{workflow.name}: not SHA-pinned: {line.strip()}"
        assert re.search(r"#\s*v\d+\.\d+\.\d+", match.group(2)), f"{workflow.name}: {line.strip()}"


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_workflow_limits_its_token_and_every_job_its_time(workflow):
    text = workflow.read_text(encoding="utf-8")
    assert re.search(r"^permissions:", text, re.M), f"{workflow.name}: no top-level permissions"
    jobs = re.split(r"^  (?=[A-Za-z0-9_-]+:\s*$)", text.split("\njobs:\n", 1)[1], flags=re.M)
    for job in filter(str.strip, jobs):
        if "runs-on:" in job:  # a reusable-workflow call may not set a timeout
            assert "timeout-minutes:" in job, f"{workflow.name}: {job.splitlines()[0]} has no timeout"


def test_only_a_tag_push_can_publish():
    release = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    publish = release.split("\n  publish:\n", 1)[1]
    assert "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')" in publish
    assert "needs: [ci, version]" in publish


def test_the_release_publishes_the_distributions_ci_checked():
    # release.yml used to build again and publish that, so what reached PyPI had never
    # been through CI's sdist checks.
    release = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python -m build" in ci and "python -m build" not in release
    uploaded = re.findall(r"upload-artifact@.*\n\s+with:\n\s+name: (\S+)", ci)
    downloaded = re.findall(r"download-artifact@.*\n\s+with:\n\s+name: (\S+)", release)
    assert "dist" in uploaded and set(downloaded) == {"dist"}, (uploaded, downloaded)


def test_ruff_is_pinned_once_in_the_dev_extra():
    dev = _pyproject()["project"]["optional-dependencies"]["dev"]
    assert any(re.fullmatch(r"ruff==\d+\.\d+\.\d+", spec) for spec in dev)


# ---------------------------------------------------------------- check_environment


def test_pins_are_read_from_a_constraints_file(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("# comment\nnumpy==1.23.5  # floor\nscipy>=1.10\n\nmlx == 0.31.2\n")
    assert check_environment.read_pins(path) == {"numpy": "1.23.5", "mlx": "0.31.2"}


def test_a_pin_that_did_not_install_exactly_is_reported():
    from importlib import metadata

    installed = metadata.version("numpy")
    assert check_environment.pin_mismatches({"numpy": installed}) == []
    (problem,) = check_environment.pin_mismatches({"numpy": "0.0.1"})
    assert "numpy" in problem and "0.0.1" in problem
    (problem,) = check_environment.pin_mismatches({"not-a-real-package-xyz": "1.0"})
    assert "not installed" in problem


def test_unmet_requirements_of_an_installed_distribution():
    # scikit-learn's own runtime requirements are necessarily met wherever it imports.
    assert check_environment.unmet_requirements("scikit-learn") == []
