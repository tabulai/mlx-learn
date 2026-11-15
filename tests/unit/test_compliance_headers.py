"""The header gate permits only explicitly recorded derived-file notices."""

from __future__ import annotations

import json

import pytest
from tools import compliance

SOURCE = "src/mlxlearn/_kernels/e_means_bf16.py"
SNAPSHOT = "benchmarks/e_means/results/optimization/bf16-dispatch-sweep.json"


def _header(year=2024, owner="Apple" + " Inc."):
    return f"Copyright © {year} {owner}"


def _repository(tmp_path, monkeypatch, *, path=SOURCE, text=None, notices=None):
    monkeypatch.setattr(compliance, "REPO", tmp_path)
    source = tmp_path / path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(text if text is not None else "# " + _header() + "\n")
    if notices is not None:
        (tmp_path / "THIRD_PARTY_NOTICES.md").write_text(notices)
    return source


def _notice(path=SOURCE, header=None, section=1):
    title = "Copied or derived source" if section == 1 else "Runtime dependencies"
    return f"## {section}. {title}\n\n`{path}` retains `{header or _header()}` under MIT.\n"


def test_exact_declared_source_and_documented_notice_pass(tmp_path, monkeypatch):
    source = _repository(tmp_path, monkeypatch, notices=_notice())
    result = compliance.check_headers([source])
    assert result.ok and result.checked == 1


@pytest.mark.parametrize("notices", [
    None, "## 1. Copied or derived source\nNo derived source.\n",
    _notice(path="src/mlxlearn/_kernels/unrelated.py"),
    _notice(header=_header(2025)), _notice(section=3),
])
def test_exact_path_allowlist_still_requires_actual_derived_source_notice(tmp_path, monkeypatch, notices):
    source = _repository(tmp_path, monkeypatch, notices=notices)
    result = compliance.check_headers([source])
    assert not result.ok
    assert "not recorded in THIRD_PARTY_NOTICES.md" in result.failures[0]


def test_documenting_sibling_does_not_create_blanket_directory_exemption(tmp_path, monkeypatch):
    path = "src/mlxlearn/_kernels/unlisted_derived.py"
    source = _repository(tmp_path, monkeypatch, path=path, notices=_notice(path=path))
    assert not compliance.check_headers([source]).ok


def test_file_cannot_borrow_header_from_unrelated_notice_entry(tmp_path, monkeypatch):
    notices = (
        f"## 1. Copied or derived source\n### Unrelated\n`other.py` retains `{_header()}`.\n"
        f"### Actual file\n`{SOURCE}` retains `{_header(2025)}`.\n"
    )
    source = _repository(tmp_path, monkeypatch, notices=notices)
    assert not compliance.check_headers([source]).ok


@pytest.mark.parametrize("extra", [_header(2025), _header(owner="NumPy Developers.")])
def test_declared_file_cannot_hide_an_additional_unrecorded_upstream_header(tmp_path, monkeypatch, extra):
    source = _repository(tmp_path, monkeypatch, text=_header() + "\n" + extra, notices=_notice())
    assert not compliance.check_headers([source]).ok


@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_recorded_json_source_snapshot_preserves_unicode_notice(tmp_path, monkeypatch, ensure_ascii):
    text = json.dumps({"experimental_source": "# " + _header() + "\npass\n"}, ensure_ascii=ensure_ascii)
    source = _repository(tmp_path, monkeypatch, path=SNAPSHOT, text=text, notices=_notice(path=SNAPSHOT))
    assert compliance.check_headers([source]).ok


def test_original_source_without_upstream_header_needs_no_notice(tmp_path, monkeypatch):
    source = _repository(tmp_path, monkeypatch, path="src/mlxlearn/original.py", text="def local():\n    return 1\n")
    assert compliance.check_headers([source]).ok
