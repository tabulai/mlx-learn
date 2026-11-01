"""Verify refresh evidence is confined to its two intended directories."""
import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('sklearn_refresh_evidence', TOOLS / 'sklearn_refresh_evidence.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def test_refresh_paths_confined():
    prefix = 'benchmarks/results/sklearn-refresh-2026-10-31/'
    assert str(evidence.refresh_path(prefix + 'knn-broad.json')) == prefix + 'knn-broad.json'
    assert str(evidence.refresh_path(prefix + 'initial-lr-gradient-audit/lr-default.json')) == prefix + 'initial-lr-gradient-audit/lr-default.json'
    for name in (prefix + '../other', prefix + '/bad', prefix + 'other/file.json',
                 '/etc/passwd', prefix + 'file\\name', prefix + 'initial-lr-gradient-audit/../../x'):
        with pytest.raises(ValueError):
            evidence.refresh_path(name)
