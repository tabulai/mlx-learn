"""The public surface, pinned.

Growing or shrinking what ``import mlxlearn`` exposes should be a decision made in
a diff, not a side effect noticed by a user. And ``import mlxlearn`` must stay
cheap: a user reading ``__version__`` should not pay for scikit-learn or Metal.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import mlxlearn
from mlxlearn import exceptions

EXPECTED_ALL = sorted(
    [
        # patching
        "patch_sklearn",
        "unpatch_sklearn",
        "is_patched",
        "get_patch_map",
        "ESTIMATOR_REGISTRY",
        # configuration
        "set_config",
        "get_config",
        "config_context",
        "reset_config",
        "Config",
        # diagnostics
        "get_backend_diagnostics",
        "get_last_backend_event",
        "clear_backend_diagnostics",
        "reset_fallback_warnings",
        "diagnostics_summary",
        "BackendEvent",
        # environment
        "device_info",
        "gpu_available",
        # errors
        "MLXLearnError",
        "CapabilityError",
        "UnsupportedParameterError",
        "UnsupportedInputError",
        "SparseInputError",
        "InvalidInputError",
        "DeviceError",
        "BackendExecutionError",
        "MLXLearnFallbackWarning",
        "MLXLearnPerformanceWarning",
        "MLXLearnPatchWarning",
        "__version__",
    ]
)

EXPECTED_EXCEPTIONS = sorted(
    [
        "MLXLearnError",
        "CapabilityError",
        "UnsupportedParameterError",
        "UnsupportedInputError",
        "SparseInputError",
        "InvalidInputError",
        "UnsupportedEstimatorStateError",
        "DeviceError",
        "BackendExecutionError",
        "MLXLearnFallbackWarning",
        "MLXLearnPerformanceWarning",
        "MLXLearnPatchWarning",
    ]
)

SUBPACKAGES = {
    "linear_model": ["LogisticRegression"],
    "neighbors": ["KNeighborsClassifier", "KNeighborsRegressor", "NearestNeighbors"],
    "svm": ["SVC"],
}


def test_top_level_all_is_pinned():
    assert sorted(mlxlearn.__all__) == EXPECTED_ALL
    assert len(set(mlxlearn.__all__)) == len(mlxlearn.__all__)


def test_every_exported_name_resolves():
    for name in mlxlearn.__all__:
        assert getattr(mlxlearn, name) is not None, name


def test_exported_errors_are_the_exceptions_module_classes():
    """Including SparseInputError, which every Layer 1 estimator raises on sparse input."""
    for name in exceptions.__all__:
        if name in mlxlearn.__all__:
            assert getattr(mlxlearn, name) is getattr(exceptions, name), name
    assert mlxlearn.SparseInputError is exceptions.SparseInputError


def test_exceptions_all_is_pinned():
    assert sorted(exceptions.__all__) == EXPECTED_EXCEPTIONS
    for name in exceptions.__all__:
        assert issubclass(getattr(exceptions, name), (exceptions.MLXLearnError, UserWarning))


def test_dir_lists_the_subpackages():
    listing = dir(mlxlearn)
    for name in [*SUBPACKAGES, "exceptions", *mlxlearn.__all__]:
        assert name in listing, name


@pytest.mark.parametrize("subpackage", sorted(SUBPACKAGES))
def test_subpackages_are_attributes(subpackage):
    module = getattr(mlxlearn, subpackage)
    assert module.__name__ == f"mlxlearn.{subpackage}"
    for name in SUBPACKAGES[subpackage]:
        assert hasattr(module, name)


def test_unknown_attribute_is_an_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
        mlxlearn.nonexistent
    assert not hasattr(mlxlearn, "patching_internals")


def test_import_is_light_and_subpackages_load_on_first_access():
    """Nothing heavy at import, and ``mlxlearn.svm`` works without importing it first."""
    script = textwrap.dedent(
        """
        import sys
        import mlxlearn
        from mlxlearn import *  # noqa: F403 - must not pull in the subpackages either
        heavy = [m for m in ("sklearn", "mlx.core") if m in sys.modules]
        assert not heavy, f"import mlxlearn imported {heavy}"
        assert "svm" in dir(mlxlearn)
        assert mlxlearn.svm.SVC.__module__ == "mlxlearn.svm._classes"
        assert mlxlearn.neighbors.KNeighborsClassifier is not None
        assert mlxlearn.linear_model.LogisticRegression is not None
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
