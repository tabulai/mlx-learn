"""The patching contract.

Everything asserted here is a promise the README makes, including the one about
what patching *cannot* do.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import sklearn.linear_model
import sklearn.neighbors
import sklearn.svm

import mlxlearn
from mlxlearn.exceptions import MLXLearnPatchWarning
from mlxlearn.patching import get_patch_map, is_patched, patch_names, patched_estimators

# ----------------------------------------------------------------------------------
# the locked two-liner
# ----------------------------------------------------------------------------------


def test_locked_two_liner_runs_verbatim():
    """The exact script the refactoring plan locks. It runs in CI too."""
    script = textwrap.dedent(
        """
        from mlxlearn import patch_sklearn
        patch_sklearn()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr


def test_patching_is_quiet_by_default(capsys):
    """A library that prints on import corrupts someone's piped output."""
    mlxlearn.patch_sklearn()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_verbose_reports_to_stderr(capsys):
    mlxlearn.patch_sklearn(verbose=True)
    assert "mlxlearn" in capsys.readouterr().err


# ----------------------------------------------------------------------------------
# module attribute replacement
# ----------------------------------------------------------------------------------


def test_attribute_access_resolves_to_mlxlearn_after_patching():
    original = sklearn.svm.SVC
    mlxlearn.patch_sklearn()
    assert sklearn.svm.SVC is not original
    assert sklearn.svm.SVC.__module__.startswith("mlxlearn")


def test_import_before_patch_still_sees_the_patched_class():
    """`import sklearn.svm` first, then patch: attribute lookup happens at access."""
    import sklearn.svm as svm_module

    mlxlearn.patch_sklearn()
    assert svm_module.SVC.__module__.startswith("mlxlearn")


def test_symbols_bound_before_patching_cannot_be_rebound():
    """Documented Python semantics, asserted so the docs cannot drift from reality.

    ``from sklearn.svm import SVC`` creates a binding in *this* namespace. Nothing
    mlxlearn does can reach it. The README says the supported pattern is
    patch-first; this is the test that keeps that honest.
    """
    from sklearn.svm import SVC as captured_before

    mlxlearn.patch_sklearn()
    assert not captured_before.__module__.startswith("mlxlearn")
    assert sklearn.svm.SVC.__module__.startswith("mlxlearn")


def test_patch_then_import_gets_the_accelerated_class():
    mlxlearn.patch_sklearn()
    from sklearn.svm import SVC as captured_after

    assert captured_after.__module__.startswith("mlxlearn")


# ----------------------------------------------------------------------------------
# idempotence and reversibility
# ----------------------------------------------------------------------------------


def test_double_patch_is_a_no_op():
    """Crucially, the second patch must not record mlxlearn's class as the original."""
    original = sklearn.svm.SVC
    mlxlearn.patch_sklearn()
    patched = sklearn.svm.SVC
    mlxlearn.patch_sklearn()
    assert sklearn.svm.SVC is patched

    mlxlearn.unpatch_sklearn()
    assert sklearn.svm.SVC is original


def test_unpatch_restores_every_estimator():
    originals = {
        name: getattr(sys.modules[spec["module"]], str(spec["attribute"]))
        for name, spec in get_patch_map().items()
    }
    mlxlearn.patch_sklearn()
    mlxlearn.unpatch_sklearn()
    for name, spec in get_patch_map().items():
        assert getattr(sys.modules[spec["module"]], str(spec["attribute"])) is originals[name]


def test_double_unpatch_is_a_no_op():
    mlxlearn.patch_sklearn()
    mlxlearn.unpatch_sklearn()
    mlxlearn.unpatch_sklearn()
    assert not is_patched()


def test_selective_patching():
    mlxlearn.patch_sklearn("SVC")
    assert sklearn.svm.SVC.__module__.startswith("mlxlearn")
    assert not sklearn.neighbors.KNeighborsClassifier.__module__.startswith("mlxlearn")


def test_patching_one_estimator_patches_only_that_estimator():
    """The ancestor appended five config and parallelism hooks to every request.

    ``patch_sklearn("SVC")`` there also replaced ``sklearn.set_config``, which is
    a surprising amount of global state to change on behalf of one class.
    """
    original_set_config = sklearn.set_config
    mlxlearn.patch_sklearn("SVC")
    assert sklearn.set_config is original_set_config
    assert patched_estimators() == ["sklearn.svm.SVC"]


def test_dotted_and_bare_names_both_resolve():
    mlxlearn.patch_sklearn("sklearn.svm.SVC")
    assert is_patched("SVC")
    assert is_patched("sklearn.svm.SVC")


@pytest.mark.parametrize("action", ["patch", "unpatch"])
def test_unknown_name_raises_in_both_directions(action):
    """Symmetry. The ancestor raised on patch and returned silently on unpatch, so
    a typo left the process patched with nothing to indicate it."""
    call = mlxlearn.patch_sklearn if action == "patch" else mlxlearn.unpatch_sklearn
    with pytest.raises(ValueError, match="no accelerated implementation"):
        call("RandomForestClassifier")


def test_third_party_patch_is_not_clobbered():
    class Impostor:
        pass

    mlxlearn.patch_sklearn("SVC")
    sklearn.svm.SVC = Impostor
    try:
        with pytest.warns(MLXLearnPatchWarning, match="no longer the class mlxlearn installed"):
            mlxlearn.unpatch_sklearn("SVC")
        assert sklearn.svm.SVC is Impostor
    finally:
        # Restore by hand: mlxlearn deliberately refused to.
        import importlib

        importlib.reload(sklearn.svm)


def test_is_patched_uses_identity_not_a_module_string():
    """The ancestor substring-matched its own package name against ``__module__``."""

    class PretendMlxlearn:
        pass

    PretendMlxlearn.__module__ = "mlxlearn.patching._estimators"
    sklearn.svm.SVC = PretendMlxlearn
    try:
        assert not is_patched("SVC")
    finally:
        import importlib

        importlib.reload(sklearn.svm)


# ----------------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------------


def test_registry_contains_only_gated_estimators():
    """Nothing gets a registry entry before it passes the gates."""
    assert set(patch_names()) == {
        "sklearn.neighbors.NearestNeighbors",
        "sklearn.neighbors.KNeighborsClassifier",
        "sklearn.neighbors.KNeighborsRegressor",
        "sklearn.linear_model.LogisticRegression",
        "sklearn.svm.SVC",
    }


@pytest.mark.parametrize("name", ["SVR", "NuSVR", "NuSVC", "RandomForestClassifier", "KMeans"])
def test_deferred_and_out_of_scope_estimators_are_not_registered(name):
    """``SVR`` and ``NuSVR`` stay unregistered until they implement the true objective.

    The ancestor patched them with a ridge surrogate, so ``sklearn.svm.SVR`` solved
    a different optimization problem than its name promises.
    """
    assert name not in {n.rsplit(".", 1)[1] for n in patch_names()}


def test_patch_map_reports_state():
    before = get_patch_map()
    assert all(entry["patched"] is False for entry in before.values())
    mlxlearn.patch_sklearn()
    after = get_patch_map()
    assert all(entry["patched"] is True for entry in after.values())
    assert all(entry["original"] is not None for entry in after.values())


# ----------------------------------------------------------------------------------
# patched estimators behave
# ----------------------------------------------------------------------------------


def test_patched_estimator_falls_back_for_capability_mismatch(small_classification_data, recwarn):
    """Layer 2 falls back where Layer 1 raises; the diagnostics record says so."""
    X, y = small_classification_data
    mlxlearn.patch_sklearn()

    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree").fit(X, y)
    assert model.execution_backend_ == "sklearn"

    event = mlxlearn.get_last_backend_event()
    assert event is not None
    events = [e for e in mlxlearn.get_backend_diagnostics() if e.fallback]
    assert any("algorithm" in e.reason for e in events)


def test_layer1_raises_where_layer2_falls_back(small_classification_data):
    from mlxlearn.exceptions import UnsupportedParameterError
    from mlxlearn.neighbors import KNeighborsClassifier as StrictKNN

    X, y = small_classification_data
    with pytest.raises(UnsupportedParameterError):
        StrictKNN(n_neighbors=3, algorithm="kd_tree").fit(X, y)


def test_fallback_policy_raise_turns_a_fallback_into_an_error(small_classification_data):
    from mlxlearn.exceptions import CapabilityError

    X, y = small_classification_data
    mlxlearn.patch_sklearn()
    with mlxlearn.config_context(fallback_policy="raise"):
        with pytest.raises(CapabilityError, match="fallback_policy='raise'"):
            sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree").fit(X, y)


def test_fallback_policy_silent_still_records(small_classification_data):
    X, y = small_classification_data
    mlxlearn.patch_sklearn()
    with mlxlearn.config_context(fallback_policy="silent"):
        sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree").fit(X, y)
    assert any(e.fallback for e in mlxlearn.get_backend_diagnostics())


def test_fallback_warns_once_per_estimator_and_reason(small_classification_data):
    X, y = small_classification_data
    mlxlearn.patch_sklearn()
    with pytest.warns(mlxlearn.MLXLearnFallbackWarning):
        sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree").fit(X, y)

    import warnings

    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree").fit(X, y)
    assert not [w for w in second if issubclass(w.category, mlxlearn.MLXLearnFallbackWarning)]


def test_patched_estimator_pickles(classification_data):
    X, y = classification_data
    mlxlearn.patch_sklearn()
    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(model.predict(X[:20]), restored.predict(X[:20]))


def test_patched_class_reprs_with_the_sklearn_name():
    mlxlearn.patch_sklearn()
    assert repr(sklearn.svm.SVC()).startswith("SVC(")


def test_instances_survive_unpatching(classification_data):
    """Unpatching changes what a name resolves to, not what existing objects are."""
    X, y = classification_data
    mlxlearn.patch_sklearn()
    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    mlxlearn.unpatch_sklearn()
    assert model.predict(X[:5]).shape == (5,)


def test_isinstance_against_the_sklearn_class_still_holds():
    """Meta-estimators and user code do this; subclassing is what keeps it true."""
    mlxlearn.patch_sklearn()
    model = sklearn.neighbors.KNeighborsClassifier()
    assert isinstance(model, sklearn.neighbors.KNeighborsClassifier)


def test_patched_estimator_works_in_a_pipeline(classification_data):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = classification_data
    mlxlearn.patch_sklearn()
    pipeline = make_pipeline(
        StandardScaler(), sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    ).fit(X, y)
    assert pipeline.predict(X[:10]).shape == (10,)


def test_patched_estimator_works_in_grid_search(classification_data):
    from sklearn.model_selection import GridSearchCV

    X, y = classification_data
    mlxlearn.patch_sklearn()
    search = GridSearchCV(
        sklearn.neighbors.KNeighborsClassifier(),
        {"n_neighbors": [3, 5]},
        cv=2,
    ).fit(X[:1500], y[:1500])
    assert search.best_params_["n_neighbors"] in (3, 5)
