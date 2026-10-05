"""The patching contract.

Everything asserted here is a promise the README makes, including the one about
what patching *cannot* do.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pickle
import subprocess
import sys
import textwrap
import types
import warnings

import numpy as np
import pytest
import sklearn.linear_model
import sklearn.neighbors
import sklearn.svm

import mlxlearn
from mlxlearn.exceptions import MLXLearnPatchWarning
from mlxlearn.patching import _estimators as layer2
from mlxlearn.patching import _patch, get_patch_map, is_patched, patch_names, patched_estimators

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
        importlib.reload(sklearn.svm)


# ----------------------------------------------------------------------------------
# warnings: never mid-loop, and attributed to the caller
# ----------------------------------------------------------------------------------


class _Impostor:
    """Another library's replacement for a scikit-learn class."""


def test_unpatch_completes_under_warnings_as_errors():
    """Under ``-W error`` (or pytest's ``filterwarnings = error``) the tamper warning
    is an exception. Raised mid-loop, it left every later estimator patched and the
    stale entry in place, so every retry raised again at the same spot."""
    stock = sklearn.neighbors.NearestNeighbors
    mlxlearn.patch_sklearn()
    sklearn.neighbors.NearestNeighbors = _Impostor  # first in the registry
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(MLXLearnPatchWarning, match="NearestNeighbors"):
                mlxlearn.unpatch_sklearn()
            assert patched_estimators() == []
            assert sklearn.svm.SVC.__module__.startswith("sklearn")
            assert sklearn.neighbors.NearestNeighbors is _Impostor
            mlxlearn.unpatch_sklearn()  # the entry is gone: a silent no-op now
    finally:
        sklearn.neighbors.NearestNeighbors = stock


def test_patch_completes_under_warnings_as_errors():
    """The mirror case: the refused estimator raises, the rest are still patched."""
    stock = sklearn.neighbors.NearestNeighbors
    mlxlearn.patch_sklearn()  # imports the consumers now, outside the error filter
    mlxlearn.unpatch_sklearn()
    mlxlearn.patch_sklearn("NearestNeighbors")
    sklearn.neighbors.NearestNeighbors = _Impostor
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(MLXLearnPatchWarning, match="replaced by something other"):
                mlxlearn.patch_sklearn()
        assert patched_estimators() == [
            name for name in patch_names() if name != "sklearn.neighbors.NearestNeighbors"
        ]
        assert sklearn.neighbors.NearestNeighbors is _Impostor
    finally:
        sklearn.neighbors.NearestNeighbors = stock


def test_patch_warnings_point_at_the_caller():
    """``stacklevel`` must land on the line that called mlxlearn. The unpatch
    warning was one frame too far out: the caller's caller, or ``<sys>:0`` at a
    script's top level."""
    stock = sklearn.svm.SVC
    mlxlearn.patch_sklearn("SVC")
    sklearn.svm.SVC = _Impostor
    try:
        with pytest.warns(MLXLearnPatchWarning) as refused_patch:
            mlxlearn.patch_sklearn("SVC")
        with pytest.warns(MLXLearnPatchWarning) as refused_unpatch:
            mlxlearn.unpatch_sklearn("SVC")
    finally:
        sklearn.svm.SVC = stock
    assert [w.filename for w in refused_patch] == [__file__]
    assert [w.filename for w in refused_unpatch] == [__file__]


# ----------------------------------------------------------------------------------
# bookkeeping versus the live attribute
# ----------------------------------------------------------------------------------


def test_repatch_after_sklearn_was_restored_behind_mlxlearns_back():
    """``importlib.reload(sklearn.svm)`` puts back the very class mlxlearn recorded
    as the original. Nothing of anyone's would be lost by patching again, so it is
    not a third-party patch and deserves neither a refusal nor a warning."""
    stock = sklearn.svm.SVC
    mlxlearn.patch_sklearn("SVC")
    importlib.reload(sklearn.svm)
    assert sklearn.svm.SVC is stock

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mlxlearn.patch_sklearn("SVC")
        assert is_patched("SVC")
        mlxlearn.unpatch_sklearn("SVC")
    assert sklearn.svm.SVC is stock


def test_unpatch_after_sklearn_was_restored_is_silent():
    mlxlearn.patch_sklearn("SVC")
    importlib.reload(sklearn.svm)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mlxlearn.unpatch_sklearn("SVC")
    assert patched_estimators() == []


def test_lost_bookkeeping_does_not_make_mlxlearn_the_original():
    """Re-executing ``_patch.py`` (``importlib.reload``, IPython's ``%autoreload``)
    empties its record while scikit-learn stays patched. Patching again used to
    record mlxlearn's own class as the "original", so unpatching restored it."""
    stock = sklearn.svm.SVC
    mlxlearn.patch_sklearn("SVC")
    _patch._INSTALLED.clear()

    mlxlearn.patch_sklearn("SVC")
    assert get_patch_map()["sklearn.svm.SVC"]["original"] is stock
    mlxlearn.unpatch_sklearn("SVC")
    assert sklearn.svm.SVC is stock


def test_unpatch_restores_even_after_the_bookkeeping_was_lost():
    stock = sklearn.svm.SVC
    mlxlearn.patch_sklearn("SVC")
    _patch._INSTALLED.clear()
    assert sklearn.svm.SVC is layer2.SVC

    mlxlearn.unpatch_sklearn()
    assert sklearn.svm.SVC is stock


def test_an_mlxlearn_class_with_no_scikit_learn_base_is_left_alone():
    """Recovering the original relies on the Layer 2 class subclassing the stock
    one. When it cannot, mlxlearn says so instead of guessing."""

    class Unrecognizable:
        _mlxlearn_allow_fallback = True

    stock = sklearn.svm.SVC
    sklearn.svm.SVC = Unrecognizable
    try:
        with pytest.warns(MLXLearnPatchWarning, match="cannot be determined"):
            mlxlearn.patch_sklearn("SVC")
        assert sklearn.svm.SVC is Unrecognizable
        mlxlearn.unpatch_sklearn("SVC")
        assert sklearn.svm.SVC is Unrecognizable
    finally:
        sklearn.svm.SVC = stock


# ----------------------------------------------------------------------------------
# scikit-learn's own consumers of the patched names
# ----------------------------------------------------------------------------------

#: Run in a fresh interpreter: in this one, scikit-learn's subpackages were imported
#: long before any test patched, which is exactly the order that hides the bug.
_PATCH_FIRST_SCRIPT = textwrap.dedent(
    """
    import importlib
    import json
    import pkgutil
    import sys

    import numpy as np

    import mlxlearn
    from mlxlearn.patching import _estimators

    mlxlearn.patch_sklearn()

    import sklearn

    for module in pkgutil.iter_modules(sklearn.__path__):
        if not module.name.startswith("_") and module.name != "tests":
            importlib.import_module(f"sklearn.{module.name}")

    LAYER2 = [getattr(_estimators, name) for name in _estimators.__all__]

    def bound_to_mlxlearn():
        return sorted(
            f"{name}.{attribute}"
            for name, module in list(sys.modules.items())
            if name == "sklearn" or name.startswith("sklearn.")
            for attribute, value in list(getattr(module, "__dict__", {}).items())
            if any(value is cls for cls in LAYER2)
        )

    from sklearn.feature_selection import mutual_info_classif

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    y = (X[:, 0] > 0).astype(int)
    result = {
        "while_patched": bound_to_mlxlearn(),
        "mutual_info": mutual_info_classif(X, y, random_state=0).tolist(),
    }

    # A scikit-learn module outside every consumer list, first imported while
    # patched: it binds the Layer 2 classes, and unpatching must still undo that.
    import sklearn.utils.estimator_checks  # noqa: F401

    result["late_import"] = bound_to_mlxlearn()
    mlxlearn.unpatch_sklearn()
    result["after_unpatch"] = bound_to_mlxlearn()
    print(json.dumps(result))
    """
)


@pytest.fixture(scope="module")
def patch_first():
    result = subprocess.run(
        [sys.executable, "-c", _PATCH_FIRST_SCRIPT], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_sklearn_internals_keep_the_stock_classes_under_patch_first(patch_first):
    """Patch, then import all of scikit-learn: only the public names are rebound.

    DBSCAN, OPTICS, TSNE, Isomap, LabelPropagation, ``mutual_info_*`` and
    StackingClassifier bind these names when their module is first imported. Under
    patch-first they captured the mlxlearn classes — un-gated acceleration of
    algorithms nobody asked mlxlearn to run. This also catches a new consumer in a
    future scikit-learn release that the registry's ``consumers`` do not list.
    """
    assert patch_first["while_patched"] == sorted(patch_names())


def test_mutual_info_is_unbiased_under_patch_first(patch_first):
    """The Kraskov estimator counts neighbors strictly inside ``nextafter`` of the
    k-th distance, which float32 distances overshoot about half the time. With the
    patched NearestNeighbors inside it, MI(x0; x0 > 0) came out near 0.52, not ln 2."""
    informative, noise = patch_first["mutual_info"]
    assert informative == pytest.approx(np.log(2), abs=0.03)
    assert noise < 0.05


def test_unpatch_restores_sklearn_modules_imported_while_patched(patch_first):
    """Their module attributes: containers they filled while patched are not rewritten
    (see ``_restore_internal_bindings``)."""
    late = set(patch_first["late_import"]) - set(patch_names())
    assert late, "the probe no longer imports a module that binds the patched names"
    assert patch_first["after_unpatch"] == []


def test_unpatch_rewrites_internal_bindings_by_identity(monkeypatch):
    """Aliases are found, and a binding outside scikit-learn is not touched."""
    stock = sklearn.neighbors.NearestNeighbors
    mlxlearn.patch_sklearn("NearestNeighbors")
    installed = sklearn.neighbors.NearestNeighbors

    inside = types.ModuleType("sklearn._mlxlearn_test_consumer")
    inside.NearestNeighbors = installed
    inside.NN = installed
    outside = types.ModuleType("mlxlearn_test_third_party")
    outside.NearestNeighbors = installed
    monkeypatch.setitem(sys.modules, inside.__name__, inside)
    monkeypatch.setitem(sys.modules, outside.__name__, outside)

    mlxlearn.unpatch_sklearn("NearestNeighbors")

    assert sklearn.neighbors.NearestNeighbors is stock
    assert inside.NearestNeighbors is stock
    assert inside.NN is stock
    assert outside.NearestNeighbors is installed  # documented: not scikit-learn's
    event = mlxlearn.get_last_backend_event()
    assert event.operation == "unpatch"
    assert "sklearn._mlxlearn_test_consumer.NN" in event.detail


# ----------------------------------------------------------------------------------
# documentation of the patched classes
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", layer2.__all__)
def test_patched_class_keeps_its_documentation(name):
    """``help(sklearn.svm.SVC)`` under patching showed only a 392-character note:
    assigning the note to the Layer 2 class replaced the docstring it inherits."""
    cls = getattr(layer2, name)
    doc = inspect.getdoc(cls)
    strict = inspect.getdoc(cls.__mro__[1])

    head, _, rest = strict.partition("\nParameters\n")
    assert rest, "the Layer 1 docstring has no Parameters section"
    assert doc.startswith(head)
    assert doc.endswith(rest)
    assert "patch_sklearn" in doc


@pytest.mark.parametrize("name", layer2.__all__)
def test_the_patched_note_does_not_call_the_crossover_a_capability_mismatch(name):
    """It listed "a problem below the measured crossover" among the capability
    mismatches Layer 1 raises on. Layer 1 runs such a problem on MLX's CPU; the
    patched class hands it to scikit-learn by design, not for want of capability."""
    doc = " ".join(inspect.getdoc(getattr(layer2, name)).split())
    note = doc[doc.index("This is the patched variant") :]
    capability, _, by_design = note.partition("by design")
    assert "crossover" not in capability
    assert "below the measured crossover" in by_design


@pytest.mark.parametrize("name", layer2.__all__)
def test_patched_class_renders_in_a_notebook(name):
    """scikit-learn's HTML repr parses the docstring with numpydoc, which raises on a
    duplicated section and warns on an unknown one."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        getattr(layer2, name)()._repr_html_()


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
