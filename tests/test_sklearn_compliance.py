"""scikit-learn's own estimator check suite, run against every mlxlearn estimator.

Both layers are checked. The Layer 1 classes advertise narrowed tags (no sparse,
no multi-output) and must honor them; the Layer 2 classes advertise scikit-learn's
tags, because they really do accept sparse input and multi-output targets — by
handing them to scikit-learn. Running only one layer would leave the other's tag
declarations unverified, and a tag that lies is worse than a missing feature: every
meta-estimator in the ecosystem reads them.

Two kinds of exemption, kept strictly apart:

``INHERITED_FAILURES``
    Checks that **scikit-learn's own estimator fails**, declared in its source. An
    mlxlearn estimator that subclasses it is not expected to pass a check its base
    class does not. Nothing about mlxlearn is being excused here.

``EXPECTED_XFAILS``
    Checks mlxlearn fails that scikit-learn passes. Each entry names a rule from
    ``docs/fp32_policy.md`` §4 and is asserted to do so, so an xfail cannot be
    added quietly to turn a red test green. **It is currently empty.**
"""

from __future__ import annotations

import pytest
from sklearn.utils.estimator_checks import check_estimator

from mlxlearn.linear_model import LogisticRegression
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from mlxlearn.patching import _estimators as patched
from mlxlearn.svm import SVC

#: Checks scikit-learn declares as expected failures for these estimators, copied
#: from ``sklearn.utils._test_common.instance_generator``. Both are the same
#: property, stated by scikit-learn about its own implementations:
#: "sample_weight is not equivalent to removing/repeating samples."
#:
#: Hardcoded rather than imported, because the source is a private module. The
#: test below re-derives it from scikit-learn when that module is importable, so
#: this list cannot silently drift out of date.
INHERITED_FAILURES: dict[str, set[str]] = {
    "LogisticRegression": {
        "check_sample_weight_equivalence_on_dense_data",
        "check_sample_weight_equivalence_on_sparse_data",
    },
    "SVC": {
        "check_sample_weight_equivalence_on_dense_data",
        "check_sample_weight_equivalence_on_sparse_data",
    },
}

#: (estimator class name, check name) -> "RULE: justification", for checks
#: scikit-learn passes and mlxlearn does not.
#:
#: Rules from docs/fp32_policy.md §4:
#:   A -- the check asserts float64-level exactness
#:   B -- the check asserts float32 input yields float32 fitted attributes
#:   C -- the check asserts random_state=None varies, which deterministic=True denies
EXPECTED_XFAILS: dict[tuple[str, str], str] = {}

LAYER1 = [NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, LogisticRegression, SVC]
LAYER2 = [
    patched.NearestNeighbors,
    patched.KNeighborsClassifier,
    patched.KNeighborsRegressor,
    patched.LogisticRegression,
    patched.SVC,
]


def _run(estimator_class, layer: str):
    estimator = estimator_class()
    # __name__ is set to the scikit-learn name on the Layer 2 classes, so this
    # key works for both layers.
    inherited = INHERITED_FAILURES.get(estimator_class.__name__, set())

    results = check_estimator(estimator, on_fail=None)
    assert results, f"no checks ran for {estimator_class}"

    failures = []
    for result in results:
        if result["status"] != "failed":
            continue
        name = result["check_name"]
        if name in inherited:
            continue
        if (estimator_class.__name__, name) in EXPECTED_XFAILS:
            continue
        failures.append(f"{name}: {result['exception']}")

    assert not failures, (
        f"{layer} {estimator_class.__name__} failed {len(failures)} scikit-learn checks:\n"
        + "\n".join(f"  - {f}" for f in failures)
    )
    return results


@pytest.mark.parametrize("estimator_class", LAYER1, ids=lambda c: c.__name__)
def test_layer1_passes_sklearn_checks(estimator_class):
    _run(estimator_class, "Layer 1")


@pytest.mark.parametrize("estimator_class", LAYER2, ids=lambda c: c.__module__ + "." + c.__name__)
def test_layer2_passes_sklearn_checks(estimator_class):
    """The patched classes must be at least as compatible as the strict ones.

    They accept strictly more — sparse input and multi-output targets reach
    scikit-learn — so a failure here means the fallback is not transparent.
    """
    _run(estimator_class, "Layer 2")


def test_inherited_failures_match_what_sklearn_declares():
    """Re-derive ``INHERITED_FAILURES`` from scikit-learn itself.

    The source is a private module, so this skips rather than fails when
    scikit-learn moves it — but while it is reachable, the hardcoded list cannot
    drift. Without this, an exemption could outlive the scikit-learn behavior that
    justified it and quietly start covering a real mlxlearn regression.
    """
    instance_generator = pytest.importorskip("sklearn.utils._test_common.instance_generator")
    get_expected = getattr(instance_generator, "_get_expected_failed_checks", None)
    if get_expected is None:
        pytest.skip("scikit-learn no longer exposes _get_expected_failed_checks")

    import sklearn.linear_model
    import sklearn.svm

    for base, name in ((sklearn.svm.SVC, "SVC"), (sklearn.linear_model.LogisticRegression, "LogisticRegression")):
        declared = set(get_expected(base()))
        assert INHERITED_FAILURES[name] == declared, (
            f"scikit-learn's expected failures for {name} changed to {sorted(declared)}; "
            f"update INHERITED_FAILURES and re-check whether mlxlearn now passes them."
        )


def test_xfail_list_is_what_it_claims_to_be():
    """An xfail must be a decision, not an accident.

    If this fails, either an estimator regressed or someone added an xfail without
    updating docs/fp32_policy.md §4. Both need attention.
    """
    for (estimator_name, check_name), justification in EXPECTED_XFAILS.items():
        assert justification.startswith(("A:", "B:", "C:")), (
            f"xfail for {estimator_name}.{check_name} does not name a rule from "
            f"docs/fp32_policy.md §4: {justification!r}"
        )


def test_layer1_tags_narrow_what_layer2_allows():
    """Layer 1 declares it does not take sparse or multi-output; Layer 2 declares it does.

    An estimator that advertises a capability and then raises is lying to every
    meta-estimator that reads its tags.
    """
    from sklearn.utils import get_tags

    strict = get_tags(KNeighborsClassifier())
    lenient = get_tags(patched.KNeighborsClassifier())

    assert strict.input_tags.sparse is False
    assert strict.target_tags.multi_output is False
    assert lenient.input_tags.sparse is True
    assert lenient.target_tags.multi_output is True
