"""Compatibility across the scikit-learn range ``pyproject.toml`` declares.

mlxlearn subclasses scikit-learn estimators, so it inherits their parameters —
including ones scikit-learn is in the middle of deprecating. A deprecation
typically lands as a **string sentinel default** (``penalty="deprecated"``,
``probability="deprecated"``), and code that reads the attribute directly then
sees a truthy string where it expected a bool, or an unknown value where it
expected one of a fixed set.

Every one of these was found by installing the built wheel into a clean
environment, which resolved scikit-learn 1.9 where development ran on 1.7. The
declared range is ``>=1.7,<1.10``, so all of it has to work; these tests fail on
whichever version is installed rather than pinning the problem away.
"""

from __future__ import annotations

import numpy as np
import pytest
import sklearn
from packaging.version import Version
from sklearn.datasets import make_classification

from mlxlearn._common.config import tuning_context
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.neighbors import KNeighborsClassifier
from mlxlearn.svm import SVC

SKLEARN = Version(sklearn.__version__)


@pytest.fixture(scope="module")
def data():
    return make_classification(
        n_samples=600, n_features=8, n_informative=5, n_redundant=1, random_state=0
    )


def test_declared_range_is_what_is_installed():
    assert Version("1.7") <= SKLEARN < Version("1.10"), (
        f"scikit-learn {SKLEARN} is outside the range pyproject.toml declares"
    )


@pytest.mark.parametrize(
    "cls, kwargs", [(KNeighborsClassifier, {"n_neighbors": 3}), (LogisticRegression, {}), (SVC, {})]
)
def test_default_construction_fits(cls, kwargs, data):
    """A default-constructed estimator must fit on every supported version.

    Both regressions this catches presented exactly here: on 1.9,
    ``LogisticRegression()`` raised ``AttributeError: no attribute 'multi_class'``
    and ``SVC()`` was rejected because ``probability`` defaulted to the truthy
    string ``"deprecated"``.
    """
    X, y = data
    with tuning_context(force_backend="mlx"):
        model = cls(**kwargs).fit(X, y)
    assert model.predict(X[:20]).shape == (20,)


def test_effective_penalty_resolves_on_both_spellings(data):
    """``penalty`` (≤1.9) and ``l1_ratio`` (≥1.8) name the same thing."""
    X, y = data

    assert LogisticRegression()._effective_penalty() == "l2"

    if SKLEARN >= Version("1.8"):
        # The new spelling: l1_ratio decides, and C=inf means unpenalized.
        assert LogisticRegression(l1_ratio=1.0)._effective_penalty() == "l1"
        assert LogisticRegression(l1_ratio=0.5)._effective_penalty() == "elasticnet"
        assert LogisticRegression(C=np.inf)._effective_penalty() is None
    else:
        assert LogisticRegression(penalty=None)._effective_penalty() is None

    with tuning_context(force_backend="mlx"):
        LogisticRegression().fit(X, y)


def test_unsupported_penalty_still_raises_on_both_spellings(data):
    """An elastic-net request must never be quietly served as ridge.

    That was the ancestor's behavior. The spelling changed between versions; the
    refusal must not.
    """
    from mlxlearn.exceptions import UnsupportedParameterError

    X, y = data
    if SKLEARN >= Version("1.8"):
        estimator = LogisticRegression(l1_ratio=0.5)
    else:
        estimator = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga")

    with pytest.raises(UnsupportedParameterError):
        estimator.fit(X, y)


def test_unpenalized_fit_is_recognized_on_both_spellings(data):
    """``C=inf`` (1.9) and ``penalty=None`` (1.7) both mean no penalty."""
    from sklearn.linear_model import LogisticRegression as SkLogisticRegression

    X, y = data
    estimator = LogisticRegression(C=np.inf) if SKLEARN >= Version("1.8") else LogisticRegression(penalty=None)
    reference = (
        SkLogisticRegression(C=np.inf) if SKLEARN >= Version("1.8")
        else SkLogisticRegression(penalty=None)
    )

    assert estimator._effective_penalty() is None
    with tuning_context(force_backend="mlx"):
        mlx = estimator.fit(X, y)
    ref = reference.fit(X, y)
    assert np.mean(mlx.predict(X) == ref.predict(X)) >= 0.99


def test_svc_probability_sentinel_is_not_a_request_for_probabilities(data):
    """``probability="deprecated"`` is scikit-learn 1.9's default, and is falsy in intent.

    Treating the string as truthy rejected every default-constructed SVC.
    """
    from mlxlearn.exceptions import UnsupportedParameterError

    X, y = data
    with tuning_context(force_backend="mlx"):
        SVC().fit(X, y)

    # An explicit request is still refused, on every version.
    with pytest.raises(UnsupportedParameterError, match="CalibratedClassifierCV"):
        SVC(probability=True).fit(X, y)


def test_all_zero_sample_weight_error_is_recognizable():
    """scikit-learn 1.9 matches the message against ``weight.*zero|zero.*weight``."""
    import re

    from mlxlearn._solvers.logistic import LogisticConfig, solve_logistic

    X = np.zeros((6, 3), dtype=np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError) as excinfo:
        solve_logistic(
            X, y, n_classes=2, sample_weight=np.zeros(6), config=LogisticConfig()
        )
    assert re.search(r"(weight.*zero)|(zero.*weight)", str(excinfo.value), re.IGNORECASE)
