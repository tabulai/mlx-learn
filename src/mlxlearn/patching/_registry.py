"""The registry of estimators the patching layer knows how to replace.

Kept as plain data — module path, attribute name, and where the mlxlearn class
lives — so that ``import mlxlearn`` does not drag in scikit-learn's estimator
modules or initialize Metal. The classes are imported the first time
:func:`get_patch_map` is called, which is to say when someone actually patches.

An estimator earns an entry here only when it has passed the gates in the
refactoring plan §9. Registering a name whose implementation is not ready would
mean silently swapping in something untested, which is precisely the failure mode
the two-layer split exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

__all__ = ["EstimatorSpec", "ESTIMATOR_REGISTRY", "resolve_names", "load_spec"]


@dataclass(frozen=True, slots=True)
class EstimatorSpec:
    """One patchable estimator.

    Attributes
    ----------
    name : str
        The canonical dotted name, e.g. ``"sklearn.svm.SVC"``. Also addressable by
        its bare class name, ``"SVC"``.
    sklearn_module : str
        The module whose attribute is replaced.
    attribute : str
        The attribute name on that module.
    mlxlearn_module : str
        Where the Layer 2 class lives.
    mlxlearn_class : str
        The Layer 2 class name.
    since : str
        The mlxlearn version that first patched this estimator.
    """

    name: str
    sklearn_module: str
    attribute: str
    mlxlearn_module: str
    mlxlearn_class: str
    since: str = "0.1.0"

    @property
    def short_name(self) -> str:
        return self.attribute


def _spec(sklearn_module: str, attribute: str, mlxlearn_module: str, mlxlearn_class: str):
    return EstimatorSpec(
        name=f"{sklearn_module}.{attribute}",
        sklearn_module=sklearn_module,
        attribute=attribute,
        mlxlearn_module=mlxlearn_module,
        mlxlearn_class=mlxlearn_class,
    )


#: Everything mlxlearn patches, in a deterministic order.
#:
#: Deliberately short. Tree ensembles, pass-through estimators, and the
#: rename-or-implement cases (``SVR``, ``NuSVR``, ``NuSVC``) are absent, and will
#: stay absent until they implement the objective their scikit-learn name promises.
ESTIMATOR_REGISTRY: tuple[EstimatorSpec, ...] = (
    _spec("sklearn.neighbors", "NearestNeighbors", "mlxlearn.patching._estimators", "NearestNeighbors"),
    _spec("sklearn.neighbors", "KNeighborsClassifier", "mlxlearn.patching._estimators", "KNeighborsClassifier"),
    _spec("sklearn.neighbors", "KNeighborsRegressor", "mlxlearn.patching._estimators", "KNeighborsRegressor"),
    _spec("sklearn.linear_model", "LogisticRegression", "mlxlearn.patching._estimators", "LogisticRegression"),
    _spec("sklearn.svm", "SVC", "mlxlearn.patching._estimators", "SVC"),
)

_BY_NAME: dict[str, EstimatorSpec] = {}
for _s in ESTIMATOR_REGISTRY:
    _BY_NAME[_s.name] = _s
    _BY_NAME[_s.short_name] = _s


def patch_names() -> list[str]:
    """The canonical dotted names, in patch order."""
    return [s.name for s in ESTIMATOR_REGISTRY]


def resolve_names(estimators: str | list[str] | tuple[str, ...] | None) -> list[EstimatorSpec]:
    """Turn a user's estimator selection into specs.

    Accepts ``None`` (everything), a single name, or an iterable. Both
    ``"SVC"`` and ``"sklearn.svm.SVC"`` resolve.

    Raises
    ------
    ValueError
        On an unknown name, listing what is available. The ancestor raised on an
        unknown name when patching and silently ignored it when unpatching; a
        typo therefore left the process patched with no indication. Both
        directions raise here.
    """
    if estimators is None:
        return list(ESTIMATOR_REGISTRY)
    if isinstance(estimators, str):
        requested = [estimators]
    else:
        requested = list(estimators)

    specs: list[EstimatorSpec] = []
    seen: set[str] = set()
    for name in requested:
        spec = _BY_NAME.get(name)
        if spec is None:
            available = ", ".join(sorted({s.short_name for s in ESTIMATOR_REGISTRY}))
            raise ValueError(
                f"mlxlearn has no accelerated implementation registered for {name!r}. "
                f"Available: {available}. Pass estimators=None to patch everything."
            )
        if spec.name not in seen:
            seen.add(spec.name)
            specs.append(spec)
    return specs


def load_spec(spec: EstimatorSpec) -> tuple[ModuleType, type]:
    """Import the scikit-learn module and the mlxlearn replacement class."""
    from importlib import import_module

    sk_module = import_module(spec.sklearn_module)
    mlx_module = import_module(spec.mlxlearn_module)
    replacement = getattr(mlx_module, spec.mlxlearn_class)
    return sk_module, replacement
