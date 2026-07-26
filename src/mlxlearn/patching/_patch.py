"""Patching scikit-learn's module attributes, and putting them back.

This is the only part of mlxlearn that goes near another library's namespace, and
it is quarantined here for that reason.

## What patching can and cannot do

``patch_sklearn()`` rebinds attributes **on scikit-learn's modules**. So
``sklearn.svm.SVC`` resolves to the mlxlearn class whether ``import sklearn``
happened before or after patching — attribute lookup happens at access time.

A name that was already bound into someone else's namespace cannot be reached::

    from sklearn.svm import SVC   # SVC is now a local name for the stock class
    patch_sklearn()               # rebinds sklearn.svm.SVC, not your SVC
    SVC()                         # still the stock class

That is Python name binding, and no patching implementation can work around it.
The supported pattern is patch-first, and the documentation says so rather than
claiming an import-order safety that does not exist.

## Where this differs from the ancestor, on purpose

* **Symmetry.** An unknown name raises in *both* directions. The ancestor raised
  when patching and returned silently when unpatching, so a typo left the process
  patched with nothing to indicate it.
* **No hidden extras.** Patching one estimator patches one estimator. The ancestor
  appended five config and parallelism hooks to every request, so
  ``patch_sklearn("SVC")`` also replaced ``sklearn.set_config``.
* **Thread safety.** A lock guards the mutation and the bookkeeping.
* **Tamper detection.** ``unpatch_sklearn`` checks that the attribute is still the
  object it installed before restoring, and warns instead of clobbering a third
  party's patch.
* **Quiet by default.** ``verbose=False``. A library that prints to stderr on
  import is a library that corrupts someone's piped output.
"""

from __future__ import annotations

import sys
import threading
import warnings

from .._common.diagnostics import record_event
from ..exceptions import MLXLearnPatchWarning
from ._registry import ESTIMATOR_REGISTRY, EstimatorSpec, load_spec, patch_names, resolve_names

__all__ = [
    "patch_sklearn",
    "unpatch_sklearn",
    "is_patched",
    "get_patch_map",
    "patch_names",
    "patched_estimators",
]

_lock = threading.RLock()

#: name -> (module, attribute, original object, object mlxlearn installed)
_INSTALLED: dict[str, tuple[object, str, object, object]] = {}


def patch_sklearn(estimators=None, verbose: bool = False) -> None:
    """Point scikit-learn's estimator names at mlxlearn's accelerated classes.

    Parameters
    ----------
    estimators : str or list of str, default=None
        Which estimators to patch. Accepts bare names (``"SVC"``) or dotted
        names (``"sklearn.svm.SVC"``). ``None`` patches everything mlxlearn
        implements.
    verbose : bool, default=False
        Print a one-line confirmation to stderr.

    Raises
    ------
    ValueError
        If a requested estimator has no mlxlearn implementation. The message
        lists what is available.

    Notes
    -----
    Idempotent: patching twice is a no-op, and the original class is recorded
    once, on the first patch, so a second call cannot record mlxlearn's own class
    as the "original" and make :func:`unpatch_sklearn` a no-op.

    Patch **before** importing the symbols you want accelerated; see the module
    docstring for why.

    Examples
    --------
    >>> from mlxlearn import patch_sklearn, unpatch_sklearn
    >>> patch_sklearn()
    >>> import sklearn.svm
    >>> sklearn.svm.SVC.__module__.startswith("mlxlearn")
    True
    >>> unpatch_sklearn()
    >>> sklearn.svm.SVC.__module__.startswith("sklearn")
    True
    """
    specs = resolve_names(estimators)

    with _lock:
        for spec in specs:
            _patch_one(spec)

    if verbose and sys.stderr is not None:
        names = ", ".join(s.short_name for s in specs)
        print(f"mlxlearn: patched scikit-learn estimators [{names}]", file=sys.stderr)


def _patch_one(spec: EstimatorSpec) -> None:
    sk_module, replacement = load_spec(spec)
    current = getattr(sk_module, spec.attribute, None)

    if spec.name in _INSTALLED:
        _, _, _, installed = _INSTALLED[spec.name]
        if current is installed:
            return  # already patched; idempotent
        # Someone replaced the attribute after we did. Re-patching would lose
        # their object with no way to get it back, so leave it alone and say so.
        warnings.warn(
            f"{spec.name} was replaced by something other than mlxlearn after "
            "patch_sklearn() ran. mlxlearn is leaving it alone rather than "
            "discarding another library's patch. Call unpatch_sklearn() first if "
            "you meant to re-patch.",
            MLXLearnPatchWarning,
            stacklevel=3,
        )
        return

    _INSTALLED[spec.name] = (sk_module, spec.attribute, current, replacement)
    setattr(sk_module, spec.attribute, replacement)
    record_event(
        spec.short_name, "patch", "sklearn", reason="patched",
        detail=f"{spec.name} -> {replacement.__module__}.{replacement.__qualname__}",
    )


def unpatch_sklearn(estimators=None) -> None:
    """Restore scikit-learn's original estimator classes.

    Parameters
    ----------
    estimators : str or list of str, default=None
        Which estimators to restore. ``None`` restores everything mlxlearn
        patched.

    Raises
    ------
    ValueError
        On an unknown estimator name — symmetric with :func:`patch_sklearn`.

    Notes
    -----
    Restoring an estimator that was never patched is a no-op, not an error;
    that is the idempotent case, distinct from naming an estimator that does not
    exist.

    Instances already created from a patched class keep working. They are
    mlxlearn objects and remain so; unpatching changes what the *name* resolves
    to, not what existing objects are.
    """
    specs = resolve_names(estimators)

    with _lock:
        for spec in specs:
            entry = _INSTALLED.get(spec.name)
            if entry is None:
                continue
            sk_module, attribute, original, installed = entry
            current = getattr(sk_module, attribute, None)
            if current is not installed:
                warnings.warn(
                    f"{spec.name} is no longer the class mlxlearn installed, so "
                    "mlxlearn did not restore it. Something else patched it after "
                    "mlxlearn did; restoring now would discard that.",
                    MLXLearnPatchWarning,
                    stacklevel=3,
                )
                _INSTALLED.pop(spec.name, None)
                continue
            setattr(sk_module, attribute, original)
            _INSTALLED.pop(spec.name, None)
            record_event(spec.short_name, "unpatch", "sklearn", reason="unpatched")


def is_patched(estimators=None) -> bool:
    """Whether the named estimators currently resolve to mlxlearn classes.

    Parameters
    ----------
    estimators : str or list of str, default=None
        ``None`` means "all of them", and returns ``True`` only if every
        registered estimator is patched.

    Returns
    -------
    bool

    Notes
    -----
    Determined by identity against what mlxlearn installed, not by inspecting a
    module string. The ancestor's equivalent tested whether its own package name
    appeared anywhere in an object's ``__module__``, which any unrelated class
    could satisfy by living in a similarly named module.
    """
    with _lock:
        specs = resolve_names(estimators)
        if not specs:
            return False
        return all(_is_patched_one(spec) for spec in specs)


def _is_patched_one(spec: EstimatorSpec) -> bool:
    entry = _INSTALLED.get(spec.name)
    if entry is None:
        return False
    sk_module, attribute, _original, installed = entry
    return getattr(sk_module, attribute, None) is installed


def get_patch_map() -> dict[str, dict[str, object]]:
    """Describe every registered estimator and its current state.

    Returns
    -------
    dict
        Keyed by dotted name. Each value has ``module``, ``attribute``,
        ``patched`` (bool), ``mlxlearn_class``, and ``original`` (the class that
        will be restored, or ``None`` when not patched).
    """
    with _lock:
        out: dict[str, dict[str, object]] = {}
        for spec in ESTIMATOR_REGISTRY:
            entry = _INSTALLED.get(spec.name)
            out[spec.name] = {
                "module": spec.sklearn_module,
                "attribute": spec.attribute,
                "patched": _is_patched_one(spec),
                "mlxlearn_class": f"{spec.mlxlearn_module}.{spec.mlxlearn_class}",
                "original": entry[2] if entry else None,
                "since": spec.since,
            }
        return out


def patched_estimators() -> list[str]:
    """The dotted names currently patched."""
    with _lock:
        return [spec.name for spec in ESTIMATOR_REGISTRY if _is_patched_one(spec)]
