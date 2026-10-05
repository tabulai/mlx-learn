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

## scikit-learn's own internals are deliberately not patched

The same name binding cuts the other way. ``sklearn/cluster/_dbscan.py`` does
``from sklearn.neighbors import NearestNeighbors`` at import time, and so do OPTICS,
TSNE, Isomap, ``mutual_info_classif`` and others; ``StackingClassifier`` binds
``LogisticRegression``. Under patch-first those modules would load *after* the
rebinding and capture the mlxlearn class for the rest of the process. That is not
acceleration anyone asked for: those algorithms have not been through the gates, and
some are wrong on float32 distances — the Kraskov mutual-information estimator
counts neighbors strictly inside ``nextafter(kth_distance)``, so ``mutual_info_classif``
came out about a quarter low and ``SelectKBest`` picked different features.

So :func:`patch_sklearn` imports each estimator's ``consumers`` (from the registry)
before rebinding it, and they keep the stock class whatever the import order.
:func:`unpatch_sklearn` additionally points any ``sklearn.*`` module attribute that
is still bound to an mlxlearn class back at the original — a module first imported
while patched, outside that list. Names bound *outside* scikit-learn while patched
(your own ``from sklearn.svm import SVC``, or a third-party library doing the same)
are ordinary bindings in someone else's namespace and are not rewritten.

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
  party's patch. The warnings are issued only after every estimator has been
  handled and the lock released, so ``-W error`` cannot stop either function
  half-way and leave the bookkeeping stuck on the offending entry.
* **Quiet by default.** ``verbose=False``. A library that prints to stderr on
  import is a library that corrupts someone's piped output.
"""

from __future__ import annotations

import sys
import threading
import warnings
from importlib import import_module

from .._common.diagnostics import external_stacklevel, record_event
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

    Warns
    -----
    MLXLearnPatchWarning
        For an estimator another library replaced after mlxlearn patched it.
        That one is left alone; the rest of the selection is still patched.

    Notes
    -----
    Idempotent: patching twice is a no-op, and the original class is recorded
    once, on the first patch, so a second call cannot record mlxlearn's own class
    as the "original" and make :func:`unpatch_sklearn` a no-op.

    Patch **before** importing the symbols you want accelerated; see the module
    docstring for why.

    Only the public names are patched. scikit-learn's own users of them —
    ``DBSCAN``, ``TSNE``, ``mutual_info_classif``, ``StackingClassifier`` and the
    like — keep the stock classes whatever the import order.

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

    # Before anything is rebound: see "scikit-learn's own internals are
    # deliberately not patched" in the module docstring. Outside the lock, because
    # it is only imports, and the import system has its own.
    for module_name in dict.fromkeys(m for spec in specs for m in spec.consumers):
        import_module(module_name)

    refusals: list[tuple[str, str]] = []
    with _lock:
        for spec in specs:
            refusal = _patch_one(spec)
            if refusal is not None:
                refusals.append((spec.name, refusal))

    if verbose and sys.stderr is not None:
        refused = {name for name, _ in refusals}
        names = ", ".join(s.short_name for s in specs if s.name not in refused)
        print(f"mlxlearn: patched scikit-learn estimators [{names}]", file=sys.stderr)

    # Attributed to the first frame outside mlxlearn — the caller's line, however
    # many mlxlearn frames sit between it and here.
    for _name, message in refusals:
        warnings.warn(message, MLXLearnPatchWarning, stacklevel=external_stacklevel())


def _patch_one(spec: EstimatorSpec) -> str | None:
    """Install one replacement. Returns why it refused, or ``None``.

    Never warns itself: the caller does, once the whole selection has been
    handled, so an error filter on the warning cannot interrupt the loop.
    """
    sk_module, replacement = load_spec(spec)
    current = getattr(sk_module, spec.attribute, None)

    entry = _INSTALLED.get(spec.name)
    if entry is not None:
        _, _, original, installed = entry
        if current is installed:
            return None  # already patched; idempotent
        if current is not original:
            # Someone replaced the attribute after we did. Re-patching would lose
            # their object with no way to get it back, so leave it alone and say so.
            return (
                f"{spec.name} was replaced by something other than mlxlearn after "
                "patch_sklearn() ran. mlxlearn is leaving it alone rather than "
                "discarding another library's patch. Call unpatch_sklearn() first if "
                "you meant to re-patch."
            )
        # Back to the recorded original without going through unpatch_sklearn —
        # importlib.reload(sklearn.svm), or a restore by hand. Nothing of anyone's
        # would be lost, so the stale entry is simply replaced below.
    elif _is_layer2_class(current):
        # sklearn is patched but this module has no record of it: it was
        # re-executed (importlib.reload, IPython's %autoreload) and the bookkeeping
        # went with it. Recording mlxlearn's own class as the "original" would make
        # every later unpatch_sklearn() restore mlxlearn, so recover the stock class.
        original = _stock_class(current, spec.attribute)
        if original is None:
            return (
                f"{spec.name} is an mlxlearn class that mlxlearn has no record of "
                "installing, and the scikit-learn class it replaced cannot be "
                "determined. mlxlearn is leaving it alone; restore it by hand, e.g. "
                f"importlib.reload({spec.sklearn_module})."
            )
    else:
        original = current

    _INSTALLED[spec.name] = (sk_module, spec.attribute, original, replacement)
    setattr(sk_module, spec.attribute, replacement)
    record_event(
        spec.short_name, "patch", "sklearn", reason="patched",
        detail=f"{spec.name} -> {replacement.__module__}.{replacement.__qualname__}",
    )
    return None


def _is_layer2_class(obj: object) -> bool:
    """Whether ``obj`` is one of the Layer 2 classes in ``_estimators``.

    They are the only classes that *define* ``_mlxlearn_allow_fallback = True``;
    reading the class's own ``__dict__`` rather than ``getattr`` keeps a subclass
    someone else wrote from qualifying by inheritance. Identity against the current
    replacement would not do here: the case this serves is a module having been
    re-executed, which may have produced new class objects.
    """
    return isinstance(obj, type) and vars(obj).get("_mlxlearn_allow_fallback") is True


def _stock_class(layer2: type, attribute: str) -> type | None:
    """The scikit-learn class a Layer 2 class stands in for.

    Every mlxlearn estimator subclasses the scikit-learn estimator of the same
    name (docs/development.md, "Subclass the scikit-learn estimator"), so it is the
    first class of that name in the MRO that scikit-learn defines — the class the
    public attribute pointed at before patching.
    """
    for base in layer2.__mro__[1:]:
        if base.__name__ == attribute and base.__module__.partition(".")[0] == "sklearn":
            return base
    return None


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

    Warns
    -----
    MLXLearnPatchWarning
        For an estimator another library replaced after mlxlearn patched it.
        That one is left alone, mlxlearn forgets it, and the rest of the
        selection is still restored.

    Notes
    -----
    Restoring an estimator that was never patched is a no-op, not an error;
    that is the idempotent case, distinct from naming an estimator that does not
    exist.

    Instances already created from a patched class keep working. They are
    mlxlearn objects and remain so; unpatching changes what the *name* resolves
    to, not what existing objects are.

    Besides the public attribute, any ``sklearn.*`` module attribute still bound
    to the mlxlearn class — a scikit-learn module first imported while patched —
    is pointed back at the original. Names bound while patched outside
    scikit-learn, by your code or another library, are not rewritten, and neither
    are containers a module filled with classes as it was imported: import
    ``sklearn.utils.estimator_checks`` before patching if you will run
    scikit-learn's estimator checks after unpatching.
    """
    specs = resolve_names(estimators)

    refused: list[str] = []
    with _lock:
        undone: list[tuple[EstimatorSpec, object, object, bool]] = []
        for spec in specs:
            # Popped first, whatever happens next: a stale entry left behind would
            # make every later call trip over it again.
            entry = _INSTALLED.pop(spec.name, None) or _orphaned_entry(spec)
            if entry is None:
                continue
            sk_module, attribute, original, installed = entry
            current = getattr(sk_module, attribute, None)
            if current is installed:
                setattr(sk_module, attribute, original)
            elif current is not original:
                # Something else patched it after mlxlearn did. (If it is already
                # the original — a reload, a restore by hand — there is nothing to
                # say and nothing to do.)
                refused.append(spec.name)
            undone.append((spec, installed, original, current is installed))

        internal = _restore_internal_bindings([(inst, orig) for _, inst, orig, _ in undone])
        for spec, installed, _original, restored in undone:
            also = internal.get(id(installed), [])
            if restored or also:
                record_event(
                    spec.short_name, "unpatch", "sklearn", reason="unpatched",
                    detail=(
                        f"also restored scikit-learn's internal binding(s): {', '.join(also)}"
                        if also else ""
                    ),
                )

    # After the loop and outside the lock, for the reason given in _patch_one.
    for name in refused:
        warnings.warn(
            f"{name} is no longer the class mlxlearn installed, so "
            "mlxlearn did not restore it. Something else patched it after "
            "mlxlearn did; restoring now would discard that.",
            MLXLearnPatchWarning,
            stacklevel=external_stacklevel(),
        )


def _orphaned_entry(spec: EstimatorSpec) -> tuple[object, str, object, object] | None:
    """Rebuild the bookkeeping for a patch this module has no record of.

    See the matching branch of :func:`_patch_one`. Nothing is imported: if the
    scikit-learn module is not loaded, nothing in it can be patched.
    """
    sk_module = sys.modules.get(spec.sklearn_module)
    current = getattr(sk_module, spec.attribute, None)
    if not _is_layer2_class(current):
        return None
    original = _stock_class(current, spec.attribute)
    if original is None:
        return None
    return (sk_module, spec.attribute, original, current)


def _restore_internal_bindings(pairs: list[tuple[object, object]]) -> dict[int, list[str]]:
    """Point ``sklearn.*`` module attributes bound to an installed class at its original.

    ``pairs`` is ``(installed, original)``. Returns the rewritten dotted names,
    keyed by ``id(installed)`` — every key is alive in ``pairs`` for the whole call.

    The consumer imports in :func:`patch_sklearn` keep scikit-learn's own modules
    off the mlxlearn classes, so usually this finds nothing. It restores the module
    attributes of a scikit-learn module outside the registry's ``consumers`` that
    was first imported while patched: a consumer added in a newer release, or
    ``sklearn.utils.estimator_checks``. Identity, not names, decides — the same
    rule as tamper detection — so ``from sklearn.neighbors import NearestNeighbors
    as NN`` is caught and a third party's class is never touched. About a
    millisecond over the few hundred ``sklearn.*`` modules a typical process has
    loaded.

    Module attributes only. A container such a module filled with classes as it
    was imported keeps the mlxlearn ones: the ``INIT_PARAMS``,
    ``PER_ESTIMATOR_CHECK_PARAMS`` and ``PER_ESTIMATOR_XFAIL_CHECKS`` tables of
    scikit-learn's test instance generator are the known case, so its estimator
    checks, run after unpatching, miss the stock classes' entries.
    """
    if not pairs:
        return {}
    by_id = {id(installed): (installed, original) for installed, original in pairs}
    found: dict[int, list[str]] = {}
    for module_name, module in list(sys.modules.items()):
        if module_name != "sklearn" and not module_name.startswith("sklearn."):
            continue
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for attribute, value in list(namespace.items()):
            match = by_id.get(id(value))
            if match is None or match[0] is not value:
                continue
            setattr(module, attribute, match[1])
            found.setdefault(id(value), []).append(f"{module_name}.{attribute}")
    return found


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
