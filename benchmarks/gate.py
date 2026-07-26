"""The performance gate.

The design goal is "never slower than scikit-learn". The *gate* is narrower and
measurable, because a goal you cannot fail is not a gate:

> On every benchmarked workload class at or above its published crossover,
> mlxlearn shows no statistically significant regression against stock
> scikit-learn.

Two consequences worth being explicit about.

**Below the crossover, being slower is not a failure — it is the design.** Small
problems are dispatched to scikit-learn on purpose; measuring mlxlearn's own path
there measures a path that never runs in patched mode.

**"No significant regression" is not "faster".** A case that lands within the
measured noise passes. Requiring a win on every case would make the gate a
lottery on a shared machine; requiring no *loss* is the claim mlxlearn actually
makes.

An operation with no measured crossover is **not** silently exempt. If mlxlearn
is significantly slower there, it either gets fixed or gets an explicit entry in
``ACCEPTED_REGRESSIONS`` with a reason — because "we never measured a win here"
is the most comfortable place for a real regression to hide.

Per-operation, never averaged. An average over fit and predict describes a
workload nobody runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main", "evaluate"]

#: Regressions smaller than this are ignored even when they clear the noise
#: threshold, so that a 1% drift does not fail a nightly run.
TOLERANCE = 0.05

#: Operations where mlxlearn is knowingly slower than scikit-learn, with the reason.
#:
#: An entry here is a decision, not an excuse: it says the regression is
#: structural, understood, and paid back elsewhere. Without this map, an operation
#: that never reaches a crossover would be silently exempt from the gate — which
#: is how a genuine regression hides behind "we never measured a win here".
ACCEPTED_REGRESSIONS: dict[str, str] = {
    "knn/fit": (
        "mlxlearn's KNN fit uploads the training matrix, its transpose, and its row "
        "norms to the device; scikit-learn's brute-force fit stores a reference and "
        "does no work. The gap is a memory copy, it is linear in the data, and the "
        "first kneighbors call repays it several times over. Measured 0.14x-0.6x on "
        "fit against 1.8x-14.3x on kneighbors. See docs/benchmarks.md."
    ),
}


def evaluate(payload: dict) -> tuple[list[str], list[str]]:
    """Return ``(failures, notes)`` for a results payload."""
    failures: list[str] = []
    notes: list[str] = []

    crossovers = payload.get("crossovers", {})
    for result in payload.get("results", []):
        key = f"{result.get('algorithm')}/{result.get('operation')}"
        crossover = crossovers.get(key, {}).get("crossover_n_samples")
        n_samples = result["shape"][0]
        regressing = result.get("significant", False) and result["speedup"] < 1.0 - TOLERANCE

        if key in ACCEPTED_REGRESSIONS:
            notes.append(f"{result['label']}: {result['speedup']:.2f}x, accepted regression for {key}")
            continue

        if crossover is None:
            # No measured win anywhere. That is only acceptable if mlxlearn is also
            # not significantly *losing* — otherwise the absence of a crossover is
            # exactly the thing that needs reporting.
            if regressing:
                failures.append(
                    f"{result['label']}: {result['speedup']:.2f}x, and {key} never reaches a "
                    "crossover. Either fix it, or add it to ACCEPTED_REGRESSIONS with a "
                    "reason and record it in docs/benchmarks.md."
                )
            else:
                notes.append(f"{result['label']}: no crossover measured for {key}; not gated")
            continue
        if n_samples < crossover:
            notes.append(
                f"{result['label']}: below the {key} crossover ({n_samples} < {crossover}); "
                "dispatched to scikit-learn by design"
            )
            continue
        if not result.get("significant", False):
            notes.append(f"{result['label']}: {result['speedup']:.2f}x, within noise")
            continue
        if regressing:
            failures.append(
                f"{result['label']}: {result['speedup']:.2f}x is a significant regression "
                f"against scikit-learn at or above the {key} crossover "
                f"({n_samples} >= {crossover})"
            )

    return failures, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    failures, notes = evaluate(payload)

    if args.verbose:
        for note in notes:
            print(f"  note: {note}")

    if failures:
        print("performance gate FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"performance gate passed ({len(notes)} cases not gated or within noise)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
