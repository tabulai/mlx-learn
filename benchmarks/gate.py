"""The performance gate.

The design goal is "never slower than scikit-learn". The *gate* is narrower and
measurable, because a goal you cannot fail is not a gate:

> On every benchmarked workload, patched dispatch — with the crossover thresholds
> configured in ``Tuning`` — shows no statistically significant regression against
> stock scikit-learn.

Consequences worth being explicit about.

**The thresholds under test are the configured ones.** Each result records the
backend the mlxlearn side actually ran on, which is ``Tuning``'s decision. An
earlier version exempted everything below a crossover *measured in the same run*;
a regression on the MLX path pushed that crossover up and then excused itself as
"dispatched to scikit-learn by design" while dispatch was still sending those
problems to MLX.

**A problem handed to scikit-learn is still gated.** Patched dispatch is the
product, and when it hands a small problem to scikit-learn, the handoff's own
overhead is the only thing that can regress there. That is a real cost to users.

**A problem ``Tuning`` sends to MLX that loses means ``Tuning`` is stale** — or the
MLX path regressed. Either way it fails, and the message says which threshold.

**What the gate does not judge.** The Layer 1 CPU path under ``--layer direct``,
which is a documented contract for directly imported classes rather than the
performance claim; and a backend *forced* below the configured threshold, which is
a kernel measurement that dispatch never runs. A forced MLX win below the threshold
is reported, since it says ``Tuning`` may be leaving speed on the table.

**"No significant regression" is not "faster".** A case that lands within the
measured noise passes. Requiring a win on every case would make the gate a
lottery on a shared machine; requiring no *loss* is the claim mlxlearn actually
makes.

**A loss has to reproduce.** The significance test bounds the false-alarm rate per
case, near 1%, and a nightly run gates over a hundred cases, which on noise alone
would fail most nights. So the runner times any case that would fail again, over
more iterations, and records that as the result's ``confirmation``; a case whose
loss does not reproduce there passes, with a note. A result without a confirmation
(an older payload) is judged on its one measurement.

A structural regression is **not** silently exempt. If mlxlearn is significantly
slower somewhere by design, it gets an explicit entry in ``ACCEPTED_REGRESSIONS``
with a reason — because "we never measured a win here" is the most comfortable
place for a real regression to hide.

Per-operation, never averaged. An average over fit and predict describes a
workload nobody runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._harness import MIN_SIGNIFICANCE_SAMPLES

__all__ = ["main", "evaluate"]

#: Regressions smaller than this are ignored even when they clear the noise
#: threshold, so that a 1% drift does not fail a nightly run.
TOLERANCE = 0.05

#: Operations where mlxlearn is knowingly slower than scikit-learn, with the reason.
#:
#: An entry here is a decision, not an excuse: it says the regression is
#: structural, understood, and paid back elsewhere. Without this map, an operation
#: that never wins would have no way to pass the gate except by hiding.
ACCEPTED_REGRESSIONS: dict[str, str] = {
    "knn/fit": (
        "mlxlearn's KNN fit uploads the training rows once and builds, on the device, "
        "the translated and scaled copy the search runs on, with its row norms (plus a "
        "float32 residual for float64 input); scikit-learn's brute-force fit stores a "
        "reference and does no work. Under patch_sklearn() the fit also probes 512 "
        "of its own rows with the first search pass, and a fit that probe hands to "
        "scikit-learn has paid for the index and the probe first. The gap is linear in "
        "the data, and where patched dispatch uses MLX the faster queries repay it "
        "after a few hundred queries. The measured ranges are in docs/benchmarks.md."
    ),
}


def _is_win(result: dict) -> bool:
    return (
        result.get("significant", False)
        and result["speedup"] > 1.0
        and result.get("speedup_best", result["speedup"]) > 1.0
    )


def _regressing(result: dict) -> bool:
    return result.get("significant", False) and result["speedup"] < 1.0 - TOLERANCE


def evaluate(payload: dict) -> tuple[list[str], list[str]]:
    """Return ``(failures, notes)`` for a results payload.

    Accepts schema 1 payloads, which recorded no backend: every result in them is
    gated, since nothing says any of them was handed off by design.
    """
    failures: list[str] = []
    notes: list[str] = []
    payload_layer = payload.get("layer", "patched")

    for result in payload.get("results", []):
        key = f"{result.get('algorithm')}/{result.get('operation')}"
        label = result["label"]
        speedup = result["speedup"]
        significant = result.get("significant", False)
        regressing = _regressing(result)
        layer = result.get("layer", payload_layer)
        backend = result.get("backend")
        forced = result.get("force_backend") or payload.get("force_backend")
        dispatch = result.get("dispatch", backend)

        if key in ACCEPTED_REGRESSIONS:
            notes.append(f"{label}: {speedup:.2f}x, accepted regression for {key}")
            continue

        if forced and dispatch != forced:
            # A kernel measurement below the configured threshold: dispatch would not
            # have run it, so it is not the claim. A win here is still news.
            if _is_win(result):
                notes.append(
                    f"{label}: {speedup:.2f}x on forced {forced}, but the configured Tuning "
                    f"hands this to {dispatch}. Tuning may be too conservative for {key}; "
                    "re-measure before lowering it."
                )
            else:
                notes.append(f"{label}: forced onto {forced} below the Tuning threshold; not gated")
            continue

        if layer == "direct" and backend == "cpu":
            notes.append(
                f"{label}: Layer 1 CPU path below the Tuning threshold; the performance claim "
                "is about patched dispatch, so not gated"
            )
            continue

        if not significant:
            iterations = min(
                result.get("mlxlearn", {}).get("iterations", MIN_SIGNIFICANCE_SAMPLES),
                result.get("sklearn", {}).get("iterations", MIN_SIGNIFICANCE_SAMPLES),
            )
            why = (
                f"{iterations} iteration(s), too few to judge"
                if iterations < MIN_SIGNIFICANCE_SAMPLES
                else "within noise"
            )
            notes.append(f"{label}: {speedup:.2f}x on {backend or 'unrecorded backend'}, {why}")
            continue
        if not regressing:
            continue
        shown = f"{speedup:.2f}x"
        confirmation = result.get("confirmation")
        if confirmation is not None:
            again = (
                f"{confirmation['speedup']:.2f}x over "
                f"{confirmation.get('mlxlearn', {}).get('iterations', '?')} iterations"
            )
            if not _regressing(confirmation):
                notes.append(f"{label}: {shown} did not reproduce ({again}); not gated")
                continue
            shown = f"{shown} (reproduced at {again})"

        if backend == "sklearn":
            failures.append(
                f"{label}: {shown}. The configured Tuning hands this to scikit-learn, so "
                "the loss is patched dispatch's own overhead. Fix it, or add it to "
                "ACCEPTED_REGRESSIONS with a reason and record it in docs/benchmarks.md."
            )
        elif backend == "mlx":
            failures.append(
                f"{label}: {shown} is a significant regression on the MLX path, which "
                f"the configured Tuning selects for this size. Either the MLX path regressed or "
                f"the {key} threshold in Tuning is too low; re-measure it."
            )
        else:
            failures.append(
                f"{label}: {shown} is a significant regression against scikit-learn "
                f"(backend {backend or 'not recorded'})."
            )

    notes.extend(_threshold_notes(payload))
    return failures, notes


def _threshold_notes(payload: dict) -> list[str]:
    """Compare the crossover measured in this run with where Tuning switches to MLX.

    Informational: the two disagreeing is how a stale threshold first shows up, but
    a measured crossover is only as good as the grid's resolution.
    """
    first_mlx: dict[str, int] = {}
    for result in payload.get("results", []):
        if result.get("dispatch", result.get("backend")) == "mlx":
            key = f"{result.get('algorithm')}/{result.get('operation')}"
            n = result["shape"][0]
            first_mlx[key] = min(n, first_mlx.get(key, n))
    notes = []
    for key, entry in sorted(payload.get("crossovers", {}).items()):
        measured = entry.get("crossover_n_samples")
        configured = first_mlx.get(key)
        if measured is None and configured is None:
            continue
        notes.append(
            f"{key}: measured crossover {measured if measured is not None else 'not reached'}; "
            f"the configured Tuning first selects MLX at n_samples="
            f"{configured if configured is not None else 'none in this run'}"
        )
    return notes


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

    print(f"performance gate passed ({len(notes)} notes: cases not gated, within noise, or drift)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
