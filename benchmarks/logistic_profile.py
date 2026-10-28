"""Profile complete wide fits by stage against the frozen native baseline.

    python -m benchmarks.logistic_profile --dtype both \
        --output bench/logistic-profile.json

These instrumented timings explain costs; use ``logistic_wide`` for speedup
claims. Wrappers do not add GPU synchronization inside the solver. In particular,
centering and initialization are lazy and execute inside the optimizer timer.
Run standalone because the temporary instrumentation is not thread-safe.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from threadpoolctl import threadpool_limits

from mlxlearn._solvers import logistic, native_logistic
from mlxlearn.linear_model import _logistic as estimator_module

from .logistic_gpu import Case, make_data, record_environment
from .logistic_wide import audit_sources, fit_once, frozen_reference, package_source_paths

CASES = (
    Case("wide_20k_1024", 20_000, 1024),
    Case("wide_60k_1024", 60_000, 1024),
    Case("wide_40k_2048", 40_000, 2048),
)


@contextlib.contextmanager
def instrument(estimator, solver, native):
    """Time nested calls, restoring every module even if a fit fails."""
    totals = {}
    with contextlib.ExitStack() as restore:

        def timed(function, key):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    totals[key] = totals.get(key, 0.0) + 1000 * (time.perf_counter() - start)

            return wrapper

        def replace(module, name, value):
            previous = getattr(module, name)
            restore.callback(setattr, module, name, previous)
            setattr(module, name, value)

        def timed_context(factory):
            def wrapper(*args, **kwargs):
                manager = factory(*args, **kwargs)

                class TimedContext:
                    # Time the actual enter/exit operations. Timing only factory
                    # invocation would measure generator construction; wrapping
                    # the full with-block would also include the optimizer.
                    def __enter__(self):
                        return timed(manager.__enter__, "centered_input_prepare_ms")()

                    def __exit__(self, *exception):
                        return timed(manager.__exit__, "centered_input_exit_drain_ms")(*exception)

                return TimedContext()

            return wrapper

        for name, key in (
            ("_prepare_logistic_input", "conversion_and_range_check_ms"),
            ("validate_data", "public_validation_ms"),
            # Only frozen reference packages still define this stage.
            ("_upload_logistic_input", "owned_upload_and_finite_check_ms"),
            ("solve_logistic", "solver_inclusive_ms"),
        ):
            if hasattr(estimator, name):
                replace(estimator, name, timed(getattr(estimator, name), key))
        name = "_prepare_centered_logistic_input"
        if hasattr(estimator, name):
            replace(estimator, name, timed_context(getattr(estimator, name)))
        replace(native, "minimize_binary", timed(native.minimize_binary, "optimizer_inclusive_ms"))
        original_factory = solver._native_binary_objective

        def objective_factory(*args, **kwargs):
            return timed(original_factory(*args, **kwargs), "direct_verification_ms")

        replace(solver, "_native_binary_objective", objective_factory)
        yield totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--reference-dir", type=Path)
    reference.add_argument("--reference-archive", type=Path)
    parser.add_argument("--dtype", choices=("float32", "float64", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("bench/logistic-profile.json"))
    args = parser.parse_args(argv)
    if args.repeats < 3 or args.warmup < 1 or args.threads < 1:
        parser.error(
            "at least three repetitions, one warmup, and a positive thread limit are required"
        )
    dtypes = ("float32", "float64") if args.dtype == "both" else (args.dtype,)
    payload = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": record_environment(),
        "method": "Rotated fresh fits; nested stage timers, no extra GPU synchronization",
        "repeats": args.repeats,
        "warmup": args.warmup,
        "requested_thread_limit": args.threads,
        "caveat": "Inclusive optimizer time contains deferred centering and initialization; "
        "stage medians need not add up to median total fit time. These are instrumented diagnostics.",
        "results": [],
    }
    paths = [Path(__file__), Path(fit_once.__code__.co_filename)] + package_source_paths()
    payload["environment"]["source_sha256"].update(
        {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    )
    with (
        frozen_reference(args.reference_dir, archive=args.reference_archive) as reference,
        threadpool_limits(limits=args.threads),
    ):
        payload["reference"] = {**reference.provenance, "manifest": reference.manifest}
        for dtype in dtypes:
            for case in CASES:
                X, y, weights = make_data(case, 41, dtype=dtype)
                samples = {"previous_native": [], "optimized_gpu": []}
                for repetition in range(args.repeats + args.warmup):
                    variants = tuple(samples) if repetition % 2 == 0 else tuple(samples)[::-1]
                    for variant in variants:
                        modules = (
                            (reference.estimator, reference.solver, reference.native)
                            if variant == "previous_native"
                            else (estimator_module, logistic, native_logistic)
                        )
                        with instrument(*modules) as stages:
                            elapsed, _, metadata = fit_once(
                                variant, X, y, weights, reference=reference, max_iter=200, tol=1e-4
                            )
                        stages["fit_ms"] = elapsed * 1000
                        stages["optimizer_excluding_verification_ms"] = (
                            stages["optimizer_inclusive_ms"] - stages["direct_verification_ms"]
                        )
                        if repetition >= args.warmup:
                            samples[variant].append(
                                {"stages": dict(stages), "fit_details": metadata}
                            )
                medians = {
                    variant: {
                        key: statistics.median(sample["stages"][key] for sample in entries)
                        for key in entries[0]["stages"]
                    }
                    for variant, entries in samples.items()
                }
                result = {
                    "case": case.name,
                    "shape": [case.n_samples, case.n_features],
                    "dtype": dtype,
                    "median_ms": medians,
                    "samples": samples,
                }
                payload["results"].append(result)
                print({key: value for key, value in result.items() if key != "samples"}, flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
        payload["source_integrity"] = audit_sources(payload, reference)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if payload["source_integrity"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
