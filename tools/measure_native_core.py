#!/usr/bin/env python3
"""Reproduce the compiled-extension measurement behind `phase0/cython_decision.md`.

Runs the frozen ancestor's ``SVC.fit`` end to end with its Cython SMO core enabled
and disabled, and reports the ratio per problem size. End to end on purpose: the
earlier microbenchmark that justified the extension timed the SMO inner loop in
isolation, found a real 1.5x, and missed that the 20 ms it saved sat inside a
3.7-second fit.

The ancestor is opened read-only. Nothing in it is modified or imported into
mlxlearn.

    python tools/measure_native_core.py --source /path/to/ancestor
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

DEFAULT_SOURCE = Path("/Users/tunguz/Programming/scikit_learn_mps")
SIZES = ((2_000, 16), (6_000, 16), (12_000, 32))


def _bench(source: Path, native: str, sizes, repeats: int) -> dict[str, float]:
    import numpy as np

    os.environ["SKMPS_SVC_SMO_USE_NATIVE_CORE"] = native
    for module in [m for m in list(sys.modules) if m.split(".")[0] in ("mpsbackend", "sklearnex")]:
        del sys.modules[module]

    from sklearnex.svm import SVC  # noqa: PLC0415 - deliberately re-imported per setting

    results: dict[str, float] = {}
    for n_samples, n_features in sizes:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n_samples, n_features))
        y = (X[:, 0] + 0.7 * rng.normal(size=n_samples) > 0).astype(int)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            SVC(kernel="rbf", C=1.0).fit(X, y)
            samples.append(time.perf_counter() - start)
        results[f"{n_samples}x{n_features}"] = round(statistics.median(samples), 4)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    if not (args.source / "mpsbackend").is_dir():
        print(f"no ancestor checkout at {args.source}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(args.source))

    compiled = _bench(args.source, "1", SIZES, args.repeats)
    pure = _bench(args.source, "0", SIZES, args.repeats)

    print(
        json.dumps(
            {
                "compiled_core_seconds": compiled,
                "pure_python_seconds": pure,
                "pure_over_compiled": {k: round(pure[k] / compiled[k], 2) for k in compiled},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
