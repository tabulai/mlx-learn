"""Lloyd at its 25-iteration reference budget versus e-means at its own budget.

Addendum to kmeans_compare.py, whose campaign caps Lloyd at the e-means pass
budget. Here Metal KMeans runs 25 iterations and Metal e-means is re-measured
at its frozen budget from the same sources, so the pair tests the paper's
claim. The frozen worker, controller and validator are reused unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kmeans_compare as benchmark  # noqa: E402
import large_compare as previous  # noqa: E402

import run as base  # noqa: E402

KIND = "lloyd_reference_addendum_v1"
MODES = ("metal-kmeans", "metal-emeans")


def load_protocol(path):
    """Return the addendum and the frozen dataset entries it applies to."""
    protocol = json.loads(Path(path).read_text())
    frozen = base.REPO / protocol["applies_to"]["path"]
    iterations = protocol["lloyd_iterations"]
    if (protocol.get("kind") != KIND or tuple(protocol["modes"]) != MODES
            or isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1):
        raise ValueError("Not a Lloyd reference addendum")
    if base.file_sha256(frozen) != protocol["applies_to"]["sha256"]:
        raise ValueError("Frozen comparison protocol differs from the one this addendum applies to")
    return protocol, json.loads(frozen.read_text())["configurations"]


def protocol_entry(entry, mode, iterations):
    """The frozen entry; only the Lloyd arm's pass cap becomes the reference budget."""
    return {**entry, "epochs": iterations} if mode == "metal-kmeans" else entry


def addendum_sources():
    """Measured sources the frozen harness does not hash: this driver and the device package."""
    paths = [Path(__file__), *sorted((base.REPO / "src/mlxlearn_device").rglob("*.py"))]
    return {str(p.resolve().relative_to(base.REPO)): base.file_sha256(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", help="run these, in this order (default: all, protocol order)")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--rows", type=int, help="diagnostic subset")
    parser.add_argument("--k", type=int, help="diagnostic subset")
    parser.add_argument("--lloyd-iterations", type=int, help="diagnostic override of the Lloyd budget")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=8 * 3600)
    parser.add_argument("--protocol", type=Path, default=HERE / "kmeans_lloyd25_protocol.json")
    parser.add_argument("--cache", default=str(base.DEFAULT_CACHE))
    parser.add_argument("--faiss-source", default=str(base.REPO / "tmp/e_means/faiss"))
    parser.add_argument("--out", type=Path, default=HERE / "results/kmeans_lloyd25")
    args = parser.parse_args()
    overrides = (args.rows, args.k, args.lloyd_iterations)
    if (args.threads < 1 or any(v is not None and (not math.isfinite(v) or v <= 0) for v in (*overrides, args.timeout))
            or len(set(args.modes)) != len(args.modes)):
        parser.error("Invalid sizes, timeout, threads or duplicate modes")
    protocol_path = args.protocol.resolve()
    protocol, entries = load_protocol(protocol_path)
    by_name = {entry["dataset"]: entry for entry in entries}
    names = args.datasets or list(by_name)
    if len(set(names)) != len(names) or not set(names) <= by_name.keys():
        parser.error("Unknown or duplicate datasets")
    iterations = args.lloyd_iterations if args.lloyd_iterations is not None else protocol["lloyd_iterations"]
    captured = {"expected_source_sha256": benchmark.sources(protocol_path),
                "expected_reference_source_sha256": previous.reference_sources(args.faiss_source),
                "expected_sklearn_sha256": benchmark.sklearn_sources(),
                "expected_versions": base.package_versions(), "expected_host": base.host_identity(),
                "addendum_source_sha256": addendum_sources()}
    for index, name in enumerate(names):
        # Alternate which arm runs first, as the frozen campaign rotates its modes.
        for mode in args.modes[index % len(args.modes):] + args.modes[:index % len(args.modes)]:
            entry = protocol_entry(by_name[name], mode, iterations)
            config = {"dataset": name, "mode": mode, "seed": 10, "epochs": entry["epochs"],
                      "rows": args.rows, "k_override": args.k,
                      "epochs_override": args.lloyd_iterations if mode == "metal-kmeans" else None,
                      "diagnostic_subset": any(v is not None for v in overrides) or args.threads != 8,
                      "precision": "float32", "n_init": 1, "tol": 0.0, "final_labels": True, "warmup": True,
                      "threads": args.threads, "train_score_rows": 10000, "memory": True,
                      "memory_interval_seconds": .01, "protocol_path": str(protocol_path),
                      "protocol_config": entry, "cache": str(Path(args.cache).resolve()),
                      "faiss_source": str(Path(args.faiss_source).resolve()), "addendum": KIND, **captured}
            digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
            path = args.out / f"{name}-{mode}-s10-{digest}.json"
            config.update(output=str(path.resolve()), worker_output=str(path.resolve()) + ".worker.tmp")
            if path.exists():
                saved = json.loads(path.read_text())
                benchmark.validate_result(saved, config, saved.get("memory_samples", []))
                print(f"SKIP {path}", flush=True)
                continue
            benchmark.measured_run(config, args.timeout)
            if addendum_sources() != captured["addendum_source_sha256"]:
                path.unlink()
                raise RuntimeError("Addendum driver or device sources changed during the measurement")


if __name__ == "__main__":
    main()
