"""Paired resident fits of frozen and current native-BF16 Metal e-means.

The unchanged h100 worker supplies timing, warm-up, adaptive repetitions,
memory handshakes, dataset loading and quality evaluation. Both variants use
assignment='auto'. Only the explicitly recorded runtime modules may differ.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import sys
import tempfile
import types
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h100  # noqa: E402

base = h100.base
KIND = "metal_gap_paired"
MODES = ("previous", "optimized")
ARCHIVE = HERE / "reference/metal-v2-source.zip"
ARCHIVE_MANIFEST = ARCHIVE.with_suffix(".json")
CORE_MODULES = {
    "mlxlearn.cluster": "src/mlxlearn/cluster/__init__.py",
    "mlxlearn.cluster._e_means": "src/mlxlearn/cluster/_e_means.py",
    "mlxlearn.cluster._recipe": "src/mlxlearn/cluster/_recipe.py",
    "mlxlearn._kernels.e_means": "src/mlxlearn/_kernels/e_means.py",
    "mlxlearn._kernels.e_means_bf16": "src/mlxlearn/_kernels/e_means_bf16.py",
}
_ORIGINAL_VALIDATE = h100.validate_result
_ORIGINAL_CHECK = h100.check_environment


def archive_contents(archive=ARCHIVE, manifest=ARCHIVE_MANIFEST):
    """Verify the whole snapshot and every member before using archived code."""
    archive, manifest = Path(archive), Path(manifest)
    metadata = json.loads(manifest.read_text())
    if base.file_sha256(archive) != metadata["archive_sha256"]:
        raise RuntimeError("Frozen runtime archive SHA-256 mismatch")
    expected = metadata["source_sha256"]
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise RuntimeError("Frozen runtime archive member list mismatch")
        blobs = {}
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Unsafe frozen archive member")
            blob = source.read(name)
            if hashlib.sha256(blob).hexdigest() != expected[name]:
                raise RuntimeError(f"Frozen runtime member SHA-256 mismatch: {name}")
            if name.startswith("src/mlxlearn/"):
                blobs[name] = blob
    if not set(CORE_MODULES.values()) <= set(blobs):
        raise RuntimeError("Frozen runtime archive is missing required e-means modules")
    return metadata, blobs


def runtime_sources(mode, archive=ARCHIVE, manifest=ARCHIVE_MANIFEST):
    if mode == "previous":
        _, blobs = archive_contents(archive, manifest)
        return {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}
    if mode != "optimized":
        raise ValueError(f"Unknown runtime mode: {mode}")
    paths = [*sorted((base.REPO / "src/mlxlearn/cluster").glob("*.py")),
             *sorted((base.REPO / "src/mlxlearn/_kernels").glob("e_means*.py"))]
    result = {str(path.relative_to(base.REPO)): base.file_sha256(path) for path in paths}
    if not set(CORE_MODULES.values()) <= set(result):
        raise RuntimeError("Current tree is missing required e-means modules")
    return result


def shared_sources(protocol, archive=ARCHIVE, manifest=ARCHIVE_MANIFEST):
    paths = [Path(__file__), HERE / "h100.py", HERE / "run.py", HERE / "memory.py",
             HERE / "vibe_manifest.json", Path(protocol), Path(archive), Path(manifest)]
    return {(str(path.resolve().relative_to(base.REPO)) if path.resolve().is_relative_to(base.REPO)
             else str(path.resolve())): base.file_sha256(path) for path in paths}


def selected_sources(config):
    return {**shared_sources(config["protocol_path"], config["archive"], config["archive_manifest"]),
            **runtime_sources(config["mode"], config["archive"], config["archive_manifest"])}


def _namespace(name, folder):
    module = types.ModuleType(name)
    module.__path__ = [str(folder)]
    module.__package__ = name
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = module.__path__
    sys.modules[name] = module


class _VerifiedSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, digest):
        super().__init__(name, path)
        self.digest = digest

    def get_code(self, fullname):
        # Compile the checked bytes directly: timestamp-based .pyc reuse would
        # not prove that the code executed matches the recorded source hash.
        content = Path(self.path).read_bytes()
        if hashlib.sha256(content).hexdigest() != self.digest:
            raise RuntimeError(f"Runtime source changed before import: {fullname}")
        return compile(content, self.path, "exec", dont_inherit=True)

    def exec_module(self, module):
        super().exec_module(module)
        module.__loaded_source_sha256__ = self.digest


class _RuntimeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, expected):
        self.root, self.expected = root, expected

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("mlxlearn."):
            return None
        stem = "src/" + fullname.replace(".", "/")
        for relative in (stem + "/__init__.py", stem + ".py"):
            if relative in self.expected:
                filename = str(self.root / relative)
                loader = _VerifiedSourceLoader(fullname, filename, self.expected[relative])
                return importlib.util.spec_from_file_location(fullname, filename, loader=loader)
        raise ImportError(f"Undeclared runtime module: {fullname}")


def install_runtime(root, expected=None):
    """Import only the selected e-means tree, without current package startup."""
    if any(name == "mlxlearn" or name.startswith("mlxlearn.") for name in sys.modules):
        raise RuntimeError("Runtime isolation requires a fresh process without imported mlxlearn modules")
    root = Path(root).resolve()
    if expected is None:
        expected = {str(path.relative_to(root)): base.file_sha256(path)
                    for path in (root / "src/mlxlearn").rglob("*.py")}
    _namespace("mlxlearn", root / "src/mlxlearn")
    _namespace("mlxlearn._kernels", root / "src/mlxlearn/_kernels")
    finder = _RuntimeFinder(root, expected)
    sys.meta_path.insert(0, finder)
    try:
        importlib.import_module("mlxlearn.cluster")
    finally:
        sys.meta_path.remove(finder)


def runtime_proof(root, expected):
    """Prove each actual loaded package module belongs to the selected tree."""
    root = Path(root).resolve()
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        if name != "mlxlearn" and not name.startswith("mlxlearn."):
            continue
        if name in {"mlxlearn", "mlxlearn._kernels"}:
            suffix = "src/mlxlearn" if name == "mlxlearn" else "src/mlxlearn/_kernels"
            if list(getattr(module, "__path__", ())) != [str(root / suffix)]:
                raise RuntimeError(f"Runtime namespace escaped selected tree: {name}")
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            raise RuntimeError(f"Loaded runtime module lacks source provenance: {name}")
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"Loaded runtime module escaped selected tree: {name}")
        relative = str(path.relative_to(root))
        digest = base.file_sha256(path)
        if expected.get(relative) != digest or getattr(module, "__loaded_source_sha256__", None) != digest:
            raise RuntimeError(f"Loaded runtime module hash is undeclared or changed: {name}")
        loaded[name] = {"relative_path": relative, "path": str(path), "sha256": digest}
    if not set(CORE_MODULES) <= set(loaded):
        raise RuntimeError("Not all selected e-means modules were loaded")
    return {"verified": True, "root": str(root), "modules": loaded,
            "synthetic_namespace_packages": ["mlxlearn", "mlxlearn._kernels"]}


def validate_runtime_record(result, config):
    if config.get("mode") not in MODES or config.get("assignment") != "auto":
        raise RuntimeError("Gap benchmark requires previous/optimized native assignment")
    comparison = config.get("runtime_comparison", {})
    if set(comparison) != set(MODES) or not all(comparison.values()):
        raise RuntimeError("Missing explicit two-runtime comparison provenance")
    expected = comparison[config["mode"]]
    shared = config.get("shared_source_sha256", {})
    if (not shared or config.get("expected_source_sha256") != {**shared, **expected}
        or result.get("runtime_source_sha256") != expected or result.get("shared_source_sha256") != shared):
        raise RuntimeError("Shared/runtime source provenance mismatch")
    proof = result.get("loaded_runtime", {})
    modules = proof.get("modules", {})
    if proof.get("verified") is not True or not set(CORE_MODULES) <= set(modules):
        raise RuntimeError("Missing actual loaded-runtime proof")
    root = Path(proof.get("root", ""))
    if not root.is_absolute():
        raise RuntimeError("Loaded runtime root must be absolute")
    for name, item in modules.items():
        relative = item.get("relative_path", "")
        if (not name.startswith("mlxlearn.") or Path(relative).is_absolute() or ".." in Path(relative).parts
            or expected.get(relative) != item.get("sha256")
            or Path(item.get("path", "")) != root / relative
            or name in CORE_MODULES and relative != CORE_MODULES[name]):
            raise RuntimeError("Loaded runtime proof differs from declared sources")
    if config["mode"] == "previous" and proof.get("origin") != config.get("archive_sha256"):
        raise RuntimeError("Loaded previous runtime is not bound to the frozen archive")
    if config["mode"] == "optimized" and proof.get("origin") != "current_source_tree":
        raise RuntimeError("Loaded optimized runtime has invalid origin")


def validate_result(result, config, memories):
    _ORIGINAL_VALIDATE(result, config, memories)
    validate_runtime_record(result, config)


@contextlib.contextmanager
def adapter(config):
    """Redirect the unchanged controller's subprocess to this worker adapter."""
    overrides = {"__file__": str(Path(__file__).resolve()), "KIND": KIND,
                 "ASSIGNMENT": dict.fromkeys(MODES, "auto"),
                 "sources": lambda protocol: selected_sources(config),
                 "validate_result": validate_result}
    original = {key: getattr(h100, key) for key in overrides}
    for key, value in overrides.items():
        setattr(h100, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(h100, key, value)


def worker(config):
    with adapter(config), tempfile.TemporaryDirectory(prefix="emeans-frozen-") as temporary:
        if config["mode"] == "previous":
            metadata, blobs = archive_contents(config["archive"], config["archive_manifest"])
            if metadata["archive_sha256"] != config["archive_sha256"]:
                raise RuntimeError("Frozen archive differs from campaign configuration")
            root = Path(temporary)
            for name, blob in blobs.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
        else:
            root = base.REPO
        expected = config["runtime_comparison"][config["mode"]]
        install_runtime(root, expected)

        def check_environment(current_config):
            _ORIGINAL_CHECK(current_config)
            runtime_proof(root, expected)

        original_check = h100.check_environment
        h100.check_environment = check_environment
        try:
            # No timing, training, memory-window or evaluator code is replaced.
            h100.worker(config)
            proof = runtime_proof(root, expected)
            proof["origin"] = config["archive_sha256"] if config["mode"] == "previous" else "current_source_tree"
            path = Path(config["worker_output"])
            result = json.loads(path.read_text())
            result.update(runtime_source_sha256=expected, shared_source_sha256=config["shared_source_sha256"],
                          loaded_runtime=proof,
                          comparison_scope="fresh paired native-BF16 fits; only explicitly recorded runtime sources differ; historical campaign timings are not paired")
            base.atomic_json(path, result)
        finally:
            h100.check_environment = original_check


def pair_results(records):
    """Pair new measurements, allowing only the explicit runtime difference."""
    groups = {}
    for row in records:
        config = row["config"]
        with adapter(config):
            validate_result(row, config, row.get("memory_samples", []))
        common_config = {key: value for key, value in config.items()
                         if key not in {"mode", "expected_source_sha256", "output", "worker_output"}}
        identity = {"config": common_config,
                    **{key: row.get(key) for key in ("n", "d", "k", "dataset", "initial_centroids_sha256",
                       "versions", "hardware", "metal_device", "recipe_resolved", "timing_scope", "scoring", "memory_scope")}}
        key = json.dumps(identity, sort_keys=True)
        modes = groups.setdefault(key, {})
        if config["mode"] in modes:
            raise RuntimeError("Duplicate gap benchmark variant for the same comparison")
        modes[config["mode"]] = row
    paired, unpaired = [], []
    for modes in groups.values():
        if set(modes) != set(MODES):
            unpaired.extend(modes.values())
            continue
        previous, optimized = modes["previous"], modes["optimized"]
        paired.append({"previous": previous, "optimized": optimized,
                       "speedup": previous["train_seconds"] / optimized["train_seconds"],
                       "quality_delta_pct": 100 * (optimized["test_mse"] / previous["test_mse"] - 1) if previous["test_mse"] else None})
    return paired, unpaired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--suite", choices=["core1m", "vibe", "all"], default="all")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12])
    parser.add_argument("--protocol", type=Path, default=HERE / "h100_protocol.json")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--archive-manifest", type=Path, default=ARCHIVE_MANIFEST)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--train-score-rows", type=int, default=10000)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--memory-interval", type=float, default=.01)
    parser.add_argument("--timeout", type=float, default=8 * 3600)
    parser.add_argument("--cache", default=str(base.DEFAULT_CACHE))
    parser.add_argument("--faiss-source", default=str(base.REPO / "tmp/e_means/faiss"))
    parser.add_argument("--out", type=Path, default=HERE / "results/gap_protocol")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker))
        return
    if (args.threads < 1 or args.train_score_rows < 0
        or any(v is not None and (not math.isfinite(v) or v <= 0) for v in (args.rows, args.k, args.epochs, args.memory_interval, args.timeout))
        or any(not 0 <= seed < 2**32 for seed in args.seeds)
        or len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes)):
        parser.error("Invalid sizes/timing/threads/seeds or duplicate modes/seeds")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    entries = protocol["configurations"]
    if args.datasets and not set(args.datasets) <= {entry["dataset"] for entry in entries}:
        parser.error("Unknown dataset")
    metadata, _ = archive_contents(args.archive, args.archive_manifest)
    comparison = {mode: runtime_sources(mode, args.archive, args.archive_manifest) for mode in MODES}
    shared = shared_sources(protocol_path, args.archive, args.archive_manifest)
    versions, hardware = base.package_versions(), base.host_identity()
    for index, entry in enumerate(entries):
        if args.datasets and entry["dataset"] not in args.datasets or args.suite != "all" and entry["suite"] != args.suite:
            continue
        for seed in args.seeds:
            modes = args.modes if (index + seed) % 2 == 0 else args.modes[::-1]
            for mode in modes:
                config = {"dataset": entry["dataset"], "mode": mode, "seed": seed,
                          "epochs": args.epochs if args.epochs is not None else entry["epochs"],
                          "epochs_override": args.epochs, "rows": args.rows, "k_override": args.k,
                          "precision": "bfloat16", "recipe": "faiss", "finalization": "last_epoch", "assignment": "auto",
                          "warmup": True, "capture_assignments": True, "threads": args.threads,
                          "train_score_rows": args.train_score_rows, "memory": not args.no_memory,
                          "memory_interval_seconds": args.memory_interval, "protocol_path": str(protocol_path),
                          "protocol_config": entry, "cache": str(Path(args.cache).resolve()),
                          "faiss_source": str(Path(args.faiss_source).resolve()), "archive": str(args.archive.resolve()),
                          "archive_manifest": str(args.archive_manifest.resolve()), "archive_sha256": metadata["archive_sha256"],
                          "runtime_comparison": comparison, "shared_source_sha256": shared,
                          "expected_source_sha256": {**shared, **comparison[mode]},
                          "expected_versions": versions, "expected_host": hardware}
                digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
                path = args.out / f"{entry['dataset']}-{mode}-s{seed}-{digest}.json"
                config.update(output=str(path.resolve()), worker_output=str(path.resolve()) + ".worker.tmp")
                with adapter(config):
                    if path.exists() and not args.force:
                        saved = json.loads(path.read_text())
                        validate_result(saved, config, saved.get("memory_samples", []))
                        print(f"SKIP {path}", flush=True)
                        continue
                    h100.measured_run(config, args.timeout)


if __name__ == "__main__":
    main()
