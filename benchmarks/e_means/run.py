"""Reproducible Apple Silicon e-means benchmarks on the paper's public data.

See README.md beside this file. Every measured configuration runs in a fresh
process, with identical starting centroids across implementations. Downloads,
data loading, initialization sampling, warm-up, and scoring are excluded from
fit timing. NumPy-to-device conversion and validation inside fit are included.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FAISS_REVISION = "ebab8f784c1d177129602269d5ad966c525fbdcb"
DEFAULT_CACHE = Path.home() / ".cache/mlxlearn/e_means"
CORE_1M = [
    "deep1b-1M", "spacev-1M", "fb-ssnpp-1M", "dino10b-1M-div4096",
    "imagenet-clip-512-normalized", "sift1b-1M-div512",
]
CORE_10M = [
    "deep1b-10M", "spacev-10M", "fb-ssnpp-10M", "dino10b-10M-div4096",
    "sift1b-10M-div512",
]
VIBE = {
    "vibe-agnews": "agnews-mxbai-1024-euclidean.hdf5",
    "vibe-arxiv": "arxiv-nomic-768-normalized.hdf5",
    "vibe-ccnews": "ccnews-nomic-768-normalized.hdf5",
    "vibe-celeba": "celeba-resnet-2048-cosine.hdf5",
    "vibe-codesearchnet": "codesearchnet-jina-768-cosine.hdf5",
    "vibe-glove": "glove-200-cosine.hdf5",
    "vibe-gooaq": "gooaq-distilroberta-768-normalized.hdf5",
    "vibe-imagenet": "imagenet-clip-512-normalized.hdf5",
    "vibe-landmark-dino": "landmark-dino-768-cosine.hdf5",
    "vibe-landmark-nomic": "landmark-nomic-768-normalized.hdf5",
    "vibe-simplewiki": "simplewiki-openai-3072-normalized.hdf5",
}
VIBE_SPLIT_VERSION = 1


def source_hashes():
    """Files defining this benchmark and its measured Metal implementation."""
    paths = [HERE / "run.py", HERE / "vibe_manifest.json",
             *sorted((REPO / "src/mlxlearn/cluster").glob("*.py")),
             REPO / "src/mlxlearn/_kernels/e_means.py"]
    return {str(path.relative_to(REPO)): file_sha256(path) for path in paths if path.exists()}


def package_versions():
    versions = {"python": platform.python_version()}
    for package in ["numpy", "scipy", "mlx", "scikit-learn", "faiss-cpu", "h5py"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def host_identity():
    return {
        "machine": platform.machine(), "platform": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "ram_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
    }


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def reference_modules(source, cache):
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != FAISS_REVISION:
        raise ValueError(f"Expected Faiss {FAISS_REVISION}; found {revision}")
    clean = subprocess.run(
        ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", "contrib/e_means", "benchs/e_means"],
        check=False,
    )
    if clean.returncode:
        raise ValueError("Reference checkout has changes; restore the pinned Faiss source first")
    bench = source / "benchs/e_means"
    sys.path.insert(0, str(bench))
    import paper_data
    import paper_download

    paper_data.ROOT = str(Path(cache) / "data")
    paper_data.CACHE = str(Path(cache) / "arrays")
    package = source / "contrib/e_means"
    spec = importlib.util.spec_from_file_location(
        "mlxlearn_faiss_reference", package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return paper_data, paper_download, module


def dataset_names(suite):
    return {"core1m": CORE_1M, "core10m": CORE_10M,
            "vibe": list(VIBE), "all": CORE_1M + CORE_10M + list(VIBE)}[suite]


def download(args):
    data, downloader, _ = reference_modules(args.faiss_source, args.cache)
    downloader.CHUNK = args.download_chunk_mib * (1 << 20)
    names = args.datasets or dataset_names(args.suite)
    files, held = {}, {}
    manifest = json.loads((HERE / "vibe_manifest.json").read_text())
    for name in names:
        if name == "sift1b-1M-div512":
            need = {"bigann/base.1B.u8bin": [(0, 8 + 1_010_000 * 128)]}
        elif name in VIBE:
            filename = VIBE[name]
            local = "vibe/" + filename
            meta = manifest["files"][filename]
            downloader.SOURCES[local] = {
                "url": "https://huggingface.co/datasets/" + manifest["repository"]
                + "/resolve/" + manifest["revision"] + "/" + filename,
                "layout": "whole", "size": meta["size"],
            }
            need = {local: [(0, meta["size"])]}
        else:
            need = downloader.needs(name)
            hold = downloader.heldout_needs(name)
            if hold:
                held[hold[0]] = hold[1]
        for f, ranges in need.items():
            files.setdefault(f, []).extend(ranges)
    files = {f: downloader.merge(ranges) for f, ranges in files.items()}
    total = sum(b - a for ranges in files.values() for a, b in ranges)
    print(f"Dataset bytes requested: {total / 1e9:.3f} GB (completed ranges resume)", flush=True)
    if args.dry_run:
        print(json.dumps(files, indent=2))
        return
    downloader.download(data.ROOT, files, args.download_threads)
    downloader.download_heldout(data.ROOT, held, args.download_threads)
    for name in names:
        if name in VIBE:
            filename = VIBE[name]
            path = Path(data.ROOT) / "vibe" / filename
            actual = file_sha256(path)
            if actual != manifest["files"][filename]["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {path}")
            print(f"SHA-256 verified {name}", flush=True)
        elif name != "sift1b-1M-div512":
            data.load(name)
            print(f"Paper fingerprints verified {name}", flush=True)


def _vibe_arrays(name, cache):
    """Public VIBE corpus, own deterministic, duplicate-disjoint holdout.

    The paper does not ship the eleven Appendix D.3 split indices. These must
    never be described as the exact paper split. Hashes and holdout indices are
    saved so the new split can be reproduced independently.
    """
    import h5py

    folder = Path(cache) / "arrays" / name
    train_path, test_path = folder / "train.npy", folder / "test.npy"
    meta_path = folder / "manifest.json"
    path = Path(cache) / "data/vibe" / VIBE[name]
    source_manifest = json.loads((HERE / "vibe_manifest.json").read_text())
    expected = source_manifest["files"][VIBE[name]]["sha256"]
    identity = {"split_version": VIBE_SPLIT_VERSION, "split_seed": 1234,
                "source_sha256": expected, "source_revision": source_manifest["revision"]}
    if meta_path.exists() and train_path.exists() and test_path.exists():
        metadata = json.loads(meta_path.read_text())
        if all(metadata.get(key) == value for key, value in identity.items()):
            train, test = np.load(train_path, mmap_mode="r"), np.load(test_path)
            samples = train[np.linspace(0, len(train) - 1, 1000).astype(int)]
            if (len(train) == metadata["train_rows"] and len(test) == metadata["test_rows"]
                and array_sha256(test) == metadata["test_sha256"]
                and array_sha256(samples) == metadata["train_sample_sha256"]):
                return train, test, metadata
            raise ValueError(f"Corrupt VIBE derived cache: {folder}; remove it and prepare again")
    if file_sha256(path) != expected:
        raise ValueError(f"VIBE source SHA-256 mismatch: {path}")
    with h5py.File(path, "r") as h:
        corpus = np.asarray(h["train"], dtype=np.float32)
    # Fixed-width bytes allow fast vectorized comparison of exact duplicates.
    byte_rows = np.ascontiguousarray(corpus).view(
        np.dtype((np.void, corpus.dtype.itemsize * corpus.shape[1]))
    ).ravel()
    hold = np.random.RandomState(1234).choice(len(corpus), 20_000, replace=False)
    keep = ~np.isin(byte_rows, byte_rows[hold])
    eval_order = np.random.RandomState(1234).permutation(len(hold))
    test = corpus[hold[eval_order[10_000:]]]
    train = corpus[keep]
    metadata = {
        **identity,
        "split": "new deterministic 20,000-row holdout, exact duplicates removed from train",
        "paper_split_exact": False,
        "source_sha256": expected,
        "source_revision": source_manifest["revision"],
        "source_train_rows": len(corpus), "train_rows": len(train),
        "removed_rows": int((~keep).sum()), "test_rows": len(test),
        "holdout_indices_sha256": array_sha256(hold),
        "test_sha256": array_sha256(test),
        "train_sample_sha256": array_sha256(train[np.linspace(0, len(train) - 1, 1000).astype(int)]),
        "normalization": "as stored in the VIBE file; Euclidean clustering",
    }
    folder.mkdir(parents=True, exist_ok=True)
    np.save(train_path, train)
    np.save(test_path, test)
    np.save(folder / "holdout_indices.npy", hold)
    atomic_json(meta_path, metadata)
    return np.load(train_path, mmap_mode="r"), test, metadata


def load_dataset(name, data, cache, rows=None):
    if name in VIBE:
        train, test, metadata = _vibe_arrays(name, cache)
        x = np.ascontiguousarray(train[:rows], dtype=np.float32)
    elif name == "sift1b-1M-div512":
        mat = data.matrix(data.find(data.SIFT))
        if len(mat) < 1_010_000:
            raise ValueError("SIFT1M requires 1,010,000 source vectors")
        x = np.array(mat[:min(rows or 1_000_000, 1_000_000)], dtype=np.float32) / 512.0
        test = data.halves(np.array(mat[1_000_000:1_010_000], dtype=np.float32) / 512.0)[1]
        metadata = {"paper_split_exact": False,
                    "split": "1M prefix with next 10,000 evaluation rows, seed1234 second half",
                    "normalization": "divide by 512 (exact power of two)",
                    "note": "same SIFT corpus/protocol; upstream has no SIFT1M fingerprints"}
    else:
        train, test = data.load(name)
        x = data.rows(train, name, stop=rows)
        fingerprints = json.loads((Path(data.SPLITS) / "fingerprints.json").read_text())[name]
        metadata = {"paper_split_exact": True, "fingerprints": fingerprints,
                    "split": "official upstream paper_data.py + paper_splits fingerprints"}
    metadata.update({"train_rows_used": len(x), "test_rows_used": len(test),
                     "dimension": x.shape[1], "requested_row_limit": rows,
                     "train_sample_sha256": array_sha256(
                         x[np.linspace(0, len(x) - 1, min(len(x), 1000)).astype(int)]),
                     "test_sha256": array_sha256(test)})
    return x, test, metadata


def mse(centroids, vectors, chunk=8192):
    """Faiss FP32 nearest assignment, independently recomputed FP64 residual."""
    import faiss

    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    total = 0.0
    for start in range(0, len(vectors), chunk):
        x = np.ascontiguousarray(vectors[start:start + chunk], dtype=np.float32)
        _, labels = index.search(x, 1)
        residual = x.astype(np.float64) - centroids[labels[:, 0]].astype(np.float64)
        total += np.square(residual).sum(dtype=np.float64)
    return float(total / len(vectors))


def worker(config):
    captured_sources, captured_versions = source_hashes(), package_versions()
    if config.get("expected_source_sha256", captured_sources) != captured_sources:
        raise RuntimeError("Benchmark source changed since campaign launch; start a fresh campaign")
    if config.get("expected_versions", captured_versions) != captured_versions:
        raise RuntimeError("Benchmark environment changed since campaign launch; start a fresh campaign")
    import faiss

    faiss.omp_set_num_threads(config["threads"])
    data, _, reference = reference_modules(config["faiss_source"], config["cache"])
    x, test, data_meta = load_dataset(config["dataset"], data, config["cache"], config["rows"])
    n, d = x.shape
    k = config["k"]
    if k is None:
        k = (int(2 ** round(np.log2(16 * np.sqrt(n)))) if config["dataset"] in VIBE
             else 16384 if n <= 2_000_000 else 65536 if n <= 20_000_000 else 262144)
    if k > n:
        raise ValueError("n_clusters exceeds the loaded training rows")
    initial_indices = np.random.RandomState(config["seed"]).permutation(n)[:k]
    initial = x[initial_indices].copy()
    method = config["method"]
    def sync():
        return None

    if method in {"metal", "metal-lloyd"}:
        import mlx.core as mx

        from mlxlearn.cluster import EMeans

        sync = mx.synchronize

    if method == "metal":

        def run():
            model = EMeans(
                n_clusters=k, max_iter=config["epochs"], random_state=config["seed"],
                init=initial, compute_labels=False, precision=config["precision"],
                recipe=config["recipe"], finalization=config["finalization"],
                batch_size=config["batch_size"], batch_law=config["batch_law"],
                assignment=config["assignment"],
            ).fit(x)
            return model, np.asarray(model.cluster_centers_, dtype=np.float32)
    elif method == "metal-lloyd":
        from mlxlearn._kernels.e_means import assign, batch_stats

        def run():
            # Same Metal assignment/reduction primitives as EMeans, held fixed
            # for a complete pass before each centroid update. Empty clusters
            # retain their previous location (a valid Lloyd fixed point).
            if not np.isfinite(x).all():
                raise ValueError("Non-finite input")
            with mx.stream(mx.gpu):
                data_mx, centers = mx.array(x), mx.array(initial)
                origin = mx.mean(data_mx, axis=0)
                mx.eval(data_mx, centers, origin)
                for _ in range(config["lloyd_iters"]):
                    sums = mx.zeros((k, d), mx.float32)
                    counts = mx.zeros((k,), mx.float32)
                    # Bound scatter sizes and uint indexing even for wide 10M data.
                    for begin in range(0, n, 65536):
                        batch = data_mx[begin:begin + 65536]
                        assignment_batch = batch - origin
                        labels, _ = assign(assignment_batch, centers - origin, method=config["assignment"],
                                           precision=config["precision"])
                        partial_sums, partial_counts = batch_stats(batch, labels, k)
                        sums, counts = sums + partial_sums, counts + partial_counts
                        mx.async_eval(sums, counts)
                    centers = mx.where((counts > 0)[:, None],
                                       sums / mx.maximum(counts[:, None], 1), centers)
                    mx.eval(centers)
                return SimpleNamespace(recipe={
                    "algorithm": "Lloyd", "iterations": config["lloyd_iters"],
                    "precision": config["precision"], "assignment": config["assignment"],
                    "empty_clusters": "retain previous centroid", "subsample": False,
                    "assignment_centering": "global mean, same as EMeans",
                }), np.array(centers)
    elif method == "faiss-emeans":
        def run():
            model = reference.NumpyEmeans(
                d, k, nepochs=config["epochs"], seed=config["seed"],
                capture_assignments=False, batch_size=config["batch_size"],
                batch_law=config["batch_law"],
            )
            model.train(x, init_centroids=initial)
            return model, model.centroids
    elif method == "faiss-lloyd":
        def run():
            model = faiss.Kmeans(
                d, k, niter=config["lloyd_iters"], seed=config["seed"],
                nredo=1, max_points_per_centroid=n, min_points_per_centroid=1,
            )
            model.train(x, init_centroids=initial)
            return model, model.centroids
    else:
        raise ValueError(method)
    if config["warmup"]:
        run()
        sync()
    samples = []
    for _ in range(config["repeats"]):
        sync()
        start = time.perf_counter()
        model, centroids = run()
        sync()
        samples.append(time.perf_counter() - start)
    test_mse = mse(centroids, test)
    # A deterministic sample is explicitly labelled; do not claim full train MSE.
    sample_indices = np.linspace(0, n - 1, min(n, config["train_score_rows"])).astype(int)
    train_mse = mse(centroids, x[sample_indices]) if len(sample_indices) else None
    result = {
        "status": "completed", "config": config, "dataset": data_meta,
        "n": n, "d": d, "k": k, "train_seconds": statistics.median(samples),
        "train_seconds_samples": samples, "test_mse": test_mse,
        "train_sample_mse": train_mse, "train_score_rows": len(sample_indices),
        "initial_centroids_sha256": array_sha256(initial),
        "final_centroids_sha256": array_sha256(centroids),
        "recipe_resolved": getattr(model, "recipe_", getattr(model, "recipe", None)),
        "algorithm_applied": (
            {"recipe": "current upstream Faiss", "finalization": "last_epoch", "precision": "float32"}
            if method == "faiss-emeans" else
            {"recipe": "Lloyd", "iterations": config["lloyd_iters"]}
            if method.endswith("lloyd") else
            {"recipe": config["recipe"], "finalization": config["finalization"], "precision": config["precision"]}
        ),
        "n_splits": getattr(model, "n_splits_", getattr(model, "n_splits", None)),
        "timing_scope": "synchronized fit from NumPy input; includes fit validation/upload; excludes data load, init sampling, warmup, scoring",
        "scoring": "Faiss IndexFlatL2 float32 assignment, residual recomputed by direct float64 difference",
        "initialization": "NumPy RandomState(seed) permutation shared across methods; differs paper CUDA RNG",
        "hardware": host_identity(),
        "versions": captured_versions, "faiss_reference_revision": FAISS_REVISION,
        "source_sha256": captured_sources,
    }
    if data_meta.get("paper_split_exact") and config["rows"] is None:
        targets = json.loads((Path(config["faiss_source"]) / "benchs/e_means/paper_targets.json").read_text())
        target = targets["table3"].get(config["dataset"])
        expected_k = 16384 if n <= 2_000_000 else 65536 if n <= 20_000_000 else 262144
        if target and k == expected_k:
            result["paper_quality_reference"] = {
                "method": "Flash-KMeans25, paper median over seeds10-19",
                "median_test_mse": target["median_test_mse"],
                "relative_mse": test_mse / target["median_test_mse"] - 1,
                "comparison": "same dataset/split/k; different initialization RNG and assignment precision",
                "runtime_comparison": "not applicable: paper timings are on an H100",
            }
    if method in {"metal", "metal-lloyd"}:
        result["hardware"]["metal"] = mx.metal.device_info()
        result["peak_gpu_memory_bytes"] = mx.get_peak_memory()
    if source_hashes() != captured_sources or package_versions() != captured_versions:
        raise RuntimeError("Source or environment changed during measurement; refusing to save mixed provenance")
    atomic_json(config["output"], result)
    print(json.dumps({"dataset": config["dataset"], "method": method, "n": n, "k": k,
                      "seconds": result["train_seconds"], "test_mse": test_mse,
                      "output": config["output"]}), flush=True)


def run_campaign(args):
    captured_sources, captured_versions = source_hashes(), package_versions()
    hardware = host_identity()
    for dataset in args.datasets or dataset_names(args.suite):
        for seed in args.seeds:
            for method in args.methods:
                epochs_values = args.epochs if not method.endswith("lloyd") else [args.epochs[0]]
                for epochs in epochs_values:
                    config = {
                        "dataset": dataset, "seed": seed, "method": method, "epochs": epochs,
                        "lloyd_iters": args.lloyd_iters, "k": args.k, "rows": args.rows,
                        "threads": args.threads, "precision": args.precision, "recipe": args.recipe,
                        "finalization": args.finalization, "assignment": args.assignment,
                        "batch_size": args.batch_size, "batch_law": args.batch_law,
                        "repeats": args.repeats, "warmup": not args.no_warmup,
                        "train_score_rows": args.train_score_rows,
                        "cache": str(Path(args.cache).resolve()),
                        "faiss_source": str(Path(args.faiss_source).resolve()),
                        "expected_source_sha256": captured_sources,
                        "expected_versions": captured_versions,
                        "expected_host": hardware,
                    }
                    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
                    out = Path(args.out) / f"{dataset}-{method}-s{seed}-{digest}.json"
                    if out.exists() and not args.force:
                        saved = json.loads(out.read_text())
                        if (saved.get("status") == "completed"
                            and saved.get("source_sha256") == captured_sources
                            and saved.get("versions") == captured_versions):
                            print(f"Already completed with matching source/environment: {out}", flush=True)
                            continue
                    config["output"] = str(out.resolve())
                    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    "--worker", json.dumps(config)], check=True, cwd=REPO,
                                   timeout=args.timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["download", "run", "list", "summary"])
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--faiss-source", default=str(REPO / "tmp/e_means/faiss"))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--suite", choices=["core1m", "core10m", "vibe", "all"], default="core1m")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=["metal", "metal-lloyd", "faiss-emeans", "faiss-lloyd"], default=["metal"])
    parser.add_argument("--epochs", nargs="+", type=float, default=[3.0])
    parser.add_argument("--lloyd-iters", type=int, default=25)
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12])
    parser.add_argument("--k", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--precision", default="float32", choices=["float32", "float16"])
    parser.add_argument("--recipe", default="paper", choices=["paper", "faiss"])
    parser.add_argument("--finalization", default="last_epoch", choices=["last_epoch", "ema", "slow_ema"])
    parser.add_argument("--assignment", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-law", default="auto", choices=["auto", "throughput", "quality"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--download-threads", type=int, default=8)
    parser.add_argument("--download-chunk-mib", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--train-score-rows", type=int, default=10_000)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker))
    elif args.command == "download":
        if args.download_threads < 1 or args.download_chunk_mib < 1:
            parser.error("download-threads and download-chunk-mib must be positive")
        download(args)
    elif args.command == "run":
        if args.repeats < 1 or args.threads < 1 or any(t <= 0 for t in args.epochs):
            parser.error("positive repeats, threads and epoch budgets are required")
        if (args.k is not None and args.k < 1) or (args.rows is not None and args.rows < 1):
            parser.error("k and rows must be positive")
        if args.lloyd_iters < 1 or args.train_score_rows < 0:
            parser.error("lloyd-iters must be positive and train-score-rows nonnegative")
        run_campaign(args)
    elif args.command == "list":
        print("\n".join(dataset_names(args.suite)))
    elif args.command == "summary":
        summarize(args.out)
    else:
        parser.error("choose download, run or list")


def summarize(folder):
    """Print individual measurements plus only initialization-matched ratios."""
    results = []
    for file in sorted(Path(folder).rglob("*.json")):
        row = json.loads(file.read_text())
        if row.get("status") == "completed":
            results.append(row)
    print("| Dataset | n | k | Method | Precision | Seed | Fit seconds | Test MSE |")
    print("|---|---:|---:|---|---|---:|---:|---:|")
    for r in results:
        c = r["config"]
        precision = c["precision"] if c["method"].startswith("metal") else "float32"
        print(f"| {c['dataset']} | {r['n']:,} | {r['k']:,} | {c['method']} | {precision} | "
              f"{c['seed']} | {r['train_seconds']:.6f} | {r['test_mse']:.9g} |")
    print("\nSame-device comparisons (paired initialization, dataset/split, precision and kernels):")
    for r in results:
        c = r["config"]
        if c["method"] != "metal":
            continue
        for b in results:
            bc = b["config"]
            if (bc["method"] == "metal-lloyd" and c["dataset"] == bc["dataset"]
                and c["precision"] == bc["precision"] and r["n"] == b["n"]
                and c["assignment"] == bc["assignment"]
                and r["k"] == b["k"] and r["initial_centroids_sha256"] == b["initial_centroids_sha256"]
                and r["dataset"] == b["dataset"]
                and r.get("source_sha256") and r["source_sha256"] == b.get("source_sha256")
                and r["versions"] == b["versions"] and r["hardware"] == b["hardware"]):
                print(f"- {c['dataset']} seed {c['seed']}, {c['precision']}, {c['epochs']:g} epochs: "
                      f"{b['train_seconds'] / r['train_seconds']:.2f}x faster than "
                      f"Metal Lloyd{bc['lloyd_iters']}; test MSE "
                      f"{100 * (r['test_mse'] / b['test_mse'] - 1):+.3f}%.")


if __name__ == "__main__":
    main()
