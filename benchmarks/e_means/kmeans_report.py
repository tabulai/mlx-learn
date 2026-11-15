"""Audit and report fresh four-method clustering comparisons without GPU work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import comparison_report as common  # noqa: E402
import kmeans_compare as benchmark  # noqa: E402

MODES = benchmark.MODES
NAMES = {"metal-kmeans": "Metal KMeans", "sklearn-kmeans": "scikit-learn KMeans",
         "metal-emeans": "Metal e-means", "faiss-emeans": "Faiss e-means"}
GLOBAL_FIELDS = ("source_sha256", "reference_source_sha256", "sklearn_source_sha256",
                 "versions", "hardware", "thread_environment", "effective_omp_threads",
                 "timing_scope", "memory_scope", "numerical_notes")
PAIR_FIELDS = ("n", "d", "k", "dataset", "initial_centroids_sha256", "train_score_sample_sha256", "train_score_rows")
IGNORE_CONFIG = {"mode", "output", "worker_output"}
CSV_FIELDS = ("dataset", "mode", "seed", "n", "d", "k", "epoch_budget", "actual_iterations",
              "seconds", "warmup_seconds", "timed_calls", "test_mse", "train_sample_mse", "final_inertia",
              "label_dtype", "label_sample_agreement", "label_sample_mse", "peak_physical_footprint_bytes",
              "peak_rss_bytes", "baseline_physical_footprint_bytes", "baseline_rss_bytes",
              "physical_footprint_peak_increase_bytes", "rss_peak_increase_bytes", "maximum_sample_gap_seconds",
              "mlx_peak_active_bytes", "mlx_cached_after_fit_bytes", "preparation_lifetime_peak_footprint_bytes",
              "through_fit_lifetime_peak_footprint_bytes", "evidence_path", "evidence_sha256")
COMPARISON_FIELDS = ("dataset", "sklearn_over_metal_kmeans", "metal_kmeans_over_metal_emeans",
                     "faiss_over_metal_emeans", "metal_kmeans_vs_sklearn_mse_change_pct",
                     "metal_emeans_vs_metal_kmeans_mse_change_pct", "metal_emeans_vs_faiss_mse_change_pct")


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def hash_mapping(value):
    return isinstance(value, dict) and bool(value) and all(digest(v) for v in value.values())


def load_rows(folder):
    rows, rejected = [], []
    for path in sorted(Path(folder).glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            rejected.append({"path": str(path.resolve()), "reason": f"unreadable JSON: {error}"})
            continue
        if isinstance(row, dict) and row.get("kind") == benchmark.KIND:
            row["_path"] = path.resolve()
            rows.append(row)
    return rows, rejected


def memory_problem(row):
    for index, memory in enumerate(row.get("memory_samples", [])):
        issue = common.memory_problem({"config": row["config"], "memory": memory})
        if issue:
            return f"memory sample {index}: {issue}"
    samples = row.get("mlx_memory_samples")
    if not isinstance(samples, list) or len(samples) != len(row["train_seconds_samples"]):
        return "allocator sample count differs from timed fits"
    if row["config"]["mode"].startswith("metal-"):
        for sample in samples:
            if not isinstance(sample, dict) or any(not common.nonnegative(sample.get(key)) for key in
                                                   ("peak_active_bytes", "active_after_fit_bytes", "cached_after_fit_bytes")):
                return "invalid MLX allocator sample"
            if sample["peak_active_bytes"] < sample["active_after_fit_bytes"]:
                return "MLX peak is below active allocation"
    elif any(sample is not None for sample in samples):
        return "CPU method has MLX allocator samples"
    return None


def row_problem(row, protocol, protocol_hash):
    try:
        config = row["config"]
        benchmark.validate_result(row, config, row.get("memory_samples", []))
        entry = next((e for e in protocol["configurations"] if e["dataset"] == config["dataset"]), None)
        if entry is None or config["protocol_config"] != entry:
            return "dataset entry differs from frozen protocol"
        fixed = {"seed": 10, "epochs": entry["epochs"], "precision": "float32", "n_init": 1,
                 "tol": 0.0, "final_labels": True, "threads": 8, "train_score_rows": 10000,
                 "memory": True, "memory_interval_seconds": .01, "rows": None, "k_override": None,
                 "diagnostic_subset": False}
        if any(config.get(key) != value for key, value in fixed.items()):
            return "record is outside the full fixed comparison protocol"
        if any(row.get(key) != entry[key] for key in ("n", "d", "k")):
            return "record is not the full dataset shape"
        if row.get("quality_fit_index") != 0 or not common.positive(row.get("warmup_seconds")):
            return "missing full warmup or first-fit quality"
        for key in ("source_sha256", "reference_source_sha256", "sklearn_source_sha256"):
            if not hash_mapping(row.get(key)):
                return f"missing valid {key}"
        if row["source_sha256"].get("benchmarks/e_means/kmeans_protocol.json") != protocol_hash:
            return "recorded protocol hash differs from report contract"
        for key in (*GLOBAL_FIELDS, *PAIR_FIELDS):
            if key not in row or row[key] is None:
                return f"missing {key}"
        for key in ("initial_centroids_sha256", "final_centroids_sha256", "final_labels_sha256", "train_score_sample_sha256"):
            if not digest(row.get(key)):
                return f"invalid {key}"
        if row.get("train_score_rows") != 10000:
            return "quality sample size differs from protocol"
        if row.get("final_labels", {}).get("semantics") != "nearest final centroids; no update after final assignment":
            return "labels are not final-center assignments"
        agreement = row.get("label_sample_agreement")
        if not isinstance(agreement, (int, float)) or not math.isfinite(agreement) or not 0 <= agreement <= 1:
            return "invalid final-label agreement diagnostic"
        metal = config["mode"].startswith("metal-")
        if row.get("full_host_input_released") is not metal:
            return "input ownership policy differs from mode"
        if metal and not isinstance(row.get("metal_device"), dict) or not metal and row.get("metal_device") is not None:
            return "device metadata differs from mode"
        if config["mode"].endswith("emeans"):
            recipe = row["recipe_resolved"]
            if recipe["seed"] != 10 or row["actual_iterations"] != config["epochs"]:
                return "e-means seed or observed epoch count differs"
            if any(key not in recipe for key in common.RECIPE_FIELDS):
                return "incomplete e-means statistical recipe"
            if config["mode"] == "metal-emeans" and (recipe.get("precision") != "float32"
                                                       or recipe.get("finalization") != "last_epoch"):
                return "Metal e-means precision or finalizer differs"
        elif row.get("recipe_resolved") is not None:
            return "Lloyd record unexpectedly declares an e-means recipe"
        return memory_problem(row)
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return f"invalid measurement: {error}"


def archive_evidence(metadata_path):
    """Verify the ZIP itself and every repo-relative archived source member."""
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    archive = metadata_path.with_suffix(".zip")
    actual = common.sha256(archive)
    if actual != metadata["archive_sha256"]:
        raise ValueError("source archive digest mismatch")
    if not hash_mapping(metadata.get("source_sha256")):
        raise ValueError("source archive metadata lacks source hashes")
    with zipfile.ZipFile(archive) as contents:
        for name, expected in metadata["source_sha256"].items():
            if hashlib.sha256(contents.read(name)).hexdigest() != expected:
                raise ValueError(f"archived source member differs: {name}")
    return metadata, {"metadata_path": str(metadata_path.resolve()), "metadata_sha256": common.sha256(metadata_path),
                      "archive_path": str(archive.resolve()), "archive_sha256": actual, "verified": True}


def validate(rows, protocol, protocol_hash, archive_metadata=None):
    candidates, rejected = defaultdict(list), []
    for row in rows:
        issue = row_problem(row, protocol, protocol_hash)
        if not issue and archive_metadata is not None:
            for field in ("source_sha256", "reference_source_sha256", "sklearn_source_sha256"):
                if field in archive_metadata and row.get(field) != archive_metadata[field]:
                    issue = f"{field} differs from frozen source archive"
                    break
        if issue:
            rejected.append({"path": str(row.get("_path", "")), "reason": issue})
        else:
            candidates[(row["config"]["dataset"], row["config"]["mode"])].append(row)
    accepted = {}
    for identity, members in candidates.items():
        if len(members) == 1:
            accepted[identity] = members[0]
        else:
            rejected.extend({"path": str(row.get("_path", "")), "reason": "duplicate dataset/seed/mode"} for row in members)
    campaigns = {common.stable({key: row[key] for key in GLOBAL_FIELDS}) for row in accepted.values()}
    devices = {common.stable(row["metal_device"]) for row in accepted.values() if row["config"]["mode"].startswith("metal-")}
    if len(campaigns) > 1 or len(devices) > 1:
        rejected.extend({"path": str(row.get("_path", "")), "reason": "mixed source/environment/device campaigns"}
                        for row in accepted.values())
        accepted = {}
    groups = []
    for entry in protocol["configurations"]:
        group = {mode: accepted[(entry["dataset"], mode)] for mode in MODES if (entry["dataset"], mode) in accepted}
        if not group:
            continue
        first = next(iter(group.values()))
        config = {k: v for k, v in first["config"].items() if k not in IGNORE_CONFIG}
        issue = any(any(row[key] != first[key] for key in PAIR_FIELDS)
                    or {k: v for k, v in row["config"].items() if k not in IGNORE_CONFIG} != config
                    for row in group.values())
        if all(mode in group for mode in ("metal-emeans", "faiss-emeans")):
            a, b = (group[mode]["recipe_resolved"] for mode in ("metal-emeans", "faiss-emeans"))
            issue |= any(not common.semantic_equal(a[key], b[key]) for key in common.RECIPE_FIELDS)
        if issue:
            for mode, row in group.items():
                rejected.append({"path": str(row.get("_path", "")), "reason": "data/init/config/e-means recipe differs across methods"})
                del accepted[(entry["dataset"], mode)]
        else:
            groups.append(group)
    expected = {(entry["dataset"], mode) for entry in protocol["configurations"] for mode in MODES}
    missing = sorted(expected - accepted.keys())
    return groups, {"complete": not missing and not rejected and len(rows) == len(expected),
                    "expected_records": len(expected), "observed_records": len(rows), "accepted_records": len(accepted),
                    "missing": [{"dataset": name, "seed": 10, "mode": mode} for name, mode in missing], "rejected": rejected}


def summarize(groups):
    rows, comparisons = [], []
    for group in groups:
        first = next(iter(group.values()))
        comparison = {"dataset": first["config"]["dataset"]}
        for left, right, key in (("sklearn-kmeans", "metal-kmeans", "sklearn_over_metal_kmeans"),
                                 ("metal-kmeans", "metal-emeans", "metal_kmeans_over_metal_emeans"),
                                 ("faiss-emeans", "metal-emeans", "faiss_over_metal_emeans")):
            if left in group and right in group:
                comparison[key] = group[left]["train_seconds"] / group[right]["train_seconds"]
        for numerator, denominator, key in (
            ("metal-kmeans", "sklearn-kmeans", "metal_kmeans_vs_sklearn_mse_change_pct"),
            ("metal-emeans", "metal-kmeans", "metal_emeans_vs_metal_kmeans_mse_change_pct"),
            ("metal-emeans", "faiss-emeans", "metal_emeans_vs_faiss_mse_change_pct"),
        ):
            if numerator in group and denominator in group and group[denominator]["test_mse"]:
                comparison[key] = 100 * (group[numerator]["test_mse"] / group[denominator]["test_mse"] - 1)
        comparisons.append(comparison)
        for mode in MODES:
            if mode not in group:
                continue
            raw = group[mode]
            memories = raw["memory_samples"]
            row = {"dataset": raw["config"]["dataset"], "mode": mode, "seed": 10,
                   **{key: raw[key] for key in ("n", "d", "k", "actual_iterations", "warmup_seconds", "test_mse",
                                               "train_sample_mse", "final_inertia", "label_sample_agreement", "label_sample_mse")},
                   "epoch_budget": raw["config"]["epochs"], "seconds": raw["train_seconds"],
                   "timed_calls": len(raw["train_seconds_samples"]), "label_dtype": raw["final_labels"]["dtype"]}
            for field in ("peak_physical_footprint_bytes", "peak_rss_bytes", "physical_footprint_peak_increase_bytes",
                          "rss_peak_increase_bytes", "maximum_sample_gap_seconds"):
                row[field] = max(m[field] for m in memories)
            for field in ("physical_footprint_bytes", "rss_bytes"):
                row["baseline_" + field] = max(m["baseline"][field] for m in memories)
            row["mlx_peak_active_bytes"] = max(m["peak_active_bytes"] for m in raw["mlx_memory_samples"]) if mode.startswith("metal-") else None
            row["mlx_cached_after_fit_bytes"] = max(m["cached_after_fit_bytes"] for m in raw["mlx_memory_samples"]) if mode.startswith("metal-") else None
            row["preparation_lifetime_peak_footprint_bytes"] = raw.get("preparation_process_memory", {}).get("lifetime_peak_physical_footprint_bytes")
            lifetime = [m.get("end", {}).get("lifetime_peak_physical_footprint_bytes") for m in memories]
            row["through_fit_lifetime_peak_footprint_bytes"] = max(lifetime) if all(common.nonnegative(v) for v in lifetime) else None
            path = raw.get("_path")
            row.update(evidence_path=str(path or ""), evidence_sha256=common.sha256(path) if path else None)
            rows.append(row)
    return rows, comparisons


def headline(comparisons):
    def span(key, *, percent=False):
        values = [row[key] for row in comparisons if row.get(key) is not None]
        if len(values) != len(comparisons) or not values:
            return None
        if percent:
            return f"{min(values):+.5g}% to {max(values):+.5g}%"
        return f"{min(values):.3g}–{max(values):.3g}×"

    lloyd = span("sklearn_over_metal_kmeans")
    emeans = span("metal_kmeans_over_metal_emeans")
    sentences = []
    if lloyd:
        sentences.append(f"Across the five datasets, Metal KMeans ran at {lloyd} the speed of scikit-learn KMeans.")
    if emeans:
        sentences.append(f"The Metal KMeans/Metal e-means runtime ratio ranged from {emeans}; values above one favor e-means.")
    for key, label in (("metal_kmeans_vs_sklearn_mse_change_pct", "Metal KMeans versus scikit-learn"),
                       ("metal_emeans_vs_metal_kmeans_mse_change_pct", "Metal e-means versus Metal KMeans")):
        values = span(key, percent=True)
        if values:
            sentences.append(f"Observed test-MSE changes for {label} ranged from {values}.")
    return " ".join(sentences)


def figure_markdown(report, figure_folder=None):
    folder = Path(figure_folder) if figure_folder is not None else HERE / "kmeans_figures"
    lines = []
    png, pdf = folder / "comparison.png", folder / "comparison.pdf"
    if png.is_file():
        lines += ["", "!" + common.link(png, "Runtime, fit memory and test MSE for all four methods", report)]
    if pdf.is_file():
        lines += ["", common.link(pdf, "Download the comparison figure as PDF", report)]
    return lines


def markdown(rows, comparisons, validation, report, folder, protocol_path):
    def link(path, title):
        return common.link(path, title, report)

    def number(value, digits=3):
        return common.fmt(value, digits) if value is not None else "—"

    lines = ["# KMeans and e-means on five 10M datasets", "",
             f"**{'Complete' if validation['complete'] else 'INCOMPLETE'}: {validation['accepted_records']}/"
             f"{validation['expected_records']} accepted measurements.** Each configuration uses 10,000,000 rows, "
             "65,536 centers, FP32 input and seed 10. These are single-seed measurements, with no error bars or statistical-significance claim.", "",
             "All four methods start from identical explicit centers and use the same integer pass budget. "
             "The timed result includes final nearest-center labels and training inertia. Lower test MSE is better.", ""]
    if validation["complete"]:
        lines += [headline(comparisons), ""]
    lines += ["| Dataset | Method | Budget / actual passes | Time (s) | Fit footprint (GiB) | Test MSE |",
              "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {common.label(row['dataset'])} | {NAMES[row['mode']]} | {row['epoch_budget']} / {row['actual_iterations']} | "
                     f"{number(row['seconds'])} | {row['peak_physical_footprint_bytes']/2**30:.3f} | {number(row['test_mse'], 8)} |")
    lines += ["", "| Dataset | sklearn / Metal KMeans | Metal KMeans / Metal e-means | Faiss / Metal e-means | "
              "Metal KMeans MSE vs sklearn | Metal e-means MSE vs Metal KMeans | Metal e-means MSE vs Faiss |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        values = [(number(row[key], 3) + "×" if "over" in key else f"{row[key]:+.5g}%") if row.get(key) is not None else "—"
                  for key in COMPARISON_FIELDS[1:]]
        lines.append(f"| {common.label(row['dataset'])} | " + " | ".join(values) + " |")
    lines += ["", "Ratios above one mean the denominator is faster. Positive MSE change means the named numerator has higher error. "
              "The Lloyd/e-means ratio compares different algorithms at a shared pass budget, not matched clustering quality."]
    lines += figure_markdown(report)
    lines += ["", "## Memory and training-quality diagnostics", "",
              "| Dataset / method | Fit RSS (GiB) | MLX peak active (GiB) | Lifetime through preparation / timed fit (GiB) | Train-sample MSE | Native labels |",
              "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lifetime = " / ".join(number(row[key]/2**30 if row[key] is not None else None) for key in
                              ("preparation_lifetime_peak_footprint_bytes", "through_fit_lifetime_peak_footprint_bytes"))
        active = row["mlx_peak_active_bytes"]
        lines.append(f"| {common.label(row['dataset'])} / {NAMES[row['mode']]} | {row['peak_rss_bytes']/2**30:.3f} | "
                     f"{number(active/2**30 if active is not None else None)} | {lifetime} | {number(row['train_sample_mse'], 8)} | {row['label_dtype']} |")
    lines += ["", "Process footprint and RSS are sampled every 10 ms during fit handshakes, including resident input and retained allocator cache. "
              "Short peaks may be missed. MLX allocation is separate and is never added to process memory. Lifetime peaks include earlier preparation "
              "and warmup; they are not fit-window peaks. CSV retains baseline, peak increase, maximum sample gaps and allocator cache values. "
              "Across repeated calls, each memory column is a maximum and need not describe the same call.", "",
              "## Protocol and limits", "",
              "DEEP, SPACEV, SimSearchNet++ and DINO use two full passes; SIFT uses three. Lloyd can stop earlier on unchanged labels or zero center shift "
              "even with tol=0, so actual iterations are reported. E-means uses the pinned Faiss quality batch law and last-epoch moving-assignment means. "
              "It is then reassigned once against its final centers. Earlier fractional-budget or online-label-only measurements are historical and are not reused.", "",
              "Each method runs in a fresh process, with one identical full untimed warmup. Data loading, explicit initialization, Metal validation/upload/preparation, "
              "cache clearing, host export and shared quality scoring are excluded. Timed work includes a fresh model, training, final labels/inertia and GPU synchronization "
              "where applicable. CPU validation and sklearn's centering/sample-weight allocation remain inside their public fit calls. "
              "A first timed fit under two seconds triggers two more; reported time is the median, while quality is from the first fit. OS file and shader caches are not flushed.", "",
              "scikit-learn uses unmodified Lloyd KMeans, n_init=1, tol=0 and copy_x=False. It subtracts and restores the input mean in place; exact cached data is reloaded "
              "and checked outside timing before each timed call. Its empty-cluster relocation can allocate full n×d temporary matrices. DINO's FP32 input alone occupies "
              "38.147 GiB, so preparation and any relocation transients matter beyond fit-only memory summaries. Metal releases its complete NumPy input reference after "
              "device preparation. Neither method is an out-of-core implementation.", "",
              "Eight OpenMP threads are requested before imports; sklearn's observed thread count and loaded thread pools are recorded. Accelerate threading is unrestricted. "
              "scikit-learn's BLAS-limit wrapper does not establish a measured single-thread Accelerate configuration when threadpoolctl cannot control that library.", "",
              "The four implementations differ in reduction order, empty-cluster treatment, centroid-update arithmetic and label dtype. Lloyd holds centers fixed for each pass; "
              "e-means moves them between mini-batches. Identical initialization does not imply identical centers or bitwise trajectories. Native full-training inertia reductions "
              "also differ; quality comparisons use shared FP32 Faiss assignment followed by direct FP64 residuals on the test set and a fixed 10,000-row training sample. "
              "The training-sample metric is not full-training MSE. Label agreement and selected-label residual diagnostics are preserved in CSV/raw records.", "",
              "Dataset identity covers complete official test arrays and 1,000 evenly spaced training fingerprints, not a full training-file digest. DINO and SIFT retain "
              "the pinned /4096 and /512 scaling. Published combined validation/test duplicate overlap with training is four vectors for DEEP and 45 for SPACEV; "
              "these are not test-only counts and the holdouts are not claimed to be duplicate-disjoint. This campaign does not reproduce H100 timing or per-seed crossing budgets.", "",
              "## Evidence", "",
              f"{link(protocol_path, 'Frozen protocol')}; {link(folder/'validation.json', 'validation and raw hashes')}; "
              f"{link(folder/'summary.csv', 'per-method CSV')}; {link(folder/'comparisons.csv', 'paired-comparison CSV')}; "
              f"{link(REPO/'docs/k-means.md', 'KMeans API documentation')}.", ""]
    archive = validation.get("archive") or {}
    if archive.get("verified"):
        lines += [f"{link(archive['archive_path'], 'Frozen source archive')} SHA-256: `{archive['archive_sha256']}`.", ""]
    if validation.get("provenance"):
        lines += ["Recorded machine and software:", "", "```json", json.dumps(validation["provenance"], indent=2, sort_keys=True), "```", ""]
    lines += ["| Dataset / method | Raw evidence |", "|---|---|"]
    for row in rows:
        lines.append(f"| {common.label(row['dataset'])} / {NAMES[row['mode']]} | {link(row['evidence_path'], 'JSON')} |")
    if not validation["complete"]:
        lines += ["", "**Incomplete campaign:** missing or rejected records are listed in the validation manifest. Partial tables do not establish full five-dataset coverage."]
    return "\n".join(lines) + "\n"


def generate(folder, output, protocol_path, archive_path, require_complete=False):
    folder, output, protocol_path = Path(folder).resolve(), Path(output).resolve(), Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    rows, parse_errors = load_rows(folder)
    archive_info, archive_error = None, None
    try:
        metadata, archive_info = archive_evidence(archive_path)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        metadata, archive_error = None, str(error)
    groups, validation = validate(rows, protocol, common.sha256(protocol_path), metadata)
    validation["rejected"].extend(parse_errors)
    if archive_error:
        validation["rejected"].append({"path": str(archive_path), "reason": archive_error})
        groups = []
        validation["accepted_records"] = 0
        validation["missing"] = [{"dataset": entry["dataset"], "seed": 10, "mode": mode}
                                 for entry in protocol["configurations"] for mode in MODES]
    validation["complete"] &= not validation["rejected"] and len(protocol["configurations"]) == 5
    validation.update(protocol_path=str(protocol_path), protocol_sha256=common.sha256(protocol_path), archive=archive_info,
                      report_generator_sha256=common.sha256(__file__), harness_validator_sha256=common.sha256(HERE/"kmeans_compare.py"),
                      common_report_sha256=common.sha256(HERE/"comparison_report.py"),
                      raw_evidence=[{"path": str(row["_path"]), "sha256": common.sha256(row["_path"])} for row in rows])
    if groups:
        first = next(iter(groups[0].values()))
        validation["provenance"] = {key: first[key] for key in ("hardware", "versions", "thread_environment")}
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    if require_complete and not validation["complete"]:
        raise ValueError("Incomplete or incompatible evidence; validation saved, report and CSV not written")
    summary, comparisons = summarize(groups)
    for name, fields, values in (("summary.csv", CSV_FIELDS, summary), ("comparisons.csv", COMPARISON_FIELDS, comparisons)):
        with (folder/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown(summary, comparisons, validation, output, folder, protocol_path))
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--results", type=Path, default=HERE/"results/kmeans_comparison")
    parser.add_argument("--output", "--report", type=Path, default=REPO/"docs/k-means-10m-comparison-2026-11-15.md")
    parser.add_argument("--protocol", type=Path, default=HERE/"kmeans_protocol.json")
    parser.add_argument("--archive-metadata", type=Path, default=HERE/"reference/kmeans-comparison-source.json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        result = generate(args.input, args.output, args.protocol, args.archive_metadata, args.require_complete)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({key: result[key] for key in ("complete", "accepted_records", "expected_records")}))


if __name__ == "__main__":
    main()
