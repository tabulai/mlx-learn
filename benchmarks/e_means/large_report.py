"""Validate and report the five 10M resident comparisons using only stdlib.

Primary ratios compare FP32 Metal with unmodified FP32 NumpyEmeans. BF16 is
an explicitly separate precision experiment. No runtime or dataset is loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import comparison_report as common  # noqa: E402

KIND = "large_faiss_resident"
MODES = ("metal-fp32", "faiss-fp32", "metal-bf16")
REVISION = "ebab8f784c1d177129602269d5ad966c525fbdcb"
RUNTIME_ARCHIVE_SHA256 = "c24dfaf84b3a75c1d741fb1708bb37addda013dc9301d9d6b3166d5f52077413"
SHARED_ARCHIVE_SHA256 = "a61613a6c3dabb59690545d9a4b43796c6b104b8e817f502558ee1279287cf39"
RECIPE_FIELDS = tuple(key for key in common.RECIPE_FIELDS if key != "seed")
IDENTITY_FIELDS = ("train_sample_sha256", "test_sha256", "train_rows_used", "test_rows_used", "dimension")
SOURCE_FIELDS = (
    "benchmarks/e_means/large_compare.py", "benchmarks/e_means/memory.py",
    "benchmarks/e_means/run.py", "benchmarks/e_means/large_protocol.json",
    "src/mlxlearn/cluster/_e_means.py", "src/mlxlearn/cluster/_recipe.py",
    "src/mlxlearn/_kernels/e_means.py", "src/mlxlearn/_kernels/e_means_bf16.py",
)
REFERENCE_FIELDS = ("contrib/e_means/base.py", "contrib/e_means/e_means.py", "benchs/e_means/paper_data.py")
GLOBAL_FIELDS = (
    "source_sha256", "reference_source_sha256", "versions", "hardware", "faiss_reference_revision",
    "faiss_compile_options", "effective_omp_threads", "thread_environment", "timing_scope", "scoring",
    "memory_scope", "memory_policy", "backend_timing_notes", "reference_backend", "numerical_differences",
)
PAIR_FIELDS = ("n", "d", "k", "dataset", "initial_centroids_sha256", "train_score_sample_sha256", "train_score_rows")
CONFIG_IGNORED = {"mode", "precision", "output", "worker_output"}
MEMORY_FIELDS = (
    "peak_physical_footprint_bytes", "peak_rss_bytes", "physical_footprint_peak_increase_bytes",
    "rss_peak_increase_bytes", "maximum_sample_gap_seconds",
)
CSV_FIELDS = [
    "dataset", "n", "d", "k", "epochs", "batch_size", "alpha0", "seed",
    "metal_fp32_seconds", "faiss_fp32_seconds", "cpu_over_metal_fp32",
    "metal_fp32_test_mse", "faiss_fp32_test_mse", "fp32_test_mse_change_pct",
    "metal_fp32_train_sample_mse", "faiss_fp32_train_sample_mse",
    "metal_bf16_seconds", "metal_fp32_over_metal_bf16", "metal_bf16_test_mse",
    "metal_bf16_train_sample_mse", "bf16_vs_fp32_test_mse_change_pct",
]
for _mode in MODES:
    _prefix = _mode.replace("-", "_")
    CSV_FIELDS.extend(f"{_prefix}_{field}" for field in (*MEMORY_FIELDS, "baseline_physical_footprint_bytes",
                                                       "baseline_rss_bytes", "mlx_peak_active_bytes",
                                                       "lifetime_peak_footprint_through_fit_bytes",
                                                       "lifetime_peak_footprint_through_preparation_bytes",
                                                       "timed_calls", "evidence_path", "evidence_sha256"))


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def source_map(value, required):
    return isinstance(value, dict) and set(required) <= value.keys() and all(digest(v) for v in value.values())


def verify_source_archive(metadata_path, expected_archive_sha256=None):
    """Verify the ZIP digest, exact member set and every member's recorded bytes."""
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    archive = metadata_path.with_suffix(".zip")
    recorded_digest = metadata.get("archive_sha256")
    if (not digest(recorded_digest)
        or expected_archive_sha256 is not None and recorded_digest != expected_archive_sha256):
        raise ValueError(f"Frozen archive metadata digest differs from the published snapshot: {metadata_path.name}")
    if common.sha256(archive) != recorded_digest:
        raise ValueError(f"Frozen archive does not match its recorded digest: {archive.name}")
    sources = metadata.get("source_sha256")
    if not source_map(sources, ()) or not sources:
        raise ValueError(f"Frozen archive has an invalid source hash map: {metadata_path.name}")
    try:
        with zipfile.ZipFile(archive) as stream:
            names = stream.namelist()
            if len(names) != len(set(names)):
                raise ValueError(f"Frozen archive contains duplicate member names: {archive.name}")
            if set(names) != sources.keys():
                raise ValueError(f"Frozen archive member names differ from the source hash map: {archive.name}")
            for name, expected in sources.items():
                if hashlib.sha256(stream.read(name)).hexdigest() != expected:
                    raise ValueError(f"Frozen archive member bytes differ from their source hash: {name}")
    except zipfile.BadZipFile as error:
        raise ValueError(f"Invalid frozen source archive: {archive.name}") from error
    return {**metadata, "archive": str(archive.resolve()), "metadata": str(metadata_path.resolve()),
            "metadata_sha256": common.sha256(metadata_path), "verified_members": len(sources)}


def archive_record_problem(row, runtime, shared):
    if any(row.get("source_sha256", {}).get(key) != value for key, value in runtime["source_sha256"].items()):
        return "runtime differs from frozen Metal v3 snapshot"
    if row.get("source_sha256") != shared["source_sha256"]:
        return "shared source map differs from frozen comparison snapshot"
    if (row.get("reference_source_sha256") != shared["reference_source_sha256"]
        or row.get("faiss_reference_revision") != shared["faiss_reference_revision"]):
        return "reference source map/revision differs from frozen comparison snapshot"
    return None


def equal_mapping(left, right):
    """Use the original report's numeric tolerance on each canonical field."""
    return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
            and all(common.semantic_equal(left[k], right[k]) for k in left))


def evidence_rows(folder):
    rows = []
    for path in sorted(Path(folder).rglob("*.json")):
        item = json.loads(path.read_text())
        if isinstance(item, dict) and item.get("kind") == KIND:
            item["_path"] = path.resolve()
            rows.append(item)
    return rows


def memory_problem(row):
    memories = row.get("memory_samples")
    samples = row.get("train_seconds_samples", [])
    if not isinstance(memories, list) or len(memories) != len(samples) or not memories:
        return "memory_samples count differs from timed fits"
    for index, memory in enumerate(memories):
        if not isinstance(memory, dict):
            return f"memory_samples[{index}] missing"
        issue = common.memory_problem({"config": row["config"], "memory": memory})
        if issue:
            return f"memory_samples[{index}]: {issue}"
    allocator = row.get("mlx_memory_samples")
    if not isinstance(allocator, list) or len(allocator) != len(samples):
        return "MLX allocator sample count differs from timed fits"
    if row["config"]["mode"] == "faiss-fp32":
        if any(sample is not None for sample in allocator):
            return "CPU record unexpectedly has MLX allocator samples"
    else:
        for sample in allocator:
            if not isinstance(sample, dict) or any(not common.nonnegative(sample.get(field)) for field in
                                                   ("peak_active_bytes", "active_after_fit_bytes", "cached_after_fit_bytes")):
                return "invalid MLX allocator sample"
            if sample["peak_active_bytes"] < sample["active_after_fit_bytes"]:
                return "MLX peak below active allocation"
    return None


def row_problem(row, protocol, protocol_sha256=None):
    if row.get("status") != "completed" or row.get("kind") != KIND:
        return "record is not a completed large comparison"
    c = row.get("config")
    if not isinstance(c, dict):
        return "missing config"
    entries = {entry["dataset"]: entry for entry in protocol["configurations"]}
    entry = entries.get(c.get("dataset"))
    mode = c.get("mode")
    if entry is None or mode not in MODES:
        return "unexpected dataset or mode"
    fixed = {"seed": 10, "recipe": "faiss", "batch_law": "quality", "finalization": "last_epoch",
             "capture_assignments": True, "warmup": True, "threads": 8,
             "train_score_rows": 10000, "memory": True, "memory_interval_seconds": .01,
             "rows": None, "k_override": None, "epochs_override": None, "diagnostic_subset": False,
             "precision": "bfloat16" if mode == "metal-bf16" else "float32", "epochs": entry["epochs"]}
    for key, expected in fixed.items():
        if key not in c or not common.semantic_equal(c[key], expected):
            return f"outside fixed protocol: config.{key}"
    if c.get("protocol_config") != entry:
        return "config.protocol_config differs from frozen contract"
    if any(row.get(key) != entry[key] for key in ("n", "d", "k")):
        return "n/d/k differs from full 10M workload"
    data = row.get("dataset")
    if not isinstance(data, dict) or any(data.get(k) != entry["dataset_identity"][k] for k in IDENTITY_FIELDS):
        return "dataset fingerprint or shape differs from contract"
    if not source_map(row.get("source_sha256"), SOURCE_FIELDS):
        return "missing or invalid runtime source hashes"
    if not source_map(row.get("reference_source_sha256"), REFERENCE_FIELDS):
        return "missing or invalid reference source hashes"
    if protocol_sha256 and row["source_sha256"]["benchmarks/e_means/large_protocol.json"] != protocol_sha256:
        return "hashed protocol differs from report contract"
    for actual, expected in (("source_sha256", "expected_source_sha256"),
                             ("reference_source_sha256", "expected_reference_source_sha256"),
                             ("versions", "expected_versions"), ("hardware", "expected_host")):
        if not row.get(actual) or row[actual] != c.get(expected):
            return f"actual {actual} differs from expected provenance"
    if row.get("faiss_reference_revision") != REVISION or protocol.get("reference_revision") != REVISION:
        return "reference revision differs from pinned Faiss"
    for field in GLOBAL_FIELDS:
        if field not in row or row[field] is None:
            return f"missing {field}"
    if row.get("effective_omp_threads") != 8 or row.get("thread_environment", {}).get("OMP_NUM_THREADS") != "8":
        return "OpenMP threads differ from protocol"
    if row["thread_environment"].get("VECLIB_MAXIMUM_THREADS") is not None:
        return "Accelerate threads were restricted"
    if row.get("full_host_input_released") is not (mode != "faiss-fp32"):
        return "input ownership policy differs from protocol"
    if (mode == "faiss-fp32" and row.get("metal_device") is not None
        or mode != "faiss-fp32" and not isinstance(row.get("metal_device"), dict)):
        return "Metal device provenance differs from mode"
    samples = row.get("train_seconds_samples")
    if not isinstance(samples, list) or not samples or not all(common.positive(t) for t in samples):
        return "invalid timed samples"
    if len(samples) != (3 if samples[0] < 2 else 1):
        return "adaptive timed-call count differs from protocol"
    if not common.semantic_equal(row.get("train_seconds"), median(samples)):
        return "reported time is not median of timed samples"
    if row.get("quality_fit_index") != 0 or any(not common.nonnegative(row.get(key)) for key in ("test_mse", "train_sample_mse")):
        return "quality is invalid or not from first timed fit"
    if row.get("train_score_rows") != 10000:
        return "training quality sample size differs from protocol"
    for key in ("initial_centroids_sha256", "final_centroids_sha256", "train_score_sample_sha256"):
        if not digest(row.get(key)):
            return f"invalid {key}"
    recipe = row.get("recipe_resolved")
    if not isinstance(recipe, dict) or any(key not in recipe for key in (*RECIPE_FIELDS, "seed")):
        return "incomplete resolved statistical recipe"
    if not equal_mapping(row.get("algorithm_comparison"), {k: recipe[k] for k in RECIPE_FIELDS}):
        return "canonical statistical recipe differs from resolved recipe"
    expected_recipe = {**entry["recipe_expected"], "batch_law": "quality", "nepochs": entry["epochs"], "seed": 10}
    if any(not common.semantic_equal(recipe.get(key), expected) for key, expected in expected_recipe.items()):
        return "resolved statistical recipe differs from frozen contract"
    if mode != "faiss-fp32" and (recipe.get("precision") != c["precision"] or recipe.get("finalization") != "last_epoch"
                                or recipe.get("assignment") != "auto"):
        return "Metal resolved precision/finalization/assignment differs from requested protocol"
    labels = row.get("captured_assignments", {})
    if labels.get("shape") != [entry["n"]] or "int64" not in str(labels.get("dtype", "")):
        return "missing full online int64 assignments"
    if labels.get("semantics") != "online final-pass labels; no extra reassignment" or not row.get("iteration_history"):
        return "missing online assignment semantics or iteration history"
    history = row["iteration_history"]
    if len(history) != len(recipe["epoch_rows"]) or row.get("n_steps") != recipe["total_steps"]:
        return "observed pass/update count differs from recipe"
    if mode != "faiss-fp32" and ([stat.get("rows") for stat in history] != recipe["epoch_rows"]
                                or history[-1].get("steps") != recipe["total_steps"]):
        return "observed Metal history differs from recipe"
    return memory_problem(row)


def pair_problem(left, right):
    for key in (*GLOBAL_FIELDS, *PAIR_FIELDS):
        if left[key] != right[key]:
            return f"{key} differs across modes"
    a = {k: v for k, v in left["config"].items() if k not in CONFIG_IGNORED}
    b = {k: v for k, v in right["config"].items() if k not in CONFIG_IGNORED}
    if a != b:
        return "common configuration differs across modes"
    if not equal_mapping(left["algorithm_comparison"], right["algorithm_comparison"]):
        return "statistical recipe differs across modes"
    if (left["config"]["mode"] != "faiss-fp32" and right["config"]["mode"] != "faiss-fp32"
        and left["metal_device"] != right["metal_device"]):
        return "Metal device differs across precision modes"
    return None


def validate(rows, protocol, protocol_sha256=None):
    """Reject ambiguous records and mixed campaigns; never pick a preferred trial."""
    candidates, rejected = defaultdict(list), []
    for row in rows:
        issue = row_problem(row, protocol, protocol_sha256)
        if issue:
            rejected.append({"path": str(row.get("_path", "")), "reason": issue})
        else:
            candidates[(row["config"]["dataset"], row["config"]["seed"], row["config"]["mode"])].append(row)
    accepted = {}
    for key, members in candidates.items():
        if len(members) != 1:
            rejected.extend({"path": str(r.get("_path", "")), "reason": "duplicate dataset/seed/mode"} for r in members)
        else:
            accepted[key] = members[0]
    global_identities = {common.stable({key: row[key] for key in GLOBAL_FIELDS}) for row in accepted.values()}
    if len(global_identities) > 1:
        rejected.extend({"path": str(r.get("_path", "")), "reason": "mixed source/environment/protocol campaigns"}
                        for r in accepted.values())
        accepted = {}
    groups = []
    for entry in protocol["configurations"]:
        group = {mode: accepted[(entry["dataset"], 10, mode)] for mode in MODES
                 if (entry["dataset"], 10, mode) in accepted}
        issue = next((problem for row in list(group.values())[1:]
                      if (problem := pair_problem(next(iter(group.values())), row))), None)
        if issue:
            for mode, row in group.items():
                rejected.append({"path": str(row.get("_path", "")), "reason": issue})
                del accepted[(entry["dataset"], 10, mode)]
        elif group:
            groups.append(group)
    expected = {(e["dataset"], 10, mode) for e in protocol["configurations"] for mode in MODES}
    missing = sorted(expected - accepted.keys())
    validation = {"complete": not missing and not rejected and len(rows) == len(expected),
                  "expected_records": len(expected), "observed_records": len(rows), "accepted_records": len(accepted),
                  "missing": [{"dataset": d, "seed": seed, "mode": mode} for d, seed, mode in missing],
                  "rejected": rejected}
    return groups, validation


def memory_peak(row, field):
    return max(sample[field] for sample in row["memory_samples"])


def change_pct(value, reference):
    return 100 * (value / reference - 1) if reference else None


def summarize(groups):
    output = []
    for group in groups:
        if not all(mode in group for mode in ("metal-fp32", "faiss-fp32")):
            continue
        metal, faiss = group["metal-fp32"], group["faiss-fp32"]
        row = {"dataset": metal["config"]["dataset"], **{k: metal[k] for k in ("n", "d", "k")},
               "epochs": metal["config"]["epochs"], "seed": 10,
               "batch_size": metal["algorithm_comparison"]["batch_size"], "alpha0": metal["algorithm_comparison"]["alpha0"],
               "cpu_over_metal_fp32": faiss["train_seconds"] / metal["train_seconds"],
               "fp32_test_mse_change_pct": change_pct(metal["test_mse"], faiss["test_mse"])}
        for mode, measurement in group.items():
            prefix = mode.replace("-", "_")
            for suffix, key in (("seconds", "train_seconds"), ("test_mse", "test_mse"), ("train_sample_mse", "train_sample_mse")):
                row[f"{prefix}_{suffix}"] = measurement[key]
            for field in MEMORY_FIELDS:
                row[f"{prefix}_{field}"] = memory_peak(measurement, field)
            for field in ("physical_footprint_bytes", "rss_bytes"):
                row[f"{prefix}_baseline_{field}"] = max(m["baseline"][field] for m in measurement["memory_samples"])
            row[f"{prefix}_mlx_peak_active_bytes"] = (None if mode == "faiss-fp32" else
                                                       max(m["peak_active_bytes"] for m in measurement["mlx_memory_samples"]))
            lifetime = [m.get("end", {}).get("lifetime_peak_physical_footprint_bytes") for m in measurement["memory_samples"]]
            row[f"{prefix}_lifetime_peak_footprint_through_fit_bytes"] = (
                max(lifetime) if all(common.nonnegative(value) for value in lifetime) else None)
            preparation_lifetime = measurement.get("preparation_process_memory", {}).get("lifetime_peak_physical_footprint_bytes")
            row[f"{prefix}_lifetime_peak_footprint_through_preparation_bytes"] = (
                preparation_lifetime if common.nonnegative(preparation_lifetime) else None)
            row[f"{prefix}_timed_calls"] = len(measurement["train_seconds_samples"])
            path = measurement.get("_path")
            row[f"{prefix}_evidence_path"] = str(path or "")
            row[f"{prefix}_evidence_sha256"] = common.sha256(path) if path else None
        if "metal-bf16" in group:
            bf16 = group["metal-bf16"]
            row["metal_fp32_over_metal_bf16"] = metal["train_seconds"] / bf16["train_seconds"]
            row["bf16_vs_fp32_test_mse_change_pct"] = change_pct(bf16["test_mse"], metal["test_mse"])
        output.append(row)
    return output


def markdown(rows, validation, protocol, report, folder):
    fmt = common.fmt
    lines = ["# E-means on five 10M datasets: Metal and original Faiss CPU", "",
             f"Validation: **{'complete' if validation['complete'] else 'INCOMPLETE'}**, "
             f"{validation['accepted_records']}/{validation['expected_records']} accepted records. "
             "Each dataset has 10,000,000 training vectors, 65,536 centroids, and seed 10. "
             "This is a single-seed comparison; there are no error bars or claims of statistical significance.", "",
             "## FP32 comparison", "",
             "Both primary implementations use FP32 input, centroid state and assignment operands. "
             "CPU/Metal is original Faiss NumpyEmeans time divided by Metal FP32 time. "
             "Lower MSE is better; positive MSE change means Metal has higher error.", "",
             "| Dataset | d | Epochs | Metal FP32 (s) | Faiss FP32 (s) | CPU/Metal | Metal test MSE | Faiss test MSE | Change |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {common.label(r['dataset'])} | {r['d']} | {r['epochs']:g} | {fmt(r['metal_fp32_seconds'])} | "
                     f"{fmt(r['faiss_fp32_seconds'])} | {fmt(r['cpu_over_metal_fp32'], 2)}× | "
                     f"{fmt(r['metal_fp32_test_mse'], 8)} | {fmt(r['faiss_fp32_test_mse'], 8)} | {fmt(r['fp32_test_mse_change_pct'], 3)}% |")
    lines += ["", "| Dataset | Metal footprint / RSS (GiB) | Faiss footprint / RSS (GiB) | Metal train-sample MSE | Faiss train-sample MSE |",
              "|---|---:|---:|---:|---:|"]
    for r in rows:
        memory = [f"{r[p + '_peak_physical_footprint_bytes'] / 2**30:.3f} / {r[p + '_peak_rss_bytes'] / 2**30:.3f}"
                  for p in ("metal_fp32", "faiss_fp32")]
        lines.append(f"| {common.label(r['dataset'])} | {memory[0]} | {memory[1]} | "
                     f"{fmt(r['metal_fp32_train_sample_mse'], 8)} | {fmt(r['faiss_fp32_train_sample_mse'], 8)} |")
    lines += ["", "## BF16 supplement: changed precision", "",
              "The supplemental Metal path rounds assignment operands and statistics inputs to BF16, with FP32 accumulation, "
              "norms, scores and centroid state. Its comparison below is against Metal FP32 only; it is not a matched-precision CPU speedup.", "",
              "| Dataset | BF16 (s) | Metal FP32/BF16 | BF16 footprint / RSS (GiB) | MLX peak active (GiB) | BF16 test MSE | vs FP32 | BF16 train-sample MSE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if "metal_bf16_seconds" not in r:
            continue
        lines.append(f"| {common.label(r['dataset'])} | {fmt(r['metal_bf16_seconds'])} | {fmt(r['metal_fp32_over_metal_bf16'], 2)}× | "
                     f"{r['metal_bf16_peak_physical_footprint_bytes'] / 2**30:.3f} / {r['metal_bf16_peak_rss_bytes'] / 2**30:.3f} | "
                     f"{r['metal_bf16_mlx_peak_active_bytes'] / 2**30:.3f} | {fmt(r['metal_bf16_test_mse'], 8)} | "
                     f"{fmt(r['bf16_vs_fp32_test_mse_change_pct'], 3)}% | {fmt(r['metal_bf16_train_sample_mse'], 8)} |")
    def links(path, title):
        return common.link(path, title, report)
    lines += ["", "## Preparation and lifetime memory diagnostics", "",
              "These process lifetime peaks include loading, preparation and warmup where applicable. "
              "They are not the matched fit-window measurements above, and do not include scoring after the fit. "
              "For DINO, two FP32 input copies alone require 76.3 GiB before host release, plus other allocations. "
              "Fit-window peaks alone do not describe the memory needed to prepare this workload.", "",
              "| Dataset / mode | Lifetime peak through preparation (GiB) | Lifetime peak through timed fit (GiB) |",
              "|---|---:|---:|"]
    for r in rows:
        for mode in MODES:
            prefix = mode.replace("-", "_")
            if f"{prefix}_seconds" not in r:
                continue
            values = [r.get(f"{prefix}_lifetime_peak_footprint_through_{phase}_bytes") for phase in ("preparation", "fit")]
            displayed = [fmt(value / 2**30) if value is not None else "—" for value in values]
            lines.append(f"| {common.label(r['dataset'])} / {mode} | {displayed[0]} | {displayed[1]} |")
    dino = next((row for row in rows if row["dataset"] == "dino10b-10M-div4096"), None)
    if dino:
        preparations = []
        for mode in ("metal-fp32", "metal-bf16"):
            value = dino.get(mode.replace("-", "_") + "_lifetime_peak_footprint_through_preparation_bytes")
            if value is not None:
                preparations.append(f"{mode}: {value / 2**30:.3f} GiB")
        if preparations:
            lines += ["", "DINO's observed process lifetime footprint through preparation was "
                      + "; ".join(preparations) + ". These measurements include allocations beyond the two input copies."]
    for filename in ("fp32-comparison.png", "bf16-supplement.png"):
        figure = HERE / "large_figures" / filename
        if figure.exists():
            lines += ["", "!" + common.link(figure, filename.removesuffix(".png").replace("-", " "), report)]
    lines += ["", "## Protocol and limits", "",
              "The explicit quality batch law matches pinned `paper_xb.py` for the 10M regime. "
              "DEEP and SPACEV use two epochs, SimSearchNet++ 1.6 and DINO 1.4: these are Table 4 aggregate "
              "crossing budgets applied as fixed local budgets, not recovered per-seed stopping points. "
              "SIFT uses an explicit local three-epoch budget; SIFT is absent from Table 4. "
              "No H100 timing or reproduction claim is made.", "",
              "Each mode runs in a fresh process with one full untimed warmup and resident FP32 input. "
              "Preparation, validation, upload, cache clearing, host centroid export and scoring are outside the timer. "
              "A timed fit includes a new model, permutations, updates, revival, history, online int64 assignments, "
              "closing final-pass means and synchronization. Original CPU recipe/state setup remains inside its unmodified train call; "
              "Metal preparation (including its FP32 global-mean assignment origin) is outside, with signature checks inside. "
              "CPU and BF16 assignment do not require that mean. A first timed call below two seconds triggers "
              "three timed calls; otherwise there is one. Time is their median; quality comes from the first timed fit.", "",
              "The original CPU implementation uses Faiss IndexFlat with eight OpenMP threads and unrestricted Accelerate. "
              "Metal FP32 assignment subtracts a global mean; original Faiss and Metal BF16 operate in original coordinates. "
              "Parallel reductions, equivalent EMA formulas, objective/count precision, and revival sampling arithmetic and "
              "random-number consumption differ. Identical initialization, recipe and seed do not imply bitwise-identical trajectories.", "",
              "Test MSE and the 10,000-row evenly spaced training-sample MSE use the same FP32 Faiss nearest-center assignment "
              "followed by direct FP64 residual evaluation. The training statistic is not full-training MSE. "
              "Dataset verification covers all test rows and 1,000 evenly spaced training fingerprints, not a complete training-file hash. "
              "DINO and SIFT retain the pinned /4096 and /512 scaling. "
              "Published splits preserve known duplicate vectors shared between evaluation and training: combined "
              "validation-plus-test overlaps are DEEP 4 and SPACEV 45 at 10M. These counts are not test-only counts, "
              "and the holdouts are not claimed to be duplicate-disjoint.", "",
              "Process physical footprint and RSS are sampled every 10 ms between the fit handshakes, including the resident-input "
              "baseline. Values shown are the maximum sampled peak over timed calls; short peaks can be missed and the measured "
              "maximum sample gap is retained in CSV/raw evidence. Baseline and peak increase are separately exported as maxima "
              "over calls and need not belong to the same call. MLX peak active allocation is a separate allocator statistic, "
              "not process memory. Metal releases its full NumPy input reference after preparation; CPU retains its contiguous "
              "input. Warmup and preparation transients are excluded; the raw post-preparation process snapshot includes lifetime "
              "diagnostics and is not a fit-window peak. DINO's resident FP32 input alone is 38.147 GiB.", "",
              "## Evidence and reproduction", "",
              f"The {links(HERE / 'large_protocol.json', 'frozen protocol')} records dataset fingerprints, scaling, "
              f"budgets and recipe constants. {links(folder / 'validation.json', 'Validation manifest')} records raw SHA-256 values, "
              f"coverage, source hashes and exclusions; {links(folder / 'summary.csv', 'CSV')} retains numerical values and raw paths. "
              f"Runtime snapshot: {links(HERE / 'reference/metal-v3-source.zip', 'Metal v3 archive')} "
              "(SHA-256 `c24dfaf84b3a75c1d741fb1708bb37addda013dc9301d9d6b3166d5f52077413`). "
              f"The {links(HERE / 'reference/large-comparison-source.zip', 'shared source archive')} "
              "also preserves the exact benchmark driver, loader, memory sampler and protocol "
              "(SHA-256 `a61613a6c3dabb59690545d9a4b43796c6b104b8e817f502558ee1279287cf39`). "
              f"Original Faiss revision: `{REVISION}`.", "",
              f"The full-DINO {links(HERE / 'results/large_validation/dino-index-boundaries.json', 'index-boundary probe')} "
              f"and {links(HERE / 'large_index_probe.py', 'probe source')} check exact FP32/BF16 gathers at rows "
              "0, 4,194,303/4,194,304, 8,388,607/8,388,608 and 9,999,999, across 2³²-element offsets. "
              "This correctness probe is excluded from performance measurements.", "",
              f"The {links(HERE / 'results/large_validation/preflight.json', 'preflight record')} records the "
              "three-mode smoke test, imported runtime paths and source/environment checks before the campaign.", "",
              "```sh", ".venv/bin/python benchmarks/e_means/large_compare.py",
              ".venv/bin/python benchmarks/e_means/large_report.py --require-complete", "```", ""]
    if validation.get("provenance"):
        lines += ["Recorded host and software:", "", "```json",
                  json.dumps({key: validation["provenance"][key] for key in ("hardware", "versions")}, indent=2, sort_keys=True), "```", ""]
    lines += ["| Dataset | Metal FP32 evidence | Faiss FP32 evidence | Metal BF16 evidence |",
              "|---|---|---|---|"]
    for row in rows:
        evidence = []
        for mode in MODES:
            path = row.get(mode.replace("-", "_") + "_evidence_path")
            evidence.append(links(path, "raw JSON") if path else "—")
        lines.append(f"| {common.label(row['dataset'])} | " + " | ".join(evidence) + " |")
    lines.append("")
    if not validation["complete"]:
        lines += ["**Incomplete evidence:** consult the validation manifest for missing or rejected records. "
                  "This draft must not be read as full five-dataset coverage.", ""]
    return "\n".join(lines)


def generate(folder, report, protocol_path, require_complete=False):
    folder, report, protocol_path = Path(folder).resolve(), Path(report).resolve(), Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    raw = evidence_rows(folder)
    snapshot = verify_source_archive(HERE / "reference/metal-v3-source.json", RUNTIME_ARCHIVE_SHA256)
    shared = verify_source_archive(HERE / "reference/large-comparison-source.json", SHARED_ARCHIVE_SHA256)
    if not source_map(shared.get("reference_source_sha256"), REFERENCE_FIELDS) or shared.get("faiss_reference_revision") != REVISION:
        raise ValueError("Frozen comparison metadata has an invalid reference source map/revision")
    issues = [(row, archive_record_problem(row, snapshot, shared)) for row in raw]
    valid_runtime = [row for row, issue in issues if issue is None]
    mismatches = [(row, issue) for row, issue in issues if issue is not None]
    groups, validation = validate(valid_runtime, protocol, common.sha256(protocol_path))
    validation["observed_records"] = len(raw)
    if mismatches:
        validation["complete"] = False
        validation["rejected"].extend({"path": str(row["_path"]), "reason": issue} for row, issue in mismatches)
    validation.update(protocol_path=str(protocol_path), protocol_sha256=common.sha256(protocol_path),
                      report_generator_sha256=common.sha256(__file__),
                      comparison_report_sha256=common.sha256(HERE / "comparison_report.py"),
                      frozen_runtime={**snapshot, "matches_all_records": all(
                          all(row.get("source_sha256", {}).get(key) == value for key, value in snapshot["source_sha256"].items())
                          for row in raw)},
                      frozen_shared_sources={**shared, "matches_all_records": not mismatches},
                      raw_evidence=[{"path": str(row["_path"]), "sha256": common.sha256(row["_path"])} for row in raw])
    if groups:
        first = next(iter(groups[0].values()))
        validation["provenance"] = {key: first[key] for key in GLOBAL_FIELDS}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    if require_complete and not validation["complete"]:
        raise ValueError("Incomplete or incompatible evidence; validation.json saved, final CSV/report not written")
    summary = summarize(groups)
    with (folder / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown(summary, validation, protocol, report, folder))
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results/large_comparison")
    parser.add_argument("--protocol", type=Path, default=HERE / "large_protocol.json")
    parser.add_argument("--report", type=Path, default=REPO / "docs/e-means-10m-comparison-2026-11-14.md")
    parser.add_argument("--require-complete", action="store_true", help="Refuse final CSV/report unless all 15 records validate")
    args = parser.parse_args()
    try:
        validation = generate(args.results, args.report, args.protocol, args.require_complete)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({key: validation[key] for key in ("complete", "accepted_records", "expected_records")}))


if __name__ == "__main__":
    main()
