"""Read recorded results and calculate useful-work throughput; never loads MLX/data."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
REPO = BENCH.parents[1]
sys.path.insert(0, str(BENCH))
import h100_report as report  # noqa: E402

SOURCES = {
    "paper": "https://arxiv.org/abs/2611.09501v1",
    "nvidia_final_whitepaper": "https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf",
    "nvidia_product": "https://www.nvidia.com/en-us/data-center/h100/",
    "nvidia_pcie_spec": "https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf",
    "apple_m4max_specs": "https://support.apple.com/en-us/121553",
    "apple_m4max_launch": "https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/",
    "apple_m5_matrix_accelerators": "https://machinelearning.apple.com/research/exploring-llms-mlx-m5",
    "reference_base": "https://github.com/facebookresearch/faiss/blob/ebab8f784c1d177129602269d5ad966c525fbdcb/",
}
PAPER_MILLIONS = {"gooaq": 1.48, "codesearchnet": 1.37, "arxiv": 1.34, "imagenet": 1.28,
                  "glove": 1.19, "agnews": .77, "landmark-dino": .76, "landmark-nomic": .76,
                  "ccnews": .50, "simplewiki": .26, "celeba": .20}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    raw = report.evidence_rows(BENCH / "results/h100_protocol")
    valid, pairs, excluded = report.validate_and_pair(raw)
    groups = report.groups(valid, pairs)
    metrics = []
    for group in groups:
        records = group["optimized_rows"]
        first = records[0]
        recipe = first["recipe_resolved"]
        epoch_rows = recipe["epoch_rows"]
        processed = sum(epoch_rows)
        for record in records:
            assert record["recipe_resolved"]["epoch_rows"] == epoch_rows
            assert [entry["rows"] for entry in record["iteration_history"]] == epoch_rows
            assert record["iteration_history"][-1]["steps"] == recipe["total_steps"]
        work = 2 * processed * first["k"] * first["d"]
        seconds = median(record["train_seconds"] for record in records)
        ref_seconds = group["h100_seconds"]
        metric = {
            "dataset": group["dataset"], "label": report.label(group["dataset"]),
            "n": first["n"], "d": first["d"], "k": first["k"], "epochs": group["epochs"],
            "epoch_rows": epoch_rows, "processed_rows": processed,
            "batch_size": recipe["batch_size"], "statistical_steps": recipe["total_steps"],
            "permutations_of_full_n": len(epoch_rows), "seeds": [r["config"]["seed"] for r in records],
            "useful_assignment_flops": work, "median_whole_fit_seconds": seconds,
            "fit_equivalent_useful_tflops_per_second": work / seconds / 1e12,
            "processed_million_rows_per_second": processed / seconds / 1e6,
            "reported_h100_seconds": ref_seconds,
            "contextual_time_ratio": seconds / ref_seconds if ref_seconds else None,
            "conditional_same_local_work_h100_tflops_per_second": work / ref_seconds / 1e12 if ref_seconds else None,
            "conditional_h100_sxm_dense_peak_fraction": work / ref_seconds / 989.4e12 if ref_seconds else None,
            "conditional_h100_pcie_dense_peak_fraction": work / ref_seconds / 756e12 if ref_seconds else None,
            "reference_seconds_rounding_assumption": "nearest 0.01 second, for sensitivity only" if ref_seconds else None,
            "ratio_range_from_time_rounding_only": [seconds / (ref_seconds + .005), seconds / (ref_seconds - .005)] if ref_seconds else None,
            "paper_table11_rounded_n": int(PAPER_MILLIONS[group["dataset"][5:]] * 1e6) if group["dataset"].startswith("vibe-") else None,
            "raw_evidence": [{"path": str(r["_path"].relative_to(REPO)), "sha256": digest(r["_path"])} for r in records],
        }
        if metric["paper_table11_rounded_n"] is not None:
            metric["conditional_table11_nominal_work_h100_tflops_per_second"] = (
                2 * metric["paper_table11_rounded_n"] * group["epochs"] * first["d"] * first["k"] / ref_seconds / 1e12)
        metrics.append(metric)
    sweep_path = BENCH / "results/optimization/bf16-dispatch-sweep.json"
    sweep = json.loads(sweep_path.read_text())
    observed_micro = []
    for entry in sweep["results"]:
        method = min((m for m in entry["methods"] if m["method"] == "native"), key=lambda m: m["median_seconds"])
        observed_micro.append({"n": entry["n"], "k": entry["k"], "d": entry["d"],
                               "seconds": method["median_seconds"], "config": method["config"],
                               "useful_tflops_per_second": 2 * entry["n"] * entry["k"] * entry["d"] / method["median_seconds"] / 1e12})
    profile_path = BENCH / "results/optimization/profile-bf16-optimized.json"
    profiles = []
    for item in json.loads(profile_path.read_text())["records"]:
        assignment = item["stage_totals"]["assign"]["seconds"]
        profiles.append({"dataset": item["dataset"], "instrumented_seconds": item["instrumented_fit_seconds"],
                         "assignment_seconds": assignment,
                         "assignment_fraction_of_serialized_profile": assignment / item["instrumented_fit_seconds"],
                         "optimistic_zero_nonassignment_time_speedup": item["instrumented_fit_seconds"] / assignment})
    fresh_stage_path, roofline_path = HERE / "stages-before.json", HERE / "roofline.json"
    fresh_stages = []
    fresh = json.loads(fresh_stage_path.read_text()) if fresh_stage_path.exists() else None
    if fresh:
        for item in fresh["records"]:
            metric = next(m for m in metrics if m["dataset"] == item["dataset"])
            original = next(r for r in valid if r["config"]["dataset"] == item["dataset"]
                            and r["config"]["seed"] == 10 and r["config"]["mode"] == "optimized")
            assert sum(item["recipe"]["epoch_rows"]) == metric["processed_rows"]
            runtime_keys = [key for key in original["source_sha256"] if key.startswith("src/")]
            assignment = item["stage_totals"]["assign"]["seconds"]
            fresh_stages.append({"dataset": item["dataset"], "seed": 10,
                                 "runtime_source_hashes_match_campaign": all(fresh["source_sha256"].get(key) == original["source_sha256"][key] for key in runtime_keys),
                                 "fresh_no_sampler_fit_seconds": item["fit_seconds"],
                                 "campaign_seed10_seconds": original["train_seconds"],
                                 "campaign_all_seed_median_seconds": metric["median_whole_fit_seconds"],
                                 "fresh_effective_useful_tflops_per_second": metric["useful_assignment_flops"] / item["fit_seconds"] / 1e12,
                                 "fresh_contextual_h100_ratio": item["fit_seconds"] / metric["reported_h100_seconds"] if metric["reported_h100_seconds"] else None,
                                 "assignment_fraction_of_serialized_profile": assignment / item["instrumented_fit_seconds"],
                                 "stage_totals": item["stage_totals"], "other_seconds": item["other_seconds"],
                                 "limitation": "Fresh diagnostic excludes external sampler; run-state, timing-controller and sampler effects are not isolated. Stage instrumentation serializes work."})
    roofline = json.loads(roofline_path.read_text()) if roofline_path.exists() else None
    output = {
        "kind": "read_only_h100_gap_audit", "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "valid_records": len(valid), "paired_seeds": len(pairs), "excluded_records": len(excluded),
        "no_new_gpu_measurements": True, "sources": SOURCES,
        "work_definition": "2 * sum(recorded epoch_rows) * k * d; FMA counts as 2 FLOPs. Useful dense assignment work only, divided by full-fit median time, not an instruction counter or isolated GEMM throughput.",
        "work_exclusions": ["tile/dimension padding", "argmin/norms/residuals", "statistics and EMA", "revival", "permutations", "host/framework work"],
        "reference_limit": "H100 rates are conditional same-local-work estimates, not H100 measurements: GPU epoch association, per-seed stopping, exact VIBE train sizes/holdouts and original execution source are unverified.",
        "hardware": {"local_gpu": "Apple M4 Max", "local_gpu_cores": 40,
                     "core_count_evidence": "system_profiler SPDisplaysDataType -json, sppci_cores=40, read 2026-11-14",
                     "local_memory_bytes": 137438953472, "apple_official_bandwidth_GBps": 546,
                     "apple_official_dense_bf16_tflops": None,
                     "h100_paper": "80GB; form factor not explicit",
                     "h100_readme_log": "NVIDIA H100 80GB HBM3; supports SXM-class inference for reproduction, not proof of Table11 SKU",
                     "h100_sxm_dense_bf16_tflops": 989.4, "h100_sxm_sparse_bf16_tflops": 1978.9,
                     "h100_sxm_bandwidth_GBps": 3350, "h100_pcie_dense_bf16_tflops": 756,
                     "h100_pcie_bandwidth_GBps": 2000, "sxm_to_apple_bandwidth_ratio": 3350 / 546,
                     "pcie_to_apple_bandwidth_ratio": 2000 / 546},
        "metrics": metrics, "historical_best_observed_native_assignment_by_shape": observed_micro,
        "historical_profile_diagnostics": profiles,
        "historical_evidence": [{"path": str(p.relative_to(REPO)), "sha256": digest(p)} for p in (sweep_path, profile_path)],
        "analysis_source_sha256": digest(__file__),
        "report_helper_sha256": digest(BENCH / "h100_report.py"),
        "fresh_unchanged_runtime_diagnostics": fresh_stages,
        "fresh_arithmetic_proxies": roofline["records"] if roofline else [],
        "fresh_evidence": [{"path": str(p.relative_to(REPO)), "sha256": digest(p)}
                           for p in (fresh_stage_path, roofline_path) if p.exists()],
        "arithmetic_proxy_limit": "Diagnostic pure FP16/BF16-output GEMM omits assignment and changes output precision; it is not a drop-in valid clustering kernel or a proven hardware ceiling. Attained throughput is not an official Apple peak.",
    }
    assert all(math.isfinite(m["fit_equivalent_useful_tflops_per_second"]) for m in metrics)
    (HERE / "workload_analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    table = ["| Dataset | n | d | k | Epochs | Processed rows | Assignment TFLOP | Metal s | Metal effective TFLOP/s | H100 s | Conditional H100 TFLOP/s | Context ratio |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in metrics:
        table.append(f"| {m['label']} | {m['n']:,} | {m['d']} | {m['k']:,} | {m['epochs']:g} | {m['processed_rows']:,} | "
                     f"{m['useful_assignment_flops']/1e12:.2f} | {m['median_whole_fit_seconds']:.3f} | {m['fit_equivalent_useful_tflops_per_second']:.2f} | "
                     f"{report.f(m['reported_h100_seconds'],2)} | {report.f(m['conditional_same_local_work_h100_tflops_per_second'],1)} | "
                     f"{report.f(m['contextual_time_ratio'],1)} |")
    (HERE / "workload_table.md").write_text("\n".join(table) + "\n")
    print(json.dumps({"records": len(valid), "paired_seeds": len(pairs), "metrics": len(metrics),
                      "output": str(HERE / "workload_analysis.json")}))


if __name__ == "__main__":
    main()
