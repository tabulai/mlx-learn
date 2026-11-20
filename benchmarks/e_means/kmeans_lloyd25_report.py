"""Audit and report the Lloyd-25 reference addendum without GPU work."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import comparison_report as common  # noqa: E402
import kmeans_compare as benchmark  # noqa: E402
import kmeans_lloyd25 as driver  # noqa: E402
import kmeans_report as frozen  # noqa: E402

MODES = driver.MODES
LLOYD, EMEANS = MODES
GLOBAL_FIELDS = ("source_sha256", "reference_source_sha256", "sklearn_source_sha256", "versions", "hardware",
                 "thread_environment", "effective_omp_threads", "metal_device")
PAIR_FIELDS = frozen.PAIR_FIELDS
# Only the arm, its budget and its output path may differ inside a dataset pair.
ARM_CONFIG = {"mode", "epochs", "protocol_config", "output", "worker_output"}
CSV_FIELDS = ("dataset", "n", "d", "k", "lloyd_budget", "lloyd_iterations", "emeans_epochs",
              "lloyd_seconds", "emeans_seconds", "runtime_ratio",
              "lloyd_training_seconds", "emeans_training_seconds", "training_ratio",
              "lloyd_iteration_seconds_min", "lloyd_iteration_seconds_median", "lloyd_iteration_seconds_max",
              "lloyd_test_mse", "emeans_test_mse", "test_mse_change_pct",
              "lloyd_train_sample_mse", "emeans_train_sample_mse", "lloyd_final_inertia", "emeans_final_inertia",
              "inertia_change_pct", "crossing_updates", "crossing_seconds", "matched_training_ratio",
              "lloyd_peak_footprint_bytes", "emeans_peak_footprint_bytes",
              "lloyd_evidence_path", "lloyd_evidence_sha256", "emeans_evidence_path", "emeans_evidence_sha256")


def row_problem(row, protocol, entries, protocol_hash):
    try:
        config = row["config"]
        benchmark.validate_result(row, config, row.get("memory_samples", []))
        entry = next((e for e in entries if e["dataset"] == config["dataset"]), None)
        if entry is None or config["mode"] not in MODES or config.get("addendum") != driver.KIND:
            return "record is not an addendum arm of a frozen dataset"
        expected = driver.protocol_entry(entry, config["mode"], protocol["lloyd_iterations"])
        fixed = {"seed": 10, "epochs": expected["epochs"], "protocol_config": expected, "precision": "float32",
                 "n_init": 1, "tol": 0.0, "final_labels": True, "threads": 8, "train_score_rows": 10000,
                 "memory": True, "memory_interval_seconds": .01, "rows": None, "k_override": None,
                 "epochs_override": None, "diagnostic_subset": False}
        if any(config.get(key) != value for key, value in fixed.items()):
            return "record is outside the full fixed addendum protocol"
        if any(row.get(key) != entry[key] for key in ("n", "d", "k")):
            return "record is not the full dataset shape"
        if not common.positive(row.get("warmup_seconds")):
            return "missing full warmup"
        hashes = [row.get(key) for key in ("source_sha256", "reference_source_sha256", "sklearn_source_sha256")]
        if not all(frozen.hash_mapping(value) for value in (*hashes, config.get("addendum_source_sha256"))):
            return "missing valid source hashes"
        if protocol_hash not in row["source_sha256"].values():
            return "recorded addendum protocol hash differs from report contract"
        if any(row.get(key) is None for key in (*GLOBAL_FIELDS, *PAIR_FIELDS)):
            return "missing provenance or pairing field"
        history = row["iteration_history"]
        if len(history) != row["actual_iterations"]:
            return "per-pass history does not cover the fit"
        if config["mode"] == LLOYD:
            if row.get("recipe_resolved") is not None:
                return "Lloyd record unexpectedly declares an e-means recipe"
            if any(not common.positive(step.get("seconds")) or not common.nonnegative(step.get("inertia_before_update"))
                   for step in history):
                return "invalid Lloyd per-iteration timing or inertia"
        elif not common.positive(history[-1].get("cumulative_seconds")):
            return "missing e-means cumulative training time"
        return frozen.memory_problem(row)
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, IndexError) as error:
        return f"invalid measurement: {error}"


def accept(rows, protocol, entries, protocol_hash):
    """Records that pass every check, at most one per dataset and arm, from a single campaign."""
    candidates, rejected = defaultdict(list), []
    for row in rows:
        issue = row_problem(row, protocol, entries, protocol_hash)
        if issue:
            rejected.append({"path": str(row.get("_path", "")), "reason": issue})
        else:
            candidates[(row["config"]["dataset"], row["config"]["mode"])].append(row)
    accepted = {}
    for identity, members in candidates.items():
        if len(members) == 1:
            accepted[identity] = members[0]
        else:
            rejected.extend({"path": str(row.get("_path", "")), "reason": "duplicate dataset/mode"} for row in members)
    campaigns = {common.stable([{key: row[key] for key in GLOBAL_FIELDS}, row["config"]["addendum_source_sha256"]])
                 for row in accepted.values()}
    if len(campaigns) > 1:
        rejected.extend({"path": str(row.get("_path", "")), "reason": "mixed source/environment/device campaigns"}
                        for row in accepted.values())
        accepted = {}
    return accepted, rejected


def validate(rows, protocol, entries, protocol_hash):
    accepted, rejected = accept(rows, protocol, entries, protocol_hash)
    pairs = []
    for entry in entries:
        group = {mode: accepted[(entry["dataset"], mode)] for mode in MODES if (entry["dataset"], mode) in accepted}
        if len(group) < 2:
            continue
        lloyd, emeans = group[LLOYD], group[EMEANS]
        shared = [{k: v for k, v in row["config"].items() if k not in ARM_CONFIG} for row in (lloyd, emeans)]
        if any(lloyd[key] != emeans[key] for key in PAIR_FIELDS) or shared[0] != shared[1]:
            for mode, row in group.items():
                rejected.append({"path": str(row.get("_path", "")), "reason": "data/init/config differs across arms"})
                del accepted[(entry["dataset"], mode)]
        else:
            pairs.append(group)
    expected = {(entry["dataset"], mode) for entry in entries for mode in MODES}
    missing = sorted(expected - accepted.keys())
    return pairs, {"complete": not missing and not rejected and len(rows) == len(expected),
                   "expected_records": len(expected), "observed_records": len(rows), "accepted_records": len(accepted),
                   "missing": [{"dataset": name, "seed": 10, "mode": mode} for name, mode in missing], "rejected": rejected}


def lloyd_inertias(row):
    """Full-training inertia after 0, 1, ... Lloyd updates, ending with the returned centers."""
    return [step["inertia_before_update"] for step in row["iteration_history"]] + [row["final_inertia"]]


def summarize(pairs):
    rows = []
    for pair in pairs:
        lloyd, emeans = pair[LLOYD], pair[EMEANS]
        steps = [step["seconds"] for step in lloyd["iteration_history"]]
        training = {"lloyd": sum(steps), "emeans": emeans["iteration_history"][-1]["cumulative_seconds"]}
        crossing = next((updates for updates, inertia in enumerate(lloyd_inertias(lloyd))
                         if inertia <= emeans["final_inertia"]), None)
        crossing_seconds = sum(steps[:crossing]) if crossing is not None else None
        row = {"dataset": lloyd["config"]["dataset"], **{key: lloyd[key] for key in ("n", "d", "k")},
               "lloyd_budget": lloyd["config"]["epochs"], "lloyd_iterations": lloyd["actual_iterations"],
               "emeans_epochs": emeans["actual_iterations"],
               "runtime_ratio": lloyd["train_seconds"] / emeans["train_seconds"],
               "lloyd_training_seconds": training["lloyd"], "emeans_training_seconds": training["emeans"],
               "training_ratio": training["lloyd"] / training["emeans"],
               # Identical work every iteration: the spread shows drift or foreign load.
               "lloyd_iteration_seconds_min": min(steps), "lloyd_iteration_seconds_median": median(steps),
               "lloyd_iteration_seconds_max": max(steps),
               "test_mse_change_pct": 100 * (emeans["test_mse"] / lloyd["test_mse"] - 1),
               "inertia_change_pct": 100 * (emeans["final_inertia"] / lloyd["final_inertia"] - 1),
               "crossing_updates": crossing, "crossing_seconds": crossing_seconds,
               "matched_training_ratio": crossing_seconds / training["emeans"] if crossing is not None else None}
        for name, raw in (("lloyd", lloyd), ("emeans", emeans)):
            row.update({f"{name}_seconds": raw["train_seconds"], f"{name}_test_mse": raw["test_mse"],
                        f"{name}_train_sample_mse": raw["train_sample_mse"], f"{name}_final_inertia": raw["final_inertia"],
                        f"{name}_peak_footprint_bytes": max(m["peak_physical_footprint_bytes"] for m in raw["memory_samples"]),
                        f"{name}_evidence_path": str(raw.get("_path") or ""),
                        f"{name}_evidence_sha256": common.sha256(raw["_path"]) if raw.get("_path") else None})
        rows.append(row)
    return rows


def changed_sources(pair, earlier):
    """Measured source files whose hashes differ between two campaigns' records."""
    def hashes(row):
        return {**row["source_sha256"], **row["config"]["addendum_source_sha256"]}

    new, old = hashes(pair[LLOYD]), hashes(earlier[LLOYD])
    return sorted(name for name in new.keys() | old.keys() if new.get(name) != old.get(name))


def control_rows(rows, protocol, entries, protocol_hash):
    """Same-session controls measured from an earlier campaign's sources, by (dataset, arm).

    The e-means arm is cheap, so its control is a full addendum record. A Lloyd
    control only has to time identical iterations: it may stop after a few, as
    a diagnostic record the harness validator accepts on the full dataset.
    """
    controls, _ = accept([row for row in rows if row["config"]["mode"] == EMEANS], protocol, entries, protocol_hash)
    for row in rows:
        config = row["config"]
        if config["mode"] != LLOYD or config.get("rows") is not None or config.get("k_override") is not None:
            continue
        try:
            benchmark.validate_result(row, config, row.get("memory_samples", []))
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, IndexError):
            continue
        if row["iteration_history"] and all(common.positive(step.get("seconds")) for step in row["iteration_history"]):
            controls[(config["dataset"], LLOYD)] = row
    return controls


def changes(pairs, earlier_pairs, replicate=None):
    """Each dataset of this campaign against an earlier campaign.

    ``replicate`` holds ``control_rows`` measured from the earlier campaign's
    sources in this campaign's session. They separate the source change from
    whatever else differs between two sessions on one machine.
    """
    earlier = {pair[LLOYD]["config"]["dataset"]: pair for pair in earlier_pairs}
    rows = []
    for pair, new in zip(pairs, summarize(pairs), strict=True):
        before = earlier.get(new["dataset"])
        if before is None:
            continue
        old = summarize([before])[0]
        # A fit that stopped early has fewer states; compare the shared prefix and say so.
        states = list(zip(lloyd_inertias(pair[LLOYD]), lloyd_inertias(before[LLOYD]), strict=False))
        row = {"dataset": new["dataset"], "changed_sources": changed_sources(pair, before),
               "lloyd_speedup": old["lloyd_seconds"] / new["lloyd_seconds"],
               "emeans_speedup": old["emeans_seconds"] / new["emeans_seconds"],
               "lloyd_iteration_speedup": old["lloyd_iteration_seconds_median"] / new["lloyd_iteration_seconds_median"],
               # Both fits start from the same centers, so every Lloyd state is comparable.
               "lloyd_inertia_states": len(states) if len(lloyd_inertias(pair[LLOYD])) == len(lloyd_inertias(before[LLOYD])) else None,
               "lloyd_inertia_max_relative_difference": max(abs(a / b - 1) if b else float(a != b) for a, b in states),
               "lloyd_reused_rows": sum(step.get("pruned_rows", 0) for step in pair[LLOYD]["iteration_history"]),
               "replicate_emeans_seconds": None, "replicate_emeans_speedup": None, "replicate_reproduces_earlier": None,
               "replicate_lloyd_iterations": None, "replicate_lloyd_iteration_seconds_median": None,
               "replicate_lloyd_iteration_speedup": None, "replicate_lloyd_reproduces_earlier": None}
        for key in ("lloyd_seconds", "emeans_seconds", "lloyd_iteration_seconds_median", "runtime_ratio", "training_ratio",
                    "lloyd_test_mse", "emeans_test_mse", "lloyd_final_inertia", "emeans_final_inertia",
                    "lloyd_peak_footprint_bytes", "emeans_peak_footprint_bytes"):
            row[key], row[f"earlier_{key}"] = new[key], old[key]
        # Only a replicate of exactly the earlier sources, data and start is a control.
        control = {mode: (replicate or {}).get((new["dataset"], mode)) for mode in MODES}
        control = {mode: found for mode, found in control.items() if found is not None
                   and all(found[key] == before[mode][key] for key in (*GLOBAL_FIELDS, *PAIR_FIELDS))}
        if EMEANS in control:
            found = control[EMEANS]
            row.update(replicate_emeans_seconds=found["train_seconds"],
                       replicate_emeans_speedup=found["train_seconds"] / new["emeans_seconds"],
                       replicate_reproduces_earlier=all(found[key] == before[EMEANS][key]
                                                        for key in ("final_inertia", "test_mse", "train_sample_mse")))
        if LLOYD in control:
            steps = control[LLOYD]["iteration_history"]
            seconds = median(step["seconds"] for step in steps)
            row.update(replicate_lloyd_iterations=len(steps), replicate_lloyd_iteration_seconds_median=seconds,
                       replicate_lloyd_iteration_speedup=seconds / new["lloyd_iteration_seconds_median"],
                       replicate_lloyd_reproduces_earlier=[step["inertia_before_update"] for step in steps]
                       == lloyd_inertias(before[LLOYD])[:len(steps)])
        rows.append(row)
    return rows


def same(new, old, form):
    """Render a quantity that the source change may or may not have altered."""
    if new == old:
        return f"{form(new)}, identical"
    return f"{form(old)} → {form(new)} ({100 * (new / old - 1):+.2g}%)"


def comparison_markdown(rows, earlier_folder, replicate_folder, link):
    lines = ["## Change from the earlier campaign", "",
             f"The {link(earlier_folder/'validation.json', 'earlier campaign')} used the same protocol, harness, data, "
             "initial centers and machine. "]
    sources = sorted({name for row in rows for name in row["changed_sources"]})
    lines[-1] += (f"{len(sources)} measured source file{'s differ' if len(sources) != 1 else ' differs'} between the two: "
                  + ", ".join(f"`{name}`" for name in sources) + ".")
    lines += ["", "| Dataset | Lloyd-25 (s), earlier → now | Speedup | Lloyd s / iteration, earlier → now | Speedup | "
              "E-means (s), earlier → now | Speedup | Lloyd / e-means, earlier → now |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {common.label(row['dataset'])} | {row['earlier_lloyd_seconds']:.1f} → {row['lloyd_seconds']:.1f} | "
                     f"{row['lloyd_speedup']:.2f}× | {row['earlier_lloyd_iteration_seconds_median']:.2f} → "
                     f"{row['lloyd_iteration_seconds_median']:.2f} | {row['lloyd_iteration_speedup']:.2f}× | "
                     f"{row['earlier_emeans_seconds']:.1f} → {row['emeans_seconds']:.1f} | {row['emeans_speedup']:.2f}× | "
                     f"{row['earlier_runtime_ratio']:.2f}× → {row['runtime_ratio']:.2f}× |")
    lines += ["", "Each campaign is one timed fit per method, taken in a different session on the same machine, so these "
              "speedups combine the source change with any difference in machine state."]
    controls = [row for row in rows if row["replicate_emeans_seconds"] is not None
                or row["replicate_lloyd_iteration_seconds_median"] is not None]
    if controls:
        def cells(row, earlier, control, now, speedup, form):
            if row[control] is None:
                return f"{form(row[earlier])} | not measured | {form(row[now])} | |"
            return f"{form(row[earlier])} | {form(row[control])} | {form(row[now])} | {row[speedup]:.2f}× |"

        def reproduced(row):
            names = [name for name, key in (("e-means", "replicate_reproduces_earlier"),
                                            ("Lloyd", "replicate_lloyd_reproduces_earlier")) if row[key]]
            return " and ".join(names) if names else "neither"

        lloyd_iterations = sorted({row["replicate_lloyd_iterations"] for row in controls} - {None})
        lines += ["", f"The {link(replicate_folder, 'same-session control')} measures the earlier sources again in the "
                  "session of this campaign: the full e-means fit"
                  + (f", and a Lloyd fit stopped after {' or '.join(map(str, lloyd_iterations))} of its identical iterations"
                     if lloyd_iterations else "") + ". Its speedups hold the machine state fixed.", "",
                  "| Dataset | E-means (s), earlier record | Earlier sources, this session | This campaign | Speedup | "
                  "Lloyd s / iteration, earlier record | Earlier sources, this session | This campaign | Speedup | "
                  "Control reproduces the earlier result bit for bit |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for row in controls:
            lines.append(f"| {common.label(row['dataset'])} | "
                         + cells(row, "earlier_emeans_seconds", "replicate_emeans_seconds", "emeans_seconds",
                                 "replicate_emeans_speedup", "{:.1f}".format) + " "
                         + cells(row, "earlier_lloyd_iteration_seconds_median", "replicate_lloyd_iteration_seconds_median",
                                 "lloyd_iteration_seconds_median", "replicate_lloyd_iteration_speedup", "{:.2f}".format)
                         + f" {reproduced(row)} |")
        lines += ["", "The last column compares the control's e-means inertia and MSE, and the Lloyd inertia of each "
                  "iteration it ran, with the earlier records."]
    lines += ["", "| Dataset | Lloyd test MSE | Largest Lloyd training-inertia difference over the fit | Lloyd rows reused | "
              "E-means test MSE | Fit footprint GiB, Lloyd | Fit footprint GiB, e-means |",
              "|---|---|---:|---:|---|---:|---:|"]
    for row in rows:
        drift = row["lloyd_inertia_max_relative_difference"]
        states = "different iteration counts" if row["lloyd_inertia_states"] is None else (
            f"none in {row['lloyd_inertia_states']} states" if drift == 0 else f"{drift:.2g} relative")
        lines.append(f"| {common.label(row['dataset'])} | {same(row['lloyd_test_mse'], row['earlier_lloyd_test_mse'], '{:.8f}'.format)} | "
                     f"{states} | {row['lloyd_reused_rows']:,} | "
                     f"{same(row['emeans_test_mse'], row['earlier_emeans_test_mse'], '{:.8f}'.format)} | "
                     f"{row['earlier_lloyd_peak_footprint_bytes']/2**30:.3f} → {row['lloyd_peak_footprint_bytes']/2**30:.3f} | "
                     f"{row['earlier_emeans_peak_footprint_bytes']/2**30:.3f} → {row['emeans_peak_footprint_bytes']/2**30:.3f} |")
    lines += ["", "A Lloyd state is the full-training FP32 inertia before each update and after the last. Rows reused counts "
              "training rows whose previous label was certified without a dense search, summed over the iterations.", ""]
    return lines


def headline(rows):
    if not rows:
        return "No paired dataset has both arms yet."

    def span(key, form):
        values = [row[key] for row in rows]
        return f"{form(min(values))} to {form(max(values))}"

    wins = sum(row["test_mse_change_pct"] <= 0 for row in rows)
    unreached = sum(row["crossing_updates"] is None for row in rows)
    return (f"Across the {len(rows)} paired dataset{'s' if len(rows) != 1 else ''}, 25-iteration Metal Lloyd took "
            f"{span('runtime_ratio', '{:.2f}×'.format)} the time of e-means with final labels in both timers, and "
            f"{span('training_ratio', '{:.2f}×'.format)} on training alone. E-means test MSE relative to Lloyd ranged "
            f"from {span('test_mse_change_pct', '{:+.3f}%'.format)}; it was equal or lower on {wins}/{len(rows)}. "
            f"Lloyd did not reach the e-means training inertia within its budget on {unreached}/{len(rows)}.")


def markdown(rows, validation, protocol, report, folder, protocol_path, comparison=None):
    def link(path, title):
        return common.link(path, title, report)

    lines = ["# E-means versus 25-iteration Lloyd on the 10M datasets", "",
             f"**{'Complete' if validation['complete'] else 'INCOMPLETE'}: {validation['accepted_records']}/"
             f"{validation['expected_records']} accepted measurements.** Each configuration uses 10,000,000 rows, "
             "65,536 centers, FP32 input and seed 10. These are single-seed, single-fit measurements, with no error bars.", ""]
    unpaired = list(dict.fromkeys(common.label(item["dataset"]) for item in validation["missing"]))
    if unpaired:
        lines += [f"Without a complete pair, and absent from every table and range below: {', '.join(unpaired)}.", ""]
    lines += [
             f"The {link(REPO/'docs/k-means-10m-comparison-2026-11-15.md', 'matched-budget comparison')} stops Lloyd at the "
             "e-means pass budget, so its runtimes are equal by construction. Here Metal KMeans runs the paper's "
             f"{protocol['lloyd_iterations']}-iteration reference budget and Metal e-means keeps its own two- or three-epoch "
             "budget. Both start from identical explicit centers and share the assignment and statistics kernels.", "",
             headline(rows), "",
             "| Dataset | Lloyd budget / actual | E-means epochs | Lloyd (s) | E-means (s) | Lloyd / e-means | E-means test MSE vs Lloyd |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {common.label(row['dataset'])} | {row['lloyd_budget']} / {row['lloyd_iterations']} | {row['emeans_epochs']} | "
                     f"{row['lloyd_seconds']:.3f} | {row['emeans_seconds']:.3f} | {row['runtime_ratio']:.2f}× | "
                     f"{row['test_mse_change_pct']:+.3f}% |")
    lines += ["", "Times include training, fresh final labels and inertia. A ratio above one means e-means is faster. "
              "A negative MSE change means e-means has lower held-out error.", "",
              "## Training time and inertia crossing", "",
              "| Dataset | Lloyd training (s) | Lloyd s / iteration, median [min, max] | E-means training (s) | Training ratio | "
              "Lloyd updates to reach e-means inertia | Matched training ratio |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["crossing_updates"] is None:
            reached, matched = f"not reached in {row['lloyd_iterations']}", f"> {row['training_ratio']:.2f}×"
        else:
            reached, matched = str(row["crossing_updates"]), f"{row['matched_training_ratio']:.2f}×"
        lines.append(f"| {common.label(row['dataset'])} | {row['lloyd_training_seconds']:.3f} | "
                     f"{row['lloyd_iteration_seconds_median']:.2f} [{row['lloyd_iteration_seconds_min']:.2f}, "
                     f"{row['lloyd_iteration_seconds_max']:.2f}] | {row['emeans_training_seconds']:.3f} | "
                     f"{row['training_ratio']:.2f}× | {reached} | {matched} |")
    lines += ["", "Training time excludes the final label pass. Every Lloyd iteration does the same work, so the spread of its "
              "per-iteration times shows thermal drift or other load during the fit. The crossing is the "
              + protocol["derived_metrics"]["inertia_crossing"][0].lower() + protocol["derived_metrics"]["inertia_crossing"][1:]
              + " Where Lloyd never reaches it, the full training ratio is a lower bound.", "",
              "## Quality", "",
              "| Dataset | Lloyd test MSE | E-means test MSE | Lloyd train-sample MSE | E-means train-sample MSE | "
              "E-means training inertia vs Lloyd | Fit footprint GiB L / E |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {common.label(row['dataset'])} | {row['lloyd_test_mse']:.8f} | {row['emeans_test_mse']:.8f} | "
                     f"{row['lloyd_train_sample_mse']:.8f} | {row['emeans_train_sample_mse']:.8f} | {row['inertia_change_pct']:+.3f}% | "
                     f"{row['lloyd_peak_footprint_bytes']/2**30:.3f} / {row['emeans_peak_footprint_bytes']/2**30:.3f} |")
    lines += ["", "Test and train-sample MSE use shared FP32 Faiss assignment and direct FP64 residuals on the test set and a fixed "
              "10,000-row training sample. The inertia column compares each method's own FP32 final inertia over all training rows; "
              "a negative change means e-means fits the training set more tightly.", ""]
    if comparison and comparison["rows"]:
        lines += comparison_markdown(comparison["rows"], comparison["earlier"], comparison["replicate"], link)
    lines += ["## Protocol and limits", "", protocol["budget"], "", protocol["inherits"], ""]
    lines += [f"- {item}" for item in protocol["limitations"]]
    lines += ["", "## Evidence", "",
              f"{link(protocol_path, 'Addendum protocol')}; {link(REPO/protocol['applies_to']['path'], 'frozen protocol')}; "
              f"{link(folder/'validation.json', 'validation and raw hashes')}; {link(folder/'summary.csv', 'paired CSV')}.", ""]
    if validation.get("provenance"):
        lines += ["Recorded machine and software:", "", "```json", json.dumps(validation["provenance"], indent=2, sort_keys=True), "```", ""]
    lines += ["| Dataset | Lloyd record | E-means record |", "|---|---|---|"]
    for row in rows:
        lines.append(f"| {common.label(row['dataset'])} | {link(row['lloyd_evidence_path'], 'JSON')} | {link(row['emeans_evidence_path'], 'JSON')} |")
    if not validation["complete"]:
        lines += ["", "**Incomplete campaign:** missing or rejected records are listed in the validation manifest. "
                  "Partial tables do not establish full five-dataset coverage."]
    return "\n".join(lines) + "\n"


def addendum_rows(folder):
    loaded, parse_errors = frozen.load_rows(folder)
    return [row for row in loaded if row.get("config", {}).get("addendum") == driver.KIND], parse_errors


def generate(folder, output, protocol_path, require_complete=False, earlier=None, replicate=None):
    folder, output, protocol_path = Path(folder).resolve(), Path(output).resolve(), Path(protocol_path).resolve()
    protocol, entries = driver.load_protocol(protocol_path)
    rows, parse_errors = addendum_rows(folder)
    pairs, validation = validate(rows, protocol, entries, common.sha256(protocol_path))
    validation["rejected"].extend(parse_errors)
    validation["complete"] &= not validation["rejected"]
    validation.update(protocol_path=str(protocol_path), protocol_sha256=common.sha256(protocol_path),
                      report_generator_sha256=common.sha256(__file__),
                      raw_evidence=[{"path": str(row["_path"]), "sha256": common.sha256(row["_path"])} for row in rows])
    if pairs:
        first = pairs[0][LLOYD]
        validation["provenance"] = {key: first[key] for key in ("hardware", "versions", "thread_environment")}
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    if require_complete and not validation["complete"]:
        raise ValueError("Incomplete or incompatible evidence; validation saved, report and CSV not written")
    summary = summarize(pairs)
    comparison = None
    if earlier is not None:
        earlier = Path(earlier).resolve()
        audit = (protocol, entries, common.sha256(protocol_path))
        earlier_pairs, earlier_validation = validate(addendum_rows(earlier)[0], *audit)
        controls = {}
        if replicate is not None:
            replicate = Path(replicate).resolve()
            controls = control_rows(addendum_rows(replicate)[0], *audit)
        comparison = {"earlier": earlier, "replicate": replicate, "rows": changes(pairs, earlier_pairs, controls)}
        (folder/"comparison.json").write_text(json.dumps(
            {"earlier": str(earlier), "earlier_accepted_records": earlier_validation["accepted_records"],
             "replicate": str(replicate) if replicate else None, "replicate_accepted_records": len(controls),
             "datasets": comparison["rows"]}, indent=2, sort_keys=True) + "\n")
    with (folder/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown(summary, validation, protocol, output, folder, protocol_path, comparison))
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--results", type=Path, default=HERE/"results/kmeans_lloyd25")
    parser.add_argument("--output", "--report", type=Path, default=REPO/"docs/k-means-lloyd25-reference-2026-11-17.md")
    parser.add_argument("--protocol", type=Path, default=HERE/"kmeans_lloyd25_protocol.json")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--earlier", type=Path, help="results folder of an earlier campaign to compare against")
    parser.add_argument("--replicate", type=Path,
                        help="e-means records measured from the earlier campaign's sources in this session")
    args = parser.parse_args()
    try:
        result = generate(args.input, args.output, args.protocol, args.require_complete, args.earlier, args.replicate)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({key: result[key] for key in ("complete", "accepted_records", "expected_records")}))


if __name__ == "__main__":
    main()
