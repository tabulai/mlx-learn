"""CPU-only join of saved telemetry invocation and benchmark memory windows."""

import datetime as dt
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE.parent / "gap_protocol"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_ns(value):
    stamp = dt.datetime.fromisoformat(value)
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    delta = stamp - epoch
    return (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000


def iso(value):
    return dt.datetime.fromtimestamp(value / 1e9, dt.timezone.utc).isoformat()


def main():
    anchored = json.loads((HERE / "gpu-telemetry-window.json").read_text())
    unanchored = json.loads((HERE / "gpu-telemetry-30s.json").read_text())
    probes = json.loads((HERE / "telemetry-probe-brackets.json").read_text())
    anchors = [anchored["start"], anchored["end"]]
    anchors += [record[key] for record in anchored["records"] for key in ("read_started", "received")]
    # Each wall read is enclosed by two monotonic reads. Preserve the envelope.
    offset_min = min(a["wall_time_ns"] - a["monotonic_bracket_ns"][1] for a in anchors)
    offset_max = max(a["wall_time_ns"] - a["monotonic_bracket_ns"][0] for a in anchors)
    offset_mid = (offset_min + offset_max) // 2

    def wall_window(identifier, start, end, source, certainty):
        return {
            "id": identifier,
            "wall_start_utc": iso(start),
            "wall_end_utc": iso(end),
            "monotonic_outer_bracket_ns": [start - offset_max, end - offset_min],
            "source": source,
            "certainty": certainty,
            "wall_to_monotonic_extrapolation": True,
        }

    windows = []
    for probe in probes["probes"]:
        windows.append(wall_window(
            probe["id"],
            probe["started_at_wall_ms"] * 10**6 - 10**6,
            probe["completed_at_wall_ms"] * 10**6 + 10**6,
            "telemetry-probe-brackets.json",
            "Recorded execution wrapper bracket; millisecond precision expanded by 1 ms. "
            "Monotonic conversion assumes no intervening wall-clock step.",
        ))
    windows.append(wall_window(
        "collection_41s",
        utc_ns(unanchored["started_at_utc"]),
        utc_ns(unanchored["ended_at_utc"]),
        "gpu-telemetry-30s.json",
        "Recorded UTC wrapper interval (30 samples took 41.29 s); monotonic conversion "
        "assumes no intervening wall-clock step. Filename does not denote actual duration.",
    ))
    windows.append({
        "id": "collection_30s",
        "wall_start_utc": anchored["start"]["wall_utc"],
        "wall_end_utc": anchored["end"]["wall_utc"],
        "monotonic_outer_bracket_ns": [
            anchored["start"]["monotonic_bracket_ns"][0],
            anchored["end"]["monotonic_bracket_ns"][1],
        ],
        "source": "gpu-telemetry-window.json",
        "certainty": "Direct same-host monotonic anchors enclosing monitor invocation; no wall conversion needed.",
        "wall_to_monotonic_extrapolation": False,
    })

    records, fits = [], []
    ignored = []
    for path in sorted(RESULTS.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("kind") != "metal_gap_paired":
            ignored.append(path.name)
            continue
        config = record["config"]
        identity = {key: config[key] for key in ("dataset", "seed", "mode")}
        identity.update(path=str(path.relative_to(ROOT)), sha256=digest(path))
        records.append(identity)
        for index, memory in enumerate(record["memory_samples"]):
            baseline, end = memory["baseline"], memory["end"]
            # Memory observation timestamps are syscall midpoints, not starts.
            start_ns = baseline["monotonic_ns"] - (baseline["collection_duration_ns"] + 1) // 2
            end_ns = end["monotonic_ns"] + (end["collection_duration_ns"] + 1) // 2
            fits.append({
                **identity,
                "fit_index": index,
                "inner_timed_seconds": record["train_seconds_samples"][index],
                "memory_window_monotonic_outer_bracket_ns": [start_ns, end_ns],
                "estimated_memory_window_utc": [iso(start_ns + offset_mid), iso(end_ns + offset_mid)],
            })

    def intersect(fit, window, padding=0):
        a, b = fit["memory_window_monotonic_outer_bracket_ns"]
        c, d = window["monotonic_outer_bracket_ns"]
        return max(0, min(b, d + padding) - max(a, c - padding))

    hits = []
    for window in windows:
        for fit in fits:
            overlap_ns = intersect(fit, window)
            if overlap_ns:
                hits.append({
                    **fit,
                    "telemetry_window_id": window["id"],
                    "overlap_with_invocation_envelope_seconds": overlap_ns / 1e9,
                    "certainty": window["certainty"],
                })
    affected_pairs = sorted({(hit["dataset"], hit["seed"]) for hit in hits})
    pairs = [
        {"dataset": dataset, "seed": seed,
         "overlapping_modes": sorted({h["mode"] for h in hits if (h["dataset"], h["seed"]) == (dataset, seed)}),
         "records_to_rerun_together": [r for r in records if (r["dataset"], r["seed"]) == (dataset, seed)]}
        for dataset, seed in affected_pairs
    ]
    guarded = {(fit["dataset"], fit["seed"], fit["mode"], fit["fit_index"])
               for window in windows for fit in fits if intersect(fit, window, 10**9)}
    exact = {(h["dataset"], h["seed"], h["mode"], h["fit_index"]) for h in hits}
    output = {
        "kind": "telemetry_fit_window_overlap_audit",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "record_count": len(records),
        "timed_fit_memory_window_count": len(fits),
        "ignored_non_result_files": ignored,
        "method": {
            "interval": "Half-open intervals; positive overlap only.",
            "benchmark_window": "Memory baseline and end observations enclose the FIT_READY/GO through FIT_DONE/STOP_ACK phase. "
            "Each midpoint is expanded by half its syscall duration. These are slightly wider than the inner fit timer.",
            "monitor_window": "Entire invocation wrapper envelope, including startup and shutdown, because monitoring CPU overhead "
            "can occur outside IOReport counter-sample intervals. Overlap does not prove measurable slowdown or continuous CPU activity.",
            "wall_mapping": "UTC-only windows are mapped using the full observed wall-minus-monotonic offset envelope from "
            "44 bracketed anchors in the later 30 s trace. Applying that envelope earlier is an extrapolation assuming no wall-clock step; "
            "it is not a formally measured bound on earlier clock drift. A separate ±1 s sensitivity check is included.",
            "limitations": "Only directly overlapping timed-fit windows are identified. This does not bound lingering thermal/cache "
            "effects or monitor overlap with unrecorded warmup intervals. No thermal or timing-causality claim is made.",
        },
        "wall_mapping": {
            "anchor_count": len(anchors),
            "wall_minus_monotonic_offset_ns": [offset_min, offset_max],
            "observed_offset_envelope_width_ns": offset_max - offset_min,
            "anchor_wall_range": [min(a["wall_utc"] for a in anchors), max(a["wall_utc"] for a in anchors)],
            "one_second_padding_additional_fit_hits": [
                dict(zip(("dataset", "seed", "mode", "fit_index"), identity, strict=True)) for identity in sorted(guarded - exact)
            ],
            "one_second_padding_additional_dataset_seed_pairs": [
                {"dataset": dataset, "seed": seed}
                for dataset, seed in sorted({identity[:2] for identity in guarded} - set(affected_pairs))
            ],
        },
        "telemetry_windows": windows,
        "overlapping_fits": hits,
        "affected_dataset_seed_pairs": pairs,
        "summary": {
            "overlapping_record_count": len({h["path"] for h in hits}),
            "overlapping_fit_count": len(exact),
            "affected_pair_count": len(pairs),
            "whole_pair_rerun_record_count": sum(len(pair["records_to_rerun_together"]) for pair in pairs),
            "probe_1_overlapping_fit_count": sum(h["telemetry_window_id"] == "probe_1" for h in hits),
            "probe_1_scope_note": "Probe 1 lies between recorded timed fits; it may overlap an unrecorded warmup. "
            "CodeSearchNet seed 10 is already in the rerun set because probe 2 overlaps its previous-runtime timed fit.",
            "raw_records_modified": False,
        },
        "evidence_sha256": {
            name: digest(HERE / name) for name in (
                "gpu-telemetry-30s.json", "gpu-telemetry-window.json", "telemetry-probe-brackets.json", "telemetry_overlap.py"
            )
        },
        "input_record_manifest": records,
    }
    (HERE / "telemetry-overlap.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))
    print("Pairs:", affected_pairs)
    print("Additional hits with ±1 s padding:", len(guarded - exact))


if __name__ == "__main__":
    main()
