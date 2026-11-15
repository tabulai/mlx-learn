# Telemetry overlap audit

The two telemetry collections and two initial probes overlap five recorded fit windows, spanning **four dataset/seed pairs**. Rerunning both modes for CodeSearchNet seeds 10, 11 and 12 and GooAQ seed 11 replaces eight records. No benchmark record was changed by this audit; all 102 input hashes were rechecked unchanged.

| Monitor invocation | UTC invocation envelope | Overlapping timed-fit memory windows |
| --- | --- | --- |
| Initial probe 1 | 22:00:11.911–22:00:15.353 | None |
| Initial probe 2 | 22:00:22.985–22:00:26.610 | CodeSearchNet, seed 10, previous: approximately 3.585 s |
| First collection, 30 samples | 22:01:04.654693–22:01:45.942556 | CodeSearchNet, previous, seed 11: 8.262 s; seed 12: 8.287 s |
| Anchored 30-second collection | 22:03:35.554045–22:04:05.558630 | GooAQ, seed 11, optimized: 10.657 s; previous: 4.274 s |

All timestamps are on 2026-11-14. The first collection's filename contains `30s`, but its recorded invocation lasted 41.29 seconds. The initial probes' wrapper brackets were recovered from local command execution events; only their timing metadata and event hashes were retained in [telemetry-probe-brackets.json](telemetry-probe-brackets.json).

The benchmark intervals come from `memory_samples[*].baseline/end.monotonic_ns`, expanded by half each observation's syscall duration. These enclose the fit handshake and are slightly wider than the inner timer. The anchored collection uses the same monotonic clock directly. UTC-only intervals use the observed wall-minus-monotonic offset envelope from 44 anchors in the later collection; its width is 0.376 ms. Extrapolating it earlier assumes no wall-clock step and does not prove a bound on earlier drift. The millisecond probe timestamps are expanded by 1 ms. Adding a further one-second guard to every monitor envelope adds CodeSearchNet seed 12 optimized but **adds no dataset/seed pairs** to the rerun set.

These are overlaps with invocation envelopes, including monitor startup and shutdown. They do not establish measurable interference or continuous monitoring CPU activity. The first probe may overlap an unrecorded warmup, although it overlaps no recorded timed fit; CodeSearchNet seed 10 is already included because of the second probe. This audit does not bound lingering thermal or cache effects and makes no thermal-cause claim.

The [machine-readable audit](telemetry-overlap.json) preserves all 102 input hashes, individual fit timestamps, overlap calculations, the eight-record rerun list, evidence hashes and uncertainty statements. [telemetry_overlap.py](telemetry_overlap.py) performs this CPU-only join. Paths in its input manifest refer to the campaign location at audit time; hashes identify the same records if subsequently archived. The script and audit should be preserved before replacement records are generated.
