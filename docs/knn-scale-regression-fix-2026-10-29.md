# Unequal-feature-scale kNN regression fix — October 29, 2026

Follow-up: the remaining fresh-fit overhead reported here is addressed by the
[input-validation optimization and new v3 comparisons](knn-fresh-fit-fix-2026-10-29.md).
This report preserves the measurements of the initial guard implementation.

The large unequal-scale query regression is fixed on the reproduced workload and three additional held-out seeds. Gaussian queries retain their integer-search advantage. Across six datasets in the controlled same-process benchmark, unequal-scale query latency is within **1.1% of v3**, while Gaussian queries remain **1.18–1.20× faster**. The earlier integer route was approximately 3.7× slower on unequal-scale data. All 513 kNN tests pass.

The fix adds a cached routing guard in [`_neighbors.py`](../src/mlxlearn_device/_neighbors.py). Once an index passes the existing integer shape and runtime gates, the guard reads 16 evenly spaced rows of its owned training storage through the MLX buffer interface. Eight pair differences estimate how concentrated the feature variation is. A participation ratio below 16 retains the existing compact filter; ordinary Gaussian data keeps integer screening. Pair differences remove sensitivity to a common coordinate offset, and Python double arithmetic avoids diagnostic overflow/underflow for finite float32 inputs.

Only the routing decision is cached. Input coordinates, the requested Euclidean metric, original float32 distance refinement, tie ordering, numerical certificates, and exact fallback kernels are unchanged. Buffer access respects strides, releases its view, and conservatively retains compact search if the interface is unavailable. A new fit starts with an unset decision. Explicit small blocks, small query batches and other ineligible shapes never perform the host sample read.

This is a bounded-cost performance heuristic, not an additional correctness certificate. Sixteen sampled rows can miss heterogeneous data or depend on training order. It does not promise the fastest route for every distribution. The selected filter must independently certify its answer or use its existing complete scan.

The final production source is frozen in [this archive](../benchmarks/reference/knn-scale-guard-2026-10-29.zip), with a [SHA-256 manifest](../benchmarks/reference/knn-scale-guard-2026-10-29.manifest.json). Relative to v4a, only `mlxlearn_device/_neighbors.py` changes. The compact and integer numerical kernels are byte-identical to the pre-fix versions. v3 is the frozen `knn-pre-v4-2026-10-28.zip` implementation.

The main controlled comparison uses 50,000 training rows, 32 features, 256 queries, k=5, float32 inputs, one-second warmups and 31 alternating paired observations per operation. It retains public NumPy validation/output boundaries and fresh preparation for fit-plus-query. Seeds 151–153 reproduce the earlier investigation; seeds 251–253 are additional held-out cases. Unequal feature scales use `logspace(-2, 2, 32)`. Every Gaussian production call uses integer search; every unequal-scale production call uses compact search. Forced compact calls provide a same-process v3-kernel comparison, with frozen-source verification. Independent frozen-v3 workers are also rerun below.

| Workload and operation | Fixed current median range, ms | v3 compact median range, ms | Median paired effect |
|---|---:|---:|---|
| Gaussian queries | 0.395–0.423 | 0.476–0.500 | **1.182–1.200× faster** |
| Unequal-scale queries | 0.496–0.505 | 0.495–0.503 | **Within 1.1% of v3** |
| Gaussian fresh fit + query | 0.754–0.776 | 0.800–0.829 | **1.061–1.070× faster** |
| Unequal-scale fresh fit + query | 0.837–0.847 | 0.816–0.827 | **2.1–2.9% slower** |

The residual fresh-fit cost is disclosed: the guard adds a small initial inspection instead of the earlier multi-fold regression. A dedicated same-process control that disables only the guard measures **12–21 microseconds** of median paired fresh-fit overhead. Gaussian median paired query ratios differ from parity by at most **0.5%** with the guard disabled; subsequent calls reuse the decision. The Gaussian fresh-fit advantage over v3 remains positive, although slightly smaller than before the fix. In the main six-seed test, identical-code A/A median paired ratios range from **0.991 to 1.006**. Those are median ratios, not bounds on individual calls.

Evidence: [six-seed controlled benchmark](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-f60fadfb8a3663f45863), [guard-only cost control](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-220f722ab375637f4b17), and the [pre-fix evaluation](knn-review-2026-10-29.md). The new [adaptive-dispatch harness](../benchmarks/knn_scale_dispatch.py) labels the production path as current and checks its selected route, rather than requiring integer execution on every dataset. The [guard control](../benchmarks/knn_scale_guard_control.py) restores its temporary override after every call and checks restoration and source integrity.

The unchanged original isolated-worker benchmark was rerun with three warmups, nine retained observations, default 16-thread sklearn, single-thread sklearn, native resident search, and all nine search/inference operations on dataset seeds 51–53. Against frozen v3, query medians improve **1.29–1.38×**, fresh fit-plus-query improves **1.12–1.17×**, and indices-only queries improve **1.31–1.33×**. All implementation accuracy and source-integrity checks pass. The same original workload was run against frozen unguarded v4a: query median ratios range from **0.999 to 1.134×**, consistent with preserved query performance but substantial multiprocess variability. Fresh-fit ratios range from 0.922 to 1.046×; the dedicated control above isolates the actual guard cost more reliably.

The previous strict per-call 2×-sklearn gate still is not a general guarantee. The v3 comparison retains one fresh-fit miss at **1.970×**, passing 242/243 target pairs. The v4a comparison passes 243/243. This fix targets the unequal-scale regression and preservation of Gaussian gains; neither exit code nor missed samples are hidden.

Evidence: [original benchmark versus v3](../benchmarks/results/knn-scale-fix-vs-v3-2026-10-29.json), [original benchmark versus v4a](../benchmarks/results/knn-scale-fix-vs-v4a-2026-10-29.json), and the [eleven-shape panel versus v3](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-ccf3ebda7ca5c4e1b971). The latter passes current/reference/resident audits; its strict aggregate quality flag retains the known sklearn copied-row residual at 1,024 features. Unchanged paths' timing differences are not attributed to this guard.

The four-distribution isolated-worker panel was repeated for seeds 151, 251 and 252: Gaussian, separated clusters, common offset 10,000, and unequal feature scales. All Gaussian and unequal-scale current/v3 implementation audits pass. Unequal-scale current query medians are **1.478 versus 1.718 ms**, **1.482 versus 1.480 ms**, and **1.021 versus 1.076 ms**, respectively. The same-process experiment is the stronger basis for estimating a small difference from v3 because it removes the large context-dependent variation observed in the prior review.

The seed-251 panel also retains two audit limitations outside the fixed regression: current and frozen v3 both disagree with direct-float64 index ordering on the large-offset fixture; their original float32 scan and both filters agree bit-for-bit. sklearn's unequal-scale copied-row distance residual also exceeds the existing absolute tolerance while its indices match. These failures remain visible in the JSON and command exit code; no tolerance was relaxed and no failing dataset was discarded. The [offset diagnostic](../benchmarks/results/knn-scale-fix-offset-audit-2026-10-29.json) records the discrepant positions and squared distances. These are separate from the successful original-coordinate unequal-scale audits.

Evidence: [distribution seed 151](../benchmarks/results/knn-scale-fix-distributions-seed151-2026-10-29.json), [seed 251, including failures](../benchmarks/results/knn-scale-fix-distributions-seed251-2026-10-29.json), and [seed 252](../benchmarks/results/knn-scale-fix-distributions-seed252-2026-10-29.json).

Validation includes **513 passed tests**, with 25 new cases covering executed integer/compact routes, k=1/5/8, exact original-coordinate neighbors, global scaling, feature permutations, common offsets, strided and lazy buffers, zero/constant/outlier data, input ownership, cached decisions, changing queries, outstanding graphs, block tails, refits in both directions and buffer-interface fallback. See the [new tests](../tests/unit/test_neighbors_scale_dispatch.py) and [complete test log](../benchmarks/results/knn-scale-fix-tests-2026-10-29.log). Ruff passes for the production change, new tests and new benchmark harnesses. No timing threshold is imposed on ordinary correctness tests.

Hardware/software remain Apple M4 Max, 128 GiB RAM, Python 3.13, MLX 0.32.2, NumPy 2.5.2 and sklearn 1.9.0. Performance workloads and tests ran sequentially. Source hashes stayed unchanged across acceptance runs. The [execution manifest](../benchmarks/results/knn-scale-fix-execution-2026-10-29.json) retains exact commands, elapsed times and exit statuses. An exploratory increase in integer candidate capacity was rejected: even a carry of 32 left uncertified unequal-scale queries and expensive scans; its [raw probe](../benchmarks/results/knn-scale-fix-carry-probe-2026-10-29.json) remains separate from acceptance evidence.

Representative reproduction commands:

```sh
python3 -m benchmarks.knn_scale_dispatch \
  --seed 151 --seed 152 --seed 153 --seed 251 --seed 252 --seed 253 \
  --output benchmarks/results/knn-scale-fix-dispatch-2026-10-29.json
python3 -m benchmarks.knn_acceleration \
  --reference-archive benchmarks/reference/knn-pre-v4-2026-10-28.zip \
  --output benchmarks/results/knn-scale-fix-vs-v3-2026-10-29.json
python3 -m benchmarks.knn_scale_guard_control \
  --output benchmarks/results/knn-scale-guard-control-2026-10-29.json
python3 -m benchmarks.knn_review_distributions --seed 251 \
  --output benchmarks/results/knn-scale-fix-distributions-seed251-2026-10-29.json
```
