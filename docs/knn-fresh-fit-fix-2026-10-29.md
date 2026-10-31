# kNN fresh fit-plus-query regression eliminated — October 29, 2026

The remaining unequal-feature-scale fresh fit-plus-query slowdown is eliminated on all six controlled datasets. Relative to frozen v3, latency is now **7.7–10.3% lower**, replacing the previous 2.1–2.9% overhead. Gaussian query gains remain: **1.18–1.23×** speedup, with Gaussian fresh fit-plus-query **1.21–1.24×** faster. These are measured results on the workload below, not guarantees for other hardware or distributions.

The only production change since the [initial scale guard](knn-scale-regression-fix-2026-10-29.md) is in [`_prepare_neighbors_input`](../src/mlxlearn/neighbors/_base.py). Plain native float32 NumPy arrays containing at least 262,144 elements now use `np.isfinite(X).all()` before sklearn validation. Every value is checked. For valid inputs, sklearn skips its redundant sum-based finiteness scan; invalid inputs retain its normal validation/error path. Shape, feature-count, target and classification checks still run. The input remains caller-owned and fit still makes its independent copy. The check temporarily allocates one boolean per element. Smaller arrays, subclasses, non-native byte order and wide-float conversion retain their existing paths.

The cached scale guard and every search kernel are unchanged. Unequal-scale data continues to use compact screening; Gaussian data continues to use integer screening. Original coordinates, distance arithmetic, certificates and exact fallbacks are untouched. Only `mlxlearn/neighbors/_base.py` differs from the previous guard archive. The final 56-source snapshot is preserved in [this archive](../benchmarks/reference/knn-fresh-validation-2026-10-29.zip), with its [SHA-256 manifest](../benchmarks/reference/knn-fresh-validation-2026-10-29.manifest.json).

The main comparison uses 50,000 training rows, 32 features, 256 queries, k=5 and float32 inputs. Unequal scales are `logspace(-2, 2, 32)`. Dataset seeds are 151–153 and 251–253, matching the earlier guard evaluation. Each variant receives one second of warmup; 31 retained pairs alternate execution order, with starting order rotating across seeds. Timings include the public NumPy interface, GPU synchronization, and all fresh preparation for fit-plus-query. Every output is audited outside timing against exhaustive direct float64 distances on the original float32 coordinates, with exact indices and unchanged distance tolerances.

| Workload and operation | Current median range, ms | v3 median range, ms | Median paired speedup, v3/current |
|---|---:|---:|---:|
| Unequal-scale fresh fit + query | 0.720–0.779 | 0.798–0.856 | **1.084–1.114×** |
| Gaussian fresh fit + query | 0.656–0.918 | 0.793–1.167 | **1.210–1.242×** |
| Gaussian queries | 0.411–0.501 | 0.489–0.628 | **1.178–1.232×** |
| Unequal-scale queries | 0.484–0.504 | 0.478–0.504 | **0.988–1.010×** |

Each range spans the six seeds; speedups are medians of paired ratios, not ratios of the displayed range endpoints. Median paired unequal-scale query latency ratios remain within 1.2% of v3. Identical-code A/A median paired ratios span 0.982–1.009 for fresh fit-plus-query and 0.995–1.009 for queries. On unequal-scale data specifically, fresh-fit A/A spans 0.998–1.009, much smaller than the measured improvement. These controls describe observed variability, not confidence bounds or per-call guarantees. All warmup and retained output, route, backend and hook-restoration audits pass.

The [same-process harness](../benchmarks/knn_fresh_dispatch.py) loads verified archived v3 public validation methods, device query orchestration and compact builders. Crucially, v3 fit and query methods retain their archived helper globals: v3 does **not** receive the new finiteness check. All other archived dependencies must match current source hashes. The diagnostic shares the current index constructor's unused cache fields; the separate isolated-worker runs below compare complete packages. Sources, archived files, harness hashes and restored hooks all pass integrity checks.

Evidence: [six-seed controlled results](../benchmarks/results/knn-fresh-dispatch-vs-v3-2026-10-29.json), [compact numerical summary](../benchmarks/results/knn-fresh-summary-2026-10-29.json), and [frozen harnesses/tests](../benchmarks/reference/knn-fresh-harnesses-2026-10-29.zip). v3 is the unchanged `knn-pre-v4-2026-10-28.zip` reference.

The original isolated-worker benchmark was rerun on dataset seeds 51–53 with three warmups, nine retained rounds, both sklearn thread configurations, native resident search, and all nine search/inference operations. Fresh fit-plus-query improves **1.21–1.45×** versus v3; Gaussian queries improve **1.33–1.51×**. All implementation accuracy and source-integrity checks pass. All **243/243** search/inference pairs also exceed the original 2× default-sklearn target in this run; the minimum is **2.179×**. Fit-only timing is reported separately and is not part of that gate. This successful run does not establish a universal per-call guarantee. See the [complete results](../benchmarks/results/knn-fresh-original-vs-v3-2026-10-29.json).

The four-distribution isolated panel was repeated on seeds 151, 251 and 252. Unequal-scale fresh fit-plus-query medians improve in all three: **1.329 versus 1.687 ms**, **1.530 versus 1.875 ms**, and **1.881 versus 2.184 ms**. Current and v3 pass every Gaussian and unequal-scale implementation audit. Separate-process query timings vary more than the controlled experiment; unchanged compact-kernel timing differences are not attributed to this validation change. Results: [seed 151](../benchmarks/results/knn-fresh-distributions-seed151-2026-10-29.json), [seed 251](../benchmarks/results/knn-fresh-distributions-seed251-2026-10-29.json), and [seed 252](../benchmarks/results/knn-fresh-distributions-seed252-2026-10-29.json).

The broader audits retain the previously documented limitations. On seed 251's offset-10,000 fixture, current and v3 both disagree with the float64 oracle's ordering for a pair tied after float32 rounding; the earlier [bitwise diagnostic](../benchmarks/results/knn-scale-fix-offset-audit-2026-10-29.json) explains that case. sklearn also retains its unequal-scale copied-row distance residual of 5.048e-6. The seed-251 run therefore exits 1. The [eleven-shape panel](../benchmarks/results/knn-fresh-panel-vs-v3-2026-10-29.json) passes current, v3 and resident audits, while its aggregate quality flag retains sklearn's 1,024-feature copied-row residual of 2.473e-6. No tolerance or failed sample was removed.

Validation totals **598 passed tests**: the [513-test kNN suite](../benchmarks/results/knn-fresh-core-tests-2026-10-29.log) and [85 validation/patching/neighbor-compliance checks](../benchmarks/results/knn-fresh-validation-tests-final-2026-10-29.log), including 25 [new cases](../tests/unit/test_neighbors_float32_validation.py). These cover the threshold, native dtype and subclasses, finite extremes, NaN/Inf at the tail, layouts, read-only views, storage ownership, invalid targets and structural validation. An initial ownership-test fixture used a large monotone matrix that exposes a pre-existing CPU search mismatch ([before/after diagnostic](../benchmarks/results/knn-fresh-ownership-fixture-audit-2026-10-29.json)); the final fixture uses seeded moderate-scale data and retains the exact ownership and query assertions. The initial failed test log remains in the execution record. Ruff passes for the production change, new tests and new harness.

All performance workloads ran sequentially on Apple M4 Max, Python 3.13, MLX 0.32.2, NumPy 2.5.2 and sklearn 1.9.0. Production source hashes match the frozen final archive throughout acceptance runs. The [execution record](../benchmarks/results/knn-fresh-execution-2026-10-29.json) retains commands, elapsed times and exit statuses. The exploratory [fit-only finiteness probe](../benchmarks/results/knn-fresh-finite-probe-2026-10-29.json) identified the saving; the end-to-end results above determine acceptance.

Reproduction:

```sh
python3 -m benchmarks.knn_fresh_dispatch \
  --output benchmarks/results/knn-fresh-dispatch-vs-v3-2026-10-29.json
python3 -m benchmarks.knn_acceleration \
  --reference-archive benchmarks/reference/knn-pre-v4-2026-10-28.zip \
  --output benchmarks/results/knn-fresh-original-vs-v3-2026-10-29.json
python3 -m benchmarks.knn_review_distributions --seed 251 \
  --output benchmarks/results/knn-fresh-distributions-seed251-2026-10-29.json
```

The later [precision and ordering fix](knn-precision-fix-2026-10-30.md) addresses
the offset ordering and monotone copied-row failures recorded above, with a new
full v3 comparison. The measurements on this page remain the historical result.
