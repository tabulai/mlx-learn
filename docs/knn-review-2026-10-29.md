# kNN progress review — October 29, 2026

Follow-up: the [unequal-scale regression fix](knn-scale-regression-fix-2026-10-29.md)
adds a guarded dispatch and new verification. The measurements below describe
the earlier v4a source snapshot before that fix.

There is real progress on the original Gaussian benchmark, but the newest integer-search route is **not a uniform improvement over v3**. A controlled same-process comparison isolates a **19% query speedup on Gaussian data** and a **3.7× query slowdown on unequal-scale data**, across three held-out seeds; identical-code controls' median paired ratios remain within approximately 1.2%. Correctness checks pass. The integer dispatch needs a performance safeguard before this should be described as a general upgrade.

The review covers the October 28 work and fresh October 29 measurements. All 56 production-source hashes exactly match the archived **v4a** candidate. There is no further source change beyond that snapshot. The work is mostly uncommitted; Git history alone would miss it. This review adds benchmarks and evidence, without modifying production code.

| Frozen source | Role in this review |
|---|---|
| `knn-2026-10-28.zip` | Original implementation before the two-day kNN work |
| `knn-pre-v4-2026-10-28.zip` | v3 compact search, strongest distinct predecessor on the original target |
| `knn-v4a-2026-10-28.zip` | Identical to current; A/A control, not a distinct predecessor |

The successive changes are substantive: a shared device search core, fused distance/selection and voting, packed/cooperative loads, compact candidate screening with conservative bounds, lazy owned training storage, then integer matrix screening. The newest step replaces compact screening only for eligible batches: 32 features, 32,768–2,097,152 training rows, at least 64 queries per block, k≤8, and a supported Metal runtime. Other search paths and public fit/voting code are unchanged from v3. Both filters refine original float32 coordinates and run a complete scan for uncertified queries.

The machine is an Apple M4 Max with 128 GiB RAM, Python 3.13, MLX 0.32.2, NumPy 2.5.2 and sklearn 1.9.0. Benchmarks ran sequentially, without overlapping test or GPU benchmark jobs. The established isolated-worker protocol uses identical float32 inputs, three warmups and nine retained calls, alternating variant order, synchronized public NumPy input/output boundaries, fresh-model preparation in fit-plus-query, and accuracy checks outside every timer. sklearn default retains 16 OpenMP threads; the single-thread comparator remains separate. These are local synthetic comparisons, not a ranking against every CPU tuning or external GPU library.

Fresh results on the original **50,000 training rows × 32 features, 256 queries, k=5**, dataset seeds 51–53:

| Comparison and operation | Current median range, ms | Reference median range, ms | Reference/current ratio |
|---|---:|---:|---:|
| Original → current, query | 1.026–1.161 | 7.994–11.660 | **7.79–10.04×** |
| Original → current, fresh fit + query | 1.459–1.854 | 10.055–16.391 | **6.89–8.84×** |
| v3 → current, query | 1.361–1.388 | 1.804–1.818 | **1.30–1.33×** |
| v3 → current, fresh fit + query | 1.902–1.927 | 2.231–2.257 | **1.16–1.18×** |
| v3 → current, indices only | 1.387–1.424 | 1.737–1.828 | **1.24–1.32×** |

Ratios are computed within each dataset/run; endpoints in separate ranges need not correspond. The original and v3 comparisons produce different absolute current latencies, so their fastest values must not be combined. Over v3, query latency falls **23–25%**, and fresh fit-plus-query latency falls **14–15%** in the established interleaved protocol. All 27 target operation/dataset medians improve, all 243 paired observations favor current, and all 54 chronological half-run medians favor current. Query medians are 3.01–3.14× default sklearn in that run. Fit alone remains slower than sklearn and has no implementation change relative to v3.

Evidence: [current versus v3](../benchmarks/results/knn-review-vs-v3-2026-10-29.json), [current versus original](../benchmarks/results/knn-review-vs-original-2026-10-29.json). Each summary links the retained per-seed observations and source identities.

Continuous one-second warmups followed by 25 retained calls provide a separate throughput check. In the initial three datasets, query medians are **0.404–0.417 ms current versus 0.488–0.502 ms v3**, a **1.17–1.22×** speedup, or **15–18% less latency**. Fresh fit-plus-query is **0.744–0.764 versus 0.804–0.827 ms**, a **1.08–1.09×** speedup. These measurements do not replace the intermittent-use results.

The sustained protocol also exposes unresolved slow blocks. For dataset 52, current uniform regression takes 1.194 ms versus v3's 0.510 ms. Repeating that dataset restores current regression to 0.424 ms versus 0.511 ms, but current uniform probabilities then take 1.216 ms versus 0.514 ms; the v3 query also moves to a slow 1.458 ms block. The slow blocks move between operations and variants. Their cause is not established, and the original observations are retained. The repeat's apparent 3.55× query gain must not be attributed to the algorithm. Uniformly improved prediction latency and tail behavior remain unproven.

Evidence: [sustained run](../benchmarks/results/knn-review-warmed-2026-10-29.json), [dataset-52 repeat](../benchmarks/results/knn-review-warmed-repeat-2026-10-29.json).

The identical-source controls materially limit interpretation of the broad benchmark panel. A/A target query ratios are 1.03–1.07×, while target operation medians overall range from 0.955–1.206× despite identical code. On the eleven-shape A/A panel, an unchanged 8,192×32 query reports an apparent **1.36×** gain, and the 1,024-feature case reports **1.28×**. The corresponding v3 comparison reports 1.39× and 1.27×, respectively. Those broad-panel differences are therefore not evidence of code progress. Only the 50,000×32 case in that panel activates the new integer path. Small single-query cases are still slower than single-thread sklearn: 0.313 versus 0.251 ms at eight features and 0.331 versus 0.252 ms at 32 features.

Evidence: [A/A target](../benchmarks/results/knn-review-aa-2026-10-29.json), [A/A panel](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-d0bd8c81819b6000c210), [v3 full panel](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-41a24685e6e827a84bc7). A/A diagnoses variability; its ratios are not used to “correct” other ratios or derive confidence bounds.

The strict claim that every call is at least twice as fast as default sklearn fails. The first dedicated v3 comparison passes 243/243 retained pairs, but the identical-source target run has a fresh-fit miss: **1.921 ms current versus 3.717 ms sklearn, or 1.935×**. This independently reproduces the limitation in the historical v4a combined audit, which passed 566/567 pairs with a minimum of 1.915×. Median gains do not imply a per-call guarantee.

Held-out distribution checks use seed 151, the same 50,000×32/256/k=5 shape, and an exhaustive direct-float64 oracle over every training point. The cases include Gaussian data, 16 separated clusters, a common offset of 10,000, and Gaussian features scaled by `logspace(-2, 2, 32)`. All inputs are rounded to float32 before defining the oracle. Every public call passes exact-index comparison and the unchanged distance tolerance.

| Distribution | Current query, ms | v3 query, ms | Default sklearn, ms | Integer / compact certified queries |
|---|---:|---:|---:|---:|
| Gaussian | 1.400 | 1.760 | 4.959 | 256 / 256 |
| Separated clusters | 3.140 | 3.419 | 4.038 | 0 / 0 |
| Common offset 10,000 | 3.188 | 3.349 | 4.479 | 0 / 0 |
| Unequal feature scales | **7.170** | **1.733** | 4.231 | **217 / 256** |

The last row is a performance regression, not a correctness failure. v3 certifies every query, while the integer filter must fall back on 39. An isolated rerun on the same data gives **3.153 versus 0.821 ms**, making current **3.84× slower**. A new seed 152 gives **3.151 versus 1.151 ms**, or **2.74× slower**, with 47 integer fallbacks and zero compact fallbacks. Fresh fit-plus-query also regresses by **2.33–2.49×** in those two repeats. Absolute times vary, but the direction and large magnitude of the regression repeat. Both algorithms return bit-identical indices and squared distances in the untimed diagnostics. The compact diagnostic's loaded source hash is verified against the frozen v3 archive.

Separated clusters and the large offset force both filters to fall back on every query, greatly reducing the benefit over sklearn. Clustered fresh fit-plus-query is 7.467 ms versus sklearn's 4.543 ms. The favorable Gaussian benchmark does not capture these behaviors.

Evidence: [distribution panel and certificates](../benchmarks/results/knn-review-distributions-2026-10-29.json), [isolated repeat, seed 151](../benchmarks/results/knn-review-anisotropic-repeat-seed151-2026-10-29.json), [new seed 152](../benchmarks/results/knn-review-anisotropic-repeat-seed152-2026-10-29.json).

A final same-process diagnostic isolates the dispatch change from differences between GPU worker contexts. All public validation, ownership and NumPy boundaries remain in place. Compact calls temporarily disable only integer-search eligibility, with restoration after every call; the compact source is verified against frozen v3. Each variant receives a one-second warmup, followed by 31 alternating paired observations. Separate fitted models and fresh fits preserve their respective operation boundaries. Seeds 151, 152 and 153 cover both Gaussian and unequal-scale inputs. A second pair of integer models supplies an A/A control in each case.

| Same-process workload | Integer query, ms | Compact query, ms | Median paired effect |
|---|---:|---:|---|
| Gaussian | 0.399–0.402 | 0.474–0.481 | Integer **1.190–1.193× faster** |
| Unequal feature scales | 1.820–1.825 | 0.488–0.492 | Integer **3.72–3.75× slower** |

Gaussian fresh fit-plus-query improves by 1.085–1.104×; unequal-scale fresh fit-plus-query regresses by 2.60–2.74×. The six A/A median paired ratios range only from **0.991 to 1.011**. All accuracy, route, restoration and source-integrity checks pass. This supports an algorithmic gain on Gaussian data and an algorithmic regression on unequal-scale data, independently of the noisy multiprocess panel. It remains a supplemental diagnostic, not a replacement for the established isolated-worker benchmark. See the [complete same-process report](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-f47fafab8498a214519a) and [reproducible harness](../benchmarks/knn_review_dispatch.py).

Correctness validation: **488 tests passed**, covering public parity, float32 behavior, the device API, fused/paired/compact/integer kernels, numerical bounds, voting, ownership/refits, runtime fallback, and benchmark gates. See the [test log](../benchmarks/results/knn-review-tests-2026-10-29.log). All fresh implementation audits and source-integrity checks pass. The full-panel aggregate quality flag remains false because both sklearn comparators reproduce the already-recorded copied-row distance residual at 1,024 features; current, v3 and resident implementations pass. Tolerances were not relaxed. The general high-dimensional Gram candidate-selection limitation remains outside the new certified path; passing these tests does not prove universal agreement with real arithmetic.

The practical assessment is to retain the genuine device-pipeline and Gaussian-query progress, while treating the integer route as conditional. Improve its dispatch or fallback strategy for unequal feature scales, add this regression workload to performance acceptance, and resolve the moving slow blocks before claiming uniform inference or tail-latency gains. README/CHANGELOG still describe historical v3 results rather than a completed v4 evaluation.

The added [distribution harness](../benchmarks/knn_review_distributions.py) preserves fixtures' hashes, raw timings, correctness checks, source provenance and untimed certificates. Existing benchmark protocols were left unchanged. Exact measurement-time harness sources are preserved in a [source archive](../benchmarks/reference/knn-review-harnesses-2026-10-29.zip) with a [SHA-256 manifest](../benchmarks/reference/knn-review-harnesses-2026-10-29.manifest.json), before import-order cleanup and explicit binding of synchronous loop closures. The cleaned harnesses pass Ruff and a separate correctness/dispatch smoke run; its timings are not used above. Commands and exit statuses for the main and follow-up batches are in the [execution log](../benchmarks/results/knn-review-execution-2026-10-29.json) and [follow-up execution log](../benchmarks/results/knn-review-followup-execution-2026-10-29.json). The A/A target command deliberately returns 1 when its strict 2× pair gate fails; this is not a numerical failure.

Representative reproduction commands:

```sh
python3 -m benchmarks.knn_acceleration \
  --reference-archive benchmarks/reference/knn-pre-v4-2026-10-28.zip \
  --output benchmarks/results/knn-review-vs-v3-2026-10-29.json
python3 -m benchmarks.knn_warmed \
  --reference-archive benchmarks/reference/knn-pre-v4-2026-10-28.zip \
  --output benchmarks/results/knn-review-warmed-2026-10-29.json
python3 -m benchmarks.knn_pipeline --profile full --resident --repeats 9 --warmup 3 \
  --reference-archive benchmarks/reference/knn-v4a-2026-10-28.zip \
  --output benchmarks/results/knn-review-aa-panel-2026-10-29.json
python3 -m benchmarks.knn_review_distributions \
  --output benchmarks/results/knn-review-distributions-2026-10-29.json
python3 -m benchmarks.knn_review_dispatch \
  --output benchmarks/results/knn-review-dispatch-2026-10-29.json
```
