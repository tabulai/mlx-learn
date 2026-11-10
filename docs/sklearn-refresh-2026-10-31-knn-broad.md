# KNN current main versus sklearn — October 31, 2026

Completed `2026-10-31T11:11:29.699173+00:00`. This run covers **72 dataset instances in 24 families**, 9 search/inference operations, and **15 retained calls after 5 warmups** per dataset-operation-variant. Fit-only is recorded separately and excluded from the headline 2× counts. Production source and fixture identities are recorded in the raw results.

| sklearn baseline | Medians ≥2× | Individual paired calls ≥2× | All medians ≥2×? | Every pair ≥2×? |
| --- | --- | --- | --- | --- |
| Brute, default threads | 587/648 (90.6%) | 8,726/9,720 (89.8%) | False | False |
| Auto, default threads | 602/648 (92.9%) | 8,892/9,720 (91.5%) | False | False |
| Brute, threadpool-limited | 294/648 (45.4%) | 4,750/9,720 (48.9%) | False | False |

Against the fastest median among the three measured sklearn configurations in each dataset-operation cell, **257/648 (39.7%)** reach 2×. This is a best-of-measured-configurations comparison, not one fixed sklearn setting. The panel weights each dataset-operation cell equally; percentages do not estimate real-world workload frequencies. Raw timing counts retain every baseline accuracy failure.

**8/24 families** reach 2× in every seed-operation median against all three baselines; **7/24** also reach 2× in every retained paired call. A family passes only when all its constituent cases and operations pass, without averaging away misses.

## Timing and dataset scope

The threadpool-limited worker requests `threadpool_limits(1)` and records one OpenMP thread. NumPy/SciPy use Apple Accelerate, which this threadpoolctl installation does not control; this is not a verified one-thread limit for every BLAS call. Default-thread workers retain the inherited environment.

Hardware: Apple M4 Max, 128 GiB memory; platform `macOS-26.6.2-arm64-arm-64bit-Mach-O`. Python `3.14.2`, MLX `0.32.4`, NumPy `2.5.4`, sklearn `1.9.1`. Reported OpenMP thread counts: `{"sklearn_1": [1], "sklearn_auto": [16], "sklearn_default": [16]}`.

Persistent isolated public NumPy API workers. All GPU/CPU timed calls are serial. Cold first fit/query is separate; every operation gets the stated count of warmups per variant. Variant starting order rotates by dataset and operation, and each round reverses it. Each call is synchronized and audited outside timing. fit_query includes a fresh fit and all first-query preparation. No result caching. Inputs are the same float32 arrays for all variants, with no rescaling or jitter. Real data use raw features and stratified holdouts. Synthetic inputs and real-data split seeds are preserved per fixture.

The current path uses mlxlearn automatic CPU/GPU dispatch with the public brute-search API. Every call records its execution policy, fitted storage backend and actual query backend separately. Automatic calls must select the native CPU or MLX path; sklearn fallback is rejected. The 'cold' sample is first fit/query for a fixture in a persistent worker, not process/compiler startup. Timed work includes public validation, NumPy boundaries and GPU synchronization; IPC and numerical audits are outside timing. Auto's actual fitted algorithm is recorded per call. Distinct auto/brute workers may differ from timing variation even when both select brute.

Speedup = sklearn median milliseconds / current median milliseconds. Below 1× means current is slower. Family ranges span seed-specific ratios, not ratios of independently selected timing endpoints. The paired-call count uses corresponding retained rounds. These are descriptive results, not confidence intervals.

Fixtures cover the previous eleven Gaussian shapes; Gaussian, clusters, +1e4 offsets and unequal scales at 50k × 32 and 4096 × 128; 100k × 32 Gaussian; and bundled raw Digits, Breast Cancer, Iris and Wine. All inputs are float32 Euclidean data with no rescaling or jitter. Real data use stratified holdouts. Class labels reused numerically for real-data regression provide timing fixtures, not regression-quality evidence. Large real datasets, sparse input, other metrics, float64 and other hardware are outside this run.

## All query workloads

Ranges span seeds. The final column counts all search/inference operation medians reaching 2×, in the order default brute / auto / threadpool-limited brute.

| Family | Train × features; queries; k | Current query ms | vs default brute | vs auto | vs threadpool-limited | Medians ≥2× |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 8,192 × 8; 1; 5 | 0.110–0.114 | 6.26–6.71× | 1.86–1.96× | 2.20–2.28× | 27 / 8 / 24 (each /27) |
| Gaussian 8D, k5 | 8,192 × 8; 256; 5 | 0.749–0.812 | 1.79–2.20× | 5.35–5.70× | 4.71–4.98× | 11 / 27 / 27 (each /27) |
| Gaussian 32D, k5 | 8,192 × 32; 256; 5 | 0.997–1.078 | 3.62–4.17× | 3.77–3.85× | 3.71–3.98× | 27 / 27 / 27 (each /27) |
| Gaussian 32D, k32 | 8,192 × 32; 128; 32 | 1.898–1.992 | 2.03–2.06× | 2.03–2.12× | 1.44–1.52× | 10 / 13 / 0 (each /27) |
| Gaussian 128D, k5 | 4,096 × 128; 128; 5 | 0.879–0.960 | 3.60–3.81× | 3.50–3.83× | 1.69–1.85× | 27 / 27 / 0 (each /27) |
| Gaussian 1024D, k5 | 2,048 × 1024; 32; 5 | 1.170–1.309 | 2.73–3.15× | 2.58–3.28× | 1.71–1.93× | 27 / 27 / 0 (each /27) |
| Gaussian 32D, k64 | 4,096 × 32; 128; 64 | 1.514–1.549 | 2.63–2.82× | 2.75–2.81× | 1.63–1.68× | 27 / 27 / 0 (each /27) |
| Gaussian 16D, k5 | 8,192 × 16; 256; 5 | 0.803–0.877 | 4.28–4.58× | 4.47–4.76× | 4.41–4.79× | 27 / 27 / 27 (each /27) |
| Gaussian 32D, single query | 8,192 × 32; 1; 5 | 0.171–0.193 | 15.63–16.53× | 16.56–18.17× | 1.53–1.67× | 27 / 27 / 0 (each /27) |
| Gaussian 8D, k32 | 8,192 × 8; 128; 32 | 1.839–1.907 | 0.82–1.02× | 2.19–2.27× | 1.44–1.49× | 0 / 27 / 0 (each /27) |
| Gaussian 50k × 32 (original panel) | 50,000 × 32; 256; 5 | 1.391–1.494 | 4.17–4.32× | 4.12–4.38× | 15.03–16.10× | 27 / 27 / 27 (each /27) |
| Gaussian 50k × 32 | 50,000 × 32; 256; 5 | 1.411–1.515 | 4.21–4.53× | 4.10–4.41× | 14.88–15.90× | 27 / 27 / 27 (each /27) |
| Clusters 50k × 32 | 50,000 × 32; 256; 5 | 2.728–2.742 | 2.17–2.29× | 2.14–2.26× | 8.17–8.22× | 26 / 26 / 27 (each /27) |
| Offset +1e4, 50k × 32 | 50,000 × 32; 256; 5 | 1.503–1.600 | 3.93–4.17× | 3.90–4.09× | 14.00–14.88× | 27 / 27 / 27 (each /27) |
| Unequal scales 50k × 32 | 50,000 × 32; 256; 5 | 1.770–1.867 | 3.26–3.47× | 3.27–3.49× | 12.02–12.67× | 27 / 27 / 27 (each /27) |
| Gaussian 4096 × 128 | 4,096 × 128; 128; 5 | 0.875–0.936 | 3.62–3.87× | 3.66–3.87× | 1.69–1.84× | 27 / 27 / 0 (each /27) |
| Clusters 4096 × 128 | 4,096 × 128; 128; 5 | 1.143–1.209 | 2.80–2.81× | 2.76–2.86× | 1.31–1.40× | 27 / 27 / 0 (each /27) |
| Offset +1e4, 4096 × 128 | 4,096 × 128; 128; 5 | 0.868–0.981 | 3.42–3.87× | 3.36–3.93× | 1.63–1.86× | 27 / 27 / 0 (each /27) |
| Unequal scales 4096 × 128 | 4,096 × 128; 128; 5 | 0.870–0.960 | 3.54–3.80× | 3.59–3.94× | 1.70–1.87× | 27 / 27 / 0 (each /27) |
| Gaussian 100k × 32 | 100,000 × 32; 256; 5 | 2.004–2.112 | 4.03–4.44× | 3.98–4.49× | 21.11–22.24× | 27 / 27 / 27 (each /27) |
| Digits (raw) | 1,541 × 64; 256; 5 | 0.368–0.419 | 7.08–7.29× | 7.25–7.40× | 2.51–2.85× | 27 / 27 / 27 (each /27) |
| Breast Cancer (raw) | 455 × 30; 114; 5 | 0.191–0.223 | 9.61–12.13× | 9.92–11.74× | 1.31–1.54× | 27 / 27 / 0 (each /27) |
| Iris (raw) | 120 × 4; 30; 5 | 0.093–0.101 | 6.16–7.32× | 2.16–2.21× | 1.40–1.49× | 27 / 25 / 0 (each /27) |
| Wine (raw) | 142 × 13; 36; 5 | 0.099–0.105 | 5.75–5.97× | 2.09–2.15× | 1.40–1.45× | 27 / 17 / 0 (each /27) |

## Fresh fit + query

| Family | Current fresh ms | vs default brute | vs auto | vs threadpool-limited |
| --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 0.191–0.197 | 4.33–4.61× | 8.65–8.82× | 1.75–1.82× |
| Gaussian 8D, k5 | 0.818–0.870 | 1.72–1.85× | 6.65–7.41× | 4.19–4.47× |
| Gaussian 32D, k5 | 1.108–1.163 | 3.45–3.72× | 3.55–3.58× | 3.38–3.54× |
| Gaussian 32D, k32 | 2.123–2.172 | 1.89–1.91× | 1.88–1.95× | 1.33–1.36× |
| Gaussian 128D, k5 | 1.190–1.294 | 2.75–3.02× | 2.72–2.94× | 1.36–1.47× |
| Gaussian 1024D, k5 | 1.775–1.925 | 2.11–2.23× | 2.01–2.28× | 1.35–1.48× |
| Gaussian 32D, k64 | 1.724–1.777 | 2.32–2.55× | 2.51–2.57× | 1.47–1.51× |
| Gaussian 16D, k5 | 0.955–0.972 | 3.83–4.14× | 4.05–4.24× | 3.87–3.92× |
| Gaussian 32D, single query | 0.261–0.283 | 11.09–11.47× | 11.64–12.59× | 1.37–1.53× |
| Gaussian 8D, k32 | 2.006–2.074 | 0.79–0.83× | 2.75–2.86× | 1.31–1.36× |
| Gaussian 50k × 32 (original panel) | 1.853–1.915 | 3.41–3.55× | 3.37–3.61× | 11.48–11.93× |
| Gaussian 50k × 32 | 1.820–1.956 | 3.42–3.67× | 3.41–3.56× | 11.25–12.10× |
| Clusters 50k × 32 | 2.950–3.123 | 2.09–2.21× | 2.07–2.25× | 7.03–7.47× |
| Offset +1e4, 50k × 32 | 2.054–2.139 | 3.11–3.28× | 3.11–3.23× | 10.32–10.71× |
| Unequal scales 50k × 32 | 2.236–2.250 | 2.92–3.06× | 2.89–3.01× | 9.79–9.84× |
| Gaussian 4096 × 128 | 1.234–1.302 | 2.73–3.01× | 2.67–3.15× | 1.34–1.41× |
| Clusters 4096 × 128 | 1.271–1.406 | 2.50–2.76× | 2.49–2.79× | 1.25–1.35× |
| Offset +1e4, 4096 × 128 | 1.119–1.268 | 2.87–3.18× | 2.76–3.12× | 1.38–1.53× |
| Unequal scales 4096 × 128 | 1.150–1.285 | 2.71–3.04× | 2.71–3.08× | 1.36–1.52× |
| Gaussian 100k × 32 | 2.754–2.797 | 3.33–3.41× | 3.24–3.32× | 15.53–15.77× |
| Digits (raw) | 0.519–0.531 | 5.34–5.85× | 5.54–6.21× | 2.17–2.19× |
| Breast Cancer (raw) | 0.242–0.281 | 7.96–9.81× | 8.35–9.53× | 1.35–1.40× |
| Iris (raw) | 0.116–0.129 | 5.75–7.03× | 2.55–2.78× | 1.69–1.74× |
| Wine (raw) | 0.126–0.138 | 4.87–4.99× | 2.34–2.58× | 1.61–1.70× |

## Observed automatic routing

Actual query backends, including fit-only's untimed accuracy query. Fit storage and execution policy are counted separately; no routing is inferred from timings.

| Phase | Requests | Query backends | Fit storage | Policies |
| --- | --- | --- | --- | --- |
| cold | 72 | {"cpu": 18, "mlx": 54} | {"cpu": 36, "mlx": 36} | {"auto": 72} |
| warmup | 3600 | {"cpu": 900, "mlx": 2700} | {"cpu": 1800, "mlx": 1800} | {"auto": 3600} |
| retained | 10800 | {"cpu": 2700, "mlx": 8100} | {"cpu": 5400, "mlx": 5400} | {"auto": 10800} |

Per-family and per-operation routing counts are recorded in the validated summary.

## Every operation and paired call

Operation cells report median counts, then individual paired-call counts. Fit-only excludes first-query preparation and is not a substitute for fresh fit + query.

| Operation | Brute, default threads | Auto, default threads | Brute, threadpool-limited |
| --- | --- | --- | --- |
| `fit` (excluded from headline) | 0/72; 37/1,080 | 15/72; 223/1,080 | 0/72; 33/1,080 |
| `fit_query` | 63/72; 949/1,080 | 69/72; 1019/1,080 | 30/72; 464/1,080 |
| `kneighbors` | 68/72; 987/1,080 | 69/72; 1003/1,080 | 33/72; 528/1,080 |
| `indices` | 63/72; 947/1,080 | 65/72; 971/1,080 | 33/72; 531/1,080 |
| `predict_uniform` | 64/72; 957/1,080 | 70/72; 1027/1,080 | 33/72; 554/1,080 |
| `proba_uniform` | 63/72; 942/1,080 | 68/72; 979/1,080 | 33/72; 544/1,080 |
| `regress_uniform` | 64/72; 961/1,080 | 63/72; 944/1,080 | 33/72; 530/1,080 |
| `predict_distance` | 68/72; 1000/1,080 | 69/72; 1019/1,080 | 33/72; 544/1,080 |
| `proba_distance` | 68/72; 998/1,080 | 64/72; 966/1,080 | 33/72; 526/1,080 |
| `regress_distance` | 66/72; 985/1,080 | 65/72; 964/1,080 | 33/72; 529/1,080 |

Family-level paired-call counts retain every individual threshold miss:

| Family | Brute, default threads | Auto, default threads | Brute, threadpool-limited |
| --- | --- | --- | --- |
| Gaussian 8D, single query | 405/405 | 197/405 | 328/405 |
| Gaussian 8D, k5 | 176/405 | 405/405 | 405/405 |
| Gaussian 32D, k5 | 405/405 | 405/405 | 405/405 |
| Gaussian 32D, k32 | 159/405 | 157/405 | 0/405 |
| Gaussian 128D, k5 | 404/405 | 404/405 | 57/405 |
| Gaussian 1024D, k5 | 387/405 | 382/405 | 63/405 |
| Gaussian 32D, k64 | 395/405 | 400/405 | 33/405 |
| Gaussian 16D, k5 | 405/405 | 405/405 | 405/405 |
| Gaussian 32D, single query | 405/405 | 405/405 | 11/405 |
| Gaussian 8D, k32 | 0/405 | 379/405 | 0/405 |
| Gaussian 50k × 32 (original panel) | 405/405 | 405/405 | 405/405 |
| Gaussian 50k × 32 | 405/405 | 405/405 | 405/405 |
| Clusters 50k × 32 | 330/405 | 317/405 | 405/405 |
| Offset +1e4, 50k × 32 | 405/405 | 405/405 | 405/405 |
| Unequal scales 50k × 32 | 405/405 | 405/405 | 405/405 |
| Gaussian 4096 × 128 | 403/405 | 404/405 | 57/405 |
| Clusters 4096 × 128 | 402/405 | 404/405 | 0/405 |
| Offset +1e4, 4096 × 128 | 404/405 | 404/405 | 57/405 |
| Unequal scales 4096 × 128 | 402/405 | 403/405 | 60/405 |
| Gaussian 100k × 32 | 405/405 | 405/405 | 405/405 |
| Digits (raw) | 405/405 | 405/405 | 393/405 |
| Breast Cancer (raw) | 405/405 | 405/405 | 2/405 |
| Iris (raw) | 405/405 | 318/405 | 33/405 |
| Wine (raw) | 404/405 | 268/405 | 11/405 |

## Accuracy and integrity

Exhaustive direct float64 differences on actual float32 coordinates; canonical distance/index order. Strict index and prediction mismatches remain failures. A separate query audit recognizes exact-distance-valid alternative tie choices; it never changes strict pass. Prediction mismatches on tied fixtures are marked ambiguous, not automatically accepted. Distance atol/rtol1e-6; existing prediction tolerance retained.

| Variant | Strict audit passes | Failure-kind counts | Maximum absolute errors |
| --- | --- | --- | --- |
| current | 14,472/14,472 | {} | {"distance_max_abs": 7.287521839316469e-06, "distance_max_abs_from_returned_points": 7.287521839316469e-06, "max_abs": 3.902618989082285e-08} |
| sklearn_default | 13,114/14,472 | {"canonical_prediction_mismatch": 160, "reported_distance_error": 793, "valid_exact_tie_alternative": 405} | {"distance_max_abs": 0.007307924795895815, "distance_max_abs_from_returned_points": 0.007307924795895815, "max_abs": 0.0033060949847009136} |
| sklearn_1 | 13,034/14,472 | {"canonical_prediction_mismatch": 160, "canonical_prediction_mismatch_on_boundary_ties": 80, "reported_distance_error": 793, "valid_exact_tie_alternative": 405} | {"distance_max_abs": 0.007307924795895815, "distance_max_abs_from_returned_points": 0.007307924795895815, "max_abs": 0.5999999999999996} |
| sklearn_auto | 13,114/14,472 | {"canonical_prediction_mismatch": 160, "reported_distance_error": 793, "valid_exact_tie_alternative": 405} | {"distance_max_abs": 0.007307924795895815, "distance_max_abs_from_returned_points": 0.007307924795895815, "max_abs": 0.0033060949847009136} |

Audit counts include cold, warmup and retained calls, and fit-only's untimed query audit. They count repeated audit requests rather than distinct datasets. Exact-distance-valid alternative ties are valid sklearn neighbor outputs; their canonical-index differences are kept separate from incorrect membership, ordering and distance-tolerance failures. Prediction differences on boundary ties are flagged as ambiguous and not automatically validated. The summary contains per-phase counts and all failing case/operation evidence pointers. Accuracy qualifications never remove timings from speed denominators.

Integrity checks: `{"fixture_integrity_pass": true, "harness_integrity_pass": true, "initial_worker_source_integrity_pass": true, "source_integrity_pass": true, "worker_source_integrity_pass": true}`. Current accuracy plus integrity acceptance: **True**.

## Artifacts

[Raw measurements and every audit](../benchmarks/results/sklearn-refresh-2026-10-31/knn-broad.json)

[Validated derived summary](../benchmarks/results/sklearn-refresh-2026-10-31/knn-broad-summary.json)

Production manifest: `benchmarks/reference/sklearn-refresh-2026-10-31.manifest.json`. Snapshot: `{"path": "benchmarks/reference/sklearn-refresh-2026-10-31.zip", "sha256": "9d905605703391da7ea8baf257327abf34e997280afd29435ab3ac213320b7c8"}`. The raw report records source, loaded-worker, harness and fixture hashes before and after timing.

Generated from recorded measurements. No source optimization or performance claim is inferred from a separate historical run.
