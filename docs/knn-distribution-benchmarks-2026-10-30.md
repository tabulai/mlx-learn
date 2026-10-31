# Clustered and offset kNN fix: full benchmark results

Completed `2026-10-30T16:44:07.865256+00:00`. This run covers **72 dataset instances in 24 families**, 9 search/inference operations, and **15 retained calls after 5 warmups** per dataset-operation-variant. Fit-only is recorded separately and excluded from the headline 2× counts. Production source and fixture identities are recorded in the raw results.

| sklearn baseline | Medians ≥2× | Individual paired calls ≥2× | All medians ≥2×? | Every pair ≥2×? |
| --- | --- | --- | --- | --- |
| Brute, default threads | 482/648 (74.4%) | 7,288/9,720 (75.0%) | False | False |
| Auto, default threads | 524/648 (80.9%) | 7,774/9,720 (80.0%) | False | False |
| Brute, one thread | 246/648 (38.0%) | 4,083/9,720 (42.0%) | False | False |

Against the fastest median among the three measured sklearn configurations in each dataset-operation cell, **228/648 (35.2%)** reach 2×. This is a best-of-measured-configurations comparison, not one fixed sklearn setting. The panel weights each dataset-operation cell equally; percentages do not estimate real-world workload frequencies. Raw timing counts retain every baseline accuracy failure.

**8/24 families** reach 2× in every seed-operation median against all three baselines; **6/24** also reach 2× in every retained paired call. A family passes only when all its constituent cases and operations pass, without averaging away misses.

![Minimum speedup across seeds](assets/knn-distribution-final-2026-10-30.png)

## Controlled comparison with the preceding source

The frozen reference is `knn-128-final-2026-10-30.zip`. Current, an identical-code A/A worker, and reference ran serially with 31 interleaved paired rounds after 1 seconds of continuous warmup per worker/operation. These are contemporaneous comparisons on identical fixtures, not ratios of separate historical runs. Speedup is reference/current time; the A/A column uses identical-code second-worker/current time. Displayed ranges span dataset seeds. Near-parity differences should be assessed against A/A variation.

Control accuracy and integrity acceptance: **True**. Per-variant strict accuracy: `{"current": true, "current_b": true, "reference": true}`.

| Family | Operation | Seeds | Current ms | Reference ms | Median paired speedup | A/A paired ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 50k × 32 | kneighbors | 151, 251, 252 | 0.438–0.453 | 0.428–0.458 | 0.96–1.04× | 0.98–1.02× |
| Gaussian 50k × 32 | fit_query | 151, 251, 252 | 0.735–0.835 | 0.760–0.813 | 0.98–1.00× | 1.00–1.02× |
| Clusters 50k × 32 | kneighbors | 151, 251, 252 | 0.782–0.822 | 2.071–2.097 | 2.59–2.70× | 0.98–1.00× |
| Clusters 50k × 32 | fit_query | 151, 251, 252 | 1.083–1.106 | 2.281–2.330 | 2.09–2.16× | 0.96–1.07× |
| Offset +1e4, 50k × 32 | kneighbors | 151, 251, 252 | 0.462–0.463 | 2.071–2.087 | 4.39–4.44× | 0.99–1.01× |
| Offset +1e4, 50k × 32 | fit_query | 151, 251, 252 | 0.832–0.856 | 2.300–2.355 | 2.74–2.76× | 0.99–1.05× |
| Unequal scales 50k × 32 | kneighbors | 151, 251, 252 | 0.506–0.515 | 0.504–0.520 | 0.96–1.01× | 0.98–1.01× |
| Unequal scales 50k × 32 | fit_query | 151, 251, 252 | 0.827–0.863 | 0.823–0.857 | 0.95–1.01× | 0.98–1.02× |
| Gaussian 4096 × 128 | kneighbors | 151, 251, 252 | 0.389–0.440 | 0.386–0.439 | 0.99–1.03× | 0.94–1.04× |
| Gaussian 4096 × 128 | fit_query | 151, 251, 252 | 0.773–0.801 | 0.751–0.788 | 0.96–0.99× | 0.96–1.05× |
| Clusters 4096 × 128 | kneighbors | 151, 251, 252 | 0.397–0.435 | 0.874–0.985 | 2.15–2.41× | 0.95–1.02× |
| Clusters 4096 × 128 | fit_query | 151, 251, 252 | 0.561–0.566 | 1.314–1.339 | 2.28–2.37× | 0.98–1.03× |
| Offset +1e4, 4096 × 128 | kneighbors | 151, 251, 252 | 0.386–0.481 | 0.406–0.431 | 0.93–1.07× | 0.89–1.08× |
| Offset +1e4, 4096 × 128 | fit_query | 151, 251, 252 | 0.790–0.854 | 0.756–0.896 | 0.95–1.01× | 0.97–1.03× |
| Unequal scales 4096 × 128 | kneighbors | 151, 251, 252 | 0.401–0.434 | 0.378–0.449 | 0.91–1.07× | 0.97–1.02× |
| Unequal scales 4096 × 128 | fit_query | 151, 251, 252 | 0.754–0.824 | 0.773–0.785 | 0.93–1.01× | 0.98–1.05× |

## Timing and dataset scope

Hardware: Apple M4 Max, 128 GiB memory; platform `macOS-26.6.2-arm64-arm-64bit-Mach-O`. Python `3.13.3`, MLX `0.32.2`, NumPy `2.5.2`, sklearn `1.9.0`. Reported OpenMP thread counts: `{"sklearn_1": [1], "sklearn_auto": [16], "sklearn_default": [16]}`.

Persistent isolated public NumPy API workers. All GPU/CPU timed calls are serial. Cold first fit/query is separate; every operation gets the stated count of warmups per variant. Variant starting order rotates by dataset and operation, and each round reverses it. Each call is synchronized and audited outside timing. fit_query includes a fresh fit and all first-query preparation. No result caching. Inputs are the same float32 arrays for all variants, with no rescaling or jitter. Real data use raw features and stratified holdouts. Synthetic inputs and real-data split seeds are preserved per fixture.

The current path explicitly forces the MLX GPU backend and brute search. These results do not evaluate automatic mlxlearn CPU/GPU dispatch. The 'cold' sample is first fit/query for a fixture in a persistent worker, not process/compiler startup. Timed work includes public validation, NumPy boundaries and GPU synchronization; IPC and numerical audits are outside timing. Auto's actual fitted algorithm is recorded per call. Distinct auto/brute workers may differ from timing variation even when both select brute.

Speedup = sklearn median milliseconds / current median milliseconds. Below 1× means current is slower. Family ranges span seed-specific ratios, not ratios of independently selected timing endpoints. The paired-call count uses corresponding retained rounds. These are descriptive results, not confidence intervals.

Fixtures cover the previous eleven Gaussian shapes; Gaussian, clusters, +1e4 offsets and unequal scales at 50k × 32 and 4096 × 128; 100k × 32 Gaussian; and bundled raw Digits, Breast Cancer, Iris and Wine. All inputs are float32 Euclidean data with no rescaling or jitter. Real data use stratified holdouts. Class labels reused numerically for real-data regression provide timing fixtures, not regression-quality evidence. Large real datasets, sparse input, other metrics, float64 and other hardware are outside this run.

## All query workloads

Ranges span seeds. The final column counts all search/inference operation medians reaching 2×, in the order default brute / auto / single-thread brute.

| Family | Train × features; queries; k | Current query ms | vs default brute | vs auto | vs one thread | Medians ≥2× |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 8,192 × 8; 1; 5 | 0.430–0.576 | 1.30–2.18× | 0.36–0.48× | 0.44–0.58× | 5 / 3 / 0 (each /27) |
| Gaussian 8D, k5 | 8,192 × 8; 256; 5 | 0.708–0.817 | 1.82–2.02× | 5.35–6.67× | 5.00–5.84× | 9 / 27 / 27 (each /27) |
| Gaussian 32D, k5 | 8,192 × 32; 256; 5 | 0.999–1.060 | 3.48–4.12× | 3.41–3.82× | 4.12–4.35× | 27 / 27 / 27 (each /27) |
| Gaussian 32D, k32 | 8,192 × 32; 128; 32 | 1.872–1.955 | 1.94–2.06× | 1.97–2.04× | 1.54–1.61× | 6 / 6 / 0 (each /27) |
| Gaussian 128D, k5 | 4,096 × 128; 128; 5 | 0.937–0.995 | 3.22–3.48× | 3.31–3.47× | 1.72–1.82× | 27 / 27 / 0 (each /27) |
| Gaussian 1024D, k5 | 2,048 × 1024; 32; 5 | 1.194–1.352 | 2.62–3.00× | 2.67–3.01× | 1.65–1.91× | 24 / 24 / 1 (each /27) |
| Gaussian 32D, k64 | 4,096 × 32; 128; 64 | 1.488–1.564 | 2.57–2.71× | 2.56–2.63× | 1.64–1.72× | 25 / 25 / 1 (each /27) |
| Gaussian 16D, k5 | 8,192 × 16; 256; 5 | 0.822–0.888 | 4.19–4.57× | 4.14–4.60× | 4.66–5.01× | 27 / 27 / 27 (each /27) |
| Gaussian 32D, single query | 8,192 × 32; 1; 5 | 0.656–0.671 | 4.20–4.41× | 4.24–4.33× | 0.42–0.44× | 27 / 27 / 0 (each /27) |
| Gaussian 8D, k32 | 8,192 × 8; 128; 32 | 1.846–1.873 | 0.82–0.94× | 2.22–2.26× | 1.52–1.56× | 0 / 27 / 0 (each /27) |
| Gaussian 50k × 32 (original panel) | 50,000 × 32; 256; 5 | 1.429–1.529 | 4.28–4.62× | 4.54–4.59× | 16.30–17.49× | 27 / 27 / 27 (each /27) |
| Gaussian 50k × 32 | 50,000 × 32; 256; 5 | 1.469–1.545 | 4.26–4.73× | 4.19–4.63× | 16.15–16.97× | 27 / 27 / 27 (each /27) |
| Clusters 50k × 32 | 50,000 × 32; 256; 5 | 2.764–2.792 | 2.29–2.45× | 2.35–2.45× | 8.93–9.03× | 27 / 27 / 27 (each /27) |
| Offset +1e4, 50k × 32 | 50,000 × 32; 256; 5 | 1.518–1.635 | 3.93–4.33× | 4.09–4.46× | 15.24–16.47× | 27 / 27 / 27 (each /27) |
| Unequal scales 50k × 32 | 50,000 × 32; 256; 5 | 1.753–1.899 | 3.49–3.85× | 3.44–3.92× | 13.11–14.27× | 27 / 27 / 27 (each /27) |
| Gaussian 4096 × 128 | 4,096 × 128; 128; 5 | 0.953–0.991 | 3.29–3.36× | 3.39–3.45× | 1.71–1.80× | 26 / 27 / 0 (each /27) |
| Clusters 4096 × 128 | 4,096 × 128; 128; 5 | 1.160–1.241 | 2.56–2.87× | 2.62–2.78× | 1.35–1.47× | 27 / 27 / 0 (each /27) |
| Offset +1e4, 4096 × 128 | 4,096 × 128; 128; 5 | 0.913–1.003 | 3.27–3.48× | 3.18–3.58× | 1.71–1.87× | 26 / 27 / 1 (each /27) |
| Unequal scales 4096 × 128 | 4,096 × 128; 128; 5 | 0.914–0.997 | 3.20–3.58× | 3.24–3.55× | 1.72–1.87× | 27 / 27 / 0 (each /27) |
| Gaussian 100k × 32 | 100,000 × 32; 256; 5 | 2.041–2.143 | 4.46–4.51× | 4.45–4.79× | 23.18–24.29× | 27 / 27 / 27 (each /27) |
| Digits (raw) | 1,541 × 64; 256; 5 | 1.344–1.722 | 1.78–2.41× | 1.73–2.31× | 0.66–0.84× | 8 / 7 / 0 (each /27) |
| Breast Cancer (raw) | 455 × 30; 114; 5 | 0.614–0.647 | 3.35–3.48× | 3.38–3.50× | 0.44–0.47× | 27 / 27 / 0 (each /27) |
| Iris (raw) | 120 × 4; 30; 5 | 0.351–0.622 | 0.98–1.83× | 0.39–0.56× | 0.23–0.37× | 0 / 0 / 0 (each /27) |
| Wine (raw) | 142 × 13; 36; 5 | 0.326–0.429 | 1.61–2.19× | 0.51–0.60× | 0.36–0.43× | 2 / 0 / 0 (each /27) |

## Fresh fit + query

| Family | Current fresh ms | vs default brute | vs auto | vs one thread |
| --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 0.488–0.715 | 1.33–2.02× | 2.27–3.47× | 0.49–0.73× |
| Gaussian 8D, k5 | 0.869–0.919 | 1.70–2.14× | 6.50–6.77× | 4.61–4.88× |
| Gaussian 32D, k5 | 1.131–1.193 | 3.28–3.74× | 3.25–3.84× | 3.76–3.95× |
| Gaussian 32D, k32 | 2.494–2.581 | 1.54–1.60× | 1.55–1.58× | 1.22–1.27× |
| Gaussian 128D, k5 | 1.580–1.602 | 2.16–2.21× | 2.08–2.24× | 1.14–1.18× |
| Gaussian 1024D, k5 | 2.211–2.462 | 1.59–1.79× | 1.61–1.78× | 1.04–1.17× |
| Gaussian 32D, k64 | 2.044–2.188 | 1.86–2.01× | 1.90–2.08× | 1.23–1.32× |
| Gaussian 16D, k5 | 0.958–1.010 | 3.68–3.97× | 3.67–3.98× | 4.18–4.39× |
| Gaussian 32D, single query | 0.659–0.792 | 3.72–4.70× | 3.56–4.55× | 0.49–0.61× |
| Gaussian 8D, k32 | 1.726–2.470 | 0.65–0.94× | 2.29–3.33× | 1.19–1.73× |
| Gaussian 50k × 32 (original panel) | 1.934–2.000 | 3.57–3.87× | 3.52–3.76× | 12.64–13.11× |
| Gaussian 50k × 32 | 1.954–2.001 | 3.50–3.64× | 3.51–3.72× | 12.64–12.95× |
| Clusters 50k × 32 | 3.092–3.161 | 2.17–2.33× | 2.13–2.32× | 8.01–8.17× |
| Offset +1e4, 50k × 32 | 2.073–2.141 | 3.25–3.45× | 3.17–3.61× | 11.80–12.21× |
| Unequal scales 50k × 32 | 2.192–2.287 | 3.10–3.24× | 3.04–3.15× | 11.03–11.58× |
| Gaussian 4096 × 128 | 1.548–1.727 | 1.98–2.22× | 2.04–2.21× | 1.07–1.20× |
| Clusters 4096 × 128 | 1.090–1.408 | 2.42–3.19× | 2.40–3.09× | 1.31–1.67× |
| Offset +1e4, 4096 × 128 | 1.590–1.739 | 1.99–2.14× | 2.01–2.10× | 1.07–1.17× |
| Unequal scales 4096 × 128 | 1.583–1.635 | 2.05–2.14× | 2.09–2.14× | 1.13–1.18× |
| Gaussian 100k × 32 | 2.833–2.862 | 3.50–3.57× | 3.50–3.61× | 17.40–17.73× |
| Digits (raw) | 1.972–2.326 | 1.40–1.81× | 1.35–1.71× | 0.52–0.61× |
| Breast Cancer (raw) | 0.707–0.748 | 3.14–3.22× | 3.01–3.28× | 0.50–0.52× |
| Iris (raw) | 0.579–0.687 | 1.06–1.49× | 0.45–0.65× | 0.29–0.45× |
| Wine (raw) | 0.647–0.692 | 1.03–1.23× | 0.46–0.56× | 0.33–0.37× |

## Every operation and paired call

Operation cells report median counts, then individual paired-call counts. Fit-only excludes first-query preparation and is not a substitute for fresh fit + query.

| Operation | Brute, default threads | Auto, default threads | Brute, one thread |
| --- | --- | --- | --- |
| `fit` (excluded from headline) | 0/72; 0/1,080 | 9/72; 135/1,080 | 0/72; 0/1,080 |
| `fit_query` | 46/72; 676/1,080 | 55/72; 773/1,080 | 27/72; 407/1,080 |
| `kneighbors` | 56/72; 861/1,080 | 60/72; 891/1,080 | 27/72; 447/1,080 |
| `indices` | 53/72; 797/1,080 | 59/72; 866/1,080 | 28/72; 462/1,080 |
| `predict_uniform` | 53/72; 792/1,080 | 58/72; 854/1,080 | 27/72; 463/1,080 |
| `proba_uniform` | 52/72; 784/1,080 | 57/72; 845/1,080 | 28/72; 464/1,080 |
| `regress_uniform` | 52/72; 785/1,080 | 57/72; 853/1,080 | 27/72; 455/1,080 |
| `predict_distance` | 58/72; 872/1,080 | 60/72; 911/1,080 | 28/72; 466/1,080 |
| `proba_distance` | 55/72; 861/1,080 | 59/72; 895/1,080 | 27/72; 453/1,080 |
| `regress_distance` | 57/72; 860/1,080 | 59/72; 886/1,080 | 27/72; 466/1,080 |

Family-level paired-call counts retain every individual threshold miss:

| Family | Brute, default threads | Auto, default threads | Brute, one thread |
| --- | --- | --- | --- |
| Gaussian 8D, single query | 100/405 | 41/405 | 0/405 |
| Gaussian 8D, k5 | 175/405 | 405/405 | 405/405 |
| Gaussian 32D, k5 | 405/405 | 405/405 | 405/405 |
| Gaussian 32D, k32 | 107/405 | 122/405 | 11/405 |
| Gaussian 128D, k5 | 390/405 | 391/405 | 58/405 |
| Gaussian 1024D, k5 | 355/405 | 345/405 | 73/405 |
| Gaussian 32D, k64 | 362/405 | 362/405 | 54/405 |
| Gaussian 16D, k5 | 405/405 | 404/405 | 405/405 |
| Gaussian 32D, single query | 405/405 | 404/405 | 0/405 |
| Gaussian 8D, k32 | 0/405 | 364/405 | 1/405 |
| Gaussian 50k × 32 (original panel) | 405/405 | 405/405 | 405/405 |
| Gaussian 50k × 32 | 405/405 | 405/405 | 405/405 |
| Clusters 50k × 32 | 392/405 | 388/405 | 405/405 |
| Offset +1e4, 50k × 32 | 405/405 | 405/405 | 405/405 |
| Unequal scales 50k × 32 | 405/405 | 405/405 | 405/405 |
| Gaussian 4096 × 128 | 382/405 | 384/405 | 66/405 |
| Clusters 4096 × 128 | 396/405 | 402/405 | 12/405 |
| Offset +1e4, 4096 × 128 | 384/405 | 386/405 | 91/405 |
| Unequal scales 4096 × 128 | 385/405 | 388/405 | 72/405 |
| Gaussian 100k × 32 | 405/405 | 405/405 | 405/405 |
| Digits (raw) | 136/405 | 156/405 | 0/405 |
| Breast Cancer (raw) | 402/405 | 402/405 | 0/405 |
| Iris (raw) | 21/405 | 0/405 | 0/405 |
| Wine (raw) | 61/405 | 0/405 | 0/405 |

## Accuracy and integrity

Exhaustive direct float64 differences on actual float32 coordinates; canonical distance/index order. Strict index and prediction mismatches remain failures. A separate query audit recognizes exact-distance-valid alternative tie choices; it never changes strict pass. Prediction mismatches on tied fixtures are marked ambiguous, not automatically accepted. Distance atol/rtol1e-6; existing prediction tolerance retained.

| Variant | Strict audit passes | Failure-kind counts | Maximum absolute errors |
| --- | --- | --- | --- |
| current | 14,472/14,472 | {} | {"distance_max_abs": 2.295977981248143e-05, "distance_max_abs_from_returned_points": 2.295977981248143e-05, "max_abs": 3.902618989082285e-08} |
| sklearn_default | 13,114/14,472 | {"canonical_prediction_mismatch": 160, "reported_distance_error": 793, "valid_exact_tie_alternative": 405} | {"distance_max_abs": 0.007307924795895815, "distance_max_abs_from_returned_points": 0.007307924795895815, "max_abs": 0.0033060949847009136} |
| sklearn_1 | 13,034/14,472 | {"canonical_prediction_mismatch": 160, "canonical_prediction_mismatch_on_boundary_ties": 80, "reported_distance_error": 793, "valid_exact_tie_alternative": 405} | {"distance_max_abs": 0.007307924795895815, "distance_max_abs_from_returned_points": 0.007307924795895815, "max_abs": 0.5999999999999996} |
| sklearn_auto | 13,114/14,472 | {"canonical_prediction_mismatch": 160, "reported_distance_error": 793, "valid_exact_tie_alternative": 405} | {"distance_max_abs": 0.007307924795895815, "distance_max_abs_from_returned_points": 0.007307924795895815, "max_abs": 0.0033060949847009136} |

Audit counts include cold, warmup and retained calls, and fit-only's untimed query audit. They count repeated audit requests rather than distinct datasets. Exact-distance-valid alternative ties are valid sklearn neighbor outputs; their canonical-index differences are kept separate from incorrect membership, ordering and distance-tolerance failures. Prediction differences on boundary ties are flagged as ambiguous and not automatically validated. The summary contains per-phase counts and all failing case/operation evidence pointers. Accuracy qualifications never remove timings from speed denominators.

Integrity checks: `{"fixture_integrity_pass": true, "harness_integrity_pass": true, "initial_worker_source_integrity_pass": true, "source_integrity_pass": true, "worker_source_integrity_pass": true}`. Current accuracy plus integrity acceptance: **True**.

## Artifacts

[Raw measurements and every audit](../benchmarks/results/knn-distribution-final-2026-10-30-broad.json)

[Validated derived summary](../benchmarks/results/knn-distribution-final-2026-10-30-summary.json)

[Controlled source comparison](../benchmarks/results/knn-distribution-final-2026-10-30-control.json)

[Vector figure](assets/knn-distribution-final-2026-10-30.svg)

Production manifest: `benchmarks/reference/knn-distribution-final-2026-10-30.manifest.json`. Snapshot: `{"path": "benchmarks/reference/knn-distribution-final-2026-10-30.zip", "sha256": "ce743a3a69c57c6b87701e0b929b9e2c91030aeabf90b7f439ad01232c08fa37"}`. The raw report records source, loaded-worker, harness and fixture hashes before and after timing.

Generated from recorded measurements. No source optimization or performance claim is inferred from a separate historical run.
