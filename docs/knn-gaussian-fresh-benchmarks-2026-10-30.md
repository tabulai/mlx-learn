# Gaussian fresh-fit fix: full benchmark results

Completed `2026-10-30T17:17:14.524677+00:00`. This run covers **72 dataset instances in 24 families**, 9 search/inference operations, and **15 retained calls after 5 warmups** per dataset-operation-variant. Fit-only is recorded separately and excluded from the headline 2× counts. Production source and fixture identities are recorded in the raw results.

| sklearn baseline | Medians ≥2× | Individual paired calls ≥2× | All medians ≥2×? | Every pair ≥2×? |
| --- | --- | --- | --- | --- |
| Brute, default threads | 492/648 (75.9%) | 7,367/9,720 (75.8%) | False | False |
| Auto, default threads | 528/648 (81.5%) | 7,801/9,720 (80.3%) | False | False |
| Brute, one thread | 246/648 (38.0%) | 4,052/9,720 (41.7%) | False | False |

Against the fastest median among the three measured sklearn configurations in each dataset-operation cell, **233/648 (36.0%)** reach 2×. This is a best-of-measured-configurations comparison, not one fixed sklearn setting. The panel weights each dataset-operation cell equally; percentages do not estimate real-world workload frequencies. Raw timing counts retain every baseline accuracy failure.

**8/24 families** reach 2× in every seed-operation median against all three baselines; **6/24** also reach 2× in every retained paired call. A family passes only when all its constituent cases and operations pass, without averaging away misses.

![Minimum speedup across seeds](assets/knn-gaussian-fresh-final-2026-10-30.png)

## Controlled comparison with both preceding sources

The pre-geometry baseline is `knn-128-final-2026-10-30.zip`; it checks removal of the Gaussian fresh-fit regression. The distribution-fixed baseline is `knn-distribution-final-2026-10-30.zip`; it checks preservation of the clustered/offset gains. Current, an identical-code A/A worker, and both frozen baselines ran serially with 31 interleaved paired rounds after 1 seconds of continuous warmup per worker/operation. These are contemporaneous comparisons on identical fixtures. Displayed ranges span the listed dataset seeds, including the additional Gaussian seeds recorded by the control harness.

All speedups are baseline/current time. The two ratio columns per baseline distinguish the ratio of per-worker medians from the median of corresponding paired-call ratios; neither replaces the other. The A/A column is the median paired ratio of identical-code second-worker/current time. Small changes should be assessed against this process/timing variation and the underlying samples.

Control accuracy and integrity acceptance: **True**. Per-variant strict accuracy: `{"current": true, "current_b": true, "distribution": true, "pre_geometry": true}`.

| Family | Operation | Seeds | Current ms | Pre-geometry ms | Distribution-fixed ms | Pre-geometry ratio of medians | Pre-geometry median paired ratio | Distribution-fixed ratio of medians | Distribution-fixed median paired ratio | A/A paired ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 50k × 32 | kneighbors | 151, 251, 252, 351, 451, 552 | 0.436–2.362 | 0.449–2.348 | 0.439–2.349 | 0.984–1.035× | 0.982–1.032× | 0.958–1.048× | 0.958–1.050× | 0.972–1.010× |
| Gaussian 50k × 32 | fit_query | 151, 251, 252, 351, 451, 552 | 0.779–2.480 | 0.775–2.454 | 0.781–2.476 | 0.981–1.021× | 0.959–1.024× | 0.988–1.039× | 0.953–1.031× | 0.963–1.006× |
| Clusters 50k × 32 | kneighbors | 151, 251, 252 | 0.810–0.831 | 2.108–2.185 | 0.808–0.823 | 2.594–2.697× | 2.610–2.682× | 0.991–1.007× | 0.997–1.034× | 0.978–1.000× |
| Clusters 50k × 32 | fit_query | 151, 251, 252 | 1.109–1.119 | 2.323–2.327 | 1.098–1.131 | 2.077–2.096× | 2.060–2.099× | 0.981–1.019× | 0.979–1.014× | 0.982–1.022× |
| Offset +1e4, 50k × 32 | kneighbors | 151, 251, 252 | 0.456–0.490 | 2.135–2.207 | 0.463–0.481 | 4.369–4.835× | 4.364–4.772× | 0.978–1.014× | 0.996–1.007× | 0.969–1.038× |
| Offset +1e4, 50k × 32 | fit_query | 151, 251, 252 | 0.888–0.899 | 2.333–2.368 | 0.868–0.900 | 2.604–2.667× | 2.598–2.711× | 0.969–1.010× | 0.938–1.005× | 0.959–0.977× |
| Unequal scales 50k × 32 | kneighbors | 151, 251, 252 | 0.516–0.529 | 0.520–0.526 | 0.522–0.529 | 0.991–1.012× | 0.982–1.011× | 0.991–1.024× | 0.983–1.011× | 0.981–1.027× |
| Unequal scales 50k × 32 | fit_query | 151, 251, 252 | 0.874–0.881 | 0.872–0.898 | 0.858–0.892 | 0.990–1.028× | 1.015–1.034× | 0.975–1.013× | 0.982–1.013× | 0.984–1.045× |
| Gaussian 4096 × 128 | kneighbors | 151, 251, 252, 351, 451, 552 | 0.422–0.457 | 0.415–0.450 | 0.416–0.441 | 0.946–1.034× | 0.948–1.050× | 0.952–1.042× | 0.942–1.046× | 0.942–1.025× |
| Gaussian 4096 × 128 | fit_query | 151, 251, 252, 351, 451, 552 | 0.647–0.708 | 0.796–0.860 | 0.821–0.876 | 1.197–1.317× | 1.196–1.281× | 1.232–1.326× | 1.220–1.330× | 0.962–1.010× |
| Clusters 4096 × 128 | kneighbors | 151, 251, 252 | 0.405–0.431 | 0.997–1.019 | 0.412–0.424 | 2.315–2.517× | 2.374–2.494× | 0.985–1.018× | 0.999–1.006× | 0.950–1.023× |
| Clusters 4096 × 128 | fit_query | 151, 251, 252 | 0.564–0.583 | 1.299–1.366 | 0.565–0.593 | 2.273–2.423× | 2.262–2.423× | 1.000–1.032× | 0.989–1.033× | 0.955–0.993× |
| Offset +1e4, 4096 × 128 | kneighbors | 151, 251, 252 | 0.407–0.423 | 0.415–0.438 | 0.414–0.425 | 1.012–1.054× | 0.997–1.069× | 0.992–1.023× | 0.982–1.060× | 0.970–1.057× |
| Offset +1e4, 4096 × 128 | fit_query | 151, 251, 252 | 0.657–0.703 | 0.814–0.837 | 0.836–0.854 | 1.191–1.266× | 1.256–1.292× | 1.216–1.280× | 1.279–1.314× | 0.979–1.020× |
| Unequal scales 4096 × 128 | kneighbors | 151, 251, 252 | 0.411–0.489 | 0.427–0.465 | 0.433–0.479 | 0.950–1.037× | 0.971–1.039× | 0.943–1.054× | 0.920–1.017× | 0.965–1.088× |
| Unequal scales 4096 × 128 | fit_query | 151, 251, 252 | 0.622–0.675 | 0.810–0.845 | 0.812–0.848 | 1.203–1.359× | 1.212–1.368× | 1.226–1.306× | 1.213–1.304× | 0.934–1.031× |

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
| Gaussian 8D, single query | 8,192 × 8; 1; 5 | 0.421–0.579 | 1.33–2.24× | 0.35–0.55× | 0.43–0.61× | 8 / 3 / 0 (each /27) |
| Gaussian 8D, k5 | 8,192 × 8; 256; 5 | 0.774–0.915 | 1.82–2.16× | 4.94–5.82× | 4.55–5.34× | 14 / 27 / 27 (each /27) |
| Gaussian 32D, k5 | 8,192 × 32; 256; 5 | 1.066–1.079 | 3.43–4.04× | 3.46–4.33× | 4.07–4.13× | 27 / 27 / 27 (each /27) |
| Gaussian 32D, k32 | 8,192 × 32; 128; 32 | 1.953–1.967 | 1.94–2.24× | 1.96–2.34× | 1.56–1.61× | 11 / 13 / 0 (each /27) |
| Gaussian 128D, k5 | 4,096 × 128; 128; 5 | 0.900–1.013 | 3.50–4.11× | 3.27–4.51× | 1.76–1.86× | 27 / 27 / 0 (each /27) |
| Gaussian 1024D, k5 | 2,048 × 1024; 32; 5 | 1.262–1.380 | 2.57–2.82× | 2.67–2.90× | 1.66–1.80× | 25 / 25 / 1 (each /27) |
| Gaussian 32D, k64 | 4,096 × 32; 128; 64 | 1.539–1.573 | 2.49–2.70× | 2.48–2.63× | 1.66–1.70× | 27 / 27 / 0 (each /27) |
| Gaussian 16D, k5 | 8,192 × 16; 256; 5 | 0.829–0.859 | 4.23–4.54× | 4.18–4.33× | 4.91–5.08× | 27 / 27 / 27 (each /27) |
| Gaussian 32D, single query | 8,192 × 32; 1; 5 | 0.641–0.700 | 4.23–4.56× | 4.16–4.21× | 0.42–0.46× | 27 / 27 / 0 (each /27) |
| Gaussian 8D, k32 | 8,192 × 8; 128; 32 | 1.856–1.896 | 0.85–0.87× | 2.20–2.23× | 1.53–1.55× | 0 / 27 / 0 (each /27) |
| Gaussian 50k × 32 (original panel) | 50,000 × 32; 256; 5 | 1.500–1.555 | 4.35–4.72× | 4.23–4.54× | 16.12–16.69× | 27 / 27 / 27 (each /27) |
| Gaussian 50k × 32 | 50,000 × 32; 256; 5 | 1.418–1.592 | 4.31–4.70× | 4.24–5.02× | 15.76–17.47× | 27 / 27 / 27 (each /27) |
| Clusters 50k × 32 | 50,000 × 32; 256; 5 | 2.716–2.804 | 2.42–2.51× | 2.45–2.50× | 8.91–9.13× | 27 / 27 / 27 (each /27) |
| Offset +1e4, 50k × 32 | 50,000 × 32; 256; 5 | 1.483–1.645 | 4.10–4.45× | 4.30–4.41× | 15.38–16.99× | 27 / 27 / 27 (each /27) |
| Unequal scales 50k × 32 | 50,000 × 32; 256; 5 | 1.810–1.916 | 3.62–3.83× | 3.57–3.63× | 13.70–14.22× | 27 / 27 / 27 (each /27) |
| Gaussian 4096 × 128 | 4,096 × 128; 128; 5 | 0.915–0.971 | 2.83–3.01× | 2.91–3.00× | 1.71–1.84× | 27 / 27 / 2 (each /27) |
| Clusters 4096 × 128 | 4,096 × 128; 128; 5 | 1.171–1.232 | 2.18–2.40× | 2.29–2.39× | 1.35–1.41× | 23 / 21 / 0 (each /27) |
| Offset +1e4, 4096 × 128 | 4,096 × 128; 128; 5 | 0.905–0.973 | 2.86–3.17× | 2.96–3.13× | 1.71–1.85× | 27 / 27 / 0 (each /27) |
| Unequal scales 4096 × 128 | 4,096 × 128; 128; 5 | 0.924–1.005 | 3.28–3.70× | 3.33–3.58× | 1.67–1.84× | 27 / 27 / 0 (each /27) |
| Gaussian 100k × 32 | 100,000 × 32; 256; 5 | 2.093–2.172 | 4.48–4.85× | 4.22–4.80× | 23.98–25.37× | 27 / 27 / 27 (each /27) |
| Digits (raw) | 1,541 × 64; 256; 5 | 1.364–1.708 | 1.72–2.27× | 1.78–2.28× | 0.66–0.86× | 6 / 7 / 0 (each /27) |
| Breast Cancer (raw) | 455 × 30; 114; 5 | 0.628–0.717 | 3.26–3.43× | 3.13–3.50× | 0.43–0.44× | 27 / 27 / 0 (each /27) |
| Iris (raw) | 120 × 4; 30; 5 | 0.400–0.617 | 0.89–1.78× | 0.35–0.57× | 0.23–0.37× | 0 / 0 / 0 (each /27) |
| Wine (raw) | 142 × 13; 36; 5 | 0.360–0.372 | 1.64–1.86× | 0.58–0.64× | 0.38–0.42× | 0 / 0 / 0 (each /27) |

## Fresh fit + query

| Family | Current fresh ms | vs default brute | vs auto | vs one thread |
| --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 0.571–0.708 | 1.23–1.87× | 2.38–3.10× | 0.49–0.67× |
| Gaussian 8D, k5 | 0.938–0.945 | 2.01–2.35× | 6.43–6.77× | 4.55–4.75× |
| Gaussian 32D, k5 | 1.118–1.304 | 2.96–4.25× | 3.03–4.14× | 3.52–4.02× |
| Gaussian 32D, k32 | 2.182–2.244 | 1.81–2.13× | 1.82–2.13× | 1.44–1.46× |
| Gaussian 128D, k5 | 1.229–1.306 | 2.77–3.30× | 2.70–3.14× | 1.41–1.58× |
| Gaussian 1024D, k5 | 1.888–1.999 | 1.96–2.11× | 1.95–2.13× | 1.31–1.39× |
| Gaussian 32D, k64 | 1.672–1.886 | 2.27–2.43× | 2.37–2.46× | 1.47–1.63× |
| Gaussian 16D, k5 | 0.946–1.004 | 3.83–4.09× | 3.66–4.07× | 4.37–4.57× |
| Gaussian 32D, single query | 0.624–0.809 | 3.66–5.21× | 3.55–4.71× | 0.51–0.66× |
| Gaussian 8D, k32 | 2.069–2.168 | 0.75–0.92× | 2.69–2.80× | 1.38–1.44× |
| Gaussian 50k × 32 (original panel) | 1.848–1.988 | 3.56–3.71× | 3.66–3.91× | 12.80–13.72× |
| Gaussian 50k × 32 | 1.947–1.981 | 3.55–3.75× | 3.48–3.76× | 12.85–13.12× |
| Clusters 50k × 32 | 3.167–3.204 | 2.20–2.28× | 2.10–2.37× | 7.91–8.06× |
| Offset +1e4, 50k × 32 | 2.082–2.164 | 3.40–3.57× | 3.41–3.50× | 12.07–12.49× |
| Unequal scales 50k × 32 | 2.296–2.349 | 2.99–3.14× | 3.01–3.21× | 10.97–11.55× |
| Gaussian 4096 × 128 | 1.231–1.317 | 2.32–2.42× | 2.27–2.42× | 1.38–1.49× |
| Clusters 4096 × 128 | 1.368–1.400 | 2.09–2.22× | 2.11–2.22× | 1.29–1.33× |
| Offset +1e4, 4096 × 128 | 1.278–1.345 | 2.23–2.34× | 2.17–2.38× | 1.37–1.46× |
| Unequal scales 4096 × 128 | 1.220–1.337 | 2.63–2.93× | 2.63–2.90× | 1.40–1.51× |
| Gaussian 100k × 32 | 2.812–2.945 | 3.51–3.63× | 3.57–3.77× | 18.51–19.16× |
| Digits (raw) | 1.638–1.974 | 1.63–1.97× | 1.51–1.91× | 0.62–0.78× |
| Breast Cancer (raw) | 0.717–0.742 | 3.12–3.19× | 3.13–3.19× | 0.47–0.51× |
| Iris (raw) | 0.514–0.702 | 0.94–1.52× | 0.46–0.68× | 0.30–0.46× |
| Wine (raw) | 0.643–0.668 | 1.04–1.08× | 0.49–0.50× | 0.32–0.34× |

## Every operation and paired call

Operation cells report median counts, then individual paired-call counts. Fit-only excludes first-query preparation and is not a substitute for fresh fit + query.

| Operation | Brute, default threads | Auto, default threads | Brute, one thread |
| --- | --- | --- | --- |
| `fit` (excluded from headline) | 0/72; 0/1,080 | 9/72; 135/1,080 | 0/72; 0/1,080 |
| `fit_query` | 53/72; 792/1,080 | 59/72; 889/1,080 | 27/72; 410/1,080 |
| `kneighbors` | 57/72; 876/1,080 | 60/72; 896/1,080 | 27/72; 446/1,080 |
| `indices` | 53/72; 766/1,080 | 58/72; 834/1,080 | 29/72; 451/1,080 |
| `predict_uniform` | 52/72; 772/1,080 | 56/72; 835/1,080 | 28/72; 468/1,080 |
| `proba_uniform` | 51/72; 774/1,080 | 57/72; 830/1,080 | 27/72; 466/1,080 |
| `regress_uniform` | 53/72; 776/1,080 | 58/72; 839/1,080 | 27/72; 449/1,080 |
| `predict_distance` | 59/72; 888/1,080 | 62/72; 904/1,080 | 27/72; 463/1,080 |
| `proba_distance` | 56/72; 860/1,080 | 59/72; 887/1,080 | 27/72; 449/1,080 |
| `regress_distance` | 58/72; 863/1,080 | 59/72; 887/1,080 | 27/72; 450/1,080 |

Family-level paired-call counts retain every individual threshold miss:

| Family | Brute, default threads | Auto, default threads | Brute, one thread |
| --- | --- | --- | --- |
| Gaussian 8D, single query | 128/405 | 43/405 | 0/405 |
| Gaussian 8D, k5 | 204/405 | 405/405 | 405/405 |
| Gaussian 32D, k5 | 404/405 | 404/405 | 405/405 |
| Gaussian 32D, k32 | 187/405 | 204/405 | 9/405 |
| Gaussian 128D, k5 | 402/405 | 400/405 | 78/405 |
| Gaussian 1024D, k5 | 361/405 | 371/405 | 67/405 |
| Gaussian 32D, k64 | 383/405 | 384/405 | 52/405 |
| Gaussian 16D, k5 | 405/405 | 405/405 | 405/405 |
| Gaussian 32D, single query | 405/405 | 405/405 | 0/405 |
| Gaussian 8D, k32 | 0/405 | 379/405 | 4/405 |
| Gaussian 50k × 32 (original panel) | 405/405 | 405/405 | 405/405 |
| Gaussian 50k × 32 | 405/405 | 405/405 | 405/405 |
| Clusters 50k × 32 | 384/405 | 378/405 | 405/405 |
| Offset +1e4, 50k × 32 | 405/405 | 405/405 | 405/405 |
| Unequal scales 50k × 32 | 405/405 | 405/405 | 405/405 |
| Gaussian 4096 × 128 | 391/405 | 392/405 | 77/405 |
| Clusters 4096 × 128 | 306/405 | 299/405 | 0/405 |
| Offset +1e4, 4096 × 128 | 396/405 | 389/405 | 72/405 |
| Unequal scales 4096 × 128 | 404/405 | 404/405 | 48/405 |
| Gaussian 100k × 32 | 405/405 | 405/405 | 405/405 |
| Digits (raw) | 109/405 | 112/405 | 0/405 |
| Breast Cancer (raw) | 404/405 | 402/405 | 0/405 |
| Iris (raw) | 20/405 | 0/405 | 0/405 |
| Wine (raw) | 49/405 | 0/405 | 0/405 |

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

[Raw measurements and every audit](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-f8f82b657c035b95a37d)

[Validated derived summary](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-e4c267f4b694d73c2156)

[Controlled source comparison](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-2c25c338eb8511a10104)

[Vector figure](assets/knn-gaussian-fresh-final-2026-10-30.svg)

Production manifest: `benchmarks/reference/knn-gaussian-fresh-final-2026-10-30.manifest.json`. Snapshot: `{"path": "benchmarks/reference/knn-gaussian-fresh-final-2026-10-30.zip", "sha256": "df09cfae53f2eeb979456f4a7a9e35b67319b49affe7d48209a5ab1a13314f81"}`. The raw report records source, loaded-worker, harness and fixture hashes before and after timing.

Generated from recorded measurements. No source optimization or performance claim is inferred from a separate historical run.
