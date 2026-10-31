# Automatic CPU/GPU kNN benchmark against three sklearn configurations

Completed `2026-10-30T20:06:43.952021+00:00`. This run covers **72 dataset instances in 24 families**, 9 search/inference operations, and **15 retained calls after 5 warmups** per dataset-operation-variant. Fit-only is recorded separately and excluded from the headline 2× counts. Production source and fixture identities are recorded in the raw results.

| sklearn baseline | Medians ≥2× | Individual paired calls ≥2× | All medians ≥2×? | Every pair ≥2×? |
| --- | --- | --- | --- | --- |
| Brute, default threads | 592/648 (91.4%) | 8,867/9,720 (91.2%) | False | False |
| Auto, default threads | 585/648 (90.3%) | 8,722/9,720 (89.7%) | False | False |
| Brute, one thread | 267/648 (41.2%) | 4,507/9,720 (46.4%) | False | False |

Against the fastest median among the three measured sklearn configurations in each dataset-operation cell, **233/648 (36.0%)** reach 2×. This is a best-of-measured-configurations comparison, not one fixed sklearn setting. The panel weights each dataset-operation cell equally; percentages do not estimate real-world workload frequencies. Raw timing counts retain every baseline accuracy failure.

**8/24 families** reach 2× in every seed-operation median against all three baselines; **7/24** also reach 2× in every retained paired call. A family passes only when all its constituent cases and operations pass, without averaging away misses.

![Minimum speedup across seeds](assets/knn-auto-benchmarks-2026-10-30.png)

## Controlled automatic dispatch and GPU comparison

Four serial workers compare current automatic dispatch, an identical automatic A/A worker, current forced GPU and the preceding frozen small-workload GPU implementation. The unchanged control covers 48 fixtures and 222 case-operations, with 31 paired rounds after 0.25 seconds of continuous warmup per worker/operation. Default continuous warmup is 0.25 seconds (the preceding small control used 1 second); this is configurable and absolute timings must not be compared across those runs. Each worker/operation warms continuously for at least the requested duration. Its one post-warmup probe is audited and retained separately, excluded from paired ratios. All warmup audits and probe samples remain in the report.

Ratios are comparator/automatic time. The ratio of per-worker medians and median paired ratio are shown separately. A/A exposes process/timing variation; these are descriptive comparisons, not confidence intervals. Each worker retains its actual routing metadata.

Control accuracy and integrity acceptance: **True**. Strict quality: `{"current": true, "current_b": true, "current_gpu": true, "previous_gpu": true}`.

| Family | Operation | Seeds | Auto ms | Current GPU/auto medians | Current GPU/auto paired | Previous GPU/auto medians | Previous GPU/auto paired | Auto A/A paired |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | fit_query | 41, 42, 43 | 0.184–0.205 | 1.785–1.949× | 1.805–2.110× | 1.648–2.038× | 1.651–1.946× | 0.902–1.023× |
| Gaussian 8D, single query | kneighbors | 41, 42, 43 | 0.085–0.099 | 2.204–3.216× | 2.122–3.246× | 2.290–2.952× | 2.201–3.110× | 0.968–1.112× |
| Gaussian 8D, single query | indices | 41, 42, 43 | 0.086–0.091 | 2.347–3.174× | 2.336–3.596× | 2.062–3.833× | 2.084–4.380× | 0.961–1.017× |
| Gaussian 8D, single query | predict_uniform | 41, 42, 43 | 0.089–0.101 | 2.036–3.296× | 2.111–3.389× | 2.221–3.378× | 2.300–3.172× | 0.977–1.114× |
| Gaussian 8D, single query | proba_uniform | 41, 42, 43 | 0.094–0.101 | 2.189–3.156× | 2.195–3.206× | 2.043–3.176× | 2.080–3.429× | 0.974–1.020× |
| Gaussian 8D, single query | regress_uniform | 41, 42, 43 | 0.089–0.099 | 2.079–3.068× | 2.091–3.153× | 2.182–2.789× | 2.222–3.056× | 1.031–1.138× |
| Gaussian 8D, single query | predict_distance | 41, 42, 43 | 0.102–0.104 | 2.304–3.514× | 2.341–3.217× | 2.107–4.634× | 1.973–4.537× | 0.947–1.007× |
| Gaussian 8D, single query | proba_distance | 41, 42, 43 | 0.098–0.109 | 1.920–3.042× | 1.872–3.330× | 2.088–3.075× | 2.076–3.150× | 0.989–1.150× |
| Gaussian 8D, single query | regress_distance | 41, 42, 43 | 0.102–0.112 | 2.153–3.385× | 2.119–3.181× | 1.982–2.767× | 1.953–2.767× | 0.985–1.024× |
| Gaussian 32D, single query | fit_query | 49, 50, 51 | 0.244–0.282 | 1.584–1.907× | 1.598–1.871× | 1.657–1.825× | 1.641–1.798× | 0.960–1.046× |
| Gaussian 32D, single query | kneighbors | 49, 50, 51 | 0.140–0.167 | 1.918–2.398× | 2.022–2.323× | 1.405–2.193× | 1.441–2.193× | 0.962–0.973× |
| Gaussian 32D, single query | indices | 49, 50, 51 | 0.133–0.147 | 1.370–2.312× | 1.386–2.454× | 1.468–2.237× | 1.455–2.370× | 0.974–1.027× |
| Gaussian 32D, single query | predict_uniform | 49, 50, 51 | 0.140–0.148 | 1.520–2.656× | 1.508–2.704× | 1.517–2.690× | 1.491–2.566× | 0.966–1.020× |
| Gaussian 32D, single query | proba_uniform | 49, 50, 51 | 0.144–0.153 | 1.537–2.462× | 1.511–2.559× | 1.659–2.401× | 1.616–2.434× | 0.957–1.036× |
| Gaussian 32D, single query | regress_uniform | 49, 50, 51 | 0.144–0.151 | 1.544–2.416× | 1.486–2.351× | 1.422–2.511× | 1.418–2.559× | 0.922–0.997× |
| Gaussian 32D, single query | predict_distance | 49, 50, 51 | 0.148–0.160 | 1.462–2.368× | 1.446–2.446× | 1.540–2.308× | 1.420–2.343× | 0.992–1.120× |
| Gaussian 32D, single query | proba_distance | 49, 50, 51 | 0.156–0.189 | 1.599–2.565× | 1.544–2.507× | 1.375–2.361× | 1.425–2.316× | 0.929–0.997× |
| Gaussian 32D, single query | regress_distance | 49, 50, 51 | 0.152–0.156 | 1.566–2.161× | 1.545–2.314× | 1.765–2.143× | 1.657–2.198× | 0.977–1.112× |
| Digits (raw) | fit_query | 151, 251, 252 | 1.096–1.123 | 1.013–1.146× | 1.016–1.145× | 0.939–1.244× | 0.876–1.192× | 0.989–0.993× |
| Digits (raw) | kneighbors | 151, 251, 252 | 0.882–0.917 | 0.942–1.063× | 0.967–1.009× | 0.480–1.012× | 0.482–1.016× | 1.004–1.014× |
| Digits (raw) | indices | 151, 251, 252 | 0.872–0.893 | 0.554–0.995× | 0.568–1.013× | 0.432–1.069× | 0.438–1.060× | 1.001–1.016× |
| Digits (raw) | predict_uniform | 151, 251, 252 | 0.907–0.912 | 0.457–1.141× | 0.469–1.142× | 0.491–1.074× | 0.490–1.117× | 0.982–1.007× |
| Digits (raw) | proba_uniform | 151, 251, 252 | 0.904–0.930 | 0.520–1.138× | 0.525–1.118× | 0.432–1.123× | 0.427–1.066× | 0.989–1.005× |
| Digits (raw) | regress_uniform | 151, 251, 252 | 0.891–0.930 | 0.522–1.044× | 0.510–1.039× | 0.465–0.949× | 0.468–0.957× | 0.978–1.012× |
| Digits (raw) | predict_distance | 151, 251, 252 | 0.906–1.033 | 1.007–1.050× | 1.002–1.022× | 1.009–1.092× | 0.993–1.113× | 0.993–1.030× |
| Digits (raw) | proba_distance | 151, 251, 252 | 0.913–0.949 | 0.435–1.015× | 0.440–0.928× | 0.894–0.999× | 0.916–0.979× | 0.977–1.017× |
| Digits (raw) | regress_distance | 151, 251, 252 | 0.936–0.951 | 0.526–1.030× | 0.525–1.066× | 0.446–1.095× | 0.462–1.045× | 0.988–1.009× |
| Breast Cancer (raw) | fit_query | 151, 251, 252 | 0.285–0.318 | 1.038–1.473× | 1.039–1.587× | 1.099–2.179× | 1.130–1.960× | 0.979–1.038× |
| Breast Cancer (raw) | kneighbors | 151, 251, 252 | 0.186–0.197 | 1.195–1.379× | 1.197–1.440× | 1.097–2.533× | 1.097–2.029× | 0.969–1.080× |
| Breast Cancer (raw) | indices | 151, 251, 252 | 0.165–0.194 | 1.111–1.425× | 1.080–1.464× | 0.965–1.426× | 0.951–1.416× | 1.066–1.163× |
| Breast Cancer (raw) | predict_uniform | 151, 251, 252 | 0.177–0.189 | 1.336–1.593× | 1.323–1.571× | 1.225–2.369× | 1.195–2.332× | 1.017–1.123× |
| Breast Cancer (raw) | proba_uniform | 151, 251, 252 | 0.172–0.197 | 1.064–1.725× | 1.023–1.683× | 1.137–2.503× | 1.127–2.576× | 1.004–1.186× |
| Breast Cancer (raw) | regress_uniform | 151, 251, 252 | 0.189–0.200 | 1.106–1.516× | 1.104–1.487× | 1.099–1.393× | 1.100–1.353× | 0.991–1.105× |
| Breast Cancer (raw) | predict_distance | 151, 251, 252 | 0.200–0.215 | 1.022–1.372× | 1.042–1.370× | 0.978–1.385× | 0.962–1.394× | 1.006–1.162× |
| Breast Cancer (raw) | proba_distance | 151, 251, 252 | 0.199–0.220 | 1.058–1.434× | 1.089–1.374× | 0.947–1.350× | 0.954–1.385× | 1.005–1.119× |
| Breast Cancer (raw) | regress_distance | 151, 251, 252 | 0.196–0.212 | 1.135–1.403× | 1.098–1.405× | 1.157–2.196× | 1.115–1.892× | 0.976–1.112× |
| Iris (raw) | fit_query | 151, 251, 252 | 0.117–0.168 | 2.475–3.228× | 2.701–3.344× | 2.084–3.040× | 2.214–3.193× | 0.856–1.083× |
| Iris (raw) | kneighbors | 151, 251, 252 | 0.068–0.078 | 2.704–3.905× | 2.729–4.299× | 2.791–3.795× | 2.763–4.139× | 0.921–1.117× |
| Iris (raw) | indices | 151, 251, 252 | 0.069–0.072 | 2.885–3.843× | 2.875–4.047× | 2.703–3.687× | 2.737–3.983× | 0.924–0.974× |
| Iris (raw) | predict_uniform | 151, 251, 252 | 0.070–0.083 | 2.600–4.425× | 2.592–4.701× | 2.774–4.486× | 2.884–4.737× | 1.005–1.152× |
| Iris (raw) | proba_uniform | 151, 251, 252 | 0.077–0.091 | 2.580–4.115× | 2.528–4.734× | 2.532–4.706× | 2.330–4.906× | 0.934–1.142× |
| Iris (raw) | regress_uniform | 151, 251, 252 | 0.078–0.089 | 2.440–3.754× | 2.493–3.992× | 2.496–3.722× | 2.508–3.825× | 0.952–1.143× |
| Iris (raw) | predict_distance | 151, 251, 252 | 0.094–0.098 | 2.288–3.388× | 2.284–3.621× | 2.054–4.109× | 2.199–3.952× | 0.924–1.012× |
| Iris (raw) | proba_distance | 151, 251, 252 | 0.084–0.102 | 2.231–3.621× | 2.147–3.613× | 2.257–3.673× | 2.199–3.623× | 0.928–1.099× |
| Iris (raw) | regress_distance | 151, 251, 252 | 0.087–0.096 | 2.359–3.584× | 2.399–4.262× | 2.179–3.477× | 2.132–3.715× | 0.976–1.248× |
| Wine (raw) | fit_query | 151, 251, 252 | 0.160–0.177 | 1.913–2.364× | 1.904–2.798× | 1.934–2.252× | 1.975–2.516× | 0.973–1.116× |
| Wine (raw) | kneighbors | 151, 251, 252 | 0.077–0.085 | 2.759–3.244× | 2.586–3.496× | 2.588–3.055× | 2.408–3.250× | 0.964–1.103× |
| Wine (raw) | indices | 151, 251, 252 | 0.074–0.099 | 2.392–3.051× | 2.300–3.374× | 2.477–2.933× | 2.494–3.136× | 0.980–1.090× |
| Wine (raw) | predict_uniform | 151, 251, 252 | 0.084–0.106 | 2.083–3.333× | 2.254–3.297× | 2.180–3.276× | 2.182–3.103× | 0.964–1.069× |
| Wine (raw) | proba_uniform | 151, 251, 252 | 0.092–0.141 | 2.180–2.765× | 2.181–3.297× | 2.087–2.934× | 2.112–3.199× | 0.948–1.029× |
| Wine (raw) | regress_uniform | 151, 251, 252 | 0.091–0.094 | 2.262–2.879× | 2.284–2.947× | 2.049–2.687× | 1.990–2.737× | 0.953–1.168× |
| Wine (raw) | predict_distance | 151, 251, 252 | 0.091–0.107 | 2.014–3.139× | 1.939–3.444× | 2.179–3.029× | 2.172–2.961× | 0.964–1.178× |
| Wine (raw) | proba_distance | 151, 251, 252 | 0.101–0.109 | 1.951–2.699× | 1.958–2.792× | 1.949–2.888× | 1.900–2.840× | 0.933–1.007× |
| Wine (raw) | regress_distance | 151, 251, 252 | 0.091–0.109 | 1.974–2.849× | 2.023–3.310× | 1.993–2.685× | 2.028–2.822× | 1.002–1.162× |
| Gaussian 50k × 32 | kneighbors | 151, 251, 252, 351, 451, 552 | 0.545–3.266 | 0.911–1.012× | 0.972–1.039× | 0.972–1.010× | 0.947–1.022× | 0.984–1.025× |
| Gaussian 50k × 32 | fit_query | 151, 251, 252, 351, 451, 552 | 0.791–2.472 | 0.986–1.056× | 0.999–1.021× | 0.984–1.047× | 0.942–1.020× | 0.970–1.012× |
| Clusters 50k × 32 | kneighbors | 151, 251, 252 | 1.033–1.500 | 0.990–1.022× | 0.971–1.022× | 0.975–1.030× | 0.988–1.023× | 0.976–1.025× |
| Clusters 50k × 32 | fit_query | 151, 251, 252 | 1.074–1.125 | 0.972–1.022× | 0.976–1.010× | 0.979–1.032× | 0.982–1.032× | 0.987–1.001× |
| Offset +1e4, 50k × 32 | kneighbors | 151, 251, 252 | 0.515–0.615 | 0.992–1.068× | 0.982–1.021× | 0.984–1.069× | 0.985–1.001× | 0.979–1.008× |
| Offset +1e4, 50k × 32 | fit_query | 151, 251, 252 | 0.839–0.859 | 0.951–1.014× | 0.958–1.016× | 0.954–1.008× | 0.902–1.015× | 0.991–1.075× |
| Unequal scales 50k × 32 | kneighbors | 151, 251, 252 | 0.675–1.030 | 0.994–1.038× | 0.984–1.013× | 0.983–1.020× | 1.001–1.009× | 0.988–1.010× |
| Unequal scales 50k × 32 | fit_query | 151, 251, 252 | 0.827–0.878 | 0.961–1.029× | 0.949–1.045× | 0.945–1.033× | 0.913–1.025× | 0.996–1.087× |
| Gaussian 4096 × 128 | kneighbors | 151, 251, 252, 351, 451, 552 | 0.420–0.505 | 0.962–1.052× | 0.992–1.041× | 0.979–1.011× | 0.994–1.018× | 0.979–1.005× |
| Gaussian 4096 × 128 | fit_query | 151, 251, 252, 351, 451, 552 | 0.632–0.665 | 0.953–1.037× | 0.928–1.036× | 0.966–1.073× | 0.893–1.055× | 0.974–1.031× |
| Clusters 4096 × 128 | kneighbors | 151, 251, 252 | 0.382–0.491 | 0.997–1.131× | 0.997–1.073× | 0.948–1.077× | 0.982–1.017× | 1.021–1.095× |
| Clusters 4096 × 128 | fit_query | 151, 251, 252 | 0.547–0.568 | 1.002–1.026× | 0.990–1.038× | 0.969–1.012× | 0.976–1.013× | 0.975–1.043× |
| Offset +1e4, 4096 × 128 | kneighbors | 151, 251, 252 | 0.455–0.457 | 1.019–1.046× | 1.009–1.049× | 1.011–1.050× | 0.987–1.050× | 0.968–1.020× |
| Offset +1e4, 4096 × 128 | fit_query | 151, 251, 252 | 0.635–0.687 | 0.977–1.042× | 0.992–1.008× | 0.963–1.026× | 0.990–1.030× | 0.987–1.035× |
| Unequal scales 4096 × 128 | kneighbors | 151, 251, 252 | 0.449–0.467 | 0.988–1.019× | 0.942–0.999× | 0.983–1.025× | 1.008–1.010× | 0.983–1.014× |
| Unequal scales 4096 × 128 | fit_query | 151, 251, 252 | 0.616–0.756 | 0.971–1.062× | 0.962–1.065× | 0.903–1.080× | 0.954–1.037× | 0.983–1.012× |

## Timing and dataset scope

Hardware: Apple M4 Max, 128 GiB memory; platform `macOS-26.6.2-arm64-arm-64bit-Mach-O`. Python `3.13.3`, MLX `0.32.2`, NumPy `2.5.2`, sklearn `1.9.0`. Reported OpenMP thread counts: `{"sklearn_1": [1], "sklearn_auto": [16], "sklearn_default": [16]}`.

Persistent isolated public NumPy API workers. All GPU/CPU timed calls are serial. Cold first fit/query is separate; every operation gets the stated count of warmups per variant. Variant starting order rotates by dataset and operation, and each round reverses it. Each call is synchronized and audited outside timing. fit_query includes a fresh fit and all first-query preparation. No result caching. Inputs are the same float32 arrays for all variants, with no rescaling or jitter. Real data use raw features and stratified holdouts. Synthetic inputs and real-data split seeds are preserved per fixture.

The current path uses mlxlearn automatic CPU/GPU dispatch with the public brute-search API. Every call records its execution policy, fitted storage backend and actual query backend separately. Automatic calls must select the native CPU or MLX path; sklearn fallback is rejected. The 'cold' sample is first fit/query for a fixture in a persistent worker, not process/compiler startup. Timed work includes public validation, NumPy boundaries and GPU synchronization; IPC and numerical audits are outside timing. Auto's actual fitted algorithm is recorded per call. Distinct auto/brute workers may differ from timing variation even when both select brute.

Speedup = sklearn median milliseconds / current median milliseconds. Below 1× means current is slower. Family ranges span seed-specific ratios, not ratios of independently selected timing endpoints. The paired-call count uses corresponding retained rounds. These are descriptive results, not confidence intervals.

Fixtures cover the previous eleven Gaussian shapes; Gaussian, clusters, +1e4 offsets and unequal scales at 50k × 32 and 4096 × 128; 100k × 32 Gaussian; and bundled raw Digits, Breast Cancer, Iris and Wine. All inputs are float32 Euclidean data with no rescaling or jitter. Real data use stratified holdouts. Class labels reused numerically for real-data regression provide timing fixtures, not regression-quality evidence. Large real datasets, sparse input, other metrics, float64 and other hardware are outside this run.

## All query workloads

Ranges span seeds. The final column counts all search/inference operation medians reaching 2×, in the order default brute / auto / single-thread brute.

| Family | Train × features; queries; k | Current query ms | vs default brute | vs auto | vs one thread | Medians ≥2× |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 8,192 × 8; 1; 5 | 0.113–0.118 | 6.74–7.05× | 1.77–1.79× | 2.08–2.16× | 27 / 6 / 18 (each /27) |
| Gaussian 8D, k5 | 8,192 × 8; 256; 5 | 0.737–0.828 | 2.24–2.49× | 5.20–6.23× | 5.30–5.91× | 8 / 27 / 27 (each /27) |
| Gaussian 32D, k5 | 8,192 × 32; 256; 5 | 0.977–1.047 | 4.32–4.54× | 4.24–4.70× | 4.32–4.61× | 27 / 27 / 27 (each /27) |
| Gaussian 32D, k32 | 8,192 × 32; 128; 32 | 1.847–1.938 | 2.00–2.47× | 1.98–2.55× | 1.59–1.67× | 17 / 18 / 1 (each /27) |
| Gaussian 128D, k5 | 4,096 × 128; 128; 5 | 0.881–0.972 | 3.40–4.10× | 3.38–4.02× | 1.79–1.97× | 27 / 27 / 0 (each /27) |
| Gaussian 1024D, k5 | 2,048 × 1024; 32; 5 | 1.162–1.281 | 2.89–3.10× | 2.79–3.13× | 1.74–1.94× | 27 / 27 / 0 (each /27) |
| Gaussian 32D, k64 | 4,096 × 32; 128; 64 | 1.259–1.502 | 2.68–3.31× | 2.62–3.24× | 1.70–2.10× | 27 / 27 / 2 (each /27) |
| Gaussian 16D, k5 | 8,192 × 16; 256; 5 | 0.801–0.864 | 4.31–4.71× | 4.30–4.45× | 5.02–5.42× | 27 / 27 / 27 (each /27) |
| Gaussian 32D, single query | 8,192 × 32; 1; 5 | 0.172–0.186 | 15.67–16.06× | 16.04–17.29× | 1.46–1.60× | 27 / 27 / 0 (each /27) |
| Gaussian 8D, k32 | 8,192 × 8; 128; 32 | 1.830–1.848 | 0.84–1.10× | 2.24–2.28× | 1.61–1.63× | 0 / 27 / 0 (each /27) |
| Gaussian 50k × 32 (original panel) | 50,000 × 32; 256; 5 | 1.323–1.497 | 4.26–5.23× | 4.27–5.32× | 17.30–19.51× | 27 / 27 / 27 (each /27) |
| Gaussian 50k × 32 | 50,000 × 32; 256; 5 | 1.432–1.496 | 4.38–4.69× | 4.65–4.75× | 17.26–18.03× | 27 / 27 / 27 (each /27) |
| Clusters 50k × 32 | 50,000 × 32; 256; 5 | 2.695–2.747 | 2.42–2.76× | 2.38–2.55× | 9.39–9.59× | 27 / 27 / 27 (each /27) |
| Offset +1e4, 50k × 32 | 50,000 × 32; 256; 5 | 1.412–1.593 | 4.27–4.67× | 4.38–4.88× | 16.25–18.34× | 27 / 27 / 27 (each /27) |
| Unequal scales 50k × 32 | 50,000 × 32; 256; 5 | 1.743–1.812 | 3.85–4.09× | 3.87–4.04× | 14.25–14.83× | 27 / 27 / 27 (each /27) |
| Gaussian 4096 × 128 | 4,096 × 128; 128; 5 | 0.889–0.955 | 3.44–3.75× | 3.48–3.76× | 1.80–1.97× | 27 / 27 / 1 (each /27) |
| Clusters 4096 × 128 | 4,096 × 128; 128; 5 | 1.143–1.264 | 2.60–2.94× | 2.57–2.83× | 1.40–1.51× | 27 / 27 / 0 (each /27) |
| Offset +1e4, 4096 × 128 | 4,096 × 128; 128; 5 | 0.928–0.950 | 3.32–3.56× | 3.48–3.63× | 1.82–1.88× | 27 / 27 / 1 (each /27) |
| Unequal scales 4096 × 128 | 4,096 × 128; 128; 5 | 0.889–0.945 | 3.41–3.73× | 3.38–3.68× | 1.82–1.96× | 27 / 27 / 1 (each /27) |
| Gaussian 100k × 32 | 100,000 × 32; 256; 5 | 1.982–2.087 | 4.60–5.12× | 4.51–5.07× | 24.56–25.93× | 27 / 27 / 27 (each /27) |
| Digits (raw) | 1,541 × 64; 256; 5 | 1.015–1.041 | 2.77–2.96× | 2.73–3.21× | 1.11–1.14× | 27 / 27 / 0 (each /27) |
| Breast Cancer (raw) | 455 × 30; 114; 5 | 0.238–0.252 | 8.94–9.41× | 8.67–8.89× | 1.13–1.21× | 27 / 27 / 0 (each /27) |
| Iris (raw) | 120 × 4; 30; 5 | 0.090–0.093 | 6.49–8.47× | 2.03–2.37× | 1.34–1.56× | 27 / 17 / 0 (each /27) |
| Wine (raw) | 142 × 13; 36; 5 | 0.105–0.120 | 5.70–6.32× | 1.83–2.01× | 1.27–1.44× | 27 / 4 / 0 (each /27) |

## Fresh fit + query

| Family | Current fresh ms | vs default brute | vs auto | vs one thread |
| --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 0.212–0.242 | 3.36–3.87× | 6.57–7.82× | 1.40–1.59× |
| Gaussian 8D, k5 | 0.827–0.879 | 1.74–2.15× | 6.75–7.10× | 5.01–5.36× |
| Gaussian 32D, k5 | 1.122–1.155 | 3.71–3.94× | 3.85–4.04× | 3.97–4.11× |
| Gaussian 32D, k32 | 2.099–2.143 | 1.92–2.27× | 1.89–2.32× | 1.49–1.55× |
| Gaussian 128D, k5 | 1.208–1.282 | 2.81–3.21× | 2.84–3.24× | 1.43–1.57× |
| Gaussian 1024D, k5 | 1.846–1.890 | 2.10–2.16× | 2.06–2.11× | 1.36–1.41× |
| Gaussian 32D, k64 | 1.703–1.722 | 2.38–2.51× | 2.33–2.53× | 1.56–1.61× |
| Gaussian 16D, k5 | 0.991–1.023 | 3.82–3.84× | 3.78–3.90× | 4.37–4.52× |
| Gaussian 32D, single query | 0.302–0.305 | 9.83–10.06× | 9.57–10.19× | 1.24–1.26× |
| Gaussian 8D, k32 | 2.043–2.115 | 0.76–0.80× | 2.67–2.81× | 1.46–1.52× |
| Gaussian 50k × 32 (original panel) | 1.777–1.978 | 3.74–4.17× | 3.75–4.23× | 13.25–14.74× |
| Gaussian 50k × 32 | 1.927–1.964 | 3.58–3.72× | 3.69–3.75× | 13.33–13.64× |
| Clusters 50k × 32 | 3.027–3.129 | 2.16–2.33× | 2.20–2.41× | 8.34–8.65× |
| Offset +1e4, 50k × 32 | 2.088–2.157 | 3.42–3.58× | 3.38–3.40× | 12.12–12.55× |
| Unequal scales 50k × 32 | 2.134–2.264 | 3.21–3.52× | 3.28–3.52× | 11.56–12.25× |
| Gaussian 4096 × 128 | 1.275–1.299 | 2.62–2.68× | 2.63–2.68× | 1.44–1.48× |
| Clusters 4096 × 128 | 1.311–1.415 | 2.41–2.56× | 2.39–2.68× | 1.32–1.43× |
| Offset +1e4, 4096 × 128 | 1.220–1.291 | 2.61–2.82× | 2.58–2.78× | 1.46–1.56× |
| Unequal scales 4096 × 128 | 1.203–1.303 | 2.67–2.81× | 2.69–2.80× | 1.44–1.58× |
| Gaussian 100k × 32 | 2.736–2.809 | 3.55–3.93× | 3.63–3.79× | 18.45–18.94× |
| Digits (raw) | 1.171–1.198 | 2.53–2.80× | 2.50–3.04× | 1.05–1.06× |
| Breast Cancer (raw) | 0.333–0.353 | 6.36–6.53× | 6.28–6.81× | 1.01–1.03× |
| Iris (raw) | 0.152–0.157 | 3.96–4.51× | 1.85–1.98× | 1.26–1.31× |
| Wine (raw) | 0.160–0.178 | 3.96–4.54× | 1.84–1.89× | 1.22–1.32× |

## Observed automatic routing

Actual query backends, including fit-only's untimed accuracy query. Fit storage and execution policy are counted separately; no routing is inferred from timings.

| Phase | Requests | Query backends | Fit storage | Policies |
| --- | --- | --- | --- | --- |
| cold | 72 | {"cpu": 18, "mlx": 54} | {"cpu": 12, "mlx": 60} | {"auto": 72} |
| warmup | 3600 | {"cpu": 900, "mlx": 2700} | {"cpu": 600, "mlx": 3000} | {"auto": 3600} |
| retained | 10800 | {"cpu": 2700, "mlx": 8100} | {"cpu": 1800, "mlx": 9000} | {"auto": 10800} |

Per-family and per-operation routing counts are recorded in the validated summary.

## Every operation and paired call

Operation cells report median counts, then individual paired-call counts. Fit-only excludes first-query preparation and is not a substitute for fresh fit + query.

| Operation | Brute, default threads | Auto, default threads | Brute, one thread |
| --- | --- | --- | --- |
| `fit` (excluded from headline) | 0/72; 0/1,080 | 9/72; 138/1,080 | 0/72; 1/1,080 |
| `fit_query` | 66/72; 982/1,080 | 65/72; 980/1,080 | 27/72; 422/1,080 |
| `kneighbors` | 68/72; 1011/1,080 | 66/72; 981/1,080 | 31/72; 515/1,080 |
| `indices` | 66/72; 984/1,080 | 66/72; 972/1,080 | 31/72; 516/1,080 |
| `predict_uniform` | 65/72; 973/1,080 | 70/72; 1035/1,080 | 30/72; 522/1,080 |
| `proba_uniform` | 64/72; 962/1,080 | 64/72; 952/1,080 | 30/72; 523/1,080 |
| `regress_uniform` | 64/72; 962/1,080 | 62/72; 933/1,080 | 33/72; 513/1,080 |
| `predict_distance` | 68/72; 1013/1,080 | 66/72; 985/1,080 | 30/72; 527/1,080 |
| `proba_distance` | 66/72; 993/1,080 | 64/72; 949/1,080 | 28/72; 494/1,080 |
| `regress_distance` | 65/72; 987/1,080 | 62/72; 935/1,080 | 27/72; 475/1,080 |

Family-level paired-call counts retain every individual threshold miss:

| Family | Brute, default threads | Auto, default threads | Brute, one thread |
| --- | --- | --- | --- |
| Gaussian 8D, single query | 405/405 | 117/405 | 250/405 |
| Gaussian 8D, k5 | 182/405 | 405/405 | 405/405 |
| Gaussian 32D, k5 | 405/405 | 405/405 | 405/405 |
| Gaussian 32D, k32 | 238/405 | 242/405 | 23/405 |
| Gaussian 128D, k5 | 400/405 | 402/405 | 83/405 |
| Gaussian 1024D, k5 | 384/405 | 381/405 | 93/405 |
| Gaussian 32D, k64 | 392/405 | 391/405 | 64/405 |
| Gaussian 16D, k5 | 405/405 | 405/405 | 405/405 |
| Gaussian 32D, single query | 405/405 | 405/405 | 7/405 |
| Gaussian 8D, k32 | 2/405 | 380/405 | 4/405 |
| Gaussian 50k × 32 (original panel) | 405/405 | 405/405 | 405/405 |
| Gaussian 50k × 32 | 405/405 | 405/405 | 405/405 |
| Clusters 50k × 32 | 394/405 | 397/405 | 405/405 |
| Offset +1e4, 50k × 32 | 405/405 | 405/405 | 405/405 |
| Unequal scales 50k × 32 | 405/405 | 405/405 | 405/405 |
| Gaussian 4096 × 128 | 405/405 | 405/405 | 98/405 |
| Clusters 4096 × 128 | 400/405 | 401/405 | 11/405 |
| Offset +1e4, 4096 × 128 | 405/405 | 405/405 | 112/405 |
| Unequal scales 4096 × 128 | 404/405 | 403/405 | 105/405 |
| Gaussian 100k × 32 | 405/405 | 405/405 | 405/405 |
| Digits (raw) | 402/405 | 405/405 | 0/405 |
| Breast Cancer (raw) | 405/405 | 405/405 | 0/405 |
| Iris (raw) | 405/405 | 229/405 | 7/405 |
| Wine (raw) | 404/405 | 114/405 | 5/405 |

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

[Raw measurements and every audit](../benchmarks/results/knn-auto-final-2026-10-30-broad.json)

[Validated derived summary](../benchmarks/results/knn-auto-final-2026-10-30-summary.json)

[Controlled source comparison](../benchmarks/results/knn-auto-final-2026-10-30-control.json)

[Vector figure](assets/knn-auto-benchmarks-2026-10-30.svg)

Production manifest: `benchmarks/reference/knn-auto-final-2026-10-30.manifest.json`. Snapshot: `{"path": "benchmarks/reference/knn-auto-final-2026-10-30.zip", "sha256": "5805fe856856b07e451d807cab936db6f7a36f7b9b2cb8c228676f28bc66b2f6"}`. The raw report records source, loaded-worker, harness and fixture hashes before and after timing.

Generated from recorded measurements. No source optimization or performance claim is inferred from a separate historical run.
