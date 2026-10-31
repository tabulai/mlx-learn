# Small-workload controlled benchmark

Completed `2026-10-30T17:54:44.690226+00:00`. Acceptance: **True**. [Raw results](../benchmarks/results/knn-small-final-2026-10-30-control.json) (SHA-256 `f4c349f1b5b2bebd8884a6c8d65467d44561bde4a3892f7023ca5ae5aef2df9d`).

Three isolated public-API workers: current, identical-code A/A and frozen `knn-gaussian-fresh-final-2026-10-30.zip`. All GPU calls are serial. Each worker/operation gets 1 seconds of continuous warmup followed by 31 interleaved paired rounds. Every call includes public validation, NumPy boundaries and synchronization. Cold first-fit/query records come from persistent workers, excluding process startup.

The 18 small-dataset/single-query fixtures cover all nine search/inference operations. The 30 preservation fixtures cover query and fresh fit-plus-query, including the six expanded Gaussian seeds. Inputs, exact canonical neighbor ordering, distance tolerances and prediction oracles are unchanged. Every cold, warmup, probe and retained audit stays in the raw file, including failures. No result caching or production hooks are used.

All speedups are comparator/current time. Ranges span dataset seeds. Ratio of medians and median paired ratios are distinct descriptive statistics; neither is a confidence interval. A/A shows the corresponding process/timing variation. Values below 1× are slower observations and are retained.

Per-variant strict quality: `{"current": true, "current_b": true, "previous": true}`.

## Small datasets and single queries

| Family | Operation | Seeds | Current ms | Previous ms | Previous/current medians | Median paired speedup | A/A paired ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `shape_single_8_k5` | fit_query | 41, 42, 43 | 0.311–0.355 | 0.386–0.407 | 1.140–1.258× | 1.155–1.250× | 0.907–1.082× |
| `shape_single_8_k5` | kneighbors | 41, 42, 43 | 0.203–0.235 | 0.272–0.302 | 1.195–1.431× | 1.206–1.365× | 0.927–1.059× |
| `shape_single_8_k5` | indices | 41, 42, 43 | 0.182–0.228 | 0.240–0.292 | 1.281–1.319× | 1.270–1.354× | 0.934–1.071× |
| `shape_single_8_k5` | predict_uniform | 41, 42, 43 | 0.220–0.259 | 0.280–0.301 | 1.162–1.336× | 1.131–1.345× | 0.876–1.069× |
| `shape_single_8_k5` | proba_uniform | 41, 42, 43 | 0.247–0.288 | 0.295–0.336 | 1.166–1.351× | 1.160–1.272× | 0.936–1.037× |
| `shape_single_8_k5` | regress_uniform | 41, 42, 43 | 0.212–0.247 | 0.282–0.296 | 1.190–1.329× | 1.171–1.334× | 0.949–1.061× |
| `shape_single_8_k5` | predict_distance | 41, 42, 43 | 0.206–0.266 | 0.266–0.310 | 1.076–1.288× | 1.154–1.296× | 0.959–1.118× |
| `shape_single_8_k5` | proba_distance | 41, 42, 43 | 0.246–0.291 | 0.322–0.339 | 1.114–1.311× | 1.170–1.282× | 0.923–1.023× |
| `shape_single_8_k5` | regress_distance | 41, 42, 43 | 0.229–0.256 | 0.304–0.311 | 1.205–1.359× | 1.159–1.313× | 0.931–1.054× |
| `shape_single_32_k5` | fit_query | 49, 50, 51 | 0.339–0.418 | 0.412–0.459 | 1.098–1.217× | 1.115–1.218× | 0.950–1.049× |
| `shape_single_32_k5` | kneighbors | 49, 50, 51 | 0.201–0.274 | 0.261–0.309 | 1.088–1.298× | 1.139–1.285× | 0.959–1.020× |
| `shape_single_32_k5` | indices | 49, 50, 51 | 0.200–0.225 | 0.253–0.263 | 1.123–1.318× | 1.155–1.336× | 0.945–1.076× |
| `shape_single_32_k5` | predict_uniform | 49, 50, 51 | 0.259–0.305 | 0.301–0.360 | 1.108–1.215× | 1.081–1.232× | 0.918–1.059× |
| `shape_single_32_k5` | proba_uniform | 49, 50, 51 | 0.251–0.289 | 0.291–0.315 | 1.006–1.247× | 1.100–1.243× | 0.934–1.055× |
| `shape_single_32_k5` | regress_uniform | 49, 50, 51 | 0.223–0.235 | 0.264–0.297 | 1.148–1.262× | 1.165–1.311× | 0.951–1.061× |
| `shape_single_32_k5` | predict_distance | 49, 50, 51 | 0.255–0.260 | 0.288–0.316 | 1.111–1.241× | 1.143–1.237× | 0.938–1.023× |
| `shape_single_32_k5` | proba_distance | 49, 50, 51 | 0.248–0.300 | 0.311–0.348 | 1.141–1.262× | 1.154–1.266× | 0.930–1.059× |
| `shape_single_32_k5` | regress_distance | 49, 50, 51 | 0.235–0.277 | 0.293–0.360 | 1.086–1.402× | 1.124–1.271× | 0.929–1.075× |
| `real_digits_raw_k5` | fit_query | 151, 251, 252 | 0.477–0.506 | 0.634–0.718 | 1.321–1.418× | 1.342–1.421× | 0.955–1.044× |
| `real_digits_raw_k5` | kneighbors | 151, 251, 252 | 0.376–0.433 | 0.451–0.553 | 1.184–1.468× | 1.152–1.483× | 0.944–1.035× |
| `real_digits_raw_k5` | indices | 151, 251, 252 | 0.357–1.074 | 0.455–1.503 | 1.272–1.400× | 1.273–1.399× | 0.991–1.020× |
| `real_digits_raw_k5` | predict_uniform | 151, 251, 252 | 0.393–0.437 | 0.463–0.555 | 1.176–1.330× | 1.191–1.355× | 0.976–1.004× |
| `real_digits_raw_k5` | proba_uniform | 151, 251, 252 | 0.393–0.446 | 0.467–0.554 | 1.154–1.407× | 1.140–1.421× | 0.956–1.043× |
| `real_digits_raw_k5` | regress_uniform | 151, 251, 252 | 0.369–0.435 | 0.453–0.541 | 1.228–1.403× | 1.213–1.419× | 0.970–1.008× |
| `real_digits_raw_k5` | predict_distance | 151, 251, 252 | 0.393–0.433 | 0.469–0.552 | 1.195–1.347× | 1.193–1.334× | 0.984–1.001× |
| `real_digits_raw_k5` | proba_distance | 151, 251, 252 | 0.402–0.447 | 0.463–0.561 | 1.120–1.396× | 1.135–1.405× | 0.957–1.005× |
| `real_digits_raw_k5` | regress_distance | 151, 251, 252 | 0.397–1.023 | 0.543–1.041 | 1.018–1.376× | 1.114–1.378× | 0.984–0.988× |
| `real_breast_cancer_raw_k5` | fit_query | 151, 251, 252 | 0.306–0.326 | 0.406–0.417 | 1.279–1.326× | 1.261–1.384× | 0.978–1.038× |
| `real_breast_cancer_raw_k5` | kneighbors | 151, 251, 252 | 0.205–0.240 | 0.286–0.330 | 1.295–1.396× | 1.296–1.408× | 0.907–1.024× |
| `real_breast_cancer_raw_k5` | indices | 151, 251, 252 | 0.209–0.223 | 0.284–0.316 | 1.307–1.448× | 1.307–1.450× | 0.895–1.045× |
| `real_breast_cancer_raw_k5` | predict_uniform | 151, 251, 252 | 0.212–0.255 | 0.301–0.324 | 1.181–1.446× | 1.205–1.423× | 0.928–1.024× |
| `real_breast_cancer_raw_k5` | proba_uniform | 151, 251, 252 | 0.227–0.254 | 0.301–0.333 | 1.189–1.403× | 1.217–1.382× | 0.959–1.056× |
| `real_breast_cancer_raw_k5` | regress_uniform | 151, 251, 252 | 0.222–0.242 | 0.278–0.334 | 1.201–1.481× | 1.229–1.445× | 0.949–1.102× |
| `real_breast_cancer_raw_k5` | predict_distance | 151, 251, 252 | 0.207–0.243 | 0.275–0.322 | 1.161–1.339× | 1.145–1.347× | 0.935–1.054× |
| `real_breast_cancer_raw_k5` | proba_distance | 151, 251, 252 | 0.210–0.241 | 0.267–0.328 | 1.215–1.563× | 1.203–1.355× | 0.926–1.075× |
| `real_breast_cancer_raw_k5` | regress_distance | 151, 251, 252 | 0.238–0.252 | 0.312–0.362 | 1.268–1.436× | 1.277–1.455× | 0.933–1.047× |
| `real_iris_raw_k5` | fit_query | 151, 251, 252 | 0.345–0.361 | 0.393–0.416 | 1.131–1.202× | 1.134–1.202× | 0.943–1.034× |
| `real_iris_raw_k5` | kneighbors | 151, 251, 252 | 0.208–0.244 | 0.250–0.319 | 1.190–1.532× | 1.140–1.390× | 0.918–1.084× |
| `real_iris_raw_k5` | indices | 151, 251, 252 | 0.193–0.206 | 0.238–0.258 | 1.154–1.322× | 1.170–1.319× | 0.951–1.098× |
| `real_iris_raw_k5` | predict_uniform | 151, 251, 252 | 0.257–0.277 | 0.294–0.339 | 1.061–1.293× | 1.177–1.270× | 0.968–1.021× |
| `real_iris_raw_k5` | proba_uniform | 151, 251, 252 | 0.244–0.288 | 0.297–0.387 | 1.106–1.343× | 1.128–1.294× | 0.912–1.049× |
| `real_iris_raw_k5` | regress_uniform | 151, 251, 252 | 0.216–0.254 | 0.261–0.324 | 1.077–1.385× | 1.113–1.351× | 0.922–1.079× |
| `real_iris_raw_k5` | predict_distance | 151, 251, 252 | 0.218–0.247 | 0.279–0.301 | 1.192–1.280× | 1.191–1.277× | 0.937–1.094× |
| `real_iris_raw_k5` | proba_distance | 151, 251, 252 | 0.248–0.289 | 0.306–0.337 | 1.165–1.354× | 1.100–1.250× | 0.956–1.041× |
| `real_iris_raw_k5` | regress_distance | 151, 251, 252 | 0.231–0.284 | 0.285–0.336 | 1.001–1.447× | 1.153–1.341× | 0.931–1.056× |
| `real_wine_raw_k5` | fit_query | 151, 251, 252 | 0.306–0.326 | 0.362–0.388 | 1.146–1.271× | 1.082–1.310× | 0.882–1.075× |
| `real_wine_raw_k5` | kneighbors | 151, 251, 252 | 0.194–0.237 | 0.274–0.285 | 1.204–1.414× | 1.199–1.435× | 0.919–1.063× |
| `real_wine_raw_k5` | indices | 151, 251, 252 | 0.168–0.210 | 0.246–0.258 | 1.168–1.530× | 1.202–1.424× | 0.901–1.079× |
| `real_wine_raw_k5` | predict_uniform | 151, 251, 252 | 0.219–0.288 | 0.286–0.330 | 1.146–1.309× | 1.146–1.336× | 0.933–1.045× |
| `real_wine_raw_k5` | proba_uniform | 151, 251, 252 | 0.231–0.271 | 0.310–0.323 | 1.193–1.342× | 1.170–1.310× | 0.969–1.069× |
| `real_wine_raw_k5` | regress_uniform | 151, 251, 252 | 0.202–0.228 | 0.236–0.310 | 1.170–1.356× | 1.167–1.414× | 0.917–1.061× |
| `real_wine_raw_k5` | predict_distance | 151, 251, 252 | 0.203–0.238 | 0.254–0.312 | 1.130–1.312× | 1.142–1.314× | 0.923–1.075× |
| `real_wine_raw_k5` | proba_distance | 151, 251, 252 | 0.219–0.259 | 0.280–0.310 | 1.085–1.418× | 1.185–1.371× | 0.942–1.082× |
| `real_wine_raw_k5` | regress_distance | 151, 251, 252 | 0.225–0.236 | 0.267–0.308 | 1.187–1.341× | 1.151–1.373× | 0.943–1.035× |

## Gaussian, clustered, offset and unequal-scale preservation

| Family | Operation | Seeds | Current ms | Previous ms | Previous/current medians | Median paired speedup | A/A paired ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `distribution_gaussian_50000x32_k5` | kneighbors | 151, 251, 252, 351, 451, 552 | 0.412–2.334 | 0.431–2.408 | 1.032–1.093× | 1.034–1.089× | 0.953–1.010× |
| `distribution_gaussian_50000x32_k5` | fit_query | 151, 251, 252, 351, 451, 552 | 0.737–2.438 | 0.770–2.446 | 0.992–1.063× | 1.004–1.068× | 0.961–1.041× |
| `distribution_clusters_50000x32_k5` | kneighbors | 151, 251, 252 | 0.771–0.789 | 0.793–0.812 | 1.027–1.034× | 1.012–1.029× | 0.976–1.002× |
| `distribution_clusters_50000x32_k5` | fit_query | 151, 251, 252 | 1.062–1.115 | 1.109–1.138 | 1.021–1.049× | 0.999–1.056× | 0.971–0.999× |
| `distribution_offset_1e4_50000x32_k5` | kneighbors | 151, 251, 252 | 0.427–0.439 | 0.447–0.464 | 1.035–1.056× | 1.030–1.057× | 0.969–1.012× |
| `distribution_offset_1e4_50000x32_k5` | fit_query | 151, 251, 252 | 0.782–0.818 | 0.831–0.853 | 1.016–1.063× | 0.965–1.045× | 0.985–1.031× |
| `distribution_anisotropic_50000x32_k5` | kneighbors | 151, 251, 252 | 0.493–0.503 | 0.512–0.527 | 1.017–1.056× | 1.023–1.063× | 0.976–1.024× |
| `distribution_anisotropic_50000x32_k5` | fit_query | 151, 251, 252 | 0.799–0.835 | 0.827–0.855 | 0.991–1.058× | 1.024–1.059× | 0.981–1.020× |
| `distribution_gaussian_4096x128_k5` | kneighbors | 151, 251, 252, 351, 451, 552 | 0.373–0.605 | 0.379–0.634 | 0.984–1.162× | 1.010–1.189× | 0.942–1.081× |
| `distribution_gaussian_4096x128_k5` | fit_query | 151, 251, 252, 351, 451, 552 | 0.563–0.668 | 0.569–0.709 | 0.961–1.125× | 0.962–1.083× | 0.925–1.067× |
| `distribution_clusters_4096x128_k5` | kneighbors | 151, 251, 252 | 0.371–0.387 | 0.383–0.411 | 0.991–1.109× | 0.988–1.119× | 0.947–1.049× |
| `distribution_clusters_4096x128_k5` | fit_query | 151, 251, 252 | 0.514–0.558 | 0.533–0.597 | 0.961–1.087× | 0.959–1.088× | 0.968–1.038× |
| `distribution_offset_1e4_4096x128_k5` | kneighbors | 151, 251, 252 | 0.360–0.381 | 0.387–0.412 | 1.016–1.134× | 0.997–1.111× | 0.944–1.048× |
| `distribution_offset_1e4_4096x128_k5` | fit_query | 151, 251, 252 | 0.589–0.632 | 0.581–0.657 | 0.988–1.063× | 1.013–1.064× | 0.925–1.078× |
| `distribution_anisotropic_4096x128_k5` | kneighbors | 151, 251, 252 | 0.358–0.385 | 0.394–0.419 | 1.024–1.124× | 1.027–1.081× | 0.927–1.052× |
| `distribution_anisotropic_4096x128_k5` | fit_query | 151, 251, 252 | 0.596–0.863 | 0.622–0.933 | 1.044–1.080× | 0.995–1.059× | 0.966–1.013× |

The full sklearn panel is a separate run on its unchanged 72 fixtures and all nine operations. These controlled comparisons isolate source changes; absolute timings should not be compared across runs with different warmup protocols.
