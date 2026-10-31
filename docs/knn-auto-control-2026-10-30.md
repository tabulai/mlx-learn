# Automatic CPU/GPU controlled benchmark

Completed `2026-10-30T20:13:40.843561+00:00`. Acceptance: **True**. [Raw results](../benchmarks/results/knn-auto-final-2026-10-30-control.json) (SHA-256 `6caf1e937417d6690f084e5faabd8d3bb33f0010becb520e1791f333b20f41af`).

Four isolated public-API workers: current automatic dispatch, an identical automatic A/A worker, current forced GPU and frozen forced-GPU `knn-small-final-2026-10-30.zip`. All CPU/GPU calls are serial. Each worker/operation gets 0.25 seconds of continuous warmup followed by 31 interleaved paired rounds. Every call includes public validation, NumPy boundaries and synchronization. Cold first-fit/query records come from persistent workers, excluding process startup. Default continuous warmup is 0.25 seconds (the preceding small control used 1 second); this is configurable and absolute timings must not be compared across those runs. Each worker/operation warms continuously for at least the requested duration. Its one post-warmup probe is audited and retained separately, excluded from paired ratios. All warmup audits and probe samples remain in the report.

The 18 small-dataset/single-query fixtures cover all nine search/inference operations. The 30 preservation fixtures cover query and fresh fit-plus-query, including the six expanded Gaussian seeds. Inputs, exact canonical neighbor ordering, distance tolerances and prediction oracles are unchanged. Every cold, warmup, probe and retained audit stays in the raw file, including failures. No result caching or production hooks are used.

All speedups are comparator/current time. Ranges span dataset seeds. Ratio of medians and median paired ratios are distinct descriptive statistics; neither is a confidence interval. A/A shows the corresponding process/timing variation. Values below 1× are slower observations and are retained.

Per-variant strict quality: `{"current": true, "current_b": true, "current_gpu": true, "previous_gpu": true}`.

## Small datasets and single queries

| Family | Operation | Seeds | Auto ms | Current GPU ms | Previous GPU ms | Current GPU/auto medians | Current GPU/auto paired | Previous GPU/auto medians | Previous GPU/auto paired | Auto A/A paired |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shape_single_8_k5` | fit_query | 41, 42, 43 | 0.184–0.205 | 0.341–0.396 | 0.314–0.418 | 1.785–1.949× | 1.805–2.110× | 1.648–2.038× | 1.651–1.946× | 0.902–1.023× |
| `shape_single_8_k5` | kneighbors | 41, 42, 43 | 0.085–0.099 | 0.217–0.275 | 0.226–0.260 | 2.204–3.216× | 2.122–3.246× | 2.290–2.952× | 2.201–3.110× | 0.968–1.112× |
| `shape_single_8_k5` | indices | 41, 42, 43 | 0.086–0.091 | 0.213–0.283 | 0.187–0.330 | 2.347–3.174× | 2.336–3.596× | 2.062–3.833× | 2.084–4.380× | 0.961–1.017× |
| `shape_single_8_k5` | predict_uniform | 41, 42, 43 | 0.089–0.101 | 0.205–0.292 | 0.223–0.300 | 2.036–3.296× | 2.111–3.389× | 2.221–3.378× | 2.300–3.172× | 0.977–1.114× |
| `shape_single_8_k5` | proba_uniform | 41, 42, 43 | 0.094–0.101 | 0.217–0.296 | 0.202–0.298 | 2.189–3.156× | 2.195–3.206× | 2.043–3.176× | 2.080–3.429× | 0.974–1.020× |
| `shape_single_8_k5` | regress_uniform | 41, 42, 43 | 0.089–0.099 | 0.202–0.274 | 0.212–0.270 | 2.079–3.068× | 2.091–3.153× | 2.182–2.789× | 2.222–3.056× | 1.031–1.138× |
| `shape_single_8_k5` | predict_distance | 41, 42, 43 | 0.102–0.104 | 0.238–0.367 | 0.218–0.484 | 2.304–3.514× | 2.341–3.217× | 2.107–4.634× | 1.973–4.537× | 0.947–1.007× |
| `shape_single_8_k5` | proba_distance | 41, 42, 43 | 0.098–0.109 | 0.202–0.299 | 0.224–0.302 | 1.920–3.042× | 1.872–3.330× | 2.088–3.075× | 2.076–3.150× | 0.989–1.150× |
| `shape_single_8_k5` | regress_distance | 41, 42, 43 | 0.102–0.112 | 0.231–0.347 | 0.217–0.283 | 2.153–3.385× | 2.119–3.181× | 1.982–2.767× | 1.953–2.767× | 0.985–1.024× |
| `shape_single_32_k5` | fit_query | 49, 50, 51 | 0.244–0.282 | 0.406–0.538 | 0.425–0.488 | 1.584–1.907× | 1.598–1.871× | 1.657–1.825× | 1.641–1.798× | 0.960–1.046× |
| `shape_single_32_k5` | kneighbors | 49, 50, 51 | 0.140–0.167 | 0.315–0.343 | 0.235–0.311 | 1.918–2.398× | 2.022–2.323× | 1.405–2.193× | 1.441–2.193× | 0.962–0.973× |
| `shape_single_32_k5` | indices | 49, 50, 51 | 0.133–0.147 | 0.196–0.307 | 0.206–0.297 | 1.370–2.312× | 1.386–2.454× | 1.468–2.237× | 1.455–2.370× | 0.974–1.027× |
| `shape_single_32_k5` | predict_uniform | 49, 50, 51 | 0.140–0.148 | 0.224–0.371 | 0.223–0.376 | 1.520–2.656× | 1.508–2.704× | 1.517–2.690× | 1.491–2.566× | 0.966–1.020× |
| `shape_single_32_k5` | proba_uniform | 49, 50, 51 | 0.144–0.153 | 0.221–0.354 | 0.239–0.360 | 1.537–2.462× | 1.511–2.559× | 1.659–2.401× | 1.616–2.434× | 0.957–1.036× |
| `shape_single_32_k5` | regress_uniform | 49, 50, 51 | 0.144–0.151 | 0.233–0.349 | 0.214–0.363 | 1.544–2.416× | 1.486–2.351× | 1.422–2.511× | 1.418–2.559× | 0.922–0.997× |
| `shape_single_32_k5` | predict_distance | 49, 50, 51 | 0.148–0.160 | 0.230–0.350 | 0.243–0.342 | 1.462–2.368× | 1.446–2.446× | 1.540–2.308× | 1.420–2.343× | 0.992–1.120× |
| `shape_single_32_k5` | proba_distance | 49, 50, 51 | 0.156–0.189 | 0.250–0.414 | 0.230–0.381 | 1.599–2.565× | 1.544–2.507× | 1.375–2.361× | 1.425–2.316× | 0.929–0.997× |
| `shape_single_32_k5` | regress_distance | 49, 50, 51 | 0.152–0.156 | 0.238–0.337 | 0.268–0.329 | 1.566–2.161× | 1.545–2.314× | 1.765–2.143× | 1.657–2.198× | 0.977–1.112× |
| `real_digits_raw_k5` | fit_query | 151, 251, 252 | 1.096–1.123 | 1.129–1.281 | 1.047–1.397 | 1.013–1.146× | 1.016–1.145× | 0.939–1.244× | 0.876–1.192× | 0.989–0.993× |
| `real_digits_raw_k5` | kneighbors | 151, 251, 252 | 0.882–0.917 | 0.839–0.975 | 0.428–0.928 | 0.942–1.063× | 0.967–1.009× | 0.480–1.012× | 0.482–1.016× | 1.004–1.014× |
| `real_digits_raw_k5` | indices | 151, 251, 252 | 0.872–0.893 | 0.495–0.868 | 0.386–0.943 | 0.554–0.995× | 0.568–1.013× | 0.432–1.069× | 0.438–1.060× | 1.001–1.016× |
| `real_digits_raw_k5` | predict_uniform | 151, 251, 252 | 0.907–0.912 | 0.415–1.041 | 0.446–0.980 | 0.457–1.141× | 0.469–1.142× | 0.491–1.074× | 0.490–1.117× | 0.982–1.007× |
| `real_digits_raw_k5` | proba_uniform | 151, 251, 252 | 0.904–0.930 | 0.484–1.029 | 0.401–1.016 | 0.520–1.138× | 0.525–1.118× | 0.432–1.123× | 0.427–1.066× | 0.989–1.005× |
| `real_digits_raw_k5` | regress_uniform | 151, 251, 252 | 0.891–0.930 | 0.485–0.930 | 0.431–0.846 | 0.522–1.044× | 0.510–1.039× | 0.465–0.949× | 0.468–0.957× | 0.978–1.012× |
| `real_digits_raw_k5` | predict_distance | 151, 251, 252 | 0.906–1.033 | 0.917–1.084 | 0.944–1.057 | 1.007–1.050× | 1.002–1.022× | 1.009–1.092× | 0.993–1.113× | 0.993–1.030× |
| `real_digits_raw_k5` | proba_distance | 151, 251, 252 | 0.913–0.949 | 0.412–0.950 | 0.848–0.912 | 0.435–1.015× | 0.440–0.928× | 0.894–0.999× | 0.916–0.979× | 0.977–1.017× |
| `real_digits_raw_k5` | regress_distance | 151, 251, 252 | 0.936–0.951 | 0.495–0.980 | 0.418–1.041 | 0.526–1.030× | 0.525–1.066× | 0.446–1.095× | 0.462–1.045× | 0.988–1.009× |
| `real_breast_cancer_raw_k5` | fit_query | 151, 251, 252 | 0.285–0.318 | 0.314–0.420 | 0.332–0.621 | 1.038–1.473× | 1.039–1.587× | 1.099–2.179× | 1.130–1.960× | 0.979–1.038× |
| `real_breast_cancer_raw_k5` | kneighbors | 151, 251, 252 | 0.186–0.197 | 0.222–0.259 | 0.216–0.476 | 1.195–1.379× | 1.197–1.440× | 1.097–2.533× | 1.097–2.029× | 0.969–1.080× |
| `real_breast_cancer_raw_k5` | indices | 151, 251, 252 | 0.165–0.194 | 0.195–0.235 | 0.187–0.235 | 1.111–1.425× | 1.080–1.464× | 0.965–1.426× | 0.951–1.416× | 1.066–1.163× |
| `real_breast_cancer_raw_k5` | predict_uniform | 151, 251, 252 | 0.177–0.189 | 0.252–0.300 | 0.231–0.447 | 1.336–1.593× | 1.323–1.571× | 1.225–2.369× | 1.195–2.332× | 1.017–1.123× |
| `real_breast_cancer_raw_k5` | proba_uniform | 151, 251, 252 | 0.172–0.197 | 0.209–0.324 | 0.224–0.470 | 1.064–1.725× | 1.023–1.683× | 1.137–2.503× | 1.127–2.576× | 1.004–1.186× |
| `real_breast_cancer_raw_k5` | regress_uniform | 151, 251, 252 | 0.189–0.200 | 0.221–0.286 | 0.209–0.263 | 1.106–1.516× | 1.104–1.487× | 1.099–1.393× | 1.100–1.353× | 0.991–1.105× |
| `real_breast_cancer_raw_k5` | predict_distance | 151, 251, 252 | 0.200–0.215 | 0.212–0.274 | 0.202–0.277 | 1.022–1.372× | 1.042–1.370× | 0.978–1.385× | 0.962–1.394× | 1.006–1.162× |
| `real_breast_cancer_raw_k5` | proba_distance | 151, 251, 252 | 0.199–0.220 | 0.232–0.286 | 0.208–0.288 | 1.058–1.434× | 1.089–1.374× | 0.947–1.350× | 0.954–1.385× | 1.005–1.119× |
| `real_breast_cancer_raw_k5` | regress_distance | 151, 251, 252 | 0.196–0.212 | 0.237–0.298 | 0.241–0.466 | 1.135–1.403× | 1.098–1.405× | 1.157–2.196× | 1.115–1.892× | 0.976–1.112× |
| `real_iris_raw_k5` | fit_query | 151, 251, 252 | 0.117–0.168 | 0.377–0.429 | 0.350–0.428 | 2.475–3.228× | 2.701–3.344× | 2.084–3.040× | 2.214–3.193× | 0.856–1.083× |
| `real_iris_raw_k5` | kneighbors | 151, 251, 252 | 0.068–0.078 | 0.202–0.289 | 0.208–0.283 | 2.704–3.905× | 2.729–4.299× | 2.791–3.795× | 2.763–4.139× | 0.921–1.117× |
| `real_iris_raw_k5` | indices | 151, 251, 252 | 0.069–0.072 | 0.202–0.265 | 0.189–0.254 | 2.885–3.843× | 2.875–4.047× | 2.703–3.687× | 2.737–3.983× | 0.924–0.974× |
| `real_iris_raw_k5` | predict_uniform | 151, 251, 252 | 0.070–0.083 | 0.210–0.310 | 0.224–0.312 | 2.600–4.425× | 2.592–4.701× | 2.774–4.486× | 2.884–4.737× | 1.005–1.152× |
| `real_iris_raw_k5` | proba_uniform | 151, 251, 252 | 0.077–0.091 | 0.235–0.322 | 0.231–0.374 | 2.580–4.115× | 2.528–4.734× | 2.532–4.706× | 2.330–4.906× | 0.934–1.142× |
| `real_iris_raw_k5` | regress_uniform | 151, 251, 252 | 0.078–0.089 | 0.205–0.292 | 0.210–0.289 | 2.440–3.754× | 2.493–3.992× | 2.496–3.722× | 2.508–3.825× | 0.952–1.143× |
| `real_iris_raw_k5` | predict_distance | 151, 251, 252 | 0.094–0.098 | 0.225–0.318 | 0.202–0.386 | 2.288–3.388× | 2.284–3.621× | 2.054–4.109× | 2.199–3.952× | 0.924–1.012× |
| `real_iris_raw_k5` | proba_distance | 151, 251, 252 | 0.084–0.102 | 0.224–0.306 | 0.230–0.310 | 2.231–3.621× | 2.147–3.613× | 2.257–3.673× | 2.199–3.623× | 0.928–1.099× |
| `real_iris_raw_k5` | regress_distance | 151, 251, 252 | 0.087–0.096 | 0.227–0.311 | 0.210–0.302 | 2.359–3.584× | 2.399–4.262× | 2.179–3.477× | 2.132–3.715× | 0.976–1.248× |
| `real_wine_raw_k5` | fit_query | 151, 251, 252 | 0.160–0.177 | 0.338–0.377 | 0.322–0.359 | 1.913–2.364× | 1.904–2.798× | 1.934–2.252× | 1.975–2.516× | 0.973–1.116× |
| `real_wine_raw_k5` | kneighbors | 151, 251, 252 | 0.077–0.085 | 0.230–0.262 | 0.215–0.261 | 2.759–3.244× | 2.586–3.496× | 2.588–3.055× | 2.408–3.250× | 0.964–1.103× |
| `real_wine_raw_k5` | indices | 151, 251, 252 | 0.074–0.099 | 0.216–0.268 | 0.218–0.244 | 2.392–3.051× | 2.300–3.374× | 2.477–2.933× | 2.494–3.136× | 0.980–1.090× |
| `real_wine_raw_k5` | predict_uniform | 151, 251, 252 | 0.084–0.106 | 0.215–0.281 | 0.202–0.276 | 2.083–3.333× | 2.254–3.297× | 2.180–3.276× | 2.182–3.103× | 0.964–1.069× |
| `real_wine_raw_k5` | proba_uniform | 151, 251, 252 | 0.092–0.141 | 0.200–0.389 | 0.208–0.413 | 2.180–2.765× | 2.181–3.297× | 2.087–2.934× | 2.112–3.199× | 0.948–1.029× |
| `real_wine_raw_k5` | regress_uniform | 151, 251, 252 | 0.091–0.094 | 0.206–0.271 | 0.187–0.253 | 2.262–2.879× | 2.284–2.947× | 2.049–2.687× | 1.990–2.737× | 0.953–1.168× |
| `real_wine_raw_k5` | predict_distance | 151, 251, 252 | 0.091–0.107 | 0.215–0.286 | 0.233–0.300 | 2.014–3.139× | 1.939–3.444× | 2.179–3.029× | 2.172–2.961× | 0.964–1.178× |
| `real_wine_raw_k5` | proba_distance | 151, 251, 252 | 0.101–0.109 | 0.198–0.288 | 0.212–0.308 | 1.951–2.699× | 1.958–2.792× | 1.949–2.888× | 1.900–2.840× | 0.933–1.007× |
| `real_wine_raw_k5` | regress_distance | 151, 251, 252 | 0.091–0.109 | 0.203–0.258 | 0.205–0.243 | 1.974–2.849× | 2.023–3.310× | 1.993–2.685× | 2.028–2.822× | 1.002–1.162× |

## Gaussian, clustered, offset and unequal-scale preservation

| Family | Operation | Seeds | Auto ms | Current GPU ms | Previous GPU ms | Current GPU/auto medians | Current GPU/auto paired | Previous GPU/auto medians | Previous GPU/auto paired | Auto A/A paired |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `distribution_gaussian_50000x32_k5` | kneighbors | 151, 251, 252, 351, 451, 552 | 0.545–3.266 | 0.496–3.184 | 0.530–3.224 | 0.911–1.012× | 0.972–1.039× | 0.972–1.010× | 0.947–1.022× | 0.984–1.025× |
| `distribution_gaussian_50000x32_k5` | fit_query | 151, 251, 252, 351, 451, 552 | 0.791–2.472 | 0.783–2.455 | 0.784–2.457 | 0.986–1.056× | 0.999–1.021× | 0.984–1.047× | 0.942–1.020× | 0.970–1.012× |
| `distribution_clusters_50000x32_k5` | kneighbors | 151, 251, 252 | 1.033–1.500 | 1.056–1.485 | 1.064–1.463 | 0.990–1.022× | 0.971–1.022× | 0.975–1.030× | 0.988–1.023× | 0.976–1.025× |
| `distribution_clusters_50000x32_k5` | fit_query | 151, 251, 252 | 1.074–1.125 | 1.077–1.097 | 1.087–1.109 | 0.972–1.022× | 0.976–1.010× | 0.979–1.032× | 0.982–1.032× | 0.987–1.001× |
| `distribution_offset_1e4_50000x32_k5` | kneighbors | 151, 251, 252 | 0.515–0.615 | 0.550–0.610 | 0.550–0.613 | 0.992–1.068× | 0.982–1.021× | 0.984–1.069× | 0.985–1.001× | 0.979–1.008× |
| `distribution_offset_1e4_50000x32_k5` | fit_query | 151, 251, 252 | 0.839–0.859 | 0.817–0.851 | 0.819–0.849 | 0.951–1.014× | 0.958–1.016× | 0.954–1.008× | 0.902–1.015× | 0.991–1.075× |
| `distribution_anisotropic_50000x32_k5` | kneighbors | 151, 251, 252 | 0.675–1.030 | 0.671–1.069 | 0.666–1.012 | 0.994–1.038× | 0.984–1.013× | 0.983–1.020× | 1.001–1.009× | 0.988–1.010× |
| `distribution_anisotropic_50000x32_k5` | fit_query | 151, 251, 252 | 0.827–0.878 | 0.844–0.856 | 0.830–0.854 | 0.961–1.029× | 0.949–1.045× | 0.945–1.033× | 0.913–1.025× | 0.996–1.087× |
| `distribution_gaussian_4096x128_k5` | kneighbors | 151, 251, 252, 351, 451, 552 | 0.420–0.505 | 0.433–0.486 | 0.425–0.495 | 0.962–1.052× | 0.992–1.041× | 0.979–1.011× | 0.994–1.018× | 0.979–1.005× |
| `distribution_gaussian_4096x128_k5` | fit_query | 151, 251, 252, 351, 451, 552 | 0.632–0.665 | 0.626–0.664 | 0.630–0.681 | 0.953–1.037× | 0.928–1.036× | 0.966–1.073× | 0.893–1.055× | 0.974–1.031× |
| `distribution_clusters_4096x128_k5` | kneighbors | 151, 251, 252 | 0.382–0.491 | 0.432–0.489 | 0.411–0.505 | 0.997–1.131× | 0.997–1.073× | 0.948–1.077× | 0.982–1.017× | 1.021–1.095× |
| `distribution_clusters_4096x128_k5` | fit_query | 151, 251, 252 | 0.547–0.568 | 0.547–0.574 | 0.530–0.560 | 1.002–1.026× | 0.990–1.038× | 0.969–1.012× | 0.976–1.013× | 0.975–1.043× |
| `distribution_offset_1e4_4096x128_k5` | kneighbors | 151, 251, 252 | 0.455–0.457 | 0.465–0.477 | 0.462–0.478 | 1.019–1.046× | 1.009–1.049× | 1.011–1.050× | 0.987–1.050× | 0.968–1.020× |
| `distribution_offset_1e4_4096x128_k5` | fit_query | 151, 251, 252 | 0.635–0.687 | 0.626–0.698 | 0.637–0.662 | 0.977–1.042× | 0.992–1.008× | 0.963–1.026× | 0.990–1.030× | 0.987–1.035× |
| `distribution_anisotropic_4096x128_k5` | kneighbors | 151, 251, 252 | 0.449–0.467 | 0.457–0.461 | 0.441–0.478 | 0.988–1.019× | 0.942–0.999× | 0.983–1.025× | 1.008–1.010× | 0.983–1.014× |
| `distribution_anisotropic_4096x128_k5` | fit_query | 151, 251, 252 | 0.616–0.756 | 0.631–0.734 | 0.618–0.683 | 0.971–1.062× | 0.962–1.065× | 0.903–1.080× | 0.954–1.037× | 0.983–1.012× |

## Observed backend routing

Counts include every warmup, probe and retained query, with cold calls separate. Fit storage is reported independently of actual query execution.

| Variant | Phase | Requests | Query backends | Fit storage |
| --- | --- | ---: | --- | --- |
| current | cold | 48 | `{"cpu": 18, "mlx": 30}` | `{"cpu": 12, "mlx": 36}` |
| current | warmup | 247589 | `{"cpu": 225759, "mlx": 21830}` | `{"cpu": 140767, "mlx": 106822}` |
| current | probe | 222 | `{"cpu": 162, "mlx": 60}` | `{"cpu": 108, "mlx": 114}` |
| current | retained | 6882 | `{"cpu": 5022, "mlx": 1860}` | `{"cpu": 3348, "mlx": 3534}` |
| current_b | cold | 48 | `{"cpu": 18, "mlx": 30}` | `{"cpu": 12, "mlx": 36}` |
| current_b | warmup | 244893 | `{"cpu": 223054, "mlx": 21839}` | `{"cpu": 138665, "mlx": 106228}` |
| current_b | probe | 222 | `{"cpu": 162, "mlx": 60}` | `{"cpu": 108, "mlx": 114}` |
| current_b | retained | 6882 | `{"cpu": 5022, "mlx": 1860}` | `{"cpu": 3348, "mlx": 3534}` |
| current_gpu | cold | 48 | `{"mlx": 48}` | `{"mlx": 48}` |
| current_gpu | warmup | 160799 | `{"mlx": 160799}` | `{"mlx": 160799}` |
| current_gpu | probe | 222 | `{"mlx": 222}` | `{"mlx": 222}` |
| current_gpu | retained | 6882 | `{"mlx": 6882}` | `{"mlx": 6882}` |
| previous_gpu | cold | 48 | `{"mlx": 48}` | `{"mlx": 48}` |
| previous_gpu | warmup | 167423 | `{"mlx": 167423}` | `{"mlx": 167423}` |
| previous_gpu | probe | 222 | `{"mlx": 222}` | `{"mlx": 222}` |
| previous_gpu | retained | 6882 | `{"mlx": 6882}` | `{"mlx": 6882}` |

The full sklearn panel is a separate run on its unchanged 72 fixtures and all nine operations. These controlled comparisons isolate source changes; absolute timings should not be compared across runs with different warmup protocols.
