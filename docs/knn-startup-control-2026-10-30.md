# Automatic CPU/GPU controlled benchmark

Completed `2026-10-30T23:56:12.055906+00:00`. Acceptance: **True**. [Raw results](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-7ddc36ba1227fd47229a) (SHA-256 `9b312857a241b40fd1184b428baba4c923ad909623e67d0bb57ce37beaa6033f`).

5 isolated public-API workers: current automatic dispatch, an identical automatic A/A worker, current forced GPU and frozen forced-GPU `knn-small-final-2026-10-30.zip`. An additional previous-auto worker uses the preceding source and exact ABI-matched archived native binary. All CPU/GPU calls are serial. Each worker/operation gets 0.25 seconds of continuous warmup followed by 31 interleaved paired rounds. Every call includes public validation, NumPy boundaries and synchronization. Cold first-fit/query records come from persistent workers, excluding process startup. Default continuous warmup is 0.25 seconds (the preceding small control used 1 second); this is configurable and absolute timings must not be compared across those runs. Each worker/operation warms continuously for at least the requested duration. Its one post-warmup probe is audited and retained separately, excluded from paired ratios. All warmup audits and probe samples remain in the report.

The 18 small-dataset/single-query fixtures cover all nine search/inference operations. The 30 preservation fixtures cover query and fresh fit-plus-query, including the six expanded Gaussian seeds. Inputs, exact canonical neighbor ordering, distance tolerances and prediction oracles are unchanged. Every cold, warmup, probe and retained audit stays in the raw file, including failures. No result caching or production hooks are used.

All speedups are comparator/current time. Ranges span dataset seeds. Ratio of medians and median paired ratios are distinct descriptive statistics; neither is a confidence interval. A/A shows the corresponding process/timing variation. Values below 1× are slower observations and are retained.

Per-variant strict quality: `{"current": true, "current_b": true, "current_gpu": true, "previous_auto": true, "previous_gpu": true}`.

## Retained request order and phase diagnostics

Order policy: **balanced**. Legacy alternates one order and its reverse. Balanced uses a Williams complete block: five workers have ten rows, balancing every position and each directed within-round predecessor twice per block. First 30 of 31 rounds are balanced; round 31 is retained. Cross-round predecessors and parent timing context are recorded separately. No clock state is inferred from these descriptive phase diagnostics.

The raw file keeps each request's position, within-round predecessor, actual previous request, and parent elapsed time since that worker's warmup probe. Each operation also contains all/first/middle/last and even/odd-round summaries, plus summaries by position and predecessor. These diagnostics do not remove rounds, infer clock state or establish statistical significance. Historical results without this field retain their original legacy order; absolute timings must not be compared as if the protocols were identical.

## Small datasets and single queries

| Family | Operation | Seeds | Auto ms | Current GPU ms | Previous GPU ms | Current GPU/auto medians | Current GPU/auto paired | Previous GPU/auto medians | Previous GPU/auto paired | Auto A/A paired |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shape_single_8_k5` | fit_query | 41, 42, 43 | 0.144–0.146 | 0.299–0.339 | 0.312–0.375 | 2.050–2.348× | 2.175–2.436× | 2.138–2.611× | 2.224–2.702× | 0.980–1.002× |
| `shape_single_8_k5` | kneighbors | 41, 42, 43 | 0.084–0.089 | 0.215–0.252 | 0.195–0.250 | 2.406–3.011× | 2.313–3.116× | 2.198–2.986× | 2.183–3.088× | 0.963–1.054× |
| `shape_single_8_k5` | indices | 41, 42, 43 | 0.083–0.086 | 0.246–0.259 | 0.250–0.268 | 2.915–3.135× | 3.028–3.284× | 2.918–3.203× | 3.043–3.511× | 0.980–1.051× |
| `shape_single_8_k5` | predict_uniform | 41, 42, 43 | 0.087–0.110 | 0.216–0.322 | 0.207–0.351 | 2.295–2.918× | 2.286–2.956× | 2.380–3.182× | 2.395–3.232× | 0.934–1.046× |
| `shape_single_8_k5` | proba_uniform | 41, 42, 43 | 0.089–0.097 | 0.233–0.293 | 0.247–0.282 | 2.507–3.279× | 2.569–3.512× | 2.663–3.145× | 2.677–3.397× | 0.969–1.064× |
| `shape_single_8_k5` | regress_uniform | 41, 42, 43 | 0.092–0.096 | 0.234–0.268 | 0.218–0.269 | 2.506–2.915× | 2.457–3.134× | 2.331–2.924× | 2.297–3.124× | 0.956–1.016× |
| `shape_single_8_k5` | predict_distance | 41, 42, 43 | 0.094–0.104 | 0.261–0.298 | 0.236–0.288 | 2.562–3.166× | 2.368–3.255× | 2.356–3.063× | 2.239–3.158× | 1.004–1.072× |
| `shape_single_8_k5` | proba_distance | 41, 42, 43 | 0.096–0.103 | 0.286–0.315 | 0.293–0.306 | 2.784–3.267× | 2.866–3.612× | 2.853–3.161× | 2.878–3.139× | 0.938–1.116× |
| `shape_single_8_k5` | regress_distance | 41, 42, 43 | 0.100–0.105 | 0.219–0.290 | 0.220–0.310 | 2.183–2.761× | 2.091–2.914× | 2.170–2.949× | 2.079–3.034× | 0.942–0.986× |
| `shape_single_32_k5` | fit_query | 49, 50, 51 | 0.209–0.217 | 0.374–0.408 | 0.410–0.431 | 1.727–1.952× | 1.795–2.082× | 1.885–2.059× | 1.890–2.175× | 0.981–1.027× |
| `shape_single_32_k5` | kneighbors | 49, 50, 51 | 0.133–0.137 | 0.307–0.316 | 0.315–0.333 | 2.274–2.366× | 2.303–2.425× | 2.298–2.496× | 2.375–2.585× | 0.984–1.017× |
| `shape_single_32_k5` | indices | 49, 50, 51 | 0.129–0.138 | 0.196–0.325 | 0.199–0.312 | 1.518–2.355× | 1.488–2.349× | 1.503–2.263× | 1.503–2.319× | 0.973–1.028× |
| `shape_single_32_k5` | predict_uniform | 49, 50, 51 | 0.138–0.141 | 0.238–0.353 | 0.299–0.353 | 1.682–2.568× | 1.709–2.649× | 2.116–2.567× | 1.876–2.809× | 0.969–1.051× |
| `shape_single_32_k5` | proba_uniform | 49, 50, 51 | 0.139–0.149 | 0.235–0.381 | 0.221–0.358 | 1.683–2.743× | 1.647–2.646× | 1.587–2.550× | 1.578–2.535× | 0.957–1.019× |
| `shape_single_32_k5` | regress_uniform | 49, 50, 51 | 0.138–0.153 | 0.220–0.327 | 0.233–0.311 | 1.505–2.366× | 1.532–2.350× | 1.524–2.248× | 1.504–2.242× | 0.950–1.055× |
| `shape_single_32_k5` | predict_distance | 49, 50, 51 | 0.149–0.160 | 0.336–0.371 | 0.350–0.366 | 2.094–2.493× | 2.122–2.533× | 2.182–2.445× | 2.158–2.457× | 0.976–1.058× |
| `shape_single_32_k5` | proba_distance | 49, 50, 51 | 0.151–0.164 | 0.234–0.372 | 0.232–0.371 | 1.550–2.262× | 1.513–2.381× | 1.512–2.257× | 1.523–2.364× | 0.933–0.992× |
| `shape_single_32_k5` | regress_distance | 49, 50, 51 | 0.144–0.154 | 0.225–0.342 | 0.240–0.329 | 1.463–2.353× | 1.467–2.440× | 1.565–2.232× | 1.533–2.285× | 1.001–1.050× |
| `real_digits_raw_k5` | fit_query | 151, 251, 252 | 0.427–0.544 | 0.818–1.088 | 1.008–1.101 | 1.503–2.432× | 1.621–2.317× | 1.880–2.362× | 1.854–2.369× | 0.846–1.089× |
| `real_digits_raw_k5` | kneighbors | 151, 251, 252 | 0.369–0.393 | 0.584–0.958 | 0.809–0.952 | 1.486–2.597× | 1.483–2.771× | 2.057–2.579× | 1.990–2.846× | 0.805–0.987× |
| `real_digits_raw_k5` | indices | 151, 251, 252 | 0.367–0.386 | 0.410–0.963 | 0.398–0.955 | 1.119–2.584× | 1.110–2.509× | 1.086–2.562× | 1.177–2.443× | 0.966–1.025× |
| `real_digits_raw_k5` | predict_uniform | 151, 251, 252 | 0.322–0.466 | 0.434–0.828 | 0.418–0.666 | 1.136–2.571× | 1.206–2.623× | 1.243–1.430× | 1.274–1.499× | 0.985–1.168× |
| `real_digits_raw_k5` | proba_uniform | 151, 251, 252 | 0.324–0.370 | 0.433–0.937 | 0.418–0.966 | 1.171–2.623× | 1.195–2.471× | 1.130–2.746× | 1.180–2.574× | 0.871–1.130× |
| `real_digits_raw_k5` | regress_uniform | 151, 251, 252 | 0.299–0.366 | 0.384–0.969 | 0.398–0.971 | 1.067–2.981× | 1.038–2.949× | 1.105–2.654× | 1.092–2.736× | 0.846–1.280× |
| `real_digits_raw_k5` | predict_distance | 151, 251, 252 | 0.388–0.405 | 0.445–0.805 | 0.437–0.917 | 1.101–1.990× | 1.171–1.801× | 1.079–2.268× | 1.198–2.120× | 0.858–0.933× |
| `real_digits_raw_k5` | proba_distance | 151, 251, 252 | 0.396–0.442 | 0.405–0.540 | 0.426–0.559 | 1.024–1.222× | 1.083–1.332× | 1.075–1.265× | 1.058–1.501× | 0.918–1.001× |
| `real_digits_raw_k5` | regress_distance | 151, 251, 252 | 0.348–0.397 | 0.491–0.931 | 0.465–0.926 | 1.253–2.675× | 1.326–2.301× | 1.186–2.660× | 1.229–2.244× | 0.995–1.205× |
| `real_breast_cancer_raw_k5` | fit_query | 151, 251, 252 | 0.193–0.226 | 0.298–0.431 | 0.322–0.392 | 1.463–2.238× | 1.483–2.244× | 1.621–1.995× | 1.660–2.032× | 0.976–1.127× |
| `real_breast_cancer_raw_k5` | kneighbors | 151, 251, 252 | 0.134–0.147 | 0.214–0.255 | 0.208–0.312 | 1.454–1.898× | 1.315–1.847× | 1.416–2.325× | 1.340–2.337× | 1.010–1.210× |
| `real_breast_cancer_raw_k5` | indices | 151, 251, 252 | 0.138–0.191 | 0.219–0.266 | 0.220–0.270 | 1.389–1.850× | 1.367–1.814× | 1.412–1.814× | 1.421–1.784× | 0.869–1.117× |
| `real_breast_cancer_raw_k5` | predict_uniform | 151, 251, 252 | 0.154–0.159 | 0.227–0.287 | 0.241–0.289 | 1.442–1.864× | 1.327–1.955× | 1.529–1.874× | 1.537–1.828× | 0.922–1.036× |
| `real_breast_cancer_raw_k5` | proba_uniform | 151, 251, 252 | 0.149–0.150 | 0.202–0.299 | 0.214–0.308 | 1.353–2.003× | 1.378–2.070× | 1.426–2.062× | 1.445–2.084× | 1.023–1.078× |
| `real_breast_cancer_raw_k5` | regress_uniform | 151, 251, 252 | 0.138–0.164 | 0.252–0.278 | 0.246–0.267 | 1.566–2.017× | 1.600–1.829× | 1.538–1.787× | 1.625–1.777× | 0.968–1.171× |
| `real_breast_cancer_raw_k5` | predict_distance | 151, 251, 252 | 0.151–0.172 | 0.245–0.302 | 0.227–0.340 | 1.533–1.851× | 1.471–1.973× | 1.421–1.978× | 1.475–2.127× | 0.947–1.143× |
| `real_breast_cancer_raw_k5` | proba_distance | 151, 251, 252 | 0.159–0.189 | 0.223–0.322 | 0.230–0.316 | 1.204–1.733× | 1.198–1.761× | 1.245–1.865× | 1.307–1.904× | 0.895–1.094× |
| `real_breast_cancer_raw_k5` | regress_distance | 151, 251, 252 | 0.166–0.180 | 0.240–0.293 | 0.250–0.315 | 1.452–1.675× | 1.421–1.756× | 1.481–1.802× | 1.462–1.750× | 0.962–1.031× |
| `real_iris_raw_k5` | fit_query | 151, 251, 252 | 0.093–0.096 | 0.317–0.343 | 0.366–0.382 | 3.290–3.704× | 3.242–4.126× | 3.807–4.065× | 3.745–4.269× | 0.900–1.072× |
| `real_iris_raw_k5` | kneighbors | 151, 251, 252 | 0.069–0.074 | 0.248–0.276 | 0.224–0.278 | 3.345–3.848× | 3.085–4.432× | 3.018–3.875× | 2.927–3.916× | 0.988–1.030× |
| `real_iris_raw_k5` | indices | 151, 251, 252 | 0.061–0.070 | 0.198–0.282 | 0.187–0.269 | 2.955–4.615× | 2.827–5.133× | 2.791–4.398× | 2.785–4.966× | 0.993–1.139× |
| `real_iris_raw_k5` | predict_uniform | 151, 251, 252 | 0.068–0.077 | 0.227–0.324 | 0.209–0.324 | 3.026–4.640× | 2.874–4.754× | 2.787–4.495× | 2.866–4.653× | 0.973–1.178× |
| `real_iris_raw_k5` | proba_uniform | 151, 251, 252 | 0.075–0.078 | 0.226–0.319 | 0.249–0.314 | 2.886–4.117× | 2.938–4.414× | 3.173–4.047× | 3.625–4.179× | 0.954–1.083× |
| `real_iris_raw_k5` | regress_uniform | 151, 251, 252 | 0.076–0.079 | 0.205–0.301 | 0.216–0.293 | 2.655–3.952× | 2.693–4.643× | 2.798–3.728× | 2.794–4.160× | 0.937–0.999× |
| `real_iris_raw_k5` | predict_distance | 151, 251, 252 | 0.086–0.104 | 0.211–0.323 | 0.226–0.314 | 2.437–3.103× | 2.440–3.165× | 2.598–3.014× | 2.684–3.121× | 0.956–1.029× |
| `real_iris_raw_k5` | proba_distance | 151, 251, 252 | 0.085–0.101 | 0.279–0.331 | 0.266–0.370 | 2.922–3.644× | 2.780–4.104× | 2.787–4.352× | 2.391–4.365× | 0.965–1.150× |
| `real_iris_raw_k5` | regress_distance | 151, 251, 252 | 0.087–0.088 | 0.246–0.344 | 0.223–0.312 | 2.819–3.918× | 2.730–4.578× | 2.559–3.539× | 2.493–4.047× | 0.924–1.065× |
| `real_wine_raw_k5` | fit_query | 151, 251, 252 | 0.096–0.108 | 0.283–0.314 | 0.295–0.361 | 2.608–3.229× | 2.596–3.328× | 2.719–3.462× | 2.805–3.506× | 0.920–1.050× |
| `real_wine_raw_k5` | kneighbors | 151, 251, 252 | 0.076–0.083 | 0.198–0.249 | 0.208–0.246 | 2.397–3.292× | 2.390–3.452× | 2.520–3.243× | 2.516–3.487× | 0.899–1.113× |
| `real_wine_raw_k5` | indices | 151, 251, 252 | 0.080–0.083 | 0.203–0.268 | 0.219–0.252 | 2.548–3.236× | 2.600–3.093× | 2.743–3.033× | 2.868–3.097× | 0.891–1.038× |
| `real_wine_raw_k5` | predict_uniform | 151, 251, 252 | 0.084–0.092 | 0.207–0.274 | 0.219–0.270 | 2.387–3.268× | 2.305–3.313× | 2.521–3.225× | 2.555–3.341× | 0.970–0.990× |
| `real_wine_raw_k5` | proba_uniform | 151, 251, 252 | 0.080–0.093 | 0.231–0.273 | 0.226–0.271 | 2.472–3.390× | 2.447–3.459× | 2.419–3.201× | 2.431–3.339× | 0.959–1.206× |
| `real_wine_raw_k5` | regress_uniform | 151, 251, 252 | 0.084–0.091 | 0.226–0.262 | 0.217–0.332 | 2.471–3.022× | 2.386–3.265× | 2.374–3.826× | 2.374–3.486× | 0.960–1.048× |
| `real_wine_raw_k5` | predict_distance | 151, 251, 252 | 0.092–0.097 | 0.206–0.299 | 0.234–0.329 | 2.125–3.161× | 2.167–3.125× | 2.406–3.473× | 2.386–3.240× | 0.926–1.070× |
| `real_wine_raw_k5` | proba_distance | 151, 251, 252 | 0.093–0.100 | 0.232–0.313 | 0.240–0.282 | 2.318–3.344× | 2.222–3.383× | 2.398–3.014× | 2.270–3.041× | 0.930–0.967× |
| `real_wine_raw_k5` | regress_distance | 151, 251, 252 | 0.095–0.105 | 0.227–0.260 | 0.226–0.265 | 2.363–2.654× | 2.381–2.862× | 2.156–2.700× | 2.215–2.782× | 0.937–1.020× |

## Gaussian, clustered, offset and unequal-scale preservation

| Family | Operation | Seeds | Auto ms | Current GPU ms | Previous GPU ms | Current GPU/auto medians | Current GPU/auto paired | Previous GPU/auto medians | Previous GPU/auto paired | Auto A/A paired |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `distribution_gaussian_50000x32_k5` | kneighbors | 151, 251, 252, 351, 451, 552 | 0.426–2.374 | 0.420–2.363 | 0.424–2.390 | 0.955–1.020× | 0.965–1.021× | 0.972–1.007× | 0.970–1.008× | 0.966–1.031× |
| `distribution_gaussian_50000x32_k5` | fit_query | 151, 251, 252, 351, 451, 552 | 0.762–2.456 | 0.758–2.437 | 0.792–2.488 | 0.966–1.012× | 0.976–1.018× | 1.013–1.067× | 1.016–1.061× | 0.991–1.027× |
| `distribution_clusters_50000x32_k5` | kneighbors | 151, 251, 252 | 0.754–0.815 | 0.750–0.798 | 0.741–0.795 | 0.920–1.058× | 0.913–1.055× | 0.943–1.044× | 0.962–1.023× | 0.987–1.005× |
| `distribution_clusters_50000x32_k5` | fit_query | 151, 251, 252 | 1.050–1.082 | 1.062–1.081 | 1.091–1.155 | 0.986–1.029× | 0.994–1.014× | 1.038–1.068× | 1.027–1.049× | 1.014–1.022× |
| `distribution_offset_1e4_50000x32_k5` | kneighbors | 151, 251, 252 | 0.446–0.506 | 0.433–0.482 | 0.439–0.494 | 0.905–0.993× | 0.945–0.979× | 0.923–1.016× | 0.965–0.999× | 0.973–0.994× |
| `distribution_offset_1e4_50000x32_k5` | fit_query | 151, 251, 252 | 0.831–0.857 | 0.807–0.834 | 0.865–0.870 | 0.970–0.985× | 0.960–0.993× | 1.010–1.044× | 0.999–1.056× | 0.980–0.994× |
| `distribution_anisotropic_50000x32_k5` | kneighbors | 151, 251, 252 | 0.502–1.504 | 0.489–1.499 | 0.505–1.508 | 0.967–1.013× | 0.974–1.005× | 0.998–1.019× | 0.994–1.011× | 0.981–1.010× |
| `distribution_anisotropic_50000x32_k5` | fit_query | 151, 251, 252 | 0.837–0.859 | 0.837–0.852 | 0.879–0.887 | 0.979–1.001× | 0.978–1.006× | 1.023–1.056× | 1.025–1.056× | 0.985–1.022× |
| `distribution_gaussian_4096x128_k5` | kneighbors | 151, 251, 252, 351, 451, 552 | 0.391–0.416 | 0.383–0.412 | 0.399–0.424 | 0.921–1.003× | 0.941–1.019× | 0.969–1.035× | 0.971–1.035× | 0.968–1.040× |
| `distribution_gaussian_4096x128_k5` | fit_query | 151, 251, 252, 351, 451, 552 | 0.604–0.665 | 0.605–0.642 | 0.643–0.711 | 0.940–1.034× | 0.925–1.025× | 1.016–1.125× | 0.998–1.106× | 0.974–1.027× |
| `distribution_clusters_4096x128_k5` | kneighbors | 151, 251, 252 | 0.386–0.388 | 0.381–0.387 | 0.380–0.395 | 0.982–1.002× | 0.996–1.018× | 0.985–1.021× | 0.971–1.011× | 1.003–1.025× |
| `distribution_clusters_4096x128_k5` | fit_query | 151, 251, 252 | 0.509–0.548 | 0.518–0.529 | 0.544–0.549 | 0.952–1.017× | 0.964–1.025× | 0.992–1.078× | 1.003–1.095× | 0.983–1.038× |
| `distribution_offset_1e4_4096x128_k5` | kneighbors | 151, 251, 252 | 0.395–0.635 | 0.371–0.616 | 0.395–0.636 | 0.939–0.997× | 0.946–0.995× | 0.998–1.035× | 0.998–1.014× | 0.965–1.005× |
| `distribution_offset_1e4_4096x128_k5` | fit_query | 151, 251, 252 | 0.615–0.653 | 0.572–0.639 | 0.641–0.681 | 0.931–0.979× | 0.941–0.982× | 1.039–1.064× | 1.026–1.055× | 0.971–0.980× |
| `distribution_anisotropic_4096x128_k5` | kneighbors | 151, 251, 252 | 0.393–0.405 | 0.384–0.413 | 0.393–0.422 | 0.951–1.052× | 0.967–1.045× | 0.973–1.046× | 0.988–1.027× | 0.985–1.076× |
| `distribution_anisotropic_4096x128_k5` | fit_query | 151, 251, 252 | 0.630–0.650 | 0.602–0.690 | 0.662–0.754 | 0.956–1.063× | 0.956–1.022× | 1.046–1.160× | 1.052–1.095× | 0.977–1.066× |

## Preceding automatic implementation

The prior automatic source and its exact native runtime run in an isolated worker. Speedups are previous-auto/current-auto time, with every seed and operation retained.

| Family | Operation | Seeds | Current auto ms | Previous auto ms | Median ratio | Paired ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `shape_single_8_k5` | fit_query | [41, 42, 43] | 0.144–0.146 | 0.165–0.174 | 1.150–1.192× | 1.149–1.239× |
| `shape_single_8_k5` | kneighbors | [41, 42, 43] | 0.084–0.089 | 0.092–0.094 | 1.039–1.099× | 1.004–1.104× |
| `shape_single_8_k5` | indices | [41, 42, 43] | 0.083–0.086 | 0.091–0.094 | 1.087–1.118× | 1.101–1.140× |
| `shape_single_8_k5` | predict_uniform | [41, 42, 43] | 0.087–0.110 | 0.093–0.122 | 1.034–1.105× | 1.051–1.093× |
| `shape_single_8_k5` | proba_uniform | [41, 42, 43] | 0.089–0.097 | 0.096–0.102 | 0.992–1.142× | 0.997–1.143× |
| `shape_single_8_k5` | regress_uniform | [41, 42, 43] | 0.092–0.096 | 0.095–0.100 | 0.990–1.081× | 0.998–1.106× |
| `shape_single_8_k5` | predict_distance | [41, 42, 43] | 0.094–0.104 | 0.106–0.115 | 1.063–1.125× | 1.062–1.177× |
| `shape_single_8_k5` | proba_distance | [41, 42, 43] | 0.096–0.103 | 0.103–0.109 | 1.001–1.115× | 0.997–1.109× |
| `shape_single_8_k5` | regress_distance | [41, 42, 43] | 0.100–0.105 | 0.109–0.110 | 1.033–1.093× | 1.066–1.164× |
| `shape_single_32_k5` | fit_query | [49, 50, 51] | 0.209–0.217 | 0.247–0.248 | 1.141–1.181× | 1.175–1.186× |
| `shape_single_32_k5` | kneighbors | [49, 50, 51] | 0.133–0.137 | 0.135–0.145 | 1.008–1.055× | 0.979–1.058× |
| `shape_single_32_k5` | indices | [49, 50, 51] | 0.129–0.138 | 0.136–0.143 | 1.022–1.107× | 1.023–1.071× |
| `shape_single_32_k5` | predict_uniform | [49, 50, 51] | 0.138–0.141 | 0.140–0.147 | 0.999–1.039× | 1.002–1.051× |
| `shape_single_32_k5` | proba_uniform | [49, 50, 51] | 0.139–0.149 | 0.144–0.151 | 1.015–1.051× | 0.978–1.055× |
| `shape_single_32_k5` | regress_uniform | [49, 50, 51] | 0.138–0.153 | 0.146–0.154 | 0.999–1.071× | 0.981–1.113× |
| `shape_single_32_k5` | predict_distance | [49, 50, 51] | 0.149–0.160 | 0.155–0.168 | 1.015–1.132× | 0.978–1.148× |
| `shape_single_32_k5` | proba_distance | [49, 50, 51] | 0.151–0.164 | 0.156–0.165 | 1.001–1.078× | 1.014–1.031× |
| `shape_single_32_k5` | regress_distance | [49, 50, 51] | 0.144–0.154 | 0.150–0.159 | 1.008–1.048× | 0.989–1.079× |
| `real_digits_raw_k5` | fit_query | [151, 251, 252] | 0.427–0.544 | 0.571–0.621 | 1.141–1.338× | 1.167–1.316× |
| `real_digits_raw_k5` | kneighbors | [151, 251, 252] | 0.369–0.393 | 0.348–0.456 | 0.909–1.175× | 0.948–1.194× |
| `real_digits_raw_k5` | indices | [151, 251, 252] | 0.367–0.386 | 0.394–0.446 | 1.074–1.156× | 1.066–1.186× |
| `real_digits_raw_k5` | predict_uniform | [151, 251, 252] | 0.322–0.466 | 0.401–0.527 | 1.052–1.370× | 1.044–1.414× |
| `real_digits_raw_k5` | proba_uniform | [151, 251, 252] | 0.324–0.370 | 0.396–0.459 | 1.071–1.415× | 1.030–1.337× |
| `real_digits_raw_k5` | regress_uniform | [151, 251, 252] | 0.299–0.366 | 0.420–0.431 | 1.167–1.406× | 1.156–1.434× |
| `real_digits_raw_k5` | predict_distance | [151, 251, 252] | 0.388–0.405 | 0.395–0.461 | 0.977–1.140× | 0.965–1.160× |
| `real_digits_raw_k5` | proba_distance | [151, 251, 252] | 0.396–0.442 | 0.403–0.498 | 0.961–1.153× | 0.961–1.153× |
| `real_digits_raw_k5` | regress_distance | [151, 251, 252] | 0.348–0.397 | 0.358–0.457 | 0.900–1.258× | 0.892–1.271× |
| `real_breast_cancer_raw_k5` | fit_query | [151, 251, 252] | 0.193–0.226 | 0.266–0.284 | 1.178–1.473× | 1.207–1.492× |
| `real_breast_cancer_raw_k5` | kneighbors | [151, 251, 252] | 0.134–0.147 | 0.184–0.193 | 1.251–1.443× | 1.282–1.471× |
| `real_breast_cancer_raw_k5` | indices | [151, 251, 252] | 0.138–0.191 | 0.205–0.227 | 1.185–1.489× | 1.188–1.477× |
| `real_breast_cancer_raw_k5` | predict_uniform | [151, 251, 252] | 0.154–0.159 | 0.187–0.200 | 1.172–1.271× | 1.158–1.285× |
| `real_breast_cancer_raw_k5` | proba_uniform | [151, 251, 252] | 0.149–0.150 | 0.202–0.204 | 1.349–1.362× | 1.308–1.361× |
| `real_breast_cancer_raw_k5` | regress_uniform | [151, 251, 252] | 0.138–0.164 | 0.187–0.224 | 1.191–1.433× | 1.143–1.379× |
| `real_breast_cancer_raw_k5` | predict_distance | [151, 251, 252] | 0.151–0.172 | 0.211–0.220 | 1.237–1.457× | 1.267–1.455× |
| `real_breast_cancer_raw_k5` | proba_distance | [151, 251, 252] | 0.159–0.189 | 0.214–0.237 | 1.159–1.366× | 1.164–1.351× |
| `real_breast_cancer_raw_k5` | regress_distance | [151, 251, 252] | 0.166–0.180 | 0.196–0.218 | 1.089–1.317× | 1.117–1.313× |
| `real_iris_raw_k5` | fit_query | [151, 251, 252] | 0.093–0.096 | 0.120–0.139 | 1.274–1.463× | 1.248–1.450× |
| `real_iris_raw_k5` | kneighbors | [151, 251, 252] | 0.069–0.074 | 0.071–0.078 | 0.991–1.112× | 0.964–1.119× |
| `real_iris_raw_k5` | indices | [151, 251, 252] | 0.061–0.070 | 0.069–0.073 | 1.006–1.189× | 0.961–1.227× |
| `real_iris_raw_k5` | predict_uniform | [151, 251, 252] | 0.068–0.077 | 0.077–0.085 | 1.033–1.200× | 1.024–1.207× |
| `real_iris_raw_k5` | proba_uniform | [151, 251, 252] | 0.075–0.078 | 0.085–0.092 | 1.087–1.216× | 1.073–1.152× |
| `real_iris_raw_k5` | regress_uniform | [151, 251, 252] | 0.076–0.079 | 0.079–0.087 | 0.999–1.142× | 0.943–1.077× |
| `real_iris_raw_k5` | predict_distance | [151, 251, 252] | 0.086–0.104 | 0.095–0.102 | 0.983–1.096× | 0.993–1.138× |
| `real_iris_raw_k5` | proba_distance | [151, 251, 252] | 0.085–0.101 | 0.089–0.093 | 0.881–1.087× | 0.884–1.076× |
| `real_iris_raw_k5` | regress_distance | [151, 251, 252] | 0.087–0.088 | 0.087–0.095 | 0.991–1.081× | 0.971–1.081× |
| `real_wine_raw_k5` | fit_query | [151, 251, 252] | 0.096–0.108 | 0.131–0.137 | 1.267–1.358× | 1.220–1.370× |
| `real_wine_raw_k5` | kneighbors | [151, 251, 252] | 0.076–0.083 | 0.087–0.089 | 1.044–1.174× | 1.044–1.239× |
| `real_wine_raw_k5` | indices | [151, 251, 252] | 0.080–0.083 | 0.083–0.085 | 1.005–1.044× | 0.994–1.056× |
| `real_wine_raw_k5` | predict_uniform | [151, 251, 252] | 0.084–0.092 | 0.082–0.095 | 0.979–1.038× | 0.981–1.046× |
| `real_wine_raw_k5` | proba_uniform | [151, 251, 252] | 0.080–0.093 | 0.085–0.095 | 0.975–1.129× | 0.966–1.164× |
| `real_wine_raw_k5` | regress_uniform | [151, 251, 252] | 0.084–0.091 | 0.092–0.095 | 1.005–1.131× | 0.985–1.158× |
| `real_wine_raw_k5` | predict_distance | [151, 251, 252] | 0.092–0.097 | 0.098–0.103 | 1.039–1.115× | 1.045–1.099× |
| `real_wine_raw_k5` | proba_distance | [151, 251, 252] | 0.093–0.100 | 0.097–0.102 | 0.965–1.070× | 0.952–1.066× |
| `real_wine_raw_k5` | regress_distance | [151, 251, 252] | 0.095–0.105 | 0.096–0.105 | 0.973–1.111× | 0.976–1.152× |
| `distribution_gaussian_50000x32_k5` | kneighbors | [151, 251, 252, 351, 451, 552] | 0.426–2.374 | 0.426–2.367 | 0.975–1.012× | 0.963–1.013× |
| `distribution_gaussian_50000x32_k5` | fit_query | [151, 251, 252, 351, 451, 552] | 0.762–2.456 | 0.795–2.480 | 1.010–1.063× | 1.004–1.068× |
| `distribution_clusters_50000x32_k5` | kneighbors | [151, 251, 252] | 0.754–0.815 | 0.770–0.801 | 0.980–1.045× | 0.985–1.021× |
| `distribution_clusters_50000x32_k5` | fit_query | [151, 251, 252] | 1.050–1.082 | 1.088–1.167 | 1.020–1.079× | 1.032–1.061× |
| `distribution_offset_1e4_50000x32_k5` | kneighbors | [151, 251, 252] | 0.446–0.506 | 0.450–0.483 | 0.944–1.009× | 0.979–1.019× |
| `distribution_offset_1e4_50000x32_k5` | fit_query | [151, 251, 252] | 0.831–0.857 | 0.873–0.886 | 1.031–1.052× | 1.018–1.043× |
| `distribution_anisotropic_50000x32_k5` | kneighbors | [151, 251, 252] | 0.502–1.504 | 0.498–1.510 | 0.985–1.028× | 1.002–1.025× |
| `distribution_anisotropic_50000x32_k5` | fit_query | [151, 251, 252] | 0.837–0.859 | 0.886–0.904 | 1.052–1.059× | 1.037–1.071× |
| `distribution_gaussian_4096x128_k5` | kneighbors | [151, 251, 252, 351, 451, 552] | 0.391–0.416 | 0.395–0.423 | 0.950–1.034× | 0.979–1.044× |
| `distribution_gaussian_4096x128_k5` | fit_query | [151, 251, 252, 351, 451, 552] | 0.604–0.665 | 0.624–0.705 | 1.035–1.106× | 1.029–1.114× |
| `distribution_clusters_4096x128_k5` | kneighbors | [151, 251, 252] | 0.386–0.388 | 0.387–0.392 | 0.999–1.015× | 0.993–1.022× |
| `distribution_clusters_4096x128_k5` | fit_query | [151, 251, 252] | 0.509–0.548 | 0.561–0.568 | 1.033–1.102× | 1.033–1.120× |
| `distribution_offset_1e4_4096x128_k5` | kneighbors | [151, 251, 252] | 0.395–0.635 | 0.396–0.625 | 0.985–1.058× | 0.973–1.003× |
| `distribution_offset_1e4_4096x128_k5` | fit_query | [151, 251, 252] | 0.615–0.653 | 0.620–0.690 | 1.004–1.069× | 0.986–1.054× |
| `distribution_anisotropic_4096x128_k5` | kneighbors | [151, 251, 252] | 0.393–0.405 | 0.397–0.423 | 0.982–1.077× | 1.002–1.045× |
| `distribution_anisotropic_4096x128_k5` | fit_query | [151, 251, 252] | 0.630–0.650 | 0.665–0.759 | 1.057–1.168× | 1.049–1.129× |

## Observed backend routing

Counts include every warmup, probe and retained query, with cold calls separate. Fit storage is reported independently of actual query execution.

| Variant | Phase | Requests | Query backends | Fit storage |
| --- | --- | ---: | --- | --- |
| current | cold | 48 | `{"cpu": 18, "mlx": 30}` | `{"cpu": 18, "mlx": 30}` |
| current | warmup | 281130 | `{"cpu": 259439, "mlx": 21691}` | `{"cpu": 259439, "mlx": 21691}` |
| current | probe | 222 | `{"cpu": 162, "mlx": 60}` | `{"cpu": 162, "mlx": 60}` |
| current | retained | 6882 | `{"cpu": 5022, "mlx": 1860}` | `{"cpu": 5022, "mlx": 1860}` |
| current_b | cold | 48 | `{"cpu": 18, "mlx": 30}` | `{"cpu": 18, "mlx": 30}` |
| current_b | warmup | 279450 | `{"cpu": 257793, "mlx": 21657}` | `{"cpu": 257793, "mlx": 21657}` |
| current_b | probe | 222 | `{"cpu": 162, "mlx": 60}` | `{"cpu": 162, "mlx": 60}` |
| current_b | retained | 6882 | `{"cpu": 5022, "mlx": 1860}` | `{"cpu": 5022, "mlx": 1860}` |
| current_gpu | cold | 48 | `{"mlx": 48}` | `{"mlx": 48}` |
| current_gpu | warmup | 162217 | `{"mlx": 162217}` | `{"mlx": 162217}` |
| current_gpu | probe | 222 | `{"mlx": 222}` | `{"mlx": 222}` |
| current_gpu | retained | 6882 | `{"mlx": 6882}` | `{"mlx": 6882}` |
| previous_auto | cold | 48 | `{"cpu": 18, "mlx": 30}` | `{"cpu": 18, "mlx": 30}` |
| previous_auto | warmup | 258933 | `{"cpu": 237636, "mlx": 21297}` | `{"cpu": 237636, "mlx": 21297}` |
| previous_auto | probe | 222 | `{"cpu": 162, "mlx": 60}` | `{"cpu": 162, "mlx": 60}` |
| previous_auto | retained | 6882 | `{"cpu": 5022, "mlx": 1860}` | `{"cpu": 5022, "mlx": 1860}` |
| previous_gpu | cold | 48 | `{"mlx": 48}` | `{"mlx": 48}` |
| previous_gpu | warmup | 162301 | `{"mlx": 162301}` | `{"mlx": 162301}` |
| previous_gpu | probe | 222 | `{"mlx": 222}` | `{"mlx": 222}` |
| previous_gpu | retained | 6882 | `{"mlx": 6882}` | `{"mlx": 6882}` |

The full sklearn panel is a separate run on its unchanged 72 fixtures and all nine operations. These controlled comparisons isolate source changes; absolute timings should not be compared across runs with different warmup protocols.
