# Automatic CPU/GPU controlled benchmark

Completed `2026-10-31T01:32:53.506535+00:00`. Acceptance: **True**. [Raw results](../benchmarks/results/knn-digits-final-2026-10-30-control.json) (SHA-256 `5b965f5f149b02c37f7522bcf9a277f9b2e00855cac76643000ea9a45f8144fe`).

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
| `shape_single_8_k5` | fit_query | 41, 42, 43 | 0.134–0.145 | 0.278–0.357 | 0.284–0.397 | 1.922–2.651× | 1.895–2.693× | 1.963–2.954× | 1.912–3.112× | 0.947–1.002× |
| `shape_single_8_k5` | kneighbors | 41, 42, 43 | 0.080–0.087 | 0.193–0.258 | 0.184–0.252 | 2.228–3.211× | 2.207–3.623× | 2.120–3.137× | 2.075–3.502× | 0.960–1.067× |
| `shape_single_8_k5` | indices | 41, 42, 43 | 0.078–0.085 | 0.253–0.525 | 0.249–0.513 | 2.968–6.755× | 3.262–6.406× | 2.931–6.329× | 3.213–5.392× | 0.986–1.028× |
| `shape_single_8_k5` | predict_uniform | 41, 42, 43 | 0.086–0.092 | 0.219–0.291 | 0.209–0.286 | 2.416–3.405× | 2.477–3.395× | 2.278–3.349× | 2.420–3.383× | 0.938–1.021× |
| `shape_single_8_k5` | proba_uniform | 41, 42, 43 | 0.085–0.092 | 0.220–0.297 | 0.240–0.375 | 2.381–3.488× | 2.426–3.516× | 2.603–4.248× | 2.627–3.928× | 0.973–1.034× |
| `shape_single_8_k5` | regress_uniform | 41, 42, 43 | 0.085–0.097 | 0.208–0.442 | 0.195–0.515 | 2.279–5.183× | 2.305–5.418× | 2.132–6.033× | 2.117–5.647× | 0.988–0.992× |
| `shape_single_8_k5` | predict_distance | 41, 42, 43 | 0.087–0.101 | 0.207–0.301 | 0.206–0.300 | 2.140–3.292× | 2.191–4.017× | 2.032–3.275× | 2.067–3.642× | 0.945–1.071× |
| `shape_single_8_k5` | proba_distance | 41, 42, 43 | 0.095–0.098 | 0.438–0.533 | 0.293–0.457 | 4.588–5.442× | 4.065–5.324× | 3.030–4.668× | 3.240–4.626× | 0.983–1.075× |
| `shape_single_8_k5` | regress_distance | 41, 42, 43 | 0.097–0.099 | 0.219–0.304 | 0.211–0.267 | 2.210–3.100× | 2.212–3.170× | 2.128–2.725× | 2.151–2.770× | 0.947–1.001× |
| `shape_single_32_k5` | fit_query | 49, 50, 51 | 0.204–0.212 | 0.320–0.442 | 0.342–0.450 | 1.518–2.168× | 1.524–2.278× | 1.614–2.204× | 1.665–2.308× | 0.978–1.023× |
| `shape_single_32_k5` | kneighbors | 49, 50, 51 | 0.131–0.139 | 0.327–0.564 | 0.315–0.542 | 2.411–4.316× | 2.432–4.152× | 2.275–4.143× | 2.418–4.024× | 0.972–1.040× |
| `shape_single_32_k5` | indices | 49, 50, 51 | 0.128–0.137 | 0.202–0.521 | 0.199–0.503 | 1.583–3.826× | 1.545–3.581× | 1.557–3.698× | 1.591–3.637× | 0.982–1.004× |
| `shape_single_32_k5` | predict_uniform | 49, 50, 51 | 0.134–0.140 | 0.227–0.335 | 0.234–0.330 | 1.621–2.465× | 1.641–2.430× | 1.671–2.454× | 1.757–2.518× | 0.994–1.076× |
| `shape_single_32_k5` | proba_uniform | 49, 50, 51 | 0.135–0.148 | 0.232–0.405 | 0.215–0.421 | 1.713–2.868× | 1.689–2.616× | 1.586–2.981× | 1.600–2.848× | 0.992–1.012× |
| `shape_single_32_k5` | regress_uniform | 49, 50, 51 | 0.146–0.151 | 0.222–0.350 | 0.205–0.394 | 1.469–2.405× | 1.473–2.416× | 1.353–2.706× | 1.408–2.819× | 0.955–1.010× |
| `shape_single_32_k5` | predict_distance | 49, 50, 51 | 0.142–0.150 | 0.329–0.648 | 0.353–0.450 | 2.190–4.561× | 2.170–4.311× | 2.351–3.166× | 2.372–2.983× | 0.979–1.039× |
| `shape_single_32_k5` | proba_distance | 49, 50, 51 | 0.144–0.160 | 0.223–0.407 | 0.231–0.401 | 1.543–2.548× | 1.539–2.349× | 1.558–2.508× | 1.564–2.608× | 0.947–1.016× |
| `shape_single_32_k5` | regress_distance | 49, 50, 51 | 0.146–0.157 | 0.237–0.582 | 0.265–0.359 | 1.508–3.989× | 1.520–3.621× | 1.684–2.458× | 1.631–2.255× | 0.957–1.088× |
| `real_digits_raw_k5` | fit_query | 151, 251, 252 | 0.363–0.473 | 0.716–1.060 | 0.996–1.154 | 1.512–2.922× | 1.519–2.861× | 2.287–2.744× | 2.299–2.763× | 0.900–1.226× |
| `real_digits_raw_k5` | kneighbors | 151, 251, 252 | 0.315–0.337 | 0.445–0.868 | 0.449–0.915 | 1.378–2.757× | 1.455–2.610× | 1.388–2.908× | 1.424–2.638× | 0.858–1.011× |
| `real_digits_raw_k5` | indices | 151, 251, 252 | 0.293–0.332 | 0.386–0.544 | 0.396–0.440 | 1.249–1.658× | 1.251–1.761× | 1.194–1.355× | 1.363–1.460× | 0.921–0.965× |
| `real_digits_raw_k5` | predict_uniform | 151, 251, 252 | 0.291–0.346 | 0.448–0.477 | 0.435–0.474 | 1.354–1.602× | 1.364–1.737× | 1.316–1.630× | 1.303–1.649× | 0.984–1.267× |
| `real_digits_raw_k5` | proba_uniform | 151, 251, 252 | 0.251–0.369 | 0.458–0.966 | 0.435–0.985 | 1.241–3.844× | 1.266–3.567× | 1.178–3.662× | 1.232–3.140× | 0.856–1.345× |
| `real_digits_raw_k5` | regress_uniform | 151, 251, 252 | 0.260–0.331 | 0.406–0.970 | 0.409–0.461 | 1.229–3.051× | 1.296–2.965× | 1.245–1.576× | 1.374–1.656× | 0.776–1.290× |
| `real_digits_raw_k5` | predict_distance | 151, 251, 252 | 0.319–0.373 | 0.449–0.914 | 0.469–0.981 | 1.202–2.862× | 1.301–2.811× | 1.296–3.072× | 1.506–3.169× | 0.889–0.942× |
| `real_digits_raw_k5` | proba_distance | 151, 251, 252 | 0.345–0.362 | 0.404–0.812 | 0.389–0.937 | 1.158–2.245× | 1.175–2.159× | 1.114–2.592× | 1.115–2.592× | 0.926–1.005× |
| `real_digits_raw_k5` | regress_distance | 151, 251, 252 | 0.289–0.361 | 0.472–1.031 | 0.434–0.876 | 1.308–3.562× | 1.327–3.033× | 1.203–2.446× | 1.238–2.621× | 0.946–1.233× |
| `real_breast_cancer_raw_k5` | fit_query | 151, 251, 252 | 0.178–0.217 | 0.301–0.320 | 0.315–0.373 | 1.452–1.713× | 1.470–1.728× | 1.519–1.838× | 1.435–1.900× | 0.895–1.056× |
| `real_breast_cancer_raw_k5` | kneighbors | 151, 251, 252 | 0.129–0.148 | 0.226–0.504 | 0.222–0.432 | 1.529–3.741× | 1.536–3.513× | 1.500–3.204× | 1.570–3.119× | 0.961–1.177× |
| `real_breast_cancer_raw_k5` | indices | 151, 251, 252 | 0.121–0.139 | 0.204–0.489 | 0.218–0.486 | 1.460–3.787× | 1.498–3.709× | 1.566–3.760× | 1.567–3.524× | 0.913–1.211× |
| `real_breast_cancer_raw_k5` | predict_uniform | 151, 251, 252 | 0.140–0.161 | 0.219–0.514 | 0.251–0.539 | 1.439–3.676× | 1.465–3.555× | 1.604–3.855× | 1.625–3.613× | 0.879–0.935× |
| `real_breast_cancer_raw_k5` | proba_uniform | 151, 251, 252 | 0.155–0.168 | 0.234–0.247 | 0.218–0.259 | 1.402–1.563× | 1.363–1.766× | 1.413–1.638× | 1.428–1.839× | 0.905–0.993× |
| `real_breast_cancer_raw_k5` | regress_uniform | 151, 251, 252 | 0.144–0.156 | 0.214–0.288 | 0.198–0.286 | 1.369–1.938× | 1.397–1.939× | 1.268–1.924× | 1.300–1.815× | 0.964–0.990× |
| `real_breast_cancer_raw_k5` | predict_distance | 151, 251, 252 | 0.158–0.163 | 0.216–0.285 | 0.203–0.280 | 1.327–1.793× | 1.307–1.824× | 1.249–1.777× | 1.280–1.811× | 0.951–1.204× |
| `real_breast_cancer_raw_k5` | proba_distance | 151, 251, 252 | 0.165–0.167 | 0.207–0.338 | 0.218–0.515 | 1.255–2.029× | 1.222–1.811× | 1.323–3.088× | 1.361–2.702× | 0.898–1.170× |
| `real_breast_cancer_raw_k5` | regress_distance | 151, 251, 252 | 0.163–0.170 | 0.226–0.534 | 0.223–0.515 | 1.388–3.145× | 1.364–2.664× | 1.367–3.032× | 1.409–2.755× | 0.895–0.992× |
| `real_iris_raw_k5` | fit_query | 151, 251, 252 | 0.090–0.098 | 0.310–0.514 | 0.344–0.595 | 3.153–5.703× | 3.134–5.193× | 3.502–6.613× | 3.755–5.609× | 0.897–1.019× |
| `real_iris_raw_k5` | kneighbors | 151, 251, 252 | 0.063–0.074 | 0.197–0.271 | 0.200–0.263 | 2.950–4.297× | 2.941–4.088× | 2.749–4.177× | 2.837–4.367× | 0.936–1.010× |
| `real_iris_raw_k5` | indices | 151, 251, 252 | 0.064–0.072 | 0.195–0.284 | 0.185–0.266 | 3.025–4.468× | 3.109–4.991× | 2.870–4.181× | 3.091–4.476× | 0.997–1.056× |
| `real_iris_raw_k5` | predict_uniform | 151, 251, 252 | 0.070–0.088 | 0.237–0.595 | 0.224–0.585 | 3.210–8.442× | 3.117–7.661× | 3.037–8.306× | 3.247–7.446× | 0.940–1.038× |
| `real_iris_raw_k5` | proba_uniform | 151, 251, 252 | 0.067–0.079 | 0.216–0.558 | 0.224–0.535 | 3.064–7.103× | 2.969–6.693× | 3.177–6.807× | 3.248–6.023× | 0.948–1.092× |
| `real_iris_raw_k5` | regress_uniform | 151, 251, 252 | 0.074–0.075 | 0.197–0.297 | 0.204–0.286 | 2.645–4.030× | 2.599–4.393× | 2.735–3.884× | 2.731–4.469× | 0.965–0.982× |
| `real_iris_raw_k5` | predict_distance | 151, 251, 252 | 0.085–0.094 | 0.211–0.358 | 0.215–0.322 | 2.408–3.802× | 2.468–3.944× | 2.535–3.418× | 2.495–3.566× | 0.975–1.036× |
| `real_iris_raw_k5` | proba_distance | 151, 251, 252 | 0.079–0.097 | 0.220–0.317 | 0.218–0.325 | 2.365–3.736× | 2.455–3.878× | 2.336–3.335× | 2.341–3.788× | 0.970–1.110× |
| `real_iris_raw_k5` | regress_distance | 151, 251, 252 | 0.082–0.101 | 0.273–0.340 | 0.242–0.326 | 2.689–4.143× | 2.404–4.039× | 2.384–3.909× | 2.337–3.878× | 0.889–1.126× |
| `real_wine_raw_k5` | fit_query | 151, 251, 252 | 0.101–0.108 | 0.274–0.320 | 0.302–0.358 | 2.527–3.026× | 2.535–3.182× | 2.786–3.539× | 2.755–3.507× | 0.940–1.016× |
| `real_wine_raw_k5` | kneighbors | 151, 251, 252 | 0.075–0.080 | 0.190–0.499 | 0.204–0.500 | 2.380–6.455× | 2.591–6.365× | 2.559–6.467× | 2.632–5.872× | 0.919–1.128× |
| `real_wine_raw_k5` | indices | 151, 251, 252 | 0.071–0.076 | 0.176–0.499 | 0.184–0.490 | 2.437–7.043× | 2.430–6.663× | 2.545–6.918× | 2.503–6.418× | 0.943–1.016× |
| `real_wine_raw_k5` | predict_uniform | 151, 251, 252 | 0.078–0.114 | 0.198–0.273 | 0.209–0.271 | 2.393–2.956× | 2.473–2.937× | 2.228–2.950× | 2.320–3.150× | 0.965–1.051× |
| `real_wine_raw_k5` | proba_uniform | 151, 251, 252 | 0.082–0.090 | 0.217–0.296 | 0.207–0.287 | 2.655–3.286× | 2.737–3.348× | 2.531–3.188× | 2.637–3.310× | 0.975–1.058× |
| `real_wine_raw_k5` | regress_uniform | 151, 251, 252 | 0.077–0.086 | 0.196–0.507 | 0.186–0.510 | 2.286–6.107× | 2.319–6.078× | 2.178–6.418× | 2.204–5.939× | 0.942–1.089× |
| `real_wine_raw_k5` | predict_distance | 151, 251, 252 | 0.086–0.091 | 0.203–0.293 | 0.210–0.465 | 2.362–3.228× | 2.324–3.462× | 2.440–5.205× | 2.432–5.036× | 0.960–1.104× |
| `real_wine_raw_k5` | proba_distance | 151, 251, 252 | 0.093–0.108 | 0.216–0.284 | 0.251–0.276 | 2.314–2.730× | 2.345–3.082× | 2.436–2.698× | 2.526–2.814× | 0.934–0.965× |
| `real_wine_raw_k5` | regress_distance | 151, 251, 252 | 0.093–0.097 | 0.227–0.273 | 0.232–0.307 | 2.413–2.879× | 2.390–2.871× | 2.398–3.235× | 2.406–3.031× | 0.992–1.012× |

## Gaussian, clustered, offset and unequal-scale preservation

| Family | Operation | Seeds | Auto ms | Current GPU ms | Previous GPU ms | Current GPU/auto medians | Current GPU/auto paired | Previous GPU/auto medians | Previous GPU/auto paired | Auto A/A paired |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `distribution_gaussian_50000x32_k5` | kneighbors | 151, 251, 252, 351, 451, 552 | 0.428–2.396 | 0.417–2.379 | 0.427–2.410 | 0.919–0.998× | 0.966–1.009× | 0.988–1.006× | 0.968–1.003× | 0.988–1.005× |
| `distribution_gaussian_50000x32_k5` | fit_query | 151, 251, 252, 351, 451, 552 | 0.750–2.424 | 0.748–2.418 | 0.793–2.482 | 0.967–1.015× | 0.958–1.012× | 1.024–1.071× | 1.005–1.062× | 0.977–1.020× |
| `distribution_clusters_50000x32_k5` | kneighbors | 151, 251, 252 | 0.748–0.796 | 0.779–0.802 | 0.738–0.794 | 0.982–1.072× | 0.983–1.069× | 0.929–1.000× | 0.981–0.991× | 1.000–1.026× |
| `distribution_clusters_50000x32_k5` | fit_query | 151, 251, 252 | 1.059–2.169 | 1.056–1.930 | 1.085–2.206 | 0.890–0.999× | 0.981–1.013× | 1.015–1.067× | 1.020–1.056× | 0.974–1.021× |
| `distribution_offset_1e4_50000x32_k5` | kneighbors | 151, 251, 252 | 0.449–0.782 | 0.449–0.791 | 0.451–0.790 | 0.951–1.011× | 0.937–0.999× | 0.938–1.010× | 0.970–0.991× | 0.947–0.982× |
| `distribution_offset_1e4_50000x32_k5` | fit_query | 151, 251, 252 | 0.819–0.860 | 0.786–0.867 | 0.852–0.883 | 0.960–1.007× | 0.969–1.000× | 1.017–1.040× | 1.027–1.040× | 0.992–1.000× |
| `distribution_anisotropic_50000x32_k5` | kneighbors | 151, 251, 252 | 0.501–0.515 | 0.501–0.511 | 0.502–0.512 | 0.983–1.020× | 0.987–1.008× | 0.984–1.021× | 0.971–1.003× | 0.989–1.016× |
| `distribution_anisotropic_50000x32_k5` | fit_query | 151, 251, 252 | 0.817–0.831 | 0.802–0.839 | 0.854–0.877 | 0.965–1.018× | 0.959–1.016× | 1.029–1.056× | 1.044–1.068× | 0.975–1.012× |
| `distribution_gaussian_4096x128_k5` | kneighbors | 151, 251, 252, 351, 451, 552 | 0.385–0.414 | 0.366–0.407 | 0.386–0.406 | 0.926–1.038× | 0.917–1.039× | 0.935–1.015× | 0.968–1.027× | 0.955–1.051× |
| `distribution_gaussian_4096x128_k5` | fit_query | 151, 251, 252, 351, 451, 552 | 0.577–0.713 | 0.561–0.705 | 0.618–0.743 | 0.939–1.080× | 0.956–1.040× | 1.038–1.101× | 1.012–1.077× | 0.973–1.047× |
| `distribution_clusters_4096x128_k5` | kneighbors | 151, 251, 252 | 0.378–0.406 | 0.377–0.399 | 0.376–0.389 | 0.981–1.012× | 0.986–1.021× | 0.956–1.006× | 0.963–1.007× | 0.995–1.014× |
| `distribution_clusters_4096x128_k5` | fit_query | 151, 251, 252 | 0.497–0.521 | 0.509–0.513 | 0.532–0.549 | 0.980–1.026× | 0.970–1.036× | 1.022–1.096× | 1.008–1.100× | 0.979–1.037× |
| `distribution_offset_1e4_4096x128_k5` | kneighbors | 151, 251, 252 | 0.381–0.421 | 0.368–0.424 | 0.383–0.409 | 0.934–1.007× | 0.966–1.023× | 0.969–1.015× | 0.994–1.015× | 0.976–0.994× |
| `distribution_offset_1e4_4096x128_k5` | fit_query | 151, 251, 252 | 0.586–0.604 | 0.589–0.709 | 0.612–0.680 | 0.975–1.210× | 0.965–1.059× | 1.025–1.160× | 1.025–1.099× | 0.962–1.003× |
| `distribution_anisotropic_4096x128_k5` | kneighbors | 151, 251, 252 | 0.376–0.428 | 0.369–0.435 | 0.365–0.414 | 0.932–1.036× | 0.991–1.034× | 0.923–1.031× | 0.981–1.016× | 0.941–1.033× |
| `distribution_anisotropic_4096x128_k5` | fit_query | 151, 251, 252 | 0.586–0.656 | 0.571–0.655 | 0.646–0.711 | 0.953–1.036× | 0.915–1.043× | 1.078–1.126× | 1.067–1.129× | 0.952–1.030× |

## Preceding automatic implementation

The prior automatic source and its exact native runtime run in an isolated worker. Speedups are previous-auto/current-auto time, with every seed and operation retained.

| Family | Operation | Seeds | Current auto ms | Previous auto ms | Median ratio | Paired ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `shape_single_8_k5` | fit_query | [41, 42, 43] | 0.134–0.145 | 0.134–0.136 | 0.929–1.008× | 0.946–1.020× |
| `shape_single_8_k5` | kneighbors | [41, 42, 43] | 0.080–0.087 | 0.081–0.085 | 0.982–1.011× | 0.979–1.033× |
| `shape_single_8_k5` | indices | [41, 42, 43] | 0.078–0.085 | 0.078–0.089 | 0.947–1.047× | 0.950–1.074× |
| `shape_single_8_k5` | predict_uniform | [41, 42, 43] | 0.086–0.092 | 0.087–0.090 | 0.971–1.021× | 0.967–1.012× |
| `shape_single_8_k5` | proba_uniform | [41, 42, 43] | 0.085–0.092 | 0.081–0.094 | 0.915–1.019× | 0.951–1.019× |
| `shape_single_8_k5` | regress_uniform | [41, 42, 43] | 0.085–0.097 | 0.086–0.091 | 0.937–1.002× | 0.880–1.003× |
| `shape_single_8_k5` | predict_distance | [41, 42, 43] | 0.087–0.101 | 0.097–0.100 | 0.986–1.135× | 0.967–1.090× |
| `shape_single_8_k5` | proba_distance | [41, 42, 43] | 0.095–0.098 | 0.094–0.100 | 0.965–1.025× | 0.980–1.041× |
| `shape_single_8_k5` | regress_distance | [41, 42, 43] | 0.097–0.099 | 0.098–0.101 | 1.000–1.020× | 0.972–1.050× |
| `shape_single_32_k5` | fit_query | [49, 50, 51] | 0.204–0.212 | 0.207–0.217 | 0.982–1.034× | 0.999–1.038× |
| `shape_single_32_k5` | kneighbors | [49, 50, 51] | 0.131–0.139 | 0.131–0.141 | 0.986–1.039× | 0.976–1.052× |
| `shape_single_32_k5` | indices | [49, 50, 51] | 0.128–0.137 | 0.129–0.133 | 0.943–1.018× | 0.932–1.004× |
| `shape_single_32_k5` | predict_uniform | [49, 50, 51] | 0.134–0.140 | 0.125–0.140 | 0.922–1.003× | 0.938–1.025× |
| `shape_single_32_k5` | proba_uniform | [49, 50, 51] | 0.135–0.148 | 0.134–0.146 | 0.947–1.032× | 0.928–1.020× |
| `shape_single_32_k5` | regress_uniform | [49, 50, 51] | 0.146–0.151 | 0.141–0.153 | 0.933–1.046× | 0.949–1.030× |
| `shape_single_32_k5` | predict_distance | [49, 50, 51] | 0.142–0.150 | 0.144–0.151 | 0.959–1.042× | 0.954–1.031× |
| `shape_single_32_k5` | proba_distance | [49, 50, 51] | 0.144–0.160 | 0.142–0.158 | 0.957–1.001× | 0.928–1.000× |
| `shape_single_32_k5` | regress_distance | [49, 50, 51] | 0.146–0.157 | 0.145–0.162 | 0.985–1.057× | 0.955–1.041× |
| `real_digits_raw_k5` | fit_query | [151, 251, 252] | 0.363–0.473 | 0.464–0.527 | 1.085–1.278× | 1.100–1.276× |
| `real_digits_raw_k5` | kneighbors | [151, 251, 252] | 0.315–0.337 | 0.301–0.388 | 0.955–1.151× | 0.963–1.170× |
| `real_digits_raw_k5` | indices | [151, 251, 252] | 0.293–0.332 | 0.289–0.361 | 0.880–1.230× | 0.864–1.276× |
| `real_digits_raw_k5` | predict_uniform | [151, 251, 252] | 0.291–0.346 | 0.295–0.378 | 0.891–1.299× | 0.897–1.307× |
| `real_digits_raw_k5` | proba_uniform | [151, 251, 252] | 0.251–0.369 | 0.367–0.382 | 1.035–1.461× | 1.060–1.396× |
| `real_digits_raw_k5` | regress_uniform | [151, 251, 252] | 0.260–0.331 | 0.353–0.382 | 1.121–1.361× | 1.073–1.347× |
| `real_digits_raw_k5` | predict_distance | [151, 251, 252] | 0.319–0.373 | 0.310–0.391 | 0.830–1.225× | 0.852–1.244× |
| `real_digits_raw_k5` | proba_distance | [151, 251, 252] | 0.345–0.362 | 0.345–0.394 | 0.955–1.142× | 0.944–1.064× |
| `real_digits_raw_k5` | regress_distance | [151, 251, 252] | 0.289–0.361 | 0.312–0.362 | 0.867–1.251× | 0.876–1.298× |
| `real_breast_cancer_raw_k5` | fit_query | [151, 251, 252] | 0.178–0.217 | 0.192–0.219 | 0.935–1.078× | 0.955–1.054× |
| `real_breast_cancer_raw_k5` | kneighbors | [151, 251, 252] | 0.129–0.148 | 0.136–0.145 | 0.978–1.053× | 0.968–1.096× |
| `real_breast_cancer_raw_k5` | indices | [151, 251, 252] | 0.121–0.139 | 0.133–0.142 | 1.021–1.099× | 1.079–1.134× |
| `real_breast_cancer_raw_k5` | predict_uniform | [151, 251, 252] | 0.140–0.161 | 0.140–0.154 | 0.873–1.033× | 0.886–1.054× |
| `real_breast_cancer_raw_k5` | proba_uniform | [151, 251, 252] | 0.155–0.168 | 0.135–0.159 | 0.856–0.948× | 0.841–0.999× |
| `real_breast_cancer_raw_k5` | regress_uniform | [151, 251, 252] | 0.144–0.156 | 0.138–0.158 | 0.901–1.102× | 0.913–1.073× |
| `real_breast_cancer_raw_k5` | predict_distance | [151, 251, 252] | 0.158–0.163 | 0.152–0.171 | 0.931–1.088× | 0.906–1.091× |
| `real_breast_cancer_raw_k5` | proba_distance | [151, 251, 252] | 0.165–0.167 | 0.163–0.184 | 0.980–1.110× | 0.953–1.083× |
| `real_breast_cancer_raw_k5` | regress_distance | [151, 251, 252] | 0.163–0.170 | 0.146–0.165 | 0.896–1.011× | 0.865–0.961× |
| `real_iris_raw_k5` | fit_query | [151, 251, 252] | 0.090–0.098 | 0.085–0.100 | 0.900–1.113× | 0.891–1.066× |
| `real_iris_raw_k5` | kneighbors | [151, 251, 252] | 0.063–0.074 | 0.059–0.070 | 0.920–1.065× | 0.890–1.066× |
| `real_iris_raw_k5` | indices | [151, 251, 252] | 0.064–0.072 | 0.062–0.068 | 0.855–1.067× | 0.904–1.098× |
| `real_iris_raw_k5` | predict_uniform | [151, 251, 252] | 0.070–0.088 | 0.076–0.094 | 1.025–1.100× | 0.953–1.083× |
| `real_iris_raw_k5` | proba_uniform | [151, 251, 252] | 0.067–0.079 | 0.075–0.075 | 0.953–1.120× | 0.971–1.104× |
| `real_iris_raw_k5` | regress_uniform | [151, 251, 252] | 0.074–0.075 | 0.073–0.077 | 0.973–1.047× | 0.901–1.046× |
| `real_iris_raw_k5` | predict_distance | [151, 251, 252] | 0.085–0.094 | 0.083–0.088 | 0.934–1.004× | 0.937–1.045× |
| `real_iris_raw_k5` | proba_distance | [151, 251, 252] | 0.079–0.097 | 0.074–0.087 | 0.889–0.944× | 0.912–1.017× |
| `real_iris_raw_k5` | regress_distance | [151, 251, 252] | 0.082–0.101 | 0.083–0.093 | 0.921–1.051× | 0.928–1.054× |
| `real_wine_raw_k5` | fit_query | [151, 251, 252] | 0.101–0.108 | 0.100–0.108 | 0.927–1.062× | 0.922–1.080× |
| `real_wine_raw_k5` | kneighbors | [151, 251, 252] | 0.075–0.080 | 0.079–0.081 | 0.991–1.082× | 0.996–1.068× |
| `real_wine_raw_k5` | indices | [151, 251, 252] | 0.071–0.076 | 0.067–0.075 | 0.879–1.049× | 0.912–1.022× |
| `real_wine_raw_k5` | predict_uniform | [151, 251, 252] | 0.078–0.114 | 0.083–0.104 | 0.913–1.062× | 0.916–1.047× |
| `real_wine_raw_k5` | proba_uniform | [151, 251, 252] | 0.082–0.090 | 0.079–0.095 | 0.896–1.060× | 0.894–1.137× |
| `real_wine_raw_k5` | regress_uniform | [151, 251, 252] | 0.077–0.086 | 0.082–0.083 | 0.971–1.069× | 0.947–1.081× |
| `real_wine_raw_k5` | predict_distance | [151, 251, 252] | 0.086–0.091 | 0.089–0.096 | 1.037–1.061× | 1.000–1.083× |
| `real_wine_raw_k5` | proba_distance | [151, 251, 252] | 0.093–0.108 | 0.091–0.110 | 0.892–1.011× | 0.843–1.025× |
| `real_wine_raw_k5` | regress_distance | [151, 251, 252] | 0.093–0.097 | 0.086–0.095 | 0.906–1.018× | 0.879–0.977× |
| `distribution_gaussian_50000x32_k5` | kneighbors | [151, 251, 252, 351, 451, 552] | 0.428–2.396 | 0.425–2.386 | 0.982–1.021× | 0.982–1.018× |
| `distribution_gaussian_50000x32_k5` | fit_query | [151, 251, 252, 351, 451, 552] | 0.750–2.424 | 0.749–2.420 | 0.983–1.013× | 0.966–1.026× |
| `distribution_clusters_50000x32_k5` | kneighbors | [151, 251, 252] | 0.748–0.796 | 0.750–0.797 | 0.942–1.065× | 0.963–0.999× |
| `distribution_clusters_50000x32_k5` | fit_query | [151, 251, 252] | 1.059–2.169 | 1.082–2.125 | 0.980–1.021× | 0.984–1.023× |
| `distribution_offset_1e4_50000x32_k5` | kneighbors | [151, 251, 252] | 0.449–0.782 | 0.457–0.792 | 1.012–1.024× | 0.976–1.007× |
| `distribution_offset_1e4_50000x32_k5` | fit_query | [151, 251, 252] | 0.819–0.860 | 0.823–0.863 | 0.975–1.005× | 0.964–1.001× |
| `distribution_anisotropic_50000x32_k5` | kneighbors | [151, 251, 252] | 0.501–0.515 | 0.506–0.517 | 0.993–1.031× | 0.988–1.018× |
| `distribution_anisotropic_50000x32_k5` | fit_query | [151, 251, 252] | 0.817–0.831 | 0.820–0.844 | 0.987–1.033× | 0.982–1.030× |
| `distribution_gaussian_4096x128_k5` | kneighbors | [151, 251, 252, 351, 451, 552] | 0.385–0.414 | 0.380–0.405 | 0.940–1.047× | 0.972–1.035× |
| `distribution_gaussian_4096x128_k5` | fit_query | [151, 251, 252, 351, 451, 552] | 0.577–0.713 | 0.563–0.696 | 0.953–1.109× | 0.955–1.035× |
| `distribution_clusters_4096x128_k5` | kneighbors | [151, 251, 252] | 0.378–0.406 | 0.380–0.406 | 0.996–1.019× | 0.970–1.031× |
| `distribution_clusters_4096x128_k5` | fit_query | [151, 251, 252] | 0.497–0.521 | 0.511–0.524 | 0.981–1.038× | 0.973–1.042× |
| `distribution_offset_1e4_4096x128_k5` | kneighbors | [151, 251, 252] | 0.381–0.421 | 0.378–0.440 | 0.992–1.045× | 0.946–1.031× |
| `distribution_offset_1e4_4096x128_k5` | fit_query | [151, 251, 252] | 0.586–0.604 | 0.601–0.651 | 1.006–1.111× | 0.998–1.007× |
| `distribution_anisotropic_4096x128_k5` | kneighbors | [151, 251, 252] | 0.376–0.428 | 0.381–0.427 | 0.963–1.071× | 0.984–1.052× |
| `distribution_anisotropic_4096x128_k5` | fit_query | [151, 251, 252] | 0.586–0.656 | 0.611–0.669 | 1.019–1.070× | 1.014–1.067× |

## Observed backend routing

Counts include every warmup, probe and retained query, with cold calls separate. Fit storage is reported independently of actual query execution.

| Variant | Phase | Requests | Query backends | Fit storage |
| --- | --- | ---: | --- | --- |
| current | cold | 48 | `{"cpu": 18, "mlx": 30}` | `{"cpu": 18, "mlx": 30}` |
| current | warmup | 286866 | `{"cpu": 264827, "mlx": 22039}` | `{"cpu": 264827, "mlx": 22039}` |
| current | probe | 222 | `{"cpu": 162, "mlx": 60}` | `{"cpu": 162, "mlx": 60}` |
| current | retained | 6882 | `{"cpu": 5022, "mlx": 1860}` | `{"cpu": 5022, "mlx": 1860}` |
| current_b | cold | 48 | `{"cpu": 18, "mlx": 30}` | `{"cpu": 18, "mlx": 30}` |
| current_b | warmup | 288372 | `{"cpu": 266258, "mlx": 22114}` | `{"cpu": 266258, "mlx": 22114}` |
| current_b | probe | 222 | `{"cpu": 162, "mlx": 60}` | `{"cpu": 162, "mlx": 60}` |
| current_b | retained | 6882 | `{"cpu": 5022, "mlx": 1860}` | `{"cpu": 5022, "mlx": 1860}` |
| current_gpu | cold | 48 | `{"mlx": 48}` | `{"mlx": 48}` |
| current_gpu | warmup | 161354 | `{"mlx": 161354}` | `{"mlx": 161354}` |
| current_gpu | probe | 222 | `{"mlx": 222}` | `{"mlx": 222}` |
| current_gpu | retained | 6882 | `{"mlx": 6882}` | `{"mlx": 6882}` |
| previous_auto | cold | 48 | `{"cpu": 18, "mlx": 30}` | `{"cpu": 18, "mlx": 30}` |
| previous_auto | warmup | 285238 | `{"cpu": 263164, "mlx": 22074}` | `{"cpu": 263164, "mlx": 22074}` |
| previous_auto | probe | 222 | `{"cpu": 162, "mlx": 60}` | `{"cpu": 162, "mlx": 60}` |
| previous_auto | retained | 6882 | `{"cpu": 5022, "mlx": 1860}` | `{"cpu": 5022, "mlx": 1860}` |
| previous_gpu | cold | 48 | `{"mlx": 48}` | `{"mlx": 48}` |
| previous_gpu | warmup | 162930 | `{"mlx": 162930}` | `{"mlx": 162930}` |
| previous_gpu | probe | 222 | `{"mlx": 222}` | `{"mlx": 222}` |
| previous_gpu | retained | 6882 | `{"mlx": 6882}` | `{"mlx": 6882}` |

The full sklearn panel is a separate run on its unchanged 72 fixtures and all nine operations. These controlled comparisons isolate source changes; absolute timings should not be compared across runs with different warmup protocols.
