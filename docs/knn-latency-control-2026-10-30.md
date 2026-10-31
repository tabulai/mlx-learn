# Automatic CPU/GPU controlled benchmark

Completed `2026-10-30T21:59:31.366565+00:00`. Acceptance: **True**. [Raw results](../benchmarks/results/knn-latency-final-2026-10-30-control.json) (SHA-256 `e12764076f2aaa5665324f2ec6a8ab2288eef9c5569bed33ba2a64434f305e95`).

5 isolated public-API workers: current automatic dispatch, an identical automatic A/A worker, current forced GPU and frozen forced-GPU `knn-small-final-2026-10-30.zip`. An additional previous-auto worker uses the preceding source and exact ABI-matched archived native binary. All CPU/GPU calls are serial. Each worker/operation gets 0.25 seconds of continuous warmup followed by 31 interleaved paired rounds. Every call includes public validation, NumPy boundaries and synchronization. Cold first-fit/query records come from persistent workers, excluding process startup. Default continuous warmup is 0.25 seconds (the preceding small control used 1 second); this is configurable and absolute timings must not be compared across those runs. Each worker/operation warms continuously for at least the requested duration. Its one post-warmup probe is audited and retained separately, excluded from paired ratios. All warmup audits and probe samples remain in the report.

The 18 small-dataset/single-query fixtures cover all nine search/inference operations. The 30 preservation fixtures cover query and fresh fit-plus-query, including the six expanded Gaussian seeds. Inputs, exact canonical neighbor ordering, distance tolerances and prediction oracles are unchanged. Every cold, warmup, probe and retained audit stays in the raw file, including failures. No result caching or production hooks are used.

All speedups are comparator/current time. Ranges span dataset seeds. Ratio of medians and median paired ratios are distinct descriptive statistics; neither is a confidence interval. A/A shows the corresponding process/timing variation. Values below 1× are slower observations and are retained.

Per-variant strict quality: `{"current": true, "current_b": true, "current_gpu": true, "previous_auto": true, "previous_gpu": true}`.

## Small datasets and single queries

| Family | Operation | Seeds | Auto ms | Current GPU ms | Previous GPU ms | Current GPU/auto medians | Current GPU/auto paired | Previous GPU/auto medians | Previous GPU/auto paired | Auto A/A paired |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shape_single_8_k5` | fit_query | 41, 42, 43 | 0.165–0.179 | 0.357–0.390 | 0.311–0.459 | 2.076–2.182× | 2.092–2.500× | 1.799–2.780× | 1.753–2.571× | 0.947–1.027× |
| `shape_single_8_k5` | kneighbors | 41, 42, 43 | 0.092–0.120 | 0.196–0.283 | 0.208–0.319 | 2.143–2.485× | 2.135–2.531× | 2.274–2.652× | 2.260–2.623× | 0.984–1.048× |
| `shape_single_8_k5` | indices | 41, 42, 43 | 0.084–0.093 | 0.245–0.413 | 0.239–0.260 | 2.627–4.484× | 2.837–4.803× | 2.601–2.994× | 2.672–3.049× | 0.975–1.182× |
| `shape_single_8_k5` | predict_uniform | 41, 42, 43 | 0.091–0.101 | 0.216–0.299 | 0.213–0.427 | 2.221–3.288× | 2.276–3.454× | 2.111–4.688× | 2.131–3.717× | 1.000–1.018× |
| `shape_single_8_k5` | proba_uniform | 41, 42, 43 | 0.096–0.109 | 0.211–0.286 | 0.247–0.293 | 1.939–2.964× | 1.979–3.407× | 2.263–2.843× | 2.221–3.022× | 0.993–1.084× |
| `shape_single_8_k5` | regress_uniform | 41, 42, 43 | 0.098–0.111 | 0.258–0.299 | 0.269–0.274 | 2.585–2.958× | 2.516–3.043× | 2.462–2.735× | 2.416–2.876× | 0.980–1.028× |
| `shape_single_8_k5` | predict_distance | 41, 42, 43 | 0.100–0.132 | 0.229–0.301 | 0.212–0.303 | 2.101–3.016× | 2.129–3.223× | 2.101–3.037× | 2.149–3.091× | 0.985–1.131× |
| `shape_single_8_k5` | proba_distance | 41, 42, 43 | 0.102–0.112 | 0.294–0.329 | 0.292–0.346 | 2.624–3.126× | 2.743–3.314× | 2.613–3.284× | 2.757–3.284× | 1.020–1.074× |
| `shape_single_8_k5` | regress_distance | 41, 42, 43 | 0.105–0.112 | 0.216–0.300 | 0.224–0.484 | 2.028–2.850× | 2.052–2.827× | 2.003–4.594× | 2.094–4.625× | 0.978–1.021× |
| `shape_single_32_k5` | fit_query | 49, 50, 51 | 0.236–0.251 | 0.358–0.456 | 0.356–0.432 | 1.448–1.932× | 1.455–2.031× | 1.421–1.831× | 1.431–1.869× | 0.979–1.034× |
| `shape_single_32_k5` | kneighbors | 49, 50, 51 | 0.137–0.152 | 0.305–0.326 | 0.302–0.336 | 2.093–2.272× | 2.296–2.304× | 2.200–2.277× | 2.240–2.293× | 0.926–1.050× |
| `shape_single_32_k5` | indices | 49, 50, 51 | 0.139–0.147 | 0.198–0.327 | 0.211–0.351 | 1.422–2.274× | 1.454–2.300× | 1.521–2.440× | 1.585–2.360× | 0.980–1.028× |
| `shape_single_32_k5` | predict_uniform | 49, 50, 51 | 0.144–0.147 | 0.227–0.367 | 0.239–0.368 | 1.558–2.544× | 1.576–2.596× | 1.638–2.549× | 1.651–2.600× | 0.989–1.011× |
| `shape_single_32_k5` | proba_uniform | 49, 50, 51 | 0.150–0.153 | 0.313–0.373 | 0.304–0.351 | 2.046–2.491× | 2.026–2.424× | 1.989–2.335× | 1.994–2.412× | 0.967–1.013× |
| `shape_single_32_k5` | regress_uniform | 49, 50, 51 | 0.143–0.152 | 0.252–0.323 | 0.222–0.332 | 1.659–2.263× | 1.720–2.297× | 1.460–2.322× | 1.538–2.353× | 0.993–1.084× |
| `shape_single_32_k5` | predict_distance | 49, 50, 51 | 0.151–0.155 | 0.343–0.353 | 0.331–0.374 | 2.260–2.340× | 2.354–2.545× | 2.184–2.463× | 2.140–2.514× | 0.982–1.043× |
| `shape_single_32_k5` | proba_distance | 49, 50, 51 | 0.154–0.165 | 0.231–0.346 | 0.252–0.347 | 1.502–2.230× | 1.509–2.317× | 1.640–2.234× | 1.614–2.179× | 0.976–1.008× |
| `shape_single_32_k5` | regress_distance | 49, 50, 51 | 0.154–0.156 | 0.238–0.344 | 0.278–0.333 | 1.528–2.234× | 1.610–2.261× | 1.787–2.164× | 1.772–2.236× | 1.016–1.023× |
| `real_digits_raw_k5` | fit_query | 151, 251, 252 | 0.561–0.618 | 1.121–1.166 | 1.117–1.257 | 1.887–1.997× | 1.952–2.049× | 1.917–2.054× | 1.932–2.104× | 0.896–1.040× |
| `real_digits_raw_k5` | kneighbors | 151, 251, 252 | 0.405–0.537 | 0.855–1.217 | 0.991–1.430 | 2.112–2.290× | 2.146–2.386× | 2.333–2.666× | 2.472–2.609× | 0.866–0.985× |
| `real_digits_raw_k5` | indices | 151, 251, 252 | 0.409–0.446 | 0.426–0.979 | 0.498–0.951 | 1.042–2.242× | 1.142–2.286× | 1.147–2.318× | 1.297–2.271× | 0.853–0.985× |
| `real_digits_raw_k5` | predict_uniform | 151, 251, 252 | 0.393–0.442 | 0.478–0.866 | 0.450–1.000 | 1.115–2.060× | 1.230–1.996× | 1.049–2.544× | 1.080–2.316× | 0.985–1.135× |
| `real_digits_raw_k5` | proba_uniform | 151, 251, 252 | 0.438–0.449 | 0.858–0.967 | 0.418–0.966 | 1.962–2.151× | 1.782–2.116× | 0.956–2.205× | 0.964–2.374× | 0.829–1.044× |
| `real_digits_raw_k5` | regress_uniform | 151, 251, 252 | 0.365–0.481 | 0.448–1.008 | 0.769–1.005 | 0.931–2.578× | 0.930–2.505× | 1.600–2.499× | 1.536–2.427× | 0.836–1.159× |
| `real_digits_raw_k5` | predict_distance | 151, 251, 252 | 0.448–0.467 | 0.453–0.939 | 0.812–0.974 | 0.990–2.011× | 1.066–1.937× | 1.737–2.127× | 1.666–2.205× | 0.864–0.992× |
| `real_digits_raw_k5` | proba_distance | 151, 251, 252 | 0.436–0.480 | 0.411–1.019 | 0.401–0.942 | 0.943–2.122× | 1.004–2.133× | 0.857–1.991× | 0.882–1.989× | 0.949–0.983× |
| `real_digits_raw_k5` | regress_distance | 151, 251, 252 | 0.386–0.478 | 0.967–1.004 | 0.447–0.974 | 2.081–2.602× | 2.055–2.601× | 0.969–2.524× | 1.001–2.499× | 0.977–1.183× |
| `real_breast_cancer_raw_k5` | fit_query | 151, 251, 252 | 0.263–0.298 | 0.372–0.381 | 0.333–0.364 | 1.278–1.415× | 1.293–1.424× | 1.148–1.384× | 1.193–1.336× | 1.033–1.136× |
| `real_breast_cancer_raw_k5` | kneighbors | 151, 251, 252 | 0.178–0.200 | 0.235–0.265 | 0.213–0.483 | 1.297–1.470× | 1.280–1.438× | 1.194–2.681× | 1.124–2.483× | 1.057–1.140× |
| `real_breast_cancer_raw_k5` | indices | 151, 251, 252 | 0.167–0.188 | 0.214–0.245 | 0.213–0.269 | 1.139–1.433× | 1.163–1.421× | 1.133–1.493× | 1.133–1.484× | 1.015–1.127× |
| `real_breast_cancer_raw_k5` | predict_uniform | 151, 251, 252 | 0.180–0.197 | 0.224–0.284 | 0.242–0.494 | 1.221–1.510× | 1.218–1.554× | 1.316–2.742× | 1.287–2.613× | 0.965–1.018× |
| `real_breast_cancer_raw_k5` | proba_uniform | 151, 251, 252 | 0.193–0.199 | 0.217–0.297 | 0.239–0.277 | 1.128–1.534× | 1.158–1.492× | 1.220–1.427× | 1.181–1.427× | 1.041–1.043× |
| `real_breast_cancer_raw_k5` | regress_uniform | 151, 251, 252 | 0.185–0.195 | 0.239–0.279 | 0.223–0.257 | 1.223–1.478× | 1.215–1.543× | 1.141–1.393× | 1.142–1.404× | 1.044–1.098× |
| `real_breast_cancer_raw_k5` | predict_distance | 151, 251, 252 | 0.192–0.214 | 0.225–0.277 | 0.199–0.513 | 1.113–1.441× | 1.094–1.457× | 0.984–2.393× | 0.955–2.312× | 1.000–1.153× |
| `real_breast_cancer_raw_k5` | proba_distance | 151, 251, 252 | 0.201–0.236 | 0.254–0.314 | 0.253–0.511 | 1.229–1.369× | 1.195–1.399× | 1.225–2.166× | 1.246–2.135× | 1.014–1.124× |
| `real_breast_cancer_raw_k5` | regress_distance | 151, 251, 252 | 0.197–0.219 | 0.234–0.305 | 0.240–0.289 | 1.131–1.548× | 1.132–1.701× | 1.156–1.468× | 1.152–1.487× | 1.012–1.069× |
| `real_iris_raw_k5` | fit_query | 151, 251, 252 | 0.123–0.146 | 0.348–0.403 | 0.365–0.507 | 2.749–2.972× | 2.779–3.434× | 2.971–3.464× | 2.916–3.706× | 0.873–1.115× |
| `real_iris_raw_k5` | kneighbors | 151, 251, 252 | 0.068–0.077 | 0.215–0.277 | 0.201–0.276 | 2.786–3.668× | 2.698–4.148× | 2.931–3.654× | 2.808–4.001× | 0.984–1.072× |
| `real_iris_raw_k5` | indices | 151, 251, 252 | 0.064–0.075 | 0.231–0.289 | 0.203–0.282 | 3.059–4.502× | 3.057–4.529× | 2.690–4.395× | 2.709–4.461× | 1.015–1.151× |
| `real_iris_raw_k5` | predict_uniform | 151, 251, 252 | 0.075–0.084 | 0.238–0.309 | 0.214–0.313 | 2.828–4.087× | 2.829–4.283× | 2.536–4.135× | 2.557–4.318× | 0.982–1.172× |
| `real_iris_raw_k5` | proba_uniform | 151, 251, 252 | 0.077–0.083 | 0.223–0.303 | 0.258–0.357 | 2.920–3.878× | 3.087–4.494× | 3.377–4.313× | 3.358–4.171× | 0.922–1.107× |
| `real_iris_raw_k5` | regress_uniform | 151, 251, 252 | 0.079–0.083 | 0.203–0.304 | 0.212–0.509 | 2.456–3.812× | 2.541–3.822× | 2.563–6.383× | 2.650–5.114× | 0.985–1.018× |
| `real_iris_raw_k5` | predict_distance | 151, 251, 252 | 0.092–0.096 | 0.216–0.302 | 0.224–0.331 | 2.358–3.144× | 2.305–3.350× | 2.348–3.453× | 2.259–3.509× | 0.923–1.026× |
| `real_iris_raw_k5` | proba_distance | 151, 251, 252 | 0.075–0.105 | 0.262–0.328 | 0.232–0.309 | 2.491–4.377× | 2.478–4.551× | 2.207–4.122× | 2.263–4.093× | 0.982–1.170× |
| `real_iris_raw_k5` | regress_distance | 151, 251, 252 | 0.085–0.093 | 0.266–0.319 | 0.259–0.317 | 2.866–3.685× | 3.053–4.215× | 2.788–3.715× | 2.836–4.214× | 0.979–1.127× |
| `real_wine_raw_k5` | fit_query | 151, 251, 252 | 0.127–0.135 | 0.313–0.337 | 0.288–0.580 | 2.387–2.657× | 2.458–2.730× | 2.196–4.302× | 2.197–3.649× | 0.975–1.075× |
| `real_wine_raw_k5` | kneighbors | 151, 251, 252 | 0.081–0.093 | 0.200–0.246 | 0.211–0.366 | 2.157–3.045× | 2.188–3.417× | 2.278–4.261× | 2.292–3.477× | 0.966–1.139× |
| `real_wine_raw_k5` | indices | 151, 251, 252 | 0.078–0.104 | 0.228–0.272 | 0.226–0.249 | 2.195–3.265× | 2.324–3.427× | 2.173–3.020× | 2.357–3.020× | 0.956–1.031× |
| `real_wine_raw_k5` | predict_uniform | 151, 251, 252 | 0.093–0.110 | 0.213–0.285 | 0.207–0.305 | 2.285–2.582× | 2.267–2.737× | 2.214–2.759× | 2.169–2.828× | 0.970–1.026× |
| `real_wine_raw_k5` | proba_uniform | 151, 251, 252 | 0.082–0.096 | 0.253–0.284 | 0.238–0.273 | 2.644–3.442× | 2.560–3.719× | 2.489–3.048× | 2.450–3.033× | 0.998–1.222× |
| `real_wine_raw_k5` | regress_uniform | 151, 251, 252 | 0.084–0.095 | 0.203–0.439 | 0.189–0.300 | 2.127–4.954× | 2.146–4.269× | 1.983–3.380× | 1.925–3.160× | 0.996–1.129× |
| `real_wine_raw_k5` | predict_distance | 151, 251, 252 | 0.095–0.099 | 0.214–0.448 | 0.235–0.502 | 2.162–4.704× | 2.130–4.206× | 2.370–5.263× | 2.278–4.276× | 0.987–1.172× |
| `real_wine_raw_k5` | proba_distance | 151, 251, 252 | 0.100–0.109 | 0.204–0.390 | 0.224–0.441 | 1.937–3.889× | 1.982–3.470× | 2.127–4.406× | 2.137–3.735× | 0.930–1.013× |
| `real_wine_raw_k5` | regress_distance | 151, 251, 252 | 0.103–0.107 | 0.235–0.253 | 0.235–0.258 | 2.249–2.448× | 2.238–2.522× | 2.192–2.488× | 2.031–2.575× | 1.005–1.075× |

## Gaussian, clustered, offset and unequal-scale preservation

| Family | Operation | Seeds | Auto ms | Current GPU ms | Previous GPU ms | Current GPU/auto medians | Current GPU/auto paired | Previous GPU/auto medians | Previous GPU/auto paired | Auto A/A paired |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `distribution_gaussian_50000x32_k5` | kneighbors | 151, 251, 252, 351, 451, 552 | 0.424–2.390 | 0.420–2.358 | 0.424–2.368 | 0.965–1.020× | 0.972–0.998× | 0.960–1.031× | 0.976–1.025× | 0.978–1.014× |
| `distribution_gaussian_50000x32_k5` | fit_query | 151, 251, 252, 351, 451, 552 | 0.800–2.501 | 0.796–2.487 | 0.808–2.529 | 0.952–1.033× | 0.966–1.055× | 0.974–1.017× | 0.965–1.042× | 0.987–1.132× |
| `distribution_clusters_50000x32_k5` | kneighbors | 151, 251, 252 | 0.793–0.803 | 0.769–0.799 | 0.771–0.791 | 0.969–1.000× | 0.984–1.002× | 0.972–0.994× | 0.976–1.035× | 0.989–0.996× |
| `distribution_clusters_50000x32_k5` | fit_query | 151, 251, 252 | 1.093–1.149 | 1.106–1.152 | 1.119–1.125 | 0.984–1.047× | 0.985–1.037× | 0.974–1.029× | 0.981–1.024× | 0.973–1.058× |
| `distribution_offset_1e4_50000x32_k5` | kneighbors | 151, 251, 252 | 0.436–0.478 | 0.450–0.488 | 0.442–0.472 | 0.979–1.032× | 0.973–0.998× | 0.987–1.014× | 0.986–1.009× | 0.998–1.016× |
| `distribution_offset_1e4_50000x32_k5` | fit_query | 151, 251, 252 | 0.846–0.909 | 0.858–0.891 | 0.866–0.907 | 0.980–1.015× | 0.979–1.014× | 0.998–1.026× | 0.982–1.005× | 0.985–1.025× |
| `distribution_anisotropic_50000x32_k5` | kneighbors | 151, 251, 252 | 0.512–0.520 | 0.514–0.524 | 0.506–0.520 | 0.999–1.010× | 0.995–1.014× | 0.987–1.001× | 0.960–1.017× | 0.994–1.039× |
| `distribution_anisotropic_50000x32_k5` | fit_query | 151, 251, 252 | 0.879–0.904 | 0.881–0.934 | 0.866–0.920 | 1.000–1.033× | 0.948–1.025× | 0.975–1.022× | 0.982–1.060× | 0.990–1.012× |
| `distribution_gaussian_4096x128_k5` | kneighbors | 151, 251, 252, 351, 451, 552 | 0.396–0.460 | 0.389–0.442 | 0.403–0.444 | 0.951–1.043× | 0.949–1.027× | 0.913–1.017× | 0.972–1.024× | 0.997–1.059× |
| `distribution_gaussian_4096x128_k5` | fit_query | 151, 251, 252, 351, 451, 552 | 0.672–0.720 | 0.658–0.677 | 0.661–0.700 | 0.941–1.001× | 0.935–0.996× | 0.963–1.023× | 0.984–1.017× | 0.955–1.073× |
| `distribution_clusters_4096x128_k5` | kneighbors | 151, 251, 252 | 0.386–0.398 | 0.378–0.397 | 0.379–0.397 | 0.972–1.006× | 0.974–0.992× | 0.975–0.998× | 0.976–0.991× | 0.997–1.003× |
| `distribution_clusters_4096x128_k5` | fit_query | 151, 251, 252 | 0.584–0.592 | 0.578–0.585 | 0.552–0.579 | 0.976–1.002× | 0.978–1.039× | 0.933–0.991× | 0.950–1.030× | 0.999–1.052× |
| `distribution_offset_1e4_4096x128_k5` | kneighbors | 151, 251, 252 | 0.397–0.425 | 0.389–0.416 | 0.400–0.428 | 0.959–1.036× | 0.965–0.995× | 0.987–1.010× | 0.996–1.002× | 0.988–1.015× |
| `distribution_offset_1e4_4096x128_k5` | fit_query | 151, 251, 252 | 0.688–0.700 | 0.618–0.678 | 0.625–0.675 | 0.888–0.968× | 0.945–0.965× | 0.898–0.981× | 0.827–0.999× | 0.974–1.005× |
| `distribution_anisotropic_4096x128_k5` | kneighbors | 151, 251, 252 | 0.377–0.403 | 0.386–0.411 | 0.375–0.423 | 1.001–1.025× | 0.970–1.047× | 0.973–1.048× | 0.968–1.027× | 0.981–1.069× |
| `distribution_anisotropic_4096x128_k5` | fit_query | 151, 251, 252 | 0.675–0.688 | 0.681–0.730 | 0.665–0.701 | 0.994–1.082× | 0.983–1.067× | 0.965–1.022× | 0.984–1.027× | 0.961–1.043× |

## Preceding automatic implementation

The prior automatic source and its exact native runtime run in an isolated worker. Speedups are previous-auto/current-auto time, with every seed and operation retained.

| Family | Operation | Seeds | Current auto ms | Previous auto ms | Median ratio | Paired ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `shape_single_8_k5` | fit_query | [41, 42, 43] | 0.165–0.179 | 0.182–0.186 | 1.040–1.123× | 1.028–1.070× |
| `shape_single_8_k5` | kneighbors | [41, 42, 43] | 0.092–0.120 | 0.095–0.122 | 1.015–1.035× | 0.988–1.022× |
| `shape_single_8_k5` | indices | [41, 42, 43] | 0.084–0.093 | 0.083–0.093 | 0.885–1.105× | 0.897–1.128× |
| `shape_single_8_k5` | predict_uniform | [41, 42, 43] | 0.091–0.101 | 0.097–0.106 | 1.007–1.061× | 1.001–1.043× |
| `shape_single_8_k5` | proba_uniform | [41, 42, 43] | 0.096–0.109 | 0.098–0.108 | 0.919–1.087× | 0.901–1.085× |
| `shape_single_8_k5` | regress_uniform | [41, 42, 43] | 0.098–0.111 | 0.098–0.118 | 0.997–1.083× | 1.020–1.107× |
| `shape_single_8_k5` | predict_distance | [41, 42, 43] | 0.100–0.132 | 0.105–0.130 | 0.986–1.069× | 0.990–1.059× |
| `shape_single_8_k5` | proba_distance | [41, 42, 43] | 0.102–0.112 | 0.097–0.111 | 0.865–1.092× | 0.885–1.126× |
| `shape_single_8_k5` | regress_distance | [41, 42, 43] | 0.105–0.112 | 0.107–0.113 | 1.011–1.019× | 0.996–1.028× |
| `shape_single_32_k5` | fit_query | [49, 50, 51] | 0.236–0.251 | 0.253–0.266 | 1.025–1.126× | 1.055–1.118× |
| `shape_single_32_k5` | kneighbors | [49, 50, 51] | 0.137–0.152 | 0.136–0.148 | 0.946–1.048× | 0.952–1.018× |
| `shape_single_32_k5` | indices | [49, 50, 51] | 0.139–0.147 | 0.133–0.146 | 0.955–1.017× | 0.962–1.016× |
| `shape_single_32_k5` | predict_uniform | [49, 50, 51] | 0.144–0.147 | 0.140–0.147 | 0.955–1.006× | 0.943–1.040× |
| `shape_single_32_k5` | proba_uniform | [49, 50, 51] | 0.150–0.153 | 0.147–0.154 | 0.974–1.026× | 0.953–1.042× |
| `shape_single_32_k5` | regress_uniform | [49, 50, 51] | 0.143–0.152 | 0.151–0.152 | 0.991–1.066× | 0.977–1.073× |
| `shape_single_32_k5` | predict_distance | [49, 50, 51] | 0.151–0.155 | 0.148–0.155 | 0.953–1.026× | 0.935–1.068× |
| `shape_single_32_k5` | proba_distance | [49, 50, 51] | 0.154–0.165 | 0.151–0.171 | 0.973–1.032× | 0.993–1.024× |
| `shape_single_32_k5` | regress_distance | [49, 50, 51] | 0.154–0.156 | 0.150–0.164 | 0.969–1.052× | 0.969–1.065× |
| `real_digits_raw_k5` | fit_query | [151, 251, 252] | 0.561–0.618 | 1.058–1.129 | 1.821–1.884× | 1.818–1.915× |
| `real_digits_raw_k5` | kneighbors | [151, 251, 252] | 0.405–0.537 | 0.887–1.099 | 2.040–2.191× | 2.016–2.202× |
| `real_digits_raw_k5` | indices | [151, 251, 252] | 0.409–0.446 | 0.901–0.914 | 2.025–2.202× | 2.041–2.185× |
| `real_digits_raw_k5` | predict_uniform | [151, 251, 252] | 0.393–0.442 | 0.895–0.909 | 2.024–2.313× | 2.016–2.310× |
| `real_digits_raw_k5` | proba_uniform | [151, 251, 252] | 0.438–0.449 | 0.901–0.916 | 2.038–2.069× | 2.082–2.119× |
| `real_digits_raw_k5` | regress_uniform | [151, 251, 252] | 0.365–0.481 | 0.890–0.981 | 2.009–2.439× | 2.043–2.446× |
| `real_digits_raw_k5` | predict_distance | [151, 251, 252] | 0.448–0.467 | 0.912–0.940 | 1.952–2.095× | 1.940–2.121× |
| `real_digits_raw_k5` | proba_distance | [151, 251, 252] | 0.436–0.480 | 0.911–0.935 | 1.897–2.144× | 1.900–2.091× |
| `real_digits_raw_k5` | regress_distance | [151, 251, 252] | 0.386–0.478 | 0.894–0.909 | 1.899–2.317× | 1.892–2.335× |
| `real_breast_cancer_raw_k5` | fit_query | [151, 251, 252] | 0.263–0.298 | 0.283–0.291 | 0.978–1.075× | 0.996–1.075× |
| `real_breast_cancer_raw_k5` | kneighbors | [151, 251, 252] | 0.178–0.200 | 0.168–0.208 | 0.934–1.043× | 0.930–1.062× |
| `real_breast_cancer_raw_k5` | indices | [151, 251, 252] | 0.167–0.188 | 0.169–0.190 | 0.939–1.112× | 0.943–1.092× |
| `real_breast_cancer_raw_k5` | predict_uniform | [151, 251, 252] | 0.180–0.197 | 0.170–0.195 | 0.862–1.063× | 0.866–1.066× |
| `real_breast_cancer_raw_k5` | proba_uniform | [151, 251, 252] | 0.193–0.199 | 0.175–0.189 | 0.903–0.971× | 0.896–0.975× |
| `real_breast_cancer_raw_k5` | regress_uniform | [151, 251, 252] | 0.185–0.195 | 0.179–0.188 | 0.920–1.017× | 0.911–1.033× |
| `real_breast_cancer_raw_k5` | predict_distance | [151, 251, 252] | 0.192–0.214 | 0.194–0.209 | 0.959–1.059× | 0.955–1.060× |
| `real_breast_cancer_raw_k5` | proba_distance | [151, 251, 252] | 0.201–0.236 | 0.206–0.241 | 0.995–1.065× | 0.983–1.081× |
| `real_breast_cancer_raw_k5` | regress_distance | [151, 251, 252] | 0.197–0.219 | 0.186–0.206 | 0.847–0.992× | 0.830–0.996× |
| `real_iris_raw_k5` | fit_query | [151, 251, 252] | 0.123–0.146 | 0.120–0.147 | 0.951–1.060× | 0.948–1.076× |
| `real_iris_raw_k5` | kneighbors | [151, 251, 252] | 0.068–0.077 | 0.068–0.075 | 0.899–1.047× | 0.859–0.986× |
| `real_iris_raw_k5` | indices | [151, 251, 252] | 0.064–0.075 | 0.070–0.076 | 0.958–1.109× | 0.927–1.107× |
| `real_iris_raw_k5` | predict_uniform | [151, 251, 252] | 0.075–0.084 | 0.080–0.084 | 0.955–1.114× | 0.960–1.113× |
| `real_iris_raw_k5` | proba_uniform | [151, 251, 252] | 0.077–0.083 | 0.080–0.086 | 0.967–1.127× | 1.044–1.236× |
| `real_iris_raw_k5` | regress_uniform | [151, 251, 252] | 0.079–0.083 | 0.072–0.087 | 0.893–1.086× | 0.924–1.069× |
| `real_iris_raw_k5` | predict_distance | [151, 251, 252] | 0.092–0.096 | 0.090–0.095 | 0.943–1.037× | 0.938–1.006× |
| `real_iris_raw_k5` | proba_distance | [151, 251, 252] | 0.075–0.105 | 0.089–0.095 | 0.881–1.189× | 0.879–1.196× |
| `real_iris_raw_k5` | regress_distance | [151, 251, 252] | 0.085–0.093 | 0.090–0.094 | 0.977–1.086× | 0.990–1.108× |
| `real_wine_raw_k5` | fit_query | [151, 251, 252] | 0.127–0.135 | 0.142–0.146 | 1.063–1.149× | 1.046–1.140× |
| `real_wine_raw_k5` | kneighbors | [151, 251, 252] | 0.081–0.093 | 0.083–0.085 | 0.907–1.053× | 0.933–1.065× |
| `real_wine_raw_k5` | indices | [151, 251, 252] | 0.078–0.104 | 0.077–0.101 | 0.968–1.062× | 0.949–1.071× |
| `real_wine_raw_k5` | predict_uniform | [151, 251, 252] | 0.093–0.110 | 0.091–0.097 | 0.880–0.975× | 0.968–0.977× |
| `real_wine_raw_k5` | proba_uniform | [151, 251, 252] | 0.082–0.096 | 0.089–0.092 | 0.927–1.116× | 0.884–1.123× |
| `real_wine_raw_k5` | regress_uniform | [151, 251, 252] | 0.084–0.095 | 0.091–0.095 | 0.984–1.123× | 1.020–1.094× |
| `real_wine_raw_k5` | predict_distance | [151, 251, 252] | 0.095–0.099 | 0.095–0.103 | 0.958–1.065× | 0.983–1.092× |
| `real_wine_raw_k5` | proba_distance | [151, 251, 252] | 0.100–0.109 | 0.097–0.101 | 0.922–0.968× | 0.902–0.969× |
| `real_wine_raw_k5` | regress_distance | [151, 251, 252] | 0.103–0.107 | 0.094–0.102 | 0.908–0.988× | 0.908–1.002× |
| `distribution_gaussian_50000x32_k5` | kneighbors | [151, 251, 252, 351, 451, 552] | 0.424–2.390 | 0.423–2.347 | 0.982–1.026× | 0.972–1.021× |
| `distribution_gaussian_50000x32_k5` | fit_query | [151, 251, 252, 351, 451, 552] | 0.800–2.501 | 0.799–2.482 | 0.967–1.027× | 0.972–1.066× |
| `distribution_clusters_50000x32_k5` | kneighbors | [151, 251, 252] | 0.793–0.803 | 0.781–0.803 | 0.982–1.001× | 0.974–1.012× |
| `distribution_clusters_50000x32_k5` | fit_query | [151, 251, 252] | 1.093–1.149 | 1.120–1.158 | 1.001–1.025× | 0.992–1.017× |
| `distribution_offset_1e4_50000x32_k5` | kneighbors | [151, 251, 252] | 0.436–0.478 | 0.440–0.494 | 0.972–1.033× | 0.968–1.016× |
| `distribution_offset_1e4_50000x32_k5` | fit_query | [151, 251, 252] | 0.846–0.909 | 0.861–0.891 | 0.979–1.018× | 0.944–1.037× |
| `distribution_anisotropic_50000x32_k5` | kneighbors | [151, 251, 252] | 0.512–0.520 | 0.511–0.525 | 0.997–1.010× | 0.993–1.027× |
| `distribution_anisotropic_50000x32_k5` | fit_query | [151, 251, 252] | 0.879–0.904 | 0.886–0.931 | 0.997–1.031× | 0.999–1.051× |
| `distribution_gaussian_4096x128_k5` | kneighbors | [151, 251, 252, 351, 451, 552] | 0.396–0.460 | 0.410–0.444 | 0.935–1.051× | 0.950–1.043× |
| `distribution_gaussian_4096x128_k5` | fit_query | [151, 251, 252, 351, 451, 552] | 0.672–0.720 | 0.624–0.681 | 0.920–1.014× | 0.926–1.019× |
| `distribution_clusters_4096x128_k5` | kneighbors | [151, 251, 252] | 0.386–0.398 | 0.381–0.401 | 0.980–1.009× | 0.985–1.005× |
| `distribution_clusters_4096x128_k5` | fit_query | [151, 251, 252] | 0.584–0.592 | 0.575–0.577 | 0.974–0.984× | 0.981–1.014× |
| `distribution_offset_1e4_4096x128_k5` | kneighbors | [151, 251, 252] | 0.397–0.425 | 0.393–0.438 | 0.970–1.082× | 0.975–1.014× |
| `distribution_offset_1e4_4096x128_k5` | fit_query | [151, 251, 252] | 0.688–0.700 | 0.634–0.683 | 0.911–0.977× | 0.919–1.007× |
| `distribution_anisotropic_4096x128_k5` | kneighbors | [151, 251, 252] | 0.377–0.403 | 0.374–0.429 | 0.971–1.063× | 0.968–1.036× |
| `distribution_anisotropic_4096x128_k5` | fit_query | [151, 251, 252] | 0.675–0.688 | 0.633–0.726 | 0.920–1.059× | 0.998–1.065× |

## Observed backend routing

Counts include every warmup, probe and retained query, with cold calls separate. Fit storage is reported independently of actual query execution.

| Variant | Phase | Requests | Query backends | Fit storage |
| --- | --- | ---: | --- | --- |
| current | cold | 48 | `{"cpu": 18, "mlx": 30}` | `{"cpu": 18, "mlx": 30}` |
| current | warmup | 259706 | `{"cpu": 238794, "mlx": 20912}` | `{"cpu": 238794, "mlx": 20912}` |
| current | probe | 222 | `{"cpu": 162, "mlx": 60}` | `{"cpu": 162, "mlx": 60}` |
| current | retained | 6882 | `{"cpu": 5022, "mlx": 1860}` | `{"cpu": 5022, "mlx": 1860}` |
| current_b | cold | 48 | `{"cpu": 18, "mlx": 30}` | `{"cpu": 18, "mlx": 30}` |
| current_b | warmup | 257134 | `{"cpu": 235730, "mlx": 21404}` | `{"cpu": 235730, "mlx": 21404}` |
| current_b | probe | 222 | `{"cpu": 162, "mlx": 60}` | `{"cpu": 162, "mlx": 60}` |
| current_b | retained | 6882 | `{"cpu": 5022, "mlx": 1860}` | `{"cpu": 5022, "mlx": 1860}` |
| current_gpu | cold | 48 | `{"mlx": 48}` | `{"mlx": 48}` |
| current_gpu | warmup | 159990 | `{"mlx": 159990}` | `{"mlx": 159990}` |
| current_gpu | probe | 222 | `{"mlx": 222}` | `{"mlx": 222}` |
| current_gpu | retained | 6882 | `{"mlx": 6882}` | `{"mlx": 6882}` |
| previous_auto | cold | 48 | `{"cpu": 18, "mlx": 30}` | `{"cpu": 12, "mlx": 36}` |
| previous_auto | warmup | 250522 | `{"cpu": 229042, "mlx": 21480}` | `{"cpu": 143474, "mlx": 107048}` |
| previous_auto | probe | 222 | `{"cpu": 162, "mlx": 60}` | `{"cpu": 108, "mlx": 114}` |
| previous_auto | retained | 6882 | `{"cpu": 5022, "mlx": 1860}` | `{"cpu": 3348, "mlx": 3534}` |
| previous_gpu | cold | 48 | `{"mlx": 48}` | `{"mlx": 48}` |
| previous_gpu | warmup | 161539 | `{"mlx": 161539}` | `{"mlx": 161539}` |
| previous_gpu | probe | 222 | `{"mlx": 222}` | `{"mlx": 222}` |
| previous_gpu | retained | 6882 | `{"mlx": 6882}` | `{"mlx": 6882}` |

The full sklearn panel is a separate run on its unchanged 72 fixtures and all nine operations. These controlled comparisons isolate source changes; absolute timings should not be compared across runs with different warmup protocols.
