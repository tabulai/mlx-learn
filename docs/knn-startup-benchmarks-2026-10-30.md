# KNN startup and first-query fix against three sklearn configurations

Completed `2026-10-30T23:48:24.584153+00:00`. This run covers **72 dataset instances in 24 families**, 9 search/inference operations, and **15 retained calls after 5 warmups** per dataset-operation-variant. Fit-only is recorded separately and excluded from the headline 2× counts. Production source and fixture identities are recorded in the raw results.

| sklearn baseline | Medians ≥2× | Individual paired calls ≥2× | All medians ≥2×? | Every pair ≥2×? |
| --- | --- | --- | --- | --- |
| Brute, default threads | 583/648 (90.0%) | 8,744/9,720 (90.0%) | False | False |
| Auto, default threads | 600/648 (92.6%) | 8,887/9,720 (91.4%) | False | False |
| Brute, one thread | 302/648 (46.6%) | 5,036/9,720 (51.8%) | False | False |

Against the fastest median among the three measured sklearn configurations in each dataset-operation cell, **263/648 (40.6%)** reach 2×. This is a best-of-measured-configurations comparison, not one fixed sklearn setting. The panel weights each dataset-operation cell equally; percentages do not estimate real-world workload frequencies. Raw timing counts retain every baseline accuracy failure.

**9/24 families** reach 2× in every seed-operation median against all three baselines; **7/24** also reach 2× in every retained paired call. A family passes only when all its constituent cases and operations pass, without averaging away misses.

![Minimum speedup across seeds](assets/knn-startup-benchmarks-2026-10-30.png)

## Controlled automatic dispatch and GPU comparison

5 serial workers compare current automatic dispatch, an identical automatic A/A worker, current forced GPU and the preceding frozen small-workload GPU implementation. A fifth worker uses the preceding automatic source and its exact archived native binary. The unchanged control covers 48 fixtures and 222 case-operations, with 31 paired rounds after 0.25 seconds of continuous warmup per worker/operation. Default continuous warmup is 0.25 seconds (the preceding small control used 1 second); this is configurable and absolute timings must not be compared across those runs. Each worker/operation warms continuously for at least the requested duration. Its one post-warmup probe is audited and retained separately, excluded from paired ratios. All warmup audits and probe samples remain in the report.

Ratios are comparator/automatic time. The ratio of per-worker medians and median paired ratio are shown separately. A/A exposes process/timing variation; these are descriptive comparisons, not confidence intervals. Each worker retains its actual routing metadata.

Control accuracy and integrity acceptance: **True**. Strict quality: `{"current": true, "current_b": true, "current_gpu": true, "previous_auto": true, "previous_gpu": true}`.

| Family | Operation | Seeds | Auto ms | Current GPU/auto medians | Current GPU/auto paired | Previous GPU/auto medians | Previous GPU/auto paired | Auto A/A paired |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | fit_query | 41, 42, 43 | 0.144–0.146 | 2.050–2.348× | 2.175–2.436× | 2.138–2.611× | 2.224–2.702× | 0.980–1.002× |
| Gaussian 8D, single query | kneighbors | 41, 42, 43 | 0.084–0.089 | 2.406–3.011× | 2.313–3.116× | 2.198–2.986× | 2.183–3.088× | 0.963–1.054× |
| Gaussian 8D, single query | indices | 41, 42, 43 | 0.083–0.086 | 2.915–3.135× | 3.028–3.284× | 2.918–3.203× | 3.043–3.511× | 0.980–1.051× |
| Gaussian 8D, single query | predict_uniform | 41, 42, 43 | 0.087–0.110 | 2.295–2.918× | 2.286–2.956× | 2.380–3.182× | 2.395–3.232× | 0.934–1.046× |
| Gaussian 8D, single query | proba_uniform | 41, 42, 43 | 0.089–0.097 | 2.507–3.279× | 2.569–3.512× | 2.663–3.145× | 2.677–3.397× | 0.969–1.064× |
| Gaussian 8D, single query | regress_uniform | 41, 42, 43 | 0.092–0.096 | 2.506–2.915× | 2.457–3.134× | 2.331–2.924× | 2.297–3.124× | 0.956–1.016× |
| Gaussian 8D, single query | predict_distance | 41, 42, 43 | 0.094–0.104 | 2.562–3.166× | 2.368–3.255× | 2.356–3.063× | 2.239–3.158× | 1.004–1.072× |
| Gaussian 8D, single query | proba_distance | 41, 42, 43 | 0.096–0.103 | 2.784–3.267× | 2.866–3.612× | 2.853–3.161× | 2.878–3.139× | 0.938–1.116× |
| Gaussian 8D, single query | regress_distance | 41, 42, 43 | 0.100–0.105 | 2.183–2.761× | 2.091–2.914× | 2.170–2.949× | 2.079–3.034× | 0.942–0.986× |
| Gaussian 32D, single query | fit_query | 49, 50, 51 | 0.209–0.217 | 1.727–1.952× | 1.795–2.082× | 1.885–2.059× | 1.890–2.175× | 0.981–1.027× |
| Gaussian 32D, single query | kneighbors | 49, 50, 51 | 0.133–0.137 | 2.274–2.366× | 2.303–2.425× | 2.298–2.496× | 2.375–2.585× | 0.984–1.017× |
| Gaussian 32D, single query | indices | 49, 50, 51 | 0.129–0.138 | 1.518–2.355× | 1.488–2.349× | 1.503–2.263× | 1.503–2.319× | 0.973–1.028× |
| Gaussian 32D, single query | predict_uniform | 49, 50, 51 | 0.138–0.141 | 1.682–2.568× | 1.709–2.649× | 2.116–2.567× | 1.876–2.809× | 0.969–1.051× |
| Gaussian 32D, single query | proba_uniform | 49, 50, 51 | 0.139–0.149 | 1.683–2.743× | 1.647–2.646× | 1.587–2.550× | 1.578–2.535× | 0.957–1.019× |
| Gaussian 32D, single query | regress_uniform | 49, 50, 51 | 0.138–0.153 | 1.505–2.366× | 1.532–2.350× | 1.524–2.248× | 1.504–2.242× | 0.950–1.055× |
| Gaussian 32D, single query | predict_distance | 49, 50, 51 | 0.149–0.160 | 2.094–2.493× | 2.122–2.533× | 2.182–2.445× | 2.158–2.457× | 0.976–1.058× |
| Gaussian 32D, single query | proba_distance | 49, 50, 51 | 0.151–0.164 | 1.550–2.262× | 1.513–2.381× | 1.512–2.257× | 1.523–2.364× | 0.933–0.992× |
| Gaussian 32D, single query | regress_distance | 49, 50, 51 | 0.144–0.154 | 1.463–2.353× | 1.467–2.440× | 1.565–2.232× | 1.533–2.285× | 1.001–1.050× |
| Digits (raw) | fit_query | 151, 251, 252 | 0.427–0.544 | 1.503–2.432× | 1.621–2.317× | 1.880–2.362× | 1.854–2.369× | 0.846–1.089× |
| Digits (raw) | kneighbors | 151, 251, 252 | 0.369–0.393 | 1.486–2.597× | 1.483–2.771× | 2.057–2.579× | 1.990–2.846× | 0.805–0.987× |
| Digits (raw) | indices | 151, 251, 252 | 0.367–0.386 | 1.119–2.584× | 1.110–2.509× | 1.086–2.562× | 1.177–2.443× | 0.966–1.025× |
| Digits (raw) | predict_uniform | 151, 251, 252 | 0.322–0.466 | 1.136–2.571× | 1.206–2.623× | 1.243–1.430× | 1.274–1.499× | 0.985–1.168× |
| Digits (raw) | proba_uniform | 151, 251, 252 | 0.324–0.370 | 1.171–2.623× | 1.195–2.471× | 1.130–2.746× | 1.180–2.574× | 0.871–1.130× |
| Digits (raw) | regress_uniform | 151, 251, 252 | 0.299–0.366 | 1.067–2.981× | 1.038–2.949× | 1.105–2.654× | 1.092–2.736× | 0.846–1.280× |
| Digits (raw) | predict_distance | 151, 251, 252 | 0.388–0.405 | 1.101–1.990× | 1.171–1.801× | 1.079–2.268× | 1.198–2.120× | 0.858–0.933× |
| Digits (raw) | proba_distance | 151, 251, 252 | 0.396–0.442 | 1.024–1.222× | 1.083–1.332× | 1.075–1.265× | 1.058–1.501× | 0.918–1.001× |
| Digits (raw) | regress_distance | 151, 251, 252 | 0.348–0.397 | 1.253–2.675× | 1.326–2.301× | 1.186–2.660× | 1.229–2.244× | 0.995–1.205× |
| Breast Cancer (raw) | fit_query | 151, 251, 252 | 0.193–0.226 | 1.463–2.238× | 1.483–2.244× | 1.621–1.995× | 1.660–2.032× | 0.976–1.127× |
| Breast Cancer (raw) | kneighbors | 151, 251, 252 | 0.134–0.147 | 1.454–1.898× | 1.315–1.847× | 1.416–2.325× | 1.340–2.337× | 1.010–1.210× |
| Breast Cancer (raw) | indices | 151, 251, 252 | 0.138–0.191 | 1.389–1.850× | 1.367–1.814× | 1.412–1.814× | 1.421–1.784× | 0.869–1.117× |
| Breast Cancer (raw) | predict_uniform | 151, 251, 252 | 0.154–0.159 | 1.442–1.864× | 1.327–1.955× | 1.529–1.874× | 1.537–1.828× | 0.922–1.036× |
| Breast Cancer (raw) | proba_uniform | 151, 251, 252 | 0.149–0.150 | 1.353–2.003× | 1.378–2.070× | 1.426–2.062× | 1.445–2.084× | 1.023–1.078× |
| Breast Cancer (raw) | regress_uniform | 151, 251, 252 | 0.138–0.164 | 1.566–2.017× | 1.600–1.829× | 1.538–1.787× | 1.625–1.777× | 0.968–1.171× |
| Breast Cancer (raw) | predict_distance | 151, 251, 252 | 0.151–0.172 | 1.533–1.851× | 1.471–1.973× | 1.421–1.978× | 1.475–2.127× | 0.947–1.143× |
| Breast Cancer (raw) | proba_distance | 151, 251, 252 | 0.159–0.189 | 1.204–1.733× | 1.198–1.761× | 1.245–1.865× | 1.307–1.904× | 0.895–1.094× |
| Breast Cancer (raw) | regress_distance | 151, 251, 252 | 0.166–0.180 | 1.452–1.675× | 1.421–1.756× | 1.481–1.802× | 1.462–1.750× | 0.962–1.031× |
| Iris (raw) | fit_query | 151, 251, 252 | 0.093–0.096 | 3.290–3.704× | 3.242–4.126× | 3.807–4.065× | 3.745–4.269× | 0.900–1.072× |
| Iris (raw) | kneighbors | 151, 251, 252 | 0.069–0.074 | 3.345–3.848× | 3.085–4.432× | 3.018–3.875× | 2.927–3.916× | 0.988–1.030× |
| Iris (raw) | indices | 151, 251, 252 | 0.061–0.070 | 2.955–4.615× | 2.827–5.133× | 2.791–4.398× | 2.785–4.966× | 0.993–1.139× |
| Iris (raw) | predict_uniform | 151, 251, 252 | 0.068–0.077 | 3.026–4.640× | 2.874–4.754× | 2.787–4.495× | 2.866–4.653× | 0.973–1.178× |
| Iris (raw) | proba_uniform | 151, 251, 252 | 0.075–0.078 | 2.886–4.117× | 2.938–4.414× | 3.173–4.047× | 3.625–4.179× | 0.954–1.083× |
| Iris (raw) | regress_uniform | 151, 251, 252 | 0.076–0.079 | 2.655–3.952× | 2.693–4.643× | 2.798–3.728× | 2.794–4.160× | 0.937–0.999× |
| Iris (raw) | predict_distance | 151, 251, 252 | 0.086–0.104 | 2.437–3.103× | 2.440–3.165× | 2.598–3.014× | 2.684–3.121× | 0.956–1.029× |
| Iris (raw) | proba_distance | 151, 251, 252 | 0.085–0.101 | 2.922–3.644× | 2.780–4.104× | 2.787–4.352× | 2.391–4.365× | 0.965–1.150× |
| Iris (raw) | regress_distance | 151, 251, 252 | 0.087–0.088 | 2.819–3.918× | 2.730–4.578× | 2.559–3.539× | 2.493–4.047× | 0.924–1.065× |
| Wine (raw) | fit_query | 151, 251, 252 | 0.096–0.108 | 2.608–3.229× | 2.596–3.328× | 2.719–3.462× | 2.805–3.506× | 0.920–1.050× |
| Wine (raw) | kneighbors | 151, 251, 252 | 0.076–0.083 | 2.397–3.292× | 2.390–3.452× | 2.520–3.243× | 2.516–3.487× | 0.899–1.113× |
| Wine (raw) | indices | 151, 251, 252 | 0.080–0.083 | 2.548–3.236× | 2.600–3.093× | 2.743–3.033× | 2.868–3.097× | 0.891–1.038× |
| Wine (raw) | predict_uniform | 151, 251, 252 | 0.084–0.092 | 2.387–3.268× | 2.305–3.313× | 2.521–3.225× | 2.555–3.341× | 0.970–0.990× |
| Wine (raw) | proba_uniform | 151, 251, 252 | 0.080–0.093 | 2.472–3.390× | 2.447–3.459× | 2.419–3.201× | 2.431–3.339× | 0.959–1.206× |
| Wine (raw) | regress_uniform | 151, 251, 252 | 0.084–0.091 | 2.471–3.022× | 2.386–3.265× | 2.374–3.826× | 2.374–3.486× | 0.960–1.048× |
| Wine (raw) | predict_distance | 151, 251, 252 | 0.092–0.097 | 2.125–3.161× | 2.167–3.125× | 2.406–3.473× | 2.386–3.240× | 0.926–1.070× |
| Wine (raw) | proba_distance | 151, 251, 252 | 0.093–0.100 | 2.318–3.344× | 2.222–3.383× | 2.398–3.014× | 2.270–3.041× | 0.930–0.967× |
| Wine (raw) | regress_distance | 151, 251, 252 | 0.095–0.105 | 2.363–2.654× | 2.381–2.862× | 2.156–2.700× | 2.215–2.782× | 0.937–1.020× |
| Gaussian 50k × 32 | kneighbors | 151, 251, 252, 351, 451, 552 | 0.426–2.374 | 0.955–1.020× | 0.965–1.021× | 0.972–1.007× | 0.970–1.008× | 0.966–1.031× |
| Gaussian 50k × 32 | fit_query | 151, 251, 252, 351, 451, 552 | 0.762–2.456 | 0.966–1.012× | 0.976–1.018× | 1.013–1.067× | 1.016–1.061× | 0.991–1.027× |
| Clusters 50k × 32 | kneighbors | 151, 251, 252 | 0.754–0.815 | 0.920–1.058× | 0.913–1.055× | 0.943–1.044× | 0.962–1.023× | 0.987–1.005× |
| Clusters 50k × 32 | fit_query | 151, 251, 252 | 1.050–1.082 | 0.986–1.029× | 0.994–1.014× | 1.038–1.068× | 1.027–1.049× | 1.014–1.022× |
| Offset +1e4, 50k × 32 | kneighbors | 151, 251, 252 | 0.446–0.506 | 0.905–0.993× | 0.945–0.979× | 0.923–1.016× | 0.965–0.999× | 0.973–0.994× |
| Offset +1e4, 50k × 32 | fit_query | 151, 251, 252 | 0.831–0.857 | 0.970–0.985× | 0.960–0.993× | 1.010–1.044× | 0.999–1.056× | 0.980–0.994× |
| Unequal scales 50k × 32 | kneighbors | 151, 251, 252 | 0.502–1.504 | 0.967–1.013× | 0.974–1.005× | 0.998–1.019× | 0.994–1.011× | 0.981–1.010× |
| Unequal scales 50k × 32 | fit_query | 151, 251, 252 | 0.837–0.859 | 0.979–1.001× | 0.978–1.006× | 1.023–1.056× | 1.025–1.056× | 0.985–1.022× |
| Gaussian 4096 × 128 | kneighbors | 151, 251, 252, 351, 451, 552 | 0.391–0.416 | 0.921–1.003× | 0.941–1.019× | 0.969–1.035× | 0.971–1.035× | 0.968–1.040× |
| Gaussian 4096 × 128 | fit_query | 151, 251, 252, 351, 451, 552 | 0.604–0.665 | 0.940–1.034× | 0.925–1.025× | 1.016–1.125× | 0.998–1.106× | 0.974–1.027× |
| Clusters 4096 × 128 | kneighbors | 151, 251, 252 | 0.386–0.388 | 0.982–1.002× | 0.996–1.018× | 0.985–1.021× | 0.971–1.011× | 1.003–1.025× |
| Clusters 4096 × 128 | fit_query | 151, 251, 252 | 0.509–0.548 | 0.952–1.017× | 0.964–1.025× | 0.992–1.078× | 1.003–1.095× | 0.983–1.038× |
| Offset +1e4, 4096 × 128 | kneighbors | 151, 251, 252 | 0.395–0.635 | 0.939–0.997× | 0.946–0.995× | 0.998–1.035× | 0.998–1.014× | 0.965–1.005× |
| Offset +1e4, 4096 × 128 | fit_query | 151, 251, 252 | 0.615–0.653 | 0.931–0.979× | 0.941–0.982× | 1.039–1.064× | 1.026–1.055× | 0.971–0.980× |
| Unequal scales 4096 × 128 | kneighbors | 151, 251, 252 | 0.393–0.405 | 0.951–1.052× | 0.967–1.045× | 0.973–1.046× | 0.988–1.027× | 0.985–1.076× |
| Unequal scales 4096 × 128 | fit_query | 151, 251, 252 | 0.630–0.650 | 0.956–1.063× | 0.956–1.022× | 1.046–1.160× | 1.052–1.095× | 0.977–1.066× |

### Comparison with preceding automatic dispatch

| Family | Operation | Seeds | Current auto ms | Previous auto ms | Previous/current medians | Previous/current paired |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | fit_query | 41, 42, 43 | 0.144–0.146 | 0.165–0.174 | 1.150–1.192× | 1.149–1.239× |
| Gaussian 8D, single query | kneighbors | 41, 42, 43 | 0.084–0.089 | 0.092–0.094 | 1.039–1.099× | 1.004–1.104× |
| Gaussian 8D, single query | indices | 41, 42, 43 | 0.083–0.086 | 0.091–0.094 | 1.087–1.118× | 1.101–1.140× |
| Gaussian 8D, single query | predict_uniform | 41, 42, 43 | 0.087–0.110 | 0.093–0.122 | 1.034–1.105× | 1.051–1.093× |
| Gaussian 8D, single query | proba_uniform | 41, 42, 43 | 0.089–0.097 | 0.096–0.102 | 0.992–1.142× | 0.997–1.143× |
| Gaussian 8D, single query | regress_uniform | 41, 42, 43 | 0.092–0.096 | 0.095–0.100 | 0.990–1.081× | 0.998–1.106× |
| Gaussian 8D, single query | predict_distance | 41, 42, 43 | 0.094–0.104 | 0.106–0.115 | 1.063–1.125× | 1.062–1.177× |
| Gaussian 8D, single query | proba_distance | 41, 42, 43 | 0.096–0.103 | 0.103–0.109 | 1.001–1.115× | 0.997–1.109× |
| Gaussian 8D, single query | regress_distance | 41, 42, 43 | 0.100–0.105 | 0.109–0.110 | 1.033–1.093× | 1.066–1.164× |
| Gaussian 32D, single query | fit_query | 49, 50, 51 | 0.209–0.217 | 0.247–0.248 | 1.141–1.181× | 1.175–1.186× |
| Gaussian 32D, single query | kneighbors | 49, 50, 51 | 0.133–0.137 | 0.135–0.145 | 1.008–1.055× | 0.979–1.058× |
| Gaussian 32D, single query | indices | 49, 50, 51 | 0.129–0.138 | 0.136–0.143 | 1.022–1.107× | 1.023–1.071× |
| Gaussian 32D, single query | predict_uniform | 49, 50, 51 | 0.138–0.141 | 0.140–0.147 | 0.999–1.039× | 1.002–1.051× |
| Gaussian 32D, single query | proba_uniform | 49, 50, 51 | 0.139–0.149 | 0.144–0.151 | 1.015–1.051× | 0.978–1.055× |
| Gaussian 32D, single query | regress_uniform | 49, 50, 51 | 0.138–0.153 | 0.146–0.154 | 0.999–1.071× | 0.981–1.113× |
| Gaussian 32D, single query | predict_distance | 49, 50, 51 | 0.149–0.160 | 0.155–0.168 | 1.015–1.132× | 0.978–1.148× |
| Gaussian 32D, single query | proba_distance | 49, 50, 51 | 0.151–0.164 | 0.156–0.165 | 1.001–1.078× | 1.014–1.031× |
| Gaussian 32D, single query | regress_distance | 49, 50, 51 | 0.144–0.154 | 0.150–0.159 | 1.008–1.048× | 0.989–1.079× |
| Digits (raw) | fit_query | 151, 251, 252 | 0.427–0.544 | 0.571–0.621 | 1.141–1.338× | 1.167–1.316× |
| Digits (raw) | kneighbors | 151, 251, 252 | 0.369–0.393 | 0.348–0.456 | 0.909–1.175× | 0.948–1.194× |
| Digits (raw) | indices | 151, 251, 252 | 0.367–0.386 | 0.394–0.446 | 1.074–1.156× | 1.066–1.186× |
| Digits (raw) | predict_uniform | 151, 251, 252 | 0.322–0.466 | 0.401–0.527 | 1.052–1.370× | 1.044–1.414× |
| Digits (raw) | proba_uniform | 151, 251, 252 | 0.324–0.370 | 0.396–0.459 | 1.071–1.415× | 1.030–1.337× |
| Digits (raw) | regress_uniform | 151, 251, 252 | 0.299–0.366 | 0.420–0.431 | 1.167–1.406× | 1.156–1.434× |
| Digits (raw) | predict_distance | 151, 251, 252 | 0.388–0.405 | 0.395–0.461 | 0.977–1.140× | 0.965–1.160× |
| Digits (raw) | proba_distance | 151, 251, 252 | 0.396–0.442 | 0.403–0.498 | 0.961–1.153× | 0.961–1.153× |
| Digits (raw) | regress_distance | 151, 251, 252 | 0.348–0.397 | 0.358–0.457 | 0.900–1.258× | 0.892–1.271× |
| Breast Cancer (raw) | fit_query | 151, 251, 252 | 0.193–0.226 | 0.266–0.284 | 1.178–1.473× | 1.207–1.492× |
| Breast Cancer (raw) | kneighbors | 151, 251, 252 | 0.134–0.147 | 0.184–0.193 | 1.251–1.443× | 1.282–1.471× |
| Breast Cancer (raw) | indices | 151, 251, 252 | 0.138–0.191 | 0.205–0.227 | 1.185–1.489× | 1.188–1.477× |
| Breast Cancer (raw) | predict_uniform | 151, 251, 252 | 0.154–0.159 | 0.187–0.200 | 1.172–1.271× | 1.158–1.285× |
| Breast Cancer (raw) | proba_uniform | 151, 251, 252 | 0.149–0.150 | 0.202–0.204 | 1.349–1.362× | 1.308–1.361× |
| Breast Cancer (raw) | regress_uniform | 151, 251, 252 | 0.138–0.164 | 0.187–0.224 | 1.191–1.433× | 1.143–1.379× |
| Breast Cancer (raw) | predict_distance | 151, 251, 252 | 0.151–0.172 | 0.211–0.220 | 1.237–1.457× | 1.267–1.455× |
| Breast Cancer (raw) | proba_distance | 151, 251, 252 | 0.159–0.189 | 0.214–0.237 | 1.159–1.366× | 1.164–1.351× |
| Breast Cancer (raw) | regress_distance | 151, 251, 252 | 0.166–0.180 | 0.196–0.218 | 1.089–1.317× | 1.117–1.313× |
| Iris (raw) | fit_query | 151, 251, 252 | 0.093–0.096 | 0.120–0.139 | 1.274–1.463× | 1.248–1.450× |
| Iris (raw) | kneighbors | 151, 251, 252 | 0.069–0.074 | 0.071–0.078 | 0.991–1.112× | 0.964–1.119× |
| Iris (raw) | indices | 151, 251, 252 | 0.061–0.070 | 0.069–0.073 | 1.006–1.189× | 0.961–1.227× |
| Iris (raw) | predict_uniform | 151, 251, 252 | 0.068–0.077 | 0.077–0.085 | 1.033–1.200× | 1.024–1.207× |
| Iris (raw) | proba_uniform | 151, 251, 252 | 0.075–0.078 | 0.085–0.092 | 1.087–1.216× | 1.073–1.152× |
| Iris (raw) | regress_uniform | 151, 251, 252 | 0.076–0.079 | 0.079–0.087 | 0.999–1.142× | 0.943–1.077× |
| Iris (raw) | predict_distance | 151, 251, 252 | 0.086–0.104 | 0.095–0.102 | 0.983–1.096× | 0.993–1.138× |
| Iris (raw) | proba_distance | 151, 251, 252 | 0.085–0.101 | 0.089–0.093 | 0.881–1.087× | 0.884–1.076× |
| Iris (raw) | regress_distance | 151, 251, 252 | 0.087–0.088 | 0.087–0.095 | 0.991–1.081× | 0.971–1.081× |
| Wine (raw) | fit_query | 151, 251, 252 | 0.096–0.108 | 0.131–0.137 | 1.267–1.358× | 1.220–1.370× |
| Wine (raw) | kneighbors | 151, 251, 252 | 0.076–0.083 | 0.087–0.089 | 1.044–1.174× | 1.044–1.239× |
| Wine (raw) | indices | 151, 251, 252 | 0.080–0.083 | 0.083–0.085 | 1.005–1.044× | 0.994–1.056× |
| Wine (raw) | predict_uniform | 151, 251, 252 | 0.084–0.092 | 0.082–0.095 | 0.979–1.038× | 0.981–1.046× |
| Wine (raw) | proba_uniform | 151, 251, 252 | 0.080–0.093 | 0.085–0.095 | 0.975–1.129× | 0.966–1.164× |
| Wine (raw) | regress_uniform | 151, 251, 252 | 0.084–0.091 | 0.092–0.095 | 1.005–1.131× | 0.985–1.158× |
| Wine (raw) | predict_distance | 151, 251, 252 | 0.092–0.097 | 0.098–0.103 | 1.039–1.115× | 1.045–1.099× |
| Wine (raw) | proba_distance | 151, 251, 252 | 0.093–0.100 | 0.097–0.102 | 0.965–1.070× | 0.952–1.066× |
| Wine (raw) | regress_distance | 151, 251, 252 | 0.095–0.105 | 0.096–0.105 | 0.973–1.111× | 0.976–1.152× |
| Gaussian 50k × 32 | kneighbors | 151, 251, 252, 351, 451, 552 | 0.426–2.374 | 0.426–2.367 | 0.975–1.012× | 0.963–1.013× |
| Gaussian 50k × 32 | fit_query | 151, 251, 252, 351, 451, 552 | 0.762–2.456 | 0.795–2.480 | 1.010–1.063× | 1.004–1.068× |
| Clusters 50k × 32 | kneighbors | 151, 251, 252 | 0.754–0.815 | 0.770–0.801 | 0.980–1.045× | 0.985–1.021× |
| Clusters 50k × 32 | fit_query | 151, 251, 252 | 1.050–1.082 | 1.088–1.167 | 1.020–1.079× | 1.032–1.061× |
| Offset +1e4, 50k × 32 | kneighbors | 151, 251, 252 | 0.446–0.506 | 0.450–0.483 | 0.944–1.009× | 0.979–1.019× |
| Offset +1e4, 50k × 32 | fit_query | 151, 251, 252 | 0.831–0.857 | 0.873–0.886 | 1.031–1.052× | 1.018–1.043× |
| Unequal scales 50k × 32 | kneighbors | 151, 251, 252 | 0.502–1.504 | 0.498–1.510 | 0.985–1.028× | 1.002–1.025× |
| Unequal scales 50k × 32 | fit_query | 151, 251, 252 | 0.837–0.859 | 0.886–0.904 | 1.052–1.059× | 1.037–1.071× |
| Gaussian 4096 × 128 | kneighbors | 151, 251, 252, 351, 451, 552 | 0.391–0.416 | 0.395–0.423 | 0.950–1.034× | 0.979–1.044× |
| Gaussian 4096 × 128 | fit_query | 151, 251, 252, 351, 451, 552 | 0.604–0.665 | 0.624–0.705 | 1.035–1.106× | 1.029–1.114× |
| Clusters 4096 × 128 | kneighbors | 151, 251, 252 | 0.386–0.388 | 0.387–0.392 | 0.999–1.015× | 0.993–1.022× |
| Clusters 4096 × 128 | fit_query | 151, 251, 252 | 0.509–0.548 | 0.561–0.568 | 1.033–1.102× | 1.033–1.120× |
| Offset +1e4, 4096 × 128 | kneighbors | 151, 251, 252 | 0.395–0.635 | 0.396–0.625 | 0.985–1.058× | 0.973–1.003× |
| Offset +1e4, 4096 × 128 | fit_query | 151, 251, 252 | 0.615–0.653 | 0.620–0.690 | 1.004–1.069× | 0.986–1.054× |
| Unequal scales 4096 × 128 | kneighbors | 151, 251, 252 | 0.393–0.405 | 0.397–0.423 | 0.982–1.077× | 1.002–1.045× |
| Unequal scales 4096 × 128 | fit_query | 151, 251, 252 | 0.630–0.650 | 0.665–0.759 | 1.057–1.168× | 1.049–1.129× |

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
| Gaussian 8D, single query | 8,192 × 8; 1; 5 | 0.106–0.110 | 7.20–8.50× | 1.76–2.05× | 2.20–2.43× | 27 / 9 / 24 (each /27) |
| Gaussian 8D, k5 | 8,192 × 8; 256; 5 | 0.747–0.825 | 2.06–2.73× | 5.26–6.05× | 5.28–5.86× | 6 / 27 / 27 (each /27) |
| Gaussian 32D, k5 | 8,192 × 32; 256; 5 | 0.970–1.021 | 4.19–4.65× | 4.36–4.65× | 4.44–4.65× | 27 / 27 / 27 (each /27) |
| Gaussian 32D, k32 | 8,192 × 32; 128; 32 | 1.914–1.957 | 1.94–2.45× | 2.00–2.45× | 1.59–1.62× | 10 / 13 / 0 (each /27) |
| Gaussian 128D, k5 | 4,096 × 128; 128; 5 | 0.843–0.960 | 3.38–3.75× | 3.34–3.81× | 1.82–2.06× | 27 / 27 / 1 (each /27) |
| Gaussian 1024D, k5 | 2,048 × 1024; 32; 5 | 1.084–1.301 | 2.82–3.30× | 2.84–3.21× | 1.74–2.07× | 27 / 27 / 2 (each /27) |
| Gaussian 32D, k64 | 4,096 × 32; 128; 64 | 1.495–1.530 | 2.60–2.72× | 2.64–2.75× | 1.70–1.78× | 27 / 27 / 0 (each /27) |
| Gaussian 16D, k5 | 8,192 × 16; 256; 5 | 0.788–0.827 | 4.38–4.70× | 4.42–4.79× | 5.22–5.48× | 27 / 27 / 27 (each /27) |
| Gaussian 32D, single query | 8,192 × 32; 1; 5 | 0.168–0.179 | 16.18–17.73× | 16.95–17.51× | 1.50–1.63× | 27 / 27 / 0 (each /27) |
| Gaussian 8D, k32 | 8,192 × 8; 128; 32 | 1.819–1.863 | 0.83–0.87× | 2.20–2.26× | 1.60–1.64× | 0 / 27 / 0 (each /27) |
| Gaussian 50k × 32 (original panel) | 50,000 × 32; 256; 5 | 1.442–1.499 | 4.52–4.62× | 4.36–4.80× | 17.32–17.91× | 27 / 27 / 27 (each /27) |
| Gaussian 50k × 32 | 50,000 × 32; 256; 5 | 1.388–1.477 | 4.57–4.73× | 4.49–4.65× | 17.49–18.58× | 27 / 27 / 27 (each /27) |
| Clusters 50k × 32 | 50,000 × 32; 256; 5 | 2.740–2.750 | 2.44–2.45× | 2.38–2.55× | 9.39–9.41× | 27 / 27 / 27 (each /27) |
| Offset +1e4, 50k × 32 | 50,000 × 32; 256; 5 | 1.413–1.589 | 4.33–4.92× | 4.24–4.82× | 16.23–18.52× | 27 / 27 / 27 (each /27) |
| Unequal scales 50k × 32 | 50,000 × 32; 256; 5 | 1.700–1.845 | 3.61–4.05× | 3.56–3.99× | 14.02–15.20× | 27 / 27 / 27 (each /27) |
| Gaussian 4096 × 128 | 4,096 × 128; 128; 5 | 0.881–0.953 | 3.38–3.72× | 3.52–3.79× | 1.81–1.98× | 27 / 27 / 1 (each /27) |
| Clusters 4096 × 128 | 4,096 × 128; 128; 5 | 1.142–1.200 | 2.64–2.80× | 2.69–2.84× | 1.43–1.52× | 27 / 27 / 1 (each /27) |
| Offset +1e4, 4096 × 128 | 4,096 × 128; 128; 5 | 0.940–0.965 | 3.44–3.46× | 3.39–3.48× | 1.80–1.87× | 27 / 27 / 1 (each /27) |
| Unequal scales 4096 × 128 | 4,096 × 128; 128; 5 | 0.861–0.954 | 3.41–3.79× | 3.36–3.73× | 1.85–2.01× | 27 / 27 / 2 (each /27) |
| Gaussian 100k × 32 | 100,000 × 32; 256; 5 | 2.020–2.110 | 4.48–4.99× | 4.66–4.84× | 24.46–25.54× | 27 / 27 / 27 (each /27) |
| Digits (raw) | 1,541 × 64; 256; 5 | 0.431–0.449 | 6.58–6.82× | 6.59–6.95× | 2.63–2.66× | 27 / 27 / 27 (each /27) |
| Breast Cancer (raw) | 455 × 30; 114; 5 | 0.183–0.235 | 10.92–11.71× | 9.68–11.56× | 1.48–1.75× | 27 / 27 / 0 (each /27) |
| Iris (raw) | 120 × 4; 30; 5 | 0.084–0.091 | 6.45–6.78× | 2.26–2.41× | 1.42–1.63× | 27 / 23 / 0 (each /27) |
| Wine (raw) | 142 × 13; 36; 5 | 0.094–0.103 | 5.94–6.76× | 1.95–2.09× | 1.31–1.38× | 27 / 15 / 0 (each /27) |

## Fresh fit + query

| Family | Current fresh ms | vs default brute | vs auto | vs one thread |
| --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 0.173–0.183 | 4.78–5.49× | 9.01–9.58× | 1.89–1.94× |
| Gaussian 8D, k5 | 0.808–0.848 | 1.81–2.49× | 6.94–7.55× | 5.20–5.54× |
| Gaussian 32D, k5 | 1.065–1.172 | 4.14–4.54× | 3.79–4.21× | 3.99–4.33× |
| Gaussian 32D, k32 | 2.081–2.173 | 1.82–2.28× | 1.82–2.38× | 1.49–1.53× |
| Gaussian 128D, k5 | 0.970–1.280 | 2.60–3.46× | 2.65–3.46× | 1.48–1.96× |
| Gaussian 1024D, k5 | 1.743–1.893 | 2.04–2.29× | 2.15–2.32× | 1.37–1.49× |
| Gaussian 32D, k64 | 1.697–1.722 | 2.41–2.46× | 2.37–2.46× | 1.55–1.60× |
| Gaussian 16D, k5 | 0.901–0.978 | 4.02–4.14× | 3.94–4.20× | 4.54–4.91× |
| Gaussian 32D, single query | 0.248–0.252 | 11.53–12.17× | 12.09–12.47× | 1.43–1.48× |
| Gaussian 8D, k32 | 2.018–2.093 | 0.80–0.82× | 2.68–2.82× | 1.48–1.54× |
| Gaussian 50k × 32 (original panel) | 1.784–1.897 | 3.89–3.93× | 3.74–3.97× | 13.98–14.82× |
| Gaussian 50k × 32 | 1.849–1.915 | 3.68–3.99× | 3.85–4.14× | 13.79–14.20× |
| Clusters 50k × 32 | 2.960–3.060 | 2.37–2.45× | 2.28–2.40× | 8.54–8.94× |
| Offset +1e4, 50k × 32 | 2.009–2.103 | 3.41–3.61× | 3.42–3.60× | 12.51–13.06× |
| Unequal scales 50k × 32 | 2.108–2.266 | 3.08–3.28× | 2.95–3.37× | 11.53–12.41× |
| Gaussian 4096 × 128 | 1.229–1.267 | 2.68–2.85× | 2.66–2.78× | 1.51–1.52× |
| Clusters 4096 × 128 | 1.245–1.363 | 2.47–2.70× | 2.45–2.68× | 1.37–1.50× |
| Offset +1e4, 4096 × 128 | 1.140–1.228 | 2.76–3.06× | 2.74–3.03× | 1.54–1.66× |
| Unequal scales 4096 × 128 | 1.155–1.207 | 2.81–2.94× | 2.80–2.95× | 1.57–1.64× |
| Gaussian 100k × 32 | 2.713–2.775 | 3.61–3.77× | 3.52–3.86× | 18.66–19.09× |
| Digits (raw) | 0.545–0.555 | 5.34–6.08× | 5.12–5.96× | 2.25–2.28× |
| Breast Cancer (raw) | 0.247–0.259 | 9.03–9.07× | 8.98–9.19× | 1.45–1.52× |
| Iris (raw) | 0.114–0.118 | 5.74–6.11× | 2.51–2.60× | 1.76–1.82× |
| Wine (raw) | 0.121–0.127 | 5.27–5.78× | 2.36–2.49× | 1.55–1.74× |

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

| Operation | Brute, default threads | Auto, default threads | Brute, one thread |
| --- | --- | --- | --- |
| `fit` (excluded from headline) | 0/72; 26/1,080 | 15/72; 219/1,080 | 0/72; 36/1,080 |
| `fit_query` | 65/72; 967/1,080 | 70/72; 1041/1,080 | 30/72; 490/1,080 |
| `kneighbors` | 67/72; 1011/1,080 | 68/72; 998/1,080 | 36/72; 575/1,080 |
| `indices` | 64/72; 946/1,080 | 66/72; 973/1,080 | 33/72; 566/1,080 |
| `predict_uniform` | 63/72; 958/1,080 | 69/72; 1035/1,080 | 33/72; 575/1,080 |
| `proba_uniform` | 63/72; 950/1,080 | 66/72; 960/1,080 | 35/72; 566/1,080 |
| `regress_uniform` | 63/72; 949/1,080 | 63/72; 943/1,080 | 33/72; 561/1,080 |
| `predict_distance` | 66/72; 991/1,080 | 71/72; 1016/1,080 | 35/72; 582/1,080 |
| `proba_distance` | 67/72; 992/1,080 | 65/72; 976/1,080 | 34/72; 567/1,080 |
| `regress_distance` | 65/72; 980/1,080 | 62/72; 945/1,080 | 33/72; 554/1,080 |

Family-level paired-call counts retain every individual threshold miss:

| Family | Brute, default threads | Auto, default threads | Brute, one thread |
| --- | --- | --- | --- |
| Gaussian 8D, single query | 405/405 | 172/405 | 348/405 |
| Gaussian 8D, k5 | 160/405 | 405/405 | 405/405 |
| Gaussian 32D, k5 | 405/405 | 405/405 | 405/405 |
| Gaussian 32D, k32 | 149/405 | 183/405 | 11/405 |
| Gaussian 128D, k5 | 401/405 | 401/405 | 91/405 |
| Gaussian 1024D, k5 | 377/405 | 380/405 | 84/405 |
| Gaussian 32D, k64 | 392/405 | 391/405 | 36/405 |
| Gaussian 16D, k5 | 405/405 | 405/405 | 405/405 |
| Gaussian 32D, single query | 405/405 | 405/405 | 6/405 |
| Gaussian 8D, k32 | 1/405 | 381/405 | 8/405 |
| Gaussian 50k × 32 (original panel) | 405/405 | 405/405 | 405/405 |
| Gaussian 50k × 32 | 405/405 | 405/405 | 405/405 |
| Clusters 50k × 32 | 400/405 | 396/405 | 405/405 |
| Offset +1e4, 50k × 32 | 405/405 | 405/405 | 405/405 |
| Unequal scales 50k × 32 | 405/405 | 405/405 | 405/405 |
| Gaussian 4096 × 128 | 404/405 | 403/405 | 110/405 |
| Clusters 4096 × 128 | 397/405 | 397/405 | 38/405 |
| Offset +1e4, 4096 × 128 | 404/405 | 404/405 | 103/405 |
| Unequal scales 4096 × 128 | 396/405 | 396/405 | 114/405 |
| Gaussian 100k × 32 | 405/405 | 405/405 | 405/405 |
| Digits (raw) | 405/405 | 405/405 | 404/405 |
| Breast Cancer (raw) | 405/405 | 405/405 | 7/405 |
| Iris (raw) | 404/405 | 298/405 | 23/405 |
| Wine (raw) | 404/405 | 230/405 | 8/405 |

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

[Raw measurements and every audit](../benchmarks/results/knn-startup-final2-2026-10-30-broad.json)

[Validated derived summary](../benchmarks/results/knn-startup-final2-2026-10-30-summary.json)

[Controlled source comparison](../benchmarks/results/knn-startup-final2-2026-10-30-control.json)

[Vector figure](assets/knn-startup-benchmarks-2026-10-30.svg)

Production manifest: `benchmarks/reference/knn-startup-final2-2026-10-30.manifest.json`. Snapshot: `{"path": "benchmarks/reference/knn-startup-final2-2026-10-30.zip", "sha256": "56418ec9e026afaff5c586a1de5202270453125421c6ccc05751838ae1f6bc1d"}`. The raw report records source, loaded-worker, harness and fixture hashes before and after timing.

Generated from recorded measurements. No source optimization or performance claim is inferred from a separate historical run.
