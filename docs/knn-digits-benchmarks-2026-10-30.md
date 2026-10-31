# KNN Digits revision: full sklearn benchmark comparison

Completed `2026-10-31T01:26:13.545916+00:00`. This run covers **72 dataset instances in 24 families**, 9 search/inference operations, and **15 retained calls after 5 warmups** per dataset-operation-variant. Fit-only is recorded separately and excluded from the headline 2× counts. Production source and fixture identities are recorded in the raw results.

| sklearn baseline | Medians ≥2× | Individual paired calls ≥2× | All medians ≥2×? | Every pair ≥2×? |
| --- | --- | --- | --- | --- |
| Brute, default threads | 587/648 (90.6%) | 8,753/9,720 (90.1%) | False | False |
| Auto, default threads | 592/648 (91.4%) | 8,831/9,720 (90.9%) | False | False |
| Brute, one thread | 300/648 (46.3%) | 4,912/9,720 (50.5%) | False | False |

Against the fastest median among the three measured sklearn configurations in each dataset-operation cell, **261/648 (40.3%)** reach 2×. This is a best-of-measured-configurations comparison, not one fixed sklearn setting. The panel weights each dataset-operation cell equally; percentages do not estimate real-world workload frequencies. Raw timing counts retain every baseline accuracy failure.

**9/24 families** reach 2× in every seed-operation median against all three baselines; **8/24** also reach 2× in every retained paired call. A family passes only when all its constituent cases and operations pass, without averaging away misses.

![Minimum speedup across seeds](assets/knn-digits-benchmarks-2026-10-30.png)

## Controlled automatic dispatch and GPU comparison

5 serial workers compare current automatic dispatch, an identical automatic A/A worker, current forced GPU and the preceding frozen small-workload GPU implementation. A fifth worker uses the preceding automatic source and its exact archived native binary. The unchanged control covers 48 fixtures and 222 case-operations, with 31 paired rounds after 0.25 seconds of continuous warmup per worker/operation. Default continuous warmup is 0.25 seconds (the preceding small control used 1 second); this is configurable and absolute timings must not be compared across those runs. Each worker/operation warms continuously for at least the requested duration. Its one post-warmup probe is audited and retained separately, excluded from paired ratios. All warmup audits and probe samples remain in the report.

Ratios are comparator/automatic time. The ratio of per-worker medians and median paired ratio are shown separately. A/A exposes process/timing variation; these are descriptive comparisons, not confidence intervals. Each worker retains its actual routing metadata.

Control accuracy and integrity acceptance: **True**. Strict quality: `{"current": true, "current_b": true, "current_gpu": true, "previous_auto": true, "previous_gpu": true}`.

| Family | Operation | Seeds | Auto ms | Current GPU/auto medians | Current GPU/auto paired | Previous GPU/auto medians | Previous GPU/auto paired | Auto A/A paired |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | fit_query | 41, 42, 43 | 0.134–0.145 | 1.922–2.651× | 1.895–2.693× | 1.963–2.954× | 1.912–3.112× | 0.947–1.002× |
| Gaussian 8D, single query | kneighbors | 41, 42, 43 | 0.080–0.087 | 2.228–3.211× | 2.207–3.623× | 2.120–3.137× | 2.075–3.502× | 0.960–1.067× |
| Gaussian 8D, single query | indices | 41, 42, 43 | 0.078–0.085 | 2.968–6.755× | 3.262–6.406× | 2.931–6.329× | 3.213–5.392× | 0.986–1.028× |
| Gaussian 8D, single query | predict_uniform | 41, 42, 43 | 0.086–0.092 | 2.416–3.405× | 2.477–3.395× | 2.278–3.349× | 2.420–3.383× | 0.938–1.021× |
| Gaussian 8D, single query | proba_uniform | 41, 42, 43 | 0.085–0.092 | 2.381–3.488× | 2.426–3.516× | 2.603–4.248× | 2.627–3.928× | 0.973–1.034× |
| Gaussian 8D, single query | regress_uniform | 41, 42, 43 | 0.085–0.097 | 2.279–5.183× | 2.305–5.418× | 2.132–6.033× | 2.117–5.647× | 0.988–0.992× |
| Gaussian 8D, single query | predict_distance | 41, 42, 43 | 0.087–0.101 | 2.140–3.292× | 2.191–4.017× | 2.032–3.275× | 2.067–3.642× | 0.945–1.071× |
| Gaussian 8D, single query | proba_distance | 41, 42, 43 | 0.095–0.098 | 4.588–5.442× | 4.065–5.324× | 3.030–4.668× | 3.240–4.626× | 0.983–1.075× |
| Gaussian 8D, single query | regress_distance | 41, 42, 43 | 0.097–0.099 | 2.210–3.100× | 2.212–3.170× | 2.128–2.725× | 2.151–2.770× | 0.947–1.001× |
| Gaussian 32D, single query | fit_query | 49, 50, 51 | 0.204–0.212 | 1.518–2.168× | 1.524–2.278× | 1.614–2.204× | 1.665–2.308× | 0.978–1.023× |
| Gaussian 32D, single query | kneighbors | 49, 50, 51 | 0.131–0.139 | 2.411–4.316× | 2.432–4.152× | 2.275–4.143× | 2.418–4.024× | 0.972–1.040× |
| Gaussian 32D, single query | indices | 49, 50, 51 | 0.128–0.137 | 1.583–3.826× | 1.545–3.581× | 1.557–3.698× | 1.591–3.637× | 0.982–1.004× |
| Gaussian 32D, single query | predict_uniform | 49, 50, 51 | 0.134–0.140 | 1.621–2.465× | 1.641–2.430× | 1.671–2.454× | 1.757–2.518× | 0.994–1.076× |
| Gaussian 32D, single query | proba_uniform | 49, 50, 51 | 0.135–0.148 | 1.713–2.868× | 1.689–2.616× | 1.586–2.981× | 1.600–2.848× | 0.992–1.012× |
| Gaussian 32D, single query | regress_uniform | 49, 50, 51 | 0.146–0.151 | 1.469–2.405× | 1.473–2.416× | 1.353–2.706× | 1.408–2.819× | 0.955–1.010× |
| Gaussian 32D, single query | predict_distance | 49, 50, 51 | 0.142–0.150 | 2.190–4.561× | 2.170–4.311× | 2.351–3.166× | 2.372–2.983× | 0.979–1.039× |
| Gaussian 32D, single query | proba_distance | 49, 50, 51 | 0.144–0.160 | 1.543–2.548× | 1.539–2.349× | 1.558–2.508× | 1.564–2.608× | 0.947–1.016× |
| Gaussian 32D, single query | regress_distance | 49, 50, 51 | 0.146–0.157 | 1.508–3.989× | 1.520–3.621× | 1.684–2.458× | 1.631–2.255× | 0.957–1.088× |
| Digits (raw) | fit_query | 151, 251, 252 | 0.363–0.473 | 1.512–2.922× | 1.519–2.861× | 2.287–2.744× | 2.299–2.763× | 0.900–1.226× |
| Digits (raw) | kneighbors | 151, 251, 252 | 0.315–0.337 | 1.378–2.757× | 1.455–2.610× | 1.388–2.908× | 1.424–2.638× | 0.858–1.011× |
| Digits (raw) | indices | 151, 251, 252 | 0.293–0.332 | 1.249–1.658× | 1.251–1.761× | 1.194–1.355× | 1.363–1.460× | 0.921–0.965× |
| Digits (raw) | predict_uniform | 151, 251, 252 | 0.291–0.346 | 1.354–1.602× | 1.364–1.737× | 1.316–1.630× | 1.303–1.649× | 0.984–1.267× |
| Digits (raw) | proba_uniform | 151, 251, 252 | 0.251–0.369 | 1.241–3.844× | 1.266–3.567× | 1.178–3.662× | 1.232–3.140× | 0.856–1.345× |
| Digits (raw) | regress_uniform | 151, 251, 252 | 0.260–0.331 | 1.229–3.051× | 1.296–2.965× | 1.245–1.576× | 1.374–1.656× | 0.776–1.290× |
| Digits (raw) | predict_distance | 151, 251, 252 | 0.319–0.373 | 1.202–2.862× | 1.301–2.811× | 1.296–3.072× | 1.506–3.169× | 0.889–0.942× |
| Digits (raw) | proba_distance | 151, 251, 252 | 0.345–0.362 | 1.158–2.245× | 1.175–2.159× | 1.114–2.592× | 1.115–2.592× | 0.926–1.005× |
| Digits (raw) | regress_distance | 151, 251, 252 | 0.289–0.361 | 1.308–3.562× | 1.327–3.033× | 1.203–2.446× | 1.238–2.621× | 0.946–1.233× |
| Breast Cancer (raw) | fit_query | 151, 251, 252 | 0.178–0.217 | 1.452–1.713× | 1.470–1.728× | 1.519–1.838× | 1.435–1.900× | 0.895–1.056× |
| Breast Cancer (raw) | kneighbors | 151, 251, 252 | 0.129–0.148 | 1.529–3.741× | 1.536–3.513× | 1.500–3.204× | 1.570–3.119× | 0.961–1.177× |
| Breast Cancer (raw) | indices | 151, 251, 252 | 0.121–0.139 | 1.460–3.787× | 1.498–3.709× | 1.566–3.760× | 1.567–3.524× | 0.913–1.211× |
| Breast Cancer (raw) | predict_uniform | 151, 251, 252 | 0.140–0.161 | 1.439–3.676× | 1.465–3.555× | 1.604–3.855× | 1.625–3.613× | 0.879–0.935× |
| Breast Cancer (raw) | proba_uniform | 151, 251, 252 | 0.155–0.168 | 1.402–1.563× | 1.363–1.766× | 1.413–1.638× | 1.428–1.839× | 0.905–0.993× |
| Breast Cancer (raw) | regress_uniform | 151, 251, 252 | 0.144–0.156 | 1.369–1.938× | 1.397–1.939× | 1.268–1.924× | 1.300–1.815× | 0.964–0.990× |
| Breast Cancer (raw) | predict_distance | 151, 251, 252 | 0.158–0.163 | 1.327–1.793× | 1.307–1.824× | 1.249–1.777× | 1.280–1.811× | 0.951–1.204× |
| Breast Cancer (raw) | proba_distance | 151, 251, 252 | 0.165–0.167 | 1.255–2.029× | 1.222–1.811× | 1.323–3.088× | 1.361–2.702× | 0.898–1.170× |
| Breast Cancer (raw) | regress_distance | 151, 251, 252 | 0.163–0.170 | 1.388–3.145× | 1.364–2.664× | 1.367–3.032× | 1.409–2.755× | 0.895–0.992× |
| Iris (raw) | fit_query | 151, 251, 252 | 0.090–0.098 | 3.153–5.703× | 3.134–5.193× | 3.502–6.613× | 3.755–5.609× | 0.897–1.019× |
| Iris (raw) | kneighbors | 151, 251, 252 | 0.063–0.074 | 2.950–4.297× | 2.941–4.088× | 2.749–4.177× | 2.837–4.367× | 0.936–1.010× |
| Iris (raw) | indices | 151, 251, 252 | 0.064–0.072 | 3.025–4.468× | 3.109–4.991× | 2.870–4.181× | 3.091–4.476× | 0.997–1.056× |
| Iris (raw) | predict_uniform | 151, 251, 252 | 0.070–0.088 | 3.210–8.442× | 3.117–7.661× | 3.037–8.306× | 3.247–7.446× | 0.940–1.038× |
| Iris (raw) | proba_uniform | 151, 251, 252 | 0.067–0.079 | 3.064–7.103× | 2.969–6.693× | 3.177–6.807× | 3.248–6.023× | 0.948–1.092× |
| Iris (raw) | regress_uniform | 151, 251, 252 | 0.074–0.075 | 2.645–4.030× | 2.599–4.393× | 2.735–3.884× | 2.731–4.469× | 0.965–0.982× |
| Iris (raw) | predict_distance | 151, 251, 252 | 0.085–0.094 | 2.408–3.802× | 2.468–3.944× | 2.535–3.418× | 2.495–3.566× | 0.975–1.036× |
| Iris (raw) | proba_distance | 151, 251, 252 | 0.079–0.097 | 2.365–3.736× | 2.455–3.878× | 2.336–3.335× | 2.341–3.788× | 0.970–1.110× |
| Iris (raw) | regress_distance | 151, 251, 252 | 0.082–0.101 | 2.689–4.143× | 2.404–4.039× | 2.384–3.909× | 2.337–3.878× | 0.889–1.126× |
| Wine (raw) | fit_query | 151, 251, 252 | 0.101–0.108 | 2.527–3.026× | 2.535–3.182× | 2.786–3.539× | 2.755–3.507× | 0.940–1.016× |
| Wine (raw) | kneighbors | 151, 251, 252 | 0.075–0.080 | 2.380–6.455× | 2.591–6.365× | 2.559–6.467× | 2.632–5.872× | 0.919–1.128× |
| Wine (raw) | indices | 151, 251, 252 | 0.071–0.076 | 2.437–7.043× | 2.430–6.663× | 2.545–6.918× | 2.503–6.418× | 0.943–1.016× |
| Wine (raw) | predict_uniform | 151, 251, 252 | 0.078–0.114 | 2.393–2.956× | 2.473–2.937× | 2.228–2.950× | 2.320–3.150× | 0.965–1.051× |
| Wine (raw) | proba_uniform | 151, 251, 252 | 0.082–0.090 | 2.655–3.286× | 2.737–3.348× | 2.531–3.188× | 2.637–3.310× | 0.975–1.058× |
| Wine (raw) | regress_uniform | 151, 251, 252 | 0.077–0.086 | 2.286–6.107× | 2.319–6.078× | 2.178–6.418× | 2.204–5.939× | 0.942–1.089× |
| Wine (raw) | predict_distance | 151, 251, 252 | 0.086–0.091 | 2.362–3.228× | 2.324–3.462× | 2.440–5.205× | 2.432–5.036× | 0.960–1.104× |
| Wine (raw) | proba_distance | 151, 251, 252 | 0.093–0.108 | 2.314–2.730× | 2.345–3.082× | 2.436–2.698× | 2.526–2.814× | 0.934–0.965× |
| Wine (raw) | regress_distance | 151, 251, 252 | 0.093–0.097 | 2.413–2.879× | 2.390–2.871× | 2.398–3.235× | 2.406–3.031× | 0.992–1.012× |
| Gaussian 50k × 32 | kneighbors | 151, 251, 252, 351, 451, 552 | 0.428–2.396 | 0.919–0.998× | 0.966–1.009× | 0.988–1.006× | 0.968–1.003× | 0.988–1.005× |
| Gaussian 50k × 32 | fit_query | 151, 251, 252, 351, 451, 552 | 0.750–2.424 | 0.967–1.015× | 0.958–1.012× | 1.024–1.071× | 1.005–1.062× | 0.977–1.020× |
| Clusters 50k × 32 | kneighbors | 151, 251, 252 | 0.748–0.796 | 0.982–1.072× | 0.983–1.069× | 0.929–1.000× | 0.981–0.991× | 1.000–1.026× |
| Clusters 50k × 32 | fit_query | 151, 251, 252 | 1.059–2.169 | 0.890–0.999× | 0.981–1.013× | 1.015–1.067× | 1.020–1.056× | 0.974–1.021× |
| Offset +1e4, 50k × 32 | kneighbors | 151, 251, 252 | 0.449–0.782 | 0.951–1.011× | 0.937–0.999× | 0.938–1.010× | 0.970–0.991× | 0.947–0.982× |
| Offset +1e4, 50k × 32 | fit_query | 151, 251, 252 | 0.819–0.860 | 0.960–1.007× | 0.969–1.000× | 1.017–1.040× | 1.027–1.040× | 0.992–1.000× |
| Unequal scales 50k × 32 | kneighbors | 151, 251, 252 | 0.501–0.515 | 0.983–1.020× | 0.987–1.008× | 0.984–1.021× | 0.971–1.003× | 0.989–1.016× |
| Unequal scales 50k × 32 | fit_query | 151, 251, 252 | 0.817–0.831 | 0.965–1.018× | 0.959–1.016× | 1.029–1.056× | 1.044–1.068× | 0.975–1.012× |
| Gaussian 4096 × 128 | kneighbors | 151, 251, 252, 351, 451, 552 | 0.385–0.414 | 0.926–1.038× | 0.917–1.039× | 0.935–1.015× | 0.968–1.027× | 0.955–1.051× |
| Gaussian 4096 × 128 | fit_query | 151, 251, 252, 351, 451, 552 | 0.577–0.713 | 0.939–1.080× | 0.956–1.040× | 1.038–1.101× | 1.012–1.077× | 0.973–1.047× |
| Clusters 4096 × 128 | kneighbors | 151, 251, 252 | 0.378–0.406 | 0.981–1.012× | 0.986–1.021× | 0.956–1.006× | 0.963–1.007× | 0.995–1.014× |
| Clusters 4096 × 128 | fit_query | 151, 251, 252 | 0.497–0.521 | 0.980–1.026× | 0.970–1.036× | 1.022–1.096× | 1.008–1.100× | 0.979–1.037× |
| Offset +1e4, 4096 × 128 | kneighbors | 151, 251, 252 | 0.381–0.421 | 0.934–1.007× | 0.966–1.023× | 0.969–1.015× | 0.994–1.015× | 0.976–0.994× |
| Offset +1e4, 4096 × 128 | fit_query | 151, 251, 252 | 0.586–0.604 | 0.975–1.210× | 0.965–1.059× | 1.025–1.160× | 1.025–1.099× | 0.962–1.003× |
| Unequal scales 4096 × 128 | kneighbors | 151, 251, 252 | 0.376–0.428 | 0.932–1.036× | 0.991–1.034× | 0.923–1.031× | 0.981–1.016× | 0.941–1.033× |
| Unequal scales 4096 × 128 | fit_query | 151, 251, 252 | 0.586–0.656 | 0.953–1.036× | 0.915–1.043× | 1.078–1.126× | 1.067–1.129× | 0.952–1.030× |

### Comparison with preceding automatic dispatch

| Family | Operation | Seeds | Current auto ms | Previous auto ms | Previous/current medians | Previous/current paired |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | fit_query | 41, 42, 43 | 0.134–0.145 | 0.134–0.136 | 0.929–1.008× | 0.946–1.020× |
| Gaussian 8D, single query | kneighbors | 41, 42, 43 | 0.080–0.087 | 0.081–0.085 | 0.982–1.011× | 0.979–1.033× |
| Gaussian 8D, single query | indices | 41, 42, 43 | 0.078–0.085 | 0.078–0.089 | 0.947–1.047× | 0.950–1.074× |
| Gaussian 8D, single query | predict_uniform | 41, 42, 43 | 0.086–0.092 | 0.087–0.090 | 0.971–1.021× | 0.967–1.012× |
| Gaussian 8D, single query | proba_uniform | 41, 42, 43 | 0.085–0.092 | 0.081–0.094 | 0.915–1.019× | 0.951–1.019× |
| Gaussian 8D, single query | regress_uniform | 41, 42, 43 | 0.085–0.097 | 0.086–0.091 | 0.937–1.002× | 0.880–1.003× |
| Gaussian 8D, single query | predict_distance | 41, 42, 43 | 0.087–0.101 | 0.097–0.100 | 0.986–1.135× | 0.967–1.090× |
| Gaussian 8D, single query | proba_distance | 41, 42, 43 | 0.095–0.098 | 0.094–0.100 | 0.965–1.025× | 0.980–1.041× |
| Gaussian 8D, single query | regress_distance | 41, 42, 43 | 0.097–0.099 | 0.098–0.101 | 1.000–1.020× | 0.972–1.050× |
| Gaussian 32D, single query | fit_query | 49, 50, 51 | 0.204–0.212 | 0.207–0.217 | 0.982–1.034× | 0.999–1.038× |
| Gaussian 32D, single query | kneighbors | 49, 50, 51 | 0.131–0.139 | 0.131–0.141 | 0.986–1.039× | 0.976–1.052× |
| Gaussian 32D, single query | indices | 49, 50, 51 | 0.128–0.137 | 0.129–0.133 | 0.943–1.018× | 0.932–1.004× |
| Gaussian 32D, single query | predict_uniform | 49, 50, 51 | 0.134–0.140 | 0.125–0.140 | 0.922–1.003× | 0.938–1.025× |
| Gaussian 32D, single query | proba_uniform | 49, 50, 51 | 0.135–0.148 | 0.134–0.146 | 0.947–1.032× | 0.928–1.020× |
| Gaussian 32D, single query | regress_uniform | 49, 50, 51 | 0.146–0.151 | 0.141–0.153 | 0.933–1.046× | 0.949–1.030× |
| Gaussian 32D, single query | predict_distance | 49, 50, 51 | 0.142–0.150 | 0.144–0.151 | 0.959–1.042× | 0.954–1.031× |
| Gaussian 32D, single query | proba_distance | 49, 50, 51 | 0.144–0.160 | 0.142–0.158 | 0.957–1.001× | 0.928–1.000× |
| Gaussian 32D, single query | regress_distance | 49, 50, 51 | 0.146–0.157 | 0.145–0.162 | 0.985–1.057× | 0.955–1.041× |
| Digits (raw) | fit_query | 151, 251, 252 | 0.363–0.473 | 0.464–0.527 | 1.085–1.278× | 1.100–1.276× |
| Digits (raw) | kneighbors | 151, 251, 252 | 0.315–0.337 | 0.301–0.388 | 0.955–1.151× | 0.963–1.170× |
| Digits (raw) | indices | 151, 251, 252 | 0.293–0.332 | 0.289–0.361 | 0.880–1.230× | 0.864–1.276× |
| Digits (raw) | predict_uniform | 151, 251, 252 | 0.291–0.346 | 0.295–0.378 | 0.891–1.299× | 0.897–1.307× |
| Digits (raw) | proba_uniform | 151, 251, 252 | 0.251–0.369 | 0.367–0.382 | 1.035–1.461× | 1.060–1.396× |
| Digits (raw) | regress_uniform | 151, 251, 252 | 0.260–0.331 | 0.353–0.382 | 1.121–1.361× | 1.073–1.347× |
| Digits (raw) | predict_distance | 151, 251, 252 | 0.319–0.373 | 0.310–0.391 | 0.830–1.225× | 0.852–1.244× |
| Digits (raw) | proba_distance | 151, 251, 252 | 0.345–0.362 | 0.345–0.394 | 0.955–1.142× | 0.944–1.064× |
| Digits (raw) | regress_distance | 151, 251, 252 | 0.289–0.361 | 0.312–0.362 | 0.867–1.251× | 0.876–1.298× |
| Breast Cancer (raw) | fit_query | 151, 251, 252 | 0.178–0.217 | 0.192–0.219 | 0.935–1.078× | 0.955–1.054× |
| Breast Cancer (raw) | kneighbors | 151, 251, 252 | 0.129–0.148 | 0.136–0.145 | 0.978–1.053× | 0.968–1.096× |
| Breast Cancer (raw) | indices | 151, 251, 252 | 0.121–0.139 | 0.133–0.142 | 1.021–1.099× | 1.079–1.134× |
| Breast Cancer (raw) | predict_uniform | 151, 251, 252 | 0.140–0.161 | 0.140–0.154 | 0.873–1.033× | 0.886–1.054× |
| Breast Cancer (raw) | proba_uniform | 151, 251, 252 | 0.155–0.168 | 0.135–0.159 | 0.856–0.948× | 0.841–0.999× |
| Breast Cancer (raw) | regress_uniform | 151, 251, 252 | 0.144–0.156 | 0.138–0.158 | 0.901–1.102× | 0.913–1.073× |
| Breast Cancer (raw) | predict_distance | 151, 251, 252 | 0.158–0.163 | 0.152–0.171 | 0.931–1.088× | 0.906–1.091× |
| Breast Cancer (raw) | proba_distance | 151, 251, 252 | 0.165–0.167 | 0.163–0.184 | 0.980–1.110× | 0.953–1.083× |
| Breast Cancer (raw) | regress_distance | 151, 251, 252 | 0.163–0.170 | 0.146–0.165 | 0.896–1.011× | 0.865–0.961× |
| Iris (raw) | fit_query | 151, 251, 252 | 0.090–0.098 | 0.085–0.100 | 0.900–1.113× | 0.891–1.066× |
| Iris (raw) | kneighbors | 151, 251, 252 | 0.063–0.074 | 0.059–0.070 | 0.920–1.065× | 0.890–1.066× |
| Iris (raw) | indices | 151, 251, 252 | 0.064–0.072 | 0.062–0.068 | 0.855–1.067× | 0.904–1.098× |
| Iris (raw) | predict_uniform | 151, 251, 252 | 0.070–0.088 | 0.076–0.094 | 1.025–1.100× | 0.953–1.083× |
| Iris (raw) | proba_uniform | 151, 251, 252 | 0.067–0.079 | 0.075–0.075 | 0.953–1.120× | 0.971–1.104× |
| Iris (raw) | regress_uniform | 151, 251, 252 | 0.074–0.075 | 0.073–0.077 | 0.973–1.047× | 0.901–1.046× |
| Iris (raw) | predict_distance | 151, 251, 252 | 0.085–0.094 | 0.083–0.088 | 0.934–1.004× | 0.937–1.045× |
| Iris (raw) | proba_distance | 151, 251, 252 | 0.079–0.097 | 0.074–0.087 | 0.889–0.944× | 0.912–1.017× |
| Iris (raw) | regress_distance | 151, 251, 252 | 0.082–0.101 | 0.083–0.093 | 0.921–1.051× | 0.928–1.054× |
| Wine (raw) | fit_query | 151, 251, 252 | 0.101–0.108 | 0.100–0.108 | 0.927–1.062× | 0.922–1.080× |
| Wine (raw) | kneighbors | 151, 251, 252 | 0.075–0.080 | 0.079–0.081 | 0.991–1.082× | 0.996–1.068× |
| Wine (raw) | indices | 151, 251, 252 | 0.071–0.076 | 0.067–0.075 | 0.879–1.049× | 0.912–1.022× |
| Wine (raw) | predict_uniform | 151, 251, 252 | 0.078–0.114 | 0.083–0.104 | 0.913–1.062× | 0.916–1.047× |
| Wine (raw) | proba_uniform | 151, 251, 252 | 0.082–0.090 | 0.079–0.095 | 0.896–1.060× | 0.894–1.137× |
| Wine (raw) | regress_uniform | 151, 251, 252 | 0.077–0.086 | 0.082–0.083 | 0.971–1.069× | 0.947–1.081× |
| Wine (raw) | predict_distance | 151, 251, 252 | 0.086–0.091 | 0.089–0.096 | 1.037–1.061× | 1.000–1.083× |
| Wine (raw) | proba_distance | 151, 251, 252 | 0.093–0.108 | 0.091–0.110 | 0.892–1.011× | 0.843–1.025× |
| Wine (raw) | regress_distance | 151, 251, 252 | 0.093–0.097 | 0.086–0.095 | 0.906–1.018× | 0.879–0.977× |
| Gaussian 50k × 32 | kneighbors | 151, 251, 252, 351, 451, 552 | 0.428–2.396 | 0.425–2.386 | 0.982–1.021× | 0.982–1.018× |
| Gaussian 50k × 32 | fit_query | 151, 251, 252, 351, 451, 552 | 0.750–2.424 | 0.749–2.420 | 0.983–1.013× | 0.966–1.026× |
| Clusters 50k × 32 | kneighbors | 151, 251, 252 | 0.748–0.796 | 0.750–0.797 | 0.942–1.065× | 0.963–0.999× |
| Clusters 50k × 32 | fit_query | 151, 251, 252 | 1.059–2.169 | 1.082–2.125 | 0.980–1.021× | 0.984–1.023× |
| Offset +1e4, 50k × 32 | kneighbors | 151, 251, 252 | 0.449–0.782 | 0.457–0.792 | 1.012–1.024× | 0.976–1.007× |
| Offset +1e4, 50k × 32 | fit_query | 151, 251, 252 | 0.819–0.860 | 0.823–0.863 | 0.975–1.005× | 0.964–1.001× |
| Unequal scales 50k × 32 | kneighbors | 151, 251, 252 | 0.501–0.515 | 0.506–0.517 | 0.993–1.031× | 0.988–1.018× |
| Unequal scales 50k × 32 | fit_query | 151, 251, 252 | 0.817–0.831 | 0.820–0.844 | 0.987–1.033× | 0.982–1.030× |
| Gaussian 4096 × 128 | kneighbors | 151, 251, 252, 351, 451, 552 | 0.385–0.414 | 0.380–0.405 | 0.940–1.047× | 0.972–1.035× |
| Gaussian 4096 × 128 | fit_query | 151, 251, 252, 351, 451, 552 | 0.577–0.713 | 0.563–0.696 | 0.953–1.109× | 0.955–1.035× |
| Clusters 4096 × 128 | kneighbors | 151, 251, 252 | 0.378–0.406 | 0.380–0.406 | 0.996–1.019× | 0.970–1.031× |
| Clusters 4096 × 128 | fit_query | 151, 251, 252 | 0.497–0.521 | 0.511–0.524 | 0.981–1.038× | 0.973–1.042× |
| Offset +1e4, 4096 × 128 | kneighbors | 151, 251, 252 | 0.381–0.421 | 0.378–0.440 | 0.992–1.045× | 0.946–1.031× |
| Offset +1e4, 4096 × 128 | fit_query | 151, 251, 252 | 0.586–0.604 | 0.601–0.651 | 1.006–1.111× | 0.998–1.007× |
| Unequal scales 4096 × 128 | kneighbors | 151, 251, 252 | 0.376–0.428 | 0.381–0.427 | 0.963–1.071× | 0.984–1.052× |
| Unequal scales 4096 × 128 | fit_query | 151, 251, 252 | 0.586–0.656 | 0.611–0.669 | 1.019–1.070× | 1.014–1.067× |

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
| Gaussian 8D, single query | 8,192 × 8; 1; 5 | 0.102–0.105 | 6.85–9.53× | 1.88–1.95× | 2.36–2.37× | 27 / 6 / 25 (each /27) |
| Gaussian 8D, k5 | 8,192 × 8; 256; 5 | 0.719–0.784 | 1.87–2.56× | 5.49–6.43× | 5.54–5.94× | 9 / 27 / 27 (each /27) |
| Gaussian 32D, k5 | 8,192 × 32; 256; 5 | 0.958–1.004 | 3.61–4.54× | 3.61–4.47× | 4.50–4.65× | 27 / 27 / 27 (each /27) |
| Gaussian 32D, k32 | 8,192 × 32; 128; 32 | 1.867–1.913 | 2.04–2.07× | 2.01–2.02× | 1.59–1.65× | 11 / 10 / 0 (each /27) |
| Gaussian 128D, k5 | 4,096 × 128; 128; 5 | 0.883–0.919 | 3.49–3.72× | 3.53–3.65× | 1.88–1.95× | 27 / 27 / 0 (each /27) |
| Gaussian 1024D, k5 | 2,048 × 1024; 32; 5 | 1.149–1.284 | 2.80–3.28× | 2.73–3.20× | 1.74–1.98× | 27 / 27 / 2 (each /27) |
| Gaussian 32D, k64 | 4,096 × 32; 128; 64 | 1.493–1.525 | 2.58–2.67× | 2.55–2.65× | 1.69–1.76× | 27 / 27 / 0 (each /27) |
| Gaussian 16D, k5 | 8,192 × 16; 256; 5 | 0.795–0.866 | 4.22–4.67× | 4.15–4.51× | 4.99–5.40× | 27 / 27 / 27 (each /27) |
| Gaussian 32D, single query | 8,192 × 32; 1; 5 | 0.163–0.178 | 14.98–17.56× | 15.67–16.67× | 1.53–1.61× | 27 / 27 / 0 (each /27) |
| Gaussian 8D, k32 | 8,192 × 8; 128; 32 | 1.817–1.911 | 0.79–1.08× | 2.19–2.30× | 1.57–1.63× | 0 / 27 / 0 (each /27) |
| Gaussian 50k × 32 (original panel) | 50,000 × 32; 256; 5 | 1.468–1.512 | 4.34–4.41× | 4.31–4.58× | 17.05–17.50× | 27 / 27 / 27 (each /27) |
| Gaussian 50k × 32 | 50,000 × 32; 256; 5 | 1.382–1.495 | 4.35–4.94× | 4.61–4.77× | 17.18–18.62× | 27 / 27 / 27 (each /27) |
| Clusters 50k × 32 | 50,000 × 32; 256; 5 | 2.595–2.731 | 2.34–2.61× | 2.37–2.59× | 9.42–9.92× | 27 / 27 / 27 (each /27) |
| Offset +1e4, 50k × 32 | 50,000 × 32; 256; 5 | 1.462–1.583 | 4.31–4.73× | 4.23–4.55× | 16.26–17.57× | 27 / 27 / 27 (each /27) |
| Unequal scales 50k × 32 | 50,000 × 32; 256; 5 | 1.444–1.849 | 3.56–4.86× | 3.59–4.51× | 13.91–17.81× | 27 / 27 / 27 (each /27) |
| Gaussian 4096 × 128 | 4,096 × 128; 128; 5 | 0.872–0.916 | 3.69–5.33× | 3.71–5.32× | 1.86–1.99× | 27 / 27 / 1 (each /27) |
| Clusters 4096 × 128 | 4,096 × 128; 128; 5 | 1.166–1.197 | 2.65–2.78× | 2.69–2.78× | 1.44–1.48× | 27 / 27 / 0 (each /27) |
| Offset +1e4, 4096 × 128 | 4,096 × 128; 128; 5 | 0.857–0.955 | 3.43–3.80× | 3.41–3.83× | 1.83–1.99× | 27 / 27 / 1 (each /27) |
| Unequal scales 4096 × 128 | 4,096 × 128; 128; 5 | 0.865–0.926 | 3.50–3.71× | 3.50–3.70× | 1.86–1.98× | 27 / 27 / 1 (each /27) |
| Gaussian 100k × 32 | 100,000 × 32; 256; 5 | 2.009–2.111 | 4.61–5.40× | 4.58–5.09× | 24.18–25.40× | 27 / 27 / 27 (each /27) |
| Digits (raw) | 1,541 × 64; 256; 5 | 0.374–0.399 | 7.46–8.24× | 7.23–8.16× | 2.84–3.04× | 27 / 27 / 27 (each /27) |
| Breast Cancer (raw) | 455 × 30; 114; 5 | 0.181–0.183 | 11.62–11.83× | 11.65–12.04× | 1.48–1.58× | 27 / 27 / 0 (each /27) |
| Iris (raw) | 120 × 4; 30; 5 | 0.085–0.088 | 6.55–7.57× | 2.19–2.39× | 1.46–1.56× | 27 / 21 / 0 (each /27) |
| Wine (raw) | 142 × 13; 36; 5 | 0.096–0.106 | 5.82–7.14× | 1.96–2.06× | 1.35–1.42× | 27 / 15 / 0 (each /27) |

## Fresh fit + query

| Family | Current fresh ms | vs default brute | vs auto | vs one thread |
| --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 0.164–0.176 | 4.57–5.13× | 8.65–9.73× | 1.88–2.04× |
| Gaussian 8D, k5 | 0.782–0.843 | 1.77–1.86× | 7.18–7.27× | 5.21–5.62× |
| Gaussian 32D, k5 | 1.068–1.137 | 3.43–4.27× | 3.35–4.18× | 4.03–4.27× |
| Gaussian 32D, k32 | 2.100–2.156 | 1.83–1.88× | 1.84–1.89× | 1.49–1.51× |
| Gaussian 128D, k5 | 1.121–1.257 | 2.68–2.95× | 2.69–3.07× | 1.47–1.66× |
| Gaussian 1024D, k5 | 1.501–1.882 | 2.10–2.64× | 2.06–2.64× | 1.38–1.73× |
| Gaussian 32D, k64 | 1.632–1.686 | 2.48–2.50× | 2.39–2.48× | 1.58–1.68× |
| Gaussian 16D, k5 | 0.923–0.929 | 3.86–4.14× | 3.91–4.08× | 4.76–4.77× |
| Gaussian 32D, single query | 0.236–0.249 | 11.17–12.34× | 11.37–11.63× | 1.49–1.51× |
| Gaussian 8D, k32 | 1.965–2.095 | 0.85–0.89× | 2.75–2.86× | 1.50–1.57× |
| Gaussian 50k × 32 (original panel) | 1.810–1.920 | 3.57–3.88× | 3.67–4.00× | 13.59–14.42× |
| Gaussian 50k × 32 | 1.801–1.902 | 3.72–3.73× | 3.57–3.96× | 13.68–14.45× |
| Clusters 50k × 32 | 3.047–3.101 | 2.24–2.38× | 2.24–2.35× | 8.42–8.56× |
| Offset +1e4, 50k × 32 | 1.845–2.076 | 3.46–3.92× | 3.54–3.91× | 12.51–14.12× |
| Unequal scales 50k × 32 | 2.134–2.260 | 3.05–3.50× | 3.09–3.37× | 11.56–12.20× |
| Gaussian 4096 × 128 | 1.172–1.272 | 2.67–3.39× | 2.72–3.32× | 1.48–1.59× |
| Clusters 4096 × 128 | 1.310–1.353 | 2.48–2.55× | 2.51–2.62× | 1.36–1.45× |
| Offset +1e4, 4096 × 128 | 1.114–1.203 | 2.81–3.04× | 2.83–3.00× | 1.57–1.67× |
| Unequal scales 4096 × 128 | 1.181–1.267 | 2.71–2.84× | 2.63–2.92× | 1.47–1.58× |
| Gaussian 100k × 32 | 2.719–2.784 | 3.56–3.88× | 3.62–4.02× | 18.50–19.02× |
| Digits (raw) | 0.499–0.529 | 5.68–6.26× | 5.85–6.83× | 2.37–2.45× |
| Breast Cancer (raw) | 0.237–0.257 | 8.53–9.28× | 8.82–9.17× | 1.40–1.47× |
| Iris (raw) | 0.108–0.122 | 5.51–6.27× | 2.55–2.75× | 1.64–1.81× |
| Wine (raw) | 0.124–0.128 | 5.23–5.67× | 2.33–2.48× | 1.63–1.66× |

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
| `fit` (excluded from headline) | 0/72; 25/1,080 | 15/72; 214/1,080 | 0/72; 24/1,080 |
| `fit_query` | 63/72; 950/1,080 | 69/72; 1029/1,080 | 31/72; 476/1,080 |
| `kneighbors` | 68/72; 1000/1,080 | 68/72; 996/1,080 | 33/72; 552/1,080 |
| `indices` | 63/72; 949/1,080 | 66/72; 962/1,080 | 34/72; 545/1,080 |
| `predict_uniform` | 64/72; 960/1,080 | 69/72; 1032/1,080 | 34/72; 553/1,080 |
| `proba_uniform` | 64/72; 945/1,080 | 65/72; 958/1,080 | 34/72; 558/1,080 |
| `regress_uniform` | 64/72; 956/1,080 | 62/72; 942/1,080 | 34/72; 563/1,080 |
| `predict_distance` | 68/72; 1008/1,080 | 67/72; 1001/1,080 | 33/72; 566/1,080 |
| `proba_distance` | 66/72; 993/1,080 | 64/72; 970/1,080 | 34/72; 554/1,080 |
| `regress_distance` | 67/72; 992/1,080 | 62/72; 941/1,080 | 33/72; 545/1,080 |

Family-level paired-call counts retain every individual threshold miss:

| Family | Brute, default threads | Auto, default threads | Brute, one thread |
| --- | --- | --- | --- |
| Gaussian 8D, single query | 405/405 | 144/405 | 360/405 |
| Gaussian 8D, k5 | 162/405 | 405/405 | 405/405 |
| Gaussian 32D, k5 | 405/405 | 405/405 | 405/405 |
| Gaussian 32D, k32 | 145/405 | 136/405 | 1/405 |
| Gaussian 128D, k5 | 402/405 | 402/405 | 73/405 |
| Gaussian 1024D, k5 | 381/405 | 382/405 | 83/405 |
| Gaussian 32D, k64 | 395/405 | 395/405 | 24/405 |
| Gaussian 16D, k5 | 405/405 | 405/405 | 405/405 |
| Gaussian 32D, single query | 405/405 | 405/405 | 0/405 |
| Gaussian 8D, k32 | 1/405 | 392/405 | 3/405 |
| Gaussian 50k × 32 (original panel) | 405/405 | 405/405 | 405/405 |
| Gaussian 50k × 32 | 405/405 | 405/405 | 405/405 |
| Clusters 50k × 32 | 395/405 | 395/405 | 405/405 |
| Offset +1e4, 50k × 32 | 405/405 | 405/405 | 405/405 |
| Unequal scales 50k × 32 | 405/405 | 405/405 | 405/405 |
| Gaussian 4096 × 128 | 404/405 | 404/405 | 87/405 |
| Clusters 4096 × 128 | 399/405 | 397/405 | 12/405 |
| Offset +1e4, 4096 × 128 | 404/405 | 405/405 | 85/405 |
| Unequal scales 4096 × 128 | 405/405 | 404/405 | 90/405 |
| Gaussian 100k × 32 | 405/405 | 405/405 | 405/405 |
| Digits (raw) | 405/405 | 405/405 | 405/405 |
| Breast Cancer (raw) | 405/405 | 405/405 | 6/405 |
| Iris (raw) | 404/405 | 295/405 | 27/405 |
| Wine (raw) | 401/405 | 225/405 | 11/405 |

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

[Raw measurements and every audit](../benchmarks/results/knn-digits-final-2026-10-30-broad.json)

[Validated derived summary](../benchmarks/results/knn-digits-final-2026-10-30-broad-summary.json)

[Controlled source comparison](../benchmarks/results/knn-digits-final-2026-10-30-control.json)

[Vector figure](assets/knn-digits-benchmarks-2026-10-30.svg)

Production manifest: `benchmarks/reference/knn-digits-final-2026-10-30.manifest.json`. Snapshot: `{"path": "benchmarks/reference/knn-digits-final-2026-10-30.zip", "sha256": "5a4b3c5c130c043640b8755b3e26aaf53911729cd5e56acc37699e681da268db"}`. The raw report records source, loaded-worker, harness and fixture hashes before and after timing.

Generated from recorded measurements. No source optimization or performance claim is inferred from a separate historical run.
