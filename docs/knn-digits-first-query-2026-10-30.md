# Fresh-model first queries in initialized processes

Initialized isolated persistent processes; normal NumPy/estimator imports and configuration are recorded separately. No model or query prewarm. Every request constructs and fits a new estimator, then times its first query and separately its second query. Public NumPy results establish completion; no harness MLX import/device probe/synchronization runs before the first query. Both outputs are audited after both timers. All requested rounds (default 30), including first model per fixture and first model in worker, remain visible. Earlier models may initialize runtime/compiler caches. This is model-first-use in an initialized process, not fresh-process or machine-cold latency. Five-worker Williams order has a complete balanced 10-round block, repeated three times by default. Any incomplete final block is also retained. No classification/regression operations are added here; the separate full broad and preservation control panels retain those operations and fixtures.

First query includes model preparation deferred by fit. Second query reuses only that fresh model. All rounds remain in the headline median; first-per-fixture and subsequent-model views are shown separately, not selected according to speed. Process initialization/import cost is separate, and must not be confused with the dedicated fresh-process cold benchmark.

Acceptance: **True**. 18 fixtures × 30 fresh models × 5 workers; each model has exactly one retained first and one retained second query. No numerical audit failure removes a valid timing. Ratios are comparator/current; bold values below 1× are slower observations. These descriptive ratios are not confidence intervals.

## Initial process setup (once per worker)

| Worker | NumPy import ms | Estimator import/setup ms | Worker entry→initialized ms | Parent process→initialized ms |
| --- | --- | --- | --- | --- |
| Current automatic | 24.844 | 638.446 | 665.200 | 689.292 |
| Default-thread brute | 24.894 | 773.279 | 800.062 | 823.645 |
| Single-thread brute | 25.018 | 785.869 | 812.827 | 836.741 |
| sklearn auto | 25.096 | 783.357 | 810.437 | 834.378 |
| Preceding automatic | 24.828 | 908.810 | 935.576 | 959.071 |

## fit_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.074 | 0.108 / 1.459× / 1.434× | 0.103 / 1.392× / 1.303× | 1.477 / 19.987× / 19.571× | 0.077 / 1.038× / **0.976×** |
| Gaussian 32D, single query | 49 | 0.113 | 0.153 / 1.349× / 1.389× | 0.154 / 1.358× / 1.349× | 0.164 / 1.441× / 1.398× | 0.116 / 1.026× / **0.973×** |
| Gaussian 8D, single query | 42 | 0.080 | 0.111 / 1.384× / 1.299× | 0.109 / 1.352× / 1.278× | 1.549 / 19.285× / 19.059× | 0.079 / **0.978×** / **0.994×** |
| Gaussian 32D, single query | 50 | 0.094 | 0.132 / 1.400× / 1.354× | 0.133 / 1.412× / 1.417× | 0.129 / 1.375× / 1.387× | 0.093 / **0.993×** / 1.009× |
| Gaussian 8D, single query | 43 | 0.086 | 0.115 / 1.326× / 1.277× | 0.116 / 1.345× / 1.363× | 1.553 / 17.953× / 17.950× | 0.089 / 1.029× / 1.018× |
| Gaussian 32D, single query | 51 | 0.084 | 0.122 / 1.458× / 1.436× | 0.123 / 1.465× / 1.430× | 0.120 / 1.430× / 1.466× | 0.085 / 1.012× / 1.048× |
| Digits (raw) | 151 | 0.069 | 0.144 / 2.072× / 2.069× | 0.144 / 2.073× / 2.075× | 0.143 / 2.063× / 2.105× | 0.063 / **0.905×** / **0.945×** |
| Digits (raw) | 251 | 0.083 | 0.158 / 1.913× / 1.938× | 0.165 / 2.005× / 2.076× | 0.148 / 1.794× / 1.910× | 0.081 / **0.985×** / 1.015× |
| Digits (raw) | 252 | 0.085 | 0.162 / 1.903× / 1.883× | 0.178 / 2.087× / 2.152× | 0.171 / 2.016× / 1.904× | 0.088 / 1.032× / 1.023× |
| Breast Cancer (raw) | 151 | 0.069 | 0.144 / 2.079× / 1.979× | 0.150 / 2.166× / 2.195× | 0.141 / 2.037× / 2.112× | 0.069 / **0.996×** / 1.010× |
| Breast Cancer (raw) | 251 | 0.051 | 0.113 / 2.213× / 2.314× | 0.119 / 2.317× / 2.209× | 0.110 / 2.142× / 2.280× | 0.046 / **0.893×** / **0.972×** |
| Breast Cancer (raw) | 252 | 0.044 | 0.107 / 2.448× / 2.461× | 0.112 / 2.572× / 2.399× | 0.099 / 2.279× / 2.308× | 0.045 / 1.031× / **0.997×** |
| Iris (raw) | 151 | 0.045 | 0.102 / 2.264× / 2.244× | 0.112 / 2.483× / 2.443× | 0.144 / 3.192× / 3.125× | 0.042 / **0.927×** / **0.930×** |
| Iris (raw) | 251 | 0.035 | 0.094 / 2.677× / 2.486× | 0.096 / 2.733× / 2.621× | 0.129 / 3.665× / 3.537× | 0.038 / 1.095× / **0.966×** |
| Iris (raw) | 252 | 0.042 | 0.108 / 2.566× / 2.598× | 0.104 / 2.457× / 2.512× | 0.148 / 3.491× / 3.340× | 0.042 / **0.994×** / 1.027× |
| Wine (raw) | 151 | 0.045 | 0.105 / 2.327× / 2.390× | 0.107 / 2.378× / 2.478× | 0.152 / 3.370× / 3.279× | 0.041 / **0.914×** / **0.954×** |
| Wine (raw) | 251 | 0.034 | 0.096 / 2.806× / 2.691× | 0.092 / 2.691× / 2.584× | 0.127 / 3.695× / 3.880× | 0.036 / 1.037× / 1.023× |
| Wine (raw) | 252 | 0.032 | 0.090 / 2.820× / 2.704× | 0.084 / 2.657× / 2.587× | 0.122 / 3.834× / 3.839× | 0.033 / 1.053× / 1.018× |

## first_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.063 | 0.719 / 11.459× / 10.922× | 0.204 / 3.254× / 3.216× | 0.169 / 2.693× / 2.616× | 0.062 / **0.994×** / 1.008× |
| Gaussian 32D, single query | 49 | 0.125 | 3.080 / 24.683× / 24.581× | 0.232 / 1.860× / 1.857× | 3.107 / 24.899× / 25.096× | 0.126 / 1.007× / **0.998×** |
| Gaussian 8D, single query | 42 | 0.064 | 2.662 / 41.293× / 41.766× | 0.212 / 3.282× / 3.274× | 0.188 / 2.922× / 2.952× | 0.065 / 1.002× / **0.998×** |
| Gaussian 32D, single query | 50 | 0.120 | 2.648 / 22.158× / 22.347× | 0.221 / 1.846× / 1.877× | 2.634 / 22.036× / 22.356× | 0.119 / **0.994×** / 1.005× |
| Gaussian 8D, single query | 43 | 0.068 | 2.637 / 38.941× / 37.335× | 0.213 / 3.142× / 3.236× | 0.191 / 2.813× / 2.649× | 0.069 / 1.021× / 1.004× |
| Gaussian 32D, single query | 51 | 0.123 | 2.760 / 22.490× / 22.287× | 0.215 / 1.750× / 1.740× | 2.804 / 22.851× / 22.545× | 0.124 / 1.012× / 1.012× |
| Digits (raw) | 151 | 0.442 | 3.193 / 7.219× / 7.204× | 1.092 / 2.470× / 2.491× | 3.181 / 7.192× / 7.135× | 0.467 / 1.055× / 1.070× |
| Digits (raw) | 251 | 0.453 | 2.774 / 6.121× / 6.191× | 1.126 / 2.485× / 2.487× | 2.787 / 6.151× / 6.197× | 0.502 / 1.107× / 1.091× |
| Digits (raw) | 252 | 0.465 | 2.908 / 6.257× / 6.348× | 1.128 / 2.427× / 2.467× | 3.021 / 6.501× / 6.624× | 0.498 / 1.073× / 1.064× |
| Breast Cancer (raw) | 151 | 0.206 | 2.150 / 10.417× / 10.799× | 0.267 / 1.294× / 1.306× | 2.119 / 10.263× / 10.431× | 0.203 / **0.982×** / **0.965×** |
| Breast Cancer (raw) | 251 | 0.192 | 2.111 / 10.993× / 10.692× | 0.239 / 1.243× / 1.244× | 2.134 / 11.113× / 11.031× | 0.182 / **0.950×** / **0.954×** |
| Breast Cancer (raw) | 252 | 0.180 | 2.147 / 11.923× / 11.375× | 0.240 / 1.333× / 1.273× | 2.069 / 11.488× / 11.029× | 0.177 / **0.982×** / **0.961×** |
| Iris (raw) | 151 | 0.057 | 1.496 / 26.212× / 25.993× | 0.107 / 1.878× / 1.865× | 0.178 / 3.118× / 3.074× | 0.054 / **0.940×** / **0.922×** |
| Iris (raw) | 251 | 0.052 | 0.644 / 12.406× / 12.715× | 0.090 / 1.745× / 1.803× | 0.161 / 3.113× / 3.216× | 0.049 / **0.941×** / 1.002× |
| Iris (raw) | 252 | 0.056 | 0.606 / 10.820× / 12.034× | 0.101 / 1.802× / 1.922× | 0.173 / 3.094× / 3.131× | 0.052 / **0.923×** / 1.025× |
| Wine (raw) | 151 | 0.067 | 0.668 / 10.010× / 10.161× | 0.108 / 1.619× / 1.717× | 0.180 / 2.702× / 2.703× | 0.062 / **0.928×** / **0.936×** |
| Wine (raw) | 251 | 0.054 | 0.532 / 9.852× / 10.057× | 0.096 / 1.769× / 1.736× | 0.158 / 2.926× / 2.807× | 0.059 / 1.094× / 1.057× |
| Wine (raw) | 252 | 0.054 | 0.512 / 9.517× / 9.444× | 0.092 / 1.702× / 1.720× | 0.149 / 2.772× / 2.852× | 0.057 / 1.054× / 1.029× |

## fit_first_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.137 | 0.842 / 6.160× / 5.801× | 0.309 / 2.263× / 2.197× | 1.648 / 12.062× / 11.716× | 0.140 / 1.024× / **0.988×** |
| Gaussian 32D, single query | 49 | 0.237 | 3.256 / 13.757× / 13.589× | 0.388 / 1.639× / 1.644× | 3.252 / 13.740× / 14.096× | 0.241 / 1.020× / **0.988×** |
| Gaussian 8D, single query | 42 | 0.145 | 2.769 / 19.139× / 18.680× | 0.320 / 2.209× / 2.187× | 1.731 / 11.965× / 11.877× | 0.143 / **0.988×** / **0.996×** |
| Gaussian 32D, single query | 50 | 0.213 | 2.785 / 13.097× / 12.540× | 0.354 / 1.666× / 1.667× | 2.767 / 13.012× / 13.240× | 0.213 / 1.002× / 1.011× |
| Gaussian 8D, single query | 43 | 0.156 | 2.754 / 17.701× / 15.685× | 0.329 / 2.116× / 2.148× | 1.748 / 11.236× / 11.152× | 0.161 / 1.036× / 1.011× |
| Gaussian 32D, single query | 51 | 0.207 | 2.883 / 13.933× / 13.603× | 0.337 / 1.630× / 1.613× | 2.917 / 14.096× / 13.958× | 0.212 / 1.025× / 1.020× |
| Digits (raw) | 151 | 0.505 | 3.333 / 6.599× / 6.435× | 1.241 / 2.458× / 2.441× | 3.337 / 6.607× / 6.419× | 0.528 / 1.045× / 1.057× |
| Digits (raw) | 251 | 0.535 | 2.939 / 5.491× / 5.525× | 1.289 / 2.408× / 2.424× | 2.923 / 5.460× / 5.485× | 0.584 / 1.090× / 1.080× |
| Digits (raw) | 252 | 0.547 | 3.084 / 5.634× / 5.778× | 1.310 / 2.393× / 2.426× | 3.179 / 5.806× / 6.006× | 0.586 / 1.070× / 1.061× |
| Breast Cancer (raw) | 151 | 0.272 | 2.301 / 8.474× / 8.433× | 0.420 / 1.548× / 1.544× | 2.272 / 8.367× / 8.450× | 0.271 / **0.997×** / **0.992×** |
| Breast Cancer (raw) | 251 | 0.242 | 2.258 / 9.338× / 8.864× | 0.358 / 1.480× / 1.493× | 2.240 / 9.264× / 9.164× | 0.236 / **0.974×** / **0.957×** |
| Breast Cancer (raw) | 252 | 0.222 | 2.250 / 10.146× / 9.491× | 0.353 / 1.593× / 1.526× | 2.165 / 9.760× / 9.354× | 0.222 / 1.000× / **0.950×** |
| Iris (raw) | 151 | 0.104 | 1.603 / 15.401× / 15.659× | 0.222 / 2.135× / 2.148× | 0.329 / 3.166× / 2.976× | 0.095 / **0.917×** / **0.924×** |
| Iris (raw) | 251 | 0.088 | 0.737 / 8.410× / 8.416× | 0.184 / 2.105× / 2.180× | 0.291 / 3.325× / 3.318× | 0.085 / **0.965×** / **0.989×** |
| Iris (raw) | 252 | 0.098 | 0.736 / 7.517× / 7.706× | 0.203 / 2.071× / 2.143× | 0.324 / 3.305× / 3.235× | 0.091 / **0.928×** / 1.056× |
| Wine (raw) | 151 | 0.112 | 0.779 / 6.956× / 7.122× | 0.214 / 1.909× / 2.002× | 0.336 / 3.004× / 2.951× | 0.102 / **0.908×** / **0.931×** |
| Wine (raw) | 251 | 0.089 | 0.628 / 7.088× / 7.287× | 0.190 / 2.142× / 2.090× | 0.284 / 3.202× / 3.171× | 0.095 / 1.071× / 1.057× |
| Wine (raw) | 252 | 0.086 | 0.607 / 7.093× / 6.944× | 0.177 / 2.071× / 2.035× | 0.272 / 3.174× / 3.211× | 0.091 / 1.065× / 1.025× |

## second_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.055 | 0.741 / 13.449× / 13.414× | 0.189 / 3.432× / 3.403× | 0.141 / 2.561× / 2.563× | 0.055 / **0.995×** / **0.991×** |
| Gaussian 32D, single query | 49 | 0.112 | 3.273 / 29.233× / 29.445× | 0.207 / 1.849× / 1.844× | 3.306 / 29.525× / 29.195× | 0.113 / 1.013× / 1.007× |
| Gaussian 8D, single query | 42 | 0.055 | 2.856 / 51.690× / 49.953× | 0.190 / 3.436× / 3.458× | 0.148 / 2.686× / 2.680× | 0.055 / 1.002× / **0.994×** |
| Gaussian 32D, single query | 50 | 0.110 | 2.668 / 24.231× / 24.082× | 0.205 / 1.857× / 1.859× | 2.638 / 23.954× / 24.059× | 0.110 / **0.998×** / 1.003× |
| Gaussian 8D, single query | 43 | 0.056 | 2.678 / 47.706× / 46.251× | 0.190 / 3.391× / 3.372× | 0.144 / 2.562× / 2.509× | 0.057 / 1.009× / **0.990×** |
| Gaussian 32D, single query | 51 | 0.117 | 2.955 / 25.321× / 24.825× | 0.203 / 1.743× / 1.737× | 2.921 / 25.036× / 24.917× | 0.117 / 1.005× / 1.006× |
| Digits (raw) | 151 | 0.285 | 3.145 / 11.027× / 11.316× | 1.058 / 3.710× / 3.755× | 3.238 / 11.351× / 12.071× | 0.311 / 1.090× / 1.123× |
| Digits (raw) | 251 | 0.291 | 2.908 / 9.979× / 10.096× | 1.068 / 3.667× / 3.695× | 2.818 / 9.671× / 9.761× | 0.331 / 1.135× / 1.134× |
| Digits (raw) | 252 | 0.292 | 2.989 / 10.225× / 10.436× | 1.078 / 3.687× / 3.739× | 3.046 / 10.422× / 10.256× | 0.327 / 1.119× / 1.120× |
| Breast Cancer (raw) | 151 | 0.141 | 2.164 / 15.306× / 15.212× | 0.224 / 1.581× / 1.567× | 2.193 / 15.513× / 15.591× | 0.139 / **0.983×** / **0.984×** |
| Breast Cancer (raw) | 251 | 0.127 | 2.139 / 16.776× / 16.763× | 0.210 / 1.650× / 1.656× | 2.095 / 16.430× / 17.074× | 0.128 / 1.003× / **0.988×** |
| Breast Cancer (raw) | 252 | 0.134 | 2.107 / 15.695× / 15.251× | 0.213 / 1.585× / 1.581× | 2.157 / 16.069× / 16.017× | 0.128 / **0.957×** / **0.950×** |
| Iris (raw) | 151 | 0.041 | 1.462 / 35.375× / 35.015× | 0.079 / 1.901× / 1.927× | 0.138 / 3.333× / 3.432× | 0.042 / 1.009× / **0.984×** |
| Iris (raw) | 251 | 0.041 | 0.664 / 16.363× / 15.780× | 0.073 / 1.810× / 1.867× | 0.140 / 3.459× / 3.490× | 0.040 / **0.987×** / **0.980×** |
| Iris (raw) | 252 | 0.040 | 0.633 / 15.833× / 15.888× | 0.078 / 1.948× / 1.940× | 0.142 / 3.558× / 3.390× | 0.039 / **0.967×** / 1.013× |
| Wine (raw) | 151 | 0.051 | 0.648 / 12.713× / 12.724× | 0.085 / 1.659× / 1.667× | 0.144 / 2.827× / 2.847× | 0.049 / **0.964×** / **0.952×** |
| Wine (raw) | 251 | 0.048 | 0.573 / 11.897× / 11.981× | 0.081 / 1.672× / 1.699× | 0.141 / 2.924× / 2.905× | 0.048 / 1.003× / 1.023× |
| Wine (raw) | 252 | 0.047 | 0.534 / 11.408× / 11.035× | 0.081 / 1.722× / 1.740× | 0.134 / 2.869× / 2.921× | 0.047 / 1.009× / 1.005× |

## Every slower or missing all-round median

| Metric | Comparator | Case | Current ms | Comparator ms | M | P |
| --- | --- | --- | --- | --- | --- | --- |
| fit_ms | Preceding automatic | shape_single_8_k5_seed42 | 0.080 | 0.079 | **0.978×** | **0.994×** |
| fit_ms | Preceding automatic | shape_single_32_k5_seed50 | 0.094 | 0.093 | **0.993×** | 1.009× |
| fit_ms | Preceding automatic | real_digits_raw_k5_seed151 | 0.069 | 0.063 | **0.905×** | **0.945×** |
| fit_ms | Preceding automatic | real_digits_raw_k5_seed251 | 0.083 | 0.081 | **0.985×** | 1.015× |
| fit_ms | Preceding automatic | real_breast_cancer_raw_k5_seed151 | 0.069 | 0.069 | **0.996×** | 1.010× |
| fit_ms | Preceding automatic | real_breast_cancer_raw_k5_seed251 | 0.051 | 0.046 | **0.893×** | **0.972×** |
| fit_ms | Preceding automatic | real_iris_raw_k5_seed151 | 0.045 | 0.042 | **0.927×** | **0.930×** |
| fit_ms | Preceding automatic | real_iris_raw_k5_seed252 | 0.042 | 0.042 | **0.994×** | 1.027× |
| fit_ms | Preceding automatic | real_wine_raw_k5_seed151 | 0.045 | 0.041 | **0.914×** | **0.954×** |
| first_query_ms | Preceding automatic | shape_single_8_k5_seed41 | 0.063 | 0.062 | **0.994×** | 1.008× |
| first_query_ms | Preceding automatic | shape_single_32_k5_seed50 | 0.120 | 0.119 | **0.994×** | 1.005× |
| first_query_ms | Preceding automatic | real_breast_cancer_raw_k5_seed151 | 0.206 | 0.203 | **0.982×** | **0.965×** |
| first_query_ms | Preceding automatic | real_breast_cancer_raw_k5_seed251 | 0.192 | 0.182 | **0.950×** | **0.954×** |
| first_query_ms | Preceding automatic | real_breast_cancer_raw_k5_seed252 | 0.180 | 0.177 | **0.982×** | **0.961×** |
| first_query_ms | Preceding automatic | real_iris_raw_k5_seed151 | 0.057 | 0.054 | **0.940×** | **0.922×** |
| first_query_ms | Preceding automatic | real_iris_raw_k5_seed251 | 0.052 | 0.049 | **0.941×** | 1.002× |
| first_query_ms | Preceding automatic | real_iris_raw_k5_seed252 | 0.056 | 0.052 | **0.923×** | 1.025× |
| first_query_ms | Preceding automatic | real_wine_raw_k5_seed151 | 0.067 | 0.062 | **0.928×** | **0.936×** |
| fit_first_query_ms | Preceding automatic | shape_single_8_k5_seed42 | 0.145 | 0.143 | **0.988×** | **0.996×** |
| fit_first_query_ms | Preceding automatic | real_breast_cancer_raw_k5_seed151 | 0.272 | 0.271 | **0.997×** | **0.992×** |
| fit_first_query_ms | Preceding automatic | real_breast_cancer_raw_k5_seed251 | 0.242 | 0.236 | **0.974×** | **0.957×** |
| fit_first_query_ms | Preceding automatic | real_iris_raw_k5_seed151 | 0.104 | 0.095 | **0.917×** | **0.924×** |
| fit_first_query_ms | Preceding automatic | real_iris_raw_k5_seed251 | 0.088 | 0.085 | **0.965×** | **0.989×** |
| fit_first_query_ms | Preceding automatic | real_iris_raw_k5_seed252 | 0.098 | 0.091 | **0.928×** | 1.056× |
| fit_first_query_ms | Preceding automatic | real_wine_raw_k5_seed151 | 0.112 | 0.102 | **0.908×** | **0.931×** |
| second_query_ms | Preceding automatic | shape_single_8_k5_seed41 | 0.055 | 0.055 | **0.995×** | **0.991×** |
| second_query_ms | Preceding automatic | shape_single_32_k5_seed50 | 0.110 | 0.110 | **0.998×** | 1.003× |
| second_query_ms | Preceding automatic | real_breast_cancer_raw_k5_seed151 | 0.141 | 0.139 | **0.983×** | **0.984×** |
| second_query_ms | Preceding automatic | real_breast_cancer_raw_k5_seed252 | 0.134 | 0.128 | **0.957×** | **0.950×** |
| second_query_ms | Preceding automatic | real_iris_raw_k5_seed251 | 0.041 | 0.040 | **0.987×** | **0.980×** |
| second_query_ms | Preceding automatic | real_iris_raw_k5_seed252 | 0.040 | 0.039 | **0.967×** | 1.013× |
| second_query_ms | Preceding automatic | real_wine_raw_k5_seed151 | 0.051 | 0.049 | **0.964×** | **0.952×** |

## first_model_for_fixture

All configurations and metrics are retained in this separate view. Earlier fixtures may already have initialized process/runtime/compiler state; this table is not fresh-process cold latency.

| Case | Worker | fit_ms median ms | first_query_ms median ms | fit_first_query_ms median ms | second_query_ms median ms |
| --- | --- | --- | --- | --- | --- |
| shape_single_8_k5_seed41 | Current automatic | 0.340 | 0.197 | 0.537 | 0.151 |
| shape_single_8_k5_seed41 | Default-thread brute | 0.406 | 18.856 | 19.262 | 1.273 |
| shape_single_8_k5_seed41 | Single-thread brute | 0.289 | 8.372 | 8.660 | 0.530 |
| shape_single_8_k5_seed41 | sklearn auto | 2.003 | 0.333 | 2.336 | 0.213 |
| shape_single_8_k5_seed41 | Preceding automatic | 0.224 | 0.121 | 0.345 | 0.094 |
| shape_single_32_k5_seed49 | Current automatic | 0.141 | 0.125 | 0.266 | 0.112 |
| shape_single_32_k5_seed49 | Default-thread brute | 0.149 | 0.726 | 0.875 | 0.821 |
| shape_single_32_k5_seed49 | Single-thread brute | 0.131 | 0.226 | 0.358 | 0.204 |
| shape_single_32_k5_seed49 | sklearn auto | 0.141 | 17.496 | 17.637 | 3.770 |
| shape_single_32_k5_seed49 | Preceding automatic | 0.161 | 0.129 | 0.290 | 0.114 |
| shape_single_8_k5_seed42 | Current automatic | 0.108 | 0.074 | 0.181 | 0.058 |
| shape_single_8_k5_seed42 | Default-thread brute | 0.139 | 3.482 | 3.621 | 4.892 |
| shape_single_8_k5_seed42 | Single-thread brute | 0.108 | 0.216 | 0.324 | 0.195 |
| shape_single_8_k5_seed42 | sklearn auto | 1.616 | 0.325 | 1.941 | 0.166 |
| shape_single_8_k5_seed42 | Preceding automatic | 0.120 | 0.073 | 0.193 | 0.058 |
| shape_single_32_k5_seed50 | Current automatic | 0.143 | 0.128 | 0.271 | 0.112 |
| shape_single_32_k5_seed50 | Default-thread brute | 0.160 | 2.877 | 3.037 | 2.598 |
| shape_single_32_k5_seed50 | Single-thread brute | 0.163 | 0.238 | 0.401 | 0.204 |
| shape_single_32_k5_seed50 | sklearn auto | 0.135 | 2.406 | 2.541 | 2.603 |
| shape_single_32_k5_seed50 | Preceding automatic | 0.138 | 0.123 | 0.260 | 0.109 |
| shape_single_8_k5_seed43 | Current automatic | 0.108 | 0.070 | 0.178 | 0.055 |
| shape_single_8_k5_seed43 | Default-thread brute | 0.146 | 2.697 | 2.843 | 2.768 |
| shape_single_8_k5_seed43 | Single-thread brute | 0.148 | 0.235 | 0.383 | 0.191 |
| shape_single_8_k5_seed43 | sklearn auto | 1.590 | 0.222 | 1.812 | 0.151 |
| shape_single_8_k5_seed43 | Preceding automatic | 0.112 | 0.070 | 0.182 | 0.055 |
| shape_single_32_k5_seed51 | Current automatic | 0.098 | 0.127 | 0.225 | 0.117 |
| shape_single_32_k5_seed51 | Default-thread brute | 0.152 | 0.776 | 0.928 | 0.742 |
| shape_single_32_k5_seed51 | Single-thread brute | 0.144 | 0.231 | 0.375 | 0.203 |
| shape_single_32_k5_seed51 | sklearn auto | 0.138 | 1.995 | 2.134 | 3.140 |
| shape_single_32_k5_seed51 | Preceding automatic | 0.107 | 0.130 | 0.237 | 0.118 |
| real_digits_raw_k5_seed151 | Current automatic | 0.070 | 0.573 | 0.644 | 0.313 |
| real_digits_raw_k5_seed151 | Default-thread brute | 0.144 | 3.090 | 3.233 | 4.039 |
| real_digits_raw_k5_seed151 | Single-thread brute | 0.143 | 1.140 | 1.283 | 1.076 |
| real_digits_raw_k5_seed151 | sklearn auto | 0.156 | 3.102 | 3.258 | 2.442 |
| real_digits_raw_k5_seed151 | Preceding automatic | 0.075 | 0.602 | 0.677 | 0.329 |
| real_digits_raw_k5_seed251 | Current automatic | 0.077 | 0.491 | 0.568 | 0.299 |
| real_digits_raw_k5_seed251 | Default-thread brute | 0.159 | 2.625 | 2.784 | 3.020 |
| real_digits_raw_k5_seed251 | Single-thread brute | 0.169 | 1.141 | 1.310 | 1.077 |
| real_digits_raw_k5_seed251 | sklearn auto | 0.161 | 2.897 | 3.058 | 3.171 |
| real_digits_raw_k5_seed251 | Preceding automatic | 0.082 | 0.533 | 0.615 | 0.332 |
| real_digits_raw_k5_seed252 | Current automatic | 0.075 | 0.502 | 0.577 | 0.339 |
| real_digits_raw_k5_seed252 | Default-thread brute | 0.162 | 3.169 | 3.331 | 2.986 |
| real_digits_raw_k5_seed252 | Single-thread brute | 0.170 | 1.150 | 1.320 | 1.054 |
| real_digits_raw_k5_seed252 | sklearn auto | 0.190 | 3.438 | 3.628 | 3.395 |
| real_digits_raw_k5_seed252 | Preceding automatic | 0.082 | 0.542 | 0.624 | 0.326 |
| real_breast_cancer_raw_k5_seed151 | Current automatic | 0.068 | 0.221 | 0.289 | 0.145 |
| real_breast_cancer_raw_k5_seed151 | Default-thread brute | 0.151 | 1.983 | 2.134 | 2.273 |
| real_breast_cancer_raw_k5_seed151 | Single-thread brute | 0.193 | 0.306 | 0.499 | 0.229 |
| real_breast_cancer_raw_k5_seed151 | sklearn auto | 0.164 | 2.167 | 2.332 | 2.371 |
| real_breast_cancer_raw_k5_seed151 | Preceding automatic | 0.074 | 0.230 | 0.304 | 0.144 |
| real_breast_cancer_raw_k5_seed251 | Current automatic | 0.056 | 0.214 | 0.271 | 0.131 |
| real_breast_cancer_raw_k5_seed251 | Default-thread brute | 0.145 | 2.123 | 2.268 | 2.050 |
| real_breast_cancer_raw_k5_seed251 | Single-thread brute | 0.142 | 0.268 | 0.409 | 0.216 |
| real_breast_cancer_raw_k5_seed251 | sklearn auto | 0.157 | 2.230 | 2.386 | 2.242 |
| real_breast_cancer_raw_k5_seed251 | Preceding automatic | 0.062 | 0.214 | 0.276 | 0.162 |
| real_breast_cancer_raw_k5_seed252 | Current automatic | 0.061 | 0.211 | 0.272 | 0.166 |
| real_breast_cancer_raw_k5_seed252 | Default-thread brute | 0.143 | 1.774 | 1.917 | 2.263 |
| real_breast_cancer_raw_k5_seed252 | Single-thread brute | 0.144 | 0.269 | 0.413 | 0.219 |
| real_breast_cancer_raw_k5_seed252 | sklearn auto | 0.138 | 1.841 | 1.979 | 2.093 |
| real_breast_cancer_raw_k5_seed252 | Preceding automatic | 0.062 | 0.206 | 0.268 | 0.162 |
| real_iris_raw_k5_seed151 | Current automatic | 0.061 | 0.060 | 0.121 | 0.042 |
| real_iris_raw_k5_seed151 | Default-thread brute | 0.133 | 1.599 | 1.733 | 1.730 |
| real_iris_raw_k5_seed151 | Single-thread brute | 0.147 | 0.151 | 0.298 | 0.090 |
| real_iris_raw_k5_seed151 | sklearn auto | 0.207 | 0.219 | 0.426 | 0.153 |
| real_iris_raw_k5_seed151 | Preceding automatic | 0.097 | 0.066 | 0.163 | 0.048 |
| real_iris_raw_k5_seed251 | Current automatic | 0.060 | 0.064 | 0.124 | 0.047 |
| real_iris_raw_k5_seed251 | Default-thread brute | 0.141 | 0.548 | 0.689 | 0.577 |
| real_iris_raw_k5_seed251 | Single-thread brute | 0.146 | 0.149 | 0.295 | 0.087 |
| real_iris_raw_k5_seed251 | sklearn auto | 0.229 | 0.229 | 0.458 | 0.156 |
| real_iris_raw_k5_seed251 | Preceding automatic | 0.061 | 0.064 | 0.126 | 0.045 |
| real_iris_raw_k5_seed252 | Current automatic | 0.062 | 0.064 | 0.126 | 0.047 |
| real_iris_raw_k5_seed252 | Default-thread brute | 0.164 | 0.843 | 1.007 | 0.942 |
| real_iris_raw_k5_seed252 | Single-thread brute | 0.163 | 0.148 | 0.311 | 0.092 |
| real_iris_raw_k5_seed252 | sklearn auto | 0.219 | 0.229 | 0.448 | 0.152 |
| real_iris_raw_k5_seed252 | Preceding automatic | 0.069 | 0.066 | 0.136 | 0.045 |
| real_wine_raw_k5_seed151 | Current automatic | 0.065 | 0.074 | 0.139 | 0.051 |
| real_wine_raw_k5_seed151 | Default-thread brute | 0.147 | 0.890 | 1.037 | 0.994 |
| real_wine_raw_k5_seed151 | Single-thread brute | 0.141 | 0.151 | 0.292 | 0.086 |
| real_wine_raw_k5_seed151 | sklearn auto | 0.219 | 0.219 | 0.438 | 0.148 |
| real_wine_raw_k5_seed151 | Preceding automatic | 0.063 | 0.080 | 0.143 | 0.057 |
| real_wine_raw_k5_seed251 | Current automatic | 0.052 | 0.066 | 0.118 | 0.049 |
| real_wine_raw_k5_seed251 | Default-thread brute | 0.124 | 0.686 | 0.810 | 0.644 |
| real_wine_raw_k5_seed251 | Single-thread brute | 0.133 | 0.121 | 0.254 | 0.084 |
| real_wine_raw_k5_seed251 | sklearn auto | 0.203 | 0.223 | 0.426 | 0.146 |
| real_wine_raw_k5_seed251 | Preceding automatic | 0.067 | 0.073 | 0.140 | 0.050 |
| real_wine_raw_k5_seed252 | Current automatic | 0.061 | 0.072 | 0.133 | 0.056 |
| real_wine_raw_k5_seed252 | Default-thread brute | 0.144 | 0.717 | 0.861 | 0.702 |
| real_wine_raw_k5_seed252 | Single-thread brute | 0.148 | 0.147 | 0.295 | 0.087 |
| real_wine_raw_k5_seed252 | sklearn auto | 0.193 | 0.260 | 0.453 | 0.205 |
| real_wine_raw_k5_seed252 | Preceding automatic | 0.062 | 0.071 | 0.132 | 0.055 |

## subsequent_fresh_models

All configurations and metrics are retained in this separate view. Earlier fixtures may already have initialized process/runtime/compiler state; this table is not fresh-process cold latency.

| Case | Worker | fit_ms median ms | first_query_ms median ms | fit_first_query_ms median ms | second_query_ms median ms |
| --- | --- | --- | --- | --- | --- |
| shape_single_8_k5_seed41 | Current automatic | 0.073 | 0.063 | 0.136 | 0.055 |
| shape_single_8_k5_seed41 | Default-thread brute | 0.107 | 0.706 | 0.811 | 0.738 |
| shape_single_8_k5_seed41 | Single-thread brute | 0.102 | 0.203 | 0.308 | 0.188 |
| shape_single_8_k5_seed41 | sklearn auto | 1.473 | 0.167 | 1.644 | 0.141 |
| shape_single_8_k5_seed41 | Preceding automatic | 0.077 | 0.062 | 0.140 | 0.055 |
| shape_single_32_k5_seed49 | Current automatic | 0.113 | 0.125 | 0.237 | 0.112 |
| shape_single_32_k5_seed49 | Default-thread brute | 0.153 | 3.080 | 3.258 | 3.297 |
| shape_single_32_k5_seed49 | Single-thread brute | 0.154 | 0.232 | 0.390 | 0.207 |
| shape_single_32_k5_seed49 | sklearn auto | 0.164 | 3.094 | 3.248 | 3.303 |
| shape_single_32_k5_seed49 | Preceding automatic | 0.116 | 0.126 | 0.241 | 0.113 |
| shape_single_8_k5_seed42 | Current automatic | 0.079 | 0.064 | 0.144 | 0.055 |
| shape_single_8_k5_seed42 | Default-thread brute | 0.110 | 2.570 | 2.672 | 2.772 |
| shape_single_8_k5_seed42 | Single-thread brute | 0.110 | 0.211 | 0.318 | 0.190 |
| shape_single_8_k5_seed42 | sklearn auto | 1.544 | 0.187 | 1.730 | 0.147 |
| shape_single_8_k5_seed42 | Preceding automatic | 0.078 | 0.065 | 0.143 | 0.055 |
| shape_single_32_k5_seed50 | Current automatic | 0.092 | 0.119 | 0.212 | 0.110 |
| shape_single_32_k5_seed50 | Default-thread brute | 0.128 | 2.643 | 2.776 | 2.677 |
| shape_single_32_k5_seed50 | Single-thread brute | 0.132 | 0.220 | 0.352 | 0.205 |
| shape_single_32_k5_seed50 | sklearn auto | 0.128 | 2.635 | 2.768 | 2.642 |
| shape_single_32_k5_seed50 | Preceding automatic | 0.093 | 0.119 | 0.212 | 0.110 |
| shape_single_8_k5_seed43 | Current automatic | 0.085 | 0.068 | 0.154 | 0.056 |
| shape_single_8_k5_seed43 | Default-thread brute | 0.114 | 2.634 | 2.751 | 2.677 |
| shape_single_8_k5_seed43 | Single-thread brute | 0.116 | 0.213 | 0.328 | 0.190 |
| shape_single_8_k5_seed43 | sklearn auto | 1.551 | 0.190 | 1.746 | 0.143 |
| shape_single_8_k5_seed43 | Preceding automatic | 0.087 | 0.069 | 0.161 | 0.057 |
| shape_single_32_k5_seed51 | Current automatic | 0.084 | 0.122 | 0.206 | 0.117 |
| shape_single_32_k5_seed51 | Default-thread brute | 0.121 | 2.763 | 2.888 | 2.959 |
| shape_single_32_k5_seed51 | Single-thread brute | 0.122 | 0.215 | 0.337 | 0.203 |
| shape_single_32_k5_seed51 | sklearn auto | 0.119 | 2.809 | 2.925 | 2.917 |
| shape_single_32_k5_seed51 | Preceding automatic | 0.085 | 0.124 | 0.212 | 0.117 |
| real_digits_raw_k5_seed151 | Current automatic | 0.069 | 0.442 | 0.504 | 0.285 |
| real_digits_raw_k5_seed151 | Default-thread brute | 0.143 | 3.196 | 3.336 | 3.120 |
| real_digits_raw_k5_seed151 | Single-thread brute | 0.144 | 1.092 | 1.239 | 1.058 |
| real_digits_raw_k5_seed151 | sklearn auto | 0.142 | 3.199 | 3.351 | 3.248 |
| real_digits_raw_k5_seed151 | Preceding automatic | 0.062 | 0.465 | 0.526 | 0.305 |
| real_digits_raw_k5_seed251 | Current automatic | 0.083 | 0.453 | 0.534 | 0.290 |
| real_digits_raw_k5_seed251 | Default-thread brute | 0.157 | 2.782 | 2.943 | 2.907 |
| real_digits_raw_k5_seed251 | Single-thread brute | 0.165 | 1.125 | 1.289 | 1.068 |
| real_digits_raw_k5_seed251 | sklearn auto | 0.148 | 2.779 | 2.920 | 2.799 |
| real_digits_raw_k5_seed251 | Preceding automatic | 0.081 | 0.501 | 0.580 | 0.330 |
| real_digits_raw_k5_seed252 | Current automatic | 0.085 | 0.464 | 0.543 | 0.292 |
| real_digits_raw_k5_seed252 | Default-thread brute | 0.161 | 2.875 | 3.051 | 2.992 |
| real_digits_raw_k5_seed252 | Single-thread brute | 0.181 | 1.127 | 1.307 | 1.079 |
| real_digits_raw_k5_seed252 | sklearn auto | 0.171 | 3.015 | 3.170 | 3.045 |
| real_digits_raw_k5_seed252 | Preceding automatic | 0.088 | 0.498 | 0.585 | 0.327 |
| real_breast_cancer_raw_k5_seed151 | Current automatic | 0.070 | 0.206 | 0.271 | 0.138 |
| real_breast_cancer_raw_k5_seed151 | Default-thread brute | 0.143 | 2.164 | 2.305 | 2.161 |
| real_breast_cancer_raw_k5_seed151 | Single-thread brute | 0.149 | 0.266 | 0.417 | 0.223 |
| real_breast_cancer_raw_k5_seed151 | sklearn auto | 0.141 | 2.118 | 2.260 | 2.190 |
| real_breast_cancer_raw_k5_seed151 | Preceding automatic | 0.068 | 0.202 | 0.266 | 0.139 |
| real_breast_cancer_raw_k5_seed251 | Current automatic | 0.050 | 0.191 | 0.240 | 0.126 |
| real_breast_cancer_raw_k5_seed251 | Default-thread brute | 0.112 | 2.098 | 2.248 | 2.139 |
| real_breast_cancer_raw_k5_seed251 | Single-thread brute | 0.117 | 0.235 | 0.356 | 0.210 |
| real_breast_cancer_raw_k5_seed251 | sklearn auto | 0.109 | 2.129 | 2.236 | 2.087 |
| real_breast_cancer_raw_k5_seed251 | Preceding automatic | 0.046 | 0.181 | 0.235 | 0.127 |
| real_breast_cancer_raw_k5_seed252 | Current automatic | 0.043 | 0.178 | 0.222 | 0.134 |
| real_breast_cancer_raw_k5_seed252 | Default-thread brute | 0.107 | 2.150 | 2.251 | 2.106 |
| real_breast_cancer_raw_k5_seed252 | Single-thread brute | 0.112 | 0.240 | 0.353 | 0.212 |
| real_breast_cancer_raw_k5_seed252 | sklearn auto | 0.099 | 2.091 | 2.183 | 2.161 |
| real_breast_cancer_raw_k5_seed252 | Preceding automatic | 0.043 | 0.177 | 0.217 | 0.128 |
| real_iris_raw_k5_seed151 | Current automatic | 0.044 | 0.057 | 0.103 | 0.041 |
| real_iris_raw_k5_seed151 | Default-thread brute | 0.102 | 1.490 | 1.602 | 1.454 |
| real_iris_raw_k5_seed151 | Single-thread brute | 0.111 | 0.107 | 0.222 | 0.079 |
| real_iris_raw_k5_seed151 | sklearn auto | 0.144 | 0.176 | 0.329 | 0.136 |
| real_iris_raw_k5_seed151 | Preceding automatic | 0.041 | 0.053 | 0.095 | 0.042 |
| real_iris_raw_k5_seed251 | Current automatic | 0.035 | 0.051 | 0.087 | 0.041 |
| real_iris_raw_k5_seed251 | Default-thread brute | 0.093 | 0.645 | 0.742 | 0.665 |
| real_iris_raw_k5_seed251 | Single-thread brute | 0.096 | 0.089 | 0.183 | 0.073 |
| real_iris_raw_k5_seed251 | sklearn auto | 0.129 | 0.160 | 0.291 | 0.140 |
| real_iris_raw_k5_seed251 | Preceding automatic | 0.038 | 0.049 | 0.083 | 0.040 |
| real_iris_raw_k5_seed252 | Current automatic | 0.042 | 0.056 | 0.098 | 0.040 |
| real_iris_raw_k5_seed252 | Default-thread brute | 0.107 | 0.604 | 0.726 | 0.630 |
| real_iris_raw_k5_seed252 | Single-thread brute | 0.103 | 0.101 | 0.202 | 0.077 |
| real_iris_raw_k5_seed252 | sklearn auto | 0.147 | 0.171 | 0.322 | 0.141 |
| real_iris_raw_k5_seed252 | Preceding automatic | 0.042 | 0.052 | 0.090 | 0.038 |
| real_wine_raw_k5_seed151 | Current automatic | 0.045 | 0.067 | 0.112 | 0.051 |
| real_wine_raw_k5_seed151 | Default-thread brute | 0.104 | 0.652 | 0.768 | 0.644 |
| real_wine_raw_k5_seed151 | Single-thread brute | 0.107 | 0.107 | 0.213 | 0.084 |
| real_wine_raw_k5_seed151 | sklearn auto | 0.150 | 0.177 | 0.333 | 0.143 |
| real_wine_raw_k5_seed151 | Preceding automatic | 0.041 | 0.062 | 0.101 | 0.049 |
| real_wine_raw_k5_seed251 | Current automatic | 0.034 | 0.053 | 0.088 | 0.048 |
| real_wine_raw_k5_seed251 | Default-thread brute | 0.096 | 0.530 | 0.628 | 0.569 |
| real_wine_raw_k5_seed251 | Single-thread brute | 0.092 | 0.096 | 0.187 | 0.080 |
| real_wine_raw_k5_seed251 | sklearn auto | 0.127 | 0.157 | 0.284 | 0.140 |
| real_wine_raw_k5_seed251 | Preceding automatic | 0.036 | 0.058 | 0.094 | 0.048 |
| real_wine_raw_k5_seed252 | Current automatic | 0.032 | 0.054 | 0.085 | 0.047 |
| real_wine_raw_k5_seed252 | Default-thread brute | 0.089 | 0.512 | 0.605 | 0.522 |
| real_wine_raw_k5_seed252 | Single-thread brute | 0.083 | 0.091 | 0.177 | 0.080 |
| real_wine_raw_k5_seed252 | sklearn auto | 0.121 | 0.148 | 0.271 | 0.134 |
| real_wine_raw_k5_seed252 | Preceding automatic | 0.033 | 0.056 | 0.090 | 0.047 |

## Every audit failure

| Case | Worker | Round | Phase | Raw pointer |
| --- | --- | --- | --- | --- |
| real_digits_raw_k5_seed151 | Default-thread brute | 0 | first | `/cases/6/variants/sklearn_default/samples/0` |
| real_digits_raw_k5_seed151 | Default-thread brute | 0 | second | `/cases/6/variants/sklearn_default/samples/0` |
| real_digits_raw_k5_seed151 | Default-thread brute | 1 | first | `/cases/6/variants/sklearn_default/samples/1` |
| real_digits_raw_k5_seed151 | Default-thread brute | 1 | second | `/cases/6/variants/sklearn_default/samples/1` |
| real_digits_raw_k5_seed151 | Default-thread brute | 2 | first | `/cases/6/variants/sklearn_default/samples/2` |
| real_digits_raw_k5_seed151 | Default-thread brute | 2 | second | `/cases/6/variants/sklearn_default/samples/2` |
| real_digits_raw_k5_seed151 | Default-thread brute | 3 | first | `/cases/6/variants/sklearn_default/samples/3` |
| real_digits_raw_k5_seed151 | Default-thread brute | 3 | second | `/cases/6/variants/sklearn_default/samples/3` |
| real_digits_raw_k5_seed151 | Default-thread brute | 4 | first | `/cases/6/variants/sklearn_default/samples/4` |
| real_digits_raw_k5_seed151 | Default-thread brute | 4 | second | `/cases/6/variants/sklearn_default/samples/4` |
| real_digits_raw_k5_seed151 | Default-thread brute | 5 | first | `/cases/6/variants/sklearn_default/samples/5` |
| real_digits_raw_k5_seed151 | Default-thread brute | 5 | second | `/cases/6/variants/sklearn_default/samples/5` |
| real_digits_raw_k5_seed151 | Default-thread brute | 6 | first | `/cases/6/variants/sklearn_default/samples/6` |
| real_digits_raw_k5_seed151 | Default-thread brute | 6 | second | `/cases/6/variants/sklearn_default/samples/6` |
| real_digits_raw_k5_seed151 | Default-thread brute | 7 | first | `/cases/6/variants/sklearn_default/samples/7` |
| real_digits_raw_k5_seed151 | Default-thread brute | 7 | second | `/cases/6/variants/sklearn_default/samples/7` |
| real_digits_raw_k5_seed151 | Default-thread brute | 8 | first | `/cases/6/variants/sklearn_default/samples/8` |
| real_digits_raw_k5_seed151 | Default-thread brute | 8 | second | `/cases/6/variants/sklearn_default/samples/8` |
| real_digits_raw_k5_seed151 | Default-thread brute | 9 | first | `/cases/6/variants/sklearn_default/samples/9` |
| real_digits_raw_k5_seed151 | Default-thread brute | 9 | second | `/cases/6/variants/sklearn_default/samples/9` |
| real_digits_raw_k5_seed151 | Default-thread brute | 10 | first | `/cases/6/variants/sklearn_default/samples/10` |
| real_digits_raw_k5_seed151 | Default-thread brute | 10 | second | `/cases/6/variants/sklearn_default/samples/10` |
| real_digits_raw_k5_seed151 | Default-thread brute | 11 | first | `/cases/6/variants/sklearn_default/samples/11` |
| real_digits_raw_k5_seed151 | Default-thread brute | 11 | second | `/cases/6/variants/sklearn_default/samples/11` |
| real_digits_raw_k5_seed151 | Default-thread brute | 12 | first | `/cases/6/variants/sklearn_default/samples/12` |
| real_digits_raw_k5_seed151 | Default-thread brute | 12 | second | `/cases/6/variants/sklearn_default/samples/12` |
| real_digits_raw_k5_seed151 | Default-thread brute | 13 | first | `/cases/6/variants/sklearn_default/samples/13` |
| real_digits_raw_k5_seed151 | Default-thread brute | 13 | second | `/cases/6/variants/sklearn_default/samples/13` |
| real_digits_raw_k5_seed151 | Default-thread brute | 14 | first | `/cases/6/variants/sklearn_default/samples/14` |
| real_digits_raw_k5_seed151 | Default-thread brute | 14 | second | `/cases/6/variants/sklearn_default/samples/14` |
| real_digits_raw_k5_seed151 | Default-thread brute | 15 | first | `/cases/6/variants/sklearn_default/samples/15` |
| real_digits_raw_k5_seed151 | Default-thread brute | 15 | second | `/cases/6/variants/sklearn_default/samples/15` |
| real_digits_raw_k5_seed151 | Default-thread brute | 16 | first | `/cases/6/variants/sklearn_default/samples/16` |
| real_digits_raw_k5_seed151 | Default-thread brute | 16 | second | `/cases/6/variants/sklearn_default/samples/16` |
| real_digits_raw_k5_seed151 | Default-thread brute | 17 | first | `/cases/6/variants/sklearn_default/samples/17` |
| real_digits_raw_k5_seed151 | Default-thread brute | 17 | second | `/cases/6/variants/sklearn_default/samples/17` |
| real_digits_raw_k5_seed151 | Default-thread brute | 18 | first | `/cases/6/variants/sklearn_default/samples/18` |
| real_digits_raw_k5_seed151 | Default-thread brute | 18 | second | `/cases/6/variants/sklearn_default/samples/18` |
| real_digits_raw_k5_seed151 | Default-thread brute | 19 | first | `/cases/6/variants/sklearn_default/samples/19` |
| real_digits_raw_k5_seed151 | Default-thread brute | 19 | second | `/cases/6/variants/sklearn_default/samples/19` |
| real_digits_raw_k5_seed151 | Default-thread brute | 20 | first | `/cases/6/variants/sklearn_default/samples/20` |
| real_digits_raw_k5_seed151 | Default-thread brute | 20 | second | `/cases/6/variants/sklearn_default/samples/20` |
| real_digits_raw_k5_seed151 | Default-thread brute | 21 | first | `/cases/6/variants/sklearn_default/samples/21` |
| real_digits_raw_k5_seed151 | Default-thread brute | 21 | second | `/cases/6/variants/sklearn_default/samples/21` |
| real_digits_raw_k5_seed151 | Default-thread brute | 22 | first | `/cases/6/variants/sklearn_default/samples/22` |
| real_digits_raw_k5_seed151 | Default-thread brute | 22 | second | `/cases/6/variants/sklearn_default/samples/22` |
| real_digits_raw_k5_seed151 | Default-thread brute | 23 | first | `/cases/6/variants/sklearn_default/samples/23` |
| real_digits_raw_k5_seed151 | Default-thread brute | 23 | second | `/cases/6/variants/sklearn_default/samples/23` |
| real_digits_raw_k5_seed151 | Default-thread brute | 24 | first | `/cases/6/variants/sklearn_default/samples/24` |
| real_digits_raw_k5_seed151 | Default-thread brute | 24 | second | `/cases/6/variants/sklearn_default/samples/24` |
| real_digits_raw_k5_seed151 | Default-thread brute | 25 | first | `/cases/6/variants/sklearn_default/samples/25` |
| real_digits_raw_k5_seed151 | Default-thread brute | 25 | second | `/cases/6/variants/sklearn_default/samples/25` |
| real_digits_raw_k5_seed151 | Default-thread brute | 26 | first | `/cases/6/variants/sklearn_default/samples/26` |
| real_digits_raw_k5_seed151 | Default-thread brute | 26 | second | `/cases/6/variants/sklearn_default/samples/26` |
| real_digits_raw_k5_seed151 | Default-thread brute | 27 | first | `/cases/6/variants/sklearn_default/samples/27` |
| real_digits_raw_k5_seed151 | Default-thread brute | 27 | second | `/cases/6/variants/sklearn_default/samples/27` |
| real_digits_raw_k5_seed151 | Default-thread brute | 28 | first | `/cases/6/variants/sklearn_default/samples/28` |
| real_digits_raw_k5_seed151 | Default-thread brute | 28 | second | `/cases/6/variants/sklearn_default/samples/28` |
| real_digits_raw_k5_seed151 | Default-thread brute | 29 | first | `/cases/6/variants/sklearn_default/samples/29` |
| real_digits_raw_k5_seed151 | Default-thread brute | 29 | second | `/cases/6/variants/sklearn_default/samples/29` |
| real_digits_raw_k5_seed151 | Single-thread brute | 0 | first | `/cases/6/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed151 | Single-thread brute | 0 | second | `/cases/6/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed151 | Single-thread brute | 1 | first | `/cases/6/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed151 | Single-thread brute | 1 | second | `/cases/6/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed151 | Single-thread brute | 2 | first | `/cases/6/variants/sklearn_1/samples/2` |
| real_digits_raw_k5_seed151 | Single-thread brute | 2 | second | `/cases/6/variants/sklearn_1/samples/2` |
| real_digits_raw_k5_seed151 | Single-thread brute | 3 | first | `/cases/6/variants/sklearn_1/samples/3` |
| real_digits_raw_k5_seed151 | Single-thread brute | 3 | second | `/cases/6/variants/sklearn_1/samples/3` |
| real_digits_raw_k5_seed151 | Single-thread brute | 4 | first | `/cases/6/variants/sklearn_1/samples/4` |
| real_digits_raw_k5_seed151 | Single-thread brute | 4 | second | `/cases/6/variants/sklearn_1/samples/4` |
| real_digits_raw_k5_seed151 | Single-thread brute | 5 | first | `/cases/6/variants/sklearn_1/samples/5` |
| real_digits_raw_k5_seed151 | Single-thread brute | 5 | second | `/cases/6/variants/sklearn_1/samples/5` |
| real_digits_raw_k5_seed151 | Single-thread brute | 6 | first | `/cases/6/variants/sklearn_1/samples/6` |
| real_digits_raw_k5_seed151 | Single-thread brute | 6 | second | `/cases/6/variants/sklearn_1/samples/6` |
| real_digits_raw_k5_seed151 | Single-thread brute | 7 | first | `/cases/6/variants/sklearn_1/samples/7` |
| real_digits_raw_k5_seed151 | Single-thread brute | 7 | second | `/cases/6/variants/sklearn_1/samples/7` |
| real_digits_raw_k5_seed151 | Single-thread brute | 8 | first | `/cases/6/variants/sklearn_1/samples/8` |
| real_digits_raw_k5_seed151 | Single-thread brute | 8 | second | `/cases/6/variants/sklearn_1/samples/8` |
| real_digits_raw_k5_seed151 | Single-thread brute | 9 | first | `/cases/6/variants/sklearn_1/samples/9` |
| real_digits_raw_k5_seed151 | Single-thread brute | 9 | second | `/cases/6/variants/sklearn_1/samples/9` |
| real_digits_raw_k5_seed151 | Single-thread brute | 10 | first | `/cases/6/variants/sklearn_1/samples/10` |
| real_digits_raw_k5_seed151 | Single-thread brute | 10 | second | `/cases/6/variants/sklearn_1/samples/10` |
| real_digits_raw_k5_seed151 | Single-thread brute | 11 | first | `/cases/6/variants/sklearn_1/samples/11` |
| real_digits_raw_k5_seed151 | Single-thread brute | 11 | second | `/cases/6/variants/sklearn_1/samples/11` |
| real_digits_raw_k5_seed151 | Single-thread brute | 12 | first | `/cases/6/variants/sklearn_1/samples/12` |
| real_digits_raw_k5_seed151 | Single-thread brute | 12 | second | `/cases/6/variants/sklearn_1/samples/12` |
| real_digits_raw_k5_seed151 | Single-thread brute | 13 | first | `/cases/6/variants/sklearn_1/samples/13` |
| real_digits_raw_k5_seed151 | Single-thread brute | 13 | second | `/cases/6/variants/sklearn_1/samples/13` |
| real_digits_raw_k5_seed151 | Single-thread brute | 14 | first | `/cases/6/variants/sklearn_1/samples/14` |
| real_digits_raw_k5_seed151 | Single-thread brute | 14 | second | `/cases/6/variants/sklearn_1/samples/14` |
| real_digits_raw_k5_seed151 | Single-thread brute | 15 | first | `/cases/6/variants/sklearn_1/samples/15` |
| real_digits_raw_k5_seed151 | Single-thread brute | 15 | second | `/cases/6/variants/sklearn_1/samples/15` |
| real_digits_raw_k5_seed151 | Single-thread brute | 16 | first | `/cases/6/variants/sklearn_1/samples/16` |
| real_digits_raw_k5_seed151 | Single-thread brute | 16 | second | `/cases/6/variants/sklearn_1/samples/16` |
| real_digits_raw_k5_seed151 | Single-thread brute | 17 | first | `/cases/6/variants/sklearn_1/samples/17` |
| real_digits_raw_k5_seed151 | Single-thread brute | 17 | second | `/cases/6/variants/sklearn_1/samples/17` |
| real_digits_raw_k5_seed151 | Single-thread brute | 18 | first | `/cases/6/variants/sklearn_1/samples/18` |
| real_digits_raw_k5_seed151 | Single-thread brute | 18 | second | `/cases/6/variants/sklearn_1/samples/18` |
| real_digits_raw_k5_seed151 | Single-thread brute | 19 | first | `/cases/6/variants/sklearn_1/samples/19` |
| real_digits_raw_k5_seed151 | Single-thread brute | 19 | second | `/cases/6/variants/sklearn_1/samples/19` |
| real_digits_raw_k5_seed151 | Single-thread brute | 20 | first | `/cases/6/variants/sklearn_1/samples/20` |
| real_digits_raw_k5_seed151 | Single-thread brute | 20 | second | `/cases/6/variants/sklearn_1/samples/20` |
| real_digits_raw_k5_seed151 | Single-thread brute | 21 | first | `/cases/6/variants/sklearn_1/samples/21` |
| real_digits_raw_k5_seed151 | Single-thread brute | 21 | second | `/cases/6/variants/sklearn_1/samples/21` |
| real_digits_raw_k5_seed151 | Single-thread brute | 22 | first | `/cases/6/variants/sklearn_1/samples/22` |
| real_digits_raw_k5_seed151 | Single-thread brute | 22 | second | `/cases/6/variants/sklearn_1/samples/22` |
| real_digits_raw_k5_seed151 | Single-thread brute | 23 | first | `/cases/6/variants/sklearn_1/samples/23` |
| real_digits_raw_k5_seed151 | Single-thread brute | 23 | second | `/cases/6/variants/sklearn_1/samples/23` |
| real_digits_raw_k5_seed151 | Single-thread brute | 24 | first | `/cases/6/variants/sklearn_1/samples/24` |
| real_digits_raw_k5_seed151 | Single-thread brute | 24 | second | `/cases/6/variants/sklearn_1/samples/24` |
| real_digits_raw_k5_seed151 | Single-thread brute | 25 | first | `/cases/6/variants/sklearn_1/samples/25` |
| real_digits_raw_k5_seed151 | Single-thread brute | 25 | second | `/cases/6/variants/sklearn_1/samples/25` |
| real_digits_raw_k5_seed151 | Single-thread brute | 26 | first | `/cases/6/variants/sklearn_1/samples/26` |
| real_digits_raw_k5_seed151 | Single-thread brute | 26 | second | `/cases/6/variants/sklearn_1/samples/26` |
| real_digits_raw_k5_seed151 | Single-thread brute | 27 | first | `/cases/6/variants/sklearn_1/samples/27` |
| real_digits_raw_k5_seed151 | Single-thread brute | 27 | second | `/cases/6/variants/sklearn_1/samples/27` |
| real_digits_raw_k5_seed151 | Single-thread brute | 28 | first | `/cases/6/variants/sklearn_1/samples/28` |
| real_digits_raw_k5_seed151 | Single-thread brute | 28 | second | `/cases/6/variants/sklearn_1/samples/28` |
| real_digits_raw_k5_seed151 | Single-thread brute | 29 | first | `/cases/6/variants/sklearn_1/samples/29` |
| real_digits_raw_k5_seed151 | Single-thread brute | 29 | second | `/cases/6/variants/sklearn_1/samples/29` |
| real_digits_raw_k5_seed151 | sklearn auto | 0 | first | `/cases/6/variants/sklearn_auto/samples/0` |
| real_digits_raw_k5_seed151 | sklearn auto | 0 | second | `/cases/6/variants/sklearn_auto/samples/0` |
| real_digits_raw_k5_seed151 | sklearn auto | 1 | first | `/cases/6/variants/sklearn_auto/samples/1` |
| real_digits_raw_k5_seed151 | sklearn auto | 1 | second | `/cases/6/variants/sklearn_auto/samples/1` |
| real_digits_raw_k5_seed151 | sklearn auto | 2 | first | `/cases/6/variants/sklearn_auto/samples/2` |
| real_digits_raw_k5_seed151 | sklearn auto | 2 | second | `/cases/6/variants/sklearn_auto/samples/2` |
| real_digits_raw_k5_seed151 | sklearn auto | 3 | first | `/cases/6/variants/sklearn_auto/samples/3` |
| real_digits_raw_k5_seed151 | sklearn auto | 3 | second | `/cases/6/variants/sklearn_auto/samples/3` |
| real_digits_raw_k5_seed151 | sklearn auto | 4 | first | `/cases/6/variants/sklearn_auto/samples/4` |
| real_digits_raw_k5_seed151 | sklearn auto | 4 | second | `/cases/6/variants/sklearn_auto/samples/4` |
| real_digits_raw_k5_seed151 | sklearn auto | 5 | first | `/cases/6/variants/sklearn_auto/samples/5` |
| real_digits_raw_k5_seed151 | sklearn auto | 5 | second | `/cases/6/variants/sklearn_auto/samples/5` |
| real_digits_raw_k5_seed151 | sklearn auto | 6 | first | `/cases/6/variants/sklearn_auto/samples/6` |
| real_digits_raw_k5_seed151 | sklearn auto | 6 | second | `/cases/6/variants/sklearn_auto/samples/6` |
| real_digits_raw_k5_seed151 | sklearn auto | 7 | first | `/cases/6/variants/sklearn_auto/samples/7` |
| real_digits_raw_k5_seed151 | sklearn auto | 7 | second | `/cases/6/variants/sklearn_auto/samples/7` |
| real_digits_raw_k5_seed151 | sklearn auto | 8 | first | `/cases/6/variants/sklearn_auto/samples/8` |
| real_digits_raw_k5_seed151 | sklearn auto | 8 | second | `/cases/6/variants/sklearn_auto/samples/8` |
| real_digits_raw_k5_seed151 | sklearn auto | 9 | first | `/cases/6/variants/sklearn_auto/samples/9` |
| real_digits_raw_k5_seed151 | sklearn auto | 9 | second | `/cases/6/variants/sklearn_auto/samples/9` |
| real_digits_raw_k5_seed151 | sklearn auto | 10 | first | `/cases/6/variants/sklearn_auto/samples/10` |
| real_digits_raw_k5_seed151 | sklearn auto | 10 | second | `/cases/6/variants/sklearn_auto/samples/10` |
| real_digits_raw_k5_seed151 | sklearn auto | 11 | first | `/cases/6/variants/sklearn_auto/samples/11` |
| real_digits_raw_k5_seed151 | sklearn auto | 11 | second | `/cases/6/variants/sklearn_auto/samples/11` |
| real_digits_raw_k5_seed151 | sklearn auto | 12 | first | `/cases/6/variants/sklearn_auto/samples/12` |
| real_digits_raw_k5_seed151 | sklearn auto | 12 | second | `/cases/6/variants/sklearn_auto/samples/12` |
| real_digits_raw_k5_seed151 | sklearn auto | 13 | first | `/cases/6/variants/sklearn_auto/samples/13` |
| real_digits_raw_k5_seed151 | sklearn auto | 13 | second | `/cases/6/variants/sklearn_auto/samples/13` |
| real_digits_raw_k5_seed151 | sklearn auto | 14 | first | `/cases/6/variants/sklearn_auto/samples/14` |
| real_digits_raw_k5_seed151 | sklearn auto | 14 | second | `/cases/6/variants/sklearn_auto/samples/14` |
| real_digits_raw_k5_seed151 | sklearn auto | 15 | first | `/cases/6/variants/sklearn_auto/samples/15` |
| real_digits_raw_k5_seed151 | sklearn auto | 15 | second | `/cases/6/variants/sklearn_auto/samples/15` |
| real_digits_raw_k5_seed151 | sklearn auto | 16 | first | `/cases/6/variants/sklearn_auto/samples/16` |
| real_digits_raw_k5_seed151 | sklearn auto | 16 | second | `/cases/6/variants/sklearn_auto/samples/16` |
| real_digits_raw_k5_seed151 | sklearn auto | 17 | first | `/cases/6/variants/sklearn_auto/samples/17` |
| real_digits_raw_k5_seed151 | sklearn auto | 17 | second | `/cases/6/variants/sklearn_auto/samples/17` |
| real_digits_raw_k5_seed151 | sklearn auto | 18 | first | `/cases/6/variants/sklearn_auto/samples/18` |
| real_digits_raw_k5_seed151 | sklearn auto | 18 | second | `/cases/6/variants/sklearn_auto/samples/18` |
| real_digits_raw_k5_seed151 | sklearn auto | 19 | first | `/cases/6/variants/sklearn_auto/samples/19` |
| real_digits_raw_k5_seed151 | sklearn auto | 19 | second | `/cases/6/variants/sklearn_auto/samples/19` |
| real_digits_raw_k5_seed151 | sklearn auto | 20 | first | `/cases/6/variants/sklearn_auto/samples/20` |
| real_digits_raw_k5_seed151 | sklearn auto | 20 | second | `/cases/6/variants/sklearn_auto/samples/20` |
| real_digits_raw_k5_seed151 | sklearn auto | 21 | first | `/cases/6/variants/sklearn_auto/samples/21` |
| real_digits_raw_k5_seed151 | sklearn auto | 21 | second | `/cases/6/variants/sklearn_auto/samples/21` |
| real_digits_raw_k5_seed151 | sklearn auto | 22 | first | `/cases/6/variants/sklearn_auto/samples/22` |
| real_digits_raw_k5_seed151 | sklearn auto | 22 | second | `/cases/6/variants/sklearn_auto/samples/22` |
| real_digits_raw_k5_seed151 | sklearn auto | 23 | first | `/cases/6/variants/sklearn_auto/samples/23` |
| real_digits_raw_k5_seed151 | sklearn auto | 23 | second | `/cases/6/variants/sklearn_auto/samples/23` |
| real_digits_raw_k5_seed151 | sklearn auto | 24 | first | `/cases/6/variants/sklearn_auto/samples/24` |
| real_digits_raw_k5_seed151 | sklearn auto | 24 | second | `/cases/6/variants/sklearn_auto/samples/24` |
| real_digits_raw_k5_seed151 | sklearn auto | 25 | first | `/cases/6/variants/sklearn_auto/samples/25` |
| real_digits_raw_k5_seed151 | sklearn auto | 25 | second | `/cases/6/variants/sklearn_auto/samples/25` |
| real_digits_raw_k5_seed151 | sklearn auto | 26 | first | `/cases/6/variants/sklearn_auto/samples/26` |
| real_digits_raw_k5_seed151 | sklearn auto | 26 | second | `/cases/6/variants/sklearn_auto/samples/26` |
| real_digits_raw_k5_seed151 | sklearn auto | 27 | first | `/cases/6/variants/sklearn_auto/samples/27` |
| real_digits_raw_k5_seed151 | sklearn auto | 27 | second | `/cases/6/variants/sklearn_auto/samples/27` |
| real_digits_raw_k5_seed151 | sklearn auto | 28 | first | `/cases/6/variants/sklearn_auto/samples/28` |
| real_digits_raw_k5_seed151 | sklearn auto | 28 | second | `/cases/6/variants/sklearn_auto/samples/28` |
| real_digits_raw_k5_seed151 | sklearn auto | 29 | first | `/cases/6/variants/sklearn_auto/samples/29` |
| real_digits_raw_k5_seed151 | sklearn auto | 29 | second | `/cases/6/variants/sklearn_auto/samples/29` |
| real_digits_raw_k5_seed251 | Default-thread brute | 0 | first | `/cases/7/variants/sklearn_default/samples/0` |
| real_digits_raw_k5_seed251 | Default-thread brute | 0 | second | `/cases/7/variants/sklearn_default/samples/0` |
| real_digits_raw_k5_seed251 | Default-thread brute | 1 | first | `/cases/7/variants/sklearn_default/samples/1` |
| real_digits_raw_k5_seed251 | Default-thread brute | 1 | second | `/cases/7/variants/sklearn_default/samples/1` |
| real_digits_raw_k5_seed251 | Default-thread brute | 2 | first | `/cases/7/variants/sklearn_default/samples/2` |
| real_digits_raw_k5_seed251 | Default-thread brute | 2 | second | `/cases/7/variants/sklearn_default/samples/2` |
| real_digits_raw_k5_seed251 | Default-thread brute | 3 | first | `/cases/7/variants/sklearn_default/samples/3` |
| real_digits_raw_k5_seed251 | Default-thread brute | 3 | second | `/cases/7/variants/sklearn_default/samples/3` |
| real_digits_raw_k5_seed251 | Default-thread brute | 4 | first | `/cases/7/variants/sklearn_default/samples/4` |
| real_digits_raw_k5_seed251 | Default-thread brute | 4 | second | `/cases/7/variants/sklearn_default/samples/4` |
| real_digits_raw_k5_seed251 | Default-thread brute | 5 | first | `/cases/7/variants/sklearn_default/samples/5` |
| real_digits_raw_k5_seed251 | Default-thread brute | 5 | second | `/cases/7/variants/sklearn_default/samples/5` |
| real_digits_raw_k5_seed251 | Default-thread brute | 6 | first | `/cases/7/variants/sklearn_default/samples/6` |
| real_digits_raw_k5_seed251 | Default-thread brute | 6 | second | `/cases/7/variants/sklearn_default/samples/6` |
| real_digits_raw_k5_seed251 | Default-thread brute | 7 | first | `/cases/7/variants/sklearn_default/samples/7` |
| real_digits_raw_k5_seed251 | Default-thread brute | 7 | second | `/cases/7/variants/sklearn_default/samples/7` |
| real_digits_raw_k5_seed251 | Default-thread brute | 8 | first | `/cases/7/variants/sklearn_default/samples/8` |
| real_digits_raw_k5_seed251 | Default-thread brute | 8 | second | `/cases/7/variants/sklearn_default/samples/8` |
| real_digits_raw_k5_seed251 | Default-thread brute | 9 | first | `/cases/7/variants/sklearn_default/samples/9` |
| real_digits_raw_k5_seed251 | Default-thread brute | 9 | second | `/cases/7/variants/sklearn_default/samples/9` |
| real_digits_raw_k5_seed251 | Default-thread brute | 10 | first | `/cases/7/variants/sklearn_default/samples/10` |
| real_digits_raw_k5_seed251 | Default-thread brute | 10 | second | `/cases/7/variants/sklearn_default/samples/10` |
| real_digits_raw_k5_seed251 | Default-thread brute | 11 | first | `/cases/7/variants/sklearn_default/samples/11` |
| real_digits_raw_k5_seed251 | Default-thread brute | 11 | second | `/cases/7/variants/sklearn_default/samples/11` |
| real_digits_raw_k5_seed251 | Default-thread brute | 12 | first | `/cases/7/variants/sklearn_default/samples/12` |
| real_digits_raw_k5_seed251 | Default-thread brute | 12 | second | `/cases/7/variants/sklearn_default/samples/12` |
| real_digits_raw_k5_seed251 | Default-thread brute | 13 | first | `/cases/7/variants/sklearn_default/samples/13` |
| real_digits_raw_k5_seed251 | Default-thread brute | 13 | second | `/cases/7/variants/sklearn_default/samples/13` |
| real_digits_raw_k5_seed251 | Default-thread brute | 14 | first | `/cases/7/variants/sklearn_default/samples/14` |
| real_digits_raw_k5_seed251 | Default-thread brute | 14 | second | `/cases/7/variants/sklearn_default/samples/14` |
| real_digits_raw_k5_seed251 | Default-thread brute | 15 | first | `/cases/7/variants/sklearn_default/samples/15` |
| real_digits_raw_k5_seed251 | Default-thread brute | 15 | second | `/cases/7/variants/sklearn_default/samples/15` |
| real_digits_raw_k5_seed251 | Default-thread brute | 16 | first | `/cases/7/variants/sklearn_default/samples/16` |
| real_digits_raw_k5_seed251 | Default-thread brute | 16 | second | `/cases/7/variants/sklearn_default/samples/16` |
| real_digits_raw_k5_seed251 | Default-thread brute | 17 | first | `/cases/7/variants/sklearn_default/samples/17` |
| real_digits_raw_k5_seed251 | Default-thread brute | 17 | second | `/cases/7/variants/sklearn_default/samples/17` |
| real_digits_raw_k5_seed251 | Default-thread brute | 18 | first | `/cases/7/variants/sklearn_default/samples/18` |
| real_digits_raw_k5_seed251 | Default-thread brute | 18 | second | `/cases/7/variants/sklearn_default/samples/18` |
| real_digits_raw_k5_seed251 | Default-thread brute | 19 | first | `/cases/7/variants/sklearn_default/samples/19` |
| real_digits_raw_k5_seed251 | Default-thread brute | 19 | second | `/cases/7/variants/sklearn_default/samples/19` |
| real_digits_raw_k5_seed251 | Default-thread brute | 20 | first | `/cases/7/variants/sklearn_default/samples/20` |
| real_digits_raw_k5_seed251 | Default-thread brute | 20 | second | `/cases/7/variants/sklearn_default/samples/20` |
| real_digits_raw_k5_seed251 | Default-thread brute | 21 | first | `/cases/7/variants/sklearn_default/samples/21` |
| real_digits_raw_k5_seed251 | Default-thread brute | 21 | second | `/cases/7/variants/sklearn_default/samples/21` |
| real_digits_raw_k5_seed251 | Default-thread brute | 22 | first | `/cases/7/variants/sklearn_default/samples/22` |
| real_digits_raw_k5_seed251 | Default-thread brute | 22 | second | `/cases/7/variants/sklearn_default/samples/22` |
| real_digits_raw_k5_seed251 | Default-thread brute | 23 | first | `/cases/7/variants/sklearn_default/samples/23` |
| real_digits_raw_k5_seed251 | Default-thread brute | 23 | second | `/cases/7/variants/sklearn_default/samples/23` |
| real_digits_raw_k5_seed251 | Default-thread brute | 24 | first | `/cases/7/variants/sklearn_default/samples/24` |
| real_digits_raw_k5_seed251 | Default-thread brute | 24 | second | `/cases/7/variants/sklearn_default/samples/24` |
| real_digits_raw_k5_seed251 | Default-thread brute | 25 | first | `/cases/7/variants/sklearn_default/samples/25` |
| real_digits_raw_k5_seed251 | Default-thread brute | 25 | second | `/cases/7/variants/sklearn_default/samples/25` |
| real_digits_raw_k5_seed251 | Default-thread brute | 26 | first | `/cases/7/variants/sklearn_default/samples/26` |
| real_digits_raw_k5_seed251 | Default-thread brute | 26 | second | `/cases/7/variants/sklearn_default/samples/26` |
| real_digits_raw_k5_seed251 | Default-thread brute | 27 | first | `/cases/7/variants/sklearn_default/samples/27` |
| real_digits_raw_k5_seed251 | Default-thread brute | 27 | second | `/cases/7/variants/sklearn_default/samples/27` |
| real_digits_raw_k5_seed251 | Default-thread brute | 28 | first | `/cases/7/variants/sklearn_default/samples/28` |
| real_digits_raw_k5_seed251 | Default-thread brute | 28 | second | `/cases/7/variants/sklearn_default/samples/28` |
| real_digits_raw_k5_seed251 | Default-thread brute | 29 | first | `/cases/7/variants/sklearn_default/samples/29` |
| real_digits_raw_k5_seed251 | Default-thread brute | 29 | second | `/cases/7/variants/sklearn_default/samples/29` |
| real_digits_raw_k5_seed251 | Single-thread brute | 0 | first | `/cases/7/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed251 | Single-thread brute | 0 | second | `/cases/7/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed251 | Single-thread brute | 1 | first | `/cases/7/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed251 | Single-thread brute | 1 | second | `/cases/7/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed251 | Single-thread brute | 2 | first | `/cases/7/variants/sklearn_1/samples/2` |
| real_digits_raw_k5_seed251 | Single-thread brute | 2 | second | `/cases/7/variants/sklearn_1/samples/2` |
| real_digits_raw_k5_seed251 | Single-thread brute | 3 | first | `/cases/7/variants/sklearn_1/samples/3` |
| real_digits_raw_k5_seed251 | Single-thread brute | 3 | second | `/cases/7/variants/sklearn_1/samples/3` |
| real_digits_raw_k5_seed251 | Single-thread brute | 4 | first | `/cases/7/variants/sklearn_1/samples/4` |
| real_digits_raw_k5_seed251 | Single-thread brute | 4 | second | `/cases/7/variants/sklearn_1/samples/4` |
| real_digits_raw_k5_seed251 | Single-thread brute | 5 | first | `/cases/7/variants/sklearn_1/samples/5` |
| real_digits_raw_k5_seed251 | Single-thread brute | 5 | second | `/cases/7/variants/sklearn_1/samples/5` |
| real_digits_raw_k5_seed251 | Single-thread brute | 6 | first | `/cases/7/variants/sklearn_1/samples/6` |
| real_digits_raw_k5_seed251 | Single-thread brute | 6 | second | `/cases/7/variants/sklearn_1/samples/6` |
| real_digits_raw_k5_seed251 | Single-thread brute | 7 | first | `/cases/7/variants/sklearn_1/samples/7` |
| real_digits_raw_k5_seed251 | Single-thread brute | 7 | second | `/cases/7/variants/sklearn_1/samples/7` |
| real_digits_raw_k5_seed251 | Single-thread brute | 8 | first | `/cases/7/variants/sklearn_1/samples/8` |
| real_digits_raw_k5_seed251 | Single-thread brute | 8 | second | `/cases/7/variants/sklearn_1/samples/8` |
| real_digits_raw_k5_seed251 | Single-thread brute | 9 | first | `/cases/7/variants/sklearn_1/samples/9` |
| real_digits_raw_k5_seed251 | Single-thread brute | 9 | second | `/cases/7/variants/sklearn_1/samples/9` |
| real_digits_raw_k5_seed251 | Single-thread brute | 10 | first | `/cases/7/variants/sklearn_1/samples/10` |
| real_digits_raw_k5_seed251 | Single-thread brute | 10 | second | `/cases/7/variants/sklearn_1/samples/10` |
| real_digits_raw_k5_seed251 | Single-thread brute | 11 | first | `/cases/7/variants/sklearn_1/samples/11` |
| real_digits_raw_k5_seed251 | Single-thread brute | 11 | second | `/cases/7/variants/sklearn_1/samples/11` |
| real_digits_raw_k5_seed251 | Single-thread brute | 12 | first | `/cases/7/variants/sklearn_1/samples/12` |
| real_digits_raw_k5_seed251 | Single-thread brute | 12 | second | `/cases/7/variants/sklearn_1/samples/12` |
| real_digits_raw_k5_seed251 | Single-thread brute | 13 | first | `/cases/7/variants/sklearn_1/samples/13` |
| real_digits_raw_k5_seed251 | Single-thread brute | 13 | second | `/cases/7/variants/sklearn_1/samples/13` |
| real_digits_raw_k5_seed251 | Single-thread brute | 14 | first | `/cases/7/variants/sklearn_1/samples/14` |
| real_digits_raw_k5_seed251 | Single-thread brute | 14 | second | `/cases/7/variants/sklearn_1/samples/14` |
| real_digits_raw_k5_seed251 | Single-thread brute | 15 | first | `/cases/7/variants/sklearn_1/samples/15` |
| real_digits_raw_k5_seed251 | Single-thread brute | 15 | second | `/cases/7/variants/sklearn_1/samples/15` |
| real_digits_raw_k5_seed251 | Single-thread brute | 16 | first | `/cases/7/variants/sklearn_1/samples/16` |
| real_digits_raw_k5_seed251 | Single-thread brute | 16 | second | `/cases/7/variants/sklearn_1/samples/16` |
| real_digits_raw_k5_seed251 | Single-thread brute | 17 | first | `/cases/7/variants/sklearn_1/samples/17` |
| real_digits_raw_k5_seed251 | Single-thread brute | 17 | second | `/cases/7/variants/sklearn_1/samples/17` |
| real_digits_raw_k5_seed251 | Single-thread brute | 18 | first | `/cases/7/variants/sklearn_1/samples/18` |
| real_digits_raw_k5_seed251 | Single-thread brute | 18 | second | `/cases/7/variants/sklearn_1/samples/18` |
| real_digits_raw_k5_seed251 | Single-thread brute | 19 | first | `/cases/7/variants/sklearn_1/samples/19` |
| real_digits_raw_k5_seed251 | Single-thread brute | 19 | second | `/cases/7/variants/sklearn_1/samples/19` |
| real_digits_raw_k5_seed251 | Single-thread brute | 20 | first | `/cases/7/variants/sklearn_1/samples/20` |
| real_digits_raw_k5_seed251 | Single-thread brute | 20 | second | `/cases/7/variants/sklearn_1/samples/20` |
| real_digits_raw_k5_seed251 | Single-thread brute | 21 | first | `/cases/7/variants/sklearn_1/samples/21` |
| real_digits_raw_k5_seed251 | Single-thread brute | 21 | second | `/cases/7/variants/sklearn_1/samples/21` |
| real_digits_raw_k5_seed251 | Single-thread brute | 22 | first | `/cases/7/variants/sklearn_1/samples/22` |
| real_digits_raw_k5_seed251 | Single-thread brute | 22 | second | `/cases/7/variants/sklearn_1/samples/22` |
| real_digits_raw_k5_seed251 | Single-thread brute | 23 | first | `/cases/7/variants/sklearn_1/samples/23` |
| real_digits_raw_k5_seed251 | Single-thread brute | 23 | second | `/cases/7/variants/sklearn_1/samples/23` |
| real_digits_raw_k5_seed251 | Single-thread brute | 24 | first | `/cases/7/variants/sklearn_1/samples/24` |
| real_digits_raw_k5_seed251 | Single-thread brute | 24 | second | `/cases/7/variants/sklearn_1/samples/24` |
| real_digits_raw_k5_seed251 | Single-thread brute | 25 | first | `/cases/7/variants/sklearn_1/samples/25` |
| real_digits_raw_k5_seed251 | Single-thread brute | 25 | second | `/cases/7/variants/sklearn_1/samples/25` |
| real_digits_raw_k5_seed251 | Single-thread brute | 26 | first | `/cases/7/variants/sklearn_1/samples/26` |
| real_digits_raw_k5_seed251 | Single-thread brute | 26 | second | `/cases/7/variants/sklearn_1/samples/26` |
| real_digits_raw_k5_seed251 | Single-thread brute | 27 | first | `/cases/7/variants/sklearn_1/samples/27` |
| real_digits_raw_k5_seed251 | Single-thread brute | 27 | second | `/cases/7/variants/sklearn_1/samples/27` |
| real_digits_raw_k5_seed251 | Single-thread brute | 28 | first | `/cases/7/variants/sklearn_1/samples/28` |
| real_digits_raw_k5_seed251 | Single-thread brute | 28 | second | `/cases/7/variants/sklearn_1/samples/28` |
| real_digits_raw_k5_seed251 | Single-thread brute | 29 | first | `/cases/7/variants/sklearn_1/samples/29` |
| real_digits_raw_k5_seed251 | Single-thread brute | 29 | second | `/cases/7/variants/sklearn_1/samples/29` |
| real_digits_raw_k5_seed251 | sklearn auto | 0 | first | `/cases/7/variants/sklearn_auto/samples/0` |
| real_digits_raw_k5_seed251 | sklearn auto | 0 | second | `/cases/7/variants/sklearn_auto/samples/0` |
| real_digits_raw_k5_seed251 | sklearn auto | 1 | first | `/cases/7/variants/sklearn_auto/samples/1` |
| real_digits_raw_k5_seed251 | sklearn auto | 1 | second | `/cases/7/variants/sklearn_auto/samples/1` |
| real_digits_raw_k5_seed251 | sklearn auto | 2 | first | `/cases/7/variants/sklearn_auto/samples/2` |
| real_digits_raw_k5_seed251 | sklearn auto | 2 | second | `/cases/7/variants/sklearn_auto/samples/2` |
| real_digits_raw_k5_seed251 | sklearn auto | 3 | first | `/cases/7/variants/sklearn_auto/samples/3` |
| real_digits_raw_k5_seed251 | sklearn auto | 3 | second | `/cases/7/variants/sklearn_auto/samples/3` |
| real_digits_raw_k5_seed251 | sklearn auto | 4 | first | `/cases/7/variants/sklearn_auto/samples/4` |
| real_digits_raw_k5_seed251 | sklearn auto | 4 | second | `/cases/7/variants/sklearn_auto/samples/4` |
| real_digits_raw_k5_seed251 | sklearn auto | 5 | first | `/cases/7/variants/sklearn_auto/samples/5` |
| real_digits_raw_k5_seed251 | sklearn auto | 5 | second | `/cases/7/variants/sklearn_auto/samples/5` |
| real_digits_raw_k5_seed251 | sklearn auto | 6 | first | `/cases/7/variants/sklearn_auto/samples/6` |
| real_digits_raw_k5_seed251 | sklearn auto | 6 | second | `/cases/7/variants/sklearn_auto/samples/6` |
| real_digits_raw_k5_seed251 | sklearn auto | 7 | first | `/cases/7/variants/sklearn_auto/samples/7` |
| real_digits_raw_k5_seed251 | sklearn auto | 7 | second | `/cases/7/variants/sklearn_auto/samples/7` |
| real_digits_raw_k5_seed251 | sklearn auto | 8 | first | `/cases/7/variants/sklearn_auto/samples/8` |
| real_digits_raw_k5_seed251 | sklearn auto | 8 | second | `/cases/7/variants/sklearn_auto/samples/8` |
| real_digits_raw_k5_seed251 | sklearn auto | 9 | first | `/cases/7/variants/sklearn_auto/samples/9` |
| real_digits_raw_k5_seed251 | sklearn auto | 9 | second | `/cases/7/variants/sklearn_auto/samples/9` |
| real_digits_raw_k5_seed251 | sklearn auto | 10 | first | `/cases/7/variants/sklearn_auto/samples/10` |
| real_digits_raw_k5_seed251 | sklearn auto | 10 | second | `/cases/7/variants/sklearn_auto/samples/10` |
| real_digits_raw_k5_seed251 | sklearn auto | 11 | first | `/cases/7/variants/sklearn_auto/samples/11` |
| real_digits_raw_k5_seed251 | sklearn auto | 11 | second | `/cases/7/variants/sklearn_auto/samples/11` |
| real_digits_raw_k5_seed251 | sklearn auto | 12 | first | `/cases/7/variants/sklearn_auto/samples/12` |
| real_digits_raw_k5_seed251 | sklearn auto | 12 | second | `/cases/7/variants/sklearn_auto/samples/12` |
| real_digits_raw_k5_seed251 | sklearn auto | 13 | first | `/cases/7/variants/sklearn_auto/samples/13` |
| real_digits_raw_k5_seed251 | sklearn auto | 13 | second | `/cases/7/variants/sklearn_auto/samples/13` |
| real_digits_raw_k5_seed251 | sklearn auto | 14 | first | `/cases/7/variants/sklearn_auto/samples/14` |
| real_digits_raw_k5_seed251 | sklearn auto | 14 | second | `/cases/7/variants/sklearn_auto/samples/14` |
| real_digits_raw_k5_seed251 | sklearn auto | 15 | first | `/cases/7/variants/sklearn_auto/samples/15` |
| real_digits_raw_k5_seed251 | sklearn auto | 15 | second | `/cases/7/variants/sklearn_auto/samples/15` |
| real_digits_raw_k5_seed251 | sklearn auto | 16 | first | `/cases/7/variants/sklearn_auto/samples/16` |
| real_digits_raw_k5_seed251 | sklearn auto | 16 | second | `/cases/7/variants/sklearn_auto/samples/16` |
| real_digits_raw_k5_seed251 | sklearn auto | 17 | first | `/cases/7/variants/sklearn_auto/samples/17` |
| real_digits_raw_k5_seed251 | sklearn auto | 17 | second | `/cases/7/variants/sklearn_auto/samples/17` |
| real_digits_raw_k5_seed251 | sklearn auto | 18 | first | `/cases/7/variants/sklearn_auto/samples/18` |
| real_digits_raw_k5_seed251 | sklearn auto | 18 | second | `/cases/7/variants/sklearn_auto/samples/18` |
| real_digits_raw_k5_seed251 | sklearn auto | 19 | first | `/cases/7/variants/sklearn_auto/samples/19` |
| real_digits_raw_k5_seed251 | sklearn auto | 19 | second | `/cases/7/variants/sklearn_auto/samples/19` |
| real_digits_raw_k5_seed251 | sklearn auto | 20 | first | `/cases/7/variants/sklearn_auto/samples/20` |
| real_digits_raw_k5_seed251 | sklearn auto | 20 | second | `/cases/7/variants/sklearn_auto/samples/20` |
| real_digits_raw_k5_seed251 | sklearn auto | 21 | first | `/cases/7/variants/sklearn_auto/samples/21` |
| real_digits_raw_k5_seed251 | sklearn auto | 21 | second | `/cases/7/variants/sklearn_auto/samples/21` |
| real_digits_raw_k5_seed251 | sklearn auto | 22 | first | `/cases/7/variants/sklearn_auto/samples/22` |
| real_digits_raw_k5_seed251 | sklearn auto | 22 | second | `/cases/7/variants/sklearn_auto/samples/22` |
| real_digits_raw_k5_seed251 | sklearn auto | 23 | first | `/cases/7/variants/sklearn_auto/samples/23` |
| real_digits_raw_k5_seed251 | sklearn auto | 23 | second | `/cases/7/variants/sklearn_auto/samples/23` |
| real_digits_raw_k5_seed251 | sklearn auto | 24 | first | `/cases/7/variants/sklearn_auto/samples/24` |
| real_digits_raw_k5_seed251 | sklearn auto | 24 | second | `/cases/7/variants/sklearn_auto/samples/24` |
| real_digits_raw_k5_seed251 | sklearn auto | 25 | first | `/cases/7/variants/sklearn_auto/samples/25` |
| real_digits_raw_k5_seed251 | sklearn auto | 25 | second | `/cases/7/variants/sklearn_auto/samples/25` |
| real_digits_raw_k5_seed251 | sklearn auto | 26 | first | `/cases/7/variants/sklearn_auto/samples/26` |
| real_digits_raw_k5_seed251 | sklearn auto | 26 | second | `/cases/7/variants/sklearn_auto/samples/26` |
| real_digits_raw_k5_seed251 | sklearn auto | 27 | first | `/cases/7/variants/sklearn_auto/samples/27` |
| real_digits_raw_k5_seed251 | sklearn auto | 27 | second | `/cases/7/variants/sklearn_auto/samples/27` |
| real_digits_raw_k5_seed251 | sklearn auto | 28 | first | `/cases/7/variants/sklearn_auto/samples/28` |
| real_digits_raw_k5_seed251 | sklearn auto | 28 | second | `/cases/7/variants/sklearn_auto/samples/28` |
| real_digits_raw_k5_seed251 | sklearn auto | 29 | first | `/cases/7/variants/sklearn_auto/samples/29` |
| real_digits_raw_k5_seed251 | sklearn auto | 29 | second | `/cases/7/variants/sklearn_auto/samples/29` |
| real_digits_raw_k5_seed252 | Default-thread brute | 0 | first | `/cases/8/variants/sklearn_default/samples/0` |
| real_digits_raw_k5_seed252 | Default-thread brute | 0 | second | `/cases/8/variants/sklearn_default/samples/0` |
| real_digits_raw_k5_seed252 | Default-thread brute | 1 | first | `/cases/8/variants/sklearn_default/samples/1` |
| real_digits_raw_k5_seed252 | Default-thread brute | 1 | second | `/cases/8/variants/sklearn_default/samples/1` |
| real_digits_raw_k5_seed252 | Default-thread brute | 2 | first | `/cases/8/variants/sklearn_default/samples/2` |
| real_digits_raw_k5_seed252 | Default-thread brute | 2 | second | `/cases/8/variants/sklearn_default/samples/2` |
| real_digits_raw_k5_seed252 | Default-thread brute | 3 | first | `/cases/8/variants/sklearn_default/samples/3` |
| real_digits_raw_k5_seed252 | Default-thread brute | 3 | second | `/cases/8/variants/sklearn_default/samples/3` |
| real_digits_raw_k5_seed252 | Default-thread brute | 4 | first | `/cases/8/variants/sklearn_default/samples/4` |
| real_digits_raw_k5_seed252 | Default-thread brute | 4 | second | `/cases/8/variants/sklearn_default/samples/4` |
| real_digits_raw_k5_seed252 | Default-thread brute | 5 | first | `/cases/8/variants/sklearn_default/samples/5` |
| real_digits_raw_k5_seed252 | Default-thread brute | 5 | second | `/cases/8/variants/sklearn_default/samples/5` |
| real_digits_raw_k5_seed252 | Default-thread brute | 6 | first | `/cases/8/variants/sklearn_default/samples/6` |
| real_digits_raw_k5_seed252 | Default-thread brute | 6 | second | `/cases/8/variants/sklearn_default/samples/6` |
| real_digits_raw_k5_seed252 | Default-thread brute | 7 | first | `/cases/8/variants/sklearn_default/samples/7` |
| real_digits_raw_k5_seed252 | Default-thread brute | 7 | second | `/cases/8/variants/sklearn_default/samples/7` |
| real_digits_raw_k5_seed252 | Default-thread brute | 8 | first | `/cases/8/variants/sklearn_default/samples/8` |
| real_digits_raw_k5_seed252 | Default-thread brute | 8 | second | `/cases/8/variants/sklearn_default/samples/8` |
| real_digits_raw_k5_seed252 | Default-thread brute | 9 | first | `/cases/8/variants/sklearn_default/samples/9` |
| real_digits_raw_k5_seed252 | Default-thread brute | 9 | second | `/cases/8/variants/sklearn_default/samples/9` |
| real_digits_raw_k5_seed252 | Default-thread brute | 10 | first | `/cases/8/variants/sklearn_default/samples/10` |
| real_digits_raw_k5_seed252 | Default-thread brute | 10 | second | `/cases/8/variants/sklearn_default/samples/10` |
| real_digits_raw_k5_seed252 | Default-thread brute | 11 | first | `/cases/8/variants/sklearn_default/samples/11` |
| real_digits_raw_k5_seed252 | Default-thread brute | 11 | second | `/cases/8/variants/sklearn_default/samples/11` |
| real_digits_raw_k5_seed252 | Default-thread brute | 12 | first | `/cases/8/variants/sklearn_default/samples/12` |
| real_digits_raw_k5_seed252 | Default-thread brute | 12 | second | `/cases/8/variants/sklearn_default/samples/12` |
| real_digits_raw_k5_seed252 | Default-thread brute | 13 | first | `/cases/8/variants/sklearn_default/samples/13` |
| real_digits_raw_k5_seed252 | Default-thread brute | 13 | second | `/cases/8/variants/sklearn_default/samples/13` |
| real_digits_raw_k5_seed252 | Default-thread brute | 14 | first | `/cases/8/variants/sklearn_default/samples/14` |
| real_digits_raw_k5_seed252 | Default-thread brute | 14 | second | `/cases/8/variants/sklearn_default/samples/14` |
| real_digits_raw_k5_seed252 | Default-thread brute | 15 | first | `/cases/8/variants/sklearn_default/samples/15` |
| real_digits_raw_k5_seed252 | Default-thread brute | 15 | second | `/cases/8/variants/sklearn_default/samples/15` |
| real_digits_raw_k5_seed252 | Default-thread brute | 16 | first | `/cases/8/variants/sklearn_default/samples/16` |
| real_digits_raw_k5_seed252 | Default-thread brute | 16 | second | `/cases/8/variants/sklearn_default/samples/16` |
| real_digits_raw_k5_seed252 | Default-thread brute | 17 | first | `/cases/8/variants/sklearn_default/samples/17` |
| real_digits_raw_k5_seed252 | Default-thread brute | 17 | second | `/cases/8/variants/sklearn_default/samples/17` |
| real_digits_raw_k5_seed252 | Default-thread brute | 18 | first | `/cases/8/variants/sklearn_default/samples/18` |
| real_digits_raw_k5_seed252 | Default-thread brute | 18 | second | `/cases/8/variants/sklearn_default/samples/18` |
| real_digits_raw_k5_seed252 | Default-thread brute | 19 | first | `/cases/8/variants/sklearn_default/samples/19` |
| real_digits_raw_k5_seed252 | Default-thread brute | 19 | second | `/cases/8/variants/sklearn_default/samples/19` |
| real_digits_raw_k5_seed252 | Default-thread brute | 20 | first | `/cases/8/variants/sklearn_default/samples/20` |
| real_digits_raw_k5_seed252 | Default-thread brute | 20 | second | `/cases/8/variants/sklearn_default/samples/20` |
| real_digits_raw_k5_seed252 | Default-thread brute | 21 | first | `/cases/8/variants/sklearn_default/samples/21` |
| real_digits_raw_k5_seed252 | Default-thread brute | 21 | second | `/cases/8/variants/sklearn_default/samples/21` |
| real_digits_raw_k5_seed252 | Default-thread brute | 22 | first | `/cases/8/variants/sklearn_default/samples/22` |
| real_digits_raw_k5_seed252 | Default-thread brute | 22 | second | `/cases/8/variants/sklearn_default/samples/22` |
| real_digits_raw_k5_seed252 | Default-thread brute | 23 | first | `/cases/8/variants/sklearn_default/samples/23` |
| real_digits_raw_k5_seed252 | Default-thread brute | 23 | second | `/cases/8/variants/sklearn_default/samples/23` |
| real_digits_raw_k5_seed252 | Default-thread brute | 24 | first | `/cases/8/variants/sklearn_default/samples/24` |
| real_digits_raw_k5_seed252 | Default-thread brute | 24 | second | `/cases/8/variants/sklearn_default/samples/24` |
| real_digits_raw_k5_seed252 | Default-thread brute | 25 | first | `/cases/8/variants/sklearn_default/samples/25` |
| real_digits_raw_k5_seed252 | Default-thread brute | 25 | second | `/cases/8/variants/sklearn_default/samples/25` |
| real_digits_raw_k5_seed252 | Default-thread brute | 26 | first | `/cases/8/variants/sklearn_default/samples/26` |
| real_digits_raw_k5_seed252 | Default-thread brute | 26 | second | `/cases/8/variants/sklearn_default/samples/26` |
| real_digits_raw_k5_seed252 | Default-thread brute | 27 | first | `/cases/8/variants/sklearn_default/samples/27` |
| real_digits_raw_k5_seed252 | Default-thread brute | 27 | second | `/cases/8/variants/sklearn_default/samples/27` |
| real_digits_raw_k5_seed252 | Default-thread brute | 28 | first | `/cases/8/variants/sklearn_default/samples/28` |
| real_digits_raw_k5_seed252 | Default-thread brute | 28 | second | `/cases/8/variants/sklearn_default/samples/28` |
| real_digits_raw_k5_seed252 | Default-thread brute | 29 | first | `/cases/8/variants/sklearn_default/samples/29` |
| real_digits_raw_k5_seed252 | Default-thread brute | 29 | second | `/cases/8/variants/sklearn_default/samples/29` |
| real_digits_raw_k5_seed252 | Single-thread brute | 0 | first | `/cases/8/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed252 | Single-thread brute | 0 | second | `/cases/8/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed252 | Single-thread brute | 1 | first | `/cases/8/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed252 | Single-thread brute | 1 | second | `/cases/8/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed252 | Single-thread brute | 2 | first | `/cases/8/variants/sklearn_1/samples/2` |
| real_digits_raw_k5_seed252 | Single-thread brute | 2 | second | `/cases/8/variants/sklearn_1/samples/2` |
| real_digits_raw_k5_seed252 | Single-thread brute | 3 | first | `/cases/8/variants/sklearn_1/samples/3` |
| real_digits_raw_k5_seed252 | Single-thread brute | 3 | second | `/cases/8/variants/sklearn_1/samples/3` |
| real_digits_raw_k5_seed252 | Single-thread brute | 4 | first | `/cases/8/variants/sklearn_1/samples/4` |
| real_digits_raw_k5_seed252 | Single-thread brute | 4 | second | `/cases/8/variants/sklearn_1/samples/4` |
| real_digits_raw_k5_seed252 | Single-thread brute | 5 | first | `/cases/8/variants/sklearn_1/samples/5` |
| real_digits_raw_k5_seed252 | Single-thread brute | 5 | second | `/cases/8/variants/sklearn_1/samples/5` |
| real_digits_raw_k5_seed252 | Single-thread brute | 6 | first | `/cases/8/variants/sklearn_1/samples/6` |
| real_digits_raw_k5_seed252 | Single-thread brute | 6 | second | `/cases/8/variants/sklearn_1/samples/6` |
| real_digits_raw_k5_seed252 | Single-thread brute | 7 | first | `/cases/8/variants/sklearn_1/samples/7` |
| real_digits_raw_k5_seed252 | Single-thread brute | 7 | second | `/cases/8/variants/sklearn_1/samples/7` |
| real_digits_raw_k5_seed252 | Single-thread brute | 8 | first | `/cases/8/variants/sklearn_1/samples/8` |
| real_digits_raw_k5_seed252 | Single-thread brute | 8 | second | `/cases/8/variants/sklearn_1/samples/8` |
| real_digits_raw_k5_seed252 | Single-thread brute | 9 | first | `/cases/8/variants/sklearn_1/samples/9` |
| real_digits_raw_k5_seed252 | Single-thread brute | 9 | second | `/cases/8/variants/sklearn_1/samples/9` |
| real_digits_raw_k5_seed252 | Single-thread brute | 10 | first | `/cases/8/variants/sklearn_1/samples/10` |
| real_digits_raw_k5_seed252 | Single-thread brute | 10 | second | `/cases/8/variants/sklearn_1/samples/10` |
| real_digits_raw_k5_seed252 | Single-thread brute | 11 | first | `/cases/8/variants/sklearn_1/samples/11` |
| real_digits_raw_k5_seed252 | Single-thread brute | 11 | second | `/cases/8/variants/sklearn_1/samples/11` |
| real_digits_raw_k5_seed252 | Single-thread brute | 12 | first | `/cases/8/variants/sklearn_1/samples/12` |
| real_digits_raw_k5_seed252 | Single-thread brute | 12 | second | `/cases/8/variants/sklearn_1/samples/12` |
| real_digits_raw_k5_seed252 | Single-thread brute | 13 | first | `/cases/8/variants/sklearn_1/samples/13` |
| real_digits_raw_k5_seed252 | Single-thread brute | 13 | second | `/cases/8/variants/sklearn_1/samples/13` |
| real_digits_raw_k5_seed252 | Single-thread brute | 14 | first | `/cases/8/variants/sklearn_1/samples/14` |
| real_digits_raw_k5_seed252 | Single-thread brute | 14 | second | `/cases/8/variants/sklearn_1/samples/14` |
| real_digits_raw_k5_seed252 | Single-thread brute | 15 | first | `/cases/8/variants/sklearn_1/samples/15` |
| real_digits_raw_k5_seed252 | Single-thread brute | 15 | second | `/cases/8/variants/sklearn_1/samples/15` |
| real_digits_raw_k5_seed252 | Single-thread brute | 16 | first | `/cases/8/variants/sklearn_1/samples/16` |
| real_digits_raw_k5_seed252 | Single-thread brute | 16 | second | `/cases/8/variants/sklearn_1/samples/16` |
| real_digits_raw_k5_seed252 | Single-thread brute | 17 | first | `/cases/8/variants/sklearn_1/samples/17` |
| real_digits_raw_k5_seed252 | Single-thread brute | 17 | second | `/cases/8/variants/sklearn_1/samples/17` |
| real_digits_raw_k5_seed252 | Single-thread brute | 18 | first | `/cases/8/variants/sklearn_1/samples/18` |
| real_digits_raw_k5_seed252 | Single-thread brute | 18 | second | `/cases/8/variants/sklearn_1/samples/18` |
| real_digits_raw_k5_seed252 | Single-thread brute | 19 | first | `/cases/8/variants/sklearn_1/samples/19` |
| real_digits_raw_k5_seed252 | Single-thread brute | 19 | second | `/cases/8/variants/sklearn_1/samples/19` |
| real_digits_raw_k5_seed252 | Single-thread brute | 20 | first | `/cases/8/variants/sklearn_1/samples/20` |
| real_digits_raw_k5_seed252 | Single-thread brute | 20 | second | `/cases/8/variants/sklearn_1/samples/20` |
| real_digits_raw_k5_seed252 | Single-thread brute | 21 | first | `/cases/8/variants/sklearn_1/samples/21` |
| real_digits_raw_k5_seed252 | Single-thread brute | 21 | second | `/cases/8/variants/sklearn_1/samples/21` |
| real_digits_raw_k5_seed252 | Single-thread brute | 22 | first | `/cases/8/variants/sklearn_1/samples/22` |
| real_digits_raw_k5_seed252 | Single-thread brute | 22 | second | `/cases/8/variants/sklearn_1/samples/22` |
| real_digits_raw_k5_seed252 | Single-thread brute | 23 | first | `/cases/8/variants/sklearn_1/samples/23` |
| real_digits_raw_k5_seed252 | Single-thread brute | 23 | second | `/cases/8/variants/sklearn_1/samples/23` |
| real_digits_raw_k5_seed252 | Single-thread brute | 24 | first | `/cases/8/variants/sklearn_1/samples/24` |
| real_digits_raw_k5_seed252 | Single-thread brute | 24 | second | `/cases/8/variants/sklearn_1/samples/24` |
| real_digits_raw_k5_seed252 | Single-thread brute | 25 | first | `/cases/8/variants/sklearn_1/samples/25` |
| real_digits_raw_k5_seed252 | Single-thread brute | 25 | second | `/cases/8/variants/sklearn_1/samples/25` |
| real_digits_raw_k5_seed252 | Single-thread brute | 26 | first | `/cases/8/variants/sklearn_1/samples/26` |
| real_digits_raw_k5_seed252 | Single-thread brute | 26 | second | `/cases/8/variants/sklearn_1/samples/26` |
| real_digits_raw_k5_seed252 | Single-thread brute | 27 | first | `/cases/8/variants/sklearn_1/samples/27` |
| real_digits_raw_k5_seed252 | Single-thread brute | 27 | second | `/cases/8/variants/sklearn_1/samples/27` |
| real_digits_raw_k5_seed252 | Single-thread brute | 28 | first | `/cases/8/variants/sklearn_1/samples/28` |
| real_digits_raw_k5_seed252 | Single-thread brute | 28 | second | `/cases/8/variants/sklearn_1/samples/28` |
| real_digits_raw_k5_seed252 | Single-thread brute | 29 | first | `/cases/8/variants/sklearn_1/samples/29` |
| real_digits_raw_k5_seed252 | Single-thread brute | 29 | second | `/cases/8/variants/sklearn_1/samples/29` |
| real_digits_raw_k5_seed252 | sklearn auto | 0 | first | `/cases/8/variants/sklearn_auto/samples/0` |
| real_digits_raw_k5_seed252 | sklearn auto | 0 | second | `/cases/8/variants/sklearn_auto/samples/0` |
| real_digits_raw_k5_seed252 | sklearn auto | 1 | first | `/cases/8/variants/sklearn_auto/samples/1` |
| real_digits_raw_k5_seed252 | sklearn auto | 1 | second | `/cases/8/variants/sklearn_auto/samples/1` |
| real_digits_raw_k5_seed252 | sklearn auto | 2 | first | `/cases/8/variants/sklearn_auto/samples/2` |
| real_digits_raw_k5_seed252 | sklearn auto | 2 | second | `/cases/8/variants/sklearn_auto/samples/2` |
| real_digits_raw_k5_seed252 | sklearn auto | 3 | first | `/cases/8/variants/sklearn_auto/samples/3` |
| real_digits_raw_k5_seed252 | sklearn auto | 3 | second | `/cases/8/variants/sklearn_auto/samples/3` |
| real_digits_raw_k5_seed252 | sklearn auto | 4 | first | `/cases/8/variants/sklearn_auto/samples/4` |
| real_digits_raw_k5_seed252 | sklearn auto | 4 | second | `/cases/8/variants/sklearn_auto/samples/4` |
| real_digits_raw_k5_seed252 | sklearn auto | 5 | first | `/cases/8/variants/sklearn_auto/samples/5` |
| real_digits_raw_k5_seed252 | sklearn auto | 5 | second | `/cases/8/variants/sklearn_auto/samples/5` |
| real_digits_raw_k5_seed252 | sklearn auto | 6 | first | `/cases/8/variants/sklearn_auto/samples/6` |
| real_digits_raw_k5_seed252 | sklearn auto | 6 | second | `/cases/8/variants/sklearn_auto/samples/6` |
| real_digits_raw_k5_seed252 | sklearn auto | 7 | first | `/cases/8/variants/sklearn_auto/samples/7` |
| real_digits_raw_k5_seed252 | sklearn auto | 7 | second | `/cases/8/variants/sklearn_auto/samples/7` |
| real_digits_raw_k5_seed252 | sklearn auto | 8 | first | `/cases/8/variants/sklearn_auto/samples/8` |
| real_digits_raw_k5_seed252 | sklearn auto | 8 | second | `/cases/8/variants/sklearn_auto/samples/8` |
| real_digits_raw_k5_seed252 | sklearn auto | 9 | first | `/cases/8/variants/sklearn_auto/samples/9` |
| real_digits_raw_k5_seed252 | sklearn auto | 9 | second | `/cases/8/variants/sklearn_auto/samples/9` |
| real_digits_raw_k5_seed252 | sklearn auto | 10 | first | `/cases/8/variants/sklearn_auto/samples/10` |
| real_digits_raw_k5_seed252 | sklearn auto | 10 | second | `/cases/8/variants/sklearn_auto/samples/10` |
| real_digits_raw_k5_seed252 | sklearn auto | 11 | first | `/cases/8/variants/sklearn_auto/samples/11` |
| real_digits_raw_k5_seed252 | sklearn auto | 11 | second | `/cases/8/variants/sklearn_auto/samples/11` |
| real_digits_raw_k5_seed252 | sklearn auto | 12 | first | `/cases/8/variants/sklearn_auto/samples/12` |
| real_digits_raw_k5_seed252 | sklearn auto | 12 | second | `/cases/8/variants/sklearn_auto/samples/12` |
| real_digits_raw_k5_seed252 | sklearn auto | 13 | first | `/cases/8/variants/sklearn_auto/samples/13` |
| real_digits_raw_k5_seed252 | sklearn auto | 13 | second | `/cases/8/variants/sklearn_auto/samples/13` |
| real_digits_raw_k5_seed252 | sklearn auto | 14 | first | `/cases/8/variants/sklearn_auto/samples/14` |
| real_digits_raw_k5_seed252 | sklearn auto | 14 | second | `/cases/8/variants/sklearn_auto/samples/14` |
| real_digits_raw_k5_seed252 | sklearn auto | 15 | first | `/cases/8/variants/sklearn_auto/samples/15` |
| real_digits_raw_k5_seed252 | sklearn auto | 15 | second | `/cases/8/variants/sklearn_auto/samples/15` |
| real_digits_raw_k5_seed252 | sklearn auto | 16 | first | `/cases/8/variants/sklearn_auto/samples/16` |
| real_digits_raw_k5_seed252 | sklearn auto | 16 | second | `/cases/8/variants/sklearn_auto/samples/16` |
| real_digits_raw_k5_seed252 | sklearn auto | 17 | first | `/cases/8/variants/sklearn_auto/samples/17` |
| real_digits_raw_k5_seed252 | sklearn auto | 17 | second | `/cases/8/variants/sklearn_auto/samples/17` |
| real_digits_raw_k5_seed252 | sklearn auto | 18 | first | `/cases/8/variants/sklearn_auto/samples/18` |
| real_digits_raw_k5_seed252 | sklearn auto | 18 | second | `/cases/8/variants/sklearn_auto/samples/18` |
| real_digits_raw_k5_seed252 | sklearn auto | 19 | first | `/cases/8/variants/sklearn_auto/samples/19` |
| real_digits_raw_k5_seed252 | sklearn auto | 19 | second | `/cases/8/variants/sklearn_auto/samples/19` |
| real_digits_raw_k5_seed252 | sklearn auto | 20 | first | `/cases/8/variants/sklearn_auto/samples/20` |
| real_digits_raw_k5_seed252 | sklearn auto | 20 | second | `/cases/8/variants/sklearn_auto/samples/20` |
| real_digits_raw_k5_seed252 | sklearn auto | 21 | first | `/cases/8/variants/sklearn_auto/samples/21` |
| real_digits_raw_k5_seed252 | sklearn auto | 21 | second | `/cases/8/variants/sklearn_auto/samples/21` |
| real_digits_raw_k5_seed252 | sklearn auto | 22 | first | `/cases/8/variants/sklearn_auto/samples/22` |
| real_digits_raw_k5_seed252 | sklearn auto | 22 | second | `/cases/8/variants/sklearn_auto/samples/22` |
| real_digits_raw_k5_seed252 | sklearn auto | 23 | first | `/cases/8/variants/sklearn_auto/samples/23` |
| real_digits_raw_k5_seed252 | sklearn auto | 23 | second | `/cases/8/variants/sklearn_auto/samples/23` |
| real_digits_raw_k5_seed252 | sklearn auto | 24 | first | `/cases/8/variants/sklearn_auto/samples/24` |
| real_digits_raw_k5_seed252 | sklearn auto | 24 | second | `/cases/8/variants/sklearn_auto/samples/24` |
| real_digits_raw_k5_seed252 | sklearn auto | 25 | first | `/cases/8/variants/sklearn_auto/samples/25` |
| real_digits_raw_k5_seed252 | sklearn auto | 25 | second | `/cases/8/variants/sklearn_auto/samples/25` |
| real_digits_raw_k5_seed252 | sklearn auto | 26 | first | `/cases/8/variants/sklearn_auto/samples/26` |
| real_digits_raw_k5_seed252 | sklearn auto | 26 | second | `/cases/8/variants/sklearn_auto/samples/26` |
| real_digits_raw_k5_seed252 | sklearn auto | 27 | first | `/cases/8/variants/sklearn_auto/samples/27` |
| real_digits_raw_k5_seed252 | sklearn auto | 27 | second | `/cases/8/variants/sklearn_auto/samples/27` |
| real_digits_raw_k5_seed252 | sklearn auto | 28 | first | `/cases/8/variants/sklearn_auto/samples/28` |
| real_digits_raw_k5_seed252 | sklearn auto | 28 | second | `/cases/8/variants/sklearn_auto/samples/28` |
| real_digits_raw_k5_seed252 | sklearn auto | 29 | first | `/cases/8/variants/sklearn_auto/samples/29` |
| real_digits_raw_k5_seed252 | sklearn auto | 29 | second | `/cases/8/variants/sklearn_auto/samples/29` |
| real_iris_raw_k5_seed151 | Default-thread brute | 0 | first | `/cases/12/variants/sklearn_default/samples/0` |
| real_iris_raw_k5_seed151 | Default-thread brute | 0 | second | `/cases/12/variants/sklearn_default/samples/0` |
| real_iris_raw_k5_seed151 | Default-thread brute | 1 | first | `/cases/12/variants/sklearn_default/samples/1` |
| real_iris_raw_k5_seed151 | Default-thread brute | 1 | second | `/cases/12/variants/sklearn_default/samples/1` |
| real_iris_raw_k5_seed151 | Default-thread brute | 2 | first | `/cases/12/variants/sklearn_default/samples/2` |
| real_iris_raw_k5_seed151 | Default-thread brute | 2 | second | `/cases/12/variants/sklearn_default/samples/2` |
| real_iris_raw_k5_seed151 | Default-thread brute | 3 | first | `/cases/12/variants/sklearn_default/samples/3` |
| real_iris_raw_k5_seed151 | Default-thread brute | 3 | second | `/cases/12/variants/sklearn_default/samples/3` |
| real_iris_raw_k5_seed151 | Default-thread brute | 4 | first | `/cases/12/variants/sklearn_default/samples/4` |
| real_iris_raw_k5_seed151 | Default-thread brute | 4 | second | `/cases/12/variants/sklearn_default/samples/4` |
| real_iris_raw_k5_seed151 | Default-thread brute | 5 | first | `/cases/12/variants/sklearn_default/samples/5` |
| real_iris_raw_k5_seed151 | Default-thread brute | 5 | second | `/cases/12/variants/sklearn_default/samples/5` |
| real_iris_raw_k5_seed151 | Default-thread brute | 6 | first | `/cases/12/variants/sklearn_default/samples/6` |
| real_iris_raw_k5_seed151 | Default-thread brute | 6 | second | `/cases/12/variants/sklearn_default/samples/6` |
| real_iris_raw_k5_seed151 | Default-thread brute | 7 | first | `/cases/12/variants/sklearn_default/samples/7` |
| real_iris_raw_k5_seed151 | Default-thread brute | 7 | second | `/cases/12/variants/sklearn_default/samples/7` |
| real_iris_raw_k5_seed151 | Default-thread brute | 8 | first | `/cases/12/variants/sklearn_default/samples/8` |
| real_iris_raw_k5_seed151 | Default-thread brute | 8 | second | `/cases/12/variants/sklearn_default/samples/8` |
| real_iris_raw_k5_seed151 | Default-thread brute | 9 | first | `/cases/12/variants/sklearn_default/samples/9` |
| real_iris_raw_k5_seed151 | Default-thread brute | 9 | second | `/cases/12/variants/sklearn_default/samples/9` |
| real_iris_raw_k5_seed151 | Default-thread brute | 10 | first | `/cases/12/variants/sklearn_default/samples/10` |
| real_iris_raw_k5_seed151 | Default-thread brute | 10 | second | `/cases/12/variants/sklearn_default/samples/10` |
| real_iris_raw_k5_seed151 | Default-thread brute | 11 | first | `/cases/12/variants/sklearn_default/samples/11` |
| real_iris_raw_k5_seed151 | Default-thread brute | 11 | second | `/cases/12/variants/sklearn_default/samples/11` |
| real_iris_raw_k5_seed151 | Default-thread brute | 12 | first | `/cases/12/variants/sklearn_default/samples/12` |
| real_iris_raw_k5_seed151 | Default-thread brute | 12 | second | `/cases/12/variants/sklearn_default/samples/12` |
| real_iris_raw_k5_seed151 | Default-thread brute | 13 | first | `/cases/12/variants/sklearn_default/samples/13` |
| real_iris_raw_k5_seed151 | Default-thread brute | 13 | second | `/cases/12/variants/sklearn_default/samples/13` |
| real_iris_raw_k5_seed151 | Default-thread brute | 14 | first | `/cases/12/variants/sklearn_default/samples/14` |
| real_iris_raw_k5_seed151 | Default-thread brute | 14 | second | `/cases/12/variants/sklearn_default/samples/14` |
| real_iris_raw_k5_seed151 | Default-thread brute | 15 | first | `/cases/12/variants/sklearn_default/samples/15` |
| real_iris_raw_k5_seed151 | Default-thread brute | 15 | second | `/cases/12/variants/sklearn_default/samples/15` |
| real_iris_raw_k5_seed151 | Default-thread brute | 16 | first | `/cases/12/variants/sklearn_default/samples/16` |
| real_iris_raw_k5_seed151 | Default-thread brute | 16 | second | `/cases/12/variants/sklearn_default/samples/16` |
| real_iris_raw_k5_seed151 | Default-thread brute | 17 | first | `/cases/12/variants/sklearn_default/samples/17` |
| real_iris_raw_k5_seed151 | Default-thread brute | 17 | second | `/cases/12/variants/sklearn_default/samples/17` |
| real_iris_raw_k5_seed151 | Default-thread brute | 18 | first | `/cases/12/variants/sklearn_default/samples/18` |
| real_iris_raw_k5_seed151 | Default-thread brute | 18 | second | `/cases/12/variants/sklearn_default/samples/18` |
| real_iris_raw_k5_seed151 | Default-thread brute | 19 | first | `/cases/12/variants/sklearn_default/samples/19` |
| real_iris_raw_k5_seed151 | Default-thread brute | 19 | second | `/cases/12/variants/sklearn_default/samples/19` |
| real_iris_raw_k5_seed151 | Default-thread brute | 20 | first | `/cases/12/variants/sklearn_default/samples/20` |
| real_iris_raw_k5_seed151 | Default-thread brute | 20 | second | `/cases/12/variants/sklearn_default/samples/20` |
| real_iris_raw_k5_seed151 | Default-thread brute | 21 | first | `/cases/12/variants/sklearn_default/samples/21` |
| real_iris_raw_k5_seed151 | Default-thread brute | 21 | second | `/cases/12/variants/sklearn_default/samples/21` |
| real_iris_raw_k5_seed151 | Default-thread brute | 22 | first | `/cases/12/variants/sklearn_default/samples/22` |
| real_iris_raw_k5_seed151 | Default-thread brute | 22 | second | `/cases/12/variants/sklearn_default/samples/22` |
| real_iris_raw_k5_seed151 | Default-thread brute | 23 | first | `/cases/12/variants/sklearn_default/samples/23` |
| real_iris_raw_k5_seed151 | Default-thread brute | 23 | second | `/cases/12/variants/sklearn_default/samples/23` |
| real_iris_raw_k5_seed151 | Default-thread brute | 24 | first | `/cases/12/variants/sklearn_default/samples/24` |
| real_iris_raw_k5_seed151 | Default-thread brute | 24 | second | `/cases/12/variants/sklearn_default/samples/24` |
| real_iris_raw_k5_seed151 | Default-thread brute | 25 | first | `/cases/12/variants/sklearn_default/samples/25` |
| real_iris_raw_k5_seed151 | Default-thread brute | 25 | second | `/cases/12/variants/sklearn_default/samples/25` |
| real_iris_raw_k5_seed151 | Default-thread brute | 26 | first | `/cases/12/variants/sklearn_default/samples/26` |
| real_iris_raw_k5_seed151 | Default-thread brute | 26 | second | `/cases/12/variants/sklearn_default/samples/26` |
| real_iris_raw_k5_seed151 | Default-thread brute | 27 | first | `/cases/12/variants/sklearn_default/samples/27` |
| real_iris_raw_k5_seed151 | Default-thread brute | 27 | second | `/cases/12/variants/sklearn_default/samples/27` |
| real_iris_raw_k5_seed151 | Default-thread brute | 28 | first | `/cases/12/variants/sklearn_default/samples/28` |
| real_iris_raw_k5_seed151 | Default-thread brute | 28 | second | `/cases/12/variants/sklearn_default/samples/28` |
| real_iris_raw_k5_seed151 | Default-thread brute | 29 | first | `/cases/12/variants/sklearn_default/samples/29` |
| real_iris_raw_k5_seed151 | Default-thread brute | 29 | second | `/cases/12/variants/sklearn_default/samples/29` |
| real_iris_raw_k5_seed151 | Single-thread brute | 0 | first | `/cases/12/variants/sklearn_1/samples/0` |
| real_iris_raw_k5_seed151 | Single-thread brute | 0 | second | `/cases/12/variants/sklearn_1/samples/0` |
| real_iris_raw_k5_seed151 | Single-thread brute | 1 | first | `/cases/12/variants/sklearn_1/samples/1` |
| real_iris_raw_k5_seed151 | Single-thread brute | 1 | second | `/cases/12/variants/sklearn_1/samples/1` |
| real_iris_raw_k5_seed151 | Single-thread brute | 2 | first | `/cases/12/variants/sklearn_1/samples/2` |
| real_iris_raw_k5_seed151 | Single-thread brute | 2 | second | `/cases/12/variants/sklearn_1/samples/2` |
| real_iris_raw_k5_seed151 | Single-thread brute | 3 | first | `/cases/12/variants/sklearn_1/samples/3` |
| real_iris_raw_k5_seed151 | Single-thread brute | 3 | second | `/cases/12/variants/sklearn_1/samples/3` |
| real_iris_raw_k5_seed151 | Single-thread brute | 4 | first | `/cases/12/variants/sklearn_1/samples/4` |
| real_iris_raw_k5_seed151 | Single-thread brute | 4 | second | `/cases/12/variants/sklearn_1/samples/4` |
| real_iris_raw_k5_seed151 | Single-thread brute | 5 | first | `/cases/12/variants/sklearn_1/samples/5` |
| real_iris_raw_k5_seed151 | Single-thread brute | 5 | second | `/cases/12/variants/sklearn_1/samples/5` |
| real_iris_raw_k5_seed151 | Single-thread brute | 6 | first | `/cases/12/variants/sklearn_1/samples/6` |
| real_iris_raw_k5_seed151 | Single-thread brute | 6 | second | `/cases/12/variants/sklearn_1/samples/6` |
| real_iris_raw_k5_seed151 | Single-thread brute | 7 | first | `/cases/12/variants/sklearn_1/samples/7` |
| real_iris_raw_k5_seed151 | Single-thread brute | 7 | second | `/cases/12/variants/sklearn_1/samples/7` |
| real_iris_raw_k5_seed151 | Single-thread brute | 8 | first | `/cases/12/variants/sklearn_1/samples/8` |
| real_iris_raw_k5_seed151 | Single-thread brute | 8 | second | `/cases/12/variants/sklearn_1/samples/8` |
| real_iris_raw_k5_seed151 | Single-thread brute | 9 | first | `/cases/12/variants/sklearn_1/samples/9` |
| real_iris_raw_k5_seed151 | Single-thread brute | 9 | second | `/cases/12/variants/sklearn_1/samples/9` |
| real_iris_raw_k5_seed151 | Single-thread brute | 10 | first | `/cases/12/variants/sklearn_1/samples/10` |
| real_iris_raw_k5_seed151 | Single-thread brute | 10 | second | `/cases/12/variants/sklearn_1/samples/10` |
| real_iris_raw_k5_seed151 | Single-thread brute | 11 | first | `/cases/12/variants/sklearn_1/samples/11` |
| real_iris_raw_k5_seed151 | Single-thread brute | 11 | second | `/cases/12/variants/sklearn_1/samples/11` |
| real_iris_raw_k5_seed151 | Single-thread brute | 12 | first | `/cases/12/variants/sklearn_1/samples/12` |
| real_iris_raw_k5_seed151 | Single-thread brute | 12 | second | `/cases/12/variants/sklearn_1/samples/12` |
| real_iris_raw_k5_seed151 | Single-thread brute | 13 | first | `/cases/12/variants/sklearn_1/samples/13` |
| real_iris_raw_k5_seed151 | Single-thread brute | 13 | second | `/cases/12/variants/sklearn_1/samples/13` |
| real_iris_raw_k5_seed151 | Single-thread brute | 14 | first | `/cases/12/variants/sklearn_1/samples/14` |
| real_iris_raw_k5_seed151 | Single-thread brute | 14 | second | `/cases/12/variants/sklearn_1/samples/14` |
| real_iris_raw_k5_seed151 | Single-thread brute | 15 | first | `/cases/12/variants/sklearn_1/samples/15` |
| real_iris_raw_k5_seed151 | Single-thread brute | 15 | second | `/cases/12/variants/sklearn_1/samples/15` |
| real_iris_raw_k5_seed151 | Single-thread brute | 16 | first | `/cases/12/variants/sklearn_1/samples/16` |
| real_iris_raw_k5_seed151 | Single-thread brute | 16 | second | `/cases/12/variants/sklearn_1/samples/16` |
| real_iris_raw_k5_seed151 | Single-thread brute | 17 | first | `/cases/12/variants/sklearn_1/samples/17` |
| real_iris_raw_k5_seed151 | Single-thread brute | 17 | second | `/cases/12/variants/sklearn_1/samples/17` |
| real_iris_raw_k5_seed151 | Single-thread brute | 18 | first | `/cases/12/variants/sklearn_1/samples/18` |
| real_iris_raw_k5_seed151 | Single-thread brute | 18 | second | `/cases/12/variants/sklearn_1/samples/18` |
| real_iris_raw_k5_seed151 | Single-thread brute | 19 | first | `/cases/12/variants/sklearn_1/samples/19` |
| real_iris_raw_k5_seed151 | Single-thread brute | 19 | second | `/cases/12/variants/sklearn_1/samples/19` |
| real_iris_raw_k5_seed151 | Single-thread brute | 20 | first | `/cases/12/variants/sklearn_1/samples/20` |
| real_iris_raw_k5_seed151 | Single-thread brute | 20 | second | `/cases/12/variants/sklearn_1/samples/20` |
| real_iris_raw_k5_seed151 | Single-thread brute | 21 | first | `/cases/12/variants/sklearn_1/samples/21` |
| real_iris_raw_k5_seed151 | Single-thread brute | 21 | second | `/cases/12/variants/sklearn_1/samples/21` |
| real_iris_raw_k5_seed151 | Single-thread brute | 22 | first | `/cases/12/variants/sklearn_1/samples/22` |
| real_iris_raw_k5_seed151 | Single-thread brute | 22 | second | `/cases/12/variants/sklearn_1/samples/22` |
| real_iris_raw_k5_seed151 | Single-thread brute | 23 | first | `/cases/12/variants/sklearn_1/samples/23` |
| real_iris_raw_k5_seed151 | Single-thread brute | 23 | second | `/cases/12/variants/sklearn_1/samples/23` |
| real_iris_raw_k5_seed151 | Single-thread brute | 24 | first | `/cases/12/variants/sklearn_1/samples/24` |
| real_iris_raw_k5_seed151 | Single-thread brute | 24 | second | `/cases/12/variants/sklearn_1/samples/24` |
| real_iris_raw_k5_seed151 | Single-thread brute | 25 | first | `/cases/12/variants/sklearn_1/samples/25` |
| real_iris_raw_k5_seed151 | Single-thread brute | 25 | second | `/cases/12/variants/sklearn_1/samples/25` |
| real_iris_raw_k5_seed151 | Single-thread brute | 26 | first | `/cases/12/variants/sklearn_1/samples/26` |
| real_iris_raw_k5_seed151 | Single-thread brute | 26 | second | `/cases/12/variants/sklearn_1/samples/26` |
| real_iris_raw_k5_seed151 | Single-thread brute | 27 | first | `/cases/12/variants/sklearn_1/samples/27` |
| real_iris_raw_k5_seed151 | Single-thread brute | 27 | second | `/cases/12/variants/sklearn_1/samples/27` |
| real_iris_raw_k5_seed151 | Single-thread brute | 28 | first | `/cases/12/variants/sklearn_1/samples/28` |
| real_iris_raw_k5_seed151 | Single-thread brute | 28 | second | `/cases/12/variants/sklearn_1/samples/28` |
| real_iris_raw_k5_seed151 | Single-thread brute | 29 | first | `/cases/12/variants/sklearn_1/samples/29` |
| real_iris_raw_k5_seed151 | Single-thread brute | 29 | second | `/cases/12/variants/sklearn_1/samples/29` |
| real_iris_raw_k5_seed151 | sklearn auto | 0 | first | `/cases/12/variants/sklearn_auto/samples/0` |
| real_iris_raw_k5_seed151 | sklearn auto | 0 | second | `/cases/12/variants/sklearn_auto/samples/0` |
| real_iris_raw_k5_seed151 | sklearn auto | 1 | first | `/cases/12/variants/sklearn_auto/samples/1` |
| real_iris_raw_k5_seed151 | sklearn auto | 1 | second | `/cases/12/variants/sklearn_auto/samples/1` |
| real_iris_raw_k5_seed151 | sklearn auto | 2 | first | `/cases/12/variants/sklearn_auto/samples/2` |
| real_iris_raw_k5_seed151 | sklearn auto | 2 | second | `/cases/12/variants/sklearn_auto/samples/2` |
| real_iris_raw_k5_seed151 | sklearn auto | 3 | first | `/cases/12/variants/sklearn_auto/samples/3` |
| real_iris_raw_k5_seed151 | sklearn auto | 3 | second | `/cases/12/variants/sklearn_auto/samples/3` |
| real_iris_raw_k5_seed151 | sklearn auto | 4 | first | `/cases/12/variants/sklearn_auto/samples/4` |
| real_iris_raw_k5_seed151 | sklearn auto | 4 | second | `/cases/12/variants/sklearn_auto/samples/4` |
| real_iris_raw_k5_seed151 | sklearn auto | 5 | first | `/cases/12/variants/sklearn_auto/samples/5` |
| real_iris_raw_k5_seed151 | sklearn auto | 5 | second | `/cases/12/variants/sklearn_auto/samples/5` |
| real_iris_raw_k5_seed151 | sklearn auto | 6 | first | `/cases/12/variants/sklearn_auto/samples/6` |
| real_iris_raw_k5_seed151 | sklearn auto | 6 | second | `/cases/12/variants/sklearn_auto/samples/6` |
| real_iris_raw_k5_seed151 | sklearn auto | 7 | first | `/cases/12/variants/sklearn_auto/samples/7` |
| real_iris_raw_k5_seed151 | sklearn auto | 7 | second | `/cases/12/variants/sklearn_auto/samples/7` |
| real_iris_raw_k5_seed151 | sklearn auto | 8 | first | `/cases/12/variants/sklearn_auto/samples/8` |
| real_iris_raw_k5_seed151 | sklearn auto | 8 | second | `/cases/12/variants/sklearn_auto/samples/8` |
| real_iris_raw_k5_seed151 | sklearn auto | 9 | first | `/cases/12/variants/sklearn_auto/samples/9` |
| real_iris_raw_k5_seed151 | sklearn auto | 9 | second | `/cases/12/variants/sklearn_auto/samples/9` |
| real_iris_raw_k5_seed151 | sklearn auto | 10 | first | `/cases/12/variants/sklearn_auto/samples/10` |
| real_iris_raw_k5_seed151 | sklearn auto | 10 | second | `/cases/12/variants/sklearn_auto/samples/10` |
| real_iris_raw_k5_seed151 | sklearn auto | 11 | first | `/cases/12/variants/sklearn_auto/samples/11` |
| real_iris_raw_k5_seed151 | sklearn auto | 11 | second | `/cases/12/variants/sklearn_auto/samples/11` |
| real_iris_raw_k5_seed151 | sklearn auto | 12 | first | `/cases/12/variants/sklearn_auto/samples/12` |
| real_iris_raw_k5_seed151 | sklearn auto | 12 | second | `/cases/12/variants/sklearn_auto/samples/12` |
| real_iris_raw_k5_seed151 | sklearn auto | 13 | first | `/cases/12/variants/sklearn_auto/samples/13` |
| real_iris_raw_k5_seed151 | sklearn auto | 13 | second | `/cases/12/variants/sklearn_auto/samples/13` |
| real_iris_raw_k5_seed151 | sklearn auto | 14 | first | `/cases/12/variants/sklearn_auto/samples/14` |
| real_iris_raw_k5_seed151 | sklearn auto | 14 | second | `/cases/12/variants/sklearn_auto/samples/14` |
| real_iris_raw_k5_seed151 | sklearn auto | 15 | first | `/cases/12/variants/sklearn_auto/samples/15` |
| real_iris_raw_k5_seed151 | sklearn auto | 15 | second | `/cases/12/variants/sklearn_auto/samples/15` |
| real_iris_raw_k5_seed151 | sklearn auto | 16 | first | `/cases/12/variants/sklearn_auto/samples/16` |
| real_iris_raw_k5_seed151 | sklearn auto | 16 | second | `/cases/12/variants/sklearn_auto/samples/16` |
| real_iris_raw_k5_seed151 | sklearn auto | 17 | first | `/cases/12/variants/sklearn_auto/samples/17` |
| real_iris_raw_k5_seed151 | sklearn auto | 17 | second | `/cases/12/variants/sklearn_auto/samples/17` |
| real_iris_raw_k5_seed151 | sklearn auto | 18 | first | `/cases/12/variants/sklearn_auto/samples/18` |
| real_iris_raw_k5_seed151 | sklearn auto | 18 | second | `/cases/12/variants/sklearn_auto/samples/18` |
| real_iris_raw_k5_seed151 | sklearn auto | 19 | first | `/cases/12/variants/sklearn_auto/samples/19` |
| real_iris_raw_k5_seed151 | sklearn auto | 19 | second | `/cases/12/variants/sklearn_auto/samples/19` |
| real_iris_raw_k5_seed151 | sklearn auto | 20 | first | `/cases/12/variants/sklearn_auto/samples/20` |
| real_iris_raw_k5_seed151 | sklearn auto | 20 | second | `/cases/12/variants/sklearn_auto/samples/20` |
| real_iris_raw_k5_seed151 | sklearn auto | 21 | first | `/cases/12/variants/sklearn_auto/samples/21` |
| real_iris_raw_k5_seed151 | sklearn auto | 21 | second | `/cases/12/variants/sklearn_auto/samples/21` |
| real_iris_raw_k5_seed151 | sklearn auto | 22 | first | `/cases/12/variants/sklearn_auto/samples/22` |
| real_iris_raw_k5_seed151 | sklearn auto | 22 | second | `/cases/12/variants/sklearn_auto/samples/22` |
| real_iris_raw_k5_seed151 | sklearn auto | 23 | first | `/cases/12/variants/sklearn_auto/samples/23` |
| real_iris_raw_k5_seed151 | sklearn auto | 23 | second | `/cases/12/variants/sklearn_auto/samples/23` |
| real_iris_raw_k5_seed151 | sklearn auto | 24 | first | `/cases/12/variants/sklearn_auto/samples/24` |
| real_iris_raw_k5_seed151 | sklearn auto | 24 | second | `/cases/12/variants/sklearn_auto/samples/24` |
| real_iris_raw_k5_seed151 | sklearn auto | 25 | first | `/cases/12/variants/sklearn_auto/samples/25` |
| real_iris_raw_k5_seed151 | sklearn auto | 25 | second | `/cases/12/variants/sklearn_auto/samples/25` |
| real_iris_raw_k5_seed151 | sklearn auto | 26 | first | `/cases/12/variants/sklearn_auto/samples/26` |
| real_iris_raw_k5_seed151 | sklearn auto | 26 | second | `/cases/12/variants/sklearn_auto/samples/26` |
| real_iris_raw_k5_seed151 | sklearn auto | 27 | first | `/cases/12/variants/sklearn_auto/samples/27` |
| real_iris_raw_k5_seed151 | sklearn auto | 27 | second | `/cases/12/variants/sklearn_auto/samples/27` |
| real_iris_raw_k5_seed151 | sklearn auto | 28 | first | `/cases/12/variants/sklearn_auto/samples/28` |
| real_iris_raw_k5_seed151 | sklearn auto | 28 | second | `/cases/12/variants/sklearn_auto/samples/28` |
| real_iris_raw_k5_seed151 | sklearn auto | 29 | first | `/cases/12/variants/sklearn_auto/samples/29` |
| real_iris_raw_k5_seed151 | sklearn auto | 29 | second | `/cases/12/variants/sklearn_auto/samples/29` |
| real_iris_raw_k5_seed252 | Default-thread brute | 0 | first | `/cases/14/variants/sklearn_default/samples/0` |
| real_iris_raw_k5_seed252 | Default-thread brute | 0 | second | `/cases/14/variants/sklearn_default/samples/0` |
| real_iris_raw_k5_seed252 | Default-thread brute | 1 | first | `/cases/14/variants/sklearn_default/samples/1` |
| real_iris_raw_k5_seed252 | Default-thread brute | 1 | second | `/cases/14/variants/sklearn_default/samples/1` |
| real_iris_raw_k5_seed252 | Default-thread brute | 2 | first | `/cases/14/variants/sklearn_default/samples/2` |
| real_iris_raw_k5_seed252 | Default-thread brute | 2 | second | `/cases/14/variants/sklearn_default/samples/2` |
| real_iris_raw_k5_seed252 | Default-thread brute | 3 | first | `/cases/14/variants/sklearn_default/samples/3` |
| real_iris_raw_k5_seed252 | Default-thread brute | 3 | second | `/cases/14/variants/sklearn_default/samples/3` |
| real_iris_raw_k5_seed252 | Default-thread brute | 4 | first | `/cases/14/variants/sklearn_default/samples/4` |
| real_iris_raw_k5_seed252 | Default-thread brute | 4 | second | `/cases/14/variants/sklearn_default/samples/4` |
| real_iris_raw_k5_seed252 | Default-thread brute | 5 | first | `/cases/14/variants/sklearn_default/samples/5` |
| real_iris_raw_k5_seed252 | Default-thread brute | 5 | second | `/cases/14/variants/sklearn_default/samples/5` |
| real_iris_raw_k5_seed252 | Default-thread brute | 6 | first | `/cases/14/variants/sklearn_default/samples/6` |
| real_iris_raw_k5_seed252 | Default-thread brute | 6 | second | `/cases/14/variants/sklearn_default/samples/6` |
| real_iris_raw_k5_seed252 | Default-thread brute | 7 | first | `/cases/14/variants/sklearn_default/samples/7` |
| real_iris_raw_k5_seed252 | Default-thread brute | 7 | second | `/cases/14/variants/sklearn_default/samples/7` |
| real_iris_raw_k5_seed252 | Default-thread brute | 8 | first | `/cases/14/variants/sklearn_default/samples/8` |
| real_iris_raw_k5_seed252 | Default-thread brute | 8 | second | `/cases/14/variants/sklearn_default/samples/8` |
| real_iris_raw_k5_seed252 | Default-thread brute | 9 | first | `/cases/14/variants/sklearn_default/samples/9` |
| real_iris_raw_k5_seed252 | Default-thread brute | 9 | second | `/cases/14/variants/sklearn_default/samples/9` |
| real_iris_raw_k5_seed252 | Default-thread brute | 10 | first | `/cases/14/variants/sklearn_default/samples/10` |
| real_iris_raw_k5_seed252 | Default-thread brute | 10 | second | `/cases/14/variants/sklearn_default/samples/10` |
| real_iris_raw_k5_seed252 | Default-thread brute | 11 | first | `/cases/14/variants/sklearn_default/samples/11` |
| real_iris_raw_k5_seed252 | Default-thread brute | 11 | second | `/cases/14/variants/sklearn_default/samples/11` |
| real_iris_raw_k5_seed252 | Default-thread brute | 12 | first | `/cases/14/variants/sklearn_default/samples/12` |
| real_iris_raw_k5_seed252 | Default-thread brute | 12 | second | `/cases/14/variants/sklearn_default/samples/12` |
| real_iris_raw_k5_seed252 | Default-thread brute | 13 | first | `/cases/14/variants/sklearn_default/samples/13` |
| real_iris_raw_k5_seed252 | Default-thread brute | 13 | second | `/cases/14/variants/sklearn_default/samples/13` |
| real_iris_raw_k5_seed252 | Default-thread brute | 14 | first | `/cases/14/variants/sklearn_default/samples/14` |
| real_iris_raw_k5_seed252 | Default-thread brute | 14 | second | `/cases/14/variants/sklearn_default/samples/14` |
| real_iris_raw_k5_seed252 | Default-thread brute | 15 | first | `/cases/14/variants/sklearn_default/samples/15` |
| real_iris_raw_k5_seed252 | Default-thread brute | 15 | second | `/cases/14/variants/sklearn_default/samples/15` |
| real_iris_raw_k5_seed252 | Default-thread brute | 16 | first | `/cases/14/variants/sklearn_default/samples/16` |
| real_iris_raw_k5_seed252 | Default-thread brute | 16 | second | `/cases/14/variants/sklearn_default/samples/16` |
| real_iris_raw_k5_seed252 | Default-thread brute | 17 | first | `/cases/14/variants/sklearn_default/samples/17` |
| real_iris_raw_k5_seed252 | Default-thread brute | 17 | second | `/cases/14/variants/sklearn_default/samples/17` |
| real_iris_raw_k5_seed252 | Default-thread brute | 18 | first | `/cases/14/variants/sklearn_default/samples/18` |
| real_iris_raw_k5_seed252 | Default-thread brute | 18 | second | `/cases/14/variants/sklearn_default/samples/18` |
| real_iris_raw_k5_seed252 | Default-thread brute | 19 | first | `/cases/14/variants/sklearn_default/samples/19` |
| real_iris_raw_k5_seed252 | Default-thread brute | 19 | second | `/cases/14/variants/sklearn_default/samples/19` |
| real_iris_raw_k5_seed252 | Default-thread brute | 20 | first | `/cases/14/variants/sklearn_default/samples/20` |
| real_iris_raw_k5_seed252 | Default-thread brute | 20 | second | `/cases/14/variants/sklearn_default/samples/20` |
| real_iris_raw_k5_seed252 | Default-thread brute | 21 | first | `/cases/14/variants/sklearn_default/samples/21` |
| real_iris_raw_k5_seed252 | Default-thread brute | 21 | second | `/cases/14/variants/sklearn_default/samples/21` |
| real_iris_raw_k5_seed252 | Default-thread brute | 22 | first | `/cases/14/variants/sklearn_default/samples/22` |
| real_iris_raw_k5_seed252 | Default-thread brute | 22 | second | `/cases/14/variants/sklearn_default/samples/22` |
| real_iris_raw_k5_seed252 | Default-thread brute | 23 | first | `/cases/14/variants/sklearn_default/samples/23` |
| real_iris_raw_k5_seed252 | Default-thread brute | 23 | second | `/cases/14/variants/sklearn_default/samples/23` |
| real_iris_raw_k5_seed252 | Default-thread brute | 24 | first | `/cases/14/variants/sklearn_default/samples/24` |
| real_iris_raw_k5_seed252 | Default-thread brute | 24 | second | `/cases/14/variants/sklearn_default/samples/24` |
| real_iris_raw_k5_seed252 | Default-thread brute | 25 | first | `/cases/14/variants/sklearn_default/samples/25` |
| real_iris_raw_k5_seed252 | Default-thread brute | 25 | second | `/cases/14/variants/sklearn_default/samples/25` |
| real_iris_raw_k5_seed252 | Default-thread brute | 26 | first | `/cases/14/variants/sklearn_default/samples/26` |
| real_iris_raw_k5_seed252 | Default-thread brute | 26 | second | `/cases/14/variants/sklearn_default/samples/26` |
| real_iris_raw_k5_seed252 | Default-thread brute | 27 | first | `/cases/14/variants/sklearn_default/samples/27` |
| real_iris_raw_k5_seed252 | Default-thread brute | 27 | second | `/cases/14/variants/sklearn_default/samples/27` |
| real_iris_raw_k5_seed252 | Default-thread brute | 28 | first | `/cases/14/variants/sklearn_default/samples/28` |
| real_iris_raw_k5_seed252 | Default-thread brute | 28 | second | `/cases/14/variants/sklearn_default/samples/28` |
| real_iris_raw_k5_seed252 | Default-thread brute | 29 | first | `/cases/14/variants/sklearn_default/samples/29` |
| real_iris_raw_k5_seed252 | Default-thread brute | 29 | second | `/cases/14/variants/sklearn_default/samples/29` |
| real_iris_raw_k5_seed252 | Single-thread brute | 0 | first | `/cases/14/variants/sklearn_1/samples/0` |
| real_iris_raw_k5_seed252 | Single-thread brute | 0 | second | `/cases/14/variants/sklearn_1/samples/0` |
| real_iris_raw_k5_seed252 | Single-thread brute | 1 | first | `/cases/14/variants/sklearn_1/samples/1` |
| real_iris_raw_k5_seed252 | Single-thread brute | 1 | second | `/cases/14/variants/sklearn_1/samples/1` |
| real_iris_raw_k5_seed252 | Single-thread brute | 2 | first | `/cases/14/variants/sklearn_1/samples/2` |
| real_iris_raw_k5_seed252 | Single-thread brute | 2 | second | `/cases/14/variants/sklearn_1/samples/2` |
| real_iris_raw_k5_seed252 | Single-thread brute | 3 | first | `/cases/14/variants/sklearn_1/samples/3` |
| real_iris_raw_k5_seed252 | Single-thread brute | 3 | second | `/cases/14/variants/sklearn_1/samples/3` |
| real_iris_raw_k5_seed252 | Single-thread brute | 4 | first | `/cases/14/variants/sklearn_1/samples/4` |
| real_iris_raw_k5_seed252 | Single-thread brute | 4 | second | `/cases/14/variants/sklearn_1/samples/4` |
| real_iris_raw_k5_seed252 | Single-thread brute | 5 | first | `/cases/14/variants/sklearn_1/samples/5` |
| real_iris_raw_k5_seed252 | Single-thread brute | 5 | second | `/cases/14/variants/sklearn_1/samples/5` |
| real_iris_raw_k5_seed252 | Single-thread brute | 6 | first | `/cases/14/variants/sklearn_1/samples/6` |
| real_iris_raw_k5_seed252 | Single-thread brute | 6 | second | `/cases/14/variants/sklearn_1/samples/6` |
| real_iris_raw_k5_seed252 | Single-thread brute | 7 | first | `/cases/14/variants/sklearn_1/samples/7` |
| real_iris_raw_k5_seed252 | Single-thread brute | 7 | second | `/cases/14/variants/sklearn_1/samples/7` |
| real_iris_raw_k5_seed252 | Single-thread brute | 8 | first | `/cases/14/variants/sklearn_1/samples/8` |
| real_iris_raw_k5_seed252 | Single-thread brute | 8 | second | `/cases/14/variants/sklearn_1/samples/8` |
| real_iris_raw_k5_seed252 | Single-thread brute | 9 | first | `/cases/14/variants/sklearn_1/samples/9` |
| real_iris_raw_k5_seed252 | Single-thread brute | 9 | second | `/cases/14/variants/sklearn_1/samples/9` |
| real_iris_raw_k5_seed252 | Single-thread brute | 10 | first | `/cases/14/variants/sklearn_1/samples/10` |
| real_iris_raw_k5_seed252 | Single-thread brute | 10 | second | `/cases/14/variants/sklearn_1/samples/10` |
| real_iris_raw_k5_seed252 | Single-thread brute | 11 | first | `/cases/14/variants/sklearn_1/samples/11` |
| real_iris_raw_k5_seed252 | Single-thread brute | 11 | second | `/cases/14/variants/sklearn_1/samples/11` |
| real_iris_raw_k5_seed252 | Single-thread brute | 12 | first | `/cases/14/variants/sklearn_1/samples/12` |
| real_iris_raw_k5_seed252 | Single-thread brute | 12 | second | `/cases/14/variants/sklearn_1/samples/12` |
| real_iris_raw_k5_seed252 | Single-thread brute | 13 | first | `/cases/14/variants/sklearn_1/samples/13` |
| real_iris_raw_k5_seed252 | Single-thread brute | 13 | second | `/cases/14/variants/sklearn_1/samples/13` |
| real_iris_raw_k5_seed252 | Single-thread brute | 14 | first | `/cases/14/variants/sklearn_1/samples/14` |
| real_iris_raw_k5_seed252 | Single-thread brute | 14 | second | `/cases/14/variants/sklearn_1/samples/14` |
| real_iris_raw_k5_seed252 | Single-thread brute | 15 | first | `/cases/14/variants/sklearn_1/samples/15` |
| real_iris_raw_k5_seed252 | Single-thread brute | 15 | second | `/cases/14/variants/sklearn_1/samples/15` |
| real_iris_raw_k5_seed252 | Single-thread brute | 16 | first | `/cases/14/variants/sklearn_1/samples/16` |
| real_iris_raw_k5_seed252 | Single-thread brute | 16 | second | `/cases/14/variants/sklearn_1/samples/16` |
| real_iris_raw_k5_seed252 | Single-thread brute | 17 | first | `/cases/14/variants/sklearn_1/samples/17` |
| real_iris_raw_k5_seed252 | Single-thread brute | 17 | second | `/cases/14/variants/sklearn_1/samples/17` |
| real_iris_raw_k5_seed252 | Single-thread brute | 18 | first | `/cases/14/variants/sklearn_1/samples/18` |
| real_iris_raw_k5_seed252 | Single-thread brute | 18 | second | `/cases/14/variants/sklearn_1/samples/18` |
| real_iris_raw_k5_seed252 | Single-thread brute | 19 | first | `/cases/14/variants/sklearn_1/samples/19` |
| real_iris_raw_k5_seed252 | Single-thread brute | 19 | second | `/cases/14/variants/sklearn_1/samples/19` |
| real_iris_raw_k5_seed252 | Single-thread brute | 20 | first | `/cases/14/variants/sklearn_1/samples/20` |
| real_iris_raw_k5_seed252 | Single-thread brute | 20 | second | `/cases/14/variants/sklearn_1/samples/20` |
| real_iris_raw_k5_seed252 | Single-thread brute | 21 | first | `/cases/14/variants/sklearn_1/samples/21` |
| real_iris_raw_k5_seed252 | Single-thread brute | 21 | second | `/cases/14/variants/sklearn_1/samples/21` |
| real_iris_raw_k5_seed252 | Single-thread brute | 22 | first | `/cases/14/variants/sklearn_1/samples/22` |
| real_iris_raw_k5_seed252 | Single-thread brute | 22 | second | `/cases/14/variants/sklearn_1/samples/22` |
| real_iris_raw_k5_seed252 | Single-thread brute | 23 | first | `/cases/14/variants/sklearn_1/samples/23` |
| real_iris_raw_k5_seed252 | Single-thread brute | 23 | second | `/cases/14/variants/sklearn_1/samples/23` |
| real_iris_raw_k5_seed252 | Single-thread brute | 24 | first | `/cases/14/variants/sklearn_1/samples/24` |
| real_iris_raw_k5_seed252 | Single-thread brute | 24 | second | `/cases/14/variants/sklearn_1/samples/24` |
| real_iris_raw_k5_seed252 | Single-thread brute | 25 | first | `/cases/14/variants/sklearn_1/samples/25` |
| real_iris_raw_k5_seed252 | Single-thread brute | 25 | second | `/cases/14/variants/sklearn_1/samples/25` |
| real_iris_raw_k5_seed252 | Single-thread brute | 26 | first | `/cases/14/variants/sklearn_1/samples/26` |
| real_iris_raw_k5_seed252 | Single-thread brute | 26 | second | `/cases/14/variants/sklearn_1/samples/26` |
| real_iris_raw_k5_seed252 | Single-thread brute | 27 | first | `/cases/14/variants/sklearn_1/samples/27` |
| real_iris_raw_k5_seed252 | Single-thread brute | 27 | second | `/cases/14/variants/sklearn_1/samples/27` |
| real_iris_raw_k5_seed252 | Single-thread brute | 28 | first | `/cases/14/variants/sklearn_1/samples/28` |
| real_iris_raw_k5_seed252 | Single-thread brute | 28 | second | `/cases/14/variants/sklearn_1/samples/28` |
| real_iris_raw_k5_seed252 | Single-thread brute | 29 | first | `/cases/14/variants/sklearn_1/samples/29` |
| real_iris_raw_k5_seed252 | Single-thread brute | 29 | second | `/cases/14/variants/sklearn_1/samples/29` |
| real_iris_raw_k5_seed252 | sklearn auto | 0 | first | `/cases/14/variants/sklearn_auto/samples/0` |
| real_iris_raw_k5_seed252 | sklearn auto | 0 | second | `/cases/14/variants/sklearn_auto/samples/0` |
| real_iris_raw_k5_seed252 | sklearn auto | 1 | first | `/cases/14/variants/sklearn_auto/samples/1` |
| real_iris_raw_k5_seed252 | sklearn auto | 1 | second | `/cases/14/variants/sklearn_auto/samples/1` |
| real_iris_raw_k5_seed252 | sklearn auto | 2 | first | `/cases/14/variants/sklearn_auto/samples/2` |
| real_iris_raw_k5_seed252 | sklearn auto | 2 | second | `/cases/14/variants/sklearn_auto/samples/2` |
| real_iris_raw_k5_seed252 | sklearn auto | 3 | first | `/cases/14/variants/sklearn_auto/samples/3` |
| real_iris_raw_k5_seed252 | sklearn auto | 3 | second | `/cases/14/variants/sklearn_auto/samples/3` |
| real_iris_raw_k5_seed252 | sklearn auto | 4 | first | `/cases/14/variants/sklearn_auto/samples/4` |
| real_iris_raw_k5_seed252 | sklearn auto | 4 | second | `/cases/14/variants/sklearn_auto/samples/4` |
| real_iris_raw_k5_seed252 | sklearn auto | 5 | first | `/cases/14/variants/sklearn_auto/samples/5` |
| real_iris_raw_k5_seed252 | sklearn auto | 5 | second | `/cases/14/variants/sklearn_auto/samples/5` |
| real_iris_raw_k5_seed252 | sklearn auto | 6 | first | `/cases/14/variants/sklearn_auto/samples/6` |
| real_iris_raw_k5_seed252 | sklearn auto | 6 | second | `/cases/14/variants/sklearn_auto/samples/6` |
| real_iris_raw_k5_seed252 | sklearn auto | 7 | first | `/cases/14/variants/sklearn_auto/samples/7` |
| real_iris_raw_k5_seed252 | sklearn auto | 7 | second | `/cases/14/variants/sklearn_auto/samples/7` |
| real_iris_raw_k5_seed252 | sklearn auto | 8 | first | `/cases/14/variants/sklearn_auto/samples/8` |
| real_iris_raw_k5_seed252 | sklearn auto | 8 | second | `/cases/14/variants/sklearn_auto/samples/8` |
| real_iris_raw_k5_seed252 | sklearn auto | 9 | first | `/cases/14/variants/sklearn_auto/samples/9` |
| real_iris_raw_k5_seed252 | sklearn auto | 9 | second | `/cases/14/variants/sklearn_auto/samples/9` |
| real_iris_raw_k5_seed252 | sklearn auto | 10 | first | `/cases/14/variants/sklearn_auto/samples/10` |
| real_iris_raw_k5_seed252 | sklearn auto | 10 | second | `/cases/14/variants/sklearn_auto/samples/10` |
| real_iris_raw_k5_seed252 | sklearn auto | 11 | first | `/cases/14/variants/sklearn_auto/samples/11` |
| real_iris_raw_k5_seed252 | sklearn auto | 11 | second | `/cases/14/variants/sklearn_auto/samples/11` |
| real_iris_raw_k5_seed252 | sklearn auto | 12 | first | `/cases/14/variants/sklearn_auto/samples/12` |
| real_iris_raw_k5_seed252 | sklearn auto | 12 | second | `/cases/14/variants/sklearn_auto/samples/12` |
| real_iris_raw_k5_seed252 | sklearn auto | 13 | first | `/cases/14/variants/sklearn_auto/samples/13` |
| real_iris_raw_k5_seed252 | sklearn auto | 13 | second | `/cases/14/variants/sklearn_auto/samples/13` |
| real_iris_raw_k5_seed252 | sklearn auto | 14 | first | `/cases/14/variants/sklearn_auto/samples/14` |
| real_iris_raw_k5_seed252 | sklearn auto | 14 | second | `/cases/14/variants/sklearn_auto/samples/14` |
| real_iris_raw_k5_seed252 | sklearn auto | 15 | first | `/cases/14/variants/sklearn_auto/samples/15` |
| real_iris_raw_k5_seed252 | sklearn auto | 15 | second | `/cases/14/variants/sklearn_auto/samples/15` |
| real_iris_raw_k5_seed252 | sklearn auto | 16 | first | `/cases/14/variants/sklearn_auto/samples/16` |
| real_iris_raw_k5_seed252 | sklearn auto | 16 | second | `/cases/14/variants/sklearn_auto/samples/16` |
| real_iris_raw_k5_seed252 | sklearn auto | 17 | first | `/cases/14/variants/sklearn_auto/samples/17` |
| real_iris_raw_k5_seed252 | sklearn auto | 17 | second | `/cases/14/variants/sklearn_auto/samples/17` |
| real_iris_raw_k5_seed252 | sklearn auto | 18 | first | `/cases/14/variants/sklearn_auto/samples/18` |
| real_iris_raw_k5_seed252 | sklearn auto | 18 | second | `/cases/14/variants/sklearn_auto/samples/18` |
| real_iris_raw_k5_seed252 | sklearn auto | 19 | first | `/cases/14/variants/sklearn_auto/samples/19` |
| real_iris_raw_k5_seed252 | sklearn auto | 19 | second | `/cases/14/variants/sklearn_auto/samples/19` |
| real_iris_raw_k5_seed252 | sklearn auto | 20 | first | `/cases/14/variants/sklearn_auto/samples/20` |
| real_iris_raw_k5_seed252 | sklearn auto | 20 | second | `/cases/14/variants/sklearn_auto/samples/20` |
| real_iris_raw_k5_seed252 | sklearn auto | 21 | first | `/cases/14/variants/sklearn_auto/samples/21` |
| real_iris_raw_k5_seed252 | sklearn auto | 21 | second | `/cases/14/variants/sklearn_auto/samples/21` |
| real_iris_raw_k5_seed252 | sklearn auto | 22 | first | `/cases/14/variants/sklearn_auto/samples/22` |
| real_iris_raw_k5_seed252 | sklearn auto | 22 | second | `/cases/14/variants/sklearn_auto/samples/22` |
| real_iris_raw_k5_seed252 | sklearn auto | 23 | first | `/cases/14/variants/sklearn_auto/samples/23` |
| real_iris_raw_k5_seed252 | sklearn auto | 23 | second | `/cases/14/variants/sklearn_auto/samples/23` |
| real_iris_raw_k5_seed252 | sklearn auto | 24 | first | `/cases/14/variants/sklearn_auto/samples/24` |
| real_iris_raw_k5_seed252 | sklearn auto | 24 | second | `/cases/14/variants/sklearn_auto/samples/24` |
| real_iris_raw_k5_seed252 | sklearn auto | 25 | first | `/cases/14/variants/sklearn_auto/samples/25` |
| real_iris_raw_k5_seed252 | sklearn auto | 25 | second | `/cases/14/variants/sklearn_auto/samples/25` |
| real_iris_raw_k5_seed252 | sklearn auto | 26 | first | `/cases/14/variants/sklearn_auto/samples/26` |
| real_iris_raw_k5_seed252 | sklearn auto | 26 | second | `/cases/14/variants/sklearn_auto/samples/26` |
| real_iris_raw_k5_seed252 | sklearn auto | 27 | first | `/cases/14/variants/sklearn_auto/samples/27` |
| real_iris_raw_k5_seed252 | sklearn auto | 27 | second | `/cases/14/variants/sklearn_auto/samples/27` |
| real_iris_raw_k5_seed252 | sklearn auto | 28 | first | `/cases/14/variants/sklearn_auto/samples/28` |
| real_iris_raw_k5_seed252 | sklearn auto | 28 | second | `/cases/14/variants/sklearn_auto/samples/28` |
| real_iris_raw_k5_seed252 | sklearn auto | 29 | first | `/cases/14/variants/sklearn_auto/samples/29` |
| real_iris_raw_k5_seed252 | sklearn auto | 29 | second | `/cases/14/variants/sklearn_auto/samples/29` |

## Audit counts and actual routing

| Worker | Audits | Query routing | Native loaded before fit | Stock sklearn.neighbors loaded before fit |
| --- | --- | --- | --- | --- |
| Current automatic | {"first_queries": 540, "first_strict_passes": 540, "models_requested": 540, "second_queries": 540, "second_strict_passes": 540} | {"first/cpu": 540, "second/cpu": 540} | {"True": 540} | {"False": 540} |
| Default-thread brute | {"first_queries": 540, "first_strict_passes": 390, "models_requested": 540, "second_queries": 540, "second_strict_passes": 390} | {"first/sklearn": 540, "second/sklearn": 540} | {"False": 540} | {"True": 540} |
| Single-thread brute | {"first_queries": 540, "first_strict_passes": 390, "models_requested": 540, "second_queries": 540, "second_strict_passes": 390} | {"first/sklearn": 540, "second/sklearn": 540} | {"False": 540} | {"True": 540} |
| sklearn auto | {"first_queries": 540, "first_strict_passes": 390, "models_requested": 540, "second_queries": 540, "second_strict_passes": 390} | {"first/sklearn": 540, "second/sklearn": 540} | {"False": 540} | {"True": 540} |
| Preceding automatic | {"first_queries": 540, "first_strict_passes": 540, "models_requested": 540, "second_queries": 540, "second_strict_passes": 540} | {"first/cpu": 540, "second/cpu": 540} | {"True": 540} | {"False": 540} |

## Evidence

[Raw requests, setup, source/runtime identities, and all audits](../benchmarks/results/knn-digits-final-2026-10-30-first-query.json)

[Validated summary](../benchmarks/results/knn-digits-final-2026-10-30-first-query-summary.json)
