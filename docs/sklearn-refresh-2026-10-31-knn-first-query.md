# Fresh-model first queries in initialized processes

The threadpool-limited worker requests `threadpool_limits(1)` and records one OpenMP thread. NumPy/SciPy use Apple Accelerate, which this threadpoolctl installation does not control; this is not a verified one-thread limit for every BLAS call. Default-thread workers retain the inherited environment.

Initialized isolated persistent processes; normal NumPy/estimator imports and configuration are recorded separately. No model or query prewarm. Every request constructs and fits a new estimator, then times its first query and separately its second query. Public NumPy results establish completion; no harness MLX import/device probe/synchronization runs before the first query. Both outputs are audited after both timers. All requested rounds (default 30), including first model per fixture and first model in worker, remain visible. Earlier models may initialize runtime/compiler caches. This is model-first-use in an initialized process, not fresh-process or machine-cold latency. Five-worker Williams order has a complete balanced 10-round block, repeated three times by default. Any incomplete final block is also retained. No classification/regression operations are added here; the separate full broad and preservation control panels retain those operations and fixtures.

First query includes model preparation deferred by fit. Second query reuses only that fresh model. All rounds remain in the headline median; first-per-fixture and subsequent-model views are shown separately, not selected according to speed. Process initialization/import cost is separate, and must not be confused with the dedicated fresh-process cold benchmark.

Acceptance: **True**. 18 fixtures × 30 fresh models × 4 workers; each model has exactly one retained first and one retained second query. No numerical audit failure removes a valid timing. Ratios are comparator/current; bold values below 1× are slower observations. These descriptive ratios are not confidence intervals.

## Initial process setup (once per worker)

| Worker | NumPy import ms | Estimator import/setup ms | Worker entry→initialized ms | Parent process→initialized ms |
| --- | --- | --- | --- | --- |
| Current automatic | 19.392 | 352.814 | 373.962 | 399.432 |
| Default-thread brute | 18.238 | 421.016 | 440.955 | 464.651 |
| Threadpool-limited brute | 19.327 | 411.540 | 432.693 | 457.933 |
| sklearn auto | 18.327 | 397.964 | 418.026 | 441.821 |

## fit_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Threadpool-limited brute ms / M / P | sklearn auto ms / M / P |
| --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.093 | 0.126 / 1.355× / 1.289× | 0.120 / 1.281× / 1.261× | 1.525 / 16.351× / 16.381× |
| Gaussian 32D, single query | 49 | 0.119 | 0.145 / 1.222× / 1.266× | 0.138 / 1.161× / 1.168× | 0.146 / 1.227× / 1.217× |
| Gaussian 8D, single query | 42 | 0.091 | 0.119 / 1.313× / 1.316× | 0.110 / 1.216× / 1.197× | 1.559 / 17.169× / 17.158× |
| Gaussian 32D, single query | 50 | 0.103 | 0.138 / 1.337× / 1.370× | 0.135 / 1.310× / 1.322× | 0.138 / 1.340× / 1.325× |
| Gaussian 8D, single query | 43 | 0.104 | 0.123 / 1.183× / 1.228× | 0.123 / 1.179× / 1.281× | 1.556 / 14.961× / 14.903× |
| Gaussian 32D, single query | 51 | 0.138 | 0.166 / 1.199× / 1.226× | 0.179 / 1.291× / 1.357× | 0.170 / 1.230× / 1.210× |
| Digits (raw) | 151 | 0.085 | 0.145 / 1.703× / 1.739× | 0.162 / 1.900× / 1.939× | 0.150 / 1.754× / 1.673× |
| Digits (raw) | 251 | 0.090 | 0.159 / 1.764× / 1.794× | 0.170 / 1.892× / 1.974× | 0.170 / 1.891× / 1.928× |
| Digits (raw) | 252 | 0.076 | 0.145 / 1.901× / 1.937× | 0.144 / 1.881× / 1.980× | 0.146 / 1.913× / 1.973× |
| Breast Cancer (raw) | 151 | 0.063 | 0.139 / 2.218× / 2.167× | 0.145 / 2.301× / 2.230× | 0.137 / 2.185× / 2.099× |
| Breast Cancer (raw) | 251 | 0.057 | 0.122 / 2.137× / 2.109× | 0.125 / 2.190× / 2.276× | 0.134 / 2.351× / 2.227× |
| Breast Cancer (raw) | 252 | 0.052 | 0.109 / 2.090× / 2.155× | 0.114 / 2.184× / 2.418× | 0.112 / 2.150× / 2.239× |
| Iris (raw) | 151 | 0.043 | 0.105 / 2.414× / 2.337× | 0.109 / 2.515× / 2.415× | 0.153 / 3.524× / 3.448× |
| Iris (raw) | 251 | 0.043 | 0.111 / 2.599× / 2.527× | 0.103 / 2.428× / 2.406× | 0.137 / 3.210× / 3.376× |
| Iris (raw) | 252 | 0.057 | 0.125 / 2.197× / 2.153× | 0.119 / 2.095× / 2.236× | 0.161 / 2.833× / 3.302× |
| Wine (raw) | 151 | 0.044 | 0.109 / 2.507× / 2.359× | 0.108 / 2.478× / 2.462× | 0.161 / 3.690× / 3.520× |
| Wine (raw) | 251 | 0.039 | 0.096 / 2.445× / 2.468× | 0.094 / 2.409× / 2.360× | 0.138 / 3.529× / 3.621× |
| Wine (raw) | 252 | 0.036 | 0.093 / 2.589× / 2.604× | 0.088 / 2.430× / 2.486× | 0.129 / 3.585× / 3.504× |

## first_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Threadpool-limited brute ms / M / P | sklearn auto ms / M / P |
| --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.069 | 0.704 / 10.215× / 9.797× | 0.219 / 3.176× / 3.140× | 0.195 / 2.834× / 2.826× |
| Gaussian 32D, single query | 49 | 0.124 | 2.537 / 20.538× / 20.453× | 0.223 / 1.808× / 1.801× | 2.476 / 20.041× / 19.621× |
| Gaussian 8D, single query | 42 | 0.067 | 2.363 / 35.240× / 35.150× | 0.208 / 3.102× / 3.052× | 0.193 / 2.875× / 2.874× |
| Gaussian 32D, single query | 50 | 0.120 | 2.452 / 20.359× / 20.260× | 0.222 / 1.846× / 1.857× | 2.426 / 20.137× / 20.106× |
| Gaussian 8D, single query | 43 | 0.070 | 2.478 / 35.647× / 34.964× | 0.217 / 3.116× / 3.177× | 0.192 / 2.766× / 2.776× |
| Gaussian 32D, single query | 51 | 0.142 | 2.806 / 19.823× / 20.067× | 0.250 / 1.765× / 1.763× | 2.832 / 20.009× / 20.105× |
| Digits (raw) | 151 | 0.445 | 2.938 / 6.596× / 6.738× | 1.006 / 2.259× / 2.265× | 2.995 / 6.725× / 6.757× |
| Digits (raw) | 251 | 0.454 | 3.057 / 6.735× / 6.782× | 1.033 / 2.276× / 2.245× | 3.063 / 6.747× / 6.690× |
| Digits (raw) | 252 | 0.438 | 3.071 / 7.017× / 7.099× | 1.000 / 2.285× / 2.289× | 3.034 / 6.931× / 7.063× |
| Breast Cancer (raw) | 151 | 0.214 | 2.155 / 10.070× / 10.278× | 0.257 / 1.199× / 1.237× | 2.226 / 10.403× / 10.579× |
| Breast Cancer (raw) | 251 | 0.199 | 2.279 / 11.478× / 11.482× | 0.244 / 1.227× / 1.257× | 2.144 / 10.799× / 10.660× |
| Breast Cancer (raw) | 252 | 0.181 | 2.187 / 12.092× / 11.656× | 0.231 / 1.278× / 1.267× | 2.234 / 12.354× / 12.370× |
| Iris (raw) | 151 | 0.053 | 1.559 / 29.442× / 29.092× | 0.103 / 1.953× / 1.959× | 0.175 / 3.308× / 3.416× |
| Iris (raw) | 251 | 0.055 | 0.662 / 12.137× / 13.019× | 0.101 / 1.842× / 1.899× | 0.171 / 3.137× / 3.333× |
| Iris (raw) | 252 | 0.063 | 0.764 / 12.160× / 13.648× | 0.118 / 1.884× / 2.018× | 0.203 / 3.227× / 3.329× |
| Wine (raw) | 151 | 0.066 | 0.717 / 10.839× / 10.848× | 0.106 / 1.608× / 1.691× | 0.193 / 2.919× / 2.820× |
| Wine (raw) | 251 | 0.058 | 0.560 / 9.650× / 9.769× | 0.097 / 1.675× / 1.653× | 0.172 / 2.957× / 2.878× |
| Wine (raw) | 252 | 0.059 | 0.536 / 9.103× / 8.844× | 0.095 / 1.616× / 1.644× | 0.160 / 2.720× / 2.731× |

## fit_first_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Threadpool-limited brute ms / M / P | sklearn auto ms / M / P |
| --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.161 | 0.829 / 5.159× / 4.968× | 0.338 / 2.100× / 2.088× | 1.722 / 10.713× / 10.573× |
| Gaussian 32D, single query | 49 | 0.239 | 2.668 / 11.157× / 11.749× | 0.360 / 1.503× / 1.497× | 2.632 / 11.003× / 10.779× |
| Gaussian 8D, single query | 42 | 0.157 | 2.470 / 15.694× / 15.306× | 0.319 / 2.024× / 1.978× | 1.747 / 11.098× / 10.908× |
| Gaussian 32D, single query | 50 | 0.224 | 2.588 / 11.555× / 11.723× | 0.359 / 1.601× / 1.628× | 2.562 / 11.442× / 11.741× |
| Gaussian 8D, single query | 43 | 0.173 | 2.595 / 15.002× / 14.425× | 0.338 / 1.957× / 2.005× | 1.757 / 10.157× / 9.915× |
| Gaussian 32D, single query | 51 | 0.278 | 2.968 / 10.662× / 11.056× | 0.430 / 1.545× / 1.560× | 2.993 / 10.753× / 10.825× |
| Digits (raw) | 151 | 0.522 | 3.089 / 5.916× / 5.934× | 1.178 / 2.255× / 2.259× | 3.151 / 6.036× / 5.973× |
| Digits (raw) | 251 | 0.538 | 3.214 / 5.969× / 5.944× | 1.216 / 2.259× / 2.245× | 3.245 / 6.027× / 5.926× |
| Digits (raw) | 252 | 0.510 | 3.209 / 6.294× / 6.244× | 1.149 / 2.254× / 2.227× | 3.189 / 6.254× / 6.246× |
| Breast Cancer (raw) | 151 | 0.280 | 2.294 / 8.198× / 8.375× | 0.398 / 1.424× / 1.483× | 2.366 / 8.457× / 8.525× |
| Breast Cancer (raw) | 251 | 0.253 | 2.414 / 9.535× / 9.538× | 0.371 / 1.465× / 1.496× | 2.282 / 9.013× / 8.544× |
| Breast Cancer (raw) | 252 | 0.233 | 2.332 / 10.020× / 9.727× | 0.347 / 1.491× / 1.517× | 2.342 / 10.067× / 9.981× |
| Iris (raw) | 151 | 0.094 | 1.661 / 17.760× / 17.260× | 0.211 / 2.252× / 2.128× | 0.327 / 3.495× / 3.424× |
| Iris (raw) | 251 | 0.099 | 0.779 / 7.829× / 8.349× | 0.204 / 2.049× / 2.066× | 0.311 / 3.127× / 3.364× |
| Iris (raw) | 252 | 0.124 | 0.887 / 7.180× / 8.561× | 0.235 / 1.901× / 2.101× | 0.365 / 2.959× / 3.364× |
| Wine (raw) | 151 | 0.106 | 0.832 / 7.854× / 7.549× | 0.215 / 2.027× / 1.993× | 0.353 / 3.331× / 3.146× |
| Wine (raw) | 251 | 0.096 | 0.659 / 6.839× / 6.680× | 0.192 / 1.990× / 1.939× | 0.312 / 3.233× / 3.173× |
| Wine (raw) | 252 | 0.096 | 0.624 / 6.466× / 6.486× | 0.185 / 1.923× / 1.934× | 0.293 / 3.034× / 3.010× |

## second_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Threadpool-limited brute ms / M / P | sklearn auto ms / M / P |
| --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.059 | 0.781 / 13.221× / 12.846× | 0.200 / 3.380× / 3.360× | 0.161 / 2.727× / 2.750× |
| Gaussian 32D, single query | 49 | 0.114 | 2.618 / 22.951× / 23.211× | 0.207 / 1.817× / 1.832× | 2.720 / 23.840× / 23.364× |
| Gaussian 8D, single query | 42 | 0.057 | 2.429 / 42.418× / 41.545× | 0.192 / 3.349× / 3.280× | 0.160 / 2.802× / 2.751× |
| Gaussian 32D, single query | 50 | 0.112 | 2.507 / 22.372× / 21.945× | 0.206 / 1.843× / 1.841× | 2.584 / 23.061× / 22.850× |
| Gaussian 8D, single query | 43 | 0.059 | 2.424 / 41.354× / 41.507× | 0.196 / 3.339× / 3.333× | 0.159 / 2.716× / 2.665× |
| Gaussian 32D, single query | 51 | 0.125 | 2.895 / 23.156× / 22.799× | 0.216 / 1.732× / 1.736× | 2.798 / 22.383× / 22.598× |
| Digits (raw) | 151 | 0.288 | 3.037 / 10.540× / 10.885× | 0.960 / 3.331× / 3.333× | 2.954 / 10.252× / 10.708× |
| Digits (raw) | 251 | 0.294 | 3.077 / 10.473× / 10.335× | 0.983 / 3.344× / 3.361× | 2.947 / 10.030× / 10.582× |
| Digits (raw) | 252 | 0.287 | 2.976 / 10.385× / 10.596× | 0.960 / 3.351× / 3.386× | 2.932 / 10.232× / 10.873× |
| Breast Cancer (raw) | 151 | 0.142 | 2.321 / 16.385× / 16.026× | 0.216 / 1.522× / 1.559× | 2.270 / 16.020× / 15.645× |
| Breast Cancer (raw) | 251 | 0.143 | 2.233 / 15.569× / 15.355× | 0.209 / 1.457× / 1.456× | 2.228 / 15.536× / 15.907× |
| Breast Cancer (raw) | 252 | 0.135 | 2.229 / 16.463× / 16.986× | 0.205 / 1.518× / 1.530× | 2.278 / 16.824× / 17.043× |
| Iris (raw) | 151 | 0.041 | 1.587 / 38.524× / 38.130× | 0.082 / 2.000× / 2.013× | 0.155 / 3.759× / 3.714× |
| Iris (raw) | 251 | 0.043 | 0.698 / 16.205× / 16.056× | 0.083 / 1.928× / 1.972× | 0.154 / 3.568× / 3.629× |
| Iris (raw) | 252 | 0.045 | 0.841 / 18.620× / 19.705× | 0.088 / 1.959× / 1.960× | 0.161 / 3.557× / 3.663× |
| Wine (raw) | 151 | 0.052 | 0.718 / 13.705× / 13.837× | 0.087 / 1.652× / 1.651× | 0.159 / 3.038× / 3.009× |
| Wine (raw) | 251 | 0.051 | 0.567 / 11.120× / 11.168× | 0.087 / 1.713× / 1.684× | 0.153 / 3.003× / 3.051× |
| Wine (raw) | 252 | 0.050 | 0.575 / 11.451× / 11.348× | 0.086 / 1.714× / 1.719× | 0.149 / 2.976× / 2.987× |

## Every slower or missing all-round median

No slower or missing all-round medians.

## first_model_for_fixture

All configurations and metrics are retained in this separate view. Earlier fixtures may already have initialized process/runtime/compiler state; this table is not fresh-process cold latency.

| Case | Worker | fit_ms median ms | first_query_ms median ms | fit_first_query_ms median ms | second_query_ms median ms |
| --- | --- | --- | --- | --- | --- |
| shape_single_8_k5_seed41 | Current automatic | 0.995 | 0.104 | 1.099 | 0.071 |
| shape_single_8_k5_seed41 | Default-thread brute | 1.039 | 12.162 | 13.201 | 1.015 |
| shape_single_8_k5_seed41 | Threadpool-limited brute | 1.015 | 5.708 | 6.723 | 0.502 |
| shape_single_8_k5_seed41 | sklearn auto | 2.533 | 0.329 | 2.861 | 0.194 |
| shape_single_32_k5_seed49 | Current automatic | 0.140 | 0.128 | 0.267 | 0.117 |
| shape_single_32_k5_seed49 | Default-thread brute | 0.148 | 0.733 | 0.880 | 0.771 |
| shape_single_32_k5_seed49 | Threadpool-limited brute | 0.152 | 0.242 | 0.393 | 0.213 |
| shape_single_32_k5_seed49 | sklearn auto | 0.141 | 14.646 | 14.786 | 3.402 |
| shape_single_8_k5_seed42 | Current automatic | 0.120 | 0.074 | 0.194 | 0.059 |
| shape_single_8_k5_seed42 | Default-thread brute | 0.135 | 2.573 | 2.708 | 2.755 |
| shape_single_8_k5_seed42 | Threadpool-limited brute | 0.127 | 0.220 | 0.347 | 0.193 |
| shape_single_8_k5_seed42 | sklearn auto | 1.634 | 0.358 | 1.992 | 0.171 |
| shape_single_32_k5_seed50 | Current automatic | 0.131 | 0.127 | 0.258 | 0.112 |
| shape_single_32_k5_seed50 | Default-thread brute | 0.169 | 2.394 | 2.563 | 1.739 |
| shape_single_32_k5_seed50 | Threadpool-limited brute | 0.173 | 0.244 | 0.417 | 0.214 |
| shape_single_32_k5_seed50 | sklearn auto | 0.151 | 2.720 | 2.871 | 2.557 |
| shape_single_8_k5_seed43 | Current automatic | 0.114 | 0.068 | 0.181 | 0.056 |
| shape_single_8_k5_seed43 | Default-thread brute | 0.154 | 2.377 | 2.531 | 2.463 |
| shape_single_8_k5_seed43 | Threadpool-limited brute | 0.130 | 0.223 | 0.352 | 0.196 |
| shape_single_8_k5_seed43 | sklearn auto | 1.591 | 0.235 | 1.826 | 0.163 |
| shape_single_32_k5_seed51 | Current automatic | 0.154 | 0.143 | 0.298 | 0.130 |
| shape_single_32_k5_seed51 | Default-thread brute | 0.161 | 1.007 | 1.167 | 0.891 |
| shape_single_32_k5_seed51 | Threadpool-limited brute | 0.154 | 0.242 | 0.396 | 0.215 |
| shape_single_32_k5_seed51 | sklearn auto | 0.171 | 3.058 | 3.229 | 2.449 |
| real_digits_raw_k5_seed151 | Current automatic | 0.087 | 0.539 | 0.625 | 0.304 |
| real_digits_raw_k5_seed151 | Default-thread brute | 0.145 | 3.371 | 3.516 | 3.176 |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 0.162 | 1.035 | 1.197 | 0.965 |
| real_digits_raw_k5_seed151 | sklearn auto | 0.133 | 3.004 | 3.136 | 2.343 |
| real_digits_raw_k5_seed251 | Current automatic | 0.089 | 0.492 | 0.581 | 0.299 |
| real_digits_raw_k5_seed251 | Default-thread brute | 0.187 | 3.148 | 3.335 | 3.391 |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 0.182 | 1.042 | 1.224 | 0.989 |
| real_digits_raw_k5_seed251 | sklearn auto | 0.196 | 2.971 | 3.168 | 3.044 |
| real_digits_raw_k5_seed252 | Current automatic | 0.076 | 0.457 | 0.533 | 0.298 |
| real_digits_raw_k5_seed252 | Default-thread brute | 0.148 | 3.362 | 3.510 | 3.133 |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 0.167 | 1.007 | 1.174 | 0.967 |
| real_digits_raw_k5_seed252 | sklearn auto | 0.163 | 3.182 | 3.345 | 3.608 |
| real_breast_cancer_raw_k5_seed151 | Current automatic | 0.074 | 0.221 | 0.295 | 0.160 |
| real_breast_cancer_raw_k5_seed151 | Default-thread brute | 0.162 | 2.122 | 2.284 | 2.336 |
| real_breast_cancer_raw_k5_seed151 | Threadpool-limited brute | 0.173 | 0.276 | 0.449 | 0.223 |
| real_breast_cancer_raw_k5_seed151 | sklearn auto | 0.137 | 1.897 | 2.033 | 2.111 |
| real_breast_cancer_raw_k5_seed251 | Current automatic | 0.075 | 0.243 | 0.318 | 0.210 |
| real_breast_cancer_raw_k5_seed251 | Default-thread brute | 0.154 | 2.393 | 2.548 | 2.258 |
| real_breast_cancer_raw_k5_seed251 | Threadpool-limited brute | 0.189 | 0.305 | 0.495 | 0.240 |
| real_breast_cancer_raw_k5_seed251 | sklearn auto | 0.166 | 2.317 | 2.483 | 2.108 |
| real_breast_cancer_raw_k5_seed252 | Current automatic | 0.066 | 0.208 | 0.274 | 0.141 |
| real_breast_cancer_raw_k5_seed252 | Default-thread brute | 0.166 | 2.301 | 2.466 | 2.173 |
| real_breast_cancer_raw_k5_seed252 | Threadpool-limited brute | 0.163 | 0.260 | 0.423 | 0.213 |
| real_breast_cancer_raw_k5_seed252 | sklearn auto | 0.148 | 2.306 | 2.454 | 2.127 |
| real_iris_raw_k5_seed151 | Current automatic | 0.063 | 0.066 | 0.129 | 0.049 |
| real_iris_raw_k5_seed151 | Default-thread brute | 0.138 | 1.455 | 1.593 | 1.764 |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 0.140 | 0.128 | 0.269 | 0.092 |
| real_iris_raw_k5_seed151 | sklearn auto | 0.219 | 0.235 | 0.453 | 0.168 |
| real_iris_raw_k5_seed251 | Current automatic | 0.068 | 0.071 | 0.139 | 0.050 |
| real_iris_raw_k5_seed251 | Default-thread brute | 0.160 | 1.909 | 2.068 | 2.001 |
| real_iris_raw_k5_seed251 | Threadpool-limited brute | 0.165 | 0.143 | 0.307 | 0.109 |
| real_iris_raw_k5_seed251 | sklearn auto | 0.197 | 0.216 | 0.413 | 0.164 |
| real_iris_raw_k5_seed252 | Current automatic | 0.083 | 0.069 | 0.153 | 0.048 |
| real_iris_raw_k5_seed252 | Default-thread brute | 0.159 | 5.789 | 5.948 | 1.612 |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 0.175 | 0.156 | 0.331 | 0.093 |
| real_iris_raw_k5_seed252 | sklearn auto | 0.221 | 0.230 | 0.452 | 0.163 |
| real_wine_raw_k5_seed151 | Current automatic | 0.062 | 0.079 | 0.141 | 0.058 |
| real_wine_raw_k5_seed151 | Default-thread brute | 0.154 | 0.730 | 0.884 | 0.907 |
| real_wine_raw_k5_seed151 | Threadpool-limited brute | 0.144 | 0.143 | 0.287 | 0.095 |
| real_wine_raw_k5_seed151 | sklearn auto | 0.214 | 0.222 | 0.436 | 0.163 |
| real_wine_raw_k5_seed251 | Current automatic | 0.070 | 0.076 | 0.146 | 0.057 |
| real_wine_raw_k5_seed251 | Default-thread brute | 0.161 | 0.797 | 0.958 | 0.555 |
| real_wine_raw_k5_seed251 | Threadpool-limited brute | 0.139 | 0.125 | 0.264 | 0.090 |
| real_wine_raw_k5_seed251 | sklearn auto | 0.221 | 0.224 | 0.445 | 0.161 |
| real_wine_raw_k5_seed252 | Current automatic | 0.058 | 0.069 | 0.126 | 0.053 |
| real_wine_raw_k5_seed252 | Default-thread brute | 0.137 | 0.567 | 0.704 | 0.737 |
| real_wine_raw_k5_seed252 | Threadpool-limited brute | 0.143 | 0.130 | 0.273 | 0.095 |
| real_wine_raw_k5_seed252 | sklearn auto | 0.203 | 0.217 | 0.419 | 0.162 |

## subsequent_fresh_models

All configurations and metrics are retained in this separate view. Earlier fixtures may already have initialized process/runtime/compiler state; this table is not fresh-process cold latency.

| Case | Worker | fit_ms median ms | first_query_ms median ms | fit_first_query_ms median ms | second_query_ms median ms |
| --- | --- | --- | --- | --- | --- |
| shape_single_8_k5_seed41 | Current automatic | 0.093 | 0.069 | 0.160 | 0.059 |
| shape_single_8_k5_seed41 | Default-thread brute | 0.126 | 0.700 | 0.820 | 0.771 |
| shape_single_8_k5_seed41 | Threadpool-limited brute | 0.119 | 0.219 | 0.337 | 0.198 |
| shape_single_8_k5_seed41 | sklearn auto | 1.523 | 0.194 | 1.722 | 0.160 |
| shape_single_32_k5_seed49 | Current automatic | 0.118 | 0.123 | 0.238 | 0.114 |
| shape_single_32_k5_seed49 | Default-thread brute | 0.142 | 2.558 | 2.670 | 2.625 |
| shape_single_32_k5_seed49 | Threadpool-limited brute | 0.138 | 0.223 | 0.358 | 0.207 |
| shape_single_32_k5_seed49 | sklearn auto | 0.147 | 2.457 | 2.626 | 2.718 |
| shape_single_8_k5_seed42 | Current automatic | 0.091 | 0.067 | 0.157 | 0.057 |
| shape_single_8_k5_seed42 | Default-thread brute | 0.118 | 2.307 | 2.413 | 2.406 |
| shape_single_8_k5_seed42 | Threadpool-limited brute | 0.110 | 0.208 | 0.318 | 0.191 |
| shape_single_8_k5_seed42 | sklearn auto | 1.557 | 0.192 | 1.746 | 0.160 |
| shape_single_32_k5_seed50 | Current automatic | 0.103 | 0.120 | 0.223 | 0.112 |
| shape_single_32_k5_seed50 | Default-thread brute | 0.138 | 2.456 | 2.589 | 2.527 |
| shape_single_32_k5_seed50 | Threadpool-limited brute | 0.135 | 0.222 | 0.358 | 0.206 |
| shape_single_32_k5_seed50 | sklearn auto | 0.138 | 2.412 | 2.551 | 2.586 |
| shape_single_8_k5_seed43 | Current automatic | 0.104 | 0.070 | 0.173 | 0.059 |
| shape_single_8_k5_seed43 | Default-thread brute | 0.122 | 2.500 | 2.622 | 2.421 |
| shape_single_8_k5_seed43 | Threadpool-limited brute | 0.122 | 0.216 | 0.338 | 0.196 |
| shape_single_8_k5_seed43 | sklearn auto | 1.550 | 0.190 | 1.757 | 0.159 |
| shape_single_32_k5_seed51 | Current automatic | 0.134 | 0.141 | 0.276 | 0.125 |
| shape_single_32_k5_seed51 | Default-thread brute | 0.166 | 2.807 | 2.971 | 2.896 |
| shape_single_32_k5_seed51 | Threadpool-limited brute | 0.183 | 0.250 | 0.431 | 0.217 |
| shape_single_32_k5_seed51 | sklearn auto | 0.169 | 2.829 | 2.986 | 2.798 |
| real_digits_raw_k5_seed151 | Current automatic | 0.084 | 0.443 | 0.522 | 0.288 |
| real_digits_raw_k5_seed151 | Default-thread brute | 0.145 | 2.937 | 3.086 | 3.028 |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 0.162 | 1.004 | 1.177 | 0.960 |
| real_digits_raw_k5_seed151 | sklearn auto | 0.152 | 2.987 | 3.166 | 2.962 |
| real_digits_raw_k5_seed251 | Current automatic | 0.091 | 0.453 | 0.537 | 0.290 |
| real_digits_raw_k5_seed251 | Default-thread brute | 0.158 | 3.031 | 3.189 | 3.055 |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 0.170 | 1.032 | 1.208 | 0.980 |
| real_digits_raw_k5_seed251 | sklearn auto | 0.170 | 3.079 | 3.267 | 2.942 |
| real_digits_raw_k5_seed252 | Current automatic | 0.077 | 0.436 | 0.507 | 0.285 |
| real_digits_raw_k5_seed252 | Default-thread brute | 0.145 | 3.070 | 3.200 | 2.974 |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 0.142 | 0.999 | 1.148 | 0.960 |
| real_digits_raw_k5_seed252 | sklearn auto | 0.145 | 3.011 | 3.151 | 2.930 |
| real_breast_cancer_raw_k5_seed151 | Current automatic | 0.062 | 0.213 | 0.276 | 0.142 |
| real_breast_cancer_raw_k5_seed151 | Default-thread brute | 0.139 | 2.164 | 2.303 | 2.316 |
| real_breast_cancer_raw_k5_seed151 | Threadpool-limited brute | 0.143 | 0.255 | 0.396 | 0.213 |
| real_breast_cancer_raw_k5_seed151 | sklearn auto | 0.138 | 2.232 | 2.383 | 2.273 |
| real_breast_cancer_raw_k5_seed251 | Current automatic | 0.057 | 0.198 | 0.251 | 0.143 |
| real_breast_cancer_raw_k5_seed251 | Default-thread brute | 0.121 | 2.274 | 2.408 | 2.231 |
| real_breast_cancer_raw_k5_seed251 | Threadpool-limited brute | 0.124 | 0.243 | 0.365 | 0.209 |
| real_breast_cancer_raw_k5_seed251 | sklearn auto | 0.134 | 2.143 | 2.279 | 2.235 |
| real_breast_cancer_raw_k5_seed252 | Current automatic | 0.052 | 0.181 | 0.232 | 0.135 |
| real_breast_cancer_raw_k5_seed252 | Default-thread brute | 0.109 | 2.185 | 2.325 | 2.232 |
| real_breast_cancer_raw_k5_seed252 | Threadpool-limited brute | 0.112 | 0.231 | 0.344 | 0.205 |
| real_breast_cancer_raw_k5_seed252 | sklearn auto | 0.111 | 2.233 | 2.340 | 2.288 |
| real_iris_raw_k5_seed151 | Current automatic | 0.042 | 0.053 | 0.093 | 0.041 |
| real_iris_raw_k5_seed151 | Default-thread brute | 0.103 | 1.563 | 1.666 | 1.587 |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 0.107 | 0.102 | 0.209 | 0.082 |
| real_iris_raw_k5_seed151 | sklearn auto | 0.149 | 0.175 | 0.324 | 0.154 |
| real_iris_raw_k5_seed251 | Current automatic | 0.042 | 0.054 | 0.099 | 0.043 |
| real_iris_raw_k5_seed251 | Default-thread brute | 0.109 | 0.659 | 0.778 | 0.696 |
| real_iris_raw_k5_seed251 | Threadpool-limited brute | 0.103 | 0.100 | 0.204 | 0.083 |
| real_iris_raw_k5_seed251 | sklearn auto | 0.137 | 0.171 | 0.308 | 0.154 |
| real_iris_raw_k5_seed252 | Current automatic | 0.056 | 0.062 | 0.122 | 0.045 |
| real_iris_raw_k5_seed252 | Default-thread brute | 0.124 | 0.757 | 0.879 | 0.834 |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 0.115 | 0.118 | 0.227 | 0.088 |
| real_iris_raw_k5_seed252 | sklearn auto | 0.158 | 0.202 | 0.361 | 0.161 |
| real_wine_raw_k5_seed151 | Current automatic | 0.043 | 0.066 | 0.106 | 0.052 |
| real_wine_raw_k5_seed151 | Default-thread brute | 0.108 | 0.716 | 0.831 | 0.718 |
| real_wine_raw_k5_seed151 | Threadpool-limited brute | 0.107 | 0.104 | 0.213 | 0.086 |
| real_wine_raw_k5_seed151 | sklearn auto | 0.158 | 0.192 | 0.350 | 0.159 |
| real_wine_raw_k5_seed251 | Current automatic | 0.039 | 0.058 | 0.096 | 0.051 |
| real_wine_raw_k5_seed251 | Default-thread brute | 0.095 | 0.559 | 0.657 | 0.570 |
| real_wine_raw_k5_seed251 | Threadpool-limited brute | 0.093 | 0.097 | 0.191 | 0.087 |
| real_wine_raw_k5_seed251 | sklearn auto | 0.137 | 0.171 | 0.311 | 0.153 |
| real_wine_raw_k5_seed252 | Current automatic | 0.036 | 0.058 | 0.095 | 0.050 |
| real_wine_raw_k5_seed252 | Default-thread brute | 0.093 | 0.531 | 0.622 | 0.575 |
| real_wine_raw_k5_seed252 | Threadpool-limited brute | 0.087 | 0.095 | 0.185 | 0.086 |
| real_wine_raw_k5_seed252 | sklearn auto | 0.129 | 0.160 | 0.292 | 0.149 |

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
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 0 | first | `/cases/6/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 0 | second | `/cases/6/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 1 | first | `/cases/6/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 1 | second | `/cases/6/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 2 | first | `/cases/6/variants/sklearn_1/samples/2` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 2 | second | `/cases/6/variants/sklearn_1/samples/2` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 3 | first | `/cases/6/variants/sklearn_1/samples/3` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 3 | second | `/cases/6/variants/sklearn_1/samples/3` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 4 | first | `/cases/6/variants/sklearn_1/samples/4` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 4 | second | `/cases/6/variants/sklearn_1/samples/4` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 5 | first | `/cases/6/variants/sklearn_1/samples/5` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 5 | second | `/cases/6/variants/sklearn_1/samples/5` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 6 | first | `/cases/6/variants/sklearn_1/samples/6` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 6 | second | `/cases/6/variants/sklearn_1/samples/6` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 7 | first | `/cases/6/variants/sklearn_1/samples/7` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 7 | second | `/cases/6/variants/sklearn_1/samples/7` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 8 | first | `/cases/6/variants/sklearn_1/samples/8` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 8 | second | `/cases/6/variants/sklearn_1/samples/8` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 9 | first | `/cases/6/variants/sklearn_1/samples/9` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 9 | second | `/cases/6/variants/sklearn_1/samples/9` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 10 | first | `/cases/6/variants/sklearn_1/samples/10` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 10 | second | `/cases/6/variants/sklearn_1/samples/10` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 11 | first | `/cases/6/variants/sklearn_1/samples/11` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 11 | second | `/cases/6/variants/sklearn_1/samples/11` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 12 | first | `/cases/6/variants/sklearn_1/samples/12` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 12 | second | `/cases/6/variants/sklearn_1/samples/12` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 13 | first | `/cases/6/variants/sklearn_1/samples/13` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 13 | second | `/cases/6/variants/sklearn_1/samples/13` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 14 | first | `/cases/6/variants/sklearn_1/samples/14` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 14 | second | `/cases/6/variants/sklearn_1/samples/14` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 15 | first | `/cases/6/variants/sklearn_1/samples/15` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 15 | second | `/cases/6/variants/sklearn_1/samples/15` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 16 | first | `/cases/6/variants/sklearn_1/samples/16` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 16 | second | `/cases/6/variants/sklearn_1/samples/16` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 17 | first | `/cases/6/variants/sklearn_1/samples/17` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 17 | second | `/cases/6/variants/sklearn_1/samples/17` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 18 | first | `/cases/6/variants/sklearn_1/samples/18` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 18 | second | `/cases/6/variants/sklearn_1/samples/18` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 19 | first | `/cases/6/variants/sklearn_1/samples/19` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 19 | second | `/cases/6/variants/sklearn_1/samples/19` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 20 | first | `/cases/6/variants/sklearn_1/samples/20` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 20 | second | `/cases/6/variants/sklearn_1/samples/20` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 21 | first | `/cases/6/variants/sklearn_1/samples/21` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 21 | second | `/cases/6/variants/sklearn_1/samples/21` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 22 | first | `/cases/6/variants/sklearn_1/samples/22` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 22 | second | `/cases/6/variants/sklearn_1/samples/22` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 23 | first | `/cases/6/variants/sklearn_1/samples/23` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 23 | second | `/cases/6/variants/sklearn_1/samples/23` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 24 | first | `/cases/6/variants/sklearn_1/samples/24` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 24 | second | `/cases/6/variants/sklearn_1/samples/24` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 25 | first | `/cases/6/variants/sklearn_1/samples/25` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 25 | second | `/cases/6/variants/sklearn_1/samples/25` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 26 | first | `/cases/6/variants/sklearn_1/samples/26` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 26 | second | `/cases/6/variants/sklearn_1/samples/26` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 27 | first | `/cases/6/variants/sklearn_1/samples/27` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 27 | second | `/cases/6/variants/sklearn_1/samples/27` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 28 | first | `/cases/6/variants/sklearn_1/samples/28` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 28 | second | `/cases/6/variants/sklearn_1/samples/28` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 29 | first | `/cases/6/variants/sklearn_1/samples/29` |
| real_digits_raw_k5_seed151 | Threadpool-limited brute | 29 | second | `/cases/6/variants/sklearn_1/samples/29` |
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
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 0 | first | `/cases/7/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 0 | second | `/cases/7/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 1 | first | `/cases/7/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 1 | second | `/cases/7/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 2 | first | `/cases/7/variants/sklearn_1/samples/2` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 2 | second | `/cases/7/variants/sklearn_1/samples/2` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 3 | first | `/cases/7/variants/sklearn_1/samples/3` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 3 | second | `/cases/7/variants/sklearn_1/samples/3` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 4 | first | `/cases/7/variants/sklearn_1/samples/4` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 4 | second | `/cases/7/variants/sklearn_1/samples/4` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 5 | first | `/cases/7/variants/sklearn_1/samples/5` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 5 | second | `/cases/7/variants/sklearn_1/samples/5` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 6 | first | `/cases/7/variants/sklearn_1/samples/6` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 6 | second | `/cases/7/variants/sklearn_1/samples/6` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 7 | first | `/cases/7/variants/sklearn_1/samples/7` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 7 | second | `/cases/7/variants/sklearn_1/samples/7` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 8 | first | `/cases/7/variants/sklearn_1/samples/8` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 8 | second | `/cases/7/variants/sklearn_1/samples/8` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 9 | first | `/cases/7/variants/sklearn_1/samples/9` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 9 | second | `/cases/7/variants/sklearn_1/samples/9` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 10 | first | `/cases/7/variants/sklearn_1/samples/10` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 10 | second | `/cases/7/variants/sklearn_1/samples/10` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 11 | first | `/cases/7/variants/sklearn_1/samples/11` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 11 | second | `/cases/7/variants/sklearn_1/samples/11` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 12 | first | `/cases/7/variants/sklearn_1/samples/12` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 12 | second | `/cases/7/variants/sklearn_1/samples/12` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 13 | first | `/cases/7/variants/sklearn_1/samples/13` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 13 | second | `/cases/7/variants/sklearn_1/samples/13` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 14 | first | `/cases/7/variants/sklearn_1/samples/14` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 14 | second | `/cases/7/variants/sklearn_1/samples/14` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 15 | first | `/cases/7/variants/sklearn_1/samples/15` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 15 | second | `/cases/7/variants/sklearn_1/samples/15` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 16 | first | `/cases/7/variants/sklearn_1/samples/16` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 16 | second | `/cases/7/variants/sklearn_1/samples/16` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 17 | first | `/cases/7/variants/sklearn_1/samples/17` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 17 | second | `/cases/7/variants/sklearn_1/samples/17` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 18 | first | `/cases/7/variants/sklearn_1/samples/18` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 18 | second | `/cases/7/variants/sklearn_1/samples/18` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 19 | first | `/cases/7/variants/sklearn_1/samples/19` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 19 | second | `/cases/7/variants/sklearn_1/samples/19` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 20 | first | `/cases/7/variants/sklearn_1/samples/20` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 20 | second | `/cases/7/variants/sklearn_1/samples/20` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 21 | first | `/cases/7/variants/sklearn_1/samples/21` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 21 | second | `/cases/7/variants/sklearn_1/samples/21` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 22 | first | `/cases/7/variants/sklearn_1/samples/22` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 22 | second | `/cases/7/variants/sklearn_1/samples/22` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 23 | first | `/cases/7/variants/sklearn_1/samples/23` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 23 | second | `/cases/7/variants/sklearn_1/samples/23` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 24 | first | `/cases/7/variants/sklearn_1/samples/24` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 24 | second | `/cases/7/variants/sklearn_1/samples/24` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 25 | first | `/cases/7/variants/sklearn_1/samples/25` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 25 | second | `/cases/7/variants/sklearn_1/samples/25` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 26 | first | `/cases/7/variants/sklearn_1/samples/26` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 26 | second | `/cases/7/variants/sklearn_1/samples/26` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 27 | first | `/cases/7/variants/sklearn_1/samples/27` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 27 | second | `/cases/7/variants/sklearn_1/samples/27` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 28 | first | `/cases/7/variants/sklearn_1/samples/28` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 28 | second | `/cases/7/variants/sklearn_1/samples/28` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 29 | first | `/cases/7/variants/sklearn_1/samples/29` |
| real_digits_raw_k5_seed251 | Threadpool-limited brute | 29 | second | `/cases/7/variants/sklearn_1/samples/29` |
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
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 0 | first | `/cases/8/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 0 | second | `/cases/8/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 1 | first | `/cases/8/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 1 | second | `/cases/8/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 2 | first | `/cases/8/variants/sklearn_1/samples/2` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 2 | second | `/cases/8/variants/sklearn_1/samples/2` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 3 | first | `/cases/8/variants/sklearn_1/samples/3` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 3 | second | `/cases/8/variants/sklearn_1/samples/3` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 4 | first | `/cases/8/variants/sklearn_1/samples/4` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 4 | second | `/cases/8/variants/sklearn_1/samples/4` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 5 | first | `/cases/8/variants/sklearn_1/samples/5` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 5 | second | `/cases/8/variants/sklearn_1/samples/5` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 6 | first | `/cases/8/variants/sklearn_1/samples/6` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 6 | second | `/cases/8/variants/sklearn_1/samples/6` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 7 | first | `/cases/8/variants/sklearn_1/samples/7` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 7 | second | `/cases/8/variants/sklearn_1/samples/7` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 8 | first | `/cases/8/variants/sklearn_1/samples/8` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 8 | second | `/cases/8/variants/sklearn_1/samples/8` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 9 | first | `/cases/8/variants/sklearn_1/samples/9` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 9 | second | `/cases/8/variants/sklearn_1/samples/9` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 10 | first | `/cases/8/variants/sklearn_1/samples/10` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 10 | second | `/cases/8/variants/sklearn_1/samples/10` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 11 | first | `/cases/8/variants/sklearn_1/samples/11` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 11 | second | `/cases/8/variants/sklearn_1/samples/11` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 12 | first | `/cases/8/variants/sklearn_1/samples/12` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 12 | second | `/cases/8/variants/sklearn_1/samples/12` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 13 | first | `/cases/8/variants/sklearn_1/samples/13` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 13 | second | `/cases/8/variants/sklearn_1/samples/13` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 14 | first | `/cases/8/variants/sklearn_1/samples/14` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 14 | second | `/cases/8/variants/sklearn_1/samples/14` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 15 | first | `/cases/8/variants/sklearn_1/samples/15` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 15 | second | `/cases/8/variants/sklearn_1/samples/15` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 16 | first | `/cases/8/variants/sklearn_1/samples/16` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 16 | second | `/cases/8/variants/sklearn_1/samples/16` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 17 | first | `/cases/8/variants/sklearn_1/samples/17` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 17 | second | `/cases/8/variants/sklearn_1/samples/17` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 18 | first | `/cases/8/variants/sklearn_1/samples/18` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 18 | second | `/cases/8/variants/sklearn_1/samples/18` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 19 | first | `/cases/8/variants/sklearn_1/samples/19` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 19 | second | `/cases/8/variants/sklearn_1/samples/19` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 20 | first | `/cases/8/variants/sklearn_1/samples/20` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 20 | second | `/cases/8/variants/sklearn_1/samples/20` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 21 | first | `/cases/8/variants/sklearn_1/samples/21` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 21 | second | `/cases/8/variants/sklearn_1/samples/21` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 22 | first | `/cases/8/variants/sklearn_1/samples/22` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 22 | second | `/cases/8/variants/sklearn_1/samples/22` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 23 | first | `/cases/8/variants/sklearn_1/samples/23` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 23 | second | `/cases/8/variants/sklearn_1/samples/23` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 24 | first | `/cases/8/variants/sklearn_1/samples/24` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 24 | second | `/cases/8/variants/sklearn_1/samples/24` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 25 | first | `/cases/8/variants/sklearn_1/samples/25` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 25 | second | `/cases/8/variants/sklearn_1/samples/25` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 26 | first | `/cases/8/variants/sklearn_1/samples/26` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 26 | second | `/cases/8/variants/sklearn_1/samples/26` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 27 | first | `/cases/8/variants/sklearn_1/samples/27` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 27 | second | `/cases/8/variants/sklearn_1/samples/27` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 28 | first | `/cases/8/variants/sklearn_1/samples/28` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 28 | second | `/cases/8/variants/sklearn_1/samples/28` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 29 | first | `/cases/8/variants/sklearn_1/samples/29` |
| real_digits_raw_k5_seed252 | Threadpool-limited brute | 29 | second | `/cases/8/variants/sklearn_1/samples/29` |
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
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 0 | first | `/cases/12/variants/sklearn_1/samples/0` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 0 | second | `/cases/12/variants/sklearn_1/samples/0` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 1 | first | `/cases/12/variants/sklearn_1/samples/1` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 1 | second | `/cases/12/variants/sklearn_1/samples/1` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 2 | first | `/cases/12/variants/sklearn_1/samples/2` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 2 | second | `/cases/12/variants/sklearn_1/samples/2` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 3 | first | `/cases/12/variants/sklearn_1/samples/3` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 3 | second | `/cases/12/variants/sklearn_1/samples/3` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 4 | first | `/cases/12/variants/sklearn_1/samples/4` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 4 | second | `/cases/12/variants/sklearn_1/samples/4` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 5 | first | `/cases/12/variants/sklearn_1/samples/5` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 5 | second | `/cases/12/variants/sklearn_1/samples/5` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 6 | first | `/cases/12/variants/sklearn_1/samples/6` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 6 | second | `/cases/12/variants/sklearn_1/samples/6` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 7 | first | `/cases/12/variants/sklearn_1/samples/7` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 7 | second | `/cases/12/variants/sklearn_1/samples/7` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 8 | first | `/cases/12/variants/sklearn_1/samples/8` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 8 | second | `/cases/12/variants/sklearn_1/samples/8` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 9 | first | `/cases/12/variants/sklearn_1/samples/9` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 9 | second | `/cases/12/variants/sklearn_1/samples/9` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 10 | first | `/cases/12/variants/sklearn_1/samples/10` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 10 | second | `/cases/12/variants/sklearn_1/samples/10` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 11 | first | `/cases/12/variants/sklearn_1/samples/11` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 11 | second | `/cases/12/variants/sklearn_1/samples/11` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 12 | first | `/cases/12/variants/sklearn_1/samples/12` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 12 | second | `/cases/12/variants/sklearn_1/samples/12` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 13 | first | `/cases/12/variants/sklearn_1/samples/13` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 13 | second | `/cases/12/variants/sklearn_1/samples/13` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 14 | first | `/cases/12/variants/sklearn_1/samples/14` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 14 | second | `/cases/12/variants/sklearn_1/samples/14` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 15 | first | `/cases/12/variants/sklearn_1/samples/15` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 15 | second | `/cases/12/variants/sklearn_1/samples/15` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 16 | first | `/cases/12/variants/sklearn_1/samples/16` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 16 | second | `/cases/12/variants/sklearn_1/samples/16` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 17 | first | `/cases/12/variants/sklearn_1/samples/17` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 17 | second | `/cases/12/variants/sklearn_1/samples/17` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 18 | first | `/cases/12/variants/sklearn_1/samples/18` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 18 | second | `/cases/12/variants/sklearn_1/samples/18` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 19 | first | `/cases/12/variants/sklearn_1/samples/19` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 19 | second | `/cases/12/variants/sklearn_1/samples/19` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 20 | first | `/cases/12/variants/sklearn_1/samples/20` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 20 | second | `/cases/12/variants/sklearn_1/samples/20` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 21 | first | `/cases/12/variants/sklearn_1/samples/21` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 21 | second | `/cases/12/variants/sklearn_1/samples/21` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 22 | first | `/cases/12/variants/sklearn_1/samples/22` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 22 | second | `/cases/12/variants/sklearn_1/samples/22` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 23 | first | `/cases/12/variants/sklearn_1/samples/23` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 23 | second | `/cases/12/variants/sklearn_1/samples/23` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 24 | first | `/cases/12/variants/sklearn_1/samples/24` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 24 | second | `/cases/12/variants/sklearn_1/samples/24` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 25 | first | `/cases/12/variants/sklearn_1/samples/25` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 25 | second | `/cases/12/variants/sklearn_1/samples/25` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 26 | first | `/cases/12/variants/sklearn_1/samples/26` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 26 | second | `/cases/12/variants/sklearn_1/samples/26` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 27 | first | `/cases/12/variants/sklearn_1/samples/27` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 27 | second | `/cases/12/variants/sklearn_1/samples/27` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 28 | first | `/cases/12/variants/sklearn_1/samples/28` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 28 | second | `/cases/12/variants/sklearn_1/samples/28` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 29 | first | `/cases/12/variants/sklearn_1/samples/29` |
| real_iris_raw_k5_seed151 | Threadpool-limited brute | 29 | second | `/cases/12/variants/sklearn_1/samples/29` |
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
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 0 | first | `/cases/14/variants/sklearn_1/samples/0` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 0 | second | `/cases/14/variants/sklearn_1/samples/0` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 1 | first | `/cases/14/variants/sklearn_1/samples/1` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 1 | second | `/cases/14/variants/sklearn_1/samples/1` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 2 | first | `/cases/14/variants/sklearn_1/samples/2` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 2 | second | `/cases/14/variants/sklearn_1/samples/2` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 3 | first | `/cases/14/variants/sklearn_1/samples/3` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 3 | second | `/cases/14/variants/sklearn_1/samples/3` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 4 | first | `/cases/14/variants/sklearn_1/samples/4` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 4 | second | `/cases/14/variants/sklearn_1/samples/4` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 5 | first | `/cases/14/variants/sklearn_1/samples/5` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 5 | second | `/cases/14/variants/sklearn_1/samples/5` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 6 | first | `/cases/14/variants/sklearn_1/samples/6` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 6 | second | `/cases/14/variants/sklearn_1/samples/6` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 7 | first | `/cases/14/variants/sklearn_1/samples/7` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 7 | second | `/cases/14/variants/sklearn_1/samples/7` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 8 | first | `/cases/14/variants/sklearn_1/samples/8` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 8 | second | `/cases/14/variants/sklearn_1/samples/8` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 9 | first | `/cases/14/variants/sklearn_1/samples/9` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 9 | second | `/cases/14/variants/sklearn_1/samples/9` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 10 | first | `/cases/14/variants/sklearn_1/samples/10` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 10 | second | `/cases/14/variants/sklearn_1/samples/10` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 11 | first | `/cases/14/variants/sklearn_1/samples/11` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 11 | second | `/cases/14/variants/sklearn_1/samples/11` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 12 | first | `/cases/14/variants/sklearn_1/samples/12` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 12 | second | `/cases/14/variants/sklearn_1/samples/12` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 13 | first | `/cases/14/variants/sklearn_1/samples/13` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 13 | second | `/cases/14/variants/sklearn_1/samples/13` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 14 | first | `/cases/14/variants/sklearn_1/samples/14` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 14 | second | `/cases/14/variants/sklearn_1/samples/14` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 15 | first | `/cases/14/variants/sklearn_1/samples/15` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 15 | second | `/cases/14/variants/sklearn_1/samples/15` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 16 | first | `/cases/14/variants/sklearn_1/samples/16` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 16 | second | `/cases/14/variants/sklearn_1/samples/16` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 17 | first | `/cases/14/variants/sklearn_1/samples/17` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 17 | second | `/cases/14/variants/sklearn_1/samples/17` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 18 | first | `/cases/14/variants/sklearn_1/samples/18` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 18 | second | `/cases/14/variants/sklearn_1/samples/18` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 19 | first | `/cases/14/variants/sklearn_1/samples/19` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 19 | second | `/cases/14/variants/sklearn_1/samples/19` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 20 | first | `/cases/14/variants/sklearn_1/samples/20` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 20 | second | `/cases/14/variants/sklearn_1/samples/20` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 21 | first | `/cases/14/variants/sklearn_1/samples/21` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 21 | second | `/cases/14/variants/sklearn_1/samples/21` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 22 | first | `/cases/14/variants/sklearn_1/samples/22` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 22 | second | `/cases/14/variants/sklearn_1/samples/22` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 23 | first | `/cases/14/variants/sklearn_1/samples/23` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 23 | second | `/cases/14/variants/sklearn_1/samples/23` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 24 | first | `/cases/14/variants/sklearn_1/samples/24` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 24 | second | `/cases/14/variants/sklearn_1/samples/24` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 25 | first | `/cases/14/variants/sklearn_1/samples/25` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 25 | second | `/cases/14/variants/sklearn_1/samples/25` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 26 | first | `/cases/14/variants/sklearn_1/samples/26` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 26 | second | `/cases/14/variants/sklearn_1/samples/26` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 27 | first | `/cases/14/variants/sklearn_1/samples/27` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 27 | second | `/cases/14/variants/sklearn_1/samples/27` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 28 | first | `/cases/14/variants/sklearn_1/samples/28` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 28 | second | `/cases/14/variants/sklearn_1/samples/28` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 29 | first | `/cases/14/variants/sklearn_1/samples/29` |
| real_iris_raw_k5_seed252 | Threadpool-limited brute | 29 | second | `/cases/14/variants/sklearn_1/samples/29` |
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
| Threadpool-limited brute | {"first_queries": 540, "first_strict_passes": 390, "models_requested": 540, "second_queries": 540, "second_strict_passes": 390} | {"first/sklearn": 540, "second/sklearn": 540} | {"False": 540} | {"True": 540} |
| sklearn auto | {"first_queries": 540, "first_strict_passes": 390, "models_requested": 540, "second_queries": 540, "second_strict_passes": 390} | {"first/sklearn": 540, "second/sklearn": 540} | {"False": 540} | {"True": 540} |

## Evidence

[Raw requests, setup, source/runtime identities, and all audits](../benchmarks/results/sklearn-refresh-2026-10-31/knn-first-query.json)

[Validated summary](../benchmarks/results/sklearn-refresh-2026-10-31/knn-first-query-summary.json)
