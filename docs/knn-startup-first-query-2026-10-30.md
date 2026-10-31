# Fresh-model first queries in initialized processes

Initialized isolated persistent processes; normal NumPy/estimator imports and configuration are recorded separately. No model or query prewarm. Every request constructs and fits a new estimator, then times its first query and separately its second query. Public NumPy results establish completion; no harness MLX import/device probe/synchronization runs before the first query. Both outputs are audited after both timers. All requested rounds (default 30), including first model per fixture and first model in worker, remain visible. Earlier models may initialize runtime/compiler caches. This is model-first-use in an initialized process, not fresh-process or machine-cold latency. Five-worker Williams order has a complete balanced 10-round block, repeated three times by default. Any incomplete final block is also retained. No classification/regression operations are added here; the separate full broad and preservation control panels retain those operations and fixtures.

First query includes model preparation deferred by fit. Second query reuses only that fresh model. All rounds remain in the headline median; first-per-fixture and subsequent-model views are shown separately, not selected according to speed. Process initialization/import cost is separate, and must not be confused with the dedicated fresh-process cold benchmark.

Acceptance: **True**. 18 fixtures × 30 fresh models × 5 workers; each model has exactly one retained first and one retained second query. No numerical audit failure removes a valid timing. Ratios are comparator/current; bold values below 1× are slower observations. These descriptive ratios are not confidence intervals.

## Initial process setup (once per worker)

| Worker | NumPy import ms | Estimator import/setup ms | Worker entry→initialized ms | Parent process→initialized ms |
| --- | --- | --- | --- | --- |
| Current automatic | 25.236 | 651.819 | 678.942 | 703.537 |
| Default-thread brute | 24.811 | 767.918 | 794.572 | 817.410 |
| Single-thread brute | 24.923 | 776.651 | 803.488 | 826.328 |
| sklearn auto | 25.732 | 763.369 | 791.160 | 814.841 |
| Preceding automatic | 25.300 | 1053.303 | 1080.602 | 1104.751 |

## fit_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.082 | 0.110 / 1.334× / 1.399× | 0.109 / 1.321× / 1.281× | 1.499 / 18.221× / 17.373× | 0.124 / 1.513× / 1.482× |
| Gaussian 32D, single query | 49 | 0.095 | 0.131 / 1.388× / 1.330× | 0.129 / 1.365× / 1.341× | 0.134 / 1.421× / 1.331× | 0.138 / 1.456× / 1.430× |
| Gaussian 8D, single query | 42 | 0.073 | 0.100 / 1.363× / 1.356× | 0.103 / 1.405× / 1.337× | 1.528 / 20.868× / 19.962× | 0.116 / 1.585× / 1.614× |
| Gaussian 32D, single query | 50 | 0.093 | 0.131 / 1.407× / 1.367× | 0.130 / 1.393× / 1.438× | 0.128 / 1.374× / 1.357× | 0.135 / 1.448× / 1.362× |
| Gaussian 8D, single query | 43 | 0.091 | 0.126 / 1.383× / 1.333× | 0.121 / 1.329× / 1.330× | 1.555 / 17.133× / 17.081× | 0.140 / 1.545× / 1.523× |
| Gaussian 32D, single query | 51 | 0.101 | 0.131 / 1.302× / 1.318× | 0.131 / 1.306× / 1.378× | 0.133 / 1.320× / 1.309× | 0.133 / 1.324× / 1.320× |
| Digits (raw) | 151 | 0.078 | 0.140 / 1.797× / 1.980× | 0.146 / 1.871× / 1.918× | 0.146 / 1.874× / 1.895× | 0.124 / 1.595× / 1.587× |
| Digits (raw) | 251 | 0.075 | 0.145 / 1.938× / 2.009× | 0.149 / 1.995× / 2.072× | 0.141 / 1.891× / 1.958× | 0.117 / 1.561× / 1.558× |
| Digits (raw) | 252 | 0.076 | 0.148 / 1.930× / 1.943× | 0.148 / 1.932× / 1.960× | 0.141 / 1.847× / 2.023× | 0.124 / 1.618× / 1.706× |
| Breast Cancer (raw) | 151 | 0.059 | 0.120 / 2.031× / 2.015× | 0.125 / 2.118× / 2.186× | 0.121 / 2.060× / 2.120× | 0.102 / 1.729× / 1.875× |
| Breast Cancer (raw) | 251 | 0.043 | 0.101 / 2.354× / 2.405× | 0.101 / 2.354× / 2.342× | 0.097 / 2.243× / 2.284× | 0.082 / 1.899× / 1.814× |
| Breast Cancer (raw) | 252 | 0.040 | 0.100 / 2.534× / 2.365× | 0.102 / 2.573× / 2.345× | 0.102 / 2.573× / 2.486× | 0.076 / 1.918× / 1.815× |
| Iris (raw) | 151 | 0.041 | 0.108 / 2.606× / 2.452× | 0.105 / 2.533× / 2.392× | 0.154 / 3.725× / 3.409× | 0.086 / 2.087× / 1.943× |
| Iris (raw) | 251 | 0.036 | 0.101 / 2.790× / 2.561× | 0.093 / 2.588× / 2.445× | 0.132 / 3.646× / 3.432× | 0.071 / 1.972× / 1.967× |
| Iris (raw) | 252 | 0.035 | 0.094 / 2.714× / 2.625× | 0.089 / 2.557× / 2.539× | 0.124 / 3.569× / 3.611× | 0.069 / 1.995× / 2.014× |
| Wine (raw) | 151 | 0.034 | 0.093 / 2.721× / 2.673× | 0.083 / 2.434× / 2.560× | 0.123 / 3.615× / 3.623× | 0.068 / 1.985× / 2.057× |
| Wine (raw) | 251 | 0.033 | 0.090 / 2.700× / 2.652× | 0.085 / 2.544× / 2.529× | 0.124 / 3.720× / 3.572× | 0.066 / 1.959× / 1.935× |
| Wine (raw) | 252 | 0.031 | 0.089 / 2.819× / 2.738× | 0.084 / 2.658× / 2.545× | 0.123 / 3.902× / 3.874× | 0.066 / 2.081× / 2.086× |

## first_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.067 | 0.695 / 10.352× / 11.130× | 0.206 / 3.063× / 3.171× | 0.172 / 2.570× / 2.661× | 0.073 / 1.090× / 1.135× |
| Gaussian 32D, single query | 49 | 0.121 | 2.679 / 22.224× / 22.399× | 0.218 / 1.812× / 1.806× | 2.688 / 22.307× / 22.494× | 0.125 / 1.036× / 1.050× |
| Gaussian 8D, single query | 42 | 0.065 | 2.571 / 39.760× / 38.594× | 0.204 / 3.155× / 3.187× | 0.173 / 2.668× / 2.685× | 0.072 / 1.114× / 1.139× |
| Gaussian 32D, single query | 50 | 0.117 | 2.752 / 23.559× / 22.630× | 0.221 / 1.890× / 1.870× | 2.675 / 22.905× / 22.788× | 0.122 / 1.048× / 1.036× |
| Gaussian 8D, single query | 43 | 0.069 | 2.670 / 38.585× / 36.729× | 0.223 / 3.227× / 3.197× | 0.194 / 2.806× / 2.856× | 0.081 / 1.164× / 1.167× |
| Gaussian 32D, single query | 51 | 0.128 | 2.707 / 21.092× / 21.164× | 0.221 / 1.725× / 1.724× | 2.681 / 20.889× / 20.694× | 0.131 / 1.019× / 1.037× |
| Digits (raw) | 151 | 0.481 | 2.851 / 5.927× / 5.850× | 1.101 / 2.289× / 2.282× | 2.931 / 6.091× / 6.385× | 0.540 / 1.122× / 1.116× |
| Digits (raw) | 251 | 0.487 | 2.875 / 5.908× / 6.018× | 1.108 / 2.278× / 2.267× | 2.913 / 5.985× / 5.904× | 0.542 / 1.114× / 1.100× |
| Digits (raw) | 252 | 0.483 | 2.999 / 6.207× / 6.076× | 1.094 / 2.264× / 2.254× | 2.898 / 5.996× / 6.000× | 0.542 / 1.122× / 1.109× |
| Breast Cancer (raw) | 151 | 0.193 | 2.196 / 11.381× / 11.547× | 0.257 / 1.334× / 1.346× | 2.111 / 10.941× / 11.458× | 0.233 / 1.207× / 1.230× |
| Breast Cancer (raw) | 251 | 0.178 | 2.023 / 11.379× / 11.268× | 0.236 / 1.327× / 1.296× | 1.992 / 11.206× / 11.472× | 0.219 / 1.234× / 1.244× |
| Breast Cancer (raw) | 252 | 0.170 | 2.085 / 12.300× / 11.961× | 0.232 / 1.370× / 1.353× | 2.002 / 11.811× / 11.655× | 0.205 / 1.210× / 1.195× |
| Iris (raw) | 151 | 0.053 | 1.491 / 28.057× / 28.889× | 0.101 / 1.898× / 1.933× | 0.164 / 3.080× / 3.274× | 0.060 / 1.137× / 1.080× |
| Iris (raw) | 251 | 0.052 | 0.599 / 11.433× / 11.179× | 0.095 / 1.805× / 1.830× | 0.168 / 3.206× / 3.274× | 0.056 / 1.078× / 1.077× |
| Iris (raw) | 252 | 0.047 | 0.644 / 13.668× / 13.243× | 0.087 / 1.836× / 1.886× | 0.154 / 3.260× / 3.375× | 0.051 / 1.072× / 1.107× |
| Wine (raw) | 151 | 0.056 | 0.529 / 9.392× / 9.701× | 0.091 / 1.607× / 1.677× | 0.157 / 2.781× / 2.716× | 0.065 / 1.146× / 1.083× |
| Wine (raw) | 251 | 0.057 | 0.555 / 9.756× / 9.687× | 0.090 / 1.583× / 1.636× | 0.152 / 2.678× / 2.744× | 0.060 / 1.057× / 1.077× |
| Wine (raw) | 252 | 0.055 | 0.519 / 9.480× / 9.868× | 0.091 / 1.664× / 1.602× | 0.152 / 2.769× / 2.729× | 0.061 / 1.113× / 1.122× |

## fit_first_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.150 | 0.808 / 5.397× / 5.871× | 0.313 / 2.093× / 2.090× | 1.670 / 11.153× / 10.656× | 0.197 / 1.313× / 1.309× |
| Gaussian 32D, single query | 49 | 0.215 | 2.808 / 13.089× / 13.233× | 0.349 / 1.626× / 1.611× | 2.817 / 13.128× / 13.156× | 0.261 / 1.218× / 1.202× |
| Gaussian 8D, single query | 42 | 0.135 | 2.657 / 19.758× / 17.826× | 0.308 / 2.288× / 2.262× | 1.697 / 12.615× / 12.129× | 0.190 / 1.410× / 1.402× |
| Gaussian 32D, single query | 50 | 0.211 | 2.867 / 13.561× / 12.860× | 0.353 / 1.670× / 1.679× | 2.815 / 13.313× / 13.294× | 0.258 / 1.219× / 1.176× |
| Gaussian 8D, single query | 43 | 0.158 | 2.785 / 17.669× / 14.986× | 0.346 / 2.194× / 2.119× | 1.746 / 11.074× / 10.853× | 0.220 / 1.393× / 1.362× |
| Gaussian 32D, single query | 51 | 0.229 | 2.839 / 12.411× / 12.394× | 0.353 / 1.542× / 1.583× | 2.801 / 12.248× / 12.157× | 0.263 / 1.152× / 1.163× |
| Digits (raw) | 151 | 0.547 | 3.013 / 5.507× / 5.358× | 1.252 / 2.288× / 2.313× | 3.067 / 5.604× / 5.756× | 0.667 / 1.219× / 1.168× |
| Digits (raw) | 251 | 0.553 | 3.014 / 5.448× / 5.470× | 1.260 / 2.277× / 2.281× | 3.037 / 5.489× / 5.358× | 0.653 / 1.179× / 1.156× |
| Digits (raw) | 252 | 0.562 | 3.148 / 5.601× / 5.515× | 1.232 / 2.193× / 2.234× | 3.038 / 5.406× / 5.420× | 0.665 / 1.183× / 1.199× |
| Breast Cancer (raw) | 151 | 0.242 | 2.310 / 9.531× / 9.627× | 0.382 / 1.576× / 1.547× | 2.220 / 9.161× / 9.189× | 0.334 / 1.380× / 1.384× |
| Breast Cancer (raw) | 251 | 0.225 | 2.122 / 9.441× / 9.307× | 0.339 / 1.510× / 1.503× | 2.103 / 9.358× / 9.860× | 0.300 / 1.336× / 1.355× |
| Breast Cancer (raw) | 252 | 0.206 | 2.191 / 10.612× / 9.886× | 0.333 / 1.613× / 1.560× | 2.104 / 10.189× / 9.806× | 0.284 / 1.374× / 1.305× |
| Iris (raw) | 151 | 0.093 | 1.586 / 17.114× / 17.044× | 0.206 / 2.219× / 2.179× | 0.323 / 3.485× / 3.455× | 0.146 / 1.574× / 1.453× |
| Iris (raw) | 251 | 0.087 | 0.693 / 7.963× / 7.759× | 0.187 / 2.146× / 2.157× | 0.300 / 3.447× / 3.433× | 0.128 / 1.466× / 1.437× |
| Iris (raw) | 252 | 0.083 | 0.730 / 8.755× / 9.132× | 0.177 / 2.121× / 2.150× | 0.280 / 3.363× / 3.368× | 0.120 / 1.443× / 1.486× |
| Wine (raw) | 151 | 0.092 | 0.624 / 6.784× / 6.988× | 0.173 / 1.884× / 2.012× | 0.280 / 3.050× / 3.025× | 0.133 / 1.445× / 1.433× |
| Wine (raw) | 251 | 0.090 | 0.653 / 7.292× / 7.218× | 0.175 / 1.955× / 1.983× | 0.279 / 3.109× / 3.054× | 0.126 / 1.406× / 1.401× |
| Wine (raw) | 252 | 0.086 | 0.615 / 7.176× / 7.368× | 0.175 / 2.048× / 1.959× | 0.279 / 3.252× / 3.108× | 0.126 / 1.473× / 1.488× |

## second_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.057 | 0.841 / 14.828× / 13.938× | 0.189 / 3.333× / 3.375× | 0.141 / 2.486× / 2.553× | 0.065 / 1.146× / 1.170× |
| Gaussian 32D, single query | 49 | 0.111 | 2.721 / 24.481× / 24.387× | 0.203 / 1.824× / 1.811× | 2.742 / 24.667× / 24.142× | 0.116 / 1.041× / 1.039× |
| Gaussian 8D, single query | 42 | 0.056 | 2.616 / 46.990× / 45.710× | 0.186 / 3.338× / 3.353× | 0.143 / 2.565× / 2.551× | 0.064 / 1.143× / 1.144× |
| Gaussian 32D, single query | 50 | 0.109 | 2.712 / 24.851× / 24.852× | 0.203 / 1.858× / 1.856× | 2.651 / 24.295× / 24.272× | 0.114 / 1.047× / 1.044× |
| Gaussian 8D, single query | 43 | 0.057 | 2.708 / 47.725× / 45.022× | 0.195 / 3.434× / 3.408× | 0.150 / 2.641× / 2.608× | 0.066 / 1.170× / 1.173× |
| Gaussian 32D, single query | 51 | 0.119 | 2.698 / 22.731× / 22.808× | 0.202 / 1.703× / 1.723× | 2.743 / 23.106× / 23.162× | 0.121 / 1.022× / 1.021× |
| Digits (raw) | 151 | 0.321 | 2.895 / 9.021× / 9.082× | 1.046 / 3.260× / 3.264× | 2.983 / 9.297× / 9.553× | 0.394 / 1.227× / 1.257× |
| Digits (raw) | 251 | 0.328 | 2.898 / 8.849× / 8.977× | 1.054 / 3.218× / 3.205× | 2.830 / 8.639× / 8.853× | 0.398 / 1.215× / 1.227× |
| Digits (raw) | 252 | 0.322 | 2.974 / 9.242× / 9.214× | 1.042 / 3.238× / 3.251× | 2.865 / 8.901× / 9.101× | 0.394 / 1.225× / 1.255× |
| Breast Cancer (raw) | 151 | 0.138 | 2.192 / 15.933× / 16.335× | 0.216 / 1.571× / 1.611× | 2.120 / 15.406× / 15.619× | 0.181 / 1.316× / 1.356× |
| Breast Cancer (raw) | 251 | 0.128 | 2.105 / 16.507× / 16.115× | 0.212 / 1.664× / 1.621× | 1.992 / 15.621× / 15.359× | 0.168 / 1.321× / 1.335× |
| Breast Cancer (raw) | 252 | 0.133 | 2.183 / 16.398× / 16.159× | 0.208 / 1.565× / 1.559× | 2.051 / 15.409× / 15.584× | 0.172 / 1.293× / 1.306× |
| Iris (raw) | 151 | 0.040 | 1.553 / 38.781× / 39.627× | 0.076 / 1.909× / 1.952× | 0.137 / 3.429× / 3.434× | 0.046 / 1.152× / 1.151× |
| Iris (raw) | 251 | 0.040 | 0.609 / 15.180× / 15.442× | 0.077 / 1.916× / 1.943× | 0.136 / 3.377× / 3.527× | 0.046 / 1.158× / 1.156× |
| Iris (raw) | 252 | 0.037 | 0.612 / 16.737× / 16.352× | 0.072 / 1.973× / 1.980× | 0.133 / 3.635× / 3.660× | 0.043 / 1.165× / 1.161× |
| Wine (raw) | 151 | 0.047 | 0.563 / 11.856× / 12.048× | 0.080 / 1.685× / 1.702× | 0.137 / 2.881× / 2.886× | 0.054 / 1.144× / 1.164× |
| Wine (raw) | 251 | 0.047 | 0.608 / 12.919× / 12.811× | 0.079 / 1.686× / 1.693× | 0.136 / 2.882× / 2.937× | 0.053 / 1.132× / 1.135× |
| Wine (raw) | 252 | 0.046 | 0.582 / 12.539× / 12.475× | 0.079 / 1.711× / 1.702× | 0.136 / 2.919× / 2.920× | 0.054 / 1.155× / 1.163× |

## Every slower or missing all-round median

No slower or missing all-round medians.

## first_model_for_fixture

All configurations and metrics are retained in this separate view. Earlier fixtures may already have initialized process/runtime/compiler state; this table is not fresh-process cold latency.

| Case | Worker | fit_ms median ms | first_query_ms median ms | fit_first_query_ms median ms | second_query_ms median ms |
| --- | --- | --- | --- | --- | --- |
| shape_single_8_k5_seed41 | Current automatic | 0.251 | 0.115 | 0.366 | 0.085 |
| shape_single_8_k5_seed41 | Default-thread brute | 0.384 | 19.873 | 20.257 | 1.142 |
| shape_single_8_k5_seed41 | Single-thread brute | 0.314 | 8.450 | 8.764 | 0.548 |
| shape_single_8_k5_seed41 | sklearn auto | 1.996 | 0.303 | 2.300 | 0.186 |
| shape_single_8_k5_seed41 | Preceding automatic | 0.306 | 0.133 | 0.439 | 0.107 |
| shape_single_32_k5_seed49 | Current automatic | 0.142 | 0.126 | 0.268 | 0.112 |
| shape_single_32_k5_seed49 | Default-thread brute | 0.139 | 0.818 | 0.956 | 1.015 |
| shape_single_32_k5_seed49 | Single-thread brute | 0.131 | 0.224 | 0.355 | 0.203 |
| shape_single_32_k5_seed49 | sklearn auto | 0.135 | 16.271 | 16.406 | 2.935 |
| shape_single_32_k5_seed49 | Preceding automatic | 0.168 | 0.139 | 0.307 | 0.125 |
| shape_single_8_k5_seed42 | Current automatic | 0.114 | 0.077 | 0.191 | 0.059 |
| shape_single_8_k5_seed42 | Default-thread brute | 0.130 | 1.840 | 1.971 | 2.780 |
| shape_single_8_k5_seed42 | Single-thread brute | 0.113 | 0.222 | 0.335 | 0.198 |
| shape_single_8_k5_seed42 | sklearn auto | 1.581 | 0.309 | 1.890 | 0.154 |
| shape_single_8_k5_seed42 | Preceding automatic | 0.163 | 0.081 | 0.244 | 0.066 |
| shape_single_32_k5_seed50 | Current automatic | 0.138 | 0.124 | 0.262 | 0.111 |
| shape_single_32_k5_seed50 | Default-thread brute | 0.150 | 2.758 | 2.908 | 2.834 |
| shape_single_32_k5_seed50 | Single-thread brute | 0.139 | 0.252 | 0.390 | 0.285 |
| shape_single_32_k5_seed50 | sklearn auto | 0.125 | 2.737 | 2.862 | 2.615 |
| shape_single_32_k5_seed50 | Preceding automatic | 0.155 | 0.125 | 0.280 | 0.113 |
| shape_single_8_k5_seed43 | Current automatic | 0.098 | 0.070 | 0.168 | 0.057 |
| shape_single_8_k5_seed43 | Default-thread brute | 0.134 | 2.597 | 2.732 | 2.926 |
| shape_single_8_k5_seed43 | Single-thread brute | 0.128 | 0.227 | 0.354 | 0.194 |
| shape_single_8_k5_seed43 | sklearn auto | 1.560 | 0.212 | 1.773 | 0.151 |
| shape_single_8_k5_seed43 | Preceding automatic | 0.143 | 0.085 | 0.228 | 0.072 |
| shape_single_32_k5_seed51 | Current automatic | 0.109 | 0.129 | 0.238 | 0.118 |
| shape_single_32_k5_seed51 | Default-thread brute | 0.153 | 1.049 | 1.202 | 0.540 |
| shape_single_32_k5_seed51 | Single-thread brute | 0.140 | 0.225 | 0.365 | 0.206 |
| shape_single_32_k5_seed51 | sklearn auto | 0.135 | 2.796 | 2.931 | 2.803 |
| shape_single_32_k5_seed51 | Preceding automatic | 0.143 | 0.134 | 0.276 | 0.120 |
| real_digits_raw_k5_seed151 | Current automatic | 0.072 | 0.595 | 0.667 | 0.330 |
| real_digits_raw_k5_seed151 | Default-thread brute | 0.133 | 2.940 | 3.073 | 3.180 |
| real_digits_raw_k5_seed151 | Single-thread brute | 0.145 | 1.142 | 1.288 | 1.055 |
| real_digits_raw_k5_seed151 | sklearn auto | 0.156 | 3.150 | 3.307 | 2.845 |
| real_digits_raw_k5_seed151 | Preceding automatic | 0.111 | 0.666 | 0.777 | 0.432 |
| real_digits_raw_k5_seed251 | Current automatic | 0.074 | 0.523 | 0.597 | 0.326 |
| real_digits_raw_k5_seed251 | Default-thread brute | 0.164 | 2.770 | 2.934 | 2.835 |
| real_digits_raw_k5_seed251 | Single-thread brute | 0.167 | 1.140 | 1.307 | 1.105 |
| real_digits_raw_k5_seed251 | sklearn auto | 0.163 | 2.959 | 3.122 | 3.029 |
| real_digits_raw_k5_seed251 | Preceding automatic | 0.131 | 0.564 | 0.695 | 0.397 |
| real_digits_raw_k5_seed252 | Current automatic | 0.071 | 0.514 | 0.586 | 0.322 |
| real_digits_raw_k5_seed252 | Default-thread brute | 0.155 | 3.101 | 3.256 | 3.029 |
| real_digits_raw_k5_seed252 | Single-thread brute | 0.171 | 1.136 | 1.308 | 1.053 |
| real_digits_raw_k5_seed252 | sklearn auto | 0.153 | 2.799 | 2.952 | 3.024 |
| real_digits_raw_k5_seed252 | Preceding automatic | 0.115 | 0.554 | 0.670 | 0.388 |
| real_breast_cancer_raw_k5_seed151 | Current automatic | 0.073 | 0.211 | 0.284 | 0.143 |
| real_breast_cancer_raw_k5_seed151 | Default-thread brute | 0.165 | 1.846 | 2.010 | 2.459 |
| real_breast_cancer_raw_k5_seed151 | Single-thread brute | 0.157 | 0.284 | 0.441 | 0.221 |
| real_breast_cancer_raw_k5_seed151 | sklearn auto | 0.143 | 1.983 | 2.126 | 2.085 |
| real_breast_cancer_raw_k5_seed151 | Preceding automatic | 0.119 | 0.258 | 0.378 | 0.190 |
| real_breast_cancer_raw_k5_seed251 | Current automatic | 0.065 | 0.221 | 0.285 | 0.133 |
| real_breast_cancer_raw_k5_seed251 | Default-thread brute | 0.159 | 2.029 | 2.188 | 2.395 |
| real_breast_cancer_raw_k5_seed251 | Single-thread brute | 0.144 | 0.267 | 0.411 | 0.213 |
| real_breast_cancer_raw_k5_seed251 | sklearn auto | 0.150 | 2.183 | 2.333 | 2.032 |
| real_breast_cancer_raw_k5_seed251 | Preceding automatic | 0.117 | 0.260 | 0.377 | 0.199 |
| real_breast_cancer_raw_k5_seed252 | Current automatic | 0.055 | 0.203 | 0.258 | 0.142 |
| real_breast_cancer_raw_k5_seed252 | Default-thread brute | 0.146 | 2.217 | 2.363 | 2.200 |
| real_breast_cancer_raw_k5_seed252 | Single-thread brute | 0.133 | 0.256 | 0.389 | 0.211 |
| real_breast_cancer_raw_k5_seed252 | sklearn auto | 0.132 | 2.091 | 2.222 | 2.000 |
| real_breast_cancer_raw_k5_seed252 | Preceding automatic | 0.105 | 0.229 | 0.334 | 0.178 |
| real_iris_raw_k5_seed151 | Current automatic | 0.060 | 0.064 | 0.124 | 0.045 |
| real_iris_raw_k5_seed151 | Default-thread brute | 0.139 | 1.572 | 1.711 | 1.484 |
| real_iris_raw_k5_seed151 | Single-thread brute | 0.136 | 0.138 | 0.274 | 0.080 |
| real_iris_raw_k5_seed151 | sklearn auto | 0.192 | 0.232 | 0.425 | 0.151 |
| real_iris_raw_k5_seed151 | Preceding automatic | 0.112 | 0.069 | 0.181 | 0.052 |
| real_iris_raw_k5_seed251 | Current automatic | 0.063 | 0.061 | 0.124 | 0.041 |
| real_iris_raw_k5_seed251 | Default-thread brute | 0.150 | 0.816 | 0.966 | 0.682 |
| real_iris_raw_k5_seed251 | Single-thread brute | 0.145 | 0.143 | 0.288 | 0.081 |
| real_iris_raw_k5_seed251 | sklearn auto | 0.203 | 0.290 | 0.493 | 0.164 |
| real_iris_raw_k5_seed251 | Preceding automatic | 0.112 | 0.069 | 0.181 | 0.053 |
| real_iris_raw_k5_seed252 | Current automatic | 0.062 | 0.060 | 0.122 | 0.040 |
| real_iris_raw_k5_seed252 | Default-thread brute | 0.128 | 0.493 | 0.622 | 0.626 |
| real_iris_raw_k5_seed252 | Single-thread brute | 0.136 | 0.120 | 0.256 | 0.078 |
| real_iris_raw_k5_seed252 | sklearn auto | 0.175 | 0.191 | 0.365 | 0.138 |
| real_iris_raw_k5_seed252 | Preceding automatic | 0.116 | 0.065 | 0.180 | 0.046 |
| real_wine_raw_k5_seed151 | Current automatic | 0.062 | 0.071 | 0.133 | 0.051 |
| real_wine_raw_k5_seed151 | Default-thread brute | 0.133 | 0.810 | 0.943 | 0.687 |
| real_wine_raw_k5_seed151 | Single-thread brute | 0.133 | 0.149 | 0.281 | 0.088 |
| real_wine_raw_k5_seed151 | sklearn auto | 0.215 | 0.225 | 0.440 | 0.149 |
| real_wine_raw_k5_seed151 | Preceding automatic | 0.103 | 0.073 | 0.177 | 0.054 |
| real_wine_raw_k5_seed251 | Current automatic | 0.059 | 0.070 | 0.129 | 0.056 |
| real_wine_raw_k5_seed251 | Default-thread brute | 0.146 | 0.547 | 0.692 | 0.520 |
| real_wine_raw_k5_seed251 | Single-thread brute | 0.140 | 0.133 | 0.273 | 0.088 |
| real_wine_raw_k5_seed251 | sklearn auto | 0.208 | 0.224 | 0.432 | 0.158 |
| real_wine_raw_k5_seed251 | Preceding automatic | 0.096 | 0.071 | 0.167 | 0.053 |
| real_wine_raw_k5_seed252 | Current automatic | 0.059 | 0.071 | 0.130 | 0.052 |
| real_wine_raw_k5_seed252 | Default-thread brute | 0.133 | 0.487 | 0.620 | 0.437 |
| real_wine_raw_k5_seed252 | Single-thread brute | 0.126 | 0.127 | 0.253 | 0.080 |
| real_wine_raw_k5_seed252 | sklearn auto | 0.182 | 0.195 | 0.378 | 0.144 |
| real_wine_raw_k5_seed252 | Preceding automatic | 0.106 | 0.076 | 0.182 | 0.062 |

## subsequent_fresh_models

All configurations and metrics are retained in this separate view. Earlier fixtures may already have initialized process/runtime/compiler state; this table is not fresh-process cold latency.

| Case | Worker | fit_ms median ms | first_query_ms median ms | fit_first_query_ms median ms | second_query_ms median ms |
| --- | --- | --- | --- | --- | --- |
| shape_single_8_k5_seed41 | Current automatic | 0.081 | 0.067 | 0.149 | 0.057 |
| shape_single_8_k5_seed41 | Default-thread brute | 0.109 | 0.689 | 0.794 | 0.835 |
| shape_single_8_k5_seed41 | Single-thread brute | 0.108 | 0.205 | 0.313 | 0.189 |
| shape_single_8_k5_seed41 | sklearn auto | 1.496 | 0.172 | 1.669 | 0.141 |
| shape_single_8_k5_seed41 | Preceding automatic | 0.124 | 0.073 | 0.196 | 0.065 |
| shape_single_32_k5_seed49 | Current automatic | 0.094 | 0.120 | 0.214 | 0.111 |
| shape_single_32_k5_seed49 | Default-thread brute | 0.130 | 2.686 | 2.811 | 2.727 |
| shape_single_32_k5_seed49 | Single-thread brute | 0.128 | 0.217 | 0.346 | 0.203 |
| shape_single_32_k5_seed49 | sklearn auto | 0.134 | 2.688 | 2.814 | 2.721 |
| shape_single_32_k5_seed49 | Preceding automatic | 0.137 | 0.125 | 0.259 | 0.116 |
| shape_single_8_k5_seed42 | Current automatic | 0.071 | 0.065 | 0.134 | 0.056 |
| shape_single_8_k5_seed42 | Default-thread brute | 0.097 | 2.598 | 2.688 | 2.612 |
| shape_single_8_k5_seed42 | Single-thread brute | 0.102 | 0.204 | 0.306 | 0.186 |
| shape_single_8_k5_seed42 | sklearn auto | 1.522 | 0.172 | 1.689 | 0.142 |
| shape_single_8_k5_seed42 | Preceding automatic | 0.114 | 0.072 | 0.186 | 0.064 |
| shape_single_32_k5_seed50 | Current automatic | 0.093 | 0.116 | 0.211 | 0.109 |
| shape_single_32_k5_seed50 | Default-thread brute | 0.131 | 2.751 | 2.859 | 2.704 |
| shape_single_32_k5_seed50 | Single-thread brute | 0.129 | 0.221 | 0.352 | 0.203 |
| shape_single_32_k5_seed50 | sklearn auto | 0.131 | 2.655 | 2.795 | 2.656 |
| shape_single_32_k5_seed50 | Preceding automatic | 0.135 | 0.122 | 0.257 | 0.114 |
| shape_single_8_k5_seed43 | Current automatic | 0.090 | 0.068 | 0.157 | 0.057 |
| shape_single_8_k5_seed43 | Default-thread brute | 0.126 | 2.685 | 2.813 | 2.705 |
| shape_single_8_k5_seed43 | Single-thread brute | 0.119 | 0.223 | 0.345 | 0.195 |
| shape_single_8_k5_seed43 | sklearn auto | 1.550 | 0.191 | 1.741 | 0.149 |
| shape_single_8_k5_seed43 | Preceding automatic | 0.138 | 0.079 | 0.216 | 0.066 |
| shape_single_32_k5_seed51 | Current automatic | 0.100 | 0.128 | 0.229 | 0.119 |
| shape_single_32_k5_seed51 | Default-thread brute | 0.131 | 2.709 | 2.844 | 2.717 |
| shape_single_32_k5_seed51 | Single-thread brute | 0.131 | 0.221 | 0.351 | 0.202 |
| shape_single_32_k5_seed51 | sklearn auto | 0.132 | 2.670 | 2.796 | 2.743 |
| shape_single_32_k5_seed51 | Preceding automatic | 0.133 | 0.131 | 0.263 | 0.122 |
| real_digits_raw_k5_seed151 | Current automatic | 0.078 | 0.479 | 0.547 | 0.320 |
| real_digits_raw_k5_seed151 | Default-thread brute | 0.140 | 2.842 | 3.011 | 2.889 |
| real_digits_raw_k5_seed151 | Single-thread brute | 0.146 | 1.101 | 1.251 | 1.046 |
| real_digits_raw_k5_seed151 | sklearn auto | 0.146 | 2.920 | 3.044 | 3.001 |
| real_digits_raw_k5_seed151 | Preceding automatic | 0.127 | 0.540 | 0.667 | 0.393 |
| real_digits_raw_k5_seed251 | Current automatic | 0.075 | 0.486 | 0.549 | 0.328 |
| real_digits_raw_k5_seed251 | Default-thread brute | 0.144 | 2.889 | 3.020 | 2.909 |
| real_digits_raw_k5_seed251 | Single-thread brute | 0.148 | 1.108 | 1.258 | 1.053 |
| real_digits_raw_k5_seed251 | sklearn auto | 0.140 | 2.912 | 3.017 | 2.829 |
| real_digits_raw_k5_seed251 | Preceding automatic | 0.117 | 0.540 | 0.652 | 0.398 |
| real_digits_raw_k5_seed252 | Current automatic | 0.077 | 0.482 | 0.559 | 0.322 |
| real_digits_raw_k5_seed252 | Default-thread brute | 0.146 | 2.999 | 3.138 | 2.970 |
| real_digits_raw_k5_seed252 | Single-thread brute | 0.146 | 1.093 | 1.232 | 1.042 |
| real_digits_raw_k5_seed252 | sklearn auto | 0.140 | 2.908 | 3.042 | 2.849 |
| real_digits_raw_k5_seed252 | Preceding automatic | 0.124 | 0.540 | 0.661 | 0.395 |
| real_breast_cancer_raw_k5_seed151 | Current automatic | 0.059 | 0.192 | 0.241 | 0.137 |
| real_breast_cancer_raw_k5_seed151 | Default-thread brute | 0.119 | 2.205 | 2.332 | 2.183 |
| real_breast_cancer_raw_k5_seed151 | Single-thread brute | 0.124 | 0.256 | 0.381 | 0.216 |
| real_breast_cancer_raw_k5_seed151 | sklearn auto | 0.120 | 2.114 | 2.222 | 2.136 |
| real_breast_cancer_raw_k5_seed151 | Preceding automatic | 0.097 | 0.232 | 0.333 | 0.180 |
| real_breast_cancer_raw_k5_seed251 | Current automatic | 0.042 | 0.177 | 0.224 | 0.127 |
| real_breast_cancer_raw_k5_seed251 | Default-thread brute | 0.101 | 2.018 | 2.119 | 2.104 |
| real_breast_cancer_raw_k5_seed251 | Single-thread brute | 0.101 | 0.233 | 0.339 | 0.212 |
| real_breast_cancer_raw_k5_seed251 | sklearn auto | 0.096 | 1.991 | 2.099 | 1.990 |
| real_breast_cancer_raw_k5_seed251 | Preceding automatic | 0.081 | 0.219 | 0.298 | 0.168 |
| real_breast_cancer_raw_k5_seed252 | Current automatic | 0.039 | 0.169 | 0.205 | 0.133 |
| real_breast_cancer_raw_k5_seed252 | Default-thread brute | 0.099 | 2.085 | 2.190 | 2.173 |
| real_breast_cancer_raw_k5_seed252 | Single-thread brute | 0.102 | 0.231 | 0.332 | 0.208 |
| real_breast_cancer_raw_k5_seed252 | sklearn auto | 0.102 | 1.999 | 2.092 | 2.054 |
| real_breast_cancer_raw_k5_seed252 | Preceding automatic | 0.076 | 0.205 | 0.283 | 0.172 |
| real_iris_raw_k5_seed151 | Current automatic | 0.041 | 0.052 | 0.092 | 0.040 |
| real_iris_raw_k5_seed151 | Default-thread brute | 0.107 | 1.489 | 1.579 | 1.553 |
| real_iris_raw_k5_seed151 | Single-thread brute | 0.104 | 0.099 | 0.203 | 0.076 |
| real_iris_raw_k5_seed151 | sklearn auto | 0.154 | 0.163 | 0.320 | 0.137 |
| real_iris_raw_k5_seed151 | Preceding automatic | 0.085 | 0.060 | 0.144 | 0.046 |
| real_iris_raw_k5_seed251 | Current automatic | 0.036 | 0.052 | 0.087 | 0.040 |
| real_iris_raw_k5_seed251 | Default-thread brute | 0.101 | 0.593 | 0.690 | 0.596 |
| real_iris_raw_k5_seed251 | Single-thread brute | 0.092 | 0.091 | 0.184 | 0.076 |
| real_iris_raw_k5_seed251 | sklearn auto | 0.130 | 0.167 | 0.299 | 0.136 |
| real_iris_raw_k5_seed251 | Preceding automatic | 0.071 | 0.056 | 0.126 | 0.046 |
| real_iris_raw_k5_seed252 | Current automatic | 0.035 | 0.046 | 0.081 | 0.036 |
| real_iris_raw_k5_seed252 | Default-thread brute | 0.094 | 0.650 | 0.735 | 0.610 |
| real_iris_raw_k5_seed252 | Single-thread brute | 0.089 | 0.086 | 0.175 | 0.072 |
| real_iris_raw_k5_seed252 | sklearn auto | 0.124 | 0.153 | 0.278 | 0.133 |
| real_iris_raw_k5_seed252 | Preceding automatic | 0.069 | 0.050 | 0.120 | 0.042 |
| real_wine_raw_k5_seed151 | Current automatic | 0.034 | 0.055 | 0.088 | 0.046 |
| real_wine_raw_k5_seed151 | Default-thread brute | 0.093 | 0.529 | 0.623 | 0.558 |
| real_wine_raw_k5_seed151 | Single-thread brute | 0.082 | 0.090 | 0.172 | 0.080 |
| real_wine_raw_k5_seed151 | sklearn auto | 0.122 | 0.157 | 0.279 | 0.135 |
| real_wine_raw_k5_seed151 | Preceding automatic | 0.068 | 0.064 | 0.132 | 0.055 |
| real_wine_raw_k5_seed251 | Current automatic | 0.033 | 0.056 | 0.089 | 0.047 |
| real_wine_raw_k5_seed251 | Default-thread brute | 0.090 | 0.558 | 0.649 | 0.608 |
| real_wine_raw_k5_seed251 | Single-thread brute | 0.084 | 0.090 | 0.174 | 0.079 |
| real_wine_raw_k5_seed251 | sklearn auto | 0.124 | 0.152 | 0.277 | 0.135 |
| real_wine_raw_k5_seed251 | Preceding automatic | 0.065 | 0.060 | 0.125 | 0.053 |
| real_wine_raw_k5_seed252 | Current automatic | 0.031 | 0.055 | 0.085 | 0.046 |
| real_wine_raw_k5_seed252 | Default-thread brute | 0.088 | 0.525 | 0.614 | 0.591 |
| real_wine_raw_k5_seed252 | Single-thread brute | 0.083 | 0.090 | 0.172 | 0.079 |
| real_wine_raw_k5_seed252 | sklearn auto | 0.122 | 0.152 | 0.276 | 0.135 |
| real_wine_raw_k5_seed252 | Preceding automatic | 0.064 | 0.060 | 0.126 | 0.053 |

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
| Preceding automatic | {"first_queries": 540, "first_strict_passes": 540, "models_requested": 540, "second_queries": 540, "second_strict_passes": 540} | {"first/cpu": 540, "second/cpu": 540} | {"True": 540} | {"True": 540} |

## Evidence

[Raw requests, setup, source/runtime identities, and all audits](../benchmarks/results/knn-startup-final2-2026-10-30-first-query.json)

[Validated summary](../benchmarks/results/knn-startup-final2-2026-10-30-first-query-summary.json)
