# Fresh-model first queries in initialized processes

Initialized isolated persistent processes; normal NumPy/estimator imports and configuration are recorded separately. No model or query prewarm. Every request constructs and fits a new estimator, then times its first query and separately its second query. Public NumPy results establish completion; no harness MLX import/device probe/synchronization runs before the first query. Both outputs are audited after both timers. All requested rounds (default 30), including first model per fixture and first model in worker, remain visible. Earlier models may initialize runtime/compiler caches. This is model-first-use in an initialized process, not fresh-process or machine-cold latency. Five-worker Williams order has a complete balanced 10-round block, repeated three times by default. Any incomplete final block is also retained. No classification/regression operations are added here; the separate full broad and preservation control panels retain those operations and fixtures.

First query includes model preparation deferred by fit. Second query reuses only that fresh model. All rounds remain in the headline median; first-per-fixture and subsequent-model views are shown separately, not selected according to speed. Process initialization/import cost is separate, and must not be confused with the dedicated fresh-process cold benchmark.

Acceptance: **True**. 18 fixtures × 30 fresh models × 5 workers; each model has exactly one retained first and one retained second query. No numerical audit failure removes a valid timing. Ratios are comparator/current; bold values below 1× are slower observations. These descriptive ratios are not confidence intervals.

## Initial process setup (once per worker)

| Worker | NumPy import ms | Estimator import/setup ms | Worker entry→initialized ms | Parent process→initialized ms |
| --- | --- | --- | --- | --- |
| Current automatic | 24.265 | 761.599 | 787.666 | 810.842 |
| Default-thread brute | 23.828 | 737.627 | 763.227 | 784.982 |
| Single-thread brute | 23.931 | 753.780 | 779.466 | 801.799 |
| sklearn auto | 23.834 | 748.629 | 774.296 | 796.633 |
| Preceding automatic | 24.496 | 1007.368 | 1033.704 | 1056.646 |

## fit_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.119 | 0.113 / **0.946×** / **0.945×** | 0.110 / **0.927×** / **0.947×** | 1.503 / 12.630× / 12.352× | 0.119 / **0.999×** / 1.000× |
| Gaussian 32D, single query | 49 | 0.140 | 0.143 / 1.019× / 1.021× | 0.139 / **0.994×** / **0.930×** | 0.139 / **0.992×** / **0.949×** | 0.139 / **0.994×** / **0.966×** |
| Gaussian 8D, single query | 42 | 0.119 | 0.109 / **0.913×** / **0.892×** | 0.107 / **0.903×** / **0.893×** | 1.531 / 12.860× / 12.654× | 0.118 / **0.995×** / **0.973×** |
| Gaussian 32D, single query | 50 | 0.134 | 0.139 / 1.038× / **0.988×** | 0.136 / 1.019× / **0.981×** | 0.135 / 1.012× / **0.988×** | 0.134 / 1.002× / **0.983×** |
| Gaussian 8D, single query | 43 | 0.134 | 0.131 / **0.976×** / **0.942×** | 0.122 / **0.912×** / **0.919×** | 1.562 / 11.668× / 11.492× | 0.131 / **0.975×** / **0.983×** |
| Gaussian 32D, single query | 51 | 0.152 | 0.144 / **0.943×** / **0.970×** | 0.144 / **0.943×** / **0.930×** | 0.141 / **0.926×** / **0.962×** | 0.147 / **0.966×** / **0.965×** |
| Digits (raw) | 151 | 0.133 | 0.158 / 1.188× / 1.182× | 0.153 / 1.150× / 1.194× | 0.155 / 1.165× / 1.138× | 0.120 / **0.900×** / **0.953×** |
| Digits (raw) | 251 | 0.119 | 0.150 / 1.264× / 1.283× | 0.142 / 1.191× / 1.218× | 0.148 / 1.240× / 1.262× | 0.125 / 1.049× / 1.020× |
| Digits (raw) | 252 | 0.113 | 0.148 / 1.312× / 1.272× | 0.145 / 1.289× / 1.237× | 0.148 / 1.310× / 1.201× | 0.121 / 1.074× / **0.983×** |
| Breast Cancer (raw) | 151 | 0.087 | 0.115 / 1.316× / 1.292× | 0.107 / 1.225× / 1.247× | 0.112 / 1.279× / 1.277× | 0.086 / **0.982×** / **0.970×** |
| Breast Cancer (raw) | 251 | 0.083 | 0.105 / 1.267× / 1.273× | 0.103 / 1.243× / 1.261× | 0.108 / 1.309× / 1.284× | 0.082 / **0.990×** / **0.960×** |
| Breast Cancer (raw) | 252 | 0.094 | 0.120 / 1.277× / 1.285× | 0.108 / 1.146× / 1.139× | 0.119 / 1.265× / 1.181× | 0.088 / **0.929×** / **0.915×** |
| Iris (raw) | 151 | 0.094 | 0.118 / 1.251× / 1.282× | 0.117 / 1.248× / 1.259× | 0.160 / 1.705× / 1.716× | 0.094 / **0.996×** / 1.006× |
| Iris (raw) | 251 | 0.069 | 0.089 / 1.286× / 1.308× | 0.088 / 1.265× / 1.355× | 0.114 / 1.642× / 1.768× | 0.069 / 1.004× / 1.033× |
| Iris (raw) | 252 | 0.064 | 0.088 / 1.368× / 1.337× | 0.086 / 1.341× / 1.333× | 0.115 / 1.798× / 1.819× | 0.058 / **0.911×** / **0.927×** |
| Wine (raw) | 151 | 0.067 | 0.091 / 1.354× / 1.335× | 0.088 / 1.303× / 1.284× | 0.126 / 1.864× / 1.922× | 0.065 / **0.970×** / 1.013× |
| Wine (raw) | 251 | 0.066 | 0.091 / 1.379× / 1.390× | 0.088 / 1.340× / 1.376× | 0.133 / 2.029× / 1.968× | 0.067 / 1.020× / 1.014× |
| Wine (raw) | 252 | 0.069 | 0.091 / 1.322× / 1.362× | 0.088 / 1.289× / 1.303× | 0.126 / 1.843× / 1.868× | 0.063 / **0.925×** / **0.928×** |

## first_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.066 | 0.696 / 10.599× / 10.738× | 0.208 / 3.172× / 3.226× | 0.175 / 2.660× / 2.773× | 0.074 / 1.129× / 1.127× |
| Gaussian 32D, single query | 49 | 0.122 | 2.968 / 24.239× / 23.805× | 0.224 / 1.832× / 1.820× | 2.918 / 23.834× / 23.408× | 0.125 / 1.023× / 1.022× |
| Gaussian 8D, single query | 42 | 0.065 | 2.580 / 39.486× / 39.294× | 0.211 / 3.226× / 3.203× | 0.183 / 2.806× / 2.898× | 0.073 / 1.112× / 1.109× |
| Gaussian 32D, single query | 50 | 0.120 | 2.949 / 24.636× / 24.628× | 0.225 / 1.877× / 1.872× | 2.920 / 24.394× / 24.169× | 0.124 / 1.038× / 1.036× |
| Gaussian 8D, single query | 43 | 0.069 | 2.918 / 42.083× / 40.077× | 0.225 / 3.240× / 3.214× | 0.191 / 2.752× / 2.818× | 0.077 / 1.108× / 1.135× |
| Gaussian 32D, single query | 51 | 0.131 | 2.918 / 22.340× / 22.593× | 0.226 / 1.734× / 1.745× | 2.849 / 21.808× / 22.060× | 0.134 / 1.027× / 1.021× |
| Digits (raw) | 151 | 0.485 | 3.334 / 6.871× / 7.067× | 1.107 / 2.282× / 2.257× | 3.197 / 6.589× / 6.677× | 0.542 / 1.117× / 1.090× |
| Digits (raw) | 251 | 0.472 | 3.090 / 6.542× / 6.382× | 1.098 / 2.325× / 2.296× | 3.083 / 6.526× / 6.720× | 0.529 / 1.120× / 1.113× |
| Digits (raw) | 252 | 0.481 | 3.162 / 6.574× / 6.407× | 1.104 / 2.295× / 2.316× | 3.165 / 6.579× / 6.816× | 0.532 / 1.106× / 1.111× |
| Breast Cancer (raw) | 151 | 0.184 | 2.124 / 11.573× / 11.129× | 0.231 / 1.261× / 1.312× | 2.068 / 11.267× / 11.357× | 0.216 / 1.177× / 1.185× |
| Breast Cancer (raw) | 251 | 0.184 | 2.259 / 12.304× / 11.775× | 0.235 / 1.278× / 1.258× | 2.040 / 11.110× / 11.161× | 0.220 / 1.199× / 1.175× |
| Breast Cancer (raw) | 252 | 0.185 | 2.198 / 11.886× / 11.903× | 0.237 / 1.281× / 1.294× | 2.130 / 11.518× / 12.158× | 0.217 / 1.172× / 1.176× |
| Iris (raw) | 151 | 0.058 | 1.533 / 26.616× / 26.702× | 0.110 / 1.908× / 1.917× | 0.182 / 3.155× / 3.287× | 0.064 / 1.107× / 1.102× |
| Iris (raw) | 251 | 0.049 | 0.536 / 11.031× / 11.481× | 0.085 / 1.756× / 1.889× | 0.146 / 2.999× / 3.258× | 0.056 / 1.142× / 1.087× |
| Iris (raw) | 252 | 0.044 | 0.513 / 11.781× / 11.999× | 0.085 / 1.947× / 2.040× | 0.143 / 3.293× / 3.367× | 0.047 / 1.074× / 1.107× |
| Wine (raw) | 151 | 0.059 | 0.567 / 9.640× / 10.640× | 0.094 / 1.604× / 1.659× | 0.155 / 2.642× / 2.779× | 0.063 / 1.064× / 1.136× |
| Wine (raw) | 251 | 0.055 | 0.560 / 10.243× / 10.060× | 0.094 / 1.718× / 1.712× | 0.156 / 2.852× / 2.767× | 0.064 / 1.171× / 1.133× |
| Wine (raw) | 252 | 0.060 | 0.531 / 8.870× / 9.501× | 0.095 / 1.582× / 1.627× | 0.156 / 2.601× / 2.810× | 0.062 / 1.040× / 1.073× |

## fit_first_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.185 | 0.800 / 4.325× / 4.290× | 0.318 / 1.720× / 1.754× | 1.678 / 9.075× / 8.963× | 0.194 / 1.049× / 1.046× |
| Gaussian 32D, single query | 49 | 0.261 | 3.126 / 11.965× / 11.891× | 0.364 / 1.394× / 1.343× | 3.060 / 11.712× / 11.534× | 0.268 / 1.025× / **0.989×** |
| Gaussian 8D, single query | 42 | 0.185 | 2.695 / 14.557× / 13.498× | 0.319 / 1.722× / 1.705× | 1.717 / 9.274× / 9.104× | 0.194 / 1.049× / 1.023× |
| Gaussian 32D, single query | 50 | 0.253 | 3.082 / 12.165× / 12.240× | 0.363 / 1.434× / 1.393× | 3.055 / 12.058× / 11.743× | 0.258 / 1.017× / 1.007× |
| Gaussian 8D, single query | 43 | 0.205 | 3.050 / 14.908× / 12.547× | 0.348 / 1.702× / 1.724× | 1.751 / 8.558× / 8.545× | 0.208 / 1.016× / 1.023× |
| Gaussian 32D, single query | 51 | 0.282 | 3.064 / 10.873× / 10.722× | 0.374 / 1.328× / 1.295× | 2.988 / 10.603× / 10.382× | 0.279 / **0.989×** / **0.993×** |
| Digits (raw) | 151 | 0.607 | 3.498 / 5.767× / 5.767× | 1.261 / 2.079× / 2.075× | 3.344 / 5.514× / 5.564× | 0.659 / 1.086× / 1.072× |
| Digits (raw) | 251 | 0.587 | 3.218 / 5.481× / 5.551× | 1.243 / 2.117× / 2.119× | 3.225 / 5.493× / 5.704× | 0.643 / 1.095× / 1.098× |
| Digits (raw) | 252 | 0.590 | 3.347 / 5.670× / 5.272× | 1.245 / 2.109× / 2.141× | 3.307 / 5.601× / 5.670× | 0.643 / 1.090× / 1.091× |
| Breast Cancer (raw) | 151 | 0.269 | 2.226 / 8.290× / 7.898× | 0.341 / 1.268× / 1.271× | 2.179 / 8.116× / 8.089× | 0.302 / 1.126× / 1.112× |
| Breast Cancer (raw) | 251 | 0.273 | 2.365 / 8.677× / 8.391× | 0.336 / 1.234× / 1.258× | 2.151 / 7.893× / 7.693× | 0.303 / 1.113× / 1.130× |
| Breast Cancer (raw) | 252 | 0.281 | 2.303 / 8.209× / 7.988× | 0.346 / 1.233× / 1.239× | 2.236 / 7.969× / 8.466× | 0.302 / 1.077× / 1.089× |
| Iris (raw) | 151 | 0.149 | 1.648 / 11.049× / 10.769× | 0.226 / 1.519× / 1.543× | 0.342 / 2.294× / 2.340× | 0.157 / 1.050× / 1.046× |
| Iris (raw) | 251 | 0.116 | 0.636 / 5.463× / 5.466× | 0.174 / 1.490× / 1.579× | 0.262 / 2.250× / 2.386× | 0.125 / 1.072× / 1.052× |
| Iris (raw) | 252 | 0.109 | 0.610 / 5.592× / 5.659× | 0.171 / 1.569× / 1.592× | 0.259 / 2.376× / 2.435× | 0.106 / **0.970×** / **0.982×** |
| Wine (raw) | 151 | 0.127 | 0.666 / 5.266× / 5.536× | 0.181 / 1.433× / 1.488× | 0.278 / 2.200× / 2.369× | 0.127 / 1.007× / 1.072× |
| Wine (raw) | 251 | 0.121 | 0.664 / 5.478× / 5.461× | 0.183 / 1.509× / 1.540× | 0.288 / 2.380× / 2.345× | 0.129 / 1.066× / 1.065× |
| Wine (raw) | 252 | 0.129 | 0.634 / 4.925× / 5.094× | 0.184 / 1.425× / 1.442× | 0.281 / 2.184× / 2.298× | 0.124 / **0.966×** / **0.985×** |

## second_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian 8D, single query | 41 | 0.056 | 0.656 / 11.665× / 11.656× | 0.192 / 3.414× / 3.346× | 0.144 / 2.564× / 2.535× | 0.065 / 1.153× / 1.153× |
| Gaussian 32D, single query | 49 | 0.112 | 2.958 / 26.427× / 26.223× | 0.204 / 1.822× / 1.817× | 2.958 / 26.433× / 26.652× | 0.116 / 1.034× / 1.030× |
| Gaussian 8D, single query | 42 | 0.056 | 2.771 / 49.719× / 48.918× | 0.187 / 3.352× / 3.360× | 0.147 / 2.646× / 2.641× | 0.064 / 1.144× / 1.135× |
| Gaussian 32D, single query | 50 | 0.110 | 3.014 / 27.372× / 27.395× | 0.203 / 1.846× / 1.836× | 2.941 / 26.714× / 26.825× | 0.115 / 1.043× / 1.043× |
| Gaussian 8D, single query | 43 | 0.057 | 2.844 / 49.709× / 48.854× | 0.196 / 3.418× / 3.424× | 0.147 / 2.574× / 2.575× | 0.065 / 1.134× / 1.165× |
| Gaussian 32D, single query | 51 | 0.119 | 3.039 / 25.594× / 25.420× | 0.204 / 1.721× / 1.727× | 2.991 / 25.192× / 25.378× | 0.122 / 1.028× / 1.033× |
| Digits (raw) | 151 | 0.322 | 3.429 / 10.650× / 11.006× | 1.052 / 3.267× / 3.296× | 3.351 / 10.407× / 10.482× | 0.394 / 1.224× / 1.247× |
| Digits (raw) | 251 | 0.320 | 3.321 / 10.387× / 10.339× | 1.050 / 3.283× / 3.239× | 3.285 / 10.275× / 10.325× | 0.387 / 1.211× / 1.218× |
| Digits (raw) | 252 | 0.324 | 3.185 / 9.818× / 10.106× | 1.050 / 3.235× / 3.267× | 3.274 / 10.093× / 10.130× | 0.392 / 1.207× / 1.226× |
| Breast Cancer (raw) | 151 | 0.136 | 2.333 / 17.107× / 16.234× | 0.208 / 1.525× / 1.517× | 2.240 / 16.424× / 16.002× | 0.174 / 1.277× / 1.277× |
| Breast Cancer (raw) | 251 | 0.130 | 2.293 / 17.691× / 17.747× | 0.209 / 1.610× / 1.605× | 2.127 / 16.410× / 16.411× | 0.169 / 1.300× / 1.318× |
| Breast Cancer (raw) | 252 | 0.137 | 2.425 / 17.750× / 17.500× | 0.211 / 1.545× / 1.558× | 2.247 / 16.447× / 16.023× | 0.176 / 1.287× / 1.284× |
| Iris (raw) | 151 | 0.041 | 1.579 / 38.060× / 37.431× | 0.080 / 1.927× / 1.963× | 0.139 / 3.342× / 3.432× | 0.048 / 1.146× / 1.157× |
| Iris (raw) | 251 | 0.039 | 0.538 / 13.918× / 13.798× | 0.073 / 1.878× / 1.872× | 0.129 / 3.338× / 3.434× | 0.045 / 1.163× / 1.149× |
| Iris (raw) | 252 | 0.036 | 0.524 / 14.368× / 14.597× | 0.073 / 2.008× / 2.053× | 0.130 / 3.573× / 3.598× | 0.041 / 1.134× / 1.150× |
| Wine (raw) | 151 | 0.048 | 0.554 / 11.467× / 11.780× | 0.080 / 1.654× / 1.662× | 0.135 / 2.790× / 2.875× | 0.055 / 1.146× / 1.189× |
| Wine (raw) | 251 | 0.047 | 0.604 / 12.801× / 12.427× | 0.081 / 1.712× / 1.741× | 0.134 / 2.838× / 2.920× | 0.055 / 1.168× / 1.186× |
| Wine (raw) | 252 | 0.049 | 0.550 / 11.273× / 11.536× | 0.081 / 1.669× / 1.689× | 0.136 / 2.784× / 2.827× | 0.055 / 1.130× / 1.133× |

## Every slower or missing all-round median

| Metric | Comparator | Case | Current ms | Comparator ms | M | P |
| --- | --- | --- | --- | --- | --- | --- |
| fit_ms | Default-thread brute | shape_single_8_k5_seed41 | 0.119 | 0.113 | **0.946×** | **0.945×** |
| fit_ms | Default-thread brute | shape_single_8_k5_seed42 | 0.119 | 0.109 | **0.913×** | **0.892×** |
| fit_ms | Default-thread brute | shape_single_8_k5_seed43 | 0.134 | 0.131 | **0.976×** | **0.942×** |
| fit_ms | Default-thread brute | shape_single_32_k5_seed51 | 0.152 | 0.144 | **0.943×** | **0.970×** |
| fit_ms | Single-thread brute | shape_single_8_k5_seed41 | 0.119 | 0.110 | **0.927×** | **0.947×** |
| fit_ms | Single-thread brute | shape_single_32_k5_seed49 | 0.140 | 0.139 | **0.994×** | **0.930×** |
| fit_ms | Single-thread brute | shape_single_8_k5_seed42 | 0.119 | 0.107 | **0.903×** | **0.893×** |
| fit_ms | Single-thread brute | shape_single_8_k5_seed43 | 0.134 | 0.122 | **0.912×** | **0.919×** |
| fit_ms | Single-thread brute | shape_single_32_k5_seed51 | 0.152 | 0.144 | **0.943×** | **0.930×** |
| fit_ms | sklearn auto | shape_single_32_k5_seed49 | 0.140 | 0.139 | **0.992×** | **0.949×** |
| fit_ms | sklearn auto | shape_single_32_k5_seed51 | 0.152 | 0.141 | **0.926×** | **0.962×** |
| fit_ms | Preceding automatic | shape_single_8_k5_seed41 | 0.119 | 0.119 | **0.999×** | 1.000× |
| fit_ms | Preceding automatic | shape_single_32_k5_seed49 | 0.140 | 0.139 | **0.994×** | **0.966×** |
| fit_ms | Preceding automatic | shape_single_8_k5_seed42 | 0.119 | 0.118 | **0.995×** | **0.973×** |
| fit_ms | Preceding automatic | shape_single_8_k5_seed43 | 0.134 | 0.131 | **0.975×** | **0.983×** |
| fit_ms | Preceding automatic | shape_single_32_k5_seed51 | 0.152 | 0.147 | **0.966×** | **0.965×** |
| fit_ms | Preceding automatic | real_digits_raw_k5_seed151 | 0.133 | 0.120 | **0.900×** | **0.953×** |
| fit_ms | Preceding automatic | real_breast_cancer_raw_k5_seed151 | 0.087 | 0.086 | **0.982×** | **0.970×** |
| fit_ms | Preceding automatic | real_breast_cancer_raw_k5_seed251 | 0.083 | 0.082 | **0.990×** | **0.960×** |
| fit_ms | Preceding automatic | real_breast_cancer_raw_k5_seed252 | 0.094 | 0.088 | **0.929×** | **0.915×** |
| fit_ms | Preceding automatic | real_iris_raw_k5_seed151 | 0.094 | 0.094 | **0.996×** | 1.006× |
| fit_ms | Preceding automatic | real_iris_raw_k5_seed252 | 0.064 | 0.058 | **0.911×** | **0.927×** |
| fit_ms | Preceding automatic | real_wine_raw_k5_seed151 | 0.067 | 0.065 | **0.970×** | 1.013× |
| fit_ms | Preceding automatic | real_wine_raw_k5_seed252 | 0.069 | 0.063 | **0.925×** | **0.928×** |
| fit_first_query_ms | Preceding automatic | shape_single_32_k5_seed51 | 0.282 | 0.279 | **0.989×** | **0.993×** |
| fit_first_query_ms | Preceding automatic | real_iris_raw_k5_seed252 | 0.109 | 0.106 | **0.970×** | **0.982×** |
| fit_first_query_ms | Preceding automatic | real_wine_raw_k5_seed252 | 0.129 | 0.124 | **0.966×** | **0.985×** |

## first_model_for_fixture

All configurations and metrics are retained in this separate view. Earlier fixtures may already have initialized process/runtime/compiler state; this table is not fresh-process cold latency.

| Case | Worker | fit_ms median ms | first_query_ms median ms | fit_first_query_ms median ms | second_query_ms median ms |
| --- | --- | --- | --- | --- | --- |
| shape_single_8_k5_seed41 | Current automatic | 0.277 | 0.123 | 0.401 | 0.149 |
| shape_single_8_k5_seed41 | Default-thread brute | 0.423 | 18.786 | 19.209 | 1.136 |
| shape_single_8_k5_seed41 | Single-thread brute | 0.285 | 8.205 | 8.490 | 0.539 |
| shape_single_8_k5_seed41 | sklearn auto | 2.035 | 0.308 | 2.343 | 0.201 |
| shape_single_8_k5_seed41 | Preceding automatic | 0.297 | 0.136 | 0.433 | 0.110 |
| shape_single_32_k5_seed49 | Current automatic | 0.174 | 0.125 | 0.299 | 0.113 |
| shape_single_32_k5_seed49 | Default-thread brute | 0.150 | 0.826 | 0.977 | 0.823 |
| shape_single_32_k5_seed49 | Single-thread brute | 0.155 | 0.236 | 0.391 | 0.202 |
| shape_single_32_k5_seed49 | sklearn auto | 0.146 | 17.098 | 17.244 | 3.293 |
| shape_single_32_k5_seed49 | Preceding automatic | 0.153 | 0.126 | 0.280 | 0.117 |
| shape_single_8_k5_seed42 | Current automatic | 0.154 | 0.071 | 0.225 | 0.057 |
| shape_single_8_k5_seed42 | Default-thread brute | 0.113 | 3.297 | 3.410 | 2.776 |
| shape_single_8_k5_seed42 | Single-thread brute | 0.106 | 0.206 | 0.312 | 0.186 |
| shape_single_8_k5_seed42 | sklearn auto | 1.593 | 0.317 | 1.910 | 0.157 |
| shape_single_8_k5_seed42 | Preceding automatic | 0.152 | 0.079 | 0.230 | 0.065 |
| shape_single_32_k5_seed50 | Current automatic | 0.145 | 0.133 | 0.278 | 0.111 |
| shape_single_32_k5_seed50 | Default-thread brute | 0.145 | 3.130 | 3.275 | 2.980 |
| shape_single_32_k5_seed50 | Single-thread brute | 0.132 | 0.229 | 0.362 | 0.206 |
| shape_single_32_k5_seed50 | sklearn auto | 0.123 | 2.889 | 3.012 | 3.004 |
| shape_single_32_k5_seed50 | Preceding automatic | 0.137 | 0.123 | 0.260 | 0.120 |
| shape_single_8_k5_seed43 | Current automatic | 0.148 | 0.072 | 0.220 | 0.058 |
| shape_single_8_k5_seed43 | Default-thread brute | 0.143 | 3.086 | 3.229 | 3.416 |
| shape_single_8_k5_seed43 | Single-thread brute | 0.152 | 0.235 | 0.388 | 0.186 |
| shape_single_8_k5_seed43 | sklearn auto | 1.581 | 0.264 | 1.845 | 0.186 |
| shape_single_8_k5_seed43 | Preceding automatic | 0.151 | 0.077 | 0.228 | 0.063 |
| shape_single_32_k5_seed51 | Current automatic | 0.162 | 0.134 | 0.297 | 0.121 |
| shape_single_32_k5_seed51 | Default-thread brute | 0.142 | 0.843 | 0.985 | 0.611 |
| shape_single_32_k5_seed51 | Single-thread brute | 0.145 | 0.234 | 0.379 | 0.205 |
| shape_single_32_k5_seed51 | sklearn auto | 0.137 | 2.348 | 2.486 | 3.074 |
| shape_single_32_k5_seed51 | Preceding automatic | 0.165 | 0.138 | 0.303 | 0.127 |
| real_digits_raw_k5_seed151 | Current automatic | 0.120 | 0.585 | 0.705 | 0.336 |
| real_digits_raw_k5_seed151 | Default-thread brute | 0.151 | 3.522 | 3.673 | 3.283 |
| real_digits_raw_k5_seed151 | Single-thread brute | 0.142 | 1.125 | 1.266 | 1.055 |
| real_digits_raw_k5_seed151 | sklearn auto | 0.156 | 3.085 | 3.241 | 2.812 |
| real_digits_raw_k5_seed151 | Preceding automatic | 0.120 | 0.657 | 0.777 | 0.399 |
| real_digits_raw_k5_seed251 | Current automatic | 0.120 | 0.514 | 0.634 | 0.335 |
| real_digits_raw_k5_seed251 | Default-thread brute | 0.169 | 4.149 | 4.318 | 4.918 |
| real_digits_raw_k5_seed251 | Single-thread brute | 0.159 | 1.123 | 1.283 | 1.053 |
| real_digits_raw_k5_seed251 | sklearn auto | 0.146 | 3.672 | 3.819 | 3.287 |
| real_digits_raw_k5_seed251 | Preceding automatic | 0.128 | 0.567 | 0.695 | 0.444 |
| real_digits_raw_k5_seed252 | Current automatic | 0.120 | 0.571 | 0.691 | 0.316 |
| real_digits_raw_k5_seed252 | Default-thread brute | 0.151 | 3.437 | 3.589 | 3.380 |
| real_digits_raw_k5_seed252 | Single-thread brute | 0.129 | 1.112 | 1.242 | 1.084 |
| real_digits_raw_k5_seed252 | sklearn auto | 0.146 | 3.616 | 3.762 | 2.536 |
| real_digits_raw_k5_seed252 | Preceding automatic | 0.122 | 0.590 | 0.712 | 0.452 |
| real_breast_cancer_raw_k5_seed151 | Current automatic | 0.134 | 0.219 | 0.353 | 0.163 |
| real_breast_cancer_raw_k5_seed151 | Default-thread brute | 0.156 | 2.338 | 2.494 | 2.296 |
| real_breast_cancer_raw_k5_seed151 | Single-thread brute | 0.157 | 0.285 | 0.443 | 0.223 |
| real_breast_cancer_raw_k5_seed151 | sklearn auto | 0.158 | 2.901 | 3.058 | 2.819 |
| real_breast_cancer_raw_k5_seed151 | Preceding automatic | 0.121 | 0.257 | 0.378 | 0.190 |
| real_breast_cancer_raw_k5_seed251 | Current automatic | 0.099 | 0.215 | 0.313 | 0.140 |
| real_breast_cancer_raw_k5_seed251 | Default-thread brute | 0.128 | 2.048 | 2.176 | 2.340 |
| real_breast_cancer_raw_k5_seed251 | Single-thread brute | 0.135 | 0.268 | 0.402 | 0.215 |
| real_breast_cancer_raw_k5_seed251 | sklearn auto | 0.132 | 2.385 | 2.517 | 2.097 |
| real_breast_cancer_raw_k5_seed251 | Preceding automatic | 0.095 | 0.236 | 0.331 | 0.174 |
| real_breast_cancer_raw_k5_seed252 | Current automatic | 0.102 | 0.200 | 0.303 | 0.159 |
| real_breast_cancer_raw_k5_seed252 | Default-thread brute | 0.142 | 2.398 | 2.540 | 2.422 |
| real_breast_cancer_raw_k5_seed252 | Single-thread brute | 0.128 | 0.258 | 0.386 | 0.211 |
| real_breast_cancer_raw_k5_seed252 | sklearn auto | 0.155 | 2.682 | 2.837 | 2.163 |
| real_breast_cancer_raw_k5_seed252 | Preceding automatic | 0.107 | 0.232 | 0.340 | 0.179 |
| real_iris_raw_k5_seed151 | Current automatic | 0.106 | 0.061 | 0.167 | 0.045 |
| real_iris_raw_k5_seed151 | Default-thread brute | 0.130 | 1.519 | 1.649 | 1.613 |
| real_iris_raw_k5_seed151 | Single-thread brute | 0.130 | 0.130 | 0.260 | 0.080 |
| real_iris_raw_k5_seed151 | sklearn auto | 0.200 | 0.231 | 0.430 | 0.152 |
| real_iris_raw_k5_seed151 | Preceding automatic | 0.104 | 0.063 | 0.167 | 0.047 |
| real_iris_raw_k5_seed251 | Current automatic | 0.101 | 0.058 | 0.159 | 0.039 |
| real_iris_raw_k5_seed251 | Default-thread brute | 0.139 | 0.528 | 0.667 | 0.538 |
| real_iris_raw_k5_seed251 | Single-thread brute | 0.141 | 0.133 | 0.274 | 0.085 |
| real_iris_raw_k5_seed251 | sklearn auto | 0.189 | 0.200 | 0.389 | 0.144 |
| real_iris_raw_k5_seed251 | Preceding automatic | 0.103 | 0.062 | 0.165 | 0.045 |
| real_iris_raw_k5_seed252 | Current automatic | 0.099 | 0.056 | 0.155 | 0.038 |
| real_iris_raw_k5_seed252 | Default-thread brute | 0.138 | 0.570 | 0.709 | 0.435 |
| real_iris_raw_k5_seed252 | Single-thread brute | 0.126 | 0.115 | 0.241 | 0.077 |
| real_iris_raw_k5_seed252 | sklearn auto | 0.183 | 0.193 | 0.377 | 0.141 |
| real_iris_raw_k5_seed252 | Preceding automatic | 0.107 | 0.068 | 0.175 | 0.044 |
| real_wine_raw_k5_seed151 | Current automatic | 0.104 | 0.069 | 0.173 | 0.049 |
| real_wine_raw_k5_seed151 | Default-thread brute | 0.142 | 0.822 | 0.964 | 0.604 |
| real_wine_raw_k5_seed151 | Single-thread brute | 0.132 | 0.136 | 0.268 | 0.086 |
| real_wine_raw_k5_seed151 | sklearn auto | 0.218 | 0.231 | 0.449 | 0.158 |
| real_wine_raw_k5_seed151 | Preceding automatic | 0.103 | 0.074 | 0.177 | 0.056 |
| real_wine_raw_k5_seed251 | Current automatic | 0.102 | 0.066 | 0.168 | 0.048 |
| real_wine_raw_k5_seed251 | Default-thread brute | 0.146 | 0.652 | 0.798 | 0.667 |
| real_wine_raw_k5_seed251 | Single-thread brute | 0.134 | 0.126 | 0.260 | 0.086 |
| real_wine_raw_k5_seed251 | sklearn auto | 0.183 | 0.198 | 0.381 | 0.147 |
| real_wine_raw_k5_seed251 | Preceding automatic | 0.107 | 0.074 | 0.181 | 0.057 |
| real_wine_raw_k5_seed252 | Current automatic | 0.103 | 0.066 | 0.169 | 0.049 |
| real_wine_raw_k5_seed252 | Default-thread brute | 0.143 | 0.603 | 0.746 | 0.447 |
| real_wine_raw_k5_seed252 | Single-thread brute | 0.152 | 0.139 | 0.291 | 0.085 |
| real_wine_raw_k5_seed252 | sklearn auto | 0.187 | 0.205 | 0.392 | 0.143 |
| real_wine_raw_k5_seed252 | Preceding automatic | 0.103 | 0.072 | 0.175 | 0.057 |

## subsequent_fresh_models

All configurations and metrics are retained in this separate view. Earlier fixtures may already have initialized process/runtime/compiler state; this table is not fresh-process cold latency.

| Case | Worker | fit_ms median ms | first_query_ms median ms | fit_first_query_ms median ms | second_query_ms median ms |
| --- | --- | --- | --- | --- | --- |
| shape_single_8_k5_seed41 | Current automatic | 0.117 | 0.065 | 0.184 | 0.056 |
| shape_single_8_k5_seed41 | Default-thread brute | 0.112 | 0.692 | 0.794 | 0.656 |
| shape_single_8_k5_seed41 | Single-thread brute | 0.106 | 0.208 | 0.316 | 0.192 |
| shape_single_8_k5_seed41 | sklearn auto | 1.501 | 0.174 | 1.678 | 0.144 |
| shape_single_8_k5_seed41 | Preceding automatic | 0.118 | 0.074 | 0.194 | 0.065 |
| shape_single_32_k5_seed49 | Current automatic | 0.139 | 0.122 | 0.261 | 0.112 |
| shape_single_32_k5_seed49 | Default-thread brute | 0.142 | 3.004 | 3.165 | 2.996 |
| shape_single_32_k5_seed49 | Single-thread brute | 0.138 | 0.224 | 0.361 | 0.204 |
| shape_single_32_k5_seed49 | sklearn auto | 0.139 | 2.904 | 3.033 | 2.947 |
| shape_single_32_k5_seed49 | Preceding automatic | 0.138 | 0.125 | 0.267 | 0.116 |
| shape_single_8_k5_seed42 | Current automatic | 0.119 | 0.065 | 0.185 | 0.056 |
| shape_single_8_k5_seed42 | Default-thread brute | 0.109 | 2.546 | 2.660 | 2.766 |
| shape_single_8_k5_seed42 | Single-thread brute | 0.108 | 0.211 | 0.320 | 0.187 |
| shape_single_8_k5_seed42 | sklearn auto | 1.525 | 0.182 | 1.716 | 0.147 |
| shape_single_8_k5_seed42 | Preceding automatic | 0.118 | 0.072 | 0.193 | 0.064 |
| shape_single_32_k5_seed50 | Current automatic | 0.130 | 0.120 | 0.248 | 0.110 |
| shape_single_32_k5_seed50 | Default-thread brute | 0.138 | 2.949 | 3.062 | 3.016 |
| shape_single_32_k5_seed50 | Single-thread brute | 0.136 | 0.222 | 0.365 | 0.202 |
| shape_single_32_k5_seed50 | sklearn auto | 0.136 | 2.930 | 3.061 | 2.921 |
| shape_single_32_k5_seed50 | Preceding automatic | 0.132 | 0.125 | 0.257 | 0.115 |
| shape_single_8_k5_seed43 | Current automatic | 0.133 | 0.069 | 0.204 | 0.057 |
| shape_single_8_k5_seed43 | Default-thread brute | 0.128 | 2.908 | 3.048 | 2.808 |
| shape_single_8_k5_seed43 | Single-thread brute | 0.121 | 0.223 | 0.342 | 0.198 |
| shape_single_8_k5_seed43 | sklearn auto | 1.562 | 0.189 | 1.749 | 0.147 |
| shape_single_8_k5_seed43 | Preceding automatic | 0.126 | 0.077 | 0.202 | 0.065 |
| shape_single_32_k5_seed51 | Current automatic | 0.151 | 0.131 | 0.281 | 0.118 |
| shape_single_32_k5_seed51 | Default-thread brute | 0.144 | 2.919 | 3.069 | 3.041 |
| shape_single_32_k5_seed51 | Single-thread brute | 0.143 | 0.226 | 0.374 | 0.204 |
| shape_single_32_k5_seed51 | sklearn auto | 0.142 | 2.855 | 2.989 | 2.984 |
| shape_single_32_k5_seed51 | Preceding automatic | 0.147 | 0.134 | 0.277 | 0.122 |
| real_digits_raw_k5_seed151 | Current automatic | 0.133 | 0.485 | 0.606 | 0.322 |
| real_digits_raw_k5_seed151 | Default-thread brute | 0.159 | 3.331 | 3.497 | 3.485 |
| real_digits_raw_k5_seed151 | Single-thread brute | 0.154 | 1.106 | 1.260 | 1.052 |
| real_digits_raw_k5_seed151 | sklearn auto | 0.154 | 3.205 | 3.345 | 3.365 |
| real_digits_raw_k5_seed151 | Preceding automatic | 0.119 | 0.540 | 0.653 | 0.393 |
| real_digits_raw_k5_seed251 | Current automatic | 0.118 | 0.469 | 0.586 | 0.319 |
| real_digits_raw_k5_seed251 | Default-thread brute | 0.150 | 3.069 | 3.207 | 3.320 |
| real_digits_raw_k5_seed251 | Single-thread brute | 0.140 | 1.098 | 1.242 | 1.049 |
| real_digits_raw_k5_seed251 | sklearn auto | 0.149 | 3.064 | 3.197 | 3.283 |
| real_digits_raw_k5_seed251 | Preceding automatic | 0.123 | 0.526 | 0.640 | 0.386 |
| real_digits_raw_k5_seed252 | Current automatic | 0.112 | 0.481 | 0.590 | 0.325 |
| real_digits_raw_k5_seed252 | Default-thread brute | 0.146 | 3.134 | 3.324 | 3.141 |
| real_digits_raw_k5_seed252 | Single-thread brute | 0.146 | 1.104 | 1.248 | 1.050 |
| real_digits_raw_k5_seed252 | sklearn auto | 0.149 | 3.162 | 3.299 | 3.298 |
| real_digits_raw_k5_seed252 | Preceding automatic | 0.121 | 0.532 | 0.642 | 0.391 |
| real_breast_cancer_raw_k5_seed151 | Current automatic | 0.086 | 0.182 | 0.266 | 0.136 |
| real_breast_cancer_raw_k5_seed151 | Default-thread brute | 0.113 | 2.092 | 2.205 | 2.335 |
| real_breast_cancer_raw_k5_seed151 | Single-thread brute | 0.107 | 0.231 | 0.337 | 0.208 |
| real_breast_cancer_raw_k5_seed151 | sklearn auto | 0.111 | 2.028 | 2.155 | 2.227 |
| real_breast_cancer_raw_k5_seed151 | Preceding automatic | 0.086 | 0.213 | 0.296 | 0.174 |
| real_breast_cancer_raw_k5_seed251 | Current automatic | 0.081 | 0.183 | 0.270 | 0.129 |
| real_breast_cancer_raw_k5_seed251 | Default-thread brute | 0.104 | 2.274 | 2.377 | 2.275 |
| real_breast_cancer_raw_k5_seed251 | Single-thread brute | 0.102 | 0.234 | 0.335 | 0.208 |
| real_breast_cancer_raw_k5_seed251 | sklearn auto | 0.107 | 2.027 | 2.125 | 2.144 |
| real_breast_cancer_raw_k5_seed251 | Preceding automatic | 0.081 | 0.220 | 0.303 | 0.168 |
| real_breast_cancer_raw_k5_seed252 | Current automatic | 0.093 | 0.183 | 0.279 | 0.136 |
| real_breast_cancer_raw_k5_seed252 | Default-thread brute | 0.120 | 2.170 | 2.261 | 2.427 |
| real_breast_cancer_raw_k5_seed252 | Single-thread brute | 0.108 | 0.235 | 0.344 | 0.211 |
| real_breast_cancer_raw_k5_seed252 | sklearn auto | 0.119 | 2.128 | 2.219 | 2.275 |
| real_breast_cancer_raw_k5_seed252 | Preceding automatic | 0.087 | 0.214 | 0.301 | 0.176 |
| real_iris_raw_k5_seed151 | Current automatic | 0.093 | 0.057 | 0.148 | 0.041 |
| real_iris_raw_k5_seed151 | Default-thread brute | 0.117 | 1.537 | 1.646 | 1.573 |
| real_iris_raw_k5_seed151 | Single-thread brute | 0.117 | 0.109 | 0.226 | 0.080 |
| real_iris_raw_k5_seed151 | sklearn auto | 0.160 | 0.179 | 0.340 | 0.138 |
| real_iris_raw_k5_seed151 | Preceding automatic | 0.093 | 0.064 | 0.155 | 0.048 |
| real_iris_raw_k5_seed251 | Current automatic | 0.069 | 0.048 | 0.114 | 0.038 |
| real_iris_raw_k5_seed251 | Default-thread brute | 0.089 | 0.544 | 0.634 | 0.537 |
| real_iris_raw_k5_seed251 | Single-thread brute | 0.087 | 0.084 | 0.172 | 0.072 |
| real_iris_raw_k5_seed251 | sklearn auto | 0.112 | 0.142 | 0.259 | 0.129 |
| real_iris_raw_k5_seed251 | Preceding automatic | 0.069 | 0.055 | 0.125 | 0.045 |
| real_iris_raw_k5_seed252 | Current automatic | 0.064 | 0.043 | 0.107 | 0.036 |
| real_iris_raw_k5_seed252 | Default-thread brute | 0.088 | 0.508 | 0.608 | 0.535 |
| real_iris_raw_k5_seed252 | Single-thread brute | 0.086 | 0.085 | 0.171 | 0.073 |
| real_iris_raw_k5_seed252 | sklearn auto | 0.114 | 0.143 | 0.258 | 0.130 |
| real_iris_raw_k5_seed252 | Preceding automatic | 0.058 | 0.047 | 0.105 | 0.041 |
| real_wine_raw_k5_seed151 | Current automatic | 0.066 | 0.059 | 0.126 | 0.048 |
| real_wine_raw_k5_seed151 | Default-thread brute | 0.089 | 0.566 | 0.664 | 0.549 |
| real_wine_raw_k5_seed151 | Single-thread brute | 0.087 | 0.094 | 0.181 | 0.080 |
| real_wine_raw_k5_seed151 | sklearn auto | 0.125 | 0.154 | 0.278 | 0.135 |
| real_wine_raw_k5_seed151 | Preceding automatic | 0.065 | 0.062 | 0.127 | 0.055 |
| real_wine_raw_k5_seed251 | Current automatic | 0.063 | 0.055 | 0.121 | 0.047 |
| real_wine_raw_k5_seed251 | Default-thread brute | 0.090 | 0.554 | 0.663 | 0.599 |
| real_wine_raw_k5_seed251 | Single-thread brute | 0.087 | 0.094 | 0.181 | 0.080 |
| real_wine_raw_k5_seed251 | sklearn auto | 0.133 | 0.156 | 0.287 | 0.133 |
| real_wine_raw_k5_seed251 | Preceding automatic | 0.066 | 0.063 | 0.128 | 0.055 |
| real_wine_raw_k5_seed252 | Current automatic | 0.068 | 0.060 | 0.127 | 0.049 |
| real_wine_raw_k5_seed252 | Default-thread brute | 0.091 | 0.531 | 0.633 | 0.556 |
| real_wine_raw_k5_seed252 | Single-thread brute | 0.088 | 0.094 | 0.183 | 0.081 |
| real_wine_raw_k5_seed252 | sklearn auto | 0.126 | 0.154 | 0.281 | 0.136 |
| real_wine_raw_k5_seed252 | Preceding automatic | 0.063 | 0.062 | 0.123 | 0.055 |

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

| Worker | Audits | Query routing | Native loaded before fit |
| --- | --- | --- | --- |
| Current automatic | {"first_queries": 540, "first_strict_passes": 540, "models_requested": 540, "second_queries": 540, "second_strict_passes": 540} | {"first/cpu": 540, "second/cpu": 540} | {"True": 540} |
| Default-thread brute | {"first_queries": 540, "first_strict_passes": 390, "models_requested": 540, "second_queries": 540, "second_strict_passes": 390} | {"first/sklearn": 540, "second/sklearn": 540} | {"False": 540} |
| Single-thread brute | {"first_queries": 540, "first_strict_passes": 390, "models_requested": 540, "second_queries": 540, "second_strict_passes": 390} | {"first/sklearn": 540, "second/sklearn": 540} | {"False": 540} |
| sklearn auto | {"first_queries": 540, "first_strict_passes": 390, "models_requested": 540, "second_queries": 540, "second_strict_passes": 390} | {"first/sklearn": 540, "second/sklearn": 540} | {"False": 540} |
| Preceding automatic | {"first_queries": 540, "first_strict_passes": 540, "models_requested": 540, "second_queries": 540, "second_strict_passes": 540} | {"first/cpu": 540, "second/cpu": 540} | {"True": 540} |

## Evidence

[Raw requests, setup, source/runtime identities, and all audits](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-e1ba0dcbe694eeac1b99)

[Validated summary](../benchmarks/results/knn-followup-final-2026-10-30-first-query-summary.json)
