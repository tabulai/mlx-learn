# Fresh-model first queries in initialized processes

Initialized isolated persistent processes; normal NumPy/estimator imports and configuration are recorded separately. No model or query prewarm. Every request constructs and fits a new estimator, then times its first query and separately its second query. Public NumPy results establish completion; no harness MLX import/device probe/synchronization runs before the first query. Both outputs are audited after both timers. All requested rounds (default 30), including first model per fixture and first model in worker, remain visible. Earlier models may initialize runtime/compiler caches. This is model-first-use in an initialized process, not fresh-process or machine-cold latency. Five-worker Williams order has a complete balanced 10-round block, repeated three times by default. Any incomplete final block is also retained. No classification/regression operations are added here; the separate full broad and preservation control panels retain those operations and fixtures.

First query includes model preparation deferred by fit. Second query reuses only that fresh model. All rounds remain in the headline median; first-per-fixture and subsequent-model views are shown separately, not selected according to speed. Process initialization/import cost is separate, and must not be confused with the dedicated fresh-process cold benchmark.

Acceptance: **True**. 2 fixtures × 2 fresh models × 5 workers; each model has exactly one retained first and one retained second query. No numerical audit failure removes a valid timing. Ratios are comparator/current; bold values below 1× are slower observations. These descriptive ratios are not confidence intervals.

## Initial process setup (once per worker)

| Worker | NumPy import ms | Estimator import/setup ms | Worker entry→initialized ms | Parent process→initialized ms |
| --- | --- | --- | --- | --- |
| Current automatic | 22.837 | 733.689 | 758.268 | 780.928 |
| Default-thread brute | 23.596 | 713.134 | 738.469 | 760.288 |
| Single-thread brute | 23.022 | 723.937 | 748.692 | 770.026 |
| sklearn auto | 23.125 | 717.604 | 742.448 | 764.095 |
| Preceding automatic | 22.926 | 1029.825 | 1054.507 | 1076.673 |

## fit_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits (raw) | 151 | 0.235 | 0.284 / 1.209× / 1.336× | 0.269 / 1.148× / 1.283× | 0.257 / 1.096× / 1.225× | 0.171 / **0.728×** / **0.772×** |
| Breast Cancer (raw) | 151 | 0.102 | 0.140 / 1.375× / 1.370× | 0.134 / 1.320× / 1.325× | 0.129 / 1.270× / 1.272× | 0.096 / **0.944×** / **0.952×** |

## first_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits (raw) | 151 | 0.828 | 10.895 / 13.151× / 11.117× | 5.319 / 6.421× / 5.262× | 9.612 / 11.602× / 9.637× | 0.637 / **0.769×** / **0.857×** |
| Breast Cancer (raw) | 151 | 0.207 | 1.670 / 8.055× / 8.046× | 0.271 / 1.305× / 1.305× | 1.802 / 8.692× / 8.703× | 0.242 / 1.169× / 1.169× |

## fit_first_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits (raw) | 151 | 1.063 | 11.178 / 10.516× / 8.996× | 5.589 / 5.257× / 4.418× | 9.869 / 9.284× / 7.823× | 0.808 / **0.760×** / **0.836×** |
| Breast Cancer (raw) | 151 | 0.309 | 1.809 / 5.859× / 5.862× | 0.405 / 1.310× / 1.310× | 1.931 / 6.252× / 6.249× | 0.338 / 1.095× / 1.095× |

## second_query_ms

All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.

| Family | Seed | Current median ms | Default-thread brute ms / M / P | Single-thread brute ms / M / P | sklearn auto ms / M / P | Preceding automatic ms / M / P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits (raw) | 151 | 0.445 | 1.945 / 4.373× / 4.829× | 1.208 / 2.716× / 2.830× | 2.786 / 6.264× / 6.507× | 0.493 / 1.108× / 1.160× |
| Breast Cancer (raw) | 151 | 0.137 | 1.785 / 12.992× / 12.994× | 0.229 / 1.666× / 1.667× | 1.923 / 13.993× / 13.989× | 0.183 / 1.333× / 1.333× |

## Every slower or missing all-round median

| Metric | Comparator | Case | Current ms | Comparator ms | M | P |
| --- | --- | --- | --- | --- | --- | --- |
| fit_ms | Preceding automatic | real_digits_raw_k5_seed151 | 0.235 | 0.171 | **0.728×** | **0.772×** |
| fit_ms | Preceding automatic | real_breast_cancer_raw_k5_seed151 | 0.102 | 0.096 | **0.944×** | **0.952×** |
| first_query_ms | Preceding automatic | real_digits_raw_k5_seed151 | 0.828 | 0.637 | **0.769×** | **0.857×** |
| fit_first_query_ms | Preceding automatic | real_digits_raw_k5_seed151 | 1.063 | 0.808 | **0.760×** | **0.836×** |

## first_model_for_fixture

All configurations and metrics are retained in this separate view. Earlier fixtures may already have initialized process/runtime/compiler state; this table is not fresh-process cold latency.

| Case | Worker | fit_ms median ms | first_query_ms median ms | fit_first_query_ms median ms | second_query_ms median ms |
| --- | --- | --- | --- | --- | --- |
| real_digits_raw_k5_seed151 | Current automatic | 0.304 | 1.142 | 1.446 | 0.563 |
| real_digits_raw_k5_seed151 | Default-thread brute | 0.275 | 18.834 | 19.109 | 1.754 |
| real_digits_raw_k5_seed151 | Single-thread brute | 0.251 | 9.507 | 9.758 | 1.354 |
| real_digits_raw_k5_seed151 | sklearn auto | 0.240 | 16.935 | 17.175 | 3.149 |
| real_digits_raw_k5_seed151 | Preceding automatic | 0.189 | 0.714 | 0.903 | 0.542 |
| real_breast_cancer_raw_k5_seed151 | Current automatic | 0.096 | 0.210 | 0.307 | 0.139 |
| real_breast_cancer_raw_k5_seed151 | Default-thread brute | 0.123 | 1.827 | 1.950 | 1.775 |
| real_breast_cancer_raw_k5_seed151 | Single-thread brute | 0.138 | 0.276 | 0.414 | 0.226 |
| real_breast_cancer_raw_k5_seed151 | sklearn auto | 0.127 | 1.674 | 1.801 | 1.995 |
| real_breast_cancer_raw_k5_seed151 | Preceding automatic | 0.109 | 0.250 | 0.358 | 0.185 |

## subsequent_fresh_models

All configurations and metrics are retained in this separate view. Earlier fixtures may already have initialized process/runtime/compiler state; this table is not fresh-process cold latency.

| Case | Worker | fit_ms median ms | first_query_ms median ms | fit_first_query_ms median ms | second_query_ms median ms |
| --- | --- | --- | --- | --- | --- |
| real_digits_raw_k5_seed151 | Current automatic | 0.165 | 0.515 | 0.680 | 0.327 |
| real_digits_raw_k5_seed151 | Default-thread brute | 0.292 | 2.956 | 3.248 | 2.137 |
| real_digits_raw_k5_seed151 | Single-thread brute | 0.288 | 1.132 | 1.420 | 1.063 |
| real_digits_raw_k5_seed151 | sklearn auto | 0.274 | 2.288 | 2.562 | 2.424 |
| real_digits_raw_k5_seed151 | Preceding automatic | 0.153 | 0.560 | 0.713 | 0.444 |
| real_breast_cancer_raw_k5_seed151 | Current automatic | 0.107 | 0.204 | 0.311 | 0.136 |
| real_breast_cancer_raw_k5_seed151 | Default-thread brute | 0.157 | 1.512 | 1.669 | 1.796 |
| real_breast_cancer_raw_k5_seed151 | Single-thread brute | 0.130 | 0.265 | 0.395 | 0.232 |
| real_breast_cancer_raw_k5_seed151 | sklearn auto | 0.131 | 1.930 | 2.061 | 1.850 |
| real_breast_cancer_raw_k5_seed151 | Preceding automatic | 0.083 | 0.235 | 0.318 | 0.181 |

## Every audit failure

| Case | Worker | Round | Phase | Raw pointer |
| --- | --- | --- | --- | --- |
| real_digits_raw_k5_seed151 | Default-thread brute | 0 | first | `/cases/0/variants/sklearn_default/samples/0` |
| real_digits_raw_k5_seed151 | Default-thread brute | 0 | second | `/cases/0/variants/sklearn_default/samples/0` |
| real_digits_raw_k5_seed151 | Default-thread brute | 1 | first | `/cases/0/variants/sklearn_default/samples/1` |
| real_digits_raw_k5_seed151 | Default-thread brute | 1 | second | `/cases/0/variants/sklearn_default/samples/1` |
| real_digits_raw_k5_seed151 | Single-thread brute | 0 | first | `/cases/0/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed151 | Single-thread brute | 0 | second | `/cases/0/variants/sklearn_1/samples/0` |
| real_digits_raw_k5_seed151 | Single-thread brute | 1 | first | `/cases/0/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed151 | Single-thread brute | 1 | second | `/cases/0/variants/sklearn_1/samples/1` |
| real_digits_raw_k5_seed151 | sklearn auto | 0 | first | `/cases/0/variants/sklearn_auto/samples/0` |
| real_digits_raw_k5_seed151 | sklearn auto | 0 | second | `/cases/0/variants/sklearn_auto/samples/0` |
| real_digits_raw_k5_seed151 | sklearn auto | 1 | first | `/cases/0/variants/sklearn_auto/samples/1` |
| real_digits_raw_k5_seed151 | sklearn auto | 1 | second | `/cases/0/variants/sklearn_auto/samples/1` |

## Audit counts and actual routing

| Worker | Audits | Query routing | Native loaded before fit |
| --- | --- | --- | --- |
| Current automatic | {"first_queries": 4, "first_strict_passes": 4, "models_requested": 4, "second_queries": 4, "second_strict_passes": 4} | {"first/cpu": 4, "second/cpu": 4} | {"True": 4} |
| Default-thread brute | {"first_queries": 4, "first_strict_passes": 2, "models_requested": 4, "second_queries": 4, "second_strict_passes": 2} | {"first/sklearn": 4, "second/sklearn": 4} | {"False": 4} |
| Single-thread brute | {"first_queries": 4, "first_strict_passes": 2, "models_requested": 4, "second_queries": 4, "second_strict_passes": 2} | {"first/sklearn": 4, "second/sklearn": 4} | {"False": 4} |
| sklearn auto | {"first_queries": 4, "first_strict_passes": 2, "models_requested": 4, "second_queries": 4, "second_strict_passes": 2} | {"first/sklearn": 4, "second/sklearn": 4} | {"False": 4} |
| Preceding automatic | {"first_queries": 4, "first_strict_passes": 4, "models_requested": 4, "second_queries": 4, "second_strict_passes": 4} | {"first/cpu": 4, "second/cpu": 4} | {"True": 4} |

## Evidence

[Raw requests, setup, source/runtime identities, and all audits](../benchmarks/results/knn-followup-firstquery-smoke-2026-10-30.json)

[Validated summary](../benchmarks/results/knn-followup-firstquery-smoke-summary-2026-10-30.json)
