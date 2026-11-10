# Current mlxlearn speedups versus scikit-learn

Complete current performance panels on the recorded machine. **Speedup = sklearn median time / mlxlearn median time**; above 1× is faster. Fit, prediction, probability estimation, first query and process startup stay separate. Every median uses the retained observations in its source panel.

Machine: **Apple M4 Max**, 128 GiB; macOS-26.6.2-arm64-arm-64bit-Mach-O. Python 3.14.2; MLX 0.32.4, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1. Source commit `d81c994e9e714301e6116d6a71fecd66aa87caed`. Benchmark workers ran serially.

**U** marks an unqualified comparison: its raw timing remains visible, but failed accuracy, convergence, reference or integrity checks prevent a speed claim. A patched estimator that used sklearn is a fallback comparison, never a GPU win. CPU/MLX labels below describe observed fit routing; KNN additionally reports actual query routing. LR host-array inference must not be described as GPU execution merely because fitting used the GPU.

The tables' medians across case ratios are **descriptive**, not significance tests. Raw summaries include failures; qualified summaries show their explicit denominators. No average combines different operations or thread policies. LR and full-grid summaries additionally separate input dtypes and query counts. KNN broad/latency aggregates describe heterogeneous fixture inventories, with different training shapes, query counts and, in the broad panel, k values.

## Operation overview

| Panel | Operation | Input | Queries | Variant | Comparator | Raw median ratio | Raw ratio range | Raw wins/cells | Qualified/cells | Qualified wins/qualified | Qualified median ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| svc-seed41 | fit | float32 | — | Forced CPU | sklearn | 2.34× | 1.673×–3.088× | 5/5 | 5/5 | 5/5 | 2.34× |
| svc-seed41 | first_predict | float32 | 257 | Forced CPU | sklearn | 6.765× | 3.669×–12.54× | 5/5 | 5/5 | 5/5 | 6.765× |
| svc-seed41 | repeat_predict | float32 | 257 | Forced CPU | sklearn | 8.552× | 4.753×–14.24× | 5/5 | 5/5 | 5/5 | 8.552× |
| svc-seed41 | fit | float32 | — | Forced GPU | sklearn | 1.88× | 0.2786×–11.14× | 2/3 | 3/3 | 2/3 | 1.88× |
| svc-seed41 | first_predict | float32 | 257 | Forced GPU | sklearn | 15.98× | 10.47×–33.38× | 3/3 | 3/3 | 3/3 | 15.98× |
| svc-seed41 | repeat_predict | float32 | 257 | Forced GPU | sklearn | 17.67× | 12.5×–40.61× | 3/3 | 3/3 | 3/3 | 17.67× |
| svc-seed42 | fit | float32 | — | Forced CPU | sklearn | 2.454× | 1.589×–2.97× | 5/5 | 5/5 | 5/5 | 2.454× |
| svc-seed42 | first_predict | float32 | 257 | Forced CPU | sklearn | 6.483× | 4.805×–12.79× | 5/5 | 5/5 | 5/5 | 6.483× |
| svc-seed42 | repeat_predict | float32 | 257 | Forced CPU | sklearn | 8.245× | 5.654×–14.94× | 5/5 | 5/5 | 5/5 | 8.245× |
| svc-seed42 | fit | float32 | — | Forced GPU | sklearn | 1.803× | 0.2949×–11.92× | 2/3 | 3/3 | 2/3 | 1.803× |
| svc-seed42 | first_predict | float32 | 257 | Forced GPU | sklearn | 16.3× | 11.27×–33.41× | 3/3 | 3/3 | 3/3 | 16.3× |
| svc-seed42 | repeat_predict | float32 | 257 | Forced GPU | sklearn | 18.39× | 14.1×–38.12× | 3/3 | 3/3 | 3/3 | 18.39× |
| lr-default | fit | float32 | — | Direct auto | sklearn | 1.104× | 0.3162×–5.149× | 15/25 | 23/25 | 14/23 | 1.104× |
| lr-default | fit | float32 | — | Patched auto | sklearn | 0.9983× | 0.8481×–6.41× | 12/25 | 24/25 | 11/24 | 0.9877× |
| lr-default | fit | float32 | — | Forced GPU | sklearn | 1.455× | 0.1921×–7.181× | 14/25 | 23/25 | 13/23 | 1.455× |
| lr-default | fit | float64 | — | Direct auto | sklearn | 1.397× | 0.3205×–4.578× | 17/25 | 23/25 | 16/23 | 1.397× |
| lr-default | fit | float64 | — | Patched auto | sklearn | 0.983× | 0.8432×–5.796× | 12/25 | 24/25 | 11/24 | 0.9757× |
| lr-default | fit | float64 | — | Forced GPU | sklearn | 1.66× | 0.1623×–8.678× | 16/25 | 23/25 | 15/23 | 1.66× |
| lr-requested1 | fit | float32 | — | Direct auto | sklearn | 1.367× | 0.3056×–5.435× | 15/25 | 23/25 | 14/23 | 1.367× |
| lr-requested1 | fit | float32 | — | Patched auto | sklearn | 0.9918× | 0.8552×–5.997× | 12/25 | 24/25 | 11/24 | 0.9827× |
| lr-requested1 | fit | float32 | — | Forced GPU | sklearn | 1.682× | 0.1445×–9.509× | 14/25 | 23/25 | 13/23 | 1.682× |
| lr-requested1 | fit | float64 | — | Direct auto | sklearn | 1.536× | 0.3256×–5.751× | 18/25 | 23/25 | 17/23 | 1.536× |
| lr-requested1 | fit | float64 | — | Patched auto | sklearn | 0.9918× | 0.8852×–6.576× | 12/25 | 24/25 | 11/24 | 0.9852× |
| lr-requested1 | fit | float64 | — | Forced GPU | sklearn | 1.907× | 0.1957×–7.675× | 16/25 | 23/25 | 15/23 | 1.907× |
| lr-strict4096-default | fit | float32 | — | Direct auto | sklearn | 2.264× | 2.264× | 1/1 | 1/1 | 1/1 | 2.264× |
| lr-strict4096-default | fit | float32 | — | Patched auto | sklearn | 2.756× | 2.756× | 1/1 | 1/1 | 1/1 | 2.756× |
| lr-strict4096-default | fit | float32 | — | Forced GPU | sklearn | 2.754× | 2.754× | 1/1 | 1/1 | 1/1 | 2.754× |
| lr-strict4096-default | fit | float64 | — | Direct auto | sklearn | 2.887× | 2.887× | 1/1 | 1/1 | 1/1 | 2.887× |
| lr-strict4096-default | fit | float64 | — | Patched auto | sklearn | 3.53× | 3.53× | 1/1 | 1/1 | 1/1 | 3.53× |
| lr-strict4096-default | fit | float64 | — | Forced GPU | sklearn | 3.544× | 3.544× | 1/1 | 1/1 | 1/1 | 3.544× |
| lr-strict4096-requested1 | fit | float32 | — | Direct auto | sklearn | 3.957× | 3.957× | 1/1 | 1/1 | 1/1 | 3.957× |
| lr-strict4096-requested1 | fit | float32 | — | Patched auto | sklearn | 4.208× | 4.208× | 1/1 | 1/1 | 1/1 | 4.208× |
| lr-strict4096-requested1 | fit | float32 | — | Forced GPU | sklearn | 4.04× | 4.04× | 1/1 | 1/1 | 1/1 | 4.04× |
| lr-strict4096-requested1 | fit | float64 | — | Direct auto | sklearn | 4.853× | 4.853× | 1/1 | 1/1 | 1/1 | 4.853× |
| lr-strict4096-requested1 | fit | float64 | — | Patched auto | sklearn | 5.631× | 5.631× | 1/1 | 1/1 | 1/1 | 5.631× |
| lr-strict4096-requested1 | fit | float64 | — | Forced GPU | sklearn | 5.477× | 5.477× | 1/1 | 1/1 | 1/1 | 5.477× |
| grid-default-seed41 | knn/fit | float64 | — | Direct auto | sklearn brute/default | 1.024× | 0.6711×–1.739× | 10/18 | 18/18 | 10/18 | 1.024× |
| grid-default-seed41 | knn/fit | float64 | — | Direct auto | sklearn auto/default | 1.283× | 0.6832×–161.5× | 10/18 | 18/18 | 10/18 | 1.283× |
| grid-default-seed41 | knn/fit | float64 | — | Patched auto | sklearn brute/default | 1.056× | 0.6604×–1.684× | 9/18 | 18/18 | 9/18 | 1.056× |
| grid-default-seed41 | knn/fit | float64 | — | Patched auto | sklearn auto/default | 1.297× | 0.6724×–179.7× | 10/18 | 18/18 | 10/18 | 1.297× |
| grid-default-seed41 | knn/kneighbors | float64 | 1 | Direct auto | sklearn brute/default | 1.396× | 0.8904×–8.526× | 15/18 | 18/18 | 15/18 | 1.396× |
| grid-default-seed41 | knn/kneighbors | float64 | 1 | Direct auto | sklearn auto/default | 1.154× | 0.2443×–5.527× | 13/18 | 18/18 | 13/18 | 1.154× |
| grid-default-seed41 | knn/kneighbors | float64 | 1 | Patched auto | sklearn brute/default | 1.579× | 0.8555×–8.812× | 15/18 | 18/18 | 15/18 | 1.579× |
| grid-default-seed41 | knn/kneighbors | float64 | 1 | Patched auto | sklearn auto/default | 1.151× | 0.2926×–5.618× | 12/18 | 18/18 | 12/18 | 1.151× |
| grid-default-seed41 | knn/kneighbors | float64 | 256 | Direct auto | sklearn brute/default | 1.618× | 0.7622×–3.493× | 14/18 | 18/18 | 14/18 | 1.618× |
| grid-default-seed41 | knn/kneighbors | float64 | 256 | Direct auto | sklearn auto/default | 1.824× | 0.712×–6.216× | 15/18 | 18/18 | 15/18 | 1.824× |
| grid-default-seed41 | knn/kneighbors | float64 | 256 | Patched auto | sklearn brute/default | 1.642× | 0.7482×–3.46× | 14/18 | 18/18 | 14/18 | 1.642× |
| grid-default-seed41 | knn/kneighbors | float64 | 256 | Patched auto | sklearn auto/default | 1.822× | 0.7117×–6.263× | 15/18 | 18/18 | 15/18 | 1.822× |
| grid-default-seed41 | knn/kneighbors | float64 | 4096 | Direct auto | sklearn brute/default | 1.813× | 0.5929×–4.951× | 15/18 | 18/18 | 15/18 | 1.813× |
| grid-default-seed41 | knn/kneighbors | float64 | 4096 | Direct auto | sklearn auto/default | 2.424× | 0.6021×–11.77× | 16/18 | 18/18 | 16/18 | 2.424× |
| grid-default-seed41 | knn/kneighbors | float64 | 4096 | Patched auto | sklearn brute/default | 1.779× | 0.5906×–6.118× | 15/18 | 18/18 | 15/18 | 1.779× |
| grid-default-seed41 | knn/kneighbors | float64 | 4096 | Patched auto | sklearn auto/default | 2.613× | 0.5998×–16.76× | 16/18 | 18/18 | 16/18 | 2.613× |
| grid-default-seed41 | svc/fit | float64 | — | Direct auto | sklearn | 2.686× | 1.318×–15.86× | 8/8 | 8/8 | 8/8 | 2.686× |
| grid-default-seed41 | svc/fit | float64 | — | Patched auto | sklearn | 0.9967× | 0.9234×–16.1× | 3/8 | 8/8 | 3/8 | 0.9967× |
| grid-default-seed41 | svc/predict | float64 | 1000 | Direct auto | sklearn | 36.18× | 19.27×–96.4× | 4/4 | 4/4 | 4/4 | 36.18× |
| grid-default-seed41 | svc/predict | float64 | 1000 | Patched auto | sklearn | 0.9974× | 0.9941×–95.47× | 1/4 | 4/4 | 1/4 | 0.9974× |
| lr-accuracy-matched-default | fit | float32 | — | Direct auto | sklearn | 1.621× | 0.5659×–1.884× | 2/3 | 1/3 | 1/1 | 1.884× |
| lr-accuracy-matched-default | fit | float32 | — | Patched auto | sklearn | 2.009× | 0.9546×–2.211× | 2/3 | 1/3 | 1/1 | 2.009× |
| lr-accuracy-matched-default | fit | float32 | — | Forced GPU | sklearn | 1.977× | 0.3635×–1.983× | 2/3 | 1/3 | 1/1 | 1.977× |
| lr-accuracy-matched-default | fit | float64 | — | Direct auto | sklearn | 1.947× | 0.5642×–3.322× | 2/3 | 3/3 | 2/3 | 1.947× |
| lr-accuracy-matched-default | fit | float64 | — | Patched auto | sklearn | 2.378× | 0.9411×–3.845× | 2/3 | 3/3 | 2/3 | 2.378× |
| lr-accuracy-matched-default | fit | float64 | — | Forced GPU | sklearn | 2.679× | 0.4058×–3.98× | 2/3 | 3/3 | 2/3 | 2.679× |
| lr-accuracy-matched-requested1 | fit | float32 | — | Direct auto | sklearn | 2.715× | 0.5662×–2.779× | 2/3 | 1/3 | 1/1 | 2.715× |
| lr-accuracy-matched-requested1 | fit | float32 | — | Patched auto | sklearn | 3.052× | 0.954×–3.615× | 2/3 | 1/3 | 1/1 | 3.052× |
| lr-accuracy-matched-requested1 | fit | float32 | — | Forced GPU | sklearn | 2.796× | 0.3653×–3.611× | 2/3 | 1/3 | 1/1 | 2.796× |
| lr-accuracy-matched-requested1 | fit | float64 | — | Direct auto | sklearn | 3.589× | 0.5664×–5.144× | 2/3 | 3/3 | 2/3 | 3.589× |
| lr-accuracy-matched-requested1 | fit | float64 | — | Patched auto | sklearn | 4.959× | 0.9314×–6.331× | 2/3 | 3/3 | 2/3 | 4.959× |
| lr-accuracy-matched-requested1 | fit | float64 | — | Forced GPU | sklearn | 4.97× | 0.3455×–6.267× | 2/3 | 3/3 | 2/3 | 4.97× |

## KNN: broad workload, first query and startup

The broad panel uses 72 float32 fixtures, including raw Iris, Wine, Breast Cancer and Digits and synthetic shape/distribution cases. It retains ten operations and three sklearn controls. The **threadpoolctl-limited** control does not manage Apple Accelerate; it must not be labeled a verified globally single-threaded process. Default brute and auto retain their default pools.

KNN ratios below are raw descriptive comparisons, retaining every comparator audit failure. Strict comparator failures include exact-distance-valid tie alternatives and distance/prediction mismatches; the audit table separates their recorded causes. A passing current implementation does not turn a failed comparator check into a qualified aggregate.

### knn-broad

72 fixtures; 15 retained repetitions; recorded current accuracy/integrity acceptance: **True**.

| Operation / phase | Comparator | Current ms range | sklearn ms range | Raw median ratio | Raw ratio range | Raw wins/cells | Aggregate qualification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fit | sklearn brute/default | 0.07371–0.5848 | 0.1266–0.6088 | 1.163× | 0.9348×–1.861× | 67/72 | **U** — strict audit failures retained |
| fit | sklearn brute/threadpoolctl-limited | 0.07371–0.5848 | 0.1263–0.5992 | 1.18× | 0.9644×–1.803× | 68/72 | **U** — strict audit failures retained |
| fit | sklearn auto/default | 0.07371–0.5848 | 0.1394–1.641 | 1.191× | 0.9414×–11.77× | 70/72 | **U** — strict audit failures retained |
| fit_query | sklearn brute/default | 0.1164–3.123 | 0.6279–9.412 | 3.153× | 0.7945×–11.47× | 69/72 | **U** — strict audit failures retained |
| fit_query | sklearn brute/threadpoolctl-limited | 0.1164–3.123 | 0.2021–43.44 | 1.698× | 1.248×–15.77× | 72/72 | **U** — strict audit failures retained |
| fit_query | sklearn auto/default | 0.1164–3.123 | 0.3087–9.153 | 3.081× | 1.879×–12.59× | 72/72 | **U** — strict audit failures retained |
| kneighbors | sklearn brute/default | 0.09342–2.742 | 0.5742–8.901 | 3.85× | 0.8185×–16.53× | 70/72 | **U** — strict audit failures retained |
| kneighbors | sklearn brute/threadpoolctl-limited | 0.09342–2.742 | 0.1342–44.6 | 1.853× | 1.308×–22.24× | 72/72 | **U** — strict audit failures retained |
| kneighbors | sklearn auto/default | 0.09342–2.742 | 0.2042–9.008 | 3.71× | 1.861×–18.17× | 72/72 | **U** — strict audit failures retained |
| indices | sklearn brute/default | 0.08729–2.772 | 0.294–8.871 | 3.334× | 0.701×–14.95× | 69/72 | **U** — strict audit failures retained |
| indices | sklearn brute/threadpoolctl-limited | 0.08729–2.772 | 0.1282–44.58 | 1.78× | 1.316×–22.14× | 72/72 | **U** — strict audit failures retained |
| indices | sklearn auto/default | 0.08729–2.772 | 0.201–8.615 | 3.122× | 1.797×–15.61× | 72/72 | **U** — strict audit failures retained |
| predict_uniform | sklearn brute/default | 0.09717–2.77 | 0.3118–8.655 | 3.167× | 0.7478×–14.83× | 69/72 | **U** — strict audit failures retained |
| predict_uniform | sklearn brute/threadpoolctl-limited | 0.09717–2.77 | 0.1705–44.52 | 1.781× | 1.33×–21.5× | 72/72 | **U** — strict audit failures retained |
| predict_uniform | sklearn auto/default | 0.09717–2.77 | 0.3935–8.361 | 3.535× | 1.881×–15.12× | 72/72 | **U** — strict audit failures retained |
| proba_uniform | sklearn brute/default | 0.1015–2.813 | 0.3085–8.697 | 3.243× | 0.7336×–14.32× | 69/72 | **U** — strict audit failures retained |
| proba_uniform | sklearn brute/threadpoolctl-limited | 0.1015–2.813 | 0.1617–44.62 | 1.807× | 1.3×–20.74× | 72/72 | **U** — strict audit failures retained |
| proba_uniform | sklearn auto/default | 0.1015–2.813 | 0.2268–8.841 | 3.141× | 1.85×–15.07× | 72/72 | **U** — strict audit failures retained |
| regress_uniform | sklearn brute/default | 0.09912–2.775 | 0.2775–8.591 | 3.233× | 0.6657×–14.46× | 69/72 | **U** — strict audit failures retained |
| regress_uniform | sklearn brute/threadpoolctl-limited | 0.09912–2.775 | 0.1482–44.56 | 1.794× | 1.295×–21.97× | 72/72 | **U** — strict audit failures retained |
| regress_uniform | sklearn auto/default | 0.09912–2.775 | 0.2103–8.356 | 3.197× | 1.808×–14.74× | 72/72 | **U** — strict audit failures retained |
| predict_distance | sklearn brute/default | 0.111–2.799 | 0.6333–9.194 | 3.791× | 0.8951×–16.13× | 69/72 | **U** — strict audit failures retained |
| predict_distance | sklearn brute/threadpoolctl-limited | 0.111–2.799 | 0.1853–44.69 | 1.838× | 1.333×–21.27× | 72/72 | **U** — strict audit failures retained |
| predict_distance | sklearn auto/default | 0.111–2.799 | 0.2503–9.411 | 3.603× | 1.891×–16.32× | 72/72 | **U** — strict audit failures retained |
| proba_distance | sklearn brute/default | 0.1151–2.755 | 0.5984–9.148 | 3.696× | 0.8663×–19.49× | 70/72 | **U** — strict audit failures retained |
| proba_distance | sklearn brute/threadpoolctl-limited | 0.1151–2.755 | 0.1641–44.69 | 1.811× | 1.301×–21.95× | 72/72 | **U** — strict audit failures retained |
| proba_distance | sklearn auto/default | 0.1151–2.755 | 0.231–8.808 | 3.503× | 1.822×–20.46× | 72/72 | **U** — strict audit failures retained |
| regress_distance | sklearn brute/default | 0.1092–2.808 | 0.5634–8.831 | 3.711× | 0.8212×–14.54× | 69/72 | **U** — strict audit failures retained |
| regress_distance | sklearn brute/threadpoolctl-limited | 0.1092–2.808 | 0.1551–44.52 | 1.802× | 1.287×–20.7× | 72/72 | **U** — strict audit failures retained |
| regress_distance | sklearn auto/default | 0.1092–2.808 | 0.2187–8.736 | 3.527× | 1.777×–15.56× | 72/72 | **U** — strict audit failures retained |

| Variant | Strict passed / requests | Recorded failure causes |
| --- | --- | --- |
| current | 14472/14472 | {} |
| sklearn brute/default | 13114/14472 | {"canonical_prediction_mismatch": 160, "reported_distance_error": 793, "valid_exact_tie_alternative": 405} |
| sklearn brute/threadpoolctl-limited | 13034/14472 | {"canonical_prediction_mismatch": 160, "canonical_prediction_mismatch_on_boundary_ties": 80, "reported_distance_error": 793, "valid_exact_tie_alternative": 405} |
| sklearn auto/default | 13114/14472 | {"canonical_prediction_mismatch": 160, "reported_distance_error": 793, "valid_exact_tie_alternative": 405} |

Actual retained query backends: `{"cpu": 2700, "mlx": 8100}`; fitted-storage backends: `{"cpu": 5400, "mlx": 5400}`.

### knn-cold

18 fixtures; 5 retained repetitions; recorded current accuracy/integrity acceptance: **True**.

| Operation / phase | Comparator | Current ms range | sklearn ms range | Raw median ratio | Raw ratio range | Raw wins/cells | Aggregate qualification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| numpy_import_ms | sklearn brute/default | 18.52–18.98 | 18.34–19.02 | 0.9951× | 0.9718×–1.015× | 4/18 | **U** — strict audit failures retained |
| fixture_load_ms | sklearn brute/default | 1.211–1.455 | 1.188–1.451 | 0.9885× | 0.9527×–1.017× | 6/18 | **U** — strict audit failures retained |
| estimator_setup_ms | sklearn brute/default | 348.7–355.1 | 399–406 | 1.141× | 1.13×–1.156× | 18/18 | **U** — strict audit failures retained |
| fit_ms | sklearn brute/default | 0.8403–1.021 | 0.9886–1.06 | 1.167× | 0.9972×–1.208× | 17/18 | **U** — strict audit failures retained |
| query_ms | sklearn brute/default | 0.08204–0.5923 | 11.24–12.89 | 98.53× | 20.83×–142.9× | 18/18 | **U** — strict audit failures retained |
| fit_query_ms | sklearn brute/default | 0.9227–1.468 | 12.24–13.95 | 11.87× | 9.086×–13.58× | 18/18 | **U** — strict audit failures retained |
| estimator_to_result_ms | sklearn brute/default | 349.6–356.5 | 411.4–419.1 | 1.173× | 1.161×–1.189× | 18/18 | **U** — strict audit failures retained |
| worker_entry_to_result_ms | sklearn brute/default | 371.2–379.1 | 433.1–440.6 | 1.163× | 1.15×–1.179× | 18/18 | **U** — strict audit failures retained |
| process_start_to_result_ms | sklearn brute/default | 395.2–403.6 | 456.7–465.2 | 1.153× | 1.141×–1.168× | 18/18 | **U** — strict audit failures retained |
| process_elapsed_ms | sklearn brute/default | 450.5–460.8 | 521.5–530.9 | 1.153× | 1.14×–1.169× | 18/18 | **U** — strict audit failures retained |
| numpy_import_ms | sklearn brute/threadpoolctl-limited | 18.52–18.98 | 18.39–19.23 | 0.9956× | 0.9703×–1.016× | 6/18 | **U** — strict audit failures retained |
| fixture_load_ms | sklearn brute/threadpoolctl-limited | 1.211–1.455 | 1.208–1.428 | 0.997× | 0.9379×–1.059× | 7/18 | **U** — strict audit failures retained |
| estimator_setup_ms | sklearn brute/threadpoolctl-limited | 348.7–355.1 | 404.9–413.5 | 1.166× | 1.147×–1.179× | 18/18 | **U** — strict audit failures retained |
| fit_ms | sklearn brute/threadpoolctl-limited | 0.8403–1.021 | 0.9789–1.054 | 1.154× | 1.011×–1.191× | 18/18 | **U** — strict audit failures retained |
| query_ms | sklearn brute/threadpoolctl-limited | 0.08204–0.5923 | 4.551–5.563 | 41.42× | 9.361×–56.15× | 18/18 | **U** — strict audit failures retained |
| fit_query_ms | sklearn brute/threadpoolctl-limited | 0.9227–1.468 | 5.535–6.607 | 5.378× | 4.434×–6.168× | 18/18 | **U** — strict audit failures retained |
| estimator_to_result_ms | sklearn brute/threadpoolctl-limited | 349.6–356.5 | 410.8–419.5 | 1.178× | 1.159×–1.192× | 18/18 | **U** — strict audit failures retained |
| worker_entry_to_result_ms | sklearn brute/threadpoolctl-limited | 371.2–379.1 | 432.4–440.8 | 1.167× | 1.15×–1.181× | 18/18 | **U** — strict audit failures retained |
| process_start_to_result_ms | sklearn brute/threadpoolctl-limited | 395.2–403.6 | 456.3–464.7 | 1.155× | 1.14×–1.17× | 18/18 | **U** — strict audit failures retained |
| process_elapsed_ms | sklearn brute/threadpoolctl-limited | 450.5–460.8 | 517.2–530.6 | 1.149× | 1.132×–1.159× | 18/18 | **U** — strict audit failures retained |
| numpy_import_ms | sklearn auto/default | 18.52–18.98 | 18.35–19.28 | 0.9946× | 0.969×–1.033× | 8/18 | **U** — strict audit failures retained |
| fixture_load_ms | sklearn auto/default | 1.211–1.455 | 1.194–1.435 | 0.9877× | 0.9554×–1.033× | 4/18 | **U** — strict audit failures retained |
| estimator_setup_ms | sklearn auto/default | 348.7–355.1 | 397.7–409.6 | 1.149× | 1.123×–1.157× | 18/18 | **U** — strict audit failures retained |
| fit_ms | sklearn auto/default | 0.8403–1.021 | 1.004–2.515 | 1.248× | 1.027×–2.682× | 18/18 | **U** — strict audit failures retained |
| query_ms | sklearn auto/default | 0.08204–0.5923 | 0.2993–12.69 | 12.4× | 3.201×–82.28× | 18/18 | **U** — strict audit failures retained |
| fit_query_ms | sklearn auto/default | 0.9227–1.468 | 1.407–13.71 | 5.888× | 1.485×–11.79× | 18/18 | **U** — strict audit failures retained |
| estimator_to_result_ms | sklearn auto/default | 349.6–356.5 | 399.2–422.6 | 1.159× | 1.124×–1.188× | 18/18 | **U** — strict audit failures retained |
| worker_entry_to_result_ms | sklearn auto/default | 371.2–379.1 | 420.3–445.6 | 1.149× | 1.116×–1.181× | 18/18 | **U** — strict audit failures retained |
| process_start_to_result_ms | sklearn auto/default | 395.2–403.6 | 443.5–469.7 | 1.139× | 1.109×–1.17× | 18/18 | **U** — strict audit failures retained |
| process_elapsed_ms | sklearn auto/default | 450.5–460.8 | 505.4–533.9 | 1.138× | 1.109×–1.171× | 18/18 | **U** — strict audit failures retained |

| Variant | Recorded query audits | Actual query routing |
| --- | --- | --- |
| current | {"completed_timings": 90, "requests": 90, "strict_passes": 90} | {"cpu": 90} |
| sklearn brute/default | {"completed_timings": 90, "requests": 90, "strict_passes": 65} | {"sklearn": 90} |
| sklearn brute/threadpoolctl-limited | {"completed_timings": 90, "requests": 90, "strict_passes": 65} | {"sklearn": 90} |
| sklearn auto/default | {"completed_timings": 90, "requests": 90, "strict_passes": 65} | {"sklearn": 90} |

### knn-first-query

18 fixtures; 30 retained repetitions; recorded current accuracy/integrity acceptance: **True**.

| Operation / phase | Comparator | Current ms range | sklearn ms range | Raw median ratio | Raw ratio range | Raw wins/cells | Aggregate qualification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fit_ms | sklearn brute/default | 0.03608–0.1383 | 0.09342–0.1659 | 1.996× | 1.183×–2.599× | 18/18 | **U** — strict audit failures retained |
| first_query_ms | sklearn brute/default | 0.05294–0.454 | 0.5363–3.071 | 11.78× | 6.596×–35.65× | 18/18 | **U** — strict audit failures retained |
| fit_first_query_ms | sklearn brute/default | 0.09354–0.5384 | 0.6236–3.214 | 8.026× | 5.159×–17.76× | 18/18 | **U** — strict audit failures retained |
| second_query_ms | sklearn brute/default | 0.04121–0.2938 | 0.5666–3.077 | 16.3× | 10.39×–42.42× | 18/18 | **U** — strict audit failures retained |
| fit_ms | sklearn brute/threadpoolctl-limited | 0.03608–0.1383 | 0.08767–0.1786 | 1.997× | 1.161×–2.515× | 18/18 | **U** — strict audit failures retained |
| first_query_ms | sklearn brute/threadpoolctl-limited | 0.05294–0.454 | 0.09519–1.033 | 1.844× | 1.199×–3.176× | 18/18 | **U** — strict audit failures retained |
| fit_first_query_ms | sklearn brute/threadpoolctl-limited | 0.09354–0.5384 | 0.1855–1.216 | 1.973× | 1.424×–2.259× | 18/18 | **U** — strict audit failures retained |
| second_query_ms | sklearn brute/threadpoolctl-limited | 0.04121–0.2938 | 0.0824–0.9826 | 1.885× | 1.457×–3.38× | 18/18 | **U** — strict audit failures retained |
| fit_ms | sklearn auto/default | 0.03608–0.1383 | 0.1119–1.559 | 2.592× | 1.227×–17.17× | 18/18 | **U** — strict audit failures retained |
| first_query_ms | sklearn auto/default | 0.05294–0.454 | 0.1602–3.063 | 5.016× | 2.72×–20.14× | 18/18 | **U** — strict audit failures retained |
| fit_first_query_ms | sklearn auto/default | 0.09354–0.5384 | 0.2926–3.245 | 7.356× | 2.959×–11.44× | 18/18 | **U** — strict audit failures retained |
| second_query_ms | sklearn auto/default | 0.04121–0.2938 | 0.1494–2.954 | 6.894× | 2.716×–23.84× | 18/18 | **U** — strict audit failures retained |

| Variant | Recorded query audits | Actual query routing |
| --- | --- | --- |
| current | {"first_queries": 540, "first_strict_passes": 540, "models_requested": 540, "second_queries": 540, "second_strict_passes": 540} | {"first/cpu": 540, "second/cpu": 540} |
| sklearn brute/default | {"first_queries": 540, "first_strict_passes": 390, "models_requested": 540, "second_queries": 540, "second_strict_passes": 390} | {"first/sklearn": 540, "second/sklearn": 540} |
| sklearn brute/threadpoolctl-limited | {"first_queries": 540, "first_strict_passes": 390, "models_requested": 540, "second_queries": 540, "second_strict_passes": 390} | {"first/sklearn": 540, "second/sklearn": 540} |
| sklearn auto/default | {"first_queries": 540, "first_strict_passes": 390, "models_requested": 540, "second_queries": 540, "second_strict_passes": 390} | {"first/sklearn": 540, "second/sklearn": 540} |

Fresh-model first/second-query measurements use initialized workers; they exclude interpreter and import startup. Fresh-process measurements create a new interpreter for every observation. Process-start-to-result includes imports, fixture loading and IPC; process elapsed additionally includes post-result audits. Fresh processes do not imply cold OS, filesystem or Metal disk caches.

## Logistic regression

The default and requested-one-thread panels each retain 25 fixtures × two public input dtypes. Direct automatic, patched automatic and forced GPU are separate comparisons. The strict 4,096-feature supplements use their own tolerance and never replace a failed default-tolerance row. Optional accuracy-matched supplements use tol=1e-6 on the three designated difficult fixtures. A failure of agreement with sklearn does not establish which optimizer is more accurate: independent objective and gradient diagnostics are retained in the summary. Different stopping points can produce different probabilities even when one has a lower objective. Thread environment and loaded pool information are retained in each input JSON. The requested one-thread panel controls Accelerate only when VECLIB_MAXIMUM_THREADS was set before process startup.

The objective gate is **one-sided**: `(current objective − sklearn objective) / abs(sklearn objective) ≤ 1e-5`. A lower current objective passes; objective equality is not required. Fits also require the independently evaluated maximum gradient to meet the requested tolerance, convergence, at least 99.5% training prediction agreement, and a passing sklearn reference. Probability comparisons separately require absolute tolerance 0.002 plus relative tolerance 1e-7, valid normalization and at least 99.5% class agreement.

**Corrected float64 certificate:** every official LR panel uses schema version 2 and `float64-original-data-v2`. Training inputs, coefficients, intercepts and weights are explicitly promoted before computing logits, residuals, objectives and gradients. The six panels were rerun with unchanged timing parameters, seeds and thread policies. Their earlier runs are preserved as superseded audit evidence and are not combined with the official results.

Fit cells show **current milliseconds / sklearn-current ratio**. Backend counts cover measured fits; the stock milliseconds are shared by all three comparisons in that row.

### lr-default

tol=0.0001; max_iter=200; pre-import thread environment: `{"BLIS_NUM_THREADS": null, "MKL_NUM_THREADS": null, "NUMEXPR_NUM_THREADS": null, "OMP_NUM_THREADS": null, "OPENBLAS_NUM_THREADS": null, "VECLIB_MAXIMUM_THREADS": null}`.

| Case | Shape | Input | Seed | sklearn ms | Direct auto ms / ratio | Patched auto ms / ratio | Forced GPU ms / ratio | Observed fit backends |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary_small | 5000×32 | float32 | 41 | 1.29 | 2.105 / 0.6128× | 1.397 / 0.9236× | 3.441 / 0.3749× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_medium | 20000×128 | float32 | 41 | 3.83 | 4.577 / 0.8369× | 4.271 / 0.8968× | 5.333 / 0.7183× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_wide | 20000×1024 | float32 | 41 | 13.06 | 11.85 / 1.102× | 10.11 / 1.292× | 8.975 / 1.455× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_large | 60000×1024 | float32 | 41 | 33.77 | 21.25 / 1.589× | 15.83 / 2.134× | 16.13 / 2.093× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| multiclass_weighted | 20000×128 | float32 | 41 | 16.07 | 15.01 / 1.071× | 16.45 / 0.977× | 23.2 / 0.6927× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_ill_conditioned | 5000×32 | float32 | 41 | 3.591 | 6.626 / 0.542× | 3.597 / 0.9983× | 7.734 / 0.4643× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_boundary | 20000×512 | float32 | 41 | 8.315 | 11.67 / 0.7125× | 9.805 / 0.8481× | 9.8 / 0.8484× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_tall_crossover | 250000×32 | float32 | 41 | 32.94 | 9.435 / 3.492× | 10.12 / 3.256× | 9.291 / 3.546× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_tall | 500000×32 | float32 | 41 | 70.24 | 15.98 / 4.397× | 14.13 / 4.972× | 13.89 / 5.058× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_tall_wide | 500000×128 | float32 | 41 | 74.2 | 14.41 / 5.149× | 12.9 / 5.751× | 14.03 / 5.29× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_40k_2048 | 40000×2048 | float32 | 41 | 40.41 | 21.85 / 1.85× | 20.24 / 1.997× | 20.28 / 1.993× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_20k_4096 | 20000×4096 | float32 | 41 | 63.56 | 34.53 / 1.841× | 30.74 / 2.067× | 32.58 / 1.951× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_weighted | 20000×1024 | float32 | 41 | 13.74 | 12.09 / 1.136× | 10.6 / 1.296× | 10.92 / 1.258× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_1000_16 | 1000×16 | float32 | 41 | 0.8708 | 1.823 / 0.4776× | 0.9341 / 0.9322× | 3.6 / 0.2419× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_1000_128 | 1000×128 | float32 | 41 | 0.967 | 3.058 / 0.3162× | 1.1 / 0.8788× | 5.034 / 0.1921× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_1000_1024 | 1000×1024 | float32 | 41 | 2.152 | 4.714 / 0.4565× **U** | 2.464 / 0.8735× | 6.899 / 0.3119× **U** | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_10000_16 | 10000×16 | float32 | 41 | 1.976 | 2.507 / 0.7884× | 2.077 / 0.9513× | 3.816 / 0.5178× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_10000_128 | 10000×128 | float32 | 41 | 2.19 | 3.135 / 0.6984× | 2.472 / 0.8859× | 4.839 / 0.4525× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_10000_1024 | 10000×1024 | float32 | 41 | 8.148 | 13.67 / 0.596× | 9.399 / 0.8669× | 16.52 / 0.4932× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_100000_16 | 100000×16 | float32 | 41 | 12.78 | 8.378 / 1.525× | 13.14 / 0.9722× | 6.296 / 2.03× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_100000_128 | 100000×128 | float32 | 41 | 17.41 | 15.78 / 1.104× | 19.42 / 0.8966× | 9.738 / 1.788× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_100000_1024 | 100000×1024 | float32 | 41 | 56.71 | 26.27 / 2.159× | 18.73 / 3.028× | 19.6 / 2.893× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_500000_16 | 500000×16 | float32 | 41 | 71.62 | 16.09 / 4.451× | 11.17 / 6.41× | 9.974 / 7.181× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_500000_1024 | 500000×1024 | float32 | 41 | 273.2 | 80.44 / 3.397× | 77.1 / 3.544× | 76.11 / 3.59× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| multiclass_weighted_wide | 20000×1024 | float32 | 41 | 73.33 | 43.85 / 1.672× **U** | 24.54 / 2.988× **U** | 24.26 / 3.023× **U** | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_small | 5000×32 | float64 | 41 | 1.237 | 2.195 / 0.5635× | 1.401 / 0.8824× | 3.499 / 0.3535× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_medium | 20000×128 | float64 | 41 | 4.863 | 4.656 / 1.044× | 5.452 / 0.8921× | 6.098 / 0.7975× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_wide | 20000×1024 | float64 | 41 | 19.61 | 17.26 / 1.136× | 17.46 / 1.123× | 11.81 / 1.66× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_large | 60000×1024 | float64 | 41 | 55.55 | 25.02 / 2.221× | 19.67 / 2.824× | 19.45 / 2.856× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| multiclass_weighted | 20000×128 | float64 | 41 | 25.7 | 15.99 / 1.607× | 26.14 / 0.983× | 23.32 / 1.102× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_ill_conditioned | 5000×32 | float64 | 41 | 3.361 | 6.496 / 0.5174× | 3.571 / 0.9413× | 5.498 / 0.6113× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_boundary | 20000×512 | float64 | 41 | 12.17 | 11.55 / 1.054× | 13.54 / 0.8989× | 11.52 / 1.056× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_tall_crossover | 250000×32 | float64 | 41 | 37.26 | 11.37 / 3.276× | 11.98 / 3.111× | 10.89 / 3.42× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_tall | 500000×32 | float64 | 41 | 79.24 | 17.31 / 4.578× | 13.67 / 5.796× | 9.131 / 8.678× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_tall_wide | 500000×128 | float64 | 41 | 92.68 | 22.01 / 4.211× | 19.52 / 4.747× | 18.1 / 5.121× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_40k_2048 | 40000×2048 | float64 | 41 | 68.62 | 29.12 / 2.356× | 25.83 / 2.656× | 25.45 / 2.696× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_20k_4096 | 20000×4096 | float64 | 41 | 118.4 | 43.4 / 2.728× | 37.83 / 3.129× | 36.44 / 3.249× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_weighted | 20000×1024 | float64 | 41 | 21.56 | 15.44 / 1.397× | 13.46 / 1.601× | 13.65 / 1.58× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_1000_16 | 1000×16 | float64 | 41 | 0.9007 | 1.965 / 0.4583× | 0.9903 / 0.9095× | 3.516 / 0.2561× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_1000_128 | 1000×128 | float64 | 41 | 0.9728 | 3.035 / 0.3205× | 1.154 / 0.8432× | 4.935 / 0.1971× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_1000_1024 | 1000×1024 | float64 | 41 | 2.327 | 4.134 / 0.563× **U** | 2.577 / 0.9032× | 14.34 / 0.1623× **U** | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_10000_16 | 10000×16 | float64 | 41 | 1.979 | 2.591 / 0.7639× | 2.069 / 0.9567× | 3.733 / 0.5302× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_10000_128 | 10000×128 | float64 | 41 | 2.365 | 3.359 / 0.704× | 2.804 / 0.8433× | 5.055 / 0.4677× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_10000_1024 | 10000×1024 | float64 | 41 | 11.8 | 13.85 / 0.8519× | 13.05 / 0.9043× | 17.16 / 0.6875× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_100000_16 | 100000×16 | float64 | 41 | 12.85 | 8.857 / 1.451× | 13.27 / 0.9684× | 6.688 / 1.921× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_100000_128 | 100000×128 | float64 | 41 | 22.12 | 16.06 / 1.377× | 23.68 / 0.9344× | 11.13 / 1.988× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_100000_1024 | 100000×1024 | float64 | 41 | 91.45 | 38.4 / 2.381× | 27.55 / 3.319× | 29.05 / 3.148× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_500000_16 | 500000×16 | float64 | 41 | 73.21 | 16.93 / 4.325× | 13.25 / 5.526× | 8.907 / 8.219× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_500000_1024 | 500000×1024 | float64 | 41 | 449.5 | 122.2 / 3.679× | 110.4 / 4.073× | 109.4 / 4.107× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| multiclass_weighted_wide | 20000×1024 | float64 | 41 | 112.6 | 43.95 / 2.563× **U** | 26.97 / 4.177× **U** | 38.93 / 2.893× **U** | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |

### lr-requested1

tol=0.0001; max_iter=200; pre-import thread environment: `{"BLIS_NUM_THREADS": null, "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": null, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}`.

| Case | Shape | Input | Seed | sklearn ms | Direct auto ms / ratio | Patched auto ms / ratio | Forced GPU ms / ratio | Observed fit backends |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary_small | 5000×32 | float32 | 41 | 1.293 | 2.23 / 0.5798× | 1.493 / 0.8663× | 4.056 / 0.3188× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_medium | 20000×128 | float32 | 41 | 4.026 | 4.745 / 0.8483× | 4.436 / 0.9074× | 5.266 / 0.7645× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_wide | 20000×1024 | float32 | 41 | 16.05 | 10.88 / 1.475× | 12.66 / 1.268× | 10.55 / 1.522× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_large | 60000×1024 | float32 | 41 | 46.77 | 17.22 / 2.716× | 15.61 / 2.996× | 14.52 / 3.221× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| multiclass_weighted | 20000×128 | float32 | 41 | 16.83 | 15.99 / 1.052× | 17.28 / 0.9736× | 23.69 / 0.7102× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_ill_conditioned | 5000×32 | float32 | 41 | 3.395 | 6.88 / 0.4934× | 3.515 / 0.9657× | 7.77 / 0.4369× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_boundary | 20000×512 | float32 | 41 | 9.41 | 12.2 / 0.7714× | 10.73 / 0.8768× | 9.707 / 0.9694× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_tall_crossover | 250000×32 | float32 | 41 | 36.69 | 10.53 / 3.484× | 10.09 / 3.636× | 9.872 / 3.716× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_tall | 500000×32 | float32 | 41 | 80.55 | 15.91 / 5.061× | 15.16 / 5.312× | 8.471 / 9.509× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_tall_wide | 500000×128 | float32 | 41 | 86.02 | 15.83 / 5.435× | 14.35 / 5.997× | 12.71 / 6.767× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_40k_2048 | 40000×2048 | float32 | 41 | 57.58 | 22.43 / 2.567× | 20.7 / 2.782× | 19.52 / 2.949× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_20k_4096 | 20000×4096 | float32 | 41 | 98.8 | 34.62 / 2.854× | 30.65 / 3.223× | 30.27 / 3.264× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_weighted | 20000×1024 | float32 | 41 | 17.3 | 12.65 / 1.367× | 11.83 / 1.462× | 10.28 / 1.682× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_1000_16 | 1000×16 | float32 | 41 | 0.8538 | 1.955 / 0.4368× | 0.9984 / 0.8552× | 3.521 / 0.2425× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_1000_128 | 1000×128 | float32 | 41 | 0.9514 | 3.114 / 0.3056× | 1.093 / 0.8701× | 4.911 / 0.1937× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_1000_1024 | 1000×1024 | float32 | 41 | 2.093 | 4.149 / 0.5045× **U** | 2.278 / 0.9185× | 14.49 / 0.1445× **U** | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_10000_16 | 10000×16 | float32 | 41 | 1.962 | 2.569 / 0.7637× | 2.107 / 0.9312× | 3.88 / 0.5056× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_10000_128 | 10000×128 | float32 | 41 | 2.244 | 3.228 / 0.6951× | 2.389 / 0.9393× | 4.624 / 0.4853× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_10000_1024 | 10000×1024 | float32 | 41 | 8.861 | 14.58 / 0.6077× | 10.18 / 0.8705× | 14.84 / 0.5973× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_100000_16 | 100000×16 | float32 | 41 | 13.15 | 8.652 / 1.52× | 13.26 / 0.9918× | 6.257 / 2.101× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_100000_128 | 100000×128 | float32 | 41 | 19.56 | 17.49 / 1.119× | 21.2 / 0.9228× | 9.564 / 2.046× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_100000_1024 | 100000×1024 | float32 | 41 | 77.59 | 29.52 / 2.628× | 19.12 / 4.057× | 20.29 / 3.823× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_500000_16 | 500000×16 | float32 | 41 | 78.2 | 15.9 / 4.917× | 14.83 / 5.273× | 15.05 / 5.197× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_500000_1024 | 500000×1024 | float32 | 41 | 379.1 | 87.25 / 4.344× | 77.12 / 4.915× | 76.93 / 4.927× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| multiclass_weighted_wide | 20000×1024 | float32 | 41 | 117.6 | 44.54 / 2.641× **U** | 36.6 / 3.214× **U** | 26.73 / 4.401× **U** | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_small | 5000×32 | float64 | 41 | 1.277 | 2.331 / 0.5478× | 1.405 / 0.9088× | 3.52 / 0.3628× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_medium | 20000×128 | float64 | 41 | 5.484 | 4.798 / 1.143× | 5.986 / 0.9162× | 6.064 / 0.9044× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_wide | 20000×1024 | float64 | 41 | 28.23 | 18.38 / 1.536× | 13.44 / 2.1× | 17.51 / 1.612× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_large | 60000×1024 | float64 | 41 | 83.46 | 20.52 / 4.068× | 18.93 / 4.41× | 18.96 / 4.402× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| multiclass_weighted | 20000×128 | float64 | 41 | 25.63 | 16.11 / 1.591× | 26.19 / 0.9786× | 25.43 / 1.008× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_ill_conditioned | 5000×32 | float64 | 41 | 3.335 | 6.794 / 0.4909× | 3.363 / 0.9918× | 7.626 / 0.4374× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_boundary | 20000×512 | float64 | 41 | 15.43 | 12.36 / 1.248× | 16.69 / 0.9239× | 11.26 / 1.37× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| binary_tall_crossover | 250000×32 | float64 | 41 | 37.98 | 11.46 / 3.315× | 11.08 / 3.427× | 7.631 / 4.977× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_tall | 500000×32 | float64 | 41 | 83.85 | 17.36 / 4.831× | 17.36 / 4.83× | 10.93 / 7.675× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| binary_tall_wide | 500000×128 | float64 | 41 | 116.9 | 20.94 / 5.583× | 17.77 / 6.576× | 17.26 / 6.769× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_40k_2048 | 40000×2048 | float64 | 41 | 108.4 | 35.47 / 3.056× | 27.87 / 3.889× | 24.98 / 4.338× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_20k_4096 | 20000×4096 | float64 | 41 | 197.2 | 47.39 / 4.161× | 38.22 / 5.16× | 37.47 / 5.264× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_weighted | 20000×1024 | float64 | 41 | 30 | 15.05 / 1.993× | 14.35 / 2.091× | 14.14 / 2.122× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_1000_16 | 1000×16 | float64 | 41 | 0.8878 | 1.967 / 0.4513× | 0.9233 / 0.9615× | 3.351 / 0.2649× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_1000_128 | 1000×128 | float64 | 41 | 1.042 | 3.199 / 0.3256× | 1.164 / 0.8949× | 5.323 / 0.1957× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_1000_1024 | 1000×1024 | float64 | 41 | 2.117 | 4.085 / 0.5181× **U** | 2.375 / 0.8914× | 6.089 / 0.3476× **U** | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_10000_16 | 10000×16 | float64 | 41 | 1.835 | 2.682 / 0.6843× | 2.038 / 0.9003× | 4.245 / 0.4323× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_10000_128 | 10000×128 | float64 | 41 | 2.284 | 3.282 / 0.6957× | 2.58 / 0.8852× | 4.89 / 0.467× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_10000_1024 | 10000×1024 | float64 | 41 | 15.62 | 14.49 / 1.078× | 16.96 / 0.9212× | 17.03 / 0.9173× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_100000_16 | 100000×16 | float64 | 41 | 13.12 | 9.085 / 1.444× | 13.69 / 0.9585× | 6.879 / 1.907× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_100000_128 | 100000×128 | float64 | 41 | 26.19 | 17.59 / 1.489× | 27.88 / 0.9394× | 11.06 / 2.368× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| grid_100000_1024 | 100000×1024 | float64 | 41 | 140.3 | 33.91 / 4.136× | 30.77 / 4.558× | 26.88 / 5.219× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_500000_16 | 500000×16 | float64 | 41 | 78.22 | 16.79 / 4.658× | 15.47 / 5.056× | 12.51 / 6.255× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_500000_1024 | 500000×1024 | float64 | 41 | 686.7 | 119.4 / 5.751× | 110.7 / 6.206× | 107.7 / 6.374× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| multiclass_weighted_wide | 20000×1024 | float64 | 41 | 197.6 | 48.56 / 4.069× **U** | 34.3 / 5.763× **U** | 28.82 / 6.857× **U** | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |

### lr-strict4096-default

tol=1e-05; max_iter=200; pre-import thread environment: `{"BLIS_NUM_THREADS": null, "MKL_NUM_THREADS": null, "NUMEXPR_NUM_THREADS": null, "OMP_NUM_THREADS": null, "OPENBLAS_NUM_THREADS": null, "VECLIB_MAXIMUM_THREADS": null}`.

| Case | Shape | Input | Seed | sklearn ms | Direct auto ms / ratio | Patched auto ms / ratio | Forced GPU ms / ratio | Observed fit backends |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wide_20k_4096 | 20000×4096 | float32 | 41 | 89.89 | 39.7 / 2.264× | 32.62 / 2.756× | 32.65 / 2.754× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_20k_4096 | 20000×4096 | float64 | 41 | 169 | 58.55 / 2.887× | 47.88 / 3.53× | 47.69 / 3.544× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |

### lr-strict4096-requested1

tol=1e-05; max_iter=200; pre-import thread environment: `{"BLIS_NUM_THREADS": null, "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": null, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}`.

| Case | Shape | Input | Seed | sklearn ms | Direct auto ms / ratio | Patched auto ms / ratio | Forced GPU ms / ratio | Observed fit backends |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wide_20k_4096 | 20000×4096 | float32 | 41 | 142.1 | 35.91 / 3.957× | 33.78 / 4.208× | 35.18 / 4.04× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_20k_4096 | 20000×4096 | float64 | 41 | 287.3 | 59.21 / 4.853× | 51.03 / 5.631× | 52.46 / 5.477× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |

### lr-accuracy-matched-default

tol=1e-06; max_iter=200; pre-import thread environment: `{"BLIS_NUM_THREADS": null, "MKL_NUM_THREADS": null, "NUMEXPR_NUM_THREADS": null, "OMP_NUM_THREADS": null, "OPENBLAS_NUM_THREADS": null, "VECLIB_MAXIMUM_THREADS": null}`.

| Case | Shape | Input | Seed | sklearn ms | Direct auto ms / ratio | Patched auto ms / ratio | Forced GPU ms / ratio | Observed fit backends |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wide_20k_4096 | 20000×4096 | float32 | 41 | 109.1 | 57.91 / 1.884× | 54.33 / 2.009× | 55.19 / 1.977× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_1000_1024 | 1000×1024 | float32 | 41 | 4.946 **U** | 8.741 / 0.5659× **U** | 5.181 / 0.9546× **U** | 13.61 / 0.3635× **U** | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| multiclass_weighted_wide | 20000×1024 | float32 | 41 | 134.3 **U** | 82.85 / 1.621× **U** | 60.74 / 2.211× **U** | 67.71 / 1.983× **U** | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_20k_4096 | 20000×4096 | float64 | 41 | 206.7 | 62.22 / 3.322× | 53.75 / 3.845× | 51.92 / 3.98× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_1000_1024 | 1000×1024 | float64 | 41 | 4.867 | 8.626 / 0.5642× | 5.171 / 0.9411× | 11.99 / 0.4058× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| multiclass_weighted_wide | 20000×1024 | float64 | 41 | 169.6 | 87.12 / 1.947× | 71.33 / 2.378× | 63.3 / 2.679× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |

### lr-accuracy-matched-requested1

tol=1e-06; max_iter=200; pre-import thread environment: `{"BLIS_NUM_THREADS": null, "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": null, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}`.

| Case | Shape | Input | Seed | sklearn ms | Direct auto ms / ratio | Patched auto ms / ratio | Forced GPU ms / ratio | Observed fit backends |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wide_20k_4096 | 20000×4096 | float32 | 41 | 172.9 | 63.68 / 2.715× | 56.65 / 3.052× | 61.84 / 2.796× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_1000_1024 | 1000×1024 | float32 | 41 | 4.86 **U** | 8.584 / 0.5662× **U** | 5.095 / 0.954× **U** | 13.31 / 0.3653× **U** | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| multiclass_weighted_wide | 20000×1024 | float32 | 41 | 216.8 **U** | 78.01 / 2.779× **U** | 59.97 / 3.615× **U** | 60.03 / 3.611× **U** | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| wide_20k_4096 | 20000×4096 | float64 | 41 | 349.3 | 67.91 / 5.144× | 55.18 / 6.331× | 55.75 / 6.267× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |
| grid_1000_1024 | 1000×1024 | float64 | 41 | 4.826 | 8.52 / 0.5664× | 5.181 / 0.9314× | 13.97 / 0.3455× | Direct auto=cpu: 7; Patched auto=sklearn: 7; Forced GPU=mlx: 7 |
| multiclass_weighted_wide | 20000×1024 | float64 | 41 | 296.4 | 82.57 / 3.589× | 59.77 / 4.959× | 59.63 / 4.97× | Direct auto=mlx: 7; Patched auto=mlx: 7; Forced GPU=mlx: 7 |

### LR inference by query count

Each fit fixture contributes one separately qualified cell per operation and query count. Inference uses the final fitted model; fit time is excluded. Fit/reference failures invalidate the corresponding inference comparison even if the output check alone passes.

| Panel | Operation | Input | Queries | Variant | Comparator | Raw median ratio | Raw ratio range | Raw wins/cells | Qualified/cells | Qualified wins/qualified | Qualified median ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr-default | predict | float32 | 1 | Direct auto | sklearn | 0.8802× | 0.7653×–0.9433× | 0/25 | 23/25 | 0/23 | 0.8802× |
| lr-default | predict_proba | float32 | 1 | Direct auto | sklearn | 0.8996× | 0.7988×–0.9868× | 0/25 | 22/25 | 0/22 | 0.8961× |
| lr-default | predict | float32 | 256 | Direct auto | sklearn | 0.8444× | 0.6443×–0.9264× | 0/25 | 23/25 | 0/23 | 0.8474× |
| lr-default | predict_proba | float32 | 256 | Direct auto | sklearn | 0.8036× | 0.6636×–0.9165× | 0/25 | 16/25 | 0/16 | 0.8476× |
| lr-default | predict | float32 | 4096 | Direct auto | sklearn | 0.7003× | 0.5383×–0.9101× | 0/25 | 23/25 | 0/23 | 0.7233× |
| lr-default | predict_proba | float32 | 4096 | Direct auto | sklearn | 0.6678× | 0.5034×–0.8849× | 0/25 | 14/25 | 0/14 | 0.7401× |
| lr-default | predict | float32 | 1 | Patched auto | sklearn | 0.8706× | 0.7668×–0.9952× | 0/25 | 24/25 | 0/24 | 0.8677× |
| lr-default | predict_proba | float32 | 1 | Patched auto | sklearn | 0.8945× | 0.8199×–1.062× | 2/25 | 23/25 | 1/23 | 0.8803× |
| lr-default | predict | float32 | 256 | Patched auto | sklearn | 0.8665× | 0.6597×–0.9818× | 0/25 | 24/25 | 0/24 | 0.8711× |
| lr-default | predict_proba | float32 | 256 | Patched auto | sklearn | 0.8427× | 0.6435×–1.012× | 1/25 | 20/25 | 1/20 | 0.8637× |
| lr-default | predict | float32 | 4096 | Patched auto | sklearn | 0.8962× | 0.5476×–1.004× | 1/25 | 24/25 | 1/24 | 0.9138× |
| lr-default | predict_proba | float32 | 4096 | Patched auto | sklearn | 0.8676× | 0.5291×–1.032× | 3/25 | 18/25 | 3/18 | 0.9295× |
| lr-default | predict | float32 | 1 | Forced GPU | sklearn | 0.8869× | 0.7065×–0.9814× | 0/25 | 23/25 | 0/23 | 0.8941× |
| lr-default | predict_proba | float32 | 1 | Forced GPU | sklearn | 0.8923× | 0.8157×–0.9967× | 0/25 | 22/25 | 0/22 | 0.8932× |
| lr-default | predict | float32 | 256 | Forced GPU | sklearn | 0.8317× | 0.6391×–1.001× | 1/25 | 23/25 | 1/23 | 0.8333× |
| lr-default | predict_proba | float32 | 256 | Forced GPU | sklearn | 0.7961× | 0.6404×–0.8891× | 0/25 | 16/25 | 0/16 | 0.8474× |
| lr-default | predict | float32 | 4096 | Forced GPU | sklearn | 0.7058× | 0.5298×–0.9027× | 0/25 | 23/25 | 0/23 | 0.7151× |
| lr-default | predict_proba | float32 | 4096 | Forced GPU | sklearn | 0.6495× | 0.5165×–0.8791× | 0/25 | 13/25 | 0/13 | 0.7318× |
| lr-default | predict | float64 | 1 | Direct auto | sklearn | 0.8955× | 0.82×–0.9472× | 0/25 | 23/25 | 0/23 | 0.8955× |
| lr-default | predict_proba | float64 | 1 | Direct auto | sklearn | 0.8983× | 0.7806×–0.9618× | 0/25 | 22/25 | 0/22 | 0.8965× |
| lr-default | predict | float64 | 256 | Direct auto | sklearn | 0.9205× | 0.8782×–0.9829× | 0/25 | 23/25 | 0/23 | 0.9205× |
| lr-default | predict_proba | float64 | 256 | Direct auto | sklearn | 0.9126× | 0.81×–0.969× | 0/25 | 16/25 | 0/16 | 0.9005× |
| lr-default | predict | float64 | 4096 | Direct auto | sklearn | 0.9538× | 0.7626×–1.012× | 5/25 | 23/25 | 3/23 | 0.953× |
| lr-default | predict_proba | float64 | 4096 | Direct auto | sklearn | 0.9611× | 0.9044×–1.005× | 3/25 | 14/25 | 1/14 | 0.9345× |
| lr-default | predict | float64 | 1 | Patched auto | sklearn | 0.8995× | 0.8237×–0.9436× | 0/25 | 24/25 | 0/24 | 0.8989× |
| lr-default | predict_proba | float64 | 1 | Patched auto | sklearn | 0.8915× | 0.8167×–0.9877× | 0/25 | 23/25 | 0/23 | 0.8867× |
| lr-default | predict | float64 | 256 | Patched auto | sklearn | 0.9162× | 0.8597×–0.9875× | 0/25 | 24/25 | 0/24 | 0.9135× |
| lr-default | predict_proba | float64 | 256 | Patched auto | sklearn | 0.9038× | 0.8615×–0.9654× | 0/25 | 20/25 | 0/20 | 0.8967× |
| lr-default | predict | float64 | 4096 | Patched auto | sklearn | 0.9615× | 0.885×–1.016× | 2/25 | 24/25 | 2/24 | 0.9544× |
| lr-default | predict_proba | float64 | 4096 | Patched auto | sklearn | 0.9627× | 0.8504×–1.038× | 6/25 | 18/25 | 1/18 | 0.9462× |
| lr-default | predict | float64 | 1 | Forced GPU | sklearn | 0.9121× | 0.823×–0.9518× | 0/25 | 23/25 | 0/23 | 0.9099× |
| lr-default | predict_proba | float64 | 1 | Forced GPU | sklearn | 0.9013× | 0.8604×–0.9581× | 0/25 | 22/25 | 0/22 | 0.8976× |
| lr-default | predict | float64 | 256 | Forced GPU | sklearn | 0.9293× | 0.8322×–0.9956× | 0/25 | 23/25 | 0/23 | 0.9293× |
| lr-default | predict_proba | float64 | 256 | Forced GPU | sklearn | 0.9227× | 0.8598×–0.9895× | 0/25 | 16/25 | 0/16 | 0.9123× |
| lr-default | predict | float64 | 4096 | Forced GPU | sklearn | 0.9479× | 0.9049×–1.053× | 2/25 | 23/25 | 1/23 | 0.9475× |
| lr-default | predict_proba | float64 | 4096 | Forced GPU | sklearn | 0.9638× | 0.8671×–1.019× | 3/25 | 13/25 | 1/13 | 0.9425× |
| lr-requested1 | predict | float32 | 1 | Direct auto | sklearn | 0.8858× | 0.8336×–0.9366× | 0/25 | 23/25 | 0/23 | 0.8842× |
| lr-requested1 | predict_proba | float32 | 1 | Direct auto | sklearn | 0.8727× | 0.8021×–0.9762× | 0/25 | 22/25 | 0/22 | 0.8714× |
| lr-requested1 | predict | float32 | 256 | Direct auto | sklearn | 0.8581× | 0.6294×–0.9168× | 0/25 | 23/25 | 0/23 | 0.8594× |
| lr-requested1 | predict_proba | float32 | 256 | Direct auto | sklearn | 0.8244× | 0.6192×–0.8928× | 0/25 | 16/25 | 0/16 | 0.8433× |
| lr-requested1 | predict | float32 | 4096 | Direct auto | sklearn | 0.6892× | 0.5141×–0.8875× | 0/25 | 23/25 | 0/23 | 0.7034× |
| lr-requested1 | predict_proba | float32 | 4096 | Direct auto | sklearn | 0.6563× | 0.514×–0.8227× | 0/25 | 14/25 | 0/14 | 0.7855× |
| lr-requested1 | predict | float32 | 1 | Patched auto | sklearn | 0.8919× | 0.6832×–0.9488× | 0/25 | 24/25 | 0/24 | 0.894× |
| lr-requested1 | predict_proba | float32 | 1 | Patched auto | sklearn | 0.8802× | 0.8179×–0.9235× | 0/25 | 23/25 | 0/23 | 0.8802× |
| lr-requested1 | predict | float32 | 256 | Patched auto | sklearn | 0.8971× | 0.6187×–0.9669× | 0/25 | 24/25 | 0/24 | 0.8983× |
| lr-requested1 | predict_proba | float32 | 256 | Patched auto | sklearn | 0.8655× | 0.6209×–1.001× | 1/25 | 20/25 | 1/20 | 0.8862× |
| lr-requested1 | predict | float32 | 4096 | Patched auto | sklearn | 0.9141× | 0.506×–1.012× | 1/25 | 24/25 | 1/24 | 0.9217× |
| lr-requested1 | predict_proba | float32 | 4096 | Patched auto | sklearn | 0.9152× | 0.5076×–1.003× | 1/25 | 18/25 | 1/18 | 0.9434× |
| lr-requested1 | predict | float32 | 1 | Forced GPU | sklearn | 0.8885× | 0.7678×–0.9512× | 0/25 | 23/25 | 0/23 | 0.8909× |
| lr-requested1 | predict_proba | float32 | 1 | Forced GPU | sklearn | 0.8795× | 0.8077×–0.9615× | 0/25 | 22/25 | 0/22 | 0.8796× |
| lr-requested1 | predict | float32 | 256 | Forced GPU | sklearn | 0.8196× | 0.6259×–0.941× | 0/25 | 23/25 | 0/23 | 0.8585× |
| lr-requested1 | predict_proba | float32 | 256 | Forced GPU | sklearn | 0.8156× | 0.6244×–1.012× | 1/25 | 16/25 | 1/16 | 0.8639× |
| lr-requested1 | predict | float32 | 4096 | Forced GPU | sklearn | 0.7065× | 0.5001×–0.9057× | 0/25 | 23/25 | 0/23 | 0.7093× |
| lr-requested1 | predict_proba | float32 | 4096 | Forced GPU | sklearn | 0.6754× | 0.5054×–0.8268× | 0/25 | 13/25 | 0/13 | 0.7816× |
| lr-requested1 | predict | float64 | 1 | Direct auto | sklearn | 0.8965× | 0.8393×–1.007× | 1/25 | 23/25 | 1/23 | 0.8965× |
| lr-requested1 | predict_proba | float64 | 1 | Direct auto | sklearn | 0.9025× | 0.814×–0.9669× | 0/25 | 22/25 | 0/22 | 0.8976× |
| lr-requested1 | predict | float64 | 256 | Direct auto | sklearn | 0.9153× | 0.8197×–0.9609× | 0/25 | 23/25 | 0/23 | 0.9132× |
| lr-requested1 | predict_proba | float64 | 256 | Direct auto | sklearn | 0.9129× | 0.7039×–0.9872× | 0/25 | 16/25 | 0/16 | 0.905× |
| lr-requested1 | predict | float64 | 4096 | Direct auto | sklearn | 0.9692× | 0.8704×–1.028× | 6/25 | 23/25 | 5/23 | 0.9603× |
| lr-requested1 | predict_proba | float64 | 4096 | Direct auto | sklearn | 0.9793× | 0.8665×–1.041× | 4/25 | 14/25 | 1/14 | 0.949× |
| lr-requested1 | predict | float64 | 1 | Patched auto | sklearn | 0.889× | 0.8542×–1.001× | 1/25 | 24/25 | 1/24 | 0.8876× |
| lr-requested1 | predict_proba | float64 | 1 | Patched auto | sklearn | 0.8828× | 0.8588×–0.9386× | 0/25 | 23/25 | 0/23 | 0.882× |
| lr-requested1 | predict | float64 | 256 | Patched auto | sklearn | 0.9222× | 0.8365×–1.003× | 1/25 | 24/25 | 1/24 | 0.9219× |
| lr-requested1 | predict_proba | float64 | 256 | Patched auto | sklearn | 0.9319× | 0.7325×–0.9951× | 0/25 | 20/25 | 0/20 | 0.9288× |
| lr-requested1 | predict | float64 | 4096 | Patched auto | sklearn | 0.9733× | 0.8528×–1.03× | 6/25 | 24/25 | 6/24 | 0.9705× |
| lr-requested1 | predict_proba | float64 | 4096 | Patched auto | sklearn | 0.9741× | 0.8781×–1.054× | 4/25 | 18/25 | 3/18 | 0.9539× |
| lr-requested1 | predict | float64 | 1 | Forced GPU | sklearn | 0.8975× | 0.812×–1.002× | 1/25 | 23/25 | 1/23 | 0.8959× |
| lr-requested1 | predict_proba | float64 | 1 | Forced GPU | sklearn | 0.8781× | 0.8323×–0.9653× | 0/25 | 22/25 | 0/22 | 0.8772× |
| lr-requested1 | predict | float64 | 256 | Forced GPU | sklearn | 0.9201× | 0.8457×–1.006× | 1/25 | 23/25 | 1/23 | 0.9193× |
| lr-requested1 | predict_proba | float64 | 256 | Forced GPU | sklearn | 0.9194× | 0.7345×–0.9965× | 0/25 | 16/25 | 0/16 | 0.9073× |
| lr-requested1 | predict | float64 | 4096 | Forced GPU | sklearn | 0.9666× | 0.871×–1.02× | 5/25 | 23/25 | 5/23 | 0.9646× |
| lr-requested1 | predict_proba | float64 | 4096 | Forced GPU | sklearn | 0.9688× | 0.7863×–1.02× | 5/25 | 13/25 | 1/13 | 0.9492× |
| lr-strict4096-default | predict | float32 | 1 | Direct auto | sklearn | 0.9051× | 0.9051× | 0/1 | 1/1 | 0/1 | 0.9051× |
| lr-strict4096-default | predict_proba | float32 | 1 | Direct auto | sklearn | 0.9127× | 0.9127× | 0/1 | 1/1 | 0/1 | 0.9127× |
| lr-strict4096-default | predict | float32 | 256 | Direct auto | sklearn | 0.6159× | 0.6159× | 0/1 | 1/1 | 0/1 | 0.6159× |
| lr-strict4096-default | predict_proba | float32 | 256 | Direct auto | sklearn | 0.6122× | 0.6122× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-default | predict | float32 | 4096 | Direct auto | sklearn | 0.5511× | 0.5511× | 0/1 | 1/1 | 0/1 | 0.5511× |
| lr-strict4096-default | predict_proba | float32 | 4096 | Direct auto | sklearn | 0.5407× | 0.5407× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-default | predict | float32 | 1 | Patched auto | sklearn | 0.9129× | 0.9129× | 0/1 | 1/1 | 0/1 | 0.9129× |
| lr-strict4096-default | predict_proba | float32 | 1 | Patched auto | sklearn | 0.8962× | 0.8962× | 0/1 | 1/1 | 0/1 | 0.8962× |
| lr-strict4096-default | predict | float32 | 256 | Patched auto | sklearn | 0.6276× | 0.6276× | 0/1 | 1/1 | 0/1 | 0.6276× |
| lr-strict4096-default | predict_proba | float32 | 256 | Patched auto | sklearn | 0.6018× | 0.6018× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-default | predict | float32 | 4096 | Patched auto | sklearn | 0.5415× | 0.5415× | 0/1 | 1/1 | 0/1 | 0.5415× |
| lr-strict4096-default | predict_proba | float32 | 4096 | Patched auto | sklearn | 0.5358× | 0.5358× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-default | predict | float32 | 1 | Forced GPU | sklearn | 0.8966× | 0.8966× | 0/1 | 1/1 | 0/1 | 0.8966× |
| lr-strict4096-default | predict_proba | float32 | 1 | Forced GPU | sklearn | 0.9113× | 0.9113× | 0/1 | 1/1 | 0/1 | 0.9113× |
| lr-strict4096-default | predict | float32 | 256 | Forced GPU | sklearn | 0.6243× | 0.6243× | 0/1 | 1/1 | 0/1 | 0.6243× |
| lr-strict4096-default | predict_proba | float32 | 256 | Forced GPU | sklearn | 0.5989× | 0.5989× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-default | predict | float32 | 4096 | Forced GPU | sklearn | 0.5462× | 0.5462× | 0/1 | 1/1 | 0/1 | 0.5462× |
| lr-strict4096-default | predict_proba | float32 | 4096 | Forced GPU | sklearn | 0.5349× | 0.5349× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-default | predict | float64 | 1 | Direct auto | sklearn | 0.9175× | 0.9175× | 0/1 | 1/1 | 0/1 | 0.9175× |
| lr-strict4096-default | predict_proba | float64 | 1 | Direct auto | sklearn | 0.9263× | 0.9263× | 0/1 | 1/1 | 0/1 | 0.9263× |
| lr-strict4096-default | predict | float64 | 256 | Direct auto | sklearn | 0.9718× | 0.9718× | 0/1 | 1/1 | 0/1 | 0.9718× |
| lr-strict4096-default | predict_proba | float64 | 256 | Direct auto | sklearn | 0.9698× | 0.9698× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-default | predict | float64 | 4096 | Direct auto | sklearn | 0.9962× | 0.9962× | 0/1 | 1/1 | 0/1 | 0.9962× |
| lr-strict4096-default | predict_proba | float64 | 4096 | Direct auto | sklearn | 0.9931× | 0.9931× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-default | predict | float64 | 1 | Patched auto | sklearn | 0.7964× | 0.7964× | 0/1 | 1/1 | 0/1 | 0.7964× |
| lr-strict4096-default | predict_proba | float64 | 1 | Patched auto | sklearn | 0.9254× | 0.9254× | 0/1 | 1/1 | 0/1 | 0.9254× |
| lr-strict4096-default | predict | float64 | 256 | Patched auto | sklearn | 0.9824× | 0.9824× | 0/1 | 1/1 | 0/1 | 0.9824× |
| lr-strict4096-default | predict_proba | float64 | 256 | Patched auto | sklearn | 0.9908× | 0.9908× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-default | predict | float64 | 4096 | Patched auto | sklearn | 0.995× | 0.995× | 0/1 | 1/1 | 0/1 | 0.995× |
| lr-strict4096-default | predict_proba | float64 | 4096 | Patched auto | sklearn | 0.9916× | 0.9916× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-default | predict | float64 | 1 | Forced GPU | sklearn | 0.8213× | 0.8213× | 0/1 | 1/1 | 0/1 | 0.8213× |
| lr-strict4096-default | predict_proba | float64 | 1 | Forced GPU | sklearn | 0.9066× | 0.9066× | 0/1 | 1/1 | 0/1 | 0.9066× |
| lr-strict4096-default | predict | float64 | 256 | Forced GPU | sklearn | 0.9642× | 0.9642× | 0/1 | 1/1 | 0/1 | 0.9642× |
| lr-strict4096-default | predict_proba | float64 | 256 | Forced GPU | sklearn | 0.9913× | 0.9913× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-default | predict | float64 | 4096 | Forced GPU | sklearn | 0.9971× | 0.9971× | 0/1 | 1/1 | 0/1 | 0.9971× |
| lr-strict4096-default | predict_proba | float64 | 4096 | Forced GPU | sklearn | 0.9962× | 0.9962× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-requested1 | predict | float32 | 1 | Direct auto | sklearn | 0.9469× | 0.9469× | 0/1 | 1/1 | 0/1 | 0.9469× |
| lr-strict4096-requested1 | predict_proba | float32 | 1 | Direct auto | sklearn | 0.9311× | 0.9311× | 0/1 | 1/1 | 0/1 | 0.9311× |
| lr-strict4096-requested1 | predict | float32 | 256 | Direct auto | sklearn | 0.6363× | 0.6363× | 0/1 | 1/1 | 0/1 | 0.6363× |
| lr-strict4096-requested1 | predict_proba | float32 | 256 | Direct auto | sklearn | 0.6204× | 0.6204× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-requested1 | predict | float32 | 4096 | Direct auto | sklearn | 0.5317× | 0.5317× | 0/1 | 1/1 | 0/1 | 0.5317× |
| lr-strict4096-requested1 | predict_proba | float32 | 4096 | Direct auto | sklearn | 0.5305× | 0.5305× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-requested1 | predict | float32 | 1 | Patched auto | sklearn | 0.9439× | 0.9439× | 0/1 | 1/1 | 0/1 | 0.9439× |
| lr-strict4096-requested1 | predict_proba | float32 | 1 | Patched auto | sklearn | 0.9257× | 0.9257× | 0/1 | 1/1 | 0/1 | 0.9257× |
| lr-strict4096-requested1 | predict | float32 | 256 | Patched auto | sklearn | 0.6389× | 0.6389× | 0/1 | 1/1 | 0/1 | 0.6389× |
| lr-strict4096-requested1 | predict_proba | float32 | 256 | Patched auto | sklearn | 0.6113× | 0.6113× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-requested1 | predict | float32 | 4096 | Patched auto | sklearn | 0.5246× | 0.5246× | 0/1 | 1/1 | 0/1 | 0.5246× |
| lr-strict4096-requested1 | predict_proba | float32 | 4096 | Patched auto | sklearn | 0.5269× | 0.5269× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-requested1 | predict | float32 | 1 | Forced GPU | sklearn | 0.931× | 0.931× | 0/1 | 1/1 | 0/1 | 0.931× |
| lr-strict4096-requested1 | predict_proba | float32 | 1 | Forced GPU | sklearn | 0.8762× | 0.8762× | 0/1 | 1/1 | 0/1 | 0.8762× |
| lr-strict4096-requested1 | predict | float32 | 256 | Forced GPU | sklearn | 0.6202× | 0.6202× | 0/1 | 1/1 | 0/1 | 0.6202× |
| lr-strict4096-requested1 | predict_proba | float32 | 256 | Forced GPU | sklearn | 0.6171× | 0.6171× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-requested1 | predict | float32 | 4096 | Forced GPU | sklearn | 0.529× | 0.529× | 0/1 | 1/1 | 0/1 | 0.529× |
| lr-strict4096-requested1 | predict_proba | float32 | 4096 | Forced GPU | sklearn | 0.5329× | 0.5329× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-requested1 | predict | float64 | 1 | Direct auto | sklearn | 0.9429× | 0.9429× | 0/1 | 1/1 | 0/1 | 0.9429× |
| lr-strict4096-requested1 | predict_proba | float64 | 1 | Direct auto | sklearn | 0.8669× | 0.8669× | 0/1 | 1/1 | 0/1 | 0.8669× |
| lr-strict4096-requested1 | predict | float64 | 256 | Direct auto | sklearn | 0.9304× | 0.9304× | 0/1 | 1/1 | 0/1 | 0.9304× |
| lr-strict4096-requested1 | predict_proba | float64 | 256 | Direct auto | sklearn | 0.9584× | 0.9584× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-requested1 | predict | float64 | 4096 | Direct auto | sklearn | 0.9883× | 0.9883× | 0/1 | 1/1 | 0/1 | 0.9883× |
| lr-strict4096-requested1 | predict_proba | float64 | 4096 | Direct auto | sklearn | 0.9931× | 0.9931× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-requested1 | predict | float64 | 1 | Patched auto | sklearn | 0.9335× | 0.9335× | 0/1 | 1/1 | 0/1 | 0.9335× |
| lr-strict4096-requested1 | predict_proba | float64 | 1 | Patched auto | sklearn | 0.9056× | 0.9056× | 0/1 | 1/1 | 0/1 | 0.9056× |
| lr-strict4096-requested1 | predict | float64 | 256 | Patched auto | sklearn | 0.9516× | 0.9516× | 0/1 | 1/1 | 0/1 | 0.9516× |
| lr-strict4096-requested1 | predict_proba | float64 | 256 | Patched auto | sklearn | 0.9622× | 0.9622× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-requested1 | predict | float64 | 4096 | Patched auto | sklearn | 0.9975× | 0.9975× | 0/1 | 1/1 | 0/1 | 0.9975× |
| lr-strict4096-requested1 | predict_proba | float64 | 4096 | Patched auto | sklearn | 0.9911× | 0.9911× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-requested1 | predict | float64 | 1 | Forced GPU | sklearn | 0.9221× | 0.9221× | 0/1 | 1/1 | 0/1 | 0.9221× |
| lr-strict4096-requested1 | predict_proba | float64 | 1 | Forced GPU | sklearn | 0.9209× | 0.9209× | 0/1 | 1/1 | 0/1 | 0.9209× |
| lr-strict4096-requested1 | predict | float64 | 256 | Forced GPU | sklearn | 0.9595× | 0.9595× | 0/1 | 1/1 | 0/1 | 0.9595× |
| lr-strict4096-requested1 | predict_proba | float64 | 256 | Forced GPU | sklearn | 0.9706× | 0.9706× | 0/1 | 0/1 | 0/0 | missing |
| lr-strict4096-requested1 | predict | float64 | 4096 | Forced GPU | sklearn | 0.9983× | 0.9983× | 0/1 | 1/1 | 0/1 | 0.9983× |
| lr-strict4096-requested1 | predict_proba | float64 | 4096 | Forced GPU | sklearn | 0.9972× | 0.9972× | 0/1 | 0/1 | 0/0 | missing |
| lr-accuracy-matched-default | predict | float32 | 1 | Direct auto | sklearn | 0.8367× | 0.8241×–0.8908× | 0/3 | 1/3 | 0/1 | 0.8908× |
| lr-accuracy-matched-default | predict_proba | float32 | 1 | Direct auto | sklearn | 0.8789× | 0.8438×–0.8925× | 0/3 | 1/3 | 0/1 | 0.8925× |
| lr-accuracy-matched-default | predict | float32 | 256 | Direct auto | sklearn | 0.714× | 0.6175×–0.7484× | 0/3 | 1/3 | 0/1 | 0.6175× |
| lr-accuracy-matched-default | predict_proba | float32 | 256 | Direct auto | sklearn | 0.7331× | 0.6154×–0.7447× | 0/3 | 1/3 | 0/1 | 0.6154× |
| lr-accuracy-matched-default | predict | float32 | 4096 | Direct auto | sklearn | 0.6075× | 0.5447×–0.623× | 0/3 | 1/3 | 0/1 | 0.5447× |
| lr-accuracy-matched-default | predict_proba | float32 | 4096 | Direct auto | sklearn | 0.593× | 0.5544×–0.6017× | 0/3 | 1/3 | 0/1 | 0.5544× |
| lr-accuracy-matched-default | predict | float32 | 1 | Patched auto | sklearn | 0.8991× | 0.895×–0.9104× | 0/3 | 1/3 | 0/1 | 0.8991× |
| lr-accuracy-matched-default | predict_proba | float32 | 1 | Patched auto | sklearn | 0.8835× | 0.8506×–0.9117× | 0/3 | 1/3 | 0/1 | 0.9117× |
| lr-accuracy-matched-default | predict | float32 | 256 | Patched auto | sklearn | 0.7203× | 0.5999×–0.9118× | 0/3 | 1/3 | 0/1 | 0.5999× |
| lr-accuracy-matched-default | predict_proba | float32 | 256 | Patched auto | sklearn | 0.7383× | 0.6006×–0.9343× | 0/3 | 1/3 | 0/1 | 0.6006× |
| lr-accuracy-matched-default | predict | float32 | 4096 | Patched auto | sklearn | 0.6116× | 0.5396×–1.069× | 1/3 | 1/3 | 0/1 | 0.5396× |
| lr-accuracy-matched-default | predict_proba | float32 | 4096 | Patched auto | sklearn | 0.5936× | 0.543×–0.9877× | 0/3 | 1/3 | 0/1 | 0.543× |
| lr-accuracy-matched-default | predict | float32 | 1 | Forced GPU | sklearn | 0.8665× | 0.82×–0.9331× | 0/3 | 1/3 | 0/1 | 0.8665× |
| lr-accuracy-matched-default | predict_proba | float32 | 1 | Forced GPU | sklearn | 0.8856× | 0.8666×–0.9038× | 0/3 | 1/3 | 0/1 | 0.8856× |
| lr-accuracy-matched-default | predict | float32 | 256 | Forced GPU | sklearn | 0.728× | 0.616×–0.7338× | 0/3 | 1/3 | 0/1 | 0.616× |
| lr-accuracy-matched-default | predict_proba | float32 | 256 | Forced GPU | sklearn | 0.7107× | 0.6041×–0.7426× | 0/3 | 1/3 | 0/1 | 0.6041× |
| lr-accuracy-matched-default | predict | float32 | 4096 | Forced GPU | sklearn | 0.6088× | 0.5399×–0.6345× | 0/3 | 1/3 | 0/1 | 0.5399× |
| lr-accuracy-matched-default | predict_proba | float32 | 4096 | Forced GPU | sklearn | 0.5886× | 0.5382×–0.6001× | 0/3 | 1/3 | 0/1 | 0.5382× |
| lr-accuracy-matched-default | predict | float64 | 1 | Direct auto | sklearn | 0.9159× | 0.9008×–0.9435× | 0/3 | 3/3 | 0/3 | 0.9159× |
| lr-accuracy-matched-default | predict_proba | float64 | 1 | Direct auto | sklearn | 0.9289× | 0.8809×–0.9484× | 0/3 | 3/3 | 0/3 | 0.9289× |
| lr-accuracy-matched-default | predict | float64 | 256 | Direct auto | sklearn | 0.9485× | 0.9374×–0.9488× | 0/3 | 3/3 | 0/3 | 0.9485× |
| lr-accuracy-matched-default | predict_proba | float64 | 256 | Direct auto | sklearn | 0.9435× | 0.9408×–0.9873× | 0/3 | 2/3 | 0/2 | 0.9654× |
| lr-accuracy-matched-default | predict | float64 | 4096 | Direct auto | sklearn | 0.9941× | 0.9758×–1.001× | 1/3 | 3/3 | 1/3 | 0.9941× |
| lr-accuracy-matched-default | predict_proba | float64 | 4096 | Direct auto | sklearn | 0.9981× | 0.9791×–1× | 1/3 | 2/3 | 1/2 | 0.9991× |
| lr-accuracy-matched-default | predict | float64 | 1 | Patched auto | sklearn | 0.9037× | 0.9013×–0.9222× | 0/3 | 3/3 | 0/3 | 0.9037× |
| lr-accuracy-matched-default | predict_proba | float64 | 1 | Patched auto | sklearn | 0.9319× | 0.921×–0.9399× | 0/3 | 3/3 | 0/3 | 0.9319× |
| lr-accuracy-matched-default | predict | float64 | 256 | Patched auto | sklearn | 0.9273× | 0.9225×–0.9759× | 0/3 | 3/3 | 0/3 | 0.9273× |
| lr-accuracy-matched-default | predict_proba | float64 | 256 | Patched auto | sklearn | 0.9702× | 0.9025×–0.992× | 0/3 | 3/3 | 0/3 | 0.9702× |
| lr-accuracy-matched-default | predict | float64 | 4096 | Patched auto | sklearn | 1.005× | 0.9891×–1.008× | 2/3 | 3/3 | 2/3 | 1.005× |
| lr-accuracy-matched-default | predict_proba | float64 | 4096 | Patched auto | sklearn | 0.9788× | 0.9649×–1× | 1/3 | 3/3 | 1/3 | 0.9788× |
| lr-accuracy-matched-default | predict | float64 | 1 | Forced GPU | sklearn | 0.9178× | 0.885×–0.951× | 0/3 | 3/3 | 0/3 | 0.9178× |
| lr-accuracy-matched-default | predict_proba | float64 | 1 | Forced GPU | sklearn | 0.9248× | 0.8909×–0.9282× | 0/3 | 3/3 | 0/3 | 0.9248× |
| lr-accuracy-matched-default | predict | float64 | 256 | Forced GPU | sklearn | 0.9442× | 0.9169×–0.9613× | 0/3 | 3/3 | 0/3 | 0.9442× |
| lr-accuracy-matched-default | predict_proba | float64 | 256 | Forced GPU | sklearn | 0.9513× | 0.9362×–0.9694× | 0/3 | 2/3 | 0/2 | 0.9603× |
| lr-accuracy-matched-default | predict | float64 | 4096 | Forced GPU | sklearn | 0.9967× | 0.9764×–1.005× | 1/3 | 3/3 | 1/3 | 0.9967× |
| lr-accuracy-matched-default | predict_proba | float64 | 4096 | Forced GPU | sklearn | 0.989× | 0.9815×–0.9998× | 0/3 | 2/3 | 0/2 | 0.9906× |
| lr-accuracy-matched-requested1 | predict | float32 | 1 | Direct auto | sklearn | 0.8674× | 0.7797×–0.8679× | 0/3 | 1/3 | 0/1 | 0.8679× |
| lr-accuracy-matched-requested1 | predict_proba | float32 | 1 | Direct auto | sklearn | 0.8685× | 0.863×–0.9068× | 0/3 | 1/3 | 0/1 | 0.9068× |
| lr-accuracy-matched-requested1 | predict | float32 | 256 | Direct auto | sklearn | 0.7523× | 0.6222×–0.7674× | 0/3 | 1/3 | 0/1 | 0.6222× |
| lr-accuracy-matched-requested1 | predict_proba | float32 | 256 | Direct auto | sklearn | 0.7371× | 0.6269×–0.7567× | 0/3 | 1/3 | 0/1 | 0.6269× |
| lr-accuracy-matched-requested1 | predict | float32 | 4096 | Direct auto | sklearn | 0.5338× | 0.5276×–0.5911× | 0/3 | 1/3 | 0/1 | 0.5276× |
| lr-accuracy-matched-requested1 | predict_proba | float32 | 4096 | Direct auto | sklearn | 0.5234× | 0.5091×–0.6055× | 0/3 | 1/3 | 0/1 | 0.5234× |
| lr-accuracy-matched-requested1 | predict | float32 | 1 | Patched auto | sklearn | 0.8984× | 0.8628×–0.9425× | 0/3 | 1/3 | 0/1 | 0.9425× |
| lr-accuracy-matched-requested1 | predict_proba | float32 | 1 | Patched auto | sklearn | 0.8873× | 0.8507×–0.9058× | 0/3 | 1/3 | 0/1 | 0.8507× |
| lr-accuracy-matched-requested1 | predict | float32 | 256 | Patched auto | sklearn | 0.7465× | 0.6104×–0.9498× | 0/3 | 1/3 | 0/1 | 0.6104× |
| lr-accuracy-matched-requested1 | predict_proba | float32 | 256 | Patched auto | sklearn | 0.7646× | 0.614×–0.9241× | 0/3 | 1/3 | 0/1 | 0.614× |
| lr-accuracy-matched-requested1 | predict | float32 | 4096 | Patched auto | sklearn | 0.5974× | 0.5297×–0.9725× | 0/3 | 1/3 | 0/1 | 0.5297× |
| lr-accuracy-matched-requested1 | predict_proba | float32 | 4096 | Patched auto | sklearn | 0.6143× | 0.5217×–0.9694× | 0/3 | 1/3 | 0/1 | 0.5217× |
| lr-accuracy-matched-requested1 | predict | float32 | 1 | Forced GPU | sklearn | 0.8686× | 0.8142×–0.9037× | 0/3 | 1/3 | 0/1 | 0.9037× |
| lr-accuracy-matched-requested1 | predict_proba | float32 | 1 | Forced GPU | sklearn | 0.8803× | 0.8778×–0.8847× | 0/3 | 1/3 | 0/1 | 0.8778× |
| lr-accuracy-matched-requested1 | predict | float32 | 256 | Forced GPU | sklearn | 0.7482× | 0.6209×–0.7714× | 0/3 | 1/3 | 0/1 | 0.6209× |
| lr-accuracy-matched-requested1 | predict_proba | float32 | 256 | Forced GPU | sklearn | 0.7244× | 0.6112×–0.7682× | 0/3 | 1/3 | 0/1 | 0.6112× |
| lr-accuracy-matched-requested1 | predict | float32 | 4096 | Forced GPU | sklearn | 0.5353× | 0.5305×–0.6026× | 0/3 | 1/3 | 0/1 | 0.5305× |
| lr-accuracy-matched-requested1 | predict_proba | float32 | 4096 | Forced GPU | sklearn | 0.5213× | 0.5197×–0.6136× | 0/3 | 1/3 | 0/1 | 0.5197× |
| lr-accuracy-matched-requested1 | predict | float64 | 1 | Direct auto | sklearn | 0.9029× | 0.8767×–0.9587× | 0/3 | 3/3 | 0/3 | 0.9029× |
| lr-accuracy-matched-requested1 | predict_proba | float64 | 1 | Direct auto | sklearn | 0.9215× | 0.8764×–0.9865× | 0/3 | 3/3 | 0/3 | 0.9215× |
| lr-accuracy-matched-requested1 | predict | float64 | 256 | Direct auto | sklearn | 0.9353× | 0.8984×–0.9545× | 0/3 | 3/3 | 0/3 | 0.9353× |
| lr-accuracy-matched-requested1 | predict_proba | float64 | 256 | Direct auto | sklearn | 0.9612× | 0.914×–0.9881× | 0/3 | 2/3 | 0/2 | 0.9747× |
| lr-accuracy-matched-requested1 | predict | float64 | 4096 | Direct auto | sklearn | 1.002× | 0.9935×–1.004× | 2/3 | 3/3 | 2/3 | 1.002× |
| lr-accuracy-matched-requested1 | predict_proba | float64 | 4096 | Direct auto | sklearn | 1.005× | 0.9904×–1.006× | 2/3 | 2/3 | 1/2 | 0.9984× |
| lr-accuracy-matched-requested1 | predict | float64 | 1 | Patched auto | sklearn | 0.9298× | 0.8538×–0.9413× | 0/3 | 3/3 | 0/3 | 0.9298× |
| lr-accuracy-matched-requested1 | predict_proba | float64 | 1 | Patched auto | sklearn | 0.9307× | 0.9037×–0.9402× | 0/3 | 3/3 | 0/3 | 0.9307× |
| lr-accuracy-matched-requested1 | predict | float64 | 256 | Patched auto | sklearn | 0.9323× | 0.9185×–0.9557× | 0/3 | 3/3 | 0/3 | 0.9323× |
| lr-accuracy-matched-requested1 | predict_proba | float64 | 256 | Patched auto | sklearn | 0.9841× | 0.9306×–0.9884× | 0/3 | 3/3 | 0/3 | 0.9841× |
| lr-accuracy-matched-requested1 | predict | float64 | 4096 | Patched auto | sklearn | 0.9969× | 0.9628×–1.004× | 1/3 | 3/3 | 1/3 | 0.9969× |
| lr-accuracy-matched-requested1 | predict_proba | float64 | 4096 | Patched auto | sklearn | 1.004× | 0.9963×–1.004× | 2/3 | 3/3 | 2/3 | 1.004× |
| lr-accuracy-matched-requested1 | predict | float64 | 1 | Forced GPU | sklearn | 0.9285× | 0.8938×–0.934× | 0/3 | 3/3 | 0/3 | 0.9285× |
| lr-accuracy-matched-requested1 | predict_proba | float64 | 1 | Forced GPU | sklearn | 0.9372× | 0.8896×–0.9854× | 0/3 | 3/3 | 0/3 | 0.9372× |
| lr-accuracy-matched-requested1 | predict | float64 | 256 | Forced GPU | sklearn | 0.9623× | 0.9081×–0.9695× | 0/3 | 3/3 | 0/3 | 0.9623× |
| lr-accuracy-matched-requested1 | predict_proba | float64 | 256 | Forced GPU | sklearn | 0.9527× | 0.9299×–0.9773× | 0/3 | 2/3 | 0/2 | 0.965× |
| lr-accuracy-matched-requested1 | predict | float64 | 4096 | Forced GPU | sklearn | 0.9965× | 0.9489×–1.002× | 1/3 | 3/3 | 1/3 | 0.9965× |
| lr-accuracy-matched-requested1 | predict_proba | float64 | 4096 | Forced GPU | sklearn | 1.006× | 1.005×–1.015× | 3/3 | 2/3 | 2/2 | 1.006× |

### Difficult-fixture accuracy diagnostics

This table follows **direct automatic** only, with primary and tighter-tolerance measurements kept in separate rows. Values are maxima across retained fits/calls, not a selected faster observation. Negative objective gaps favor the current model under the one-sided objective gate. A certified float64 stock gradient above the panel tolerance disqualifies the comparison even when current checks pass. Probability deltas use all 4,096 held-out queries.

| Panel | Case | Input | sklearn max gradient | Current max gradient | Signed relative objective gap | Maximum probability delta | Fit | 4,096-query probabilities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr-default | wide_20k_4096 | float32 | 9.862e-05 | 5.725e-06 | -0.000198 | 0.05488 | qualified | **U** |
| lr-default | grid_1000_1024 | float32 | 7.93e-05 | 7.148e-05 | 0.000303 | 0.1416 | **U** | **U** |
| lr-default | multiclass_weighted_wide | float32 | 4.741e-05 | 9.259e-05 | 1.189e-05 | 0.01112 | **U** | **U** |
| lr-default | wide_20k_4096 | float64 | 9.863e-05 | 5.725e-06 | -0.000198 | 0.05488 | qualified | **U** |
| lr-default | grid_1000_1024 | float64 | 7.93e-05 | 7.148e-05 | 0.000303 | 0.1416 | **U** | **U** |
| lr-default | multiclass_weighted_wide | float64 | 4.734e-05 | 9.259e-05 | 1.189e-05 | 0.01112 | **U** | **U** |
| lr-requested1 | wide_20k_4096 | float32 | 9.862e-05 | 5.725e-06 | -0.000198 | 0.05488 | qualified | **U** |
| lr-requested1 | grid_1000_1024 | float32 | 7.93e-05 | 7.148e-05 | 0.000303 | 0.1416 | **U** | **U** |
| lr-requested1 | multiclass_weighted_wide | float32 | 4.741e-05 | 9.259e-05 | 1.189e-05 | 0.01112 | **U** | **U** |
| lr-requested1 | wide_20k_4096 | float64 | 9.863e-05 | 5.725e-06 | -0.000198 | 0.05488 | qualified | **U** |
| lr-requested1 | grid_1000_1024 | float64 | 7.93e-05 | 7.148e-05 | 0.000303 | 0.1416 | **U** | **U** |
| lr-requested1 | multiclass_weighted_wide | float64 | 4.734e-05 | 9.259e-05 | 1.189e-05 | 0.01112 | **U** | **U** |
| lr-strict4096-default | wide_20k_4096 | float32 | 5.953e-06 | 1.824e-06 | -1.204e-06 | 0.003891 | qualified | **U** |
| lr-strict4096-default | wide_20k_4096 | float64 | 5.962e-06 | 1.824e-06 | -1.204e-06 | 0.003891 | qualified | **U** |
| lr-strict4096-requested1 | wide_20k_4096 | float32 | 5.953e-06 | 1.824e-06 | -1.204e-06 | 0.003891 | qualified | **U** |
| lr-strict4096-requested1 | wide_20k_4096 | float64 | 5.962e-06 | 1.824e-06 | -1.204e-06 | 0.003891 | qualified | **U** |
| lr-accuracy-matched-default | wide_20k_4096 | float32 | 9.938e-07 | 9.523e-07 | -2.76e-08 | 0.0006891 | qualified | qualified |
| lr-accuracy-matched-default | grid_1000_1024 | float32 | 1.8e-06 | 9.186e-07 | -2.11e-08 | 0.004104 | **U** | **U** |
| lr-accuracy-matched-default | multiclass_weighted_wide | float32 | 3.321e-06 | 8.331e-07 | -1.946e-08 | 0.0004059 | **U** | **U** |
| lr-accuracy-matched-default | wide_20k_4096 | float64 | 9.969e-07 | 9.522e-07 | -2.759e-08 | 0.0006885 | qualified | qualified |
| lr-accuracy-matched-default | grid_1000_1024 | float64 | 6.221e-07 | 9.185e-07 | 2.536e-07 | 0.003112 | qualified | **U** |
| lr-accuracy-matched-default | multiclass_weighted_wide | float64 | 9.414e-07 | 8.331e-07 | -5.75e-10 | 4.349e-05 | qualified | qualified |
| lr-accuracy-matched-requested1 | wide_20k_4096 | float32 | 9.938e-07 | 9.523e-07 | -2.76e-08 | 0.0006891 | qualified | qualified |
| lr-accuracy-matched-requested1 | grid_1000_1024 | float32 | 1.8e-06 | 9.186e-07 | -2.11e-08 | 0.004104 | **U** | **U** |
| lr-accuracy-matched-requested1 | multiclass_weighted_wide | float32 | 3.321e-06 | 8.331e-07 | -1.946e-08 | 0.0004059 | **U** | **U** |
| lr-accuracy-matched-requested1 | wide_20k_4096 | float64 | 9.969e-07 | 9.522e-07 | -2.759e-08 | 0.0006885 | qualified | qualified |
| lr-accuracy-matched-requested1 | grid_1000_1024 | float64 | 6.221e-07 | 9.185e-07 | 2.536e-07 | 0.003112 | qualified | **U** |
| lr-accuracy-matched-requested1 | multiclass_weighted_wide | float64 | 9.414e-07 | 8.331e-07 | -5.75e-10 | 4.349e-05 | qualified | qualified |

## SVC kernel/cache panel

Eight cases × seeds 41/42, seven retained fresh fits each. Matching CPU/GPU routes are forced per case, with pre-import CPU thread controls set to one. These are direct estimator comparisons. Each retained fit has separately timed first and repeated predictions on 257 held-out queries. Quality checks include convergence, prediction agreement and continuous OVO margin agreement. The original aggregate quality gate, including the historical native control, is retained in the evidence; the comparisons here are qualified from current and sklearn checks and provenance only.

Each metric cell is **sklearn ms / current ms / ratio**; U marks an unqualified raw ratio.

| Case | Seed | Train shape | Cache MB | Observed fit backend | Fit | First prediction | Repeat prediction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_narrow | 41 | 2000×20 | 200.0 | cpu: 7 | 16.41 / 9.286 / 1.767× | 4.906 / 0.3913 / 12.54× | 4.713 / 0.3309 / 14.24× |
| rbf_wide | 41 | 4000×256 | 200.0 | mlx: 7 | 396 / 35.55 / 11.14× | 42.53 / 1.274 / 33.38× | 42.69 / 1.051 / 40.61× |
| linear | 41 | 1200×20 | 200.0 | cpu: 7 | 39.52 / 23.62 / 1.673× | 0.8168 / 0.2226 / 3.669× | 0.7629 / 0.1605 / 4.753× |
| linear_medium | 41 | 2500×32 | 200.0 | cpu: 7 | 268.6 / 114.8 / 2.34× | 2.439 / 0.3606 / 6.765× | 2.373 / 0.24 / 9.884× |
| poly | 41 | 2000×32 | 200.0 | cpu: 7 | 26.05 / 10.21 / 2.551× | 2.361 / 0.404 / 5.843× | 2.386 / 0.303 / 7.875× |
| sigmoid | 41 | 2000×32 | 200.0 | cpu: 7 | 25.38 / 8.221 / 3.088× | 2.944 / 0.3759 / 7.83× | 2.963 / 0.3464 / 8.552× |
| rbf_multiclass | 41 | 1600×32 | 200.0 | mlx: 7 | 26.74 / 14.22 / 1.88× | 9.075 / 0.5678 / 15.98× | 9.273 / 0.5249 / 17.67× |
| rbf_bounded_cache | 41 | 1600×32 | 0.05 | mlx: 7 | 37.63 / 135.1 / 0.2786× | 4.193 / 0.4004 / 10.47× | 4.249 / 0.3398 / 12.5× |
| rbf_narrow | 42 | 2000×20 | 200.0 | cpu: 7 | 17.25 / 10.66 / 1.618× | 4.911 / 0.384 / 12.79× | 4.869 / 0.3259 / 14.94× |
| rbf_wide | 42 | 4000×256 | 200.0 | mlx: 7 | 456.9 / 38.31 / 11.92× | 47.8 / 1.431 / 33.41× | 47.71 / 1.251 / 38.12× |
| linear | 42 | 1200×20 | 200.0 | cpu: 7 | 76.76 / 48.29 / 1.589× | 1.135 / 0.2362 / 4.805× | 1.08 / 0.191 / 5.654× |
| linear_medium | 42 | 2500×32 | 200.0 | cpu: 7 | 370.1 / 150.8 / 2.454× | 2.838 / 0.4032 / 7.037× | 2.758 / 0.2944 / 9.369× |
| poly | 42 | 2000×32 | 200.0 | cpu: 7 | 23.77 / 9.544 / 2.49× | 2.374 / 0.3764 / 6.306× | 2.303 / 0.2955 / 7.794× |
| sigmoid | 42 | 2000×32 | 200.0 | cpu: 7 | 26.4 / 8.889 / 2.97× | 3.155 / 0.4866 / 6.483× | 3.115 / 0.3779 / 8.245× |
| rbf_multiclass | 42 | 1600×32 | 200.0 | mlx: 7 | 24.58 / 13.64 / 1.803× | 8.945 / 0.5487 / 16.3× | 8.838 / 0.4807 / 18.39× |
| rbf_bounded_cache | 42 | 1600×32 | 0.05 | mlx: 7 | 37.87 / 128.4 / 0.2949× | 4.371 / 0.3877 / 11.27× | 4.377 / 0.3105 / 14.1× |

### SVC first calls, separate from retained medians

Each cell below is sklearn ms / current ms for the first model in its worker. Imports and input generation are excluded. These observations are not the dedicated fresh-process startup measure.

| Case | Seed | First fit ms | First prediction ms | Repeat prediction ms |
| --- | --- | --- | --- | --- |
| rbf_narrow | 41 | 18.22 / 12.4 | 4.817 / 0.5989 | 4.762 / 0.3346 |
| rbf_wide | 41 | 398.5 / 57.1 | 42.63 / 1.58 | 41.12 / 1.157 |
| linear | 41 | 41.88 / 24.98 | 0.8942 / 0.3225 | 0.7499 / 0.1622 |
| linear_medium | 41 | 269.4 / 118.8 | 2.453 / 0.5704 | 2.274 / 0.2441 |
| poly | 41 | 27.93 / 13.83 | 2.53 / 0.6062 | 2.475 / 0.314 |
| sigmoid | 41 | 26.88 / 10.77 | 3.039 / 0.5545 | 2.996 / 0.3587 |
| rbf_multiclass | 41 | 28.15 / 30.59 | 9.408 / 0.7986 | 9.323 / 0.5203 |
| rbf_bounded_cache | 41 | 37.83 / 172.2 | 4.344 / 0.923 | 4.638 / 0.4343 |
| rbf_narrow | 42 | 19.27 / 13.84 | 5.1 / 0.64 | 4.866 / 0.4364 |
| rbf_wide | 42 | 459.3 / 60.35 | 47.35 / 1.545 | 47.59 / 1.241 |
| linear | 42 | 77.66 / 48.83 | 1.146 / 0.3638 | 1.135 / 0.2154 |
| linear_medium | 42 | 376.4 / 153 | 2.915 / 0.605 | 2.798 / 0.3182 |
| poly | 42 | 25.66 / 13.35 | 2.375 / 0.6535 | 2.31 / 0.3303 |
| sigmoid | 42 | 27.98 / 11.36 | 3.3 / 0.5725 | 3.145 / 0.3802 |
| rbf_multiclass | 42 | 25.86 / 28.94 | 9.038 / 0.7748 | 8.939 / 0.5235 |
| rbf_bounded_cache | 42 | 38.94 / 168 | 4.33 / 0.9156 | 4.291 / 0.3239 |

## Full KNN/SVC shape grid

84 operation cases: 18 distinct KNN fits, 54 KNN query configurations, eight SVC fits and four SVC predictions. Public inputs are float64, with original and canonical-float32 neighbor geometry checked separately. **KNN correctness samples up to nine evenly spaced query rows**, rather than claiming an exhaustive audit of all 4,096 queries. SVC queries are independent held-out Gaussian rows. Each case runs in a new worker; seven retained calls follow one warmup.

Fit and query routes are shown separately; fit rows' query route belongs to their untimed accuracy check. Cells show current ms / raw ratio. KNN columns compare against both brute and auto sklearn; SVC uses stock SVC. Raw timing remains visible when any comparison's gate fails, including its stock reference. Qualification is evaluated per variant/comparator pair across all retained audits; an unrelated variant's failure does not erase a passing pair.

| Case / operation | sklearn ms | Direct auto ms / ratio | Patched auto ms / ratio | Observed fit/query routes |
| --- | --- | --- | --- | --- |
| knn/fit/500x8 | sklearn_brute=0.1179; sklearn_auto=0.2011 | brute 0.07992 / 1.475×; auto 0.07992 / 2.517× | brute 0.07558 / 1.56×; auto 0.07558 / 2.661× | Direct auto: fit=cpu: 7; query=cpu: 7; Patched auto: fit=cpu: 7; query=cpu: 7 |
| knn/kneighbors/500x8/q1 | sklearn_brute=0.8434; sklearn_auto=0.2255 | brute 0.1023 / 8.245×; auto 0.1023 / 2.204× | brute 0.09571 / 8.812×; auto 0.09571 / 2.356× | Direct auto: fit=cpu: 7; query=cpu: 7; Patched auto: fit=cpu: 7; query=cpu: 7 |
| knn/kneighbors/500x8/q256 | sklearn_brute=1.072; sklearn_auto=1.025 | brute 0.6095 / 1.759×; auto 0.6095 / 1.682× | brute 0.6177 / 1.736×; auto 0.6177 / 1.66× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/kneighbors/500x8/q4096 | sklearn_brute=1.902; sklearn_auto=12.15 | brute 1.541 / 1.234×; auto 1.541 / 7.885× | brute 1.835 / 1.036×; auto 1.835 / 6.62× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/fit/500x64 | sklearn_brute=0.1415; sklearn_auto=0.1287 | brute 0.08138 / 1.739×; auto 0.08138 / 1.582× | brute 0.08404 / 1.684×; auto 0.08404 / 1.531× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/kneighbors/500x64/q1 | sklearn_brute=0.6841; sklearn_auto=0.6814 | brute 0.7683 / 0.8904×; auto 0.7683 / 0.8869× | brute 0.784 / 0.8727×; auto 0.784 / 0.8692× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/kneighbors/500x64/q256 | sklearn_brute=0.9563; sklearn_auto=1.048 | brute 0.8973 / 1.066×; auto 0.8973 / 1.168× | brute 0.9275 / 1.031×; auto 0.9275 / 1.13× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/kneighbors/500x64/q4096 | sklearn_brute=1.724; sklearn_auto=1.75 | brute 2.907 / 0.5929×; auto 2.907 / 0.6021× | brute 2.918 / 0.5906×; auto 2.918 / 0.5998× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/fit/500x768 | sklearn_brute=0.1826; sklearn_auto=0.1809 | brute 0.1814 / 1.007×; auto 0.1814 / 0.9975× | brute 0.1833 / 0.9964×; auto 0.1833 / 0.987× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/500x768/q1 | sklearn_brute=0.7021; sklearn_auto=0.7246 | brute 0.7723 / 0.9091×; auto 0.7723 / 0.9383× | brute 0.8206 / 0.8555×; auto 0.8206 / 0.883× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/500x768/q256 | sklearn_brute=1.423; sklearn_auto=1.528 | brute 1.22 / 1.167×; auto 1.22 / 1.253× | brute 1.171 / 1.215×; auto 1.171 / 1.305× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/500x768/q4096 | sklearn_brute=6.321; sklearn_auto=6.428 | brute 5.21 / 1.213×; auto 5.21 / 1.234× | brute 5.248 / 1.204×; auto 5.248 / 1.225× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/fit/2000x8 | sklearn_brute=0.1301; sklearn_auto=0.4244 | brute 0.09075 / 1.433×; auto 0.09075 / 4.677× | brute 0.0915 / 1.422×; auto 0.0915 / 4.638× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/kneighbors/2000x8/q1 | sklearn_brute=0.81; sklearn_auto=0.2431 | brute 0.1199 / 6.757×; auto 0.1199 / 2.028× | brute 0.1235 / 6.556×; auto 0.1235 / 1.968× | Direct auto: fit=cpu: 7; query=cpu: 7; Patched auto: fit=cpu: 7; query=cpu: 7 |
| knn/kneighbors/2000x8/q256 | sklearn_brute=1.515; sklearn_auto=2.539 | brute 0.7029 / 2.156×; auto 0.7029 / 3.613× | brute 0.6665 / 2.273×; auto 0.6665 / 3.81× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/kneighbors/2000x8/q4096 | sklearn_brute=2.509; sklearn_auto=34.87 | brute 3.207 / 0.7821×; auto 3.207 / 10.87× | brute 3.059 / 0.8201×; auto 3.059 / 11.4× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/fit/2000x64 | sklearn_brute=0.1472; sklearn_auto=0.1455 | brute 0.1064 / 1.383×; auto 0.1064 / 1.367× | brute 0.1163 / 1.265×; auto 0.1163 / 1.251× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/kneighbors/2000x64/q1 | sklearn_brute=0.7625; sklearn_auto=0.7921 | brute 0.7874 / 0.9684×; auto 0.7874 / 1.006× | brute 0.8456 / 0.9017×; auto 0.8456 / 0.9367× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/kneighbors/2000x64/q256 | sklearn_brute=1.578; sklearn_auto=1.593 | brute 1.189 / 1.327×; auto 1.189 / 1.34× | brute 1.186 / 1.33×; auto 1.186 / 1.343× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/kneighbors/2000x64/q4096 | sklearn_brute=3.649; sklearn_auto=3.37 | brute 3.147 / 1.16×; auto 3.147 / 1.071× | brute 3.203 / 1.139×; auto 3.203 / 1.052× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/fit/2000x768 | sklearn_brute=0.329; sklearn_auto=0.327 | brute 0.4572 / 0.7196×; auto 0.4572 / 0.7153× | brute 0.4604 / 0.7146×; auto 0.4604 / 0.7103× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/2000x768/q1 | sklearn_brute=1.009; sklearn_auto=1.037 | brute 0.8733 / 1.155×; auto 0.8733 / 1.187× | brute 0.8821 / 1.144×; auto 0.8821 / 1.176× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/2000x768/q256 | sklearn_brute=3.174; sklearn_auto=3.369 | brute 1.713 / 1.853×; auto 1.713 / 1.967× | brute 1.697 / 1.87×; auto 1.697 / 1.985× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/2000x768/q4096 | sklearn_brute=21.21; sklearn_auto=20.21 | brute 8.853 / 2.396×; auto 8.853 / 2.283× | brute 7.801 / 2.719×; auto 7.801 / 2.591× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/fit/8000x8 | sklearn_brute=0.1307; sklearn_auto=1.501 | brute 0.1122 / 1.164×; auto 0.1122 / 13.38× | brute 0.104 / 1.256×; auto 0.104 / 14.43× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/kneighbors/8000x8/q1 | sklearn_brute=1.065; sklearn_auto=0.2455 | brute 0.1249 / 8.526×; auto 0.1249 / 1.966× | brute 0.1474 / 7.224×; auto 0.1474 / 1.666× | Direct auto: fit=cpu: 7; query=cpu: 7; Patched auto: fit=cpu: 7; query=cpu: 7 |
| knn/kneighbors/8000x8/q256 | sklearn_brute=1.812; sklearn_auto=5.45 | brute 1.056 / 1.716×; auto 1.056 / 5.16× | brute 0.981 / 1.847×; auto 0.981 / 5.556× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/kneighbors/8000x8/q4096 | sklearn_brute=15.16; sklearn_auto=79.98 | brute 6.797 / 2.23×; auto 6.797 / 11.77× | brute 7.012 / 2.162×; auto 7.012 / 11.41× | Direct auto: fit=cpu: 7; query=mlx: 7; Patched auto: fit=cpu: 7; query=mlx: 7 |
| knn/fit/8000x64 | sklearn_brute=0.1999; sklearn_auto=0.2003 | brute 0.2126 / 0.9402×; auto 0.2126 / 0.9424× | brute 0.2244 / 0.8906×; auto 0.2244 / 0.8927× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/8000x64/q1 | sklearn_brute=0.8874; sklearn_auto=0.8508 | brute 0.8179 / 1.085×; auto 0.8179 / 1.04× | brute 0.8407 / 1.056×; auto 0.8407 / 1.012× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/8000x64/q256 | sklearn_brute=2.011; sklearn_auto=1.963 | brute 2.638 / 0.7622×; auto 2.638 / 0.7441× | brute 2.688 / 0.7482×; auto 2.688 / 0.7304× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/8000x64/q4096 | sklearn_brute=23.53; sklearn_auto=23.53 | brute 10.44 / 2.254×; auto 10.44 / 2.254× | brute 10.68 / 2.204×; auto 10.68 / 2.204× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/fit/8000x768 | sklearn_brute=0.9284; sklearn_auto=0.9361 | brute 1.335 / 0.6952×; auto 1.335 / 0.7009× | brute 1.299 / 0.7144×; auto 1.299 / 0.7203× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/8000x768/q1 | sklearn_brute=1.616; sklearn_auto=1.545 | brute 1.063 / 1.52×; auto 1.063 / 1.454× | brute 1.011 / 1.598×; auto 1.011 / 1.528× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/8000x768/q256 | sklearn_brute=6.749; sklearn_auto=6.258 | brute 4.439 / 1.52×; auto 4.439 / 1.41× | brute 4.36 / 1.548×; auto 4.36 / 1.435× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/8000x768/q4096 | sklearn_brute=98.31; sklearn_auto=97.49 | brute 23.68 / 4.151×; auto 23.68 / 4.117× | brute 23.55 / 4.174×; auto 23.55 / 4.139× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/fit/32000x8 | sklearn_brute=0.1694; sklearn_auto=6.902 | brute 0.1626 / 1.042×; auto 0.1626 / 42.44× | brute 0.1518 / 1.116×; auto 0.1518 / 45.49× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/32000x8/q1 | sklearn_brute=1.05; sklearn_auto=0.27 | brute 0.6365 / 1.649×; auto 0.6365 / 0.4241× | brute 0.6501 / 1.615×; auto 0.6501 / 0.4153× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/32000x8/q256 | sklearn_brute=2.703; sklearn_auto=9.313 | brute 1.855 / 1.457×; auto 1.855 / 5.021× | brute 1.811 / 1.492×; auto 1.811 / 5.142× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/32000x8/q4096 | sklearn_brute=34.35; sklearn_auto=145.9 | brute 13.24 / 2.594×; auto 13.24 / 11.02× | brute 9.726 / 3.532×; auto 9.726 / 15× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/fit/32000x64 | sklearn_brute=0.3796; sklearn_auto=0.3865 | brute 0.5656 / 0.6711×; auto 0.5656 / 0.6832× | brute 0.5748 / 0.6604×; auto 0.5748 / 0.6724× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/32000x64/q1 | sklearn_brute=1.123; sklearn_auto=1.095 | brute 0.9958 / 1.128×; auto 0.9958 / 1.1× | brute 0.979 / 1.147×; auto 0.979 / 1.119× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/32000x64/q256 | sklearn_brute=4.004; sklearn_auto=4.205 | brute 4.197 / 0.9541×; auto 4.197 / 1.002× | brute 4.087 / 0.9796×; auto 4.087 / 1.029× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/32000x64/q4096 | sklearn_brute=59.16; sklearn_auto=60.13 | brute 42.42 / 1.395×; auto 42.42 / 1.418× | brute 42.39 / 1.396×; auto 42.39 / 1.419× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/fit/32000x768 | sklearn_brute=3.251; sklearn_auto=3.33 | brute 3.796 / 0.8565×; auto 3.796 / 0.8772× | brute 3.663 / 0.8874×; auto 3.663 / 0.9089× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/32000x768/q1 | sklearn_brute=2.531; sklearn_auto=2.578 | brute 1.39 / 1.821×; auto 1.39 / 1.855× | brute 1.28 / 1.978×; auto 1.28 / 2.015× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/32000x768/q256 | sklearn_brute=20.27; sklearn_auto=22.65 | brute 8.248 / 2.458×; auto 8.248 / 2.746× | brute 6.682 / 3.033×; auto 6.682 / 3.389× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/32000x768/q4096 | sklearn_brute=297.7; sklearn_auto=292.5 | brute 60.12 / 4.951×; auto 60.12 / 4.864× | brute 60.68 / 4.906×; auto 60.68 / 4.82× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/fit/128000x8 | sklearn_brute=0.2637; sklearn_auto=33.66 | brute 0.3408 / 0.7736×; auto 0.3408 / 98.76× | brute 0.3121 / 0.8447×; auto 0.3121 / 107.8× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/128000x8/q1 | sklearn_brute=0.9192; sklearn_auto=0.3134 | brute 0.8444 / 1.089×; auto 0.8444 / 0.3711× | brute 0.8756 / 1.05×; auto 0.8756 / 0.3579× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/128000x8/q256 | sklearn_brute=6.328; sklearn_auto=15.09 | brute 2.427 / 2.607×; auto 2.427 / 6.216× | brute 2.409 / 2.627×; auto 2.409 / 6.263× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/128000x8/q4096 | sklearn_brute=93.35; sklearn_auto=255.8 | brute 23.23 / 4.018×; auto 23.23 / 11.01× | brute 15.26 / 6.118×; auto 15.26 / 16.76× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/fit/128000x64 | sklearn_brute=1.199; sklearn_auto=1.189 | brute 1.658 / 0.7235×; auto 1.658 / 0.7174× | brute 1.7 / 0.7055×; auto 1.7 / 0.6996× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/128000x64/q1 | sklearn_brute=1.57; sklearn_auto=1.455 | brute 1.299 / 1.209×; auto 1.299 / 1.12× | brute 1.292 / 1.215×; auto 1.292 / 1.126× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/128000x64/q256 | sklearn_brute=11.34; sklearn_auto=12.81 | brute 13.75 / 0.825×; auto 13.75 / 0.9318× | brute 13.83 / 0.8198×; auto 13.83 / 0.9259× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/128000x64/q4096 | sklearn_brute=211.1; sklearn_auto=210.2 | brute 208.8 / 1.011×; auto 208.8 / 1.007× | brute 208.8 / 1.011×; auto 208.8 / 1.007× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/fit/128000x768 | sklearn_brute=18.49; sklearn_auto=17.66 | brute 13.32 / 1.388×; auto 13.32 / 1.325× | brute 13.15 / 1.406×; auto 13.15 / 1.343× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/128000x768/q1 | sklearn_brute=8.132; sklearn_auto=8.255 | brute 2.144 / 3.794×; auto 2.144 / 3.851× | brute 2.289 / 3.553×; auto 2.289 / 3.607× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/128000x768/q256 | sklearn_brute=78.3; sklearn_auto=73.94 | brute 26.02 / 3.009×; auto 26.02 / 2.842× | brute 25.04 / 3.127×; auto 25.04 / 2.953× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/128000x768/q4096 | sklearn_brute=1309; sklearn_auto=1320 | brute 298.5 / 4.386×; auto 298.5 / 4.422× | brute 281.7 / 4.648×; auto 281.7 / 4.686× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/fit/512000x8 | sklearn_brute=1.292; sklearn_auto=194.6 | brute 1.205 / 1.072×; auto 1.205 / 161.5× | brute 1.083 / 1.193×; auto 1.083 / 179.7× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/512000x8/q1 | sklearn_brute=2.11; sklearn_auto=0.3925 | brute 1.607 / 1.313×; auto 1.607 / 0.2443× | brute 1.341 / 1.573×; auto 1.341 / 0.2926× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/512000x8/q256 | sklearn_brute=21.32; sklearn_auto=34.74 | brute 7.707 / 2.767×; auto 7.707 / 4.508× | brute 6.955 / 3.065×; auto 6.955 / 4.995× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/512000x8/q4096 | sklearn_brute=328.4; sklearn_auto=534.8 | brute 254.5 / 1.29×; auto 254.5 / 2.102× | brute 253.4 / 1.296×; auto 253.4 / 2.11× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/fit/512000x64 | sklearn_brute=4.16; sklearn_auto=4.162 | brute 4.751 / 0.8757×; auto 4.751 / 0.8761× | brute 4.575 / 0.9094×; auto 4.575 / 0.9099× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/512000x64/q1 | sklearn_brute=3.468; sklearn_auto=3.363 | brute 2.345 / 1.479×; auto 2.345 / 1.434× | brute 2.19 / 1.584×; auto 2.19 / 1.536× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/512000x64/q256 | sklearn_brute=48.84; sklearn_auto=44.76 | brute 62.86 / 0.777×; auto 62.86 / 0.712× | brute 62.89 / 0.7767×; auto 62.89 / 0.7117× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/512000x64/q4096 | sklearn_brute=739; sklearn_auto=760.9 | brute 1092 / 0.677×; auto 1092 / 0.697× | brute 1077 / 0.6865×; auto 1077 / 0.7068× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/fit/512000x768 | sklearn_brute=71.6; sklearn_auto=66.2 | brute 53.37 / 1.342×; auto 53.37 / 1.24× | brute 48.16 / 1.487×; auto 48.16 / 1.374× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/512000x768/q1 | sklearn_brute=27.44; sklearn_auto=26.44 | brute 4.784 / 5.735×; auto 4.784 / 5.527× | brute 4.707 / 5.83×; auto 4.707 / 5.618× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/512000x768/q256 | sklearn_brute=299.7; sklearn_auto=306.6 | brute 85.79 / 3.493×; auto 85.79 / 3.574× | brute 86.61 / 3.46×; auto 86.61 / 3.54× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| knn/kneighbors/512000x768/q4096 | sklearn_brute=4676; sklearn_auto=4693 | brute 1831 / 2.554×; auto 1831 / 2.564× | brute 1782 / 2.624×; auto 1782 / 2.634× | Direct auto: fit=mlx: 7; query=mlx: 7; Patched auto: fit=mlx: 7; query=mlx: 7 |
| svc/fit/500x16 | sklearn=2.1 | sklearn 1.594 / 1.318× | sklearn 2.274 / 0.9234× | Direct auto: fit=cpu: 7; query=None: 7; Patched auto: fit=sklearn: 7; query=None: 7 |
| svc/fit/500x128 | sklearn=7.215 | sklearn 2.229 / 3.237× | sklearn 7.435 / 0.9705× | Direct auto: fit=cpu: 7; query=None: 7; Patched auto: fit=sklearn: 7; query=None: 7 |
| svc/fit/2000x16 | sklearn=23.21 | sklearn 10.87 / 2.135× | sklearn 23.52 / 0.9866× | Direct auto: fit=cpu: 7; query=None: 7; Patched auto: fit=sklearn: 7; query=None: 7 |
| svc/fit/2000x128 | sklearn=88.14 | sklearn 13.56 / 6.502× | sklearn 88.29 / 0.9983× | Direct auto: fit=cpu: 7; query=None: 7; Patched auto: fit=sklearn: 7; query=None: 7 |
| svc/fit/8000x16 | sklearn=322.3 | sklearn 165.6 / 1.946× | sklearn 323.9 / 0.9951× | Direct auto: fit=cpu: 7; query=None: 7; Patched auto: fit=sklearn: 7; query=None: 7 |
| svc/fit/8000x128 | sklearn=1113 | sklearn 253.5 / 4.39× | sklearn 253.8 / 4.386× | Direct auto: fit=mlx: 7; query=None: 7; Patched auto: fit=mlx: 7; query=None: 7 |
| svc/fit/32000x16 | sklearn=4550 | sklearn 2203 / 2.066× | sklearn 4548 / 1.001× | Direct auto: fit=cpu: 7; query=None: 7; Patched auto: fit=sklearn: 7; query=None: 7 |
| svc/fit/32000x128 | sklearn=4.11e+04 | sklearn 2592 / 15.86× | sklearn 2553 / 16.1× | Direct auto: fit=mlx: 7; query=None: 7; Patched auto: fit=mlx: 7; query=None: 7 |
| svc/predict/2000x16 | sklearn=24.83 | sklearn 1.288 / 19.27× | sklearn 24.92 / 0.9962× | Direct auto: fit=cpu: 7; query=None: 7; Patched auto: fit=sklearn: 7; query=None: 7 |
| svc/predict/2000x128 | sklearn=89.29 | sklearn 1.881 / 47.46× | sklearn 89.42 / 0.9985× | Direct auto: fit=cpu: 7; query=None: 7; Patched auto: fit=sklearn: 7; query=None: 7 |
| svc/predict/8000x16 | sklearn=83.5 | sklearn 3.355 / 24.89× | sklearn 83.99 / 0.9941× | Direct auto: fit=cpu: 7; query=None: 7; Patched auto: fit=sklearn: 7; query=None: 7 |
| svc/predict/8000x128 | sklearn=272.1 | sklearn 2.823 / 96.4× | sklearn 2.85 / 95.47× | Direct auto: fit=mlx: 7; query=None: 7; Patched auto: fit=mlx: 7; query=None: 7 |

## Unqualified comparisons and causes

All failed cells remain in the tables and machine-readable summary. Repeated identical causes are grouped below; query counts and operations identify every affected comparison.

| Panel | Case | Input / seed | Variant / comparator | Operations / queries | Causes |
| --- | --- | --- | --- | --- | --- |
| lr-default | binary_medium | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_wide | float32 / 41 | Direct auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_wide | float32 / 41 | Patched auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_wide | float32 / 41 | Forced GPU / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_large | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_large | float32 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_large | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_boundary | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_boundary | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | wide_40k_2048 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | wide_40k_2048 | float32 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | wide_40k_2048 | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | wide_20k_4096 | float32 / 41 | Direct auto / sklearn | predict_proba / 1, 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | wide_20k_4096 | float32 / 41 | Patched auto / sklearn | predict_proba / 1, 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | wide_20k_4096 | float32 / 41 | Forced GPU / sklearn | predict_proba / 1, 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_1000_128 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_1000_128 | float32 / 41 | Forced GPU / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | fit /  | objective gap exceeds tolerance |
| lr-default | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | predict, predict_proba / 1 | fit prerequisite: objective gap exceeds tolerance |
| lr-default | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | predict / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995 |
| lr-default | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995; probabilities close failed |
| lr-default | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | fit /  | objective gap exceeds tolerance |
| lr-default | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | predict, predict_proba / 1 | fit prerequisite: objective gap exceeds tolerance |
| lr-default | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | predict / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995 |
| lr-default | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995; probabilities close failed |
| lr-default | grid_10000_1024 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_10000_1024 | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_100000_1024 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_100000_1024 | float32 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_100000_1024 | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_500000_1024 | float32 / 41 | Direct auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_500000_1024 | float32 / 41 | Patched auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_500000_1024 | float32 / 41 | Forced GPU / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | multiclass_weighted_wide | float32 / 41 | Direct auto / sklearn | fit /  | objective gap exceeds tolerance |
| lr-default | multiclass_weighted_wide | float32 / 41 | Direct auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: objective gap exceeds tolerance |
| lr-default | multiclass_weighted_wide | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; probabilities close failed |
| lr-default | multiclass_weighted_wide | float32 / 41 | Patched auto / sklearn | fit /  | objective gap exceeds tolerance |
| lr-default | multiclass_weighted_wide | float32 / 41 | Patched auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: objective gap exceeds tolerance |
| lr-default | multiclass_weighted_wide | float32 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; probabilities close failed |
| lr-default | multiclass_weighted_wide | float32 / 41 | Forced GPU / sklearn | fit /  | objective gap exceeds tolerance |
| lr-default | multiclass_weighted_wide | float32 / 41 | Forced GPU / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: objective gap exceeds tolerance |
| lr-default | multiclass_weighted_wide | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; probabilities close failed |
| lr-default | binary_medium | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_wide | float64 / 41 | Direct auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_wide | float64 / 41 | Patched auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_wide | float64 / 41 | Forced GPU / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_large | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_large | float64 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_large | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_boundary | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | binary_boundary | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | wide_40k_2048 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | wide_40k_2048 | float64 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | wide_40k_2048 | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | wide_20k_4096 | float64 / 41 | Direct auto / sklearn | predict_proba / 1, 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | wide_20k_4096 | float64 / 41 | Patched auto / sklearn | predict_proba / 1, 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | wide_20k_4096 | float64 / 41 | Forced GPU / sklearn | predict_proba / 1, 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_1000_128 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_1000_128 | float64 / 41 | Forced GPU / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_1000_1024 | float64 / 41 | Direct auto / sklearn | fit /  | objective gap exceeds tolerance |
| lr-default | grid_1000_1024 | float64 / 41 | Direct auto / sklearn | predict, predict_proba / 1 | fit prerequisite: objective gap exceeds tolerance |
| lr-default | grid_1000_1024 | float64 / 41 | Direct auto / sklearn | predict / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995 |
| lr-default | grid_1000_1024 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995; probabilities close failed |
| lr-default | grid_1000_1024 | float64 / 41 | Forced GPU / sklearn | fit /  | objective gap exceeds tolerance |
| lr-default | grid_1000_1024 | float64 / 41 | Forced GPU / sklearn | predict, predict_proba / 1 | fit prerequisite: objective gap exceeds tolerance |
| lr-default | grid_1000_1024 | float64 / 41 | Forced GPU / sklearn | predict / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995 |
| lr-default | grid_1000_1024 | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995; probabilities close failed |
| lr-default | grid_10000_1024 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_10000_1024 | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_100000_1024 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_100000_1024 | float64 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_100000_1024 | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_500000_1024 | float64 / 41 | Direct auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_500000_1024 | float64 / 41 | Patched auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | grid_500000_1024 | float64 / 41 | Forced GPU / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-default | multiclass_weighted_wide | float64 / 41 | Direct auto / sklearn | fit /  | objective gap exceeds tolerance |
| lr-default | multiclass_weighted_wide | float64 / 41 | Direct auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: objective gap exceeds tolerance |
| lr-default | multiclass_weighted_wide | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; probabilities close failed |
| lr-default | multiclass_weighted_wide | float64 / 41 | Patched auto / sklearn | fit /  | objective gap exceeds tolerance |
| lr-default | multiclass_weighted_wide | float64 / 41 | Patched auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: objective gap exceeds tolerance |
| lr-default | multiclass_weighted_wide | float64 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; probabilities close failed |
| lr-default | multiclass_weighted_wide | float64 / 41 | Forced GPU / sklearn | fit /  | objective gap exceeds tolerance |
| lr-default | multiclass_weighted_wide | float64 / 41 | Forced GPU / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: objective gap exceeds tolerance |
| lr-default | multiclass_weighted_wide | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; probabilities close failed |
| lr-requested1 | binary_medium | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_wide | float32 / 41 | Direct auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_wide | float32 / 41 | Patched auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_wide | float32 / 41 | Forced GPU / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_large | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_large | float32 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_large | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_boundary | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_boundary | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | wide_40k_2048 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | wide_40k_2048 | float32 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | wide_40k_2048 | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | wide_20k_4096 | float32 / 41 | Direct auto / sklearn | predict_proba / 1, 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | wide_20k_4096 | float32 / 41 | Patched auto / sklearn | predict_proba / 1, 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | wide_20k_4096 | float32 / 41 | Forced GPU / sklearn | predict_proba / 1, 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_1000_128 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_1000_128 | float32 / 41 | Forced GPU / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | fit /  | objective gap exceeds tolerance |
| lr-requested1 | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | predict, predict_proba / 1 | fit prerequisite: objective gap exceeds tolerance |
| lr-requested1 | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | predict / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995 |
| lr-requested1 | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995; probabilities close failed |
| lr-requested1 | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | fit /  | objective gap exceeds tolerance |
| lr-requested1 | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | predict, predict_proba / 1 | fit prerequisite: objective gap exceeds tolerance |
| lr-requested1 | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | predict / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995 |
| lr-requested1 | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995; probabilities close failed |
| lr-requested1 | grid_10000_1024 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_10000_1024 | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_100000_1024 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_100000_1024 | float32 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_100000_1024 | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_500000_1024 | float32 / 41 | Direct auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_500000_1024 | float32 / 41 | Patched auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_500000_1024 | float32 / 41 | Forced GPU / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | multiclass_weighted_wide | float32 / 41 | Direct auto / sklearn | fit /  | objective gap exceeds tolerance |
| lr-requested1 | multiclass_weighted_wide | float32 / 41 | Direct auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: objective gap exceeds tolerance |
| lr-requested1 | multiclass_weighted_wide | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; probabilities close failed |
| lr-requested1 | multiclass_weighted_wide | float32 / 41 | Patched auto / sklearn | fit /  | objective gap exceeds tolerance |
| lr-requested1 | multiclass_weighted_wide | float32 / 41 | Patched auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: objective gap exceeds tolerance |
| lr-requested1 | multiclass_weighted_wide | float32 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; probabilities close failed |
| lr-requested1 | multiclass_weighted_wide | float32 / 41 | Forced GPU / sklearn | fit /  | objective gap exceeds tolerance |
| lr-requested1 | multiclass_weighted_wide | float32 / 41 | Forced GPU / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: objective gap exceeds tolerance |
| lr-requested1 | multiclass_weighted_wide | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; probabilities close failed |
| lr-requested1 | binary_medium | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_wide | float64 / 41 | Direct auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_wide | float64 / 41 | Patched auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_wide | float64 / 41 | Forced GPU / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_large | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_large | float64 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_large | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_boundary | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | binary_boundary | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | wide_40k_2048 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | wide_40k_2048 | float64 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | wide_40k_2048 | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | wide_20k_4096 | float64 / 41 | Direct auto / sklearn | predict_proba / 1, 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | wide_20k_4096 | float64 / 41 | Patched auto / sklearn | predict_proba / 1, 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | wide_20k_4096 | float64 / 41 | Forced GPU / sklearn | predict_proba / 1, 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_1000_128 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_1000_128 | float64 / 41 | Forced GPU / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_1000_1024 | float64 / 41 | Direct auto / sklearn | fit /  | objective gap exceeds tolerance |
| lr-requested1 | grid_1000_1024 | float64 / 41 | Direct auto / sklearn | predict, predict_proba / 1 | fit prerequisite: objective gap exceeds tolerance |
| lr-requested1 | grid_1000_1024 | float64 / 41 | Direct auto / sklearn | predict / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995 |
| lr-requested1 | grid_1000_1024 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995; probabilities close failed |
| lr-requested1 | grid_1000_1024 | float64 / 41 | Forced GPU / sklearn | fit /  | objective gap exceeds tolerance |
| lr-requested1 | grid_1000_1024 | float64 / 41 | Forced GPU / sklearn | predict, predict_proba / 1 | fit prerequisite: objective gap exceeds tolerance |
| lr-requested1 | grid_1000_1024 | float64 / 41 | Forced GPU / sklearn | predict / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995 |
| lr-requested1 | grid_1000_1024 | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; inference prediction agreement below 0.995; probabilities close failed |
| lr-requested1 | grid_10000_1024 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_10000_1024 | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_100000_1024 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_100000_1024 | float64 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_100000_1024 | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_500000_1024 | float64 / 41 | Direct auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_500000_1024 | float64 / 41 | Patched auto / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | grid_500000_1024 | float64 / 41 | Forced GPU / sklearn | predict_proba / 4096 | inference output check failed; probabilities close failed |
| lr-requested1 | multiclass_weighted_wide | float64 / 41 | Direct auto / sklearn | fit /  | objective gap exceeds tolerance |
| lr-requested1 | multiclass_weighted_wide | float64 / 41 | Direct auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: objective gap exceeds tolerance |
| lr-requested1 | multiclass_weighted_wide | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; probabilities close failed |
| lr-requested1 | multiclass_weighted_wide | float64 / 41 | Patched auto / sklearn | fit /  | objective gap exceeds tolerance |
| lr-requested1 | multiclass_weighted_wide | float64 / 41 | Patched auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: objective gap exceeds tolerance |
| lr-requested1 | multiclass_weighted_wide | float64 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; probabilities close failed |
| lr-requested1 | multiclass_weighted_wide | float64 / 41 | Forced GPU / sklearn | fit /  | objective gap exceeds tolerance |
| lr-requested1 | multiclass_weighted_wide | float64 / 41 | Forced GPU / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: objective gap exceeds tolerance |
| lr-requested1 | multiclass_weighted_wide | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | fit prerequisite: objective gap exceeds tolerance; inference output check failed; probabilities close failed |
| lr-strict4096-default | wide_20k_4096 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-strict4096-default | wide_20k_4096 | float32 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-strict4096-default | wide_20k_4096 | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-strict4096-default | wide_20k_4096 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-strict4096-default | wide_20k_4096 | float64 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-strict4096-default | wide_20k_4096 | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-strict4096-requested1 | wide_20k_4096 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-strict4096-requested1 | wide_20k_4096 | float32 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-strict4096-requested1 | wide_20k_4096 | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-strict4096-requested1 | wide_20k_4096 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-strict4096-requested1 | wide_20k_4096 | float64 / 41 | Patched auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-strict4096-requested1 | wide_20k_4096 | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-accuracy-matched-default | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | fit /  | sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-default | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-default | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance; inference output check failed; probabilities close failed |
| lr-accuracy-matched-default | grid_1000_1024 | float32 / 41 | Patched auto / sklearn | fit /  | gradient exceeds tolerance; sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-default | grid_1000_1024 | float32 / 41 | Patched auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: gradient exceeds tolerance; fit prerequisite: sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-default | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | fit /  | sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-default | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-default | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance; inference output check failed; probabilities close failed |
| lr-accuracy-matched-default | multiclass_weighted_wide | float32 / 41 | Direct auto / sklearn | fit /  | sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-default | multiclass_weighted_wide | float32 / 41 | Direct auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-default | multiclass_weighted_wide | float32 / 41 | Patched auto / sklearn | fit /  | sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-default | multiclass_weighted_wide | float32 / 41 | Patched auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-default | multiclass_weighted_wide | float32 / 41 | Forced GPU / sklearn | fit /  | sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-default | multiclass_weighted_wide | float32 / 41 | Forced GPU / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-default | grid_1000_1024 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-accuracy-matched-default | grid_1000_1024 | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-accuracy-matched-requested1 | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | fit /  | sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-requested1 | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-requested1 | grid_1000_1024 | float32 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance; inference output check failed; probabilities close failed |
| lr-accuracy-matched-requested1 | grid_1000_1024 | float32 / 41 | Patched auto / sklearn | fit /  | gradient exceeds tolerance; sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-requested1 | grid_1000_1024 | float32 / 41 | Patched auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: gradient exceeds tolerance; fit prerequisite: sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-requested1 | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | fit /  | sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-requested1 | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-requested1 | grid_1000_1024 | float32 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance; inference output check failed; probabilities close failed |
| lr-accuracy-matched-requested1 | multiclass_weighted_wide | float32 / 41 | Direct auto / sklearn | fit /  | sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-requested1 | multiclass_weighted_wide | float32 / 41 | Direct auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-requested1 | multiclass_weighted_wide | float32 / 41 | Patched auto / sklearn | fit /  | sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-requested1 | multiclass_weighted_wide | float32 / 41 | Patched auto / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-requested1 | multiclass_weighted_wide | float32 / 41 | Forced GPU / sklearn | fit /  | sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-requested1 | multiclass_weighted_wide | float32 / 41 | Forced GPU / sklearn | predict, predict_proba / 1, 256, 4096 | fit prerequisite: sklearn reference: gradient exceeds tolerance |
| lr-accuracy-matched-requested1 | grid_1000_1024 | float64 / 41 | Direct auto / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |
| lr-accuracy-matched-requested1 | grid_1000_1024 | float64 / 41 | Forced GPU / sklearn | predict_proba / 256, 4096 | inference output check failed; probabilities close failed |

## Evidence and reproducibility

Large result files are distributed in a verified release asset. [Restore the evidence](../benchmarks/evidence/sklearn-refresh-evidence-2026-10-31.md) before following links to those files or regenerating this report.

[Machine-readable summary, all normalized comparisons and failure causes](../benchmarks/results/sklearn-refresh-2026-10-31/summary.json)

| Input panel | SHA-256 |
| --- | --- |
| [knn-broad-summary](../benchmarks/results/sklearn-refresh-2026-10-31/knn-broad-summary.json) | `6e77342fb401683871c1b1e0ae680732008d776a471f4fb78e7c6149f754cd71` |
| [knn-cold-summary](../benchmarks/results/sklearn-refresh-2026-10-31/knn-cold-summary.json) | `b9df0b1b8c050548271d807e2045986d36cde7a49c81c5cd026f43475895c57e` |
| [knn-first-query-summary](../benchmarks/results/sklearn-refresh-2026-10-31/knn-first-query-summary.json) | `881b9db66569748ca2d0e39b89c3610208584d257d86da8351a3a610bdbfdf36` |
| [svc-seed41](../benchmarks/results/sklearn-refresh-2026-10-31/svc-seed41.json) | `90ceeae55811959eeda421194a9de65283870bb122760fc7f2b652f0d8631b70` |
| [svc-seed42](../benchmarks/results/sklearn-refresh-2026-10-31/svc-seed42.json) | `e7e81cd5e12292d8b00b3e825c7bf1a92e04a6bf5ad23f726f20b0f16f52804f` |
| [lr-default](../benchmarks/results/sklearn-refresh-2026-10-31/lr-default.json) | `0bb7b21b208ef1f01dacb5b7a65ea509a408c883b00bd6991391250c4f336ce1` |
| [lr-requested1](../benchmarks/results/sklearn-refresh-2026-10-31/lr-requested1.json) | `5eb4c9f5c3d8da3f8115d535ac459913aa210f042479b4d93b8ba3c87930a96c` |
| [lr-strict4096-default](../benchmarks/results/sklearn-refresh-2026-10-31/lr-strict4096-default.json) | `2a560330d89c76a7d9998d93281c71ca975e261fa758528f990ab22908eb8ee3` |
| [lr-strict4096-requested1](../benchmarks/results/sklearn-refresh-2026-10-31/lr-strict4096-requested1.json) | `129c8763075555d1217064b0fb4afd87b88c85175f6f0c90e3b47cf61e3baf19` |
| [grid-default-seed41](../benchmarks/results/sklearn-refresh-2026-10-31/grid-default-seed41.json) | `49fb5381e761bb6a38f175a39214bca1cc2e0c8e20ecda1a16582048c78faf2d` |
| [lr-accuracy-matched-default](../benchmarks/results/sklearn-refresh-2026-10-31/lr-accuracy-matched-default.json) | `25da501193977c5921e30219211549eaffe456ea541f34babe347c410fe3a88a` |
| [lr-accuracy-matched-requested1](../benchmarks/results/sklearn-refresh-2026-10-31/lr-accuracy-matched-requested1.json) | `2565616fbe58e807e198cb40cbcb19e92f0e07524667f0770dad545ce3153ebb` |

[Superseded original LR panels and archive manifest](../benchmarks/results/sklearn-refresh-2026-10-31/initial-lr-gradient-audit/manifest.json). Every archived file hash was verified, along with matching original/current fixture hashes, production/native hashes and timing/data parameters. These original panels are provenance only; none of their timings enter the tables above.

[environment.json](../benchmarks/results/sklearn-refresh-2026-10-31/environment.json)

[run-plan.json](../benchmarks/results/sklearn-refresh-2026-10-31/run-plan.json)

[source-state.json](../benchmarks/results/sklearn-refresh-2026-10-31/source-state.json)

[queue-state.json](../benchmarks/results/sklearn-refresh-2026-10-31/queue-state.json)

[supplemental-queue-state.json](../benchmarks/results/sklearn-refresh-2026-10-31/supplemental-queue-state.json)

[corrected-lr-queue-state.json](../benchmarks/results/sklearn-refresh-2026-10-31/corrected-lr-queue-state.json)

[final-integrity.json](../benchmarks/results/sklearn-refresh-2026-10-31/final-integrity.json)

[Final benchmark and report helper archive](../benchmarks/reference/sklearn-refresh-harnesses-final-2026-10-31.zip)

[Final benchmark and report helper archive manifest](../benchmarks/reference/sklearn-refresh-harnesses-final-2026-10-31.manifest.json)

The linked panels retain source and native integrity checks for each timing run. Source hashes, native hashes, package versions, thread settings, warmups and actual backend records remain in the linked raw evidence. Historical SVC implementation controls are retained there but are not the sklearn comparison reported here. Full-grid KNN fit rows are deduplicated by training shape: 18 measured fits represent the original 54 repeated query-count fit labels.
