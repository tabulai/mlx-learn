# Intermediate followup: predeclared legacy-order control

**This is completed intermediate evidence, not the final startup implementation.** The user subsequently authorized a startup architecture change. Its full benchmark reruns will use the separate `knn-startup-final-2026-10-30` prefix; none of these retained observations is overwritten or relabelled.

Nine fixtures comprise all three seeds (151, 251, 252) of Digits, Breast Cancer, and offset +1e4 4096×128. Digits/Breast each retain all nine operations; offset retains query and fresh fit-plus-query. This 60-cell robustness panel was declared before the new full-run results to check earlier GPU/CPU exceptions. Five isolated workers use 31 paired rounds and 1 second continuous warmup per worker/operation, with the historical alternating forward/reverse order. The separate full 48-fixture control uses balanced order and 0.25 second warmup. Absolute timings must not be compared across those protocols or the faster run selected.

Each comparison below is within this run. Preceding auto is the immediate `knn-latency-final-2026-10-30` source and exact archived native runtime; historical forced GPU is `knn-small-final-2026-10-30`. All source/runtime/harness archives and raw-record integrity checks pass. Live source files are intentionally not required to match these superseded snapshots.

Ratios are comparator/current time; below 1× means the automatic replica is slower and is bold. M/P denotes ratio of medians / median paired ratio. A >5% runtime slowdown means ratio <1/1.05 for both statistics and both automatic replicas; this is descriptive, not a significance test.

## Slowdown screen

| Comparator | Cells exceeding 5% in both statistics and replicas |
| --- | --- |
| Immediate prior auto (latency-final) | 0 |
| Current forced GPU | 0 |
| Historical forced GPU (small-final) | 0 |

No cells cross this full two-replica/two-statistic threshold; every smaller or inconsistent slowdown remains in the complete tables.

## All 60 median timing cells

| Family | Seed | Operation | Current auto ms | Current auto A/A ms | Current forced GPU ms | Immediate prior auto (latency-final) ms | Historical forced GPU (small-final) ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 0.518917 | 0.522375 | 1.026084 | 0.568750 | 0.591417 |
| Digits | 151 | kneighbors | 0.324459 | 0.370292 | 0.918292 | 0.428250 | 0.916416 |
| Digits | 151 | indices | 0.361250 | 0.308125 | 0.841791 | 0.437875 | 0.928709 |
| Digits | 151 | predict_uniform | 0.432917 | 0.452292 | 0.939625 | 0.530458 | 1.107083 |
| Digits | 151 | proba_uniform | 0.385500 | 0.395500 | 0.901417 | 0.382875 | 0.918083 |
| Digits | 151 | regress_uniform | 0.371417 | 0.381084 | 0.459208 | 0.419417 | 0.406959 |
| Digits | 151 | predict_distance | 0.341500 | 0.382875 | 0.955250 | 0.463458 | 0.466833 |
| Digits | 151 | proba_distance | 0.407750 | 0.329500 | 0.977375 | 0.475292 | 1.024792 |
| Digits | 151 | regress_distance | 0.397958 | 0.382583 | 0.436167 | 0.446583 | 0.436208 |
| Digits | 251 | fit_query | 0.495708 | 0.525625 | 1.107209 | 0.596750 | 1.088792 |
| Digits | 251 | kneighbors | 0.356209 | 0.329916 | 0.781792 | 0.430125 | 0.955334 |
| Digits | 251 | indices | 0.369500 | 0.365250 | 0.456458 | 0.440458 | 0.909083 |
| Digits | 251 | predict_uniform | 0.386000 | 0.378917 | 0.491583 | 0.371542 | 0.864333 |
| Digits | 251 | proba_uniform | 0.373542 | 0.389250 | 0.466583 | 0.439917 | 0.457917 |
| Digits | 251 | regress_uniform | 0.304375 | 0.379917 | 0.483125 | 0.452041 | 0.929917 |
| Digits | 251 | predict_distance | 0.406041 | 0.330208 | 0.509333 | 0.459958 | 0.965750 |
| Digits | 251 | proba_distance | 0.427792 | 0.419584 | 0.534750 | 0.496875 | 0.583666 |
| Digits | 251 | regress_distance | 0.411667 | 0.412125 | 1.025708 | 0.419083 | 0.950833 |
| Digits | 252 | fit_query | 0.529917 | 0.492167 | 0.617375 | 0.578916 | 1.068542 |
| Digits | 252 | kneighbors | 0.383583 | 0.381167 | 0.404333 | 0.444750 | 0.919875 |
| Digits | 252 | indices | 0.369167 | 0.369083 | 0.955667 | 0.343833 | 0.873541 |
| Digits | 252 | predict_uniform | 0.363708 | 0.377292 | 0.449750 | 0.414333 | 0.413791 |
| Digits | 252 | proba_uniform | 0.323958 | 0.392459 | 0.963500 | 0.457750 | 0.888375 |
| Digits | 252 | regress_uniform | 0.383667 | 0.309917 | 0.448417 | 0.426083 | 0.934167 |
| Digits | 252 | predict_distance | 0.400375 | 0.400084 | 0.425292 | 0.452334 | 0.433500 |
| Digits | 252 | proba_distance | 0.434416 | 0.431500 | 0.776500 | 0.432042 | 0.918292 |
| Digits | 252 | regress_distance | 0.413375 | 0.418042 | 0.492916 | 0.462500 | 0.410750 |
| Breast Cancer | 151 | fit_query | 0.257167 | 0.253333 | 0.317375 | 0.296584 | 0.336500 |
| Breast Cancer | 151 | kneighbors | 0.165167 | 0.157292 | 0.281542 | 0.199667 | 0.494792 |
| Breast Cancer | 151 | indices | 0.170500 | 0.168375 | 0.246459 | 0.202958 | 0.225667 |
| Breast Cancer | 151 | predict_uniform | 0.135792 | 0.155334 | 0.289917 | 0.213334 | 0.296625 |
| Breast Cancer | 151 | proba_uniform | 0.163833 | 0.142042 | 0.292625 | 0.204209 | 0.317875 |
| Breast Cancer | 151 | regress_uniform | 0.166000 | 0.156166 | 0.233834 | 0.200958 | 0.241416 |
| Breast Cancer | 151 | predict_distance | 0.194458 | 0.198417 | 0.521042 | 0.227209 | 0.303500 |
| Breast Cancer | 151 | proba_distance | 0.201916 | 0.197375 | 0.288959 | 0.233209 | 0.266542 |
| Breast Cancer | 151 | regress_distance | 0.165875 | 0.171583 | 0.320541 | 0.223542 | 0.297292 |
| Breast Cancer | 251 | fit_query | 0.266375 | 0.262833 | 0.392292 | 0.290542 | 0.384417 |
| Breast Cancer | 251 | kneighbors | 0.170292 | 0.160625 | 0.240708 | 0.210666 | 0.233125 |
| Breast Cancer | 251 | indices | 0.127667 | 0.143250 | 0.258292 | 0.193834 | 0.287834 |
| Breast Cancer | 251 | predict_uniform | 0.172916 | 0.137625 | 0.313791 | 0.198041 | 0.323000 |
| Breast Cancer | 251 | proba_uniform | 0.184084 | 0.158833 | 0.253250 | 0.212750 | 0.261834 |
| Breast Cancer | 251 | regress_uniform | 0.170917 | 0.166875 | 0.318833 | 0.201583 | 0.273209 |
| Breast Cancer | 251 | predict_distance | 0.181167 | 0.180833 | 0.263542 | 0.222666 | 0.248500 |
| Breast Cancer | 251 | proba_distance | 0.172500 | 0.216417 | 0.287958 | 0.239834 | 0.289958 |
| Breast Cancer | 251 | regress_distance | 0.189541 | 0.158667 | 0.293208 | 0.218792 | 0.304875 |
| Breast Cancer | 252 | fit_query | 0.258709 | 0.260625 | 0.390417 | 0.295042 | 0.311916 |
| Breast Cancer | 252 | kneighbors | 0.135875 | 0.167334 | 0.248000 | 0.195459 | 0.247167 |
| Breast Cancer | 252 | indices | 0.150125 | 0.129541 | 0.279959 | 0.193291 | 0.256667 |
| Breast Cancer | 252 | predict_uniform | 0.170000 | 0.151375 | 0.242333 | 0.206333 | 0.256042 |
| Breast Cancer | 252 | proba_uniform | 0.168875 | 0.177625 | 0.302792 | 0.209459 | 0.287333 |
| Breast Cancer | 252 | regress_uniform | 0.169417 | 0.155292 | 0.258791 | 0.208792 | 0.236292 |
| Breast Cancer | 252 | predict_distance | 0.155958 | 0.188125 | 0.292166 | 0.224250 | 0.288667 |
| Breast Cancer | 252 | proba_distance | 0.195250 | 0.168375 | 0.278833 | 0.232375 | 0.417958 |
| Breast Cancer | 252 | regress_distance | 0.176708 | 0.186833 | 0.265625 | 0.213959 | 0.268917 |
| Offset +1e4,4096×128 | 151 | kneighbors | 0.412417 | 0.401958 | 0.402875 | 0.417000 | 0.402500 |
| Offset +1e4,4096×128 | 151 | fit_query | 0.683083 | 0.663917 | 0.699917 | 0.693833 | 0.675875 |
| Offset +1e4,4096×128 | 251 | kneighbors | 0.433125 | 0.412834 | 0.443292 | 0.431583 | 0.445333 |
| Offset +1e4,4096×128 | 251 | fit_query | 0.762666 | 0.727291 | 0.720041 | 0.740750 | 0.819333 |
| Offset +1e4,4096×128 | 252 | kneighbors | 0.413917 | 0.398166 | 0.389583 | 0.410750 | 0.392417 |
| Offset +1e4,4096×128 | 252 | fit_query | 0.709542 | 0.712875 | 0.710500 | 0.706375 | 0.724750 |

## All 60 comparisons against Immediate prior auto (latency-final)

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 1.0960× | 1.0990× | 1.0888× | 1.0790× |
| Digits | 151 | kneighbors | 1.3199× | 1.2893× | 1.1565× | 1.1589× |
| Digits | 151 | indices | 1.2121× | 1.2068× | 1.4211× | 1.3842× |
| Digits | 151 | predict_uniform | 1.2253× | 1.1446× | 1.1728× | 1.1731× |
| Digits | 151 | proba_uniform | **0.9932×** | 1.0009× | **0.9681×** | 1.0085× |
| Digits | 151 | regress_uniform | 1.1292× | 1.1144× | 1.1006× | 1.0927× |
| Digits | 151 | predict_distance | 1.3571× | 1.3714× | 1.2105× | 1.2226× |
| Digits | 151 | proba_distance | 1.1656× | 1.1691× | 1.4425× | 1.3634× |
| Digits | 151 | regress_distance | 1.1222× | 1.0859× | 1.1673× | 1.1523× |
| Digits | 251 | fit_query | 1.2038× | 1.2013× | 1.1353× | 1.1581× |
| Digits | 251 | kneighbors | 1.2075× | 1.2426× | 1.3037× | 1.3397× |
| Digits | 251 | indices | 1.1920× | 1.1806× | 1.2059× | 1.2096× |
| Digits | 251 | predict_uniform | **0.9625×** | 1.0224× | **0.9805×** | **0.9863×** |
| Digits | 251 | proba_uniform | 1.1777× | 1.1446× | 1.1302× | 1.0755× |
| Digits | 251 | regress_uniform | 1.4851× | 1.4637× | 1.1898× | 1.1215× |
| Digits | 251 | predict_distance | 1.1328× | 1.1352× | 1.3929× | 1.3511× |
| Digits | 251 | proba_distance | 1.1615× | 1.1355× | 1.1842× | 1.1553× |
| Digits | 251 | regress_distance | 1.0180× | 1.0646× | 1.0169× | 1.0340× |
| Digits | 252 | fit_query | 1.0925× | 1.0964× | 1.1763× | 1.1893× |
| Digits | 252 | kneighbors | 1.1595× | 1.1340× | 1.1668× | 1.1423× |
| Digits | 252 | indices | **0.9314×** | **0.9425×** | **0.9316×** | **0.9618×** |
| Digits | 252 | predict_uniform | 1.1392× | 1.1236× | 1.0982× | 1.0940× |
| Digits | 252 | proba_uniform | 1.4130× | 1.3936× | 1.1664× | 1.1515× |
| Digits | 252 | regress_uniform | 1.1106× | 1.1103× | 1.3748× | 1.4010× |
| Digits | 252 | predict_distance | 1.1298× | 1.1070× | 1.1306× | 1.1487× |
| Digits | 252 | proba_distance | **0.9945×** | 1.0204× | 1.0013× | 1.0340× |
| Digits | 252 | regress_distance | 1.1188× | 1.0873× | 1.1063× | 1.1018× |
| Breast Cancer | 151 | fit_query | 1.1533× | 1.2026× | 1.1707× | 1.1702× |
| Breast Cancer | 151 | kneighbors | 1.2089× | 1.1792× | 1.2694× | 1.2381× |
| Breast Cancer | 151 | indices | 1.1904× | 1.2764× | 1.2054× | 1.2221× |
| Breast Cancer | 151 | predict_uniform | 1.5710× | 1.6057× | 1.3734× | 1.2751× |
| Breast Cancer | 151 | proba_uniform | 1.2464× | 1.2632× | 1.4377× | 1.4568× |
| Breast Cancer | 151 | regress_uniform | 1.2106× | 1.2515× | 1.2868× | 1.2354× |
| Breast Cancer | 151 | predict_distance | 1.1684× | 1.1781× | 1.1451× | 1.1365× |
| Breast Cancer | 151 | proba_distance | 1.1550× | 1.1746× | 1.1816× | 1.1548× |
| Breast Cancer | 151 | regress_distance | 1.3477× | 1.3536× | 1.3028× | 1.2184× |
| Breast Cancer | 251 | fit_query | 1.0907× | 1.1087× | 1.1054× | 1.1225× |
| Breast Cancer | 251 | kneighbors | 1.2371× | 1.3034× | 1.3115× | 1.3676× |
| Breast Cancer | 251 | indices | 1.5183× | 1.5350× | 1.3531× | 1.3274× |
| Breast Cancer | 251 | predict_uniform | 1.1453× | 1.1748× | 1.4390× | 1.4803× |
| Breast Cancer | 251 | proba_uniform | 1.1557× | 1.2201× | 1.3395× | 1.3101× |
| Breast Cancer | 251 | regress_uniform | 1.1794× | 1.1805× | 1.2080× | 1.1573× |
| Breast Cancer | 251 | predict_distance | 1.2291× | 1.2182× | 1.2313× | 1.2237× |
| Breast Cancer | 251 | proba_distance | 1.3903× | 1.3666× | 1.1082× | 1.1150× |
| Breast Cancer | 251 | regress_distance | 1.1543× | 1.2057× | 1.3789× | 1.3513× |
| Breast Cancer | 252 | fit_query | 1.1404× | 1.1050× | 1.1321× | 1.1556× |
| Breast Cancer | 252 | kneighbors | 1.4385× | 1.4625× | 1.1681× | 1.2224× |
| Breast Cancer | 252 | indices | 1.2875× | 1.3524× | 1.4921× | 1.5414× |
| Breast Cancer | 252 | predict_uniform | 1.2137× | 1.2786× | 1.3631× | 1.2973× |
| Breast Cancer | 252 | proba_uniform | 1.2403× | 1.2016× | 1.1792× | 1.1792× |
| Breast Cancer | 252 | regress_uniform | 1.2324× | 1.2030× | 1.3445× | 1.3124× |
| Breast Cancer | 252 | predict_distance | 1.4379× | 1.4285× | 1.1920× | 1.2273× |
| Breast Cancer | 252 | proba_distance | 1.1901× | 1.2386× | 1.3801× | 1.3576× |
| Breast Cancer | 252 | regress_distance | 1.2108× | 1.2031× | 1.1452× | 1.1764× |
| Offset +1e4,4096×128 | 151 | kneighbors | 1.0111× | 1.0387× | 1.0374× | 1.0244× |
| Offset +1e4,4096×128 | 151 | fit_query | 1.0157× | 1.0048× | 1.0451× | 1.0536× |
| Offset +1e4,4096×128 | 251 | kneighbors | **0.9964×** | 1.0223× | 1.0454× | 1.0294× |
| Offset +1e4,4096×128 | 251 | fit_query | **0.9713×** | **0.9727×** | 1.0185× | 1.0047× |
| Offset +1e4,4096×128 | 252 | kneighbors | **0.9923×** | 1.0243× | 1.0316× | 1.0683× |
| Offset +1e4,4096×128 | 252 | fit_query | **0.9955×** | 1.0008× | **0.9909×** | 1.0039× |

## All 60 comparisons against Current forced GPU

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 1.9774× | 1.9366× | 1.9643× | 1.9593× |
| Digits | 151 | kneighbors | 2.8302× | 2.7808× | 2.4799× | 2.3352× |
| Digits | 151 | indices | 2.3302× | 2.4036× | 2.7320× | 2.4684× |
| Digits | 151 | predict_uniform | 2.1705× | 1.9668× | 2.0775× | 2.0787× |
| Digits | 151 | proba_uniform | 2.3383× | 2.3116× | 2.2792× | 2.2707× |
| Digits | 151 | regress_uniform | 1.2364× | 1.3599× | 1.2050× | 1.2930× |
| Digits | 151 | predict_distance | 2.7972× | 2.3954× | 2.4949× | 2.6123× |
| Digits | 151 | proba_distance | 2.3970× | 2.3808× | 2.9662× | 2.7532× |
| Digits | 151 | regress_distance | 1.0960× | 1.0631× | 1.1401× | 1.1223× |
| Digits | 251 | fit_query | 2.2336× | 2.2401× | 2.1065× | 2.2041× |
| Digits | 251 | kneighbors | 2.1948× | 2.2142× | 2.3697× | 1.9458× |
| Digits | 251 | indices | 1.2353× | 1.4290× | 1.2497× | 1.3404× |
| Digits | 251 | predict_uniform | 1.2735× | 1.5843× | 1.2973× | 1.4933× |
| Digits | 251 | proba_uniform | 1.2491× | 1.3740× | 1.1987× | 1.2477× |
| Digits | 251 | regress_uniform | 1.5873× | 1.6815× | 1.2717× | 1.3278× |
| Digits | 251 | predict_distance | 1.2544× | 1.2505× | 1.5425× | 1.5714× |
| Digits | 251 | proba_distance | 1.2500× | 1.2553× | 1.2745× | 1.2936× |
| Digits | 251 | regress_distance | 2.4916× | 2.4752× | 2.4888× | 2.4529× |
| Digits | 252 | fit_query | 1.1650× | 1.2569× | 1.2544× | 1.3378× |
| Digits | 252 | kneighbors | 1.0541× | 1.0484× | 1.0608× | 1.1492× |
| Digits | 252 | indices | 2.5887× | 2.5847× | 2.5893× | 2.6927× |
| Digits | 252 | predict_uniform | 1.2366× | 1.3489× | 1.1920× | 1.3541× |
| Digits | 252 | proba_uniform | 2.9742× | 2.7019× | 2.4550× | 2.5327× |
| Digits | 252 | regress_uniform | 1.1688× | 1.3666× | 1.4469× | 1.5878× |
| Digits | 252 | predict_distance | 1.0622× | 1.1099× | 1.0630× | 1.0411× |
| Digits | 252 | proba_distance | 1.7875× | 1.6911× | 1.7995× | 1.9587× |
| Digits | 252 | regress_distance | 1.1924× | 1.2808× | 1.1791× | 1.2283× |
| Breast Cancer | 151 | fit_query | 1.2341× | 1.2981× | 1.2528× | 1.2869× |
| Breast Cancer | 151 | kneighbors | 1.7046× | 1.8749× | 1.7899× | 1.8333× |
| Breast Cancer | 151 | indices | 1.4455× | 1.5772× | 1.4638× | 1.6057× |
| Breast Cancer | 151 | predict_uniform | 2.1350× | 2.1350× | 1.8664× | 2.1263× |
| Breast Cancer | 151 | proba_uniform | 1.7861× | 1.9355× | 2.0601× | 2.1550× |
| Breast Cancer | 151 | regress_uniform | 1.4086× | 1.3863× | 1.4973× | 1.4975× |
| Breast Cancer | 151 | predict_distance | 2.6795× | 2.1849× | 2.6260× | 2.2312× |
| Breast Cancer | 151 | proba_distance | 1.4311× | 1.4993× | 1.4640× | 1.5431× |
| Breast Cancer | 151 | regress_distance | 1.9324× | 2.0246× | 1.8681× | 1.9111× |
| Breast Cancer | 251 | fit_query | 1.4727× | 1.7631× | 1.4926× | 1.5909× |
| Breast Cancer | 251 | kneighbors | 1.4135× | 1.5925× | 1.4986× | 1.6073× |
| Breast Cancer | 251 | indices | 2.0232× | 2.0313× | 1.8031× | 1.9818× |
| Breast Cancer | 251 | predict_uniform | 1.8147× | 1.9255× | 2.2800× | 2.2897× |
| Breast Cancer | 251 | proba_uniform | 1.3757× | 1.5504× | 1.5944× | 1.6370× |
| Breast Cancer | 251 | regress_uniform | 1.8654× | 2.0547× | 1.9106× | 1.9580× |
| Breast Cancer | 251 | predict_distance | 1.4547× | 1.4929× | 1.4574× | 1.4254× |
| Breast Cancer | 251 | proba_distance | 1.6693× | 1.8405× | 1.3306× | 1.5421× |
| Breast Cancer | 251 | regress_distance | 1.5469× | 1.6701× | 1.8479× | 1.8981× |
| Breast Cancer | 252 | fit_query | 1.5091× | 1.5250× | 1.4980× | 1.4933× |
| Breast Cancer | 252 | kneighbors | 1.8252× | 1.9277× | 1.4821× | 1.7465× |
| Breast Cancer | 252 | indices | 1.8648× | 1.9542× | 2.1612× | 2.0916× |
| Breast Cancer | 252 | predict_uniform | 1.4255× | 1.5850× | 1.6009× | 1.6762× |
| Breast Cancer | 252 | proba_uniform | 1.7930× | 1.8081× | 1.7047× | 1.7004× |
| Breast Cancer | 252 | regress_uniform | 1.5275× | 1.6921× | 1.6665× | 1.7038× |
| Breast Cancer | 252 | predict_distance | 1.8734× | 1.9481× | 1.5530× | 1.7534× |
| Breast Cancer | 252 | proba_distance | 1.4281× | 1.6859× | 1.6560× | 1.6197× |
| Breast Cancer | 252 | regress_distance | 1.5032× | 1.4994× | 1.4217× | 1.4229× |
| Offset +1e4,4096×128 | 151 | kneighbors | **0.9769×** | 1.0297× | 1.0023× | 1.0233× |
| Offset +1e4,4096×128 | 151 | fit_query | 1.0246× | 1.0676× | 1.0542× | 1.0851× |
| Offset +1e4,4096×128 | 251 | kneighbors | 1.0235× | 1.0047× | 1.0738× | 1.0098× |
| Offset +1e4,4096×128 | 251 | fit_query | **0.9441×** | **0.9585×** | **0.9900×** | **0.9701×** |
| Offset +1e4,4096×128 | 252 | kneighbors | **0.9412×** | **0.9701×** | **0.9784×** | **0.9876×** |
| Offset +1e4,4096×128 | 252 | fit_query | 1.0013× | **0.9829×** | **0.9967×** | 1.0015× |

## All 60 comparisons against Historical forced GPU (small-final)

| Family | Seed | Operation | Current M | Current P | A/A M | A/A P |
| --- | --- | --- | --- | --- | --- | --- |
| Digits | 151 | fit_query | 1.1397× | 1.2242× | 1.1322× | 1.1884× |
| Digits | 151 | kneighbors | 2.8244× | 2.7208× | 2.4748× | 2.4799× |
| Digits | 151 | indices | 2.5708× | 2.7153× | 3.0141× | 2.9117× |
| Digits | 151 | predict_uniform | 2.5573× | 2.3312× | 2.4477× | 2.2929× |
| Digits | 151 | proba_uniform | 2.3815× | 2.2720× | 2.3213× | 2.3312× |
| Digits | 151 | regress_uniform | 1.0957× | 1.1800× | 1.0679× | 1.1420× |
| Digits | 151 | predict_distance | 1.3670× | 1.4587× | 1.2193× | 1.3422× |
| Digits | 151 | proba_distance | 2.5133× | 2.6485× | 3.1101× | 2.8877× |
| Digits | 151 | regress_distance | 1.0961× | 1.1005× | 1.1402× | 1.1480× |
| Digits | 251 | fit_query | 2.1964× | 2.2795× | 2.0714× | 2.1138× |
| Digits | 251 | kneighbors | 2.6819× | 2.6751× | 2.8957× | 2.9796× |
| Digits | 251 | indices | 2.4603× | 2.3687× | 2.4889× | 2.3605× |
| Digits | 251 | predict_uniform | 2.2392× | 2.0918× | 2.2811× | 2.1239× |
| Digits | 251 | proba_uniform | 1.2259× | 1.1317× | 1.1764× | 1.1696× |
| Digits | 251 | regress_uniform | 3.0552× | 2.6121× | 2.4477× | 2.3912× |
| Digits | 251 | predict_distance | 2.3785× | 2.2549× | 2.9247× | 2.7597× |
| Digits | 251 | proba_distance | 1.3644× | 1.3922× | 1.3911× | 1.3741× |
| Digits | 251 | regress_distance | 2.3097× | 2.2738× | 2.3071× | 2.3071× |
| Digits | 252 | fit_query | 2.0164× | 2.0027× | 2.1711× | 2.0998× |
| Digits | 252 | kneighbors | 2.3981× | 2.2722× | 2.4133× | 2.3830× |
| Digits | 252 | indices | 2.3662× | 2.2451× | 2.3668× | 2.3751× |
| Digits | 252 | predict_uniform | 1.1377× | 1.1336× | 1.0967× | 1.1102× |
| Digits | 252 | proba_uniform | 2.7423× | 2.3413× | 2.2636× | 2.1751× |
| Digits | 252 | regress_uniform | 2.4348× | 2.3622× | 3.0142× | 2.7838× |
| Digits | 252 | predict_distance | 1.0827× | 1.1173× | 1.0835× | 1.1119× |
| Digits | 252 | proba_distance | 2.1139× | 2.1511× | 2.1281× | 2.2967× |
| Digits | 252 | regress_distance | **0.9936×** | 1.0239× | **0.9826×** | 1.0117× |
| Breast Cancer | 151 | fit_query | 1.3085× | 1.3147× | 1.3283× | 1.2922× |
| Breast Cancer | 151 | kneighbors | 2.9957× | 2.6300× | 3.1457× | 2.4224× |
| Breast Cancer | 151 | indices | 1.3236× | 1.3503× | 1.3403× | 1.3311× |
| Breast Cancer | 151 | predict_uniform | 2.1844× | 2.1969× | 1.9096× | 1.9630× |
| Breast Cancer | 151 | proba_uniform | 1.9402× | 2.0334× | 2.2379× | 2.2963× |
| Breast Cancer | 151 | regress_uniform | 1.4543× | 1.5098× | 1.5459× | 1.5089× |
| Breast Cancer | 151 | predict_distance | 1.5607× | 1.7907× | 1.5296× | 1.6433× |
| Breast Cancer | 151 | proba_distance | 1.3201× | 1.3462× | 1.3504× | 1.3250× |
| Breast Cancer | 151 | regress_distance | 1.7923× | 1.8297× | 1.7326× | 1.8808× |
| Breast Cancer | 251 | fit_query | 1.4431× | 1.4715× | 1.4626× | 1.4959× |
| Breast Cancer | 251 | kneighbors | 1.3690× | 1.4431× | 1.4514× | 1.4747× |
| Breast Cancer | 251 | indices | 2.2546× | 2.1545× | 2.0093× | 2.0377× |
| Breast Cancer | 251 | predict_uniform | 1.8680× | 1.9766× | 2.3470× | 2.2587× |
| Breast Cancer | 251 | proba_uniform | 1.4224× | 1.5785× | 1.6485× | 1.7009× |
| Breast Cancer | 251 | regress_uniform | 1.5985× | 1.8065× | 1.6372× | 1.7518× |
| Breast Cancer | 251 | predict_distance | 1.3717× | 1.3860× | 1.3742× | 1.3060× |
| Breast Cancer | 251 | proba_distance | 1.6809× | 1.6872× | 1.3398× | 1.6361× |
| Breast Cancer | 251 | regress_distance | 1.6085× | 1.6998× | 1.9215× | 1.9090× |
| Breast Cancer | 252 | fit_query | 1.2057× | 1.2275× | 1.1968× | 1.2585× |
| Breast Cancer | 252 | kneighbors | 1.8191× | 1.8058× | 1.4771× | 1.7447× |
| Breast Cancer | 252 | indices | 1.7097× | 1.8276× | 1.9814× | 1.9205× |
| Breast Cancer | 252 | predict_uniform | 1.5061× | 1.5845× | 1.6914× | 1.7006× |
| Breast Cancer | 252 | proba_uniform | 1.7015× | 1.8073× | 1.6176× | 1.6768× |
| Breast Cancer | 252 | regress_uniform | 1.3947× | 1.3731× | 1.5216× | 1.4595× |
| Breast Cancer | 252 | predict_distance | 1.8509× | 1.8390× | 1.5344× | 1.6729× |
| Breast Cancer | 252 | proba_distance | 2.1406× | 2.1737× | 2.4823× | 2.5453× |
| Breast Cancer | 252 | regress_distance | 1.5218× | 1.5099× | 1.4393× | 1.4721× |
| Offset +1e4,4096×128 | 151 | kneighbors | **0.9760×** | 1.0331× | 1.0013× | 1.0243× |
| Offset +1e4,4096×128 | 151 | fit_query | **0.9894×** | 1.0001× | 1.0180× | 1.0517× |
| Offset +1e4,4096×128 | 251 | kneighbors | 1.0282× | 1.0069× | 1.0787× | 1.0201× |
| Offset +1e4,4096×128 | 251 | fit_query | 1.0743× | 1.0228× | 1.1266× | 1.0232× |
| Offset +1e4,4096×128 | 252 | kneighbors | **0.9481×** | 1.0043× | **0.9856×** | 1.0197× |
| Offset +1e4,4096×128 | 252 | fit_query | 1.0214× | 1.0165× | 1.0167× | 1.0367× |

## Every audit phase

| Worker | Cold passes/audits | Warmup passes/audits | Probe passes/audits | Retained passes/audits |
| --- | --- | --- | --- | --- |
| Current auto | 9/9 | 220306/220306 | 60/60 | 1860/1860 |
| Current auto A/A | 9/9 | 221059/221059 | 60/60 | 1860/1860 |
| Current forced GPU | 9/9 | 169841/169841 | 60/60 | 1860/1860 |
| Immediate prior auto (latency-final) | 9/9 | 177776/177776 | 60/60 | 1860/1860 |
| Historical forced GPU (small-final) | 9/9 | 168804/168804 | 60/60 | 1860/1860 |

All first/middle/last, parity, position and predecessor diagnostics remain in the audit JSON. No audit failure filters a timing denominator.

## Intermediate initialized-process first-query panel

The completed 18-fixture × 30-fresh-model × 5-worker panel contains 2700 fresh models and 5400 audited query calls. Normal import/setup costs are recorded separately; all first and second queries, including the first model in each worker, are retained. This panel is not fresh-process startup latency. The later cold process run was interrupted and is not treated as complete acceptance evidence.

| Comparator | Metric | Faster fixture medians | Minimum ratio | Maximum ratio |
| --- | --- | --- | --- | --- |
| sklearn_default | fit_ms | 14/18 | **0.9128×** | 1.3786× |
| sklearn_default | first_query_ms | 18/18 | 6.5422× | 42.0831× |
| sklearn_default | fit_first_query_ms | 18/18 | 4.3253× | 14.9082× |
| sklearn_default | second_query_ms | 18/18 | 9.8178× | 49.7187× |
| sklearn_1 | fit_ms | 13/18 | **0.9027×** | 1.3411× |
| sklearn_1 | first_query_ms | 18/18 | 1.2606× | 3.2401× |
| sklearn_1 | fit_first_query_ms | 18/18 | 1.2333× | 2.1171× |
| sklearn_1 | second_query_ms | 18/18 | 1.5251× | 3.4181× |
| sklearn_auto | fit_ms | 16/18 | **0.9260×** | 12.8600× |
| sklearn_auto | first_query_ms | 18/18 | 2.6007× | 24.3939× |
| sklearn_auto | fit_first_query_ms | 18/18 | 2.1837× | 12.0580× |
| sklearn_auto | second_query_ms | 18/18 | 2.5642× | 26.7138× |
| Immediate prior auto (latency-final) | fit_ms | 5/18 | **0.9000×** | 1.0744× |
| Immediate prior auto (latency-final) | first_query_ms | 18/18 | 1.0231× | 1.1989× |
| Immediate prior auto (latency-final) | fit_first_query_ms | 15/18 | **0.9656×** | 1.1259× |
| Immediate prior auto (latency-final) | second_query_ms | 18/18 | 1.0279× | 1.3004× |

## Evidence

[Legacy raw results](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-45337c132701c2fa1e75) — SHA256 `b3a07683bff86e647c4d1dc44342fd13da69500a1a90ff5b67824b44616996de`.

[Independent intermediate audit, every cell and paired ratio](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-2ed1365fc34ed9847121).

[Initialized-process first-query raw results](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-e1ba0dcbe694eeac1b99) — SHA256 `12ac0c8a05571e4f55841076f248b1698bcd468927e75a5d5a17d136c401f876`.
